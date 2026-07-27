import { Data, Effect } from "effect"
import { dual } from "effect/Function"
import { pipeArguments, type Pipeable } from "effect/Pipeable"
import native, {
  type LazyTensor as NativeLazyTensorType,
  type NativeDType,
  type NativeTensor as NativeTensorType
} from "@effect-torch/native"
import { CurrentDevice, type DeviceKind } from "./Device.ts"

const {
  CancellationToken,
  evalLazy,
  LazyTensor: NativeLazyTensor,
  loadTensors,
  reportExternalMemory,
  saveTensors
} = native

/**
 * Element data types supported by the native backend.
 *
 * @since 0.1.0
 * @category models
 */
export type DType = "f32" | "f64" | "i64" | "u8" | "u32"

/**
 * JavaScript typed arrays accepted by {@link fromTypedArray} and returned by
 * {@link toTypedArray}, matching the supported {@link DType}s.
 *
 * @since 0.1.0
 * @category models
 */
export type TypedArray = Float32Array | Float64Array | BigInt64Array | Uint8Array | Uint32Array

/**
 * Common options for tensor constructors.
 *
 * @since 0.1.0
 * @category models
 */
export interface TensorOptions {
  readonly dtype?: DType
}

/**
 * Error type raised by tensor operations, both at graph construction time
 * (shape, dtype and device validation) and at evaluation time.
 *
 * @since 0.1.0
 * @category errors
 */
export class TensorError extends Data.TaggedError("TensorError")<{
  readonly op: string
  readonly message: string
}> {}

/**
 * Common supertype of {@link LazyTensor} and {@link Tensor}. Every operation
 * accepts this type, so lazy and evaluated tensors can be mixed freely.
 *
 * @since 0.1.0
 * @category models
 */
export interface GenericTensor extends Pipeable {
  readonly [TensorTypeId]: TensorTypeId
  readonly _tag: "LazyTensor" | "Tensor"
  /** @internal */
  readonly lazy: NativeLazyTensorType
  readonly shape: ReadonlyArray<number>
  readonly dtype: DType
  readonly device: DeviceKind
}

/**
 * A tensor described by a lazy computation graph. Operations on lazy tensors
 * only extend the graph; nothing is computed until {@link evaluate} is called.
 *
 * @since 0.1.0
 * @category models
 */
export interface LazyTensor extends GenericTensor {
  readonly _tag: "LazyTensor"
}

/**
 * A materialized tensor whose data resides on the device, obtained through
 * {@link evaluate}.
 *
 * @since 0.1.0
 * @category models
 */
export interface Tensor extends GenericTensor {
  readonly _tag: "Tensor"
  /** @internal */
  readonly materialized: NativeTensorType
}

const TensorTypeId: unique symbol = Symbol.for("@effect-torch/core/Tensor")

/**
 * @since 0.1.0
 * @category symbols
 */
export type TensorTypeId = typeof TensorTypeId

const TensorProto = {
  [TensorTypeId]: TensorTypeId,
  pipe(this: GenericTensor) {
    return pipeArguments(this, arguments)
  }
}

const makeLazy = (
  lazy: NativeLazyTensorType,
  shape: ReadonlyArray<number>,
  dtype: DType,
  device: DeviceKind
): LazyTensor => {
  const self = Object.create(TensorProto)
  self._tag = "LazyTensor"
  self.lazy = lazy
  self.shape = shape
  self.dtype = dtype
  self.device = device
  return self
}

const fromHandle = (handle: NativeTensorType): Tensor => {
  const self = Object.create(TensorProto)
  self._tag = "Tensor"
  self.lazy = NativeLazyTensor.fromMaterialized(handle)
  self.materialized = handle
  self.shape = handle.shape
  self.dtype = handle.dtype as DType
  self.device = handle.device as DeviceKind
  return self
}

/**
 * Returns `true` if the tensor is still a lazy computation graph.
 *
 * @since 0.1.0
 * @category refinements
 */
export const isLazyTensor = (self: GenericTensor): self is LazyTensor => self._tag === "LazyTensor"

/**
 * Returns `true` if the tensor has been materialized on the device.
 *
 * @since 0.1.0
 * @category refinements
 */
export const isTensor = (self: GenericTensor): self is Tensor => self._tag === "Tensor"

const validateShape = (op: string, shape: ReadonlyArray<number>): Array<number> =>
  shape.map((dim) => {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new Error(`${op}: invalid shape dimension ${dim}, expected a non-negative integer`)
    }
    return dim
  })

const broadcastShapes = (
  op: string,
  a: ReadonlyArray<number>,
  b: ReadonlyArray<number>
): Array<number> => {
  const rank = Math.max(a.length, b.length)
  const out: Array<number> = []
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - 1 - i] ?? 1
    const db = b[b.length - 1 - i] ?? 1
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`${op}: shapes [${a}] and [${b}] are not broadcastable`)
    }
    out.unshift(Math.max(da, db))
  }
  return out
}

const matmulShape = (a: ReadonlyArray<number>, b: ReadonlyArray<number>): Array<number> => {
  if (a.length < 2 || b.length < 2) {
    throw new Error(`matmul: expected tensors of rank >= 2, got [${a}] and [${b}]`)
  }
  const m = a[a.length - 2]
  const ka = a[a.length - 1]
  const kb = b[b.length - 2]
  const n = b[b.length - 1]
  if (ka !== kb) {
    throw new Error(`matmul: inner dimensions mismatch, got [${a}] and [${b}]`)
  }
  const batch = broadcastShapes("matmul", a.slice(0, -2), b.slice(0, -2))
  return [...batch, m, n]
}

const checkCompatible = (op: string, a: GenericTensor, b: GenericTensor): void => {
  if (a.dtype !== b.dtype) {
    throw new Error(`${op}: dtype mismatch, got ${a.dtype} and ${b.dtype}, use cast for explicit conversion`)
  }
  if (a.device !== b.device) {
    throw new Error(`${op}: device mismatch, got ${a.device} and ${b.device}`)
  }
}

const numel = (shape: ReadonlyArray<number>): number => shape.reduce((a, b) => a * b, 1)

/**
 * Right-hand operand accepted by arithmetic and comparison operations. A
 * `number` is lifted to a scalar tensor with the same dtype and device as the
 * left operand.
 *
 * @since 0.1.0
 * @category models
 */
export type TensorOrScalar = GenericTensor | number

// Scalar operands are pure values: the same (value, dtype, device) triple
// can share one leaf node forever instead of allocating a fresh device
// buffer per use. On backends with a scanning allocator (Metal) unbounded
// tiny-buffer churn is expensive, so the cache is size-bounded — hot
// constants (lr, betas, eps, 0.5, 1, 2) stay resident; per-step varying
// scalars (bias-correction factors) rotate through.
const scalarLeafCache = new Map<string, NativeLazyTensorType>()
const SCALAR_LEAF_CACHE_LIMIT = 4096

const liftOperand = (
  self: GenericTensor,
  other: GenericTensor | number
): { readonly lazy: NativeLazyTensorType; readonly shape: ReadonlyArray<number> } => {
  if (typeof other === "number") {
    const key = `${self.device}:${self.dtype}:${other}`
    let lazy = scalarLeafCache.get(key)
    if (lazy === undefined) {
      lazy = NativeLazyTensor.full([], other, self.dtype as NativeDType, self.device)
      if (scalarLeafCache.size >= SCALAR_LEAF_CACHE_LIMIT) {
        scalarLeafCache.delete(scalarLeafCache.keys().next().value!)
      }
      scalarLeafCache.set(key, lazy)
    }
    return { lazy, shape: [] }
  }
  return { lazy: other.lazy, shape: other.shape }
}

const binaryOp = (
  op: string,
  native: (a: NativeLazyTensorType, b: NativeLazyTensorType) => NativeLazyTensorType,
  outDtype: (dtype: DType) => DType = (dtype) => dtype
): {
  (other: TensorOrScalar): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, other: TensorOrScalar): Effect.Effect<LazyTensor, TensorError>
} =>
  dual(
    2,
    (self: GenericTensor, other: TensorOrScalar): Effect.Effect<LazyTensor, TensorError> =>
      Effect.try({
        try: () => {
          if (typeof other !== "number") checkCompatible(op, self, other)
          const rhs = liftOperand(self, other)
          return makeLazy(
            native(self.lazy, rhs.lazy),
            broadcastShapes(op, self.shape, rhs.shape),
            outDtype(self.dtype),
            self.device
          )
        },
        catch: (error) =>
          new TensorError({ op, message: error instanceof Error ? error.message : String(error) })
      })
  )

const unaryOp = (
  op: string,
  native: (a: NativeLazyTensorType) => NativeLazyTensorType
): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError> =>
(self) =>
  Effect.try({
    try: () => makeLazy(native(self.lazy), self.shape, self.dtype, self.device),
    catch: (error) =>
      new TensorError({ op, message: error instanceof Error ? error.message : String(error) })
  })

const normalizeDim = (op: string, rank: number, dim: number): number => {
  const normalized = dim < 0 ? dim + rank : dim
  if (!Number.isInteger(normalized) || normalized < 0 || normalized >= rank) {
    throw new Error(`${op}: dimension ${dim} out of range for rank ${rank}`)
  }
  return normalized
}

const dualOptions = <O, R = never>(
  impl: (self: GenericTensor, options: O | undefined) => Effect.Effect<LazyTensor, TensorError, R>
): {
  (options?: O): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError, R>
  (self: GenericTensor, options?: O): Effect.Effect<LazyTensor, TensorError, R>
} =>
  dual(
    (args) =>
      args.length === 2 || (args.length === 1 && args[0] !== undefined && TensorTypeId in (args[0] as object)),
    impl
  )

/**
 * Creates a lazy tensor filled with zeros.
 *
 * @since 0.1.0
 * @category constructors
 */
export const zeros = (
  shape: ReadonlyArray<number>,
  options: TensorOptions = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    return yield* Effect.try({
      try: () => {
        const validShape = validateShape("zeros", shape)
        return makeLazy(
          NativeLazyTensor.zeros(validShape, options.dtype as NativeDType, device),
          validShape,
          options.dtype ?? "f32",
          device
        )
      },
      catch: (error) =>
        new TensorError({ op: "zeros", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Creates a lazy tensor filled with ones.
 *
 * @since 0.1.0
 * @category constructors
 */
export const ones = (
  shape: ReadonlyArray<number>,
  options: TensorOptions = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    return yield* Effect.try({
      try: () => {
        const validShape = validateShape("ones", shape)
        return makeLazy(
          NativeLazyTensor.ones(validShape, options.dtype as NativeDType, device),
          validShape,
          options.dtype ?? "f32",
          device
        )
      },
      catch: (error) =>
        new TensorError({ op: "ones", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Creates a lazy tensor filled with a constant value.
 *
 * @since 0.1.0
 * @category constructors
 */
export const full = (
  shape: ReadonlyArray<number>,
  value: number,
  options: TensorOptions = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    return yield* Effect.try({
      try: () => {
        const validShape = validateShape("full", shape)
        return makeLazy(
          NativeLazyTensor.full(validShape, value, options.dtype as NativeDType, device),
          validShape,
          options.dtype ?? "f32",
          device
        )
      },
      catch: (error) =>
        new TensorError({ op: "full", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Creates a lazy tensor sampled from a standard normal distribution. Only
 * floating point dtypes are supported.
 *
 * @since 0.1.0
 * @category constructors
 */
export const randn = (
  shape: ReadonlyArray<number>,
  options: { readonly dtype?: "f32" | "f64" } = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    return yield* Effect.try({
      try: () => {
        const validShape = validateShape("randn", shape)
        return makeLazy(
          NativeLazyTensor.randn(validShape, options.dtype as NativeDType, device),
          validShape,
          options.dtype ?? "f32",
          device
        )
      },
      catch: (error) =>
        new TensorError({ op: "randn", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Creates a lazy tensor sampled uniformly from `[min, max)`. Only floating
 * point dtypes are supported. Like {@link randn}, draws happen at
 * evaluation time: evaluate related tensors together in one walk.
 *
 * @since 0.1.0
 * @category constructors
 */
export const uniform = (
  shape: ReadonlyArray<number>,
  options: { readonly min?: number; readonly max?: number; readonly dtype?: "f32" | "f64" } = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    return yield* Effect.try({
      try: () => {
        const validShape = validateShape("uniform", shape)
        return makeLazy(
          NativeLazyTensor.uniform(
            validShape,
            options.min ?? 0,
            options.max ?? 1,
            options.dtype as NativeDType,
            device
          ),
          validShape,
          options.dtype ?? "f32",
          device
        )
      },
      catch: (error) =>
        new TensorError({ op: "uniform", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Creates a lazy 1-dimensional tensor of `steps` evenly spaced values from
 * `start` to `end`, both inclusive.
 *
 * @since 0.1.0
 * @category constructors
 */
export const linspace = (
  start: number,
  end: number,
  steps: number,
  options: TensorOptions = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (!Number.isInteger(steps) || steps < 1) {
      return yield* new TensorError({
        op: "linspace",
        message: `linspace: steps must be a positive integer, got ${steps}`
      })
    }
    if (options.dtype !== undefined && options.dtype !== "f32" && options.dtype !== "f64") {
      return yield* new TensorError({
        op: "linspace",
        message: `linspace: dtype must be f32 or f64, got ${options.dtype}`
      })
    }
    if (steps === 1) {
      return yield* full([1], start, options)
    }
    const base = yield* arange(steps, undefined, { dtype: options.dtype ?? "f32" })
    return yield* add(yield* mul(base, (end - start) / (steps - 1)), start)
  })

/**
 * Creates a lazy 1-dimensional tensor of evenly spaced values in the interval
 * `[start, end)`. When `end` is omitted the range is `[0, start)`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const arange = (
  start: number,
  end?: number,
  options: { readonly step?: number; readonly dtype?: DType } = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    return yield* Effect.try({
      try: () => {
        const from = end === undefined ? 0 : start
        const to = end === undefined ? start : end
        const step = options.step ?? 1
        const size = Math.max(0, Math.ceil((to - from) / step))
        return makeLazy(
          NativeLazyTensor.arange(from, to, step, options.dtype as NativeDType, device),
          [size],
          options.dtype ?? "f32",
          device
        )
      },
      catch: (error) =>
        new TensorError({ op: "arange", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Creates a lazy `n x n` identity matrix.
 *
 * @since 0.1.0
 * @category constructors
 */
export const eye = (
  n: number,
  options: TensorOptions = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    return yield* Effect.try({
      try: () => {
        const [size] = validateShape("eye", [n])
        if (size === 0) throw new Error("eye: n must be positive")
        return makeLazy(
          NativeLazyTensor.eye(size, options.dtype as NativeDType, device),
          [size, size],
          options.dtype ?? "f32",
          device
        )
      },
      catch: (error) =>
        new TensorError({ op: "eye", message: error instanceof Error ? error.message : String(error) })
    })
  })

const dtypeOfTypedArray = (data: TypedArray): DType => {
  if (data instanceof Float32Array) return "f32"
  if (data instanceof Float64Array) return "f64"
  if (data instanceof BigInt64Array) return "i64"
  if (data instanceof Uint8Array) return "u8"
  if (data instanceof Uint32Array) return "u32"
  throw new Error(`fromTypedArray: unsupported typed array ${(data as object).constructor.name}`)
}

/**
 * Creates a lazy tensor from a typed array. The dtype is inferred from the
 * array type and the shape defaults to `[data.length]`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromTypedArray = (
  data: TypedArray,
  shape?: ReadonlyArray<number>
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    return yield* Effect.try({
      try: () => {
        const dtype = dtypeOfTypedArray(data)
        const validShape = shape === undefined ? [data.length] : validateShape("fromTypedArray", shape)
        if (numel(validShape) !== data.length) {
          throw new Error(
            `fromTypedArray: data length ${data.length} does not match shape [${validShape}]`
          )
        }
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        return makeLazy(
          NativeLazyTensor.fromBytes(bytes, validShape, dtype as NativeDType, device),
          validShape,
          dtype,
          device
        )
      },
      catch: (error) =>
        new TensorError({
          op: "fromTypedArray",
          message: error instanceof Error ? error.message : String(error)
        })
    })
  })

/**
 * Creates a lazy tensor of zeros with the same shape and dtype as the input.
 *
 * @since 0.1.0
 * @category constructors
 */
export const zerosLike = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  zeros(self.shape, { dtype: self.dtype })

/**
 * Creates a lazy tensor of ones with the same shape and dtype as the input.
 *
 * @since 0.1.0
 * @category constructors
 */
export const onesLike = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  ones(self.shape, { dtype: self.dtype })

/**
 * Creates a lazy tensor filled with `value`, with the same shape and dtype
 * as the input.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fullLike = (
  self: GenericTensor,
  value: number
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> => full(self.shape, value, { dtype: self.dtype })

/**
 * Returns the shape of a tensor.
 *
 * @since 0.1.0
 * @category getters
 */
export const shape = (self: GenericTensor): ReadonlyArray<number> => self.shape

/**
 * Returns the dtype of a tensor.
 *
 * @since 0.1.0
 * @category getters
 */
export const dtype = (self: GenericTensor): DType => self.dtype

/**
 * Returns the device a tensor lives on.
 *
 * @since 0.1.0
 * @category getters
 */
export const device = (self: GenericTensor): DeviceKind => self.device

/**
 * Elementwise addition with broadcasting. Fails with {@link TensorError} if
 * the dtypes or devices differ.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const add = binaryOp("add", (a, b) => a.add(b))

/**
 * Elementwise subtraction with broadcasting.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sub = binaryOp("sub", (a, b) => a.sub(b))

/**
 * Elementwise multiplication with broadcasting.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const mul = binaryOp("mul", (a, b) => a.mul(b))

/**
 * Elementwise division with broadcasting.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const div = binaryOp("div", (a, b) => a.div(b))

/**
 * Elementwise maximum of two tensors with broadcasting. At equal elements
 * the gradient flows to the left operand only. (Not to be confused with
 * the reduction {@link max}.)
 *
 * @since 0.1.0
 * @category elementwise
 */
export const maximum = binaryOp("maximum", (a, b) => a.maximum(b))

/**
 * Elementwise minimum of two tensors with broadcasting. At equal elements
 * the gradient flows to the left operand only. (Not to be confused with
 * the reduction {@link min}.)
 *
 * @since 0.1.0
 * @category elementwise
 */
export const minimum = binaryOp("minimum", (a, b) => a.minimum(b))

/**
 * Elementwise equality comparison with broadcasting. Returns a `u8` tensor.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const eq = binaryOp("eq", (a, b) => a.eq(b), () => "u8")

/**
 * Elementwise greater-than comparison with broadcasting. Returns a `u8` tensor.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const gt = binaryOp("gt", (a, b) => a.gt(b), () => "u8")

/**
 * Elementwise less-than comparison with broadcasting. Returns a `u8` tensor.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const lt = binaryOp("lt", (a, b) => a.lt(b), () => "u8")

/**
 * Elementwise greater-than-or-equal comparison with broadcasting. Returns a
 * `u8` tensor.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const ge = binaryOp("ge", (a, b) => a.ge(b), () => "u8")

/**
 * Elementwise less-than-or-equal comparison with broadcasting. Returns a `u8`
 * tensor.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const le = binaryOp("le", (a, b) => a.le(b), () => "u8")

/**
 * Elementwise negation.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const neg = unaryOp("neg", (a) => a.neg())

/**
 * Elementwise absolute value.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const abs = unaryOp("abs", (a) => a.abs())

/**
 * Elementwise square root.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sqrt = unaryOp("sqrt", (a) => a.sqrt())

/**
 * Elementwise exponential.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const exp = unaryOp("exp", (a) => a.exp())

/**
 * Elementwise natural logarithm.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log = unaryOp("log", (a) => a.log())

/**
 * Elementwise sine.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sin = unaryOp("sin", (a) => a.sin())

/**
 * Elementwise cosine.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const cos = unaryOp("cos", (a) => a.cos())

/**
 * Elementwise hyperbolic tangent.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const tanh = unaryOp("tanh", (a) => a.tanh())

/**
 * Elementwise rectified linear unit, `max(x, 0)`. The gradient at `x = 0`
 * is taken to be `0`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const relu = unaryOp("relu", (a) => a.relu())

/**
 * Elementwise error function.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const erf = unaryOp("erf", (a) => a.erf())

/**
 * Elementwise floor. The gradient is `0` almost everywhere.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const floor = unaryOp("floor", (a) => a.floor())

/**
 * Elementwise ceiling. The gradient is `0` almost everywhere.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const ceil = unaryOp("ceil", (a) => a.ceil())

/**
 * Elementwise rounding to the nearest integer. The gradient is `0` almost
 * everywhere.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const round = unaryOp("round", (a) => a.round())

/**
 * Elementwise sign: `-1`, `0` or `1`. The gradient is `0` everywhere.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sign = unaryOp("sign", (a) => a.sign())

/**
 * Elementwise square, `x * x`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const square = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> => mul(self, self)

/**
 * Elementwise reciprocal square root, `x ** -0.5`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const rsqrt = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> => pow(self, -0.5)

/**
 * Elementwise reciprocal, `1 / x`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const reciprocal = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> => pow(self, -1)

/**
 * Elementwise `exp(x) - 1`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const expm1 = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.flatMap(exp(self), (e) => sub(e, 1))

/**
 * Elementwise `log(1 + x)`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log1p = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.flatMap(add(self, 1), (t) => log(t))

/**
 * Elementwise base-2 logarithm.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log2 = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.flatMap(log(self), (t) => div(t, Math.LN2))

/**
 * Elementwise base-10 logarithm.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log10 = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.flatMap(log(self), (t) => div(t, Math.LN10))

/**
 * Elementwise hyperbolic sine, `(exp(x) - exp(-x)) / 2`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sinh = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    const e = yield* exp(self)
    const ne = yield* exp(yield* neg(self))
    return yield* div(yield* sub(e, ne), 2)
  })

/**
 * Elementwise hyperbolic cosine, `(exp(x) + exp(-x)) / 2`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const cosh = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    const e = yield* exp(self)
    const ne = yield* exp(yield* neg(self))
    return yield* div(yield* add(e, ne), 2)
  })

/**
 * Elementwise tangent, `sin(x) / cos(x)`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const tan = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    return yield* div(yield* sin(self), yield* cos(self))
  })

/**
 * Elementwise not-equal comparison with broadcasting. Returns a `u8` tensor.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const ne: {
  (other: TensorOrScalar): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, other: TensorOrScalar): Effect.Effect<LazyTensor, TensorError>
} = dual(
  2,
  (self: GenericTensor, other: TensorOrScalar): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      return yield* maximum(yield* lt(self, other), yield* gt(self, other))
    })
)

/**
 * Elementwise logical AND on `u8` tensors with broadcasting.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const logicalAnd = binaryOp("logicalAnd", (a, b) => a.minimum(b))

/**
 * Elementwise logical OR on `u8` tensors with broadcasting.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const logicalOr = binaryOp("logicalOr", (a, b) => a.maximum(b))

/**
 * Elementwise logical NOT on a `u8` tensor: `0` becomes `1`, everything
 * else becomes `0`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const logicalNot = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> => eq(self, 0)

/**
 * Elementwise remainder of the division `self / other`, following the sign
 * of the divisor (Python/PyTorch semantics): `self - floor(self / other) * other`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const remainder: {
  (other: TensorOrScalar): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, other: TensorOrScalar): Effect.Effect<LazyTensor, TensorError>
} = dual(
  2,
  (self: GenericTensor, other: TensorOrScalar): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      const q = yield* floor(yield* div(self, other))
      return yield* sub(self, yield* mul(q, other))
    })
)

/**
 * Selects elements from `a` or `b` depending on a `u8` condition tensor,
 * with broadcasting across all three inputs. Gradients flow only to the
 * selected side.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const where: {
  (a: TensorOrScalar, b: TensorOrScalar): (cond: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (cond: GenericTensor, a: TensorOrScalar, b: TensorOrScalar): Effect.Effect<LazyTensor, TensorError>
} = dual(
  3,
  (cond: GenericTensor, a: TensorOrScalar, b: TensorOrScalar): Effect.Effect<LazyTensor, TensorError> =>
    Effect.try({
      try: () => {
        if (cond.dtype !== "u8") {
          throw new Error(`where: condition must be u8, got ${cond.dtype}`)
        }
        const ref = typeof a === "number" ? (typeof b === "number" ? undefined : b) : a
        if (ref === undefined) {
          throw new Error("where: at least one of a and b must be a tensor")
        }
        if (cond.device !== ref.device) {
          throw new Error(`where: device mismatch, got ${cond.device} and ${ref.device}`)
        }
        if (typeof a !== "number" && typeof b !== "number") {
          checkCompatible("where", a, b)
        }
        const lhs = liftOperand(ref, a)
        const rhs = liftOperand(ref, b)
        const shape = broadcastShapes(
          "where",
          broadcastShapes("where", cond.shape, lhs.shape),
          rhs.shape
        )
        return makeLazy(cond.lazy.whereCond(lhs.lazy, rhs.lazy), shape, ref.dtype, ref.device)
      },
      catch: (error) =>
        new TensorError({ op: "where", message: error instanceof Error ? error.message : String(error) })
    })
)

/**
 * Elementwise logistic sigmoid, `1 / (1 + exp(-x))`, computed as the
 * numerically stable `tanh(x / 2) / 2 + 1/2`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sigmoid = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    const t = yield* tanh(yield* div(self, 2))
    return yield* add(yield* div(t, 2), 0.5)
  })

/**
 * Softmax over the given dimensions (the last one by default), computed
 * with max-subtraction for numerical stability.
 *
 * @since 0.1.0
 * @category neural network
 */
export const softmax = dualOptions(
  (self: GenericTensor, options: ReduceOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      const dims = normalizeDims("softmax", self.shape.length, options.dims ?? [self.shape.length - 1])
      const m = yield* max(self, { dims, keepdims: true })
      const e = yield* exp(yield* sub(self, m))
      return yield* div(e, yield* sum(e, { dims, keepdims: true }))
    })
)

/**
 * Log-softmax over the given dimensions (the last one by default),
 * `log(softmax(x))` computed without materializing the softmax itself.
 *
 * @since 0.1.0
 * @category neural network
 */
export const logSoftmax = dualOptions(
  (self: GenericTensor, options: ReduceOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      const dims = normalizeDims("logSoftmax", self.shape.length, options.dims ?? [self.shape.length - 1])
      const m = yield* max(self, { dims, keepdims: true })
      const shifted = yield* sub(self, m)
      const s = yield* sum(yield* exp(shifted), { dims, keepdims: true })
      return yield* sub(shifted, yield* log(s))
    })
)

/**
 * SiLU / swish activation, `x * sigmoid(x)`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const silu = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    return yield* mul(self, yield* sigmoid(self))
  })

/**
 * Softplus activation, `log(1 + exp(x))`, computed in the numerically
 * stable form `max(x, 0) + log1p(exp(-|x|))`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const softplus = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    const head = yield* maximum(self, 0)
    const tail = yield* log1p(yield* exp(yield* neg(yield* abs(self))))
    return yield* add(head, tail)
  })

/**
 * Options for {@link elu}. `alpha` is the saturation magnitude for negative
 * inputs, default `1`.
 *
 * @since 0.1.0
 * @category models
 */
export interface EluOptions {
  readonly alpha?: number
}

/**
 * Exponential linear unit: `x` when `x > 0`, `alpha * (exp(x) - 1)`
 * otherwise.
 *
 * @since 0.1.0
 * @category neural network
 */
export const elu = dualOptions(
  (self: GenericTensor, options: EluOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      const alpha = options.alpha ?? 1
      const negative = yield* mul(yield* expm1(self), alpha)
      return yield* where(yield* gt(self, 0), self, negative)
    })
)

/**
 * Options for {@link leakyRelu}. `negativeSlope` defaults to `0.01`.
 *
 * @since 0.1.0
 * @category models
 */
export interface LeakyReluOptions {
  readonly negativeSlope?: number
}

/**
 * Leaky ReLU: `x` when `x > 0`, `negativeSlope * x` otherwise.
 *
 * @since 0.1.0
 * @category neural network
 */
export const leakyRelu = dualOptions(
  (self: GenericTensor, options: LeakyReluOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      return yield* maximum(self, yield* mul(self, options.negativeSlope ?? 0.01))
    })
)

/**
 * Options for {@link gelu}. `approximate: "tanh"` selects the tanh
 * approximation instead of the exact erf form.
 *
 * @since 0.1.0
 * @category models
 */
export interface GeluOptions {
  readonly approximate?: "none" | "tanh"
}

/**
 * Gaussian error linear unit. The default exact form is
 * `x * (1 + erf(x / sqrt(2))) / 2`; `approximate: "tanh"` uses the faster
 * tanh approximation.
 *
 * @since 0.1.0
 * @category neural network
 */
export const gelu = dualOptions(
  (self: GenericTensor, options: GeluOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      if (options.approximate === "tanh") {
        const c = Math.sqrt(2 / Math.PI)
        const inner = yield* mul(yield* add(self, yield* mul(yield* pow(self, 3), 0.044715)), c)
        return yield* mul(yield* mul(self, 0.5), yield* add(yield* tanh(inner), 1))
      }
      return yield* mul(yield* mul(self, 0.5), yield* add(yield* erf(yield* div(self, Math.SQRT2)), 1))
    })
)

/**
 * Mish activation, `x * tanh(softplus(x))`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const mish = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    return yield* mul(self, yield* tanh(yield* softplus(self)))
  })

/**
 * Options for {@link clamp}. At least one of `min` / `max` must be given.
 *
 * @since 0.1.0
 * @category models
 */
export interface ClampOptions {
  readonly min?: number
  readonly max?: number
}

/**
 * Clamps every element into `[min, max]`; either bound may be omitted.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const clamp = dualOptions(
  (self: GenericTensor, options: ClampOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      if (options.min === undefined && options.max === undefined) {
        return yield* new TensorError({ op: "clamp", message: "clamp: at least one of min and max is required" })
      }
      let out: GenericTensor = self
      if (options.min !== undefined) out = yield* maximum(out, options.min)
      if (options.max !== undefined) out = yield* minimum(out, options.max)
      return out as LazyTensor
    })
)

/**
 * Hardtanh activation: clamp to `[-1, 1]`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const hardtanh = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  clamp(self, { min: -1, max: 1 })

/**
 * Options for {@link dropout}. `p` is the probability of zeroing an
 * element; surviving elements are scaled by `1 / (1 - p)`.
 *
 * @since 0.1.0
 * @category models
 */
export interface DropoutOptions {
  readonly p?: number
}

/**
 * Randomly zeroes elements with probability `p` and scales the survivors by
 * `1 / (1 - p)` (inverted dropout). This is the functional form: it always
 * applies — skip it at evaluation time by construction. The mask is drawn
 * at evaluation time, so the usual `randn` rule applies: evaluate the loss
 * and its gradients together in one walk.
 *
 * @since 0.1.0
 * @category neural network
 */
export const dropout = dualOptions(
  (
    self: GenericTensor,
    options: DropoutOptions = {}
  ): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
    Effect.gen(function* () {
      const p = options.p ?? 0.5
      if (p < 0 || p >= 1) {
        return yield* new TensorError({ op: "dropout", message: `dropout: p must be in [0, 1), got ${p}` })
      }
      if (!isFloatDtype(self.dtype)) {
        return yield* new TensorError({
          op: "dropout",
          message: `dropout: dtype must be f32 or f64, got ${self.dtype}`
        })
      }
      if (p === 0) {
        return yield* add(self, 0)
      }
      const mask = yield* ge(yield* uniform(self.shape, { dtype: self.dtype === "f64" ? "f64" : "f32" }), p)
      return yield* where(mask, yield* div(self, 1 - p), 0)
    })
)

/**
 * Elementwise exponentiation to a constant power.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const pow: {
  (exponent: number): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, exponent: number): Effect.Effect<LazyTensor, TensorError>
} = dual(2, (self: GenericTensor, exponent: number): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => makeLazy(self.lazy.pow(exponent), self.shape, self.dtype, self.device),
    catch: (error) =>
      new TensorError({ op: "pow", message: error instanceof Error ? error.message : String(error) })
  })
)

/**
 * Batched matrix multiplication over the last two dimensions, with
 * broadcasting of the leading batch dimensions.
 *
 * @since 0.1.0
 * @category operations
 */
export const matmul: {
  (other: GenericTensor): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, other: GenericTensor): Effect.Effect<LazyTensor, TensorError>
} = dual(2, (self: GenericTensor, other: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => {
      checkCompatible("matmul", self, other)
      return makeLazy(
        self.lazy.matmul(other.lazy),
        matmulShape(self.shape, other.shape),
        self.dtype,
        self.device
      )
    },
    catch: (error) =>
      new TensorError({ op: "matmul", message: error instanceof Error ? error.message : String(error) })
  })
)

/**
 * Options for reduction operations.
 *
 * @since 0.1.0
 * @category models
 */
export interface ReduceOptions {
  readonly dims?: ReadonlyArray<number>
  readonly keepdims?: boolean
}

const normalizeDims = (op: string, rank: number, dims: ReadonlyArray<number>): Array<number> => {
  const normalized = dims.map((d) => {
    const dim = d < 0 ? d + rank : d
    if (!Number.isInteger(dim) || dim < 0 || dim >= rank) {
      throw new Error(`${op}: dimension ${d} out of range for rank ${rank}`)
    }
    return dim
  })
  const unique = [...new Set(normalized)]
  if (unique.length !== normalized.length) {
    throw new Error(`${op}: duplicate dimensions [${dims}]`)
  }
  return unique.sort((a, b) => a - b)
}

const reducedShape = (
  op: string,
  shape: ReadonlyArray<number>,
  dims: ReadonlyArray<number>,
  keepdims: boolean
): Array<number> => {
  const normalized = normalizeDims(op, shape.length, dims)
  if (keepdims) {
    return shape.map((d, i) => (normalized.includes(i) ? 1 : d))
  }
  return shape.filter((_, i) => !normalized.includes(i))
}

const reduceOp = (
  op: string,
  native: (a: NativeLazyTensorType, dims: Array<number>, keepdims: boolean) => NativeLazyTensorType
): {
  (options?: ReduceOptions): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, options?: ReduceOptions): Effect.Effect<LazyTensor, TensorError>
} =>
  dual(
    (args) => args.length === 2 || (args.length === 1 && args[0] !== undefined && TensorTypeId in (args[0] as object)),
    (self: GenericTensor, options: ReduceOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
      Effect.try({
        try: () => {
          const dims = options.dims ?? self.shape.map((_, i) => i)
          const keepdims = options.keepdims ?? false
          const normalized = normalizeDims(op, self.shape.length, dims)
          return makeLazy(
            native(self.lazy, normalized, keepdims),
            reducedShape(op, self.shape, dims, keepdims),
            self.dtype,
            self.device
          )
        },
        catch: (error) =>
          new TensorError({ op, message: error instanceof Error ? error.message : String(error) })
      })
  )

/**
 * Sums a tensor over the given dimensions (all of them by default). Negative
 * dimensions count from the end.
 *
 * @since 0.1.0
 * @category reductions
 */
export const sum = reduceOp("sum", (a, dims, keepdims) => a.sum(dims, keepdims))

/**
 * Computes the mean of a tensor over the given dimensions (all of them by
 * default). Negative dimensions count from the end.
 *
 * @since 0.1.0
 * @category reductions
 */
export const mean = reduceOp("mean", (a, dims, keepdims) => a.mean(dims, keepdims))

/**
 * Computes the maximum of a tensor over the given dimensions (all of them by
 * default). Negative dimensions count from the end.
 *
 * @since 0.1.0
 * @category reductions
 */
export const max = reduceOp("max", (a, dims, keepdims) => a.max(dims, keepdims))

/**
 * Computes the minimum of a tensor over the given dimensions (all of them by
 * default). Negative dimensions count from the end.
 *
 * @since 0.1.0
 * @category reductions
 */
export const min = reduceOp("min", (a, dims, keepdims) => a.min(dims, keepdims))

/**
 * Returns the indices of the maximum values along `dim` as an `i64` tensor,
 * with `dim` removed from the shape. Not differentiable.
 *
 * @since 0.1.0
 * @category reductions
 */
export const argmax: {
  (dim: number): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, dim: number): Effect.Effect<LazyTensor, TensorError>
} = dual(2, (self: GenericTensor, dim: number): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => {
      const d = normalizeDim("argmax", self.shape.length, dim)
      return makeLazy(
        self.lazy.argmax(d),
        self.shape.filter((_, i) => i !== d),
        "i64",
        self.device
      )
    },
    catch: (error) =>
      new TensorError({ op: "argmax", message: error instanceof Error ? error.message : String(error) })
  })
)

/**
 * Returns the indices of the minimum values along `dim` as an `i64` tensor,
 * with `dim` removed from the shape. Not differentiable.
 *
 * @since 0.1.0
 * @category reductions
 */
export const argmin: {
  (dim: number): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, dim: number): Effect.Effect<LazyTensor, TensorError>
} = dual(2, (self: GenericTensor, dim: number): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => {
      const d = normalizeDim("argmin", self.shape.length, dim)
      return makeLazy(
        self.lazy.argmin(d),
        self.shape.filter((_, i) => i !== d),
        "i64",
        self.device
      )
    },
    catch: (error) =>
      new TensorError({ op: "argmin", message: error instanceof Error ? error.message : String(error) })
  })
)

/**
 * Cumulative sum along `dim`, preserving the shape.
 *
 * @since 0.1.0
 * @category reductions
 */
export const cumsum: {
  (dim: number): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, dim: number): Effect.Effect<LazyTensor, TensorError>
} = dual(2, (self: GenericTensor, dim: number): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => {
      const d = normalizeDim("cumsum", self.shape.length, dim)
      return makeLazy(self.lazy.cumsum(d), self.shape, self.dtype, self.device)
    },
    catch: (error) =>
      new TensorError({ op: "cumsum", message: error instanceof Error ? error.message : String(error) })
  })
)

/**
 * Options for {@link variance} and {@link std}. `correction` is the Bessel
 * correction subtracted from the element count (`1` gives the unbiased
 * estimator, `0` the population variance).
 *
 * @since 0.1.0
 * @category models
 */
export interface VarianceOptions extends ReduceOptions {
  readonly correction?: number
}

/**
 * Computes the variance over the given dimensions (all of them by default).
 *
 * @since 0.1.0
 * @category reductions
 */
export const variance = dualOptions(
  (self: GenericTensor, options: VarianceOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      const dims = options.dims ?? self.shape.map((_, i) => i)
      const keepdims = options.keepdims ?? false
      const correction = options.correction ?? 1
      const normalized = normalizeDims("variance", self.shape.length, dims)
      const count = normalized.reduce((n, d) => n * self.shape[d], 1)
      if (count - correction <= 0) {
        return yield* new TensorError({
          op: "variance",
          message: `variance: ${count} elements with correction ${correction} gives a non-positive denominator`
        })
      }
      const m = yield* mean(self, { dims: normalized, keepdims: true })
      const centered = yield* sub(self, m)
      const ss = yield* sum(yield* square(centered), { dims: normalized, keepdims })
      return yield* div(ss, count - correction)
    })
)

/**
 * Computes the standard deviation over the given dimensions (all of them by
 * default): the square root of {@link variance}.
 *
 * @since 0.1.0
 * @category reductions
 */
export const std = dualOptions(
  (self: GenericTensor, options: VarianceOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.flatMap(variance(self, options), (v) => sqrt(v))
)

/**
 * Options for {@link norm}. `ord` selects the norm order: `1`, `2`
 * (default), any positive number for a general p-norm, `Infinity` for the
 * maximum absolute value, `-Infinity` for the minimum.
 *
 * @since 0.1.0
 * @category models
 */
export interface NormOptions extends ReduceOptions {
  readonly ord?: number
}

/**
 * Computes the p-norm over the given dimensions (all of them by default).
 *
 * @since 0.1.0
 * @category reductions
 */
export const norm = dualOptions(
  (self: GenericTensor, options: NormOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      const ord = options.ord ?? 2
      const dims = options.dims ?? self.shape.map((_, i) => i)
      const keepdims = options.keepdims ?? false
      if (ord <= 0 && !Number.isFinite(ord)) {
        const m = yield* min(yield* abs(self), { dims, keepdims })
        return m
      }
      if (ord === Infinity) {
        return yield* max(yield* abs(self), { dims, keepdims })
      }
      if (ord <= 0) {
        return yield* new TensorError({ op: "norm", message: `norm: unsupported order ${ord}` })
      }
      if (ord === 1) {
        return yield* sum(yield* abs(self), { dims, keepdims })
      }
      if (ord === 2) {
        return yield* sqrt(yield* sum(yield* square(self), { dims, keepdims }))
      }
      return yield* pow(
        yield* sum(yield* pow(yield* abs(self), ord), { dims, keepdims }),
        1 / ord
      )
    })
)

/**
 * Computes the logical AND of all elements over the given dimensions (all
 * of them by default). The input must be `u8`.
 *
 * @since 0.1.0
 * @category reductions
 */
export const all = dualOptions(
  (self: GenericTensor, options: ReduceOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      if (self.dtype !== "u8") {
        return yield* new TensorError({ op: "all", message: `all: expected a u8 tensor, got ${self.dtype}` })
      }
      return yield* min(self, options)
    })
)

/**
 * Computes the logical OR of all elements over the given dimensions (all of
 * them by default). The input must be `u8`.
 *
 * @since 0.1.0
 * @category reductions
 */
export const any = dualOptions(
  (self: GenericTensor, options: ReduceOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      if (self.dtype !== "u8") {
        return yield* new TensorError({ op: "any", message: `any: expected a u8 tensor, got ${self.dtype}` })
      }
      return yield* max(self, options)
    })
)

/**
 * Computes `log(sum(exp(x)))` over the given dimensions (all of them by
 * default) with the usual max-subtraction for numerical stability.
 *
 * @since 0.1.0
 * @category reductions
 */
export const logsumexp = dualOptions(
  (self: GenericTensor, options: ReduceOptions = {}): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      const dims = options.dims ?? self.shape.map((_, i) => i)
      const keepdims = options.keepdims ?? false
      const normalized = normalizeDims("logsumexp", self.shape.length, dims)
      const m = yield* max(self, { dims: normalized, keepdims: true })
      const s = yield* sum(yield* exp(yield* sub(self, m)), { dims: normalized, keepdims: true })
      const out = yield* add(m, yield* log(s))
      return keepdims ? out : yield* reshape(out, reducedShape("logsumexp", self.shape, dims, false))
    })
)

/**
 * Computes the product of elements over the given dimensions (all of them
 * by default). The product of an empty set of elements is `1`.
 *
 * Implemented as a fold of per-index slices — the backend has no product
 * kernel — so the graph grows linearly with the reduced dimension sizes.
 *
 * @since 0.1.0
 * @category reductions
 */
export const prod = dualOptions<ReduceOptions, CurrentDevice>(
  (
    self: GenericTensor,
    options: ReduceOptions = {}
  ): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
    Effect.gen(function* () {
      const dims = options.dims ?? self.shape.map((_, i) => i)
      const keepdims = options.keepdims ?? false
      const normalized = normalizeDims("prod", self.shape.length, dims)
      let cur: GenericTensor = self
      for (const d of normalized) {
        const n = cur.shape[d]
        if (n === 0) {
          const shape = cur.shape.map((size, i) => (i === d ? 1 : size))
          cur = yield* ones(shape, { dtype: cur.dtype })
          continue
        }
        let acc: GenericTensor = yield* slice(cur, {
          end: cur.shape.map((size, i) => (i === d ? 1 : size))
        })
        for (let i = 1; i < n; i++) {
          const next = yield* slice(cur, {
            start: cur.shape.map((_, j) => (j === d ? i : 0)),
            end: cur.shape.map((size, j) => (j === d ? i + 1 : size))
          })
          acc = yield* mul(acc, next)
        }
        cur = acc
      }
      return keepdims
        ? yield* reshape(cur, self.shape.map((size, i) => (normalized.includes(i) ? 1 : size)))
        : yield* reshape(cur, reducedShape("prod", self.shape, dims, false))
    })
)

/**
 * Reshapes a tensor. The total number of elements must stay the same.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const reshape: {
  (shape: ReadonlyArray<number>): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, shape: ReadonlyArray<number>): Effect.Effect<LazyTensor, TensorError>
} = dual(
  2,
  (self: GenericTensor, newShape: ReadonlyArray<number>): Effect.Effect<LazyTensor, TensorError> =>
    Effect.try({
      try: () => {
        const validShape = validateShape("reshape", newShape)
        if (numel(validShape) !== numel(self.shape)) {
          throw new Error(
            `reshape: cannot reshape [${self.shape}] (${numel(self.shape)} elements) to [${validShape}] (${numel(validShape)} elements)`
          )
        }
        return makeLazy(self.lazy.reshape(validShape), validShape, self.dtype, self.device)
      },
      catch: (error) =>
        new TensorError({ op: "reshape", message: error instanceof Error ? error.message : String(error) })
    })
)

/**
 * Reorders the dimensions of a tensor. `dims` must be a permutation of the
 * tensor's rank; negative dimensions count from the end.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const transpose: {
  (dims: ReadonlyArray<number>): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, dims: ReadonlyArray<number>): Effect.Effect<LazyTensor, TensorError>
} = dual(
  2,
  (self: GenericTensor, dims: ReadonlyArray<number>): Effect.Effect<LazyTensor, TensorError> =>
    Effect.try({
      try: () => {
        if (dims.length !== self.shape.length) {
          throw new Error(
            `transpose: expected ${self.shape.length} dimensions, got [${dims}]`
          )
        }
        const normalized = dims.map((d) => {
          const dim = d < 0 ? d + self.shape.length : d
          if (!Number.isInteger(dim) || dim < 0 || dim >= self.shape.length) {
            throw new Error(`transpose: dimension ${d} out of range for rank ${self.shape.length}`)
          }
          return dim
        })
        if (new Set(normalized).size !== normalized.length) {
          throw new Error(`transpose: dims [${dims}] are not a permutation`)
        }
        const outShape = normalized.map((d) => self.shape[d])
        return makeLazy(self.lazy.permute(normalized), outShape, self.dtype, self.device)
      },
      catch: (error) =>
        new TensorError({ op: "transpose", message: error instanceof Error ? error.message : String(error) })
    })
)

/**
 * Options for {@link slice}. Each field is a per-dimension array; omitted
 * entries default to the full extent of that dimension.
 *
 * @since 0.1.0
 * @category models
 */
export interface SliceOptions {
  readonly start?: ReadonlyArray<number>
  readonly end?: ReadonlyArray<number>
  readonly stride?: ReadonlyArray<number>
}

/**
 * Extracts a per-dimension range from a tensor. Negative indices resolve
 * against the dimension size; `stride` selects every n-th element.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const slice: {
  (options: SliceOptions): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, options: SliceOptions): Effect.Effect<LazyTensor, TensorError>
} = dual(
  2,
  (self: GenericTensor, options: SliceOptions): Effect.Effect<LazyTensor, TensorError> =>
    Effect.try({
      try: () => {
        const rank = self.shape.length
        const ranges: Array<[number, number, number]> = []
        const outShape: Array<number> = []
        for (let i = 0; i < rank; i++) {
          const dim = self.shape[i]
          const stride = options.stride?.[i] ?? 1
          if (!Number.isInteger(stride) || stride <= 0) {
            throw new Error(`slice: stride at dim ${i} must be a positive integer, got ${stride}`)
          }
          const rawStart = options.start?.[i] ?? 0
          const rawEnd = options.end?.[i] ?? dim
          const start = Math.min(Math.max(rawStart < 0 ? rawStart + dim : rawStart, 0), dim)
          const end = Math.min(Math.max(rawEnd < 0 ? rawEnd + dim : rawEnd, 0), dim)
          const len = Math.max(0, Math.ceil((end - start) / stride))
          const stop = len === 0 ? start : start + (len - 1) * stride + 1
          ranges.push([start, stop, stride])
          outShape.push(len)
        }
        return makeLazy(
          self.lazy.slice(ranges.map((r) => [...r])),
          outShape,
          self.dtype,
          self.device
        )
      },
      catch: (error) =>
        new TensorError({ op: "slice", message: error instanceof Error ? error.message : String(error) })
    })
)

/**
 * Concatenates two or more tensors along an existing dimension. All tensors
 * must have the same rank, dtype and device, and match on every dimension
 * except the concatenated one.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const concat = (
  tensors: readonly [GenericTensor, GenericTensor, ...ReadonlyArray<GenericTensor>],
  options: { readonly dim?: number } = {}
): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => {
      const [first, ...rest] = tensors
      const dim = options.dim ?? 0
      const rank = first.shape.length
      const axis = dim < 0 ? dim + rank : dim
      if (!Number.isInteger(axis) || axis < 0 || axis >= rank) {
        throw new Error(`concat: dimension ${dim} out of range for rank ${rank}`)
      }
      let lazy = first.lazy
      let outShape: ReadonlyArray<number> = first.shape
      for (const next of rest) {
        checkCompatible("concat", first, next)
        if (next.shape.length !== rank) {
          throw new Error(`concat: rank mismatch, [${outShape}] vs [${next.shape}]`)
        }
        for (let i = 0; i < rank; i++) {
          if (i !== axis && outShape[i] !== next.shape[i]) {
            throw new Error(`concat: shape mismatch at dim ${i}, [${outShape}] vs [${next.shape}]`)
          }
        }
        lazy = lazy.concat(next.lazy, axis)
        outShape = outShape.map((d, i) => (i === axis ? d + next.shape[i] : d))
      }
      return makeLazy(lazy, outShape, first.dtype, first.device)
    },
    catch: (error) =>
      new TensorError({ op: "concat", message: error instanceof Error ? error.message : String(error) })
  })

/**
 * Broadcasts a tensor to a larger shape. Every existing dimension must either
 * match the target or be `1`, and the target rank must be at least the current
 * rank.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const broadcastTo: {
  (shape: ReadonlyArray<number>): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, shape: ReadonlyArray<number>): Effect.Effect<LazyTensor, TensorError>
} = dual(
  2,
  (self: GenericTensor, target: ReadonlyArray<number>): Effect.Effect<LazyTensor, TensorError> =>
    Effect.try({
      try: () => {
        const validShape = validateShape("broadcastTo", target)
        if (validShape.length < self.shape.length) {
          throw new Error(`broadcastTo: cannot broadcast [${self.shape}] to lower rank [${validShape}]`)
        }
        for (let i = 0; i < self.shape.length; i++) {
          const d = self.shape[self.shape.length - 1 - i]
          const t = validShape[validShape.length - 1 - i]
          if (d !== t && d !== 1) {
            throw new Error(`broadcastTo: cannot broadcast [${self.shape}] to [${validShape}]`)
          }
        }
        return makeLazy(self.lazy.broadcastTo(validShape), validShape, self.dtype, self.device)
      },
      catch: (error) =>
        new TensorError({ op: "broadcastTo", message: error instanceof Error ? error.message : String(error) })
    })
)

/**
 * Flattens a contiguous range of dimensions into one. `startDim` and
 * `endDim` (inclusive) default to collapsing all dimensions into a vector.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const flatten = dualOptions(
  (
    self: GenericTensor,
    options: { readonly startDim?: number; readonly endDim?: number } = {}
  ): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      const rank = self.shape.length
      const start = options.startDim ?? 0
      const end = options.endDim ?? -1
      if (rank === 0) {
        if (start === 0 && (end === -1 || end === 0)) {
          return yield* reshape(self, [1])
        }
        return yield* new TensorError({
          op: "flatten",
          message: `flatten: dimension out of range for a rank-0 tensor`
        })
      }
      const s = normalizeDim("flatten", rank, start)
      const e = normalizeDim("flatten", rank, end)
      if (e < s) {
        return yield* new TensorError({ op: "flatten", message: `flatten: endDim ${end} precedes startDim ${start}` })
      }
      const collapsed = self.shape.slice(s, e + 1).reduce((a, b) => a * b, 1)
      return yield* reshape(self, [...self.shape.slice(0, s), collapsed, ...self.shape.slice(e + 1)])
    })
)

/**
 * Removes size-1 dimensions. Without `dims`, every size-1 dimension is
 * removed; with `dims`, only those (each must actually have size 1).
 *
 * @since 0.1.0
 * @category shape operations
 */
export const squeeze = dualOptions(
  (
    self: GenericTensor,
    options: { readonly dims?: ReadonlyArray<number> } = {}
  ): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      if (options.dims === undefined) {
        return yield* reshape(self, self.shape.filter((d) => d !== 1))
      }
      const normalized = normalizeDims("squeeze", self.shape.length, options.dims)
      for (const d of normalized) {
        if (self.shape[d] !== 1) {
          return yield* new TensorError({
            op: "squeeze",
            message: `squeeze: dimension ${d} has size ${self.shape[d]}, expected 1`
          })
        }
      }
      return yield* reshape(self, self.shape.filter((_, i) => !normalized.includes(i)))
    })
)

/**
 * Inserts a size-1 dimension at position `dim`.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const unsqueeze: {
  (dim: number): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, dim: number): Effect.Effect<LazyTensor, TensorError>
} = dual(2, (self: GenericTensor, dim: number): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    const rank = self.shape.length
    const d = dim < 0 ? dim + rank + 1 : dim
    if (!Number.isInteger(d) || d < 0 || d > rank) {
      return yield* new TensorError({
        op: "unsqueeze",
        message: `unsqueeze: dimension ${dim} out of range for rank ${rank}`
      })
    }
    const shape = [...self.shape]
    shape.splice(d, 0, 1)
    return yield* reshape(self, shape)
  })
)

/**
 * Stacks tensors along a new dimension inserted at `dim` (default `0`).
 * All tensors must have the same shape, dtype and device.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const stack = (
  tensors: readonly [GenericTensor, GenericTensor, ...ReadonlyArray<GenericTensor>],
  options: { readonly dim?: number } = {}
): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    const rank = tensors[0].shape.length
    const dim = options.dim ?? 0
    const d = dim < 0 ? dim + rank + 1 : dim
    if (!Number.isInteger(d) || d < 0 || d > rank) {
      return yield* new TensorError({
        op: "stack",
        message: `stack: dimension ${dim} out of range for rank ${rank}`
      })
    }
    const expanded: Array<LazyTensor> = []
    for (const t of tensors) {
      expanded.push(yield* unsqueeze(t, d))
    }
    return yield* concat(expanded as [GenericTensor, GenericTensor, ...Array<GenericTensor>], { dim: d })
  })

/**
 * Splits a tensor along `dim` into chunks. A number `sections` gives
 * equal-sized chunks (the last one smaller if the dimension does not divide
 * evenly); an array gives the exact size of each chunk and must sum to the
 * dimension size.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const split = (
  self: GenericTensor,
  sections: number | ReadonlyArray<number>,
  options: { readonly dim?: number } = {}
): Effect.Effect<Array<LazyTensor>, TensorError> =>
  Effect.gen(function* () {
    const dim = options.dim ?? 0
    const d = normalizeDim("split", self.shape.length, dim)
    const n = self.shape[d]
    let sizes: ReadonlyArray<number>
    if (typeof sections === "number") {
      if (!Number.isInteger(sections) || sections <= 0) {
        return yield* new TensorError({ op: "split", message: `split: section size must be positive, got ${sections}` })
      }
      sizes = Array.from({ length: Math.ceil(n / sections) }, (_, i) => Math.min(sections, n - i * sections))
    } else {
      if (sections.reduce((a, b) => a + b, 0) !== n) {
        return yield* new TensorError({
          op: "split",
          message: `split: section sizes sum to ${sections.reduce((a, b) => a + b, 0)}, expected ${n}`
        })
      }
      sizes = sections
    }
    const out: Array<LazyTensor> = []
    let offset = 0
    for (const size of sizes) {
      out.push(yield* slice(self, {
        start: self.shape.map((_, i) => (i === d ? offset : 0)),
        end: self.shape.map((extent, i) => (i === d ? offset + size : extent))
      }))
      offset += size
    }
    return out
  })

/**
 * Splits a tensor into at most `chunks` parts along `dim`, as evenly as
 * possible.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const chunk = (
  self: GenericTensor,
  chunks: number,
  options: { readonly dim?: number } = {}
): Effect.Effect<Array<LazyTensor>, TensorError> => {
  const dim = options.dim ?? 0
  const d = dim < 0 ? dim + self.shape.length : dim
  const n = Number.isInteger(d) && d >= 0 && d < self.shape.length ? self.shape[d] : 0
  const size = Math.ceil(n / Math.max(1, chunks))
  return split(self, Math.max(1, size), options)
}

/**
 * Tiles a tensor by repeating it `reps[i]` times along dimension `i`.
 * Extra leading entries in `reps` add new leading dimensions.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const tile = (
  self: GenericTensor,
  reps: ReadonlyArray<number>
): Effect.Effect<LazyTensor, TensorError> =>
  Effect.gen(function* () {
    for (const r of reps) {
      if (!Number.isInteger(r) || r < 1) {
        return yield* new TensorError({ op: "tile", message: `tile: reps must be positive integers, got [${reps}]` })
      }
    }
    let cur: GenericTensor = self
    if (reps.length > self.shape.length) {
      const extra = reps.length - self.shape.length
      cur = yield* reshape(cur, [...Array<number>(extra).fill(1), ...self.shape])
    }
    const rank = cur.shape.length
    const fullReps = reps.length < rank
      ? [...Array<number>(rank - reps.length).fill(1), ...reps]
      : reps
    for (let i = 0; i < rank; i++) {
      if (fullReps[i] === 1) continue
      const widened = yield* unsqueeze(cur, i)
      const broadcastShape = [...widened.shape]
      broadcastShape[i] = fullReps[i]
      const wide = yield* broadcastTo(widened, broadcastShape)
      const merged = [...wide.shape]
      merged[i] = wide.shape[i] * wide.shape[i + 1]
      merged.splice(i + 1, 1)
      cur = yield* reshape(wide, merged)
    }
    return cur as LazyTensor
  })

/**
 * Pads a tensor with zeros. `pads[i]` is `[before, after]` for dimension
 * `i`; omitted trailing dimensions are not padded.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const pad = (
  self: GenericTensor,
  pads: ReadonlyArray<readonly [before: number, after: number]>
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (pads.length > self.shape.length) {
      return yield* new TensorError({
        op: "pad",
        message: `pad: ${pads.length} pad specs for a rank-${self.shape.length} tensor`
      })
    }
    let cur: GenericTensor = self
    for (let d = 0; d < pads.length; d++) {
      const [before, after] = pads[d]
      if (before < 0 || after < 0) {
        return yield* new TensorError({ op: "pad", message: `pad: negative padding [${before}, ${after}]` })
      }
      if (before > 0) {
        const shape = [...cur.shape]
        shape[d] = before
        cur = yield* concat([yield* zeros(shape, { dtype: cur.dtype }), cur], { dim: d })
      }
      if (after > 0) {
        const shape = [...cur.shape]
        shape[d] = after
        cur = yield* concat([cur, yield* zeros(shape, { dtype: cur.dtype })], { dim: d })
      }
    }
    return cur as LazyTensor
  })

/**
 * Gathers rows (or slices along `dim`) by `i64` indexes: the inverse of
 * one-hot. `indexes` must be a 1-D `i64` tensor on the same device.
 * Not differentiable (scatter-add is not implemented yet), so use it for
 * embeddings only in evaluation graphs for now.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const take: {
  (
    indexes: GenericTensor,
    options?: { readonly dim?: number }
  ): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (
    self: GenericTensor,
    indexes: GenericTensor,
    options?: { readonly dim?: number }
  ): Effect.Effect<LazyTensor, TensorError>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: GenericTensor,
    indexes: GenericTensor,
    options: { readonly dim?: number } = {}
  ): Effect.Effect<LazyTensor, TensorError> =>
    Effect.try({
      try: () => {
        const d = normalizeDim("take", self.shape.length, options.dim ?? 0)
        if (indexes.dtype !== "i64") {
          throw new Error(`take: indexes must be i64, got ${indexes.dtype}`)
        }
        if (indexes.shape.length !== 1) {
          throw new Error(`take: indexes must be 1-D, got shape [${indexes.shape}]`)
        }
        if (indexes.device !== self.device) {
          throw new Error(`take: device mismatch, got ${indexes.device} and ${self.device}`)
        }
        const outShape = [...self.shape]
        outShape[d] = indexes.shape[0]
        return makeLazy(self.lazy.indexSelect(d, indexes.lazy), outShape, self.dtype, self.device)
      },
      catch: (error) =>
        new TensorError({ op: "take", message: error instanceof Error ? error.message : String(error) })
    })
)

/**
 * Expands `i64` class indexes of any shape into one-hot vectors of the
 * given depth, appended as a new last dimension.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const oneHot = (
  indexes: GenericTensor,
  depth: number,
  options: { readonly dtype?: "f32" | "f64" } = {}
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (indexes.dtype !== "i64") {
      return yield* new TensorError({ op: "oneHot", message: `oneHot: indexes must be i64, got ${indexes.dtype}` })
    }
    if (!Number.isInteger(depth) || depth < 1) {
      return yield* new TensorError({ op: "oneHot", message: `oneHot: depth must be a positive integer, got ${depth}` })
    }
    const classes = yield* arange(depth, undefined, { dtype: "i64" })
    const expanded = yield* reshape(indexes, [...indexes.shape, 1])
    return yield* cast(yield* eq(expanded, classes), options.dtype ?? "f32")
  })

const triangleMask = (
  op: string,
  self: GenericTensor,
  diagonal: number,
  keepUpper: boolean
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (self.shape.length < 2) {
      return yield* new TensorError({ op, message: `${op}: expected rank >= 2, got rank ${self.shape.length}` })
    }
    const m = self.shape[self.shape.length - 2]
    const n = self.shape[self.shape.length - 1]
    const rows = yield* reshape(yield* arange(m, undefined, { dtype: "i64" }), [m, 1])
    const cols = yield* reshape(yield* arange(n, undefined, { dtype: "i64" }), [1, n])
    const shifted = yield* add(rows, diagonal)
    const mask = keepUpper ? yield* ge(cols, shifted) : yield* le(cols, shifted)
    return yield* where(mask, self, 0)
  })

/**
 * Upper-triangular part of the last two dimensions, zeroing everything
 * below the `diagonal`-th diagonal (default `0`).
 *
 * @since 0.1.0
 * @category shape operations
 */
export const triu = dualOptions(
  (
    self: GenericTensor,
    options: { readonly diagonal?: number } = {}
  ): Effect.Effect<LazyTensor, TensorError, CurrentDevice> => triangleMask("triu", self, options.diagonal ?? 0, true)
)

/**
 * Lower-triangular part of the last two dimensions, zeroing everything
 * above the `diagonal`-th diagonal (default `0`).
 *
 * @since 0.1.0
 * @category shape operations
 */
export const tril = dualOptions(
  (
    self: GenericTensor,
    options: { readonly diagonal?: number } = {}
  ): Effect.Effect<LazyTensor, TensorError, CurrentDevice> => triangleMask("tril", self, options.diagonal ?? 0, false)
)

/**
 * Dot product of two rank-1 tensors (`sum(a * b)`), or matrix
 * multiplication when both are rank >= 2 (alias of {@link matmul}).
 *
 * @since 0.1.0
 * @category operations
 */
export const dot: {
  (other: GenericTensor): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, other: GenericTensor): Effect.Effect<LazyTensor, TensorError>
} = dual(
  2,
  (self: GenericTensor, other: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
    Effect.gen(function* () {
      if (self.shape.length === 1 && other.shape.length === 1) {
        return yield* sum(yield* mul(self, other))
      }
      return yield* matmul(self, other)
    })
)

/**
 * Sum of the diagonal of a square rank-2 tensor.
 *
 * @since 0.1.0
 * @category operations
 */
export const trace = (
  self: GenericTensor
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (self.shape.length !== 2 || self.shape[0] !== self.shape[1]) {
      return yield* new TensorError({
        op: "trace",
        message: `trace: expected a square rank-2 tensor, got shape [${self.shape}]`
      })
    }
    const id = yield* eye(self.shape[0], { dtype: self.dtype })
    return yield* sum(yield* mul(self, id))
  })

/**
 * Options for {@link conv2d} and {@link conv1d}. `groups` splits the
 * channel dimensions into that many independent convolutions (grouped
 * convolution; `groups = inChannels` is depthwise).
 *
 * @since 0.1.0
 * @category models
 */
export interface ConvOptions {
  readonly stride?: number
  readonly padding?: number
  readonly dilation?: number
  readonly groups?: number
}

const checkConvOptions = (
  op: string,
  self: GenericTensor,
  weight: GenericTensor,
  options: ConvOptions,
  rank: number
): Effect.Effect<
  { readonly stride: number; readonly padding: number; readonly dilation: number; readonly groups: number },
  TensorError
> =>
  Effect.gen(function* () {
    const stride = options.stride ?? 1
    const padding = options.padding ?? 0
    const dilation = options.dilation ?? 1
    const groups = options.groups ?? 1
    if (self.shape.length !== rank + 2 || weight.shape.length !== rank + 2) {
      return yield* new TensorError({
        op,
        message: `${op}: expected rank-${rank + 2} input and weight, got ranks ${self.shape.length} and ${weight.shape.length}`
      })
    }
    for (const [name, value, min] of [["stride", stride, 1], ["padding", padding, 0], [
      "dilation",
      dilation,
      1
    ], ["groups", groups, 1]] as const) {
      if (!Number.isInteger(value) || value < min) {
        return yield* new TensorError({ op, message: `${op}: ${name} must be an integer >= ${min}, got ${value}` })
      }
    }
    return { stride, padding, dilation, groups }
  })

const conv2dGroup = (
  op: string,
  x: GenericTensor,
  weight: GenericTensor,
  options: { readonly stride: number; readonly padding: number; readonly dilation: number },
  outChannels: number
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const { stride, padding, dilation } = options
    const [n, cIn, kh, kw] = [x.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]]
    const padded = padding > 0
      ? yield* pad(x, [[0, 0], [0, 0], [padding, padding], [padding, padding]])
      : x
    const oh = Math.floor((padded.shape[2] - dilation * (kh - 1) - 1) / stride) + 1
    const ow = Math.floor((padded.shape[3] - dilation * (kw - 1) - 1) / stride) + 1
    if (oh < 1 || ow < 1) {
      return yield* new TensorError({
        op,
        message: `${op}: kernel [${kh}, ${kw}] with dilation ${dilation} is larger than the padded input [${padded.shape[2]}, ${padded.shape[3]}]`
      })
    }
    const windows: Array<LazyTensor> = []
    for (let ky = 0; ky < kh; ky++) {
      for (let kx = 0; kx < kw; kx++) {
        windows.push(yield* slice(padded, {
          start: [0, 0, ky * dilation, kx * dilation],
          end: [
            padded.shape[0],
            cIn,
            ky * dilation + (oh - 1) * stride + 1,
            kx * dilation + (ow - 1) * stride + 1
          ],
          stride: [1, 1, stride, stride]
        }))
      }
    }
    // im2col: [kh*kw, n, c, oh, ow] -> [n, oh, ow, c*kh*kw] @ [c*kh*kw, cOut]
    const stacked = yield* stack(
      windows as unknown as [GenericTensor, GenericTensor, ...Array<GenericTensor>],
      { dim: 0 }
    )
    const cols = yield* reshape(yield* transpose(stacked, [1, 3, 4, 2, 0]), [
      n,
      oh,
      ow,
      cIn * kh * kw
    ])
    const wFlat = yield* transpose(yield* reshape(weight, [outChannels, cIn * kh * kw]), [1, 0])
    return yield* transpose(yield* matmul(cols, wFlat), [0, 3, 1, 2])
  })

/**
 * 2-D convolution via im2col: the input is unfolded into kernel windows and
 * contracted with the weight in a single matmul, expressed entirely in the
 * standard op vocabulary — so it runs on every backend and differentiates
 * through the ordinary adjoints. `self` is `[N, C_in, H, W]`, `weight` is
 * `[C_out, C_in/groups, KH, KW]`; a bias is added separately with `add`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const conv2d: {
  (
    weight: GenericTensor,
    options?: ConvOptions
  ): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError, CurrentDevice>
  (
    self: GenericTensor,
    weight: GenericTensor,
    options?: ConvOptions
  ): Effect.Effect<LazyTensor, TensorError, CurrentDevice>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: GenericTensor,
    weight: GenericTensor,
    options: ConvOptions = {}
  ): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
    Effect.gen(function* () {
      const opts = yield* checkConvOptions("conv2d", self, weight, options, 2)
      yield* Effect.try({
        try: () => checkCompatible("conv2d", self, weight),
        catch: (error) =>
          new TensorError({ op: "conv2d", message: error instanceof Error ? error.message : String(error) })
      })
      const cIn = self.shape[1]
      const [cOut, cPerGroup] = [weight.shape[0], weight.shape[1]]
      if (cIn % opts.groups !== 0 || cOut % opts.groups !== 0) {
        return yield* new TensorError({
          op: "conv2d",
          message: `conv2d: channels [${cIn}, ${cOut}] are not divisible into ${opts.groups} groups`
        })
      }
      if (cPerGroup !== cIn / opts.groups) {
        return yield* new TensorError({
          op: "conv2d",
          message: `conv2d: weight has ${cPerGroup} input channels per group, expected ${cIn / opts.groups}`
        })
      }
      if (opts.groups === 1) {
        return yield* conv2dGroup("conv2d", self, weight, opts, cOut)
      }
      const xs = yield* split(self, Array<number>(opts.groups).fill(cIn / opts.groups), { dim: 1 })
      const ws = yield* split(weight, Array<number>(opts.groups).fill(cOut / opts.groups), { dim: 0 })
      const outs: Array<LazyTensor> = []
      for (let i = 0; i < opts.groups; i++) {
        outs.push(yield* conv2dGroup("conv2d", xs[i], ws[i], opts, cOut / opts.groups))
      }
      return yield* concat(outs as [GenericTensor, GenericTensor, ...Array<GenericTensor>], { dim: 1 })
    })
)

/**
 * 1-D convolution over `[N, C_in, L]` with `weight` `[C_out, C_in/groups, K]`,
 * implemented as a rank-4 {@link conv2d}.
 *
 * @since 0.1.0
 * @category neural network
 */
export const conv1d: {
  (
    weight: GenericTensor,
    options?: ConvOptions
  ): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError, CurrentDevice>
  (
    self: GenericTensor,
    weight: GenericTensor,
    options?: ConvOptions
  ): Effect.Effect<LazyTensor, TensorError, CurrentDevice>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: GenericTensor,
    weight: GenericTensor,
    options: ConvOptions = {}
  ): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
    Effect.gen(function* () {
      if (self.shape.length !== 3 || weight.shape.length !== 3) {
        return yield* new TensorError({
          op: "conv1d",
          message: `conv1d: expected rank-3 input and weight, got ranks ${self.shape.length} and ${weight.shape.length}`
        })
      }
      const out = yield* conv2d(yield* unsqueeze(self, 2), yield* unsqueeze(weight, 2), options)
      return yield* squeeze(out, { dims: [2] })
    })
)

/**
 * Options for {@link maxPool2d} and {@link avgPool2d}. `kernelSize` is the
 * window `[KH, KW]` (a number for square windows); `stride` defaults to the
 * kernel size (non-overlapping windows).
 *
 * @since 0.1.0
 * @category models
 */
export interface PoolOptions {
  readonly kernelSize: number | readonly [number, number]
  readonly stride?: number | readonly [number, number]
  readonly padding?: number
}

const pool2d = (
  op: string,
  reduce: (t: GenericTensor) => Effect.Effect<LazyTensor, TensorError>,
  self: GenericTensor,
  options: PoolOptions
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (self.shape.length !== 4) {
      return yield* new TensorError({
        op,
        message: `${op}: expected a rank-4 [N, C, H, W] input, got rank ${self.shape.length}`
      })
    }
    const [kh, kw] = typeof options.kernelSize === "number"
      ? [options.kernelSize, options.kernelSize]
      : options.kernelSize
    const [sy, sx] = options.stride === undefined
      ? [kh, kw]
      : typeof options.stride === "number"
      ? [options.stride, options.stride]
      : options.stride
    const padding = options.padding ?? 0
    if (kh < 1 || kw < 1 || sy < 1 || sx < 1 || padding < 0) {
      return yield* new TensorError({
        op,
        message: `${op}: invalid kernel [${kh}, ${kw}] / stride [${sy}, ${sx}] / padding ${padding}`
      })
    }
    const padded = padding > 0
      ? yield* pad(self, [[0, 0], [0, 0], [padding, padding], [padding, padding]])
      : self
    const oh = Math.floor((padded.shape[2] - kh) / sy) + 1
    const ow = Math.floor((padded.shape[3] - kw) / sx) + 1
    if (oh < 1 || ow < 1) {
      return yield* new TensorError({
        op,
        message: `${op}: kernel [${kh}, ${kw}] is larger than the padded input [${padded.shape[2]}, ${padded.shape[3]}]`
      })
    }
    const windows: Array<LazyTensor> = []
    for (let ky = 0; ky < kh; ky++) {
      for (let kx = 0; kx < kw; kx++) {
        windows.push(yield* slice(padded, {
          start: [0, 0, ky, kx],
          end: [padded.shape[0], padded.shape[1], ky + (oh - 1) * sy + 1, kx + (ow - 1) * sx + 1],
          stride: [1, 1, sy, sx]
        }))
      }
    }
    const stacked = yield* stack(
      windows as unknown as [GenericTensor, GenericTensor, ...Array<GenericTensor>],
      { dim: 0 }
    )
    return yield* reduce(stacked)
  })

/**
 * 2-D max pooling over `[N, C, H, W]`, composed from window slices —
 * gradients route to the maximal element of each window.
 *
 * @since 0.1.0
 * @category neural network
 */
export const maxPool2d = (
  self: GenericTensor,
  options: PoolOptions
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  pool2d("maxPool2d", (t) => max(t, { dims: [0] }), self, options)

/**
 * 2-D average pooling over `[N, C, H, W]`, composed from window slices.
 *
 * @since 0.1.0
 * @category neural network
 */
export const avgPool2d = (
  self: GenericTensor,
  options: PoolOptions
): Effect.Effect<LazyTensor, TensorError, CurrentDevice> =>
  pool2d("avgPool2d", (t) => mean(t, { dims: [0] }), self, options)

/**
 * Converts a tensor to a different dtype. Dtypes are strict in this library:
 * no implicit promotion happens anywhere, so `cast` is the only way to mix
 * dtypes.
 *
 * @since 0.1.0
 * @category operations
 */
export const cast: {
  (dtype: DType): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, dtype: DType): Effect.Effect<LazyTensor, TensorError>
} = dual(2, (self: GenericTensor, dtype: DType): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => makeLazy(self.lazy.cast(dtype as NativeDType), self.shape, dtype, self.device),
    catch: (error) =>
      new TensorError({ op: "cast", message: error instanceof Error ? error.message : String(error) })
  })
)

/**
 * Error raised by {@link grad} when the graph violates the autodiff
 * contract.
 *
 * @since 0.1.0
 * @category errors
 */
export class GradError extends Data.TaggedError("GradError")<{
  readonly reason: "non-scalar-output" | "non-float-dtype" | "not-differentiable"
  readonly detail: string
}> {}

const isFloatDtype = (dtype: string): boolean => dtype === "f32" || dtype === "f64"

const toGradError = (error: unknown): GradError => {
  const detail = error instanceof Error ? error.message : String(error)
  return new GradError({
    // the scalar and float-dtype contracts are validated above, so a native
    // error here means the graph contains a non-differentiable construct
    reason: "not-differentiable",
    detail
  })
}

/**
 * Computes the gradients of a scalar loss with respect to the given tensors.
 * The loss is an ordinary lazy graph value — there is no tracing and no
 * function transformation, the backward transform runs natively on the
 * graph itself: one walk, with adjoints expressed in the same node
 * vocabulary as the forward pass, so higher-order derivatives work by
 * applying `grad` again.
 *
 * Gradients are lazy tensors sharing the forward graph; a `wrt` tensor that
 * does not influence the loss yields a zero gradient. Because the loss and
 * its gradients share the forward graph, evaluate them together with
 * {@link evaluate}: evaluating them separately recomputes the forward
 * pass and, if the graph contains `randn`, produces values from different
 * random draws.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const grad = (
  loss: GenericTensor,
  wrt: ReadonlyArray<GenericTensor>
): Effect.Effect<Array<LazyTensor>, GradError> =>
  Effect.gen(function* () {
    if (loss.shape.length !== 0) {
      return yield* new GradError({
        reason: "non-scalar-output",
        detail: `grad: expected a scalar (0-d) loss, got shape [${loss.shape}], reduce it first (e.g. with sum or mean)`
      })
    }
    if (!isFloatDtype(loss.dtype)) {
      return yield* new GradError({
        reason: "non-float-dtype",
        detail: `grad: loss dtype must be f32 or f64, got ${loss.dtype}`
      })
    }
    for (const target of wrt) {
      if (!isFloatDtype(target.dtype)) {
        return yield* new GradError({
          reason: "non-float-dtype",
          detail: `grad: cannot differentiate with respect to ${target.dtype} tensor, only f32 and f64 are differentiable`
        })
      }
    }
    const grads = yield* Effect.try({
      try: () => native.grad(loss.lazy, wrt.map((target) => target.lazy)),
      catch: toGradError
    })
    return grads.map((handle, i) => makeLazy(handle, wrt[i].shape, wrt[i].dtype, wrt[i].device))
  })

/**
 * Stops gradient flow: the returned tensor has the same value as the input,
 * but the backward walk does not continue past it, so ancestors of the input
 * receive no gradient through this path.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const stopGradient = (self: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => makeLazy(self.lazy.stopGradient(), self.shape, self.dtype, self.device),
    catch: (error) =>
      new TensorError({
        op: "stopGradient",
        message: error instanceof Error ? error.message : String(error)
      })
  })

type CancellationTokenType = InstanceType<typeof CancellationToken>

const isCancelled = (token: CancellationTokenType, error: unknown): boolean =>
  token.cancelled || (error instanceof Error && error.message.includes("aborted"))

const fromNative = <A>(
  op: string,
  register: (token: CancellationTokenType) => Promise<A>
): Effect.Effect<A, TensorError> =>
  Effect.callback<A, TensorError>((resume, signal) => {
    const token = new CancellationToken()
    if (signal.aborted) token.cancel()
    else signal.addEventListener("abort", () => token.cancel(), { once: true })
    register(token).then(
      (value) => resume(Effect.succeed(value)),
      (error) =>
        resume(
          isCancelled(token, error)
            ? Effect.interrupt
            : Effect.fail(
                new TensorError({ op, message: error instanceof Error ? error.message : String(error) })
              )
        )
    )
  })

/**
 * Evaluates one or more lazy tensors in a single graph walk, running the
 * computation off the JavaScript thread, and returns the materialized
 * tensors in the same order. All roots share one deduplication cache:
 * subgraphs shared between the roots are computed only once, and `randn`
 * nodes produce a single set of draws across all roots. This matters for
 * gradients: the loss and its gradients share the forward graph, so they
 * must be evaluated together to be consistent. Interrupting the fiber
 * aborts the native evaluation. Already materialized roots are returned
 * as-is.
 *
 * @since 0.1.0
 * @category destructors
 */
export const evaluate = (
  roots: ReadonlyArray<GenericTensor>
): Effect.Effect<Array<Tensor>, TensorError> =>
  roots.every(isTensor)
    ? Effect.succeed(roots as Array<Tensor>)
    : Effect.map(
        fromNative("evaluate", (token) => evalLazy(roots.map((root) => root.lazy), token)),
        (handles) => {
          reportExternalMemory(handles.reduce((total, handle) => total + handle.bytes, 0))
          return handles.map(fromHandle)
        }
      )

const typedArrayConstructor = (dtype: DType) => {
  switch (dtype) {
    case "f32":
      return Float32Array
    case "f64":
      return Float64Array
    case "i64":
      return BigInt64Array
    case "u8":
      return Uint8Array
    case "u32":
      return Uint32Array
  }
}

/**
 * Evaluates a tensor and reads its values back into a typed array matching
 * the tensor's dtype. Data is exported without copying when the device buffer
 * allows it.
 *
 * @since 0.1.0
 * @category destructors
 */
export const toTypedArray = (self: GenericTensor): Effect.Effect<TypedArray, TensorError> =>
  Effect.flatMap(evaluate([self]), ([evaluated]) =>
    fromNative<TypedArray>("toTypedArray", (token) =>
      evaluated.materialized.readback(token).then((buffer) => {
        const Ctor = typedArrayConstructor(evaluated.dtype)
        return new Ctor(buffer)
      })
    )
  )

/**
 * Evaluates a tensor and reads its values back as a plain JavaScript number
 * array. Fails with a `TensorError` for `i64` tensors, whose values may not
 * be representable as numbers — use {@link toTypedArray} there and handle
 * bigints explicitly.
 *
 * @since 0.1.0
 * @category destructors
 */
export const toNumberArray = (self: GenericTensor): Effect.Effect<Array<number>, TensorError> =>
  self.dtype === "i64"
    ? new TensorError({
        op: "toNumberArray",
        message: "toNumberArray: i64 tensors may contain values not representable as numbers"
      })
    : Effect.map(toTypedArray(self), (arr) =>
        Array.from(arr as Float32Array | Float64Array | Uint8Array | Uint32Array)
      )

/**
 * Explicitly releases the native buffer of a materialized tensor instead of
 * waiting for GC. In workloads that replace device-resident values every
 * iteration (like training loops replacing parameters and optimizer state),
 * prompt release lets the backend allocator reuse the buffers immediately —
 * on backends with a scanning allocator (Metal) this keeps per-iteration
 * cost flat. Using the tensor afterwards fails at evaluation time.
 *
 * @since 0.1.0
 * @category destructors
 */
export const dispose = (self: Tensor): Effect.Effect<void, TensorError> =>
  Effect.try({
    try: () => {
      self.materialized.dispose()
      self.lazy.dispose()
    },
    catch: (error) =>
      new TensorError({
        op: "dispose",
        message: error instanceof Error ? error.message : String(error)
      })
  })

/**
 * Saves tensors to a safetensors file. The tensors are evaluated and
 * serialized entirely on the native side — all entries share a single graph
 * walk (shared subgraphs are computed once, `randn` draws are consistent
 * across entries) and tensor data never crosses the JavaScript thread.
 * Interrupting the fiber aborts the native work.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = (
  path: string,
  tensors: Readonly<Record<string, GenericTensor>>
): Effect.Effect<void, TensorError> => {
  const entries = Object.entries(tensors)
  return fromNative<void>("save", (token) =>
    saveTensors(
      path,
      entries.map(([name]) => name),
      entries.map(([, tensor]) => tensor.lazy),
      token
    )
  )
}

/**
 * Loads a safetensors file straight into materialized tensors on the
 * current device; the file is read and deserialized entirely on the native
 * side, so tensor data never crosses the JavaScript thread. Interrupting
 * the fiber aborts the native work.
 *
 * @since 0.1.0
 * @category constructors
 */
export const load = (
  path: string
): Effect.Effect<Record<string, Tensor>, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    const [names, handles] = yield* fromNative("load", (token) => loadTensors(path, device, token))
    reportExternalMemory(handles.reduce((total, handle) => total + handle.bytes, 0))
    return Object.fromEntries(names.map((name, i) => [name, fromHandle(handles[i])]))
  })

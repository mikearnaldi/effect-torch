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
export type DType = "f32" | "f64" | "f16" | "i64" | "u8" | "u32"

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
 * Common supertype of {@link Lazy} and {@link Concrete}. Every operation
 * accepts this type, so lazy and evaluated tensors can be mixed freely.
 *
 * @since 0.1.0
 * @category models
 */
export interface Any extends Pipeable {
  readonly [TensorTypeId]: TensorTypeId
  readonly _tag: "LazyTensor" | "Tensor"
  /**
   * The native computation-graph handle. Internal to the library: the
   * Gradient and Optimizer modules call native methods on it and wrap
   * the result with {@link makeLazy}; the handle's methods are the
   * native op set, which userland cannot extend.
   *
   * @internal
   */
  readonly lazy: NativeLazyTensorType
  readonly shape: ReadonlyArray<number>
  readonly dtype: DType
  readonly device: DeviceKind
}

/**
 * A tensor described by a lazy computation graph. Operations on lazy tensors
 * only extend the graph; nothing is computed until {@link compute} is called.
 *
 * @since 0.1.0
 * @category models
 */
export interface Lazy extends Any {
  readonly _tag: "LazyTensor"
}

/**
 * A materialized tensor whose data resides on the device, obtained through
 * {@link compute}.
 *
 * @since 0.1.0
 * @category models
 */
export interface Concrete extends Any {
  readonly _tag: "Tensor"
  /**
   * The native device-buffer handle.
   *
   * @internal
   */
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
  pipe(this: Any) {
    return pipeArguments(this, arguments)
  }
}

/**
 * Wraps a native graph handle as a {@link Lazy}. The native graph does
 * not track shapes, so the caller owns the `shape`, `dtype` and `device`
 * metadata: it must exactly describe the handle's result, or every
 * downstream operation reads wrong shapes. Internal to the library —
 * used by the Gradient and Optimizer modules to wrap native adjoints
 * and fused update nodes.
 *
 * @since 0.1.0
 * @category constructors
 * @internal
 */
export const makeLazy = (
  lazy: NativeLazyTensorType,
  shape: ReadonlyArray<number>,
  dtype: DType,
  device: DeviceKind
): Lazy => {
  const self = Object.create(TensorProto)
  self._tag = "LazyTensor"
  self.lazy = lazy
  self.shape = shape
  self.dtype = dtype
  self.device = device
  return self
}

const fromHandle = (handle: NativeTensorType): Concrete => {
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
export const isLazyTensor = (self: Any): self is Lazy => self._tag === "LazyTensor"

/**
 * Returns `true` if the tensor has been materialized on the device.
 *
 * @since 0.1.0
 * @category refinements
 */
export const isTensor = (self: Any): self is Concrete => self._tag === "Tensor"

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

const checkCompatible = (op: string, a: Any, b: Any): void => {
  if (a.dtype !== b.dtype) {
    throw new Error(`${op}: dtype mismatch, got ${a.dtype} and ${b.dtype}, use cast for explicit conversion`)
  }
  if (a.device !== b.device) {
    throw new Error(`${op}: device mismatch, got ${a.device} and ${b.device}`)
  }
}

const numel = (shape: ReadonlyArray<number>): number => shape.reduce((a, b) => a * b, 1)

const isFloatDtype = (dtype: string): boolean => dtype === "f32" || dtype === "f64"

/**
 * Creates a shared 0-d constant tensor. The native runtime pools constant
 * leaves, so repeated calls with the same (value, dtype, device) triple
 * are backed by the same graph node — hot constants cost one native node
 * total instead of one per use. Use it for constants referenced many
 * times (a learning rate lifted per step, a scale applied in a loop); use
 * {@link full} for a fresh node or a non-scalar shape. A constant is
 * exactly that: never route a value that changes meaning per step through
 * one.
 *
 * @since 0.1.0
 * @category constructors
 */
export const constant = (
  value: number,
  options: TensorOptions = {}
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    const dtype = options.dtype ?? "f32"
    return yield* Effect.try({
      try: () => makeLazy(NativeLazyTensor.constant(value, dtype as NativeDType, device), [], dtype, device),
      catch: (error) =>
        new TensorError({ op: "constant", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Creates a shared 0-d constant tensor with the same dtype and device as
 * `self` — the scalar counterpart of {@link zerosLike} / {@link onesLike}
 * / {@link fullLike}, and the way to lift a numeric constant next to an
 * existing tensor (custom losses, optimizer updates) without threading a
 * device through the environment. The native runtime pools constant
 * leaves (see {@link constant}). A constant is exactly that: never route
 * a value that changes meaning per step through one.
 *
 * @since 0.1.0
 * @category constructors
 */
export const constantLike = (self: Any, value: number): Effect.Effect<Lazy, TensorError> =>
  Effect.try({
    try: () =>
      makeLazy(
        NativeLazyTensor.constant(value, self.dtype as NativeDType, self.device),
        [],
        self.dtype,
        self.device
      ),
    catch: (error) =>
      new TensorError({ op: "constantLike", message: error instanceof Error ? error.message : String(error) })
  })

const binaryOp = (
  op: string,
  native: (a: NativeLazyTensorType, b: NativeLazyTensorType) => NativeLazyTensorType,
  outDtype: (dtype: DType) => DType = (dtype) => dtype
): {
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError>
} =>
  dual(
    2,
    (self: Any, other: Any): Effect.Effect<Lazy, TensorError> =>
      Effect.try({
        try: () => {
          checkCompatible(op, self, other)
          return makeLazy(
            native(self.lazy, other.lazy),
            broadcastShapes(op, self.shape, other.shape),
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
): (self: Any) => Effect.Effect<Lazy, TensorError> =>
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
  impl: (self: Any, options: O | undefined) => Effect.Effect<Lazy, TensorError, R>
): {
  (options?: O): (self: Any) => Effect.Effect<Lazy, TensorError, R>
  (self: Any, options?: O): Effect.Effect<Lazy, TensorError, R>
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
    return yield* add(
      yield* mul(base, yield* constantLike(base, (end - start) / (steps - 1))),
      yield* constantLike(base, start)
    )
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
export const zerosLike = (self: Any): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  zeros(self.shape, { dtype: self.dtype })

/**
 * Creates a lazy tensor of ones with the same shape and dtype as the input.
 *
 * @since 0.1.0
 * @category constructors
 */
export const onesLike = (self: Any): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  ones(self.shape, { dtype: self.dtype })

/**
 * Creates a lazy tensor filled with `value`, with the same shape and dtype
 * as the input.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fullLike = (
  self: Any,
  value: number
): Effect.Effect<Lazy, TensorError, CurrentDevice> => full(self.shape, value, { dtype: self.dtype })

/**
 * Returns the shape of a tensor.
 *
 * @since 0.1.0
 * @category getters
 */
export const shape = (self: Any): ReadonlyArray<number> => self.shape

/**
 * Returns the dtype of a tensor.
 *
 * @since 0.1.0
 * @category getters
 */
export const dtype = (self: Any): DType => self.dtype

/**
 * Returns the device a tensor lives on.
 *
 * @since 0.1.0
 * @category getters
 */
export const device = (self: Any): DeviceKind => self.device

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
export const square = (self: Any): Effect.Effect<Lazy, TensorError> => mul(self, self)

/**
 * Elementwise reciprocal square root, `x ** -0.5`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const rsqrt = (self: Any): Effect.Effect<Lazy, TensorError> => pow(self, -0.5)

/**
 * Elementwise reciprocal, `1 / x`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const reciprocal = (self: Any): Effect.Effect<Lazy, TensorError> => pow(self, -1)

/**
 * Elementwise `exp(x) - 1`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const expm1 = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const e = yield* exp(self)
    return yield* sub(e, yield* constantLike(e, 1))
  })

/**
 * Elementwise `log(1 + x)`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log1p = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const t = yield* add(self, yield* constantLike(self, 1))
    return yield* log(t)
  })

/**
 * Elementwise base-2 logarithm.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log2 = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const t = yield* log(self)
    return yield* div(t, yield* constantLike(t, Math.LN2))
  })

/**
 * Elementwise base-10 logarithm.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log10 = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const t = yield* log(self)
    return yield* div(t, yield* constantLike(t, Math.LN10))
  })

/**
 * Elementwise hyperbolic sine, `(exp(x) - exp(-x)) / 2`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sinh = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const e = yield* exp(self)
    const ne = yield* exp(yield* neg(self))
    return yield* div(yield* sub(e, ne), yield* constantLike(e, 2))
  })

/**
 * Elementwise hyperbolic cosine, `(exp(x) + exp(-x)) / 2`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const cosh = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const e = yield* exp(self)
    const ne = yield* exp(yield* neg(self))
    return yield* div(yield* add(e, ne), yield* constantLike(e, 2))
  })

/**
 * Elementwise tangent, `sin(x) / cos(x)`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const tan = (self: Any): Effect.Effect<Lazy, TensorError> =>
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
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError>
} = dual(
  2,
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError> =>
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
export const logicalNot = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.flatMap(constantLike(self, 0), (zero) => eq(self, zero))

/**
 * Elementwise remainder of the division `self / other`, following the sign
 * of the divisor (Python/PyTorch semantics): `self - floor(self / other) * other`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const remainder: {
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError>
} = dual(
  2,
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError> =>
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
  (a: Any, b: Any): (cond: Any) => Effect.Effect<Lazy, TensorError>
  (cond: Any, a: Any, b: Any): Effect.Effect<Lazy, TensorError>
} = dual(
  3,
  (cond: Any, a: Any, b: Any): Effect.Effect<Lazy, TensorError> =>
    Effect.try({
      try: () => {
        if (cond.dtype !== "u8") {
          throw new Error(`where: condition must be u8, got ${cond.dtype}`)
        }
        checkCompatible("where", a, b)
        if (cond.device !== a.device) {
          throw new Error(`where: device mismatch, got ${cond.device} and ${a.device}`)
        }
        const shape = broadcastShapes(
          "where",
          broadcastShapes("where", cond.shape, a.shape),
          b.shape
        )
        return makeLazy(cond.lazy.whereCond(a.lazy, b.lazy), shape, a.dtype, a.device)
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
export const sigmoid = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const half = yield* constantLike(self, 2)
    const t = yield* tanh(yield* div(self, half))
    return yield* add(yield* div(t, half), yield* constantLike(self, 0.5))
  })

/**
 * Softmax over the given dimensions (the last one by default), computed
 * with max-subtraction for numerical stability.
 *
 * @since 0.1.0
 * @category neural network
 */
export const softmax = dualOptions(
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError> =>
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
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError> =>
    Effect.gen(function* () {
      const dims = normalizeDims("logSoftmax", self.shape.length, options.dims ?? [self.shape.length - 1])
      const m = yield* max(self, { dims, keepdims: true })
      const shifted = yield* sub(self, m)
      const s = yield* sum(yield* exp(shifted), { dims, keepdims: true })
      return yield* sub(shifted, yield* log(s))
    })
)

/**
 * Options for {@link scaledDotProductAttention}.
 *
 * @since 0.1.0
 * @category neural network
 */
export interface ScaledDotProductAttentionOptions {
  /** Score multiplier; defaults to `1 / sqrt(headDim)`. */
  readonly scale?: number
  /** Mask the scores causally: query `i` attends to keys `j <= i` (right-aligned when the key sequence is longer than the query sequence). */
  readonly causal?: boolean
}

/**
 * Scaled dot-product attention `softmax(q·kᵀ · scale) · v` as a single
 * semantic operation: `q` is `[..., T, D]`, `k` is `[..., S, D]` and `v`
 * is `[..., S, Dv]` with equal leading (batch/head) dimensions, giving an
 * output of `[..., T, Dv]`. The backward is closed-form and recomputes
 * the attention probabilities instead of retaining them; it is not
 * second-order differentiable.
 *
 * @since 0.1.0
 * @category neural network
 */
export const scaledDotProductAttention = (
  q: Any,
  k: Any,
  v: Any,
  options: ScaledDotProductAttentionOptions = {}
): Effect.Effect<Lazy, TensorError> =>
  Effect.try({
    try: () => {
      const op = "scaledDotProductAttention"
      const rank = q.shape.length
      if (rank < 2 || k.shape.length !== rank || v.shape.length !== rank) {
        throw new Error(
          `${op}: q, k and v must share a rank >= 2, got [${q.shape}], [${k.shape}] and [${v.shape}]`
        )
      }
      const leading = q.shape.slice(0, -2)
      if (
        !leading.every((d, i) => d === k.shape[i]) ||
        !leading.every((d, i) => d === v.shape[i])
      ) {
        throw new Error(
          `${op}: leading dims must match, got [${q.shape}], [${k.shape}] and [${v.shape}]`
        )
      }
      if (q.shape[rank - 1] !== k.shape[rank - 1]) {
        throw new Error(`${op}: q and k head dims mismatch, got [${q.shape}] and [${k.shape}]`)
      }
      if (k.shape[rank - 2] !== v.shape[rank - 2]) {
        throw new Error(`${op}: k and v sequence lengths mismatch, got [${k.shape}] and [${v.shape}]`)
      }
      if (q.dtype !== "f32" && q.dtype !== "f64") {
        throw new Error(`${op}: dtype must be f32 or f64, got ${q.dtype}`)
      }
      if (k.dtype !== q.dtype || v.dtype !== q.dtype) {
        throw new Error(`${op}: q, k and v must share a dtype, got ${q.dtype}, ${k.dtype} and ${v.dtype}`)
      }
      if (k.device !== q.device || v.device !== q.device) {
        throw new Error(`${op}: q, k and v must be on the same device`)
      }
      const scale = options.scale ?? 1 / Math.sqrt(q.shape[rank - 1])
      return makeLazy(
        q.lazy.scaledDotProductAttention(k.lazy, v.lazy, scale, options.causal ?? false),
        [...q.shape.slice(0, -1), v.shape[rank - 1]],
        q.dtype,
        q.device
      )
    },
    catch: (error) =>
      new TensorError({
        op: "scaledDotProductAttention",
        message: error instanceof Error ? error.message : String(error)
      })
  })

/**
 * SiLU / swish activation, `x * sigmoid(x)`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const silu = (self: Any): Effect.Effect<Lazy, TensorError> =>
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
export const softplus = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const head = yield* maximum(self, yield* constantLike(self, 0))
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
  (self: Any, options: EluOptions = {}): Effect.Effect<Lazy, TensorError> =>
    Effect.gen(function* () {
      const alpha = options.alpha ?? 1
      const negative = yield* mul(yield* expm1(self), yield* constantLike(self, alpha))
      return yield* where(yield* gt(self, yield* constantLike(self, 0)), self, negative)
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
  (self: Any, options: LeakyReluOptions = {}): Effect.Effect<Lazy, TensorError> =>
    Effect.gen(function* () {
      return yield* maximum(self, yield* mul(self, yield* constantLike(self, options.negativeSlope ?? 0.01)))
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
  (self: Any, options: GeluOptions = {}): Effect.Effect<Lazy, TensorError> =>
    Effect.gen(function* () {
      if (options.approximate === "tanh") {
        const c = Math.sqrt(2 / Math.PI)
        const inner = yield* mul(
          yield* add(self, yield* mul(yield* pow(self, 3), yield* constantLike(self, 0.044715))),
          yield* constantLike(self, c)
        )
        return yield* mul(
          yield* mul(self, yield* constantLike(self, 0.5)),
          yield* add(yield* tanh(inner), yield* constantLike(self, 1))
        )
      }
      return yield* mul(
        yield* mul(self, yield* constantLike(self, 0.5)),
        yield* add(yield* erf(yield* div(self, yield* constantLike(self, Math.SQRT2))), yield* constantLike(self, 1))
      )
    })
)

/**
 * Mish activation, `x * tanh(softplus(x))`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const mish = (self: Any): Effect.Effect<Lazy, TensorError> =>
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
  (self: Any, options: ClampOptions = {}): Effect.Effect<Lazy, TensorError> =>
    Effect.gen(function* () {
      if (options.min === undefined && options.max === undefined) {
        return yield* new TensorError({ op: "clamp", message: "clamp: at least one of min and max is required" })
      }
      let out: Any = self
      if (options.min !== undefined) out = yield* maximum(out, yield* constantLike(self, options.min))
      if (options.max !== undefined) out = yield* minimum(out, yield* constantLike(self, options.max))
      return out as Lazy
    })
)

/**
 * Hardtanh activation: clamp to `[-1, 1]`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const hardtanh = (self: Any): Effect.Effect<Lazy, TensorError> =>
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
    self: Any,
    options: DropoutOptions = {}
  ): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
        return yield* add(self, yield* constantLike(self, 0))
      }
      const mask = yield* ge(
        yield* uniform(self.shape, { dtype: self.dtype === "f64" ? "f64" : "f32" }),
        yield* constantLike(self, p)
      )
      return yield* where(
        mask,
        yield* div(self, yield* constantLike(self, 1 - p)),
        yield* constantLike(self, 0)
      )
    })
)

/**
 * Elementwise exponentiation to a constant power.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const pow: {
  (exponent: number): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, exponent: number): Effect.Effect<Lazy, TensorError>
} = dual(2, (self: Any, exponent: number): Effect.Effect<Lazy, TensorError> =>
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
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError>
} = dual(2, (self: Any, other: Any): Effect.Effect<Lazy, TensorError> =>
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
  (options?: ReduceOptions): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, options?: ReduceOptions): Effect.Effect<Lazy, TensorError>
} =>
  dual(
    (args) => args.length === 2 || (args.length === 1 && args[0] !== undefined && TensorTypeId in (args[0] as object)),
    (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError> =>
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
  (dim: number): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError>
} = dual(2, (self: Any, dim: number): Effect.Effect<Lazy, TensorError> =>
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
  (dim: number): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError>
} = dual(2, (self: Any, dim: number): Effect.Effect<Lazy, TensorError> =>
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
  (dim: number): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError>
} = dual(2, (self: Any, dim: number): Effect.Effect<Lazy, TensorError> =>
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
  (self: Any, options: VarianceOptions = {}): Effect.Effect<Lazy, TensorError> =>
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
      return yield* div(ss, yield* constantLike(ss, count - correction))
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
  (self: Any, options: VarianceOptions = {}): Effect.Effect<Lazy, TensorError> =>
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
  (self: Any, options: NormOptions = {}): Effect.Effect<Lazy, TensorError> =>
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
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError> =>
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
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError> =>
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
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError> =>
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
 * by default). The product of an empty set of elements is `1`. The
 * gradient is computed as `g * prod / x`, so it is undefined when any
 * factor is `0`.
 *
 * @since 0.1.0
 * @category reductions
 */
export const prod = reduceOp("prod", (a, dims, keepdims) => a.prod(dims, keepdims))

/**
 * Reshapes a tensor. The total number of elements must stay the same.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const reshape: {
  (shape: ReadonlyArray<number>): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, shape: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError>
} = dual(
  2,
  (self: Any, newShape: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError> =>
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
  (dims: ReadonlyArray<number>): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, dims: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError>
} = dual(
  2,
  (self: Any, dims: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError> =>
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
  (options: SliceOptions): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, options: SliceOptions): Effect.Effect<Lazy, TensorError>
} = dual(
  2,
  (self: Any, options: SliceOptions): Effect.Effect<Lazy, TensorError> =>
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
  tensors: readonly [Any, Any, ...ReadonlyArray<Any>],
  options: { readonly dim?: number } = {}
): Effect.Effect<Lazy, TensorError> =>
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
  (shape: ReadonlyArray<number>): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, shape: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError>
} = dual(
  2,
  (self: Any, target: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError> =>
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
    self: Any,
    options: { readonly startDim?: number; readonly endDim?: number } = {}
  ): Effect.Effect<Lazy, TensorError> =>
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
    self: Any,
    options: { readonly dims?: ReadonlyArray<number> } = {}
  ): Effect.Effect<Lazy, TensorError> =>
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
  (dim: number): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError>
} = dual(2, (self: Any, dim: number): Effect.Effect<Lazy, TensorError> =>
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
  tensors: readonly [Any, Any, ...ReadonlyArray<Any>],
  options: { readonly dim?: number } = {}
): Effect.Effect<Lazy, TensorError> =>
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
    const expanded: Array<Lazy> = []
    for (const t of tensors) {
      expanded.push(yield* unsqueeze(t, d))
    }
    return yield* concat(expanded as [Any, Any, ...Array<Any>], { dim: d })
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
  self: Any,
  sections: number | ReadonlyArray<number>,
  options: { readonly dim?: number } = {}
): Effect.Effect<Array<Lazy>, TensorError> =>
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
    const out: Array<Lazy> = []
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
  self: Any,
  chunks: number,
  options: { readonly dim?: number } = {}
): Effect.Effect<Array<Lazy>, TensorError> => {
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
  self: Any,
  reps: ReadonlyArray<number>
): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    for (const r of reps) {
      if (!Number.isInteger(r) || r < 1) {
        return yield* new TensorError({ op: "tile", message: `tile: reps must be positive integers, got [${reps}]` })
      }
    }
    let cur: Any = self
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
    return cur as Lazy
  })

/**
 * Pads a tensor with zeros. `pads[i]` is `[before, after]` for dimension
 * `i`; omitted trailing dimensions are not padded.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const pad = (
  self: Any,
  pads: ReadonlyArray<readonly [before: number, after: number]>
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (pads.length > self.shape.length) {
      return yield* new TensorError({
        op: "pad",
        message: `pad: ${pads.length} pad specs for a rank-${self.shape.length} tensor`
      })
    }
    let cur: Any = self
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
    return cur as Lazy
  })

/**
 * Gathers rows (or slices along `dim`) by integer indexes: the inverse of
 * one-hot. `indexes` must be a 1-D `i64` or `u32` tensor on the same device.
 * Differentiable: gradients scatter-add back into the input positions.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const take: {
  (
    indexes: Any,
    options?: { readonly dim?: number }
  ): (self: Any) => Effect.Effect<Lazy, TensorError>
  (
    self: Any,
    indexes: Any,
    options?: { readonly dim?: number }
  ): Effect.Effect<Lazy, TensorError>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: Any,
    indexes: Any,
    options: { readonly dim?: number } = {}
  ): Effect.Effect<Lazy, TensorError> =>
    Effect.try({
      try: () => {
        const d = normalizeDim("take", self.shape.length, options.dim ?? 0)
        if (indexes.dtype !== "i64" && indexes.dtype !== "u32") {
          throw new Error(`take: indexes must be i64 or u32, got ${indexes.dtype}`)
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
 * Gathers elements along `dim` at the given integer indexes, which must have
 * the same rank as the input; the output shape is the indexes shape. This
 * is the general take-along-dim (unlike {@link take}, which selects whole
 * slices with a 1-D index).
 *
 * @since 0.1.0
 * @category shape operations
 */
export const gather: {
  (
    indexes: Any,
    options?: { readonly dim?: number }
  ): (self: Any) => Effect.Effect<Lazy, TensorError>
  (
    self: Any,
    indexes: Any,
    options?: { readonly dim?: number }
  ): Effect.Effect<Lazy, TensorError>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: Any,
    indexes: Any,
    options: { readonly dim?: number } = {}
  ): Effect.Effect<Lazy, TensorError> =>
    Effect.try({
      try: () => {
        const d = normalizeDim("gather", self.shape.length, options.dim ?? 0)
        if (indexes.dtype !== "i64" && indexes.dtype !== "u32") {
          throw new Error(`gather: indexes must be i64 or u32, got ${indexes.dtype}`)
        }
        if (indexes.shape.length !== self.shape.length) {
          throw new Error(
            `gather: indexes rank ${indexes.shape.length} must match input rank ${self.shape.length}`
          )
        }
        for (let i = 0; i < self.shape.length; i++) {
          if (i !== d && indexes.shape[i] > self.shape[i]) {
            throw new Error(
              `gather: indexes shape [${indexes.shape}] exceeds input shape [${self.shape}] at dim ${i}`
            )
          }
        }
        if (indexes.device !== self.device) {
          throw new Error(`gather: device mismatch, got ${indexes.device} and ${self.device}`)
        }
        return makeLazy(self.lazy.gather(d, indexes.lazy), indexes.shape, self.dtype, self.device)
      },
      catch: (error) =>
        new TensorError({ op: "gather", message: error instanceof Error ? error.message : String(error) })
    })
)

/**
 * Adds `src` into `self` at positions given by `indexes` along `dim`
 * (accumulating duplicates): the differentiable inverse of {@link gather}.
 * `indexes` must be `i64` or `u32` with the same shape as `src`.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const scatterAdd = (
  self: Any,
  indexes: Any,
  src: Any,
  options: { readonly dim?: number } = {}
): Effect.Effect<Lazy, TensorError> =>
  Effect.try({
    try: () => {
      const d = normalizeDim("scatterAdd", self.shape.length, options.dim ?? 0)
      if (indexes.dtype !== "i64" && indexes.dtype !== "u32") {
        throw new Error(`scatterAdd: indexes must be i64 or u32, got ${indexes.dtype}`)
      }
      if (indexes.shape.length !== src.shape.length || !indexes.shape.every((s, i) => s === src.shape[i])) {
        throw new Error(
          `scatterAdd: indexes shape [${indexes.shape}] must match src shape [${src.shape}]`
        )
      }
      if (src.shape.length !== self.shape.length) {
        throw new Error(`scatterAdd: src rank ${src.shape.length} must match input rank ${self.shape.length}`)
      }
      for (let i = 0; i < self.shape.length; i++) {
        if (i !== d && src.shape[i] !== self.shape[i]) {
          throw new Error(
            `scatterAdd: src shape [${src.shape}] must match input shape [${self.shape}] outside dim ${d}`
          )
        }
      }
      checkCompatible("scatterAdd", self, src)
      if (indexes.device !== self.device) {
        throw new Error(`scatterAdd: device mismatch, got ${indexes.device} and ${self.device}`)
      }
      return makeLazy(self.lazy.scatterAdd(d, indexes.lazy, src.lazy), self.shape, self.dtype, self.device)
    },
    catch: (error) =>
      new TensorError({ op: "scatterAdd", message: error instanceof Error ? error.message : String(error) })
  })

/**
 * Reverses the order of elements along the given dimensions.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const flip = (
  self: Any,
  dims: ReadonlyArray<number>
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const normalized = normalizeDims("flip", self.shape.length, dims)
    let cur: Any = self
    for (const d of normalized) {
      const n = self.shape[d]
      const r = yield* arange(n, undefined, { dtype: "i64" })
      const idx = yield* add(yield* mul(r, yield* constantLike(r, -1)), yield* constantLike(r, n - 1))
      cur = yield* take(cur, idx, { dim: d })
    }
    return cur as Lazy
  })

/**
 * Expands `i64` or `u32` class indexes of any shape into one-hot vectors of the
 * given depth, appended as a new last dimension.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const oneHot = (
  indexes: Any,
  depth: number,
  options: { readonly dtype?: "f32" | "f64" } = {}
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (indexes.dtype !== "i64" && indexes.dtype !== "u32") {
      return yield* new TensorError({
        op: "oneHot",
        message: `oneHot: indexes must be i64 or u32, got ${indexes.dtype}`
      })
    }
    if (!Number.isInteger(depth) || depth < 1) {
      return yield* new TensorError({ op: "oneHot", message: `oneHot: depth must be a positive integer, got ${depth}` })
    }
    const classes = yield* arange(depth, undefined, { dtype: indexes.dtype })
    const expanded = yield* reshape(indexes, [...indexes.shape, 1])
    return yield* cast(yield* eq(expanded, classes), options.dtype ?? "f32")
  })

/**
 * Cross entropy between class logits of shape `[..., classes]` and
 * integer class-index targets of the leading shape: the scalar mean of
 * `logsumexp(logits) - logits[target]` over active positions, computed
 * stably (max subtraction) without materializing softmax intermediates or a
 * one-hot tensor in the graph. Positions where the target equals
 * `ignoreIndex` (default `-100`) contribute zero loss and zero gradient and
 * are excluded from the mean. Evaluation fails when every position is
 * ignored or an active target is out of range. The backward is not
 * second-order differentiable.
 *
 * @since 0.1.0
 * @category losses
 */
export const crossEntropy: {
  (
    options: { readonly target: Any; readonly ignoreIndex?: number }
  ): (self: Any) => Effect.Effect<Lazy, TensorError>
  (
    self: Any,
    options: { readonly target: Any; readonly ignoreIndex?: number }
  ): Effect.Effect<Lazy, TensorError>
} = dual(2, (
  self: Any,
  options: { readonly target: Any; readonly ignoreIndex?: number }
): Effect.Effect<Lazy, TensorError> =>
  Effect.try({
    try: () => {
      const { target } = options
      const ignoreIndex = options.ignoreIndex ?? -100
      if (self.shape.length < 1) {
        throw new Error("crossEntropy: logits must have rank >= 1")
      }
      if (self.dtype !== "f32" && self.dtype !== "f64") {
        throw new Error(`crossEntropy: logits must be f32 or f64, got ${self.dtype}`)
      }
      if (target.dtype !== "i64" && target.dtype !== "u32") {
        throw new Error(`crossEntropy: targets must be i64 or u32, got ${target.dtype}`)
      }
      const leading = self.shape.slice(0, -1)
      if (target.shape.length !== leading.length || !leading.every((d, i) => d === target.shape[i])) {
        throw new Error(
          `crossEntropy: targets shape [${target.shape}] does not match logits leading shape [${leading}]`
        )
      }
      if (!Number.isInteger(ignoreIndex)) {
        throw new Error(`crossEntropy: ignoreIndex must be an integer, got ${ignoreIndex}`)
      }
      if (target.device !== self.device) {
        throw new Error(`crossEntropy: device mismatch, got ${target.device} and ${self.device}`)
      }
      return makeLazy(self.lazy.crossEntropy(target.lazy, ignoreIndex), [], self.dtype, self.device)
    },
    catch: (error) =>
      new TensorError({ op: "crossEntropy", message: error instanceof Error ? error.message : String(error) })
  })
)

/**
 * Embedding lookup: selects rows from a `[vocab, hidden]` weight matrix by
 * integer indexes of any shape, giving output shape `[...indexes.shape,
 * hidden]`. Repeated indexes accumulate weight gradients. With
 * `paddingIndex`, the stored padding row is returned in the forward pass but
 * receives zero gradient (the `torch.nn.functional.embedding` contract).
 *
 * @since 0.1.0
 * @category shape operations
 */
export const embedding = (
  indexes: Any,
  options: {
    readonly weight: Any
    readonly paddingIndex?: number
  }
): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    const { paddingIndex, weight } = options
    if (weight.shape.length !== 2) {
      return yield* new TensorError({
        op: "embedding",
        message: `embedding: weight must be rank 2 [vocab, hidden], got shape [${weight.shape}]`
      })
    }
    if (weight.dtype !== "f32" && weight.dtype !== "f64") {
      return yield* new TensorError({
        op: "embedding",
        message: `embedding: weight must be f32 or f64, got ${weight.dtype}`
      })
    }
    if (indexes.dtype !== "i64" && indexes.dtype !== "u32") {
      return yield* new TensorError({
        op: "embedding",
        message: `embedding: indexes must be i64 or u32, got ${indexes.dtype}`
      })
    }
    if (indexes.device !== weight.device) {
      return yield* new TensorError({
        op: "embedding",
        message: `embedding: device mismatch, got ${indexes.device} and ${weight.device}`
      })
    }
    const [vocab, hidden] = weight.shape
    if (
      paddingIndex !== undefined &&
      (!Number.isInteger(paddingIndex) || paddingIndex < 0 || paddingIndex >= vocab)
    ) {
      return yield* new TensorError({
        op: "embedding",
        message: `embedding: paddingIndex must be an integer in [0, ${vocab}), got ${paddingIndex}`
      })
    }
    const n = indexes.shape.reduce((acc, d) => acc * d, 1)
    const flat = indexes.shape.length === 1 ? indexes : yield* reshape(indexes, [n])
    let out: Any = yield* take(weight, flat, { dim: 0 })
    if (paddingIndex !== undefined) {
      const mask = yield* broadcastTo(
        yield* reshape(yield* cast(yield* eq(flat, yield* constantLike(flat, paddingIndex)), weight.dtype), [n, 1]),
        [n, hidden]
      )
      const stopped = makeLazy(out.lazy.stopGradient(), out.shape, out.dtype, out.device)
      out = yield* add(yield* sub(out, yield* mul(mask, out)), yield* mul(mask, stopped))
    }
    return yield* reshape(out, [...indexes.shape, hidden])
  })

const triangleMask = (
  op: string,
  self: Any,
  diagonal: number,
  keepUpper: boolean
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (self.shape.length < 2) {
      return yield* new TensorError({ op, message: `${op}: expected rank >= 2, got rank ${self.shape.length}` })
    }
    const m = self.shape[self.shape.length - 2]
    const n = self.shape[self.shape.length - 1]
    const rows = yield* reshape(yield* arange(m, undefined, { dtype: "i64" }), [m, 1])
    const cols = yield* reshape(yield* arange(n, undefined, { dtype: "i64" }), [1, n])
    const shifted = yield* add(rows, yield* constantLike(rows, diagonal))
    const mask = keepUpper ? yield* ge(cols, shifted) : yield* le(cols, shifted)
    return yield* where(mask, self, yield* constantLike(self, 0))
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
    self: Any,
    options: { readonly diagonal?: number } = {}
  ): Effect.Effect<Lazy, TensorError, CurrentDevice> => triangleMask("triu", self, options.diagonal ?? 0, true)
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
    self: Any,
    options: { readonly diagonal?: number } = {}
  ): Effect.Effect<Lazy, TensorError, CurrentDevice> => triangleMask("tril", self, options.diagonal ?? 0, false)
)

/**
 * Dot product of two rank-1 tensors (`sum(a * b)`), or matrix
 * multiplication when both are rank >= 2 (alias of {@link matmul}).
 *
 * @since 0.1.0
 * @category operations
 */
export const dot: {
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError>
} = dual(
  2,
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError> =>
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
  self: Any
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
  self: Any,
  weight: Any,
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

const convOutDim = (
  op: string,
  input: number,
  kernel: number,
  stride: number,
  padding: number,
  dilation: number
): Effect.Effect<number, TensorError> => {
  const effective = dilation * (kernel - 1) + 1
  if (input + 2 * padding < effective) {
    return new TensorError({
      op,
      message: `${op}: kernel of effective size ${effective} exceeds the padded input size ${input + 2 * padding}`
    })
  }
  return Effect.succeed(Math.floor((input + 2 * padding - effective) / stride) + 1)
}

/**
 * 2-D convolution as a single native node (candle's kernel on every
 * backend), differentiable through native adjoints. `self` is
 * `[N, C_in, H, W]`, `weight` is `[C_out, C_in/groups, KH, KW]`; a bias is
 * added separately with `add`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const conv2d: {
  (
    weight: Any,
    options?: ConvOptions
  ): (self: Any) => Effect.Effect<Lazy, TensorError>
  (
    self: Any,
    weight: Any,
    options?: ConvOptions
  ): Effect.Effect<Lazy, TensorError>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: Any,
    weight: Any,
    options: ConvOptions = {}
  ): Effect.Effect<Lazy, TensorError> =>
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
      const oh = yield* convOutDim("conv2d", self.shape[2], weight.shape[2], opts.stride, opts.padding, opts.dilation)
      const ow = yield* convOutDim("conv2d", self.shape[3], weight.shape[3], opts.stride, opts.padding, opts.dilation)
      return yield* Effect.try({
        try: () =>
          makeLazy(
            self.lazy.conv2d(weight.lazy, opts.stride, opts.padding, opts.dilation, opts.groups),
            [self.shape[0], cOut, oh, ow],
            self.dtype,
            self.device
          ),
        catch: (error) =>
          new TensorError({ op: "conv2d", message: error instanceof Error ? error.message : String(error) })
      })
    })
)

/**
 * 1-D convolution over `[N, C_in, L]` with `weight` `[C_out, C_in/groups, K]`,
 * as a single native node.
 *
 * @since 0.1.0
 * @category neural network
 */
export const conv1d: {
  (
    weight: Any,
    options?: ConvOptions
  ): (self: Any) => Effect.Effect<Lazy, TensorError>
  (
    self: Any,
    weight: Any,
    options?: ConvOptions
  ): Effect.Effect<Lazy, TensorError>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: Any,
    weight: Any,
    options: ConvOptions = {}
  ): Effect.Effect<Lazy, TensorError> =>
    Effect.gen(function* () {
      const opts = yield* checkConvOptions("conv1d", self, weight, options, 1)
      yield* Effect.try({
        try: () => checkCompatible("conv1d", self, weight),
        catch: (error) =>
          new TensorError({ op: "conv1d", message: error instanceof Error ? error.message : String(error) })
      })
      const cIn = self.shape[1]
      const [cOut, cPerGroup] = [weight.shape[0], weight.shape[1]]
      if (cIn % opts.groups !== 0 || cOut % opts.groups !== 0) {
        return yield* new TensorError({
          op: "conv1d",
          message: `conv1d: channels [${cIn}, ${cOut}] are not divisible into ${opts.groups} groups`
        })
      }
      if (cPerGroup !== cIn / opts.groups) {
        return yield* new TensorError({
          op: "conv1d",
          message: `conv1d: weight has ${cPerGroup} input channels per group, expected ${cIn / opts.groups}`
        })
      }
      const ol = yield* convOutDim("conv1d", self.shape[2], weight.shape[2], opts.stride, opts.padding, opts.dilation)
      return yield* Effect.try({
        try: () =>
          makeLazy(
            self.lazy.conv1d(weight.lazy, opts.stride, opts.padding, opts.dilation, opts.groups),
            [self.shape[0], cOut, ol],
            self.dtype,
            self.device
          ),
        catch: (error) =>
          new TensorError({ op: "conv1d", message: error instanceof Error ? error.message : String(error) })
      })
    })
)

const dilateDim = (
  self: Any,
  dim: number,
  factor: number
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    if (factor === 1) {
      return yield* add(self, yield* constantLike(self, 0))
    }
    const n = self.shape[dim]
    const widened = yield* unsqueeze(self, dim + 1)
    const zshape = [...self.shape]
    zshape.splice(dim + 1, 0, factor - 1)
    const cat = yield* concat([widened, yield* zeros(zshape, { dtype: self.dtype })], { dim: dim + 1 })
    const merged = [...cat.shape]
    merged[dim] = n * factor
    merged.splice(dim + 1, 1)
    const wide = yield* reshape(cat, merged)
    const keep = (n - 1) * factor + 1
    return yield* slice(wide, { end: wide.shape.map((s, i) => (i === dim ? keep : s)) })
  })

/**
 * Options for {@link convTranspose2d} and {@link convTranspose1d}.
 * `outputPadding` appends zeros to the bottom/right of the result to
 * resolve stride ambiguity (must be smaller than `stride`).
 *
 * @since 0.1.0
 * @category models
 */
export interface ConvTransposeOptions extends ConvOptions {
  readonly outputPadding?: number
}

const convTranspose2dImpl = (
  op: string,
  self: Any,
  weight: Any,
  options: ConvTransposeOptions,
  userPadding: readonly [number, number],
  outputPads: readonly [number, number]
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const opts = yield* checkConvOptions(op, self, weight, options, 2)
    const outputPadding = options.outputPadding ?? 0
    if (!Number.isInteger(outputPadding) || outputPadding < 0) {
      return yield* new TensorError({
        op,
        message: `${op}: outputPadding must be a non-negative integer, got ${outputPadding}`
      })
    }
    if (outputPadding >= opts.stride) {
      return yield* new TensorError({
        op,
        message: `${op}: outputPadding ${outputPadding} must be smaller than stride ${opts.stride}`
      })
    }
    yield* Effect.try({
      try: () => checkCompatible(op, self, weight),
      catch: (error) =>
        new TensorError({
          op,
          message: error instanceof Error ? error.message : String(error)
        })
    })
    const cIn = self.shape[1]
    const [wIn, , kh, kw] = weight.shape
    const groups = opts.groups
    if (wIn !== cIn) {
      return yield* new TensorError({
        op,
        message: `${op}: weight has ${wIn} input channels, expected ${cIn}`
      })
    }
    if (cIn % groups !== 0) {
      return yield* new TensorError({
        op,
        message: `${op}: ${cIn} input channels are not divisible into ${groups} groups`
      })
    }
    // equivalent conv: dilated input, flipped channel-swapped kernel,
    // padding' = dilation * (k - 1) - padding
    const padY = opts.dilation * (kh - 1) - userPadding[0]
    const padX = opts.dilation * (kw - 1) - userPadding[1]
    if (padY < 0 || padX < 0) {
      return yield* new TensorError({
        op,
        message: `${op}: padding [${userPadding}] is too large for kernel [${kh}, ${kw}] with dilation ${opts.dilation}`
      })
    }
    const convGroup = (x: Any, w: Any): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
      Effect.gen(function* () {
        const dilated = yield* dilateDim(yield* dilateDim(x, 2, opts.stride), 3, opts.stride)
        const kernel = yield* flip(yield* transpose(w, [1, 0, 2, 3]), [2, 3])
        const padded = padY > 0 || padX > 0
          ? yield* pad(dilated, [[0, 0], [0, 0], [padY, padY], [padX, padX]])
          : dilated
        return yield* conv2d(padded, kernel, { dilation: opts.dilation })
      })
    let out: Lazy
    if (groups === 1) {
      out = yield* convGroup(self, weight)
    } else {
      const xs = yield* split(self, Array<number>(groups).fill(cIn / groups), { dim: 1 })
      const ws = yield* split(weight, Array<number>(groups).fill(wIn / groups), { dim: 0 })
      const outs: Array<Lazy> = []
      for (let i = 0; i < groups; i++) {
        outs.push(yield* convGroup(xs[i], ws[i]))
      }
      out = yield* concat(outs as [Any, Any, ...Array<Any>], { dim: 1 })
    }
    if (outputPads[0] > 0 || outputPads[1] > 0) {
      out = yield* pad(out, [[0, 0], [0, 0], [0, outputPads[0]], [0, outputPads[1]]])
    }
    return out
  })

/**
 * 2-D transposed convolution ("deconvolution", the gradient of conv2d):
 * `self` is `[N, C_in, H, W]`, `weight` is `[C_in, C_out/groups, KH, KW]`.
 * Composed as input dilation (zero-interleave) followed by a regular
 * {@link conv2d} with the spatially flipped, channel-swapped kernel — so
 * it runs on every backend and differentiates through ordinary adjoints.
 *
 * @since 0.1.0
 * @category neural network
 */
export const convTranspose2d: {
  (
    weight: Any,
    options?: ConvTransposeOptions
  ): (self: Any) => Effect.Effect<Lazy, TensorError, CurrentDevice>
  (
    self: Any,
    weight: Any,
    options?: ConvTransposeOptions
  ): Effect.Effect<Lazy, TensorError, CurrentDevice>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: Any,
    weight: Any,
    options: ConvTransposeOptions = {}
  ): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
    convTranspose2dImpl("convTranspose2d", self, weight, options, [
      options.padding ?? 0,
      options.padding ?? 0
    ], [
      options.outputPadding ?? 0,
      options.outputPadding ?? 0
    ])
)

/**
 * 1-D transposed convolution over `[N, C_in, L]` with `weight`
 * `[C_in, C_out/groups, K]`, implemented as a rank-4
 * {@link convTranspose2d}.
 *
 * @since 0.1.0
 * @category neural network
 */
export const convTranspose1d: {
  (
    weight: Any,
    options?: ConvTransposeOptions
  ): (self: Any) => Effect.Effect<Lazy, TensorError, CurrentDevice>
  (
    self: Any,
    weight: Any,
    options?: ConvTransposeOptions
  ): Effect.Effect<Lazy, TensorError, CurrentDevice>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && TensorTypeId in (args[1] as object)),
  (
    self: Any,
    weight: Any,
    options: ConvTransposeOptions = {}
  ): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
    Effect.gen(function* () {
      if (self.shape.length !== 3 || weight.shape.length !== 3) {
        return yield* new TensorError({
          op: "convTranspose1d",
          message: `convTranspose1d: expected rank-3 input and weight, got ranks ${self.shape.length} and ${weight.shape.length}`
        })
      }
      const out = yield* convTranspose2dImpl(
        "convTranspose1d",
        yield* unsqueeze(self, 2),
        yield* unsqueeze(weight, 2),
        options,
        [0, options.padding ?? 0],
        [0, options.outputPadding ?? 0]
      )
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
  reduce: (t: Any) => Effect.Effect<Lazy, TensorError>,
  self: Any,
  options: PoolOptions
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
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
    const windows: Array<Lazy> = []
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
      windows as unknown as [Any, Any, ...Array<Any>],
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
  self: Any,
  options: PoolOptions
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  pool2d("maxPool2d", (t) => max(t, { dims: [0] }), self, options)

/**
 * 2-D average pooling over `[N, C, H, W]`, composed from window slices.
 *
 * @since 0.1.0
 * @category neural network
 */
export const avgPool2d = (
  self: Any,
  options: PoolOptions
): Effect.Effect<Lazy, TensorError, CurrentDevice> =>
  pool2d("avgPool2d", (t) => mean(t, { dims: [0] }), self, options)

const checkSquare = (op: string, self: Any): Effect.Effect<void, TensorError> =>
  Effect.gen(function* () {
    const rank = self.shape.length
    if (rank < 2 || self.shape[rank - 2] !== self.shape[rank - 1]) {
      return yield* new TensorError({
        op,
        message: `${op}: expected a tensor square on its last two dimensions, got shape [${self.shape}]`
      })
    }
    if (!isFloatDtype(self.dtype)) {
      return yield* new TensorError({ op, message: `${op}: dtype must be f32 or f64, got ${self.dtype}` })
    }
  })

/**
 * Matrix inverse of a tensor that is square on its last two dimensions;
 * leading dimensions are treated as batch. Linear algebra runs on the
 * CPU — on other devices the matrices round-trip through the host.
 *
 * @since 0.1.0
 * @category linalg
 */
export const inverse = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    yield* checkSquare("inverse", self)
    return yield* Effect.try({
      try: () => makeLazy(self.lazy.inverse(), self.shape, self.dtype, self.device),
      catch: (error) =>
        new TensorError({ op: "inverse", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Determinant of a tensor that is square on its last two dimensions, with
 * the leading (batch) dimensions as the output shape. Linear algebra runs
 * on the CPU — on other devices the matrices round-trip through the host.
 *
 * @since 0.1.0
 * @category linalg
 */
export const det = (self: Any): Effect.Effect<Lazy, TensorError> =>
  Effect.gen(function* () {
    yield* checkSquare("det", self)
    return yield* Effect.try({
      try: () => makeLazy(self.lazy.det(), self.shape.slice(0, -2), self.dtype, self.device),
      catch: (error) =>
        new TensorError({ op: "det", message: error instanceof Error ? error.message : String(error) })
    })
  })

/**
 * Solves the linear system `a @ x = b` for `x`, with `a` square on its last
 * two dimensions and `b` of matching rank whose leading dimensions equal
 * `a`'s. Linear algebra runs on the CPU — on other devices the
 * matrices round-trip through the host.
 *
 * @since 0.1.0
 * @category linalg
 */
export const solve: {
  (b: Any): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, b: Any): Effect.Effect<Lazy, TensorError>
} = dual(
  2,
  (self: Any, b: Any): Effect.Effect<Lazy, TensorError> =>
    Effect.gen(function* () {
      yield* checkSquare("solve", self)
      const rank = self.shape.length
      if (
        b.shape.length !== rank ||
        !self.shape.slice(0, -2).every((d, i) => d === b.shape[i]) ||
        b.shape[rank - 2] !== self.shape[rank - 1]
      ) {
        return yield* new TensorError({
          op: "solve",
          message: `solve: expected a right-hand side of matching rank with leading shape [${self.shape.slice(0, -1)}], got shape [${b.shape}]`
        })
      }
      yield* Effect.try({
        try: () => checkCompatible("solve", self, b),
        catch: (error) =>
          new TensorError({ op: "solve", message: error instanceof Error ? error.message : String(error) })
      })
      return yield* Effect.try({
        try: () => makeLazy(self.lazy.solve(b.lazy), b.shape, self.dtype, self.device),
        catch: (error) =>
          new TensorError({ op: "solve", message: error instanceof Error ? error.message : String(error) })
      })
    })
)

/**
 * Converts a tensor to a different dtype. Dtypes are strict in this library:
 * no implicit promotion happens anywhere, so `cast` is the only way to mix
 * dtypes.
 *
 * @since 0.1.0
 * @category operations
 */
export const cast: {
  (dtype: DType): (self: Any) => Effect.Effect<Lazy, TensorError>
  (self: Any, dtype: DType): Effect.Effect<Lazy, TensorError>
} = dual(2, (self: Any, dtype: DType): Effect.Effect<Lazy, TensorError> =>
  Effect.try({
    try: () => makeLazy(self.lazy.cast(dtype as NativeDType), self.shape, dtype, self.device),
    catch: (error) =>
      new TensorError({ op: "cast", message: error instanceof Error ? error.message : String(error) })
  })
)
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
 * as-is. A tuple in gives the same tuple out, each element materialized.
 *
 * @since 0.1.0
 * @category destructors
 */
export const compute = <Roots extends ReadonlyArray<Any>>(
  roots: Roots
): Effect.Effect<{ readonly [K in keyof Roots]: Concrete }, TensorError> =>
  roots.every(isTensor)
    ? Effect.succeed(roots as { readonly [K in keyof Roots]: Concrete })
    : Effect.map(
        fromNative("evaluate", (token) => evalLazy(roots.map((root) => root.lazy), token)),
        (handles) => {
          reportExternalMemory(handles.reduce((total, handle) => total + handle.bytes, 0))
          return handles.map(fromHandle) as { readonly [K in keyof Roots]: Concrete }
        }
      )

const typedArrayConstructor = (dtype: DType) => {
  switch (dtype) {
    case "f32":
    // f16 reads back as f32: the native side converts before the readback
    // since JS has no f16 typed array on Node 22
    case "f16":
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
export const toTypedArray = (self: Any): Effect.Effect<TypedArray, TensorError> =>
  Effect.flatMap(compute([self]), ([evaluated]) =>
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
export const toNumberArray = (self: Any): Effect.Effect<Array<number>, TensorError> =>
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
export const dispose = (self: Concrete): Effect.Effect<void, TensorError> =>
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
  tensors: Readonly<Record<string, Any>>
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
): Effect.Effect<Record<string, Concrete>, TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    const [names, handles] = yield* fromNative("load", (token) => loadTensors(path, device, token))
    reportExternalMemory(handles.reduce((total, handle) => total + handle.bytes, 0))
    return Object.fromEntries(names.map((name, i) => [name, fromHandle(handles[i])]))
  })

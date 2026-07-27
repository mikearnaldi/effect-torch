import { Data, Effect } from "effect"
import { dual } from "effect/Function"
import { pipeArguments, type Pipeable } from "effect/Pipeable"
import native, {
  type LazyTensor as NativeLazyTensorType,
  type NativeDType,
  type NativeTensor as NativeTensorType
} from "@effect-torch/native"
import { CurrentDevice, type DeviceKind } from "./Device.js"

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

const liftOperand = (
  self: GenericTensor,
  other: GenericTensor | number
): { readonly lazy: NativeLazyTensorType; readonly shape: ReadonlyArray<number> } => {
  if (typeof other === "number") {
    return {
      lazy: NativeLazyTensor.full([], other, self.dtype as NativeDType, self.device),
      shape: []
    }
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

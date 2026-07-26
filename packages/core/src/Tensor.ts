import { Data, Effect } from "effect"
import { dual } from "effect/Function"
import { pipeArguments, type Pipeable } from "effect/Pipeable"
import native, {
  type LazyTensor as NativeLazyTensorType,
  type NativeDType,
  type NativeTensor as NativeTensorType
} from "@effect-torch/native"

const { CancellationToken, evalLazy, LazyTensor: NativeLazyTensor } = native

export type DType = "f32" | "f64"

export type DeviceKind = "cpu" | "metal" | "cuda"

export interface TensorOptions {
  readonly dtype?: DType
  readonly device?: DeviceKind
}

export class TensorError extends Data.TaggedError("TensorError")<{
  readonly op: string
  readonly message: string
}> {}

export interface GenericTensor extends Pipeable {
  readonly [TensorTypeId]: TensorTypeId
  readonly _tag: "LazyTensor" | "Tensor"
  /** @internal */
  readonly lazy: NativeLazyTensorType
  readonly shape: ReadonlyArray<number>
  readonly dtype: DType
  readonly device: DeviceKind
}

export interface LazyTensor extends GenericTensor {
  readonly _tag: "LazyTensor"
}

export interface Tensor extends GenericTensor {
  readonly _tag: "Tensor"
  /** @internal */
  readonly materialized: NativeTensorType
}

const TensorTypeId: unique symbol = Symbol.for("@effect-torch/core/Tensor")

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

export const isLazyTensor = (self: GenericTensor): self is LazyTensor => self._tag === "LazyTensor"

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
    throw new Error(`${op}: dtype mismatch, got ${a.dtype} and ${b.dtype}`)
  }
  if (a.device !== b.device) {
    throw new Error(`${op}: device mismatch, got ${a.device} and ${b.device}`)
  }
}

export const zeros = (
  shape: ReadonlyArray<number>,
  options: TensorOptions = {}
): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => {
      const validShape = validateShape("zeros", shape)
      return makeLazy(
        NativeLazyTensor.zeros(validShape, options.dtype as NativeDType, options.device),
        validShape,
        options.dtype ?? "f32",
        options.device ?? "cpu"
      )
    },
    catch: (error) =>
      new TensorError({ op: "zeros", message: error instanceof Error ? error.message : String(error) })
  })

export const randn = (
  shape: ReadonlyArray<number>,
  options: TensorOptions = {}
): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => {
      const validShape = validateShape("randn", shape)
      return makeLazy(
        NativeLazyTensor.randn(validShape, options.dtype as NativeDType, options.device),
        validShape,
        options.dtype ?? "f32",
        options.device ?? "cpu"
      )
    },
    catch: (error) =>
      new TensorError({ op: "randn", message: error instanceof Error ? error.message : String(error) })
  })

export const shape = (self: GenericTensor): ReadonlyArray<number> => self.shape

export const dtype = (self: GenericTensor): DType => self.dtype

export const device = (self: GenericTensor): DeviceKind => self.device

export const add: {
  (other: GenericTensor): (self: GenericTensor) => Effect.Effect<LazyTensor, TensorError>
  (self: GenericTensor, other: GenericTensor): Effect.Effect<LazyTensor, TensorError>
} = dual(2, (self: GenericTensor, other: GenericTensor): Effect.Effect<LazyTensor, TensorError> =>
  Effect.try({
    try: () => {
      checkCompatible("add", self, other)
      return makeLazy(
        self.lazy.add(other.lazy),
        broadcastShapes("add", self.shape, other.shape),
        self.dtype,
        self.device
      )
    },
    catch: (error) =>
      new TensorError({ op: "add", message: error instanceof Error ? error.message : String(error) })
  })
)

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

export const evaluate = (self: GenericTensor): Effect.Effect<Tensor, TensorError> =>
  isTensor(self)
    ? Effect.succeed(self)
    : Effect.map(
        fromNative("evaluate", (token) => evalLazy(self.lazy, token)),
        fromHandle
      )

export const toTypedArray = (
  self: GenericTensor
): Effect.Effect<Float32Array | Float64Array, TensorError> =>
  Effect.flatMap(evaluate(self), (evaluated) =>
    fromNative<Float32Array | Float64Array>("toTypedArray", async () => {
      const buffer = await evaluated.materialized.readback()
      return evaluated.dtype === "f64" ? new Float64Array(buffer) : new Float32Array(buffer)
    })
  )

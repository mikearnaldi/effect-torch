import { Data, Deferred, Effect, Exit } from "effect"
import { dual } from "effect/Function"
import * as Runtime from "./Runtime.ts"

/**
 * Element data types understood by the runtime API. Availability depends on
 * the active backend and placement.
 *
 * @since 0.1.0
 * @category models
 */
export type DType = Runtime.DType

/** GGML K-quant storage encodings accepted by native tensor operations. */
export type GgmlKQuant = Runtime.TensorStorageEncoding

/**
 * JavaScript typed-array representations used for tensor transfer.
 * {@link fromTypedArray} infers `f16` from `Float16Array`, but there is no
 * typed-array representation for `bf16`. Readback widens both `f16` and
 * `bf16` tensors to `Float32Array`; the other dtypes use their corresponding
 * array type.
 *
 * @since 0.1.0
 * @category models
 */
export type TypedArray = Float32Array | Float64Array | Float16Array | BigInt64Array | Uint8Array | Uint32Array

/**
 * Common options for tensor constructors.
 *
 * @since 0.1.0
 * @category models
 */
export interface TensorOptions {
  /** Element dtype of the constructed tensor; defaults to `f32`. */
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
  /** Public operation name associated with the failure. */
  readonly op: string
  /** Human-readable description of the failed validation or backend work. */
  readonly message: string
  /** Original backend failure when the error crossed the runtime boundary. */
  readonly backend?: Runtime.BackendError
}> {}

/**
 * Common supertype of {@link Lazy} and {@link Concrete}. Graph-building
 * operations generally accept either form, subject to matching runtime,
 * placement, shape, and dtype requirements.
 *
 * @since 0.1.0
 * @category models
 */
export type Any = Runtime.TensorHandle

/**
 * A tensor described by an immutable semantic graph. Graph-building operations
 * do not run tensor kernels. Materialization occurs when the graph is submitted
 * through {@link compute}, readback, saving, or as a lazy executable input.
 *
 * @since 0.1.0
 * @category models
 */
export type Lazy = Runtime.LazyTensorHandle

/**
 * A materialized tensor whose data is owned by its backend handle, obtained
 * through evaluation, compiled execution, or loading. The handle keeps that
 * storage alive until {@link clear} succeeds or native finalization runs;
 * using a cleared handle fails.
 *
 * @since 0.1.0
 * @category models
 */
export type Concrete = Runtime.ConcreteTensorHandle

/**
 * A backend-owned immutable executable and the metadata expected for its
 * outputs. The opaque handle retains its typed lowered program, static plans,
 * prepared artifacts, captured generated bindings or constants, and diagnostics.
 * There is no explicit program-release operation; drop JavaScript and cache
 * references to make the wrapper eligible for native finalization.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface CompiledProgram {
  /** Opaque runtime handle for the compiled executable. */
  readonly handle: Runtime.ExecutableHandle
  /** Output metadata recorded from the roots at freeze time. */
  readonly outputs: ReadonlyArray<{
    /** Expected output shape. */
    readonly shape: ReadonlyArray<number>
    /** Expected output dtype. */
    readonly dtype: DType
    /** Expected encoded storage, when the output is an encoded identity. */
    readonly storage?: Runtime.EncodedTensorStorage
    /** Expected output placement. */
    readonly placement: Runtime.Placement
  }>
}

/**
 * Refines a handle by its lazy tag. This does not validate runtime ownership or
 * whether concrete dependencies captured by the graph remain usable.
 *
 * @since 0.1.0
 * @category refinements
 */
export const isLazyTensor = (self: Any): self is Lazy => self._tag === "LazyTensor"

/**
 * Refines a handle by its concrete tag. This is not a liveness check: a cleared
 * concrete handle retains its tag but fails when used.
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
  if (a.placement.id !== b.placement.id) {
    throw new Error(`${op}: placement mismatch, got ${a.placement.id} and ${b.placement.id}`)
  }
}

const backendMessage = (error: Runtime.BackendError): string => error.message

const caughtTensorError = (op: string, error: unknown): TensorError =>
  error instanceof TensorError
    ? error
    : error instanceof Runtime.BackendError
    ? new TensorError({ op, message: error.message, backend: error })
    : new TensorError({ op, message: error instanceof Error ? error.message : String(error) })

const fromBackend = <A>(op: string, effect: Effect.Effect<A, Runtime.BackendError>): Effect.Effect<A, TensorError> =>
  Effect.mapError(effect, (error) => new TensorError({ op, message: backendMessage(error), backend: error }))

interface GraphResult {
  readonly request: Runtime.NodeRequest
  readonly shape: ReadonlyArray<number>
  readonly dtype: DType
  readonly storage?: Runtime.EncodedTensorStorage
  readonly placement: Runtime.Placement
}

const sameShape = (a: ReadonlyArray<number>, b: ReadonlyArray<number>): boolean =>
  a.length === b.length && a.every((dimension, index) => dimension === b[index])

const samePlacement = (a: Runtime.Placement, b: Runtime.Placement): boolean =>
  a.id === b.id && a.deviceType === b.deviceType && a.description === b.description && a.ordinal === b.ordinal &&
  a.memorySpace === b.memorySpace

const sameStorage = (
  a: Runtime.EncodedTensorStorage | undefined,
  b: Runtime.EncodedTensorStorage | undefined
): boolean =>
  a === undefined
    ? b === undefined
    : b !== undefined && a.encoding === b.encoding && a.physicalDtype === b.physicalDtype &&
      sameShape(a.physicalShape, b.physicalShape)

const encodedRowBytes = (encoding: Runtime.TensorStorageEncoding, columns: number): number | undefined => {
  if (columns % 256 !== 0) return undefined
  const blockBytes = encoding === "Q2_K"
    ? 84
    : encoding === "Q3_K"
    ? 110
    : encoding === "Q4_K"
    ? 144
    : encoding === "Q5_K"
    ? 176
    : 210
  return columns / 256 * blockBytes
}

const validStorage = (value: Runtime.TensorHandle): boolean => {
  if (value.storage === undefined) return true
  const columns = value.shape.at(-1)
  const rows = value.shape.slice(0, -1).reduce((total, dimension) => total * dimension, 1)
  const rowBytes = columns === undefined ? undefined : encodedRowBytes(value.storage.encoding, columns)
  return value.dtype === "f32" &&
    ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"].includes(value.storage.encoding) &&
    value.storage.physicalDtype === "u8" && Array.isArray(value.storage.physicalShape) &&
    value.storage.physicalShape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0) &&
    rowBytes !== undefined && sameShape(value.storage.physicalShape, [rows, rowBytes])
}

const isTensorHandleValue = (value: unknown): value is Any =>
  typeof value === "object" && value !== null &&
  ((value as { readonly _tag?: unknown })._tag === "LazyTensor" ||
    (value as { readonly _tag?: unknown })._tag === "Tensor")

const validateTensorHandle = <T extends "LazyTensor" | "Tensor">(
  op: string,
  runtime: Runtime.RuntimeService,
  value: Runtime.TensorHandle,
  expected: {
    readonly _tag: T
    readonly shape?: ReadonlyArray<number>
    readonly dtype?: DType
    readonly storage?: Runtime.EncodedTensorStorage
    readonly placement?: Runtime.Placement
  }
): T extends "LazyTensor" ? Lazy : Concrete => {
  const validDtypes: ReadonlyArray<string> = ["f32", "f64", "f16", "bf16", "i64", "u8", "u32"]
  if (
    !isTensorHandleValue(value) || value._tag !== expected._tag || !Array.isArray(value.shape) ||
    !value.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0) ||
    !validDtypes.includes(value.dtype) || !validStorage(value) || typeof value.device !== "string" ||
    typeof value.placement !== "object" || value.placement === null || value.device !== value.placement.deviceType ||
    !samePlacement(value.placement, runtime.placement) ||
    (expected.shape !== undefined && !sameShape(value.shape, expected.shape)) ||
    (expected.dtype !== undefined && value.dtype !== expected.dtype) ||
    !sameStorage(value.storage, expected.storage) ||
    (expected.placement !== undefined && !samePlacement(value.placement, expected.placement))
  ) {
    const candidate = value as Partial<Runtime.TensorHandle>
    throw new TensorError({
      op,
      message: `${op}: backend returned invalid ${
        expected._tag === "LazyTensor" ? "lazy" : "concrete"
      } tensor metadata; expected ${expected.dtype ?? "runtime dtype"} [${expected.shape ?? "runtime shape"}] on ${
        expected.placement?.id ?? runtime.placement.id
      }, got ${String(candidate.dtype)} [${Array.isArray(candidate.shape) ? candidate.shape : "invalid shape"}] on ${
        candidate.placement?.id ?? "invalid placement"
      }`
    })
  }
  return value as T extends "LazyTensor" ? Lazy : Concrete
}

const graphTry = (
  op: string,
  evaluate: (runtime: Runtime.RuntimeService) => GraphResult
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const result = yield* Effect.try({ try: () => evaluate(runtime), catch: (error) => caughtTensorError(op, error) })
    const handle = yield* fromBackend(op, runtime.node(result.request))
    return yield* Effect.try({
      try: () => validateTensorHandle(op, runtime, handle, { _tag: "LazyTensor", ...result }),
      catch: (error) => caughtTensorError(op, error)
    })
  })

const numel = (shape: ReadonlyArray<number>): number => shape.reduce((a, b) => a * b, 1)

const isFloatDtype = (dtype: string): boolean => dtype === "f32" || dtype === "f64"

/**
 * Creates a 0-d constant tensor. Native backends use a bounded process-local
 * pool keyed by value bits, dtype, and device, so a resident entry may be
 * reused; graph-node identity is not a public guarantee. Use {@link full} for
 * non-scalar shapes. Runtime-varying values must be declared as inputs: use
 * {@link makeScalarInput} for a plain number or {@link makeInput}/{@link compile}
 * for tensor bindings.
 *
 * @since 0.1.0
 * @category constructors
 */
export const constant = (
  value: number,
  options: TensorOptions = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("constant", (runtime) => {
    const dtype = options.dtype ?? "f32"
    return {
      request: { op: "constant", inputs: [], attributes: { value, dtype } },
      shape: [],
      dtype,
      placement: runtime.placement
    }
  })

/**
 * Creates a shared 0-d constant tensor with the same dtype and device as
 * `self` — the scalar counterpart of {@link zerosLike} / {@link onesLike}
 * / {@link fullLike}, and the way to lift a numeric constant next to an
 * existing tensor (custom losses, optimizer updates) without threading a
 * device through the environment. Native runtimes may pool the leaf as
 * described by {@link constant}; runtime-varying values require scalar or
 * tensor input declarations.
 *
 * @since 0.1.0
 * @category constructors
 */
export const constantLike = (self: Any, value: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("constantLike", (runtime) => ({
    request: { op: "constant", inputs: [self], attributes: { value, dtype: self.dtype } },
    shape: [],
    dtype: self.dtype,
    placement: runtime.placement
  }))

// A 0-d float scalar never promotes a float tensor's dtype (the native
// graph applies the same rule): `mul(f32Scalar, bf16Tensor)` is bf16.
const scalarCoercible = (a: Any, b: Any): boolean =>
  a.dtype !== b.dtype && isFloat(a.dtype) && isFloat(b.dtype) && (a.shape.length === 0) !== (b.shape.length === 0)

const isFloat = (dtype: DType): boolean => dtype === "f32" || dtype === "f64" || dtype === "f16" || dtype === "bf16"

const binaryOp = (
  op: string,
  request: (a: Runtime.TensorHandle, b: Runtime.TensorHandle) => Runtime.NodeRequest,
  outDtype: (dtype: DType) => DType = (dtype) => dtype
): {
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} =>
  dual(
    2,
    (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
      graphTry(op, () => {
        if (self.dtype !== other.dtype && !scalarCoercible(self, other)) {
          throw new Error(
            `${op}: dtype mismatch, got ${self.dtype} and ${other.dtype}, use cast for explicit conversion`
          )
        }
        if (self.placement.id !== other.placement.id) {
          throw new Error(`${op}: placement mismatch, got ${self.placement.id} and ${other.placement.id}`)
        }
        const dtype = scalarCoercible(self, other) && self.shape.length === 0 ? other.dtype : self.dtype
        return {
          request: request(self, other),
          shape: broadcastShapes(op, self.shape, other.shape),
          dtype: outDtype(dtype),
          placement: self.placement
        }
      })
  )

const unaryOp = (
  op: string,
  request: (a: Runtime.TensorHandle) => Runtime.NodeRequest
): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
(self) =>
  graphTry(op, () => ({
    request: request(self),
    shape: self.shape,
    dtype: self.dtype,
    placement: self.placement
  }))

const normalizeDim = (op: string, rank: number, dim: number): number => {
  const normalized = dim < 0 ? dim + rank : dim
  if (!Number.isInteger(normalized) || normalized < 0 || normalized >= rank) {
    throw new Error(`${op}: dimension ${dim} out of range for rank ${rank}`)
  }
  return normalized
}

const dualOptions = <O, R = Runtime.Runtime>(
  impl: (self: Any, options: O | undefined) => Effect.Effect<Lazy, TensorError, R>
): {
  (options?: O): (self: Any) => Effect.Effect<Lazy, TensorError, R>
  (self: Any, options?: O): Effect.Effect<Lazy, TensorError, R>
} =>
  dual(
    (args) => args.length === 2 || (args.length === 1 && isTensorHandleValue(args[0])),
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("zeros", (runtime) => {
    const validShape = validateShape("zeros", shape)
    const dtype = options.dtype ?? "f32"
    return {
      request: { op: "zeros", inputs: [], attributes: { shape: validShape, dtype } },
      shape: validShape,
      dtype,
      placement: runtime.placement
    }
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("ones", (runtime) => {
    const validShape = validateShape("ones", shape)
    const dtype = options.dtype ?? "f32"
    return {
      request: { op: "ones", inputs: [], attributes: { shape: validShape, dtype } },
      shape: validShape,
      dtype,
      placement: runtime.placement
    }
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("full", (runtime) => {
    const validShape = validateShape("full", shape)
    const dtype = options.dtype ?? "f32"
    return {
      request: { op: "full", inputs: [], attributes: { shape: validShape, value, dtype } },
      shape: validShape,
      dtype,
      placement: runtime.placement
    }
  })

/**
 * Creates a lazy tensor sampled from a standard normal distribution. Only
 * floating dtypes are supported. Sampling occurs per executable invocation:
 * reusing one random node across roots shares one draw within that invocation,
 * separately constructed nodes remain distinct, and every later invocation
 * draws afresh. Optimization preserves semantic random-source identity.
 *
 * @since 0.1.0
 * @category constructors
 */
export const randn = (
  shape: ReadonlyArray<number>,
  options: { readonly dtype?: "f32" | "f64" | "f16" | "bf16" } = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("randn", (runtime) => {
    const validShape = validateShape("randn", shape)
    const dtype = options.dtype ?? "f32"
    return {
      request: { op: "randn", inputs: [], attributes: { shape: validShape, dtype } },
      shape: validShape,
      dtype,
      placement: runtime.placement
    }
  })

/**
 * Creates a lazy floating-point tensor sampled uniformly from `[min, max)`;
 * construction rejects `max <= min`, but does not reject `NaN` bounds.
 * Sampling follows {@link randn}'s
 * invocation semantics. Submit related roots together in one {@link compute}
 * or compiled-program invocation when they must share a random source.
 *
 * @since 0.1.0
 * @category constructors
 */
export const uniform = (
  shape: ReadonlyArray<number>,
  options: { readonly min?: number; readonly max?: number; readonly dtype?: "f32" | "f64" | "f16" | "bf16" } = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("uniform", (runtime) => {
    const validShape = validateShape("uniform", shape)
    const dtype = options.dtype ?? "f32"
    return {
      request: {
        op: "uniform",
        inputs: [],
        attributes: { shape: validShape, lo: options.min ?? 0, hi: options.max ?? 1, dtype }
      },
      shape: validShape,
      dtype,
      placement: runtime.placement
    }
  })

/**
 * Creates a lazy 1-dimensional tensor of `steps` evenly spaced values from
 * `start` to `end`, both inclusive. `steps` must be a positive integer. This
 * constructor supports only `f32` and `f64` despite accepting `TensorOptions`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const linspace = (
  start: number,
  end: number,
  steps: number,
  options: TensorOptions = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
 * Creates values beginning at `start` and advancing by nonzero `step` until the
 * next value would cross exclusive `end`. Omitting `end` uses `0` as the start
 * and the first argument as the end. Positive and negative steps are supported;
 * a step whose sign moves away from `end` produces an empty tensor.
 *
 * @since 0.1.0
 * @category constructors
 */
export const arange = (
  start: number,
  end?: number,
  options: { readonly step?: number; readonly dtype?: DType } = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("arange", (runtime) => {
    const from = end === undefined ? 0 : start
    const to = end === undefined ? start : end
    const step = options.step ?? 1
    const size = Math.max(0, Math.ceil((to - from) / step))
    const dtype = options.dtype ?? "f32"
    return {
      request: { op: "arange", inputs: [], attributes: { start: from, end: to, step, dtype } },
      shape: [size],
      dtype,
      placement: runtime.placement
    }
  })

/**
 * Creates a lazy `n x n` identity matrix. `n` must be a positive integer.
 *
 * @since 0.1.0
 * @category constructors
 */
export const eye = (
  n: number,
  options: TensorOptions = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("eye", (runtime) => {
    const [size] = validateShape("eye", [n])
    if (size === 0) throw new Error("eye: n must be positive")
    const dtype = options.dtype ?? "f32"
    return {
      request: { op: "eye", inputs: [], attributes: { n: size, dtype } },
      shape: [size, size],
      dtype,
      placement: runtime.placement
    }
  })

const dtypeOfTypedArray = (data: TypedArray): DType => {
  if (data instanceof Float32Array) return "f32"
  if (data instanceof Float64Array) return "f64"
  if (typeof Float16Array !== "undefined" && data instanceof Float16Array) return "f16"
  if (data instanceof BigInt64Array) return "i64"
  if (data instanceof Uint8Array) return "u8"
  if (data instanceof Uint32Array) return "u32"
  throw new Error(`fromTypedArray: unsupported typed array ${(data as object).constructor.name}`)
}

/**
 * Creates a lazy tensor by copying the addressed bytes of `data` when this
 * graph-building Effect runs. Later source-array mutation does not change the
 * tensor. The dtype is inferred and shape defaults to `[data.length]`. This
 * semantic leaf lowers as an executable constant independently of
 * `constantWeights`; use an input placeholder for runtime-varying data.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromTypedArray = (
  data: TypedArray,
  shape?: ReadonlyArray<number>
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("fromTypedArray", (runtime) => {
    const dtype = dtypeOfTypedArray(data)
    const validShape = shape === undefined ? [data.length] : validateShape("fromTypedArray", shape)
    if (numel(validShape) !== data.length) {
      throw new Error(
        `fromTypedArray: data length ${data.length} does not match shape [${validShape}]`
      )
    }
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    return {
      request: { op: "fromBytes", inputs: [], attributes: { data: bytes, shape: validShape, dtype } },
      shape: validShape,
      dtype,
      placement: runtime.placement
    }
  })

/**
 * Creates a lazy tensor of zeros with the same shape and dtype as the input.
 *
 * @since 0.1.0
 * @category constructors
 */
export const zerosLike = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("zerosLike", (runtime) => ({
    request: { op: "zeros", inputs: [self], attributes: { shape: [...self.shape], dtype: self.dtype } },
    shape: self.shape,
    dtype: self.dtype,
    placement: runtime.placement
  }))

/**
 * Creates a lazy tensor of ones with the same shape and dtype as the input.
 *
 * @since 0.1.0
 * @category constructors
 */
export const onesLike = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("onesLike", (runtime) => ({
    request: { op: "ones", inputs: [self], attributes: { shape: [...self.shape], dtype: self.dtype } },
    shape: self.shape,
    dtype: self.dtype,
    placement: runtime.placement
  }))

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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("fullLike", (runtime) => ({
    request: { op: "full", inputs: [self], attributes: { shape: [...self.shape], value, dtype: self.dtype } },
    shape: self.shape,
    dtype: self.dtype,
    placement: runtime.placement
  }))

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
export const device = (self: Any): string => self.device

/**
 * Elementwise addition with broadcasting. Dtypes and placements must match,
 * except that when exactly one operand is a 0-d float tensor, its value is
 * coerced to the non-scalar float operand's dtype without promoting it.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const add = binaryOp("add", (a, b) => ({ op: "add", inputs: [a, b] }))

/**
 * Elementwise subtraction with broadcasting. The 0-d float coercion described
 * by {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sub = binaryOp("sub", (a, b) => ({ op: "sub", inputs: [a, b] }))

/**
 * Elementwise multiplication with broadcasting. The 0-d float coercion
 * described by {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const mul = binaryOp("mul", (a, b) => ({ op: "mul", inputs: [a, b] }))

/**
 * Elementwise division with broadcasting. The 0-d float coercion described by
 * {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const div = binaryOp("div", (a, b) => ({ op: "div", inputs: [a, b] }))

/**
 * Elementwise maximum of two tensors with broadcasting. The 0-d float
 * coercion described by {@link add} also applies. At equal elements the
 * gradient flows to the left operand only. Not to be confused with the
 * reduction {@link max}.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const maximum = binaryOp("maximum", (a, b) => ({ op: "maximum", inputs: [a, b] }))

/**
 * Elementwise minimum of two tensors with broadcasting. The 0-d float
 * coercion described by {@link add} also applies. At equal elements the
 * gradient flows to the left operand only. Not to be confused with the
 * reduction {@link min}.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const minimum = binaryOp("minimum", (a, b) => ({ op: "minimum", inputs: [a, b] }))

/**
 * Elementwise equality comparison with broadcasting. Returns a `u8` tensor;
 * the 0-d float coercion described by {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const eq = binaryOp("eq", (a, b) => ({ op: "eq", inputs: [a, b] }), () => "u8")

/**
 * Elementwise greater-than comparison with broadcasting. Returns a `u8`
 * tensor; the 0-d float coercion described by {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const gt = binaryOp("gt", (a, b) => ({ op: "gt", inputs: [a, b] }), () => "u8")

/**
 * Elementwise less-than comparison with broadcasting. Returns a `u8` tensor;
 * the 0-d float coercion described by {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const lt = binaryOp("lt", (a, b) => ({ op: "lt", inputs: [a, b] }), () => "u8")

/**
 * Elementwise greater-than-or-equal comparison with broadcasting. Returns a
 * `u8` tensor; the 0-d float coercion described by {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const ge = binaryOp("ge", (a, b) => ({ op: "ge", inputs: [a, b] }), () => "u8")

/**
 * Elementwise less-than-or-equal comparison with broadcasting. Returns a `u8`
 * tensor; the 0-d float coercion described by {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const le = binaryOp("le", (a, b) => ({ op: "le", inputs: [a, b] }), () => "u8")

/**
 * Elementwise negation.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const neg = unaryOp("neg", (a) => ({ op: "neg", inputs: [a] }))

/**
 * Elementwise absolute value.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const abs = unaryOp("abs", (a) => ({ op: "abs", inputs: [a] }))

/**
 * Elementwise square root.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sqrt = unaryOp("sqrt", (a) => ({ op: "sqrt", inputs: [a] }))

/**
 * Elementwise exponential.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const exp = unaryOp("exp", (a) => ({ op: "exp", inputs: [a] }))

/**
 * Elementwise natural logarithm.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log = unaryOp("log", (a) => ({ op: "log", inputs: [a] }))

/**
 * Elementwise sine.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sin = unaryOp("sin", (a) => ({ op: "sin", inputs: [a] }))

/**
 * Elementwise cosine.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const cos = unaryOp("cos", (a) => ({ op: "cos", inputs: [a] }))

/**
 * Elementwise hyperbolic tangent.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const tanh = unaryOp("tanh", (a) => ({ op: "tanh", inputs: [a] }))

/**
 * Elementwise rectified linear unit, `max(x, 0)`. The gradient at `x = 0`
 * is taken to be `0`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const relu = unaryOp("relu", (a) => ({ op: "relu", inputs: [a] }))

/**
 * Elementwise error function.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const erf = unaryOp("erf", (a) => ({ op: "erf", inputs: [a] }))

/**
 * Elementwise floor. The gradient is `0` almost everywhere.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const floor = unaryOp("floor", (a) => ({ op: "floor", inputs: [a] }))

/**
 * Elementwise ceiling. The gradient is `0` almost everywhere.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const ceil = unaryOp("ceil", (a) => ({ op: "ceil", inputs: [a] }))

/**
 * Elementwise rounding to the nearest integer. The gradient is `0` almost
 * everywhere.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const round = unaryOp("round", (a) => ({ op: "round", inputs: [a] }))

/**
 * Elementwise sign: `-1`, `0` or `1`. The gradient is `0` everywhere.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sign = unaryOp("sign", (a) => ({ op: "sign", inputs: [a] }))

/**
 * Elementwise square, `x * x`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const square = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> => mul(self, self)

/**
 * Elementwise reciprocal square root, `x ** -0.5`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const rsqrt = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> => pow(self, -0.5)

/**
 * Elementwise reciprocal, `1 / x`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const reciprocal = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> => pow(self, -1)

/**
 * Elementwise `exp(x) - 1`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const expm1 = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const e = yield* exp(self)
    return yield* sub(e, yield* constantLike(e, 1))
  })

/**
 * Elementwise `log(1 + x)`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log1p = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const t = yield* add(self, yield* constantLike(self, 1))
    return yield* log(t)
  })

/**
 * Elementwise base-2 logarithm.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log2 = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const t = yield* log(self)
    return yield* div(t, yield* constantLike(t, Math.LN2))
  })

/**
 * Elementwise base-10 logarithm.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const log10 = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const t = yield* log(self)
    return yield* div(t, yield* constantLike(t, Math.LN10))
  })

/**
 * Elementwise hyperbolic sine, `(exp(x) - exp(-x)) / 2`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sinh = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
export const cosh = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
export const tan = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    return yield* div(yield* sin(self), yield* cos(self))
  })

/**
 * Elementwise not-equal comparison with broadcasting. Returns a `u8` tensor;
 * the 0-d float coercion described by {@link add} also applies.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const ne: {
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
      return yield* maximum(yield* lt(self, other), yield* gt(self, other))
    })
)

/**
 * Elementwise numeric minimum with broadcasting, intended as logical AND for
 * `u8` masks containing only `0` and `1`. This function does not enforce the
 * dtype or mask values, and follows {@link minimum}'s dtype coercion rules.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const logicalAnd = binaryOp("logicalAnd", (a, b) => ({ op: "minimum", inputs: [a, b] }))

/**
 * Elementwise numeric maximum with broadcasting, intended as logical OR for
 * `u8` masks containing only `0` and `1`. This function does not enforce the
 * dtype or mask values, and follows {@link maximum}'s dtype coercion rules.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const logicalOr = binaryOp("logicalOr", (a, b) => ({ op: "maximum", inputs: [a, b] }))

/**
 * Compares every element with zero: `0` becomes `1` and every other value
 * becomes `0`, returning `u8`. The input dtype is not restricted to `u8`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const logicalNot = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.flatMap(constantLike(self, 0), (zero) => eq(self, zero))

/**
 * Elementwise remainder of the division `self / other`, following the sign
 * of the divisor (Python/PyTorch semantics): `self - floor(self / other) * other`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const remainder: {
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
      const q = yield* floor(yield* div(self, other))
      return yield* sub(self, yield* mul(q, other))
    })
)

/**
 * Selects elements from `a` or `b` depending on a `u8` condition tensor,
 * with broadcasting across all three inputs. Condition values are not
 * checked to be `0` or `1`; zero is false and nonzero is true. The value
 * tensors must have exactly matching dtypes and placements. Gradients flow
 * only to the selected side.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const where: {
  (a: Any, b: Any): (cond: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (cond: Any, a: Any, b: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  3,
  (cond: Any, a: Any, b: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("where", () => {
      if (cond.dtype !== "u8") {
        throw new Error(`where: condition must be u8, got ${cond.dtype}`)
      }
      checkCompatible("where", a, b)
      if (cond.placement.id !== a.placement.id) {
        throw new Error("where: condition must use the same placement as its values")
      }
      const shape = broadcastShapes(
        "where",
        broadcastShapes("where", cond.shape, a.shape),
        b.shape
      )
      return {
        request: { op: "whereCond", inputs: [cond, a, b] },
        shape,
        dtype: a.dtype,
        placement: a.placement
      }
    })
)

/**
 * Elementwise logistic sigmoid, `1 / (1 + exp(-x))`, computed as the
 * numerically stable `tanh(x / 2) / 2 + 1/2`.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const sigmoid = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
  /**
   * Mask scores causally: query `i` attends to keys `j <= i`, right-aligned
   * when the key sequence is longer than the query sequence.
   */
  readonly causal?: boolean
  /**
   * Causal attention window including the current token. `null` explicitly
   * selects full causal attention; omission inherits the decode configuration.
   */
  readonly window?: number | null
}

/**
 * Scaled dot-product attention `softmax(q·kᵀ · scale) · v` as a single
 * semantic operation: `q` is `[..., T, D]`, `k` is `[..., S, D]` and `v`
 * is `[..., S, Dv]`. At rank 3 and above, the final leading dimension is
 * the head dimension: query heads may be a divisible multiple of K/V heads,
 * while preceding batch dimensions must match. The output uses the query
 * heads and has shape `[..., Hq, T, Dv]`. The backward is closed-form and recomputes
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("scaledDotProductAttention", () => {
    const op = "scaledDotProductAttention"
    const rank = q.shape.length
    if (rank < 2 || k.shape.length !== rank || v.shape.length !== rank) {
      throw new Error(
        `${op}: q, k and v must share a rank >= 2, got [${q.shape}], [${k.shape}] and [${v.shape}]`
      )
    }
    const leading = q.shape.slice(0, rank < 3 ? -2 : -3)
    if (!leading.every((d, i) => d === k.shape[i]) || !leading.every((d, i) => d === v.shape[i])) {
      throw new Error(
        `${op}: leading dims must match, got [${q.shape}], [${k.shape}] and [${v.shape}]`
      )
    }
    if (rank >= 3) {
      const qHeads = q.shape[rank - 3]
      const kvHeads = k.shape[rank - 3]
      if (v.shape[rank - 3] !== kvHeads) {
        throw new Error(`${op}: k and v heads mismatch, got [${k.shape}] and [${v.shape}]`)
      }
      if (kvHeads === 0 || qHeads % kvHeads !== 0) {
        throw new Error(`${op}: query heads ${qHeads} must be divisible by K/V heads ${kvHeads}`)
      }
    }
    if (q.shape[rank - 1] !== k.shape[rank - 1]) {
      throw new Error(`${op}: q and k head dims mismatch, got [${q.shape}] and [${k.shape}]`)
    }
    if (k.shape[rank - 2] !== v.shape[rank - 2]) {
      throw new Error(`${op}: k and v sequence lengths mismatch, got [${k.shape}] and [${v.shape}]`)
    }
    if (q.dtype !== "f32" && q.dtype !== "f64" && q.dtype !== "bf16") {
      throw new Error(`${op}: dtype must be f32, f64 or bf16, got ${q.dtype}`)
    }
    if (k.dtype !== q.dtype || v.dtype !== q.dtype) {
      throw new Error(`${op}: q, k and v must share a dtype, got ${q.dtype}, ${k.dtype} and ${v.dtype}`)
    }
    if (k.placement.id !== q.placement.id || v.placement.id !== q.placement.id) {
      throw new Error(`${op}: q, k and v must use the same placement`)
    }
    if (options.window !== undefined) {
      if (options.causal !== true) {
        throw new Error(`${op}: window requires causal=true`)
      }
      if (options.window !== null && (!Number.isSafeInteger(options.window) || options.window <= 0)) {
        throw new Error(`${op}: window must be a positive integer or null`)
      }
    }
    const scale = options.scale ?? 1 / Math.sqrt(q.shape[rank - 1])
    return {
      request: {
        op: "scaledDotProductAttention",
        inputs: [q, k, v],
        attributes: {
          scale,
          causal: options.causal ?? false,
          ...(options.window === undefined ? {} : { window: options.window })
        }
      },
      shape: [...q.shape.slice(0, -1), v.shape[rank - 1]],
      dtype: q.dtype,
      placement: q.placement
    }
  })

/**
 * Options for {@link kdaChunk}.
 *
 * @since 0.1.0
 * @category neural network
 */
export interface KdaChunkOptions {
  /** Output multiplier; defaults to `1 / sqrt(headDim)`. */
  readonly scale?: number
}

/**
 * Kimi Delta Attention (KDA) as a single semantic operation: gated
 * delta-rule linear attention evaluated in the chunked parallel form.
 * With the per-channel log decay `g = logDecay` and per-head gate
 * `beta`, each head carries a matrix state `S` of shape `[Dk, Dv]`
 * updated per token as
 * `S_t = (I - beta_t k_t k_tᵀ) Diag(exp(g_t)) S_{t-1} + beta_t k_t v_tᵀ`
 * from a zero initial state, producing `o_t = scale · S_tᵀ q_t`.
 *
 * `q`, `k` and `logDecay` are `[..., H, T, Dk]`, `v` is `[..., H, T, Dv]`
 * and `beta` is `[..., H, T, 1]`, all with equal leading dimensions and
 * a shared dtype; the output is `[..., H, T, Dv]`. `logDecay` holds raw
 * per-channel log decay rates (`<= 0`, before any cumulative summation —
 * the gate activation lives upstream) and `beta` must already lie in
 * `[0, 1]` (e.g. sigmoided). Because positional information is carried
 * by the learnable decayed transition itself, KDA layers use no
 * positional encoding.
 *
 * The implementation computes in f32 (f64 stays f64) with chunk size 64
 * and sub-chunk 16, using the pivot-factored decay and sequential
 * triangular substitution of the reference algorithm — no reciprocal
 * cumulative decay is ever formed. Not yet differentiable.
 *
 * @since 0.1.0
 * @category neural network
 */
export const kdaChunk = (
  q: Any,
  k: Any,
  v: Any,
  logDecay: Any,
  beta: Any,
  options: KdaChunkOptions = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("kdaChunk", () => {
    const op = "kdaChunk"
    const rank = q.shape.length
    if (
      rank < 2 || k.shape.length !== rank || v.shape.length !== rank ||
      logDecay.shape.length !== rank || beta.shape.length !== rank
    ) {
      throw new Error(
        `${op}: q, k, v, logDecay and beta must share a rank >= 2, got [${q.shape}], [${k.shape}], [${v.shape}], [${logDecay.shape}] and [${beta.shape}]`
      )
    }
    if (!k.shape.every((d, i) => d === q.shape[i]) || !logDecay.shape.every((d, i) => d === q.shape[i])) {
      throw new Error(
        `${op}: q, k and logDecay must share a shape, got [${q.shape}], [${k.shape}] and [${logDecay.shape}]`
      )
    }
    if (!v.shape.slice(0, -1).every((d, i) => d === q.shape[i])) {
      throw new Error(`${op}: v must match q on all but the head dim, got [${v.shape}] and [${q.shape}]`)
    }
    const betaShape = [...q.shape.slice(0, -1), 1]
    if (!beta.shape.every((d, i) => d === betaShape[i])) {
      throw new Error(`${op}: beta must have shape [${betaShape}], got [${beta.shape}]`)
    }
    if (q.dtype !== "f32" && q.dtype !== "f64" && q.dtype !== "bf16") {
      throw new Error(`${op}: dtype must be f32, f64 or bf16, got ${q.dtype}`)
    }
    for (const [name, t] of [["k", k], ["v", v], ["logDecay", logDecay], ["beta", beta]] as const) {
      if (t.dtype !== q.dtype) {
        throw new Error(`${op}: all operands must share a dtype, got ${q.dtype} and ${t.dtype} for ${name}`)
      }
      if (t.placement.id !== q.placement.id) {
        throw new Error(`${op}: all operands must use the same placement`)
      }
    }
    const scale = options.scale ?? 1 / Math.sqrt(q.shape[rank - 1])
    return {
      request: {
        op: "kdaChunk",
        inputs: [q, k, v, logDecay, beta],
        attributes: { scale }
      },
      shape: [...q.shape.slice(0, -1), v.shape[rank - 1]],
      dtype: q.dtype,
      placement: q.placement
    }
  })

/**
 * Causal depthwise short convolution over `[..., T, C]` inputs with a
 * `[C, K]` weight as a single semantic operation:
 * `y[t, c] = sum_j weight[c, j] · x[t - K + 1 + j, c]` with zero history
 * (a left zero-padding of `K - 1` tokens). The output has the input's
 * shape. This is the KDA-style local mixing convolution; kept semantic so
 * compiled generation can carry the `K - 1`-token window as per-sequence
 * state instead of re-deriving it from composed ops. Not yet
 * differentiable.
 *
 * @since 0.1.0
 * @category neural network
 */
export const shortConv1d = (
  self: Any,
  weight: Any
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("shortConv1d", () => {
    const op = "shortConv1d"
    const rank = self.shape.length
    if (rank < 2 || weight.shape.length !== 2) {
      throw new Error(
        `${op}: expected input [..., T, C] and weight [C, K], got [${self.shape}] and [${weight.shape}]`
      )
    }
    const channels = self.shape[rank - 1]
    if (weight.shape[0] !== channels) {
      throw new Error(`${op}: weight has ${weight.shape[0]} channels, expected ${channels}`)
    }
    if (weight.shape[1] < 1) {
      throw new Error(`${op}: kernel size must be >= 1, got ${weight.shape[1]}`)
    }
    if (self.dtype !== weight.dtype) {
      throw new Error(`${op}: input and weight must share a dtype, got ${self.dtype} and ${weight.dtype}`)
    }
    if (self.placement.id !== weight.placement.id) {
      throw new Error(`${op}: input and weight must use the same placement`)
    }
    return {
      request: {
        op: "shortConv1d",
        inputs: [self, weight],
        attributes: {}
      },
      shape: self.shape,
      dtype: self.dtype,
      placement: self.placement
    }
  })

/**
 * SiLU / swish activation, `x * sigmoid(x)`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const silu = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    return yield* mul(self, yield* sigmoid(self))
  })

/**
 * Softplus activation, `log(1 + exp(x))`, computed in the numerically
 * stable form `max(x, 0) + log1p(exp(-|x|))`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const softplus = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
  /** Scale of the negative branch; defaults to `1`. */
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
  (self: Any, options: EluOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
      const alpha = options.alpha ?? 1
      const negative = yield* mul(yield* expm1(self), yield* constantLike(self, alpha))
      return yield* where(yield* gt(self, yield* constantLike(self, 0)), self, negative)
    })
)

/**
 * Options for {@link leakyRelu}.
 *
 * @since 0.1.0
 * @category models
 */
export interface LeakyReluOptions {
  /**
   * Negative-branch slope; defaults to `0.01`. Values above `1` do not
   * produce the usual piecewise Leaky ReLU.
   */
  readonly negativeSlope?: number
}

/**
 * Computes `maximum(x, negativeSlope * x)`, with a default slope of `0.01`.
 * This is the usual Leaky ReLU piecewise formula only when
 * `negativeSlope <= 1`; the slope is not otherwise validated.
 *
 * @since 0.1.0
 * @category neural network
 */
export const leakyRelu = dualOptions(
  (self: Any, options: LeakyReluOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
  /** Approximation mode; defaults to the exact `"none"` form. */
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
  (self: Any, options: GeluOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("gelu", () => ({
      request: {
        op: "gelu",
        inputs: [self],
        attributes: { approximate: options.approximate === "tanh" }
      },
      shape: self.shape,
      dtype: self.dtype,
      placement: self.placement
    }))
)

/**
 * Mish activation, `x * tanh(softplus(x))`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const mish = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    return yield* mul(self, yield* tanh(yield* softplus(self)))
  })

/**
 * Options for {@link clamp}. At least one of `min` / `max` must be given.
 *
 * @since 0.1.0
 * @category models
 */
export interface ClampOptions {
  /** Inclusive lower bound. */
  readonly min?: number
  /** Inclusive upper bound. */
  readonly max?: number
}

/**
 * Clamps every element into `[min, max]`; either bound may be omitted.
 *
 * @since 0.1.0
 * @category elementwise
 */
export const clamp = dualOptions(
  (self: Any, options: ClampOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
export const hardtanh = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  clamp(self, { min: -1, max: 1 })

/**
 * Options for {@link dropout}. `p` is the probability of zeroing an
 * element; surviving elements are scaled by `1 / (1 - p)`.
 *
 * @since 0.1.0
 * @category models
 */
export interface DropoutOptions {
  /** Probability of replacing an element with zero; defaults to `0.5`. */
  readonly p?: number
}

/**
 * Applies inverted dropout on every invocation; omit it from evaluation graphs
 * when dropout is disabled. `p` defaults to `0.5`; values below `0` or at least
 * `1` fail, but `NaN` is not rejected. This implementation accepts only `f32`
 * and `f64`. For `p > 0`, the mask has
 * {@link randn}'s sharing and fresh-per-invocation behavior. Submit a loss and
 * its gradients as roots of the same invocation when they must share the mask.
 *
 * @since 0.1.0
 * @category neural network
 */
export const dropout = dualOptions(
  (
    self: Any,
    options: DropoutOptions = {}
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
  (exponent: number): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, exponent: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, exponent: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("pow", () => ({
      request: { op: "pow", inputs: [self], attributes: { exponent } },
      shape: self.shape,
      dtype: self.dtype,
      placement: self.placement
    }))
)

/**
 * Batched matrix multiplication over the last two dimensions, with
 * broadcasting of the leading batch dimensions.
 *
 * @since 0.1.0
 * @category operations
 */
export const matmul: {
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("matmul", () => {
      checkCompatible("matmul", self, other)
      return {
        request: { op: "matmul", inputs: [self, other] },
        shape: matmulShape(self.shape, other.shape),
        dtype: self.dtype,
        placement: self.placement
      }
    })
)

/**
 * Options for reduction operations.
 *
 * @since 0.1.0
 * @category models
 */
export interface ReduceOptions {
  /**
   * Dimensions to reduce. Reduction operations default to every dimension;
   * {@link softmax} and {@link logSoftmax} default to the last dimension.
   * Negative indexes count from the end.
   */
  readonly dims?: ReadonlyArray<number>
  /** Whether reduced dimensions remain with size `1`; defaults to `false`. */
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
  request: (a: Runtime.TensorHandle, dims: ReadonlyArray<number>, keepdims: boolean) => Runtime.NodeRequest
): {
  (options?: ReduceOptions): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, options?: ReduceOptions): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} =>
  dual(
    (args) => args.length === 2 || (args.length === 1 && isTensorHandleValue(args[0])),
    (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
      graphTry(op, () => {
        const dims = options.dims ?? self.shape.map((_, i) => i)
        const keepdims = options.keepdims ?? false
        const normalized = normalizeDims(op, self.shape.length, dims)
        return {
          request: request(self, normalized, keepdims),
          shape: reducedShape(op, self.shape, dims, keepdims),
          dtype: self.dtype,
          placement: self.placement
        }
      })
  )

/**
 * Sums a tensor over the given dimensions (all of them by default). Negative
 * dimensions count from the end.
 *
 * @since 0.1.0
 * @category reductions
 */
export const sum = reduceOp("sum", (self, dims, keepdims) => ({
  op: "sum",
  inputs: [self],
  attributes: { dims, keepdims }
}))

/**
 * Computes the mean of a tensor over the given dimensions (all of them by
 * default). Negative dimensions count from the end.
 *
 * @since 0.1.0
 * @category reductions
 */
export const mean = reduceOp("mean", (self, dims, keepdims) => ({
  op: "mean",
  inputs: [self],
  attributes: { dims, keepdims }
}))

/**
 * Computes the maximum of a tensor over the given dimensions (all of them by
 * default). Negative dimensions count from the end.
 *
 * @since 0.1.0
 * @category reductions
 */
export const max = reduceOp("max", (self, dims, keepdims) => ({
  op: "max",
  inputs: [self],
  attributes: { dims, keepdims }
}))

/**
 * Computes the minimum of a tensor over the given dimensions (all of them by
 * default). Negative dimensions count from the end.
 *
 * @since 0.1.0
 * @category reductions
 */
export const min = reduceOp("min", (self, dims, keepdims) => ({
  op: "min",
  inputs: [self],
  attributes: { dims, keepdims }
}))

/**
 * Returns the indices of the maximum values along `dim` as an `i64` tensor,
 * with `dim` removed from the shape. Not differentiable.
 *
 * @since 0.1.0
 * @category reductions
 */
export const argmax: {
  (dim: number): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("argmax", () => {
      const d = normalizeDim("argmax", self.shape.length, dim)
      return {
        request: { op: "argmax", inputs: [self], attributes: { dim: d } },
        shape: self.shape.filter((_, i) => i !== d),
        dtype: "i64",
        placement: self.placement
      }
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
  (dim: number): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("argmin", () => {
      const d = normalizeDim("argmin", self.shape.length, dim)
      return {
        request: { op: "argmin", inputs: [self], attributes: { dim: d } },
        shape: self.shape.filter((_, i) => i !== d),
        dtype: "i64",
        placement: self.placement
      }
    })
)

/**
 * Cumulative sum along `dim`, preserving the shape.
 *
 * @since 0.1.0
 * @category reductions
 */
export const cumsum: {
  (dim: number): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("cumsum", () => {
      const d = normalizeDim("cumsum", self.shape.length, dim)
      return {
        request: { op: "cumsum", inputs: [self], attributes: { dim: d } },
        shape: self.shape,
        dtype: self.dtype,
        placement: self.placement
      }
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
  /** Value subtracted from the reduced element count; defaults to `1`. */
  readonly correction?: number
}

/**
 * Computes the variance over the given dimensions (all of them by default).
 *
 * @since 0.1.0
 * @category reductions
 */
export const variance = dualOptions(
  (self: Any, options: VarianceOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
  (self: Any, options: VarianceOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
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
  /**
   * Norm order; defaults to `2`. Positive finite values and positive or
   * negative `Infinity` are supported.
   */
  readonly ord?: number
}

/**
 * Computes the p-norm over the given dimensions (all of them by default).
 *
 * @since 0.1.0
 * @category reductions
 */
export const norm = dualOptions(
  (self: Any, options: NormOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
 * Computes the numeric minimum over the given dimensions (all by default),
 * intended as logical AND for a `u8` mask containing only `0` and `1`. The
 * dtype is enforced, but mask values are not.
 *
 * @since 0.1.0
 * @category reductions
 */
export const all = dualOptions(
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
      if (self.dtype !== "u8") {
        return yield* new TensorError({ op: "all", message: `all: expected a u8 tensor, got ${self.dtype}` })
      }
      return yield* min(self, options)
    })
)

/**
 * Computes the numeric maximum over the given dimensions (all by default),
 * intended as logical OR for a `u8` mask containing only `0` and `1`. The
 * dtype is enforced, but mask values are not.
 *
 * @since 0.1.0
 * @category reductions
 */
export const any = dualOptions(
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
  (self: Any, options: ReduceOptions = {}): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
export const prod = reduceOp("prod", (self, dims, keepdims) => ({
  op: "prod",
  inputs: [self],
  attributes: { dims, keepdims }
}))

/**
 * Reshapes a tensor. The total number of elements must stay the same.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const reshape: {
  (shape: ReadonlyArray<number>): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, shape: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, newShape: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("reshape", () => {
      const validShape = validateShape("reshape", newShape)
      if (numel(validShape) !== numel(self.shape)) {
        throw new Error(
          `reshape: cannot reshape [${self.shape}] (${numel(self.shape)} elements) to [${validShape}] (${
            numel(validShape)
          } elements)`
        )
      }
      return {
        request: { op: "reshape", inputs: [self], attributes: { shape: validShape } },
        shape: validShape,
        dtype: self.dtype,
        placement: self.placement
      }
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
  (dims: ReadonlyArray<number>): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, dims: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, dims: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("transpose", () => {
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
      return {
        request: { op: "permute", inputs: [self], attributes: { dims: normalized } },
        shape: outShape,
        dtype: self.dtype,
        placement: self.placement
      }
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
  /**
   * Inclusive start index per dimension; defaults to `0`. Negative indexes
   * count from the end.
   */
  readonly start?: ReadonlyArray<number>
  /**
   * Exclusive end index per dimension; defaults to the dimension extent.
   * Negative indexes count from the end.
   */
  readonly end?: ReadonlyArray<number>
  /** Positive step per dimension; defaults to `1`. */
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
  (options: SliceOptions): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, options: SliceOptions): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, options: SliceOptions): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("slice", () => {
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
      return {
        request: {
          op: "slice",
          inputs: [self],
          attributes: { ranges: ranges.map((range) => [...range]) }
        },
        shape: outShape,
        dtype: self.dtype,
        placement: self.placement
      }
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const [first, ...rest] = tensors
    const dim = options.dim ?? 0
    const rank = first.shape.length
    const axis = dim < 0 ? dim + rank : dim
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank) {
      return yield* new TensorError({
        op: "concat",
        message: `concat: dimension ${dim} out of range for rank ${rank}`
      })
    }
    let out: Any = first
    for (const next of rest) {
      out = yield* graphTry("concat", () => {
        checkCompatible("concat", first, next)
        if (next.shape.length !== rank) {
          throw new Error(`concat: rank mismatch, [${out.shape}] vs [${next.shape}]`)
        }
        for (let i = 0; i < rank; i++) {
          if (i !== axis && out.shape[i] !== next.shape[i]) {
            throw new Error(`concat: shape mismatch at dim ${i}, [${out.shape}] vs [${next.shape}]`)
          }
        }
        return {
          request: { op: "concat", inputs: [out, next], attributes: { dim: axis } },
          shape: out.shape.map((d, i) => (i === axis ? d + next.shape[i] : d)),
          dtype: first.dtype,
          placement: first.placement
        }
      })
    }
    return out as Lazy
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
  (shape: ReadonlyArray<number>): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, shape: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, target: ReadonlyArray<number>): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("broadcastTo", () => {
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
      return {
        request: { op: "broadcastTo", inputs: [self], attributes: { shape: validShape } },
        shape: validShape,
        dtype: self.dtype,
        placement: self.placement
      }
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
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
  (dim: number): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, dim: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(2, (self: Any, dim: number): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
  }))

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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
    return yield* concat(expanded as unknown as [Any, Any, ...Array<Any>], { dim: d })
  })

/**
 * Splits a tensor along `dim`. A numeric `sections` is the maximum size of
 * each chunk, not the number of chunks; the last chunk may be smaller. An
 * array supplies each chunk size and must sum to the dimension size. The
 * array path currently validates only that sum, so callers must supply
 * non-negative integer sizes.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const split = (
  self: Any,
  sections: number | ReadonlyArray<number>,
  options: { readonly dim?: number } = {}
): Effect.Effect<Array<Lazy>, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
      out.push(
        yield* slice(self, {
          start: self.shape.map((_, i) => (i === d ? offset : 0)),
          end: self.shape.map((extent, i) => (i === d ? offset + size : extent))
        })
      )
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
): Effect.Effect<Array<Lazy>, TensorError, Runtime.Runtime> => {
  const dim = options.dim ?? 0
  const d = dim < 0 ? dim + self.shape.length : dim
  const n = Number.isInteger(d) && d >= 0 && d < self.shape.length ? self.shape[d] : 0
  const size = Math.ceil(n / Math.max(1, chunks))
  return split(self, Math.max(1, size), options)
}

/**
 * Tiles with NumPy-style trailing alignment. When `reps` is shorter than the
 * tensor rank, leading dimensions repeat once; extra leading entries add
 * size-1 dimensions before repetition. Every repetition must be a positive
 * integer. When no reshape or repetition is needed, the original handle may
 * be returned, so fresh graph-node or storage identity is not guaranteed.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const tile = (
  self: Any,
  reps: ReadonlyArray<number>
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
  ): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (
    self: Any,
    indexes: Any,
    options?: { readonly dim?: number }
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && isTensorHandleValue(args[1])),
  (
    self: Any,
    indexes: Any,
    options: { readonly dim?: number } = {}
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("take", () => {
      const d = normalizeDim("take", self.shape.length, options.dim ?? 0)
      if (indexes.dtype !== "i64" && indexes.dtype !== "u32") {
        throw new Error(`take: indexes must be i64 or u32, got ${indexes.dtype}`)
      }
      if (indexes.shape.length !== 1) {
        throw new Error(`take: indexes must be 1-D, got shape [${indexes.shape}]`)
      }
      if (indexes.placement.id !== self.placement.id) {
        throw new Error("take: indexes must use the same placement as the input")
      }
      const outShape = [...self.shape]
      outShape[d] = indexes.shape[0]
      return {
        request: { op: "indexSelect", inputs: [self, indexes], attributes: { dim: d } },
        shape: outShape,
        dtype: self.dtype,
        placement: self.placement
      }
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
  ): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (
    self: Any,
    indexes: Any,
    options?: { readonly dim?: number }
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && isTensorHandleValue(args[1])),
  (
    self: Any,
    indexes: Any,
    options: { readonly dim?: number } = {}
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("gather", () => {
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
      if (indexes.placement.id !== self.placement.id) {
        throw new Error("gather: indexes must use the same placement as the input")
      }
      return {
        request: { op: "gather", inputs: [self, indexes], attributes: { dim: d } },
        shape: indexes.shape,
        dtype: self.dtype,
        placement: self.placement
      }
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("scatterAdd", () => {
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
    if (indexes.placement.id !== self.placement.id) {
      throw new Error("scatterAdd: indexes must use the same placement as the input")
    }
    return {
      request: {
        op: "scatterAdd",
        inputs: [self, indexes, src],
        attributes: { dim: d }
      },
      shape: self.shape,
      dtype: self.dtype,
      placement: self.placement
    }
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
 * Expands `i64` or `u32` class indexes into vectors of positive `depth`,
 * appended as a final dimension. Index values are not range-checked: values
 * outside `[0, depth)` produce all-zero vectors.
 *
 * @since 0.1.0
 * @category shape operations
 */
export const oneHot = (
  indexes: Any,
  depth: number,
  options: { readonly dtype?: "f32" | "f64" | "f16" | "bf16" } = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
  ): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (
    self: Any,
    options: { readonly target: Any; readonly ignoreIndex?: number }
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(2, (
  self: Any,
  options: { readonly target: Any; readonly ignoreIndex?: number }
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("crossEntropy", () => {
    const { target } = options
    const ignoreIndex = options.ignoreIndex ?? -100
    if (self.shape.length < 1) {
      throw new Error("crossEntropy: logits must have rank >= 1")
    }
    if (self.dtype !== "f32" && self.dtype !== "f64" && self.dtype !== "bf16") {
      throw new Error(`crossEntropy: logits must be f32, f64 or bf16, got ${self.dtype}`)
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
    if (target.placement.id !== self.placement.id) {
      throw new Error("crossEntropy: target must use the same placement as logits")
    }
    return {
      request: {
        op: "crossEntropy",
        inputs: [self, target],
        attributes: { ignoreIndex }
      },
      shape: [],
      dtype: self.dtype,
      placement: self.placement
    }
  }))

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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const { paddingIndex, weight } = options
    if (weight.shape.length !== 2) {
      return yield* new TensorError({
        op: "embedding",
        message: `embedding: weight must be rank 2 [vocab, hidden], got shape [${weight.shape}]`
      })
    }
    if (weight.dtype !== "f32" && weight.dtype !== "f64" && weight.dtype !== "f16" && weight.dtype !== "bf16") {
      return yield* new TensorError({
        op: "embedding",
        message: `embedding: weight must be a float dtype, got ${weight.dtype}`
      })
    }
    if (indexes.dtype !== "i64" && indexes.dtype !== "u32") {
      return yield* new TensorError({
        op: "embedding",
        message: `embedding: indexes must be i64 or u32, got ${indexes.dtype}`
      })
    }
    if (indexes.placement.id !== weight.placement.id) {
      return yield* new TensorError({
        op: "embedding",
        message: "embedding: indexes and weight must use the same placement"
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
    if (weight.storage !== undefined) {
      return yield* graphTry("embedding", () => ({
        request: {
          op: "quantizedEmbedding",
          inputs: [indexes, weight],
          attributes: {
            encoding: weight.storage!.encoding,
            logicalShape: [vocab, hidden],
            ...(paddingIndex === undefined ? {} : { paddingIndex })
          }
        },
        shape: [...indexes.shape, hidden],
        dtype: "f32",
        placement: weight.placement
      }))
    }
    const n = indexes.shape.reduce((acc, d) => acc * d, 1)
    const flat = indexes.shape.length === 1 ? indexes : yield* reshape(indexes, [n])
    let out: Any = yield* take(weight, flat, { dim: 0 })
    if (paddingIndex !== undefined) {
      const mask = yield* broadcastTo(
        yield* reshape(yield* cast(yield* eq(flat, yield* constantLike(flat, paddingIndex)), weight.dtype), [n, 1]),
        [n, hidden]
      )
      const stopped = yield* graphTry("stopGradient", () => ({
        request: { op: "stopGradient", inputs: [out] },
        shape: out.shape,
        dtype: out.dtype,
        placement: out.placement
      }))
      out = yield* add(yield* sub(out, yield* mul(mask, out)), yield* mul(mask, stopped))
    }
    return yield* reshape(out, [...indexes.shape, hidden])
  })

const triangleMask = (
  op: string,
  self: Any,
  diagonal: number,
  keepUpper: boolean
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> => triangleMask("triu", self, options.diagonal ?? 0, true)
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
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> => triangleMask("tril", self, options.diagonal ?? 0, false)
)

/**
 * Dot product of two rank-1 tensors (`sum(a * b)`), or matrix
 * multiplication when both are rank >= 2 (alias of {@link matmul}).
 *
 * @since 0.1.0
 * @category operations
 */
export const dot: {
  (other: Any): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, other: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
  /** Step between kernel positions; defaults to `1`. */
  readonly stride?: number
  /** Symmetric zero padding on each spatial dimension; defaults to `0`. */
  readonly padding?: number
  /** Spacing between kernel elements; defaults to `1`. */
  readonly dilation?: number
  /** Number of independent channel groups; defaults to `1`. */
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
  Effect.gen(function*() {
    const stride = options.stride ?? 1
    const padding = options.padding ?? 0
    const dilation = options.dilation ?? 1
    const groups = options.groups ?? 1
    if (self.shape.length !== rank + 2 || weight.shape.length !== rank + 2) {
      return yield* new TensorError({
        op,
        message: `${op}: expected rank-${
          rank + 2
        } input and weight, got ranks ${self.shape.length} and ${weight.shape.length}`
      })
    }
    for (
      const [name, value, min] of [["stride", stride, 1], ["padding", padding, 0], [
        "dilation",
        dilation,
        1
      ], ["groups", groups, 1]] as const
    ) {
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
 * 2-D convolution as one semantic operation, lowered to the active backend's
 * convolution implementation and differentiable through native adjoints. `self` is
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
  ): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (
    self: Any,
    weight: Any,
    options?: ConvOptions
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && isTensorHandleValue(args[1])),
  (
    self: Any,
    weight: Any,
    options: ConvOptions = {}
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
      return yield* graphTry("conv2d", () => ({
        request: {
          op: "conv2d",
          inputs: [self, weight],
          attributes: opts
        },
        shape: [self.shape[0], cOut, oh, ow],
        dtype: self.dtype,
        placement: self.placement
      }))
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
  ): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (
    self: Any,
    weight: Any,
    options?: ConvOptions
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && isTensorHandleValue(args[1])),
  (
    self: Any,
    weight: Any,
    options: ConvOptions = {}
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
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
      return yield* graphTry("conv1d", () => ({
        request: {
          op: "conv1d",
          inputs: [self, weight],
          attributes: opts
        },
        shape: [self.shape[0], cOut, ol],
        dtype: self.dtype,
        placement: self.placement
      }))
    })
)

const dilateDim = (
  self: Any,
  dim: number,
  factor: number
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
  /**
   * Extra zeros appended to each output spatial dimension; defaults to `0`
   * and must be less than `stride`.
   */
  readonly outputPadding?: number
}

const convTranspose2dImpl = (
  op: string,
  self: Any,
  weight: Any,
  options: ConvTransposeOptions,
  userPadding: readonly [number, number],
  outputPads: readonly [number, number]
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
    const convGroup = (x: Any, w: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
      Effect.gen(function*() {
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
      out = yield* concat(outs as unknown as [Any, Any, ...Array<Any>], { dim: 1 })
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
  ): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (
    self: Any,
    weight: Any,
    options?: ConvTransposeOptions
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && isTensorHandleValue(args[1])),
  (
    self: Any,
    weight: Any,
    options: ConvTransposeOptions = {}
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
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
  ): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (
    self: Any,
    weight: Any,
    options?: ConvTransposeOptions
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  (args) => args.length === 3 || (args.length === 2 && isTensorHandleValue(args[1])),
  (
    self: Any,
    weight: Any,
    options: ConvTransposeOptions = {}
  ): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
      if (self.shape.length !== 3 || weight.shape.length !== 3) {
        return yield* new TensorError({
          op: "convTranspose1d",
          message:
            `convTranspose1d: expected rank-3 input and weight, got ranks ${self.shape.length} and ${weight.shape.length}`
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
 * Options for {@link maxPool2d} and {@link avgPool2d}. Padding inserts real
 * zeros before windows are reduced: padded zeros can win an all-negative max
 * window and are included in an average's divisor.
 *
 * @since 0.1.0
 * @category models
 */
export interface PoolOptions {
  /** Window size `[KH, KW]`, or one number for a square window. */
  readonly kernelSize: number | readonly [number, number]
  /** Window step; defaults to `kernelSize`. */
  readonly stride?: number | readonly [number, number]
  /** Symmetric zero padding on height and width; defaults to `0`. */
  readonly padding?: number
}

const pool2d = (
  op: string,
  reduce: (t: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>,
  self: Any,
  options: PoolOptions
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
        windows.push(
          yield* slice(padded, {
            start: [0, 0, ky, kx],
            end: [padded.shape[0], padded.shape[1], ky + (oh - 1) * sy + 1, kx + (ow - 1) * sx + 1],
            stride: [1, 1, sy, sx]
          })
        )
      }
    }
    const stacked = yield* stack(
      windows as unknown as [Any, Any, ...Array<Any>],
      { dim: 0 }
    )
    return yield* reduce(stacked)
  })

/**
 * 2-D max pooling over `[N, C, H, W]`, composed from window slices.
 * Gradients are divided evenly among tied maximal elements in each window.
 * Padding contributes zeros, rather than negative infinity, so it can
 * determine boundary maxima for negative inputs.
 *
 * @since 0.1.0
 * @category neural network
 */
export const maxPool2d = (
  self: Any,
  options: PoolOptions
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> => pool2d("maxPool2d", (t) => max(t, { dims: [0] }), self, options)

/**
 * 2-D average pooling over `[N, C, H, W]`, composed from window slices.
 * Padding contributes zeros and remains part of the fixed window divisor.
 *
 * @since 0.1.0
 * @category neural network
 */
export const avgPool2d = (
  self: Any,
  options: PoolOptions
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  pool2d("avgPool2d", (t) => mean(t, { dims: [0] }), self, options)

const checkSquare = (op: string, self: Any): Effect.Effect<void, TensorError> =>
  Effect.gen(function*() {
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
 * Matrix inverse of a tensor square on its last two dimensions; leading
 * dimensions are treated as batch. Currently supported only on CPU and only
 * for `f32` or `f64`; non-CPU graphs fail compilation.
 *
 * @since 0.1.0
 * @category linalg
 */
export const inverse = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* checkSquare("inverse", self)
    return yield* graphTry("inverse", () => ({
      request: { op: "inverse", inputs: [self] },
      shape: self.shape,
      dtype: self.dtype,
      placement: self.placement
    }))
  })

/**
 * Determinant of a tensor square on its last two dimensions, with the leading
 * batch dimensions as output shape. Currently supported only on CPU and only
 * for `f32` or `f64`; non-CPU graphs fail compilation.
 *
 * @since 0.1.0
 * @category linalg
 */
export const det = (self: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* checkSquare("det", self)
    return yield* graphTry("det", () => ({
      request: { op: "det", inputs: [self] },
      shape: self.shape.slice(0, -2),
      dtype: self.dtype,
      placement: self.placement
    }))
  })

/**
 * Solves the linear system `a @ x = b` for `x`, with `a` square on its last
 * two dimensions and `b` of matching rank whose leading dimensions equal
 * `a`'s. Both tensors must share dtype and placement. Currently supported only
 * on CPU and only for `f32` or `f64`; non-CPU graphs fail compilation.
 *
 * @since 0.1.0
 * @category linalg
 */
export const solve: {
  (b: Any): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, b: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, b: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
      yield* checkSquare("solve", self)
      const rank = self.shape.length
      if (
        b.shape.length !== rank ||
        !self.shape.slice(0, -2).every((d, i) => d === b.shape[i]) ||
        b.shape[rank - 2] !== self.shape[rank - 1]
      ) {
        return yield* new TensorError({
          op: "solve",
          message: `solve: expected a right-hand side of matching rank with leading shape [${
            self.shape.slice(0, -1)
          }], got shape [${b.shape}]`
        })
      }
      yield* Effect.try({
        try: () => checkCompatible("solve", self, b),
        catch: (error) =>
          new TensorError({ op: "solve", message: error instanceof Error ? error.message : String(error) })
      })
      return yield* graphTry("solve", () => ({
        request: { op: "solve", inputs: [self, b] },
        shape: b.shape,
        dtype: self.dtype,
        placement: self.placement
      }))
    })
)

/**
 * Converts a tensor to a different dtype. Most mixed-dtype operations require
 * an explicit cast. Binary elementwise operations have one narrow exception:
 * a 0-d float operand is coerced to a non-scalar float operand's dtype, as
 * described by {@link add}.
 *
 * @since 0.1.0
 * @category operations
 */
export const cast: {
  (dtype: DType): (self: Any) => Effect.Effect<Lazy, TensorError, Runtime.Runtime>
  (self: Any, dtype: DType): Effect.Effect<Lazy, TensorError, Runtime.Runtime>
} = dual(
  2,
  (self: Any, dtype: DType): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
    graphTry("cast", () => ({
      request: { op: "cast", inputs: [self], attributes: { dtype } },
      shape: self.shape,
      dtype,
      placement: self.placement
    }))
)
/**
 * Materializes all roots together and returns concrete handles in root order.
 * A nonempty call submits one compile request, possibly served by the native
 * structural cache, and executes one immutable lowered plan. Shared subgraphs
 * run once and each shared random source has one draw across roots. Interruption
 * requests backend cancellation; already-submitted work may be retired safely
 * before resources are reclaimed. Returned outputs retain their escaping
 * storage across later invocations until cleared or finalized. Concrete roots
 * are accepted, but JavaScript identity and storage aliasing are unspecified;
 * every returned handle has independent ownership and may be cleared without
 * consuming its concrete root.
 *
 * @since 0.1.0
 * @category destructors
 */
export const compute = <Roots extends ReadonlyArray<Any>>(
  roots: Roots
): Effect.Effect<{ readonly [K in keyof Roots]: Concrete }, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    if (roots.length === 0) {
      yield* Runtime.Runtime
      return [] as unknown as { readonly [K in keyof Roots]: Concrete }
    }
    const runtime = yield* Runtime.Runtime
    const executable = yield* fromBackend("compile", runtime.compile({ roots }))
    const values = yield* fromBackend(
      "execute",
      runtime.execute(executable, { bindings: [], scalars: [], runtimeValues: {} })
    )
    if (values.length !== roots.length) {
      yield* releaseTensors(runtime, values)
      return yield* new TensorError({
        op: "execute",
        message: `execute: backend returned ${values.length} tensors for ${roots.length} roots`
      })
    }
    const checked = Effect.forEach(values, (value, index) =>
      Effect.try({
        try: () =>
          validateTensorHandle("execute", runtime, value, {
            _tag: "Tensor",
            shape: roots[index].shape,
            dtype: roots[index].dtype,
            ...(roots[index].storage === undefined ? {} : { storage: roots[index].storage }),
            placement: roots[index].placement
          }),
        catch: (error) => caughtTensorError("execute", error)
      }))
    return (yield* preserveOnFailure(checked, releaseTensors(runtime, values))) as {
      readonly [K in keyof Roots]: Concrete
    }
  })

const typedArrayConstructor = (dtype: DType) => {
  switch (dtype) {
    case "f32":
    // f16/bf16 read back as f32: the native side converts before the
    // readback since JS has no half typed arrays we can rely on
    case "f16":
    case "bf16":
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
 * Deterministically releases this concrete handle's ownership and invalidates
 * it and lazy graphs that captured it. Call exactly once. This does not
 * guarantee immediate physical deallocation: aliases, in-flight invocations,
 * exported readback buffers, retained generated bindings or constants, and
 * allocator caches may still retain backing storage. Independently owned
 * handles remain valid.
 *
 * @since 0.1.0
 * @category destructors
 */
export const clear = (self: Concrete): Effect.Effect<void, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    yield* fromBackend("clear", runtime.release(self))
  })

/**
 * Deterministically releases every concrete handle in input order. Each handle
 * must have independent ownership and must be supplied exactly once.
 *
 * @since 0.1.0
 * @category destructors
 */
export const clearAll = (
  tensors: ReadonlyArray<Concrete>
): Effect.Effect<void, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    yield* Effect.forEach(
      tensors,
      (tensor) => fromBackend("clearAll", runtime.release(tensor)),
      { discard: true }
    )
  })

/**
 * Evaluates a tensor and returns its values in a host typed array.
 * `f16` and `bf16` are widened to `Float32Array`; `f32`, `f64`, `i64`, `u8`,
 * and `u32` return `Float32Array`, `Float64Array`, `BigInt64Array`,
 * `Uint8Array`, and `Uint32Array`, respectively. The returned array remains
 * readable after later {@link clear} on the tensor. Its backing `ArrayBuffer`
 * may be a copy or a direct export that retains runtime storage until the
 * buffer is collected. Treat it as readback data; mutation is not a supported
 * way to update a tensor because aliasing is unspecified.
 *
 * @since 0.1.0
 * @category destructors
 */
export const toTypedArray = (self: Any): Effect.Effect<TypedArray, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    if (self.storage !== undefined) {
      return yield* new TensorError({
        op: "toTypedArray",
        message: `toTypedArray: ${self.storage.encoding} storage requires explicit dequantization`
      })
    }
    const runtime = yield* Runtime.Runtime
    const evaluated = isTensor(self) ? self : (yield* compute([self]))[0]
    const buffer = yield* fromBackend("toTypedArray", runtime.readback(evaluated))
    const Ctor = typedArrayConstructor(evaluated.dtype)
    return new Ctor(buffer)
  })

/**
 * Evaluates a tensor and reads its values back as a plain JavaScript number
 * array. Fails with a `TensorError` for `i64` tensors, whose values may not
 * be representable as numbers — use {@link toTypedArray} there and handle
 * bigints explicitly.
 *
 * @since 0.1.0
 * @category destructors
 */
export const toNumberArray = (self: Any): Effect.Effect<Array<number>, TensorError, Runtime.Runtime> =>
  self.dtype === "i64"
    ? new TensorError({
      op: "toNumberArray",
      message: "toNumberArray: i64 tensors may contain values not representable as numbers"
    })
    : Effect.map(toTypedArray(self), (arr) => Array.from(arr as Float32Array | Float64Array | Uint8Array | Uint32Array))

const validateMetadata = (op: string, metadata: Readonly<Record<string, string>>): Readonly<Record<string, string>> => {
  if (typeof metadata !== "object" || metadata === null || Array.isArray(metadata)) {
    throw new TensorError({ op, message: `${op}: metadata must be a record of strings` })
  }
  const output = Object.create(null) as Record<string, string>
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value !== "string") {
      throw new TensorError({ op, message: `${op}: metadata ${JSON.stringify(key)} must be a string` })
    }
    output[key] = value
  }
  return Object.freeze(output)
}

const releaseTensors = (
  runtime: Runtime.RuntimeService,
  values: ReadonlyArray<Concrete>
): Effect.Effect<void> => Effect.ignore(Effect.forEach(values, (value) => runtime.release(value), { discard: true }))

const preserveOnFailure = <A, E, R>(
  effect: Effect.Effect<A, E, R>,
  cleanup: Effect.Effect<void>
): Effect.Effect<A, E, R> => Effect.onExit(effect, (exit) => Exit.isFailure(exit) ? cleanup : Effect.void)

/**
 * Options for direct safetensors writes.
 *
 * @since 0.1.0
 * @category models
 */
export interface SafetensorsOptions {
  /**
   * String metadata stored in the archive; `__metadata__` is reserved as a
   * tensor name.
   */
  readonly metadata?: Readonly<Record<string, string>>
}

/**
 * Materialized tensors and string metadata loaded from a safetensors archive.
 * Each tensor handle owns runtime storage that the caller may release with
 * {@link clear} when no longer needed.
 *
 * @since 0.1.0
 * @category models
 */
export interface SafetensorsArchive {
  /** Null-prototype, frozen record of archive names to materialized tensors. */
  readonly tensors: Readonly<Record<string, Concrete>>
  /** Frozen string metadata record. */
  readonly metadata: Readonly<Record<string, string>>
}

/**
 * Saves tensors through the runtime's optional direct path. All entries are
 * compiled and materialized together before the transfer service serializes
 * the resulting concrete tensors. Encoded tensors are rejected because
 * safetensors cannot preserve their logical storage metadata.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = (
  path: string,
  tensors: Readonly<Record<string, Any>>,
  options: SafetensorsOptions = {}
): Effect.Effect<void, TensorError, Runtime.Runtime> => {
  const entries = Object.entries(tensors)
  if (entries.length === 0) return new TensorError({ op: "save", message: "save: expected at least one tensor" })
  if (entries.some(([name]) => name === "__metadata__")) {
    return new TensorError({ op: "save", message: "save: __metadata__ is reserved and cannot be a tensor name" })
  }
  const encoded = entries.find(([, tensor]) => tensor.storage !== undefined)
  if (encoded !== undefined) {
    return new TensorError({
      op: "save",
      message: `save: encoded tensor ${JSON.stringify(encoded[0])} cannot be represented by safetensors`
    })
  }
  return Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.pathSafetensors
    if (extension === undefined) {
      return yield* new TensorError({
        op: "save",
        message: `save: backend ${runtime.backend.name} does not support path-based safetensors`
      })
    }
    const metadata = yield* Effect.try({
      try: () => validateMetadata("save", options.metadata ?? {}),
      catch: (error) => caughtTensorError("save", error)
    })
    const materialized = yield* compute(entries.map(([, tensor]) => tensor))
    yield* Effect.ensuring(
      fromBackend(
        "save",
        extension.save(path, {
          entries: entries.map(([name], index) => ({ name, tensor: materialized[index] })),
          metadata
        })
      ),
      releaseTensors(runtime, materialized)
    )
  })
}

/**
 * Loads tensors and archive metadata through the runtime's optional direct
 * path. Returned concrete handles own runtime storage; release them with
 * {@link clear} when deterministic cleanup is required.
 *
 * @since 0.1.0
 * @category constructors
 */
export const loadArchive = (
  path: string
): Effect.Effect<SafetensorsArchive, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.pathSafetensors
    if (extension === undefined) {
      return yield* new TensorError({
        op: "loadArchive",
        message: `loadArchive: backend ${runtime.backend.name} does not support path-based safetensors`
      })
    }
    const archive = yield* fromBackend("loadArchive", extension.load(path))
    const candidates: Array<Concrete> = []
    if (typeof archive === "object" && archive !== null && Array.isArray(archive.entries)) {
      for (const entry of archive.entries) {
        if (
          typeof entry === "object" && entry !== null && typeof entry.tensor === "object" && entry.tensor !== null &&
          entry.tensor._tag === "Tensor"
        ) candidates.push(entry.tensor)
      }
    }
    const checked = yield* preserveOnFailure(
      Effect.try({
        try: () => {
          if (typeof archive !== "object" || archive === null || !Array.isArray(archive.entries)) {
            throw new TensorError({ op: "loadArchive", message: "loadArchive: backend returned an invalid archive" })
          }
          const metadata = validateMetadata("loadArchive", archive.metadata)
          const names = new Set<string>()
          const tensors = Object.create(null) as Record<string, Concrete>
          for (const entry of archive.entries) {
            if (
              typeof entry !== "object" || entry === null || typeof entry.name !== "string" ||
              entry.name === "__metadata__" || names.has(entry.name)
            ) {
              throw new TensorError({
                op: "loadArchive",
                message: "loadArchive: backend returned invalid tensor names"
              })
            }
            names.add(entry.name)
            tensors[entry.name] = validateTensorHandle("loadArchive", runtime, entry.tensor, { _tag: "Tensor" })
          }
          return Object.freeze({ tensors: Object.freeze(tensors), metadata })
        },
        catch: (error) => caughtTensorError("loadArchive", error)
      }),
      releaseTensors(runtime, candidates)
    )
    return checked
  })

/**
 * Loads only the tensor record from {@link loadArchive}. Returned concrete
 * handles have the same ownership and cleanup requirements.
 *
 * @since 0.1.0
 * @category constructors
 */
export const load = (
  path: string
): Effect.Effect<Readonly<Record<string, Concrete>>, TensorError, Runtime.Runtime> =>
  Effect.map(loadArchive(path), (archive) => archive.tensors).pipe(
    Effect.mapError((error) =>
      new TensorError({
        op: "load",
        message: error.message,
        ...(error.backend === undefined ? {} : { backend: error.backend })
      })
    )
  )

/**
 * Diagnostics for one JavaScript {@link ProgramCache}. `cached` is the current
 * cache-map size, including ready and pending traces that have not been cleared;
 * `compiled` counts builder trace attempts,
 * including failures and retraces after eviction. Neither field counts native
 * structural-cache misses, native compilations, or backend pipeline entries.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface CompileStats {
  /** Number of ready or in-flight entries currently in the cache. */
  readonly cached: number
  /** Total trace attempts since this cache was created. */
  readonly compiled: number
}

/**
 * Controls tracing and executable compilation for {@link compile}.
 * `optimize` defaults to `true` and records semantics-preserving code-generation
 * regions beside the unchanged semantic graph. `false` disables those optional
 * regions but uses the same typed compiler and executor. `constantWeights`
 * retains captured concrete leaves as inference constants and bypasses bundled
 * runtimes' structural executable cache; declared inputs remain dynamic.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface CompileOptions extends Runtime.ExecutableCompileOptions {
  /**
   * Capacity for this function's ready JavaScript LRU entries. Signatures
   * include runtime identity and each input's shape, dtype, and placement.
   * In-flight entries can temporarily exceed the capacity. Defaults to `32`.
   * This does not bound native structural or pipeline caches.
   */
  readonly cacheCapacity?: number
}

/**
 * A lazily traced, signature-specialized compiled function. The first call for
 * a runtime and input metadata signature traces placeholders and obtains an
 * immutable native executable; later calls reuse its typed lowered instructions
 * and static plans. Lazy arguments are materialized before invocation. Every
 * call has independent workspace, random state, and outputs, so one executable
 * is concurrently callable and later calls do not overwrite live results.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface CompiledFn<E = never, R = never> {
  /**
   * Executes the program selected by the inputs' runtime and metadata
   * signature.
   */
  readonly call: (
    inputs: ReadonlyArray<Any>
  ) => Effect.Effect<Array<Concrete>, TensorError | E, Runtime.Runtime | R>
  /** Snapshot of current entries and cumulative trace attempts. */
  readonly stats: Effect.Effect<CompileStats>
  /**
   * Drops this function's JavaScript cache entries and signature history only;
   * native structural and pipeline caches are unaffected. An already in-flight
   * trace may insert its result afterwards.
   */
  readonly clear: Effect.Effect<void>
}

/**
 * Mutable coordination state for the JavaScript tracing cache. Ready programs
 * use LRU eviction; pending traces can temporarily exceed `capacity`. Concurrent
 * misses for one key share a trace. Eviction and clearing only drop JavaScript
 * wrapper references and do not clear native structural or pipeline caches.
 * Wrapper finalization occurs once no JavaScript references remain, while
 * shareable native artifacts may remain cached.
 * Consumers should use {@link makeProgramCache} and {@link cachedProgram}
 * rather than mutate the exposed collections and counters.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface ProgramCache {
  /** Maximum number of ready programs retained after trace completion. */
  readonly capacity: number
  /** LRU-ordered ready entries and in-flight single-flight entries. */
  readonly entries: Map<string, ProgramCacheEntry>
  /**
   * Signatures observed since creation or the last clear, used for
   * diagnostics.
   */
  keys: Set<string>
  /** Cumulative trace-attempt counter; clearing entries does not reset it. */
  compiled: number
  /** Whether the distinct-signature capacity warning has been emitted. */
  warned: boolean
  /** Snapshot of current entries and cumulative trace attempts. */
  readonly stats: Effect.Effect<CompileStats>
  /**
   * Drops current JavaScript entries and signature history; an in-flight trace
   * may repopulate the cache. Native caches and the cumulative trace count are
   * unchanged.
   */
  readonly clear: Effect.Effect<void>
}

/** Mutable internals — the public surface is {@link ProgramCache}. */
type ProgramCacheState = ProgramCache

type ProgramCacheEntry =
  | { readonly _tag: "ready"; readonly program: CompiledProgram }
  | { readonly _tag: "pending"; readonly deferred: Deferred.Deferred<CompiledProgram, unknown> }

/**
 * Creates empty mutable cache state for {@link cachedProgram}. `capacity`
 * defaults to `32` and should be a non-negative integer; this constructor does
 * not validate it. A zero capacity permits tracing but retains no ready entry.
 *
 * @since 0.1.0
 * @category compilation
 */
export const makeProgramCache = (capacity: number = 32): ProgramCache => {
  const cache: ProgramCacheState = {
    capacity,
    entries: new Map(),
    keys: new Set(),
    compiled: 0,
    warned: false,
    get stats() {
      return Effect.sync(() => ({ cached: cache.entries.size, compiled: cache.compiled }))
    },
    get clear() {
      return Effect.sync(() => {
        // Dropping the last JS wrapper reference permits native finalization;
        // independently cached native artifacts may remain resident.
        cache.entries.clear()
        cache.keys.clear()
      })
    }
  }
  return cache
}

const evictProgramCache = (cache: ProgramCacheState): void => {
  while (cache.entries.size > cache.capacity) {
    let oldest: string | undefined
    for (const [key, entry] of cache.entries) {
      if (entry._tag === "ready") {
        oldest = key
        break
      }
    }
    if (oldest === undefined) {
      return
    }
    cache.entries.delete(oldest)
  }
}

/**
 * Looks up the program for `key`, running `trace` once on a miss.
 * The thunk is only invoked on a miss — cache hits never build a
 * trace effect. Concurrent misses on the same key share one trace
 * (single-flight): the first caller traces, the rest await the same
 * deferred. A failed trace is removed from `entries` but still increments
 * `compiled` and remains in signature history until clear. Ready entries are
 * evicted after successful traces; pending entries can exceed capacity.
 *
 * @since 0.1.0
 * @category compilation
 */
export const cachedProgram = <E, R>(
  cache: ProgramCacheState,
  key: string,
  trace: () => Effect.Effect<CompiledProgram, E, R>
): Effect.Effect<CompiledProgram, TensorError | E, R> =>
  Effect.suspend(() => {
    const hit = cache.entries.get(key)
    if (hit !== undefined) {
      cache.entries.delete(key)
      cache.entries.set(key, hit)
      return hit._tag === "ready"
        ? Effect.succeed(hit.program)
        : Deferred.await(hit.deferred) as Effect.Effect<CompiledProgram, E>
    }
    return Effect.gen(function*() {
      const deferred = yield* Deferred.make<CompiledProgram, unknown>()
      cache.entries.set(key, { _tag: "pending", deferred })
      cache.compiled++
      const isNewSignature = !cache.keys.has(key)
      if (isNewSignature) {
        cache.keys.add(key)
        if (!cache.warned && cache.keys.size > cache.capacity) {
          cache.warned = true
          yield* Effect.logWarning(
            `compile: more than ${cache.capacity} distinct input signatures seen; programs will be re-traced on every cache eviction (check for accidental shape polymorphism)`
          )
        }
      }
      const exit = yield* Effect.exit(trace())
      yield* Deferred.done(deferred, exit)
      if (Exit.isFailure(exit)) {
        cache.entries.delete(key)
        return yield* Effect.failCause(exit.cause)
      }
      cache.entries.set(key, { _tag: "ready", program: exit.value })
      evictProgramCache(cache)
      return exit.value
    })
  })

/** Process-local ids keep runtime object identities out of serialized keys. */
const runtimeSignatureIds = new WeakMap<object, number>()
let nextRuntimeSignatureId = 0

/**
 * Builds a process-local cache key from the object identity of
 * `runtime.identity` and each input's placement id, shape, and dtype. Tensor
 * values and handle identities are not part of the key.
 *
 * @since 0.1.0
 * @category compilation
 */
export const signatureOf = (inputs: ReadonlyArray<Any>, runtime: Runtime.RuntimeService): string => {
  let runtimeId = runtimeSignatureIds.get(runtime.identity)
  if (runtimeId === undefined) {
    runtimeId = nextRuntimeSignatureId++
    runtimeSignatureIds.set(runtime.identity, runtimeId)
  }
  const inputsKey = inputs.map((input) => {
    const storage = input.storage === undefined
      ? "dense"
      : `${input.storage.encoding}:${input.storage.physicalShape.join("x")}:${input.storage.physicalDtype}`
    return `${input.placement.id}:${input.shape.join("x")}:${input.dtype}:${storage}`
  }).join("|")
  return `${runtimeId}|${inputsKey}`
}

/**
 * Creates a tensor placeholder whose exemplar contributes metadata but not its
 * value. Tensor and scalar declarations share one unsigned slot namespace and
 * must form a contiguous range from zero. A slot may appear repeatedly only
 * when every declaration agrees on kind, shape, dtype, and device. Runtime
 * tensor bindings are supplied in ascending slot order with scalar slots omitted.
 *
 * @since 0.1.0
 * @category compilation
 */
export const makeInput = (slot: number, exemplar: Any): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("input", (runtime) => ({
    request: {
      op: "input",
      inputs: [exemplar],
      attributes: {
        slot,
        shape: [...exemplar.shape],
        dtype: exemplar.dtype,
        ...(exemplar.storage === undefined ? {} : { storage: exemplar.storage })
      }
    },
    shape: exemplar.shape,
    dtype: exemplar.dtype,
    ...(exemplar.storage === undefined ? {} : { storage: exemplar.storage }),
    placement: runtime.placement
  }))

/**
 * Creates a 0-d runtime-scalar placeholder. It shares {@link makeInput}'s
 * contiguous slot namespace and repeated-declaration rules. {@link runProgram}
 * receives scalar values separately in ascending scalar-slot order.
 * `CompiledFn.call` has no scalar argument; use the manual placeholder,
 * {@link freezeProgram}, and `runProgram` path for runtime scalars.
 *
 * @since 0.1.0
 * @category compilation
 */
export const makeScalarInput = (
  slot: number,
  dtype: DType
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("scalarInput", (runtime) => ({
    request: { op: "scalarInput", inputs: [], attributes: { slot, dtype } },
    shape: [],
    dtype,
    placement: runtime.placement
  }))

/**
 * Compiles nonempty traced roots into an immutable, concurrently callable
 * executable. Preparation validates the roots and contiguous tensor/scalar
 * slot contract and indexes the semantic graph once. With optimization enabled,
 * fusion, epilogue, and optimizer choices are recorded as side-table regions;
 * compilation does not insert fused semantic nodes or rebuild the graph.
 * Backend lowering then creates the authoritative typed instruction, memory,
 * and physical plans and prepares required artifacts.
 *
 * @since 0.1.0
 * @category compilation
 */
export const freezeProgram = (
  roots: ReadonlyArray<Any>,
  options: Runtime.ExecutableCompileOptions = {}
): Effect.Effect<CompiledProgram, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const handle = yield* fromBackend("compile", runtime.compile({ roots, options }))
    return {
      handle,
      outputs: roots.map((root) => ({
        shape: root.shape,
        dtype: root.dtype,
        ...(root.storage === undefined ? {} : { storage: root.storage }),
        placement: root.placement
      }))
    }
  })

/**
 * Executes an immutable lowered plan; execution performs no graph traversal,
 * optimization, kernel selection, or allocation discovery. Lazy tensor inputs
 * are materialized together and those temporary handles are released after the
 * call. Tensor inputs and scalars bind their declarations in ascending slot
 * order within their separate arrays. The backend validates counts, metadata,
 * layout, placement, and ownership. Inputs are borrowed, while returned outputs
 * retain escaping storage until cleared or finalized; aliasing is unspecified.
 *
 * @since 0.1.0
 * @category compilation
 */
export const runProgram = (
  program: CompiledProgram,
  inputs: ReadonlyArray<Any>,
  scalars: ReadonlyArray<number> = []
): Effect.Effect<Array<Concrete>, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const materialized = inputs.every(isTensor) ? [] : yield* compute(inputs.filter((input) => !isTensor(input)))
    let index = 0
    const concrete = inputs.map((input) => isTensor(input) ? input : materialized[index++]!)
    const values = yield* Effect.ensuring(
      fromBackend(
        "run",
        runtime.execute(program.handle, { bindings: concrete, scalars, runtimeValues: {} })
      ),
      releaseTensors(runtime, materialized)
    )
    if (values.length !== program.outputs.length) {
      yield* releaseTensors(runtime, values)
      return yield* new TensorError({
        op: "run",
        message: `run: backend returned ${values.length} tensors for ${program.outputs.length} program outputs`
      })
    }
    const checked = Effect.forEach(values, (value, index) =>
      Effect.try({
        try: () => validateTensorHandle("run", runtime, value, { _tag: "Tensor", ...program.outputs[index] }),
        catch: (error) => caughtTensorError("run", error)
      }))
    return yield* preserveOnFailure(checked, releaseTensors(runtime, values))
  })

/**
 * A backend-owned decode program with fixed batch width, attention geometry,
 * and output metadata.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface DecodeProgram extends Runtime.DecodeStateSchema {
  /** Opaque handle for the immutable decode-specialized lowered executable. */
  readonly handle: Runtime.ExecutableHandle
  /**
   * Fixed compiled batch width; batched execution accepts at most this many
   * active sequences.
   */
  readonly batch: number
  /** Number of cacheable attention layers in the decode-specialized graph. */
  readonly layers: number
  /** Number of key/value heads per cacheable attention layer. */
  readonly kvHeads: number
  /** Width of each cached key/value head. */
  readonly headDim: number
  /** Number of KDA recurrent layers with per-sequence state. */
  readonly kdaLayers: number
  /** Number of heads per KDA layer. */
  readonly kdaHeads: number
  /** Key width of each KDA head. */
  readonly kdaHeadDim: number
  /** Value width of each KDA head. */
  readonly kdaValueDim: number
  /** Number of short-conv layers with per-sequence window state. */
  readonly convLayers: number
  /** Channel count of each short-conv layer. */
  readonly convChannels: number
  /** Kernel size of each short-conv layer. */
  readonly convKernel: number
  /** Output metadata recorded from the roots at compile time. */
  readonly outputs: CompiledProgram["outputs"]
}

/**
 * A backend-owned KV sequence: a mutable block table and logical token cursor
 * over one {@link KvPool}. Release live sequences deterministically with
 * {@link releaseKvSequence}; native finalization is a fallback.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface KvSequence {
  /** Opaque runtime handle for the live sequence. */
  readonly handle: Runtime.KvSequenceHandle
}

/**
 * A backend-owned KV pool containing the per-layer key/value arenas and prefix
 * cache. There is no explicit pool-release operation; its native storage is
 * finalized after the pool and all sequences backed by it become unreachable.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface KvPool {
  /** Opaque runtime handle for the fixed-capacity pool. */
  readonly handle: Runtime.KvPoolHandle
}

/**
 * Recurrent state allocated for each sequence. Each recurrent family must be
 * either entirely zero or entirely positive. Current native pools store KDA
 * and short-convolution recurrent state as `f32`, independently of KV dtype.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface KvRecurrentGeometry {
  readonly kdaLayers: number
  readonly kdaHeads: number
  readonly kdaHeadDim: number
  readonly kdaValueDim: number
  readonly convLayers: number
  readonly convChannels: number
  readonly convKernel: number
}

/**
 * Allocates a fixed-capacity state pool. `maxTokens` and `blockSize` must be
 * positive integers with exact divisibility. Attention geometry must be either
 * entirely zero, for stateless or recurrent-only programs, or entirely positive. Each
 * recurrent family follows {@link KvRecurrentGeometry}'s corresponding rule.
 * KV storage supports `f32`, `f16`, `bf16`, or quantized `u8`; recurrent state
 * remains `f32`. The pool must exactly match the compiled decode schema.
 *
 * @since 0.1.0
 * @category compilation
 */
export const makeKvPool = (
  layers: number,
  kvHeads: number,
  headDim: number,
  maxTokens: number,
  blockSize: number,
  dtype: DType = "f32",
  recurrent: KvRecurrentGeometry = {
    kdaLayers: 0,
    kdaHeads: 0,
    kdaHeadDim: 0,
    kdaValueDim: 0,
    convLayers: 0,
    convChannels: 0,
    convKernel: 0
  }
): Effect.Effect<KvPool, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.decode
    if (extension === undefined) {
      return yield* new TensorError({
        op: "makeKvPool",
        message: `makeKvPool: backend ${runtime.backend.name} does not support compiled inference`
      })
    }
    const handle = yield* fromBackend(
      "makeKvPool",
      extension.makePool({ layers, kvHeads, headDim, maxTokens, blockSize, dtype, ...recurrent })
    )
    return { handle }
  })

/**
 * Creates an independent live sequence in `pool`.
 *
 * A sequence owns a block table, cursor, and recurrent state. It starts empty;
 * {@link kvPrefillMatch} may attach a resident prefix only for programs without
 * KDA or short-convolution recurrent state. Then execute prefill or decode with {@link runDecodeProgram} or
 * {@link runBatchedDecodeProgram}. Release it exactly once with
 * {@link releaseKvSequence} when it leaves the scheduler.
 *
 * Fails when the current runtime has no decode extension or when `pool` is not
 * owned by that runtime.
 *
 * @since 0.1.0
 * @category compilation
 */
export const makeKvSequence = (pool: KvPool): Effect.Effect<KvSequence, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.decode
    if (extension === undefined) {
      return yield* new TensorError({
        op: "makeKvSequence",
        message: "makeKvSequence: inference extension is unavailable"
      })
    }
    const handle = yield* fromBackend("makeKvSequence", extension.makeSequence(pool.handle))
    return { handle }
  })

/**
 * On a newly created empty sequence, attaches the longest resident whole-block
 * proper KV prefix of `tokens` and returns the matched length. For non-empty input, at
 * least one token is left for execution even when the complete sequence is
 * resident; empty input returns zero.
 *
 * The result is the offset at which prefill should begin. A return value of
 * zero means no reusable prefix was found. Matching updates the sequence's
 * block table and cursor; later decode runs continue from that position. Prefix
 * entries for hybrid pools carry KDA and short-convolution state snapshots at
 * completed block boundaries. Purely recurrent pools without KV blocks return
 * zero.
 *
 * Token values are expected to be non-negative integers representable as
 * `u32`; this wrapper leaves validation to the backend. Fails when the current
 * runtime has no decode extension or when `sequence` is invalid, released, or
 * owned by another runtime.
 *
 * @since 0.1.0
 * @category compilation
 */
export const kvPrefillMatch = (
  sequence: KvSequence,
  tokens: ReadonlyArray<number>
): Effect.Effect<number, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.decode
    return yield* extension === undefined
      ? new TensorError({ op: "prefillMatch", message: "prefillMatch: inference extension is unavailable" })
      : fromBackend("prefillMatch", extension.prefillMatch(sequence.handle, tokens))
  })

/**
 * Returns the current token cursor of `sequence`.
 *
 * The cursor includes tokens supplied by a matched prefix and tokens committed
 * by subsequent prefill or decode runs. It is independent of any active
 * attention window: window eviction may release old blocks without rewinding
 * the logical sequence position.
 *
 * Fails when the current runtime has no decode extension or when `sequence` is
 * invalid, released, or owned by another runtime.
 *
 * @since 0.1.0
 * @category compilation
 */
export const kvSequenceCursor = (sequence: KvSequence): Effect.Effect<number, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.decode
    return yield* extension === undefined
      ? new TensorError({ op: "sequenceCursor", message: "sequenceCursor: inference extension is unavailable" })
      : fromBackend("sequenceCursor", extension.sequenceCursor(sequence.handle))
  })

/**
 * Releases `sequence` and returns every block reference it owns to its pool.
 *
 * The handle is invalid after this Effect succeeds. Callers that manage live
 * generation sessions should release each sequence exactly once; native
 * finalization is only a fallback for abandoned handles.
 *
 * Fails when the current runtime has no decode extension or when `sequence` is
 * invalid, already released, or owned by another runtime.
 *
 * @since 0.1.0
 * @category compilation
 */
export const releaseKvSequence = (sequence: KvSequence): Effect.Effect<void, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.decode
    return yield* extension === undefined
      ? new TensorError({ op: "releaseSequence", message: "releaseSequence: inference extension is unavailable" })
      : fromBackend("releaseSequence", extension.releaseSequence(sequence.handle))
  })

/**
 * Performs decode semantic specialization and compiles the result as an
 * immutable executable. Stateless graphs are allowed. Causal
 * {@link scaledDotProductAttention}, KDA, short convolution, and position
 * operations are converted to their incremental state or cursor forms when
 * present; every attention operation must be causal, and runtime scalar inputs
 * are rejected. Specialization creates one new semantic graph before ordinary
 * single-index compilation; later fusion remains side-table planning. State capacities and
 * `batch` must be positive unsigned 32-bit integers, `blockSize` must divide
 * `maxTokens`, and `window` must be an unsigned 32-bit integer in
 * `1..=maxTokens`. With `state.lastTokenRow`, every root must be
 * `[batch, T, V]` and the program outputs become advance-selected `[V]`
 * rows: one for batch 1, otherwise `batch` rows in row order. Compilation retains captured concrete leaves as
 * constants and therefore bypasses native structural executable caching.
 *
 * @since 0.1.0
 * @category compilation
 */
export const compileDecodeProgram = (
  roots: ReadonlyArray<Any>,
  state: Runtime.DecodeStateRequest
): Effect.Effect<DecodeProgram, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const extension = runtime.extensions.decode
    if (extension === undefined) {
      return yield* new TensorError({
        op: "compileDecode",
        message: `compileDecode: backend ${runtime.backend.name} does not support compiled inference`
      })
    }
    const handle = yield* fromBackend(
      "compileDecode",
      runtime.compile({
        roots,
        options: {
          constantWeights: true
        },
        state
      })
    )
    if (handle.state === undefined) {
      return yield* new TensorError({
        op: "compileDecode",
        message: "compileDecode: backend returned a stateless executable"
      })
    }
    return {
      handle,
      ...handle.state,
      outputs: roots.flatMap((root) => {
        const base = { dtype: root.dtype, placement: root.placement }
        if (state.lastTokenRow !== true) return [{ shape: root.shape, ...base }]
        return Array.from({ length: state.batch }, () => ({ shape: [root.shape[2]!], ...base }))
      })
    }
  })

/**
 * Runs one sequence through an immutable decode executable. Lazy inputs are materialized
 * first, then one native call binds the input slots and executes against the
 * sequence's pool. Program, sequence, pool geometry, runtime, input count, and
 * input metadata must agree. `tokens` contains the real, unpadded token ids
 * represented by the query rows; on success its length advances the cursor
 * and its values feed prefix-cache hashes. Token values and advance limits are
 * backend-validated. Failure or cancellation before commit rolls back cursor,
 * block, and recurrent-state changes. Returned outputs retain their storage
 * independently of the sequence and later invocations.
 *
 * @since 0.1.0
 * @category compilation
 */
export const runDecodeProgram = (
  program: DecodeProgram,
  inputs: ReadonlyArray<Any>,
  seq: KvSequence,
  tokens: ReadonlyArray<number>
): Effect.Effect<Array<Concrete>, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    if (runtime.extensions.decode === undefined) {
      return yield* new TensorError({ op: "decode", message: "decode: inference extension is unavailable" })
    }
    const materialized = inputs.every(isTensor) ? [] : yield* compute(inputs.filter((input) => !isTensor(input)))
    let index = 0
    const concrete = inputs.map((input) => isTensor(input) ? input : materialized[index++]!)
    const values = yield* Effect.ensuring(
      fromBackend(
        "decode",
        runtime.execute(program.handle, {
          bindings: concrete,
          scalars: [],
          runtimeValues: {},
          state: { sequences: [seq.handle], tokens: [tokens] }
        })
      ),
      releaseTensors(runtime, materialized)
    )
    if (values.length !== program.outputs.length) {
      yield* releaseTensors(runtime, values)
      return yield* new TensorError({
        op: "decode",
        message: `decode: backend returned ${values.length} tensors for ${program.outputs.length} program outputs`
      })
    }
    const checked = Effect.forEach(values, (value, index) =>
      Effect.try({
        try: () => validateTensorHandle("decode", runtime, value, { _tag: "Tensor", ...program.outputs[index] }),
        catch: (error) => caughtTensorError("decode", error)
      }))
    return yield* preserveOnFailure(checked, releaseTensors(runtime, values))
  })

/**
 * Runs a frozen batched decode program against one active sequence per batch
 * row. The active count must be from `1` through `program.batch`; the backend
 * pads unused rows to the fixed compiled width. Sequences must be distinct,
 * live, from one compatible pool and runtime, and `tokens` must provide one
 * equally sized, nonempty real-token row per sequence. Every sequence advances
 * by that row length, normally `1` for decode. The highest-slot tensor input
 * may have any positive rank and use the active count as its leading dimension
 * when remaining dimensions match; the backend zero-pads that dimension to the
 * compiled width. Every other input must exactly match its declaration.
 * Returned outputs retain the fixed compiled batch width, so padded rows must
 * be ignored. Constraint or execution failure commits none of the sequences.
 *
 * @since 0.1.0
 * @category compilation
 */
export const runBatchedDecodeProgram = (
  program: DecodeProgram,
  inputs: ReadonlyArray<Any>,
  seqs: ReadonlyArray<KvSequence>,
  tokens: ReadonlyArray<ReadonlyArray<number>>
): Effect.Effect<Array<Concrete>, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    if (runtime.extensions.decode === undefined) {
      return yield* new TensorError({
        op: "decodeBatched",
        message: "decodeBatched: inference extension is unavailable"
      })
    }
    const materialized = inputs.every(isTensor) ? [] : yield* compute(inputs.filter((input) => !isTensor(input)))
    let index = 0
    const concrete = inputs.map((input) => isTensor(input) ? input : materialized[index++]!)
    const values = yield* Effect.ensuring(
      fromBackend(
        "decodeBatched",
        runtime.execute(program.handle, {
          bindings: concrete,
          scalars: [],
          runtimeValues: {},
          state: {
            sequences: seqs.map((sequence) => sequence.handle),
            tokens
          }
        })
      ),
      releaseTensors(runtime, materialized)
    )
    if (values.length !== program.outputs.length) {
      yield* releaseTensors(runtime, values)
      return yield* new TensorError({
        op: "decodeBatched",
        message:
          `decodeBatched: backend returned ${values.length} tensors for ${program.outputs.length} program outputs`
      })
    }
    const checked = Effect.forEach(values, (value, index) =>
      Effect.try({
        try: () => validateTensorHandle("decodeBatched", runtime, value, { _tag: "Tensor", ...program.outputs[index] }),
        catch: (error) => caughtTensorError("decodeBatched", error)
      }))
    return yield* preserveOnFailure(checked, releaseTensors(runtime, values))
  })

/**
 * Fused linear layer as a single semantic operation: `y = x · weight +
 * bias` over the last dim (the addmm epilogue — one gemm launch on
 * Metal).
 *
 * @since 0.1.0
 * @category neural network
 */
export const linear = (
  self: Any,
  weight: Any,
  bias: Any
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const k = self.shape[self.shape.length - 1]
    if (
      self.shape.length < 2 ||
      weight.shape.length !== 2 ||
      weight.shape[0] !== k
    ) {
      return yield* new TensorError({
        op: "linear",
        message: `linear: expected input [.., K] and weight [K, N], got [${self.shape}] x [${weight.shape}]`
      })
    }
    const n = weight.shape[1]
    const flatBias = yield* (
      bias.shape.length === 1 && bias.shape[0] === n
        ? Effect.succeed(bias)
        : bias.shape.length === 2 && bias.shape[0] === 1 && bias.shape[1] === n
        ? reshape(bias, [n])
        : new TensorError({
          op: "linear",
          message: `linear: bias must be [N] or [1, N], got [${bias.shape}] for N ${n}`
        })
    )
    yield* Effect.try({
      try: () => {
        checkCompatible("linear", self, weight)
        checkCompatible("linear", self, flatBias)
      },
      catch: (error) =>
        new TensorError({ op: "linear", message: error instanceof Error ? error.message : String(error) })
    })
    return yield* graphTry("linear", () => ({
      request: { op: "linear", inputs: [self, weight, flatBias] },
      shape: [...self.shape.slice(0, -1), n],
      dtype: self.dtype,
      placement: self.placement
    }))
  })

/**
 * Applies a row-oriented weight `[N, K]` to an input `[..., K]`. Dense
 * weights are transposed into the existing linear/matmul path; encoded
 * weights dispatch to a native packed operation. Bias is optional and may be
 * `[N]` or `[1, N]`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const linearRows = (
  self: Any,
  weight: Any,
  bias?: Any
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const k = self.shape[self.shape.length - 1]
    if (self.shape.length < 2 || weight.shape.length !== 2 || weight.shape[1] !== k) {
      return yield* new TensorError({
        op: "linearRows",
        message: `linearRows: expected input [.., K] and weight [N, K], got [${self.shape}] x [${weight.shape}]`
      })
    }
    const n = weight.shape[0]
    let flatBias: Any | undefined
    if (bias !== undefined) {
      flatBias = yield* (
        bias.shape.length === 1 && bias.shape[0] === n
          ? Effect.succeed(bias)
          : bias.shape.length === 2 && bias.shape[0] === 1 && bias.shape[1] === n
          ? reshape(bias, [n])
          : new TensorError({
            op: "linearRows",
            message: `linearRows: bias must be [N] or [1, N], got [${bias.shape}] for N ${n}`
          })
      )
    }
    yield* Effect.try({
      try: () => {
        checkCompatible("linearRows", self, weight)
        if (flatBias !== undefined) checkCompatible("linearRows", self, flatBias)
      },
      catch: (error) =>
        new TensorError({ op: "linearRows", message: error instanceof Error ? error.message : String(error) })
    })
    if (weight.storage === undefined) {
      const transposed = yield* transpose(weight, [1, 0])
      return flatBias === undefined ? yield* matmul(self, transposed) : yield* linear(self, transposed, flatBias)
    }
    return yield* graphTry("linearRows", () => ({
      request: {
        op: "quantizedLinear",
        inputs: flatBias === undefined ? [self, weight] : [self, weight, flatBias],
        attributes: { encoding: weight.storage!.encoding, logicalShape: [n, k] }
      },
      shape: [...self.shape.slice(0, -1), n],
      dtype: "f32",
      placement: self.placement
    }))
  })

/**
 * Layer normalization over the last dim as a single semantic operation:
 * `y = (x − μ)/√(σ² + eps) · weight + bias`. The semantic node lets the
 * fused Metal kernel evaluate it in one launch (RFC 0007).
 *
 * @since 0.1.0
 * @category neural network
 */
export const layerNorm = (
  self: Any,
  weight: Any,
  bias: Any,
  eps = 1e-5
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("layerNorm", () => {
    const k = weight.shape.length
    const suffix = self.shape.slice(self.shape.length - k)
    if (
      self.shape.length < k ||
      weight.shape.some((dim, i) => suffix[i] !== dim) ||
      bias.shape.length !== k ||
      bias.shape.some((dim, i) => weight.shape[i] !== dim)
    ) {
      throw new Error(
        `layerNorm: weight and bias must match the input's trailing dims [${self.shape}], got [${weight.shape}] and [${bias.shape}]`
      )
    }
    checkCompatible("layerNorm", self, weight)
    checkCompatible("layerNorm", self, bias)
    return {
      request: {
        op: "layerNorm",
        inputs: [self, weight, bias],
        attributes: { eps }
      },
      shape: self.shape,
      dtype: self.dtype,
      placement: self.placement
    }
  })

/**
 * RMS normalization over the last dimension as one semantic operation:
 * `y = x / sqrt(mean(x²) + eps)`, optionally multiplied by `weight`.
 *
 * @since 0.1.0
 * @category neural network
 */
export const rmsNorm = (
  self: Any,
  weight?: Any,
  eps = 1e-6
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("rmsNorm", () => {
    const width = self.shape.at(-1)
    if (width === undefined) throw new Error("rmsNorm: input must have rank at least 1")
    if (weight !== undefined) {
      if (weight.shape.length !== 1 || weight.shape[0] !== width) {
        throw new Error(`rmsNorm: weight must be [${width}], got [${weight.shape}]`)
      }
      checkCompatible("rmsNorm", self, weight)
    }
    return {
      request: {
        op: "rmsNorm",
        inputs: weight === undefined ? [self] : [self, weight],
        attributes: { eps }
      },
      shape: self.shape,
      dtype: self.dtype,
      placement: self.placement
    }
  })

/**
 * Rows `0..seqLen-1` of a `[maxPositions, embeddingDim]` position
 * embedding table, as a single semantic operation (the graph equivalent
 * of gathering `arange(seqLen)`). The semantic node lets decode compilation
 * offset the positions by the runtime cursor. `seqLen` must be a positive
 * integer within the table; this wrapper leaves those checks to the backend.
 *
 * @since 0.1.0
 * @category neural network
 */
export const positionEmbedding = (
  weight: Any,
  seqLen: number
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("positionEmbedding", () => {
    if (weight.shape.length !== 2) {
      throw new Error(
        `positionEmbedding: weight must be [maxPositions, E], got [${weight.shape}]`
      )
    }
    return {
      request: { op: "positionEmbedding", inputs: [weight], attributes: { seqLen } },
      shape: [seqLen, weight.shape[1]],
      dtype: weight.dtype,
      placement: weight.placement
    }
  })

/**
 * Options for {@link rotaryEmbedding}.
 *
 * @since 0.1.0
 * @category neural network
 */
export interface RotaryEmbeddingOptions {
  /** Pairing layout; defaults to GPT-NeoX-style half-split pairs. */
  readonly layout?: "HalfSplit" | "InterleavedPairs"
}

/**
 * Rotary position embedding (RoPE) over the last dimension of a
 * `[..., seqLen, D]` input (D even): positions
 * `0..seqLen-1` rotated by `theta^(-2j/D)`. The semantic node lets decode
 * compilation rebase positions on the runtime cursor. Input rank, `seqLen`,
 * even head width, and dtype constraints are backend-validated. `theta` is
 * passed through without validation and should be positive and finite.
 * Generation beyond a pool's finite capacity additionally requires an
 * attention window that can evict old blocks.
 *
 * @since 0.1.0
 * @category neural network
 */
export const rotaryEmbedding = (
  self: Any,
  seqLen: number,
  theta: number,
  options: RotaryEmbeddingOptions = {}
): Effect.Effect<Lazy, TensorError, Runtime.Runtime> =>
  graphTry("rotaryEmbedding", () => {
    const layout = options.layout ?? "HalfSplit"
    if (layout !== "HalfSplit" && layout !== "InterleavedPairs") {
      throw new Error(`rotaryEmbedding: unsupported layout ${String(layout)}`)
    }
    return {
      request: { op: "rotaryEmbedding", inputs: [self], attributes: { seqLen, theta, layout } },
      shape: self.shape,
      dtype: self.dtype,
      placement: self.placement
    }
  })

/**
 * Creates a lazily traced compiled function; calling `compile` itself does not
 * run `build` or perform native compilation. On the first call for each runtime
 * and ordered input metadata signature, the builder receives
 * lazy placeholders carrying the call inputs' metadata, not their values, and
 * runs for each trace attempt on a cache miss. Its graph, including
 * differentiation or optimizer updates already built into it, is frozen into
 * a native program. Concurrent misses for one signature share an attempt, but
 * failed traces, eviction, or clearing can cause that signature to be traced
 * again. Calls accept lazy or concrete inputs; lazy graphs are materialized
 * before program execution. Random nodes in the semantic graph draw afresh on
 * each run.
 *
 * Retracing is automatic when runtime identity, placement, shape, or dtype
 * differs from every cached signature. Ready programs use least-recently-used
 * eviction at `cacheCapacity`; an evicted signature is traced again if used.
 * Materializing a tensor inside `build` fails at trace time — a compiled
 * builder is a pure graph builder over its placeholders. Runtime-varying
 * scalars require the manual {@link makeScalarInput}, {@link freezeProgram},
 * and {@link runProgram} path because `CompiledFn.call` binds tensors only.
 * The JavaScript signature LRU, native structural executable cache, and backend
 * pipeline caches are distinct; `CompiledFn.clear` clears only the first.
 *
 * @since 0.1.0
 * @category compilation
 */
export const compile = <E = never, R = never>(
  build: (
    inputs: ReadonlyArray<Lazy>
  ) => Effect.Effect<ReadonlyArray<Any>, E, R>,
  options: CompileOptions = {}
): Effect.Effect<CompiledFn<E, R>> =>
  Effect.gen(function*() {
    const cache = makeProgramCache(options.cacheCapacity)
    const trace = (
      inputs: ReadonlyArray<Any>
    ): Effect.Effect<CompiledProgram, TensorError | E, Runtime.Runtime | R> =>
      Effect.gen(function*() {
        const placeholders: Array<Lazy> = []
        for (let i = 0; i < inputs.length; i++) {
          placeholders.push(yield* makeInput(i, inputs[i]))
        }
        const roots = yield* build(placeholders)
        const compileOptions: Runtime.ExecutableCompileOptions = {
          ...(options.optimize === undefined ? {} : { optimize: options.optimize }),
          ...(options.constantWeights === undefined ? {} : { constantWeights: options.constantWeights })
        }
        return yield* freezeProgram(roots, compileOptions)
      })
    const self: CompiledFn<E, R> = {
      call: (inputs) =>
        Effect.gen(function*() {
          const runtime = yield* Runtime.Runtime
          const program = yield* cachedProgram(cache, signatureOf(inputs, runtime), () => trace(inputs))
          return yield* runProgram(program, inputs)
        }),
      get stats() {
        return cache.stats
      },
      get clear() {
        return cache.clear
      }
    }
    return self
  })

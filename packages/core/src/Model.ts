/**
 * Pure model graphs, ordinary compiled execution, and stateful generation.
 *
 * A {@link Model} separates architecture from values. Its `parameterSpecs`
 * catalog defines a stable flat order and how to initialize fresh values, while
 * `forward` extends the current lazy tensor graph from a parameter array and one
 * input. Configuration is captured by constructors rather than stored in a
 * mutable module tree. The resulting graph can be composed, differentiated by
 * {@link Gradient.grad}, and updated by an optimizer without model-specific
 * adapters. Models have no train/eval mode or non-parameter state; notably,
 * {@link dropout} always applies, so evaluation should use a chain without it.
 *
 * There are two distinct compiled paths. {@link Model.execute} lazily traces the
 * ordinary forward graph per runtime and input-metadata signature, retaining a
 * small JavaScript LRU on that model object. It is suitable for repeated
 * stateless evaluation, while `forward` remains the path for composition,
 * training, and differentiation. {@link inference} instead materializes and
 * freezes one parameter generation, eagerly compiles fixed-shape prefill and
 * decode programs, and creates one shared decode-state pool. KV arenas and
 * prefix-cache content are pool-wide, while recurrent state belongs to each
 * sequence. The {@link InferenceProgram} is separate from the model's
 * `execute` cache and not reflected by `Model.stats`.
 *
 * Generation has three ownership levels. The inference artifact retains frozen
 * parameters, immutable native programs, and the shared pool; each
 * {@link Generation} session tracks its own live sequences; each
 * {@link GenerationSeq} owns a mutable cursor, block references, and any
 * recurrent state. Full KV blocks are addressed by chained token-prefix hashes
 * across every session of one artifact. Finished or window-evicted blocks can
 * remain as reclaimable LRU prefix-cache entries, so releasing a sequence drops
 * its live references but does not necessarily erase cached content.
 *
 * Constructors check selected configuration fields. Standard combinators
 * enforce flat parameter arity and unique names. This module does not
 * generally prove that parameter tensors match {@link ParameterSpec}, that a
 * custom {@link Definition} honors its catalog, that token ids fit a model's
 * vocabulary, or that a graph is supported by a particular backend. Those
 * errors remain graph-build, compilation, or execution failures. Training
 * loops live in the `Trainer` module.
 *
 * @since 0.1.0
 */
import { Data, Effect, Exit, Predicate, Semaphore } from "effect"
import * as Gradient from "./Gradient.ts"
import * as Runtime from "./Runtime.ts"
import type * as Speculation from "./Speculation.ts"
import * as Tensor from "./Tensor.ts"

/**
 * A failure in model construction, parameter arity, or serialization, such as
 * invalid layer configuration, duplicate parameter names, an incorrect
 * parameter count, or a missing checkpoint key. Tensor graph, compilation,
 * backend, and ownership failures remain {@link Tensor.TensorError}s.
 *
 * `op` is diagnostic rather than an exhaustive discriminant. In particular,
 * common parameter-arity checks use `"forward"` and identify the calling
 * operation in `message`.
 *
 * @since 0.1.0
 * @category errors
 */
export class ModelError extends Data.TaggedError("ModelError")<{
  /** The operation reporting the failure, such as `linear`, `forward`, or `load`. */
  readonly op: string
  /** Human-readable diagnostic text; branch on the error tag rather than parsing this value. */
  readonly message: string
}> {}

/**
 * The stable name and logical shape of one model parameter.
 *
 * @since 0.1.0
 * @category models
 */
export interface ParameterSpec {
  /** Stable checkpoint key and parameter-array identity. */
  readonly name: string
  /**
   * Declared logical shape, independent of encoded physical storage. The
   * catalog is descriptive: `forward`, {@link save}, and {@link load} do not
   * universally compare supplied tensors against it.
   */
  readonly shape: ReadonlyArray<number>
  /** Declarative recipe for creating one fresh parameter value. */
  readonly initializer: ParameterInitializer
}

/**
 * Declarative recipe for creating one fresh parameter value.
 *
 * @since 0.1.0
 * @category models
 */
export type ParameterInitializer =
  | {
    /** Selects a zero-mean normal draw. */
    readonly _tag: "Normal"
    /** Positive standard-deviation multiplier applied to the unit-normal draw. */
    readonly scale: number
  }
  | {
    /** Selects a constant-filled tensor. */
    readonly _tag: "Constant"
    /** Finite value assigned to every tensor element. */
    readonly value: number
  }

/**
 * A model's parameter values in {@link Model.parameterSpecs} order. The array
 * length is the model arity; parameterless models use `[]`. Values may be lazy
 * graph nodes or materialized tensors unless a narrower API says otherwise.
 *
 * @since 0.1.0
 * @category models
 */
export type Params = ReadonlyArray<Tensor.Any>

/**
 * The stable exposure name of the residual activation after zero-based model
 * layer `layer`. This defines the shared name used by models
 * publishing hidden states via `Tensor.expose` and speculative proposers
 * requesting them via {@link Speculation.HiddenTap}. A model publishes any
 * number of exposures once; any number of proposers may subscribe to any
 * subset of them.
 *
 * @since 0.1.0
 * @category models
 */
export const hiddenExposure = (layer: number): string => `layers.${layer}.hidden`

/**
 * A pure architecture plus a lazily allocated ordinary-execution cache.
 * Parameters are a flat array in `parameterSpecs` order. The model borrows
 * parameter and input handles; ownership transfers only for concrete outputs
 * explicitly returned by `execute` or generation APIs.
 *
 * The parameter catalog records identities and logical shapes but is not a
 * runtime schema validator. Built-in layers validate tensors while constructing
 * their graph; custom definitions are responsible for making `parameterSpecs`
 * and `forward` agree.
 *
 * @since 0.1.0
 * @category models
 */
export interface Model {
  /** Logical parameter specifications in flat parameter-array order. */
  readonly parameterSpecs: ReadonlyArray<ParameterSpec>
  /**
   * Extends the lazy graph: borrowed parameters and one input in, one lazy
   * output out. No evaluation or ownership transfer is implied. Built-in
   * parameterized layers and arity-aware combinators fail with a
   * {@link ModelError} if `params.length` is wrong. A directly invoked
   * parameterless constructor ignores the array, but callers should still pass
   * `[]`; {@link Model.execute} and {@link inference} enforce top-level arity.
   */
  readonly forward: (
    params: Params,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Lazy, ModelError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Runs the ordinary compiled forward path and returns one materialized output.
   * The first call for a runtime and ordered parameter/input metadata signature
   * traces placeholders through `forward`; later calls reuse the immutable
   * executable. The signature contains runtime identity, placement id, shape,
   * dtype, and encoded-storage metadata, but not tensor values or handle
   * identity. Ready entries use a 32-entry LRU. Failed traces, eviction, or
   * clearing can therefore retrace a previously seen signature.
   *
   * Concrete arguments execute in one native program invocation. Lazy arguments
   * are first materialized and can require additional work. Arguments are
   * borrowed and are not retained as constants; materialize a lazy initializer
   * once before an evaluation loop. Calls are independently executable and may
   * overlap. The returned concrete output is caller-owned and should be released
   * with {@link Tensor.clear} when unused. Use `forward`, not
   * `execute`, while building a graph for training or differentiation.
   */
  readonly execute: (
    params: Params,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Concrete, ModelError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Snapshot of this model's ordinary JavaScript signature cache. `cached`
   * includes ready and in-flight entries. `compiled` is the cumulative number
   * of trace attempts, including failures and retraces, not native cold
   * compilations, inference-program compilations, or backend pipeline entries.
   */
  readonly stats: Effect.Effect<Tensor.CompileStats>
  /**
   * Drops current ordinary JavaScript forward-cache entries and signature
   * history. It does not clear parameters, outputs, inference artifacts, native
   * structural/pipeline caches, or the cumulative trace-attempt count. An
   * already in-flight trace may insert its result after this effect completes.
   */
  readonly clear: Effect.Effect<void>
}

/**
 * A custom model definition. {@link define} validates only the parameter
 * catalog: names must be nonempty and unique, shape dimensions must be
 * non-negative safe integers, and initializer values must be finite. It does
 * not execute initializers or `forward`, freeze/copy the supplied arrays, or
 * validate backend support.
 *
 * @since 0.1.0
 * @category models
 */
export interface Definition {
  /** Parameter catalog in the exact flat order accepted by `forward`. */
  readonly parameterSpecs: ReadonlyArray<ParameterSpec>
  /** Pure lazy graph builder; responsible for its own tensor and arity checks. */
  readonly forward: Model["forward"]
}

interface ModelDef {
  readonly parameterSpecs: ReadonlyArray<ParameterSpec>
  readonly forward: Model["forward"]
}

type ModelInternal =
  & {
    -readonly [K in keyof Model]: Model[K]
  }
  & { _fn: Tensor.CompiledFn<ModelError | Tensor.TensorError, Runtime.Runtime> | undefined }

// The shared prototype keeps model values small. Each instance allocates its
// own CompiledFn on first execute; that function then traces once per metadata
// signature, so model construction itself remains runtime- and device-free.
const ModelProto: Pick<Model, "execute" | "stats" | "clear"> & ThisType<ModelInternal> = {
  execute(params, input) {
    const self = this
    return Effect.gen(function*() {
      yield* checkArity("execute", self.parameterSpecs.map((parameter) => parameter.name), params)
      if (self._fn === undefined) {
        self._fn = yield* Tensor.compile<ModelError | Tensor.TensorError, Runtime.Runtime>(
          (inputs) =>
            Effect.map(
              self.forward(inputs.slice(0, -1), inputs[inputs.length - 1]),
              (output) => [output]
            )
        )
      }
      const [output] = yield* self._fn.call([...params, input])
      return output
    })
  },
  get stats() {
    return Effect.suspend(() => this._fn?.stats ?? Effect.succeed({ cached: 0, compiled: 0 }))
  },
  get clear() {
    return Effect.suspend(() => this._fn?.clear ?? Effect.void)
  }
}

const make = (def: ModelDef): Model => {
  // SAFETY: ModelProto supplies the methods; all ModelInternal fields are initialized before return.
  const self = Object.create(ModelProto) as ModelInternal
  self.parameterSpecs = def.parameterSpecs
  self.forward = def.forward
  self._fn = undefined
  return self
}

/**
 * Validates a custom parameter catalog and constructs a model with the standard
 * ordinary compiled-execution path. This does not validate initializer output,
 * forward behavior, tensor shape/dtype compatibility, or inference support.
 *
 * @since 0.1.0
 * @category constructors
 */
export const define = (definition: Definition): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    const seen = new Set<string>()
    for (const parameter of definition.parameterSpecs) {
      if (!Predicate.isString(parameter.name) || parameter.name.length === 0) {
        return yield* new ModelError({ op: "define", message: "parameter name must not be empty" })
      }
      if (seen.has(parameter.name)) {
        return yield* new ModelError({ op: "define", message: `duplicate parameter name: ${parameter.name}` })
      }
      seen.add(parameter.name)
      if (!Array.isArray(parameter.shape)) {
        return yield* new ModelError({ op: "define", message: `${parameter.name}: shape must be an array` })
      }
      for (const dimension of parameter.shape) {
        if (!Number.isSafeInteger(dimension) || dimension < 0) {
          return yield* new ModelError({
            op: "define",
            message: `${parameter.name}: shape dimensions must be non-negative safe integers, got ${dimension}`
          })
        }
      }
      const initializer = parameter.initializer
      if (!Predicate.isObjectOrArray(initializer)) {
        return yield* new ModelError({ op: "define", message: `${parameter.name}: initializer must be an object` })
      }
      if (initializer._tag === "Normal") {
        if (!Number.isFinite(initializer.scale) || initializer.scale <= 0) {
          return yield* new ModelError({
            op: "define",
            message: `${parameter.name}: normal initializer scale must be positive and finite`
          })
        }
      } else if (initializer._tag === "Constant") {
        if (Number.isFinite(initializer.value)) continue
        return yield* new ModelError({
          op: "define",
          message: `${parameter.name}: constant initializer value must be finite`
        })
      } else {
        return yield* new ModelError({
          op: "define",
          message: `${parameter.name}: initializer must be Normal or Constant`
        })
      }
    }
    return make({
      parameterSpecs: definition.parameterSpecs,
      forward: definition.forward
    })
  })

/**
 * Creates one fresh lazy parameter generation from the model's declared
 * initializers, in {@link Model.parameterSpecs} order. Normal draws remain lazy;
 * materialize the returned roots together before retaining them so each draw is
 * sampled once.
 *
 * @since 0.1.0
 * @category constructors
 */
export const initialize = (model: Model): Effect.Effect<Params, Tensor.TensorError, Runtime.Runtime> =>
  Effect.forEach(model.parameterSpecs, (parameter) => {
    const initializer = parameter.initializer
    if (initializer._tag === "Constant") {
      return Tensor.full(parameter.shape, initializer.value)
    }
    return Effect.gen(function*() {
      const drawn = yield* Tensor.randn(parameter.shape)
      return initializer.scale === 1
        ? drawn
        : yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, initializer.scale))
    })
  })

const checkName = (op: string, name: string): Effect.Effect<void, ModelError> =>
  name.length === 0 ? new ModelError({ op, message: "name must not be empty" }) : Effect.void

const checkPositiveInt = (op: string, field: string, value: number): Effect.Effect<void, ModelError> =>
  Number.isInteger(value) && value >= 1
    ? Effect.void
    : new ModelError({ op, message: `${field} must be a positive integer, got ${value}` })

const normal = (scale: number): ParameterInitializer => ({ _tag: "Normal", scale })
const constant = (value: number): ParameterInitializer => ({ _tag: "Constant", value })

const checkArity = (
  who: string,
  names: ReadonlyArray<string>,
  params: Params
): Effect.Effect<void, ModelError> =>
  params.length === names.length
    ? Effect.void
    : new ModelError({
      op: "forward",
      message: `${who}: expected ${names.length} parameters [${names.join(", ")}], got ${params.length}`
    })

const parameterless = (
  apply: (self: Tensor.Any) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>
): Effect.Effect<Model> =>
  Effect.succeed(make({
    parameterSpecs: [],
    forward: (_, input) => apply(input)
  }))

/**
 * A fully-connected layer `add(matmul(input, weight), bias)` with
 * `names = ["<name>.weight", "<name>.bias"]`. The weight is initialized to
 * `randn([inFeatures, outFeatures]) * (1 / sqrt(inFeatures))`, the bias to
 * `zeros([1, outFeatures])`. Fails with a {@link ModelError} if the name
 * is empty or a feature count is not a positive integer.
 *
 * @since 0.1.0
 * @category constructors
 */
export const linear = (
  name: string,
  inFeatures: number,
  outFeatures: number
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    yield* checkName("linear", name)
    yield* checkPositiveInt("linear", "inFeatures", inFeatures)
    yield* checkPositiveInt("linear", "outFeatures", outFeatures)
    const names = [`${name}.weight`, `${name}.bias`]
    return make({
      parameterSpecs: [
        { name: names[0], shape: [inFeatures, outFeatures], initializer: normal(1 / Math.sqrt(inFeatures)) },
        { name: names[1], shape: [1, outFeatures], initializer: constant(0) }
      ],
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          const [weight, bias] = params
          return yield* Tensor.linear(input, weight, bias)
        })
    })
  })

/**
 * A 1-D convolution layer over `[N, C_in, L]` inputs with
 * `names = ["<name>.weight", "<name>.bias"]`. The weight is
 * `[C_out, C_in/groups, K]` initialized to `randn * (1 / sqrt(fan_in))`
 * with `fan_in = (C_in/groups) * K`; the bias is `zeros([C_out])`, added
 * per channel. Stride and dilation must be positive integers, padding a
 * non-negative integer, and groups a positive integer. Fails with a
 * {@link ModelError} on an empty name, channels/kernel/groups that are not
 * positive integers, or channels not divisible into groups; invalid
 * stride, padding, or dilation fails with a {@link Tensor.TensorError}
 * when `forward` builds the convolution.
 *
 * @since 0.1.0
 * @category constructors
 */
export const conv1d = (
  name: string,
  inChannels: number,
  outChannels: number,
  kernelSize: number,
  options: Tensor.ConvOptions = {}
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    yield* checkName("conv1d", name)
    yield* checkPositiveInt("conv1d", "inChannels", inChannels)
    yield* checkPositiveInt("conv1d", "outChannels", outChannels)
    yield* checkPositiveInt("conv1d", "kernelSize", kernelSize)
    const groups = options.groups ?? 1
    yield* checkPositiveInt("conv1d", "groups", groups)
    if (inChannels % groups !== 0 || outChannels % groups !== 0) {
      return yield* new ModelError({
        op: "conv1d",
        message: `channels [${inChannels}, ${outChannels}] are not divisible into ${groups} groups`
      })
    }
    const names = [`${name}.weight`, `${name}.bias`]
    const fanIn = (inChannels / groups) * kernelSize
    return make({
      parameterSpecs: [
        {
          name: names[0],
          shape: [outChannels, inChannels / groups, kernelSize],
          initializer: normal(1 / Math.sqrt(fanIn))
        },
        { name: names[1], shape: [outChannels], initializer: constant(0) }
      ],
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          const [weight, bias] = params
          const out = yield* Tensor.conv1d(input, weight, options)
          return yield* Tensor.add(out, yield* Tensor.reshape(bias, [1, outChannels, 1]))
        })
    })
  })

/**
 * A 2-D convolution layer over `[N, C_in, H, W]` inputs with
 * `names = ["<name>.weight", "<name>.bias"]`. The weight is
 * `[C_out, C_in/groups, KH, KW]` initialized to `randn * (1 / sqrt(fan_in))`
 * with `fan_in = (C_in/groups) * KH * KW`; the bias is `zeros([C_out])`,
 * added per channel. `kernelSize` is a square size or a `[KH, KW]` pair;
 * stride, padding, dilation, and groups come from `options`. Stride and
 * dilation must be positive integers, padding a non-negative integer, and
 * groups a positive integer. Fails with a {@link ModelError} on an empty
 * name, channels/kernel/groups that are not positive integers, or channels
 * not divisible into groups; invalid stride, padding, or dilation fails
 * with a {@link Tensor.TensorError} when `forward` builds the convolution.
 *
 * @since 0.1.0
 * @category constructors
 */
export const conv2d = (
  name: string,
  inChannels: number,
  outChannels: number,
  kernelSize: number | readonly [number, number],
  options: Tensor.ConvOptions = {}
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    yield* checkName("conv2d", name)
    yield* checkPositiveInt("conv2d", "inChannels", inChannels)
    yield* checkPositiveInt("conv2d", "outChannels", outChannels)
    const [kh, kw] = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize] as const
    yield* checkPositiveInt("conv2d", "kernelSize", kh)
    yield* checkPositiveInt("conv2d", "kernelSize", kw)
    const groups = options.groups ?? 1
    yield* checkPositiveInt("conv2d", "groups", groups)
    if (inChannels % groups !== 0 || outChannels % groups !== 0) {
      return yield* new ModelError({
        op: "conv2d",
        message: `channels [${inChannels}, ${outChannels}] are not divisible into ${groups} groups`
      })
    }
    const names = [`${name}.weight`, `${name}.bias`]
    const fanIn = (inChannels / groups) * kh * kw
    return make({
      parameterSpecs: [
        {
          name: names[0],
          shape: [outChannels, inChannels / groups, kh, kw],
          initializer: normal(1 / Math.sqrt(fanIn))
        },
        { name: names[1], shape: [outChannels], initializer: constant(0) }
      ],
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          const [weight, bias] = params
          const out = yield* Tensor.conv2d(input, weight, options)
          return yield* Tensor.add(out, yield* Tensor.reshape(bias, [1, outChannels, 1, 1]))
        })
    })
  })

/**
 * An embedding layer: looks up rows of a `[numEmbeddings, embeddingDim]`
 * weight by `i64` or `u32` indexes of any shape, giving
 * `[...indexes.shape, embeddingDim]`. `names = ["<name>.weight"]`; the
 * weight is initialized to `randn` (unit normal, matching PyTorch).
 * Repeated indexes accumulate weight gradients. With `paddingIndex`, the
 * initialized row is returned normally (it is not zeroed) but receives no
 * gradient. Fails with a {@link ModelError} on an empty name, counts that
 * are not positive integers, or a `paddingIndex` that is not an integer in
 * `[0, numEmbeddings)`. Index dtype, placement, and bounds are checked by the
 * tensor graph/backend rather than by this constructor.
 *
 * @since 0.1.0
 * @category constructors
 */
export const embedding = (
  name: string,
  numEmbeddings: number,
  embeddingDim: number,
  options: { readonly paddingIndex?: number } = {}
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    yield* checkName("embedding", name)
    yield* checkPositiveInt("embedding", "numEmbeddings", numEmbeddings)
    yield* checkPositiveInt("embedding", "embeddingDim", embeddingDim)
    if (
      options.paddingIndex !== undefined &&
      (!Number.isInteger(options.paddingIndex) || options.paddingIndex < 0 ||
        options.paddingIndex >= numEmbeddings)
    ) {
      return yield* new ModelError({
        op: "embedding",
        message: `paddingIndex must be an integer in [0, ${numEmbeddings}), got ${options.paddingIndex}`
      })
    }
    const names = [`${name}.weight`]
    return make({
      parameterSpecs: [{ name: names[0], shape: [numEmbeddings, embeddingDim], initializer: normal(1) }],
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          return yield* Tensor.embedding(input, {
            weight: params[0],
            paddingIndex: options.paddingIndex
          })
        })
    })
  })

/**
 * A learned absolute position embedding (GPT-style `wpe`): looks up rows
 * `0..t-1` of a `[maxPositions, embeddingDim]` table, where `t` is the
 * input's last dimension. It ignores the input's values and leading dimensions.
 * The output is `[t, embeddingDim]` with no copied batch axis.
 * `names = ["<name>.weight"]`, initialized unit-normal. Fails with a
 * {@link ModelError} on an empty name, counts that are not positive
 * integers, or an input whose sequence length exceeds `maxPositions`.
 * A zero-length/rank-zero input reaches the tensor/backend position-operation
 * checks rather than this constructor's upper-bound check.
 * Compiled inference offsets each gather by the sequence cursor, so the
 * total absolute cursor, including positions evaluated for padded prefill
 * chunks, must remain within `maxPositions`; an attention window does not
 * remove that table limit.
 *
 * @since 0.1.0
 * @category constructors
 */
export const positionEmbedding = (
  name: string,
  maxPositions: number,
  embeddingDim: number
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    yield* checkName("positionEmbedding", name)
    yield* checkPositiveInt("positionEmbedding", "maxPositions", maxPositions)
    yield* checkPositiveInt("positionEmbedding", "embeddingDim", embeddingDim)
    const names = [`${name}.weight`]
    return make({
      parameterSpecs: [{ name: names[0], shape: [maxPositions, embeddingDim], initializer: normal(1) }],
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          const t = input.shape.length === 0 ? 0 : input.shape[input.shape.length - 1]
          if (t > maxPositions) {
            return yield* new ModelError({
              op: "positionEmbedding",
              message: `${name}: sequence length ${t} exceeds maxPositions ${maxPositions}`
            })
          }
          return yield* Tensor.positionEmbedding(params[0], t)
        })
    })
  })

/**
 * A layer-normalization layer over the trailing `normalizedShape`
 * dimensions: `(x - mean) / sqrt(var + eps) * weight + bias` with the
 * biased variance and `eps` defaulting to `1e-5`.
 * `names = ["<name>.weight", "<name>.bias"]`, initialized to ones and
 * zeros of `normalizedShape` (a single feature count or a shape). Fails
 * with a {@link ModelError} on an empty name, an empty shape, dimensions
 * that are not positive integers, or a non-positive `eps`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const layerNorm = (
  name: string,
  normalizedShape: number | ReadonlyArray<number>,
  options: { readonly eps?: number } = {}
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    yield* checkName("layerNorm", name)
    const shape: ReadonlyArray<number> = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape]
    if (shape.length === 0) {
      return yield* new ModelError({ op: "layerNorm", message: "normalizedShape must not be empty" })
    }
    for (const dim of shape) {
      yield* checkPositiveInt("layerNorm", "normalizedShape", dim)
    }
    const eps = options.eps ?? 1e-5
    if (!(eps > 0)) {
      return yield* new ModelError({ op: "layerNorm", message: `eps must be positive, got ${eps}` })
    }
    const names = [`${name}.weight`, `${name}.bias`]
    return make({
      parameterSpecs: [
        { name: names[0], shape, initializer: constant(1) },
        { name: names[1], shape, initializer: constant(0) }
      ],
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          const [weight, bias] = params
          return yield* Tensor.layerNorm(input, weight, bias, eps)
        })
    })
  })

/**
 * Options for {@link multiHeadAttention}.
 *
 * @since 0.1.0
 * @category constructors
 */
export interface MultiHeadAttentionOptions {
  /** Whether to mask attention scores causally. Defaults to `false`. */
  readonly causal?: boolean
  /**
   * Applies RoPE to q and k per head with this theta base (for example,
   * `10000`). Use a positive finite base; this constructor passes it through
   * without validation. RoPE has no learned table limit, but a separately
   * composed position embedding still does. In compiled generation, positions
   * are offset by the sequence's absolute cursor. Generation can outgrow the
   * pool's finite row capacity only when every attention operation permits the
   * configured window to evict old blocks. Per-head width and dtype constraints
   * are tensor/backend validation concerns.
   */
  readonly rope?: number
}

/**
 * Multi-head scaled dot-product attention over `[..., T, embedDim]`
 * inputs (GPT-2 style): one fused q/k/v projection, an output projection,
 * the head dimension split across `numHeads` heads, and
 * {@link Tensor.scaledDotProductAttention} per head. Names are exactly
 * `["<name>.qkv.weight", "<name>.qkv.bias", "<name>.wo.weight",
 * "<name>.wo.bias"]`. The qkv weight is `[embedDim, 3 * embedDim]` and
 * its bias `[1, 3 * embedDim]`; the output projection follows
 * {@link linear}. Both weights use `randn * (1 / sqrt(embedDim))` and
 * biases are zero. Fails with a {@link ModelError} on an empty name,
 * counts that are not positive integers, or `embedDim` not divisible by
 * `numHeads`. The constructor does not validate the RoPE theta or head width.
 * RoPE-specific shape, dtype, and numeric checks can fail while building,
 * compiling, or running the graph.
 *
 * @since 0.1.0
 * @category constructors
 */
export const multiHeadAttention = (
  name: string,
  embedDim: number,
  numHeads: number,
  options: MultiHeadAttentionOptions = {}
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    yield* checkName("multiHeadAttention", name)
    yield* checkPositiveInt("multiHeadAttention", "embedDim", embedDim)
    yield* checkPositiveInt("multiHeadAttention", "numHeads", numHeads)
    if (embedDim % numHeads !== 0) {
      return yield* new ModelError({
        op: "multiHeadAttention",
        message: `embedDim ${embedDim} must be divisible by numHeads ${numHeads}`
      })
    }
    const headDim = embedDim / numHeads
    // One [E, 3E] projection exposes q/k/v slices from one semantic matmul
    // instead of constructing three independent [E, E] projections.
    const names = [
      `${name}.qkv.weight`,
      `${name}.qkv.bias`,
      `${name}.wo.weight`,
      `${name}.wo.bias`
    ]
    const causal = options.causal ?? false
    return make({
      parameterSpecs: [
        { name: names[0], shape: [embedDim, 3 * embedDim], initializer: normal(1 / Math.sqrt(embedDim)) },
        { name: names[1], shape: [1, 3 * embedDim], initializer: constant(0) },
        { name: names[2], shape: [embedDim, embedDim], initializer: normal(1 / Math.sqrt(embedDim)) },
        { name: names[3], shape: [1, embedDim], initializer: constant(0) }
      ],
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          const [qkvWeight, qkvBias, woWeight, woBias] = params
          const rank = input.shape.length
          const t = input.shape[rank - 2]
          const leading = input.shape.slice(0, -2)
          // [..., T, E] -> [..., T, H, Dh] -> [..., H, T, Dh]
          const splitHeads = (x: Tensor.Any) =>
            Effect.gen(function*() {
              const reshaped = yield* Tensor.reshape(x, [...leading, t, numHeads, headDim])
              const perm = Array.from({ length: rank + 1 }, (_, i) => i)
              perm[rank - 2] = rank - 1
              perm[rank - 1] = rank - 2
              return yield* Tensor.transpose(reshaped, perm)
            })
          // [..., H, T, Dh] -> [..., T, H, Dh] -> [..., T, E]
          const mergeHeads = (x: Tensor.Any) =>
            Effect.gen(function*() {
              const perm = Array.from({ length: rank + 1 }, (_, i) => i)
              perm[rank - 2] = rank - 1
              perm[rank - 1] = rank - 2
              const transposed = yield* Tensor.transpose(x, perm)
              return yield* Tensor.reshape(transposed, [...leading, t, embedDim])
            })
          const qkv = yield* Tensor.linear(input, qkvWeight, qkvBias)
          const q = yield* Tensor.slice(qkv, {
            start: [...leading.map(() => 0), 0, 0],
            end: [...leading.map((d) => d), t, embedDim]
          })
          const k = yield* Tensor.slice(qkv, {
            start: [...leading.map(() => 0), 0, embedDim],
            end: [...leading.map((d) => d), t, 2 * embedDim]
          })
          const v = yield* Tensor.slice(qkv, {
            start: [...leading.map(() => 0), 0, 2 * embedDim],
            end: [...leading.map((d) => d), t, 3 * embedDim]
          })
          const maybeRope = (x: Tensor.Any) =>
            options.rope !== undefined ? Tensor.rotaryEmbedding(x, t, options.rope) : Effect.succeed(x)
          const attended = yield* Tensor.scaledDotProductAttention(
            yield* maybeRope(yield* splitHeads(q)),
            yield* maybeRope(yield* splitHeads(k)),
            yield* splitHeads(v),
            { causal }
          )
          return yield* Tensor.linear(yield* mergeHeads(attended), woWeight, woBias)
        })
    })
  })

/**
 * Options for {@link kimiDeltaAttention}.
 *
 * @since 0.1.0
 * @category constructors
 */
export interface KimiDeltaAttentionOptions {
  /**
   * Epsilon of the output RMS normalization; defaults to `1e-6`. It is not
   * validated, so callers should provide a positive finite value.
   */
  readonly normEps?: number
}

/**
 * Kimi Delta Attention over `[..., T, embedDim]` inputs (Kimi Linear
 * style): one fused q/k/v projection, a causal depthwise short
 * convolution (kernel 4) plus SiLU over the fused projection, per-head L2
 * normalization of q and k, a low-rank per-channel log-decay gate
 * `logDecay = -exp(aLog) * softplus(fb(fa(x)) + dtBias)`, a sigmoid
 * per-head gate `beta`, the {@link Tensor.kdaChunk} gated delta-rule
 * core, and a sigmoid-gated per-head RMS normalization before the output
 * projection. The head dimension is `embedDim / numHeads` for both keys
 * and values. KDA layers carry positional information in their learnable
 * decayed state transition and apply no positional encoding. In a hybrid
 * stack, the full-attention layers can therefore omit RoPE, as in the
 * Kimi K3 configuration.
 *
 * Names are exactly `["<name>.qkv.weight", "<name>.qkv.bias",
 * "<name>.convqkv.weight", "<name>.fa.weight", "<name>.fb.weight",
 * "<name>.alog", "<name>.dtbias", "<name>.b.weight", "<name>.ga.weight",
 * "<name>.gb.weight", "<name>.norm.weight", "<name>.wo.weight",
 * "<name>.wo.bias"]`. Projection weights use `randn * (1 /
 * sqrt(fanIn))`, the convolution weight `randn * (1 / sqrt(4))`, `alog` and
 * `dtbias` are zero (an initial per-step decay of about `exp(-0.69)`),
 * `norm.weight` is one and biases are zero. Fails with a
 * {@link ModelError} on an empty name, counts that are not positive
 * integers, or `embedDim` not divisible by `numHeads`. The KDA and short-conv
 * cores provide first-order adjoints, so this model is trainable, including
 * mixed-bf16 training on supporting runtimes. Their backward nodes do not
 * provide second-order derivatives. `normEps` is passed through without a
 * finite/positive check; invalid values are not rejected by this constructor.
 *
 * @since 0.1.0
 * @category constructors
 */
export const kimiDeltaAttention = (
  name: string,
  embedDim: number,
  numHeads: number,
  options: KimiDeltaAttentionOptions = {}
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    yield* checkName("kimiDeltaAttention", name)
    yield* checkPositiveInt("kimiDeltaAttention", "embedDim", embedDim)
    yield* checkPositiveInt("kimiDeltaAttention", "numHeads", numHeads)
    if (embedDim % numHeads !== 0) {
      return yield* new ModelError({
        op: "kimiDeltaAttention",
        message: `embedDim ${embedDim} must be divisible by numHeads ${numHeads}`
      })
    }
    const headDim = embedDim / numHeads
    const eps = options.normEps ?? 1e-6
    const names = [
      `${name}.qkv.weight`,
      `${name}.qkv.bias`,
      `${name}.convqkv.weight`,
      `${name}.fa.weight`,
      `${name}.fb.weight`,
      `${name}.alog`,
      `${name}.dtbias`,
      `${name}.b.weight`,
      `${name}.ga.weight`,
      `${name}.gb.weight`,
      `${name}.norm.weight`,
      `${name}.wo.weight`,
      `${name}.wo.bias`
    ]
    return make({
      parameterSpecs: [
        { name: names[0], shape: [embedDim, 3 * embedDim], initializer: normal(1 / Math.sqrt(embedDim)) },
        { name: names[1], shape: [1, 3 * embedDim], initializer: constant(0) },
        { name: names[2], shape: [3 * embedDim, 4], initializer: normal(1 / Math.sqrt(4)) },
        { name: names[3], shape: [embedDim, headDim], initializer: normal(1 / Math.sqrt(embedDim)) },
        { name: names[4], shape: [headDim, embedDim], initializer: normal(1 / Math.sqrt(headDim)) },
        { name: names[5], shape: [numHeads], initializer: constant(0) },
        { name: names[6], shape: [embedDim], initializer: constant(0) },
        { name: names[7], shape: [embedDim, numHeads], initializer: normal(1 / Math.sqrt(embedDim)) },
        { name: names[8], shape: [embedDim, headDim], initializer: normal(1 / Math.sqrt(embedDim)) },
        { name: names[9], shape: [headDim, embedDim], initializer: normal(1 / Math.sqrt(headDim)) },
        { name: names[10], shape: [headDim], initializer: constant(1) },
        { name: names[11], shape: [embedDim, embedDim], initializer: normal(1 / Math.sqrt(embedDim)) },
        { name: names[12], shape: [1, embedDim], initializer: constant(0) }
      ],
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          const [
            qkvWeight,
            qkvBias,
            convqkvWeight,
            faWeight,
            fbWeight,
            aLog,
            dtBias,
            bWeight,
            gaWeight,
            gbWeight,
            normWeight,
            woWeight,
            woBias
          ] = params
          const rank = input.shape.length
          const t = input.shape[rank - 2]
          const leading = input.shape.slice(0, -2)
          // [..., T, E] -> [..., H, T, Dh]
          const splitHeads = (x: Tensor.Any, width: number) =>
            Effect.gen(function*() {
              const reshaped = yield* Tensor.reshape(x, [...leading, t, numHeads, width])
              const perm = Array.from({ length: rank + 1 }, (_, i) => i)
              perm[rank - 2] = rank - 1
              perm[rank - 1] = rank - 2
              return yield* Tensor.transpose(reshaped, perm)
            })
          // [..., H, T, Dh] -> [..., T, E]
          const mergeHeads = (x: Tensor.Any) =>
            Effect.gen(function*() {
              const perm = Array.from({ length: rank + 1 }, (_, i) => i)
              perm[rank - 2] = rank - 1
              perm[rank - 1] = rank - 2
              const transposed = yield* Tensor.transpose(x, perm)
              return yield* Tensor.reshape(transposed, [...leading, t, embedDim])
            })
          // Per-head L2 normalization along the head dim.
          const l2Norm = (x: Tensor.Any) =>
            Effect.gen(function*() {
              const ss = yield* Tensor.sum(yield* Tensor.square(x), { dims: [-1], keepdims: true })
              const epsT = yield* Tensor.constantLike(ss, 1e-6)
              return yield* Tensor.mul(x, yield* Tensor.rsqrt(yield* Tensor.add(ss, epsT)))
            })
          // Apply the causal depthwise kernel to the fused [.., T, 3E]
          // projection before taking q/k/v slices.
          const qkv = yield* Tensor.linear(input, qkvWeight, qkvBias)
          const convolved = yield* Tensor.silu(yield* Tensor.shortConv1d(qkv, convqkvWeight))
          const q = yield* Tensor.slice(convolved, {
            start: [...leading.map(() => 0), 0, 0],
            end: [...leading.map((d) => d), t, embedDim]
          })
          const k = yield* Tensor.slice(convolved, {
            start: [...leading.map(() => 0), 0, embedDim],
            end: [...leading.map((d) => d), t, 2 * embedDim]
          })
          const v = yield* Tensor.slice(convolved, {
            start: [...leading.map(() => 0), 0, 2 * embedDim],
            end: [...leading.map((d) => d), t, 3 * embedDim]
          })
          const qh = yield* l2Norm(yield* splitHeads(q, headDim))
          const kh = yield* l2Norm(yield* splitHeads(k, headDim))
          const vh = yield* splitHeads(v, headDim)
          // Zero biases follow the compute dtype (mixedBf16 runs bf16).
          const zeroBias = (n: number) => Tensor.zeros([1, n], { dtype: input.dtype })
          // Per-channel log decay: -exp(aLog) * softplus(fb(fa(x)) + dtBias).
          const gateHidden = yield* Tensor.linear(input, faWeight, yield* zeroBias(headDim))
          const gateFlat = yield* Tensor.linear(gateHidden, fbWeight, yield* zeroBias(embedDim))
          const gate = yield* splitHeads(gateFlat, headDim)
          const dt = yield* Tensor.reshape(dtBias, [numHeads, 1, headDim])
          const soft = yield* Tensor.softplus(yield* Tensor.add(gate, dt))
          const aExp = yield* Tensor.exp(yield* Tensor.reshape(aLog, [numHeads, 1, 1]))
          const logDecay = yield* Tensor.neg(yield* Tensor.mul(aExp, soft))
          // Per-head beta gate in [0, 1].
          const betaFlat = yield* Tensor.sigmoid(
            yield* Tensor.linear(input, bWeight, yield* zeroBias(numHeads))
          )
          const beta = yield* splitHeads(betaFlat, 1)
          const attended = yield* Tensor.kdaChunk(qh, kh, vh, logDecay, beta)
          // Sigmoid-gated per-head RMS normalization.
          const gateOutHidden = yield* Tensor.linear(input, gaWeight, yield* zeroBias(headDim))
          const gateOut = yield* splitHeads(
            yield* Tensor.sigmoid(
              yield* Tensor.linear(gateOutHidden, gbWeight, yield* zeroBias(embedDim))
            ),
            headDim
          )
          const ms = yield* Tensor.mean(yield* Tensor.square(attended), { dims: [-1], keepdims: true })
          const epsT = yield* Tensor.constantLike(ms, eps)
          const normed = yield* Tensor.mul(
            yield* Tensor.mul(attended, yield* Tensor.rsqrt(yield* Tensor.add(ms, epsT))),
            normWeight
          )
          const gated = yield* Tensor.mul(normed, gateOut)
          return yield* Tensor.linear(yield* mergeHeads(gated), woWeight, woBias)
        })
    })
  })

/**
 * The hyperbolic tangent activation as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const tanh: Effect.Effect<Model> = parameterless(Tensor.tanh)

/**
 * The sigmoid activation as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const sigmoid: Effect.Effect<Model> = parameterless(Tensor.sigmoid)

/**
 * The rectified linear unit activation as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const relu: Effect.Effect<Model> = parameterless(Tensor.relu)

/**
 * The SiLU / swish activation `x * sigmoid(x)` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const silu: Effect.Effect<Model> = parameterless(Tensor.silu)

/**
 * The mish activation `x * tanh(softplus(x))` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const mish: Effect.Effect<Model> = parameterless(Tensor.mish)

/**
 * The softplus activation `log(1 + exp(x))` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const softplus: Effect.Effect<Model> = parameterless(Tensor.softplus)

/**
 * The GELU activation as a parameterless model; `approximate` (`"none"`,
 * the erf form, or `"tanh"`) comes from `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const gelu = (options: Tensor.GeluOptions = {}): Effect.Effect<Model> =>
  parameterless((input) => Tensor.gelu(input, options))

/**
 * The ELU activation as a parameterless model; `alpha` comes from
 * `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const elu = (options: Tensor.EluOptions = {}): Effect.Effect<Model> =>
  parameterless((input) => Tensor.elu(input, options))

/**
 * The leaky-ReLU activation as a parameterless model; `negativeSlope`
 * comes from `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const leakyRelu = (options: Tensor.LeakyReluOptions = {}): Effect.Effect<Model> =>
  parameterless((input) => Tensor.leakyRelu(input, options))

/**
 * Softmax over integer axis `dim` (the last dimension by default) as a
 * parameterless model. The axis must be in range when `forward` is built.
 *
 * @since 0.1.0
 * @category constructors
 */
export const softmax = (dim: number = -1): Effect.Effect<Model> =>
  parameterless((input) => Tensor.softmax(input, { dims: [dim] }))

/**
 * Log-softmax over integer axis `dim` (the last dimension by default) as a
 * parameterless model. The axis must be in range when `forward` is built.
 *
 * @since 0.1.0
 * @category constructors
 */
export const logSoftmax = (dim: number = -1): Effect.Effect<Model> =>
  parameterless((input) => Tensor.logSoftmax(input, { dims: [dim] }))

/**
 * Flattens the input into `[batch, features]` as a parameterless model.
 * `startDim` defaults to `1`, preserving the batch dimension between the
 * convolutional and fully connected parts of a network. `endDim` defaults to
 * the last dimension. Both must be integer axes in range, and `endDim` must not
 * precede `startDim`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const flatten = (
  options: { readonly startDim?: number; readonly endDim?: number } = {}
): Effect.Effect<Model> =>
  parameterless((input) =>
    Tensor.flatten(input, {
      startDim: options.startDim ?? 1,
      endDim: options.endDim
    })
  )

/**
 * Inverted dropout as a parameterless model: zeroes elements with
 * probability `p` (default `0.5`) and scales survivors by `1 / (1 - p)`.
 * This functional form always applies. Build the evaluation chain without it.
 * Dropout adds nothing to the parameter array, so one checkpoint serves both
 * chains. The mask follows
 * {@link Tensor.uniform}'s per-invocation sharing rule: submit a loss and its
 * gradients as roots of the same invocation when they must share it. Fails
 * with a {@link ModelError} if `p` is numerically outside `[0, 1)`. This is not
 * a full finite-number check: `NaN` currently passes through. Input dtype is
 * checked by {@link Tensor.dropout}, which currently accepts f32 and f64 only.
 *
 * @since 0.1.0
 * @category constructors
 */
export const dropout = (options: Tensor.DropoutOptions = {}): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    const p = options.p ?? 0.5
    if (p < 0 || p >= 1) {
      return yield* new ModelError({ op: "dropout", message: `p must be in [0, 1), got ${p}` })
    }
    return make({
      parameterSpecs: [],
      forward: (_, input) => Tensor.dropout(input, { p })
    })
  })

const pool = (
  op: string,
  apply: (
    self: Tensor.Any,
    options: Tensor.PoolOptions
  ) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>,
  options: Tensor.PoolOptions
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    const [kh, kw] = Array.isArray(options.kernelSize)
      ? options.kernelSize
      : [options.kernelSize, options.kernelSize] as const
    yield* checkPositiveInt(op, "kernelSize", kh)
    yield* checkPositiveInt(op, "kernelSize", kw)
    if (options.stride !== undefined) {
      const [sh, sw] = Array.isArray(options.stride)
        ? options.stride
        : [options.stride, options.stride] as const
      yield* checkPositiveInt(op, "stride", sh)
      yield* checkPositiveInt(op, "stride", sw)
    }
    if (options.padding !== undefined && (!Number.isInteger(options.padding) || options.padding < 0)) {
      return yield* new ModelError({
        op,
        message: `padding must be a non-negative integer, got ${options.padding}`
      })
    }
    return make({
      parameterSpecs: [],
      forward: (_, input) => apply(input, options)
    })
  })

/**
 * 2-D max pooling as a parameterless model; `kernelSize` (a square size
 * or a `[KH, KW]` pair), `stride`, and `padding` come from `options`.
 * Fails with a {@link ModelError} unless kernel and stride sizes are
 * positive integers and padding is a non-negative integer.
 *
 * @since 0.1.0
 * @category constructors
 */
export const maxPool2d = (options: Tensor.PoolOptions): Effect.Effect<Model, ModelError> =>
  pool("maxPool2d", Tensor.maxPool2d, options)

/**
 * 2-D average pooling as a parameterless model; `kernelSize` (a square
 * size or a `[KH, KW]` pair), `stride`, and `padding` come from
 * `options`. Fails with a {@link ModelError} unless kernel and stride sizes
 * are positive integers and padding is a non-negative integer.
 *
 * @since 0.1.0
 * @category constructors
 */
export const avgPool2d = (options: Tensor.PoolOptions): Effect.Effect<Model, ModelError> =>
  pool("avgPool2d", Tensor.avgPool2d, options)

/**
 * Wraps a sub-model in a gradient-checkpoint boundary. The forward value
 * is unchanged, but during the backward pass the sub-model's forward
 * intermediates are recomputed from a fresh copy instead of retained. This
 * trades one extra forward evaluation of the block for lower peak activation
 * memory. Region inputs stay shared, including parameters, the incoming
 * activation, and constructor draws. Recomputation therefore matches the
 * forward pass.
 *
 * Apply it per block, not to the whole model. Checkpointing the full
 * network just moves the peak into the backward pass. The standard
 * recipe is one boundary per expensive stage:
 *
 * ```ts
 * Model.chain(
 *   yield* Model.checkpoint(yield* block1),
 *   yield* Model.checkpoint(yield* block2),
 *   head
 * )
 * ```
 *
 * This recomputation works on every target.
 *
 * @since 0.1.0
 * @category combinators
 */
export const checkpoint = (model: Model): Effect.Effect<Model> =>
  Effect.succeed(make({
    parameterSpecs: model.parameterSpecs,
    forward: (params, input) => Effect.flatMap(model.forward(params, input), Gradient.checkpoint)
  }))

/**
 * Adds a residual (skip) connection around a sub-model. Its forward pass is
 * `input + block(input)`. Parameter specs are the sub-model's; the
 * sub-model's output must be broadcast-compatible with its input. Transformer
 * blocks and ResNet stages normally use equal shapes.
 *
 * @since 0.1.0
 * @category combinators
 */
export const residual = (model: Model): Effect.Effect<Model> =>
  Effect.succeed(make({
    parameterSpecs: model.parameterSpecs,
    forward: (params, input) =>
      Effect.gen(function*() {
        const out = yield* model.forward(params, input)
        return yield* Tensor.add(input, out)
      })
  }))

/**
 * Transforms a model's input before it enters the sub-model:
 * `forward(params, input) = model.forward(params, f(input))`. Parameter
 * specs are the sub-model's. Use it for input derived from the raw
 * input's shape or values when no dedicated layer covers the case. Position
 * embeddings have their own {@link positionEmbedding} layer.
 *
 * @since 0.1.0
 * @category combinators
 */
export const mapInput = (
  model: Model,
  f: (input: Tensor.Any) => Effect.Effect<Tensor.Any, Tensor.TensorError, Runtime.Runtime>
): Effect.Effect<Model> =>
  Effect.succeed(make({
    parameterSpecs: model.parameterSpecs,
    forward: (params, input) => Effect.flatMap(f(input), (mapped) => model.forward(params, mapped))
  }))

/**
 * Fans one input into several sub-models and combines their outputs:
 * `forward(params, input) = f(...models.map(m => m.forward(mParams,
 * input)))`. Parameter specs are concatenated in model order and sliced
 * by arity in `forward`. The combiner is variadic with one argument per model, in
 * the same order (inferred from the tuple). Fails with a
 * {@link ModelError} when the array is empty or when parameter names
 * collide.
 *
 * Adding branches, such as token and position embeddings, has its own
 * combinator, {@link add}. {@link residual} is
 * the special case where one branch is the identity.
 *
 * @since 0.1.0
 * @category combinators
 */
export const merge = <const M extends ReadonlyArray<Model>>(
  models: M,
  f: (...outputs: { -readonly [K in keyof M]: Tensor.Lazy }) => Effect.Effect<
    Tensor.Lazy,
    Tensor.TensorError,
    Runtime.Runtime
  >
): Effect.Effect<Model, ModelError> => {
  if (models.length === 0) {
    return new ModelError({ op: "merge", message: "at least one model is required" })
  }
  const parameterSpecs = models.flatMap((model) => model.parameterSpecs)
  const names = parameterSpecs.map((parameter) => parameter.name)
  const seen = new Set<string>()
  const duplicates = new Set<string>()
  for (const name of names) {
    if (seen.has(name)) {
      duplicates.add(name)
    }
    seen.add(name)
  }
  if (duplicates.size > 0) {
    return new ModelError({
      op: "merge",
      message: `duplicate parameter names: [${[...duplicates].join(", ")}]`
    })
  }
  const arities = models.map((model) => model.parameterSpecs.length)
  return Effect.succeed(make({
    parameterSpecs,
    forward: (params, input) =>
      Effect.gen(function*() {
        yield* checkArity("merge", names, params)
        const outputs: Array<Tensor.Lazy> = []
        let offset = 0
        for (let i = 0; i < models.length; i++) {
          outputs.push(yield* models[i].forward(params.slice(offset, offset + arities[i]), input))
          offset += arities[i]
        }
        // SAFETY: the loop adds exactly one output for every combiner parameter, in the same order.
        return yield* f(...(outputs as Parameters<typeof f>))
      })
  }))
}

/**
 * Adds the outputs of several models over a shared input elementwise:
 * `forward(params, input) = Σᵢ models[i].forward(paramsᵢ, input)` with
 * each model's parameters sliced by arity from the concatenated array.
 * For example, token and position embeddings use `add(wte, wpe)`.
 * {@link residual} is the special case where one branch
 * is the identity. Parameter specs follow {@link merge}. Fails with a
 * {@link ModelError} when the chain is empty or parameter names collide.
 *
 * @since 0.1.0
 * @category combinators
 */
export const add = (...models: ReadonlyArray<Model>): Effect.Effect<Model, ModelError> =>
  merge(models, (first, ...rest) =>
    Effect.gen(function*() {
      let acc = first
      for (const output of rest) {
        acc = yield* Tensor.add(acc, output)
      }
      return acc
    }))

/**
 * Composes models into a single model that threads its input through each
 * child in order, slicing each child's share of the concatenated
 * parameter array by its parameter-spec arity. The result concatenates the
 * children's parameter specs.
 *
 * Fails with a {@link ModelError} when the chain is empty or when parameter
 * names collide. A collision would silently overwrite entries in a saved
 * checkpoint.
 *
 * @since 0.1.0
 * @category combinators
 */
export const chain = (...models: ReadonlyArray<Model>): Effect.Effect<Model, ModelError> => {
  if (models.length === 0) {
    return new ModelError({ op: "chain", message: "at least one model is required" })
  }
  const parameterSpecs = models.flatMap((model) => model.parameterSpecs)
  const names = parameterSpecs.map((parameter) => parameter.name)
  const seen = new Set<string>()
  const duplicates = new Set<string>()
  for (const name of names) {
    if (seen.has(name)) {
      duplicates.add(name)
    }
    seen.add(name)
  }
  if (duplicates.size > 0) {
    return new ModelError({
      op: "chain",
      message: `duplicate parameter names: [${[...duplicates].join(", ")}]`
    })
  }
  const arities = models.map((model) => model.parameterSpecs.length)
  return Effect.succeed(make({
    parameterSpecs,
    forward: (params, input) =>
      Effect.gen(function*() {
        yield* checkArity("chain", names, params)
        let current = yield* models[0].forward(params.slice(0, arities[0]), input)
        let offset = arities[0]
        for (let i = 1; i < models.length; i++) {
          current = yield* models[i].forward(params.slice(offset, offset + arities[i]), current)
          offset += arities[i]
        }
        return current
      })
  }))
}

/**
 * Saves a model's parameters to a safetensors file, zipping parameter-spec names
 * with the parameter array into the record {@link Tensor.save} takes.
 * Fails with a {@link ModelError} if the parameter array's length does
 * not match the model's arity. It does not compare tensor shapes or dtypes with
 * {@link Model.parameterSpecs}. Saving borrows parameters and does not clear them.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = (
  model: Model,
  params: Params,
  path: string
): Effect.Effect<void, ModelError | Tensor.TensorError, Runtime.Runtime> =>
  params.length !== model.parameterSpecs.length
    ? new ModelError({
      op: "save",
      message: `model has ${model.parameterSpecs.length} parameters, got ${params.length}`
    })
    : Tensor.save(
      path,
      Object.fromEntries(model.parameterSpecs.map((parameter, i) => [parameter.name, params[i]]))
    )

/**
 * Loads a safetensors file and returns tensors selected by parameter-spec names in
 * parameter-array order. A missing key fails with a {@link ModelError}; extra
 * keys are ignored. This maps names and arity but does not validate the
 * architecture. It leaves shape, dtype, storage, and placement compatibility
 * unchecked until first use.
 *
 * {@link Tensor.load} materializes the entire archive. This function releases
 * unselected tensors before success and releases all imported tensors if
 * validation fails or is interrupted. On success, the selected handles are
 * caller-owned and should be released with
 * {@link Tensor.clear} when no longer needed.
 *
 * @since 0.1.0
 * @category destructors
 */
export const load = (
  model: Model,
  path: string
): Effect.Effect<ReadonlyArray<Tensor.Concrete>, ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.flatMap(Tensor.load(path), (record) =>
    Effect.onExit(
      Effect.gen(function*() {
        const params: Array<Tensor.Concrete> = []
        for (const { name } of model.parameterSpecs) {
          const param = record[name]
          if (param === undefined) {
            return yield* new ModelError({
              op: "load",
              message: `missing parameter "${name}" in ${path}`
            })
          }
          params.push(param)
        }
        const retained = new Set(params)
        for (const tensor of Object.values(record)) {
          if (retained.has(tensor)) continue
          yield* Tensor.clear(tensor)
        }
        return params
      }),
      (exit) => Exit.isFailure(exit) ? Tensor.clearAll(Object.values(record)) : Effect.void
    ))

/**
 * A failure in inference-artifact construction or generation: invalid
 * configuration or model structure, or misuse of the generation calling
 * convention. Current operation labels include `inference`, `add`, `prefill`,
 * and `step`; treat `message` as a diagnostic rather than a stable protocol.
 * Decode compilation and pool-construction tensor errors are wrapped as
 * `InferenceError("inference")`. Errors raised earlier by `model.forward`, and
 * tensor/backend failures during `add`, `step`, cursor, or cleanup, retain their
 * original types.
 *
 * @since 0.1.0
 * @category errors
 */
export class InferenceError extends Data.TaggedError("InferenceError")<{
  /** The inference phase reporting the failure. */
  readonly op: string
  /** Human-readable diagnostic text; branch on the error tag and `op` rather than parsing it. */
  readonly message: string
}> {}

/**
 * Fixed deployment geometry for {@link inference}. Construction validates
 * these scalar fields, then eagerly traces and compiles one prefill program
 * per `prefillChunks` width `[batchSize, chunk]` and fixed-width decode
 * `[batchSize, 1]`. Batch size one
 * uses the same decode path. There is no later shape-specialization cache.
 *
 * Validation checks structure only. It does not estimate whether the
 * pool is large enough for a particular set of prompts, check token ids against
 * the model vocabulary, prove that every model operation supports decode
 * specialization, or prove that learned position tables cover future cursors.
 * Those constraints fail when the graph is compiled or a sequence is run.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface InferenceConfig {
  /**
   * Fixed pool capacity in token rows, shared by live sequences and
   * unreferenced prefix-cache blocks across every session of the artifact.
   * Must be a positive integer and an exact multiple of `blockSize`. Without
   * an effective attention window it also bounds each sequence cursor; with a
   * window, aggregate live frontiers can still exhaust the shared pool.
   */
  readonly maxTokens: number
  /**
   * KV paging granularity in tokens. Must be a positive integer that
   * divides `maxTokens`. Defaults to 16.
   */
  readonly blockSize?: number
  /**
   * Requested positive attention-retention window, no greater than
   * `maxTokens`. Omit for full history. Decode specialization permits block
   * eviction only if every attention operation resolves to bounded local
   * attention; an explicit full-attention operation makes the compiled program
   * retain full history. The effective window is part of the compiled geometry.
   *
   * With cursor-offset RoPE and no separately bounded absolute-position state,
   * eviction can let a sequence advance beyond `maxTokens` while retaining only
   * its live window and partial frontier. It does not reset the logical cursor,
   * expand a learned position table, or guarantee enough aggregate pool capacity.
   */
  readonly attentionWindow?: number
  /**
   * Fixed prompt-chunk token widths in ascending order. The compiler creates
   * one prefill program per entry. The runtime serves each prompt chunk from
   * the largest compiled width covering its remaining tokens, so smaller widths
   * only bound zero-padding waste on short prompts. Entries must be positive
   * safe integers; they need not be multiples of `blockSize`. Every prefill
   * invocation has one of the compiled shapes. The final suffix is
   * zero-padded, but only its real token ids advance the sequence, enter
   * state hashes, and select the returned logits row. Graph operations still
   * evaluate the padded extent, so a cursor-offset learned position table
   * must cover the largest compiled chunk at every invocation.
   */
  readonly prefillChunks: ReadonlyArray<number>
  /**
   * Token-tensor dtype used by all fixed programs. Defaults to `"u32"`;
   * prompts passed to {@link Generation.add} must match exactly. Decode state
   * and prefix hashes are u32-based even for `"i64"`, so prompt and step ids
   * must still be non-negative and fit u32.
   */
  readonly tokenDtype?: "u32" | "i64"
  /**
   * KV storage dtype. Defaults to `"f32"`; `"f16"` and `"bf16"` narrow
   * rows on write and attention widens them to f32. `"int8"` uses symmetric
   * per-token, per-head quantization with f32 scales. KDA and short-convolution
   * recurrent state remains f32 and is not controlled by this option.
   */
  readonly kvDtype?: "f32" | "f16" | "bf16" | "int8"
  /** Default sampling controls for generation. Defaults to `{ seed: 0 }`. */
  readonly sampling?: GenerationSamplingOptions
  /**
   * Positive fixed decode width, maximum live sequences tracked by each
   * session, and maximum active entries in one step. Defaults to `8`. The one
   * decode program has shape `[batchSize, 1]`; batch size one is the ordinary
   * single-sequence case. This is not a global limit across sessions; all
   * sessions still compete for one pool's token-row capacity.
   */
  readonly batchSize?: number
  /** Optional high-level proposer compiled with this target. */
  readonly speculation?: {
    /** Proposer artifact whose vocabulary and target contract must match this model. */
    readonly proposer: Speculation.Artifact
    /** Maximum proposal width, bounded by the artifact's trained capacity. */
    readonly maxDraftTokens: number
    /** Proposal-width policy; only `"fixed"` is currently implemented. */
    readonly schedule?: "fixed" | "adaptive"
  } | undefined
}

/**
 * One mutable sequence owned by a {@link Generation} session. Its backend state
 * consists of an absolute logical cursor, KV block references when attention is
 * present, and per-sequence KDA/short-convolution state when present. It is an
 * ordinary value rather than a scoped resource.
 *
 * Call {@link GenerationSeq.finish} when the sequence leaves a scheduler, or
 * {@link Generation.close} for all sequences in that session. Releasing drops
 * live references; completed blocks may remain in the artifact's reclaimable
 * prefix cache. Native finalization is only a fallback for abandoned handles.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationSeq {
  /** Runtime discriminant for a sampled-generation sequence. */
  readonly _tag: "GenerationSeq"
  /**
   * Returns the total logical token count, including evicted window
   * positions. Fails after the underlying sequence has been released.
   */
  readonly cursor: () => Effect.Effect<number, Tensor.TensorError, Runtime.Runtime>
  /**
   * Removes this sequence from its session and releases its backend state.
   * Completed KV blocks can become prefix-cache entries rather than immediately
   * free blocks. Calls after it has already been finished or closed are no-ops.
   */
  readonly finish: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * Sampling controls owned by generation; draw counters are sequence-managed.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationSamplingOptions {
  /** Non-negative temperature; `0` selects greedy sampling. */
  readonly temperature?: number
  /** Non-negative candidate count; `0` disables top-k filtering. */
  readonly topK?: number
  /** Nucleus probability in `(0, 1]`; `1` disables top-p filtering. */
  readonly topP?: number
  /** Unsigned 64-bit seed. Safe integer numbers remain accepted for convenience. */
  readonly seed: bigint | number
}

/**
 * One prompt admitted by {@link Generation.add}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationAdd {
  /** Nonempty `[1, T]` token tensor matching the artifact's token dtype and placement. */
  readonly prompt: Tensor.Any
  /** Overrides inference sampling defaults for the admission page only. */
  readonly sampling?: Partial<GenerationSamplingOptions>
  /** Optional positive generation limit for this sequence. */
  readonly maxTokens?: number | undefined
  /** Token ids that terminate this sequence when sampled. */
  readonly eosTokens?: ReadonlyArray<number>
}

/**
 * One live sequence selected by {@link Generation.step}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationStep {
  /** Sequence whose pending token is committed. */
  readonly seq: GenerationSeq
  /** Overrides inference sampling defaults for this page only. */
  readonly sampling?: Partial<GenerationSamplingOptions>
}

/**
 * A nonempty page of sampled tokens for one sequence.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface TokenPage {
  /** Sequence that owns this page. */
  readonly seq: GenerationSeq
  /** Sampled token ids in generation order. */
  readonly tokens: ReadonlyArray<number>
  /** Terminal policy reached by the final token, when the page ends the sequence. */
  readonly stopReason?: "eos" | "maxTokens" | undefined
}

/**
 * A caller-scheduled generation session over one {@link InferenceProgram}.
 * {@link Generation.add} creates and prefills independent sequences and samples
 * their first token. {@link Generation.step} commits each sequence's pending
 * token and samples its successor. Every active count uses the fixed
 * `[batchSize, 1]` program with explicit inactive lanes.
 *
 * Prefix matching spans the pool, not just one session. It uses chained hashes
 * to reuse the longest resident proper prefix made of complete `blockSize` blocks,
 * whether those blocks are referenced by another live sequence or retained
 * unreferenced in the LRU cache. At least the final prompt token is always
 * executed so `add` can sample the first pending token. Hybrid KV/recurrent programs also
 * require a published recurrent snapshot at the matched block boundary and
 * restore it with the KV blocks. Programs without KV blocks have no block
 * anchor and therefore no prefix match. This includes purely recurrent and
 * stateless graphs.
 *
 * Sessions are ordinary values and require no `Scope`. Sessions from the same
 * artifact may run concurrently and share pool capacity/cache content. Calls to
 * `add` and `step` on one session are serialized. `finish`, `cursor`, and
 * `close` are outside that JavaScript lock, so callers must not overlap them
 * with admission or stepping on the same session/sequence. Native sequence
 * locks are a safety backstop, not a supported concurrent lifecycle API.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface Generation {
  /**
   * Atomically admits a nonempty array of prompts. Capacity and policy are
   * validated for every entry before any sequence is allocated. Results preserve
   * input order and ordinary generation returns one token per page.
   */
  readonly add: (
    entries: ReadonlyArray<GenerationAdd>
  ) => Effect.Effect<ReadonlyArray<TokenPage>, InferenceError | ModelError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Commits each selected sequence's pending token and samples one successor.
   * Terminal sequences fail validation before native execution.
   */
  readonly step: (
    entries: ReadonlyArray<GenerationStep>
  ) => Effect.Effect<ReadonlyArray<TokenPage>, InferenceError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Returns this session's JavaScript live-sequence count. This is not a pool
   * capacity, global-session, or prefix-cache statistic.
   */
  readonly live: () => Effect.Effect<number>
  /**
   * Closes the native session and releases all live sequences atomically. A
   * successful close invalidates previously returned sequences and the session
   * accepts no later additions or rounds. Native finalizers remain a fallback.
   */
  readonly close: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * A caller-driven stateful sequence used only by {@link StatefulExecution}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface StatefulExecutionSeq {
  /** Runtime discriminant for a caller-token execution sequence. */
  readonly _tag: "StatefulExecutionSeq"
  /** Underlying decode-state sequence handle. */
  readonly sequence: Tensor.KvSequence
  /** Returns the sequence's absolute logical token count. */
  readonly cursor: () => Effect.Effect<number, Tensor.TensorError, Runtime.Runtime>
  /** Releases this sequence; repeated calls are no-ops. */
  readonly finish: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * Lower-level stateful logits execution for custom host samplers. Unlike
 * {@link Generation}, callers select tokens and own each returned logits row.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface StatefulExecution {
  /** Prefills nonempty prompts and returns one sequence and caller-owned logits row per prompt. */
  readonly add: (
    prompts: ReadonlyArray<Tensor.Any>
  ) => Effect.Effect<
    ReadonlyArray<{ readonly seq: StatefulExecutionSeq; readonly logits: Tensor.Concrete }>,
    InferenceError | ModelError | Tensor.TensorError,
    Runtime.Runtime
  >
  /** Commits one caller-selected token per sequence and returns caller-owned successor logits. */
  readonly step: (
    entries: ReadonlyArray<{ readonly seq: StatefulExecutionSeq; readonly token: number }>
  ) => Effect.Effect<ReadonlyArray<Tensor.Concrete>, InferenceError | Tensor.TensorError, Runtime.Runtime>
  /** Returns this session's current live-sequence count. */
  readonly live: () => Effect.Effect<number>
  /** Closes the session and releases all of its live sequences. */
  readonly close: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * An immutable decode-specialized artifact. It retains one materialized
 * parameter generation as native constants, fixed prefill/decode executables,
 * and one shared decode-state pool. Its KV arenas and prefix cache are shared
 * across sessions, while each sequence owns its mutable recurrent state. It is neither
 * a {@link Model} nor part of `Model.execute`'s signature cache.
 *
 * The artifact is safe to share: immutable programs can run concurrently and
 * different sessions coordinate through the native pool. It has no explicit
 * release or `Scope` lifetime. Programs, frozen constants, and pool storage are
 * finalized when the artifact and dependent sequence handles become
 * unreachable. Sequence state is the capacity-sensitive resource that callers
 * can release deterministically through {@link GenerationSeq.finish} or
 * {@link Generation.close}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface InferenceProgram {
  /**
   * Opens an empty caller-scheduled session. This allocates JavaScript
   * coordination state, not a private pool; all sessions share the artifact's
   * pool capacity and prefix cache. No `Scope` service is required. Use
   * {@link Generation.close} for deterministic session cleanup or
   * {@link GenerationSeq.finish} for one sequence.
   */
  readonly generation: () => Effect.Effect<Generation, InferenceError>
  /** Opens a lower-level caller-token/caller-owned-logits session. */
  readonly execution: () => Effect.Effect<StatefulExecution, InferenceError>
  /** Native generation counters, phase timings, acceptance, and pool pressure. */
  readonly diagnostics: () => Effect.Effect<Runtime.InferenceDiagnostics, Tensor.TensorError>
}

interface ResolvedInferenceConfig {
  readonly maxTokens: number
  readonly blockSize: number
  readonly prefillChunks: ReadonlyArray<number>
  readonly tokenDtype: "u32" | "i64"
  readonly kvDtype: Tensor.DType
  readonly batchSize: number
  readonly sampling: GenerationSamplingOptions
  readonly attentionWindow: number | undefined
  readonly speculation: {
    readonly proposer: Speculation.Artifact
    readonly maxDraftTokens: number
  } | undefined
}

const invalidInferenceConfig = (message: string): InferenceError => new InferenceError({ op: "inference", message })

const resolveInferenceConfig = (
  config: InferenceConfig
): Effect.Effect<ResolvedInferenceConfig, InferenceError> =>
  Effect.gen(function*() {
    const blockSize = config.blockSize ?? 16
    if (!Number.isInteger(blockSize) || blockSize <= 0) {
      return yield* invalidInferenceConfig(`blockSize must be a positive integer, got ${config.blockSize}`)
    }
    if (
      !Number.isInteger(config.maxTokens) || config.maxTokens <= 0 || config.maxTokens % blockSize !== 0
    ) {
      return yield* invalidInferenceConfig(
        `maxTokens must be a positive multiple of blockSize ${blockSize}, got ${config.maxTokens}`
      )
    }
    if (
      config.attentionWindow !== undefined &&
      (!Number.isInteger(config.attentionWindow) || config.attentionWindow <= 0 ||
        config.attentionWindow > config.maxTokens)
    ) {
      return yield* invalidInferenceConfig(
        `attentionWindow must be a positive integer no greater than maxTokens, got ${config.attentionWindow}`
      )
    }
    if (
      config.prefillChunks.length === 0 ||
      config.prefillChunks.some((chunk) => !Number.isSafeInteger(chunk) || chunk <= 0)
    ) {
      return yield* invalidInferenceConfig(
        `prefillChunks must be positive safe integers, got [${config.prefillChunks}]`
      )
    }
    const prefillChunks = [...new Set(config.prefillChunks)].sort((left, right) => left - right)
    const tokenDtype = config.tokenDtype ?? "u32"
    if (tokenDtype !== "u32" && tokenDtype !== "i64") {
      return yield* invalidInferenceConfig(`tokenDtype must be u32 or i64, got ${String(config.tokenDtype)}`)
    }
    const configuredKvDtype = config.kvDtype ?? "f32"
    if (!["f32", "f16", "bf16", "int8"].includes(configuredKvDtype)) {
      return yield* invalidInferenceConfig(`unsupported kvDtype ${String(config.kvDtype)}`)
    }
    const batchSize = config.batchSize ?? 8
    if (!Number.isInteger(batchSize) || batchSize <= 0) {
      return yield* invalidInferenceConfig(`batchSize must be a positive integer, got ${config.batchSize}`)
    }
    const sampling = config.sampling ?? { seed: 0 }
    if (
      (!Predicate.isBigInt(sampling.seed) && !Number.isSafeInteger(sampling.seed)) || sampling.seed < 0 ||
      BigInt(sampling.seed) > 0xffff_ffff_ffff_ffffn
    ) {
      return yield* invalidInferenceConfig(`sampling.seed must be an unsigned 64-bit integer, got ${sampling.seed}`)
    }
    if (sampling.temperature !== undefined && (!Number.isFinite(sampling.temperature) || sampling.temperature < 0)) {
      return yield* invalidInferenceConfig(
        `sampling.temperature must be finite and non-negative, got ${sampling.temperature}`
      )
    }
    if (sampling.topK !== undefined && (!Number.isSafeInteger(sampling.topK) || sampling.topK < 0)) {
      return yield* invalidInferenceConfig(`sampling.topK must be a non-negative safe integer, got ${sampling.topK}`)
    }
    if (sampling.topP !== undefined && (!Number.isFinite(sampling.topP) || sampling.topP <= 0 || sampling.topP > 1)) {
      return yield* invalidInferenceConfig(`sampling.topP must be in (0, 1], got ${sampling.topP}`)
    }
    let speculation: ResolvedInferenceConfig["speculation"]
    if (config.speculation !== undefined) {
      const proposer = config.speculation.proposer
      if (
        !Predicate.isObjectOrArray(proposer) ||
        !["Autoregressive", "HistoryLookup", "ParallelBlock"].includes(proposer._tag)
      ) {
        return yield* invalidInferenceConfig("speculation.proposer is not a supported speculation artifact")
      }
      if (
        !Number.isSafeInteger(config.speculation.maxDraftTokens) || config.speculation.maxDraftTokens <= 0 ||
        config.speculation.maxDraftTokens > proposer.maxDraftTokens
      ) {
        return yield* invalidInferenceConfig(
          `maxDraftTokens must be in [1, ${proposer.maxDraftTokens}], got ${config.speculation.maxDraftTokens}`
        )
      }
      if (config.speculation.schedule === "adaptive") {
        return yield* invalidInferenceConfig("adaptive speculative scheduling is not implemented; use fixed")
      }
      if (config.attentionWindow !== undefined) {
        return yield* invalidInferenceConfig("speculative execution does not yet support attentionWindow")
      }
      speculation = { proposer, maxDraftTokens: config.speculation.maxDraftTokens }
    }
    return {
      maxTokens: config.maxTokens,
      blockSize,
      prefillChunks,
      tokenDtype,
      kvDtype: configuredKvDtype === "int8" ? "u8" : configuredKvDtype,
      batchSize,
      sampling,
      attentionWindow: config.attentionWindow,
      speculation
    }
  })

interface DecodeGeometry {
  readonly layers: number
  readonly kvHeads: number
  readonly headDim: number
  readonly kdaLayers: number
  readonly kdaHeads: number
  readonly kdaHeadDim: number
  readonly kdaValueDim: number
  readonly convLayers: number
  readonly convChannels: number
  readonly convKernel: number
  readonly window: number | undefined
}

const decodeGeometry = (program: Tensor.DecodeProgram): DecodeGeometry => ({
  layers: program.layers,
  kvHeads: program.kvHeads,
  headDim: program.headDim,
  kdaLayers: program.kdaLayers,
  kdaHeads: program.kdaHeads,
  kdaHeadDim: program.kdaHeadDim,
  kdaValueDim: program.kdaValueDim,
  convLayers: program.convLayers,
  convChannels: program.convChannels,
  convKernel: program.convKernel,
  window: program.window
})

const sameDecodeGeometry = (left: DecodeGeometry, right: DecodeGeometry): boolean =>
  left.layers === right.layers && left.kvHeads === right.kvHeads && left.headDim === right.headDim &&
  left.kdaLayers === right.kdaLayers && left.kdaHeads === right.kdaHeads &&
  left.kdaHeadDim === right.kdaHeadDim && left.kdaValueDim === right.kdaValueDim &&
  left.convLayers === right.convLayers && left.convChannels === right.convChannels &&
  left.convKernel === right.convKernel && left.window === right.window

interface InferencePrograms {
  readonly prefill: ReadonlyArray<Tensor.DecodeProgram>
  readonly decode: Tensor.DecodeProgram
  readonly geometry: DecodeGeometry
  readonly pool: Tensor.KvPool
  readonly speculation?: {
    /** Verify programs per packed rows-per-sequence width, ascending. */
    readonly verify: ReadonlyArray<Tensor.DecodeProgram>
    readonly maxDraftTokens: number
    readonly proposer?: {
      readonly prefill: Tensor.DecodeProgram
      readonly decode: Tensor.DecodeProgram
      readonly pool: Tensor.KvPool
    }
    readonly generalized?: NonNullable<Runtime.InferenceCompileRequest["generalizedProposer"]>
  }
}

/** Speculative-plan verify widths, compiled in ascending order. */
const verifyWidths = (maxDraftTokens: number): ReadonlyArray<number> => {
  const widest = maxDraftTokens + 1
  // M=8 and M=16 have dedicated Metal MMA paths. With a 15-token DFlash
  // block, keep both so the runtime can widen only for high-acceptance
  // sessions; non-aligned narrow widths remain slower than M=8.
  if (widest === 16) return [8, 16]
  return [widest]
}

const logitsVocab = (
  output: Tensor.Any,
  batch: number,
  steps: number
): Effect.Effect<number, InferenceError> => {
  const expected = [batch, steps]
  if (output.shape.length !== 3 || output.shape[0] !== expected[0] || output.shape[1] !== expected[1]) {
    return new InferenceError({
      op: "inference",
      message: `model output must be [${batch}, ${steps}, vocab], got [${output.shape}]`
    })
  }
  return Effect.succeed(output.shape[2]!)
}

const schemaShapeMatches = (
  declared: ReadonlyArray<number | "Rows">,
  actual: ReadonlyArray<number | "Rows">
): boolean => declared.length === actual.length && declared.every((dimension, index) => dimension === actual[index])

const validateTargetContract = (
  model: Model,
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedInferenceConfig,
  vocabulary: number
): Effect.Effect<void, InferenceError | ModelError | Tensor.TensorError, Runtime.Runtime> => {
  const speculation = config.speculation
  if (speculation === undefined) return Effect.void
  const proposer = speculation.proposer
  return Effect.gen(function*() {
    if (proposer.vocabulary !== vocabulary) {
      return yield* invalidInferenceConfig(
        `proposer target vocabulary must be ${proposer.vocabulary}, got ${vocabulary}`
      )
    }
    if (proposer._tag !== "ParallelBlock") return
    for (const weight of [proposer.tokenEmbedding, proposer.lmHead]) {
      const index = model.parameterSpecs.findIndex((parameter) => parameter.name === weight.name)
      const value = index < 0 ? undefined : frozenParams[index]
      if (
        value === undefined || value.dtype !== weight.dtype ||
        !schemaShapeMatches(weight.shape, value.shape)
      ) {
        const actual = value === undefined ? "missing" : `${value.dtype}[${value.shape}]`
        return yield* invalidInferenceConfig(
          `proposer target shared weight ${
            JSON.stringify(weight.name)
          } requires ${weight.dtype}[${weight.shape}], got ${actual}`
        )
      }
    }
  })
}

interface TracedInferenceProgram {
  readonly program: Tensor.DecodeProgram
  readonly taps: ReadonlyArray<Runtime.InferenceTargetTapRoute>
}

const traceInferenceProgram = (
  model: Model,
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedInferenceConfig,
  inputShape: readonly [number, number],
  lastTokenRow = true,
  packedCausalChains?: Runtime.PackedCausalChainsLayout,
  taps: ReadonlyArray<Speculation.HiddenTap> = []
): Effect.Effect<
  TracedInferenceProgram,
  InferenceError | ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    const [graphRows, steps] = inputShape
    const tokenInput = yield* Tensor.zeros(inputShape, { dtype: config.tokenDtype })
    const input = yield* Tensor.makeInput(0, tokenInput)
    const output = yield* model.forward(frozenParams, input)
    yield* logitsVocab(output, graphRows, steps)
    // Exposures live in the graph itself (Tensor.expose identity nodes), so
    // composition can never drop them; discovery is one walk from the root.
    const runtime = yield* Runtime.Runtime
    const discovered = yield* runtime.exposures(output).pipe(
      Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message }))
    )
    const exposed = new Map(discovered.map((entry) => [entry.name, entry.tensor]))
    const roots: Array<Tensor.Any> = [output]
    const routes: Array<Runtime.InferenceTargetTapRoute> = []
    for (const tap of taps) {
      const value = exposed.get(tap.name)
      const logicalShape: ReadonlyArray<number | "Rows"> | undefined = value === undefined ||
          value.shape.length < 2 || value.shape[0] !== graphRows || value.shape[1] !== steps
        ? undefined
        : ["Rows", ...value.shape.slice(2)]
      if (
        value === undefined || logicalShape === undefined || value.dtype !== tap.dtype ||
        !schemaShapeMatches(tap.shape, logicalShape)
      ) {
        const available = discovered.map((entry) => entry.name).sort()
        const actual = value === undefined
          ? `missing; model exposes ${available.length === 0 ? "nothing" : available.join(", ")}`
          : `${value.dtype}[${value.shape}]`
        return yield* invalidInferenceConfig(
          `proposer target hidden tap "${tap.name}" requires ${tap.dtype}[${tap.shape}], got ${actual}`
        )
      }
      roots.push(value)
      routes.push({
        name: tap.name,
        outputRoot: roots.length - 1,
        value: { dtype: value.dtype, shape: value.shape }
      })
    }
    const program = yield* Tensor.compileDecodeProgram(roots, {
      maxTokens: config.maxTokens,
      blockSize: config.blockSize,
      kvDtype: config.kvDtype,
      batch: packedCausalChains === undefined ? graphRows : config.batchSize,
      ...(taps.length === 0
        ? { lastTokenRow }
        : {
          outputSelections: [
            lastTokenRow ? "splitLastTokenRow" as const : "allRows" as const,
            ...taps.map(() => "allRows" as const)
          ]
        }),
      packedCausalChains,
      window: config.attentionWindow
    }).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))
    return { program, taps: routes }
  })

const compileProposerPlan = (
  proposer: Speculation.HistoryLookup | Speculation.ParallelBlock,
  proposerParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedInferenceConfig,
  vocabulary: number,
  targetTaps: {
    /** Target hidden taps per prefill chunk shape, in ascending shape order. */
    readonly prefill: ReadonlyArray<ReadonlyArray<Runtime.InferenceTargetTapRoute>>
    readonly decode: ReadonlyArray<Runtime.InferenceTargetTapRoute>
    /** Target hidden taps per verify width, in ascending width order. */
    readonly verify: ReadonlyArray<{
      readonly width: number
      readonly taps: ReadonlyArray<Runtime.InferenceTargetTapRoute>
    }>
  },
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  targetNames: ReadonlyArray<string>
): Effect.Effect<
  NonNullable<Runtime.InferenceCompileRequest["generalizedProposer"]>,
  InferenceError | ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    if (proposer._tag === "HistoryLookup") {
      const plan: Runtime.InferenceProposerPlan = {
        vocabulary,
        tokenMapFingerprint: "identity",
        hiddenTaps: [],
        sharedTensors: [],
        stages: [{
          operationId: "HistoryLookup",
          layoutId: "suffix-ngram-v1",
          historyLookup: {
            id: "suffix-ngram-v1",
            minMatchTokens: proposer.minMatchTokens,
            maxMatchTokens: proposer.maxMatchTokens
          },
          inputs: [],
          outputs: [{ dtype: config.tokenDtype, shape: [config.batchSize * config.speculation!.maxDraftTokens] }]
        }],
        state: { kind: "None", commitKind: "None", commitStages: [] },
        output: {
          topology: "Chains",
          probabilities: "Deterministic",
          tokenIds: { kind: "StageOutput", stage: 0, output: 0 }
        },
        tokenMap: { kind: "Identity", fingerprint: "identity" },
        trainedMaxRows: config.speculation!.maxDraftTokens
      }
      return { plan, sharedTensors: [], stageExecutables: [], maxDraftTokens: config.speculation!.maxDraftTokens }
    }

    const sharedTensors: Array<Tensor.Concrete> = []
    const sharedMetadata: Array<Runtime.InferenceProposerPlan["sharedTensors"][number]> = []
    for (
      const [kind, weight] of [
        ["TokenEmbedding", proposer.tokenEmbedding],
        ["LmHead", proposer.lmHead]
      ] as const
    ) {
      const actual = frozenParams[targetNames.indexOf(weight.name)]
      if (actual === undefined) {
        return yield* invalidInferenceConfig(`proposer target shared weight ${JSON.stringify(weight.name)} is missing`)
      }
      sharedTensors.push(actual)
      sharedMetadata.push({
        kind,
        name: weight.name,
        value: { dtype: actual.dtype, shape: actual.shape }
      })
    }

    const anchorSchema: Runtime.InferenceValueSchema = { dtype: config.tokenDtype, shape: [config.batchSize] }
    const anchor = yield* Tensor.makeInput(0, yield* Tensor.zeros(anchorSchema.shape, { dtype: anchorSchema.dtype }))
    const embedding = sharedTensors[0]!
    const head = sharedTensors[1]!
    // Parallel-block drafters are trained with a fixed physical block width.
    // The requested speculative width selects a candidate prefix; it must not
    // shrink the diffusion block and change proposal conditioning.
    const output = proposer.buildWithProbabilities === undefined
      ? {
        tokenIds: yield* proposer.build(
          proposerParams,
          anchor,
          yield* Tensor.makeInput(1, embedding),
          yield* Tensor.makeInput(2, head),
          proposer.maxDraftTokens
        )
      }
      : yield* proposer.buildWithProbabilities(
        proposerParams,
        anchor,
        yield* Tensor.makeInput(1, embedding),
        yield* Tensor.makeInput(2, head),
        proposer.maxDraftTokens
      )
    const expectedShape = [config.batchSize, proposer.maxDraftTokens]
    if (output.tokenIds.dtype !== "u32" || !schemaShapeMatches(expectedShape, output.tokenIds.shape)) {
      return yield* invalidInferenceConfig(
        `parallel block output requires u32[${expectedShape}], got ${output.tokenIds.dtype}[${output.tokenIds.shape}]`
      )
    }
    const expectedProbabilityShape = [config.batchSize, proposer.maxDraftTokens, vocabulary]
    if (
      output.probabilityRows !== undefined &&
      (output.probabilityRows.dtype !== "f32" ||
        !schemaShapeMatches(expectedProbabilityShape, output.probabilityRows.shape))
    ) {
      return yield* invalidInferenceConfig(
        `parallel block probabilities require f32[${expectedProbabilityShape}], got ${output.probabilityRows.dtype}[${output.probabilityRows.shape}]`
      )
    }
    const roots = output.probabilityRows === undefined
      ? [output.tokenIds]
      : [output.tokenIds, output.probabilityRows]
    const program = yield* Tensor.compileDecodeProgram(roots, {
      maxTokens: config.maxTokens,
      blockSize: config.blockSize,
      kvDtype: config.kvDtype,
      batch: config.batchSize,
      currentBlockAttention: proposer.currentBlockAttention ?? "Causal",
      window: proposer.attentionWindow
    }).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))
    const compileReplay = (
      taps: ReadonlyArray<Runtime.InferenceTargetTapRoute>,
      packedRows?: number
    ) =>
      Effect.gen(function*() {
        const targetRows: Array<Tensor.Any> = []
        for (let index = 0; index < taps.length; index++) {
          const input = yield* Tensor.zeros(taps[index]!.value.shape, { dtype: taps[index]!.value.dtype })
          const routed = yield* Tensor.makeInput(index, input)
          targetRows.push(
            packedRows === undefined
              ? routed
              : yield* Tensor.reshape(routed, [config.batchSize, packedRows, ...routed.shape.slice(2)])
          )
        }
        const keyValues = yield* proposer.replay(proposerParams, targetRows)
        const roots: Array<Tensor.Any> = []
        // Decode specialization assigns independent state roots from last to first.
        // Reverse replay roots so semantic proposer layer N writes KV cache layer N.
        for (const { key, value } of [...keyValues].reverse()) {
          // Stateful attention appends K/V transactionally and applies cursor-relative
          // transforms. Replay discards the query outputs.
          roots.push(
            yield* Tensor.scaledDotProductAttention(yield* Tensor.zerosLike(key), key, value, {
              causal: true,
              scale: 1
            })
          )
        }
        return yield* Tensor.compileDecodeProgram(roots, {
          maxTokens: config.maxTokens,
          blockSize: config.blockSize,
          kvDtype: config.kvDtype,
          batch: config.batchSize,
          outputSelections: roots.map(() => "allRows" as const),
          window: proposer.attentionWindow
        }).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))
      })
    const replayPrefills: Array<Tensor.DecodeProgram> = []
    for (const taps of targetTaps.prefill) {
      replayPrefills.push(yield* compileReplay(taps))
    }
    const replayDecode = yield* compileReplay(targetTaps.decode)
    // One replay program per verify width: tap row counts follow the width.
    const replayVerifies: Array<Tensor.DecodeProgram> = []
    for (const { width, taps } of targetTaps.verify) {
      replayVerifies.push(yield* compileReplay(taps, width))
    }
    const replayGeometry = decodeGeometry(replayPrefills[replayPrefills.length - 1]!)
    if (
      replayPrefills.some((program) => !sameDecodeGeometry(replayGeometry, decodeGeometry(program))) ||
      !sameDecodeGeometry(replayGeometry, decodeGeometry(replayDecode)) ||
      replayVerifies.some((program) => !sameDecodeGeometry(replayGeometry, decodeGeometry(program))) ||
      !sameDecodeGeometry(replayGeometry, decodeGeometry(program))
    ) {
      return yield* invalidInferenceConfig("parallel block and replay graphs disagree on state geometry")
    }
    const pool = yield* Tensor.makeKvPool(
      replayGeometry.layers,
      replayGeometry.kvHeads,
      replayGeometry.headDim,
      config.maxTokens,
      config.blockSize,
      config.kvDtype
    ).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))
    const inputs: Runtime.InferenceProposerPlan["stages"][number]["inputs"] = [
      { slot: 0, value: { kind: "PendingTokens", value: anchorSchema } },
      { slot: 1, value: { kind: "SharedTokenEmbedding" } },
      { slot: 2, value: { kind: "SharedLmHead" } }
    ]
    const plan: Runtime.InferenceProposerPlan = {
      vocabulary,
      tokenMapFingerprint: "identity",
      hiddenTaps: targetTaps.decode,
      prefillHiddenTaps: targetTaps.prefill[targetTaps.prefill.length - 1]!,
      // Tap routes are width-independent in root order; the widest width's
      // metadata validates the plan.
      verifyHiddenTaps: targetTaps.verify[targetTaps.verify.length - 1]!.taps,
      sharedTensors: sharedMetadata,
      stages: [{
        operationId: "ParallelBlock",
        layoutId: "parallel-block",
        inputs,
        outputs: roots.map((root) => ({ dtype: root.dtype, shape: root.shape }))
      }],
      state: {
        kind: "Kv",
        schemaId: "parallel-block-kv",
        commitKind: "Replay",
        commitStages: [0]
      },
      output: {
        topology: "Chains",
        probabilities: output.probabilityRows === undefined ? "Unavailable" : "CausalNormalized",
        tokenIds: { kind: "StageOutput", stage: 0, output: 0 },
        probabilityRows: output.probabilityRows === undefined
          ? undefined
          : { kind: "StageOutput" as const, stage: 0, output: 1 }
      },
      tokenMap: { kind: "Identity", fingerprint: "identity" },
      trainedMaxRows: proposer.maxDraftTokens
    }
    return {
      plan,
      sharedTensors,
      stageExecutables: [program.handle],
      replay: {
        prefill: replayPrefills.map((program) => program.handle),
        decode: replayDecode.handle,
        verify: replayVerifies.map((program) => program.handle),
        pool: pool.handle
      },
      maxDraftTokens: config.speculation!.maxDraftTokens
    }
  })

const compileInferencePrograms = (
  model: Model,
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedInferenceConfig,
  proposerParams: ReadonlyArray<Tensor.Concrete> | undefined
): Effect.Effect<
  InferencePrograms,
  InferenceError | ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    const proposer = config.speculation?.proposer
    const taps = proposer?._tag === "ParallelBlock" ? proposer.hiddenTaps : []
    // One prefill program per compiled chunk width, ascending; the runtime
    // serves each prompt chunk from the largest width covering its remaining
    // tokens and skips the LM-head chain for non-final chunks.
    const prefillTraces: Array<TracedInferenceProgram> = []
    for (const chunk of config.prefillChunks) {
      prefillTraces.push(
        yield* traceInferenceProgram(
          model,
          frozenParams,
          config,
          [config.batchSize, chunk],
          true,
          undefined,
          taps
        )
      )
    }
    const decodeTrace = yield* traceInferenceProgram(
      model,
      frozenParams,
      config,
      [config.batchSize, 1],
      true,
      undefined,
      taps
    )
    const prefill = prefillTraces.map((trace) => trace.program)
    const decode = decodeTrace.program
    const geometry = decodeGeometry(prefill[prefill.length - 1]!)
    if (
      prefill.some((program) => !sameDecodeGeometry(geometry, decodeGeometry(program))) ||
      !sameDecodeGeometry(geometry, decodeGeometry(decode))
    ) {
      return yield* new InferenceError({
        op: "inference",
        message: "prefill and decode traces disagree on attention geometry or retention policy"
      })
    }
    const targetVocabulary = decode.outputs[0]?.shape[0]
    if (targetVocabulary === undefined) {
      return yield* new InferenceError({ op: "inference", message: "target decode did not expose a vocabulary row" })
    }
    yield* validateTargetContract(model, frozenParams, config, targetVocabulary)
    const pool = yield* Tensor.makeKvPool(
      geometry.layers,
      geometry.kvHeads,
      geometry.headDim,
      config.maxTokens,
      config.blockSize,
      config.kvDtype,
      {
        kdaLayers: geometry.kdaLayers,
        kdaHeads: geometry.kdaHeads,
        kdaHeadDim: geometry.kdaHeadDim,
        kdaValueDim: geometry.kdaValueDim,
        convLayers: geometry.convLayers,
        convChannels: geometry.convChannels,
        convKernel: geometry.convKernel
      }
    ).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))
    if (config.speculation === undefined) {
      return { prefill, decode, geometry, pool }
    }
    if (proposerParams === undefined) {
      return yield* new InferenceError({ op: "inference", message: "speculative proposer parameters are missing" })
    }
    if (geometry.layers === 0 || geometry.kdaLayers !== 0 || geometry.convLayers !== 0) {
      return yield* new InferenceError({
        op: "inference",
        message: "speculative target state must be KV-only with at least one attention layer"
      })
    }
    if (proposer?._tag !== "Autoregressive") {
      // ParallelBlock compiles one verify program per packed width and the
      // runtime adaptively selects the width per round from measured token
      // rates; HistoryLookup verifies full-width (its drafts are free).
      const widths = proposer?._tag === "ParallelBlock"
        ? verifyWidths(config.speculation.maxDraftTokens)
        : [config.speculation.maxDraftTokens + 1]
      const verifyTraces: Array<TracedInferenceProgram> = []
      for (const width of widths) {
        verifyTraces.push(
          yield* traceInferenceProgram(
            model,
            frozenParams,
            config,
            [config.batchSize * width, 1],
            false,
            { rowsPerSequence: width },
            taps
          )
        )
      }
      const generalized = yield* compileProposerPlan(
        proposer!,
        proposerParams,
        config,
        targetVocabulary,
        {
          prefill: prefillTraces.map((trace) => trace.taps),
          decode: decodeTrace.taps,
          verify: verifyTraces.map((trace, index) => ({ width: widths[index]!, taps: trace.taps }))
        },
        frozenParams,
        model.parameterSpecs.map((parameter) => parameter.name)
      )
      return {
        prefill,
        decode,
        geometry,
        pool,
        speculation: {
          verify: verifyTraces.map((trace) => trace.program),
          maxDraftTokens: config.speculation.maxDraftTokens,
          generalized
        }
      }
    }
    const proposerModel = proposer.model
    const exactParams = proposerParams
    const proposerPrefill = yield* traceInferenceProgram(
      proposerModel,
      exactParams,
      config,
      [config.batchSize, config.prefillChunks[config.prefillChunks.length - 1]!]
    ).pipe(Effect.map((trace) => trace.program))
    const proposerDecode = yield* traceInferenceProgram(
      proposerModel,
      exactParams,
      config,
      [config.batchSize, 1]
    ).pipe(Effect.map((trace) => trace.program))
    const proposerGeometry = decodeGeometry(proposerPrefill)
    if (!sameDecodeGeometry(proposerGeometry, decodeGeometry(proposerDecode))) {
      return yield* new InferenceError({
        op: "inference",
        message: "proposer prefill and decode traces disagree on state geometry"
      })
    }
    if (proposerGeometry.layers === 0 || proposerGeometry.kdaLayers !== 0 || proposerGeometry.convLayers !== 0) {
      return yield* new InferenceError({
        op: "inference",
        message: "speculative proposer state must be KV-only with at least one attention layer"
      })
    }
    const proposerVocabulary = proposerDecode.outputs[0]?.shape[0]
    if (
      targetVocabulary !== proposer.vocabulary ||
      proposerVocabulary !== targetVocabulary
    ) {
      return yield* new InferenceError({
        op: "inference",
        message:
          `speculative identity token map requires target/proposer vocabulary ${proposer.vocabulary}, got target ${targetVocabulary} and proposer ${proposerVocabulary}`
      })
    }
    const verify = yield* traceInferenceProgram(
      model,
      frozenParams,
      config,
      [config.batchSize * (config.speculation.maxDraftTokens + 1), 1],
      false,
      { rowsPerSequence: config.speculation.maxDraftTokens + 1 }
    ).pipe(Effect.map((trace) => trace.program))
    if (!sameDecodeGeometry(geometry, decodeGeometry(verify))) {
      return yield* new InferenceError({
        op: "inference",
        message: "target verification trace disagrees with target decode state geometry"
      })
    }
    const proposerPool = yield* Tensor.makeKvPool(
      proposerGeometry.layers,
      proposerGeometry.kvHeads,
      proposerGeometry.headDim,
      config.maxTokens,
      config.blockSize,
      config.kvDtype
    ).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))
    return {
      prefill,
      decode,
      geometry,
      pool,
      speculation: {
        // Exact proposers keep a single full-width verify program.
        verify: [verify],
        maxDraftTokens: config.speculation.maxDraftTokens,
        proposer: { prefill: proposerPrefill, decode: proposerDecode, pool: proposerPool }
      }
    }
  })

interface PrefillChunkPlan {
  readonly offset: number
  readonly real: number
  readonly final: boolean
}

// This checks only the public add calling convention. Reading or execution
// later validates token values and model vocabulary and position bounds.
const validatePrompt = (
  prompt: Tensor.Any,
  config: ResolvedInferenceConfig,
  runtime: Runtime.RuntimeService
): Effect.Effect<void, InferenceError> => {
  if (prompt.placement.id !== runtime.placement.id) {
    return new InferenceError({ op: "add", message: "prompt must use the inference program runtime and placement" })
  }
  if (prompt.dtype !== config.tokenDtype) {
    return new InferenceError({
      op: "add",
      message: `prompt dtype must be ${config.tokenDtype}, got ${prompt.dtype}`
    })
  }
  if (prompt.shape.length !== 2 || prompt.shape[0] !== 1 || prompt.shape[1]! < 1) {
    return new InferenceError({
      op: "add",
      message: `add expects a prompt of shape [1, T] with T >= 1, got [${prompt.shape}]`
    })
  }
  return Effect.void
}

const readTokenIds = (tokens: Tensor.Any): Effect.Effect<Array<number>, InferenceError, Runtime.Runtime> => {
  const read = tokens.dtype === "i64"
    ? Effect.gen(function*() {
      const values = yield* Tensor.toTypedArray(tokens)
      const ids: Array<number> = []
      for (const value of values) {
        if (!Predicate.isBigInt(value) || value < 0n || value > 0xffff_ffffn) {
          return yield* new InferenceError({
            op: "prefill",
            message: `token ids must fit u32 for decode state, got ${String(value)}`
          })
        }
        ids.push(Number(value))
      }
      return ids
    })
    : Tensor.toNumberArray(tokens)
  return Effect.mapError(read, (error) =>
    error instanceof InferenceError
      ? error
      : new InferenceError({ op: "prefill", message: `token ids must be readable integers: ${error.message}` }))
}

const tokenTensor = (
  ids: ReadonlyArray<number>,
  shape: ReadonlyArray<number>,
  dtype: "u32" | "i64"
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  Tensor.fromTypedArray(dtype === "i64" ? BigInt64Array.from(ids.map(BigInt)) : Uint32Array.from(ids), shape)

const slottedTokenTensor = (
  ids: ReadonlyArray<number>,
  slots: ReadonlyArray<number>,
  batchSize: number,
  dtype: "u32" | "i64"
): Effect.Effect<Tensor.Any, Tensor.TensorError, Runtime.Runtime> => {
  const values = Array<number>(batchSize).fill(0)
  for (const [index, slot] of slots.entries()) values[slot] = ids[index]!
  return tokenTensor(values, [batchSize, 1], dtype)
}

interface PrefillLane {
  readonly slot: number
  readonly sequence: Tensor.KvSequence
  readonly tokens: ReadonlyArray<number>
  offset: number
}

interface PrefillRoundLane extends PrefillLane {
  readonly chunk: PrefillChunkPlan
}

const slottedPrefillTensor = (
  lanes: ReadonlyArray<PrefillRoundLane>,
  config: ResolvedInferenceConfig
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> => {
  // The generic session driver always runs the largest compiled chunk.
  const prefillChunk = config.prefillChunks[config.prefillChunks.length - 1]!
  const values = Array<number>(config.batchSize * prefillChunk).fill(0)
  for (const lane of lanes) {
    const tokens = lane.tokens.slice(lane.chunk.offset, lane.chunk.offset + lane.chunk.real)
    for (const [index, token] of tokens.entries()) {
      values[lane.slot * prefillChunk + index] = token
    }
  }
  return tokenTensor(values, [config.batchSize, prefillChunk], config.tokenDtype)
}

const selectSlottedOutputs = (
  outputs: ReadonlyArray<Tensor.Concrete>,
  slots: ReadonlyArray<number>
): Effect.Effect<Array<Tensor.Concrete>, never, Runtime.Runtime> =>
  Effect.gen(function*() {
    const selected = slots.map((slot) => outputs[slot]!)
    const selectedSlots = new Set(slots)
    for (const [slot, output] of outputs.entries()) {
      if (!selectedSlots.has(slot)) yield* Tensor.clear(output)
    }
    return selected
  })

const runPrefillBatches = <A>(
  program: Tensor.DecodeProgram,
  config: ResolvedInferenceConfig,
  lanes: ReadonlyArray<PrefillLane>,
  runFinal: (
    lanes: ReadonlyArray<PrefillRoundLane>,
    input: Tensor.Any,
    tokens: ReadonlyArray<ReadonlyArray<number>>
  ) => Effect.Effect<ReadonlyArray<A>, Tensor.TensorError, Runtime.Runtime>,
  clearFinalValues: (values: ReadonlyArray<A>) => Effect.Effect<void, never, Runtime.Runtime>
): Effect.Effect<ReadonlyArray<A>, InferenceError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.suspend(() => {
    const results = new Map<number, A>()
    return Effect.onExit(
      Effect.gen(function*() {
        while (results.size < lanes.length) {
          const round = lanes
            .filter((lane) => !results.has(lane.slot))
            .map((lane): PrefillRoundLane => {
              const real = Math.min(
                config.prefillChunks[config.prefillChunks.length - 1]!,
                lane.tokens.length - lane.offset
              )
              return {
                ...lane,
                chunk: { offset: lane.offset, real, final: lane.offset + real === lane.tokens.length }
              }
            })
          for (const final of [false, true]) {
            const group = round.filter((lane) => lane.chunk.final === final)
            if (group.length === 0) continue
            const input = yield* slottedPrefillTensor(group, config)
            const tokens = group.map((lane) =>
              lane.tokens.slice(lane.chunk.offset, lane.chunk.offset + lane.chunk.real)
            )
            if (final) {
              const values = yield* runFinal(group, input, tokens)
              if (values.length !== group.length) {
                yield* clearFinalValues(values)
                return yield* new InferenceError({
                  op: "prefill",
                  message: `prefill returned ${values.length} final values for ${group.length} lanes`
                })
              }
              for (const [index, lane] of group.entries()) results.set(lane.slot, values[index]!)
            } else {
              const outputs = yield* Tensor.runBatchedDecodeProgram(
                program,
                [input],
                group.map((lane) => lane.sequence),
                group.map((lane) => lane.slot),
                tokens
              )
              yield* Tensor.clearAll(outputs)
            }
            for (const lane of group) lanes.find((source) => source.slot === lane.slot)!.offset += lane.chunk.real
          }
        }
        return lanes.map((lane) => results.get(lane.slot)!)
      }),
      (exit) => Exit.isFailure(exit) ? clearFinalValues(Array.from(results.values())) : Effect.void
    )
  })

interface SessionSeq {
  readonly sequence: Tensor.KvSequence
}

interface LiveEntry<Seq extends SessionSeq> {
  readonly seq: Seq
  readonly slot: number
}

// Keep entries live until backend release succeeds so a failed or interrupted
// release remains retryable.
const releaseLiveEntry = <Seq extends SessionSeq>(live: Array<LiveEntry<Seq>>, entry: LiveEntry<Seq>) =>
  Effect.gen(function*() {
    const index = live.indexOf(entry)
    if (index < 0) return
    yield* Tensor.releaseKvSequence(entry.seq.sequence)
    live.splice(index, 1)
  })

const releaseLiveEntries = <Seq extends SessionSeq>(
  live: Array<LiveEntry<Seq>>,
  entries: ReadonlyArray<LiveEntry<Seq>>
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    let failure: Tensor.TensorError | undefined
    for (const entry of entries) {
      yield* Effect.matchEffect(releaseLiveEntry(live, entry), {
        onFailure: (error) =>
          Effect.sync(() => {
            failure ??= error
          }),
        onSuccess: () => Effect.void
      })
    }
    if (failure !== undefined) {
      return yield* Effect.fail(failure)
    }
  })

const closeLiveEntries = <Seq extends SessionSeq>(
  live: Array<LiveEntry<Seq>>
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> => releaseLiveEntries(live, live.slice())

// The step semaphore does not cover lifecycle mutations. Generation requires
// callers to keep them disjoint.
const validateStepEntries = (
  live: ReadonlyArray<LiveEntry<StatefulExecutionSeq>>,
  batchSize: number,
  entries: ReadonlyArray<{ readonly seq: StatefulExecutionSeq; readonly token: number }>
): Effect.Effect<void, InferenceError> =>
  Effect.gen(function*() {
    if (entries.length === 0) {
      return yield* new InferenceError({ op: "step", message: "step expects at least one entry" })
    }
    if (entries.length > batchSize) {
      return yield* new InferenceError({
        op: "step",
        message: `step accepts at most batchSize (${batchSize}) entries, got ${entries.length}`
      })
    }
    for (const [index, entry] of entries.entries()) {
      if (!Number.isInteger(entry.token) || entry.token < 0) {
        return yield* new InferenceError({
          op: "step",
          message: `step expects token ids (non-negative integers), got ${entry.token}`
        })
      }
      if (!live.some((liveEntry) => liveEntry.seq === entry.seq)) {
        return yield* new InferenceError({
          op: "step",
          message: `entry ${index} is not a live sequence of this session`
        })
      }
      if (entries.findIndex((other) => other.seq === entry.seq) !== index) {
        return yield* new InferenceError({ op: "step", message: "step entries must be distinct sequences" })
      }
    }
  })

interface InferenceEngine {
  readonly config: ResolvedInferenceConfig
  readonly frozenParams: ReadonlyArray<Tensor.Concrete>
  readonly programs: InferencePrograms
  readonly artifact: Runtime.InferenceArtifactHandle
  readonly runtime: Runtime.RuntimeService
}

const openStatefulExecution = (engine: InferenceEngine): Effect.Effect<StatefulExecution, never> =>
  Effect.gen(function*() {
    const roundLock = yield* Semaphore.make(1)
    const live: Array<LiveEntry<StatefulExecutionSeq>> = []
    const config = engine.config
    const programs = engine.programs
    const add: StatefulExecution["add"] = (prompts) =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          if (prompts.length === 0) {
            return yield* new InferenceError({ op: "add", message: "add expects at least one prompt" })
          }
          if (live.length + prompts.length > config.batchSize) {
            return yield* new InferenceError({
              op: "add",
              message: `add needs ${prompts.length} free lanes, but only ${config.batchSize - live.length} remain`
            })
          }
          const runtime = yield* Runtime.Runtime
          for (const prompt of prompts) yield* validatePrompt(prompt, config, runtime)
          const promptValues = yield* Tensor.compute(prompts)
          const sequences: Array<Tensor.KvSequence> = []
          const added: Array<{ readonly seq: StatefulExecutionSeq; readonly logits: Tensor.Concrete }> = []
          return yield* Effect.onExit(
            Effect.gen(function*() {
              const tokenRows: Array<ReadonlyArray<number>> = []
              for (const prompt of promptValues) tokenRows.push(yield* readTokenIds(prompt))
              const freeSlots = Array.from({ length: config.batchSize }, (_, slot) => slot)
                .filter((slot) => !live.some((entry) => entry.slot === slot))
              const lanes: Array<PrefillLane> = []
              for (const [index, tokens] of tokenRows.entries()) {
                const sequence = yield* Tensor.makeKvSequence(programs.pool)
                sequences.push(sequence)
                const matched = yield* Tensor.kvPrefillMatch(sequence, tokens)
                lanes.push({ slot: freeSlots[index]!, sequence, tokens, offset: matched })
              }
              const logits = yield* runPrefillBatches(
                programs.prefill[programs.prefill.length - 1]!,
                config,
                lanes,
                (finals, input, tokens) =>
                  Effect.flatMap(
                    Tensor.runBatchedDecodeProgram(
                      programs.prefill[programs.prefill.length - 1]!,
                      [input],
                      finals.map((lane) => lane.sequence),
                      finals.map((lane) => lane.slot),
                      tokens
                    ),
                    (outputs) => selectSlottedOutputs(outputs, finals.map((lane) => lane.slot))
                  ),
                Tensor.clearAll
              )
              yield* Effect.sync(() => {
                for (const [index, lane] of lanes.entries()) {
                  let entry: LiveEntry<StatefulExecutionSeq>
                  const seq: StatefulExecutionSeq = {
                    _tag: "StatefulExecutionSeq",
                    sequence: lane.sequence,
                    cursor: () => Tensor.kvSequenceCursor(lane.sequence),
                    finish: () => releaseLiveEntry(live, entry)
                  }
                  entry = { seq, slot: lane.slot }
                  live.push(entry)
                  added.push({ seq, logits: logits[index]! })
                }
              })
              return added
            }),
            (exit) =>
              Effect.gen(function*() {
                yield* Tensor.clearAll(promptValues)
                if (Exit.isFailure(exit)) {
                  yield* Tensor.clearAll(added.map((entry) => entry.logits))
                  for (const sequence of sequences) {
                    const entry = live.find((entry) => entry.seq.sequence === sequence)
                    if (entry === undefined) {
                      yield* Tensor.releaseKvSequence(sequence)
                    } else {
                      yield* releaseLiveEntry(live, entry)
                    }
                  }
                }
              })
          )
        })
      )
    const runStep = <A, Entry extends { readonly seq: StatefulExecutionSeq; readonly token: number }>(
      entries: ReadonlyArray<Entry>,
      runBatched: (
        entries: ReadonlyArray<Entry>,
        input: Tensor.Any,
        ids: ReadonlyArray<number>,
        slots: ReadonlyArray<number>,
        program: Tensor.DecodeProgram
      ) => Effect.Effect<ReadonlyArray<A>, Tensor.TensorError, Runtime.Runtime>
    ): Effect.Effect<ReadonlyArray<A>, InferenceError | Tensor.TensorError, Runtime.Runtime> =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          yield* validateStepEntries(live, config.batchSize, entries)
          const ids = entries.map((entry) => entry.token)
          const slots = entries.map((entry) => live.find((liveEntry) => liveEntry.seq === entry.seq)!.slot)
          const input = yield* slottedTokenTensor(ids, slots, config.batchSize, config.tokenDtype)
          return yield* runBatched(entries, input, ids, slots, programs.decode)
        })
      )
    const step: StatefulExecution["step"] = (entries) =>
      runStep(
        entries,
        (entries, input, ids, slots, batched) =>
          Effect.flatMap(
            Tensor.runBatchedDecodeProgram(
              batched,
              [input],
              entries.map((entry) => entry.seq.sequence),
              slots,
              ids.map((id) => [id])
            ),
            (outputs) =>
              Effect.onExit(
                Effect.gen(function*() {
                  const selected = slots.map((slot) => outputs[slot]!)
                  const selectedSlots = new Set(slots)
                  for (const [slot, output] of outputs.entries()) {
                    if (selectedSlots.has(slot)) continue
                    yield* Tensor.clear(output)
                  }
                  return selected
                }),
                (exit) => Exit.isFailure(exit) ? Tensor.clearAll(outputs) : Effect.void
              )
          )
      )
    return {
      add,
      step,
      live: () => Effect.sync(() => live.length),
      close: () => closeLiveEntries(live)
    }
  })

interface NativeGenerationEntry {
  readonly seq: GenerationSeq
  readonly handle: Runtime.InferenceSequenceHandle
  readonly id: bigint
  terminal: "eos" | "maxTokens" | undefined
}

const inferenceBackend = <A>(op: string, effect: Effect.Effect<A, Runtime.BackendError>) =>
  Effect.mapError(effect, (backend) => new Tensor.TensorError({ op, message: backend.message, backend }))

const nativeSampling = (sampling: GenerationSamplingOptions): Runtime.InferenceSamplingOptions => {
  const seed = sampling.seed
  return {
    temperature: sampling.temperature ?? 1,
    topK: sampling.topK ?? 0,
    topP: sampling.topP ?? 1,
    seed: BigInt(seed)
  }
}

const nativeSamplingOverride = (
  sampling: Partial<GenerationSamplingOptions>
): Runtime.InferenceSamplingOverrides => ({
  temperature: sampling.temperature,
  topK: sampling.topK,
  topP: sampling.topP,
  seed: sampling.seed === undefined ? undefined : BigInt(sampling.seed)
})

const validateGenerationAdd = (
  entry: GenerationAdd,
  index: number,
  defaults: GenerationSamplingOptions
): Effect.Effect<void, InferenceError> =>
  Effect.gen(function*() {
    if (
      entry.maxTokens !== undefined &&
      (!Number.isSafeInteger(entry.maxTokens) || entry.maxTokens <= 0 || entry.maxTokens > 0xffff_ffff)
    ) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} maxTokens must be an unsigned 32-bit positive integer, got ${entry.maxTokens}`
      })
    }
    for (const token of entry.eosTokens ?? []) {
      if (!Number.isInteger(token) || token < 0 || token > 0xffff_ffff) {
        return yield* new InferenceError({
          op: "add",
          message: `entry ${index} eosTokens must contain unsigned 32-bit token ids, got ${token}`
        })
      }
    }
    const sampling = { ...defaults, ...entry.sampling }
    if (
      (!Predicate.isBigInt(sampling.seed) && !Number.isSafeInteger(sampling.seed)) || sampling.seed < 0 ||
      BigInt(sampling.seed) > 0xffff_ffff_ffff_ffffn
    ) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} seed must be an unsigned 64-bit integer, got ${sampling.seed}`
      })
    }
    if (sampling.temperature !== undefined && (!Number.isFinite(sampling.temperature) || sampling.temperature < 0)) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} temperature must be finite and non-negative, got ${sampling.temperature}`
      })
    }
    if (sampling.topK !== undefined && (!Number.isSafeInteger(sampling.topK) || sampling.topK < 0)) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} topK must be a non-negative safe integer, got ${sampling.topK}`
      })
    }
    if (sampling.topP !== undefined && (!Number.isFinite(sampling.topP) || sampling.topP <= 0 || sampling.topP > 1)) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} topP must be in (0, 1], got ${sampling.topP}`
      })
    }
  })

const openGeneration = (engine: InferenceEngine): Effect.Effect<Generation, InferenceError> =>
  Effect.gen(function*() {
    const roundLock = yield* Semaphore.make(1)
    const live: Array<NativeGenerationEntry> = []
    const config = engine.config
    const runtime = engine.runtime
    const native = runtime.extensions.inference
    const session = yield* Effect.mapError(
      native.open(engine.artifact),
      (error) => new InferenceError({ op: "generation", message: error.message })
    )

    const pagesFor = (
      op: "add" | "step",
      result: Runtime.InferenceRoundResult,
      expected: ReadonlyArray<NativeGenerationEntry>
    ): Effect.Effect<ReadonlyArray<TokenPage>, InferenceError> =>
      Effect.gen(function*() {
        if (
          result.roundId < 0n || result.roundId > 0xffff_ffff_ffff_ffffn ||
          !Predicate.isBoolean(result.recovered) ||
          result.pages.length !== expected.length
        ) {
          return yield* new InferenceError({
            op,
            message: `${op}: native inference returned a malformed round receipt`
          })
        }
        const pages: Array<TokenPage> = []
        for (const [index, page] of result.pages.entries()) {
          const entry = expected[index]!
          if (
            page.sequence !== entry.handle || page.sequenceId !== entry.id || page.tokens.length === 0 ||
            page.tokens.some((token) => !Number.isInteger(token) || token < 0 || token > 0xffff_ffff) ||
            (page.stopReason !== undefined && page.stopReason !== "eos" && page.stopReason !== "maxTokens")
          ) {
            return yield* new InferenceError({ op, message: `${op}: native inference returned a malformed token page` })
          }
          pages.push({
            seq: entry.seq,
            tokens: page.tokens,
            stopReason: page.stopReason
          })
        }
        return pages
      })

    const add: Generation["add"] = (requests) =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          if (requests.length === 0) {
            return yield* new InferenceError({ op: "add", message: "add expects at least one entry" })
          }
          if (live.length + requests.length > config.batchSize) {
            return yield* new InferenceError({
              op: "add",
              message: `add needs ${requests.length} free lanes, but only ${config.batchSize - live.length} remain`
            })
          }
          for (const [index, request] of requests.entries()) {
            yield* validateGenerationAdd(request, index, config.sampling)
          }
          for (const request of requests) yield* validatePrompt(request.prompt, config, runtime)
          const promptValues = yield* Tensor.compute(requests.map((request) => request.prompt))
          return yield* Effect.onExit(
            Effect.gen(function*() {
              const result = yield* inferenceBackend(
                "inferenceAdd",
                native.add(session, {
                  entries: requests.map((request, index) => ({
                    prompt: promptValues[index]!,
                    sampling: request.sampling === undefined
                      ? undefined
                      : nativeSamplingOverride(request.sampling),
                    maxTokens: request.maxTokens,
                    eosTokens: request.eosTokens ?? []
                  }))
                })
              )
              if (result.pages.length !== requests.length) {
                return yield* new InferenceError({
                  op: "add",
                  message: "add: native inference returned the wrong page count"
                })
              }
              const added: Array<NativeGenerationEntry> = []
              for (const page of result.pages) {
                if (
                  page.sequenceId < 0n || page.sequenceId > 0xffff_ffff_ffff_ffffn ||
                  added.some((entry) => entry.handle === page.sequence || entry.id === page.sequenceId)
                ) {
                  return yield* new InferenceError({
                    op: "add",
                    message: "add: native inference returned invalid sequence identity"
                  })
                }
                let entry: NativeGenerationEntry
                const seq: GenerationSeq = {
                  _tag: "GenerationSeq",
                  cursor: () =>
                    Effect.gen(function*() {
                      const inspected = yield* inferenceBackend(
                        "inferenceInspect",
                        native.inspect(session, entry.handle)
                      )
                      if (
                        inspected.sequenceId !== entry.id || inspected.cursor < 0n ||
                        inspected.cursor > BigInt(Number.MAX_SAFE_INTEGER)
                      ) {
                        return yield* new Tensor.TensorError({
                          op: "inferenceInspect",
                          message: "native inference returned an invalid cursor"
                        })
                      }
                      return Number(inspected.cursor)
                    }),
                  finish: () =>
                    roundLock.withPermits(1)(
                      Effect.gen(function*() {
                        const index = live.indexOf(entry)
                        if (index < 0) return
                        yield* inferenceBackend("inferenceFinish", native.finish(session, [entry.handle]))
                        live.splice(index, 1)
                      })
                    )
                }
                entry = { seq, handle: page.sequence, id: page.sequenceId, terminal: undefined }
                added.push(entry)
              }
              const pages = yield* pagesFor("add", result, added)
              return yield* Effect.uninterruptible(Effect.gen(function*() {
                yield* inferenceBackend("inferenceAcknowledge", native.acknowledge(session, result.roundId))
                for (const [index, entry] of added.entries()) entry.terminal = result.pages[index]!.stopReason
                live.push(...added)
                return pages
              }))
            }),
            () => Tensor.clearAll(promptValues)
          )
        })
      )

    const step: Generation["step"] = (requests) =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          if (requests.length === 0) {
            return yield* new InferenceError({ op: "step", message: "step expects at least one entry" })
          }
          if (requests.length > config.batchSize) {
            return yield* new InferenceError({
              op: "step",
              message: `step accepts at most batchSize (${config.batchSize}) entries, got ${requests.length}`
            })
          }
          const selected: Array<NativeGenerationEntry> = []
          for (const [index, request] of requests.entries()) {
            const entry = live.find((entry) => entry.seq === request.seq)
            if (entry === undefined) {
              return yield* new InferenceError({ op: "step", message: `entry ${index} is not a live sequence` })
            }
            if (selected.includes(entry)) {
              return yield* new InferenceError({ op: "step", message: "step entries must be distinct sequences" })
            }
            if (entry.terminal !== undefined) {
              return yield* new InferenceError({
                op: "step",
                message: `entry ${index} is terminal (${entry.terminal})`
              })
            }
            selected.push(entry)
          }
          const result = yield* inferenceBackend(
            "inferenceRound",
            native.runRound(session, {
              entries: selected.map((entry, index) => ({
                sequence: entry.handle,
                sampling: requests[index]!.sampling === undefined
                  ? undefined
                  : nativeSamplingOverride(requests[index]!.sampling)
              }))
            })
          )
          const pages = yield* pagesFor("step", result, selected)
          return yield* Effect.uninterruptible(Effect.gen(function*() {
            yield* inferenceBackend("inferenceAcknowledge", native.acknowledge(session, result.roundId))
            for (const [index, entry] of selected.entries()) entry.terminal = result.pages[index]!.stopReason
            return pages
          }))
        })
      )

    return {
      add,
      step,
      live: () => Effect.sync(() => live.length),
      close: () =>
        roundLock.withPermits(1)(
          Effect.tap(inferenceBackend("inferenceClose", native.close(session)), () => Effect.sync(() => live.splice(0)))
        )
    }
  })

/**
 * Materializes a model for stateful autoregressive generation and eagerly
 * compiles its complete deployment geometry. The same `forward` builder is
 * traced twice, once for fixed prompt chunks and once for fixed-width batched
 * decode. Decode specialization rewrites causal attention to paged KV
 * attention, KDA and short convolution to per-sequence
 * recurrent operations, and learned/rotary position nodes to absolute-cursor-
 * offset forms. There is no shape-keyed growth or later tracing.
 *
 * Every trace must return exactly `[batch, T, vocab]` with the traced batch and
 * token dimensions preserved, and all traces must agree on state geometry and
 * effective retention policy. Native `lastTokenRow` selection returns one
 * caller-owned `[vocab]` row per active sequence. Stateless graphs are allowed.
 * Non-causal attention, runtime scalar inputs, unsupported stateful operations,
 * inconsistent traces, and invalid output rank/axes fail during construction.
 * This does not establish semantic language-model correctness or validate a
 * tokenizer/vocabulary contract.
 *
 * `params` are borrowed and materialized together once with
 * {@link Tensor.compute}. This samples lazy initializers once and produces a new
 * concrete generation retained as immutable constants by every compiled
 * program. Caller-supplied concrete handles are not consumed and may be cleared
 * after this effect succeeds; the artifact's retained generation remains valid.
 * If tracing or pool construction fails or is interrupted, the newly
 * materialized parameter handles are cleared before the failure is returned.
 * There is no explicit artifact release after success; native finalization
 * reclaims its constants, programs, and pool when unreachable.
 *
 * State capacity is separate from artifact lifetime. Live sequences pin blocks
 * and recurrent state, while completed blocks may remain as evictable prefix
 * cache. Use {@link GenerationSeq.finish} or {@link Generation.close} to remove
 * live ownership promptly. An attention window can bound retained KV history
 * without resetting the absolute cursor; learned position tables and other
 * cursor-indexed state remain independently bounded.
 *
 * @since 0.1.0
 * @category compilation
 */
export const inference = (
  model: Model,
  params: Params,
  config: InferenceConfig
): Effect.Effect<InferenceProgram, InferenceError | ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    yield* checkArity("inference", model.parameterSpecs.map((parameter) => parameter.name), params)
    const resolved = yield* resolveInferenceConfig(config)
    const proposerSourceParams = resolved.speculation?.proposer._tag === "HistoryLookup"
      ? []
      : resolved.speculation?.proposer.params ?? []
    const targetArity = params.length
    return yield* Effect.flatMap(
      Tensor.compute([...params, ...proposerSourceParams]),
      (allFrozenParams) =>
        Effect.onExit(
          Effect.gen(function*() {
            const frozenParams = allFrozenParams.slice(0, targetArity)
            const proposerParams = resolved.speculation === undefined
              ? undefined
              : allFrozenParams.slice(targetArity)
            const programs = yield* compileInferencePrograms(model, frozenParams, resolved, proposerParams)
            const exactProposer = programs.speculation !== undefined &&
                programs.speculation.generalized === undefined && programs.speculation.proposer !== undefined
              ? { ...programs.speculation.proposer, maxDraftTokens: programs.speculation.maxDraftTokens }
              : undefined
            const artifact = yield* inferenceBackend(
              "inferenceCompile",
              runtime.extensions.inference.compile({
                target: {
                  prefill: programs.prefill.map((program) => program.handle),
                  decode: programs.decode.handle,
                  verify: programs.speculation?.verify.map((program) => program.handle),
                  pool: programs.pool.handle
                },
                proposer: exactProposer === undefined
                  ? undefined
                  : {
                    prefill: exactProposer.prefill.handle,
                    decode: exactProposer.decode.handle,
                    pool: exactProposer.pool.handle,
                    maxDraftTokens: exactProposer.maxDraftTokens
                  },
                generalizedProposer: programs.speculation?.generalized,
                batchSize: resolved.batchSize,
                tokenDtype: resolved.tokenDtype,
                sampling: nativeSampling(resolved.sampling)
              })
            )
            const engine: InferenceEngine = {
              config: resolved,
              frozenParams: allFrozenParams,
              programs,
              artifact,
              runtime
            }
            return {
              generation: () => openGeneration(engine),
              execution: () => openStatefulExecution(engine),
              diagnostics: () =>
                inferenceBackend("inferenceDiagnostics", runtime.extensions.inference.diagnostics(artifact))
            } satisfies InferenceProgram
          }),
          (exit) => Exit.isFailure(exit) ? Tensor.clearAll(allFrozenParams) : Effect.void
        )
    )
  })

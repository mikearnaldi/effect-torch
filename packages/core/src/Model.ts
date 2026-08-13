/**
 * Models pair pure parameter construction with a parameterised forward graph
 * builder — the Flax/Haiku design, flattened:
 * parameters are always a flat array of tensors, configuration lives in
 * the factory's closure, and the forward graph is an ordinary lazy graph,
 * so {@link Gradient.grad} differentiates it and an optimizer's `step`
 * updates it with zero model-specific code. There is no mutable module
 * state or backward mode. Each model object also lazily owns mutable
 * compiled-program cache state exposed through {@link Model.stats} and
 * {@link Model.clear}.
 *
 * Everything that can fail returns an `Effect`: factories validate their
 * configuration (positive integer feature counts, unique parameter names)
 * into a {@link ModelError}; parameterized forwards check parameter arity;
 * and serialization reports arity and missing-key problems in the error
 * channel. Training lives in the `Trainer` module.
 *
 * Constructors cover dense and convolutional layers, token and position
 * embeddings, normalization, multi-head attention, activations, dropout,
 * flattening, and pooling. {@link chain}, {@link merge}, {@link add},
 * {@link residual}, {@link mapInput}, and {@link checkpoint} compose them.
 * `names` gives every parameter a stable, checkpoint-friendly identity
 * mapped by {@link save} and {@link load}. Every model also carries a
 * compiled execution path:
 * {@link Model.execute} runs the forward as a cached native executable,
 * traced on a signature miss and reused until eviction or clearing, while
 * {@link Model.forward} stays the lazy graph builder for training,
 * composition, and differentiation. {@link inference} builds the separate
 * paged-KV artifact used for autoregressive generation.
 *
 * Stateful layers (batchnorm running stats) are deliberately absent: the
 * pure design keeps non-trainable state out of the parameter array until
 * the `stateRoots`/`rebuildState` contract generalizes to models. Note
 * that {@link dropout} is the functional form — it always applies; build
 * the evaluation chain without it (parameterless stages add nothing to
 * the array, so one checkpoint serves both chains).
 *
 * @since 0.1.0
 */
import { Data, Effect, Exit, Semaphore } from "effect"
import * as Gradient from "./Gradient.ts"
import * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * A failure in model construction, parameter arity, or serialization:
 * invalid layer configuration, duplicate parameter names, an incorrect
 * parameter count, or a missing checkpoint key. The tagged payload has an
 * operation label and a human-readable message; tensor-operation failures
 * stay {@link Tensor.TensorError}s.
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
  /** Stable parameter identity. */
  readonly name: string
  /** Logical tensor shape, independent of its physical storage. */
  readonly shape: ReadonlyArray<number>
}

/**
 * A model's parameters in {@link Model.parameters} order. Its length is
 * the model arity; parameterless models use the empty array.
 *
 * @since 0.1.0
 * @category models
 */
export type Params = ReadonlyArray<Tensor.Any>

/**
 * A model: stable parameter identities, an initializer that builds the
 * initial parameters as lazy graph values, and a forward function that
 * extends the graph — parameters and input in, lazy output out.
 *
 * Parameters are a flat array of tensors in `names` order, so the existing
 * training path (`Gradient.grad` followed by an optimizer step) works on any
 * model with zero adapter code. Storage precision is tensor metadata and does
 * not change the model parameter contract.
 *
 * @since 0.1.0
 * @category models
 */
export interface Model {
  /** Logical parameter specifications in parameter-array order. */
  readonly parameters: ReadonlyArray<ParameterSpec>
  /**
   * Stable parameter identities, one per parameter, in the same order
   * as the parameter array. Also serves as the model's arity;
   * parameterless models have no names and arity zero.
   */
  readonly names: ReadonlyArray<string>
  /**
   * Builds the initial parameters as lazy graph values. Materialize them
   * once with {@link Tensor.compute} before retaining them across separate
   * evaluations; {@link inference} performs that materialization eagerly.
   */
  readonly init: Effect.Effect<Params, ModelError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Extends the graph: parameters and input in, lazy output out.
   * Single-input, single-output; differentiated as-is by
   * `Gradient.grad`. Parameterized layers and arity-aware combinators fail
   * with a {@link ModelError} if `params.length` is wrong. A directly
   * invoked parameterless constructor ignores the array, but callers
   * should still pass `[]`; {@link Model.execute} always enforces arity.
   */
  readonly forward: (
    params: Params,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Lazy, ModelError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Runs the compiled forward program: parameters and input in,
   * materialized output out. With concrete inputs, each invocation makes one
   * native program call after the first call per signature pays the trace;
   * lazy inputs are first materialized through the common executable path. The
   * cache signature includes runtime identity and every parameter and input
   * tensor's shape, dtype, storage encoding, and placement; values do not affect
   * it. Any signature change traces a new program automatically. Ready entries
   * use a 32-entry LRU, so
   * eviction, clearing, or a failed trace can retrace a previously seen
   * signature. `execute` borrows and does not retain parameter
   * values, so materialize lazy initializers once before repeated calls. Use it
   * for evaluation loops; use `forward` wherever a graph is being built
   * (training, composition, differentiation). The returned concrete output is
   * caller-owned and should be released with {@link Tensor.clear} when unused.
   */
  readonly execute: (
    params: Params,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Concrete, ModelError | Tensor.TensorError, Runtime.Runtime>
  /**
   * JavaScript signature-cache diagnostics. `compiled` counts trace attempts,
   * including failures and retraces, rather than native cold compilations.
   */
  readonly stats: Effect.Effect<Tensor.CompileStats>
  /**
   * Clears JavaScript forward-program entries without clearing parameters,
   * outputs, native caches, or the cumulative trace-attempt count.
   */
  readonly clear: Effect.Effect<void>
}

/**
 * A public model definition. Omitting `init` creates a load-only model.
 *
 * @since 0.1.0
 * @category models
 */
export interface Definition {
  readonly parameters: ReadonlyArray<ParameterSpec>
  readonly init?: Effect.Effect<Params, Tensor.TensorError, Runtime.Runtime>
  readonly forward: Model["forward"]
}

interface ModelDef {
  readonly parameters: ReadonlyArray<ParameterSpec>
  readonly init: Effect.Effect<Params, ModelError | Tensor.TensorError, Runtime.Runtime>
  readonly forward: Model["forward"]
}

type ModelInternal =
  & {
    -readonly [K in keyof Model]: Model[K]
  }
  & { _fn: Tensor.CompiledFn<ModelError | Tensor.TensorError, Runtime.Runtime> | undefined }

// Every model is compiled: `execute` runs the forward as a frozen
// program on the shared prototype; the program cache is created on the
// first execute and the trace runs on the first call per input
// signature, so constructors stay device-free.
const ModelProto = {
  execute(this: ModelInternal, params: Params, input: Tensor.Any) {
    const self = this
    return Effect.gen(function*() {
      yield* checkArity("execute", self.names, params)
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
    const self = this as ModelInternal
    return Effect.suspend(() => self._fn?.stats ?? Effect.succeed({ cached: 0, compiled: 0 }))
  },
  get clear() {
    const self = this as ModelInternal
    return Effect.suspend(() => self._fn?.clear ?? Effect.void)
  }
}

const make = (def: ModelDef): Model => {
  const self = Object.create(ModelProto) as ModelInternal
  self.parameters = def.parameters
  self.names = def.parameters.map((parameter) => parameter.name)
  self.init = def.init
  self.forward = def.forward
  self._fn = undefined
  return self
}

/**
 * Validates and constructs a model with the standard compiled execution path.
 *
 * @since 0.1.0
 * @category constructors
 */
export const define = (definition: Definition): Effect.Effect<Model, ModelError> =>
  Effect.gen(function*() {
    const seen = new Set<string>()
    for (const parameter of definition.parameters) {
      if (typeof parameter.name !== "string" || parameter.name.length === 0) {
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
    }
    return make({
      parameters: definition.parameters,
      init: definition.init ?? new ModelError({
        op: "init",
        message: "model has no initializer; load parameters before use"
      }),
      forward: definition.forward
    })
  })

const checkName = (op: string, name: string): Effect.Effect<void, ModelError> =>
  name.length === 0 ? new ModelError({ op, message: "name must not be empty" }) : Effect.void

const checkPositiveInt = (op: string, field: string, value: number): Effect.Effect<void, ModelError> =>
  Number.isInteger(value) && value >= 1
    ? Effect.void
    : new ModelError({ op, message: `${field} must be a positive integer, got ${value}` })

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
    parameters: [],
    init: Effect.succeed<Params>([]),
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
      parameters: [
        { name: names[0], shape: [inFeatures, outFeatures] },
        { name: names[1], shape: [1, outFeatures] }
      ],
      init: Effect.gen(function*() {
        const drawn = yield* Tensor.randn([inFeatures, outFeatures])
        const weight = yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, 1 / Math.sqrt(inFeatures)))
        const bias = yield* Tensor.zeros([1, outFeatures])
        return [weight, bias] as const
      }),
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
      parameters: [
        { name: names[0], shape: [outChannels, inChannels / groups, kernelSize] },
        { name: names[1], shape: [outChannels] }
      ],
      init: Effect.gen(function*() {
        const drawn = yield* Tensor.randn([outChannels, inChannels / groups, kernelSize])
        const weight = yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, 1 / Math.sqrt(fanIn)))
        const bias = yield* Tensor.zeros([outChannels])
        return [weight, bias] as const
      }),
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
    const [kh, kw] = typeof kernelSize === "number" ? [kernelSize, kernelSize] as const : kernelSize
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
      parameters: [
        { name: names[0], shape: [outChannels, inChannels / groups, kh, kw] },
        { name: names[1], shape: [outChannels] }
      ],
      init: Effect.gen(function*() {
        const drawn = yield* Tensor.randn([outChannels, inChannels / groups, kh, kw])
        const weight = yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, 1 / Math.sqrt(fanIn)))
        const bias = yield* Tensor.zeros([outChannels])
        return [weight, bias] as const
      }),
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
 * `[0, numEmbeddings)`.
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
      parameters: [{ name: names[0], shape: [numEmbeddings, embeddingDim] }],
      init: Effect.gen(function*() {
        const weight = yield* Tensor.randn([numEmbeddings, embeddingDim])
        return [weight] as const
      }),
      forward: (params, input) =>
        Effect.gen(function*() {
          yield* checkArity(name, names, params)
          return yield* Tensor.embedding(input, {
            weight: params[0],
            ...(options.paddingIndex !== undefined ? { paddingIndex: options.paddingIndex } : {})
          })
        })
    })
  })

/**
 * A learned absolute position embedding (GPT-style `wpe`): looks up rows
 * `0..t-1` of a `[maxPositions, embeddingDim]` table, where `t` is the
 * input's last dimension — the input's values and leading dimensions are
 * ignored. The output is `[t, embeddingDim]` (with no copied batch axis).
 * `names = ["<name>.weight"]`, initialized unit-normal. Fails with a
 * {@link ModelError} on an empty name, counts that are not positive
 * integers, or an input whose sequence length exceeds `maxPositions`.
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
      parameters: [{ name: names[0], shape: [maxPositions, embeddingDim] }],
      init: Effect.gen(function*() {
        const weight = yield* Tensor.randn([maxPositions, embeddingDim])
        return [weight] as const
      }),
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
    const shape: ReadonlyArray<number> = typeof normalizedShape === "number" ? [normalizedShape] : normalizedShape
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
      parameters: [
        { name: names[0], shape },
        { name: names[1], shape }
      ],
      init: Effect.gen(function*() {
        const weight = yield* Tensor.ones(shape)
        const bias = yield* Tensor.zeros(shape)
        return [weight, bias] as const
      }),
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
   * 10000). Use a positive finite base; it is not validated here. RoPE itself
   * introduces no learned-position-table limit, but a separately composed
   * position embedding still does. Cached generation can outgrow `maxTokens`
   * only when inference also uses an attention window that evicts old KV
   * blocks, and remains subject to backend numeric and resource limits. The
   * per-head dimension must be even and the attention input must be f32 or
   * bf16 when RoPE is enabled.
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
 * `numHeads`. RoPE-specific shape or dtype violations surface as
 * {@link Tensor.TensorError}s from `forward`.
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
    // Fused QKV projection: one [E, 3E] gemm+epilogue instead of three
    // [E, E] linears — one launch forward, one per gradient direction
    // backward, at 1/3 the matmul count.
    const names = [
      `${name}.qkv.weight`,
      `${name}.qkv.bias`,
      `${name}.wo.weight`,
      `${name}.wo.bias`
    ]
    const causal = options.causal ?? false
    return make({
      parameters: [
        { name: names[0], shape: [embedDim, 3 * embedDim] },
        { name: names[1], shape: [1, 3 * embedDim] },
        { name: names[2], shape: [embedDim, embedDim] },
        { name: names[3], shape: [1, embedDim] }
      ],
      init: Effect.gen(function*() {
        const qkvDrawn = yield* Tensor.randn([embedDim, 3 * embedDim])
        const qkvWeight = yield* Tensor.mul(qkvDrawn, yield* Tensor.constantLike(qkvDrawn, 1 / Math.sqrt(embedDim)))
        const qkvBias = yield* Tensor.zeros([1, 3 * embedDim])
        const woDrawn = yield* Tensor.randn([embedDim, embedDim])
        const woWeight = yield* Tensor.mul(woDrawn, yield* Tensor.constantLike(woDrawn, 1 / Math.sqrt(embedDim)))
        const woBias = yield* Tensor.zeros([1, embedDim])
        return [qkvWeight, qkvBias, woWeight, woBias] as const
      }),
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
            options.rope !== undefined ? Tensor.rotaryEmbedding(x, t, options.rope) : Effect.succeed(x as Tensor.Any)
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
 * decayed state transition and apply **no** positional encoding; in a
 * hybrid stack the full-attention layers can therefore omit RoPE (the
 * Kimi K3 configuration).
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
 * provide second-order derivatives.
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
    const scaled = (fanIn: number, shape: ReadonlyArray<number>) =>
      Effect.gen(function*() {
        const drawn = yield* Tensor.randn(shape)
        return yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, 1 / Math.sqrt(fanIn)))
      })
    return make({
      parameters: [
        { name: names[0], shape: [embedDim, 3 * embedDim] },
        { name: names[1], shape: [1, 3 * embedDim] },
        { name: names[2], shape: [3 * embedDim, 4] },
        { name: names[3], shape: [embedDim, headDim] },
        { name: names[4], shape: [headDim, embedDim] },
        { name: names[5], shape: [numHeads] },
        { name: names[6], shape: [embedDim] },
        { name: names[7], shape: [embedDim, numHeads] },
        { name: names[8], shape: [embedDim, headDim] },
        { name: names[9], shape: [headDim, embedDim] },
        { name: names[10], shape: [headDim] },
        { name: names[11], shape: [embedDim, embedDim] },
        { name: names[12], shape: [1, embedDim] }
      ],
      init: Effect.gen(function*() {
        return [
          yield* scaled(embedDim, [embedDim, 3 * embedDim]),
          yield* Tensor.zeros([1, 3 * embedDim]),
          yield* scaled(4, [3 * embedDim, 4]),
          yield* scaled(embedDim, [embedDim, headDim]),
          yield* scaled(headDim, [headDim, embedDim]),
          yield* Tensor.zeros([numHeads]),
          yield* Tensor.zeros([embedDim]),
          yield* scaled(embedDim, [embedDim, numHeads]),
          yield* scaled(embedDim, [embedDim, headDim]),
          yield* scaled(headDim, [headDim, embedDim]),
          yield* Tensor.full([headDim], 1),
          yield* scaled(embedDim, [embedDim, embedDim]),
          yield* Tensor.zeros([1, embedDim])
        ] as const
      }),
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
          // One causal depthwise short convolution (kernel 4) + SiLU over
          // the fused [.., T, 3E] projection — contiguous input, one
          // launch — then the q/k/v slices.
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
 * Flattens the input into `[batch, features]` as a parameterless model:
 * `startDim` defaults to **1** (the batch dimension is preserved, the
 * common case between the convolutional and the fully-connected part of a
 * network) and `endDim` to the last dimension. Both must be integer axes
 * in range, and `endDim` must not precede `startDim`.
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
      ...(options.endDim !== undefined ? { endDim: options.endDim } : {})
    })
  )

/**
 * Inverted dropout as a parameterless model: zeroes elements with
 * probability `p` (default `0.5`) and scales survivors by `1 / (1 - p)`.
 * This is the functional form — it **always applies**; build the
 * evaluation chain without it (dropout adds nothing to the parameter
 * array, so one checkpoint serves both chains). The mask follows
 * {@link Tensor.randn}'s per-invocation sharing rule: submit a loss and its
 * gradients as roots of the same invocation when they must share it. Fails
 * with a {@link ModelError} if `p` is outside `[0, 1)`. The range check does
 * not reject `NaN`; do not pass it as a probability.
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
      parameters: [],
      init: Effect.succeed<Params>([]),
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
    const [kh, kw] = typeof options.kernelSize === "number"
      ? [options.kernelSize, options.kernelSize] as const
      : options.kernelSize
    yield* checkPositiveInt(op, "kernelSize", kh)
    yield* checkPositiveInt(op, "kernelSize", kw)
    if (options.stride !== undefined) {
      const [sh, sw] = typeof options.stride === "number" ? [options.stride, options.stride] as const : options.stride
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
      parameters: [],
      init: Effect.succeed<Params>([]),
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
 * Wraps a sub-model in a gradient-checkpoint boundary: the forward value
 * is unchanged, but during the backward pass the sub-model's forward
 * intermediates are recomputed from a fresh copy instead of being
 * retained — trading one extra forward evaluation of the block for its
 * peak activation memory. Region inputs (parameters, the incoming
 * activation, constructor draws) stay shared, so recomputation is
 * consistent with the forward pass.
 *
 * Apply it per block, not to the whole model: checkpointing the full
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
 * This is the recompute mechanism — meaningful on every target.
 *
 * @since 0.1.0
 * @category combinators
 */
export const checkpoint = (model: Model): Effect.Effect<Model> =>
  Effect.succeed(make({
    parameters: model.parameters,
    init: model.init,
    forward: (params, input) => Effect.flatMap(model.forward(params, input), Gradient.checkpoint)
  }))

/**
 * Adds a residual (skip) connection around a sub-model: the forward is
 * `input + block(input)`. Names and init are the sub-model's; the
 * sub-model's output must be broadcast-compatible with its input (an
 * equal shape in the standard usage — transformer blocks, ResNet
 * stages).
 *
 * @since 0.1.0
 * @category combinators
 */
export const residual = (model: Model): Effect.Effect<Model> =>
  Effect.succeed(make({
    parameters: model.parameters,
    init: model.init,
    forward: (params, input) =>
      Effect.gen(function*() {
        const out = yield* model.forward(params, input)
        return yield* Tensor.add(input, out)
      })
  }))

/**
 * Transforms a model's input before it enters the sub-model:
 * `forward(params, input) = model.forward(params, f(input))`. Names and
 * init are the sub-model's. Use it for input derived from the raw
 * input's shape or values when no dedicated layer covers the case
 * (position embeddings have their own: {@link positionEmbedding}).
 *
 * @since 0.1.0
 * @category combinators
 */
export const mapInput = (
  model: Model,
  f: (input: Tensor.Any) => Effect.Effect<Tensor.Any, Tensor.TensorError, Runtime.Runtime>
): Effect.Effect<Model> =>
  Effect.succeed(make({
    parameters: model.parameters,
    init: model.init,
    forward: (params, input) => Effect.flatMap(f(input), (mapped) => model.forward(params, mapped))
  }))

/**
 * Fans one input into several sub-models and combines their outputs:
 * `forward(params, input) = f(...models.map(m => m.forward(mParams,
 * input)))`. `names` is the concatenation of the models' names (in
 * order), sliced by arity in `forward`; `init` runs each model's `init`
 * in order. The combiner is variadic with one argument per model, in
 * the same order (inferred from the tuple). Fails with a
 * {@link ModelError} when the array is empty or when parameter names
 * collide.
 *
 * The common case — adding the branches, as in token + position
 * embeddings — has its own combinator: {@link add}. {@link residual} is
 * the special case where one branch is the identity.
 *
 * @since 0.1.0
 * @category combinators
 */
export const merge = <const M extends ReadonlyArray<Model>>(
  models: M,
  f: (...outputs: { [K in keyof M]: Tensor.Lazy }) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>
): Effect.Effect<Model, ModelError> => {
  if (models.length === 0) {
    return new ModelError({ op: "merge", message: "at least one model is required" })
  }
  const parameters = models.flatMap((model) => model.parameters)
  const names = parameters.map((parameter) => parameter.name)
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
  const arities = models.map((model) => model.names.length)
  return Effect.succeed(make({
    parameters,
    init: Effect.gen(function*() {
      const params: Array<Tensor.Any> = []
      for (const model of models) {
        params.push(...(yield* model.init))
      }
      return params
    }),
    forward: (params, input) =>
      Effect.gen(function*() {
        yield* checkArity("merge", names, params)
        const outputs: Array<Tensor.Lazy> = []
        let offset = 0
        for (let i = 0; i < models.length; i++) {
          outputs.push(yield* models[i].forward(params.slice(offset, offset + arities[i]), input))
          offset += arities[i]
        }
        return yield* f(...(outputs as { [K in keyof M]: Tensor.Lazy }))
      })
  }))
}

/**
 * Adds the outputs of several models over a shared input elementwise:
 * `forward(params, input) = Σᵢ models[i].forward(paramsᵢ, input)` with
 * each model's parameters sliced by arity from the concatenated array.
 * The standard non-sequential top — token + position embeddings is
 * `add(wte, wpe)`; {@link residual} is the special case where one branch
 * is the identity. `names` and `init` follow {@link merge}. Fails with a
 * {@link ModelError} when the chain is empty or parameter names collide.
 *
 * @since 0.1.0
 * @category combinators
 */
export const add = (...models: ReadonlyArray<Model>): Effect.Effect<Model, ModelError> =>
  merge(models, (first, ...rest) =>
    Effect.gen(function*() {
      let acc: Tensor.Any = first
      for (const output of rest) {
        acc = yield* Tensor.add(acc, output)
      }
      return acc as Tensor.Lazy
    }))

/**
 * Composes models into a single model that threads its input through each
 * child in order, slicing each child's share of the concatenated
 * parameter array by its arity (`names.length`). `names` is the
 * concatenation of the children's names and `init` runs each child's
 * `init` in order.
 *
 * Fails with a {@link ModelError} when the chain is empty or when
 * parameter names collide — a collision would silently overwrite entries
 * in a saved checkpoint.
 *
 * @since 0.1.0
 * @category combinators
 */
export const chain = (...models: ReadonlyArray<Model>): Effect.Effect<Model, ModelError> => {
  if (models.length === 0) {
    return new ModelError({ op: "chain", message: "at least one model is required" })
  }
  const parameters = models.flatMap((model) => model.parameters)
  const names = parameters.map((parameter) => parameter.name)
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
  const arities = models.map((model) => model.names.length)
  return Effect.succeed(make({
    parameters,
    init: Effect.gen(function*() {
      const params: Array<Tensor.Any> = []
      for (const model of models) {
        params.push(...(yield* model.init))
      }
      return params
    }),
    forward: (params, input) =>
      Effect.gen(function*() {
        yield* checkArity("chain", names, params)
        let current: Tensor.Any = input
        let offset = 0
        for (let i = 0; i < models.length; i++) {
          current = yield* models[i].forward(params.slice(offset, offset + arities[i]), current)
          offset += arities[i]
        }
        return current as Tensor.Lazy
      })
  }))
}

/**
 * Saves a model's parameters to a safetensors file, zipping `model.names`
 * with the parameter array into the record {@link Tensor.save} takes.
 * Fails with a {@link ModelError} if the parameter array's length does
 * not match the model's arity. Saving borrows parameters and does not clear them.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = (
  model: Model,
  params: Params,
  path: string
): Effect.Effect<void, ModelError | Tensor.TensorError, Runtime.Runtime> =>
  params.length !== model.names.length
    ? new ModelError({
      op: "save",
      message: `model has ${model.names.length} parameters, got ${params.length}`
    })
    : Tensor.save(
      path,
      Object.fromEntries(model.names.map((name, i) => [name, params[i]]))
    )

/**
 * Loads a model's parameters from a safetensors file written by
 * {@link save}, returning the materialized tensors in `model.names` order
 * — the same array `forward` and optimizer steps expect. A missing key
 * fails with a {@link ModelError}; extra keys are ignored. {@link Tensor.load}
 * materializes the whole archive, so extra entries, and all imported entries
 * after a missing-key failure, are left to native finalization. Shape or dtype
 * mismatches against the architecture surface as graph-build errors on
 * first use. Loading for a parameterless model therefore returns `[]` and
 * ignores every stored tensor. Returned handles are caller-owned and should be
 * released with {@link Tensor.clear} when no longer needed.
 *
 * @since 0.1.0
 * @category destructors
 */
export const load = (
  model: Model,
  path: string
): Effect.Effect<ReadonlyArray<Tensor.Concrete>, ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const record = yield* Tensor.load(path)
    const params: Array<Tensor.Concrete> = []
    for (const name of model.names) {
      const param = record[name]
      if (param === undefined) {
        return yield* new ModelError({
          op: "load",
          message: `missing parameter "${name}" in ${path}`
        })
      }
      params.push(param)
    }
    return params
  })

/**
 * A failure in inference-artifact construction or generation: invalid
 * configuration or model structure, or misuse of the generation calling
 * convention. The `op` payload identifies `inference`, `add`, `prefill`,
 * or `step`; `message` is diagnostic text. Pool-capacity, position-table,
 * and other native execution failures stay {@link Tensor.TensorError}s.
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
 * Deployment configuration for {@link inference}. Compilation is eager
 * and fixed-shape: prefill and single-sequence decode are always built,
 * with a third batched-decode program when `decodeBatch > 1`.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface InferenceConfig {
  /**
   * KV capacity in token rows, shared by live sequences and the resident
   * prefix cache. Must be a positive integer and an exact multiple of
   * `blockSize`.
   */
  readonly maxTokens: number
  /**
   * KV paging granularity in tokens. Must be a positive integer that
   * divides `maxTokens`. Defaults to 16.
   */
  readonly blockSize?: number
  /**
   * Positive integer number of cached positions attended per step, no greater
   * than `maxTokens`. Omit for full attention. A window permits old KV blocks to be evicted only
   * when every attention operation resolves to a local window; an explicit full-attention
   * operation retains full history. With RoPE and no other absolute-position state this can extend generation
   * beyond `maxTokens` when the pool can hold the active window and block
   * frontier, subject to backend numeric and resource limits. Learned
   * position tables still impose their absolute cursor limit.
   */
  readonly attentionWindow?: number
  /**
   * Positive integer fixed prompt-chunk length. Defaults to `blockSize`;
   * the final chunk is zero-padded and only real rows enter the KV cache.
   * Padded positions are still evaluated, so every chunk's cursor-offset
   * extent must fit any learned position table.
   */
  readonly prefillChunk?: number
  /**
   * Token-id dtype used by the fixed programs. Defaults to `"u32"`;
   * prompts passed to {@link Generation.add} must use this dtype.
   */
  readonly tokenDtype?: "u32" | "i64"
  /**
   * KV storage dtype. Defaults to `"f32"`; `"f16"` and `"bf16"` narrow
   * rows on write and attention widens them to f32. `"int8"` uses symmetric
   * per-token, per-head quantization with f32 scales.
   */
  readonly kvDtype?: "f32" | "f16" | "bf16" | "int8"
  /**
   * Positive integer maximum live sequences per session and entries per
   * step. Defaults to 8. Values above 1 compile a fixed `[decodeBatch, 1]`
   * batched program in addition to the single-sequence program.
   */
  readonly decodeBatch?: number
}

/**
 * One live sequence inside a {@link Generation} session: a block table and
 * logical token cursor. Created by {@link Generation.add}; there is no
 * `Scope` lifetime. Call {@link GenerationSeq.finish} or
 * {@link Generation.close} for deterministic release. Native finalization
 * is a fallback when an abandoned sequence becomes unreachable.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationSeq {
  /**
   * Low-level KV handle owned by this sequence. Do not release it directly;
   * it becomes invalid after {@link GenerationSeq.finish} or its session's
   * {@link Generation.close}.
   */
  readonly sequence: Tensor.KvSequence
  /**
   * Returns the total logical token count, including evicted window
   * positions. Fails after the underlying sequence has been released.
   */
  readonly cursor: () => Effect.Effect<number, Tensor.TensorError, Runtime.Runtime>
  /**
   * Removes this sequence from its session and returns its KV blocks. Calls
   * after it has already been finished or closed are no-ops.
   */
  readonly finish: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * The result of {@link Generation.add}: the new sequence's handle and
 * its prompt's final-position logits `[vocab]` — the distribution the
 * first generated token is sampled from.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationEntry {
  /** The new live sequence handle. */
  readonly seq: GenerationSeq
  /** Caller-owned final-prompt-position logits with shape `[vocab]`. */
  readonly logits: Tensor.Concrete
}

/**
 * A generation session over an {@link InferenceProgram}'s pool. Sequences
 * are added individually with {@link Generation.add}, which performs
 * chunked prefill; {@link Generation.step} advances one or more in one
 * run. The entries are the batch: one entry uses the `[1, 1]` program and
 * more use one fixed-width batched run with native padding. The pool keeps a content-addressed prefix cache: prompts whose
 * leading blocks are already resident (computed by an earlier, since
 * finished or still-live sequence) reuse them and compute only their suffix.
 * Hybrid recurrent artifacts also publish KDA and short-convolution state
 * snapshots at completed block boundaries, so matching restores both. Purely
 * recurrent artifacts without KV blocks have no block-anchored prefix cache.
 *
 * Sessions are ordinary values and require no `Scope`. Independent
 * sessions may run concurrently. Calls to `step` on one session are
 * serialized; `add`, `finish`, and `close` are not covered by that lock,
 * so callers must not overlap lifecycle mutations with other operations
 * on the same session.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface Generation {
  /**
   * Prefills `prompt` (`[1, T]` token ids) as a new live sequence and
   * returns its handle and final-position logits. `T` must be at least 1,
   * and the tensor must use the inference runtime, placement, and configured
   * `tokenDtype`. Adding beyond `decodeBatch` live sequences fails with an
   * {@link InferenceError}; token/vocabulary and pool failures may surface
   * as {@link Tensor.TensorError}s.
   */
  readonly add: (
    prompt: Tensor.Any
  ) => Effect.Effect<GenerationEntry, InferenceError | ModelError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Advances every entry's sequence by one token in one run and returns
   * caller-owned materialized `[vocab]` logits in entry order. Finishing a
   * sequence or closing the session does not release previously returned
   * logits. The array must be nonempty
   * and contain at most `decodeBatch` distinct live sequences from this
   * session. Each token must be a non-negative integer representable by the
   * configured token dtype and valid for the model vocabulary. Calls on the
   * same session serialize; admission and scheduling stay with the caller.
   */
  readonly step: (
    entries: ReadonlyArray<{
      /** A distinct live sequence created by this session. */
      readonly seq: GenerationSeq
      /** The next token id, as a non-negative integer. */
      readonly token: number
    }>
  ) => Effect.Effect<
    ReadonlyArray<Tensor.Concrete>,
    InferenceError | ModelError | Tensor.TensorError,
    Runtime.Runtime
  >
  /** Returns the number of sequences currently live in this session. */
  readonly live: () => Effect.Effect<number>
  /**
   * Releases every live sequence's blocks and invalidates their low-level
   * handles. The session remains usable and can accept new sequences after
   * closing. Without this call, native finalizers release abandoned
   * sequences when the session and handles become unreachable.
   */
  readonly close: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * A compiled inference artifact: frozen chunked-prefill and
 * single-sequence-decode programs, plus a fixed-width batched-decode
 * program when `decodeBatch > 1`, and their shared KV pool. All programs
 * are derived from the model and compiled eagerly at construction. This is
 * not a {@link Model}; generation runs through {@link Generation} sessions.
 * The artifact is immutable and parallel-safe. It has no `Scope` or
 * explicit lifetime: native finalizers release unreachable programs, pool,
 * and abandoned sequences, while {@link Generation.close} and
 * {@link GenerationSeq.finish} provide deterministic block release.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface InferenceProgram {
  /**
   * Opens an independent generation session. No `Scope` service is
   * required. Use {@link Generation.close} for deterministic session
   * cleanup or {@link GenerationSeq.finish} for one sequence; native
   * finalizers are the fallback.
   */
  readonly generation: () => Effect.Effect<Generation, InferenceError>
}

interface ResolvedInferenceConfig {
  readonly maxTokens: number
  readonly blockSize: number
  readonly prefillChunk: number
  readonly tokenDtype: "u32" | "i64"
  readonly kvDtype: Tensor.DType
  readonly decodeBatch: number
  readonly attentionWindow?: number
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
    const prefillChunk = config.prefillChunk ?? blockSize
    if (!Number.isInteger(prefillChunk) || prefillChunk <= 0) {
      return yield* invalidInferenceConfig(`prefillChunk must be a positive integer, got ${config.prefillChunk}`)
    }
    const tokenDtype = config.tokenDtype ?? "u32"
    if (tokenDtype !== "u32" && tokenDtype !== "i64") {
      return yield* invalidInferenceConfig(`tokenDtype must be u32 or i64, got ${String(config.tokenDtype)}`)
    }
    const configuredKvDtype = config.kvDtype ?? "f32"
    if (!["f32", "f16", "bf16", "int8"].includes(configuredKvDtype)) {
      return yield* invalidInferenceConfig(`unsupported kvDtype ${String(config.kvDtype)}`)
    }
    const decodeBatch = config.decodeBatch ?? 8
    if (!Number.isInteger(decodeBatch) || decodeBatch <= 0) {
      return yield* invalidInferenceConfig(`decodeBatch must be a positive integer, got ${config.decodeBatch}`)
    }
    return {
      maxTokens: config.maxTokens,
      blockSize,
      prefillChunk,
      tokenDtype,
      kvDtype: configuredKvDtype === "int8" ? "u8" : configuredKvDtype,
      decodeBatch,
      ...(config.attentionWindow === undefined ? {} : { attentionWindow: config.attentionWindow })
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
  readonly window?: number
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
  ...(program.window === undefined ? {} : { window: program.window })
})

const sameDecodeGeometry = (left: DecodeGeometry, right: DecodeGeometry): boolean =>
  left.layers === right.layers && left.kvHeads === right.kvHeads && left.headDim === right.headDim &&
  left.kdaLayers === right.kdaLayers && left.kdaHeads === right.kdaHeads &&
  left.kdaHeadDim === right.kdaHeadDim && left.kdaValueDim === right.kdaValueDim &&
  left.convLayers === right.convLayers && left.convChannels === right.convChannels &&
  left.convKernel === right.convKernel && left.window === right.window

interface InferencePrograms {
  readonly prefill: Tensor.DecodeProgram
  readonly decode: Tensor.DecodeProgram
  readonly batched: Tensor.DecodeProgram | undefined
  readonly geometry: DecodeGeometry
  readonly pool: Tensor.KvPool
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

const traceInferenceProgram = (
  model: Model,
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedInferenceConfig,
  inputShape: readonly [number, number]
): Effect.Effect<
  Tensor.DecodeProgram,
  InferenceError | ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    const [programBatch, steps] = inputShape
    const tokenInput = yield* Tensor.zeros(inputShape, { dtype: config.tokenDtype })
    const input = yield* Tensor.makeInput(0, tokenInput)
    const output = yield* model.forward(frozenParams, input)
    yield* logitsVocab(output, programBatch, steps)
    return yield* Tensor.compileDecodeProgram([output], {
      maxTokens: config.maxTokens,
      blockSize: config.blockSize,
      kvDtype: config.kvDtype,
      batch: programBatch,
      lastTokenRow: true,
      ...(config.attentionWindow === undefined ? {} : { window: config.attentionWindow })
    }).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))
  })

const compileInferencePrograms = (
  model: Model,
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedInferenceConfig
): Effect.Effect<
  InferencePrograms,
  InferenceError | ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    const prefill = yield* traceInferenceProgram(model, frozenParams, config, [1, config.prefillChunk])
    const decode = yield* traceInferenceProgram(model, frozenParams, config, [1, 1])
    const batched = config.decodeBatch > 1
      ? yield* traceInferenceProgram(model, frozenParams, config, [config.decodeBatch, 1])
      : undefined
    const geometry = decodeGeometry(prefill)
    if (
      !sameDecodeGeometry(geometry, decodeGeometry(decode)) ||
      (batched !== undefined && !sameDecodeGeometry(geometry, decodeGeometry(batched)))
    ) {
      return yield* new InferenceError({
        op: "inference",
        message: "prefill and decode traces disagree on attention geometry or retention policy"
      })
    }
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
    return { prefill, decode, batched, geometry, pool }
  })

interface PrefillChunkPlan {
  readonly offset: number
  readonly real: number
  readonly final: boolean
}

const planPrefillChunks = (
  total: number,
  matched: number,
  chunk: number
): ReadonlyArray<PrefillChunkPlan> => {
  const chunks: Array<PrefillChunkPlan> = []
  for (let offset = matched; offset < total; offset += chunk) {
    const real = Math.min(chunk, total - offset)
    chunks.push({ offset, real, final: offset + real === total })
  }
  return chunks
}

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
        if (typeof value !== "bigint" || value < 0n || value > 0xffff_ffffn) {
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

const prefillInput = (
  prompt: Tensor.Any,
  chunk: PrefillChunkPlan,
  config: ResolvedInferenceConfig
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    let input = yield* Tensor.slice(prompt, {
      start: [0, chunk.offset],
      end: [1, chunk.offset + chunk.real]
    })
    if (chunk.real < config.prefillChunk) {
      input = yield* Tensor.concat([
        input,
        yield* Tensor.zeros([1, config.prefillChunk - chunk.real], { dtype: config.tokenDtype })
      ], { dim: 1 })
    }
    return input
  })

interface LiveEntry {
  readonly seq: GenerationSeq
}

const releaseLiveEntry = (live: Array<LiveEntry>, entry: LiveEntry) =>
  Effect.gen(function*() {
    const index = live.indexOf(entry)
    if (index < 0) return
    yield* Tensor.releaseKvSequence(entry.seq.sequence)
    live.splice(index, 1)
  })

const closeLiveEntries = (
  live: Array<LiveEntry>
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const entries = live.splice(0)
    let failure: Exit.Failure<void, Tensor.TensorError> | undefined
    for (const entry of entries) {
      const exit = yield* Effect.exit(Tensor.releaseKvSequence(entry.seq.sequence))
      if (Exit.isFailure(exit)) {
        live.push(entry)
        failure ??= exit
      }
    }
    if (failure !== undefined) {
      return yield* Effect.failCause(failure.cause)
    }
  })

const validateStepEntries = (
  live: ReadonlyArray<LiveEntry>,
  decodeBatch: number,
  entries: ReadonlyArray<{ readonly seq: GenerationSeq; readonly token: number }>
): Effect.Effect<void, InferenceError> =>
  Effect.gen(function*() {
    if (entries.length === 0) {
      return yield* new InferenceError({ op: "step", message: "step expects at least one entry" })
    }
    if (entries.length > decodeBatch) {
      return yield* new InferenceError({
        op: "step",
        message: `step accepts at most decodeBatch (${decodeBatch}) entries, got ${entries.length}`
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
}

const openGeneration = (engine: InferenceEngine): Effect.Effect<Generation, never> =>
  Effect.gen(function*() {
    const roundLock = yield* Semaphore.make(1)
    const live: Array<LiveEntry> = []
    const config = engine.config
    const programs = engine.programs
    const add: Generation["add"] = (prompt) =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        if (live.length >= config.decodeBatch) {
          return yield* new InferenceError({
            op: "add",
            message: `a session holds at most decodeBatch (${config.decodeBatch}) live sequences; finish one first`
          })
        }
        yield* validatePrompt(prompt, config, runtime)
        const [materializedPrompt] = yield* Tensor.compute([prompt])
        return yield* Effect.gen(function*() {
          const ids = yield* readTokenIds(materializedPrompt)
          const sequence = yield* Tensor.makeKvSequence(programs.pool)
          return yield* Effect.gen(function*() {
            const matched = yield* Tensor.kvPrefillMatch(sequence, ids)
            let logits: Tensor.Concrete | undefined
            for (const chunk of planPrefillChunks(ids.length, matched, config.prefillChunk)) {
              const input = yield* prefillInput(materializedPrompt, chunk, config)
              const [output] = yield* Tensor.runDecodeProgram(
                programs.prefill,
                [input],
                sequence,
                ids.slice(chunk.offset, chunk.offset + chunk.real)
              )
              if (chunk.final) {
                logits = output
              } else {
                yield* Tensor.clear(output)
              }
            }
            if (logits === undefined) {
              return yield* new InferenceError({ op: "prefill", message: "prefill produced no logits" })
            }
            const entry: LiveEntry = {
              seq: {
                sequence,
                cursor: () => Tensor.kvSequenceCursor(sequence),
                finish: () => releaseLiveEntry(live, entry)
              }
            }
            live.push(entry)
            return { seq: entry.seq, logits } satisfies GenerationEntry
          }).pipe(
            Effect.onExit((exit) =>
              Exit.isFailure(exit) ? Effect.ignore(Tensor.releaseKvSequence(sequence)) : Effect.void
            )
          )
        }).pipe(Effect.ensuring(Effect.ignore(Tensor.clear(materializedPrompt))))
      })
    const step: Generation["step"] = (entries) =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          yield* validateStepEntries(live, config.decodeBatch, entries)
          if (entries.length === 1) {
            const entry = entries[0]!
            const input = yield* tokenTensor([entry.token], [1, 1], config.tokenDtype)
            const [output] = yield* Tensor.runDecodeProgram(
              programs.decode,
              [input],
              entry.seq.sequence,
              [entry.token]
            )
            return [output]
          }
          if (programs.batched === undefined) {
            return yield* new InferenceError({
              op: "step",
              message: `stepping ${entries.length} sequences needs decodeBatch > 1`
            })
          }
          const ids = entries.map((entry) => entry.token)
          const input = yield* tokenTensor(ids, [entries.length, 1], config.tokenDtype)
          const outputs = yield* Tensor.runBatchedDecodeProgram(
            programs.batched,
            [input],
            entries.map((entry) => entry.seq.sequence),
            ids.map((id) => [id])
          )
          const selected = outputs.slice(0, entries.length)
          yield* Tensor.clearAll(outputs.slice(entries.length))
          return selected
        })
      )
    return {
      add,
      step,
      live: () => Effect.sync(() => live.length),
      close: () => closeLiveEntries(live)
    }
  })

/**
 * Compiles a model for generation. The same `forward` graph builder is
 * traced with placeholders and decode-specialized natively: causal attention
 * becomes paged KV attention and position nodes become cursor-offset operations.
 * Construction eagerly freezes prefill `[1, prefillChunk]` and decode
 * `[1, 1]` programs, plus batched decode `[decodeBatch, 1]` when
 * `decodeBatch > 1`. There is no shape-keyed growth or later tracing.
 *
 * The model must return logits exactly as `[batch, T, vocab]`, preserving
 * the two token-input axes; generation returns the advance-selected
 * final-position row as `[vocab]`, extracted natively by the decode
 * specialization (`lastTokenRow`). Stateless graphs are accepted. Causal attention, KDA,
 * short convolution, and position operations use incremental state or cursor
 * specialization when present. Non-causal attention and runtime scalar inputs
 * fail with an {@link InferenceError}. Learned position
 * embeddings remain bounded by their table's total absolute cursor, even
 * with an attention window.
 *
 * `params` are borrowed and eagerly materialized once with
 * {@link Tensor.compute} before tracing. This freezes lazy initializer draws for every later prefill and
 * step. Concrete parameters are retained by the compiled artifacts as
 * immutable constants, so generation calls bind only token rows. Caller-supplied
 * concrete handles are not consumed; the artifact retains its own materialized
 * generation until it becomes unreachable.
 *
 * The artifact and sessions require no `Scope`. Static programs and pool
 * memory are released by native finalizers when unreachable. Live sequences
 * pin pool blocks, so use {@link Generation.close} or
 * {@link GenerationSeq.finish} for prompt release under capacity pressure
 * rather than relying on GC timing.
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
    yield* checkArity("inference", model.names, params)
    const resolved = yield* resolveInferenceConfig(config)
    const frozenParams = yield* Tensor.compute(params)
    return yield* Effect.onExit(
      Effect.gen(function*() {
        const programs = yield* compileInferencePrograms(model, frozenParams, resolved)
        return {
          generation: () => openGeneration({ config: resolved, frozenParams, programs })
        } satisfies InferenceProgram
      }),
      (exit) =>
        Exit.isFailure(exit)
          ? Effect.forEach(frozenParams, (parameter) => Effect.ignore(Tensor.clear(parameter)), { discard: true })
          : Effect.void
    )
  })

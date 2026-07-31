/**
 * Models as pure values pairing parameter construction with a
 * parameterised forward graph builder — the Flax/Haiku design, flattened:
 * parameters are always a flat array of tensors, configuration lives in
 * the factory's closure, and the forward graph is an ordinary lazy graph,
 * so {@link Gradient.grad} differentiates it and {@link Optimizer.step}
 * updates it with zero model-specific code. There is no mutable module
 * state and no backward mode.
 *
 * Everything that can fail returns an `Effect`: factories validate their
 * configuration (positive feature counts, unique parameter names) into a
 * {@link ModelError}, `forward` checks the parameter array's length
 * against the model's arity, and checkpoints report arity and
 * missing-key problems in the error channel. Training lives in the
 * `Trainer` module.
 *
 * The layer catalog covers the standard MLP / CNN / embedding stack:
 * parameterised layers ({@link linear}, {@link conv1d}, {@link conv2d},
 * {@link embedding}, {@link layerNorm}) and parameterless ones (the
 * activations, {@link softmax}, {@link logSoftmax}, {@link flatten},
 * {@link dropout}, {@link maxPool2d}, {@link avgPool2d}). {@link chain}
 * composes models into a model with the same interface, concatenating
 * the parameter arrays in order. `names` gives every parameter a stable,
 * checkpoint-friendly identity that maps directly onto
 * {@link Tensor.save} / {@link Tensor.load} via {@link save} and
 * {@link load}. {@link compile} freezes the forward graph into a cached
 * native program — a compiled model is still a `Model`.
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
import { Data, Effect } from "effect"
import type { CurrentDevice } from "./Device.ts"
import * as Gradient from "./Gradient.ts"
import * as Tensor from "./Tensor.ts"

/**
 * A failure in model construction, checkpointing, or training:
 * invalid layer configuration, duplicate parameter names, parameter
 * array arity mismatches, missing checkpoint keys, or an invalid step
 * count. Failures from the underlying tensor operations stay
 * {@link Tensor.TensorError}s.
 *
 * @since 0.1.0
 * @category errors
 */
export class ModelError extends Data.TaggedError("ModelError")<{
  readonly op: string
  readonly message: string
}> {}

/**
 * A model's parameters: a flat array of tensors in `names` order.
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
 * Parameters are a flat array of tensors in `names` order, so the
 * existing training path (`Gradient.grad`, `Optimizer.step`) works on
 * any model with zero adapter code. Configuration (feature counts,
 * activation choice) lives in the closure of the factory, never in the
 * parameters.
 *
 * @since 0.1.0
 * @category models
 */
export interface Model {
  /**
   * Stable parameter identities, one per parameter, in the same order
   * as the parameter array. Also serves as the model's arity.
   */
  readonly names: ReadonlyArray<string>
  /**
   * Builds the initial parameters as lazy graph values. Materialization
   * happens in the first `Optimizer.step` walk (or an explicit
   * `Tensor.compute`), so initial `randn` draws are consistent with the
   * first loss within that walk.
   */
  readonly init: Effect.Effect<Params, Tensor.TensorError, CurrentDevice>
  /**
   * Extends the graph: parameters and input in, lazy output out.
   * Single-input, single-output; differentiated as-is by
   * `Gradient.grad`. Fails with a {@link ModelError} if `params.length`
   * does not match the model's arity. May require the current device
   * (layers that draw randomness or reshape on-device, like `dropout`
   * and the pools).
   */
  readonly forward: (
    params: Params,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Lazy, ModelError | Tensor.TensorError, CurrentDevice>
}

const CompiledTypeId: unique symbol = Symbol.for("@effect-torch/core/Model/Compiled")

/**
 * @since 0.1.0
 * @category symbols
 */
export type CompiledTypeId = typeof CompiledTypeId

/**
 * A model with a compiled execution path (see {@link compile}): the
 * `forward` contract is unchanged — it is still the original graph
 * builder, composable and differentiable like any other model — and the
 * frozen forward program is exposed as {@link CompiledModel.execute},
 * the fast materialized path for evaluation, with the shape-keyed
 * program cache's diagnostics and release added as required members.
 *
 * @since 0.1.0
 * @category models
 */
export interface CompiledModel extends Model {
  readonly [CompiledTypeId]: CompiledTypeId
  /**
   * Runs the frozen forward program: parameters and input in,
   * materialized output out — one native call per invocation after the
   * first call per input signature pays the trace. A new input shape
   * traces a new program automatically. Use it for evaluation loops;
   * use `forward` wherever a graph is being built (training,
   * composition, differentiation).
   */
  readonly execute: (
    params: Params,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Concrete, ModelError | Tensor.TensorError, CurrentDevice>
  /**
   * Shape-cache diagnostics: programs cached, traces performed.
   */
  readonly stats: () => Tensor.CompileStats
  /**
   * Releases the cached forward programs.
   */
  readonly dispose: () => Effect.Effect<void>
}

/**
 * Returns `true` if the model was produced by {@link compile} and
 * narrows it to {@link CompiledModel}.
 *
 * @since 0.1.0
 * @category refinements
 */
export const isCompiled = (model: Model): model is CompiledModel => CompiledTypeId in model

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
  apply: (self: Tensor.Any) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, CurrentDevice>
): Effect.Effect<Model> =>
  Effect.succeed({
    names: [],
    init: Effect.succeed<Params>([]),
    forward: (_, input) => apply(input)
  })

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
  Effect.gen(function* () {
    yield* checkName("linear", name)
    yield* checkPositiveInt("linear", "inFeatures", inFeatures)
    yield* checkPositiveInt("linear", "outFeatures", outFeatures)
    const names = [`${name}.weight`, `${name}.bias`]
    return {
      names,
      init: Effect.gen(function* () {
        const drawn = yield* Tensor.randn([inFeatures, outFeatures])
        const weight = yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, 1 / Math.sqrt(inFeatures)))
        const bias = yield* Tensor.zeros([1, outFeatures])
        return [weight, bias] as const
      }),
      forward: (params, input) =>
        Effect.gen(function* () {
          yield* checkArity(name, names, params)
          const [weight, bias] = params
          return yield* Tensor.add(yield* Tensor.matmul(input, weight), bias)
        })
    }
  })

/**
 * A 1-D convolution layer over `[N, C_in, L]` inputs with
 * `names = ["<name>.weight", "<name>.bias"]`. The weight is
 * `[C_out, C_in/groups, K]` initialized to `randn * (1 / sqrt(fan_in))`
 * with `fan_in = (C_in/groups) * K`; the bias is `zeros([C_out])`, added
 * per channel. Stride, padding, dilation, and groups come from
 * `options`. Fails with a {@link ModelError} on an empty name,
 * non-positive channels/kernel/groups, or channels not divisible into
 * groups.
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
  Effect.gen(function* () {
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
    return {
      names,
      init: Effect.gen(function* () {
        const drawn = yield* Tensor.randn([outChannels, inChannels / groups, kernelSize])
        const weight = yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, 1 / Math.sqrt(fanIn)))
        const bias = yield* Tensor.zeros([outChannels])
        return [weight, bias] as const
      }),
      forward: (params, input) =>
        Effect.gen(function* () {
          yield* checkArity(name, names, params)
          const [weight, bias] = params
          const out = yield* Tensor.conv1d(input, weight, options)
          return yield* Tensor.add(out, yield* Tensor.reshape(bias, [1, outChannels, 1]))
        })
    }
  })

/**
 * A 2-D convolution layer over `[N, C_in, H, W]` inputs with
 * `names = ["<name>.weight", "<name>.bias"]`. The weight is
 * `[C_out, C_in/groups, KH, KW]` initialized to `randn * (1 / sqrt(fan_in))`
 * with `fan_in = (C_in/groups) * KH * KW`; the bias is `zeros([C_out])`,
 * added per channel. `kernelSize` is a square size or a `[KH, KW]` pair;
 * stride, padding, dilation, and groups come from `options`. Fails with a
 * {@link ModelError} on an empty name, non-positive channels/kernel/groups,
 * or channels not divisible into groups.
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
  Effect.gen(function* () {
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
    return {
      names,
      init: Effect.gen(function* () {
        const drawn = yield* Tensor.randn([outChannels, inChannels / groups, kh, kw])
        const weight = yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, 1 / Math.sqrt(fanIn)))
        const bias = yield* Tensor.zeros([outChannels])
        return [weight, bias] as const
      }),
      forward: (params, input) =>
        Effect.gen(function* () {
          yield* checkArity(name, names, params)
          const [weight, bias] = params
          const out = yield* Tensor.conv2d(input, weight, options)
          return yield* Tensor.add(out, yield* Tensor.reshape(bias, [1, outChannels, 1, 1]))
        })
    }
  })

/**
 * An embedding layer: looks up rows of a `[numEmbeddings, embeddingDim]`
 * weight by integer indexes of any shape, giving
 * `[...indexes.shape, embeddingDim]`. `names = ["<name>.weight"]`; the
 * weight is initialized to `randn` (unit normal, matching PyTorch).
 * Repeated indexes accumulate weight gradients. Fails with a
 * {@link ModelError} on an empty name, non-positive counts, or a
 * `paddingIndex` outside `[0, numEmbeddings)`.
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
  Effect.gen(function* () {
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
    return {
      names,
      init: Effect.gen(function* () {
        const weight = yield* Tensor.randn([numEmbeddings, embeddingDim])
        return [weight] as const
      }),
      forward: (params, input) =>
        Effect.gen(function* () {
          yield* checkArity(name, names, params)
          return yield* Tensor.embedding(input, {
            weight: params[0],
            ...(options.paddingIndex !== undefined ? { paddingIndex: options.paddingIndex } : {})
          })
        })
    }
  })

/**
 * A learned absolute position embedding (GPT-style `wpe`): looks up rows
 * `0..t-1` of a `[maxPositions, embeddingDim]` table, where `t` is the
 * input's last dimension — the input's values are ignored, only its
 * sequence length matters. `names = ["<name>.weight"]`, initialized
 * unit-normal. Fails with a {@link ModelError} on an empty name,
 * non-positive counts, or an input whose sequence length exceeds
 * `maxPositions`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const positionEmbedding = (
  name: string,
  maxPositions: number,
  embeddingDim: number
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function* () {
    yield* checkName("positionEmbedding", name)
    yield* checkPositiveInt("positionEmbedding", "maxPositions", maxPositions)
    yield* checkPositiveInt("positionEmbedding", "embeddingDim", embeddingDim)
    const names = [`${name}.weight`]
    return {
      names,
      init: Effect.gen(function* () {
        const weight = yield* Tensor.randn([maxPositions, embeddingDim])
        return [weight] as const
      }),
      forward: (params, input) =>
        Effect.gen(function* () {
          yield* checkArity(name, names, params)
          const t = input.shape.length === 0 ? 0 : input.shape[input.shape.length - 1]
          if (t > maxPositions) {
            return yield* new ModelError({
              op: "positionEmbedding",
              message: `${name}: sequence length ${t} exceeds maxPositions ${maxPositions}`
            })
          }
          return yield* Tensor.embedding(
            yield* Tensor.arange(t, undefined, { dtype: "i64" }),
            { weight: params[0] }
          )
        })
    }
  })

/**
 * A layer-normalization layer over the trailing `normalizedShape`
 * dimensions: `(x - mean) / sqrt(var + eps) * weight + bias` with the
 * biased variance and `eps` defaulting to `1e-5`.
 * `names = ["<name>.weight", "<name>.bias"]`, initialized to ones and
 * zeros of `normalizedShape` (a single feature count or a shape). Fails
 * with a {@link ModelError} on an empty name, an empty or non-positive
 * shape, or a non-positive `eps`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const layerNorm = (
  name: string,
  normalizedShape: number | ReadonlyArray<number>,
  options: { readonly eps?: number } = {}
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function* () {
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
    const dims = shape.map((_, i) => i - shape.length)
    return {
      names,
      init: Effect.gen(function* () {
        const weight = yield* Tensor.ones(shape)
        const bias = yield* Tensor.zeros(shape)
        return [weight, bias] as const
      }),
      forward: (params, input) =>
        Effect.gen(function* () {
          yield* checkArity(name, names, params)
          const [weight, bias] = params
          const mu = yield* Tensor.mean(input, { dims, keepdims: true })
          const centered = yield* Tensor.sub(input, mu)
          const variance = yield* Tensor.variance(input, { dims, keepdims: true, correction: 0 })
          const inv = yield* Tensor.rsqrt(yield* Tensor.add(variance, yield* Tensor.constantLike(variance, eps)))
          return yield* Tensor.add(yield* Tensor.mul(yield* Tensor.mul(centered, inv), weight), bias)
        })
    }
  })

/**
 * Options for {@link multiHeadAttention}.
 *
 * @since 0.1.0
 * @category constructors
 */
export interface MultiHeadAttentionOptions {
  /** Mask the attention scores causally (autoregressive transformers). */
  readonly causal?: boolean
}

/**
 * Multi-head scaled dot-product attention over `[..., T, embedDim]`
 * inputs (GPT-2 style): learned `wq`, `wk`, `wv` and `wo` projections,
 * the head dim split across `numHeads` heads, and
 * {@link Tensor.scaledDotProductAttention} per head. Names are
 * `["<name>.wq.weight", "<name>.wq.bias", "<name>.wk.weight", ...]` —
 * each projection follows the {@link linear} conventions (weight
 * `[embedDim, embedDim]` initialized to `randn * (1 / sqrt(embedDim))`,
 * bias `zeros([1, embedDim])`). Fails with a {@link ModelError} on an
 * empty name, non-positive counts, or `embedDim` not divisible by
 * `numHeads`.
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
  Effect.gen(function* () {
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
    const projections = ["wq", "wk", "wv", "wo"] as const
    const names = projections.flatMap((p) => [`${name}.${p}.weight`, `${name}.${p}.bias`])
    const causal = options.causal ?? false
    return {
      names,
      init: Effect.gen(function* () {
        const params: Array<Tensor.Any> = []
        for (const _ of projections) {
          const drawn = yield* Tensor.randn([embedDim, embedDim])
          params.push(
            yield* Tensor.mul(drawn, yield* Tensor.constantLike(drawn, 1 / Math.sqrt(embedDim))),
            yield* Tensor.zeros([1, embedDim])
          )
        }
        return params
      }),
      forward: (params, input) =>
        Effect.gen(function* () {
          yield* checkArity(name, names, params)
          const rank = input.shape.length
          const t = input.shape[rank - 2]
          const leading = input.shape.slice(0, -2)
          // [..., T, E] -> [..., T, H, Dh] -> [..., H, T, Dh]
          const splitHeads = (x: Tensor.Any) =>
            Effect.gen(function* () {
              const reshaped = yield* Tensor.reshape(x, [...leading, t, numHeads, headDim])
              const perm = Array.from({ length: rank + 1 }, (_, i) => i)
              perm[rank - 2] = rank - 1
              perm[rank - 1] = rank - 2
              return yield* Tensor.transpose(reshaped, perm)
            })
          // [..., H, T, Dh] -> [..., T, H, Dh] -> [..., T, E]
          const mergeHeads = (x: Tensor.Any) =>
            Effect.gen(function* () {
              const perm = Array.from({ length: rank + 1 }, (_, i) => i)
              perm[rank - 2] = rank - 1
              perm[rank - 1] = rank - 2
              const transposed = yield* Tensor.transpose(x, perm)
              return yield* Tensor.reshape(transposed, [...leading, t, embedDim])
            })
          const project = (x: Tensor.Any, weight: Tensor.Any, bias: Tensor.Any) =>
            Effect.gen(function* () {
              return yield* Tensor.add(yield* Tensor.matmul(x, weight), bias)
            })
          const projected: Array<Tensor.Any> = []
          for (let p = 0; p < 3; p++) {
            projected.push(yield* project(input, params[p * 2], params[p * 2 + 1]))
          }
          const attended = yield* Tensor.scaledDotProductAttention(
            yield* splitHeads(projected[0]),
            yield* splitHeads(projected[1]),
            yield* splitHeads(projected[2]),
            { causal }
          )
          return yield* project(yield* mergeHeads(attended), params[6], params[7])
        })
    }
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
 * Softmax over `dim` (the last dimension by default) as a parameterless
 * model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const softmax = (dim: number = -1): Effect.Effect<Model> =>
  parameterless((input) => Tensor.softmax(input, { dims: [dim] }))

/**
 * Log-softmax over `dim` (the last dimension by default) as a
 * parameterless model.
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
 * network) and `endDim` to the last dimension.
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
 * array, so one checkpoint serves both chains). The mask is drawn at
 * evaluation time, so the usual `randn` rule applies: evaluate the loss
 * and its gradients together in one walk. Fails with a {@link ModelError}
 * if `p` is outside `[0, 1)`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const dropout = (options: Tensor.DropoutOptions = {}): Effect.Effect<Model, ModelError> =>
  Effect.gen(function* () {
    const p = options.p ?? 0.5
    if (p < 0 || p >= 1) {
      return yield* new ModelError({ op: "dropout", message: `p must be in [0, 1), got ${p}` })
    }
    return {
      names: [],
      init: Effect.succeed<Params>([]),
      forward: (_, input) => Tensor.dropout(input, { p })
    }
  })

const pool = (
  op: string,
  apply: (
    self: Tensor.Any,
    options: Tensor.PoolOptions
  ) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, CurrentDevice>,
  options: Tensor.PoolOptions
): Effect.Effect<Model, ModelError> =>
  Effect.gen(function* () {
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
    return {
      names: [],
      init: Effect.succeed<Params>([]),
      forward: (_, input) => apply(input, options)
    }
  })

/**
 * 2-D max pooling as a parameterless model; `kernelSize` (a square size
 * or a `[KH, KW]` pair), `stride`, and `padding` come from `options`.
 * Fails with a {@link ModelError} on non-positive sizes or negative
 * padding.
 *
 * @since 0.1.0
 * @category constructors
 */
export const maxPool2d = (options: Tensor.PoolOptions): Effect.Effect<Model, ModelError> =>
  pool("maxPool2d", Tensor.maxPool2d, options)

/**
 * 2-D average pooling as a parameterless model; `kernelSize` (a square
 * size or a `[KH, KW]` pair), `stride`, and `padding` come from
 * `options`. Fails with a {@link ModelError} on non-positive sizes or
 * negative padding.
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
  Effect.succeed({
    names: model.names,
    init: model.init,
    forward: (params, input) => Effect.flatMap(model.forward(params, input), Gradient.checkpoint)
  })

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
  Effect.succeed({
    names: model.names,
    init: model.init,
    forward: (params, input) =>
      Effect.gen(function* () {
        const out = yield* model.forward(params, input)
        return yield* Tensor.add(input, out)
      })
  })

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
  f: (input: Tensor.Any) => Effect.Effect<Tensor.Any, Tensor.TensorError, CurrentDevice>
): Effect.Effect<Model> =>
  Effect.succeed({
    names: model.names,
    init: model.init,
    forward: (params, input) => Effect.flatMap(f(input), (mapped) => model.forward(params, mapped))
  })

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
  f: (...outputs: { [K in keyof M]: Tensor.Lazy }) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, CurrentDevice>
): Effect.Effect<Model, ModelError> => {
  if (models.length === 0) {
    return new ModelError({ op: "merge", message: "at least one model is required" })
  }
  const names = models.flatMap((model) => model.names)
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
  return Effect.succeed({
    names,
    init: Effect.gen(function* () {
      const params: Array<Tensor.Any> = []
      for (const model of models) {
        params.push(...(yield* model.init))
      }
      return params
    }),
    forward: (params, input) =>
      Effect.gen(function* () {
        yield* checkArity("merge", names, params)
        const outputs: Array<Tensor.Lazy> = []
        let offset = 0
        for (let i = 0; i < models.length; i++) {
          outputs.push(yield* models[i].forward(params.slice(offset, offset + arities[i]), input))
          offset += arities[i]
        }
        return yield* f(...(outputs as { [K in keyof M]: Tensor.Lazy }))
      })
  })
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
    Effect.gen(function* () {
      let acc: Tensor.Any = first
      for (const output of rest) {
        acc = yield* Tensor.add(acc, output)
      }
      return acc as Tensor.Lazy
    })
  )

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
  const names = models.flatMap((model) => model.names)
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
  return Effect.succeed({
    names,
    init: Effect.gen(function* () {
      const params: Array<Tensor.Any> = []
      for (const model of models) {
        params.push(...(yield* model.init))
      }
      return params
    }),
    forward: (params, input) =>
      Effect.gen(function* () {
        yield* checkArity("chain", names, params)
        let current: Tensor.Any = input
        let offset = 0
        for (let i = 0; i < models.length; i++) {
          current = yield* models[i].forward(params.slice(offset, offset + arities[i]), current)
          offset += arities[i]
        }
        return current as Tensor.Lazy
      })
  })
}

/**
 * Saves a model's parameters to a safetensors file, zipping `model.names`
 * with the parameter array into the record {@link Tensor.save} takes.
 * Fails with a {@link ModelError} if the parameter array's length does
 * not match the model's arity.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = (
  model: Model,
  params: Params,
  path: string
): Effect.Effect<void, ModelError | Tensor.TensorError> =>
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
 * — the same array `forward` and `Optimizer.step` expect. A missing key
 * fails with a {@link ModelError}; shape/dtype mismatches against the
 * architecture surface as graph-build errors on first use.
 *
 * @since 0.1.0
 * @category destructors
 */
export const load = (
  model: Model,
  path: string
): Effect.Effect<ReadonlyArray<Tensor.Concrete>, ModelError | Tensor.TensorError, CurrentDevice> =>
  Effect.gen(function* () {
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
 * Options for {@link compile}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface CompileOptions {
  /**
   * Shape-cache capacity in programs. The first forward with a new input
   * signature traces and freezes a program; later calls with the same
   * signature reuse it. Defaults to 32.
   */
  readonly cacheCapacity?: number
}

/**
 * Returns a model with a compiled execution path: same `names`, `init`,
 * and `forward` (the graph-builder contract is untouched — the compiled
 * model composes, differentiates, and trains exactly like the original),
 * plus {@link CompiledModel.execute}, which runs the forward as a frozen
 * native program — parameters and input in, materialized output out, in
 * one native call per invocation after the first call per input
 * signature pays the trace. Parameter shapes are fixed by the
 * architecture, so in practice the cache key varies only on the data
 * shape; a new input shape traces a new program automatically, up to
 * `cacheCapacity` programs with least-recently-used eviction.
 *
 * The compiled model is still a {@link Model} — a {@link CompiledModel},
 * to be precise (`isCompiled` narrows a `Model` to one). Use `execute`
 * for evaluation loops; use `forward` wherever a graph is being built.
 *
 * @since 0.1.0
 * @category compilation
 */
export const compile = (
  model: Model,
  options: CompileOptions = {}
): Effect.Effect<CompiledModel, never, CurrentDevice> =>
  Effect.map(
    Tensor.compile<ModelError | Tensor.TensorError, CurrentDevice>(
      (inputs) =>
        Effect.map(
          model.forward(inputs.slice(0, -1), inputs[inputs.length - 1]),
          (output) => [output]
        ),
      { ...(options.cacheCapacity !== undefined ? { cacheCapacity: options.cacheCapacity } : {}) }
    ),
    (fn): CompiledModel => ({
      [CompiledTypeId]: CompiledTypeId,
      names: model.names,
      init: model.init,
      forward: model.forward,
      execute: (params, input) =>
        Effect.gen(function* () {
          yield* checkArity("execute", model.names, params)
          const [output] = yield* fn.call([...params, input])
          return output
        }),
      stats: fn.stats,
      dispose: fn.dispose
    })
  )

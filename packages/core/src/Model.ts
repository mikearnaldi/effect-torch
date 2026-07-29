/**
 * Models as pure values pairing parameter construction with a
 * parameterised forward graph builder — the Flax/Haiku design, flattened:
 * parameters are always a plain tuple of tensors, configuration lives in
 * the factory's closure, and the forward graph is an ordinary lazy graph,
 * so {@link Gradient.grad} differentiates it and {@link Optimizer.step}
 * updates it with zero model-specific code. There is no mutable module
 * state and no backward mode.
 *
 * Everything that can fail returns an `Effect`: factories validate their
 * configuration (positive feature counts, unique parameter names) into a
 * {@link ModelError}, checkpoints report arity and missing-key problems in
 * the error channel, and {@link train} runs the whole training loop —
 * init, forward, loss, gradients, update, one graph walk per step — with
 * the tensor, gradient, and callback error channels in the union.
 *
 * The layer catalog covers the standard MLP / CNN / embedding stack:
 * parameterised layers ({@link linear}, {@link conv1d}, {@link conv2d},
 * {@link embedding}, {@link layerNorm}) and parameterless ones (the
 * activations, {@link softmax}, {@link logSoftmax}, {@link flatten},
 * {@link dropout}, {@link maxPool2d}, {@link avgPool2d}). {@link chain}
 * composes models into a model with the same interface, computing the
 * concatenated parameter tuple at the type level ({@link ParamsOf}).
 * `names` gives every parameter a stable, checkpoint-friendly identity
 * that maps directly onto {@link Tensor.save} / {@link Tensor.load} via
 * {@link save} and {@link load}.
 *
 * Stateful layers (batchnorm running stats) are deliberately absent: the
 * pure design keeps non-trainable state out of the parameter tuple until
 * the `stateRoots`/`rebuildState` contract generalizes to models. Note
 * that {@link dropout} is the functional form — it always applies; build
 * the evaluation chain without it (parameterless stages add nothing to
 * the tuple, so one checkpoint serves both chains).
 *
 * @since 0.1.0
 */
import { Data, Effect } from "effect"
import type { CurrentDevice } from "./Device.ts"
import type * as Gradient from "./Gradient.ts"
import * as Optimizer from "./Optimizer.ts"
import * as Tensor from "./Tensor.ts"

/**
 * A failure in model construction, checkpointing, or training:
 * invalid layer configuration, duplicate parameter names, parameter
 * tuple arity mismatches, missing checkpoint keys, or an invalid step
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
 * A model: stable parameter identities, an initializer that builds the
 * initial parameters as lazy graph values, and a forward function that
 * extends the graph — parameters and input in, lazy output out.
 *
 * `P` is constrained to a tuple of tensors — "expect a tuple, ask for a
 * tuple" — so the existing training path (`Gradient.grad`,
 * `Optimizer.step`) works on any model with zero adapter code.
 * Configuration (feature counts, activation choice) lives in the closure
 * of the factory, never in `P`.
 *
 * @since 0.1.0
 * @category models
 */
export interface Model<P extends ReadonlyArray<Tensor.GenericTensor>> {
  /**
   * Stable parameter identities, one per tensor in `P`, in the same
   * order. Also serves as the arity of `P` at runtime.
   */
  readonly names: ReadonlyArray<string>
  /**
   * Builds the initial parameters as lazy graph values. Materialization
   * happens in the first `Optimizer.step` walk (or an explicit
   * `Tensor.compute`), so initial `randn` draws are consistent with the
   * first loss within that walk.
   */
  readonly init: Effect.Effect<P, Tensor.TensorError, CurrentDevice>
  /**
   * Extends the graph: parameters and input in, lazy output out.
   * Single-input, single-output; differentiated as-is by
   * `Gradient.grad`. May require the current device (layers that draw
   * randomness or reshape on-device, like `dropout` and the pools).
   */
  readonly forward: (
    params: P,
    input: Tensor.GenericTensor
  ) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError, CurrentDevice>
}

/**
 * Any model, for constraints.
 *
 * @since 0.1.0
 * @category models
 */
export type Any = Model<any>

/**
 * Computes the concatenated parameter tuple of a list of models at the
 * type level: `ParamsOf<[Linear, Tanh, Linear]>` is
 * `readonly [w1, b1, w2, b2]`.
 *
 * @since 0.1.0
 * @category models
 */
export type ParamsOf<Ms extends ReadonlyArray<Any>> = Ms extends readonly [infer H, ...infer T]
  ? H extends Model<infer P>
    ? T extends ReadonlyArray<Any>
      ? readonly [...P, ...ParamsOf<T>]
    : readonly [...P]
  : readonly []
  : readonly []

/**
 * Extracts the parameter tuple of a model: `Params<typeof model>` is the
 * tuple `init` returns and `forward` takes. Useful for naming the type of
 * a chained model's parameters.
 *
 * @since 0.1.0
 * @category models
 */
export type Params<M extends Any> = M extends Model<infer P> ? P : never

const checkName = (op: string, name: string): Effect.Effect<void, ModelError> =>
  name.length === 0 ? new ModelError({ op, message: "name must not be empty" }) : Effect.void

const checkPositiveInt = (op: string, field: string, value: number): Effect.Effect<void, ModelError> =>
  Number.isInteger(value) && value >= 1
    ? Effect.void
    : new ModelError({ op, message: `${field} must be a positive integer, got ${value}` })

const parameterless = (
  apply: (self: Tensor.GenericTensor) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError, CurrentDevice>
): Effect.Effect<Model<readonly []>> =>
  Effect.succeed({
    names: [],
    init: Effect.succeed<readonly []>([]),
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
): Effect.Effect<
  Model<readonly [weight: Tensor.GenericTensor, bias: Tensor.GenericTensor]>,
  ModelError
> =>
  Effect.gen(function* () {
    yield* checkName("linear", name)
    yield* checkPositiveInt("linear", "inFeatures", inFeatures)
    yield* checkPositiveInt("linear", "outFeatures", outFeatures)
    return {
      names: [`${name}.weight`, `${name}.bias`],
      init: Effect.gen(function* () {
        const weight = yield* Tensor.mul(
          yield* Tensor.randn([inFeatures, outFeatures]),
          1 / Math.sqrt(inFeatures)
        )
        const bias = yield* Tensor.zeros([1, outFeatures])
        return [weight, bias] as const
      }),
      forward: ([weight, bias], input) =>
        Effect.gen(function* () {
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
): Effect.Effect<
  Model<readonly [weight: Tensor.GenericTensor, bias: Tensor.GenericTensor]>,
  ModelError
> =>
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
    const fanIn = (inChannels / groups) * kernelSize
    return {
      names: [`${name}.weight`, `${name}.bias`],
      init: Effect.gen(function* () {
        const weight = yield* Tensor.mul(
          yield* Tensor.randn([outChannels, inChannels / groups, kernelSize]),
          1 / Math.sqrt(fanIn)
        )
        const bias = yield* Tensor.zeros([outChannels])
        return [weight, bias] as const
      }),
      forward: ([weight, bias], input) =>
        Effect.gen(function* () {
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
): Effect.Effect<
  Model<readonly [weight: Tensor.GenericTensor, bias: Tensor.GenericTensor]>,
  ModelError
> =>
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
    const fanIn = (inChannels / groups) * kh * kw
    return {
      names: [`${name}.weight`, `${name}.bias`],
      init: Effect.gen(function* () {
        const weight = yield* Tensor.mul(
          yield* Tensor.randn([outChannels, inChannels / groups, kh, kw]),
          1 / Math.sqrt(fanIn)
        )
        const bias = yield* Tensor.zeros([outChannels])
        return [weight, bias] as const
      }),
      forward: ([weight, bias], input) =>
        Effect.gen(function* () {
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
): Effect.Effect<Model<readonly [weight: Tensor.GenericTensor]>, ModelError> =>
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
    return {
      names: [`${name}.weight`],
      init: Effect.gen(function* () {
        const weight = yield* Tensor.randn([numEmbeddings, embeddingDim])
        return [weight] as const
      }),
      forward: ([weight], input) =>
        Tensor.embedding(input, {
          weight,
          ...(options.paddingIndex !== undefined ? { paddingIndex: options.paddingIndex } : {})
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
): Effect.Effect<
  Model<readonly [weight: Tensor.GenericTensor, bias: Tensor.GenericTensor]>,
  ModelError
> =>
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
    const dims = shape.map((_, i) => i - shape.length)
    return {
      names: [`${name}.weight`, `${name}.bias`],
      init: Effect.gen(function* () {
        const weight = yield* Tensor.ones(shape)
        const bias = yield* Tensor.zeros(shape)
        return [weight, bias] as const
      }),
      forward: ([weight, bias], input) =>
        Effect.gen(function* () {
          const mu = yield* Tensor.mean(input, { dims, keepdims: true })
          const centered = yield* Tensor.sub(input, mu)
          const variance = yield* Tensor.variance(input, { dims, keepdims: true, correction: 0 })
          const inv = yield* Tensor.rsqrt(yield* Tensor.add(variance, eps))
          return yield* Tensor.add(yield* Tensor.mul(yield* Tensor.mul(centered, inv), weight), bias)
        })
    }
  })

/**
 * The hyperbolic tangent activation as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const tanh: Effect.Effect<Model<readonly []>> = parameterless(Tensor.tanh)

/**
 * The sigmoid activation as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const sigmoid: Effect.Effect<Model<readonly []>> = parameterless(Tensor.sigmoid)

/**
 * The rectified linear unit activation as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const relu: Effect.Effect<Model<readonly []>> = parameterless(Tensor.relu)

/**
 * The SiLU / swish activation `x * sigmoid(x)` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const silu: Effect.Effect<Model<readonly []>> = parameterless(Tensor.silu)

/**
 * The mish activation `x * tanh(softplus(x))` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const mish: Effect.Effect<Model<readonly []>> = parameterless(Tensor.mish)

/**
 * The softplus activation `log(1 + exp(x))` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const softplus: Effect.Effect<Model<readonly []>> = parameterless(Tensor.softplus)

/**
 * The GELU activation as a parameterless model; `approximate` (`"none"`,
 * the erf form, or `"tanh"`) comes from `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const gelu = (options: Tensor.GeluOptions = {}): Effect.Effect<Model<readonly []>> =>
  parameterless((input) => Tensor.gelu(input, options))

/**
 * The ELU activation as a parameterless model; `alpha` comes from
 * `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const elu = (options: Tensor.EluOptions = {}): Effect.Effect<Model<readonly []>> =>
  parameterless((input) => Tensor.elu(input, options))

/**
 * The leaky-ReLU activation as a parameterless model; `negativeSlope`
 * comes from `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const leakyRelu = (
  options: Tensor.LeakyReluOptions = {}
): Effect.Effect<Model<readonly []>> => parameterless((input) => Tensor.leakyRelu(input, options))

/**
 * Softmax over `dim` (the last dimension by default) as a parameterless
 * model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const softmax = (dim: number = -1): Effect.Effect<Model<readonly []>> =>
  parameterless((input) => Tensor.softmax(input, { dims: [dim] }))

/**
 * Log-softmax over `dim` (the last dimension by default) as a
 * parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const logSoftmax = (dim: number = -1): Effect.Effect<Model<readonly []>> =>
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
): Effect.Effect<Model<readonly []>> =>
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
 * tuple, so one checkpoint serves both chains). The mask is drawn at
 * evaluation time, so the usual `randn` rule applies: evaluate the loss
 * and its gradients together in one walk. Fails with a {@link ModelError}
 * if `p` is outside `[0, 1)`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const dropout = (
  options: Tensor.DropoutOptions = {}
): Effect.Effect<Model<readonly []>, ModelError> =>
  Effect.gen(function* () {
    const p = options.p ?? 0.5
    if (p < 0 || p >= 1) {
      return yield* new ModelError({ op: "dropout", message: `p must be in [0, 1), got ${p}` })
    }
    return {
      names: [],
      init: Effect.succeed<readonly []>([]),
      forward: (_, input) => Tensor.dropout(input, { p })
    }
  })

const pool = (
  op: string,
  apply: (
    self: Tensor.GenericTensor,
    options: Tensor.PoolOptions
  ) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError, CurrentDevice>,
  options: Tensor.PoolOptions
): Effect.Effect<Model<readonly []>, ModelError> =>
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
      init: Effect.succeed<readonly []>([]),
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
export const maxPool2d = (options: Tensor.PoolOptions): Effect.Effect<Model<readonly []>, ModelError> =>
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
export const avgPool2d = (options: Tensor.PoolOptions): Effect.Effect<Model<readonly []>, ModelError> =>
  pool("avgPool2d", Tensor.avgPool2d, options)

/**
 * Composes models into a single model that threads its input through each
 * child in order, slicing each child's share of the concatenated
 * parameter tuple by its arity (`names.length`). `names` is the
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
export const chain = <Ms extends ReadonlyArray<Any>>(
  ...models: Ms
): Effect.Effect<Model<ParamsOf<Ms>>, ModelError> => {
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
      const params: Array<Tensor.GenericTensor> = []
      for (const model of models) {
        params.push(...(yield* model.init))
      }
      return params as unknown as ParamsOf<Ms>
    }),
    forward: (params, input) =>
      Effect.gen(function* () {
        let current: Tensor.GenericTensor = input
        let offset = 0
        for (let i = 0; i < models.length; i++) {
          current = yield* models[i].forward(
            params.slice(offset, offset + arities[i]) as ReadonlyArray<Tensor.GenericTensor>,
            current
          )
          offset += arities[i]
        }
        return current as Tensor.LazyTensor
      })
  })
}

/**
 * Saves a model's parameters to a safetensors file, zipping `model.names`
 * with the parameter tuple into the record {@link Tensor.save} takes.
 * Fails with a {@link ModelError} if the parameter tuple's length does
 * not match the model's arity.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = (
  model: Any,
  params: ReadonlyArray<Tensor.GenericTensor>,
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
 * — the same tuple `forward` and `Optimizer.step` expect. A missing key
 * fails with a {@link ModelError}; shape/dtype mismatches against the
 * architecture surface as graph-build errors on first use.
 *
 * @since 0.1.0
 * @category destructors
 */
export const load = <P extends ReadonlyArray<Tensor.GenericTensor>>(
  model: Model<P>,
  path: string
): Effect.Effect<Optimizer.Materialized<P>, ModelError | Tensor.TensorError, CurrentDevice> =>
  Effect.gen(function* () {
    const record = yield* Tensor.load(path)
    const params: Array<Tensor.Tensor> = []
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
    return params as Optimizer.Materialized<P>
  })

/**
 * The training data for {@link train}: a full-batch input and its target.
 * (A `Dataset` module with batching is future work.)
 *
 * @since 0.1.0
 * @category training
 */
export interface TrainData {
  readonly input: Tensor.GenericTensor
  readonly target: Tensor.GenericTensor
}

/**
 * Per-step progress reported to {@link TrainConfig.onStep} and
 * {@link TrainConfig.stop}: the 1-based step number and the step's loss
 * value.
 *
 * @since 0.1.0
 * @category training
 */
export interface TrainStep {
  readonly step: number
  readonly loss: number
}

/**
 * Configuration for {@link train}. `loss` is any loss function in the
 * shape of {@link Loss.mse} — `(prediction, target) => Effect<Lazy>` —
 * so the `Loss` module's exports slot in directly. `params` overrides the
 * initial parameters (continued training, fine-tuning from a checkpoint);
 * when omitted, `model.init` runs. `onStep` runs after every step with
 * the step's loss value — throttle inside the callback.
 *
 * `stop` decides when training ends; it is checked after every step (at
 * least one step always runs), so any policy is a plain function:
 * `({ step }) => step >= 3000` stops on a step count,
 * `({ loss }) => loss < 0.01` stops on a loss target, and the two compose
 * with `||` — or close over any other state you track.
 *
 * @since 0.1.0
 * @category training
 */
export interface TrainConfig<P extends ReadonlyArray<Tensor.GenericTensor>, S, E, R> {
  readonly optimizer: Optimizer.Optimizer<S>
  readonly loss: (
    prediction: Tensor.GenericTensor,
    target: Tensor.GenericTensor
  ) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError>
  readonly data: TrainData
  readonly stop: (info: TrainStep) => boolean
  readonly params?: P
  readonly onStep?: (info: TrainStep) => Effect.Effect<void, E, R>
}

/**
 * The result of {@link train}: the trained parameters (materialized
 * leaves, ready for `forward`, `save`, or more training), the final
 * optimizer state, and the final step's loss.
 *
 * @since 0.1.0
 * @category training
 */
export interface Trained<P extends ReadonlyArray<Tensor.GenericTensor>, S> {
  readonly params: Optimizer.Materialized<P>
  readonly state: S
  readonly loss: number
}

/**
 * Runs the training loop: initialize (or take `config.params`), then
 * repeatedly build `loss(forward(params, input), target)`, differentiate
 * it, extend the graph with the optimizer update, and evaluate loss,
 * parameters, and state in a single walk — one forward pass, one backward
 * pass, one async boundary per step, with graph depth staying O(model
 * depth). After each step `onStep` runs and `stop` decides whether the
 * loop ends (at least one step always runs).
 *
 * @since 0.1.0
 * @category training
 */
export const train = <P extends ReadonlyArray<Tensor.GenericTensor>, S, E = never, R = never>(
  model: Model<P>,
  config: TrainConfig<P, S, E, R>
): Effect.Effect<
  Trained<P, S>,
  Tensor.TensorError | Gradient.GradError | E,
  CurrentDevice | R
> =>
  Effect.gen(function* () {
    let params: P = config.params !== undefined ? config.params : yield* model.init
    let state = yield* config.optimizer.init(params)
    let step = 0
    let loss = Number.NaN
    while (true) {
      step++
      const prediction = yield* model.forward(params, config.data.input)
      const lossTensor = yield* config.loss(prediction, config.data.target)
      const result = yield* Optimizer.step(config.optimizer, lossTensor, params, state)
      loss = (yield* Tensor.toNumberArray(result.loss))[0]
      params = result.params as P
      state = result.state
      if (config.onStep !== undefined) {
        yield* config.onStep({ step, loss })
      }
      if (config.stop({ step, loss })) {
        break
      }
    }
    return { params: params as unknown as Optimizer.Materialized<P>, state, loss }
  })

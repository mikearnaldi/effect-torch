/**
 * Pure model graphs, parameters, composition, and ordinary compiled execution.
 *
 * A {@link Definition} separates architecture from values. Its `parameterSpecs`
 * catalog defines a stable flat order and how to initialize fresh values, while
 * `forward` extends the current lazy tensor graph from a parameter array and one
 * input. Configuration is captured by constructors rather than stored in a
 * mutable module tree. The resulting graph can be composed, differentiated by
 * {@link Gradient.grad}, and updated by an optimizer without model-specific
 * adapters. Models have no train/eval mode or non-parameter state; notably,
 * {@link dropout} always applies, so evaluation should use a chain without it.
 *
 * {@link compile} pairs a definition with one caller-owned parameter generation
 * and creates a {@link Program} for repeated stateless evaluation. `forward`
 * remains the path for composition, training, and differentiation.
 * Autoregressive compilation and generation live in `AutoRegressive`; their
 * artifacts have independent caches and state.
 *
 * Diffusion compilation, immutable prefixes, and block generation live in
 * `Diffusion`. {@link executeLayers} supports ordinary layer-wise reference
 * execution and diagnostics.
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
import { Data, Effect, Exit, Predicate } from "effect"
import { checkpoint as gradientCheckpoint } from "./Gradient.ts"
import type * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * A failure in model construction, parameter preparation, arity, or serialization, such as
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
   * catalog is descriptive: `forward` does not universally compare supplied
   * tensors against it.
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
 * A model's parameter values in {@link Definition.parameterSpecs} order. The array
 * length is the model arity; parameterless models use `[]`. Values may be lazy
 * graph nodes or materialized tensors unless a narrower API says otherwise.
 *
 * @since 0.1.0
 * @category models
 */
export type Parameters = ReadonlyArray<Tensor.Any>

type FunctionParameters<F> = F extends (...args: infer A) => infer _Result ? A : never

/**
 * Materializes each distinct dense parameter once, deduplicating by tensor
 * identity, and borrows concrete packed parameters in their original order.
 * The callback may compile several entry points against this one generation.
 *
 * The callback must capture parameters in native programs before returning.
 * Its result must not depend on these temporary handles remaining live. On
 * every exit, only newly materialized handles are cleared. Source parameters
 * remain caller-owned. Acquisition inherits the caller's interruptibility;
 * native acquisition owns partial and late results. A packed parameter that is
 * not concrete fails with a {@link ModelError} whose `op` is `"withParameters"`.
 *
 * @since 0.1.0
 * @category compilation
 */
export const withParameters = <A, E, R>(
  sourceParams: ReadonlyArray<Tensor.Any>,
  use: (params: ReadonlyArray<Tensor.Concrete>) => Effect.Effect<A, E, R>
): Effect.Effect<A, E | ModelError | Tensor.TensorError, R | Runtime.Runtime> =>
  Effect.suspend(() => {
    let materialized: ReadonlyArray<Tensor.Concrete> = []
    const distinct = [...new Set(sourceParams)]

    const acquire = Tensor.compute(distinct.filter((parameter) => parameter.storage === undefined)).pipe(
      Effect.onExit((exit) => {
        if (Exit.isSuccess(exit)) materialized = exit.value

        return Effect.void
      })
    )

    return Effect.onExit(
      Effect.gen(function*() {
        yield* acquire
        let denseIndex = 0
        const prepared = new Map<Tensor.Any, Tensor.Concrete>()

        for (const parameter of distinct) {
          if (parameter.storage === undefined) {
            prepared.set(parameter, materialized[denseIndex++]!)
          } else if (Tensor.isTensor(parameter)) {
            prepared.set(parameter, parameter)
          } else {
            return yield* new ModelError({
              op: "withParameters",
              message: "packed inference parameters must be concrete tensors"
            })
          }
        }

        return yield* use(sourceParams.map((parameter) => prepared.get(parameter)!))
      }),
      () => Tensor.clearAll(materialized)
    )
  })

/**
 * The stable exposure name of the residual activation after zero-based model
 * layer `layer`. This defines the shared name used by models
 * publishing hidden states via `Tensor.expose` and speculative proposers
 * requesting them via `Speculation.HiddenTap`. A model publishes any
 * number of exposures once; any number of proposers may subscribe to any
 * subset of them.
 *
 * @since 0.1.0
 * @category models
 */
export const hiddenExposure = (layer: number): string => `layers.${layer}.hidden`

/**
 * A pure model graph. Parameters use the exact order declared by
 * `parameterSpecs`. A definition has no compiled state and owns no tensors.
 *
 * @since 0.1.0
 * @category models
 */
export interface Definition {
  /** Logical parameter specifications in flat parameter-array order. */
  readonly parameterSpecs: ReadonlyArray<ParameterSpec>
  /** Extends a lazy graph from borrowed parameters and one borrowed input. */
  readonly forward: (
    parameters: Parameters,
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Lazy, ModelError | Tensor.TensorError, Runtime.Runtime>
}

/**
 * Input accepted by {@link define}.
 *
 * @since 0.1.0
 * @category models
 */
export interface Options {
  readonly parameterSpecs: ReadonlyArray<ParameterSpec>
  readonly forward: Definition["forward"]
}

/**
 * A definition paired with caller-owned parameter values.
 *
 * @since 0.1.0
 * @category models
 */
export interface Loaded<D = Definition> {
  readonly definition: D
  readonly parameters: Parameters
}

/**
 * Repeated ordinary execution for one definition and parameter generation.
 * Inputs are borrowed. Each returned tensor belongs to the caller.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface Program {
  readonly run: (
    input: Tensor.Any
  ) => Effect.Effect<Tensor.Concrete, ModelError | Tensor.TensorError, Runtime.Runtime>
  readonly stats: Effect.Effect<Tensor.CompileStats>
  readonly clear: Effect.Effect<void>
}

/**
 * Runs a stack one layer at a time. Each builder returns the next hidden tensor
 * followed by any tensors to retain for that layer, such as attention K/V.
 * Builders and observers borrow their inputs for the duration of their effect.
 *
 * The final output and retained tensors belong to the caller. Intermediate
 * hidden tensors are released as execution advances. Failure or interruption
 * releases all outputs produced by this call. The original input is borrowed.
 *
 * @since 0.1.0
 * @category execution
 */
export const executeLayers = <E, R, OE = never, OR = never>(
  input: Tensor.Any,
  count: number,
  build: (
    layer: number,
    hidden: Tensor.Concrete
  ) => Effect.Effect<readonly [Tensor.Any, ...Array<Tensor.Any>], E, R>,
  options: Runtime.ExecutableCompileOptions & {
    readonly observeLayer?: ((layer: number, hidden: Tensor.Concrete) => Effect.Effect<void, OE, OR>) | undefined
  } = {}
): Effect.Effect<
  {
    readonly output: Tensor.Concrete
    readonly retained: ReadonlyArray<ReadonlyArray<Tensor.Concrete>>
  },
  E | OE | ModelError | Tensor.TensorError,
  R | OR | Runtime.Runtime
> =>
  Effect.suspend(() => {
    const owned = new Set<Tensor.Concrete>()
    const retained: Array<ReadonlyArray<Tensor.Concrete>> = []

    const compute = (roots: ReadonlyArray<Tensor.Any>) =>
      Tensor.compute(roots, { optimize: options.optimize, constantWeights: options.constantWeights }).pipe(
        Effect.onExit((exit) => {
          if (Exit.isSuccess(exit)) {
            for (const tensor of exit.value) {
              owned.add(tensor)
            }
          }

          return Effect.void
        })
      )

    return Effect.onExit(
      Effect.gen(function*() {
        if (!Number.isSafeInteger(count) || count < 0) {
          return yield* new ModelError({ op: "executeLayers", message: "layer count must be a non-negative integer" })
        }

        let hidden = (yield* compute([input]))[0]

        for (let layer = 0; layer < count; layer++) {
          const roots = yield* build(layer, hidden)
          const outputs: ReadonlyArray<Tensor.Concrete> = yield* compute(roots)
          const previous = hidden
          hidden = outputs[0]
          retained.push(outputs.slice(1))

          yield* Tensor.clear(previous)
          owned.delete(previous)

          if (options.observeLayer !== undefined) {
            yield* options.observeLayer(layer, hidden)
          }
        }

        return { output: hidden, retained }
      }),
      (exit) => Exit.isFailure(exit) ? Tensor.clearAll(owned) : Effect.void
    )
  })

const make = (options: Options): Definition => ({
  parameterSpecs: options.parameterSpecs,
  forward: options.forward
})

/**
 * Validates a custom parameter catalog and constructs a model with the standard
 * ordinary compiled-execution path. This does not validate initializer output,
 * forward behavior, tensor shape/dtype compatibility, or inference support.
 *
 * @since 0.1.0
 * @category constructors
 */
export const define = (options: Options): Effect.Effect<Definition, ModelError> =>
  Effect.gen(function*() {
    const seen = new Set<string>()

    for (const parameter of options.parameterSpecs) {
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
      parameterSpecs: options.parameterSpecs,
      forward: options.forward
    })
  })

/**
 * Creates an ordinary compiled program for one definition and parameter
 * generation. The program borrows its parameters on every call. Clear the
 * returned output when it is no longer needed.
 *
 * @since 0.1.0
 * @category compilation
 */
export const compile = (
  definition: Definition,
  parameters: Parameters,
  options: Tensor.CompileOptions = {}
): Effect.Effect<Program, ModelError> =>
  Effect.gen(function*() {
    yield* checkArity("compile", definition.parameterSpecs.map((parameter) => parameter.name), parameters)
    const fn = yield* Tensor.compile<ModelError | Tensor.TensorError, Runtime.Runtime>(
      (inputs) => Effect.map(definition.forward(inputs.slice(0, -1), inputs[inputs.length - 1]), (output) => [output]),
      options
    )

    return {
      run: (input) => Effect.map(fn.call([...parameters, input]), ([output]) => output),
      get stats() {
        return fn.stats
      },
      get clear() {
        return fn.clear
      }
    }
  })

/**
 * Creates one fresh lazy parameter generation from the model's declared
 * initializers, in {@link Definition.parameterSpecs} order. Normal draws remain lazy;
 * materialize the returned roots together before retaining them so each draw is
 * sampled once.
 *
 * @since 0.1.0
 * @category constructors
 */
export const initialize = (model: Definition): Effect.Effect<Parameters, Tensor.TensorError, Runtime.Runtime> =>
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
  params: Parameters
): Effect.Effect<void, ModelError> =>
  params.length === names.length
    ? Effect.void
    : new ModelError({
      op: "forward",
      message: `${who}: expected ${names.length} parameters [${names.join(", ")}], got ${params.length}`
    })

const parameterless = (
  apply: (self: Tensor.Any) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>
): Effect.Effect<Definition> =>
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
): Effect.Effect<Definition, ModelError> =>
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
): Effect.Effect<Definition, ModelError> =>
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
): Effect.Effect<Definition, ModelError> =>
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
): Effect.Effect<Definition, ModelError> =>
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
): Effect.Effect<Definition, ModelError> =>
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
): Effect.Effect<Definition, ModelError> =>
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
): Effect.Effect<Definition, ModelError> =>
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
): Effect.Effect<Definition, ModelError> =>
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
export const tanh: Effect.Effect<Definition> = parameterless(Tensor.tanh)

/**
 * The sigmoid activation as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const sigmoid: Effect.Effect<Definition> = parameterless(Tensor.sigmoid)

/**
 * The rectified linear unit activation as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const relu: Effect.Effect<Definition> = parameterless(Tensor.relu)

/**
 * The SiLU / swish activation `x * sigmoid(x)` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const silu: Effect.Effect<Definition> = parameterless(Tensor.silu)

/**
 * The mish activation `x * tanh(softplus(x))` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const mish: Effect.Effect<Definition> = parameterless(Tensor.mish)

/**
 * The softplus activation `log(1 + exp(x))` as a parameterless model.
 *
 * @since 0.1.0
 * @category constructors
 */
export const softplus: Effect.Effect<Definition> = parameterless(Tensor.softplus)

/**
 * The GELU activation as a parameterless model; `approximate` (`"none"`,
 * the erf form, or `"tanh"`) comes from `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const gelu = (options: Tensor.GeluOptions = {}): Effect.Effect<Definition> =>
  parameterless((input) => Tensor.gelu(input, options))

/**
 * The ELU activation as a parameterless model; `alpha` comes from
 * `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const elu = (options: Tensor.EluOptions = {}): Effect.Effect<Definition> =>
  parameterless((input) => Tensor.elu(input, options))

/**
 * The leaky-ReLU activation as a parameterless model; `negativeSlope`
 * comes from `options`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const leakyRelu = (options: Tensor.LeakyReluOptions = {}): Effect.Effect<Definition> =>
  parameterless((input) => Tensor.leakyRelu(input, options))

/**
 * Softmax over integer axis `dim` (the last dimension by default) as a
 * parameterless model. The axis must be in range when `forward` is built.
 *
 * @since 0.1.0
 * @category constructors
 */
export const softmax = (dim: number = -1): Effect.Effect<Definition> =>
  parameterless((input) => Tensor.softmax(input, { dims: [dim] }))

/**
 * Log-softmax over integer axis `dim` (the last dimension by default) as a
 * parameterless model. The axis must be in range when `forward` is built.
 *
 * @since 0.1.0
 * @category constructors
 */
export const logSoftmax = (dim: number = -1): Effect.Effect<Definition> =>
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
  options: {
    readonly startDim?: number
    readonly endDim?: number
  } = {}
): Effect.Effect<Definition> =>
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
export const dropout = (options: Tensor.DropoutOptions = {}): Effect.Effect<Definition, ModelError> =>
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
): Effect.Effect<Definition, ModelError> =>
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
export const maxPool2d = (options: Tensor.PoolOptions): Effect.Effect<Definition, ModelError> =>
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
export const avgPool2d = (options: Tensor.PoolOptions): Effect.Effect<Definition, ModelError> =>
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
 * Definition.chain(
 *   yield* Definition.checkpoint(yield* block1),
 *   yield* Definition.checkpoint(yield* block2),
 *   head
 * )
 * ```
 *
 * This recomputation works on every target.
 *
 * @since 0.1.0
 * @category combinators
 */
export const checkpoint = (model: Definition): Effect.Effect<Definition> =>
  Effect.succeed(make({
    parameterSpecs: model.parameterSpecs,
    forward: (params, input) => Effect.flatMap(model.forward(params, input), gradientCheckpoint)
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
export const residual = (model: Definition): Effect.Effect<Definition> =>
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
  model: Definition,
  f: (input: Tensor.Any) => Effect.Effect<Tensor.Any, Tensor.TensorError, Runtime.Runtime>
): Effect.Effect<Definition> =>
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
export const merge = <const M extends ReadonlyArray<Definition>>(
  models: M,
  f: (...outputs: { -readonly [K in keyof M]: Tensor.Lazy }) => Effect.Effect<
    Tensor.Lazy,
    Tensor.TensorError,
    Runtime.Runtime
  >
): Effect.Effect<Definition, ModelError> => {
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
        return yield* f(...(outputs as FunctionParameters<typeof f>))
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
export const add = (...models: ReadonlyArray<Definition>): Effect.Effect<Definition, ModelError> =>
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
export const chain = (...models: ReadonlyArray<Definition>): Effect.Effect<Definition, ModelError> => {
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

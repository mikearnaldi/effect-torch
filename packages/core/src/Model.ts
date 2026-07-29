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
 * Primitives ({@link linear}, {@link tanh}, {@link sigmoid},
 * {@link relu}) are models; {@link chain} composes models into a model
 * with the same interface, computing the concatenated parameter tuple at
 * the type level ({@link ParamsOf}). `names` gives every parameter a
 * stable, checkpoint-friendly identity that maps directly onto
 * {@link Tensor.save} / {@link Tensor.load} via {@link save} and
 * {@link load}.
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
   * `Tensor.evaluate`), so initial `randn` draws are consistent with the
   * first loss within that walk.
   */
  readonly init: Effect.Effect<P, Tensor.TensorError, CurrentDevice>
  /**
   * Extends the graph: parameters and input in, lazy output out.
   * Single-input, single-output; differentiated as-is by
   * `Gradient.grad`.
   */
  readonly forward: (
    params: P,
    input: Tensor.GenericTensor
  ) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError>
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
> => {
  if (name.length === 0) {
    return new ModelError({ op: "linear", message: "name must not be empty" })
  }
  if (!Number.isInteger(inFeatures) || inFeatures < 1) {
    return new ModelError({
      op: "linear",
      message: `inFeatures must be a positive integer, got ${inFeatures}`
    })
  }
  if (!Number.isInteger(outFeatures) || outFeatures < 1) {
    return new ModelError({
      op: "linear",
      message: `outFeatures must be a positive integer, got ${outFeatures}`
    })
  }
  return Effect.succeed({
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
  })
}

const parameterless = (
  apply: (self: Tensor.GenericTensor) => Effect.Effect<Tensor.LazyTensor, Tensor.TensorError>
): Effect.Effect<Model<readonly []>> =>
  Effect.succeed({
    names: [],
    init: Effect.succeed<readonly []>([]),
    forward: (_, input) => apply(input)
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

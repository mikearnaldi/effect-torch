/**
 * Trainer checkpoints: a whole training position — parameters, optimizer
 * state, global step, and optionally the data sampler's state — as a
 * single safetensors file.
 *
 * Serialization never inspects the optimizer state `S` itself: the
 * `stateRoots`/`rebuildState` contract (plus an `init` template) makes
 * any state a canonical list of tensors, so every entry rides the same
 * file — `param:<name>`, `state:<i>`, the step as a 0-d `meta:step`, and
 * the sampler as `sampler:*`. Roots the current device cannot represent
 * (f64 on Metal) load onto the CPU, restored with their recorded dtype —
 * resuming is bit-exact.
 *
 * @since 0.1.0
 */
import { Data, Effect } from "effect"
import type * as Device from "./Device.ts"
import * as Model from "./Model.ts"
import type * as Sampler from "./Sampler.ts"
import * as Tensor from "./Tensor.ts"
import type * as Trainer from "./Trainer.ts"

/**
 * @since 0.1.0
 * @category errors
 */
export class CheckpointError extends Data.TaggedError("CheckpointError")<{
  readonly op: string
  readonly message: string
}> {}

const PARAM_PREFIX = "param:"
const STATE_PREFIX = "state:"
const STEP_KEY = "meta:step"
const SAMPLER_ORDER_KEY = "sampler:order"
const SAMPLER_CURSOR_KEY = "sampler:cursor"
const SAMPLER_EPOCH_KEY = "sampler:epoch"

/**
 * A restored training position: the parameters and the {@link Trainer.Resume}
 * to pass back to `trainer.train(params, resume)`.
 *
 * @since 0.1.0
 * @category models
 */
export interface Checkpoint<S> {
  readonly params: Model.Params
  readonly resume: Trainer.Resume<S>
}

/**
 * A {@link Checkpoint} with the data sampler's state, for
 * {@link Sampler.restore}.
 *
 * @since 0.1.0
 * @category models
 */
export interface CheckpointWithSampler<S> extends Checkpoint<S> {
  readonly sampler: Sampler.SamplerState
}

/**
 * Saves parameters, optimizer state roots, and the global step to a
 * single safetensors file at `path`.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  trained: Trainer.Trained<S>
): Effect.Effect<void, Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function* () {
    const entries = yield* trainerEntries(trainer, trained)
    yield* Tensor.save(path, entries)
  })

/**
 * {@link save} plus the data sampler's state, so resuming continues the
 * epoch exactly where it stopped.
 *
 * @since 0.1.0
 * @category destructors
 */
export const saveWithSampler = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  trained: Trainer.Trained<S>,
  sampler: Sampler.Sampler
): Effect.Effect<void, Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function* () {
    const entries = yield* trainerEntries(trainer, trained)
    const state = sampler.state()
    entries[SAMPLER_ORDER_KEY] = yield* Tensor.fromTypedArray(state.order, [state.order.length])
    entries[SAMPLER_CURSOR_KEY] = yield* Tensor.full([], state.cursor, { dtype: "u32" })
    entries[SAMPLER_EPOCH_KEY] = yield* Tensor.full([], state.epoch, { dtype: "u32" })
    yield* Tensor.save(path, entries)
  })

/**
 * Loads a checkpoint saved by {@link save}. The optimizer state is
 * rebuilt generically: an `init` template supplies the structure, the
 * saved roots supply the values, `rebuildState` injects them.
 *
 * @since 0.1.0
 * @category constructors
 */
export const load = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<Checkpoint<S>, CheckpointError | Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function* () {
    const tensors = yield* Tensor.load(path)
    const checkpoint = yield* trainerCheckpoint(path, trainer, tensors)
    return checkpoint
  })

/**
 * {@link load} plus the sampler state saved by {@link saveWithSampler}.
 *
 * @since 0.1.0
 * @category constructors
 */
export const loadWithSampler = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<CheckpointWithSampler<S>, CheckpointError | Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function* () {
    const tensors = yield* Tensor.load(path)
    const checkpoint = yield* trainerCheckpoint(path, trainer, tensors)
    const orderTensor = yield* required(path, tensors, SAMPLER_ORDER_KEY)
    const cursorTensor = yield* required(path, tensors, SAMPLER_CURSOR_KEY)
    const epochTensor = yield* required(path, tensors, SAMPLER_EPOCH_KEY)
    const order = yield* Tensor.toNumberArray(orderTensor)
    const [cursor] = yield* Tensor.toNumberArray(cursorTensor)
    const [epoch] = yield* Tensor.toNumberArray(epochTensor)
    return { ...checkpoint, sampler: { order: Uint32Array.from(order), cursor, epoch } }
  })

const trainerEntries = <S, EL, RL, ED, RD, EO, RO>(
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  trained: Trainer.Trained<S>
): Effect.Effect<Record<string, Tensor.Any>, Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function* () {
    const entries: Record<string, Tensor.Any> = Object.fromEntries(
      trainer.model.names.map((name, i) => [`${PARAM_PREFIX}${name}`, trained.params[i]])
    )
    for (const [i, root] of trainer.config.optimizer.stateRoots(trained.state).entries()) {
      entries[`${STATE_PREFIX}${i}`] = root
    }
    entries[STEP_KEY] = yield* Tensor.full([], trained.step, { dtype: "u32" })
    return entries
  })

const required = (
  path: string,
  tensors: Record<string, Tensor.Concrete>,
  key: string
): Effect.Effect<Tensor.Concrete, CheckpointError> => {
  const tensor = tensors[key]
  return tensor === undefined
    ? new CheckpointError({ op: "checkpoint.load", message: `checkpoint ${path} is missing ${key}` })
    : Effect.succeed(tensor)
}

const trainerCheckpoint = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  tensors: Record<string, Tensor.Concrete>
): Effect.Effect<Checkpoint<S>, CheckpointError | Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function* () {
    const optimizer = trainer.config.optimizer
    const params: Array<Tensor.Any> = []
    for (const name of trainer.model.names) {
      params.push(yield* required(path, tensors, `${PARAM_PREFIX}${name}`))
    }
    const template = yield* optimizer.init(params)
    const roots: Array<Tensor.Any> = []
    for (const i of optimizer.stateRoots(template).keys()) {
      roots.push(yield* required(path, tensors, `${STATE_PREFIX}${i}`))
    }
    const stepTensor = yield* required(path, tensors, STEP_KEY)
    const [step] = yield* Tensor.toNumberArray(stepTensor)
    return { params, resume: { state: optimizer.rebuildState(template, roots), step } }
  })

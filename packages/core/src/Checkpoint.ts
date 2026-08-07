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
import type * as Model from "./Model.ts"
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
const SAMPLER_VERSION_KEY = "sampler:version"
const SAMPLER_LENGTH_KEY = "sampler:length"
const SAMPLER_BLOCK_KEY = "sampler:block"
const SAMPLER_BATCH_KEY = "sampler:batch"
const SAMPLER_VERSION = 1
const U32_MAX = 0xffff_ffff

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
 * {@link Sampler.restoreCheckpoint}.
 *
 * @since 0.1.0
 * @category models
 */
export interface CheckpointWithSampler<S> extends Checkpoint<S> {
  readonly sampler: Sampler.CheckpointSamplerState
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
  Effect.gen(function*() {
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
  Effect.gen(function*() {
    const entries = yield* trainerEntries(trainer, trained)
    const state = sampler.state()
    entries[SAMPLER_ORDER_KEY] = yield* Tensor.fromTypedArray(state.order, [state.order.length])
    entries[SAMPLER_CURSOR_KEY] = yield* Tensor.full([], state.cursor, { dtype: "u32" })
    entries[SAMPLER_EPOCH_KEY] = yield* Tensor.full([], state.epoch, { dtype: "u32" })
    entries[SAMPLER_VERSION_KEY] = yield* Tensor.full([], SAMPLER_VERSION, { dtype: "u32" })
    entries[SAMPLER_LENGTH_KEY] = yield* Tensor.full([], state.config.length, { dtype: "u32" })
    entries[SAMPLER_BLOCK_KEY] = yield* Tensor.full([], state.config.block, { dtype: "u32" })
    entries[SAMPLER_BATCH_KEY] = yield* Tensor.full([], state.config.batch, { dtype: "u32" })
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
  Effect.gen(function*() {
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
  Effect.gen(function*() {
    const tensors = yield* Tensor.load(path)
    const checkpoint = yield* trainerCheckpoint(path, trainer, tensors)
    const versionTensor = tensors[SAMPLER_VERSION_KEY]
    if (versionTensor === undefined) {
      const v1Key = [SAMPLER_LENGTH_KEY, SAMPLER_BLOCK_KEY, SAMPLER_BATCH_KEY].find((key) => tensors[key] !== undefined)
      if (v1Key !== undefined) {
        return yield* new CheckpointError({
          op: "checkpoint.load",
          message: `checkpoint ${path} has ${v1Key} without ${SAMPLER_VERSION_KEY}`
        })
      }
      const order = yield* readU32Vector(path, tensors, SAMPLER_ORDER_KEY)
      const cursor = yield* readU32Scalar(path, tensors, SAMPLER_CURSOR_KEY)
      const epoch = yield* readU32Scalar(path, tensors, SAMPLER_EPOCH_KEY)
      return {
        ...checkpoint,
        sampler: { _tag: "LegacySamplerState", order, cursor, epoch }
      }
    }
    const version = yield* decodeU32Scalar(path, SAMPLER_VERSION_KEY, versionTensor)
    if (version !== SAMPLER_VERSION) {
      return yield* new CheckpointError({
        op: "checkpoint.load",
        message: `checkpoint ${path} has unsupported sampler version ${version}`
      })
    }
    const order = yield* readU32Vector(path, tensors, SAMPLER_ORDER_KEY)
    const cursor = yield* readU32Scalar(path, tensors, SAMPLER_CURSOR_KEY)
    const epoch = yield* readU32Scalar(path, tensors, SAMPLER_EPOCH_KEY)
    const length = yield* readU32Scalar(path, tensors, SAMPLER_LENGTH_KEY)
    const block = yield* readU32Scalar(path, tensors, SAMPLER_BLOCK_KEY)
    const batch = yield* readU32Scalar(path, tensors, SAMPLER_BATCH_KEY)
    return {
      ...checkpoint,
      sampler: {
        _tag: "SamplerState",
        config: { length, block, batch },
        order,
        cursor,
        epoch
      }
    }
  })

const trainerEntries = <S, EL, RL, ED, RD, EO, RO>(
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  trained: Trainer.Trained<S>
): Effect.Effect<Record<string, Tensor.Any>, Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function*() {
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

const decodeU32Scalar = (
  path: string,
  key: string,
  tensor: Tensor.Concrete
): Effect.Effect<number, CheckpointError | Tensor.TensorError> =>
  Effect.gen(function*() {
    if (tensor.dtype !== "u32" || tensor.shape.length !== 0) {
      return yield* new CheckpointError({
        op: "checkpoint.load",
        message: `checkpoint ${path} has invalid ${key}: expected a u32 scalar, got ${tensor.dtype} [${tensor.shape}]`
      })
    }
    const values = yield* Tensor.toNumberArray(tensor)
    const value = values[0]
    if (values.length !== 1 || !Number.isInteger(value) || value < 0 || value > U32_MAX) {
      return yield* new CheckpointError({
        op: "checkpoint.load",
        message: `checkpoint ${path} has invalid ${key}: expected exactly one u32 value`
      })
    }
    return value
  })

const readU32Scalar = (
  path: string,
  tensors: Record<string, Tensor.Concrete>,
  key: string
): Effect.Effect<number, CheckpointError | Tensor.TensorError> =>
  Effect.flatMap(required(path, tensors, key), (tensor) => decodeU32Scalar(path, key, tensor))

const readU32Vector = (
  path: string,
  tensors: Record<string, Tensor.Concrete>,
  key: string
): Effect.Effect<Uint32Array, CheckpointError | Tensor.TensorError> =>
  Effect.gen(function*() {
    const tensor = yield* required(path, tensors, key)
    if (tensor.dtype !== "u32" || tensor.shape.length !== 1) {
      return yield* new CheckpointError({
        op: "checkpoint.load",
        message: `checkpoint ${path} has invalid ${key}: expected a u32 vector, got ${tensor.dtype} [${tensor.shape}]`
      })
    }
    const values = yield* Tensor.toNumberArray(tensor)
    if (
      values.length !== tensor.shape[0] ||
      values.some((value) => !Number.isInteger(value) || value < 0 || value > U32_MAX)
    ) {
      return yield* new CheckpointError({
        op: "checkpoint.load",
        message: `checkpoint ${path} has invalid ${key}: expected ${tensor.shape[0]} u32 values`
      })
    }
    return Uint32Array.from(values)
  })

const trainerCheckpoint = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  tensors: Record<string, Tensor.Concrete>
): Effect.Effect<Checkpoint<S>, CheckpointError | Tensor.TensorError, Device.CurrentDevice> =>
  Effect.gen(function*() {
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
    const step = yield* readU32Scalar(path, tensors, STEP_KEY)
    return { params, resume: { state: optimizer.rebuildState(template, roots), step } }
  })

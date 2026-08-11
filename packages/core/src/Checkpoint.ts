/**
 * Trainer checkpoint persistence: parameters, optimizer tensor state, global
 * step, and optionally data-sampler state in one safetensors file. The last
 * loss, training data, and {@link Trainer.Resume.startedAt} elapsed-time anchor
 * are not persisted; a loaded resume starts a new elapsed clock unless the
 * caller supplies an anchor.
 *
 * Optimizer state `S` is opaque. Saving serializes only the tensors returned
 * by `optimizer.stateRoots(state)`, in positional `state:<i>` order. Loading
 * calls the supplied optimizer's `init` to make a structural template, takes
 * the expected root count from that template, and passes the loaded roots to
 * `rebuildState`. Thus every resumable dynamic value must be represented by a
 * tensor root, while all non-root structure must be reproducible by `init`;
 * the root count, meaning, and order must remain stable.
 *
 * Archives contain no model or optimizer identity, hyperparameters, state
 * schema, library version, or other provenance (the sampler payload alone is
 * versioned). Compatibility is the caller's responsibility: load with the
 * same model parameter names and semantics, optimizer implementation and
 * configuration, and state-root schema. Obvious missing metadata is rejected,
 * but semantic mismatches may load and fail or diverge only when used. Every
 * restored tensor is imported by the selected runtime without placement
 * fallback.
 *
 * @since 0.1.0
 */
import { Data, Effect } from "effect"
import type * as Model from "./Model.ts"
import type * as Runtime from "./Runtime.ts"
import type * as Sampler from "./Sampler.ts"
import * as Tensor from "./Tensor.ts"
import type * as Trainer from "./Trainer.ts"

/**
 * A structurally invalid checkpoint payload, such as a missing required key,
 * malformed u32 metadata, or unsupported sampler payload version. Path access,
 * safetensors parsing, tensor import, and backend failures remain
 * {@link Tensor.TensorError}s.
 *
 * @since 0.1.0
 * @category errors
 */
export class CheckpointError extends Data.TaggedError("CheckpointError")<{
  /** The checkpoint operation that detected the invalid payload. */
  readonly op: string
  /** A diagnostic naming the path, entry, or unsupported version. */
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
 * A restored training position for `trainer.train(params, resume)`. It is
 * reconstructed under the trainer and runtime supplied to {@link load}, not
 * from provenance embedded in the archive.
 *
 * @since 0.1.0
 * @category models
 */
export interface Checkpoint<S> {
  /** Caller-owned materialized parameters in the supplied trainer's `model.names` order. */
  readonly params: Model.Params
  /**
   * The optimizer state rebuilt from loaded roots and the persisted u32 global
   * step. `startedAt` is absent because elapsed-time state is not persisted.
   */
  readonly resume: Trainer.Resume<S>
}

/**
 * A {@link Checkpoint} with decoded sampler state from
 * {@link loadWithSampler}. Sampler invariants are checked later by
 * {@link Sampler.restore} against the requested configuration.
 *
 * @since 0.1.0
 * @category models
 */
export interface CheckpointWithSampler<S> extends Checkpoint<S> {
  /** Decoded u32 sampler payload to validate and copy with {@link Sampler.restore}. */
  readonly sampler: Sampler.SamplerState
}

/**
 * Saves parameters by `trainer.model.names`, optimizer state roots by stable
 * positional index, and the global step as a u32 scalar. For a faithful round
 * trip, `trained.step` must be an integer in `0..4294967295`; this function
 * does not validate that range before backend conversion. Optimizer values not
 * exposed by `stateRoots`, the last loss, `Resume.startedAt`, and trainer
 * provenance are not written.
 *
 * `trained` must have been produced by this trainer with matching model arity,
 * parameter-name order, and optimizer root schema. That relationship is not
 * validated before entries are zipped. Saving borrows tensors and does not
 * clear them.
 *
 * `path` is handled by the selected runtime's direct safetensors extension.
 * Unsupported path I/O, an unwritable path, tensor evaluation, unsupported
 * dtypes, and backend serialization failures are {@link Tensor.TensorError}s.
 *
 * @since 0.1.0
 * @category destructors
 */
export const save = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  trained: Trainer.Trained<S>
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const entries = yield* trainerEntries(trainer, trained)
    yield* Tensor.save(path, entries)
  })

/**
 * {@link save} plus the sampler's configuration, order, cursor, epoch, and
 * payload version. All sampler metadata is represented as u32; configuration
 * and epoch are positive u32 values, while cursor and order indices may be
 * zero. The same path and backend failures surface as
 * {@link Tensor.TensorError}.
 *
 * This preserves the remaining draws in the current permutation. JavaScript
 * RNG state is not saved, so the next reshuffle is not guaranteed to match an
 * uninterrupted sampler. This function assumes `sampler.state()` already
 * satisfies every sampler invariant; it does not independently validate the
 * permutation, cursor, epoch, or unsigned-32-bit scalar ranges.
 *
 * @since 0.1.0
 * @category destructors
 */
export const saveWithSampler = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  trained: Trainer.Trained<S>,
  sampler: Sampler.Sampler
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> =>
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
 * Loads a checkpoint written by {@link save} (or the trainer portion of
 * {@link saveWithSampler}). Parameters are selected by the supplied model's
 * names. The supplied optimizer's `init` state determines how many positional
 * roots are required, and `rebuildState` injects those roots into that
 * template. Extra archive entries are ignored.
 *
 * The archive does not prove trainer compatibility. In particular, loaded
 * parameter and state-root shapes or dtypes are not compared with a persisted
 * schema, and optimizer identity and hyperparameters are not stored. Use the
 * same trainer semantics and a stable `stateRoots`/`rebuildState` contract.
 * The returned resume contains the u32 global step but no `startedAt` anchor.
 * Loaded parameters and state roots are caller-owned; retain them while
 * resuming and release them with {@link Tensor.clear} when no longer needed.
 * All archive tensors are imported before selection; metadata and extra entries
 * are not returned and rely on native finalization.
 *
 * Missing required checkpoint entries or malformed `meta:step` metadata fail
 * with {@link CheckpointError}. Missing paths, malformed safetensors files,
 * unavailable path I/O, unsupported imports, optimizer initialization, and
 * backend failures surface as {@link Tensor.TensorError}.
 *
 * @since 0.1.0
 * @category constructors
 */
export const load = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<Checkpoint<S>, CheckpointError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const tensors = yield* Tensor.load(path)
    const checkpoint = yield* trainerCheckpoint(path, trainer, tensors)
    return checkpoint
  })

/**
 * {@link load} plus the versioned sampler payload written by
 * {@link saveWithSampler}. Sampler scalars must be rank-0 u32 tensors and
 * `sampler:order` a rank-1 u32 tensor; only payload version 1 is accepted.
 * The decoded state is not a full sampler validation: pass it with the exact
 * saved configuration to {@link Sampler.restore}, which checks the window
 * count, permutation, batch-aligned cursor, and positive u32 epoch.
 *
 * Missing or malformed checkpoint entries and unsupported sampler versions
 * fail with {@link CheckpointError}; path, safetensors parsing, tensor import,
 * optimizer initialization, and backend failures remain
 * {@link Tensor.TensorError}s. RNG state is not part of the payload.
 *
 * @since 0.1.0
 * @category constructors
 */
export const loadWithSampler = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<CheckpointWithSampler<S>, CheckpointError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const tensors = yield* Tensor.load(path)
    const checkpoint = yield* trainerCheckpoint(path, trainer, tensors)
    const version = yield* readU32Scalar(path, tensors, SAMPLER_VERSION_KEY)
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
): Effect.Effect<Record<string, Tensor.Any>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const entries: Record<string, Tensor.Any> = Object.fromEntries(
      trainer.model.names.map((name, i) => [`${PARAM_PREFIX}${name}`, trained.params[i]])
    )
    for (const [i, root] of trainer.config.optimizer.stateRoots(trained.state).entries()) {
      entries[`${STATE_PREFIX}${i}`] = root
    }
    const first = trained.params[0]
    if (first === undefined) {
      return yield* new Tensor.TensorError({
        op: "checkpoint.save",
        message: "checkpoint.save: expected model parameters"
      })
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
): Effect.Effect<number, CheckpointError | Tensor.TensorError, Runtime.Runtime> =>
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
): Effect.Effect<number, CheckpointError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.flatMap(required(path, tensors, key), (tensor) => decodeU32Scalar(path, key, tensor))

const readU32Vector = (
  path: string,
  tensors: Record<string, Tensor.Concrete>,
  key: string
): Effect.Effect<Uint32Array, CheckpointError | Tensor.TensorError, Runtime.Runtime> =>
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
): Effect.Effect<Checkpoint<S>, CheckpointError | Tensor.TensorError, Runtime.Runtime> =>
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

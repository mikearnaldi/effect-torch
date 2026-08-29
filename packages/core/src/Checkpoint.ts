/**
 * Trainer checkpoint persistence: parameters, optimizer tensor state, global
 * step, and optionally data-sampler state in one safetensors file. This is a
 * continuation snapshot, not a self-describing trainer artifact. The last
 * loss, training data, learning-rate schedule, stop policy, and
 * {@link Trainer.Resume.startedAt} elapsed-time anchor are not persisted; a
 * loaded resume starts a new elapsed clock unless the caller supplies an
 * anchor.
 *
 * Optimizer state `S` is opaque. Saving serializes only the tensors returned
 * by `optimizer.stateRoots(state)`, in positional `state:<i>` order. Loading
 * calls the supplied optimizer's `init` to make a state template, takes the
 * expected root count from that template, and passes the loaded roots to
 * `rebuildState`. Every changing value needed to resume must therefore be a
 * tensor root. `init` must reproduce all non-root structure, and the root
 * count, meaning, and order must remain stable.
 *
 * Archives contain no model or optimizer identity, hyperparameters, state
 * schema, library version, or other identifying metadata. Only the sampler
 * payload is versioned. Compatibility is the caller's responsibility. Load
 * with the same model parameter names and behavior, optimizer implementation
 * and configuration, and state-root schema. The loader rejects missing required
 * entries, but other mismatches may load and fail or diverge only when used.
 * The selected runtime imports every restored tensor without placement
 * fallback.
 *
 * Persistence uses the runtime's direct path-based safetensors extension.
 * Source tensors are borrowed, and loading transfers ownership of selected
 * concrete handles to the caller. Atomic replacement is not part of the
 * runtime extension contract; the bundled CPU and Metal runtimes currently
 * write a temporary file and rename it, while custom runtimes may differ.
 *
 * @since 0.1.0
 */
import { Data, Effect, Exit } from "effect"
import type * as Runtime from "./Runtime.ts"
import type * as Sampler from "./Sampler.ts"
import * as Tensor from "./Tensor.ts"
import type * as Trainer from "./Trainer.ts"

/**
 * A structurally invalid checkpoint payload detected after safetensors import,
 * such as a missing required key, malformed u32 metadata, or unsupported
 * sampler payload version. Path access, safetensors parsing, tensor import,
 * serialization, and backend failures remain {@link Tensor.TensorError}s.
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
 * A restored training position for `trainer.train(params, resume)`.
 * {@link load} reconstructs it with the supplied trainer and runtime. The
 * archive contains no identifying metadata for either one.
 *
 * @since 0.1.0
 * @category models
 */
export interface Checkpoint<S> {
  /** Caller-owned materialized parameters in the supplied trainer's parameter-spec order. */
  readonly params: ReadonlyArray<Tensor.Concrete>
  /**
   * The optimizer state rebuilt from caller-owned loaded roots and the
   * persisted u32 global step. `startedAt` is absent because elapsed-time state
   * is not persisted.
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
 * Saves parameters by the trainer model's parameter specs, optimizer state
 * roots by stable positional index, and the global step as a u32 scalar. For a
 * faithful round trip, `trained.step` must be an integer in `0..4294967295`.
 * Values outside that range fail before serialization. Optimizer values not
 * exposed by `stateRoots`, the last loss, `Resume.startedAt`, and trainer
 * identity and configuration are not written.
 *
 * `trained` must have been produced by this trainer with matching parameter
 * order and optimizer root schema. Parameter arity and the persisted u32 step
 * are validated before serialization. A parameterless checkpoint is rejected.
 * Saving borrows tensors and does not clear them.
 *
 * `path` is handled by the selected runtime's direct safetensors extension.
 * The write materializes all entries together. It then performs
 * {@link Tensor.save}'s ordered best-effort cleanup and attempts to release
 * each temporary independently. `trained` remains untouched. Atomic replacement
 * and durability depend on the backend; this function does not guarantee them.
 * In particular, interruption or failure does not portably imply that the
 * destination was unchanged; bundled runtimes can observe cancellation after
 * their rename completed.
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
 * {@link save} plus a snapshot of the sampler's configuration, order, cursor,
 * epoch, and payload version. All sampler metadata is represented as u32;
 * configuration and epoch are positive u32 values, while cursor and order
 * indices may be zero. The same path and backend failures are reported as
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
 * same trainer behavior and a stable `stateRoots`/`rebuildState` contract.
 * The returned resume contains the u32 global step but no `startedAt` anchor.
 * Loaded parameters and state roots are caller-owned independent handles;
 * retain them while resuming and release each owned handle with
 * {@link Tensor.clear} when no longer needed. All archive tensors are imported
 * before selection. Unreturned metadata and extra entries are released before
 * success; every imported entry is released if validation fails or is
 * interrupted.
 *
 * Missing required checkpoint entries or malformed `meta:step` metadata fail
 * with {@link CheckpointError}. Missing paths, malformed safetensors files,
 * unavailable path I/O, unsupported imports, optimizer initialization, and
 * backend failures are reported as {@link Tensor.TensorError}.
 *
 * @since 0.1.0
 * @category constructors
 */
export const load = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<Checkpoint<S>, CheckpointError | Tensor.TensorError, Runtime.Runtime> =>
  withLoadedTensors(path, (tensors) => trainerCheckpoint(path, trainer, tensors), (checkpoint) => [
    ...checkpoint.params.filter(Tensor.isTensor),
    ...trainer.config.optimizer.stateRoots(checkpoint.resume.state).filter(Tensor.isTensor)
  ])

/**
 * {@link load} plus the versioned sampler payload written by
 * {@link saveWithSampler}. Sampler scalars must be rank-0 u32 tensors and
 * `sampler:order` a rank-1 u32 tensor; only payload version 1 is accepted.
 * The decoded state is a detached JavaScript value, not a full sampler
 * validation. Pass it with the exact saved configuration to
 * {@link Sampler.restore}, which checks the window count, permutation,
 * batch-aligned cursor, and positive u32 epoch.
 *
 * Missing or malformed checkpoint entries and unsupported sampler versions
 * fail with {@link CheckpointError}; path, safetensors parsing, tensor import,
 * optimizer initialization, scalar/vector readback, and backend failures remain
 * {@link Tensor.TensorError}s. Sampler metadata tensors and extra entries are
 * released before success. RNG state is not part of the payload.
 *
 * @since 0.1.0
 * @category constructors
 */
export const loadWithSampler = <S, EL, RL, ED, RD, EO, RO>(
  path: string,
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<CheckpointWithSampler<S>, CheckpointError | Tensor.TensorError, Runtime.Runtime> =>
  withLoadedTensors(
    path,
    (tensors) =>
      Effect.gen(function*() {
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
      }),
    (checkpoint) => [
      ...checkpoint.params.filter(Tensor.isTensor),
      ...trainer.config.optimizer.stateRoots(checkpoint.resume.state).filter(Tensor.isTensor)
    ]
  )

const withLoadedTensors = <A, E, R>(
  path: string,
  use: (tensors: Record<string, Tensor.Concrete>) => Effect.Effect<A, E, R>,
  retain: (value: A) => ReadonlyArray<Tensor.Concrete>
): Effect.Effect<A, E | Tensor.TensorError, R | Runtime.Runtime> =>
  Effect.flatMap(Tensor.load(path), (tensors) =>
    Effect.onExit(
      Effect.gen(function*() {
        const value = yield* use(tensors)
        const retained = new Set(retain(value))
        for (const tensor of Object.values(tensors)) {
          if (retained.has(tensor)) {
            continue
          }
          yield* Tensor.clear(tensor)
        }
        return value
      }),
      (exit) => Exit.isFailure(exit) ? Tensor.clearAll(Object.values(tensors)) : Effect.void
    ))

const trainerEntries = <S, EL, RL, ED, RD, EO, RO>(
  trainer: Trainer.Trainer<S, EL, RL, ED, RD, EO, RO>,
  trained: Trainer.Trained<S>
): Effect.Effect<Record<string, Tensor.Any>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    if (trained.params.length !== trainer.model.parameterSpecs.length) {
      return yield* new Tensor.TensorError({
        op: "checkpoint.save",
        message:
          `checkpoint.save: model has ${trainer.model.parameterSpecs.length} parameters, got ${trained.params.length}`
      })
    }
    if (!Number.isSafeInteger(trained.step) || trained.step < 0 || trained.step > U32_MAX) {
      return yield* new Tensor.TensorError({
        op: "checkpoint.save",
        message: `checkpoint.save: step must be a non-negative u32 integer, got ${trained.step}`
      })
    }
    const entries: Record<string, Tensor.Any> = Object.fromEntries(
      trainer.model.parameterSpecs.map((parameter, i) => [`${PARAM_PREFIX}${parameter.name}`, trained.params[i]])
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
    const params: Array<Tensor.Concrete> = []
    for (const { name } of trainer.model.parameterSpecs) {
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

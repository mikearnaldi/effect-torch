/**
 * Epoch samplers for language-model training: the non-overlapping
 * `block`-windows of a token sequence in a shuffled permutation, drawn
 * without replacement within each epoch. If the window count is not a
 * multiple of `batch`, the trailing partial batch is skipped before the
 * permutation reshuffles. A sampler deals only in window offsets —
 * materializing the windows into tensors is the caller's job, so any
 * token storage works.
 *
 * The full state (permutation, cursor, epoch) is exposed and restorable,
 * so a checkpoint can resume the data layout exactly where it stopped
 * (see {@link Checkpoint}).
 *
 * @since 0.1.0
 */
import { Data, Effect } from "effect"

/**
 * @since 0.1.0
 * @category errors
 */
export class SamplerError extends Data.TaggedError("SamplerError")<{
  readonly op: string
  readonly message: string
}> {}

/**
 * @since 0.1.0
 * @category models
 */
export interface SamplerConfig {
  /** The token count of the training data. */
  readonly length: number
  /** The window size in tokens. */
  readonly block: number
  /** The number of windows per draw. */
  readonly batch: number
}

/**
 * The restorable sampler state: the epoch permutation, the cursor into
 * it, and the 1-based epoch number.
 *
 * @since 0.1.0
 * @category models
 */
export interface SamplerState {
  readonly _tag: "SamplerState"
  readonly config: SamplerConfig
  readonly order: Uint32Array
  readonly cursor: number
  readonly epoch: number
}

/**
 * Sampler state written before checkpoints recorded {@link SamplerConfig}.
 * Kept only so existing checkpoints can be migrated with
 * {@link restoreCheckpoint}.
 *
 * @since 0.1.0
 * @category models
 */
export interface LegacySamplerState {
  readonly _tag: "LegacySamplerState"
  readonly order: Uint32Array
  readonly cursor: number
  readonly epoch: number
}

/**
 * A sampler state loaded from a checkpoint: current states validate their
 * saved configuration directly; legacy states require the one-batch-per-step
 * migration in {@link restoreCheckpoint}.
 *
 * @since 0.1.0
 * @category models
 */
export type CheckpointSamplerState = SamplerState | LegacySamplerState

/**
 * @since 0.1.0
 * @category models
 */
export interface Sampler {
  /** Draws the next batch of window start offsets. */
  readonly next: () => ReadonlyArray<number>
  /** The current restorable state. */
  readonly state: () => SamplerState
}

const shuffle = (order: Uint32Array) => {
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    const t = order[i]
    order[i] = order[j]
    order[j] = t
  }
}

const U32_MAX = 0xffff_ffff

const checkConfig = (op: string, config: SamplerConfig): Effect.Effect<number, SamplerError> => {
  if (
    !Number.isSafeInteger(config.length) ||
    !Number.isSafeInteger(config.block) ||
    !Number.isSafeInteger(config.batch) ||
    config.length < 1 ||
    config.block < 1 ||
    config.batch < 1 ||
    config.length > U32_MAX ||
    config.block > U32_MAX ||
    config.batch > U32_MAX
  ) {
    return new SamplerError({
      op,
      message:
        `${op}: length, block, and batch must be positive u32 integers, got length ${config.length}, block ${config.block}, batch ${config.batch}`
    })
  }
  const windows = Math.floor((config.length - 1) / config.block)
  if (windows < config.batch) {
    return new SamplerError({
      op,
      message:
        `${op}: need length > block and at least batch windows, got length ${config.length}, block ${config.block}, batch ${config.batch}`
    })
  }
  return Effect.succeed(windows)
}

const fromOrder = (config: SamplerConfig, order: Uint32Array, cursor: number, epoch: number): Sampler => {
  const stableConfig = { ...config }
  const windowCount = order.length
  const next = () => {
    if (cursor + stableConfig.batch > windowCount) {
      shuffle(order)
      cursor = 0
      epoch += 1
    }
    const starts = new Array<number>(stableConfig.batch)
    for (let b = 0; b < stableConfig.batch; b++) {
      starts[b] = order[cursor + b] * stableConfig.block
    }
    cursor += stableConfig.batch
    return starts
  }
  return {
    next,
    state: () => ({
      _tag: "SamplerState",
      config: { ...stableConfig },
      order: order.slice(),
      cursor,
      epoch
    })
  }
}

const checkState = (
  op: string,
  state: SamplerState,
  windows: number
): Effect.Effect<void, SamplerError> => {
  if (state.order.length !== windows) {
    return new SamplerError({
      op,
      message: `${op}: state holds ${state.order.length} windows, config implies ${windows}`
    })
  }
  if (
    !Number.isSafeInteger(state.cursor) ||
    state.cursor < 0 ||
    state.cursor > windows ||
    state.cursor % state.config.batch !== 0
  ) {
    return new SamplerError({
      op,
      message: `${op}: invalid cursor ${state.cursor} for ${windows} windows and batch ${state.config.batch}`
    })
  }
  if (!Number.isSafeInteger(state.epoch) || state.epoch < 1 || state.epoch > U32_MAX) {
    return new SamplerError({ op, message: `${op}: epoch must be a positive u32 integer, got ${state.epoch}` })
  }
  const seen = new Uint8Array(windows)
  for (const index of state.order) {
    if (index >= windows || seen[index] !== 0) {
      return new SamplerError({ op, message: `${op}: order is not a permutation of 0..${windows - 1}` })
    }
    seen[index] = 1
  }
  return Effect.void
}

/**
 * Creates a sampler over a fresh shuffled permutation.
 *
 * @since 0.1.0
 * @category constructors
 */
export const make = (config: SamplerConfig): Effect.Effect<Sampler, SamplerError> => {
  const stableConfig = { ...config }
  return Effect.map(checkConfig("sampler.make", stableConfig), (windows) => {
    const order = new Uint32Array(windows)
    for (let i = 0; i < windows; i++) order[i] = i
    shuffle(order)
    return fromOrder(stableConfig, order, 0, 1)
  })
}

/**
 * Restores a sampler from a previously captured {@link SamplerState}
 * (fails if the state does not match the config's window count).
 *
 * @since 0.1.0
 * @category constructors
 */
export const restore = (
  config: SamplerConfig,
  state: SamplerState
): Effect.Effect<Sampler, SamplerError> => {
  const stableConfig = { ...config }
  const stableState: SamplerState = {
    _tag: "SamplerState",
    config: { ...state.config },
    order: state.order.slice(),
    cursor: state.cursor,
    epoch: state.epoch
  }
  if (
    stableState.config.length !== stableConfig.length ||
    stableState.config.block !== stableConfig.block ||
    stableState.config.batch !== stableConfig.batch
  ) {
    return new SamplerError({
      op: "sampler.restore",
      message:
        `sampler.restore: checkpoint config length=${stableState.config.length}, block=${stableState.config.block}, batch=${stableState.config.batch}; requested length=${stableConfig.length}, block=${stableConfig.block}, batch=${stableConfig.batch}`
    })
  }
  return Effect.flatMap(
    checkConfig("sampler.restore", stableConfig),
    (windows) =>
      Effect.as(
        checkState("sampler.restore", stableState, windows),
        fromOrder(stableConfig, stableState.order, stableState.cursor, stableState.epoch)
      )
  )
}

/**
 * Restores sampler state loaded from a trainer checkpoint. Current
 * checkpoints validate their persisted config exactly. For legacy
 * checkpoints, the epoch and cursor are reconstructed from the checkpoint
 * step under the checkpoint convention of one sampler draw per optimizer
 * step.
 *
 * @since 0.1.0
 * @category constructors
 */
export const restoreCheckpoint = (
  config: SamplerConfig,
  state: CheckpointSamplerState,
  step: number
): Effect.Effect<Sampler, SamplerError> => {
  if (state._tag === "SamplerState") return restore(config, state)
  const stableConfig = { ...config }
  const stableState: LegacySamplerState = {
    _tag: "LegacySamplerState",
    order: state.order.slice(),
    cursor: state.cursor,
    epoch: state.epoch
  }
  if (!Number.isSafeInteger(step) || step < 1) {
    return new SamplerError({
      op: "sampler.restoreCheckpoint",
      message: `sampler.restoreCheckpoint: legacy checkpoint step must be a positive integer, got ${step}`
    })
  }
  return Effect.flatMap(checkConfig("sampler.restoreCheckpoint", stableConfig), (windows) => {
    const drawsPerEpoch = Math.floor(windows / stableConfig.batch)
    const expectedEpoch = Math.floor((step - 1) / drawsPerEpoch) + 1
    const expectedCursor = (((step - 1) % drawsPerEpoch) + 1) * stableConfig.batch
    if (stableState.epoch !== expectedEpoch || stableState.cursor !== expectedCursor) {
      return new SamplerError({
        op: "sampler.restoreCheckpoint",
        message:
          `sampler.restoreCheckpoint: legacy state epoch=${stableState.epoch}, cursor=${stableState.cursor} does not match step ${step} under batch ${stableConfig.batch} (expected epoch=${expectedEpoch}, cursor=${expectedCursor})`
      })
    }
    return restore(stableConfig, {
      _tag: "SamplerState",
      config: stableConfig,
      order: stableState.order,
      cursor: stableState.cursor,
      epoch: stableState.epoch
    })
  })
}

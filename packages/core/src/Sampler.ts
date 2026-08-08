/**
 * Epoch samplers for language-model training. They draw a shuffled
 * permutation of the non-overlapping `block` windows in a token sequence,
 * without replacement within an epoch. If the window count is not a multiple
 * of `batch`, the trailing partial batch is skipped before reshuffling. A
 * sampler returns only window offsets; the caller materializes the windows,
 * so any token storage can be used.
 *
 * {@link Sampler.state} snapshots the configuration, permutation, cursor, and
 * epoch for checkpoint persistence and {@link restore}. A restored sampler
 * reproduces the remaining draws in that permutation exactly. The global
 * `Math.random` state is not captured, so a later reshuffle is a new random
 * event and need not match an uninterrupted sampler.
 *
 * @since 0.1.0
 */
import { Data, Effect } from "effect"

/**
 * An invalid sampler configuration or restorable state. `op` identifies the
 * failed constructor and `message` describes the violated invariant.
 *
 * @since 0.1.0
 * @category errors
 */
export class SamplerError extends Data.TaggedError("SamplerError")<{
  /** The operation that rejected the configuration or state. */
  readonly op: string
  /** A diagnostic describing the invalid value or invariant. */
  readonly message: string
}> {}

/**
 * The token geometry and batch size of an epoch sampler. All fields must be
 * positive unsigned 32-bit integers, and the derived window count
 * `floor((length - 1) / block)` must be at least `batch`.
 *
 * @since 0.1.0
 * @category models
 */
export interface SamplerConfig {
  /** Total token count used to derive the available window starts. */
  readonly length: number
  /** Window size and stride in tokens. */
  readonly block: number
  /** Number of window offsets returned by each draw. */
  readonly batch: number
}

/**
 * A detached, restorable snapshot of a sampler. {@link restore} validates the
 * configuration, permutation, cursor, and epoch before copying them. The
 * snapshot fixes only the current permutation; it does not contain
 * `Math.random` state for future reshuffles.
 *
 * @since 0.1.0
 * @category models
 */
export interface SamplerState {
  /** Discriminant for serialized sampler state. */
  readonly _tag: "SamplerState"
  /** The exact configuration under which `order` was generated. */
  readonly config: SamplerConfig
  /** A permutation of every index from `0` through `floor((length - 1) / block) - 1`. */
  readonly order: Uint32Array
  /** Next unread position in `order`, in range and divisible by `config.batch`. */
  readonly cursor: number
  /** One-based epoch number; restorable values are limited to a positive u32. */
  readonly epoch: number
}

/**
 * A synchronous source of shuffled, non-overlapping token-window offsets with
 * detached state snapshots for restoration.
 *
 * @since 0.1.0
 * @category models
 */
export interface Sampler {
  /**
   * Draws the next complete batch of token offsets. At an epoch boundary it
   * drops any trailing partial batch, reshuffles, and then draws.
   */
  readonly next: () => ReadonlyArray<number>
  /** Returns a copy of the current configuration, permutation, cursor, and epoch. */
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
 * Creates a sampler over a fresh permutation shuffled with `Math.random`.
 * The configuration is copied and must contain positive u32 integers with at
 * least `batch` derived windows; otherwise it fails with {@link SamplerError}.
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
 * Restores a sampler from a previously captured {@link SamplerState}. The
 * requested configuration must exactly equal `state.config`; both must imply
 * the same window count. `order` must be a complete permutation, `cursor`
 * must be in range and batch-aligned, and `epoch` must be a positive u32.
 * Invalid input fails with {@link SamplerError}; accepted input is copied.
 *
 * Remaining draws in the saved permutation are exact. A reshuffle after that
 * uses the process's current `Math.random` state, which is not persisted.
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

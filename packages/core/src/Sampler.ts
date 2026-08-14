/**
 * Synchronous epoch samplers for next-token language-model training. They draw
 * a shuffled permutation of input blocks at offsets `i * block`, where a caller
 * can consume `block + 1` tokens to form a `block`-token input and its shifted
 * target. Input blocks do not overlap; adjacent full input/target spans share
 * their boundary token. Sampling is without replacement within the
 * usable complete batches of an epoch. If the window count is not a multiple
 * of `batch`, the trailing partial batch is skipped before reshuffling. A
 * sampler returns only offsets; the caller owns and materializes token storage.
 *
 * {@link Sampler.state} snapshots the configuration, permutation, cursor, and
 * epoch for checkpoint persistence and {@link restore}. A restored sampler
 * reproduces every remaining complete batch in that permutation exactly. The
 * global `Math.random` state is not captured, so a boundary that immediately
 * reshuffles, or any later reshuffle, is a new random event and need not match
 * an uninterrupted sampler.
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
  /** Total token count; each start must leave `block` following tokens available. */
  readonly length: number
  /** Input length and stride in tokens; a shifted-target sample uses `block + 1` tokens. */
  readonly block: number
  /** Number of window offsets returned by each draw. */
  readonly batch: number
}

/**
 * A detached, restorable snapshot of a sampler. {@link restore} validates the
 * configuration, permutation, cursor, and epoch before copying them. The
 * snapshot fixes only the current permutation; it does not contain
 * `Math.random` state for future reshuffles. Runtime validation reconstructs
 * rather than checks the incoming `_tag` discriminant.
 *
 * @since 0.1.0
 * @category models
 */
export interface SamplerState {
  /** Discriminant emitted by snapshots and checkpoint decoding. */
  readonly _tag: "SamplerState"
  /** The exact configuration under which `order` was generated. */
  readonly config: SamplerConfig
  /** A permutation of every index from `0` through `floor((length - 1) / block) - 1`. */
  readonly order: Uint32Array
  /**
   * Next unread position in `order`, in `0..order.length` and divisible by
   * `config.batch`. It may point at a trailing incomplete batch, which the next
   * draw skips before reshuffling.
   */
  readonly cursor: number
  /**
   * One-based current-permutation label. It increments when `next` reshuffles,
   * immediately before returning the new epoch's first batch, and wraps from
   * `0xffff_ffff` to `1`.
   */
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
   * Draws the next complete batch of token offsets synchronously. At an epoch
   * boundary it drops any trailing partial batch, reshuffles the whole order,
   * increments the epoch, and then draws. The returned number array is detached
   * from sampler state.
   */
  readonly next: () => ReadonlyArray<number>
  /** Returns detached copies of the current configuration and permutation plus cursor and epoch. */
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
      epoch = epoch === U32_MAX ? 1 : epoch + 1
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
 * Creates a sampler over a fresh permutation shuffled with the process-global
 * `Math.random`. The configuration is copied synchronously when `make` is
 * called and must contain positive u32 integers with at least `batch` derived
 * windows; otherwise the returned effect fails with {@link SamplerError}.
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
 * requested configuration must exactly equal `state.config`; both inputs are
 * copied synchronously when `restore` is called, and both must imply the same
 * window count. `order` must be a complete permutation, `cursor` must be in
 * range and batch-aligned, and `epoch` must be a positive u32.
 * Invalid input fails with {@link SamplerError}; accepted input is copied.
 *
 * Remaining complete batches in the saved permutation are exact. If fewer than
 * `batch` entries remain, the first restored draw already reshuffles and is not
 * reproducible from the snapshot alone. Reshuffling uses the process's current
 * `Math.random` state, which is not persisted. Epoch is a one-based u32
 * permutation counter and wraps from `0xffff_ffff` to `1`.
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

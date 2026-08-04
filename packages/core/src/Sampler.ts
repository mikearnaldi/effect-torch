/**
 * Epoch samplers for language-model training: the non-overlapping
 * `block`-windows of a token sequence in a shuffled permutation, so every
 * window is drawn exactly once per epoch (no replacement); the
 * permutation reshuffles at each epoch boundary. A sampler deals only in
 * window offsets — materializing the windows into tensors is the
 * caller's job, so any token storage works.
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
  readonly order: Uint32Array
  readonly cursor: number
  readonly epoch: number
}

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

const checkConfig = (op: string, config: SamplerConfig): Effect.Effect<number, SamplerError> => {
  const windows = Math.floor((config.length - 1) / config.block)
  if (config.block < 1 || config.batch < 1 || windows < config.batch) {
    return new SamplerError({
      op,
      message:
        `${op}: need length > block and at least batch windows, got length ${config.length}, block ${config.block}, batch ${config.batch}`
    })
  }
  return Effect.succeed(windows)
}

const fromOrder = (config: SamplerConfig, order: Uint32Array, cursor: number, epoch: number): Sampler => {
  const windowCount = order.length
  const next = () => {
    if (cursor + config.batch > windowCount) {
      shuffle(order)
      cursor = 0
      epoch += 1
    }
    const starts = new Array<number>(config.batch)
    for (let b = 0; b < config.batch; b++) {
      starts[b] = order[cursor + b] * config.block
    }
    cursor += config.batch
    return starts
  }
  return { next, state: () => ({ order, cursor, epoch }) }
}

/**
 * Creates a sampler over a fresh shuffled permutation.
 *
 * @since 0.1.0
 * @category constructors
 */
export const make = (config: SamplerConfig): Effect.Effect<Sampler, SamplerError> =>
  Effect.map(checkConfig("sampler.make", config), (windows) => {
    const order = new Uint32Array(windows)
    for (let i = 0; i < windows; i++) order[i] = i
    shuffle(order)
    return fromOrder(config, order, 0, 1)
  })

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
): Effect.Effect<Sampler, SamplerError> =>
  Effect.flatMap(checkConfig("sampler.restore", config), (windows) =>
    state.order.length !== windows
      ? new SamplerError({
        op: "sampler.restore",
        message: `sampler.restore: state holds ${state.order.length} windows, config implies ${windows}`
      })
      : Effect.succeed(fromOrder(config, state.order, state.cursor, state.epoch)))

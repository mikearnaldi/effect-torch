/**
 * Family-neutral token generation contracts.
 *
 * @since 0.1.0
 */
import type { Effect } from "effect"

/** @since 0.1.0 @category models */
export type StopReason = "stop" | "maxTokens"

/** @since 0.1.0 @category models */
export type PageStopReason = "eos" | "maxTokens"

/**
 * A nonempty page of committed token ids.
 *
 * @since 0.1.0
 * @category models
 */
export interface Page {
  readonly tokens: ReadonlyArray<number> | Uint32Array
  readonly stopReason?: PageStopReason | undefined
}

/**
 * Prepared input and sequential page publication.
 *
 * @since 0.1.0
 * @category models
 */
export interface Request<E = never> {
  readonly prompt: Uint32Array
  readonly maxTokens: number | undefined
  readonly eosTokens: ReadonlyArray<number>
  readonly onPage: (page: Page) => Effect.Effect<void, E>
}

/**
 * A generation request carrying consumer-specific settings.
 *
 * @since 0.1.0
 * @category models
 */
export interface RequestWithSettings<Settings, E = never> extends Request<E> {
  readonly settings: Settings
}

/**
 * A model-family adapter that publishes committed token pages.
 *
 * @since 0.1.0
 * @category models
 */
export type Generator<E = never, R = never, Settings = never, PageError = never> = (
  request: [Settings] extends [never] ? Request<PageError> : RequestWithSettings<Settings, PageError>
) => Effect.Effect<StopReason, E | PageError, R>

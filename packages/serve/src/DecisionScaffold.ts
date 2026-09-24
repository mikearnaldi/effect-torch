import { Effect } from "effect"
import * as Noise from "./DecisionNoise.ts"

/**
 * Label order matches Planner.optionsFor and Readout, including Noul false/true.
 *
 * @since 0.1.0
 * @category models
 */
export interface Label {
  readonly code: string
  readonly optionName: string
  readonly tokenId: number
}

/**
 * Verified single-question scaffold. Metadata is frozen. Buffer getters return
 * fresh caller-owned copies; the uncorrupted prefix/canvas stay private.
 *
 * @since 0.1.0
 * @category models
 */
export interface Scaffold {
  readonly semanticKey: string
  readonly model: string
  readonly prompt: string
  readonly tokenizerSha256: string
  readonly chatTemplateSha256: string
  readonly vocabularySize: number
  readonly labels: ReadonlyArray<Label>
  readonly slot: number
  /** Unpadded canvas length, including the turn-close token. */
  readonly contentLength: number
  readonly prefixIds: Uint32Array
  readonly canvasIds: Uint32Array
  readonly allowedTokenIds: Uint32Array
}

/**
 * Caller-owned one-step buffers. Only the answer slot contains fresh vocabulary noise.
 *
 * @since 0.1.0
 * @category models
 */
export interface Read {
  readonly semanticKey: string
  readonly readIndex: number
  readonly slot: number
  readonly prefixIds: Uint32Array
  readonly canvasIds: Uint32Array
  readonly allowedTokenIds: Uint32Array
}

/**
 * Corrupt only the verified slot, preserving the fixed thought/answer scaffold, close, and padding.
 *
 * @since 0.1.0
 * @category models
 */
export const prepareRead = (
  scaffold: Scaffold,
  options: {
    readonly seed: string
    readonly readIndex: number
  }
): Effect.Effect<Read, Noise.NoiseError> =>
  Effect.gen(function*() {
    const noise = yield* Noise.make({
      semanticKey: scaffold.semanticKey,
      seed: options.seed,
      readIndex: options.readIndex,
      length: 1,
      vocabularySize: scaffold.vocabularySize
    })

    const canvasIds = scaffold.canvasIds.slice()
    canvasIds[scaffold.slot] = noise[0]

    return Object.freeze({
      semanticKey: scaffold.semanticKey,
      readIndex: options.readIndex,
      slot: scaffold.slot,
      prefixIds: scaffold.prefixIds,
      canvasIds,
      allowedTokenIds: scaffold.allowedTokenIds
    })
  })

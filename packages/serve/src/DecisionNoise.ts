import { Effect, Schema } from "effect"
import { createHash } from "node:crypto"

/**
 * Invalid deterministic noise configuration.
 *
 * @since 0.1.0
 * @category models
 */
export class NoiseError extends Schema.TaggedErrorClass<NoiseError>()("NoiseError", { message: Schema.String }) {}

const NonNegativeInteger = Schema.Number.check(Schema.isInt(), Schema.isGreaterThanOrEqualTo(0))

/**
 * Semantic identity and deterministic independent-read noise controls.
 *
 * @since 0.1.0
 * @category models
 */
export const Options = Schema.Struct({
  semanticKey: Schema.String.check(Schema.isMinLength(1)),
  seed: Schema.String,
  readIndex: NonNegativeInteger,
  length: NonNegativeInteger,
  vocabularySize: Schema.Number.check(Schema.isInt(), Schema.isBetween({ minimum: 1, maximum: 2 ** 32 }))
})

/**
 * Validated semantic identity, seed, read index and vocabulary bounds for noise generation.
 *
 * @since 0.1.0
 * @category models
 */
export type Options = typeof Options.Type

/**
 * Uniform vocabulary IDs from SHA-256 hash-counter words with rejection sampling.
 * The explicit readIndex distinguishes independent reads. The semantic key must
 * come from an isolated question plan, never a caller ID or scheduler slot.
 * Length does not enter the seed, so extending a canvas preserves its prefix.
 * This samples initial token corruption only; it does not execute a denoiser.
 *
 * @since 0.1.0
 * @category models
 */
export const make = (options: Options): Effect.Effect<Uint32Array, NoiseError> =>
  Effect.gen(function*() {
    const config = yield* Schema.decodeUnknownEffect(Options, { onExcessProperty: "error" })(options).pipe(
      Effect.mapError((error) => new NoiseError({ message: error.message }))
    )

    return yield* Effect.try({
      try: () => {
        const prefix = JSON.stringify([
          "decision-model/uniform-noise/v1",
          config.seed,
          config.semanticKey,
          config.readIndex,
          config.vocabularySize
        ])

        const limit = Math.floor(2 ** 32 / config.vocabularySize) * config.vocabularySize
        const tokens = new Uint32Array(config.length)
        let position = 0
        let counter = 0n

        while (position < tokens.length) {
          const bytes = createHash("sha256").update(prefix).update(":" + counter++).digest()

          for (let offset = 0; offset < bytes.length && position < tokens.length; offset += 4) {
            const word = bytes.readUInt32BE(offset)

            // Discard the incomplete final bucket. Direct modulo would bias IDs.
            if (word < limit) tokens[position++] = word % config.vocabularySize
          }
        }

        return tokens
      },
      catch: (error) => new NoiseError({ message: String(error) })
    })
  })

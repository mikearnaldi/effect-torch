import { type Runtime, Tensor } from "@effect-torch/core"
import { Effect } from "effect"

/** @internal All possible token and position IDs must be exact F32 integers. */
export const canPackIds = (canvasLength: number, vocabSize: number): boolean =>
  Number.isSafeInteger(canvasLength) && canvasLength > 0 && canvasLength - 1 <= 2 ** 24 &&
  Number.isSafeInteger(vocabSize) && vocabSize > 0 && vocabSize - 1 <= 2 ** 24

/** @internal Preserve feedback separately and consolidate only exact statistics. */
export const pack = (
  statistics: ReadonlyArray<Tensor.Any>,
  canvasLength: number,
  vocabSize: number
): Effect.Effect<ReadonlyArray<Tensor.Any>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    if (
      !canPackIds(canvasLength, vocabSize) || statistics[3]!.dtype !== "f32" ||
      statistics[5]!.dtype !== "f32"
    ) return statistics

    const [sampled, argmax, entropy, order, mean] = yield* Effect.forEach(
      statistics.slice(1),
      (tensor) =>
        Effect.gen(function*() {
          const floating = yield* Tensor.cast(tensor, "f32")
          return yield* Tensor.reshape(floating, [tensor.shape.reduce((size, dimension) => size * dimension, 1)])
        })
    )
    return [statistics[0]!, yield* Tensor.concat([sampled!, argmax!, entropy!, order!, mean!])]
  })

/** @internal Slicing preserves every existing F32 statistic, including signed zero. */
export const unpack = (values: ReadonlyArray<number>, canvasLength: number) =>
  [
    values.slice(0, canvasLength),
    values.slice(canvasLength, 2 * canvasLength),
    values.slice(2 * canvasLength, 3 * canvasLength),
    values.slice(3 * canvasLength, 4 * canvasLength),
    values.slice(4 * canvasLength)
  ] as const

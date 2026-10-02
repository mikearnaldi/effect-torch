/** Private policy-placement diagnostic; no production API or policy changes. */
import { type Runtime, Tensor } from "@effect-torch/core"
import type { DiffusionGemma } from "@effect-torch/models"
import { Effect } from "effect"

/** CPU interpretation of the proposed GPU tail, independent of model execution. */
export const cpuTail = (
  prediction: DiffusionGemma.GenerationPrediction,
  noise: Uint32Array,
  history: ReadonlyArray<Uint32Array>,
  stabilityThreshold: number,
  entropyBound: number,
  confidenceThreshold: number
) => {
  const ordered = Float64Array.from(prediction.entropyOrder, (position) => prediction.tokenEntropy[position]!)
  // CUDA's F64 cumsum kernel visits prefix elements sequentially, starting at
  // +0. Preserve both JS Math.fround boundaries instead of an F32 scan.
  const prefixes = Float64Array.from(ordered, (_, last) => {
    let sum = 0
    for (let index = 0; index <= last; index++) sum += ordered[index]!
    return sum
  })
  const accepted = new Uint8Array(noise.length)
  prediction.entropyOrder.forEach((position, rank) => {
    accepted[position] = Math.fround(Math.fround(prefixes[rank]!) - ordered[rank]!) <= Math.fround(entropyBound) ? 1 : 0
  })
  const canvas = Uint32Array.from(
    noise,
    (token, position) => accepted[position] ? prediction.sampledTokens[position]! : token
  )
  const stable = history.length === stabilityThreshold &&
    history.every((old) => old.every((token, position) => token === prediction.argmaxTokens[position]))
  return {
    accepted,
    canvas,
    draft: prediction.argmaxTokens.slice(),
    history: stabilityThreshold === 0 ? [] : [...history, prediction.argmaxTokens.slice()].slice(-stabilityThreshold),
    done: stable && prediction.meanEntropy < Math.fround(confidenceThreshold)
  }
}

/** Bounded ST=1 tail. Existing sampler statistics and their arithmetic remain intact. */
export const gpuTail = (
  statistics: ReadonlyArray<Tensor.Any>,
  noise: Tensor.Any,
  priorArgmax: Tensor.Any,
  historyAvailable: Tensor.Any,
  entropyBound: number,
  confidenceThreshold: number
): Effect.Effect<ReadonlyArray<Tensor.Any>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const sampled = statistics[1]!
    const argmax = statistics[2]!
    const entropy = statistics[3]!
    const order = statistics[4]!
    const mean = statistics[5]!
    const sorted = yield* Tensor.gather(entropy, order, { dim: 1 })
    const sum64 = yield* Tensor.cumsum(yield* Tensor.cast(sorted, "f64"), 1)
    const roundedSum = yield* Tensor.cast(sum64, "f32")
    const preceding = yield* Tensor.sub(roundedSum, sorted)
    const acceptedSorted = yield* Tensor.le(preceding, yield* Tensor.full([], Math.fround(entropyBound)))
    const accepted = yield* Tensor.scatterAdd(
      yield* Tensor.zeros(entropy.shape, { dtype: "u8" }),
      order,
      acceptedSorted,
      { dim: 1 }
    )
    const canvas = yield* Tensor.where(accepted, sampled, noise)
    const stable = yield* Tensor.logicalAnd(
      historyAvailable,
      yield* Tensor.all(yield* Tensor.eq(priorArgmax, argmax))
    )
    const done = yield* Tensor.logicalAnd(
      stable,
      yield* Tensor.lt(mean, yield* Tensor.full([], Math.fround(confidenceThreshold)))
    )
    return [canvas, argmax, done]
  })

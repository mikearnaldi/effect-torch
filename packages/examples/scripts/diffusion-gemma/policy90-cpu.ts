/** Explicit CPU equivalence checks; no CUDA import, allocation, or inference. */
import { DiffusionGemma } from "@effect-torch/models"
import { Effect } from "effect"
import assert from "node:assert/strict"
import { cpuTail } from "./policy90.ts"

let cases = 0
const check = (
  entropy: ReadonlyArray<number>,
  bound: number,
  mean: number,
  confidence: number,
  history: ReadonlyArray<Uint32Array>,
  threshold: number
) => {
  const width = entropy.length
  const prediction: DiffusionGemma.GenerationPrediction = {
    tokenEntropy: Float32Array.from(entropy),
    entropyOrder: Uint32Array.from(
      Array.from({ length: width }, (_, i) => i).sort((a, b) => entropy[a]! - entropy[b]!)
    ),
    sampledTokens: Uint32Array.from({ length: width }, (_, i) => 100 + i),
    argmaxTokens: Uint32Array.from({ length: width }, (_, i) => i),
    meanEntropy: Math.fround(mean)
  }
  const noise = Uint32Array.from({ length: width }, (_, i) => 200 + i)
  const oldHistory = history.map((entry) => entry.slice())
  const expected = DiffusionGemma.sampleGenerationCanvas(new Uint32Array(width), prediction, noise, bound)
  const stopped = DiffusionGemma.stopGeneration({ history }, prediction, threshold, confidence)
  const actual = cpuTail(prediction, noise, history, threshold, bound, confidence)
  assert.deepEqual(actual.accepted, expected.acceptedMask)
  assert.deepEqual(actual.canvas, expected.canvas)
  assert.deepEqual(actual.draft, prediction.argmaxTokens)
  assert.deepEqual(actual.history, stopped.state.history)
  assert.equal(actual.done, stopped.done)
  assert.deepEqual(history, oldHistory, "prior history must remain unchanged")
  assert.deepEqual(noise, Uint32Array.from({ length: width }, (_, i) => 200 + i))
  cases++
}

const values = [
  [0, 0, 0, 0],
  [-0, 0, -0, 0],
  [.1, .1, .1, .1],
  [.01, .02, .04, .08],
  [1e-30, .1, 1e10, 1e-30],
  [NaN, .01, .1, Infinity],
  [-Infinity, .1, Infinity, NaN]
]
for (const entropy of values) {
  for (const bound of [.1, Math.fround(.1) - 2 ** -27, Math.fround(.1) + 2 ** -27]) {
    for (const threshold of [0, 1, 2]) {
      for (
        const history of [[], [Uint32Array.from([0, 1, 2, 3])], [
          Uint32Array.from([0, 1, 2, 3]),
          Uint32Array.from([0, 1, 2, 3])
        ], [Uint32Array.from([3, 2, 1, 0])]]
      ) {
        for (const mean of [0, .005 - 2 ** -31, .005, .005 + 2 ** -31, NaN, Infinity]) {
          check(entropy, bound, mean, .005, history, threshold)
        }
      }
    }
  }
}
// Random finite entropy/order fixtures include many equal-entropy positions.
let bits = 20261001
for (let trial = 0; trial < 100; trial++) {
  const entropy = Array.from({ length: 256 }, () => {
    bits = (Math.imul(bits, 1664525) + 1013904223) >>> 0
    return Math.fround((bits % 31) / 200)
  })
  check(entropy, .1, .005, .005, [], 1)
}
// A terminal refinement still consumes exactly one entire canvas draw.
for (const seed of [0, 1, 20261001, 0xffff_ffff]) {
  const reference = Effect.runSync(DiffusionGemma.generationRandom(seed))
  const candidate = Effect.runSync(DiffusionGemma.generationRandom(seed))
  const initial = reference.canvas(256, 262144)
  assert.deepEqual(candidate.canvas(256, 262144), initial)
  let draws = 0
  const policy = DiffusionGemma.generationPolicy({
    canvasLength: 256,
    maxSteps: 48,
    entropyBound: .1,
    stabilityThreshold: 1,
    confidenceThreshold: .005,
    eosTokenIds: [1, 106, 50],
    padTokenId: 0
  }, () =>
    Effect.sync(() => {
      draws++
      return reference.canvas(256, 262144)
    }))
  const prediction: DiffusionGemma.GenerationPrediction = {
    sampledTokens: new Uint32Array(256),
    argmaxTokens: new Uint32Array(256),
    tokenEntropy: new Float32Array(256),
    entropyOrder: Uint32Array.from({ length: 256 }, (_, i) => i),
    meanEntropy: 0
  }
  const history = [new Uint32Array(256)]
  const result = Effect.runSync(policy.refine({
    canvas: initial,
    prediction,
    state: { history },
    block: { index: 0, position: 32, remainingTokens: 64, canvasLength: 256 },
    step: { index: 1, remaining: 47 }
  }))
  const gpuInterpretation = cpuTail(prediction, candidate.canvas(256, 262144), history, 1, .1, .005)
  assert(result.done && gpuInterpretation.done)
  assert.equal(draws, 1)
  assert.deepEqual(result.canvas, gpuInterpretation.canvas)
  assert.deepEqual(
    reference.canvas(256, 262144),
    candidate.canvas(256, 262144),
    "terminal draw must advance RNG identically"
  )
  cases++
}
process.stdout.write(
  JSON.stringify({ status: "passed", cases, scope: "CPU mathematical policy equivalence only; GPU fixture pending" }) +
    "\n"
)

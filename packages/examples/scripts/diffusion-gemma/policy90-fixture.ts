/** Ignored-by-default manual CUDA policy fixture; run only in the reserved GPU queue. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Tensor } from "@effect-torch/core"
import type { DiffusionGemma } from "@effect-torch/models"
import { Effect } from "effect"
import assert from "node:assert/strict"
import { writeFileSync } from "node:fs"
import { cpuTail, gpuTail } from "./policy90.ts"

const [output, extra] = process.argv.slice(2)
assert(output && extra === undefined, "usage: policy90-fixture.ts NEW_OUTPUT_JSON")
const main = Effect.scoped(Effect.gen(function*() {
  let cases = 0
  for (const width of [4, 256]) {
    const placeholder = (slot: number, shape: ReadonlyArray<number>, dtype: Tensor.DType) =>
      Tensor.zeros(shape, { dtype }).pipe(Effect.flatMap((x) => Tensor.makeInput(slot, x)))
    const sampled = yield* placeholder(0, [1, width], "u32")
    const argmax = yield* placeholder(1, [1, width], "u32")
    const entropy = yield* placeholder(2, [1, width], "f32")
    const order = yield* placeholder(3, [1, width], "u32")
    const mean = yield* placeholder(4, [], "f32")
    const noise = yield* placeholder(5, [1, width], "u32")
    const history = yield* placeholder(6, [1, width], "u32")
    const available = yield* Tensor.makeScalarInput(7, "u8")
    const roots = yield* gpuTail([entropy, sampled, argmax, entropy, order, mean], noise, history, available, .1, .005)
    const program = yield* Tensor.freezeProgram(roots)
    const fixtures = [
      Array(width).fill(0),
      Array.from({ length: width }, (_, i) => i % 2 === 0 ? -0 : 0),
      Array(width).fill(Math.fround(.1)),
      Array.from({ length: width }, (_, i) => Math.fround((i % 31) / 200)),
      Array.from({ length: width }, (_, i) => [NaN, .01, .1, Infinity][i % 4]!),
      Array.from({ length: width }, (_, i) => [-Infinity, .1, Infinity, NaN][i % 4]!)
    ]
    for (const values of fixtures) {
      for (const ready of [0, 1, 2]) {
        for (const meanValue of [0, Math.fround(.005), Math.fround(.005 - 2 ** -31), NaN]) {
          const prediction: DiffusionGemma.GenerationPrediction = {
            sampledTokens: Uint32Array.from({ length: width }, (_, i) => i + 10),
            argmaxTokens: Uint32Array.from({ length: width }, (_, i) => i),
            tokenEntropy: Float32Array.from(values),
            entropyOrder: Uint32Array.from(
              Array.from({ length: width }, (_, i) => i).sort((a, b) => values[a]! - values[b]!)
            ),
            meanEntropy: meanValue
          }
          const random = Uint32Array.from({ length: width }, (_, i) => i + 1000)
          const prior = prediction.argmaxTokens.slice()
          if (ready === 2) prior[width - 1] = prior[width - 1]! + 1
          const expected = cpuTail(prediction, random, ready ? [prior] : [], 1, .1, .005)
          const inputs = yield* Effect.acquireRelease(
            Tensor.compute([
              yield* Tensor.fromTypedArray(prediction.sampledTokens, [1, width]),
              yield* Tensor.fromTypedArray(prediction.argmaxTokens, [1, width]),
              yield* Tensor.fromTypedArray(prediction.tokenEntropy, [1, width]),
              yield* Tensor.fromTypedArray(prediction.entropyOrder, [1, width]),
              yield* Tensor.fromTypedArray(Float32Array.of(meanValue), []),
              yield* Tensor.fromTypedArray(random, [1, width]),
              yield* Tensor.fromTypedArray(prior, [1, width])
            ]),
            Tensor.clearAll,
            { interruptible: true }
          )
          const actual = yield* Effect.acquireRelease(
            Tensor.runProgram(program, inputs, [Math.min(ready, 1)]),
            Tensor.clearAll,
            {
              interruptible: true
            }
          )
          assert.deepEqual(yield* Tensor.toTypedArray(actual[0]!), expected.canvas)
          assert.deepEqual(yield* Tensor.toTypedArray(actual[1]!), expected.draft)
          assert.deepEqual(yield* Tensor.toTypedArray(actual[2]!), Uint8Array.of(expected.done ? 1 : 0))
          cases++
        }
      }
    }
  }
  return {
    status: "passed",
    cases,
    scope: "actual GPU policy math; finite/ties/nonfinite/history/strict thresholds; no inference timing"
  }
}))
const result = await Effect.runPromise(main.pipe(Effect.provide(BackendCuda.layer())))
writeFileSync(output, JSON.stringify(result, null, 2) + "\n", { flag: "wx" })
process.stdout.write(JSON.stringify(result) + "\n")

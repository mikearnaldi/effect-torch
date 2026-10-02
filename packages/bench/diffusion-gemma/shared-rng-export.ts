/** Export fresh CUDA uniforms and canonical canvas streams for cross-engine RNG parity. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/models"
import { Effect } from "effect"
import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"

const directory = process.env.RNG_EVIDENCE_DIR
assert(directory !== undefined, "RNG_EVIDENCE_DIR must name a fresh output directory")
await mkdir(directory)
const records: Array<unknown> = []
const canvases: Array<unknown> = []
await Effect.runPromise(Effect.provide(
  Effect.scoped(Effect.gen(function*() {
    for (const seed of [0, 12345, 0xffff_ffff]) {
      const random = yield* DiffusionGemma.generationRandom(seed)
      for (let draw = 0; draw < 5; draw++) {
        canvases.push({ seed, draw, vocabSize: 262144, values: Array.from(random.canvas(256, 262144)) })
      }
      const rejection = yield* DiffusionGemma.generationRandom(seed)
      for (let draw = 0; draw < 3; draw++) {
        canvases.push({ seed, draw, vocabSize: 2147483649, values: Array.from(rejection.canvas(31, 2147483649)) })
      }
      for (const shape of seed === 12345 ? [[1, 4099], [1, 256, 262144]] : [[1, 4099]]) {
        const uniform = yield* Tensor.uniform(shape, { dtype: "f32" })
        const program = yield* Tensor.freezeProgram([uniform], { randomSeed: seed })
        let first: Uint8Array | undefined
        const draws = shape.length === 3 ? 1 : 3
        for (let draw = 0; draw < draws; draw++) {
          yield* Effect.scoped(Effect.gen(function*() {
            const output = yield* Effect.acquireRelease(Tensor.runProgram(program, []), Tensor.clearAll, {
              interruptible: true
            })
            const values = yield* Tensor.toTypedArray(output[0]!)
            assert(values instanceof Float32Array)
            const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
            if (draw === 0 && shape.length === 2) first = bytes.slice()
            const file = `uniform-${seed}-${shape.length}-${draw}.bin`
            yield* Effect.promise(() => writeFile(join(directory, file), bytes))
            records.push({ seed, draw, shape, file, sha256: createHash("sha256").update(bytes).digest("hex") })
          }))
        }
        if (first !== undefined) {
          // Fresh compilation with the same graph/seed must reset the invocation
          // stream, including when earlier requests and warmups used that seed.
          const repeated = yield* Tensor.freezeProgram([uniform], { randomSeed: seed })
          yield* Effect.scoped(Effect.gen(function*() {
            const output = yield* Effect.acquireRelease(Tensor.runProgram(repeated, []), Tensor.clearAll, {
              interruptible: true
            })
            const values = yield* Tensor.toTypedArray(output[0]!)
            assert.deepEqual(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), first)
          }))
        }
      }
    }
  })),
  BackendCuda.layer()
))
await writeFile(
  join(directory, "manifest.json"),
  JSON.stringify({ records, canvases, freshCompileResetsStream: true }, null, 2)
)
process.stdout.write(
  JSON.stringify({
    directory,
    uniformRecords: records.length,
    canvasRecords: canvases.length,
    freshCompileResetsStream: true
  }) + "\n"
)

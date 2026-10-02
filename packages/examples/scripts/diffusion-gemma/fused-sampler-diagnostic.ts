/** Private fixed-refinement diagnostic; excluded from generation acceptance. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Runtime, Safetensors, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/models"
import { Effect } from "effect"
import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import * as Statistics from "../../../models/src/internal/diffusionGemmaStatistics.ts"

const sha = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex")
const [checkpoint, initializedPath, caseDirectory, outputDirectory, extra] = process.argv.slice(2)
assert(
  checkpoint && initializedPath && caseDirectory && outputDirectory && extra === undefined,
  "usage: fused-sampler-diagnostic.ts CHECKPOINT INITIALIZED_STATE REPLAY_CASE NEW_OUTPUT"
)
mkdirSync(outputDirectory) // Fresh output only.
const replayBytes = readFileSync(join(caseDirectory, "manifest.json"))
// SAFETY: The caller supplies a validated replay bank; the selected feedback hash,
// tensor metadata, canvas width, temperature, and prefix schema are checked below.
const replay = JSON.parse(replayBytes.toString()) as {
  request: { promptTokenIds: Array<number>; seed: number }
  steps: Array<{
    canvasTokenIds: Array<number>
    temperature: number
    feedbackInput: null | {
      file: string
      sha256: string
      dtype: string
      shape: Array<number>
    }
  }>
}
const fixed = replay.steps.find((step) => step.feedbackInput !== null)
assert(fixed?.feedbackInput && Number.isFinite(fixed.temperature))
assert.equal(sha(readFileSync(join(caseDirectory, fixed.feedbackInput.file))), fixed.feedbackInput.sha256)
const modelSource = readFileSync(
  fileURLToPath(new URL("../../../models/src/DiffusionGemma.ts", import.meta.url)),
  "utf8"
)
const samplerStart = modelSource.indexOf("const generationStatisticsWithDeviceNoise = (")
const samplerEnd = modelSource.indexOf("\n/**", samplerStart)
assert(samplerStart > 0 && samplerEnd > samplerStart)
const expectedBuilder = modelSource.slice(samplerStart, samplerEnd)
  .replace("const generationStatisticsWithDeviceNoise = (", "const deviceStatistics = (")
const ownSource = readFileSync(fileURLToPath(import.meta.url), "utf8")
assert(ownSource.includes(expectedBuilder), "private sampler copy differs from current model builder")

const deviceStatistics = (
  logits: Tensor.Any,
  temperature: Tensor.Any,
  predictionDtype: Tensor.DType
): Effect.Effect<ReadonlyArray<Tensor.Any>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const processed = yield* Tensor.div(logits, temperature)
    const uniform = yield* Tensor.uniform(logits.shape, { dtype: "f32" })
    const gumbel = yield* Tensor.neg(yield* Tensor.log(yield* Tensor.neg(yield* Tensor.log(uniform))))
    const sampled = yield* Tensor.cast(yield* Tensor.argmax(yield* Tensor.add(processed, gumbel), 2), "u32")
    const argmax = yield* Tensor.cast(yield* Tensor.argmax(processed, 2), "u32")
    const normalized = yield* Tensor.sub(processed, yield* Tensor.logsumexp(processed, { dims: [2], keepdims: true }))
    const clamped = yield* Tensor.clamp(normalized, { min: -3.4028234663852886e38 })
    const entropy = yield* Tensor.neg(
      yield* Tensor.sum(
        yield* Tensor.mul(clamped, yield* Tensor.softmax(normalized)),
        { dims: [2] }
      )
    )
    const order = yield* Tensor.topKIndices(yield* Tensor.neg(entropy), logits.shape[1]!)
    const mean = yield* Tensor.mean(entropy)

    return [yield* Tensor.cast(processed, predictionDtype), sampled, argmax, entropy, order, mean]
  })

const main = Effect.scoped(Effect.gen(function*() {
  const loaded = yield* Effect.acquireRelease(
    DiffusionGemma.loadCheckpoint(checkpoint),
    (value) => Tensor.clearAll(value.ownedParameters),
    { interruptible: true }
  )
  const initialized = yield* Effect.acquireRelease(
    Safetensors.load(initializedPath),
    (value) => Tensor.clearAll(Object.values(value)),
    { interruptible: true }
  )
  const feedback = yield* Effect.acquireRelease(
    Safetensors.load(join(caseDirectory, fixed.feedbackInput!.file)),
    (value) => Tensor.clearAll(Object.values(value)),
    { interruptible: true }
  )
  const model = DiffusionGemma.fromTensors(loaded, { ...loaded.tensors, ...initialized })
  const width = fixed.canvasTokenIds.length
  assert.equal(width, loaded.config.canvas_length)
  assert.equal(feedback.feedback!.dtype, model.definition.predictionDtype)
  assert.deepEqual(feedback.feedback!.shape, [1, width, model.definition.vocabSize])
  const input = (slot: number, shape: ReadonlyArray<number>, dtype: Tensor.DType) =>
    Tensor.zeros(shape, { dtype }).pipe(Effect.flatMap((value) => Tensor.makeInput(slot, value)))
  const tokens = yield* input(0, [1, width], "u32")
  const positions = yield* input(1, [1, width], "u32")
  const previous = yield* input(2, feedback.feedback!.shape, feedback.feedback!.dtype)
  const temperature = yield* input(3, [], "f32")
  const hidden = yield* model.definition.denoise(model.parameters, tokens, positions, {
    _tag: "Refinement",
    logits: previous
  })
  const logits = yield* model.definition.readout(model.parameters, hidden, { _tag: "Full" })
  const fusedRoots = yield* deviceStatistics(logits, temperature, model.definition.predictionDtype)
    .pipe(Effect.flatMap((values) => Statistics.pack(values, width, model.definition.vocabSize)))
  const state: Runtime.DecodeStateRequest = {
    maxTokens: 4096,
    blockSize: 16,
    kvDtype: model.definition.dtype,
    batch: 1,
    access: "ReadOnly",
    currentBlockAttention: "Bidirectional"
  }
  const fused = yield* Tensor.compileDecodeProgram(fusedRoots, state, { randomSeed: replay.request.seed })
  const bodyAndHead = yield* Tensor.compileDecodeProgram([logits], state)
  const samplerLogits = yield* input(0, logits.shape, logits.dtype)
  const samplerTemperature = yield* input(1, [], "f32")
  const samplerRoots = yield* deviceStatistics(samplerLogits, samplerTemperature, model.definition.predictionDtype)
    .pipe(Effect.flatMap((values) => Statistics.pack(values, width, model.definition.vocabSize)))
  const sampler = yield* Tensor.freezeProgram(samplerRoots, { randomSeed: replay.request.seed })
  const promptWidth = replay.request.promptTokenIds.length
  const promptTokens = yield* input(0, [1, promptWidth], "u32")
  const promptPositions = yield* input(1, [1, promptWidth], "u32")
  const encoderRoot = yield* model.definition.encode(model.parameters, promptTokens, promptPositions)
  const encoder = yield* Tensor.compileDecodeProgram([encoderRoot], {
    ...state,
    access: "Append",
    currentBlockAttention: "Causal"
  })
  assert(Runtime.sameDecodeStateSchema(fused, bodyAndHead) && Runtime.sameDecodeStateSchema(fused, encoder))
  const pool = yield* Tensor.makeKvPoolFromSchema(encoder)
  const sequence = yield* Effect.acquireRelease(
    Tensor.makeKvSequence(pool),
    (value) => Effect.orDie(Tensor.releaseKvSequence(value)),
    { interruptible: true }
  )
  const prefixOutputs = yield* Tensor.runDecodeProgram(
    encoder,
    [
      yield* Tensor.fromTypedArray(Uint32Array.from(replay.request.promptTokenIds), [1, promptWidth]),
      yield* Tensor.fromTypedArray(Uint32Array.from({ length: promptWidth }, (_, i) => i), [1, promptWidth])
    ],
    sequence,
    replay.request.promptTokenIds
  )
  yield* Tensor.clearAll(prefixOutputs)
  const prefix = yield* Effect.acquireRelease(
    Tensor.snapshotKvSequence(sequence),
    (value) => Effect.orDie(Tensor.releaseKvPrefix(value)),
    { interruptible: true }
  )
  const bindings = yield* Effect.acquireRelease(
    Tensor.compute([
      yield* Tensor.fromTypedArray(Uint32Array.from(fixed.canvasTokenIds), [1, width]),
      yield* Tensor.fromTypedArray(Uint32Array.from({ length: width }, (_, i) => promptWidth + i), [1, width]),
      feedback.feedback!,
      yield* Tensor.full([], fixed.temperature, { dtype: "f32" })
    ]),
    Tensor.clearAll,
    { interruptible: true }
  )
  const measure = (mode: "fused" | "separate") =>
    Effect.scoped(Effect.gen(function*() {
      const start = performance.now()
      const outputs = yield* Effect.acquireRelease(
        mode === "fused"
          ? Tensor.runReadOnlyDecodeProgram(fused, bindings, prefix, width)
          : Effect.gen(function*() {
            const read = yield* Effect.acquireRelease(
              Tensor.runReadOnlyDecodeProgram(bodyAndHead, bindings.slice(0, 3), prefix, width),
              Tensor.clearAll,
              { interruptible: true }
            )
            return yield* Tensor.runProgram(sampler, [read[0]!, bindings[3]!])
          }),
        Tensor.clearAll,
        { interruptible: true }
      )
      assert.equal(outputs.length, 2)
      const packed = yield* Tensor.toTypedArray(outputs[1]!)
      const elapsedMs = performance.now() - start
      assert(packed instanceof Float32Array && packed.length === 4 * width + 1)
      const carried = yield* Tensor.toTypedArray(outputs[0]!) // Full feedback readback excluded from timing.
      assert(carried instanceof Float32Array)
      return {
        elapsedMs,
        statisticsSha256: sha(new Uint8Array(packed.buffer, packed.byteOffset, packed.byteLength)),
        feedbackSha256: sha(new Uint8Array(carried.buffer, carried.byteOffset, carried.byteLength))
      }
    }))
  const records = []
  for (let round = -2; round < 22; round++) {
    const order = round < 11 ? ["separate", "fused"] as const : ["fused", "separate"] as const
    const first = yield* measure(order[0])
    const second = yield* measure(order[1])
    assert.equal(first.statisticsSha256, second.statisticsSha256, `statistics mismatch at invocation ${round + 2}`)
    assert.equal(first.feedbackSha256, second.feedbackSha256, `feedback mismatch at invocation ${round + 2}`)
    const record = { round, rngInvocation: round + 2, order, [order[0]]: first, [order[1]]: second }
    records.push(record)
    writeFileSync(join(outputDirectory, `pair-${round + 2}.json`), JSON.stringify(record, null, 2), { flag: "wx" })
    process.stdout.write(JSON.stringify(record) + "\n")
  }
  writeFileSync(
    join(outputDirectory, "complete.json"),
    JSON.stringify(
      {
        status: "passed",
        scope: "fixed-refinement diagnostic; not natural generation or acceptance timings",
        timedBoundary: "full refinement+full readout+sampler completion+packed statistics readback",
        fixedInputs: true,
        warmupPairs: 2,
        measuredPairsPerOrder: 11,
        seed: replay.request.seed,
        replayManifestSha256: sha(replayBytes),
        initializedStateSha256: sha(readFileSync(initializedPath)),
        sourceSha256: sha(Buffer.from(ownSource)),
        samplerBuilderSha256: sha(Buffer.from(expectedBuilder)),
        records
      },
      null,
      2
    ),
    { flag: "wx" }
  )
}))

await Effect.runPromise(main.pipe(Effect.provide(BackendCuda.layer())))

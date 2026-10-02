/** Private combined99 diagnostic. Root alone schedules GPU execution. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Runtime, Safetensors, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/models"
import { Effect } from "effect"
import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { performance } from "node:perf_hooks"
import { fileURLToPath } from "node:url"
import { forkRequestRng99 } from "../../../backend-cuda/src/internal/requestRng99.ts"
import { loadManifest } from "../../../bench/diffusion-gemma/common.ts"
import { validateReplayManifest } from "../../../bench/diffusion-gemma/controlled-replay-validation.ts"
import { preloadReplay, type ReplayManifest } from "../../../bench/diffusion-gemma/controlled-replay.ts"
import * as Statistics from "../../../models/src/internal/diffusionGemmaStatistics.ts"

const [checkpoint, initializedPath, caseDirectory, outputDirectory, extra] = process.argv.slice(2)
assert(checkpoint && initializedPath && caseDirectory && outputDirectory && extra === undefined)
assert(process.env.MANIFEST)
const manifest = loadManifest(process.env.MANIFEST)
const config = manifest.generation
assert.equal(config.canvasLength, 256)
assert.equal(config.stabilityThreshold, 1)
const topology = "fused"
for (
  const flag of [
    "GRAPHS",
    "OVERLAP_PRIMARY_GRAPHS",
    "WHOLE_READ71",
    "EXPERT_SEQUENCE_GRAPHS",
    "EXPERT_BRANCH_GRAPHS",
    "EXPERT_PAIR_GRAPH61"
  ]
) {
  assert.equal(process.env[`EFFECT_TORCH_CUDA_${flag}`], "0")
}
const replayBytes = readFileSync(join(caseDirectory, "manifest.json"))
// SAFETY: all replay fields are validated before use.
const replay = JSON.parse(replayBytes.toString()) as ReplayManifest
validateReplayManifest(replay, {
  promptTokenIds: replay.request.promptTokenIds,
  seed: replay.request.seed,
  maxNewTokens: 64
})
assert([32, 128].includes(replay.request.promptTokenIds.length))
const sha = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex")
assert.equal(sha(readFileSync(initializedPath)), manifest.model.initializedStateSha256)
mkdirSync(outputDirectory)
const ownSource = readFileSync(fileURLToPath(import.meta.url), "utf8")
const modelSource = readFileSync(
  fileURLToPath(new URL("../../../models/src/DiffusionGemma.ts", import.meta.url)),
  "utf8"
)
const start = modelSource.indexOf("const generationStatisticsWithDeviceNoise = (")
const end = modelSource.indexOf("\n/**", start)
assert(start > 0 && end > start)
const copiedBuilder = modelSource.slice(start, end).replace(
  "const generationStatisticsWithDeviceNoise = (",
  "const deviceStatistics = ("
)
assert(ownSource.includes(copiedBuilder), "sampler builder differs from guarded model")
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
  const runtime = yield* Runtime.Runtime
  const loaded = yield* Effect.acquireRelease(
    DiffusionGemma.loadCheckpoint(checkpoint),
    (x) => Tensor.clearAll(x.ownedParameters),
    { interruptible: true }
  )
  const initialized = yield* Effect.acquireRelease(
    Safetensors.load(initializedPath),
    (x) => Tensor.clearAll(Object.values(x)),
    { interruptible: true }
  )
  const model = DiffusionGemma.fromTensors(loaded, { ...loaded.tensors, ...initialized })
  const width = 256
  const input = (slot: number, shape: ReadonlyArray<number>, dtype: Tensor.DType) =>
    Tensor.zeros(shape, { dtype }).pipe(Effect.flatMap((x) => Tensor.makeInput(slot, x)))
  const state: Runtime.DecodeStateRequest = {
    maxTokens: manifest.deployment.maxTokens,
    blockSize: 16,
    kvDtype: model.definition.dtype,
    batch: 1,
    access: "ReadOnly",
    currentBlockAttention: "Bidirectional"
  }
  const tokens = yield* input(0, [1, width], "u32")
  const positions = yield* input(1, [1, width], "u32")
  const previous = yield* input(2, [1, width, model.definition.vocabSize], model.definition.predictionDtype)
  const initialHidden = yield* model.definition.denoise(model.parameters, tokens, positions, { _tag: "Initial" })
  const refinementHidden = yield* model.definition.denoise(model.parameters, tokens, positions, {
    _tag: "Refinement",
    logits: previous
  })
  const initialLogits = yield* model.definition.readout(model.parameters, initialHidden, { _tag: "Full" })
  const refinementLogits = yield* model.definition.readout(model.parameters, refinementHidden, { _tag: "Full" })
  const initial = yield* Tensor.compileDecodeProgram([initialLogits], state, { constantWeights: true })
  const refinement = yield* Tensor.compileDecodeProgram([refinementLogits], state, { constantWeights: true })
  const initialTemperature = yield* input(2, [], "f32")
  const refinementTemperature = yield* input(3, [], "f32")
  const combine = (logits: Tensor.Any, temperature: Tensor.Any) =>
    Effect.gen(function*() {
      const statistics = yield* deviceStatistics(logits, temperature, model.definition.predictionDtype)
      const roots = yield* Statistics.pack(statistics, width, model.definition.vocabSize)
      assert.equal(roots.length, 2)
      // Root the softcapped logits to keep the temperature division outside its
      // elementwise region, preserving the existing sampler83 fusion boundary.
      return yield* Tensor.compileDecodeProgram([...roots, logits], state, { constantWeights: true, randomSeed: 0 })
    })
  const combinedInitial = yield* combine(initialLogits, initialTemperature)
  const combinedRefinement = yield* combine(refinementLogits, refinementTemperature)
  assert(Runtime.sameDecodeStateSchema(initial, combinedInitial))
  assert(Runtime.sameDecodeStateSchema(initial, combinedRefinement))
  const encoders = new Map<number, Tensor.DecodeProgram>()
  for (const size of [64, 256]) {
    const t = yield* input(0, [1, size], "u32")
    const p = yield* input(1, [1, size], "u32")
    const program = yield* Tensor.compileDecodeProgram([yield* model.definition.encode(model.parameters, t, p)], {
      ...state,
      access: "Append",
      currentBlockAttention: "Causal"
    }, { constantWeights: true })
    assert(Runtime.sameDecodeStateSchema(initial, program))
    encoders.set(size, program)
  }
  assert(Runtime.sameDecodeStateSchema(initial, refinement))
  writeFileSync(
    join(outputDirectory, "compile-coverage.json"),
    JSON.stringify(
      {
        scope:
          "Compiled program metadata before admission assertions; combined native fork independently requires seed0/source ordinal0",
        programs: Object.fromEntries([
          ["initial", initial],
          ["refinement", refinement],
          ["combinedInitial", combinedInitial],
          ["combinedRefinement", combinedRefinement],
          ["encoder64", encoders.get(64)!],
          ["encoder256", encoders.get(256)!]
        ].map(([name, program]) => [name, program]))
      },
      null,
      2
    ) + "\n",
    { flag: "wx" }
  )
  const coverage: Record<string, Runtime.ExecutableDiagnostics> = {}
  for (
    const [name, program, projections] of [
      ["initial", initial, 55],
      ["refinement", refinement, 56],
      ["combinedInitial", combinedInitial, 55],
      ["combinedRefinement", combinedRefinement, 56],
      [
        "encoder64",
        encoders.get(64)!,
        0
      ],
      ["encoder256", encoders.get(256)!, 55]
    ] as const
  ) {
    const d = program.handle.diagnostics
    const count = (kind: string) => d.instructions.find((x) => x.kind === kind)?.count ?? 0
    assert.equal(count("packed_projection77"), projections)
    assert.equal(count("fused_moe75"), 30)
    assert.equal(count("kv_stepwise_bf16_gemm_active_rows"), 30)
    coverage[name] = d
  }
  const pool = yield* Tensor.makeKvPoolFromSchema(initial)
  const logits = yield* input(0, [1, width, model.definition.vocabSize], "f32")
  const temperature = yield* Tensor.makeScalarInput(1, "f32")
  const roots = yield* Statistics.pack(
    yield* deviceStatistics(logits, temperature, model.definition.predictionDtype),
    width,
    model.definition.vocabSize
  )
  assert.equal(roots.length, 2)
  const feedbackBank = yield* preloadReplay(caseDirectory, replay)
  for (const program of [combinedInitial, combinedRefinement]) {
    assert.equal(program.outputs.length, 3)
    assert.equal(program.handle.diagnostics.instructions.find((x) => x.kind === "et_random_sampler83_f32")?.count, 1)
  }
  const execute = (arm: "separate" | "combined", quality: boolean) =>
    Effect.scoped(Effect.gen(function*() {
      const own = <A extends ReadonlyArray<Tensor.Concrete>>(
        effect: Effect.Effect<A, Tensor.TensorError | Runtime.BackendError, Runtime.Runtime>
      ) => Effect.acquireRelease(effect, Tensor.clearAll, { interruptible: true })
      const began = performance.now()
      // Control compiles its request-private sampler; combined templates share
      // one fresh request seed/counter across their initial/refinement wrappers.
      const sampler = arm === "separate"
        ? yield* Tensor.freezeProgram(roots, { randomSeed: replay.request.seed })
        : undefined
      const family = arm === "combined"
        ? yield* forkRequestRng99(runtime, {
          initial: combinedInitial.handle,
          refinement: combinedRefinement.handle,
          seed: replay.request.seed
        })
        : undefined
      if (sampler !== undefined) {
        assert.equal(
          sampler.handle.diagnostics.instructions.find((x) => x.kind === "et_random_sampler83_f32")?.count,
          1
        )
        coverage[arm] = sampler.handle.diagnostics
      }
      const random = yield* DiffusionGemma.generationRandom(replay.request.seed)
      const prompt = Uint32Array.from(replay.request.promptTokenIds)
      const sequence = yield* Effect.acquireRelease(
        Tensor.makeKvSequence(pool),
        (x) => Effect.orDie(Tensor.releaseKvSequence(x)),
        { interruptible: true }
      )
      const promptWidth = prompt.length <= 64 ? 64 : 256
      const padded = new Uint32Array(promptWidth)
      padded.set(prompt)
      const prefill = yield* own(
        Tensor.runDecodeProgram(
          encoders.get(promptWidth)!,
          [
            yield* Tensor.fromTypedArray(padded, [1, promptWidth]),
            yield* Tensor.fromTypedArray(
              Uint32Array.from({ length: promptWidth }, (_, i) => i),
              [1, promptWidth]
            )
          ],
          sequence,
          Array.from(prompt)
        )
      )
      yield* Tensor.clearAll(prefill)
      const prefix = yield* Effect.acquireRelease(
        Tensor.snapshotKvSequence(sequence),
        (x) => Effect.orDie(Tensor.releaseKvPrefix(x)),
        { interruptible: true }
      )
      // Shared fixed positions and host canonical body bindings isolate the dispatch
      // boundary. Neither arm includes a position-cache advantage.
      const [positionTensor] = yield* own(
        Tensor.compute([
          yield* Tensor.fromTypedArray(
            Uint32Array.from({ length: width }, (_, i) => prompt.length + i),
            [1, width]
          )
        ])
      )
      const initialDraw = random.canvas(width, model.definition.vocabSize)
      let history: ReadonlyArray<Uint32Array> = []
      let canvas = Uint32Array.from(replay.steps[0]!.canvasTokenIds)
      let draft = canvas
      let oldOutputs: Array<Tensor.Concrete> = []
      let lastStats: Float32Array | undefined
      const witnesses: Array<unknown> = []
      const steps: Array<{ step: number; nativeDone: boolean }> = []
      for (let step = 0; step < replay.steps.length; step++) {
        const record = replay.steps[step]!
        const program = step === 0 ? initial : refinement
        const hostCanvas = Uint32Array.from(record.canvasTokenIds)
        const bindings: Array<Tensor.Concrete> = [positionTensor!]
        if (step > 0) {
          assert(record.feedbackInput !== undefined && record.feedbackInput !== null)
          const feedback = feedbackBank.get(record.feedbackInput.file)
          assert(feedback !== undefined && Tensor.isTensor(feedback))
          bindings.push(feedback)
        }
        const randomCanvas = random.canvas(width, model.definition.vocabSize)
        let outputs: Array<Tensor.Concrete>
        let values: Float32Array
        // Both arms materialize the actual host canvas inside each invocation;
        // the combined arm batches its real temperature upload in the same call.
        const hostInputs = [yield* Tensor.fromTypedArray(hostCanvas, [1, width])]
        if (arm === "combined") hostInputs.push(yield* Tensor.full([], record.temperature))
        const uploaded = yield* own(Tensor.compute(hostInputs))
        const [canvasTensor, temperatureTensor] = uploaded
        if (arm === "combined") {
          assert(family !== undefined)
          const template = step === 0 ? combinedInitial : combinedRefinement
          const handle = step === 0 ? family[0] : family[1]
          assert(temperatureTensor !== undefined)
          outputs = yield* own(
            Tensor.runReadOnlyDecodeProgram(
              { ...template, handle },
              [canvasTensor!, ...bindings, temperatureTensor!],
              prefix,
              width
            )
          )
          assert.equal(outputs.length, 3)
          yield* Tensor.clear(outputs[2]!)
          outputs = outputs.slice(0, 2)
        } else {
          assert(sampler !== undefined)
          const body = yield* own(Tensor.runReadOnlyDecodeProgram(program, [canvasTensor!, ...bindings], prefix, width))
          outputs = yield* own(Tensor.runProgram(sampler, [body[0]!], [record.temperature]))
          yield* Tensor.clearAll(body)
        }
        yield* Tensor.clearAll(uploaded)
        const read = yield* Tensor.toTypedArray(outputs[1]!)
        assert(read instanceof Float32Array)
        values = read
        assert(values instanceof Float32Array && values.length === width * 4 + 1)
        lastStats = values
        const [sampled, argmax, entropy, order, mean] = Statistics.unpack(Array.from(values), width)
        const prediction: DiffusionGemma.GenerationPrediction = {
          sampledTokens: Uint32Array.from(sampled),
          argmaxTokens: Uint32Array.from(argmax),
          tokenEntropy: Float32Array.from(entropy),
          entropyOrder: Uint32Array.from(order),
          meanEntropy: mean[0]!
        }
        const sampledCanvas = DiffusionGemma.sampleGenerationCanvas(
          canvas,
          prediction,
          randomCanvas,
          config.entropyBound
        )
        const stopped = DiffusionGemma.stopGeneration({ history }, prediction, 1, config.confidenceThreshold)
        history = stopped.state.history
        steps.push({ step, nativeDone: stopped.done })
        if (quality) {
          const feedback = yield* Tensor.toTypedArray(outputs[0]!)
          witnesses.push({
            step,
            canvas: Array.from(sampledCanvas.canvas),
            draft: Array.from(prediction.argmaxTokens),
            done: stopped.done,
            feedbackSha256: sha(new Uint8Array(feedback.buffer, feedback.byteOffset, feedback.byteLength)),
            statsSha256: sha(new Uint8Array(values.buffer, values.byteOffset, values.byteLength)),
            noiseSha256: sha(new Uint8Array(randomCanvas.buffer, randomCanvas.byteOffset, randomCanvas.byteLength))
          })
        }
        yield* Tensor.clearAll(oldOutputs)
        oldOutputs = outputs
        // Actual model, sampler, policy and history ran before canonical feedback
        // or canvas is selected for the following model invocation.
        canvas = Uint32Array.from(record.postCanvasTokenIds)
        draft = Uint32Array.from(record.draftTokenIds)
        assert.equal(record.done, step + 1 === replay.steps.length)
      }
      assert(lastStats !== undefined)
      const completed = DiffusionGemma.finishGenerationCanvas(draft, config.eosTokenIds, config.padTokenId)
      const committed = yield* Effect.acquireRelease(
        Tensor.forkKvPrefix(prefix),
        (x) => Effect.orDie(Tensor.releaseKvSequence(x)),
        { interruptible: true }
      )
      const commit = yield* own(
        Tensor.runDecodeProgram(
          encoders.get(width)!,
          [yield* Tensor.fromTypedArray(completed.tokens, [1, width]), positionTensor!],
          committed,
          Array.from(completed.tokens)
        )
      )
      yield* Tensor.clearAll(commit)
      const elapsedMilliseconds = performance.now() - began
      return {
        arm,
        quality,
        elapsedMilliseconds,
        refinements: steps.length,
        steps,
        tokens: Array.from(completed.tokens.slice(0, 64)),
        witnesses,
        initialDrawSha256: sha(new Uint8Array(initialDraw.buffer, initialDraw.byteOffset, initialDraw.byteLength)),
        finalStatsSha256: sha(new Uint8Array(lastStats.buffer, lastStats.byteOffset, lastStats.byteLength))
      }
    }))
  const separateQuality = yield* execute("separate", true)
  const combinedQuality = yield* execute("combined", true)
  assert.deepEqual(combinedQuality.witnesses, separateQuality.witnesses)
  assert.deepEqual(combinedQuality.tokens, replay.outputTokenIds)
  assert.equal(combinedQuality.initialDrawSha256, separateQuality.initialDrawSha256)
  writeFileSync(
    join(outputDirectory, "quality.json"),
    JSON.stringify({ separateQuality, combinedQuality }, null, 2) + "\n",
    { flag: "wx" }
  )
  const pairs = []
  for (let round = -1; round < 4; round++) {
    const order = round % 2 === 0 ? ["separate", "combined"] as const : ["combined", "separate"] as const
    const first = yield* execute(order[0], false)
    const second = yield* execute(order[1], false)
    assert.deepEqual(first.tokens, second.tokens)
    assert.deepEqual(first.steps, second.steps)
    assert.equal(first.initialDrawSha256, second.initialDrawSha256)
    assert.equal(first.finalStatsSha256, second.finalStatsSha256)
    pairs.push({ round, order, first, second })
  }
  return {
    status: "passed",
    topology,
    scope: "fixedcase paired host-boundary diagnostic; not acceptance timing",
    replayManifestSha256: sha(replayBytes),
    sourceSha256: sha(Buffer.from(ownSource)),
    samplerBuilderSha256: sha(Buffer.from(copiedBuilder)),
    controls:
      "Control uses current fused93 body/readout plus private scalar83 sampler; combined templates compile once with seed0 and fork one shared request seed/counter for initial/refinement inside timing; combined pays actual tensor-temperature upload; same sampler builder, actual host canvas upload, prefill/positions/canonical inputs/CPU draws/host policy/terminal commit; packed statistics readback each step; full feedback hashes quality only; graph capture disabled",
    coverage,
    pairs
  }
}))
const result = await Effect.runPromise(main.pipe(Effect.provide(BackendCuda.layer())))
writeFileSync(join(outputDirectory, "complete.json"), JSON.stringify(result, null, 2) + "\n", { flag: "wx" })
process.stdout.write(JSON.stringify(result) + "\n")

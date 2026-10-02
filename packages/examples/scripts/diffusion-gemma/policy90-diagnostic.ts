/** Private paired policy-placement diagnostic. Root owns the GPU queue. */
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
import { loadManifest } from "../../../bench/diffusion-gemma/common.ts"
import { validateReplayManifest } from "../../../bench/diffusion-gemma/controlled-replay-validation.ts"
import {
  preloadReplay,
  type ReplayManifest,
  type ReplayStep
} from "../../../bench/diffusion-gemma/controlled-replay.ts"
import * as Statistics from "../../../models/src/internal/diffusionGemmaStatistics.ts"
import { gpuTail } from "./policy90.ts"

const [mode, checkpoint, initializedPath, caseDirectory, outputDirectory, extra] = process.argv.slice(2)
assert(
  (mode === "natural" || mode === "replay") && checkpoint && initializedPath && caseDirectory && outputDirectory &&
    extra === undefined,
  "usage: policy90-diagnostic.ts natural|replay CHECKPOINT INITIALIZED_STATE REPLAY_CASE NEW_OUTPUT"
)
assert(process.env.MANIFEST, "MANIFEST must pin the matched initialized state and generation settings")
const manifest = loadManifest(process.env.MANIFEST)
const config = manifest.generation
assert.equal(config.canvasLength, 256)
assert.equal(config.stabilityThreshold, 1, "diagnostic only supports the matched ST=1 configuration")
const replayBytes = readFileSync(join(caseDirectory, "manifest.json"))
// SAFETY: validateReplayManifest checks every field before any use below.
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
assert(ownSource.includes(copiedBuilder), "sampler copy differs from the guarded current model source")

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
    (x) => Tensor.clearAll(x.ownedParameters),
    { interruptible: true }
  )
  const initialized = yield* Effect.acquireRelease(
    Safetensors.load(initializedPath),
    (x) => Tensor.clearAll(Object.values(x)),
    { interruptible: true }
  )
  const model = DiffusionGemma.fromTensors(loaded, { ...loaded.tensors, ...initialized })
  const width = config.canvasLength
  assert.equal(model.definition.canvasLength, width)
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
  const initial = yield* Tensor.compileDecodeProgram(
    [
      yield* model.definition.denoise(model.parameters, tokens, positions, { _tag: "Initial" })
    ],
    state,
    { constantWeights: true }
  )
  const refinement = yield* Tensor.compileDecodeProgram(
    [
      yield* model.definition.denoise(model.parameters, tokens, positions, { _tag: "Refinement", logits: previous })
    ],
    state,
    { constantWeights: true }
  )
  const hidden = yield* input(0, initial.outputs[0]!.shape, initial.outputs[0]!.dtype)
  const full = yield* Tensor.freezeProgram([
    yield* model.definition.readout(model.parameters, hidden, { _tag: "Full" })
  ], { constantWeights: true })
  const encoders = new Map<number, Tensor.DecodeProgram>()
  for (const size of [64, 256]) {
    const t = yield* input(0, [1, size], "u32")
    const p = yield* input(1, [1, size], "u32")
    const program = yield* Tensor.compileDecodeProgram(
      [
        yield* model.definition.encode(model.parameters, t, p)
      ],
      { ...state, access: "Append", currentBlockAttention: "Causal" },
      { constantWeights: true }
    )
    assert(Runtime.sameDecodeStateSchema(initial, program))
    encoders.set(size, program)
  }
  assert(Runtime.sameDecodeStateSchema(initial, refinement))
  // Exact same static coverage observed for current85/current87 at these
  // four geometries. Attention82 retains the native attention instruction
  // name; its BF16 IO and PTX are established by root's pinned addon/flags.
  const modelCoverage: Record<string, Runtime.ExecutableDiagnostics> = {}
  for (
    const [name, program, projections] of [
      ["initial", initial, 55],
      ["refinement", refinement, 56],
      ["encoder64", encoders.get(64)!, 0],
      ["encoder256", encoders.get(256)!, 55]
    ] as const
  ) {
    const diagnostics = program.handle.diagnostics
    const count = (kind: string) => diagnostics.instructions.find((x) => x.kind === kind)?.count ?? 0
    assert.equal(count("packed_projection77"), projections, `${name}: projection77 coverage changed`)
    assert.equal(count("fused_moe75"), 30, `${name}: MoE75 coverage changed`)
    assert.equal(count("kv_stepwise_bf16_gemm_active_rows"), 30, `${name}: native attention coverage changed`)
    modelCoverage[name] = diagnostics
  }
  const pool = yield* Tensor.makeKvPoolFromSchema(initial)
  const logits = yield* input(0, [1, width, model.definition.vocabSize], "f32")
  const temperature = yield* Tensor.makeScalarInput(1, "f32")
  const statistics = yield* deviceStatistics(logits, temperature, model.definition.predictionDtype)
  const packed = yield* Statistics.pack(statistics, width, model.definition.vocabSize)
  assert.equal(packed.length, 2)
  const noise = yield* input(2, [1, width], "u32")
  const priorArgmax = yield* input(3, [1, width], "u32")
  const available = yield* Tensor.makeScalarInput(4, "u8")
  const tail = yield* gpuTail(
    statistics,
    noise,
    priorArgmax,
    available,
    config.entropyBound,
    config.confidenceThreshold
  )
  const roots = { host: packed, gpu: [...packed, ...tail] }
  const feedbackBank = mode === "replay" ? yield* preloadReplay(caseDirectory, replay) : new Map<string, Tensor.Any>()
  const canonical = mode === "replay" ?
    yield* Effect.acquireRelease(
      Tensor.compute(
        yield* Effect.forEach(
          replay.steps,
          (step) => Tensor.fromTypedArray(Uint32Array.from(step.canvasTokenIds), [1, width])
        )
      ),
      Tensor.clearAll,
      { interruptible: true }
    ) :
    []
  const coverage: Partial<Record<"host" | "gpu", Runtime.ExecutableDiagnostics>> = {}
  const execute = (arm: "host" | "gpu", quality: boolean) =>
    Effect.scoped(Effect.gen(function*() {
      const own = <A extends ReadonlyArray<Tensor.Concrete>>(
        effect: Effect.Effect<A, Tensor.TensorError, Runtime.Runtime>
      ) => Effect.acquireRelease(effect, Tensor.clearAll, { interruptible: true })
      const began = performance.now()
      // Exactly one seeded sampler executable per request, including initial and
      // refinement reads. Freeze cost is inside both timed boundaries.
      const sampler = yield* Tensor.freezeProgram(roots[arm], { randomSeed: replay.request.seed })
      const instructions = sampler.handle.diagnostics.instructions
      const fused = instructions.find((x) => x.kind === "et_random_sampler83_f32")?.count ?? 0
      assert.equal(fused, 1, "policy roots or scalar input disabled current87 sampler83 fusion")
      coverage[arm] = sampler.handle.diagnostics
      const random = yield* DiffusionGemma.generationRandom(replay.request.seed)
      const prompt = Uint32Array.from(replay.request.promptTokenIds)
      const sequence = yield* Effect.acquireRelease(Tensor.makeKvSequence(pool), (x) =>
        Effect.orDie(Tensor.releaseKvSequence(x)), { interruptible: true })
      const promptWidth = prompt.length <= 64 ? 64 : 256
      const padded = new Uint32Array(promptWidth)
      padded.set(prompt)
      const prefill = yield* own(Tensor.runDecodeProgram(
        encoders.get(promptWidth)!,
        [
          yield* Tensor.fromTypedArray(padded, [1, promptWidth]),
          yield* Tensor.fromTypedArray(
            Uint32Array.from({ length: promptWidth }, (_, i) =>
              i),
            [1, promptWidth]
          )
        ],
        sequence,
        Array.from(prompt)
      ))
      yield* Tensor.clearAll(prefill)
      const prefix = yield* Effect.acquireRelease(Tensor.snapshotKvSequence(sequence), (x) =>
        Effect.orDie(Tensor.releaseKvPrefix(x)), { interruptible: true })
      // Positions are concrete once per prefix in BOTH arms; this diagnostic
      // does not attribute the independently parked position-cache experiment.
      const [positionTensor] = yield* own(Tensor.compute([
        yield* Tensor.fromTypedArray(
          Uint32Array.from({ length: width }, (_, i) =>
            prompt.length + i),
          [1, width]
        )
      ]))
      const drawn = random.canvas(width, model.definition.vocabSize)
      let canvas = mode === "replay" ? Uint32Array.from(replay.steps[0]!.canvasTokenIds) : drawn
      let gpuCanvas: Tensor.Concrete | undefined
      let gpuHistory: Tensor.Concrete | undefined
      if (arm === "gpu") {
        const owned = yield* own(Tensor.compute([
          yield* Tensor.fromTypedArray(canvas, [1, width]),
          yield* Tensor.zeros([1, width], { dtype: "u32" })
        ]))
        gpuCanvas = owned[0]!
        gpuHistory = owned[1]!
      }
      let feedback: Tensor.Any | undefined
      let oldOutputs: Array<Tensor.Concrete> = []
      let history: ReadonlyArray<Uint32Array> = []
      let draft = canvas
      let lastStats: Float32Array | undefined
      let finalOutputs: Array<Tensor.Concrete> = []
      const witnesses: Array<unknown> = []
      let refinements = 0
      let statisticsBytes = 0
      let boundaryMs = 0
      let lastSamplerEnd = performance.now()
      for (let step = 0; step < config.maxSteps; step++) {
        const record: ReplayStep | undefined = mode === "replay" ? replay.steps[step] : undefined
        if (mode === "replay") {
          assert(record !== undefined, "replay exceeded canonical work")
        }
        const currentCanvas = arm === "gpu"
          ? mode === "replay" ? canonical[step]! : gpuCanvas!
          : (yield* own(Tensor.compute([yield* Tensor.fromTypedArray(canvas, [1, width])])))[0]!
        const bodyBindings: Array<Tensor.Any> = [currentCanvas, positionTensor!]
        if (step > 0) {
          const file = record?.feedbackInput
          bodyBindings.push(file ? feedbackBank.get(file.file)! : feedback!)
        }
        const bodyStart = performance.now()
        if (step > 0) {
          boundaryMs += bodyStart - lastSamplerEnd
        }
        const body = yield* own(
          Tensor.runReadOnlyDecodeProgram(step === 0 ? initial : refinement, bodyBindings, prefix, width)
        )
        if (arm === "host") {
          yield* Tensor.clear(currentCanvas)
        }
        const head = yield* own(Tensor.runProgram(full, [body[0]!]))
        yield* Tensor.clearAll(body)
        const temp: number = record?.temperature ??
          DiffusionGemma.generationTemperature(
            config.minTemperature,
            config.maxTemperature,
            config.maxSteps,
            config.maxSteps - step
          )
        // The shared Mulberry canvas stream draws a whole canvas even on terminal
        // reads. It remains CPU-generated and uploaded, matching the vLLM overlay.
        const randomCanvas = random.canvas(width, model.definition.vocabSize)
        let noiseTensor: Tensor.Concrete | undefined
        if (arm === "gpu") {
          ;[noiseTensor] = yield* own(Tensor.compute([yield* Tensor.fromTypedArray(randomCanvas, [1, width])]))
        }
        const outputs: Array<Tensor.Concrete> = yield* own(
          Tensor.runProgram(
            sampler,
            arm === "gpu" ? [head[0]!, noiseTensor!, gpuHistory!] : [head[0]!],
            arm === "gpu" ? [temp, step > 0 ? 1 : 0] : [temp]
          )
        )
        lastSamplerEnd = performance.now()
        yield* Tensor.clearAll(head)
        if (noiseTensor !== undefined) {
          yield* Tensor.clear(noiseTensor)
        }
        let nativeDone: boolean
        if (arm === "host" || quality) {
          const values = yield* Tensor.toTypedArray(outputs[1]!)
          assert(values instanceof Float32Array && values.length === width * 4 + 1)
          statisticsBytes += values.byteLength
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
          canvas = sampledCanvas.canvas
          draft = prediction.argmaxTokens.slice()
          nativeDone = stopped.done
          history = stopped.state.history
          if (arm === "gpu") {
            assert.deepEqual(yield* Tensor.toTypedArray(outputs[2]!), canvas)
            assert.deepEqual(yield* Tensor.toTypedArray(outputs[3]!), draft)
            assert.deepEqual(yield* Tensor.toTypedArray(outputs[4]!), Uint8Array.of(nativeDone ? 1 : 0))
          }
          if (quality) {
            witnesses.push({
              step,
              canvas: Array.from(canvas),
              draft: Array.from(draft),
              done: nativeDone,
              statisticsSha256: sha(new Uint8Array(values.buffer, values.byteOffset, values.byteLength))
            })
          }
        } else {
          const flag = yield* Tensor.toTypedArray(outputs[4]!)
          assert(flag instanceof Uint8Array && flag.length === 1)
          statisticsBytes += flag.byteLength
          nativeDone = flag[0] !== 0
        }
        if (oldOutputs.length) {
          yield* Tensor.clearAll(oldOutputs)
        }
        oldOutputs = outputs
        finalOutputs = outputs
        feedback = outputs[0]!
        if (arm === "gpu") {
          gpuCanvas = outputs[2]!
          gpuHistory = outputs[3]!
        }
        refinements++
        if (record !== undefined) {
          canvas = Uint32Array.from(record.postCanvasTokenIds)
          draft = Uint32Array.from(record.draftTokenIds)
        }
        if ((record?.done ?? nativeDone) || step + 1 === config.maxSteps) {
          break
        }
      }
      if (mode === "replay") {
        assert.equal(refinements, replay.steps.length)
      }
      if (arm === "gpu" && !quality) {
        // Include the actual final draft transfer in BOTH timed modes. Replay
        // still commits the canonical draft, while preserving the same native
        // output boundary that production natural generation requires.
        const value = yield* Tensor.toTypedArray(finalOutputs[3]!)
        assert(value instanceof Uint32Array)
        if (mode === "natural") {
          draft = value
        }
      }
      const completed = DiffusionGemma.finishGenerationCanvas(draft, config.eosTokenIds, config.padTokenId)
      // Same mandatory terminal full-canvas encoder/KV commit as matched replay.
      const committed = yield* Effect.acquireRelease(Tensor.forkKvPrefix(prefix), (x) =>
        Effect.orDie(Tensor.releaseKvSequence(x)), { interruptible: true })
      const commit = yield* own(Tensor.runDecodeProgram(
        encoders.get(width)!,
        [
          yield* Tensor.fromTypedArray(completed.tokens, [1, width]),
          positionTensor!
        ],
        committed,
        Array.from(completed.tokens)
      ))
      yield* Tensor.clearAll(commit)
      const elapsedMilliseconds = performance.now() - began
      // Preserve one actual sampler witness outside timing. Native stochastic
      // work is never replaced by replay; only following model inputs are.
      if (arm === "gpu" && !quality) {
        const value = yield* Tensor.toTypedArray(finalOutputs[1]!)
        assert(value instanceof Float32Array)
        lastStats = value
      }
      assert(lastStats !== undefined)
      return {
        arm,
        quality,
        elapsedMilliseconds,
        refinements,
        tokens: Array.from(completed.tokens.slice(0, 64)),
        statisticsBytes,
        boundaryMs,
        finalStatisticsSha256: sha(new Uint8Array(lastStats.buffer, lastStats.byteOffset, lastStats.byteLength)),
        witnesses
      }
    }))
  const hostQuality = yield* execute("host", true)
  const gpuQuality = yield* execute("gpu", true)
  assert.deepEqual(gpuQuality.witnesses, hostQuality.witnesses, "actual policy/statistics witness mismatch")
  assert.deepEqual(gpuQuality.tokens, hostQuality.tokens)
  assert.equal(gpuQuality.refinements, hostQuality.refinements)
  if (mode === "replay") assert.deepEqual(gpuQuality.tokens, replay.outputTokenIds)
  writeFileSync(join(outputDirectory, "quality.json"), JSON.stringify({ hostQuality, gpuQuality }, null, 2) + "\n", {
    flag: "wx"
  })
  const pairs = []
  for (let round = -1; round < 4; round++) {
    const order = round % 2 === 0 ? ["host", "gpu"] as const : ["gpu", "host"] as const
    const first = yield* execute(order[0], false)
    const second = yield* execute(order[1], false)
    assert.deepEqual(first.tokens, second.tokens)
    assert.equal(first.refinements, second.refinements)
    assert.equal(first.finalStatisticsSha256, second.finalStatisticsSha256)
    pairs.push({ round, order, first, second })
  }
  return {
    status: "passed",
    mode,
    scope: "paired standalone policy-placement diagnostic; not acceptance timing",
    promptTokens: replay.request.promptTokenIds.length,
    seed: replay.request.seed,
    replayManifestSha256: sha(replayBytes),
    samplerBuilderSha256: sha(Buffer.from(copiedBuilder)),
    sourceSha256: sha(Buffer.from(ownSource)),
    timedBoundary:
      "fresh sampler compilation + prefill + separate body/readout/sampler + actual policy + output + terminal encoder commit",
    commonControls:
      "positions cached once per prefix; exact host temperature scalar; CPU canvas RNG retained; no callbacks; ST1; sampler83 count1 required both arms",
    coverage,
    modelCoverage,
    pairs
  }
}))
const result = await Effect.runPromise(main.pipe(Effect.provide(BackendCuda.layer())))
writeFileSync(join(outputDirectory, "complete.json"), JSON.stringify(result, null, 2) + "\n", { flag: "wx" })
process.stdout.write(JSON.stringify(result) + "\n")

/** Full selected BF16 checkpoint against a saved official CUDA oracle, with identical inputs. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Model, Safetensors, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/core/models"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"

const Inputs = Schema.Struct({
  case: Schema.Struct({ promptIds: Schema.Array(Schema.Int), labelIds: Schema.Array(Schema.Int), slot: Schema.Int }),
  reads: Schema.Array(Schema.Struct({ canvas_ids: Schema.Array(Schema.Int), index: Schema.Int }))
})
const Manifest = Schema.Struct({
  model: Schema.String,
  revision: Schema.String,
  inputs_sha256: Schema.String,
  status: Schema.String,
  reads: Schema.Array(Schema.Struct({ file: Schema.String, sha256: Schema.String }))
})
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex")
const compare = (actual: ReadonlyArray<number>, expected: ReadonlyArray<number>) => {
  if (actual.length !== expected.length) throw new Error("oracle/output length mismatch")
  let exact = 0
  let maxAbsoluteError = 0
  let squaredError = 0
  let squaredReference = 0
  let beyondOneBf16Step = 0
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) throw new Error("nonfinite model/oracle output")
    const error = Math.abs(actual[i] - expected[i])
    if (error === 0) exact++
    maxAbsoluteError = Math.max(maxAbsoluteError, error)
    squaredError += error * error
    squaredReference += expected[i] * expected[i]
    if (error > Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(expected[i]))) - 7)) + 2e-6) beyondOneBf16Step++
  }
  return {
    elements: actual.length,
    exact,
    maxAbsoluteError,
    relativeL2: Math.sqrt(squaredError / Math.max(squaredReference, 1e-30)),
    beyondOneBf16Step
  }
}
const probabilities = (values: ReadonlyArray<number>) => {
  const max = Math.max(...values)
  const weights = values.map((n) => Math.exp(n - max))
  const sum = weights.reduce((a, b) => a + b, 0)
  return weights.map((n) => n / sum)
}

const program = Effect.scoped(Effect.gen(function*() {
  const [checkpoint, oracleDirectory, output, requested = "4"] = process.argv.slice(2)
  if (
    checkpoint === undefined || oracleDirectory === undefined || output === undefined || !["1", "4"].includes(requested)
  ) throw new Error("usage: verify-full.ts <checkpoint> <oracle-directory> <new-output-directory> [1|4]")
  if (!(yield* BackendCuda.isAvailable)) throw new Error("CUDA is required for this gate")
  if (existsSync(join(output, "run.json"))) {
    throw new Error("preserve existing run evidence; choose a new output directory")
  }
  mkdirSync(output, { recursive: true })
  const manifest = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Manifest))(
    readFileSync(join(oracleDirectory, "manifest.json"), "utf8")
  )
  if (manifest.revision !== "f7f5b7f5fa82ffc52addd066915886d497f5517b") {
    throw new Error("unexpected oracle model revision")
  }
  const inputsBytes = readFileSync(join(oracleDirectory, "inputs.json"))
  if (hash(inputsBytes) !== manifest.inputs_sha256) throw new Error("oracle inputs hash mismatch")
  const inputs = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Inputs))(inputsBytes.toString("utf8"))
  const config = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Schema.Json))(
    readFileSync(join(checkpoint, "config.json"), "utf8")
  )
  const started = performance.now()
  console.log("Loading selected BF16 text weights")
  const loaded = yield* Effect.acquireRelease(
    DiffusionGemma.loadParameters(join(checkpoint, "model.safetensors.index.json"), config),
    (model) => Tensor.clearAll(model.ownedParameters),
    { interruptible: true }
  )
  const loadMs = performance.now() - started
  let oracleRows: Readonly<Record<string, ReadonlyArray<number>>> = {}
  let capture = false
  let currentLayers: Array<{ layer: number; comparison: ReturnType<typeof compare> }> = []
  const execution = Model.executor(DiffusionGemma.make(loaded), {
    observeLayer: ({ phase, layer, hidden }) =>
      Effect.gen(function*() {
        if (phase === "prefill") {
          console.log(`Prefill layer ${layer + 1}/${loaded.config.text_config.num_hidden_layers}`)
        } else if (capture) {
          const data = yield* Tensor.toNumberArray(hidden)
          const width = loaded.config.text_config.hidden_size
          const row = data.slice(inputs.case.slot * width, (inputs.case.slot + 1) * width)
          const expected = oracleRows[`decoder.layers.${layer}.hidden`]
          if (expected === undefined) throw new Error("missing oracle hidden state")
          currentLayers.push({ layer, comparison: compare(row, expected) })
        }
      })
  })
  const prefillStarted = performance.now()
  const prefix = yield* Effect.acquireRelease(
    execution.prefill(Uint32Array.from(inputs.case.promptIds)),
    Tensor.clearKvPrefix,
    { interruptible: true }
  )
  const read = (canvas: Uint32Array) =>
    Effect.scoped(
      Effect.acquireRelease(execution.read(prefix, canvas, inputs.case.slot), Tensor.clear, { interruptible: true })
        .pipe(
          Effect.flatMap(Tensor.toNumberArray)
        )
    )
  const prefillMs = performance.now() - prefillStarted
  const results = []
  for (const input of inputs.reads.slice(0, Number(requested))) {
    const artifact = manifest.reads[input.index]
    if (artifact === undefined || artifact.file.includes("/") || artifact.file.includes("\\")) {
      throw new Error("invalid reference filename")
    }
    const file = join(oracleDirectory, artifact.file)
    if (hash(readFileSync(file)) !== artifact.sha256) throw new Error("oracle tensor file hash mismatch")
    oracleRows = yield* Effect.acquireUseRelease(Safetensors.load(file), (tensors) =>
      Effect.gen(function*() {
        const entries = yield* Effect.forEach(Object.entries(tensors), ([name, tensor]) =>
          Tensor.toNumberArray(tensor).pipe(Effect.map((values) => [name, values] as const)))
        return Object.fromEntries(entries)
      }), (tensors) => Tensor.clearAll(Object.values(tensors)))
    currentLayers = []
    capture = true
    const canvas = Uint32Array.from(input.canvas_ids)
    const started = performance.now()
    const logits = yield* read(canvas)
    const readMs = performance.now() - started
    capture = false
    const replay = yield* read(canvas)
    const replayExact = logits.length === replay.length && logits.every((value, index) => value === replay[index])
    if (!replayExact) throw new Error("read-only replay changed logits")
    const raw = Buffer.from(Float32Array.from(logits).buffer)
    writeFileSync(join(output, `read-${input.index}.f32`), raw, { flag: "wx" })
    const expected = oracleRows["logits.answer"]
    if (expected === undefined) throw new Error("missing oracle logits")
    const selected = inputs.case.labelIds.map((id) => logits[id])
    const expectedSelected = inputs.case.labelIds.map((id) => expected[id])
    const probability = probabilities(selected)
    const expectedProbability = probabilities(expectedSelected)
    results.push({
      index: input.index,
      readMs,
      replayExact,
      logitsSha256: hash(raw),
      logits: compare(logits, expected),
      layers: currentLayers,
      selected,
      expectedSelected,
      probability,
      expectedProbability,
      maximumProbabilityDifference: Math.max(...probability.map((value, i) => Math.abs(value - expectedProbability[i])))
    })
    writeFileSync(
      join(output, "run.json"),
      JSON.stringify(
        { modelRevision: manifest.revision, loadMs, prefillMs, prefixBytes: prefix.bytes, results },
        null,
        2
      ) + "\n"
    )
    console.log(JSON.stringify(results.at(-1)))
  }
  if (results.some((result) => result.logits.beyondOneBf16Step > 0)) {
    throw new Error("full-model comparison exceeds one BF16 step; evidence saved for investigation")
  }
})).pipe(Effect.provide(BackendCuda.layer()))

NodeRuntime.runMain(program)

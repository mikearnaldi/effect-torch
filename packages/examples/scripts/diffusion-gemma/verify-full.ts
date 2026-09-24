/**
 * Full selected BF16 checkpoint against a saved official CUDA oracle, with identical inputs.
 *
 * pnpm --filter @effect-torch/examples exec tsx scripts/diffusion-gemma/verify-full.ts \
 *   <checkpoint> <oracle-directory> <output-directory> [1|4] [prefill-chunks-csv]
 *
 * EFFECT_TORCH_DIFFUSION_GEMMA_INITIALIZED_STATE selects the configuration-derived
 * manifest and sibling safetensors asset required for pinned RoPE initialization.
 */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Diffusion, Runtime, Safetensors, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/models"
import { NodeRuntime } from "@effect/platform-node"
import { Cause, Effect, Exit, Schema } from "effect"
import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"

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

const InitializedState = Schema.Struct({
  status: Schema.Literal("passed"),
  kind: Schema.Literal("diffusion-gemma-initialized-rope-v1"),
  configSha256: Schema.String,
  file: Schema.Literal("initialized-rope.safetensors"),
  sha256: Schema.String,
  payloadBytes: Schema.Int,
  producerSha256: Schema.String,
  torchVersion: Schema.String,
  torchCommit: Schema.String,
  cpuCapability: Schema.String,
  sourcePins: Schema.Struct({
    model: Schema.Struct({ revision: Schema.Literal("f7f5b7f5fa82ffc52addd066915886d497f5517b") }),
    transformers: Schema.Struct({
      archive_url: Schema.Literal(
        "https://github.com/huggingface/transformers/archive/93ebf6b11127967f2725cf4d012aae55c3654f5a.tar.gz"
      )
    })
  }),
  tensors: Schema.Array(
    Schema.Struct({
      name: Schema.String,
      dtype: Schema.Literal("f32"),
      shape: Schema.Array(Schema.Int),
      sha256: Schema.String
    })
  )
})

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

const summarize = (samplesMs: ReadonlyArray<number>) => {
  const sorted = [...samplesMs].sort((a, b) => a - b)
  const percentile = (p: number) => sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)] ?? null

  return {
    count: samplesMs.length,
    samplesMs,
    minMs: sorted[0] ?? null,
    maxMs: sorted.at(-1) ?? null,
    p50Ms: percentile(0.5),
    p95Ms: percentile(0.95)
  }
}

type Status = "running" | "passed" | "failed" | "interrupted"

interface Measurement {
  phase: string
  readIndex: number | null
  status: Status
  startedAt: string
  elapsedMs: number | null
  error: string | null
}

interface CompiledProgram {
  index: number
  status: Status
  roots: ReadonlyArray<{
    shape: ReadonlyArray<number>
    dtype: Tensor.DType
  }>
  state: Runtime.DecodeStateRequest | null
  options: Runtime.CompileRequest["options"]
  wallMs: number | null
  diagnostics: Runtime.ExecutableDiagnostics | null
  error: string | null
}

interface ReadResult {
  index: number
  readMs: number
  replayMs: number
  selectedMs: number
  replayExact: boolean
  logitsSha256: string
  logits: ReturnType<typeof compare>
  selectedAgreement: ReturnType<typeof compare>
  rows: ReadonlyArray<number>
  labelIds: ReadonlyArray<number>
  selected: ReadonlyArray<number>
  selectedFromFull: ReadonlyArray<number>
  expectedSelected: ReadonlyArray<number>
  probability: ReadonlyArray<number>
  expectedProbability: ReadonlyArray<number>
  maximumProbabilityDifference: number
}

const program = Effect.gen(function*() {
  const [checkpoint, oracleDirectory, output, requested = "4", chunks, extra] = process.argv.slice(2)

  if (
    checkpoint === undefined || oracleDirectory === undefined || output === undefined ||
    !["1", "4"].includes(requested) || extra !== undefined
  ) {
    throw new Error(
      "usage: verify-full.ts <checkpoint> <oracle-directory> <new-output-directory> [1|4] [prefill-chunks-csv]"
    )
  }

  const prefillChunks = chunks === undefined ? [278] : chunks.split(",").map(Number)

  if (
    (chunks !== undefined && !/^\d+(,\d+)*$/.test(chunks)) ||
    prefillChunks.some((width, index) =>
      !Number.isSafeInteger(width) || width <= 0 || width > 0xffff_ffff ||
      (index > 0 && width <= prefillChunks[index - 1]!)
    )
  ) throw new Error("prefill chunks must be ascending distinct positive u32 integers")

  if (existsSync(join(output, "run.json"))) {
    throw new Error("preserve existing run evidence; choose a new output directory")
  }

  mkdirSync(output, { recursive: true })
  const runtime = yield* Runtime.Runtime
  const measurements: Array<Measurement> = []
  const compiledPrograms: Array<CompiledProgram> = []
  const results: Array<ReadResult> = []

  const memory: Array<{
    stage: string
    externalMemoryBytes: number
    at: string
  }> = []

  const concurrentResults: Array<{
    index: number
    isolatedExact: boolean
    comparison: ReturnType<typeof compare>
  }> = []

  interface Report {
    status: Status
    stage: string
    error: string | null
    startedAt: string
    modelRevision: string | null
    manifestSha256: string | null
    inputsSha256: string | null
    execution: string
    requestedReads: number
    canvasLengths: Array<number>
    geometry: {
      prefillChunks: ReadonlyArray<number>
      prefillSource: "default" | "explicit"
      blockSize: number
      maxTokens: number | null
      modelCanvasLength: number | null
      promptTokens: number | null
    }
    checkpointParameters: {
      uniqueCount: number
      uniqueBytes: number
      dtypes: ReadonlyArray<Tensor.DType>
    } | null
    initializedState: {
      manifestSha256: string
      manifest: typeof InitializedState.Type
    } | null
    prefixBytes: number | null
    prefixInspection: {
      cursor: number
      retainedBytes: number
      sharedBytes: number
      copiedBytes: number
      layers: ReadonlyArray<
        {
          layerId: number
          startPosition: number
          kvHeads: number
          headDim: number
          dtype: Tensor.DType
          retainedRows: number
        }
      >
    } | null
    externalMemoryZeroAfterLoad: boolean
    measurements: typeof measurements
    compiledPrograms: typeof compiledPrograms
    memory: typeof memory
    results: typeof results
    concurrent: {
      concurrency: number
      results: typeof concurrentResults
    }
  }

  const report: Report = {
    status: "running",
    stage: "setup",
    error: null,
    startedAt: new Date().toISOString(),
    modelRevision: null,
    manifestSha256: null,
    inputsSha256: null,
    execution: "compiled-diffusion",
    requestedReads: Number(requested),
    canvasLengths: [],
    geometry: {
      prefillChunks,
      prefillSource: chunks === undefined ? "default" : "explicit",
      blockSize: 16,
      maxTokens: null,
      modelCanvasLength: null,
      promptTokens: null
    },
    checkpointParameters: null,
    initializedState: null,
    prefixBytes: null,
    prefixInspection: null,
    externalMemoryZeroAfterLoad: false,
    measurements,
    compiledPrograms,
    memory,
    results,
    concurrent: { concurrency: 2, results: concurrentResults }
  }

  const samples = (phase: string) =>
    measurements.filter((m) => m.phase === phase && m.status === "passed").map((m) => m.elapsedMs!)

  let evidenceWriteMs = 0

  const persist = () => {
    const started = performance.now()

    const evidence = {
      ...report,
      updatedAt: new Date().toISOString(),
      loadMs: samples("load")[0] ?? null,
      compileMs: samples("compile")[0] ?? null,
      prefillMs: samples("prefill")[0] ?? null,
      timings: {
        unit: "milliseconds",
        percentileMethod: "nearest-rank",
        scope:
          "Host wall time including input preparation, answer-row readback and temporary cleanup; evidence writes and memory/prefix inspection excluded.",
        interpretation:
          "Small fixed samples, including first invocation. Descriptive only; no throughput or steady-state performance claim.",
        read: summarize(samples("read")),
        replay: summarize(samples("replay")),
        selected: summarize(samples("selected")),
        concurrentRead: summarize(samples("concurrent-read")),
        concurrentBatch: summarize(samples("concurrent-batch"))
      },
      memoryAccounting: {
        checkpoint:
          "Unique loaded.ownedParameters handles, Tensor.byteLength metadata; aliases excluded, allocation padding excluded.",
        programs:
          "Per-compile-call static plans. persistentBytes can reference the same weights across programs and must not be summed as unique storage. Cached handles may repeat diagnostics and native compile timings.",
        external:
          "Runtime.extensions.diagnostics.externalMemoryBytes samples, not device capacity, total process memory or peak allocation. A zero value after nonempty load is flagged as unavailable accounting.",
        cleanup:
          "After scoped handle release; native program caches and pools may remain reachable or await finalization."
      }
    }

    // Rename keeps the last complete report readable if the process is killed.
    writeFileSync(join(output, "run.json.next"), JSON.stringify(evidence, null, 2) + "\n")
    renameSync(join(output, "run.json.next"), join(output, "run.json"))
    evidenceWriteMs += performance.now() - started
  }

  const stage = (name: string) =>
    Effect.sync(() => {
      report.stage = name
      persist()
    })

  const sampleMemory = (name: string) =>
    Effect.gen(function*() {
      memory.push({
        stage: name,
        externalMemoryBytes: yield* runtime.extensions.diagnostics.externalMemoryBytes,
        at: new Date().toISOString()
      })
      persist()
    })

  const measured = <A, E, R>(phase: string, effect: Effect.Effect<A, E, R>, readIndex: number | null = null) =>
    Effect.suspend(() => {
      const measurement: Measurement = {
        phase,
        readIndex,
        status: "running",
        startedAt: new Date().toISOString(),
        elapsedMs: null,
        error: null
      }

      measurements.push(measurement)
      persist()
      const started = performance.now()
      const writesBefore = evidenceWriteMs

      return effect.pipe(Effect.onExit((exit) =>
        Effect.sync(() => {
          measurement.elapsedMs = performance.now() - started - (evidenceWriteMs - writesBefore)
          measurement.status = Exit.isSuccess(exit)
            ? "passed"
            : Cause.hasInterruptsOnly(exit.cause)
            ? "interrupted"
            : "failed"
          measurement.error = Exit.isFailure(exit) ? Cause.pretty(exit.cause) : null
          persist()
        })
      ))
    })

  persist()
  yield* Effect.scoped(Effect.gen(function*() {
    if (!(yield* BackendCuda.isAvailable)) throw new Error("CUDA is required for this gate")

    yield* sampleMemory("before-load")
    const manifestBytes = readFileSync(join(oracleDirectory, "manifest.json"))
    const manifest = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Manifest))(manifestBytes.toString("utf8"))
    report.manifestSha256 = hash(manifestBytes)
    report.modelRevision = manifest.revision

    if (manifest.revision !== "f7f5b7f5fa82ffc52addd066915886d497f5517b") {
      throw new Error("unexpected oracle model revision")
    }

    const inputsBytes = readFileSync(join(oracleDirectory, "inputs.json"))
    report.inputsSha256 = hash(inputsBytes)

    if (report.inputsSha256 !== manifest.inputs_sha256) throw new Error("oracle inputs hash mismatch")

    const inputs = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Inputs))(inputsBytes.toString("utf8"))
    report.geometry.promptTokens = inputs.case.promptIds.length
    const requestedInputs = inputs.reads.slice(0, Number(requested))

    if (requestedInputs.length !== Number(requested)) throw new Error("oracle has fewer reads than requested")

    const config = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Schema.Json))(
      readFileSync(join(checkpoint, "config.json"), "utf8")
    )

    yield* stage("load")
    console.log("Loading selected BF16 text weights")

    const loaded = yield* Effect.acquireRelease(
      measured("load", DiffusionGemma.loadParameters(join(checkpoint, "model.safetensors.index.json"), config)),
      (model) => Tensor.clearAll(model.ownedParameters),
      { interruptible: true }
    )

    const uniqueParameters = [...new Set(loaded.ownedParameters)]
    report.checkpointParameters = {
      uniqueCount: uniqueParameters.length,
      uniqueBytes: uniqueParameters.reduce((total, tensor) => total + Tensor.byteLength(tensor), 0),
      dtypes: [...new Set(uniqueParameters.map((tensor) => tensor.dtype))]
    }
    yield* sampleMemory("after-load")
    report.externalMemoryZeroAfterLoad = report.checkpointParameters.uniqueBytes > 0 &&
      memory.at(-1)!.externalMemoryBytes === 0

    const observed: Runtime.RuntimeService = {
      ...runtime,
      compile: (request) =>
        Effect.suspend(() => {
          const record: CompiledProgram = {
            index: compiledPrograms.length,
            status: "running",
            roots: request.roots.map(({ shape, dtype }) => ({ shape, dtype })),
            state: request.state ?? null,
            options: request.options,
            wallMs: null,
            diagnostics: null,
            error: null
          }

          compiledPrograms.push(record)
          persist()
          const started = performance.now()

          return runtime.compile(request).pipe(Effect.onExit((exit) =>
            Effect.sync(() => {
              record.wallMs = performance.now() - started
              record.status = Exit.isSuccess(exit)
                ? "passed"
                : Cause.hasInterruptsOnly(exit.cause)
                ? "interrupted"
                : "failed"

              if (Exit.isSuccess(exit)) record.diagnostics = exit.value.diagnostics
              else record.error = Cause.pretty(exit.cause)

              persist()
            })
          ))
        })
    }

    let prepared = loaded.tensors
    const statePath = process.env.EFFECT_TORCH_DIFFUSION_GEMMA_INITIALIZED_STATE

    if (statePath !== undefined) {
      const bytes = readFileSync(statePath)
      const state = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(InitializedState))(bytes.toString("utf8"))

      if (state.configSha256 !== hash(readFileSync(join(checkpoint, "config.json")))) {
        throw new Error("initialized state configuration hash differs")
      }

      if (
        state.torchVersion.split("+")[0] !== "2.10.0" ||
        state.torchCommit !== "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
      ) throw new Error("initialized state Torch pin differs")

      const file = join(dirname(statePath), state.file)

      if (hash(readFileSync(file)) !== state.sha256) throw new Error("initialized state file hash differs")

      const tensors = yield* Effect.acquireRelease(
        Safetensors.load(file),
        (values) => Tensor.clearAll(Object.values(values)),
        { interruptible: true }
      )

      const required = new Set(
        loaded.config.text_config.layer_types.map((kind) => "model.decoder.rotary_emb." + kind + "_inv_freq")
      )

      if (
        Object.keys(tensors).length !== required.size || state.tensors.length !== required.size ||
        new Set(state.tensors.map((entry) => entry.name)).size !== required.size
      ) throw new Error("initialized state tensor names differ")

      for (const entry of state.tensors) {
        const tensor = tensors[entry.name]

        if (
          !required.has(entry.name) || tensor === undefined || tensor.dtype !== "f32" ||
          tensor.shape.join(",") !== entry.shape.join(",")
        ) throw new Error("initialized state tensor metadata differs")

        if (hash(new Uint8Array(yield* runtime.readback(tensor))) !== entry.sha256) {
          throw new Error("initialized state tensor hash differs")
        }
      }

      for (const [layer, kind] of loaded.config.text_config.layer_types.entries()) {
        const tensor = tensors["model.decoder.rotary_emb." + kind + "_inv_freq"]!

        if (
          tensor.shape.length !== 1 ||
          tensor.shape[0] !== loaded.config.text_config.per_layer_config[String(layer)]!.head_dim / 2
        ) throw new Error("initialized state head geometry differs")
      }

      if (
        Object.values(tensors).reduce((total, tensor) => total + Tensor.byteLength(tensor), 0) !== state.payloadBytes
      ) throw new Error("initialized state byte accounting differs")

      report.initializedState = { manifestSha256: hash(bytes), manifest: state }
      prepared = { ...prepared, ...tensors }
      persist()
    }

    yield* stage("compile")
    const blockSize = report.geometry.blockSize
    const maxTokens = Math.ceil((inputs.case.promptIds.length + loaded.config.canvas_length) / blockSize) * blockSize
    report.geometry.maxTokens = maxTokens
    report.geometry.modelCanvasLength = loaded.config.canvas_length
    report.canvasLengths = [...new Set(inputs.reads.map((read) => read.canvas_ids.length))].sort((a, b) => a - b)

    const model = DiffusionGemma.fromTensors(loaded, prepared)
    const execution = yield* measured(
      "compile",
      Diffusion.compile(model.definition, model.parameters, {
        maxTokens,
        blockSize,
        prefillChunks,
        canvasLengths: report.canvasLengths,
        selectedReadouts: [{ rows: 1, labels: inputs.case.labelIds.length }]
      }).pipe(Effect.provideService(Runtime.Runtime, observed))
    )

    yield* sampleMemory("after-compile")
    yield* stage("prefill")

    const prefix = yield* Effect.acquireRelease(
      measured("prefill", execution.encode(Uint32Array.from(inputs.case.promptIds))),
      (prefix) => Effect.orDie(execution.release(prefix)),
      { interruptible: true }
    )

    report.prefixBytes = prefix.bytes
    yield* sampleMemory("after-prefill")
    yield* stage("prefix-inspection")
    // One host export outside read timings; retain only geometry and byte counters.
    report.prefixInspection = yield* execution.inspect(prefix).pipe(Effect.map((inspection) => ({
      cursor: inspection.cursor,
      retainedBytes: inspection.retainedBytes,
      sharedBytes: inspection.sharedBytes,
      copiedBytes: inspection.copiedBytes,
      layers: inspection.layers.map(({ layerId, startPosition, kvHeads, headDim, dtype, keys }) => ({
        layerId,
        startPosition,
        kvHeads,
        headDim,
        dtype,
        retainedRows: keys.length / (kvHeads * headDim)
      }))
    })))
    yield* sampleMemory("after-prefix-inspection")

    const read = (canvas: Uint32Array) =>
      Effect.scoped(Effect.gen(function*() {
        const logits = yield* Effect.acquireRelease(
          execution.evaluate(prefix, canvas, { _tag: "Initial" }),
          Tensor.clear,
          { interruptible: true }
        )

        const answer = yield* Tensor.slice(logits, {
          start: [0, inputs.case.slot, 0],
          end: [1, inputs.case.slot + 1, loaded.config.text_config.vocab_size]
        })

        const [row] = yield* Effect.acquireRelease(Tensor.compute([answer]), Tensor.clearAll, { interruptible: true })

        return yield* Tensor.toNumberArray(row!)
      }))

    const readSelected = (canvas: Uint32Array) =>
      Effect.scoped(
        Effect.acquireRelease(
          execution.score(prefix, canvas, [inputs.case.slot], inputs.case.labelIds),
          Tensor.clear,
          { interruptible: true }
        ).pipe(Effect.flatMap(Tensor.toNumberArray))
      )

    const isolated = new Map<number, ReadonlyArray<number>>()

    for (const input of requestedInputs) {
      yield* stage("oracle-" + input.index)
      const artifact = manifest.reads[input.index]

      if (artifact === undefined || artifact.file.includes("/") || artifact.file.includes("\\")) {
        throw new Error("invalid reference filename")
      }

      const file = join(oracleDirectory, artifact.file)

      if (hash(readFileSync(file)) !== artifact.sha256) throw new Error("oracle tensor file hash mismatch")

      const expected = yield* Effect.scoped(Effect.gen(function*() {
        const tensors = yield* Effect.acquireRelease(
          Safetensors.load(file, { names: ["logits.answer"] }),
          (tensors) => Tensor.clearAll(Object.values(tensors)),
          { interruptible: true }
        )

        const logits = tensors["logits.answer"]

        if (logits === undefined) throw new Error("missing oracle logits")

        return yield* Tensor.toNumberArray(logits)
      }))

      const canvas = Uint32Array.from(input.canvas_ids)
      yield* stage("read-" + input.index)
      const logits = yield* measured("read", read(canvas), input.index)
      isolated.set(input.index, logits)
      const raw = Buffer.from(Float32Array.from(logits).buffer)
      writeFileSync(join(output, "read-" + input.index + ".f32"), raw, { flag: "wx" })
      yield* sampleMemory("after-read-" + input.index)
      yield* stage("replay-" + input.index)
      const replay = yield* measured("replay", read(canvas), input.index)
      const replayExact = logits.length === replay.length && logits.every((value, index) => value === replay[index])

      if (!replayExact) throw new Error("read-only replay changed logits")

      yield* sampleMemory("after-replay-" + input.index)
      yield* stage("selected-" + input.index)
      const selected = yield* measured("selected", readSelected(canvas), input.index)
      const selectedFromFull = inputs.case.labelIds.map((id) => logits[id]!)
      const selectedAgreement = compare(selected, selectedFromFull)
      const expectedSelected = inputs.case.labelIds.map((id) => expected[id]!)
      const probability = probabilities(selected)
      const expectedProbability = probabilities(expectedSelected)
      results.push({
        index: input.index,
        readMs: samples("read").at(-1)!,
        replayMs: samples("replay").at(-1)!,
        selectedMs: samples("selected").at(-1)!,
        replayExact,
        logitsSha256: hash(raw),
        logits: compare(logits, expected),
        selectedAgreement,
        rows: [inputs.case.slot],
        labelIds: inputs.case.labelIds,
        selected,
        selectedFromFull,
        expectedSelected,
        probability,
        expectedProbability,
        maximumProbabilityDifference: Math.max(
          ...probability.map((value, i) => Math.abs(value - expectedProbability[i]!))
        )
      })
      yield* sampleMemory("after-selected-" + input.index)
      console.log(JSON.stringify(results.at(-1)))
    }

    // Two additional calls reuse the first two isolated baselines. A one-read
    // smoke run borrows the same prefix/canvas twice to keep concurrency two.
    const concurrentInputs = [requestedInputs[0]!, requestedInputs[1] ?? requestedInputs[0]!]
    yield* stage("concurrent-shared-prefix")

    const concurrent = yield* measured(
      "concurrent-batch",
      Effect.forEach(
        concurrentInputs,
        (input) => measured("concurrent-read", read(Uint32Array.from(input.canvas_ids)), input.index),
        { concurrency: 2 }
      )
    )

    for (const [slot, logits] of concurrent.entries()) {
      const input = concurrentInputs[slot]!
      const comparison = compare(logits, isolated.get(input.index)!)
      concurrentResults.push({
        index: input.index,
        isolatedExact: comparison.exact === comparison.elements,
        comparison
      })
    }

    persist()

    if (concurrentResults.some((result) => !result.isolatedExact)) {
      throw new Error("shared-prefix concurrent evaluation changed isolated logits")
    }

    yield* sampleMemory("after-concurrent-reads")
    yield* stage("comparison-gate")

    if (
      results.some((result) => result.logits.beyondOneBf16Step > 0 || result.selectedAgreement.beyondOneBf16Step > 0)
    ) {
      throw new Error("full-model comparison exceeds one BF16 step; evidence saved for investigation")
    }
  })).pipe(Effect.onExit((exit) =>
    Effect.gen(function*() {
      report.status = Exit.isSuccess(exit) ? "passed" : Cause.hasInterruptsOnly(exit.cause) ? "interrupted" : "failed"
      report.error = Exit.isFailure(exit) ? Cause.pretty(exit.cause) : null

      if (Exit.isSuccess(exit)) report.stage = "complete"

      persist()
      yield* sampleMemory("after-scoped-cleanup")
    })
  ))
}).pipe(Effect.provide(BackendCuda.layer()))

NodeRuntime.runMain(program)

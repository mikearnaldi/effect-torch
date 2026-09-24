/**
 * pnpm --filter @effect-torch/examples exec tsx scripts/diffusion-gemma/verify-generation.ts \
 *   <cpu|metal|cuda> <checkpoint-directory> <inputs.json> [tokenizer.json]
 *
 * inputs.json contains tokenized model input, including its chat template:
 * {"promptIds":[2,7,4],"labelIds":[8,9],"answerRow":0,"maxNewTokens":64,"seed":42}
 * Optional canvasIds supplies the independent decision canvas. Output includes
 * selected-answer probabilities, generated IDs, optional decoded text, and
 * measured sampler/host-random-input costs. Uses one compiled model artifact.
 * For a bounded two-block run, set maxNewTokens=257, maxSteps=3,
 * stabilityThreshold=3, eosTokenIds=[], outputLimit="whole-block".
 * Optional generationReference points to full_generation.py's generation.json.
 * Replay uses its settings and hashed F32 random files, then checks every
 * step's argmax canvas and final output. Replay timing includes file I/O/hashing.
 * Optional evidenceDirectory retains atomic partial native-phase observations.
 * Optional prefillChunks configures ascending encoder widths. For the pinned
 * 278-token prompt and 256-token commits, use [256,278] to match GEMM geometry.
 * Runtime timings distinguish prefill, denoiser, readout, commit, and sampler;
 * physical GPU memory is measured separately by the validation launcher.
 * EFFECT_TORCH_DIFFUSION_GEMMA_INITIALIZED_STATE selects the same generated
 * manifest/safetensors asset as packages/examples/scripts/diffusion-gemma/verify-full.ts for pinned RoPE initialization.
 */
import * as BackendAppleNative from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Decision, Diffusion, Runtime, Safetensors, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/models"
import * as Tokenizers from "@effect-torch/tokenizers"
import { NodeRuntime } from "@effect/platform-node"
import { Cause, Effect, Exit, Schema } from "effect"
import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs"
import { basename, dirname, join } from "node:path"

const TokenId = Schema.Int.check(Schema.isGreaterThanOrEqualTo(0), Schema.isLessThanOrEqualTo(0xffff_ffff))

const Inputs = Schema.Struct({
  promptIds: Schema.Array(TokenId),
  labelIds: Schema.Array(TokenId),
  canvasIds: Schema.optionalKey(Schema.Array(TokenId)),
  answerRow: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  maxNewTokens: Schema.Int.check(Schema.isGreaterThanOrEqualTo(0)),
  maxSteps: Schema.optionalKey(Schema.Int.check(Schema.isGreaterThan(0))),
  eosTokenIds: Schema.optionalKey(Schema.Array(TokenId)),
  stabilityThreshold: Schema.optionalKey(Schema.Int.check(Schema.isGreaterThanOrEqualTo(0))),
  outputLimit: Schema.optionalKey(Schema.Literals(["exact", "whole-block"])),
  generationReference: Schema.optionalKey(Schema.String),
  evidenceDirectory: Schema.optionalKey(Schema.String),
  prefillChunks: Schema.optionalKey(
    Schema.Array(Schema.Int.check(Schema.isGreaterThan(0), Schema.isLessThanOrEqualTo(0xffff_ffff)))
  ),
  seed: TokenId
})

const GenerationReference = Schema.Struct({
  status: Schema.Literal("passed"),
  prompt: Schema.Array(TokenId),
  canvas_length: Schema.Int,
  vocab_size: Schema.Int,
  config: Schema.Struct({
    max_new_tokens: Schema.Int,
    max_denoising_steps: Schema.Int,
    t_min: Schema.Number,
    t_max: Schema.Number,
    stability_threshold: Schema.Int,
    confidence_threshold: Schema.Number,
    eos_token_id: Schema.Null,
    pad_token_id: TokenId,
    sampler_config: Schema.Struct({ entropy_bound: Schema.Number })
  }),
  random_canvases: Schema.Array(Schema.Array(TokenId)),
  blocks: Schema.Array(Schema.Struct({
    steps: Schema.Array(Schema.Struct({
      argmax_tokens: Schema.Array(TokenId),
      exponentials: Schema.Struct({ file: Schema.String, bytes: Schema.Int, sha256: Schema.String })
    }))
  })),
  sequences: Schema.Array(TokenId)
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

interface InitializedStateRecord {
  manifestSha256: string
  manifest: typeof InitializedState.Type
}

type Phase = "load" | "compile" | "decision" | "generation"

type Status = "running" | "passed" | "failed" | "interrupted"

interface Cleanup {
  externalBytesAfterCleanup: number | null
}

interface NativeSample {
  startedAt: string
  phase: Phase
  kind: string
  program: number | null
  status: Status
  elapsedMilliseconds: number | null
  error: string | null
  inputShapes: ReadonlyArray<ReadonlyArray<number>>
  outputShapes: ReadonlyArray<ReadonlyArray<number>>
  validLengths: ReadonlyArray<number>
}

interface ProgramRecord {
  index: number
  kind: string
  state: Runtime.CompileRequest["state"]
  roots: ReadonlyArray<{
    shape: ReadonlyArray<number>
    dtype: Tensor.DType
  }>
  compile: NativeSample
  diagnostics: Runtime.ExecutableDiagnostics | null
}

const summarize = (values: ReadonlyArray<number>) => {
  const sorted = [...values].sort((a, b) => a - b)

  return {
    count: values.length,
    samplesMilliseconds: values,
    totalMilliseconds: values.reduce((a, b) => a + b, 0),
    p50Milliseconds: sorted[Math.max(0, Math.ceil(sorted.length * 0.5) - 1)] ?? null,
    p95Milliseconds: sorted[Math.max(0, Math.ceil(sorted.length * 0.95) - 1)] ?? null
  }
}

const main = Effect.gen(function*() {
  const [device, checkpoint, inputFile, tokenizerFile, extra] = process.argv.slice(2)

  if (
    (device !== "cpu" && device !== "metal" && device !== "cuda") ||
    checkpoint === undefined || inputFile === undefined || extra !== undefined
  ) throw new Error("usage: generate.ts <cpu|metal|cuda> <checkpoint-directory> <inputs.json> [tokenizer.json]")

  if (device === "cuda" && !(yield* BackendCuda.isAvailable)) throw new Error("CUDA is required")

  if (device === "metal" && !(yield* BackendAppleNative.isAvailable)) throw new Error("Metal is required")

  const backend = device === "cpu"
    ? BackendCpu.layer
    : device === "metal"
    ? BackendAppleNative.layer()
    : BackendCuda.layer()

  const inputs = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Inputs))(readFileSync(inputFile, "utf8"))

  const config = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Schema.Json))(
    readFileSync(join(checkpoint, "config.json"), "utf8")
  )

  if (inputs.labelIds.length === 0) throw new Error("labelIds must be nonempty")

  const prefillChunks = inputs.prefillChunks ?? [1, 16]

  if (prefillChunks.length === 0 || prefillChunks.some((n, i) => i > 0 && n <= prefillChunks[i - 1]!)) {
    throw new Error("prefillChunks must be nonempty, ascending and unique")
  }

  const geometry = { prefillChunks, source: inputs.prefillChunks === undefined ? "default" : "explicit" }

  const reference = inputs.generationReference === undefined ? undefined : yield* Schema.decodeUnknownEffect(
    Schema.fromJsonString(GenerationReference)
  )(readFileSync(inputs.generationReference, "utf8"))

  if (reference !== undefined && JSON.stringify(reference.prompt) !== JSON.stringify(inputs.promptIds)) {
    throw new Error("generation reference prompt differs from inputs")
  }

  const result = yield* Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const programs: Array<ProgramRecord> = []
    const executions: Array<NativeSample> = []
    const readbacks: Array<NativeSample> = []

    const randomSources: Array<{
      kind: "canvas" | "exponentials"
      milliseconds: number
      elements: number
    }> = []

    const progress: Array<
      {
        block: number
        step: number
        done: boolean
        draft: ReadonlyArray<number>
        elapsedMilliseconds: number
      }
    > = []

    let checkpointParameters: {
      uniqueCount: number
      uniqueBytes: number
      dtypes: ReadonlyArray<Tensor.DType>
    } | null = null

    const handles = new WeakMap<Runtime.ExecutableHandle, ProgramRecord>()
    let initializedState: InitializedStateRecord | null = null
    const hasRead = new Set<Phase>()
    let phase: Phase = "load"
    let phaseStarted = performance.now()
    let status: Status = "running"
    let error: string | null = null
    let evidenceWriteMilliseconds = 0
    const cleanup: Cleanup = { externalBytesAfterCleanup: null }
    const phaseMilliseconds: Partial<Record<Phase, number>> = {}
    const evidenceFile = inputs.evidenceDirectory === undefined ? undefined : join(inputs.evidenceDirectory, "run.json")

    if (evidenceFile !== undefined) {
      if (existsSync(evidenceFile)) throw new Error("choose a new evidenceDirectory")

      mkdirSync(dirname(evidenceFile), { recursive: true })
    }

    const persist = () => {
      if (evidenceFile === undefined) return

      const started = performance.now()
      writeFileSync(
        evidenceFile + ".next",
        JSON.stringify(
          {
            status,
            error,
            phase,
            phaseElapsedMilliseconds: performance.now() - phaseStarted,
            phaseMilliseconds,
            checkpointParameters,
            initializedState,
            geometry,
            progress,
            programs,
            executions,
            readbacks,
            randomSources,
            evidenceWriteMilliseconds,
            cleanup
          },
          null,
          2
        ) + "\n"
      )
      renameSync(evidenceFile + ".next", evidenceFile)
      evidenceWriteMilliseconds += performance.now() - started
    }

    const enter = (next: Phase) => {
      phase = next
      phaseStarted = performance.now()
      persist()
    }

    const sample = (kind: string, program: number | null): NativeSample => ({
      startedAt: new Date().toISOString(),
      phase,
      kind,
      program,
      status: "running",
      elapsedMilliseconds: null,
      error: null,
      inputShapes: [],
      outputShapes: [],
      validLengths: []
    })

    const measured = <A, E, R>(record: NativeSample, effect: Effect.Effect<A, E, R>) =>
      Effect.suspend(() => {
        // Persist before invoking the backend. After success, only synchronous
        // metadata updates run before output ownership reaches the caller.
        persist()
        const started = performance.now()

        return effect.pipe(Effect.onExit((exit) =>
          Effect.sync(() => {
            record.elapsedMilliseconds = performance.now() - started
            record.status = Exit.isSuccess(exit)
              ? "passed"
              : Cause.hasInterruptsOnly(exit.cause)
              ? "interrupted"
              : "failed"
            record.error = Exit.isFailure(exit) ? Cause.pretty(exit.cause) : null
          })
        ))
      })

    const observed: Runtime.RuntimeService = {
      ...runtime,
      compile: (request) =>
        Effect.suspend(() => {
          const kind = request.state?.access === "Append" ?
            "encoder"
            : request.state?.access === "ReadOnly" ?
            "denoiser"
            : request.roots.length === 6 && request.roots[1]!.dtype === "u32" && request.roots[2]!.dtype === "u32" ?
            "sampler"
            : request.roots.length === 1 && request.roots[0]!.shape.length === 3 && request.roots[0]!.dtype === "f32" ?
            "readout"
            : "auxiliary"

          const record: ProgramRecord = {
            index: programs.length,
            kind,
            state: request.state,
            roots: request.roots.map(({ shape, dtype }) => ({ shape, dtype })),
            compile: sample("compile", programs.length),
            diagnostics: null
          }

          programs.push(record)

          return measured(record.compile, runtime.compile(request)).pipe(Effect.tap((handle) =>
            Effect.sync(() => {
              handles.set(handle, record)
              record.diagnostics = handle.diagnostics
            })
          ))
        }),
      execute: (handle, invocation) =>
        Effect.suspend(() => {
          const program = handles.get(handle)

          const kind = invocation.state?.access === "Append" ?
            hasRead.has(phase) ? "commit" : "encoderPrefill"
            : invocation.state?.access === "ReadOnly"
            ? "denoiser"
            : program?.kind ?? "unclassified"

          if (invocation.state?.access === "ReadOnly") hasRead.add(phase)

          const record = sample(kind, program?.index ?? null)
          record.inputShapes = invocation.bindings.map((tensor) => tensor.shape)
          record.validLengths = invocation.state?.validLengths ?? []
          executions.push(record)

          return measured(record, runtime.execute(handle, invocation)).pipe(Effect.tap((outputs) =>
            Effect.sync(() => {
              record.outputShapes = outputs.map((tensor) => tensor.shape)
            })
          ))
        }),
      readback: (tensor) =>
        Effect.suspend(() => {
          const record = sample("readback", null)
          record.inputShapes = [tensor.shape]
          readbacks.push(record)

          return measured(record, runtime.readback(tensor))
        })
    }

    enter("load")

    return yield* Effect.scoped(Effect.gen(function*() {
      const loadStarted = performance.now()

      const loaded = yield* Effect.acquireRelease(
        DiffusionGemma.loadParameters(join(checkpoint, "model.safetensors.index.json"), config),
        (loaded) => Tensor.clearAll(loaded.ownedParameters),
        { interruptible: true }
      )

      const loadMilliseconds = performance.now() - loadStarted
      phaseMilliseconds.load = loadMilliseconds
      const uniqueParameters = [...new Set(loaded.ownedParameters)]
      checkpointParameters = {
        uniqueCount: uniqueParameters.length,
        uniqueBytes: uniqueParameters.reduce((sum, tensor) => sum + Tensor.byteLength(tensor), 0),
        dtypes: [...new Set(uniqueParameters.map((tensor) => tensor.dtype))]
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

        initializedState = { manifestSha256: hash(bytes), manifest: state }
        prepared = { ...prepared, ...tensors }
        persist()
      }

      const model = DiffusionGemma.fromTensors(loaded, prepared)
      const definition = model.definition
      const blockSize = 16
      const maxNewTokens = reference?.config.max_new_tokens ?? inputs.maxNewTokens
      const blocks = Math.ceil(maxNewTokens / definition.canvasLength)
      const context = inputs.promptIds.length + Math.max(1, blocks) * definition.canvasLength
      enter("compile")
      const compileStarted = performance.now()

      const artifact = yield* Diffusion.compile(definition, model.parameters, {
        maxTokens: Math.ceil((context * 2 + blockSize) / blockSize) * blockSize,
        blockSize,
        prefillChunks,
        canvasLengths: [...new Set([definition.canvasLength, inputs.canvasIds?.length ?? definition.canvasLength])]
          .sort((
            a,
            b
          ) => a - b),
        selectedReadouts: [{ rows: 1, labels: inputs.labelIds.length }]
      })

      const compileMilliseconds = performance.now() - compileStarted
      phaseMilliseconds.compile = compileMilliseconds
      yield* Tensor.clearAll(loaded.ownedParameters)
      const prompt = Uint32Array.from(inputs.promptIds)
      const random = yield* DiffusionGemma.generationRandom(inputs.seed)

      const decisionCanvas = inputs.canvasIds === undefined
        ? random.canvas(artifact.canvasLength, artifact.vocabSize)
        : Uint32Array.from(inputs.canvasIds)

      enter("decision")
      const decisionStarted = performance.now()

      const decision = yield* Effect.scoped(Effect.gen(function*() {
        const prefix = yield* Effect.acquireRelease(
          artifact.encode(prompt),
          (prefix) => Effect.orDie(artifact.release(prefix)),
          { interruptible: true }
        )

        const logits = yield* Effect.acquireRelease(
          artifact.score(prefix, decisionCanvas, [inputs.answerRow], inputs.labelIds),
          Tensor.clear,
          { interruptible: true }
        )

        const selectedLogits = yield* Tensor.toNumberArray(logits)

        return { selectedLogits, probabilities: yield* Decision.restrictedSoftmax(selectedLogits) }
      }))

      const decisionMilliseconds = performance.now() - decisionStarted
      phaseMilliseconds.decision = decisionMilliseconds

      if (
        reference !== undefined &&
        (reference.canvas_length !== artifact.canvasLength || reference.vocab_size !== artifact.vocabSize)
      ) {
        throw new Error("generation reference geometry differs from checkpoint")
      }

      let canvasIndex = 0
      let exponentialIndex = 0
      const recordedSteps = reference?.blocks.flatMap((block) => block.steps) ?? []

      const replayRandom: DiffusionGemma.GenerationRandom | undefined = reference === undefined ? undefined : {
        canvas: () => {
          const canvas = reference.random_canvases[canvasIndex++]

          if (canvas === undefined) throw new Error("generation consumed too many random canvases")

          return Uint32Array.from(canvas)
        },
        exponentials: (length) => {
          const record = recordedSteps[exponentialIndex++]?.exponentials

          if (record === undefined || basename(record.file) !== record.file) {
            throw new Error("invalid random replay file")
          }

          const bytes = readFileSync(join(dirname(inputs.generationReference!), record.file))

          if (
            bytes.byteLength !== record.bytes || bytes.byteLength !== length * 4 ||
            createHash("sha256").update(bytes).digest("hex") !== record.sha256
          ) {
            throw new Error("random replay file hash/length mismatch")
          }

          return new Float32Array(bytes.buffer, bytes.byteOffset, length)
        }
      }

      const source = replayRandom ?? (yield* DiffusionGemma.generationRandom(inputs.seed))

      const measuredRandom: DiffusionGemma.GenerationRandom = {
        canvas: (length, vocabSize) => {
          const started = performance.now()

          try {
            return source.canvas(length, vocabSize)
          } finally {
            randomSources.push({ kind: "canvas", milliseconds: performance.now() - started, elements: length })
          }
        },
        exponentials: (length) => {
          const started = performance.now()

          try {
            return source.exponentials(length)
          } finally {
            randomSources.push({ kind: "exponentials", milliseconds: performance.now() - started, elements: length })
          }
        }
      }

      enter("generation")
      const generationStarted = performance.now()

      const generated = yield* DiffusionGemma.generate(artifact, prompt, {
        maxNewTokens,
        maxSteps: reference?.config.max_denoising_steps ?? inputs.maxSteps ??
          DiffusionGemma.generationDefaults.maxSteps,
        minTemperature: reference?.config.t_min ?? DiffusionGemma.generationDefaults.minTemperature,
        maxTemperature: reference?.config.t_max ?? DiffusionGemma.generationDefaults.maxTemperature,
        entropyBound: reference?.config.sampler_config.entropy_bound ?? DiffusionGemma.generationDefaults.entropyBound,
        confidenceThreshold: reference?.config.confidence_threshold ??
          DiffusionGemma.generationDefaults.confidenceThreshold,
        stabilityThreshold: reference?.config.stability_threshold ?? inputs.stabilityThreshold ??
          DiffusionGemma.generationDefaults.stabilityThreshold,
        eosTokenIds: reference === undefined ? inputs.eosTokenIds ?? DiffusionGemma.generationDefaults.eosTokenIds : [],
        padTokenId: reference?.config.pad_token_id ?? DiffusionGemma.generationDefaults.padTokenId,
        random: measuredRandom,
        outputLimit: reference === undefined ? inputs.outputLimit ?? "exact" : "whole-block",
        onProgress: (event) =>
          Effect.sync(() => {
            progress.push({
              block: event.block.index,
              step: event.step.index,
              done: event.done,
              draft: Array.from(event.draft),
              elapsedMilliseconds: performance.now() - generationStarted
            })
          })
      })

      const generationMilliseconds = performance.now() - generationStarted
      phaseMilliseconds.generation = generationMilliseconds

      return {
        device,
        geometry,
        checkpointParameters,
        initializedState,
        cleanup,
        loadMilliseconds,
        compileMilliseconds,
        decisionMilliseconds,
        generationMilliseconds,
        decision: { ...decision, labelIds: inputs.labelIds, canvasIds: Array.from(decisionCanvas) },
        generation: { ...generated, tokens: Array.from(generated.tokens), progress },
        executionProfile: {
          programs,
          executions,
          readbacks,
          randomSources,
          evidenceWriteMilliseconds,
          generation: Object.fromEntries(
            ["encoderPrefill", "denoiser", "readout", "commit", "sampler", "auxiliary"].map((
              kind
            ) => [
              kind,
              summarize(
                executions.filter((sample) =>
                  sample.phase === "generation" && sample.kind === kind && sample.status === "passed"
                ).map((sample) => sample.elapsedMilliseconds!)
              )
            ])
          ),
          hostRandomSource: summarize(randomSources.map((sample) => sample.milliseconds)),
          hostRandomPreparationIncludingValidationMilliseconds: generated.randomMilliseconds,
          samplerProcessingIncludingCompileReadbackMilliseconds: generated.processingMilliseconds,
          timerBoundaries: {
            native:
              "Runtime.execute Effect entry to completion, including backend queue wait, internal grouped-expert control readback and returned handles; excludes input materialization, graph compilation, separate Runtime.readback calls and evidence writes. No residual is labeled denoiser.",
            append:
              "Append before the first ReadOnly in each decision/generation request is prefill; subsequent Append is continuing-block commit. Each sample is one fixed-width invocation; validLengths records real tokens.",
            stateless:
              "Single F32 rank-3 output is readout. Six outputs with sampled/argmax u32 roots are sampler. Roots, binding and output shapes are retained for classification audit.",
            random:
              "Source callbacks include generation or replay file I/O/hashing; library randomMilliseconds additionally includes input validation.",
            aggregate:
              "High-level and library sampler timings include observer evidence writes. Native samples exclude those writes. Phase totals overlap native samples and must not be added to them.",
            memory:
              "Per-program persistentBytes may share weights and must not be summed as unique storage. Physical device memory is sampled separately by the run launcher."
          }
        },
        replay: reference === undefined ? null : {
          reference: inputs.generationReference,
          tokensExact:
            JSON.stringify(Array.from(generated.tokens)) === JSON.stringify(reference.sequences.slice(prompt.length)),
          randomCanvasesConsumed: canvasIndex,
          expectedRandomCanvases: reference.random_canvases.length,
          exponentialInputsConsumed: exponentialIndex,
          expectedExponentialInputs: recordedSteps.length,
          stepArgmaxExact: progress.map((event, index) =>
            JSON.stringify(event.draft) === JSON.stringify(recordedSteps[index]?.argmax_tokens)
          )
        }
      }
    })).pipe(
      Effect.provideService(Runtime.Runtime, observed),
      Effect.onExit((exit) =>
        Effect.gen(function*() {
          status = Exit.isSuccess(exit) ? "passed" : Cause.hasInterruptsOnly(exit.cause) ? "interrupted" : "failed"
          error = Exit.isFailure(exit) ? Cause.pretty(exit.cause) : null

          if (
            Exit.isSuccess(exit) && exit.value.replay !== null &&
            (!exit.value.replay.tokensExact || exit.value.replay.stepArgmaxExact.some((exact) => !exact) ||
              exit.value.replay.randomCanvasesConsumed !== exit.value.replay.expectedRandomCanvases ||
              exit.value.replay.exponentialInputsConsumed !== exit.value.replay.expectedExponentialInputs)
          ) {
            status = "failed"
            error = "generation replay differs from pinned reference"
          }

          cleanup.externalBytesAfterCleanup = yield* runtime.extensions.diagnostics.externalMemoryBytes
          persist()
        })
      )
    )
  }).pipe(Effect.provide(backend))

  const text = tokenizerFile === undefined ? null : yield* Effect.gen(function*() {
    const tokenizer = yield* Tokenizers.fromFile(tokenizerFile, Tokenizers.strictConfig)

    return yield* tokenizer.decode(Uint32Array.from(result.generation.tokens), { skipSpecialTokens: true })
  })

  console.log(JSON.stringify({ ...result, text }, null, 2))

  if (
    result.replay !== null &&
    (!result.replay.tokensExact || result.replay.randomCanvasesConsumed !== result.replay.expectedRandomCanvases ||
      result.replay.exponentialInputsConsumed !== result.replay.expectedExponentialInputs ||
      result.replay.stepArgmaxExact.some((exact) => !exact))
  ) {
    throw new Error("generation replay differs from pinned reference; output includes comparison evidence")
  }
})

NodeRuntime.runMain(main)

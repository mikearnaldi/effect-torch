/**
 * Compiled autoregressive inference, sampled generation, and caller-driven
 * stateful logits execution over generic `Model` graphs.
 *
 * An inference artifact retains one frozen parameter generation, fixed-shape
 * prefill/decode programs, and a shared decode-state pool. Each generation or
 * execution session owns its live sequences. Sequences track absolute cursors,
 * K/V block references, and recurrent state. Completed K/V blocks may remain
 * in the artifact's reclaimable prefix cache after a sequence is released.
 *
 * Sampling and speculative acceptance run through the native inference
 * artifact. Custom host samplers use caller-owned logits from `execution()`.
 * Model construction, ordinary forward execution, and graph exposure names
 * belong to `Model`.
 *
 * @since 0.1.0
 */
import { Data, Effect, Exit, Predicate, Semaphore } from "effect"
import * as Model from "./Model.ts"
import * as Runtime from "./Runtime.ts"
import type * as Speculation from "./Speculation.ts"
import * as Tensor from "./Tensor.ts"

/**
 * A failure in inference-artifact construction or generation: invalid
 * configuration or model structure, or misuse of the generation calling
 * convention. Current operation labels include `inference`, `add`, `prefill`,
 * and `step`; treat `message` as a diagnostic rather than a stable protocol.
 * Decode compilation and pool-construction tensor errors are wrapped as
 * `InferenceError("inference")`. Errors raised earlier by `model.forward`, and
 * tensor/backend failures during `add`, `step`, cursor, or cleanup, retain their
 * original types.
 *
 * @since 0.1.0
 * @category errors
 */
export class InferenceError extends Data.TaggedError("InferenceError")<{
  /** The inference phase reporting the failure. */
  readonly op: string
  /** Human-readable diagnostic text; branch on the error tag and `op` rather than parsing it. */
  readonly message: string
}> {}

/**
 * Fixed deployment geometry for {@link inference}. Construction validates
 * these scalar fields, then eagerly traces and compiles one prefill program
 * per `prefillChunks` width `[batchSize, chunk]` and fixed-width decode
 * `[batchSize, 1]`. Batch size one
 * uses the same decode path. There is no later shape-specialization cache.
 *
 * Validation checks structure only. It does not estimate whether the
 * pool is large enough for a particular set of prompts, check token ids against
 * the model vocabulary, prove that every model operation supports decode
 * specialization, or prove that learned position tables cover future cursors.
 * Those constraints fail when the graph is compiled or a sequence is run.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface CompileOptions {
  /**
   * Fixed pool capacity in token rows, shared by live sequences and
   * unreferenced prefix-cache blocks across every session of the artifact.
   * Must be a positive integer and an exact multiple of `blockSize`. Without
   * an effective attention window it also bounds each sequence cursor; with a
   * window, aggregate live frontiers can still exhaust the shared pool.
   */
  readonly maxTokens: number
  /**
   * KV paging granularity in tokens. Must be a positive integer that
   * divides `maxTokens`. Defaults to 16.
   */
  readonly blockSize?: number
  /**
   * Requested positive attention-retention window, no greater than
   * `maxTokens`. Omit for full history. Decode specialization permits block
   * eviction only if every attention operation resolves to bounded local
   * attention; an explicit full-attention operation makes the compiled program
   * retain full history. The effective window is part of the compiled geometry.
   *
   * With cursor-offset RoPE and no separately bounded absolute-position state,
   * eviction can let a sequence advance beyond `maxTokens` while retaining only
   * its live window and partial frontier. It does not reset the logical cursor,
   * expand a learned position table, or guarantee enough aggregate pool capacity.
   */
  readonly attentionWindow?: number
  /**
   * Fixed prompt-chunk token widths in ascending order. The compiler creates
   * one prefill program per entry. The runtime serves each prompt chunk from
   * the largest compiled width covering its remaining tokens, so smaller widths
   * only bound zero-padding waste on short prompts. Entries must be positive
   * safe integers; they need not be multiples of `blockSize`. Every prefill
   * invocation has one of the compiled shapes. The final suffix is
   * zero-padded, but only its real token ids advance the sequence, enter
   * state hashes, and select the returned logits row. Graph operations still
   * evaluate the padded extent, so a cursor-offset learned position table
   * must cover the largest compiled chunk at every invocation.
   */
  readonly prefillChunks: ReadonlyArray<number>
  /**
   * Token-tensor dtype used by all fixed programs. Defaults to `"u32"`;
   * prompts passed to {@link Generation.add} must match exactly. Decode state
   * and prefix hashes are u32-based even for `"i64"`, so prompt and step ids
   * must still be non-negative and fit u32.
   */
  readonly tokenDtype?: "u32" | "i64"
  /**
   * KV storage dtype. Defaults to `"f32"`; `"f16"` and `"bf16"` narrow
   * rows on write and attention widens them to f32. `"int8"` uses symmetric
   * per-token, per-head quantization with f32 scales. KDA and short-convolution
   * recurrent state remains f32 and is not controlled by this option.
   */
  readonly kvDtype?: "f32" | "f16" | "bf16" | "int8"
  /** Default sampling controls for generation. Defaults to `{ seed: 0 }`. */
  readonly sampling?: GenerationSamplingOptions
  /**
   * Positive fixed decode width, maximum live sequences tracked by each
   * session, and maximum active entries in one step. Defaults to `8`. The one
   * decode program has shape `[batchSize, 1]`; batch size one is the ordinary
   * single-sequence case. This is not a global limit across sessions; all
   * sessions still compete for one pool's token-row capacity.
   */
  readonly batchSize?: number
  /** Optional high-level proposer compiled with this target. */
  readonly speculation?: {
    /** Proposer artifact whose vocabulary and target contract must match this model. */
    readonly proposer: Speculation.Proposer
    /** Maximum proposal width, bounded by the artifact's trained capacity. */
    readonly maxDraftTokens: number
    /** Proposal-width policy; only `"fixed"` is currently implemented. */
    readonly schedule?: "fixed" | "adaptive"
  } | undefined
}

/**
 * One mutable sequence owned by a {@link Generation} session. Its backend state
 * consists of an absolute logical cursor, KV block references when attention is
 * present, and per-sequence KDA/short-convolution state when present. It is an
 * ordinary value rather than a scoped resource.
 *
 * Call {@link GenerationSeq.finish} when the sequence leaves a scheduler, or
 * {@link Generation.close} for all sequences in that session. Releasing drops
 * live references; completed blocks may remain in the artifact's reclaimable
 * prefix cache. Native finalization is only a fallback for abandoned handles.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationSeq {
  /** Runtime discriminant for a sampled-generation sequence. */
  readonly _tag: "GenerationSeq"
  /**
   * Returns the total logical token count, including evicted window
   * positions. Fails after the underlying sequence has been released.
   */
  readonly cursor: () => Effect.Effect<number, Tensor.TensorError, Runtime.Runtime>
  /**
   * Removes this sequence from its session and releases its backend state.
   * Completed KV blocks can become prefix-cache entries rather than immediately
   * free blocks. Calls after it has already been finished or closed are no-ops.
   */
  readonly finish: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * Sampling controls owned by generation; draw counters are sequence-managed.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationSamplingOptions {
  /** Non-negative temperature; `0` selects greedy sampling. */
  readonly temperature?: number
  /** Non-negative candidate count; `0` disables top-k filtering. */
  readonly topK?: number
  /** Nucleus probability in `(0, 1]`; `1` disables top-p filtering. */
  readonly topP?: number
  /** Unsigned 64-bit seed. Safe integer numbers remain accepted for convenience. */
  readonly seed: bigint | number
}

/**
 * One prompt admitted by {@link Generation.add}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationAdd {
  /** Nonempty `[1, T]` token tensor matching the artifact's token dtype and placement. */
  readonly prompt: Tensor.Any
  /** Overrides inference sampling defaults for the admission page only. */
  readonly sampling?: Partial<GenerationSamplingOptions>
  /** Optional positive generation limit for this sequence. */
  readonly maxTokens?: number | undefined
  /** Token ids that terminate this sequence when sampled. */
  readonly eosTokens?: ReadonlyArray<number>
}

/**
 * One live sequence selected by {@link Generation.step}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface GenerationStep {
  /** Sequence whose pending token is committed. */
  readonly seq: GenerationSeq
  /** Overrides inference sampling defaults for this page only. */
  readonly sampling?: Partial<GenerationSamplingOptions>
}

/**
 * A nonempty page of sampled tokens for one sequence.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface TokenPage {
  /** Sequence that owns this page. */
  readonly seq: GenerationSeq
  /** Sampled token ids in generation order. */
  readonly tokens: ReadonlyArray<number>
  /** Terminal policy reached by the final token, when the page ends the sequence. */
  readonly stopReason?: "eos" | "maxTokens" | undefined
}

/**
 * A caller-scheduled generation session over one {@link Artifact}.
 * {@link Generation.add} creates and prefills independent sequences and samples
 * their first token. {@link Generation.step} commits each sequence's pending
 * token and samples its successor. Every active count uses the fixed
 * `[batchSize, 1]` program with explicit inactive lanes.
 *
 * Prefix matching spans the pool, not just one session. It uses chained hashes
 * to reuse the longest resident proper prefix made of complete `blockSize` blocks,
 * whether those blocks are referenced by another live sequence or retained
 * unreferenced in the LRU cache. At least the final prompt token is always
 * executed so `add` can sample the first pending token. Hybrid KV/recurrent programs also
 * require a published recurrent snapshot at the matched block boundary and
 * restore it with the KV blocks. Programs without KV blocks have no block
 * anchor and therefore no prefix match. This includes purely recurrent and
 * stateless graphs.
 *
 * Sessions are ordinary values and require no `Scope`. Sessions from the same
 * artifact may run concurrently and share pool capacity/cache content. Calls to
 * `add` and `step` on one session are serialized. `finish`, `cursor`, and
 * `close` are outside that JavaScript lock, so callers must not overlap them
 * with admission or stepping on the same session/sequence. Native sequence
 * locks are a safety backstop, not a supported concurrent lifecycle API.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface Generation {
  /**
   * Atomically admits a nonempty array of prompts. Capacity and policy are
   * validated for every entry before any sequence is allocated. Results preserve
   * input order and ordinary generation returns one token per page.
   */
  readonly add: (
    entries: ReadonlyArray<GenerationAdd>
  ) => Effect.Effect<ReadonlyArray<TokenPage>, InferenceError | Model.ModelError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Commits each selected sequence's pending token and samples one successor.
   * Terminal sequences fail validation before native execution.
   */
  readonly step: (
    entries: ReadonlyArray<GenerationStep>
  ) => Effect.Effect<ReadonlyArray<TokenPage>, InferenceError | Tensor.TensorError, Runtime.Runtime>
  /**
   * Returns this session's JavaScript live-sequence count. This is not a pool
   * capacity, global-session, or prefix-cache statistic.
   */
  readonly live: () => Effect.Effect<number>
  /**
   * Closes the native session and releases all live sequences atomically. A
   * successful close invalidates previously returned sequences and the session
   * accepts no later additions or rounds. Native finalizers remain a fallback.
   */
  readonly close: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * A caller-driven stateful sequence used only by {@link StatefulExecution}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface StatefulExecutionSeq {
  /** Runtime discriminant for a caller-token execution sequence. */
  readonly _tag: "StatefulExecutionSeq"
  /** Underlying decode-state sequence handle. */
  readonly sequence: Tensor.KvSequence
  /** Returns the sequence's absolute logical token count. */
  readonly cursor: () => Effect.Effect<number, Tensor.TensorError, Runtime.Runtime>
  /** Releases this sequence; repeated calls are no-ops. */
  readonly finish: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * Lower-level stateful logits execution for custom host samplers. Unlike
 * {@link Generation}, callers select tokens and own each returned logits row.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface StatefulExecution {
  /** Prefills nonempty prompts and returns one sequence and caller-owned logits row per prompt. */
  readonly add: (
    prompts: ReadonlyArray<Tensor.Any>
  ) => Effect.Effect<
    ReadonlyArray<{
      readonly seq: StatefulExecutionSeq
      readonly logits: Tensor.Concrete
    }>,
    InferenceError | Model.ModelError | Tensor.TensorError,
    Runtime.Runtime
  >
  /** Commits one caller-selected token per sequence and returns caller-owned successor logits. */
  readonly step: (
    entries: ReadonlyArray<{
      readonly seq: StatefulExecutionSeq
      readonly token: number
    }>
  ) => Effect.Effect<ReadonlyArray<Tensor.Concrete>, InferenceError | Tensor.TensorError, Runtime.Runtime>
  /** Returns this session's current live-sequence count. */
  readonly live: () => Effect.Effect<number>
  /** Closes the session and releases all of its live sequences. */
  readonly close: () => Effect.Effect<void, Tensor.TensorError, Runtime.Runtime>
}

/**
 * An immutable decode-specialized artifact. It retains one materialized
 * parameter generation as native constants, fixed prefill/decode executables,
 * and one shared decode-state pool. Its KV arenas and prefix cache are shared
 * across sessions, while each sequence owns its mutable recurrent state. It is neither
 * a {@link Model.Definition} nor part of `Model.Definition.execute`'s signature cache.
 *
 * The artifact is safe to share: immutable programs can run concurrently and
 * different sessions coordinate through the native pool. It has no explicit
 * release or `Scope` lifetime. Programs, frozen constants, and pool storage are
 * finalized when the artifact and dependent sequence handles become
 * unreachable. Sequence state is the capacity-sensitive resource that callers
 * can release deterministically through {@link GenerationSeq.finish} or
 * {@link Generation.close}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface Artifact {
  /**
   * Opens an empty caller-scheduled session. This allocates JavaScript
   * coordination state, not a private pool; all sessions share the artifact's
   * pool capacity and prefix cache. No `Scope` service is required. Use
   * {@link Generation.close} for deterministic session cleanup or
   * {@link GenerationSeq.finish} for one sequence.
   */
  readonly generation: () => Effect.Effect<Generation, InferenceError>
  /** Opens a lower-level caller-token/caller-owned-logits session. */
  readonly execution: () => Effect.Effect<StatefulExecution, InferenceError>
  /** Native generation counters, phase timings, acceptance, and pool pressure. */
  readonly diagnostics: () => Effect.Effect<Runtime.InferenceDiagnostics, Tensor.TensorError>
}

/**
 * Single-prompt generation with callback backpressure. Sampling overrides apply
 * to every page. The prompt is borrowed for the call; the session is closed on
 * completion, failure, callback failure, or interruption.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerateOptions<E = never, R = never> {
  readonly maxTokens: number
  readonly eosTokens: ReadonlyArray<number>
  readonly sampling?: Partial<GenerationSamplingOptions>
  readonly onPage: (tokens: ReadonlyArray<number>) => Effect.Effect<void, E, R>
}

/**
 * Number of published tokens and the session's terminal reason.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerationResult {
  readonly generatedTokens: number
  readonly stop: "eos" | "maxTokens"
}

/**
 * Generate from an existing artifact with deterministic session cleanup. This
 * convenience driver shares the artifact with independently derived decision
 * scorers and does not load, compile, or own its model parameters.
 *
 * @since 0.1.0
 * @category generation
 */
export const generate = <E = never, R = never>(
  program: Artifact,
  prompt: Tensor.Any,
  options: GenerateOptions<E, R>
): Effect.Effect<GenerationResult, E | InferenceError | Model.ModelError | Tensor.TensorError, R | Runtime.Runtime> =>
  Effect.scoped(Effect.gen(function*() {
    const session = yield* Effect.acquireRelease(
      program.generation(),
      (session) => Effect.orDie(session.close()),
      { interruptible: true }
    )

    const sampling = options.sampling ?? {}
    let pages = yield* session.add([{
      prompt,
      maxTokens: options.maxTokens,
      eosTokens: options.eosTokens,
      sampling
    }])
    let generatedTokens = 0

    while (true) {
      const page = pages[0]!
      yield* options.onPage(page.tokens)
      generatedTokens += page.tokens.length

      if (page.stopReason !== undefined) return { generatedTokens, stop: page.stopReason }

      pages = yield* session.step([{ seq: page.seq, sampling }])
    }
  }))

interface ResolvedCompileOptions {
  readonly maxTokens: number
  readonly blockSize: number
  readonly prefillChunks: ReadonlyArray<number>
  readonly tokenDtype: "u32" | "i64"
  readonly kvDtype: Tensor.DType
  readonly batchSize: number
  readonly sampling: GenerationSamplingOptions
  readonly attentionWindow: number | undefined
  readonly speculation: {
    readonly proposer: Speculation.Proposer
    readonly maxDraftTokens: number
  } | undefined
}

const invalidCompileOptions = (message: string): InferenceError => new InferenceError({ op: "inference", message })

const resolveCompileOptions = (
  config: CompileOptions
): Effect.Effect<ResolvedCompileOptions, InferenceError> =>
  Effect.gen(function*() {
    const blockSize = config.blockSize ?? 16

    if (!Number.isInteger(blockSize) || blockSize <= 0) {
      return yield* invalidCompileOptions(`blockSize must be a positive integer, got ${config.blockSize}`)
    }

    if (
      !Number.isInteger(config.maxTokens) || config.maxTokens <= 0 || config.maxTokens % blockSize !== 0
    ) {
      return yield* invalidCompileOptions(
        `maxTokens must be a positive multiple of blockSize ${blockSize}, got ${config.maxTokens}`
      )
    }

    if (
      config.attentionWindow !== undefined &&
      (!Number.isInteger(config.attentionWindow) || config.attentionWindow <= 0 ||
        config.attentionWindow > config.maxTokens)
    ) {
      return yield* invalidCompileOptions(
        `attentionWindow must be a positive integer no greater than maxTokens, got ${config.attentionWindow}`
      )
    }

    if (
      config.prefillChunks.length === 0 ||
      config.prefillChunks.some((chunk) => !Number.isSafeInteger(chunk) || chunk <= 0)
    ) {
      return yield* invalidCompileOptions(
        `prefillChunks must be positive safe integers, got [${config.prefillChunks}]`
      )
    }

    const prefillChunks = [...new Set(config.prefillChunks)].sort((left, right) => left - right)
    const tokenDtype = config.tokenDtype ?? "u32"

    if (tokenDtype !== "u32" && tokenDtype !== "i64") {
      return yield* invalidCompileOptions(`tokenDtype must be u32 or i64, got ${String(config.tokenDtype)}`)
    }

    const configuredKvDtype = config.kvDtype ?? "f32"

    if (!["f32", "f16", "bf16", "int8"].includes(configuredKvDtype)) {
      return yield* invalidCompileOptions(`unsupported kvDtype ${String(config.kvDtype)}`)
    }

    const batchSize = config.batchSize ?? 8

    if (!Number.isInteger(batchSize) || batchSize <= 0) {
      return yield* invalidCompileOptions(`batchSize must be a positive integer, got ${config.batchSize}`)
    }

    const sampling = config.sampling ?? { seed: 0 }

    if (
      (!Predicate.isBigInt(sampling.seed) && !Number.isSafeInteger(sampling.seed)) || sampling.seed < 0 ||
      BigInt(sampling.seed) > 0xffff_ffff_ffff_ffffn
    ) {
      return yield* invalidCompileOptions(`sampling.seed must be an unsigned 64-bit integer, got ${sampling.seed}`)
    }

    if (sampling.temperature !== undefined && (!Number.isFinite(sampling.temperature) || sampling.temperature < 0)) {
      return yield* invalidCompileOptions(
        `sampling.temperature must be finite and non-negative, got ${sampling.temperature}`
      )
    }

    if (sampling.topK !== undefined && (!Number.isSafeInteger(sampling.topK) || sampling.topK < 0)) {
      return yield* invalidCompileOptions(`sampling.topK must be a non-negative safe integer, got ${sampling.topK}`)
    }

    if (sampling.topP !== undefined && (!Number.isFinite(sampling.topP) || sampling.topP <= 0 || sampling.topP > 1)) {
      return yield* invalidCompileOptions(`sampling.topP must be in (0, 1], got ${sampling.topP}`)
    }

    let speculation: ResolvedCompileOptions["speculation"]

    if (config.speculation !== undefined) {
      const proposer = config.speculation.proposer

      if (
        !Predicate.isObjectOrArray(proposer) ||
        !["Autoregressive", "HistoryLookup", "ParallelBlock"].includes(proposer._tag)
      ) {
        return yield* invalidCompileOptions("speculation.proposer is not a supported speculation artifact")
      }

      if (
        !Number.isSafeInteger(config.speculation.maxDraftTokens) || config.speculation.maxDraftTokens <= 0 ||
        config.speculation.maxDraftTokens > proposer.maxDraftTokens
      ) {
        return yield* invalidCompileOptions(
          `maxDraftTokens must be in [1, ${proposer.maxDraftTokens}], got ${config.speculation.maxDraftTokens}`
        )
      }

      if (config.speculation.schedule === "adaptive") {
        return yield* invalidCompileOptions("adaptive speculative scheduling is not implemented; use fixed")
      }

      if (config.attentionWindow !== undefined) {
        return yield* invalidCompileOptions("speculative execution does not yet support attentionWindow")
      }

      speculation = { proposer, maxDraftTokens: config.speculation.maxDraftTokens }
    }

    return {
      maxTokens: config.maxTokens,
      blockSize,
      prefillChunks,
      tokenDtype,
      kvDtype: configuredKvDtype === "int8" ? "u8" : configuredKvDtype,
      batchSize,
      sampling,
      attentionWindow: config.attentionWindow,
      speculation
    }
  })

interface Artifacts {
  readonly prefill: ReadonlyArray<Tensor.DecodeProgram>
  readonly decode: Tensor.DecodeProgram
  readonly geometry: Runtime.DecodeStateSchema
  readonly pool: Tensor.KvPool
  readonly speculation?: {
    /** Verify programs per packed rows-per-sequence width, ascending. */
    readonly verify: ReadonlyArray<Tensor.DecodeProgram>
    readonly maxDraftTokens: number
    readonly proposer?: {
      readonly prefill: Tensor.DecodeProgram
      readonly decode: Tensor.DecodeProgram
      readonly pool: Tensor.KvPool
    }
    readonly generalized?: NonNullable<Runtime.InferenceCompileRequest["generalizedProposer"]>
  }
}

/** Speculative-plan verify widths, compiled in ascending order. */
const verifyWidths = (maxDraftTokens: number): ReadonlyArray<number> => {
  const widest = maxDraftTokens + 1

  // M=8 and M=16 have dedicated Metal MMA paths. With a 15-token DFlash
  // block, keep both so the runtime can widen only for high-acceptance
  // sessions; non-aligned narrow widths remain slower than M=8.
  if (widest === 16) return [8, 16]

  return [widest]
}

const logitsVocab = (
  output: Tensor.Any,
  batch: number,
  steps: number
): Effect.Effect<number, InferenceError> => {
  const expected = [batch, steps]

  if (output.shape.length !== 3 || output.shape[0] !== expected[0] || output.shape[1] !== expected[1]) {
    return new InferenceError({
      op: "inference",
      message: `model output must be [${batch}, ${steps}, vocab], got [${output.shape}]`
    })
  }

  return Effect.succeed(output.shape[2]!)
}

const schemaShapeMatches = (
  declared: ReadonlyArray<number | "Rows">,
  actual: ReadonlyArray<number | "Rows">
): boolean => declared.length === actual.length && declared.every((dimension, index) => dimension === actual[index])

const validateTargetContract = (
  model: Model.Definition,
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedCompileOptions,
  vocabulary: number
): Effect.Effect<void, InferenceError | Model.ModelError | Tensor.TensorError, Runtime.Runtime> => {
  const speculation = config.speculation

  if (speculation === undefined) return Effect.void

  const proposer = speculation.proposer

  return Effect.gen(function*() {
    if (proposer.vocabulary !== vocabulary) {
      return yield* invalidCompileOptions(
        `proposer target vocabulary must be ${proposer.vocabulary}, got ${vocabulary}`
      )
    }

    if (proposer._tag !== "ParallelBlock") return

    for (const weight of [proposer.tokenEmbedding, proposer.lmHead]) {
      const index = model.parameterSpecs.findIndex((parameter) => parameter.name === weight.name)
      const value = index < 0 ? undefined : frozenParams[index]

      if (
        value === undefined || value.dtype !== weight.dtype ||
        !schemaShapeMatches(weight.shape, value.shape)
      ) {
        const actual = value === undefined ? "missing" : `${value.dtype}[${value.shape}]`

        return yield* invalidCompileOptions(
          `proposer target shared weight ${
            JSON.stringify(weight.name)
          } requires ${weight.dtype}[${weight.shape}], got ${actual}`
        )
      }
    }
  })
}

interface TracedArtifact {
  readonly program: Tensor.DecodeProgram
  readonly taps: ReadonlyArray<Runtime.InferenceTargetTapRoute>
}

const traceArtifact = (
  model: Model.Definition,
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedCompileOptions,
  inputShape: readonly [number, number],
  lastTokenRow = true,
  packedCausalChains?: Runtime.PackedCausalChainsLayout,
  taps: ReadonlyArray<Speculation.HiddenTap> = []
): Effect.Effect<
  TracedArtifact,
  InferenceError | Model.ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    const [graphRows, steps] = inputShape
    const tokenInput = yield* Tensor.zeros(inputShape, { dtype: config.tokenDtype })
    const input = yield* Tensor.makeInput(0, tokenInput)
    const output = yield* model.forward(frozenParams, input)
    yield* logitsVocab(output, graphRows, steps)
    // Exposures live in the graph itself (Tensor.expose identity nodes), so
    // composition can never drop them; discovery is one walk from the root.
    const runtime = yield* Runtime.Runtime

    const discovered = yield* runtime.exposures(output).pipe(
      Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message }))
    )

    const exposed = new Map(discovered.map((entry) => [entry.name, entry.tensor]))
    const roots: Array<Tensor.Any> = [output]
    const routes: Array<Runtime.InferenceTargetTapRoute> = []

    for (const tap of taps) {
      const value = exposed.get(tap.name)

      const logicalShape: ReadonlyArray<number | "Rows"> | undefined = value === undefined ||
          value.shape.length < 2 || value.shape[0] !== graphRows || value.shape[1] !== steps
        ? undefined
        : ["Rows", ...value.shape.slice(2)]

      if (
        value === undefined || logicalShape === undefined || value.dtype !== tap.dtype ||
        !schemaShapeMatches(tap.shape, logicalShape)
      ) {
        const available = discovered.map((entry) => entry.name).sort()

        const actual = value === undefined
          ? `missing; model exposes ${available.length === 0 ? "nothing" : available.join(", ")}`
          : `${value.dtype}[${value.shape}]`

        return yield* invalidCompileOptions(
          `proposer target hidden tap "${tap.name}" requires ${tap.dtype}[${tap.shape}], got ${actual}`
        )
      }

      roots.push(value)
      routes.push({
        name: tap.name,
        outputRoot: roots.length - 1,
        value: { dtype: value.dtype, shape: value.shape }
      })
    }

    const program = yield* Tensor.compileDecodeProgram(roots, {
      access: "Append",
      maxTokens: config.maxTokens,
      blockSize: config.blockSize,
      kvDtype: config.kvDtype,
      batch: packedCausalChains === undefined ? graphRows : config.batchSize,
      ...(taps.length === 0
        ? { lastTokenRow }
        : {
          outputSelections: [
            lastTokenRow ? "splitLastTokenRow" as const : "allRows" as const,
            ...taps.map(() => "allRows" as const)
          ]
        }),
      packedCausalChains,
      window: config.attentionWindow
    }).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))

    return { program, taps: routes }
  })

const compileProposerPlan = (
  proposer: Speculation.HistoryLookup | Speculation.ParallelBlock,
  proposerParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedCompileOptions,
  vocabulary: number,
  targetTaps: {
    /** Target hidden taps per prefill chunk shape, in ascending shape order. */
    readonly prefill: ReadonlyArray<ReadonlyArray<Runtime.InferenceTargetTapRoute>>
    readonly decode: ReadonlyArray<Runtime.InferenceTargetTapRoute>
    /** Target hidden taps per verify width, in ascending width order. */
    readonly verify: ReadonlyArray<{
      readonly width: number
      readonly taps: ReadonlyArray<Runtime.InferenceTargetTapRoute>
    }>
  },
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  targetNames: ReadonlyArray<string>
): Effect.Effect<
  NonNullable<Runtime.InferenceCompileRequest["generalizedProposer"]>,
  InferenceError | Model.ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    if (proposer._tag === "HistoryLookup") {
      const plan: Runtime.InferenceProposerPlan = {
        vocabulary,
        tokenMapFingerprint: "identity",
        hiddenTaps: [],
        sharedTensors: [],
        stages: [{
          operationId: "HistoryLookup",
          layoutId: "suffix-ngram-v1",
          historyLookup: {
            id: "suffix-ngram-v1",
            minMatchTokens: proposer.minMatchTokens,
            maxMatchTokens: proposer.maxMatchTokens
          },
          inputs: [],
          outputs: [{ dtype: config.tokenDtype, shape: [config.batchSize * config.speculation!.maxDraftTokens] }]
        }],
        state: { kind: "None", commitKind: "None", commitStages: [] },
        output: {
          topology: "Chains",
          probabilities: "Deterministic",
          tokenIds: { kind: "StageOutput", stage: 0, output: 0 }
        },
        tokenMap: { kind: "Identity", fingerprint: "identity" },
        trainedMaxRows: config.speculation!.maxDraftTokens
      }

      return { plan, sharedTensors: [], stageExecutables: [], maxDraftTokens: config.speculation!.maxDraftTokens }
    }

    const sharedTensors: Array<Tensor.Concrete> = []
    const sharedMetadata: Array<Runtime.InferenceProposerPlan["sharedTensors"][number]> = []

    for (
      const [kind, weight] of [
        ["TokenEmbedding", proposer.tokenEmbedding],
        ["LmHead", proposer.lmHead]
      ] as const
    ) {
      const actual = frozenParams[targetNames.indexOf(weight.name)]

      if (actual === undefined) {
        return yield* invalidCompileOptions(`proposer target shared weight ${JSON.stringify(weight.name)} is missing`)
      }

      sharedTensors.push(actual)
      sharedMetadata.push({
        kind,
        name: weight.name,
        value: { dtype: actual.dtype, shape: actual.shape }
      })
    }

    const anchorSchema: Runtime.InferenceValueSchema = { dtype: config.tokenDtype, shape: [config.batchSize] }
    const anchor = yield* Tensor.makeInput(0, yield* Tensor.zeros(anchorSchema.shape, { dtype: anchorSchema.dtype }))
    const embedding = sharedTensors[0]!
    const head = sharedTensors[1]!

    // Parallel-block drafters are trained with a fixed physical block width.
    // The requested speculative width selects a candidate prefix; it must not
    // shrink the diffusion block and change proposal conditioning.
    const output = proposer.buildWithProbabilities === undefined
      ? {
        tokenIds: yield* proposer.build(
          proposerParams,
          anchor,
          yield* Tensor.makeInput(1, embedding),
          yield* Tensor.makeInput(2, head),
          proposer.maxDraftTokens
        )
      }
      : yield* proposer.buildWithProbabilities(
        proposerParams,
        anchor,
        yield* Tensor.makeInput(1, embedding),
        yield* Tensor.makeInput(2, head),
        proposer.maxDraftTokens
      )

    const expectedShape = [config.batchSize, proposer.maxDraftTokens]

    if (output.tokenIds.dtype !== "u32" || !schemaShapeMatches(expectedShape, output.tokenIds.shape)) {
      return yield* invalidCompileOptions(
        `parallel block output requires u32[${expectedShape}], got ${output.tokenIds.dtype}[${output.tokenIds.shape}]`
      )
    }

    const expectedProbabilityShape = [config.batchSize, proposer.maxDraftTokens, vocabulary]

    if (
      output.probabilityRows !== undefined &&
      (output.probabilityRows.dtype !== "f32" ||
        !schemaShapeMatches(expectedProbabilityShape, output.probabilityRows.shape))
    ) {
      return yield* invalidCompileOptions(
        `parallel block probabilities require f32[${expectedProbabilityShape}], got ${output.probabilityRows.dtype}[${output.probabilityRows.shape}]`
      )
    }

    const roots = output.probabilityRows === undefined
      ? [output.tokenIds]
      : [output.tokenIds, output.probabilityRows]

    const program = yield* Tensor.compileDecodeProgram(roots, {
      access: "Append",
      maxTokens: config.maxTokens,
      blockSize: config.blockSize,
      kvDtype: config.kvDtype,
      batch: config.batchSize,
      currentBlockAttention: proposer.currentBlockAttention ?? "Causal",
      window: proposer.attentionWindow
    }).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))

    const compileReplay = (
      taps: ReadonlyArray<Runtime.InferenceTargetTapRoute>,
      packedRows?: number
    ) =>
      Effect.gen(function*() {
        const targetRows: Array<Tensor.Any> = []

        for (let index = 0; index < taps.length; index++) {
          const input = yield* Tensor.zeros(taps[index]!.value.shape, { dtype: taps[index]!.value.dtype })
          const routed = yield* Tensor.makeInput(index, input)
          targetRows.push(
            packedRows === undefined
              ? routed
              : yield* Tensor.reshape(routed, [config.batchSize, packedRows, ...routed.shape.slice(2)])
          )
        }

        const keyValues = yield* proposer.replay(proposerParams, targetRows)
        const roots: Array<Tensor.Any> = []

        // Decode specialization assigns independent state roots from last to first.
        // Reverse replay roots so semantic proposer layer N writes KV cache layer N.
        for (const { key, value } of [...keyValues].reverse()) {
          // Stateful attention appends K/V transactionally and applies cursor-relative
          // transforms. Replay discards the query outputs.
          roots.push(
            yield* Tensor.scaledDotProductAttention(yield* Tensor.zerosLike(key), key, value, {
              causal: true,
              scale: 1
            })
          )
        }

        return yield* Tensor.compileDecodeProgram(roots, {
          access: "Append",
          maxTokens: config.maxTokens,
          blockSize: config.blockSize,
          kvDtype: config.kvDtype,
          batch: config.batchSize,
          outputSelections: roots.map(() => "allRows" as const),
          window: proposer.attentionWindow
        }).pipe(Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message })))
      })

    const replayPrefills: Array<Tensor.DecodeProgram> = []

    for (const taps of targetTaps.prefill) {
      replayPrefills.push(yield* compileReplay(taps))
    }

    const replayDecode = yield* compileReplay(targetTaps.decode)
    // One replay program per verify width: tap row counts follow the width.
    const replayVerifies: Array<Tensor.DecodeProgram> = []

    for (const { width, taps } of targetTaps.verify) {
      replayVerifies.push(yield* compileReplay(taps, width))
    }

    const replayGeometry = replayPrefills[replayPrefills.length - 1]!

    if (
      replayPrefills.some((program) => !Runtime.sameDecodeStateSchema(replayGeometry, program)) ||
      !Runtime.sameDecodeStateSchema(replayGeometry, replayDecode) ||
      replayVerifies.some((program) => !Runtime.sameDecodeStateSchema(replayGeometry, program)) ||
      !Runtime.sameDecodeStateSchema(replayGeometry, program)
    ) {
      return yield* invalidCompileOptions("parallel block and replay graphs disagree on state geometry")
    }

    const pool = yield* Tensor.makeKvPoolFromSchema(replayGeometry).pipe(
      Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message }))
    )

    const inputs: Runtime.InferenceProposerPlan["stages"][number]["inputs"] = [
      { slot: 0, value: { kind: "PendingTokens", value: anchorSchema } },
      { slot: 1, value: { kind: "SharedTokenEmbedding" } },
      { slot: 2, value: { kind: "SharedLmHead" } }
    ]

    const plan: Runtime.InferenceProposerPlan = {
      vocabulary,
      tokenMapFingerprint: "identity",
      hiddenTaps: targetTaps.decode,
      prefillHiddenTaps: targetTaps.prefill[targetTaps.prefill.length - 1]!,
      // Tap routes are width-independent in root order; the widest width's
      // metadata validates the plan.
      verifyHiddenTaps: targetTaps.verify[targetTaps.verify.length - 1]!.taps,
      sharedTensors: sharedMetadata,
      stages: [{
        operationId: "ParallelBlock",
        layoutId: "parallel-block",
        inputs,
        outputs: roots.map((root) => ({ dtype: root.dtype, shape: root.shape }))
      }],
      state: {
        kind: "Kv",
        schemaId: "parallel-block-kv",
        commitKind: "Replay",
        commitStages: [0]
      },
      output: {
        topology: "Chains",
        probabilities: output.probabilityRows === undefined ? "Unavailable" : "CausalNormalized",
        tokenIds: { kind: "StageOutput", stage: 0, output: 0 },
        probabilityRows: output.probabilityRows === undefined
          ? undefined
          : { kind: "StageOutput" as const, stage: 0, output: 1 }
      },
      tokenMap: { kind: "Identity", fingerprint: "identity" },
      trainedMaxRows: proposer.maxDraftTokens
    }

    return {
      plan,
      sharedTensors,
      stageExecutables: [program.handle],
      replay: {
        prefill: replayPrefills.map((program) => program.handle),
        decode: replayDecode.handle,
        verify: replayVerifies.map((program) => program.handle),
        pool: pool.handle
      },
      maxDraftTokens: config.speculation!.maxDraftTokens
    }
  })

const compileArtifacts = (
  model: Model.Definition,
  frozenParams: ReadonlyArray<Tensor.Concrete>,
  config: ResolvedCompileOptions,
  proposerParams: ReadonlyArray<Tensor.Concrete> | undefined
): Effect.Effect<
  Artifacts,
  InferenceError | Model.ModelError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    const proposer = config.speculation?.proposer
    const taps = proposer?._tag === "ParallelBlock" ? proposer.hiddenTaps : []
    // One prefill program per compiled chunk width, ascending; the runtime
    // serves each prompt chunk from the largest width covering its remaining
    // tokens and skips the LM-head chain for non-final chunks.
    const prefillTraces: Array<TracedArtifact> = []

    for (const chunk of config.prefillChunks) {
      prefillTraces.push(
        yield* traceArtifact(
          model,
          frozenParams,
          config,
          [config.batchSize, chunk],
          true,
          undefined,
          taps
        )
      )
    }

    const decodeTrace = yield* traceArtifact(
      model,
      frozenParams,
      config,
      [config.batchSize, 1],
      true,
      undefined,
      taps
    )

    const prefill = prefillTraces.map((trace) => trace.program)
    const decode = decodeTrace.program
    const geometry = prefill[prefill.length - 1]!

    if (
      prefill.some((program) => !Runtime.sameDecodeStateSchema(geometry, program)) ||
      !Runtime.sameDecodeStateSchema(geometry, decode)
    ) {
      return yield* new InferenceError({
        op: "inference",
        message: "prefill and decode traces disagree on attention geometry or retention policy"
      })
    }

    const targetVocabulary = decode.outputs[0]?.shape[0]

    if (targetVocabulary === undefined) {
      return yield* new InferenceError({ op: "inference", message: "target decode did not expose a vocabulary row" })
    }

    yield* validateTargetContract(model, frozenParams, config, targetVocabulary)

    const pool = yield* Tensor.makeKvPoolFromSchema(geometry).pipe(
      Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message }))
    )

    if (config.speculation === undefined) {
      return { prefill, decode, geometry, pool }
    }

    if (proposerParams === undefined) {
      return yield* new InferenceError({ op: "inference", message: "speculative proposer parameters are missing" })
    }

    if (geometry.layers === 0 || geometry.kdaLayers !== 0 || geometry.convLayers !== 0) {
      return yield* new InferenceError({
        op: "inference",
        message: "speculative target state must be KV-only with at least one attention layer"
      })
    }

    if (proposer?._tag !== "Autoregressive") {
      // ParallelBlock compiles one verify program per packed width and the
      // runtime adaptively selects the width per round from measured token
      // rates; HistoryLookup verifies full-width (its drafts are free).
      const widths = proposer?._tag === "ParallelBlock"
        ? verifyWidths(config.speculation.maxDraftTokens)
        : [config.speculation.maxDraftTokens + 1]

      const verifyTraces: Array<TracedArtifact> = []

      for (const width of widths) {
        verifyTraces.push(
          yield* traceArtifact(
            model,
            frozenParams,
            config,
            [config.batchSize * width, 1],
            false,
            { rowsPerSequence: width },
            taps
          )
        )
      }

      const generalized = yield* compileProposerPlan(
        proposer!,
        proposerParams,
        config,
        targetVocabulary,
        {
          prefill: prefillTraces.map((trace) => trace.taps),
          decode: decodeTrace.taps,
          verify: verifyTraces.map((trace, index) => ({ width: widths[index]!, taps: trace.taps }))
        },
        frozenParams,
        model.parameterSpecs.map((parameter) => parameter.name)
      )

      return {
        prefill,
        decode,
        geometry,
        pool,
        speculation: {
          verify: verifyTraces.map((trace) => trace.program),
          maxDraftTokens: config.speculation.maxDraftTokens,
          generalized
        }
      }
    }

    const proposerModel = proposer.model
    const exactParams = proposerParams

    const proposerPrefill = yield* traceArtifact(
      proposerModel,
      exactParams,
      config,
      [config.batchSize, config.prefillChunks[config.prefillChunks.length - 1]!]
    ).pipe(Effect.map((trace) => trace.program))

    const proposerDecode = yield* traceArtifact(
      proposerModel,
      exactParams,
      config,
      [config.batchSize, 1]
    ).pipe(Effect.map((trace) => trace.program))

    const proposerGeometry = proposerPrefill

    if (!Runtime.sameDecodeStateSchema(proposerGeometry, proposerDecode)) {
      return yield* new InferenceError({
        op: "inference",
        message: "proposer prefill and decode traces disagree on state geometry"
      })
    }

    if (proposerGeometry.layers === 0 || proposerGeometry.kdaLayers !== 0 || proposerGeometry.convLayers !== 0) {
      return yield* new InferenceError({
        op: "inference",
        message: "speculative proposer state must be KV-only with at least one attention layer"
      })
    }

    const proposerVocabulary = proposerDecode.outputs[0]?.shape[0]

    if (
      targetVocabulary !== proposer.vocabulary ||
      proposerVocabulary !== targetVocabulary
    ) {
      return yield* new InferenceError({
        op: "inference",
        message:
          `speculative identity token map requires target/proposer vocabulary ${proposer.vocabulary}, got target ${targetVocabulary} and proposer ${proposerVocabulary}`
      })
    }

    const verify = yield* traceArtifact(
      model,
      frozenParams,
      config,
      [config.batchSize * (config.speculation.maxDraftTokens + 1), 1],
      false,
      { rowsPerSequence: config.speculation.maxDraftTokens + 1 }
    ).pipe(Effect.map((trace) => trace.program))

    if (!Runtime.sameDecodeStateSchema(geometry, verify)) {
      return yield* new InferenceError({
        op: "inference",
        message: "target verification trace disagrees with target decode state geometry"
      })
    }

    const proposerPool = yield* Tensor.makeKvPoolFromSchema(proposerGeometry).pipe(
      Effect.mapError((error) => new InferenceError({ op: "inference", message: error.message }))
    )

    return {
      prefill,
      decode,
      geometry,
      pool,
      speculation: {
        // Exact proposers keep a single full-width verify program.
        verify: [verify],
        maxDraftTokens: config.speculation.maxDraftTokens,
        proposer: { prefill: proposerPrefill, decode: proposerDecode, pool: proposerPool }
      }
    }
  })

interface PrefillChunkPlan {
  readonly offset: number
  readonly real: number
  readonly final: boolean
}

// This checks only the public add calling convention. Reading or execution
// later validates token values and model vocabulary and position bounds.
const validatePrompt = (
  prompt: Tensor.Any,
  config: ResolvedCompileOptions,
  runtime: Runtime.RuntimeService
): Effect.Effect<void, InferenceError> => {
  if (prompt.placement.id !== runtime.placement.id) {
    return new InferenceError({ op: "add", message: "prompt must use the inference program runtime and placement" })
  }

  if (prompt.dtype !== config.tokenDtype) {
    return new InferenceError({
      op: "add",
      message: `prompt dtype must be ${config.tokenDtype}, got ${prompt.dtype}`
    })
  }

  if (prompt.shape.length !== 2 || prompt.shape[0] !== 1 || prompt.shape[1]! < 1) {
    return new InferenceError({
      op: "add",
      message: `add expects a prompt of shape [1, T] with T >= 1, got [${prompt.shape}]`
    })
  }

  return Effect.void
}

const readTokenIds = (tokens: Tensor.Any): Effect.Effect<Array<number>, InferenceError, Runtime.Runtime> => {
  const read = tokens.dtype === "i64"
    ? Effect.gen(function*() {
      const values = yield* Tensor.toTypedArray(tokens)
      const ids: Array<number> = []

      for (const value of values) {
        if (!Predicate.isBigInt(value) || value < 0n || value > 0xffff_ffffn) {
          return yield* new InferenceError({
            op: "prefill",
            message: `token ids must fit u32 for decode state, got ${String(value)}`
          })
        }

        ids.push(Number(value))
      }

      return ids
    })
    : Tensor.toNumberArray(tokens)

  return Effect.mapError(read, (error) =>
    error instanceof InferenceError
      ? error
      : new InferenceError({ op: "prefill", message: `token ids must be readable integers: ${error.message}` }))
}

const tokenTensor = (
  ids: ReadonlyArray<number>,
  shape: ReadonlyArray<number>,
  dtype: "u32" | "i64"
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  Tensor.fromTypedArray(dtype === "i64" ? BigInt64Array.from(ids.map(BigInt)) : Uint32Array.from(ids), shape)

const slottedTokenTensor = (
  ids: ReadonlyArray<number>,
  slots: ReadonlyArray<number>,
  batchSize: number,
  dtype: "u32" | "i64"
): Effect.Effect<Tensor.Any, Tensor.TensorError, Runtime.Runtime> => {
  const values = Array<number>(batchSize).fill(0)

  for (const [index, slot] of slots.entries()) values[slot] = ids[index]!

  return tokenTensor(values, [batchSize, 1], dtype)
}

interface PrefillLane {
  readonly slot: number
  readonly sequence: Tensor.KvSequence
  readonly tokens: ReadonlyArray<number>
  offset: number
}

interface PrefillRoundLane extends PrefillLane {
  readonly chunk: PrefillChunkPlan
}

const slottedPrefillTensor = (
  lanes: ReadonlyArray<PrefillRoundLane>,
  config: ResolvedCompileOptions
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> => {
  // The generic session driver always runs the largest compiled chunk.
  const prefillChunk = config.prefillChunks[config.prefillChunks.length - 1]!
  const values = Array<number>(config.batchSize * prefillChunk).fill(0)

  for (const lane of lanes) {
    const tokens = lane.tokens.slice(lane.chunk.offset, lane.chunk.offset + lane.chunk.real)

    for (const [index, token] of tokens.entries()) {
      values[lane.slot * prefillChunk + index] = token
    }
  }

  return tokenTensor(values, [config.batchSize, prefillChunk], config.tokenDtype)
}

const selectSlottedOutputs = (
  outputs: ReadonlyArray<Tensor.Concrete>,
  slots: ReadonlyArray<number>
): Effect.Effect<Array<Tensor.Concrete>, never, Runtime.Runtime> =>
  Effect.gen(function*() {
    const selected = slots.map((slot) => outputs[slot]!)
    const selectedSlots = new Set(slots)

    for (const [slot, output] of outputs.entries()) {
      if (!selectedSlots.has(slot)) yield* Tensor.clear(output)
    }

    return selected
  })

const runPrefillBatches = <A>(
  program: Tensor.DecodeProgram,
  config: ResolvedCompileOptions,
  lanes: ReadonlyArray<PrefillLane>,
  runFinal: (
    lanes: ReadonlyArray<PrefillRoundLane>,
    input: Tensor.Any,
    tokens: ReadonlyArray<ReadonlyArray<number>>
  ) => Effect.Effect<ReadonlyArray<A>, Tensor.TensorError, Runtime.Runtime>,
  clearFinalValues: (values: ReadonlyArray<A>) => Effect.Effect<void, never, Runtime.Runtime>
): Effect.Effect<ReadonlyArray<A>, InferenceError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.suspend(() => {
    const results = new Map<number, A>()

    return Effect.onExit(
      Effect.gen(function*() {
        while (results.size < lanes.length) {
          const round = lanes
            .filter((lane) => !results.has(lane.slot))
            .map((lane): PrefillRoundLane => {
              const real = Math.min(
                config.prefillChunks[config.prefillChunks.length - 1]!,
                lane.tokens.length - lane.offset
              )

              return {
                ...lane,
                chunk: { offset: lane.offset, real, final: lane.offset + real === lane.tokens.length }
              }
            })

          for (const final of [false, true]) {
            const group = round.filter((lane) => lane.chunk.final === final)

            if (group.length === 0) continue

            const input = yield* slottedPrefillTensor(group, config)

            const tokens = group.map((lane) =>
              lane.tokens.slice(lane.chunk.offset, lane.chunk.offset + lane.chunk.real)
            )

            if (final) {
              const values = yield* runFinal(group, input, tokens)

              if (values.length !== group.length) {
                yield* clearFinalValues(values)

                return yield* new InferenceError({
                  op: "prefill",
                  message: `prefill returned ${values.length} final values for ${group.length} lanes`
                })
              }

              for (const [index, lane] of group.entries()) results.set(lane.slot, values[index]!)
            } else {
              const outputs = yield* Tensor.runBatchedDecodeProgram(
                program,
                [input],
                group.map((lane) => lane.sequence),
                group.map((lane) => lane.slot),
                tokens
              )

              yield* Tensor.clearAll(outputs)
            }

            for (const lane of group) lanes.find((source) => source.slot === lane.slot)!.offset += lane.chunk.real
          }
        }

        return lanes.map((lane) => results.get(lane.slot)!)
      }),
      (exit) => Exit.isFailure(exit) ? clearFinalValues(Array.from(results.values())) : Effect.void
    )
  })

interface SessionSeq {
  readonly sequence: Tensor.KvSequence
}

interface LiveEntry<Seq extends SessionSeq> {
  readonly seq: Seq
  readonly slot: number
}

// Keep entries live until backend release succeeds so a failed or interrupted
// release remains retryable.
const releaseLiveEntry = <Seq extends SessionSeq>(live: Array<LiveEntry<Seq>>, entry: LiveEntry<Seq>) =>
  Effect.gen(function*() {
    const index = live.indexOf(entry)

    if (index < 0) return

    yield* Tensor.releaseKvSequence(entry.seq.sequence)
    live.splice(index, 1)
  })

const releaseLiveEntries = <Seq extends SessionSeq>(
  live: Array<LiveEntry<Seq>>,
  entries: ReadonlyArray<LiveEntry<Seq>>
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    let failure: Tensor.TensorError | undefined

    for (const entry of entries) {
      yield* Effect.matchEffect(releaseLiveEntry(live, entry), {
        onFailure: (error) =>
          Effect.sync(() => {
            failure ??= error
          }),
        onSuccess: () => Effect.void
      })
    }

    if (failure !== undefined) {
      return yield* Effect.fail(failure)
    }
  })

const closeLiveEntries = <Seq extends SessionSeq>(
  live: Array<LiveEntry<Seq>>
): Effect.Effect<void, Tensor.TensorError, Runtime.Runtime> => releaseLiveEntries(live, live.slice())

// The step semaphore does not cover lifecycle mutations. Generation requires
// callers to keep them disjoint.
const validateStepEntries = (
  live: ReadonlyArray<LiveEntry<StatefulExecutionSeq>>,
  batchSize: number,
  entries: ReadonlyArray<{
    readonly seq: StatefulExecutionSeq
    readonly token: number
  }>
): Effect.Effect<void, InferenceError> =>
  Effect.gen(function*() {
    if (entries.length === 0) {
      return yield* new InferenceError({ op: "step", message: "step expects at least one entry" })
    }

    if (entries.length > batchSize) {
      return yield* new InferenceError({
        op: "step",
        message: `step accepts at most batchSize (${batchSize}) entries, got ${entries.length}`
      })
    }

    for (const [index, entry] of entries.entries()) {
      if (!Number.isInteger(entry.token) || entry.token < 0) {
        return yield* new InferenceError({
          op: "step",
          message: `step expects token ids (non-negative integers), got ${entry.token}`
        })
      }

      if (!live.some((liveEntry) => liveEntry.seq === entry.seq)) {
        return yield* new InferenceError({
          op: "step",
          message: `entry ${index} is not a live sequence of this session`
        })
      }

      if (entries.findIndex((other) => other.seq === entry.seq) !== index) {
        return yield* new InferenceError({ op: "step", message: "step entries must be distinct sequences" })
      }
    }
  })

interface InferenceEngine {
  readonly config: ResolvedCompileOptions
  readonly programs: Artifacts
  readonly artifact: Runtime.InferenceArtifactHandle
  readonly runtime: Runtime.RuntimeService
}

const openStatefulExecution = (engine: InferenceEngine): Effect.Effect<StatefulExecution, never> =>
  Effect.gen(function*() {
    const roundLock = yield* Semaphore.make(1)
    const live: Array<LiveEntry<StatefulExecutionSeq>> = []
    const config = engine.config
    const programs = engine.programs

    const add: StatefulExecution["add"] = (prompts) =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          if (prompts.length === 0) {
            return yield* new InferenceError({ op: "add", message: "add expects at least one prompt" })
          }

          if (live.length + prompts.length > config.batchSize) {
            return yield* new InferenceError({
              op: "add",
              message: `add needs ${prompts.length} free lanes, but only ${config.batchSize - live.length} remain`
            })
          }

          const runtime = yield* Runtime.Runtime

          for (const prompt of prompts) yield* validatePrompt(prompt, config, runtime)

          const promptValues = yield* Tensor.compute(prompts)
          const sequences: Array<Tensor.KvSequence> = []

          const added: Array<{
            readonly seq: StatefulExecutionSeq
            readonly logits: Tensor.Concrete
          }> = []

          return yield* Effect.onExit(
            Effect.gen(function*() {
              const tokenRows: Array<ReadonlyArray<number>> = []

              for (const prompt of promptValues) tokenRows.push(yield* readTokenIds(prompt))

              const freeSlots = Array.from({ length: config.batchSize }, (_, slot) => slot)
                .filter((slot) => !live.some((entry) => entry.slot === slot))

              const lanes: Array<PrefillLane> = []

              for (const [index, tokens] of tokenRows.entries()) {
                const sequence = yield* Tensor.makeKvSequence(programs.pool)
                sequences.push(sequence)
                const matched = yield* Tensor.kvPrefillMatch(sequence, tokens)
                lanes.push({ slot: freeSlots[index]!, sequence, tokens, offset: matched })
              }

              const logits = yield* runPrefillBatches(
                programs.prefill[programs.prefill.length - 1]!,
                config,
                lanes,
                (finals, input, tokens) =>
                  Effect.flatMap(
                    Tensor.runBatchedDecodeProgram(
                      programs.prefill[programs.prefill.length - 1]!,
                      [input],
                      finals.map((lane) => lane.sequence),
                      finals.map((lane) => lane.slot),
                      tokens
                    ),
                    (outputs) => selectSlottedOutputs(outputs, finals.map((lane) => lane.slot))
                  ),
                Tensor.clearAll
              )

              yield* Effect.sync(() => {
                for (const [index, lane] of lanes.entries()) {
                  let entry: LiveEntry<StatefulExecutionSeq>

                  const seq: StatefulExecutionSeq = {
                    _tag: "StatefulExecutionSeq",
                    sequence: lane.sequence,
                    cursor: () => Tensor.kvSequenceCursor(lane.sequence),
                    finish: () => releaseLiveEntry(live, entry)
                  }

                  entry = { seq, slot: lane.slot }
                  live.push(entry)
                  added.push({ seq, logits: logits[index]! })
                }
              })

              return added
            }),
            (exit) =>
              Effect.gen(function*() {
                yield* Tensor.clearAll(promptValues)

                if (Exit.isFailure(exit)) {
                  yield* Tensor.clearAll(added.map((entry) => entry.logits))

                  for (const sequence of sequences) {
                    const entry = live.find((entry) => entry.seq.sequence === sequence)

                    if (entry === undefined) {
                      yield* Tensor.releaseKvSequence(sequence)
                    } else {
                      yield* releaseLiveEntry(live, entry)
                    }
                  }
                }
              })
          )
        })
      )

    const runStep = <
      A,
      Entry extends {
        readonly seq: StatefulExecutionSeq
        readonly token: number
      }
    >(
      entries: ReadonlyArray<Entry>,
      runBatched: (
        entries: ReadonlyArray<Entry>,
        input: Tensor.Any,
        ids: ReadonlyArray<number>,
        slots: ReadonlyArray<number>,
        program: Tensor.DecodeProgram
      ) => Effect.Effect<ReadonlyArray<A>, Tensor.TensorError, Runtime.Runtime>
    ): Effect.Effect<ReadonlyArray<A>, InferenceError | Tensor.TensorError, Runtime.Runtime> =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          yield* validateStepEntries(live, config.batchSize, entries)
          const ids = entries.map((entry) => entry.token)
          const slots = entries.map((entry) => live.find((liveEntry) => liveEntry.seq === entry.seq)!.slot)
          const input = yield* slottedTokenTensor(ids, slots, config.batchSize, config.tokenDtype)

          return yield* runBatched(entries, input, ids, slots, programs.decode)
        })
      )

    const step: StatefulExecution["step"] = (entries) =>
      runStep(
        entries,
        (entries, input, ids, slots, batched) =>
          Effect.flatMap(
            Tensor.runBatchedDecodeProgram(
              batched,
              [input],
              entries.map((entry) => entry.seq.sequence),
              slots,
              ids.map((id) => [id])
            ),
            (outputs) =>
              Effect.onExit(
                Effect.gen(function*() {
                  const selected = slots.map((slot) => outputs[slot]!)
                  const selectedSlots = new Set(slots)

                  for (const [slot, output] of outputs.entries()) {
                    if (selectedSlots.has(slot)) continue

                    yield* Tensor.clear(output)
                  }

                  return selected
                }),
                (exit) => Exit.isFailure(exit) ? Tensor.clearAll(outputs) : Effect.void
              )
          )
      )

    return {
      add,
      step,
      live: () => Effect.sync(() => live.length),
      close: () => closeLiveEntries(live)
    }
  })

interface NativeGenerationEntry {
  readonly seq: GenerationSeq
  readonly handle: Runtime.InferenceSequenceHandle
  readonly id: bigint
  terminal: "eos" | "maxTokens" | undefined
}

const inferenceBackend = <A>(op: string, effect: Effect.Effect<A, Runtime.BackendError>) =>
  Effect.mapError(effect, (backend) => new Tensor.TensorError({ op, message: backend.message, backend }))

const nativeSampling = (sampling: GenerationSamplingOptions): Runtime.InferenceSamplingOptions => {
  const seed = sampling.seed

  return {
    temperature: sampling.temperature ?? 1,
    topK: sampling.topK ?? 0,
    topP: sampling.topP ?? 1,
    seed: BigInt(seed)
  }
}

const nativeSamplingOverride = (
  sampling: Partial<GenerationSamplingOptions>
): Runtime.InferenceSamplingOverrides => ({
  temperature: sampling.temperature,
  topK: sampling.topK,
  topP: sampling.topP,
  seed: sampling.seed === undefined ? undefined : BigInt(sampling.seed)
})

const validateGenerationAdd = (
  entry: GenerationAdd,
  index: number,
  defaults: GenerationSamplingOptions
): Effect.Effect<void, InferenceError> =>
  Effect.gen(function*() {
    if (
      entry.maxTokens !== undefined &&
      (!Number.isSafeInteger(entry.maxTokens) || entry.maxTokens <= 0 || entry.maxTokens > 0xffff_ffff)
    ) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} maxTokens must be an unsigned 32-bit positive integer, got ${entry.maxTokens}`
      })
    }

    for (const token of entry.eosTokens ?? []) {
      if (!Number.isInteger(token) || token < 0 || token > 0xffff_ffff) {
        return yield* new InferenceError({
          op: "add",
          message: `entry ${index} eosTokens must contain unsigned 32-bit token ids, got ${token}`
        })
      }
    }

    const sampling = { ...defaults, ...entry.sampling }

    if (
      (!Predicate.isBigInt(sampling.seed) && !Number.isSafeInteger(sampling.seed)) || sampling.seed < 0 ||
      BigInt(sampling.seed) > 0xffff_ffff_ffff_ffffn
    ) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} seed must be an unsigned 64-bit integer, got ${sampling.seed}`
      })
    }

    if (sampling.temperature !== undefined && (!Number.isFinite(sampling.temperature) || sampling.temperature < 0)) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} temperature must be finite and non-negative, got ${sampling.temperature}`
      })
    }

    if (sampling.topK !== undefined && (!Number.isSafeInteger(sampling.topK) || sampling.topK < 0)) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} topK must be a non-negative safe integer, got ${sampling.topK}`
      })
    }

    if (sampling.topP !== undefined && (!Number.isFinite(sampling.topP) || sampling.topP <= 0 || sampling.topP > 1)) {
      return yield* new InferenceError({
        op: "add",
        message: `entry ${index} topP must be in (0, 1], got ${sampling.topP}`
      })
    }
  })

const openGeneration = (engine: InferenceEngine): Effect.Effect<Generation, InferenceError> =>
  Effect.gen(function*() {
    const roundLock = yield* Semaphore.make(1)
    const live: Array<NativeGenerationEntry> = []
    const config = engine.config
    const runtime = engine.runtime
    const native = runtime.extensions.inference

    const session = yield* Effect.mapError(
      native.open(engine.artifact),
      (error) => new InferenceError({ op: "generation", message: error.message })
    )

    const pagesFor = (
      op: "add" | "step",
      result: Runtime.InferenceRoundResult,
      expected: ReadonlyArray<NativeGenerationEntry>
    ): Effect.Effect<ReadonlyArray<TokenPage>, InferenceError> =>
      Effect.gen(function*() {
        if (
          result.roundId < 0n || result.roundId > 0xffff_ffff_ffff_ffffn ||
          !Predicate.isBoolean(result.recovered) ||
          result.pages.length !== expected.length
        ) {
          return yield* new InferenceError({
            op,
            message: `${op}: native inference returned a malformed round receipt`
          })
        }

        const pages: Array<TokenPage> = []

        for (const [index, page] of result.pages.entries()) {
          const entry = expected[index]!

          if (
            page.sequence !== entry.handle || page.sequenceId !== entry.id || page.tokens.length === 0 ||
            page.tokens.some((token) => !Number.isInteger(token) || token < 0 || token > 0xffff_ffff) ||
            (page.stopReason !== undefined && page.stopReason !== "eos" && page.stopReason !== "maxTokens")
          ) {
            return yield* new InferenceError({ op, message: `${op}: native inference returned a malformed token page` })
          }

          pages.push({
            seq: entry.seq,
            tokens: page.tokens,
            stopReason: page.stopReason
          })
        }

        return pages
      })

    const add: Generation["add"] = (requests) =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          if (requests.length === 0) {
            return yield* new InferenceError({ op: "add", message: "add expects at least one entry" })
          }

          if (live.length + requests.length > config.batchSize) {
            return yield* new InferenceError({
              op: "add",
              message: `add needs ${requests.length} free lanes, but only ${config.batchSize - live.length} remain`
            })
          }

          for (const [index, request] of requests.entries()) {
            yield* validateGenerationAdd(request, index, config.sampling)
          }

          for (const request of requests) yield* validatePrompt(request.prompt, config, runtime)

          const promptValues = yield* Tensor.compute(requests.map((request) => request.prompt))

          return yield* Effect.onExit(
            Effect.gen(function*() {
              const result = yield* inferenceBackend(
                "inferenceAdd",
                native.add(session, {
                  entries: requests.map((request, index) => ({
                    prompt: promptValues[index]!,
                    sampling: request.sampling === undefined
                      ? undefined
                      : nativeSamplingOverride(request.sampling),
                    maxTokens: request.maxTokens,
                    eosTokens: request.eosTokens ?? []
                  }))
                })
              )

              if (result.pages.length !== requests.length) {
                return yield* new InferenceError({
                  op: "add",
                  message: "add: native inference returned the wrong page count"
                })
              }

              const added: Array<NativeGenerationEntry> = []

              for (const page of result.pages) {
                if (
                  page.sequenceId < 0n || page.sequenceId > 0xffff_ffff_ffff_ffffn ||
                  added.some((entry) => entry.handle === page.sequence || entry.id === page.sequenceId)
                ) {
                  return yield* new InferenceError({
                    op: "add",
                    message: "add: native inference returned invalid sequence identity"
                  })
                }

                let entry: NativeGenerationEntry

                const seq: GenerationSeq = {
                  _tag: "GenerationSeq",
                  cursor: () =>
                    Effect.gen(function*() {
                      const inspected = yield* inferenceBackend(
                        "inferenceInspect",
                        native.inspect(session, entry.handle)
                      )

                      if (
                        inspected.sequenceId !== entry.id || inspected.cursor < 0n ||
                        inspected.cursor > BigInt(Number.MAX_SAFE_INTEGER)
                      ) {
                        return yield* new Tensor.TensorError({
                          op: "inferenceInspect",
                          message: "native inference returned an invalid cursor"
                        })
                      }

                      return Number(inspected.cursor)
                    }),
                  finish: () =>
                    roundLock.withPermits(1)(
                      Effect.gen(function*() {
                        const index = live.indexOf(entry)

                        if (index < 0) return

                        yield* inferenceBackend("inferenceFinish", native.finish(session, [entry.handle]))
                        live.splice(index, 1)
                      })
                    )
                }

                entry = { seq, handle: page.sequence, id: page.sequenceId, terminal: undefined }
                added.push(entry)
              }

              const pages = yield* pagesFor("add", result, added)

              return yield* Effect.uninterruptible(Effect.gen(function*() {
                yield* inferenceBackend("inferenceAcknowledge", native.acknowledge(session, result.roundId))

                for (const [index, entry] of added.entries()) entry.terminal = result.pages[index]!.stopReason

                live.push(...added)

                return pages
              }))
            }),
            () => Tensor.clearAll(promptValues)
          )
        })
      )

    const step: Generation["step"] = (requests) =>
      roundLock.withPermits(1)(
        Effect.gen(function*() {
          if (requests.length === 0) {
            return yield* new InferenceError({ op: "step", message: "step expects at least one entry" })
          }

          if (requests.length > config.batchSize) {
            return yield* new InferenceError({
              op: "step",
              message: `step accepts at most batchSize (${config.batchSize}) entries, got ${requests.length}`
            })
          }

          const selected: Array<NativeGenerationEntry> = []

          for (const [index, request] of requests.entries()) {
            const entry = live.find((entry) => entry.seq === request.seq)

            if (entry === undefined) {
              return yield* new InferenceError({ op: "step", message: `entry ${index} is not a live sequence` })
            }

            if (selected.includes(entry)) {
              return yield* new InferenceError({ op: "step", message: "step entries must be distinct sequences" })
            }

            if (entry.terminal !== undefined) {
              return yield* new InferenceError({
                op: "step",
                message: `entry ${index} is terminal (${entry.terminal})`
              })
            }

            selected.push(entry)
          }

          const result = yield* inferenceBackend(
            "inferenceRound",
            native.runRound(session, {
              entries: selected.map((entry, index) => ({
                sequence: entry.handle,
                sampling: requests[index]!.sampling === undefined
                  ? undefined
                  : nativeSamplingOverride(requests[index]!.sampling)
              }))
            })
          )

          const pages = yield* pagesFor("step", result, selected)

          return yield* Effect.uninterruptible(Effect.gen(function*() {
            yield* inferenceBackend("inferenceAcknowledge", native.acknowledge(session, result.roundId))

            for (const [index, entry] of selected.entries()) entry.terminal = result.pages[index]!.stopReason

            return pages
          }))
        })
      )

    return {
      add,
      step,
      live: () => Effect.sync(() => live.length),
      close: () =>
        roundLock.withPermits(1)(
          Effect.tap(inferenceBackend("inferenceClose", native.close(session)), () => Effect.sync(() => live.splice(0)))
        )
    }
  })

/**
 * Materializes a model for stateful autoregressive generation and eagerly
 * compiles its complete deployment geometry. The same `forward` builder is
 * traced twice, once for fixed prompt chunks and once for fixed-width batched
 * decode. Decode specialization rewrites causal attention to paged KV
 * attention, KDA and short convolution to per-sequence
 * recurrent operations, and learned/rotary position nodes to absolute-cursor-
 * offset forms. There is no shape-keyed growth or later tracing.
 *
 * Every trace must return exactly `[batch, T, vocab]` with the traced batch and
 * token dimensions preserved, and all traces must agree on state geometry and
 * effective retention policy. Native `lastTokenRow` selection returns one
 * caller-owned `[vocab]` row per active sequence. Stateless graphs are allowed.
 * Non-causal attention, runtime scalar inputs, unsupported stateful operations,
 * inconsistent traces, and invalid output rank/axes fail during construction.
 * This does not establish semantic language-model correctness or validate a
 * tokenizer/vocabulary contract.
 *
 * Dense `params` are borrowed and materialized together once with
 * {@link Tensor.compute}. This samples lazy initializers once and produces a new
 * concrete generation. Already-concrete packed parameters are borrowed directly
 * during compilation. Every compiled program retains its parameters as immutable
 * constants. Caller-supplied concrete handles are not consumed and may be cleared
 * after this effect succeeds; the artifact's retained generation remains valid.
 * Temporary materialized handles are cleared after compilation on success,
 * failure, or interruption; native programs retain captured constants.
 * There is no explicit artifact release after success; native finalization
 * reclaims its constants, programs, and pool when unreachable.
 *
 * State capacity is separate from artifact lifetime. Live sequences pin blocks
 * and recurrent state, while completed blocks may remain as evictable prefix
 * cache. Use {@link GenerationSeq.finish} or {@link Generation.close} to remove
 * live ownership promptly. An attention window can bound retained KV history
 * without resetting the absolute cursor; learned position tables and other
 * cursor-indexed state remain independently bounded.
 *
 * @since 0.1.0
 * @category compilation
 */
export const compile = (
  model: Model.Definition,
  params: Model.Parameters,
  config: CompileOptions
): Effect.Effect<Artifact, InferenceError | Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime

    if (params.length !== model.parameterSpecs.length) {
      const names = model.parameterSpecs.map((parameter) => parameter.name)

      return yield* new Model.ModelError({
        op: "forward",
        message: `inference: expected ${names.length} parameters [${names.join(", ")}], got ${params.length}`
      })
    }

    const resolved = yield* resolveCompileOptions(config)

    const proposerSourceParams = resolved.speculation?.proposer._tag === "HistoryLookup"
      ? []
      : resolved.speculation?.proposer.params ?? []

    const targetArity = params.length
    const sourceParams = [...params, ...proposerSourceParams]

    return yield* Model.withParameters(sourceParams, (allFrozenParams) =>
      Effect.gen(function*() {
        const frozenParams = allFrozenParams.slice(0, targetArity)

        const proposerParams = resolved.speculation === undefined
          ? undefined
          : allFrozenParams.slice(targetArity)

        const programs = yield* compileArtifacts(model, frozenParams, resolved, proposerParams)

        const exactProposer = programs.speculation !== undefined &&
            programs.speculation.generalized === undefined && programs.speculation.proposer !== undefined
          ? { ...programs.speculation.proposer, maxDraftTokens: programs.speculation.maxDraftTokens }
          : undefined

        const artifact = yield* inferenceBackend(
          "inferenceCompile",
          runtime.extensions.inference.compile({
            target: {
              prefill: programs.prefill.map((program) => program.handle),
              decode: programs.decode.handle,
              verify: programs.speculation?.verify.map((program) => program.handle),
              pool: programs.pool.handle
            },
            proposer: exactProposer === undefined
              ? undefined
              : {
                prefill: exactProposer.prefill.handle,
                decode: exactProposer.decode.handle,
                pool: exactProposer.pool.handle,
                maxDraftTokens: exactProposer.maxDraftTokens
              },
            generalizedProposer: programs.speculation?.generalized,
            batchSize: resolved.batchSize,
            tokenDtype: resolved.tokenDtype,
            sampling: nativeSampling(resolved.sampling)
          })
        )

        const engine: InferenceEngine = {
          config: resolved,
          programs,
          artifact,
          runtime
        }

        const inferenceProgram: Artifact = {
          generation: () => openGeneration(engine),
          execution: () => openStatefulExecution(engine),
          diagnostics: () =>
            inferenceBackend("inferenceDiagnostics", runtime.extensions.inference.diagnostics(artifact))
        }

        return inferenceProgram
      }))
  })

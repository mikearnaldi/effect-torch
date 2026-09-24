/**
 * Compiled causal encoding and read-only block diffusion over shared parameters.
 * Prefixes are immutable native snapshots. Each evaluation owns its canvas
 * storage and returned logits; completed tokens pass through the encoder when
 * committed to a new prefix.
 *
 * @since 0.1.0
 */
import { Data, Effect, Exit, Scope } from "effect"
import * as Model from "./Model.ts"
import * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * Initial conditioning or borrowed full-vocabulary logits from a previous step.
 * Refinement logits have shape `[1, canvasLength, vocabSize]` and the model
 * definition's predictionDtype. Full readout logits remain F32.
 *
 * @since 0.1.0
 * @category models
 */
export type Prediction =
  | { readonly _tag: "Initial" }
  | {
    readonly _tag: "Refinement"
    readonly logits: Tensor.Any
  }

/**
 * Full logits or ordered row and vocabulary selections supplied as rank-one
 * u32 inputs. Selected output has shape `[1, rows.length, labels.length]`.
 * Duplicate indexes retain their supplied order.
 *
 * @since 0.1.0
 * @category models
 */
export type Readout =
  | { readonly _tag: "Full" }
  | {
    readonly _tag: "Selected"
    readonly rows: Tensor.Any
    readonly labels: Tensor.Any
  }

/**
 * Lazy graph builders for a causal encoder and bidirectional denoiser sharing
 * one ordered parameter generation. Tokens and absolute positions are u32
 * `[1, T]` inputs. Both graphs must expose the same ordered persistent K/V
 * schema. Readout returns f32 logits. Builders must use the supplied parameters
 * rather than capture source handles.
 *
 * @since 0.1.0
 * @category models
 */
export interface Definition {
  readonly parameterSpecs: ReadonlyArray<Model.ParameterSpec>
  readonly vocabSize: number
  readonly canvasLength: number
  readonly maxPositions: number
  readonly dtype: Tensor.DType
  /** Storage dtype required for carried prediction feedback. */
  readonly predictionDtype: Tensor.DType
  readonly encode: (
    parameters: ReadonlyArray<Tensor.Any>,
    tokens: Tensor.Any,
    positions: Tensor.Any
  ) => Effect.Effect<Tensor.Any, Model.ModelError | Tensor.TensorError, Runtime.Runtime>
  readonly denoise: (
    parameters: ReadonlyArray<Tensor.Any>,
    tokens: Tensor.Any,
    positions: Tensor.Any,
    prediction: Prediction
  ) => Effect.Effect<Tensor.Any, Model.ModelError | Tensor.TensorError, Runtime.Runtime>
  readonly readout: (
    parameters: ReadonlyArray<Tensor.Any>,
    hidden: Tensor.Any,
    selection: Readout
  ) => Effect.Effect<Tensor.Any, Model.ModelError | Tensor.TensorError, Runtime.Runtime>
}

/**
 * A deliberately compiled selected-readout shape. Values remain runtime inputs.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface SelectedReadoutShape {
  readonly rows: number
  readonly labels: number
}

/**
 * Deployment capacities and fixed trace shapes, independent of generation policy.
 * Every canvas width gets initial/refinement hidden-state programs and a full
 * readout. Selected shapes add only stateless readout programs. No invocation
 * traces new graphs.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface CompileOptions {
  readonly maxTokens: number
  /** Paging unit, default 16; must divide maxTokens. */
  readonly blockSize?: number
  /** Ascending positive prompt chunk widths. Defaults to one paging unit. */
  readonly prefillChunks?: ReadonlyArray<number>
  /** Ascending positive exact canvas widths. Defaults to definition.canvasLength. */
  readonly canvasLengths?: ReadonlyArray<number>
  /** Default is no selected readouts. Counts do not constrain index order. */
  readonly selectedReadouts?: ReadonlyArray<SelectedReadoutShape>
  readonly compile?: { readonly optimize?: boolean }
}

/**
 * Invalid deployment geometry, incompatible graph schemas, or artifact misuse.
 * Backend and graph failures retain their TensorError or Model.ModelError types.
 *
 * @since 0.1.0
 * @category errors
 */
export class InferenceError extends Data.TaggedError("DiffusionInferenceError")<{
  readonly op: string
  readonly message: string
}> {}

const PrefixTypeId: unique symbol = Symbol("@effect-torch/core/Diffusion/Prefix")

/**
 * An owned immutable prefix tied to its creating artifact and runtime. Concurrent
 * evaluations borrow it. Release after every borrower has finished. Token count
 * includes evicted positions; bytes reports the native snapshot storage.
 *
 * @since 0.1.0
 * @category models
 */
export interface Prefix {
  readonly tokenCount: number
  readonly bytes: number
  readonly [PrefixTypeId]: {
    readonly owner: object
    readonly snapshot: Tensor.KvSnapshot
  }
}

/**
 * Fixed compiled entry points and a shared state pool. Returned prefixes and
 * logits are caller-owned. Releasing a prefix does not invalidate completed
 * outputs or another prefix produced by commit. Calls on different prefixes,
 * and concurrent evaluations of one prefix, may overlap.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface Artifact {
  readonly dtype: Tensor.DType
  readonly predictionDtype: Tensor.DType
  readonly vocabSize: number
  readonly canvasLength: number
  readonly maxPositions: number
  /** Runs the model policy using these compiled encode, evaluate, and commit operations. */
  readonly generate: <P, S, E, R>(
    options: GenerationRequest<P, S, E, R>
  ) => Effect.Effect<
    GenerationResult,
    E | InferenceError | Tensor.TensorError | DiffusionGenerationError,
    R | Runtime.Runtime
  >
  readonly encode: (tokens: Uint32Array) => Effect.Effect<Prefix, InferenceError | Tensor.TensorError, Runtime.Runtime>
  readonly commit: (
    prefix: Prefix,
    tokens: Uint32Array
  ) => Effect.Effect<Prefix, InferenceError | Tensor.TensorError, Runtime.Runtime>
  readonly evaluate: (
    prefix: Prefix,
    canvas: Uint32Array,
    prediction: Prediction
  ) => Effect.Effect<Tensor.Concrete, InferenceError | Tensor.TensorError, Runtime.Runtime>
  readonly score: (
    prefix: Prefix,
    canvas: Uint32Array,
    rows: ReadonlyArray<number>,
    labels: ReadonlyArray<number>
  ) => Effect.Effect<Tensor.Concrete, InferenceError | Tensor.TensorError, Runtime.Runtime>
  readonly release: (prefix: Prefix) => Effect.Effect<void, InferenceError | Tensor.TensorError, Runtime.Runtime>
  /** Diagnostic host copy of prefix state and physical storage accounting. */
  readonly inspect: (
    prefix: Prefix
  ) => Effect.Effect<Runtime.KvSnapshotInspection, InferenceError | Tensor.TensorError, Runtime.Runtime>
}

const invalid = (op: string, message: string) => new InferenceError({ op, message })

const positiveU32 = (value: number) => Number.isInteger(value) && value > 0 && value <= 0xffff_ffff

const shapeMatches = (actual: ReadonlyArray<number>, expected: ReadonlyArray<number>) =>
  actual.length === expected.length && actual.every((dimension, index) => dimension === expected[index])

const selectionKey = (shape: SelectedReadoutShape) => `${shape.rows}:${shape.labels}`

interface CanvasPrograms {
  readonly initial: Tensor.DecodeProgram
  readonly refinement: Tensor.DecodeProgram
  readonly full: Tensor.CompiledProgram
  readonly selected: ReadonlyMap<string, Tensor.CompiledProgram>
}

const input = (slot: number, shape: ReadonlyArray<number>, dtype: Tensor.DType) =>
  Effect.flatMap(Tensor.zeros(shape, { dtype }), (exemplar) => Tensor.makeInput(slot, exemplar))

/**
 * Materializes one parameter generation and eagerly compiles all configured
 * encoder and initial/refinement denoiser through Tensor.compileDecodeProgram.
 * Full and selected stateless readouts use Tensor.freezeProgram. All native
 * programs retain the same parameter generation.
 *
 * @since 0.1.0
 * @category compilation
 */
export const compile = (
  definition: Definition,
  parameters: Model.Parameters,
  config: CompileOptions
): Effect.Effect<Artifact, InferenceError | Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const blockSize = config.blockSize ?? 16
    const prefillChunks = [...config.prefillChunks ?? [Math.min(blockSize, definition.maxPositions)]]
    const canvasLengths = [...config.canvasLengths ?? [definition.canvasLength]]
    const selectedReadouts = [...config.selectedReadouts ?? []]

    for (
      const [name, value] of Object.entries({
        maxTokens: config.maxTokens,
        blockSize,
        vocabSize: definition.vocabSize,
        canvasLength: definition.canvasLength,
        maxPositions: definition.maxPositions
      })
    ) {
      if (!positiveU32(value)) return yield* invalid("inference", `${name} must be a positive u32 integer`)
    }

    if (config.maxTokens % blockSize !== 0) {
      return yield* invalid("inference", "blockSize must divide maxTokens")
    }

    if (definition.dtype !== "f32" && definition.dtype !== "f16" && definition.dtype !== "bf16") {
      return yield* invalid("inference", "dtype must be f32, f16, or bf16")
    }

    for (const [name, widths] of [["prefillChunks", prefillChunks], ["canvasLengths", canvasLengths]] as const) {
      if (
        widths.length === 0 ||
        widths.some((width, index) =>
          !positiveU32(width) || width > definition.maxPositions || (index > 0 && width <= widths[index - 1]!)
        )
      ) {
        return yield* invalid(
          "inference",
          `${name} must contain ascending distinct positive widths within maxPositions`
        )
      }
    }

    const selections = new Set<string>()

    for (const shape of selectedReadouts) {
      if (!positiveU32(shape.rows) || !positiveU32(shape.labels) || selections.has(selectionKey(shape))) {
        return yield* invalid("inference", "selectedReadouts must contain distinct positive row/label counts")
      }

      selections.add(selectionKey(shape))
    }

    if (parameters.length !== definition.parameterSpecs.length) {
      return yield* invalid(
        "compile",
        `expected ${definition.parameterSpecs.length} parameters, got ${parameters.length}`
      )
    }

    return yield* Model.withParameters(parameters, (parameters) =>
      Effect.gen(function*() {
        let schema: Tensor.DecodeProgram | undefined

        const compile = (root: Tensor.Any, access: "Append" | "ReadOnly") =>
          Effect.gen(function*() {
            const program = yield* Tensor.compileDecodeProgram([root], {
              maxTokens: config.maxTokens,
              blockSize,
              kvDtype: definition.dtype,
              batch: 1,
              access,
              currentBlockAttention: access === "Append" ? "Causal" : "Bidirectional"
            }, config.compile)

            if (program.kdaLayers !== 0 || program.convLayers !== 0) {
              return yield* invalid("inference", "diffusion prefixes require K/V-only state")
            }

            if (schema === undefined) schema = program
            else if (!Runtime.sameDecodeStateSchema(program, schema)) {
              return yield* invalid("inference", "encoder and denoiser entry points have incompatible K/V schemas")
            }

            return program
          })

        const encoder = new Map<number, Tensor.DecodeProgram>()

        for (const width of prefillChunks) {
          const tokens = yield* input(0, [1, width], "u32")
          const positions = yield* input(1, [1, width], "u32")
          const hidden = yield* definition.encode(parameters, tokens, positions)
          encoder.set(width, yield* compile(hidden, "Append"))
        }

        const canvases = new Map<number, CanvasPrograms>()

        for (const width of canvasLengths) {
          const tokens = yield* input(0, [1, width], "u32")
          const positions = yield* input(1, [1, width], "u32")
          const initialHidden = yield* definition.denoise(parameters, tokens, positions, { _tag: "Initial" })
          const initial = yield* compile(initialHidden, "ReadOnly")
          const previous = yield* input(2, [1, width, definition.vocabSize], definition.predictionDtype)

          const refined = yield* definition.denoise(parameters, tokens, positions, {
            _tag: "Refinement",
            logits: previous
          })

          if (refined.dtype !== initialHidden.dtype || !shapeMatches(refined.shape, initialHidden.shape)) {
            return yield* invalid("inference", "refinement hidden state must match the initial hidden state")
          }

          const refinement = yield* compile(refined, "ReadOnly")
          const hidden = yield* input(0, initialHidden.shape, initialHidden.dtype)
          const fullLogits = yield* definition.readout(parameters, hidden, { _tag: "Full" })

          if (fullLogits.dtype !== "f32" || !shapeMatches(fullLogits.shape, [1, width, definition.vocabSize])) {
            return yield* invalid("inference", "full readout must return f32 [1, canvasLength, vocabSize]")
          }

          const full = yield* Tensor.freezeProgram([fullLogits], { ...config.compile, constantWeights: true })
          const selected = new Map<string, Tensor.CompiledProgram>()

          for (const shape of selectedReadouts) {
            const rows = yield* input(1, [shape.rows], "u32")
            const labels = yield* input(2, [shape.labels], "u32")
            const logits = yield* definition.readout(parameters, hidden, { _tag: "Selected", rows, labels })

            if (logits.dtype !== "f32" || !shapeMatches(logits.shape, [1, shape.rows, shape.labels])) {
              return yield* invalid("inference", "selected readout must return f32 [1, rows, labels]")
            }

            selected.set(
              selectionKey(shape),
              yield* Tensor.freezeProgram([logits], { ...config.compile, constantWeights: true })
            )
          }

          canvases.set(width, { initial, refinement, full, selected })
        }

        const geometry = schema!
        const pool = yield* Tensor.makeKvPoolFromSchema(geometry)
        const owner = {}

        const checkRuntime = (op: string) =>
          Effect.flatMap(Runtime.Runtime, (active) =>
            active.identity === runtime.identity ? Effect.void : invalid(op, "artifact belongs to another runtime"))

        const snapshot = (prefix: Prefix, op: string) =>
          Effect.gen(function*() {
            yield* checkRuntime(op)

            if (prefix[PrefixTypeId]?.owner !== owner) {
              return yield* invalid(op, "prefix belongs to another artifact")
            }

            return prefix[PrefixTypeId].snapshot
          })

        const validateTokens = (tokens: Uint32Array, offset: number, op: string) =>
          Effect.gen(function*() {
            if (offset + tokens.length > definition.maxPositions) {
              return yield* invalid(op, "tokens exceed maxPositions")
            }

            if (
              tokens.some((token) =>
                token >= definition.vocabSize
              )
            ) {
              return yield* invalid(op, "token is outside the vocabulary")
            }
          })

        const inputs = (
          tokens: Uint32Array,
          width: number,
          offset: number
        ): Effect.Effect<Array<Tensor.Any>, Tensor.TensorError, Runtime.Runtime> =>
          Effect.gen(function*() {
            const padded = new Uint32Array(width)
            padded.set(tokens)

            const positions = Uint32Array.from({ length: width }, (_, row) =>
              Math.min(offset + row, definition.maxPositions - 1))

            return [
              yield* Tensor.fromTypedArray(padded, [1, width]),
              yield* Tensor.fromTypedArray(positions, [1, width])
            ]
          })

        const append = (
          acquire: Effect.Effect<Tensor.KvSequence, Tensor.TensorError, Runtime.Runtime>,
          tokens: Uint32Array,
          offset: number
        ) =>
          Effect.suspend(() => {
            let acquired: Tensor.KvSnapshot | undefined

            return Effect.scoped(Effect.gen(function*() {
              const sequence = yield* Effect.acquireRelease(
                acquire,
                (seq) => Effect.orDie(Tensor.releaseKvSequence(seq)),
                { interruptible: true }
              )

              for (let start = 0; start < tokens.length;) {
                const remaining = tokens.length - start

                const width = prefillChunks.find((size) => size >= remaining) ??
                  prefillChunks[prefillChunks.length - 1]!

                const chunk = tokens.slice(start, start + width)
                const bindings = yield* inputs(chunk, width, offset + start)
                yield* Effect.scoped(Effect.acquireRelease(
                  Tensor.runDecodeProgram(encoder.get(width)!, bindings, sequence, Array.from(chunk)),
                  Tensor.clearAll,
                  { interruptible: true }
                ))
                start += chunk.length
              }

              const retained = yield* Tensor.snapshotKvSequence(sequence).pipe(Effect.onExit((exit) =>
                Effect.sync(() => {
                  if (Exit.isSuccess(exit)) acquired = exit.value
                })
              ))

              return {
                tokenCount: retained.tokenCount,
                bytes: retained.retainedBytes,
                [PrefixTypeId]: { owner, snapshot: retained }
              }
            })).pipe(Effect.onExit((exit) =>
              Exit.isFailure(exit) && acquired !== undefined
                ? Effect.orDie(Tensor.releaseKvPrefix(acquired))
                : Effect.void
            ))
          })

        const run = (
          program: Tensor.DecodeProgram,
          readout: Tensor.CompiledProgram,
          bindings: ReadonlyArray<Tensor.Any>,
          selection: ReadonlyArray<Tensor.Any>,
          prefix: Tensor.KvSnapshot,
          width: number
        ) =>
          Effect.suspend(() => {
            let acquired: ReadonlyArray<Tensor.Concrete> = []

            return Effect.scoped(Effect.gen(function*() {
              const hidden = yield* Effect.acquireRelease(
                Tensor.runReadOnlyDecodeProgram(program, bindings, prefix, width),
                Tensor.clearAll,
                { interruptible: true }
              )

              const outputs = yield* Tensor.runProgram(readout, [hidden[0]!, ...selection]).pipe(Effect.onExit((exit) =>
                Effect.sync(() => {
                  if (Exit.isSuccess(exit)) acquired = exit.value
                })
              ))

              return outputs[0]!
            })).pipe(Effect.onExit((exit) =>
              Exit.isFailure(exit) ? Tensor.clearAll(acquired) : Effect.void
            ))
          })

        const program: Artifact = {
          dtype: definition.dtype,
          predictionDtype: definition.predictionDtype,
          vocabSize: definition.vocabSize,
          canvasLength: definition.canvasLength,
          maxPositions: definition.maxPositions,
          generate: (options) => generateWithProgram(program, options),
          encode: (tokens) =>
            Effect.gen(function*() {
              yield* checkRuntime("encode")
              yield* validateTokens(tokens, 0, "encode")

              return yield* append(Tensor.makeKvSequence(pool), tokens, 0)
            }),
          commit: (prefix, tokens) =>
            Effect.gen(function*() {
              const retained = yield* snapshot(prefix, "commit")
              yield* validateTokens(tokens, prefix.tokenCount, "commit")

              return yield* append(Tensor.forkKvPrefix(retained), tokens, prefix.tokenCount)
            }),
          evaluate: (prefix, canvas, prediction) =>
            Effect.gen(function*() {
              const retained = yield* snapshot(prefix, "evaluate")
              yield* validateTokens(canvas, prefix.tokenCount, "evaluate")
              const bucket = canvases.get(canvas.length)

              if (bucket === undefined) {
                return yield* invalid("evaluate", "canvas length has no compiled bucket")
              }

              const bindings = yield* inputs(canvas, canvas.length, prefix.tokenCount)

              if (prediction._tag === "Refinement") {
                if (
                  prediction.logits.dtype !== definition.predictionDtype ||
                  !shapeMatches(prediction.logits.shape, [1, canvas.length, definition.vocabSize])
                ) {
                  return yield* invalid(
                    "evaluate",
                    `refinement logits must be ${definition.predictionDtype} [1, canvasLength, vocabSize]`
                  )
                }

                bindings.push(prediction.logits)
              }

              return yield* run(
                prediction._tag === "Initial" ? bucket.initial : bucket.refinement,
                bucket.full,
                bindings,
                [],
                retained,
                canvas.length
              )
            }),
          score: (prefix, canvas, rows, labels) =>
            Effect.gen(function*() {
              const retained = yield* snapshot(prefix, "score")
              yield* validateTokens(canvas, prefix.tokenCount, "score")
              const bucket = canvases.get(canvas.length)
              const selected = bucket?.selected.get(selectionKey({ rows: rows.length, labels: labels.length }))

              if (bucket === undefined || selected === undefined) {
                return yield* invalid("score", "canvas and selection shape have no compiled bucket")
              }

              if (
                rows.some((row) => !Number.isInteger(row) || row < 0 || row >= canvas.length) ||
                labels.some((label) => !Number.isInteger(label) || label < 0 || label >= definition.vocabSize)
              ) {
                return yield* invalid("score", "row or label index is out of bounds")
              }

              const bindings = yield* inputs(canvas, canvas.length, prefix.tokenCount)

              const selection = [
                yield* Tensor.fromTypedArray(new Uint32Array(rows), [rows.length]),
                yield* Tensor.fromTypedArray(new Uint32Array(labels), [labels.length])
              ]

              return yield* run(bucket.initial, selected, bindings, selection, retained, canvas.length)
            }),
          release: (prefix) => Effect.flatMap(snapshot(prefix, "release"), Tensor.releaseKvPrefix),
          inspect: (prefix) => Effect.flatMap(snapshot(prefix, "inspect"), Tensor.inspectKvPrefix)
        }

        return program
      }))
  })

/**
 * An acquired value and its deterministic release effect.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Owned<A, R = never> {
  readonly value: A
  readonly release: Effect.Effect<void, never, R>
}

interface Acquired<A> {
  readonly value: A
  readonly scope: Scope.Closeable
}

/**
 * Initial model conditioning or caller-supplied feedback.
 *
 * @since 0.1.0
 * @category generation
 */
export type InitialFeedback<F> = { readonly _tag: "Initial" } | {
  readonly _tag: "Supplied"
  readonly value: F
}

/**
 * Conditioning for one evaluation, distinguishing fresh state from carried feedback.
 *
 * @since 0.1.0
 * @category generation
 */
export type Feedback<F> = InitialFeedback<F> | {
  readonly _tag: "Previous"
  readonly value: F
}

/**
 * One generated block and its absolute logical position.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Block {
  readonly index: number
  /** Absolute logical position, independent of retained prefix storage. */
  readonly position: number
  readonly remainingTokens: number
  /** Always the model canvas width, including a partial final output page. */
  readonly canvasLength: number
}

/**
 * One refinement iteration within a block.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Step {
  readonly index: number
  /** Counts down from maxSteps to one, as in reverse diffusion. */
  readonly remaining: number
}

/**
 * Fresh canvas tokens and their initial prediction state.
 *
 * @since 0.1.0
 * @category generation
 */
export interface InitialCanvas<F> {
  readonly canvas: Uint32Array
  readonly feedback: InitialFeedback<F>
}

/**
 * One evaluation result and the feedback carried to the next refinement.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Evaluation<P, F> {
  /** Reduced host statistics. Full logits and prediction feedback stay on-device. */
  readonly prediction: P
  readonly feedback: F
}

/**
 * Resource-owning operations used by the diffusion generation driver.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Callbacks<Prefix, F, P, E, R> {
  /** Acquisitions must clean partial resources and native late results themselves. */
  readonly encode: (prompt: Uint32Array) => Effect.Effect<Owned<Prefix, R>, E, R>
  readonly initialize: (block: Block) => Effect.Effect<Owned<InitialCanvas<F>, R>, E, R>
  readonly evaluate: (input: {
    readonly prefix: Prefix
    readonly canvas: Uint32Array
    readonly feedback: Feedback<F>
    readonly block: Block
    readonly step: Step
  }) => Effect.Effect<Owned<Evaluation<P, F>, R>, E, R>
  /** Encode completed tokens causally. Denoiser K/V cannot become encoder K/V. */
  readonly commit: (prefix: Prefix, tokens: Uint32Array, block: Block) => Effect.Effect<Owned<Prefix, R>, E, R>
}

/**
 * Updated policy state, noisy canvas, and the current completed-token candidate.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Refinement<S> {
  readonly state: S
  readonly canvas: Uint32Array
  /** The model policy chooses the completed output independently of its noisy canvas. */
  readonly draft: Uint32Array
  readonly done: boolean
}

/**
 * A completed full-width block and its sequence stopping decision.
 *
 * @since 0.1.0
 * @category generation
 */
export interface CompletedBlock {
  readonly tokens: Uint32Array
  readonly stop: boolean
}

/**
 * Model-specific canvas initialization, refinement, and block stopping policy.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Policy<S, P, E, R> {
  readonly canvasLength: number
  readonly maxSteps: number
  readonly start: (canvas: Uint32Array, block: Block) => S
  readonly refine: (input: {
    readonly state: S
    readonly canvas: Uint32Array
    readonly prediction: P
    readonly block: Block
    readonly step: Step
  }) => Effect.Effect<Refinement<S>, E, R>
  readonly finish: (draft: Uint32Array, block: Block) => CompletedBlock
}

/**
 * Internal refinement progress, which may produce no committed token page.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Progress {
  readonly block: Block
  readonly step: Step
  readonly draft: Uint32Array
  readonly done: boolean
}

/**
 * One diffusion generation request with explicit model operations and policy.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerationOptions<Prefix, F, P, S, E, R> {
  readonly prompt: Uint32Array
  readonly maxNewTokens: number
  /**
   * whole-block matches references that round the output limit up to a canvas.
   * exact clips only the published final page, never the evaluated canvas.
   */
  readonly outputLimit: "exact" | "whole-block"
  readonly callbacks: Callbacks<Prefix, F, P, E, R>
  readonly policy: Policy<S, P, E, R>
  readonly onProgress?: (progress: Progress) => Effect.Effect<void, E, R>
  readonly onPage: (tokens: Uint32Array, block: Block) => Effect.Effect<void, E, R>
}

/**
 * Generation policy and processing over a compiled artifact. Initialization
 * owns its supplied feedback. Processing borrows full F32 logits and returns
 * independently owned feedback plus sampler statistics. It must preserve any
 * model-specific temperature or rounding boundaries before returning
 * `[1, canvasLength, vocabSize]` feedback in the artifact's predictionDtype.
 * Raw logits are released after process
 * completes. No full-logit host transfer is imposed by this interface.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerationRequest<P, S, E = never, R = never>
  extends Omit<GenerationOptions<Prefix, Tensor.Any, P, S, E, R>, "callbacks">
{
  readonly initialize: (
    block: Block
  ) => Effect.Effect<Owned<InitialCanvas<Tensor.Any>, R | Runtime.Runtime>, E, R | Runtime.Runtime>
  readonly process: (
    logits: Tensor.Concrete,
    block: Block,
    step: Step
  ) => Effect.Effect<Owned<Evaluation<P, Tensor.Any>, R | Runtime.Runtime>, E, R | Runtime.Runtime>
}

/**
 * Completed generation counters and its stopping reason.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerationResult {
  readonly generatedTokens: number
  readonly blocks: number
  readonly refinements: number
  readonly stop: "length" | "policy"
}

/**
 * An invalid generation limit or policy result.
 *
 * @since 0.1.0
 * @category errors
 */
export class DiffusionGenerationError extends Data.TaggedError("DiffusionGenerationError")<{
  readonly message: string
}> {}

/**
 * Single-sequence scheduler. Refinement progress does not publish token pages.
 * Each resource has a child scope. Supported interruptible acquireRelease
 * registers its finalizer at acquisition; replacement closes the preceding
 * scope promptly. The request scope closes all children on every exit.
 *
 * @since 0.1.0
 * @category generation
 */
export const runGeneration = <Prefix, F, P, S, E, R>(
  options: GenerationOptions<Prefix, F, P, S, E, R>
): Effect.Effect<GenerationResult, E | DiffusionGenerationError, R> =>
  Effect.scopedWith((requestScope) => {
    const { callbacks, policy } = options

    const acquire = <A>(effect: Effect.Effect<Owned<A, R>, E, R>): Effect.Effect<Acquired<A>, E, R> =>
      Effect.gen(function*() {
        const scope = yield* Scope.fork(requestScope)

        const owned = yield* Effect.acquireRelease(effect, (owned) => owned.release, { interruptible: true })
          .pipe(Scope.provide(scope))

        return { value: owned.value, scope }
      })

    const checkWidth = (tokens: Uint32Array) =>
      tokens.length === policy.canvasLength
        ? Effect.void
        : Effect.fail(new DiffusionGenerationError({ message: "Policy canvas must retain the configured full width" }))

    return Effect.gen(function*() {
      if (
        !Number.isSafeInteger(options.maxNewTokens) || options.maxNewTokens < 0 ||
        !Number.isSafeInteger(policy.canvasLength) || policy.canvasLength < 1 ||
        !Number.isSafeInteger(policy.maxSteps) || policy.maxSteps < 1
      ) {
        return yield* new DiffusionGenerationError({
          message: "maxNewTokens must be nonnegative; canvasLength and maxSteps must be positive integers"
        })
      }

      let generatedTokens = 0
      let blocks = 0
      let refinements = 0

      if (options.maxNewTokens === 0) return { generatedTokens, blocks, refinements, stop: "length" as const }

      let prefix = yield* acquire(callbacks.encode(options.prompt.slice()))

      while (generatedTokens < options.maxNewTokens) {
        const block: Block = {
          index: blocks,
          position: options.prompt.length + blocks * policy.canvasLength,
          remainingTokens: options.maxNewTokens - generatedTokens,
          canvasLength: policy.canvasLength
        }

        const initial = yield* acquire(callbacks.initialize(block))
        let canvas = initial.value.canvas
        yield* checkWidth(canvas)
        let feedback: Feedback<F> = initial.value.feedback
        let feedbackScope = initial.scope
        let state = policy.start(canvas, block)
        let draft = canvas

        for (let index = 0; index < policy.maxSteps; index++) {
          const step: Step = { index, remaining: policy.maxSteps - index }

          const evaluated: Acquired<Evaluation<P, F>> = yield* acquire(callbacks.evaluate({
            prefix: prefix.value,
            canvas,
            feedback,
            block,
            step
          }))

          const next = yield* Scope.use(
            Effect.gen(function*() {
              const next = yield* policy.refine({ state, canvas, prediction: evaluated.value.prediction, block, step })
              yield* checkWidth(next.canvas)
              yield* checkWidth(next.draft)

              return next
            }),
            feedbackScope
          )

          feedbackScope = evaluated.scope
          feedback = { _tag: "Previous", value: evaluated.value.feedback }
          canvas = next.canvas
          state = next.state
          draft = next.draft
          refinements++
          const done = next.done || step.remaining === 1

          if (options.onProgress !== undefined) {
            yield* options.onProgress({ block, step, draft: draft.slice(), done })
          }

          if (done) break
        }

        const completed = yield* Scope.use(
          Effect.gen(function*() {
            const completed = policy.finish(draft, block)
            yield* checkWidth(completed.tokens)

            return completed
          }),
          feedbackScope
        )

        const count = options.outputLimit === "exact"
          ? Math.min(completed.tokens.length, block.remainingTokens)
          : completed.tokens.length

        yield* options.onPage(completed.tokens.slice(0, count), block)
        generatedTokens += count
        blocks++

        if (completed.stop || generatedTokens >= options.maxNewTokens) {
          return { generatedTokens, blocks, refinements, stop: completed.stop ? "policy" as const : "length" as const }
        }

        prefix = yield* Scope.use(acquire(callbacks.commit(prefix.value, completed.tokens, block)), prefix.scope)
      }

      return { generatedTokens, blocks, refinements, stop: "length" as const }
    })
  })

const generateWithProgram = <P, S, E, R>(
  program: Artifact,
  options: GenerationRequest<P, S, E, R>
): Effect.Effect<
  GenerationResult,
  E | InferenceError | Tensor.TensorError | DiffusionGenerationError,
  R | Runtime.Runtime
> =>
  Effect.suspend(() => {
    if (options.policy.canvasLength !== program.canvasLength) {
      return invalid("generate", "policy canvasLength must match the model canvasLength")
    }

    return runGeneration<Prefix, Tensor.Any, P, S, E | InferenceError | Tensor.TensorError, R | Runtime.Runtime>({
      ...options,
      callbacks: {
        initialize: options.initialize,
        encode: (tokens) =>
          Effect.map(program.encode(tokens), (value) => ({ value, release: Effect.orDie(program.release(value)) })),
        commit: (prefix, tokens) =>
          Effect.map(
            program.commit(prefix, tokens),
            (value) => ({ value, release: Effect.orDie(program.release(value)) })
          ),
        evaluate: ({ prefix, canvas, feedback, block, step }) =>
          Effect.suspend(() => {
            let processed: Owned<Evaluation<P, Tensor.Any>, R | Runtime.Runtime> | undefined

            return Effect.scoped(Effect.gen(function*() {
              const logits = yield* Effect.acquireRelease(
                program.evaluate(
                  prefix,
                  canvas,
                  feedback._tag === "Initial" ? { _tag: "Initial" } : { _tag: "Refinement", logits: feedback.value }
                ),
                Tensor.clear,
                { interruptible: true }
              )

              return yield* options.process(logits, block, step).pipe(Effect.onExit((exit) =>
                Effect.sync(() => {
                  if (Exit.isSuccess(exit)) processed = exit.value
                })
              ))
            })).pipe(
              Effect.onExit((exit) => Exit.isFailure(exit) && processed !== undefined ? processed.release : Effect.void)
            )
          })
      }
    })
  })

import {
  AutoRegressive,
  type Decision as CoreDecision,
  Diffusion,
  type Generation as CoreGeneration,
  type Model as CoreModel,
  Runtime,
  Tensor
} from "@effect-torch/core"
import type * as Tokenizers from "@effect-torch/tokenizers"
import { Effect, Queue, Schema, type Scope, Stream } from "effect"
import type * as Decision from "./Decision.ts"
import * as DecisionEngine from "./DecisionEngine.ts"
import type * as DecisionPlanner from "./DecisionPlanner.ts"
import type * as DecisionScaffold from "./DecisionScaffold.ts"

/**
 * Transport-independent model failure.
 *
 * @since 0.1.0
 * @category errors
 */
export class ModelError extends Schema.TaggedErrorClass<ModelError>()("ServeModelError", {
  message: Schema.String,
  invalidRequest: Schema.Boolean
}) {}

/**
 * Normalized generation input. Sampling fields remain optional for model defaults.
 *
 * @since 0.1.0
 * @category models
 */
export interface GenerationRequest {
  readonly input: { readonly prompt: string } | {
    readonly messages: ReadonlyArray<{
      readonly role: string
      readonly content: string
    }>
  }
  readonly maxTokens: number
  readonly temperature?: number | undefined
  readonly topP?: number | undefined
  readonly seed?: number | undefined
}

/**
 * Text deltas followed by exactly one terminal event.
 *
 * @since 0.1.0
 * @category models
 */
export type GenerationEvent =
  | {
    readonly _tag: "Delta"
    readonly text: string
  }
  | {
    readonly _tag: "Done"
    readonly finishReason: "stop" | "length"
    readonly promptTokens: number
    readonly completionTokens: number
  }

/**
 * Request-local noise seed and independent-read count for decision inference.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecisionSettings {
  readonly seed: string
  readonly reads: 1 | 4
}

/**
 * One loaded model, optionally serving both tasks. Its owner keeps its scope open.
 *
 * @since 0.1.0
 * @category models
 */
export interface Registration {
  readonly id: string
  readonly validateGeneration?: (request: GenerationRequest) => Effect.Effect<void, ModelError>
  readonly generate?: (request: GenerationRequest) => Stream.Stream<GenerationEvent, ModelError>
  readonly decide?: (
    request: Decision.Request,
    settings: DecisionSettings
  ) => Effect.Effect<Decision.Response, ModelError>
}

/**
 * A registered model with a required generation operation.
 *
 * @since 0.1.0
 * @category models
 */
export interface GenerationRegistration extends Registration {
  readonly validateGeneration: NonNullable<Registration["validateGeneration"]>
  readonly generate: NonNullable<Registration["generate"]>
}

/**
 * One registered model providing both generation and independent decisions.
 *
 * @since 0.1.0
 * @category models
 */
export interface SharedRegistration extends GenerationRegistration {
  readonly decide: NonNullable<Registration["decide"]>
}

/**
 * Pages are borrowed until the callback returns.
 *
 * @since 0.1.0
 * @category models
 */
export interface Generator<E, R> {
  readonly run: CoreGeneration.Generator<E, R, GenerationRequest, ModelError>
}

/**
 * Render chat or encode raw text, then decode committed pages with UTF-8 buffering.
 *
 * @since 0.1.0
 * @category constructors
 */
export const textGeneration = <E, R>(options: {
  readonly tokenizer: Pick<Tokenizers.Tokenizer, "encode" | "applyChatTemplate" | "decodeStream">
  readonly template: string
  readonly variables?: Tokenizers.ChatTemplateOptions["variables"]
  readonly generator: Generator<E, R>
  readonly eosTokens: ReadonlyArray<number>
  readonly chatHeaderEnd?: number
}): (request: GenerationRequest) => Stream.Stream<GenerationEvent, ModelError, Exclude<R, Scope.Scope>> =>
(request: GenerationRequest): Stream.Stream<GenerationEvent, ModelError, Exclude<R, Scope.Scope>> =>
  Stream.callback<GenerationEvent, ModelError, R>((queue) =>
    Effect.gen(function*() {
      const chat = "messages" in request.input

      const prompt = "prompt" in request.input ? request.input.prompt : yield* options.tokenizer.applyChatTemplate(
        options.template,
        request.input.messages,
        { addGenerationPrompt: true, variables: options.variables }
      )

      const encoded = yield* options.tokenizer.encode(prompt, { addSpecialTokens: !chat })

      if (encoded.data.length === 0) {
        return yield* new ModelError({ message: "Prompt must contain tokens", invalidRequest: true })
      }

      const decoder = options.tokenizer.decodeStream({ skipSpecialTokens: true })
      let header = chat && options.chatHeaderEnd !== undefined
      let stopped = false
      let count = 0

      const finishReason = yield* options.generator.run({
        prompt: encoded.data,
        maxTokens: request.maxTokens,
        eosTokens: options.eosTokens,
        settings: request,
        onPage: (page) =>
          Effect.gen(function*() {
            for (const token of page.tokens) {
              if (stopped) {
                break
              }

              count++

              if (options.eosTokens.includes(token)) {
                stopped = true
                break
              }

              if (header) {
                if (token === options.chatHeaderEnd) header = false

                continue
              }

              const delta = yield* decoder.step(token).pipe(
                Effect.mapError((error) => new ModelError({ message: error.message, invalidRequest: false }))
              )

              if (delta !== undefined) yield* Queue.offer(queue, { _tag: "Delta", text: delta })
            }
          })
      })

      yield* Queue.offer(queue, {
        _tag: "Done",
        finishReason: stopped || finishReason === "stop" ? "stop" : "length",
        promptTokens: encoded.data.length,
        completionTokens: count
      })
      yield* Queue.end(queue)
    }).pipe(
      Effect.mapError((error) =>
        error instanceof ModelError ? error : new ModelError({ message: String(error), invalidRequest: false })
      ),
      Effect.catchCause((cause) => Queue.failCause(queue, cause))
    ), { bufferSize: 16 })

/**
 * Text encoding and decoding shared by the generation families.
 *
 * @since 0.1.0
 * @category models
 */
export interface TextOptions {
  readonly id: string
  readonly tokenizer: Pick<Tokenizers.Tokenizer, "encode" | "applyChatTemplate" | "decodeStream">
  readonly template: string
  readonly variables?: Tokenizers.ChatTemplateOptions["variables"]
  readonly eosTokens: ReadonlyArray<number>
  readonly chatHeaderEnd?: number
}

/**
 * Bind a model-owned token generator without selecting or deriving a model family.
 * The application's generation operation owns its sessions and temporary state.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromGeneration = <E>(
  options: TextOptions & {
    readonly generator: Generator<E, Runtime.Runtime>
    readonly validateGeneration?: (request: GenerationRequest) => Effect.Effect<void, ModelError>
  }
): Effect.Effect<GenerationRegistration, never, Runtime.Runtime> =>
  Effect.gen(function*() {
    const backend = yield* Runtime.Runtime
    const validateGeneration = options.validateGeneration ?? (() => Effect.void)

    const generate = textGeneration({
      ...options,
      generator: {
        run: (request: CoreGeneration.RequestWithSettings<GenerationRequest, ModelError>) =>
          validateGeneration(request.settings).pipe(
            Effect.andThen(() => options.generator.run(request))
          )
      }
    })

    return {
      id: options.id,
      validateGeneration,
      generate: (request: GenerationRequest): Stream.Stream<GenerationEvent, ModelError> =>
        generate(request).pipe(Stream.provideService(Runtime.Runtime, backend))
    }
  })

/**
 * A registered model with a required decision operation.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecisionRegistration extends Registration {
  readonly decide: NonNullable<Registration["decide"]>
}

/**
 * Model-owned prompt preparation and conversion from an independent read to the
 * scorer's input. Prefix and temporary input acquisitions belong to their scopes.
 * A causal adapter can ignore canvas noise and select next-position row zero.
 *
 * @since 0.1.0
 * @category models
 */
export interface DecisionOptions<Prefix, Input, E, PrepareError> {
  readonly id: string
  readonly scorer: CoreDecision.Scorer<Input, E, Runtime.Runtime>
  readonly prepare: (plan: DecisionPlanner.QuestionPlan) => Effect.Effect<DecisionScaffold.Scaffold, PrepareError>
  readonly prefill: (ids: Uint32Array) => Effect.Effect<Prefix, E, Scope.Scope | Runtime.Runtime>
  readonly input: (
    prefix: Prefix,
    canvas: Uint32Array,
    slot: number
  ) => Effect.Effect<{ readonly input: Input; readonly row: number }, E, Scope.Scope | Runtime.Runtime>
  readonly mapError: (error: E | PrepareError) => ModelError
  readonly maxActiveQuestions?: number
  readonly maxCachedPrefixes?: number
  readonly maxPrefixTokens?: number
  readonly cacheIdleMs?: number
}

/**
 * Register a Decision scorer from either family or a custom evaluator.
 * Selected tensors are released after readback, including failed or interrupted
 * calls. The scoped engine drains calls before releasing cached prefixes.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromDecision = <Prefix, Input, E, PrepareError>(
  options: DecisionOptions<Prefix, Input, E, PrepareError>
): Effect.Effect<DecisionRegistration, ModelError, Scope.Scope | Runtime.Runtime> =>
  Effect.gen(function*() {
    const backend = yield* Runtime.Runtime

    const engine = yield* DecisionEngine.make({
      modelId: options.id,
      prepare: (plan) => options.prepare(plan).pipe(Effect.mapError(options.mapError)),
      runtime: {
        prefill: (ids) =>
          options.prefill(ids).pipe(
            Effect.mapError(options.mapError),
            Effect.provideService(Runtime.Runtime, backend)
          ),
        read: (prefix, canvas, slot, labels) =>
          Effect.scoped(Effect.gen(function*() {
            const prepared = yield* options.input(prefix, canvas, slot).pipe(Effect.mapError(options.mapError))
            const logits = yield* Effect.acquireRelease(
              options.scorer.score([prepared.input], { rows: [prepared.row], labels: Array.from(labels) }).pipe(
                Effect.mapError(options.mapError)
              ),
              Tensor.clear,
              { interruptible: true }
            )

            if (
              logits.shape.length !== 3 || logits.shape[0] !== 1 || logits.shape[1] !== 1 ||
              logits.shape[2] !== labels.length
            ) {
              return yield* new ModelError({
                message: "Decision scorer returned an invalid shape",
                invalidRequest: false
              })
            }

            if (!["f16", "bf16", "f32", "f64"].includes(logits.dtype)) {
              return yield* new ModelError({
                message: "Decision scorer must return floating logits",
                invalidRequest: false
              })
            }

            return yield* Tensor.toNumberArray(logits)
          })).pipe(
            Effect.mapError((error) =>
              error instanceof ModelError ? error : new ModelError({ message: String(error), invalidRequest: false })
            ),
            Effect.provideService(Runtime.Runtime, backend)
          )
      },
      maxActiveQuestions: options.maxActiveQuestions ?? 1,
      maxCachedPrefixes: options.maxCachedPrefixes ?? 2,
      maxPrefixTokens: options.maxPrefixTokens ?? 4096,
      cacheIdleMs: options.cacheIdleMs ?? 60000
    }).pipe(Effect.mapError((error) => new ModelError({ message: error.message, invalidRequest: false })))

    return {
      id: options.id,
      decide: (request: Decision.Request, settings: DecisionSettings): Effect.Effect<Decision.Response, ModelError> =>
        engine.run(request, settings).pipe(Effect.mapError((error) =>
          error instanceof ModelError
            ? error
            : new ModelError({ message: error.message, invalidRequest: error._tag === "ValidationError" })
        ))
    }
  })

type GenerationFactory<A, E> = (
  artifact: A
) => Omit<TextOptions, "id"> & {
  readonly generator: Generator<E, Runtime.Runtime>
  readonly validateGeneration?: (request: GenerationRequest) => Effect.Effect<void, ModelError>
}

type DecisionFactory<A, Prefix, Input, E, PrepareError> = (
  artifact: A
) => Omit<DecisionOptions<Prefix, Input, E, PrepareError>, "id">

interface CapabilityOptions<A, Prefix, Input, E, PrepareError, GenerationError> {
  readonly id: string
  readonly generation?: GenerationFactory<A, GenerationError>
  readonly decision?: DecisionFactory<A, Prefix, Input, E, PrepareError>
}

/**
 * Loads one autoregressive definition and derives its serving operations from
 * the same compiled artifact. The caller owns the supplied parameters.
 *
 * @since 0.1.0
 * @category loading
 */
export interface AutoRegressiveLoadOptions<Prefix, Input, E, PrepareError, GenerationError>
  extends CapabilityOptions<AutoRegressive.Artifact, Prefix, Input, E, PrepareError, GenerationError>
{
  readonly family: "AutoRegressive"
  readonly definition: CoreModel.Definition
  readonly parameters: CoreModel.Parameters
  readonly compile: AutoRegressive.CompileOptions
}

/**
 * Loads one diffusion definition and derives its serving operations from the
 * same compiled artifact. The caller owns the supplied parameters.
 *
 * @since 0.1.0
 * @category loading
 */
export interface DiffusionLoadOptions<Prefix, Input, E, PrepareError, GenerationError>
  extends CapabilityOptions<Diffusion.Artifact, Prefix, Input, E, PrepareError, GenerationError>
{
  readonly family: "Diffusion"
  readonly definition: Diffusion.Definition
  readonly parameters: CoreModel.Parameters
  readonly compile: Diffusion.CompileOptions
}

const register = <A, Prefix, Input, E, PrepareError, GenerationError>(
  options: CapabilityOptions<A, Prefix, Input, E, PrepareError, GenerationError>,
  artifact: A
): Effect.Effect<Registration, ModelError, Scope.Scope | Runtime.Runtime> =>
  Effect.gen(function*() {
    if (options.generation === undefined && options.decision === undefined) {
      return yield* new ModelError({
        message: "A loaded model must provide generation or decision",
        invalidRequest: true
      })
    }

    const generation = options.generation === undefined
      ? undefined
      : yield* fromGeneration({ id: options.id, ...options.generation(artifact) })

    const decision = options.decision === undefined
      ? undefined
      : yield* fromDecision({ id: options.id, ...options.decision(artifact) })

    if (generation !== undefined && decision !== undefined) {
      return {
        id: options.id,
        validateGeneration: generation.validateGeneration,
        generate: generation.generate,
        decide: decision.decide
      }
    }

    if (generation !== undefined) {
      return {
        id: options.id,
        validateGeneration: generation.validateGeneration,
        generate: generation.generate
      }
    }

    return { id: options.id, decide: decision!.decide }
  })

/**
 * Compiles an existing core definition and registers generation, decisions, or
 * both from that one artifact. This function does not choose a backend.
 *
 * @since 0.1.0
 * @category loading
 */
export const load = <Prefix, Input, E, PrepareError, GenerationError>(
  options:
    | AutoRegressiveLoadOptions<Prefix, Input, E, PrepareError, GenerationError>
    | DiffusionLoadOptions<Prefix, Input, E, PrepareError, GenerationError>
): Effect.Effect<Registration, ModelError, Scope.Scope | Runtime.Runtime> =>
  Effect.gen(function*() {
    if (options.id.length === 0) {
      return yield* new ModelError({ message: "Model id must not be empty", invalidRequest: true })
    }

    if (options.family === "AutoRegressive") {
      const artifact = yield* AutoRegressive.compile(options.definition, options.parameters, options.compile).pipe(
        Effect.mapError((error) => new ModelError({ message: String(error), invalidRequest: false }))
      )

      return yield* register(options, artifact)
    }

    const artifact = yield* Diffusion.compile(options.definition, options.parameters, options.compile).pipe(
      Effect.mapError((error) => new ModelError({ message: String(error), invalidRequest: false }))
    )

    return yield* register(options, artifact)
  })

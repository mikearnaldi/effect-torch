import * as BackendCpu from "@effect-torch/backend-cpu"
import {
  AutoRegressive,
  Decision as CoreDecision,
  Diffusion,
  Model as CoreModel,
  Runtime,
  Tensor
} from "@effect-torch/core"
import { describe, expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber, Stream } from "effect"
import { Decision, type DecisionPlanner, type DecisionScaffold } from "../src/index.ts"
import * as Model from "../src/Model.ts"

const tokenizer: Parameters<typeof Model.textGeneration>[0]["tokenizer"] = {
  applyChatTemplate: (_template, messages) => Effect.succeed(messages.map((message) => message.content).join(" ")),
  encode: () => Effect.succeed({ data: Uint32Array.of(1, 2), dtype: "u32", shape: [2] }),
  decodeStream: () => ({ step: (token) => Effect.succeed(String(token)) })
}

const request: Model.GenerationRequest = { input: { prompt: "hello" }, maxTokens: 3, seed: 42, temperature: 0 }

// Synthetic token mapping for family integration, independent of checkpoint templates.
const scaffoldFor = (plan: DecisionPlanner.QuestionPlan): DecisionScaffold.Scaffold => ({
  semanticKey: plan.semanticKey,
  model: plan.model,
  prompt: plan.prompt,
  tokenizerSha256: "test",
  chatTemplateSha256: "test",
  vocabularySize: 5,
  labels: plan.options.map((option, index) => ({ code: String(index), optionName: option.name, tokenId: index })),
  slot: 0,
  contentLength: 2,
  prefixIds: Uint32Array.of(1, 2),
  canvasIds: Uint32Array.of(0, 1),
  allowedTokenIds: Uint32Array.of(0, 1)
})

const makeDiffusion = Effect.gen(function*() {
  const embedding = yield* Tensor.fromTypedArray(Float32Array.of(1, 0, 0, 1, 1, 1, 2, -1, -1, 2), [5, 2])
  const head = yield* Tensor.fromTypedArray(Float32Array.of(1, 0, 1, 2, -1, 0, 1, 1, -1, 2), [2, 5])

  const hidden = (
    parameters: ReadonlyArray<Tensor.Any>,
    tokens: Tensor.Any,
    positions: Tensor.Any,
    causal: boolean,
    prediction: Diffusion.Prediction
  ) =>
    Effect.gen(function*() {
      const width = tokens.shape[1]!
      const embeddedTokens = yield* Tensor.embedding(tokens, { weight: parameters[0]! })
      const position = yield* Tensor.reshape(yield* Tensor.cast(positions, "f32"), [1, width, 1])
      const offset = yield* Tensor.mul(position, yield* Tensor.constantLike(position, 0.125))
      let embedded = yield* Tensor.add(embeddedTokens, offset)

      if (prediction._tag === "Refinement") {
        embedded = yield* Tensor.add(
          embedded,
          yield* Tensor.slice(prediction.logits, { start: [0, 0, 0], end: [1, width, 2] })
        )
      }

      const heads = yield* Tensor.transpose(yield* Tensor.reshape(embedded, [1, width, 1, 2]), [0, 2, 1, 3])
      const attended = yield* Tensor.scaledDotProductAttention(heads, heads, heads, { causal, layerId: 0 })

      return yield* Tensor.reshape(yield* Tensor.transpose(attended, [0, 2, 1, 3]), [1, width, 2])
    })

  const definition: Diffusion.Definition = {
    parameterSpecs: [embedding, head].map((parameter, index) => ({
      name: `parameter.${index}`,
      shape: parameter.shape,
      initializer: { _tag: "Constant" as const, value: 0 }
    })),
    vocabSize: 5,
    canvasLength: 2,
    maxPositions: 32,
    dtype: "f32",
    predictionDtype: "f32",
    encode: (parameters, tokens, positions) => hidden(parameters, tokens, positions, true, { _tag: "Initial" }),
    denoise: (parameters, tokens, positions, prediction) => hidden(parameters, tokens, positions, false, prediction),
    readout: (parameters, hidden, selection) =>
      Effect.gen(function*() {
        const logits = yield* Tensor.matmul(hidden, parameters[1]!)

        if (selection._tag === "Full") return logits

        return yield* Tensor.take(yield* Tensor.take(logits, selection.rows, { dim: 1 }), selection.labels, { dim: 2 })
      })
  }

  return yield* Diffusion.compile(definition, [embedding, head], {
    maxTokens: 32,
    blockSize: 4,
    prefillChunks: [4],
    selectedReadouts: [{ rows: 1, labels: 2 }]
  })
})

describe("generation adapters", () => {
  it.effect("loads generation and decisions from one compiled artifact", () =>
    Effect.gen(function*() {
      const definition = yield* CoreModel.chain(
        yield* CoreModel.embedding("embedding", 6, 4),
        yield* CoreModel.linear("head", 4, 6)
      )
      const parameters = yield* Effect.acquireRelease(
        Tensor.compute(yield* CoreModel.initialize(definition)),
        Tensor.clearAll,
        { interruptible: true }
      )
      const artifacts: Array<AutoRegressive.Artifact> = []

      const loaded = yield* Model.load({
        family: "AutoRegressive",
        id: "loaded",
        definition,
        parameters,
        compile: { maxTokens: 32, blockSize: 4, prefillChunks: [4], batchSize: 1 },
        generation: (artifact) => {
          artifacts.push(artifact)

          return {
            tokenizer,
            template: "test",
            eosTokens: [],
            generator: {
              run: ({ prompt, onPage, settings }) =>
                Effect.gen(function*() {
                  const input = yield* Tensor.fromTypedArray(prompt, [1, prompt.length])
                  const result = yield* AutoRegressive.generate(artifact, input, {
                    maxTokens: settings.maxTokens,
                    eosTokens: [],
                    onPage: (tokens) => onPage({ tokens })
                  })

                  return result.stop === "eos" ? "stop" as const : "maxTokens" as const
                })
            }
          }
        },
        decision: (artifact) => {
          artifacts.push(artifact)

          return {
            scorer: CoreDecision.fromAutoRegressive(artifact),
            prepare: (plan) => Effect.succeed(scaffoldFor(plan)),
            prefill: (ids) => Tensor.fromTypedArray(ids, [1, ids.length]),
            input: (prompt) => Effect.succeed({ input: prompt, row: 0 }),
            mapError: (
              error:
                | CoreDecision.DecisionError
                | AutoRegressive.InferenceError
                | CoreModel.ModelError
                | Tensor.TensorError
            ) => new Model.ModelError({ message: error.message, invalidRequest: false })
          }
        }
      })

      expect(artifacts).toHaveLength(2)
      expect(artifacts[0]).toBe(artifacts[1])
      expect(loaded.generate).toBeDefined()
      expect(loaded.decide).toBeDefined()
    }).pipe(Effect.scoped, Effect.provide(BackendCpu.layer)))

  it.effect("uses one real CPU artifact for chat and completion and closes both sessions", () =>
    Effect.gen(function*() {
      const definition = yield* CoreModel.chain(
        yield* CoreModel.embedding("embedding", 6, 4),
        yield* CoreModel.multiHeadAttention("attention", 4, 1, { causal: true, rope: 10000 }),
        yield* CoreModel.linear("head", 4, 6)
      )

      const parameters = yield* Effect.acquireRelease(
        Tensor.compute(yield* CoreModel.initialize(definition)),
        Tensor.clearAll,
        { interruptible: true }
      )

      const program = yield* AutoRegressive.compile(definition, parameters, {
        maxTokens: 64,
        blockSize: 4,
        prefillChunks: [4],
        batchSize: 1
      })

      let closed = 0

      const tracked: AutoRegressive.Artifact = {
        ...program,
        generation: () =>
          program.generation().pipe(Effect.map((session) => ({
            ...session,
            close: () =>
              session.close().pipe(Effect.tap(() =>
                Effect.sync(() => {
                  closed++
                })
              ))
          })))
      }

      const model = yield* Model.fromGeneration({
        id: "tiny",
        tokenizer,
        template: "test",
        eosTokens: [],
        generator: {
          run: ({ prompt, onPage, settings }) =>
            Effect.gen(function*() {
              const tensor = yield* Tensor.fromTypedArray(prompt, [1, prompt.length])
              const result = yield* AutoRegressive.generate(tracked, tensor, {
                maxTokens: settings.maxTokens,
                eosTokens: [],
                sampling: { seed: settings.seed ?? 42, temperature: settings.temperature ?? 0 },
                onPage: (tokens) => onPage({ tokens })
              })

              return result.stop === "eos" ? "stop" : "maxTokens"
            })
        }
      })

      const decision = yield* Model.fromDecision({
        id: "tiny",
        scorer: CoreDecision.fromAutoRegressive(program),
        prepare: (plan) => Effect.succeed(scaffoldFor(plan)),
        prefill: (ids) => Tensor.fromTypedArray(ids, [1, ids.length]),
        input: (prompt) => Effect.succeed({ input: prompt, row: 0 }),
        mapError: (error) => new Model.ModelError({ message: error.message, invalidRequest: false })
      })

      const question = yield* Decision.decodeRequest({
        model: "tiny",
        state: "state",
        questions: { answer: { type: "noul" } }
      })
      const before = yield* decision.decide(question, { seed: "test", reads: 4 })

      const completion = yield* Stream.runCollect(model.generate(request))

      const chat = yield* Stream.runCollect(
        model.generate({ ...request, input: { messages: [{ role: "user", content: "hello" }] } })
      )

      expect(completion).toEqual(chat)
      expect(completion.at(-1)).toEqual({ _tag: "Done", finishReason: "length", promptTokens: 2, completionTokens: 3 })
      expect(closed).toBe(2)
      expect(yield* decision.decide(question, { seed: "test", reads: 4 })).toEqual(before)
      expect(yield* decision.decide(question, { seed: "different", reads: 1 })).toEqual({
        ...before,
        usage: { ...before.usage, output_tokens: 2 }
      })

      const prompt = yield* Tensor.fromTypedArray(Uint32Array.of(1, 2), [1, 2])
      const failure = new Model.ModelError({ message: "page consumer failed", invalidRequest: false })
      expect(
        yield* Effect.flip(AutoRegressive.generate(tracked, prompt, {
          maxTokens: 3,
          eosTokens: [],
          onPage: () => Effect.fail(failure)
        }))
      ).toBe(failure)
      expect(closed).toBe(3)

      const started = yield* Deferred.make<void>()
      const fiber = yield* AutoRegressive.generate(tracked, prompt, {
        maxTokens: 3,
        eosTokens: [],
        onPage: () => Deferred.succeed(started, undefined).pipe(Effect.andThen(Effect.never))
      }).pipe(Effect.forkChild)

      yield* Deferred.await(started)
      yield* Fiber.interrupt(fiber)
      expect(closed).toBe(4)
      expect(yield* decision.decide(question, { seed: "test", reads: 4 })).toEqual(before)
    }).pipe(Effect.scoped, Effect.provide(BackendCpu.layer)))

  it.effect("keeps diffusion decisions independent across generation on the same compiled artifact", () =>
    Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime
      let releases = 0
      const outputs: Array<Tensor.Concrete> = []

      yield* Effect.scoped(Effect.gen(function*() {
        const program = yield* makeDiffusion
        const scorer = CoreDecision.fromDiffusion(program)

        const decision = yield* Model.fromDecision({
          id: "tiny",
          scorer: CoreDecision.fromEvaluator((inputs: ReadonlyArray<CoreDecision.DiffusionInput>, selection) =>
            scorer.score(inputs, selection).pipe(Effect.tap((tensor) =>
              Effect.sync(() => {
                outputs.push(tensor)
              })
            ))
          ),
          prepare: (plan) => Effect.succeed(scaffoldFor(plan)),
          prefill: (ids) =>
            Effect.acquireRelease(
              program.encode(ids),
              (prefix) =>
                Effect.orDie(program.release(prefix)).pipe(Effect.tap(() =>
                  Effect.sync(() => {
                    releases++
                  })
                )),
              { interruptible: true }
            ),
          input: (prefix, canvas, slot) => Effect.succeed({ input: { prefix, canvas }, row: slot }),
          mapError: (error) => new Model.ModelError({ message: error.message, invalidRequest: false })
        })

        const generation = yield* Model.fromGeneration({
          id: "tiny",
          tokenizer,
          template: "test",
          eosTokens: [],
          generator: {
            run: ({ prompt, onPage, settings }) =>
              program.generate<Array<number>, number, Model.ModelError | Tensor.TensorError, Runtime.Runtime>({
                prompt,
                maxNewTokens: settings.maxTokens,
                outputLimit: "exact",
                initialize: () =>
                  Effect.succeed({
                    value: { canvas: Uint32Array.of(0, 1), feedback: { _tag: "Initial" } },
                    release: Effect.void
                  }),
                process: (logits) =>
                  Effect.gen(function*() {
                    const prediction = yield* Tensor.toNumberArray(logits)
                    const feedback = (yield* Tensor.compute([yield* Tensor.zeros([1, 2, 5])]))[0]

                    return { value: { prediction, feedback }, release: Tensor.clear(feedback) }
                  }),
                policy: {
                  canvasLength: 2,
                  maxSteps: 1,
                  start: () => 0,
                  refine: ({ prediction, state }) => {
                    const draft = Uint32Array.from([0, 1], (row) => {
                      const logits = prediction.slice(row * 5, row * 5 + 5)
                      return logits.indexOf(Math.max(...logits))
                    })

                    return Effect.succeed({ state, canvas: draft.slice(), draft, done: true })
                  },
                  finish: (tokens) => ({ tokens, stop: false })
                },
                onPage: (tokens) => onPage({ tokens })
              }).pipe(Effect.map((result) => result.stop === "length" ? "maxTokens" : "stop"))
          }
        })

        const model: Model.SharedRegistration = { ...generation, decide: decision.decide }
        const question = yield* Decision.decodeRequest({
          model: "tiny",
          state: "state",
          questions: { answer: { type: "noul" } }
        })
        const settings: Model.DecisionSettings = { seed: "diffusion", reads: 4 }
        const before = yield* model.decide(question, settings)
        const pages = yield* Stream.runCollect(model.generate(request))
        expect(pages.at(-1)).toEqual({ _tag: "Done", finishReason: "length", promptTokens: 2, completionTokens: 3 })
        expect(yield* model.decide(question, settings)).toEqual(before)

        const renamed = yield* Decision.decodeRequest({
          ...question,
          questions: { renamed: question.questions.answer, duplicate: question.questions.answer }
        })
        const after = yield* model.decide(renamed, settings)
        expect(after.answers.renamed).toEqual(before.answers.answer)
        expect(after.answers.duplicate).toEqual(before.answers.answer)
        expect(after.usage).toEqual(before.usage)
        expect(releases).toBe(0)
        expect(outputs).toHaveLength(12)

        for (const tensor of outputs) {
          expect((yield* Effect.flip(runtime.readback(tensor))).reason).toBe("invalid-handle")
        }
      }))

      expect(releases).toBe(1)
    }).pipe(Effect.provide(BackendCpu.layer)))

  it.effect("strips the diffusion channel header and stops decoding at EOS inside a committed page", () =>
    Effect.gen(function*() {
      const generate = Model.textGeneration({
        tokenizer,
        template: "test",
        eosTokens: [106],
        chatHeaderEnd: 101,
        generator: {
          run: ({ onPage }) => onPage({ tokens: Uint32Array.of(100, 3, 101, 4, 5, 106, 0, 0) }).pipe(Effect.as("stop"))
        }
      })

      const events = yield* Stream.runCollect(
        generate({ input: { messages: [{ role: "user", content: "Hi" }] }, maxTokens: 8 })
      )

      expect(events).toEqual([
        { _tag: "Delta", text: "4" },
        { _tag: "Delta", text: "5" },
        { _tag: "Done", finishReason: "stop", promptTokens: 2, completionTokens: 6 }
      ])
    }))

  it.effect("propagates typed producer failures instead of leaving the stream open", () =>
    Effect.gen(function*() {
      const failure = new Model.ModelError({ message: "bad request", invalidRequest: true })

      const generate = Model.textGeneration({
        tokenizer,
        template: "test",
        eosTokens: [],
        generator: { run: () => Effect.fail(failure) }
      })

      expect(yield* Effect.flip(Stream.runCollect(generate(request)))).toBe(failure)
    }))
})

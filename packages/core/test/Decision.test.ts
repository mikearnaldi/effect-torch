import { describe, expect, it } from "@effect/vitest"
import { Context, Data, Deferred, Effect, Exit, Fiber } from "effect"
import { expectTypeOf } from "vitest"
import * as AutoRegressive from "../src/AutoRegressive.ts"
import * as Decision from "../src/Decision.ts"
import * as Diffusion from "../src/Diffusion.ts"
import * as Model from "../src/Model.ts"
import * as Runtime from "../src/Runtime.ts"
import * as Tensor from "../src/Tensor.ts"
import { deep, onDevices } from "./utils/devices.ts"

describe("Decision readouts", () => {
  it.effect("normalizes complete selected sets stably at extreme logits and uniformity", () =>
    Effect.gen(function*() {
      expect(yield* Decision.restrictedSoftmax([0, 0, 0, 0])).toEqual([0.25, 0.25, 0.25, 0.25])
      expect(yield* Decision.restrictedSoftmax([1e308, -1e308])).toEqual([1, 0])
      const p = yield* Decision.restrictedSoftmax([1000, 1001, 999])
      const shifted = yield* Decision.restrictedSoftmax([0, 1, -1])
      expect(p).toEqual(shifted)
      expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 15)
      expect(yield* Decision.restrictedSoftmax([123])).toEqual([1])
    }))

  it.effect("averages independent probabilities rather than logits", () =>
    Effect.gen(function*() {
      const p = yield* Decision.independentProbabilities([[10, 0], [-2, 0]], 2)
      expect(p[0]).toBeCloseTo((1 / (1 + Math.exp(-10)) + 1 / (1 + Math.exp(2))) / 2, 14)
      expect(p[0]).toBeLessThan(0.57)
      expect((yield* Decision.restrictedSoftmax([4, 0]))[0]).toBeGreaterThan(0.98)
    }))

  it.effect("rejects missing, sparse, non-finite and malformed distributions", () =>
    Effect.gen(function*() {
      for (const values of [[], [NaN], [Infinity], [-Infinity], Array<number>(2)]) {
        expect((yield* Effect.flip(Decision.restrictedSoftmax(values)))._tag).toBe("DecisionError")
      }

      expect((yield* Effect.flip(Decision.restrictedSoftmax([1], 2)))._tag).toBe("DecisionError")
      expect((yield* Effect.flip(Decision.independentProbabilities([], 2)))._tag).toBe("DecisionError")
      expect((yield* Effect.flip(Decision.independentProbabilities(Array<ReadonlyArray<number>>(1), 2)))._tag)
        .toBe("DecisionError")
      expect((yield* Effect.flip(Decision.meanProbabilities(Array<ReadonlyArray<number>>(1), 2)))._tag).toBe(
        "DecisionError"
      )

      for (const values of [[0, 0], [-0.1, 1.1], [0.4, 0.4], [NaN, 1], [1]]) {
        expect((yield* Effect.flip(Decision.meanProbabilities([values], 2)))._tag).toBe("DecisionError")
      }
    }))

  it.effect("decodes binary, categorical ties and explicitly ordered expected values", () =>
    Effect.gen(function*() {
      expect(yield* Decision.binaryProbability([0.2, 0.8])).toBe(0.8)
      expect(yield* Decision.binaryProbability([0.2, 0.8], 0)).toBe(0.2)
      expect(yield* Decision.categorical([0.5, 0.5], ["second", "first"])).toEqual({
        index: 0,
        value: "second",
        probabilities: [0.5, 0.5]
      })
      expect(yield* Decision.categorical([0.1, 0.9], ["same", "same"])).toEqual({
        index: 1,
        value: "same",
        probabilities: [0.1, 0.9]
      })
      expect(yield* Decision.expectedValue([0.2, 0.3, 0.5], [-10, 0, 20])).toBe(8)
      expect((yield* Effect.flip(Decision.expectedValue([1], [Infinity])))._tag).toBe("DecisionError")
      expect((yield* Effect.flip(Decision.categorical([1], [])))._tag).toBe("DecisionError")
    }))

  it.effect("retains an evaluator's typed failure and service requirement", () => {
    class EvaluationError extends Data.TaggedError("EvaluationError") {}
    class EvaluationConfig
      extends Context.Service<EvaluationConfig, { readonly fail: boolean }>()("EvaluationConfig")
    {}
    const failure = new EvaluationError()

    const scorer = Decision.fromEvaluator((_inputs: ReadonlyArray<string>, _selection: Decision.Selection) =>
      Effect.flatMap(EvaluationConfig, () => Effect.fail(failure))
    )

    expectTypeOf(scorer).toEqualTypeOf<Decision.Scorer<string, EvaluationError, EvaluationConfig>>()

    return Effect.gen(function*() {
      const error = yield* Effect.flip(scorer.score(["prepared"], { rows: [0], labels: [1] })).pipe(
        Effect.provideService(EvaluationConfig, { fail: true })
      )

      expect(error).toBe(failure)
    })
  })
})

type Prepared = ReadonlyArray<ReadonlyArray<number>>

const evaluator = Decision.fromEvaluator((inputs: ReadonlyArray<Prepared>, selection: Decision.Selection) =>
  Effect.gen(function*() {
    const values = inputs.flatMap((input) =>
      selection.rows.flatMap((row) => selection.labels.map((label) => input[row]![label]!))
    )

    const graph = yield* Tensor.fromTypedArray(
      new Float32Array(values),
      [inputs.length, selection.rows.length, selection.labels.length]
    )

    const [output] = yield* Tensor.compute([graph])

    return output
  })
)

const options = { batchSize: 2, concurrency: 2 }

const selection = { rows: [0], labels: [0, 1] }

const decision: Decision.IndependentReads<Prepared> = { inputs: [[[10, 0]], [[-2, 0]], [[0, 0]]], selection }

const makeCausal = Effect.gen(function*() {
  const embedding = yield* Model.embedding("embedding", 6, 4)
  const attention = yield* Model.multiHeadAttention("attention", 4, 1, { causal: true, rope: 10000 })
  const head = yield* Model.linear("head", 4, 6)

  return yield* Model.chain(embedding, attention, head)
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
    selectedReadouts: [{ rows: 3, labels: 3 }]
  })
})

onDevices("Decision scoring", () => (it) => {
  it.effect("diffusion selected batches match full readout and preserve borrowed prefixes", () =>
    Effect.scoped(Effect.gen(function*() {
      const program = yield* makeDiffusion

      const prefix = yield* Effect.acquireRelease(
        program.encode(Uint32Array.of(1, 2, 3)),
        (prefix) => Effect.orDie(program.release(prefix)),
        { interruptible: true }
      )

      const metadata = { tokenCount: prefix.tokenCount, bytes: prefix.bytes }
      const inputs = [{ prefix, canvas: Uint32Array.of(0, 4) }, { prefix, canvas: Uint32Array.of(3, 1) }]
      const selection = { rows: [1, 0, 1], labels: [4, 2, 4] }
      const scorer = Decision.fromDiffusion(program)

      const selected = yield* Effect.acquireRelease(scorer.score(inputs, selection), Tensor.clear, {
        interruptible: true
      })

      const values = yield* Tensor.toNumberArray(selected)
      expect(selected.shape).toEqual([2, 3, 3])

      for (const [index, input] of inputs.entries()) {
        const full = yield* Effect.acquireRelease(
          program.evaluate(prefix, input.canvas, { _tag: "Initial" }),
          Tensor.clear,
          { interruptible: true }
        )

        const all = yield* Tensor.toNumberArray(full)
        const expected = selection.rows.flatMap((row) => selection.labels.map((label) => all[row * 5 + label]!))
        deep(values.slice(index * 9, (index + 1) * 9), expected)

        const isolated = yield* Effect.acquireRelease(scorer.score([input], selection), Tensor.clear, {
          interruptible: true
        })

        deep(yield* Tensor.toNumberArray(isolated), expected)
      }

      expect(yield* Tensor.toNumberArray(selected)).toEqual(values)
      expect({ tokenCount: prefix.tokenCount, bytes: prefix.bytes }).toEqual(metadata)
    })))

  for (const stop of ["failure", "interruption"] as const) {
    it.effect(`diffusion batch ${stop} releases preceding outputs and leaves the prefix usable`, () =>
      Effect.scoped(Effect.gen(function*() {
        const program = yield* makeDiffusion

        const prefix = yield* Effect.acquireRelease(
          program.encode(Uint32Array.of(1, 2)),
          (prefix) => Effect.orDie(program.release(prefix)),
          { interruptible: true }
        )

        const reached = yield* Deferred.make<void>()
        const produced: Array<Tensor.Concrete> = []
        let calls = 0

        const scorer = Decision.fromDiffusion({
          ...program,
          score: (prefix, canvas, rows, labels) =>
            Effect.suspend(() => {
              if (++calls === 2) {
                return stop === "failure"
                  ? Effect.fail(new Diffusion.InferenceError({ op: "score", message: "injected" }))
                  : Deferred.succeed(reached, undefined).pipe(Effect.andThen(Effect.never))
              }

              return program.score(prefix, canvas, rows, labels).pipe(Effect.tap((output) =>
                Effect.sync(() => {
                  produced.push(output)
                })
              ))
            })
        })

        const input = { prefix, canvas: Uint32Array.of(0, 4) }
        const selection = { rows: [1, 0, 1], labels: [4, 2, 4] }

        if (stop === "failure") {
          expect((yield* Effect.flip(scorer.score([input, input], selection)))._tag).toBe("DiffusionInferenceError")
        } else {
          const fiber = yield* Effect.forkChild(scorer.score([input, input], selection))
          yield* Effect.raceFirst(Deferred.await(reached), Fiber.join(fiber))
          yield* Fiber.interrupt(fiber)
        }

        const runtime = yield* Runtime.Runtime
        expect(produced).toHaveLength(1)
        expect((yield* Effect.flip(runtime.readback(produced[0]!))).reason).toBe("invalid-handle")
        const result = yield* Decision.fromDiffusion(program).score([input], selection)
        expect((yield* Tensor.toNumberArray(result)).every(Number.isFinite)).toBe(true)
        yield* Tensor.clear(result)
      })))
  }

  it.effect("diffusion interruption during final input release clears the unreturned joined output", () =>
    Effect.scoped(Effect.gen(function*() {
      const program = yield* makeDiffusion

      const prefix = yield* Effect.acquireRelease(
        program.encode(Uint32Array.of(1, 2)),
        (prefix) => Effect.orDie(program.release(prefix)),
        { interruptible: true }
      )

      const runtime = yield* Runtime.Runtime
      const releasing = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      const scored: Array<Tensor.Concrete> = []
      const projected: Array<Tensor.Concrete> = []
      let paused = false

      const service: Runtime.RuntimeService = {
        ...runtime,
        execute: (executable, invocation) =>
          runtime.execute(executable, invocation).pipe(Effect.onExit((exit) =>
            Effect.sync(() => {
              if (Exit.isSuccess(exit)) {
                projected.push(...exit.value.filter((value) =>
                  value.shape.length === 3 && value.shape[0] === 2 && value.shape[1] === 3 && value.shape[2] === 3
                ))
              }
            })
          )),
        release: (tensor) =>
          Effect.gen(function*() {
            // Scope finalizers run in reverse acquisition order; scored[0] is last.
            if (!paused && tensor === scored[0] && projected.length === 1) {
              paused = true
              yield* Deferred.succeed(releasing, undefined)
              yield* Deferred.await(resume)
            }

            yield* runtime.release(tensor)
          })
      }

      const scorer = Decision.fromDiffusion({
        ...program,
        score: (prefix, canvas, rows, labels) =>
          program.score(prefix, canvas, rows, labels).pipe(Effect.tap((output) =>
            Effect.sync(() => {
              scored.push(output)
            })
          ))
      })

      const canvas = Uint32Array.of(0, 4)
      const selection = { rows: [1, 0, 1], labels: [4, 2, 4] }

      const fiber = yield* Effect.forkChild(
        scorer.score([{ prefix, canvas }, { prefix, canvas }], selection).pipe(
          Effect.provideService(Runtime.Runtime, service)
        )
      )

      yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
      const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
      yield* Deferred.succeed(resume, undefined)
      yield* Fiber.join(interruption)
      expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
      yield* Effect.gen(function*() {
        expect(projected).toHaveLength(1)
        expect(scored).toHaveLength(2)

        for (const output of [...scored, ...projected]) {
          expect((yield* Effect.flip(runtime.readback(output))).reason).toBe("invalid-handle")
        }
      }).pipe(Effect.ensuring(Tensor.clearAll([...scored, ...projected])))
      expect(Array.from(canvas)).toEqual([0, 4])
      expect(prefix.tokenCount).toBe(2)

      const fresh = yield* Effect.acquireRelease(
        Decision.fromDiffusion(program).score([{ prefix, canvas }], selection),
        Tensor.clear,
        { interruptible: true }
      )

      expect((yield* Tensor.toNumberArray(fresh)).every(Number.isFinite)).toBe(true)
    })))

  it.effect("batches unequal read counts with probability weighting and ordered duplicate selections", () =>
    Effect.gen(function*() {
      const result = yield* Decision.scoreIndependently(evaluator, [decision], options)
      const expected = yield* Decision.independentProbabilities(decision.inputs.map((input) => input[0]!), 2)
      deep(result[0]![0], expected)
      const selected = { rows: [1, 0, 1], labels: [2, 0, 2] }
      const input = [[1, 20, 3], [4, 30, 6]]
      const logits = yield* evaluator.score([input, input], selected)
      expect(logits.shape).toEqual([2, 3, 3])
      expect(yield* Tensor.toNumberArray(logits)).toEqual([6, 4, 6, 3, 1, 3, 6, 4, 6, 6, 4, 6, 3, 1, 3, 6, 4, 6])
      yield* Tensor.clear(logits)
    }))

  it.effect("isolates questions across reordering, additions and duplicates", () =>
    Effect.gen(function*() {
      const sibling = { inputs: [[[0, 100]], [[0, 100]]], selection }
      const alone = yield* Decision.scoreIndependently(evaluator, [decision], options)
      const mixed = yield* Decision.scoreIndependently(evaluator, [sibling, decision, sibling, decision], options)
      expect(mixed[1]).toEqual(alone[0])
      expect(mixed[3]).toEqual(alone[0])

      const sequential = yield* Decision.scoreIndependently(evaluator, [decision, sibling], {
        batchSize: 1,
        concurrency: 1
      })

      expect(sequential).toEqual([alone[0], mixed[0]])
    }))

  it.effect("bounds concurrent batches globally and preserves completion-independent order", () =>
    Effect.gen(function*() {
      const reached = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      let active = 0
      let peak = 0
      const sizes: Array<number> = []

      const bounded = Decision.fromEvaluator((
        inputs: ReadonlyArray<Prepared>,
        selection: Decision.Selection
      ) =>
        Effect.gen(function*() {
          active++
          peak = Math.max(peak, active)
          sizes.push(inputs.length)

          return yield* Effect.gen(function*() {
            if (active === 2) yield* Deferred.succeed(reached, undefined)

            yield* Deferred.await(resume)

            return yield* evaluator.score(inputs, selection)
          }).pipe(Effect.ensuring(Effect.sync(() => {
            active--
          })))
        })
      )

      const fiber = yield* Effect.forkChild(
        Decision.scoreIndependently(bounded, [decision, decision, decision], options)
      )

      yield* Effect.raceFirst(Deferred.await(reached), Fiber.join(fiber))
      expect(active).toBe(2)
      yield* Deferred.succeed(resume, undefined)
      const result = yield* Fiber.join(fiber)
      expect(peak).toBe(2)
      expect(active).toBe(0)
      expect(sizes).toEqual([2, 1, 2, 1, 2, 1])
      expect(result[0]).toEqual(result[1])
      expect(result[1]).toEqual(result[2])
    }))

  it.effect("rejects invalid scheduling and selections before evaluation", () =>
    Effect.gen(function*() {
      let calls = 0

      const counting = Decision.fromEvaluator(
        (inputs: ReadonlyArray<Prepared>, selection: Decision.Selection) => {
          calls++

          return evaluator.score(inputs, selection)
        }
      )

      for (const invalid of [{ batchSize: 0, concurrency: 2 }, { batchSize: 2, concurrency: 1.5 }]) {
        expect((yield* Effect.flip(Decision.scoreIndependently(counting, [decision], invalid)))._tag).toBe(
          "DecisionError"
        )
      }

      for (const selection of [{ rows: [], labels: [0] }, { rows: [0], labels: [-1] }, { rows: [NaN], labels: [0] }]) {
        expect(
          (yield* Effect.flip(Decision.scoreIndependently(counting, [{ ...decision, selection }], options)))._tag
        ).toBe("DecisionError")
      }

      expect(
        (yield* Effect.flip(Decision.scoreIndependently(counting, [{ inputs: [], selection }], options)))._tag
      ).toBe("DecisionError")
      expect(yield* Decision.scoreIndependently(counting, [], options)).toEqual([])
      expect(calls).toBe(0)
    }))

  for (const failure of ["shape", "nonfinite", "readback"] as const) {
    it.effect(`releases scoring outputs on ${failure} failure`, () =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const produced: Array<Tensor.Concrete> = []

        const broken = Decision.fromEvaluator((
          inputs: ReadonlyArray<Prepared>,
          selection: Decision.Selection
        ) =>
          evaluator.score(
            failure === "nonfinite" ? inputs.map(() => [[NaN, 0]]) : inputs,
            failure === "shape" ? { ...selection, rows: [0, 0] } : selection
          ).pipe(Effect.tap((output) =>
            Effect.sync(() => {
              produced.push(output)
            })
          ))
        )

        const service: Runtime.RuntimeService = {
          ...runtime,
          readback: (tensor) =>
            failure === "readback"
              ? Effect.fail(
                new Runtime.BackendError({
                  backend: runtime.backend.name,
                  phase: "readback",
                  operation: "readback",
                  message: "injected failure",
                  reason: "invalid-handle"
                })
              )
              : runtime.readback(tensor)
        }

        const error = yield* Effect.flip(Decision.scoreIndependently(broken, [decision], options)).pipe(
          Effect.provideService(Runtime.Runtime, service)
        )

        expect(error._tag).toBe(failure === "readback" ? "TensorError" : "DecisionError")
        expect(produced.length).toBeGreaterThan(0)

        for (const output of produced) {
          expect((yield* Effect.flip(runtime.readback(output))).reason).toBe("invalid-handle")
        }
      }))
  }

  it.effect("cancellation drains readback before releasing outputs and preserves borrowed input", () =>
    Effect.scoped(Effect.gen(function*() {
      const runtime = yield* Runtime.Runtime

      const [input] = yield* Effect.acquireRelease(
        Tensor.compute([yield* Tensor.fromTypedArray(Float32Array.of(1, 2), [1, 1, 2])]),
        Tensor.clearAll,
        { interruptible: true }
      )

      const produced: Array<Tensor.Concrete> = []
      const reached = yield* Deferred.make<void>()
      let borrowing = false

      const service: Runtime.RuntimeService = {
        ...runtime,
        readback: (tensor) =>
          Effect.gen(function*() {
            produced.push(tensor)
            borrowing = true
            yield* Deferred.succeed(reached, undefined)

            return yield* Effect.never
          }).pipe(Effect.ensuring(Effect.sync(() => {
            borrowing = false
          }))),
        release: (tensor) =>
          Effect.gen(function*() {
            expect(borrowing).toBe(false)

            return yield* runtime.release(tensor)
          })
      }

      const borrowed = Decision.fromEvaluator((
        _inputs: ReadonlyArray<Tensor.Any>,
        _selection: Decision.Selection
      ) =>
        Effect.gen(function*() {
          const [output] = yield* Tensor.compute([yield* Tensor.neg(input)])

          return output
        })
      )

      const fiber = yield* Effect.forkChild(
        Decision.scoreIndependently(borrowed, [{ inputs: [input], selection }], options).pipe(
          Effect.provideService(Runtime.Runtime, service)
        )
      )

      yield* Effect.raceFirst(Deferred.await(reached), Fiber.join(fiber))
      yield* Fiber.interrupt(fiber)
      expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
      expect(produced).toHaveLength(1)
      expect((yield* Effect.flip(runtime.readback(produced[0]!))).reason).toBe("invalid-handle")
      expect(yield* Tensor.toNumberArray(input)).toEqual([1, 2])
    })))

  it.effect("AR scoring agrees with full causal readout, batches independently and retains caller outputs", () =>
    Effect.scoped(Effect.gen(function*() {
      const model = yield* makeCausal

      const params = yield* Effect.acquireRelease(Tensor.compute(yield* Model.initialize(model)), Tensor.clearAll, {
        interruptible: true
      })

      const program = yield* AutoRegressive.compile(model, params, {
        maxTokens: 32,
        blockSize: 4,
        prefillChunks: [4],
        batchSize: 2
      })

      const sessions: Array<AutoRegressive.StatefulExecution> = []

      const scorer = Decision.fromAutoRegressive({
        ...program,
        generation: () => Effect.die("Scoring must not open generation"),
        execution: () =>
          program.execution().pipe(Effect.tap((session) =>
            Effect.sync(() => {
              sessions.push(session)
            })
          ))
      })

      const prompts = yield* Effect.acquireRelease(
        Tensor.compute([
          yield* Tensor.fromTypedArray(Uint32Array.of(1, 2, 3), [1, 3]),
          yield* Tensor.fromTypedArray(Uint32Array.of(4, 1), [1, 2])
        ]),
        Tensor.clearAll,
        { interruptible: true }
      )

      const selection = { rows: [0, 0], labels: [5, 1, 5, 0] }

      const selected = yield* Effect.acquireRelease(scorer.score(prompts, selection), Tensor.clear, {
        interruptible: true
      })

      const before = yield* Tensor.toNumberArray(selected)
      expect(selected.shape).toEqual([2, 2, 4])

      for (const [index, prompt] of prompts.entries()) {
        const [full] = yield* Effect.acquireRelease(
          Tensor.compute([yield* model.forward(params, prompt)]),
          Tensor.clearAll,
          { interruptible: true }
        )

        const values = yield* Tensor.toNumberArray(full)
        const finalRow = values.slice(-6)
        const expected = selection.labels.map((label) => finalRow[label]!)
        deep(before.slice(index * 8, (index + 1) * 8), [...expected, ...expected])

        const isolated = yield* Effect.acquireRelease(scorer.score([prompt], selection), Tensor.clear, {
          interruptible: true
        })

        deep(yield* Tensor.toNumberArray(isolated), before.slice(index * 8, (index + 1) * 8))
      }

      for (let index = 0; index < 6; index++) {
        const output = yield* scorer.score([prompts[1]!], selection)
        yield* Tensor.clear(output)
      }

      expect(yield* Tensor.toNumberArray(selected)).toEqual(before)

      for (const session of sessions) expect(yield* session.live()).toBe(0)

      expect(yield* Tensor.toNumberArray(prompts[0]!)).toEqual([1, 2, 3])
    })))

  for (const phase of ["final logits release", "session close"] as const) {
    it.effect(`AR interruption during ${phase} clears the unreturned projection and preserves borrowed prompts`, () =>
      Effect.scoped(Effect.gen(function*() {
        const model = yield* makeCausal

        const params = yield* Effect.acquireRelease(Tensor.compute(yield* Model.initialize(model)), Tensor.clearAll, {
          interruptible: true
        })

        const program = yield* AutoRegressive.compile(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2
        })

        const prompts = yield* Effect.acquireRelease(
          Tensor.compute([
            yield* Tensor.fromTypedArray(Uint32Array.of(1, 2, 3), [1, 3]),
            yield* Tensor.fromTypedArray(Uint32Array.of(4, 1), [1, 2])
          ]),
          Tensor.clearAll,
          { interruptible: true }
        )

        const runtime = yield* Runtime.Runtime
        const releasing = yield* Deferred.make<void>()
        const resume = yield* Deferred.make<void>()
        const full: Array<Tensor.Concrete> = []
        const projected: Array<Tensor.Concrete> = []
        const sessions: Array<AutoRegressive.StatefulExecution> = []
        let paused = false

        const pause = Effect.gen(function*() {
          paused = true
          yield* Deferred.succeed(releasing, undefined)
          yield* Deferred.await(resume)
        })

        const service: Runtime.RuntimeService = {
          ...runtime,
          execute: (executable, invocation) =>
            runtime.execute(executable, invocation).pipe(Effect.onExit((exit) =>
              Effect.sync(() => {
                if (Exit.isSuccess(exit)) {
                  projected.push(...exit.value.filter((value) =>
                    value.shape.length === 3 && value.shape[0] === 2 && value.shape[1] === 1 && value.shape[2] === 2
                  ))
                }
              })
            )),
          release: (tensor) =>
            Effect.gen(function*() {
              if (
                phase === "final logits release" && !paused && tensor === full[1] && projected.length === 1
              ) yield* pause

              yield* runtime.release(tensor)
            })
        }

        const scorer = Decision.fromAutoRegressive({
          ...program,
          execution: () =>
            Effect.map(program.execution(), (session) => {
              sessions.push(session)

              return {
                ...session,
                add: (inputs) =>
                  session.add(inputs).pipe(Effect.tap((outputs) =>
                    Effect.sync(() => {
                      full.push(...outputs.map(({ logits }) => logits))
                    })
                  )),
                close: () =>
                  Effect.gen(function*() {
                    if (phase === "session close" && !paused && projected.length === 1) yield* pause

                    yield* session.close()
                  })
              }
            })
        })

        const fiber = yield* Effect.forkChild(
          scorer.score(prompts, { rows: [0], labels: [5, 0] }).pipe(
            Effect.provideService(Runtime.Runtime, service)
          )
        )

        yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
        const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
        yield* Deferred.succeed(resume, undefined)
        yield* Fiber.join(interruption)
        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        yield* Effect.gen(function*() {
          expect(projected).toHaveLength(1)
          expect(full).toHaveLength(2)

          for (const output of [...full, ...projected]) {
            expect((yield* Effect.flip(runtime.readback(output))).reason).toBe("invalid-handle")
          }
        }).pipe(Effect.ensuring(Tensor.clearAll([...full, ...projected])))

        for (const session of sessions) expect(yield* session.live()).toBe(0)

        expect(yield* Tensor.toNumberArray(prompts[0]!)).toEqual([1, 2, 3])
        expect(yield* Tensor.toNumberArray(prompts[1]!)).toEqual([4, 1])

        for (const parameter of params) expect((yield* Tensor.toNumberArray(parameter)).length).toBeGreaterThan(0)
      })))
  }

  it.effect("AR selection failures close state and clear full logits before retry", () =>
    Effect.scoped(Effect.gen(function*() {
      const model = yield* makeCausal

      const params = yield* Effect.acquireRelease(Tensor.compute(yield* Model.initialize(model)), Tensor.clearAll, {
        interruptible: true
      })

      const program = yield* AutoRegressive.compile(model, params, {
        maxTokens: 8,
        blockSize: 4,
        prefillChunks: [4],
        batchSize: 1
      })

      const sessions: Array<AutoRegressive.StatefulExecution> = []
      const full: Array<Tensor.Concrete> = []

      const scorer = Decision.fromAutoRegressive({
        ...program,
        execution: () =>
          Effect.map(program.execution(), (session) => {
            sessions.push(session)

            return {
              ...session,
              add: (inputs) =>
                session.add(inputs).pipe(Effect.tap((outputs) =>
                  Effect.sync(() => {
                    full.push(...outputs.map(({ logits }) => logits))
                  })
                ))
            }
          })
      })

      const prompt = yield* Tensor.fromTypedArray(Uint32Array.of(1, 2, 3), [1, 3])
      expect((yield* Effect.flip(scorer.score([prompt], { rows: [1], labels: [0] })))._tag).toBe("DecisionError")
      expect(sessions).toHaveLength(0)

      for (let index = 0; index < 3; index++) {
        expect((yield* Effect.flip(scorer.score([prompt], { rows: [0], labels: [6] })))._tag).toBe("DecisionError")
      }

      for (const session of sessions) expect(yield* session.live()).toBe(0)

      const runtime = yield* Runtime.Runtime

      for (const output of full) expect((yield* Effect.flip(runtime.readback(output))).reason).toBe("invalid-handle")

      const output = yield* scorer.score([prompt], { rows: [0], labels: [0, 1] })
      expect((yield* Tensor.toNumberArray(output)).every(Number.isFinite)).toBe(true)
      yield* Tensor.clear(output)
    })))
})

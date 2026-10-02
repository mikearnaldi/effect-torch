import { Diffusion, Runtime, Tensor } from "@effect-torch/core"
import * as DG from "@effect-torch/models/DiffusionGemma"
import { expect } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { readFileSync } from "node:fs"
import * as GenerationStatistics from "../src/internal/diffusionGemmaStatistics.ts"
import { onDevices } from "./utils/devices.ts"

interface RecordedTensor {
  readonly shape: ReadonlyArray<number>
  readonly values: ReadonlyArray<number>
}

interface Step {
  readonly remaining: number
  readonly raw_logits: RecordedTensor
  readonly exponentials: ReadonlyArray<number>
  readonly feedback_out: RecordedTensor
  readonly sampled_tokens: ReadonlyArray<number>
  readonly argmax_tokens: ReadonlyArray<number>
  readonly token_entropy: ReadonlyArray<number>
  readonly entropy_order: ReadonlyArray<number>
  readonly mean_entropy: number
  readonly canvas: ReadonlyArray<number>
  readonly random_canvas: ReadonlyArray<number>
}

const fixture: {
  readonly runs: ReadonlyArray<{
    readonly dtype: string
    readonly prompt: ReadonlyArray<number>
    readonly sequences: ReadonlyArray<number>
    readonly blocks: ReadonlyArray<
      {
        readonly initial_canvas: ReadonlyArray<number>
        readonly steps: ReadonlyArray<Step>
      }
    >
  }>
} = JSON.parse(readFileSync(new URL("./fixtures/diffusion-gemma-generation.json", import.meta.url), "utf8"))

onDevices("DiffusionGemma native sampling", () => (it) => {
  for (const run of fixture.runs) {
    it.effect(`${run.dtype} scalar temperature preserves seeded progression and borrowed logits`, () =>
      Effect.scoped(Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const processedKey = "EFFECT_TORCH_DIFFUSION_PROCESSED_EVALUATION97"
        yield* Effect.acquireRelease(Effect.sync(() => process.env[processedKey]), (previous) =>
          Effect.sync(() => {
            if (previous === undefined) delete process.env[processedKey]
            else process.env[processedKey] = previous
          }))
        const key = "EFFECT_TORCH_DIFFUSION_SCALAR_TEMPERATURE93"
        yield* Effect.acquireRelease(Effect.sync(() => process.env[key]), (previous) =>
          Effect.sync(() => {
            if (previous === undefined) delete process.env[key]
            else process.env[key] = previous
          }))
        const reference = run.blocks[0]!.steps[0]!.raw_logits
        const raw = yield* Tensor.fromTypedArray(Float32Array.from(reference.values), reference.shape)
        const [logits] = yield* Effect.acquireRelease(Tensor.compute([raw]), Tensor.clearAll)
        const execute = (enabled: boolean, processed = false) =>
          Effect.gen(function*() {
            process.env[key] = enabled ? "1" : "0"
            process.env[processedKey] = processed ? "1" : "0"
            const records: Array<unknown> = []
            let seededCompiles = 0
            let scalarCalls = 0
            const tracked: Runtime.RuntimeService = {
              ...runtime,
              compile: (request) => {
                if (request.options?.randomSeed !== undefined) seededCompiles++
                return runtime.compile(request)
              },
              execute: (handle, invocation) => {
                if (invocation.scalars.length === 1) scalarCalls++
                return runtime.execute(handle, invocation)
              }
            }
            const program: Diffusion.Artifact = {
              dtype: "f32",
              predictionDtype: run.dtype === "bfloat16" ? "bf16" : "f32",
              vocabSize: 37,
              canvasLength: 3,
              maxPositions: 2048,
              encode: () => Effect.die("unreachable"),
              commit: () => Effect.die("unreachable"),
              evaluate: () => Effect.die("unreachable"),
              evaluateProcessed: () => Effect.die("unreachable"),
              score: () => Effect.die("unreachable"),
              release: () => Effect.die("unreachable"),
              inspect: () => Effect.die("unreachable"),
              generate: (options) =>
                Effect.gen(function*() {
                  for (let index = 0; index < 3; index++) {
                    yield* Effect.scoped(Effect.gen(function*() {
                      if (options.processed !== undefined) {
                        const step = { index, remaining: 3 - index }
                        const outputs = yield* Effect.acquireRelease(
                          Tensor.runProgram(options.processed.processor, [logits!], [options.processed.scalar(step)]),
                          Tensor.clearAll
                        )
                        const host = Float32Array.from(yield* Tensor.toNumberArray(outputs[1]!))
                        records.push(
                          options.processed.decode(
                            host,
                            { index: 0, position: 2, remainingTokens: 3, canvasLength: 3 },
                            step
                          )
                        )
                        records.push(yield* Tensor.toNumberArray(outputs[0]!))
                        return
                      }
                      const output = yield* Effect.acquireRelease(
                        options.process(logits!, {
                          index: 0,
                          position: 2,
                          remainingTokens: 3,
                          canvasLength: 3
                        }, { index, remaining: 3 - index }),
                        (value) => value.release
                      )
                      records.push(output.value.prediction)
                      records.push(yield* Tensor.toNumberArray(output.value.feedback))
                    }))
                  }
                  return { generatedTokens: 0, blocks: 0, refinements: 3, stop: "length" as const }
                })
            }
            yield* DG.generate(program, new Uint32Array([1, 2]), {
              seed: 12345,
              maxNewTokens: 3,
              maxSteps: 3,
              eosTokenIds: []
            }).pipe(Effect.provideService(Runtime.Runtime, tracked))
            expect(seededCompiles).toBe(1)
            expect(scalarCalls).toBe(enabled || processed ? 3 : 0)
            expect(yield* Tensor.toNumberArray(logits!)).toEqual(Array.from(Float32Array.from(reference.values)))
            return records
          })
        const baseline = yield* execute(false)
        expect(yield* execute(true)).toEqual(baseline)
        expect(yield* execute(true)).toEqual(baseline)
        expect(yield* execute(false, true)).toEqual(baseline)
      })))

    it.effect(`${run.dtype} ergonomic generation delegates scheduling and reads back only reduced statistics`, () =>
      Effect.scoped(Effect.gen(function*() {
        const key = "EFFECT_TORCH_DIFFUSION_PROCESSED_EVALUATION97"
        yield* Effect.acquireRelease(Effect.sync(() => process.env[key]), (previous) =>
          Effect.sync(() => {
            if (previous === undefined) delete process.env[key]
            else process.env[key] = previous
          }))
        process.env[key] = "1"
        const runtime = yield* Runtime.Runtime
        let calls = 0
        let statisticsReads = 0

        const program: Diffusion.Artifact = {
          dtype: run.dtype === "bfloat16" ? "bf16" : "f32",
          predictionDtype: run.dtype === "bfloat16" ? "bf16" : "f32",
          vocabSize: 37,
          canvasLength: 3,
          maxPositions: 2048,
          encode: () => Effect.die("unreachable"),
          commit: () => Effect.die("unreachable"),
          evaluate: () => Effect.die("unreachable"),
          evaluateProcessed: () => Effect.die("unreachable"),
          score: () => Effect.die("unreachable"),
          release: () => Effect.die("unreachable"),
          inspect: () => Effect.die("unreachable"),
          generate: (options) => {
            expect(options.processed).toBeUndefined()
            calls++

            return Diffusion.runGeneration({
              ...options,
              callbacks: {
                encode: (tokens) => Effect.succeed({ value: tokens.length, release: Effect.void }),
                commit: (prefix, tokens) => Effect.succeed({ value: prefix + tokens.length, release: Effect.void }),
                initialize: options.initialize,
                evaluate: ({ canvas, feedback, block, step }) =>
                  Effect.scoped(Effect.gen(function*() {
                    const reference = run.blocks[block.index]!.steps[step.index]!
                    expect(Array.from(canvas)).toEqual(reference.canvas)

                    if (feedback._tag !== "Initial") expect(feedback.value.dtype).toBe(program.predictionDtype)

                    const raw = yield* Tensor.fromTypedArray(
                      Float32Array.from(reference.raw_logits.values),
                      reference.raw_logits.shape
                    )

                    const [logits] = yield* Effect.acquireRelease(Tensor.compute([raw]), Tensor.clearAll, {
                      interruptible: true
                    })

                    return yield* Effect.acquireRelease(options.process(logits!, block, step), (output, exit) =>
                      Exit.isFailure(exit) ? output.release : Effect.void, { interruptible: true })
                  }))
              }
            })
          }
        }

        const canvases = run.blocks.flatMap((
          block
        ) => [block.initial_canvas, ...block.steps.map((step) => step.random_canvas)])

        const exponentials = run.blocks.flatMap((block) => block.steps.map((step) => step.exponentials))
        let canvasIndex = 0
        let stepIndex = 0

        const service: Runtime.RuntimeService = {
          ...runtime,
          readback: (tensor) => {
            expect(tensor.shape).toEqual([3 * 4 + 1])
            statisticsReads++

            return runtime.readback(tensor)
          }
        }

        const generated = yield* DG.generate(program, Uint32Array.from(run.prompt), {
          maxNewTokens: 5,
          maxSteps: 3,
          confidenceThreshold: 1e-8,
          eosTokenIds: [],
          outputLimit: "exact",
          random: {
            canvas: () => Uint32Array.from(canvases[canvasIndex++]!),
            exponentials: () => Float32Array.from(exponentials[stepIndex++]!)
          }
        }).pipe(Effect.provideService(Runtime.Runtime, service))

        expect(calls).toBe(1)
        expect(statisticsReads).toBe(6)
        expect(Array.from(generated.tokens)).toEqual(run.sequences.slice(run.prompt.length, run.prompt.length + 5))
        expect(generated.randomInputBytes).toBe(6 * 3 * 37 * 4)
        expect(generated.statisticsReadbackBytes).toBe(6 * (3 * 4 + 1) * 4)
        expect(generated.processingMilliseconds).toBeGreaterThan(0)
        expect(canvasIndex).toBe(canvases.length)
        expect(stepIndex).toBe(exponentials.length)

        canvasIndex = 0
        stepIndex = 0
        const reached = yield* Deferred.make<void>()
        const produced: Array<Tensor.Concrete> = []

        const interrupted: Runtime.RuntimeService = {
          ...runtime,
          execute: (program, invocation) =>
            runtime.execute(program, invocation).pipe(Effect.map((outputs) => {
              produced.push(...outputs)

              return outputs
            })),
          readback: () => Deferred.succeed(reached, undefined).pipe(Effect.andThen(Effect.never))
        }

        const fiber = yield* Effect.forkChild(
          DG.generate(program, Uint32Array.from(run.prompt), {
            maxNewTokens: 5,
            maxSteps: 3,
            confidenceThreshold: 1e-8,
            eosTokenIds: [],
            random: {
              canvas: () => Uint32Array.from(canvases[canvasIndex++]!),
              exponentials: () => Float32Array.from(exponentials[stepIndex++]!)
            }
          }).pipe(Effect.provideService(Runtime.Runtime, interrupted))
        )

        yield* Deferred.await(reached)
        yield* Fiber.interrupt(fiber)
        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        expect(produced.length).toBeGreaterThan(2)

        for (const tensor of produced) {
          expect((yield* Effect.flip(runtime.readback(tensor))).reason).toBe("invalid-handle")
        }

        canvasIndex = 0
        stepIndex = 0
        produced.length = 0
        const failed = yield* Effect.exit(
          DG.generate(program, Uint32Array.from(run.prompt), {
            maxNewTokens: 5,
            maxSteps: 3,
            confidenceThreshold: 1e-8,
            eosTokenIds: [],
            random: {
              canvas: () => Uint32Array.from(canvases[canvasIndex++]!),
              exponentials: () => Float32Array.from(exponentials[stepIndex++]!)
            }
          }).pipe(Effect.provideService(Runtime.Runtime, {
            ...interrupted,
            readback: () =>
              Effect.fail(
                new Runtime.BackendError({
                  reason: "transfer-failed",
                  backend: "test",
                  operation: "readback",
                  phase: "readback",
                  message: "injected statistics readback failure"
                })
              )
          }))
        )
        expect(Exit.isFailure(failed)).toBe(true)
        expect(produced.length).toBeGreaterThan(2)
        for (const tensor of produced) {
          expect((yield* Effect.flip(runtime.readback(tensor))).reason).toBe("invalid-handle")
        }
      })))

    for (const scalar93 of [false, true]) {
      it.effect(`${run.dtype} scalar93=${scalar93} interruption during final statistics cleanup releases feedback and preserves borrowed logits`, () =>
        Effect.scoped(Effect.gen(function*() {
          const key = "EFFECT_TORCH_DIFFUSION_SCALAR_TEMPERATURE93"
          yield* Effect.acquireRelease(
            Effect.sync(() => {
              const previous = process.env[key]
              process.env[key] = scalar93 ? "1" : "0"
              return previous
            }),
            (previous) =>
              Effect.sync(() => {
                if (previous === undefined) delete process.env[key]
                else process.env[key] = previous
              })
          )
          const runtime = yield* Runtime.Runtime
          const reference = run.blocks[0]!.steps[0]!

          const raw = yield* Tensor.fromTypedArray(
            Float32Array.from(reference.raw_logits.values),
            reference.raw_logits.shape
          )

          const [borrowed] = yield* Effect.acquireRelease(Tensor.compute([raw]), Tensor.clearAll, {
            interruptible: true
          })
          const releasing = yield* Deferred.make<void>()
          const resume = yield* Deferred.make<void>()
          const produced: Array<Tensor.Concrete> = []
          let outputs: ReadonlyArray<Tensor.Concrete> = []
          let paused = false
          let readbacks = 0
          let borrowedReleased = false

          const tracked: Runtime.RuntimeService = {
            ...runtime,
            execute: (executable, invocation) =>
              runtime.execute(executable, invocation).pipe(Effect.onExit((exit) =>
                Effect.sync(() => {
                  if (Exit.isSuccess(exit)) {
                    produced.push(...exit.value)

                    if (invocation.bindings.includes(borrowed!)) outputs = exit.value
                  }
                })
              )),
            readback: (tensor) => {
              if (outputs.slice(1).includes(tensor)) readbacks++

              return runtime.readback(tensor)
            },
            release: (tensor) =>
              Effect.gen(function*() {
                if (tensor === borrowed) borrowedReleased = true

                if (!paused && tensor === outputs.at(-1)) {
                  paused = true
                  yield* Deferred.succeed(releasing, undefined)
                  yield* Deferred.await(resume)
                }

                yield* runtime.release(tensor)
              })
          }

          const program: Diffusion.Artifact = {
            dtype: run.dtype === "bfloat16" ? "bf16" : "f32",
            predictionDtype: run.dtype === "bfloat16" ? "bf16" : "f32",
            vocabSize: 37,
            canvasLength: 3,
            maxPositions: 2048,
            encode: () => Effect.die("unreachable"),
            commit: () => Effect.die("unreachable"),
            evaluate: () => Effect.die("unreachable"),
            evaluateProcessed: () => Effect.die("unreachable"),
            score: () => Effect.die("unreachable"),
            release: () => Effect.die("unreachable"),
            inspect: () => Effect.die("unreachable"),
            generate: (options) =>
              Effect.scoped(Effect.gen(function*() {
                yield* Effect.acquireRelease(
                  options.process(borrowed!, {
                    index: 0,
                    position: run.prompt.length,
                    remainingTokens: 3,
                    canvasLength: 3
                  }, {
                    index: 0,
                    remaining: 3
                  }),
                  (output) => output.release,
                  { interruptible: true }
                )

                return { generatedTokens: 0, blocks: 0, refinements: 1, stop: "length" as const }
              }))
          }

          yield* Effect.gen(function*() {
            const seededOptions = { maxNewTokens: 3, maxSteps: 3, eosTokenIds: [], seed: 12345 }
            const generationOptions = scalar93 ? seededOptions : {
              ...seededOptions,
              random: {
                canvas: () => Uint32Array.from(run.blocks[0]!.initial_canvas),
                exponentials: () => Float32Array.from(reference.exponentials)
              }
            }
            const fiber = yield* DG.generate(program, Uint32Array.from(run.prompt), generationOptions)
              .pipe(Effect.provideService(Runtime.Runtime, tracked), Effect.forkChild)

            yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
            const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
            yield* Deferred.succeed(resume, undefined)
            yield* Fiber.join(interruption)
            expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
            expect(outputs).toHaveLength(2)
            expect(outputs[0]!.dtype).toBe(program.predictionDtype)
            expect(readbacks).toBe(1)
            expect(borrowedReleased).toBe(false)
            expect(yield* Tensor.toNumberArray(borrowed!)).toEqual(
              Array.from(Float32Array.from(reference.raw_logits.values))
            )

            for (const tensor of produced) {
              expect((yield* Effect.flip(runtime.readback(tensor))).reason).toBe("invalid-handle")
            }
          }).pipe(Effect.ensuring(Tensor.clearAll(produced)))
        })))
    }

    it.effect(`${run.dtype} replays exponential draws and keeps feedback in its actual dtype`, () =>
      Effect.scoped(Effect.gen(function*() {
        const dtype = run.dtype === "bfloat16" ? "bf16" : "f32"

        const compiled = yield* Tensor.compile(
          ([logits, draws, temperature]) => DG.generationStatistics(logits!, draws!, temperature!, dtype)
        )
        const packedCompiled = yield* Tensor.compile(
          ([logits, draws, temperature]) =>
            DG.generationStatistics(logits!, draws!, temperature!, dtype).pipe(
              Effect.flatMap((statistics) => GenerationStatistics.pack(statistics, 3, 37))
            )
        )

        for (const block of run.blocks) {
          for (const step of block.steps) {
            yield* Effect.scoped(Effect.gen(function*() {
              const logits = yield* Tensor.fromTypedArray(
                Float32Array.from(step.raw_logits.values),
                step.raw_logits.shape
              )

              const draws = yield* Tensor.fromTypedArray(Float32Array.from(step.exponentials), step.raw_logits.shape)
              const temperature = yield* Tensor.full([], DG.generationTemperature(0.4, 0.8, 3, step.remaining))

              const outputs = yield* Effect.acquireRelease(
                compiled.call([logits, draws, temperature]),
                Tensor.clearAll,
                { interruptible: true }
              )

              expect(outputs[0]!.dtype).toBe(dtype)
              expect(outputs).toHaveLength(6)

              const [feedback, sampled, argmax, entropy, order, mean] = yield* Effect.forEach(
                outputs,
                Tensor.toNumberArray
              )
              const packedOutputs = yield* Effect.acquireRelease(
                packedCompiled.call([logits, draws, temperature]),
                Tensor.clearAll,
                { interruptible: true }
              )
              expect(packedOutputs).toHaveLength(2)
              const packed = yield* Tensor.toNumberArray(packedOutputs[1]!)
              expect(GenerationStatistics.unpack(packed, 3)).toEqual([sampled, argmax, entropy, order, mean])
              expect(yield* Tensor.toNumberArray(packedOutputs[0]!)).toEqual(feedback)

              expect(sampled).toEqual(step.sampled_tokens)
              expect(argmax).toEqual(step.argmax_tokens)
              expect(order).toEqual(step.entropy_order)
              entropy!.forEach((value, index) => expect(value).toBeCloseTo(step.token_entropy[index]!, 5))
              expect(mean![0]).toBeCloseTo(step.mean_entropy, 5)
              feedback!.forEach((value, index) => {
                const expected = step.feedback_out.values[index]!

                const tolerance = dtype === "bf16"
                  ? Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(expected))) - 7))
                  : 2e-5 + 2e-5 * Math.abs(expected)

                expect(Math.abs(value - expected)).toBeLessThanOrEqual(tolerance)
              })
            }))
          }
        }

        expect((yield* compiled.stats).compiled).toBe(1)
      })))
  }

  it.effect("statistics packing preserves exact integer bounds, signed zero and nonfinite classes", () =>
    Effect.scoped(Effect.gen(function*() {
      expect(GenerationStatistics.canPackIds(2 ** 24 + 1, 2 ** 24 + 1)).toBe(true)
      for (const size of [0, -1, 1.5, 2 ** 24 + 2, Infinity, NaN]) {
        expect(GenerationStatistics.canPackIds(size, 37)).toBe(false)
        expect(GenerationStatistics.canPackIds(3, size)).toBe(false)
      }
      const statistics = [
        yield* Tensor.fromTypedArray(new Float32Array([1])),
        yield* Tensor.fromTypedArray(new Uint32Array([2 ** 24, 2 ** 24 - 1, 0])),
        yield* Tensor.fromTypedArray(new Uint32Array([0, 2 ** 24, 2 ** 24 - 1])),
        yield* Tensor.fromTypedArray(new Float32Array([-0, NaN, Infinity])),
        yield* Tensor.fromTypedArray(new Uint32Array([2, 1, 0])),
        yield* Tensor.fromTypedArray(new Float32Array([-0]))
      ]
      const packed = yield* GenerationStatistics.pack(statistics, 3, 2 ** 24 + 1)
      const outputs = yield* Effect.acquireRelease(Tensor.compute(packed), Tensor.clearAll, { interruptible: true })
      const decoded = GenerationStatistics.unpack(yield* Tensor.toNumberArray(outputs[1]!), 3)
      expect(decoded[0]).toEqual([2 ** 24, 2 ** 24 - 1, 0])
      expect(decoded[1]).toEqual([0, 2 ** 24, 2 ** 24 - 1])
      expect(Object.is(decoded[2][0], -0)).toBe(true)
      expect(Number.isNaN(decoded[2][1])).toBe(true)
      expect(decoded[2][2]).toBe(Infinity)
      expect(decoded[3]).toEqual([2, 1, 0])
      expect(Object.is(decoded[4][0], -0)).toBe(true)
      expect(yield* GenerationStatistics.pack(statistics, 3, 2 ** 24 + 2)).toBe(statistics)
      for (const index of [3, 5]) {
        const wider = statistics.slice()
        wider[index] = yield* Tensor.fromTypedArray(new Float64Array(index === 3 ? [1, 2, 3] : [1]))
        expect(yield* GenerationStatistics.pack(wider, 3, 37)).toBe(wider)
      }
    })))

  it.effect("seeded random streams replay while retaining positive exponential draws", () =>
    Effect.gen(function*() {
      const first = yield* DG.generationRandom(0)
      const replay = yield* DG.generationRandom(0)
      const different = yield* DG.generationRandom(1)
      const canvas = first.canvas(128, 37)
      expect(canvas).toEqual(replay.canvas(128, 37))
      expect(canvas).not.toEqual(different.canvas(128, 37))
      expect(canvas.every((token) => token < 37)).toBe(true)
      const exponentials = first.exponentials(128)
      expect(exponentials).toEqual(replay.exponentials(128))
      expect(exponentials.every((value) => value > 0 && Number.isFinite(value))).toBe(true)
      expect(() => first.canvas(1, 0x1_0000_0001)).toThrow(RangeError)
      expect((yield* Effect.flip(DG.generationRandom(-1)))._tag).toBe("ModelError")
    }))
})

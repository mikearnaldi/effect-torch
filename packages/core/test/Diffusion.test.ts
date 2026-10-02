import { expect } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import * as Diffusion from "../src/Diffusion.ts"
import * as Runtime from "../src/Runtime.ts"
import * as Tensor from "../src/Tensor.ts"
import { deep, onDevices } from "./utils/devices.ts"

const ids = (values: ReadonlyArray<number>) => new Uint32Array(values)

const initial: Diffusion.Prediction = { _tag: "Initial" }

const config: Diffusion.CompileOptions = {
  maxTokens: 64,
  blockSize: 2,
  prefillChunks: [1, 2, 4],
  canvasLengths: [2, 3],
  selectedReadouts: [{ rows: 2, labels: 3 }, { rows: 1, labels: 1 }]
}

const fixture = Effect.gen(function*() {
  const parameters = [
    yield* Tensor.fromTypedArray(new Float32Array([0.1, 0.4, 0.8, -0.2, -0.5, 0.6, 0.3, -0.7]), [4, 2]),
    yield* Tensor.fromTypedArray(new Float32Array([0.3, -0.4, 0.6, 0.1, -0.2, 0.5, 0.9, -0.8]), [2, 4])
  ]

  const embed = (
    parameters: ReadonlyArray<Tensor.Any>,
    tokens: Tensor.Any,
    positions: Tensor.Any,
    prediction: Diffusion.Prediction
  ) =>
    Effect.gen(function*() {
      const embeddings = yield* Tensor.embedding(tokens, { weight: parameters[0]! })
      const position = yield* Tensor.reshape(yield* Tensor.cast(positions, "f32"), [1, tokens.shape[1]!, 1])
      const offset = yield* Tensor.mul(position, yield* Tensor.constantLike(position, 0.125))
      let hidden = yield* Tensor.add(embeddings, offset)

      if (prediction._tag === "Refinement") {
        const feedback = yield* Tensor.slice(prediction.logits, { start: [0, 0, 0], end: [1, tokens.shape[1]!, 2] })
        hidden = yield* Tensor.add(hidden, feedback)
      }

      return yield* Tensor.reshape(hidden, [1, 1, tokens.shape[1]!, 2])
    })

  const attention = (
    parameters: ReadonlyArray<Tensor.Any>,
    tokens: Tensor.Any,
    positions: Tensor.Any,
    prediction: Diffusion.Prediction,
    causal: boolean
  ) =>
    Effect.gen(function*() {
      const heads = yield* embed(parameters, tokens, positions, prediction)
      const attended = yield* Tensor.scaledDotProductAttention(heads, heads, heads, { causal, scale: 0.5 })

      return yield* Tensor.reshape(attended, [1, tokens.shape[1]!, 2])
    })

  const definition: Diffusion.Definition = {
    parameterSpecs: parameters.map((parameter, index) => ({
      name: `parameter.${index}`,
      shape: parameter.shape,
      initializer: { _tag: "Constant", value: 0 }
    })),
    vocabSize: 4,
    canvasLength: 2,
    maxPositions: 32,
    dtype: "f32",
    predictionDtype: "f32",
    encode: (parameters, tokens, positions) => attention(parameters, tokens, positions, initial, true),
    denoise: (parameters, tokens, positions, prediction) => attention(parameters, tokens, positions, prediction, false),
    readout: (parameters, hidden, selection) =>
      Effect.gen(function*() {
        const rows = selection._tag === "Full" ? hidden : yield* Tensor.take(hidden, selection.rows, { dim: 1 })

        const head = selection._tag === "Full"
          ? parameters[1]!
          : yield* Tensor.take(parameters[1]!, selection.labels, { dim: 1 })

        return yield* Tensor.matmul(rows, head)
      })
  }

  // Ordinary attention over explicit keys is an independent numerical oracle.
  const reference = (prompt: Uint32Array, canvas: Uint32Array, prediction: Diffusion.Prediction = initial) =>
    Effect.gen(function*() {
      const tokens = yield* Tensor.fromTypedArray(canvas, [1, canvas.length])

      const positions = yield* Tensor.fromTypedArray(Uint32Array.from(canvas, (_, row) => prompt.length + row), [
        1,
        canvas.length
      ])

      const heads = yield* embed(parameters, tokens, positions, prediction)
      let keys: Tensor.Any = heads

      if (prompt.length > 0) {
        const prefix = yield* embed(
          parameters,
          yield* Tensor.fromTypedArray(prompt, [1, prompt.length]),
          yield* Tensor.fromTypedArray(Uint32Array.from(prompt, (_, row) => row), [1, prompt.length]),
          initial
        )

        keys = yield* Tensor.concat([prefix, heads], { dim: 2 })
      }

      const attended = yield* Tensor.scaledDotProductAttention(heads, keys, keys, { causal: false, scale: 0.5 })
      const hidden = yield* Tensor.reshape(attended, [1, canvas.length, 2])
      const [result] = yield* Tensor.compute([yield* definition.readout(parameters, hidden, { _tag: "Full" })])

      return result!
    })

  return { definition, parameters, reference }
})

const ownPrefix = (
  program: Diffusion.Artifact,
  effect: ReturnType<Diffusion.Artifact["encode"]>
) => Effect.acquireRelease(effect, (prefix) => Effect.orDie(program.release(prefix)), { interruptible: true })

const ownTensor = <E, R>(effect: Effect.Effect<Tensor.Concrete, E, R>) =>
  Effect.acquireRelease(effect, Tensor.clear, { interruptible: true })

onDevices("Diffusion", () => (it) => {
  it.effect("processed evaluation falls back before execution and preserves device and host outputs", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
      const logits = yield* ownTensor(program.evaluate(prefix, ids([0, 3]), initial))
      const input = yield* Tensor.makeInput(0, logits)
      const scalar = yield* Tensor.makeScalarInput(1, "f32")
      const processor = yield* Tensor.freezeProgram([
        yield* Tensor.mul(input, scalar),
        yield* Tensor.reshape(input, [8])
      ])
      const expected = yield* Tensor.toNumberArray(logits)
      const runtime = yield* Runtime.Runtime
      let admissions = 0
      let bodyExecutions = 0
      const tracked: Runtime.RuntimeService = {
        ...runtime,
        extensions: {
          ...runtime.extensions,
          decode: {
            ...runtime.extensions.decode,
            prepareProcessedReadOnly: () =>
              Effect.sync(() => {
                admissions++
                return undefined
              })
          }
        },
        execute: (executable, invocation) => {
          if (invocation.state?.access === "ReadOnly") bodyExecutions++
          return runtime.execute(executable, invocation)
        }
      }
      const result = yield* Effect.acquireRelease(
        program.evaluateProcessed(prefix, ids([0, 3]), initial, processor, 2).pipe(
          Effect.provideService(Runtime.Runtime, tracked)
        ),
        (value) => Tensor.clear(value.device),
        { interruptible: true }
      )
      expect(admissions).toBe(1)
      expect(bodyExecutions).toBe(1)
      deep(Array.from(result.host), expected)
      deep(yield* Tensor.toNumberArray(result.device), expected.map((value) => value * 2))
      const refined = yield* Effect.acquireRelease(
        program.evaluateProcessed(prefix, ids([0, 3]), { _tag: "Refinement", logits: result.device }, processor, 0.5),
        (value) => Tensor.clear(value.device),
        { interruptible: true }
      )
      expect(refined.host).toHaveLength(8)
      deep(Array.from(result.host), expected)
    })))

  it.effect("processed output is reclaimed if input finalization is interrupted", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, { ...config, cachePositions: false })
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
      const input = yield* Tensor.makeInput(0, yield* Tensor.zeros([1, 2, 4]))
      const scalar = yield* Tensor.makeScalarInput(1, "f32")
      const processor = yield* Tensor.freezeProgram([
        yield* Tensor.mul(input, scalar),
        yield* Tensor.reshape(input, [8])
      ])
      const [device] = yield* Tensor.compute([yield* Tensor.zeros([1, 2, 4])])
      const runtime = yield* Runtime.Runtime
      const releasing = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      let position: Tensor.Concrete | undefined
      let cleared = false
      const tracked: Runtime.RuntimeService = {
        ...runtime,
        extensions: {
          ...runtime.extensions,
          decode: {
            ...runtime.extensions.decode,
            prepareProcessedReadOnly: () =>
              Effect.succeed({
                execute: (request) =>
                  Effect.sync(() => {
                    position = request.bindings[0]
                    return { device: device!, host: new Float32Array(8) }
                  })
              })
          }
        },
        release: (value) =>
          Effect.gen(function*() {
            if (value === position) {
              yield* Deferred.succeed(releasing, undefined)
              yield* Deferred.await(resume)
            }
            yield* runtime.release(value)
            if (value === device) cleared = true
          })
      }
      const fiber = yield* program.evaluateProcessed(prefix, ids([0, 3]), initial, processor, 1).pipe(
        Effect.provideService(Runtime.Runtime, tracked),
        Effect.forkChild
      )
      yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
      const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
      yield* Deferred.succeed(resume, undefined)
      yield* Fiber.join(interruption)
      expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
      expect(cleared).toBe(true)
    })))

  it.effect("processed fallback readback failure releases both processor outputs", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
      const input = yield* Tensor.makeInput(0, yield* Tensor.zeros([1, 2, 4]))
      const scalar = yield* Tensor.makeScalarInput(1, "f32")
      const processor = yield* Tensor.freezeProgram([
        yield* Tensor.mul(input, scalar),
        yield* Tensor.reshape(input, [8])
      ])
      const runtime = yield* Runtime.Runtime
      let outputs: ReadonlyArray<Tensor.Concrete> = []
      const released = new Set<Tensor.Concrete>()
      const tracked: Runtime.RuntimeService = {
        ...runtime,
        extensions: {
          ...runtime.extensions,
          decode: { ...runtime.extensions.decode, prepareProcessedReadOnly: undefined }
        },
        execute: (handle, invocation) =>
          runtime.execute(handle, invocation).pipe(Effect.tap((values) =>
            Effect.sync(() => {
              if (handle === processor.handle) outputs = values
            })
          )),
        readback: () =>
          Effect.fail(
            new Runtime.BackendError({
              backend: "test",
              operation: "readback",
              phase: "execute",
              reason: "execution-failed",
              message: "injected"
            })
          ),
        release: (value) =>
          runtime.release(value).pipe(Effect.tap(() =>
            Effect.sync(() => {
              released.add(value)
            })
          ))
      }
      const exit = yield* Effect.exit(
        program.evaluateProcessed(prefix, ids([0, 3]), initial, processor, 1).pipe(
          Effect.provideService(Runtime.Runtime, tracked)
        )
      )
      expect(Exit.isFailure(exit)).toBe(true)
      expect(outputs).toHaveLength(2)
      expect(outputs.every((value) => released.has(value))).toBe(true)
    })))

  it.effect("processed execution failure never retries the body", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
      const exemplar = yield* Tensor.zeros([1, 2, 4])
      const input = yield* Tensor.makeInput(0, exemplar)
      const scalar = yield* Tensor.makeScalarInput(1, "f32")
      const processor = yield* Tensor.freezeProgram([
        yield* Tensor.mul(input, scalar),
        yield* Tensor.reshape(input, [8])
      ])
      const runtime = yield* Runtime.Runtime
      let attempts = 0
      let bodyExecutions = 0
      const tracked: Runtime.RuntimeService = {
        ...runtime,
        extensions: {
          ...runtime.extensions,
          decode: {
            ...runtime.extensions.decode,
            prepareProcessedReadOnly: () =>
              Effect.succeed({
                execute: () => {
                  attempts++
                  return Effect.fail(
                    new Runtime.BackendError({
                      backend: "test",
                      operation: "processed",
                      phase: "execute",
                      reason: "execution-failed",
                      message: "injected"
                    })
                  )
                }
              })
          }
        },
        execute: (executable, invocation) => {
          if (invocation.state?.access === "ReadOnly") bodyExecutions++
          return runtime.execute(executable, invocation)
        }
      }
      const exit = yield* Effect.exit(
        program.evaluateProcessed(prefix, ids([0, 3]), initial, processor, 1).pipe(
          Effect.provideService(Runtime.Runtime, tracked)
        )
      )
      expect(Exit.isFailure(exit)).toBe(true)
      expect(attempts).toBe(1)
      expect(bodyExecutions).toBe(0)
    })))

  it.effect("shares concrete prefix positions across concurrent reads and isolates committed prefixes", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters, reference } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, { ...config, cachePositions: true })
      const runtime = yield* Runtime.Runtime
      const reads: Array<Tensor.Concrete> = []
      const tracked: Runtime.RuntimeService = {
        ...runtime,
        execute: (executable, invocation) => {
          if (invocation.state?.access === "ReadOnly") reads.push(invocation.bindings[1]!)
          return runtime.execute(executable, invocation)
        }
      }
      const prefix = yield* program.encode(ids([1, 2]))
      const outputs = yield* Effect.all(
        [ids([0, 3]), ids([1, 0]), ids([3, 2])].map((canvas) =>
          ownTensor(program.evaluate(prefix, canvas, initial).pipe(Effect.provideService(Runtime.Runtime, tracked)))
        ),
        { concurrency: "unbounded" }
      )
      expect(reads).toHaveLength(3)
      expect(reads[0]).toBe(reads[1])
      expect(reads[0]).toBe(reads[2])
      expect(yield* Tensor.toNumberArray(reads[0]!)).toEqual([2, 3])
      const expected = yield* ownTensor(reference(ids([1, 2]), ids([0, 3])))
      deep(yield* Tensor.toNumberArray(outputs[0]!), yield* Tensor.toNumberArray(expected))
      yield* ownTensor(
        program.score(prefix, ids([0, 3]), [0], [0]).pipe(Effect.provideService(Runtime.Runtime, tracked))
      )
      expect(reads[3]).toBe(reads[0])
      yield* ownTensor(
        program.evaluate(prefix, ids([0, 3, 1]), initial).pipe(Effect.provideService(Runtime.Runtime, tracked))
      )
      expect(reads[4]).not.toBe(reads[0])
      expect(yield* Tensor.toNumberArray(reads[4]!)).toEqual([2, 3, 4])
      const next = yield* ownPrefix(program, program.commit(prefix, ids([1])))
      yield* ownTensor(
        program.evaluate(next, ids([0, 3]), initial).pipe(Effect.provideService(Runtime.Runtime, tracked))
      )
      expect(reads[5]).not.toBe(reads[0])
      expect(yield* Tensor.toNumberArray(reads[5]!)).toEqual([3, 4])
      yield* program.release(prefix)
      for (const positions of [reads[0]!, reads[4]!]) {
        expect((yield* Effect.flip(runtime.readback(positions))).reason).toBe("invalid-handle")
      }
      expect((yield* Effect.flip(program.evaluate(prefix, ids([0, 3]), initial)))._tag).toBe("TensorError")
      expect(yield* Tensor.toNumberArray(reads[5]!)).toEqual([3, 4])
      deep(yield* Tensor.toNumberArray(outputs[0]!), yield* Tensor.toNumberArray(expected))
    })))

  for (const interrupted of [false, true]) {
    it.effect(`cleans a partially constructed position cache after ${interrupted ? "interruption" : "failure"}`, () =>
      Effect.scoped(Effect.gen(function*() {
        const { definition, parameters } = yield* fixture
        const program = yield* Diffusion.compile(definition, parameters, { ...config, cachePositions: true })
        const runtime = yield* Runtime.Runtime
        const reached = yield* Deferred.make<void>()
        let snapshot: Runtime.KvPrefixHandle | undefined
        let releasedSnapshot: Runtime.KvPrefixHandle | undefined
        const cached: Array<Tensor.Concrete> = []
        const tracked: Runtime.RuntimeService = {
          ...runtime,
          compile: (request) => {
            if (snapshot !== undefined && cached.length > 0) {
              return interrupted
                ? Deferred.succeed(reached, undefined).pipe(Effect.andThen(Effect.never))
                : Effect.fail(
                  new Runtime.BackendError({
                    backend: runtime.backend.name,
                    operation: "position-cache",
                    phase: "compile",
                    reason: "execution-failed",
                    message: "injected position cache failure"
                  })
                )
            }
            return runtime.compile(request)
          },
          execute: (executable, invocation) =>
            runtime.execute(executable, invocation).pipe(Effect.tap((outputs) =>
              Effect.sync(() => {
                if (snapshot !== undefined && invocation.state === undefined) cached.push(...outputs)
              })
            )),
          extensions: {
            ...runtime.extensions,
            decode: {
              ...runtime.extensions.decode,
              snapshot: (sequence) =>
                runtime.extensions.decode.snapshot(sequence).pipe(Effect.tap((value) =>
                  Effect.sync(() => {
                    snapshot = value
                  })
                )),
              releasePrefix: (value) =>
                runtime.extensions.decode.releasePrefix(value).pipe(Effect.tap(() =>
                  Effect.sync(() => {
                    releasedSnapshot = value
                  })
                ))
            }
          }
        }
        const acquire = program.encode(ids([1, 2])).pipe(Effect.provideService(Runtime.Runtime, tracked))
        if (interrupted) {
          const fiber = yield* Effect.forkChild(acquire)
          yield* Effect.raceFirst(Deferred.await(reached), Fiber.join(fiber))
          yield* Fiber.interrupt(fiber)
          expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        } else {
          expect(Exit.isFailure(yield* Effect.exit(acquire))).toBe(true)
        }
        expect(snapshot).toBeDefined()
        expect(releasedSnapshot).toBe(snapshot)
        expect(cached).toHaveLength(1)
        expect((yield* Effect.flip(runtime.readback(cached[0]!))).reason).toBe("invalid-handle")
        const usable = yield* ownPrefix(program, program.encode(ids([1, 2])))
        yield* ownTensor(program.evaluate(usable, ids([0, 3]), initial))
      })))
  }

  it.effect("clears cached positions when native prefix release fails without masking the error", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, { ...config, cachePositions: true })
      const runtime = yield* Runtime.Runtime
      const prefix = yield* program.encode(ids([1, 2]))
      let positions: Tensor.Concrete | undefined
      const tracked: Runtime.RuntimeService = {
        ...runtime,
        execute: (executable, invocation) => {
          if (invocation.state?.access === "ReadOnly") positions = invocation.bindings[1]
          return runtime.execute(executable, invocation)
        },
        extensions: {
          ...runtime.extensions,
          decode: {
            ...runtime.extensions.decode,
            releasePrefix: (value) =>
              runtime.extensions.decode.releasePrefix(value).pipe(Effect.andThen(
                Effect.fail(
                  new Runtime.BackendError({
                    backend: runtime.backend.name,
                    operation: "release-prefix",
                    phase: "shutdown",
                    reason: "execution-failed",
                    message: "injected prefix release failure"
                  })
                )
              ))
          }
        }
      }
      yield* ownTensor(
        program.evaluate(prefix, ids([0, 3]), initial).pipe(Effect.provideService(Runtime.Runtime, tracked))
      )
      const error = yield* Effect.flip(program.release(prefix).pipe(Effect.provideService(Runtime.Runtime, tracked)))
      expect(error.message).toContain("injected prefix release failure")
      expect((yield* Effect.flip(runtime.readback(positions!))).reason).toBe("invalid-handle")
    })))

  it.effect("fuses full readout while preserving refinement, selected reads, and retained outputs", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const baseline = yield* Diffusion.compile(definition, parameters, config)
      const fused = yield* Diffusion.compile(definition, parameters, { ...config, fuseFullReadout: true })
      const prompt = ids([0, 1, 2])
      const left = yield* ownPrefix(baseline, baseline.encode(prompt))
      const right = yield* ownPrefix(fused, fused.encode(prompt))
      const canvas = ids([2, 3])
      const expected = yield* ownTensor(baseline.evaluate(left, canvas, initial))
      const actual = yield* ownTensor(fused.evaluate(right, canvas, initial))
      const values = yield* Tensor.toNumberArray(expected)
      expect(yield* Tensor.toNumberArray(actual)).toEqual(values)
      const prediction: Diffusion.Prediction = { _tag: "Refinement", logits: expected }
      const expectedRefined = yield* ownTensor(baseline.evaluate(left, canvas, prediction))
      const actualRefined = yield* ownTensor(fused.evaluate(right, canvas, prediction))
      expect(yield* Tensor.toNumberArray(actualRefined)).toEqual(yield* Tensor.toNumberArray(expectedRefined))
      const selected = yield* ownTensor(fused.score(right, canvas, [1, 0], [3, 1, 1]))
      const expectedSelected = yield* ownTensor(baseline.score(left, canvas, [1, 0], [3, 1, 1]))
      expect(yield* Tensor.toNumberArray(selected)).toEqual(yield* Tensor.toNumberArray(expectedSelected))
      yield* ownPrefix(fused, fused.commit(right, ids([1, 3])))
      expect(right.tokenCount).toBe(3)
      expect(yield* Tensor.toNumberArray(actual)).toEqual(values)
    })))

  for (const fuseFullReadout of [false, true]) {
    it.effect(`reclaims interrupted read outputs with fused readout ${fuseFullReadout}`, () =>
      Effect.scoped(Effect.gen(function*() {
        const { definition, parameters } = yield* fixture
        const program = yield* Diffusion.compile(definition, parameters, {
          ...config,
          selectedReadouts: [],
          fuseFullReadout
        })
        const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
        const runtime = yield* Runtime.Runtime
        const completed = yield* Deferred.make<void>()
        const resume = yield* Deferred.make<void>()
        let logits: Tensor.Concrete | undefined
        let input: Tensor.Concrete | undefined
        const released = new Set<Tensor.Concrete>()
        const tracked: Runtime.RuntimeService = {
          ...runtime,
          execute: (executable, invocation) =>
            runtime.execute(executable, invocation).pipe(
              Effect.tap((outputs) =>
                Effect.sync(() => {
                  if (invocation.state?.access === "ReadOnly") {
                    logits = outputs[0]
                    input = invocation.bindings[0]
                  }
                })
              )
            ),
          release: (tensor) =>
            Effect.gen(function*() {
              if (tensor === input) {
                yield* Deferred.succeed(completed, undefined)
                yield* Deferred.await(resume)
              }
              yield* runtime.release(tensor)
              released.add(tensor)
            })
        }
        const fiber = yield* program.evaluate(prefix, ids([0, 3]), initial).pipe(
          Effect.provideService(Runtime.Runtime, tracked),
          Effect.forkChild
        )
        yield* Effect.raceFirst(Deferred.await(completed), Fiber.join(fiber))
        const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
        yield* Deferred.succeed(resume, undefined)
        yield* Fiber.join(interruption)
        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        expect(logits).toBeDefined()
        expect(released.has(logits!)).toBe(true)
        const next = yield* ownTensor(program.evaluate(prefix, ids([0, 3]), initial))
        expect((yield* Tensor.toNumberArray(next)).length).toBe(8)
      })))
  }

  it.effect("reuses shape programs for concurrent independent reads and retained outputs", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters, reference } = yield* fixture
      const runtime = yield* Runtime.Runtime
      const traces: Array<Runtime.CompileRequest> = []

      const tracked: Runtime.RuntimeService = {
        ...runtime,
        compile: (request) => {
          traces.push(request)

          return runtime.compile(request)
        }
      }

      const program = yield* Diffusion.compile(definition, parameters, config).pipe(
        Effect.provideService(Runtime.Runtime, tracked)
      )

      expect(traces.filter((request) => request.state?.access === "Append")).toHaveLength(3)
      expect(traces.filter((request) => request.state?.access === "ReadOnly")).toHaveLength(4)
      const compiled = traces.length
      const prefix = yield* ownPrefix(program, program.encode(ids([0, 1, 2])))
      const canvases = [ids([1, 3]), ids([2, 0]), ids([3, 1]), ids([0, 2])]

      const outputs = yield* Effect.all(
        canvases.map((canvas) => ownTensor(program.evaluate(prefix, canvas, initial))),
        { concurrency: "unbounded" }
      )

      for (let index = 0; index < canvases.length; index++) {
        const expected = yield* ownTensor(reference(ids([0, 1, 2]), canvases[index]!))
        deep(yield* Tensor.toNumberArray(outputs[index]!), yield* Tensor.toNumberArray(expected))
      }

      const before = yield* Tensor.toNumberArray(outputs[0]!)
      const another = yield* ownTensor(program.evaluate(prefix, ids([0, 0]), initial))
      yield* Tensor.clear(another)
      deep(yield* Tensor.toNumberArray(outputs[0]!), before)
      expect(prefix.tokenCount).toBe(3)
      expect(prefix.bytes).toBeGreaterThan(0)
      expect(traces).toHaveLength(compiled)
    })))

  it.effect("forks encoder commits without changing the old prefix and preserves absolute positions", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters, reference } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([0, 1, 2])))
      const before = yield* ownTensor(program.evaluate(prefix, ids([3, 0]), initial))
      const extended = yield* ownPrefix(program, program.commit(prefix, ids([1, 3, 2])))
      const branch = yield* ownPrefix(program, program.commit(prefix, ids([2])))
      expect([prefix.tokenCount, extended.tokenCount, branch.tokenCount]).toEqual([3, 6, 4])

      for (const [retained, prompt] of [[extended, ids([0, 1, 2, 1, 3, 2])], [branch, ids([0, 1, 2, 2])]] as const) {
        const actual = yield* ownTensor(program.evaluate(retained, ids([3, 0]), initial))
        const expected = yield* ownTensor(reference(prompt, ids([3, 0])))
        deep(yield* Tensor.toNumberArray(actual), yield* Tensor.toNumberArray(expected))
      }

      const after = yield* ownTensor(program.evaluate(prefix, ids([3, 0]), initial))
      deep(yield* Tensor.toNumberArray(after), yield* Tensor.toNumberArray(before))
    })))

  it.effect("selected logits preserve repeated row and label ordering without retracing denoisers", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([0, 2])))
      const full = yield* ownTensor(program.evaluate(prefix, ids([1, 3, 0]), initial))
      const values = yield* Tensor.toNumberArray(full)

      for (const [rows, labels] of [[[2, 0], [3, 1, 3]], [[1, 1], [2, 0, 1]]] as const) {
        const selected = yield* ownTensor(program.score(prefix, ids([1, 3, 0]), rows, labels))
        expect(selected.shape).toEqual([1, 2, 3])
        deep(
          yield* Tensor.toNumberArray(selected),
          rows.flatMap((row) => labels.map((label) => values[row * 4 + label]!))
        )
      }
    })))

  it.effect("carries full prediction feedback across refinement while independent reads reset", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters, reference } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 0])))
      const canvas = ids([2, 3])
      const first = yield* ownTensor(program.evaluate(prefix, canvas, initial))
      const second = yield* ownTensor(program.evaluate(prefix, canvas, { _tag: "Refinement", logits: first }))
      const third = yield* ownTensor(program.evaluate(prefix, canvas, { _tag: "Refinement", logits: second }))
      const expectedSecond = yield* ownTensor(reference(ids([1, 0]), canvas, { _tag: "Refinement", logits: first }))
      const expectedThird = yield* ownTensor(reference(ids([1, 0]), canvas, { _tag: "Refinement", logits: second }))
      deep(yield* Tensor.toNumberArray(second), yield* Tensor.toNumberArray(expectedSecond))
      deep(yield* Tensor.toNumberArray(third), yield* Tensor.toNumberArray(expectedThird))
      expect(yield* Tensor.toNumberArray(second)).not.toEqual(yield* Tensor.toNumberArray(first))
      const independent = yield* ownTensor(program.evaluate(prefix, canvas, initial))
      deep(yield* Tensor.toNumberArray(independent), yield* Tensor.toNumberArray(first))
    })))

  it.effect("generates multiple refined blocks, commits only continuing blocks, and releases request state", () =>
    Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const runtime = yield* Runtime.Runtime
      const pages: Array<Array<number>> = []
      const positions: Array<number> = []
      const outputs: Array<Tensor.Concrete> = []
      let reads = 0
      let appends = 0
      let snapshots = 0
      let releasedSnapshots = 0

      const tracked: Runtime.RuntimeService = {
        ...runtime,
        execute: (executable, invocation) =>
          runtime.execute(executable, invocation).pipe(Effect.tap((values) =>
            Effect.sync(() => {
              outputs.push(...values)

              if (invocation.state?.access === "ReadOnly") reads++

              if (invocation.state?.access === "Append") appends++
            })
          )),
        extensions: {
          ...runtime.extensions,
          decode: {
            ...runtime.extensions.decode,
            snapshot: (sequence) =>
              runtime.extensions.decode.snapshot(sequence).pipe(Effect.tap(() =>
                Effect.sync(() => {
                  snapshots++
                })
              )),
            releasePrefix: (prefix) =>
              runtime.extensions.decode.releasePrefix(prefix).pipe(Effect.tap(() =>
                Effect.sync(() => {
                  releasedSnapshots++
                })
              ))
          }
        }
      }

      const result = yield* program.generate({
        prompt: ids([0]),
        maxNewTokens: 3,
        outputLimit: "exact",
        initialize: (block) =>
          Effect.sync(() => {
            positions.push(block.position)

            return { value: { canvas: ids([3, 3]), feedback: { _tag: "Initial" as const } }, release: Effect.void }
          }),
        process: (logits) =>
          Effect.map(Tensor.compute([logits]), ([feedback]) => ({
            value: { prediction: undefined, feedback: feedback! },
            release: Tensor.clear(feedback!)
          })),
        policy: {
          canvasLength: 2,
          maxSteps: 2,
          start: () => 0,
          refine: ({ state, canvas, block }) =>
            Effect.succeed({
              state: state + 1,
              canvas,
              draft: ids([block.index + 1, 2]),
              done: false
            }),
          finish: (tokens) => ({ tokens, stop: false })
        },
        onPage: (tokens) =>
          Effect.sync(() => {
            pages.push(Array.from(tokens))
          })
      }).pipe(Effect.provideService(Runtime.Runtime, tracked))

      expect(result).toEqual({ generatedTokens: 3, blocks: 2, refinements: 4, stop: "length" })
      expect(pages).toEqual([[1, 2], [2]])
      expect(positions).toEqual([1, 3])
      expect(reads).toBe(4)
      expect(appends).toBe(2)
      expect(snapshots).toBe(2)
      expect(releasedSnapshots).toBe(snapshots)

      for (const output of outputs) {
        expect((yield* Effect.flip(runtime.readback(output))).reason).toBe("invalid-handle")
      }
    }))

  it.effect("uses the declared feedback dtype independently of F32 readout logits", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters, reference } = yield* fixture

      const program = yield* Diffusion.compile(
        {
          ...definition,
          predictionDtype: "bf16",
          denoise: (parameters, tokens, positions, prediction) =>
            Effect.gen(function*() {
              const feedback: Diffusion.Prediction = prediction._tag === "Initial"
                ? prediction
                : { _tag: "Refinement", logits: yield* Tensor.cast(prediction.logits, "f32") }

              return yield* definition.denoise(parameters, tokens, positions, feedback)
            })
        },
        parameters,
        config
      )

      const prefix = yield* ownPrefix(program, program.encode(ids([1, 0])))
      const canvas = ids([2, 3])
      const first = yield* ownTensor(program.evaluate(prefix, canvas, initial))
      expect(first.dtype).toBe("f32")
      expect(program.predictionDtype).toBe("bf16")
      expect((yield* Effect.flip(program.evaluate(prefix, canvas, { _tag: "Refinement", logits: first })))._tag).toBe(
        "DiffusionInferenceError"
      )

      const feedback = yield* ownTensor(
        Effect.map(Tensor.compute([yield* Tensor.cast(first, "bf16")]), ([value]) => value!)
      )

      const refined = yield* ownTensor(program.evaluate(prefix, canvas, { _tag: "Refinement", logits: feedback }))

      const expected = yield* ownTensor(
        reference(ids([1, 0]), canvas, { _tag: "Refinement", logits: yield* Tensor.cast(feedback, "f32") })
      )

      deep(yield* Tensor.toNumberArray(refined), yield* Tensor.toNumberArray(expected))
      expect((yield* Tensor.toNumberArray(feedback)).length).toBe(8)
    })))

  it.effect("rejects foreign prefixes, uncompiled shapes, invalid selections, and released snapshots", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const other = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([1])))
      expect((yield* Effect.flip(other.evaluate(prefix, ids([0, 1]), initial)))._tag).toBe("DiffusionInferenceError")
      expect((yield* Effect.flip(program.evaluate(prefix, ids([0]), initial)))._tag).toBe("DiffusionInferenceError")
      expect((yield* Effect.flip(program.score(prefix, ids([0, 1]), [2], [0])))._tag).toBe("DiffusionInferenceError")
      expect((yield* Effect.flip(program.commit(prefix, ids([4]))))._tag).toBe("DiffusionInferenceError")
      const released = yield* program.encode(ids([0]))
      const retainedOutput = yield* ownTensor(program.evaluate(released, ids([2, 3]), initial))
      yield* program.release(released)
      expect((yield* Tensor.toNumberArray(retainedOutput)).length).toBe(8)
      expect((yield* Effect.flip(program.evaluate(released, ids([0, 1]), initial)))._tag).toBe("TensorError")
    })))

  it.effect("releases hidden outputs when projection fails and leaves its borrowed prefix usable", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
      const runtime = yield* Runtime.Runtime
      let hidden: ReadonlyArray<Tensor.Concrete> = []
      const released = new Set<Tensor.Concrete>()

      const tracked: Runtime.RuntimeService = {
        ...runtime,
        execute: (executable, invocation) => {
          if (hidden.length > 0 && invocation.state === undefined) {
            return Effect.fail(
              new Runtime.BackendError({
                backend: runtime.backend.name,
                operation: "projection",
                phase: "execute",
                reason: "execution-failed",
                message: "projection failure"
              })
            )
          }

          return runtime.execute(executable, invocation).pipe(Effect.tap((outputs) =>
            Effect.sync(() => {
              if (invocation.state?.access === "ReadOnly") hidden = outputs
            })
          ))
        },
        release: (tensor) =>
          runtime.release(tensor).pipe(Effect.tap(() =>
            Effect.sync(() => {
              released.add(tensor)
            })
          ))
      }

      const result = yield* Effect.exit(
        program.evaluate(prefix, ids([0, 3]), initial).pipe(Effect.provideService(Runtime.Runtime, tracked))
      )

      expect(Exit.isFailure(result)).toBe(true)
      expect(hidden).toHaveLength(1)
      expect(released.has(hidden[0]!)).toBe(true)
      const usable = yield* ownTensor(program.evaluate(prefix, ids([0, 3]), initial))
      expect((yield* Tensor.toNumberArray(usable)).length).toBe(8)
    })))

  it.effect("interruption during final hidden release reclaims unreturned logits", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
      const runtime = yield* Runtime.Runtime
      const releasing = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      let hidden: Tensor.Concrete | undefined
      let logits: Tensor.Concrete | undefined
      const released = new Set<Tensor.Concrete>()

      const tracked: Runtime.RuntimeService = {
        ...runtime,
        execute: (executable, invocation) =>
          runtime.execute(executable, invocation).pipe(Effect.tap((outputs) =>
            Effect.sync(() => {
              if (invocation.state?.access === "ReadOnly") hidden = outputs[0]
              else if (hidden !== undefined && invocation.bindings.includes(hidden)) logits = outputs[0]
            })
          )),
        release: (tensor) =>
          Effect.gen(function*() {
            if (tensor === hidden) {
              yield* Deferred.succeed(releasing, undefined)
              yield* Deferred.await(resume)
            }

            yield* runtime.release(tensor)
            released.add(tensor)
          })
      }

      const fiber = yield* program.evaluate(prefix, ids([0, 3]), initial).pipe(
        Effect.provideService(Runtime.Runtime, tracked),
        Effect.forkChild
      )

      yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
      const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
      yield* Deferred.succeed(resume, undefined)
      yield* Fiber.join(interruption)
      expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
      expect(hidden).toBeDefined()
      expect(logits).toBeDefined()
      expect(released.has(hidden!)).toBe(true)
      expect(released.has(logits!)).toBe(true)
      const next = yield* ownTensor(program.evaluate(prefix, ids([0, 3]), initial))
      expect((yield* Tensor.toNumberArray(next)).length).toBe(8)
    })))

  it.effect("interruption during final sequence release reclaims the unreturned snapshot", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, config)
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
      const runtime = yield* Runtime.Runtime
      const releasing = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      let snapshot: Runtime.KvPrefixHandle | undefined
      let released: Runtime.KvPrefixHandle | undefined

      const tracked: Runtime.RuntimeService = {
        ...runtime,
        extensions: {
          ...runtime.extensions,
          decode: {
            ...runtime.extensions.decode,
            snapshot: (sequence) =>
              runtime.extensions.decode.snapshot(sequence).pipe(Effect.tap((value) =>
                Effect.sync(() => {
                  snapshot = value
                })
              )),
            releaseSequence: (sequence) =>
              Effect.gen(function*() {
                yield* Deferred.succeed(releasing, undefined)
                yield* Deferred.await(resume)
                yield* runtime.extensions.decode.releaseSequence(sequence)
              }),
            releasePrefix: (value) =>
              runtime.extensions.decode.releasePrefix(value).pipe(Effect.tap(() =>
                Effect.sync(() => {
                  released = value
                })
              ))
          }
        }
      }

      const fiber = yield* program.commit(prefix, ids([0, 3])).pipe(
        Effect.provideService(Runtime.Runtime, tracked),
        Effect.forkChild
      )

      yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
      const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
      yield* Deferred.succeed(resume, undefined)
      yield* Fiber.join(interruption)
      expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
      expect(snapshot).toBeDefined()
      expect(released).toBe(snapshot)
      const next = yield* ownTensor(program.evaluate(prefix, ids([0, 3]), initial))
      expect((yield* Tensor.toNumberArray(next)).length).toBe(8)
      expect((yield* program.inspect(prefix)).cursor).toBe(2)
    })))

  it.effect("interruption after the final cached position compute reclaims the unpublished prefix and every position", () =>
    Effect.scoped(Effect.gen(function*() {
      const { definition, parameters } = yield* fixture
      const program = yield* Diffusion.compile(definition, parameters, { ...config, cachePositions: true })
      const prefix = yield* ownPrefix(program, program.encode(ids([1, 2])))
      const runtime = yield* Runtime.Runtime
      const releasing = yield* Deferred.make<void>()
      const resume = yield* Deferred.make<void>()
      let snapshot: Runtime.KvPrefixHandle | undefined
      let released: Runtime.KvPrefixHandle | undefined
      const cached: Array<Tensor.Concrete> = []

      const tracked: Runtime.RuntimeService = {
        ...runtime,
        execute: (executable, invocation) =>
          runtime.execute(executable, invocation).pipe(Effect.tap((outputs) =>
            Effect.sync(() => {
              if (snapshot !== undefined && invocation.state === undefined) cached.push(...outputs)
            })
          )),
        extensions: {
          ...runtime.extensions,
          decode: {
            ...runtime.extensions.decode,
            snapshot: (sequence) =>
              runtime.extensions.decode.snapshot(sequence).pipe(Effect.tap((value) =>
                Effect.sync(() => {
                  snapshot = value
                })
              )),
            releaseSequence: (sequence) =>
              Effect.gen(function*() {
                yield* Deferred.succeed(releasing, undefined)
                yield* Deferred.await(resume)
                yield* runtime.extensions.decode.releaseSequence(sequence)
              }),
            releasePrefix: (value) =>
              runtime.extensions.decode.releasePrefix(value).pipe(Effect.tap(() =>
                Effect.sync(() => {
                  released = value
                })
              ))
          }
        }
      }

      const fiber = yield* program.commit(prefix, ids([0, 3])).pipe(
        Effect.provideService(Runtime.Runtime, tracked),
        Effect.forkChild
      )

      yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
      const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
      yield* Deferred.succeed(resume, undefined)
      yield* Fiber.join(interruption)
      expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
      expect(snapshot).toBeDefined()
      expect(released).toBe(snapshot)
      expect(cached).toHaveLength(config.canvasLengths!.length)
      for (const positions of cached) {
        expect((yield* Effect.flip(runtime.readback(positions))).reason).toBe("invalid-handle")
      }
      const next = yield* ownTensor(program.evaluate(prefix, ids([0, 3]), initial))
      expect((yield* Tensor.toNumberArray(next)).length).toBe(8)
      expect((yield* program.inspect(prefix)).cursor).toBe(2)
    })))

  for (const operation of ["encode", "commit"] as const) {
    it.effect(`interrupts ${operation} acquisitions and releases the temporary append sequence`, () =>
      Effect.scoped(Effect.gen(function*() {
        const { definition, parameters } = yield* fixture
        const program = yield* Diffusion.compile(definition, parameters, { ...config, prefillChunks: [1] })
        const prefix = yield* ownPrefix(program, program.encode(ids([0, 1, 2])))
        const before = yield* ownTensor(program.evaluate(prefix, ids([1, 3]), initial))
        const runtime = yield* Runtime.Runtime
        const entered = yield* Deferred.make<void>()
        let appends = 0
        let releases = 0

        const tracked: Runtime.RuntimeService = {
          ...runtime,
          execute: (executable, invocation) => {
            if (invocation.state?.access === "Append" && ++appends === 2) {
              return Deferred.succeed(entered, undefined).pipe(Effect.andThen(Effect.never))
            }

            return runtime.execute(executable, invocation)
          },
          extensions: {
            ...runtime.extensions,
            decode: {
              ...runtime.extensions.decode,
              releaseSequence: (sequence) =>
                runtime.extensions.decode.releaseSequence(sequence).pipe(Effect.tap(() =>
                  Effect.sync(() => {
                    releases++
                  })
                ))
            }
          }
        }

        const task = operation === "encode" ? program.encode(ids([3, 2, 1])) : program.commit(prefix, ids([3, 2, 1]))
        const fiber = yield* task.pipe(Effect.provideService(Runtime.Runtime, tracked), Effect.forkChild)
        yield* Deferred.await(entered)
        yield* Fiber.interrupt(fiber)
        expect(releases).toBe(1)
        expect(prefix.tokenCount).toBe(3)
        const after = yield* ownTensor(program.evaluate(prefix, ids([1, 3]), initial))
        deep(yield* Tensor.toNumberArray(after), yield* Tensor.toNumberArray(before))
        const completed = yield* ownPrefix(program, program.commit(prefix, ids([3, 2, 1])))
        expect(completed.tokenCount).toBe(6)
      })))
  }
})

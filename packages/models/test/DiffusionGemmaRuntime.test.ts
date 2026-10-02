import { Diffusion, Runtime, Tensor } from "@effect-torch/core"
import * as DG from "@effect-torch/models/DiffusionGemma"
import { expect } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber, Scope } from "effect"
import { onDevices } from "./utils/devices.ts"

type Dtype = "f32" | "bf16"

type Parameters = Pick<DG.LoadedParameters, "config" | "parameterSpecs" | "aliases" | "ownedParameters">

const prompt = () => Uint32Array.of(0, 1, 2, 3, 4, 5, 6)

const canvas = () => Uint32Array.of(2, 4, 1)

const parameters = (dtype: Dtype) =>
  Effect.gen(function*() {
    const catalog = yield* DG.parameterCatalog({
      model_type: "diffusion_gemma",
      dtype: dtype === "bf16" ? "bfloat16" : "float32",
      canvas_length: 3,
      text_config: {
        vocab_size: 7,
        hidden_size: 8,
        intermediate_size: 4,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        global_head_dim: 16,
        layer_types: ["sliding_attention", "full_attention"],
        num_experts: 3,
        top_k_experts: 2,
        moe_intermediate_size: 2,
        sliding_window: 4,
        max_position_embeddings: 16
      }
    })

    let seed = 0x6a09e667

    const random = () => {
      seed ^= seed << 13
      seed ^= seed >>> 17
      seed ^= seed << 5

      return ((seed >>> 0) / 0x100000000 - 0.5) * 0.4
    }

    const roots = yield* Effect.forEach(catalog.parameterSpecs, (spec) =>
      Effect.gen(function*() {
        const size = spec.shape.reduce((n, dim) => n * dim, 1)

        // Nonzero independent projections exercise RoPE, both caches, and experts.
        // Keep norm scales at one and encoder/decoder scalars independent.
        const values = Float32Array.from({ length: size }, (_, index) => {
          if (spec.name.endsWith(".layer_scalar")) {
            return spec.name.startsWith("model.decoder") ? 1.125 : 0.875
          }

          if (spec.name.endsWith(".per_expert_scale")) {
            return 0.75 + index * 0.125
          }

          return spec.initializer._tag === "Constant" ? 1 : random()
        })

        const tensor = yield* Tensor.fromTypedArray(values, spec.shape)

        return dtype === "bf16" ? yield* Tensor.cast(tensor, dtype) : tensor
      }))

    const ownedParameters = yield* Tensor.compute(roots).pipe(Effect.flatMap(Tensor.clearAllScoped))
    const tensors = Object.fromEntries(catalog.parameterSpecs.map((spec, i) => [spec.name, ownedParameters[i]]))

    for (const [alias, canonical] of Object.entries(catalog.aliases)) tensors[alias] = tensors[canonical]

    return { ...catalog, ownedParameters }
  })

// Track only handles acquired after model parameters, including the artifact's
// temporary parameter generation and every invocation-owned input/output.
const observe = (runtime: Runtime.RuntimeService) => {
  const outputs: Array<Tensor.Concrete> = []
  const hidden: Array<Tensor.Concrete> = []
  const released = new Set<Tensor.Concrete>()
  const prefixes: Array<Runtime.KvPrefixHandle> = []
  const releasedPrefixes = new Set<Runtime.KvPrefixHandle>()
  const sequences: Array<Runtime.KvSequenceHandle> = []
  const releasedSequences = new Set<Runtime.KvSequenceHandle>()

  const appends: Array<{
    validLengths: ReadonlyArray<number>
    transactionBytes: number
  }> = []

  const hook = {
    beforeExecute: (_invocation: Runtime.ExecutionInvocation): Effect.Effect<void, Runtime.BackendError> => Effect.void,
    beforeRelease: (_tensor: Tensor.Concrete): Effect.Effect<void> => Effect.void,
    beforeReleaseSequence: (_sequence: Runtime.KvSequenceHandle): Effect.Effect<void> => Effect.void
  }

  let calls = 0

  const service: Runtime.RuntimeService = {
    ...runtime,
    execute: (program, invocation) =>
      Effect.gen(function*() {
        calls++
        yield* hook.beforeExecute(invocation)

        return yield* runtime.execute(program, invocation).pipe(Effect.onExit((exit) =>
          Effect.sync(() => {
            if (Exit.isSuccess(exit)) {
              outputs.push(...exit.value)

              if (invocation.state?.access === "ReadOnly") hidden.push(...exit.value)

              if (invocation.state?.access === "Append") {
                appends.push({
                  validLengths: [...invocation.state.validLengths],
                  transactionBytes: program.diagnostics.memory.transactionBytes
                })
              }
            }
          })
        ))
      }),
    release: (tensor) =>
      Effect.gen(function*() {
        yield* hook.beforeRelease(tensor)
        yield* runtime.release(tensor)
        released.add(tensor)
      }),
    extensions: {
      ...runtime.extensions,
      decode: {
        ...runtime.extensions.decode,
        makeSequence: (pool) =>
          runtime.extensions.decode.makeSequence(pool).pipe(Effect.tap((value) =>
            Effect.sync(() => {
              sequences.push(value)
            })
          )),
        fork: (prefix) =>
          runtime.extensions.decode.fork(prefix).pipe(Effect.tap((value) =>
            Effect.sync(() => {
              sequences.push(value)
            })
          )),
        releaseSequence: (sequence) =>
          Effect.gen(function*() {
            yield* hook.beforeReleaseSequence(sequence)
            yield* runtime.extensions.decode.releaseSequence(sequence)
            releasedSequences.add(sequence)
          }),
        snapshot: (sequence) =>
          runtime.extensions.decode.snapshot(sequence).pipe(Effect.tap((value) =>
            Effect.sync(() => {
              prefixes.push(value)
            })
          )),
        releasePrefix: (prefix) =>
          runtime.extensions.decode.releasePrefix(prefix).pipe(Effect.tap(() =>
            Effect.sync(() => {
              releasedPrefixes.add(prefix)
            })
          ))
      }
    }
  }

  return {
    service,
    outputs,
    hidden,
    released,
    prefixes,
    releasedPrefixes,
    sequences,
    releasedSequences,
    appends,
    hook,
    get calls() {
      return calls
    }
  }
}

type Observed = ReturnType<typeof observe>

const invalid = (runtime: Runtime.RuntimeService, values: Iterable<Tensor.Concrete>) =>
  Effect.gen(function*() {
    for (const value of new Set(values)) {
      expect((yield* Effect.flip(runtime.readback(value))).reason).toBe("invalid-handle")
    }
  })

const released = (tracker: Observed, values: Iterable<Tensor.Concrete>) =>
  Effect.gen(function*() {
    for (const value of new Set(values)) expect(tracker.released.has(value)).toBe(true)

    yield* invalid(tracker.service, values)
  })

const live = (runtime: Runtime.RuntimeService, values: Iterable<Tensor.Concrete>) =>
  Effect.gen(function*() {
    for (const value of new Set(values)) expect((yield* runtime.readback(value)).byteLength).toBeGreaterThan(0)
  })

const finiteLogits = (values: ReadonlyArray<number>, size = 7) => {
  expect(values).toHaveLength(size)
  expect(values.every(Number.isFinite)).toBe(true)
  expect(values.some((value) => value !== 0)).toBe(true)
}

const withParameters = <A, E, R>(
  dtype: Dtype,
  use: (loaded: Parameters, tracker: Observed) => Effect.Effect<A, E, R>
) =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const tracker = observe(runtime)
    const owners: Array<Tensor.Concrete> = []

    const result = yield* Effect.scoped(Effect.gen(function*() {
      const loaded = yield* parameters(dtype)
      owners.push(...loaded.ownedParameters)
      const result = yield* use(loaded, tracker).pipe(Effect.provideService(Runtime.Runtime, tracker.service))

      for (const owner of owners) {
        expect(tracker.outputs.includes(owner)).toBe(false)
        expect(tracker.released.has(owner)).toBe(false)
      }

      yield* live(runtime, owners)

      return result
    }))

    yield* released(tracker, tracker.outputs)

    for (const prefix of tracker.prefixes) expect(tracker.releasedPrefixes.has(prefix)).toBe(true)

    for (const sequence of tracker.sequences) expect(tracker.releasedSequences.has(sequence)).toBe(true)

    yield* invalid(runtime, owners)

    return result
  })

const compile = (loaded: Parameters, optimize = false) =>
  Diffusion.compile(DG.define(loaded), loaded.ownedParameters, {
    maxTokens: 64,
    blockSize: 1,
    prefillChunks: [1, 4],
    canvasLengths: [1, 2, 3],
    selectedReadouts: [{ rows: 1, labels: 4 }],
    compile: { optimize }
  })

const ownPrefix = (
  program: Diffusion.Artifact,
  acquire: ReturnType<Diffusion.Artifact["encode"]>
) => Effect.acquireRelease(acquire, (prefix) => Effect.orDie(program.release(prefix)), { interruptible: true })

const read = (program: Diffusion.Artifact, prefix: Diffusion.Prefix, ids = canvas(), slot = 1) =>
  Effect.scoped(Effect.gen(function*() {
    const output = yield* Effect.acquireRelease(program.evaluate(prefix, ids, { _tag: "Initial" }), Tensor.clear, {
      interruptible: true
    })

    return (yield* Tensor.toNumberArray(output)).slice(slot * 7, (slot + 1) * 7)
  }))

const injected = () =>
  new Runtime.BackendError({
    reason: "execution-failed",
    backend: "DiffusionGemmaRuntime.test",
    operation: "execute",
    phase: "execute",
    message: "injected execution failure"
  })

onDevices("DiffusionGemma compiled runtime", (device) => (it) => {
  for (const dtype of ["f32", "bf16"] as const) {
    for (const optimize of [false, true]) {
      it.effect(
        dtype + " optimize=" + optimize + ": concurrent reads preserve immutable heterogeneous state",
        () =>
          withParameters(dtype, (loaded, tracker) =>
            Effect.scoped(Effect.gen(function*() {
              const program = yield* compile(loaded, optimize)
              const ids = prompt()
              const prefix = yield* ownPrefix(program, program.encode(ids))
              const before = yield* program.inspect(prefix)
              expect(before.cursor).toBe(7)
              expect(prefix.bytes).toBe(before.retainedBytes)
              // Physical pages may also contain logically evicted local rows.
              const elementBytes = dtype === "bf16" ? 2 : 4
              expect(prefix.bytes).toBeGreaterThanOrEqual(2 * (3 * 8 + 7 * 16) * elementBytes)

              if (device === "cuda") {
                // CUDA retains complete transaction segments, deduplicated by
                // backing allocation. Both appends have compiled width 4; the
                // second uses only 3 rows. Each layer has K/V buffers aligned to
                // CUDA_STORAGE_ALIGNMENT (workspace.rs), including trailing capacity.
                const alignment = 256

                const appendBytes = 2 *
                  [8, 16].reduce(
                    (bytes, width) => bytes + Math.ceil(4 * width * elementBytes / alignment) * alignment,
                    0
                  )

                expect(tracker.appends).toEqual([
                  { validLengths: [4], transactionBytes: appendBytes },
                  { validLengths: [3], transactionBytes: appendBytes }
                ])
                // The global layer keeps both backing allocations alive.
                expect(prefix.bytes).toBe(tracker.appends.reduce((bytes, append) => bytes + append.transactionBytes, 0))
              } else {
                expect(prefix.bytes).toBeLessThanOrEqual(2 * 7 * (8 + 16) * elementBytes)
              }

              expect(
                before.layers.map(({ startPosition, kvHeads, headDim, dtype }) => ({
                  startPosition,
                  kvHeads,
                  headDim,
                  dtype
                }))
              ).toEqual([
                { startPosition: 4, kvHeads: 1, headDim: 8, dtype },
                { startPosition: 0, kvHeads: 1, headDim: 16, dtype }
              ])
              const firstRead = tracker.outputs.length
              const input = canvas()
              const baseline = yield* read(program, prefix, input)
              finiteLogits(baseline)

              const concurrent = yield* Effect.all(Array.from({ length: 4 }, () => read(program, prefix, input)), {
                concurrency: 4
              })

              for (const logits of concurrent) expect(logits).toEqual(baseline)

              const labels = [6, 2, 2, 0]

              const selected = yield* Effect.scoped(
                Effect.acquireRelease(program.score(prefix, input, [1], labels), Tensor.clear, { interruptible: true })
                  .pipe(Effect.flatMap(Tensor.toNumberArray))
              )

              finiteLogits(selected, labels.length)
              selected.forEach((value, i) => {
                const expected = baseline[labels[i]!]!

                const tolerance = dtype === "f32"
                  ? 2e-6 + 2e-5 * Math.abs(expected)
                  : Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(expected))) - 7))

                expect(Math.abs(value - expected)).toBeLessThanOrEqual(tolerance)
              })
              const noisy = Uint32Array.of(6, 4, 1)
              const changed = yield* read(program, prefix, noisy)
              expect(changed.some((value, i) => Math.abs(value - baseline[i]!) > 1e-5)).toBe(true)
              expect(yield* read(program, prefix, noisy)).toEqual(changed)
              expect(yield* read(program, prefix, input)).toEqual(baseline)
              expect(input).toEqual(canvas())
              expect(ids).toEqual(prompt())
              expect(labels).toEqual([6, 2, 2, 0])
              expect(yield* program.inspect(prefix)).toEqual(before)
              yield* released(tracker, tracker.outputs.slice(firstRead))
            })))
      )
    }
  }

  it.effect("outputs and forked prefixes remain independent after releasing their source", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.scoped(Effect.gen(function*() {
        const program = yield* compile(loaded)
        const prefixScope = yield* Scope.fork(yield* Effect.scope)

        const prefix = yield* ownPrefix(program, Effect.scoped(program.encode(prompt()))).pipe(
          Scope.provide(prefixScope)
        )

        const first = yield* Effect.acquireRelease(
          Effect.scoped(program.evaluate(prefix, canvas(), { _tag: "Initial" })),
          Tensor.clear,
          { interruptible: true }
        )

        const baseline = yield* Tensor.toNumberArray(first)
        const extended = yield* ownPrefix(program, program.commit(prefix, Uint32Array.of(2, 3)))
        expect(extended.tokenCount).toBe(9)

        const second = yield* Effect.acquireRelease(
          program.evaluate(extended, canvas(), { _tag: "Initial" }),
          Tensor.clear,
          { interruptible: true }
        )

        yield* Scope.close(prefixScope, Exit.void)
        expect((yield* Effect.flip(program.inspect(prefix)))._tag).toBe("TensorError")
        expect(yield* Tensor.toNumberArray(first)).toEqual(baseline)
        const next = yield* read(program, extended)
        finiteLogits(next)
        yield* Tensor.clear(second)
        expect(yield* Tensor.toNumberArray(first)).toEqual(baseline)
        yield* live(tracker.service, loaded.ownedParameters)
      }))))

  it.effect("rejects invalid tokens, selections, bucket sizes and positions before invocation", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.scoped(Effect.gen(function*() {
        const program = yield* compile(loaded)
        const compiledCalls = tracker.calls

        for (const ids of [Uint32Array.of(7), Uint32Array.of(0xffffffff), new Uint32Array(17)]) {
          expect((yield* Effect.flip(program.encode(ids)))._tag).toBe("DiffusionInferenceError")
        }

        expect(tracker.calls).toBe(compiledCalls)
        const prefix = yield* ownPrefix(program, program.encode(prompt()))
        const before = tracker.calls

        const attempts = [
          program.evaluate(prefix, new Uint32Array(), { _tag: "Initial" }),
          program.evaluate(prefix, Uint32Array.of(7), { _tag: "Initial" }),
          program.evaluate(prefix, new Uint32Array(4), { _tag: "Initial" }),
          program.score(prefix, canvas(), [1], []),
          program.score(prefix, canvas(), [1], [0, 1, 2, 7]),
          ...[-1, 3, 0.5, NaN, Infinity].map((slot) => program.score(prefix, canvas(), [slot], [0, 1, 2, 3]))
        ]

        for (const attempt of attempts) expect((yield* Effect.flip(attempt))._tag).toBe("DiffusionInferenceError")

        expect(tracker.calls).toBe(before)
        const nearLimit = yield* ownPrefix(program, program.encode(new Uint32Array(15)))
        const atLimit = tracker.calls
        expect((yield* Effect.flip(program.evaluate(nearLimit, Uint32Array.of(1, 2), { _tag: "Initial" })))._tag).toBe(
          "DiffusionInferenceError"
        )
        expect(tracker.calls).toBe(atLimit)
        finiteLogits(yield* read(program, nearLimit, Uint32Array.of(1), 0))
      }))))

  // Failure after an earlier append chunk exercises whole-request disposal;
  // read/projection failures exercise temporary input and hidden-state cleanup.
  for (const phase of ["encode", "commit", "evaluate", "projection"] as const) {
    for (const cancel of [false, true]) {
      it.effect(
        (cancel ? "interrupted " : "failed ") + phase + " releases private state and preserves borrowers",
        () =>
          withParameters("f32", (loaded, tracker) =>
            Effect.scoped(Effect.gen(function*() {
              const program = yield* compile(loaded)
              const prefix = yield* ownPrefix(program, program.encode(prompt()))
              const before = yield* program.inspect(prefix)
              const baseline = yield* read(program, prefix)
              const start = tracker.outputs.length
              const sequenceStart = tracker.sequences.length
              const entered = yield* Deferred.make<void>()
              let appends = 0
              let readFinished = false
              tracker.hook.beforeExecute = (invocation) => {
                const shouldFail = phase === "encode" || phase === "commit"
                  ? invocation.state?.access === "Append" && ++appends === 2
                  : phase === "evaluate"
                  ? invocation.state?.access === "ReadOnly"
                  : readFinished && invocation.state === undefined

                if (invocation.state?.access === "ReadOnly") readFinished = true

                return shouldFail
                  ? cancel
                    ? Deferred.succeed(entered, undefined).pipe(Effect.andThen(Effect.never))
                    : Effect.fail(injected())
                  : Effect.void
              }

              const operation: Effect.Effect<
                Diffusion.Prefix | Tensor.Concrete,
                Diffusion.InferenceError | Tensor.TensorError,
                Runtime.Runtime
              > = phase === "encode"
                ? program.encode(prompt())
                : phase === "commit"
                ? program.commit(prefix, prompt())
                : program.evaluate(prefix, canvas(), { _tag: "Initial" })

              if (cancel) {
                const fiber = yield* operation.pipe(Effect.forkChild)
                yield* Effect.raceFirst(Deferred.await(entered), Fiber.join(fiber))
                yield* Fiber.interrupt(fiber)
                expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
              } else {
                const error = yield* Effect.flip(operation)
                expect(error._tag).toBe("TensorError")
                expect(error.message).toContain("injected execution failure")
              }

              tracker.hook.beforeExecute = () => Effect.void
              yield* released(tracker, tracker.outputs.slice(start))

              for (const sequence of tracker.sequences.slice(sequenceStart)) {
                expect(tracker.releasedSequences.has(sequence)).toBe(true)
              }

              expect((yield* program.inspect(prefix)).layers).toEqual(before.layers)
              expect((yield* program.inspect(prefix)).cursor).toBe(before.cursor)
              expect(yield* read(program, prefix)).toEqual(baseline)
              yield* live(tracker.service, loaded.ownedParameters)
            })))
      )
    }
  }

  for (const phase of ["encode", "evaluate"] as const) {
    it.effect(
      "interruption during final " + phase + " cleanup reclaims the unreturned result",
      () =>
        withParameters("f32", (loaded, tracker) =>
          Effect.scoped(Effect.gen(function*() {
            const program = yield* compile(loaded)
            const prefix = yield* ownPrefix(program, program.encode(prompt()))
            const start = tracker.outputs.length
            const prefixStart = tracker.prefixes.length
            const entered = yield* Deferred.make<void>()
            const resume = yield* Deferred.make<void>()
            const pause = Deferred.succeed(entered, undefined).pipe(Effect.andThen(Deferred.await(resume)))

            if (phase === "encode") tracker.hook.beforeReleaseSequence = () => pause
            else tracker.hook.beforeRelease = (value) => tracker.hidden.includes(value) ? pause : Effect.void

            const operation: Effect.Effect<
              Diffusion.Prefix | Tensor.Concrete,
              Diffusion.InferenceError | Tensor.TensorError,
              Runtime.Runtime
            > = phase === "encode"
              ? program.encode(prompt())
              : program.evaluate(prefix, canvas(), { _tag: "Initial" })

            const fiber = yield* operation.pipe(Effect.forkChild)
            yield* Effect.raceFirst(Deferred.await(entered), Fiber.join(fiber))
            const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
            yield* Deferred.succeed(resume, undefined)
            yield* Fiber.join(interruption)
            expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
            yield* released(tracker, tracker.outputs.slice(start))

            for (const retained of tracker.prefixes.slice(prefixStart)) {
              expect(tracker.releasedPrefixes.has(retained)).toBe(true)
            }

            tracker.hook.beforeRelease = () => Effect.void
            tracker.hook.beforeReleaseSequence = () => Effect.void
            finiteLogits(yield* read(program, prefix))
          })))
    )
  }

  it.effect("encoding inherits an enclosing uninterruptible acquisition", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.gen(function*() {
        const program = yield* compile(loaded)
        const entered = yield* Deferred.make<void>()
        const resume = yield* Deferred.make<void>()
        let paused = false
        let acquired = false
        tracker.hook.beforeExecute = (invocation) => {
          if (invocation.state?.access !== "Append" || paused) return Effect.void

          paused = true

          return Deferred.succeed(entered, undefined).pipe(Effect.andThen(Deferred.await(resume)))
        }

        const fiber = yield* Effect.gen(function*() {
          const prefix = yield* program.encode(prompt())
          acquired = true
          yield* program.release(prefix)
        }).pipe(Effect.uninterruptible, Effect.forkChild)

        yield* Effect.raceFirst(Deferred.await(entered), Fiber.join(fiber))
        const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
        yield* Deferred.succeed(resume, undefined)
        yield* Fiber.join(interruption)
        expect(acquired).toBe(true)
        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
      })))
})

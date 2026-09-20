import { expect } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { Model, Runtime, Tensor } from "../src/index.ts"
import * as DG from "../src/models/DiffusionGemma.ts"
import { onDevices } from "./utils/devices.ts"

type Dtype = "f32" | "bf16"
type Parameters = Pick<DG.LoadedParameters, "config" | "tensors" | "ownedParameters">
const prompt = () => Uint32Array.of(0, 1, 2, 3, 4, 5, 6)
const canvas = () => Uint32Array.of(2, 4, 1)

const parameters = (dtype: Dtype) =>
  Effect.gen(function*() {
    const catalog = yield* DG.parameterCatalog({
      model_type: "diffusion_gemma",
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
    return { config: catalog.config, tensors, ownedParameters }
  })

// Observe only execution-owned handles. Parameters are acquired before this
// service is installed, so they cannot accidentally satisfy a cleanup check.
const observe = (runtime: Runtime.RuntimeService) => {
  const outputs: Array<Tensor.Concrete> = []
  const released = new Set<Tensor.Concrete>()
  const hook = { beforeExecute: (_call: number): Effect.Effect<void, Runtime.BackendError> => Effect.void }
  let calls = 0
  const service: Runtime.RuntimeService = {
    ...runtime,
    execute: (program, invocation) =>
      Effect.gen(function*() {
        yield* hook.beforeExecute(++calls)
        const values = yield* runtime.execute(program, invocation)
        outputs.push(...values)
        return values
      }),
    release: (tensor) =>
      runtime.release(tensor).pipe(Effect.tap(() =>
        Effect.sync(() => {
          released.add(tensor)
        })
      ))
  }
  return {
    service,
    outputs,
    released,
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
    const handles = new Set(values)
    for (const handle of handles) expect(tracker.released.has(handle)).toBe(true)
    yield* invalid(tracker.service, handles)
  })
const live = (runtime: Runtime.RuntimeService, values: Iterable<Tensor.Concrete>) =>
  Effect.gen(function*() {
    for (const value of new Set(values)) expect((yield* runtime.readback(value)).byteLength).toBeGreaterThan(0)
  })
const snapshot = (values: Iterable<Tensor.Concrete>) => Effect.forEach(values, Tensor.toNumberArray)
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
    expect(new Set(tracker.outputs).size).toBe(tracker.outputs.length)
    yield* released(tracker, tracker.outputs)
    yield* invalid(runtime, owners)
    return result
  })

// The caller chooses scoped ownership here; the model itself returns plain handles.
const scopedExecutor = <E = never, R = never>(loaded: Parameters, options: Model.ExecutorOptions<E, R> = {}) => {
  const execution = Model.executor(DG.make(loaded), options)
  return {
    prefill: (ids: Uint32Array) =>
      Effect.acquireRelease(
        execution.prefill(ids),
        Tensor.clearKvPrefix,
        { interruptible: true }
      ),
    read: (prefix: Tensor.KvPrefix, ids: Uint32Array, slot: number, labels?: Uint32Array) =>
      Effect.scoped(
        Effect.acquireRelease(execution.read(prefix, ids, slot, labels), Tensor.clear, { interruptible: true }).pipe(
          Effect.flatMap(Tensor.toNumberArray)
        )
      )
  }
}

const injected = () =>
  new Runtime.BackendError({
    reason: "execution-failed",
    backend: "DiffusionGemmaRuntime.test",
    operation: "execute",
    phase: "execute",
    message: "injected execution failure"
  })

onDevices("DiffusionGemma runtime", () => (it) => {
  for (const dtype of ["f32", "bf16"] as const) {
    for (const optimize of [false, true]) {
      it.effect(
        dtype + " optimize=" + optimize + ": repeatable concurrent reads borrow immutable prefix storage",
        () =>
          withParameters(dtype, (loaded, tracker) =>
            Effect.scoped(Effect.gen(function*() {
              const execution = scopedExecutor(loaded, { optimize })
              const ids = prompt()
              const prefix = yield* execution.prefill(ids)
              expect(prefix.tokenCount).toBe(7)
              // One KV head, two buffers, local tail of three and global history of seven.
              expect(prefix.bytes).toBe(2 * (3 * 8 + 7 * 16) * (dtype === "bf16" ? 2 : 4))
              const cache = tracker.outputs.filter((value) => !tracker.released.has(value))
              expect(cache).toHaveLength(4)
              expect(cache.map((value) => value.shape)).toEqual([[1, 1, 3, 8], [1, 1, 3, 8], [1, 1, 7, 16], [
                1,
                1,
                7,
                16
              ]])
              expect(cache.every((value) => value.dtype === dtype)).toBe(true)
              const before = yield* snapshot(cache)
              const firstRead = tracker.outputs.length
              const input = canvas()
              const baseline = yield* execution.read(prefix, input, 1)
              finiteLogits(baseline)
              yield* released(tracker, tracker.outputs.slice(firstRead))
              expect(yield* execution.read(prefix, input, 1)).toEqual(baseline)
              const concurrent = yield* Effect.all(
                Array.from({ length: 4 }, () => execution.read(prefix, input, 1)),
                { concurrency: 4 }
              )
              for (const logits of concurrent) expect(logits).toEqual(baseline)

              const labels = Uint32Array.of(6, 2, 2, 0)
              const selected = yield* execution.read(prefix, input, 1, labels)
              finiteLogits(selected, labels.length)
              selected.forEach((value, i) => {
                const expected = baseline[labels[i]]
                // Restricting the head can select a different GEMM reduction shape.
                const tolerance = dtype === "f32" ?
                  2e-6 + 2e-5 * Math.abs(expected)
                  : Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(expected))) - 7))
                expect(Math.abs(value - expected)).toBeLessThanOrEqual(tolerance)
              })
              // Change a context token, keeping the answer token and its slot fixed.
              const noisy = Uint32Array.of(6, 4, 1)
              const changed = yield* execution.read(prefix, noisy, 1)
              finiteLogits(changed)
              expect(changed.some((value, i) => Math.abs(value - baseline[i]) > 1e-5)).toBe(true)
              expect(yield* execution.read(prefix, noisy, 1)).toEqual(changed)
              expect(yield* execution.read(prefix, input, 1)).toEqual(baseline)
              expect(input).toEqual(canvas())
              expect(ids).toEqual(prompt())
              expect(labels).toEqual(Uint32Array.of(6, 2, 2, 0))
              expect(yield* snapshot(cache)).toEqual(before)
              for (const handle of cache) expect(tracker.released.has(handle)).toBe(false)
              yield* released(tracker, tracker.outputs.slice(firstRead))
            })))
      )
    }
  }

  it.effect("returns caller-owned tensors that outlive scopes and independent reads", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.gen(function*() {
        const execution = Model.executor(DG.make(loaded))
        const prefix = yield* Effect.scoped(execution.prefill(prompt()))
        const cache = prefix.layers.flatMap(({ keys, values }) => [keys, values])
        const liveOutputs = tracker.outputs.filter((tensor) => !tracker.released.has(tensor))
        expect(liveOutputs).toEqual(cache)
        yield* live(tracker.service, cache)

        const first = yield* Effect.scoped(execution.read(prefix, canvas(), 1))
        const baseline = yield* Tensor.toNumberArray(first)
        finiteLogits(baseline)
        const other = Model.executor(DG.make(loaded))
        const second = yield* other.read({ ...prefix }, canvas(), 1)
        expect(yield* Tensor.toNumberArray(second)).toEqual(baseline)
        yield* Tensor.clear(second)
        expect(yield* Tensor.toNumberArray(first)).toEqual(baseline)

        yield* Tensor.clearKvPrefix(prefix)
        yield* invalid(tracker.service, cache)
        expect(yield* Tensor.toNumberArray(first)).toEqual(baseline)
        yield* Tensor.clear(first)
        yield* released(tracker, tracker.outputs)
        yield* live(tracker.service, loaded.ownedParameters)

        const failure = yield* Effect.flip(execution.read(prefix, canvas(), 1))
        expect(failure._tag).toBe("TensorError")
        yield* released(tracker, tracker.outputs)
        const fresh = yield* execution.prefill(prompt())
        const logits = yield* execution.read(fresh, canvas(), 1)
        expect(yield* Tensor.toNumberArray(logits)).toEqual(baseline)
        yield* Tensor.clear(logits)
        yield* Tensor.clearKvPrefix(fresh)
        yield* released(tracker, tracker.outputs)
      })))

  it.effect("rejects invalid tokens, slots, labels, canvas sizes and positions before execution", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.scoped(Effect.gen(function*() {
        const execution = scopedExecutor(loaded)
        for (const ids of [new Uint32Array(), Uint32Array.of(7), Uint32Array.of(0xffffffff), new Uint32Array(17)]) {
          const error = yield* Effect.flip(execution.prefill(ids))
          expect(error._tag).toBe("ModelError")
          expect(error.op).toBe("prefill")
        }
        expect(tracker.calls).toBe(0)
        const prefix = yield* execution.prefill(prompt())
        const before = tracker.calls
        const attempts = [
          execution.read(prefix, new Uint32Array(), 0),
          execution.read(prefix, Uint32Array.of(7), 0),
          execution.read(prefix, Uint32Array.of(0xffffffff), 0),
          execution.read(prefix, new Uint32Array(4), 1),
          execution.read(prefix, canvas(), 1, new Uint32Array()),
          execution.read(prefix, canvas(), 1, Uint32Array.of(7)),
          execution.read(prefix, canvas(), 1, Uint32Array.of(0xffffffff)),
          ...[-1, 3, 0.5, NaN, Infinity].map((slot) => execution.read(prefix, canvas(), slot))
        ]
        for (const attempt of attempts) {
          const error = yield* Effect.flip(attempt)
          expect(error._tag).toBe("ModelError")
          expect(error.op).toBe("read")
        }
        expect(tracker.calls).toBe(before)
        finiteLogits(yield* execution.read(prefix, canvas(), 1))
        const nearLimit = yield* execution.prefill(new Uint32Array(15))
        const atLimit = tracker.calls
        const error = yield* Effect.flip(execution.read(nearLimit, Uint32Array.of(1, 2), 0))
        expect(error._tag).toBe("ModelError")
        expect(error.message).toContain("position limit")
        expect(tracker.calls).toBe(atLimit)
        finiteLogits(yield* execution.read(nearLimit, Uint32Array.of(1), 0))
      }))))

  for (const failAt of [2, 3]) {
    it.effect(
      "prefill failure at execution " + failAt + " immediately releases partial state",
      () =>
        withParameters("f32", (loaded, tracker) =>
          Effect.gen(function*() {
            const execution = scopedExecutor(loaded)
            tracker.hook.beforeExecute = (call) => call === failAt ? Effect.fail(injected()) : Effect.void
            // The parameter scope stays open. Failure must close the partial prefix
            // itself, including already-published layer K/V on the third invocation.
            const error = yield* Effect.flip(execution.prefill(prompt()))
            expect(error._tag).toBe("TensorError")
            expect(error.message).toContain("injected execution failure")
            expect(tracker.calls).toBe(failAt)
            expect(tracker.outputs.length).toBeGreaterThan(failAt === 3 ? 1 : 0)
            yield* released(tracker, tracker.outputs)
            yield* live(tracker.service, loaded.ownedParameters)
            yield* Effect.scoped(Effect.gen(function*() {
              const fresh = yield* execution.prefill(prompt())
              finiteLogits(yield* execution.read(fresh, canvas(), 1))
            }))
            yield* released(tracker, tracker.outputs)
          }))
    )

    it.effect(
      "read failure at execution " + failAt + " releases only canvas state and permits retry",
      () =>
        withParameters("f32", (loaded, tracker) =>
          Effect.scoped(Effect.gen(function*() {
            const execution = scopedExecutor(loaded)
            const prefix = yield* execution.prefill(prompt())
            const cache = tracker.outputs.filter((value) => !tracker.released.has(value))
            const before = yield* snapshot(cache)
            const baseline = yield* execution.read(prefix, canvas(), 1)
            const start = tracker.outputs.length
            const stopAt = tracker.calls + failAt
            tracker.hook.beforeExecute = (call) => call === stopAt ? Effect.fail(injected()) : Effect.void
            const error = yield* Effect.flip(execution.read(prefix, canvas(), 1))
            expect(error._tag).toBe("TensorError")
            expect(error.message).toContain("injected execution failure")
            expect(tracker.calls).toBe(stopAt)
            expect(tracker.outputs.length).toBeGreaterThan(start)
            yield* released(tracker, tracker.outputs.slice(start))
            for (const handle of cache) expect(tracker.released.has(handle)).toBe(false)
            expect(yield* snapshot(cache)).toEqual(before)
            yield* live(tracker.service, loaded.ownedParameters)
            expect(yield* execution.read(prefix, canvas(), 1)).toEqual(baseline)
            yield* released(tracker, tracker.outputs.slice(start))
          })))
    )
  }

  for (const phase of ["prefill", "read"] as const) {
    it.effect(
      phase + " observer failure releases outputs already returned by execution",
      () =>
        withParameters("f32", (loaded, tracker) =>
          Effect.scoped(Effect.gen(function*() {
            const execution = scopedExecutor(loaded, {
              observeLayer: (event) =>
                event.phase === phase
                  ? Effect.fail(new Tensor.TensorError({ op: "observeLayer", message: "injected observer failure" }))
                  : Effect.void
            })

            if (phase === "prefill") {
              const error = yield* Effect.flip(execution.prefill(prompt()))
              expect(error.message).toContain("injected observer failure")
              yield* released(tracker, tracker.outputs)
            } else {
              const prefix = yield* execution.prefill(prompt())
              const firstReadOutput = tracker.outputs.length
              const error = yield* Effect.flip(execution.read(prefix, canvas(), 1))
              expect(error.message).toContain("injected observer failure")
              yield* released(tracker, tracker.outputs.slice(firstReadOutput))
              yield* live(tracker.service, prefix.layers.flatMap(({ keys, values }) => [keys, values]))
            }
          })))
    )
  }

  it.effect("interruption during prefill's final hidden release also releases the unreturned prefix", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.gen(function*() {
        const releasing = yield* Deferred.make<void>()
        const resumeRelease = yield* Deferred.make<void>()
        let finalHidden: Tensor.Concrete | undefined
        let paused = false
        const service: Runtime.RuntimeService = {
          ...tracker.service,
          release: (tensor) =>
            Effect.gen(function*() {
              if (tensor === finalHidden && !paused) {
                paused = true
                yield* Deferred.succeed(releasing, undefined)
                yield* Deferred.await(resumeRelease)
              }

              yield* tracker.service.release(tensor)
            })
        }
        const execution = Model.executor(DG.make(loaded), {
          observeLayer: ({ layer, hidden }) =>
            Effect.sync(() => {
              if (layer === loaded.config.text_config.num_hidden_layers - 1) {
                finalHidden = hidden
              }
            })
        })
        const fiber = yield* execution.prefill(prompt()).pipe(
          Effect.provideService(Runtime.Runtime, service),
          Effect.forkChild
        )
        yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
        const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
        yield* Deferred.succeed(resumeRelease, undefined)
        yield* Fiber.join(interruption)

        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        yield* released(tracker, tracker.outputs).pipe(Effect.ensuring(Tensor.clearAll(tracker.outputs)))
        yield* live(tracker.service, loaded.ownedParameters)
      })))

  it.effect("interrupted prefill releases hidden and partial K/V while the parameter scope stays open", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.gen(function*() {
        const execution = scopedExecutor(loaded)
        const paused = yield* Deferred.make<void>()
        tracker.hook.beforeExecute = (call) =>
          call === 3
            ? Deferred.succeed(paused, undefined).pipe(Effect.andThen(Effect.never))
            : Effect.void
        const fiber = yield* Effect.forkChild(execution.prefill(prompt()))
        yield* Effect.raceFirst(Deferred.await(paused), Fiber.join(fiber))
        expect(tracker.calls).toBe(3)
        expect(tracker.outputs.length).toBeGreaterThan(1)
        expect(tracker.outputs.some((value) => !tracker.released.has(value))).toBe(true)
        yield* Fiber.interrupt(fiber)
        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        yield* released(tracker, tracker.outputs)
        yield* live(tracker.service, loaded.ownedParameters)
        yield* Effect.scoped(Effect.gen(function*() {
          const fresh = yield* execution.prefill(prompt())
          finiteLogits(yield* execution.read(fresh, canvas(), 1))
        }))
        yield* released(tracker, tracker.outputs)
      })))

  it.effect("prefill respects an enclosing uninterruptible acquisition", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.gen(function*() {
        const reached = yield* Deferred.make<void>()
        const resume = yield* Deferred.make<void>()
        let acquired = false
        const execution = Model.executor(DG.make(loaded), {
          observeLayer: ({ layer }) =>
            layer === 0
              ? Deferred.succeed(reached, undefined).pipe(Effect.andThen(Deferred.await(resume)))
              : Effect.void
        })
        const fiber = yield* Effect.gen(function*() {
          const prefix = yield* execution.prefill(prompt())
          acquired = true
          yield* Tensor.clearKvPrefix(prefix)
        }).pipe(Effect.uninterruptible, Effect.forkChild)

        yield* Effect.raceFirst(Deferred.await(reached), Fiber.join(fiber))
        const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
        yield* Deferred.succeed(resume, undefined)
        yield* Fiber.join(interruption)

        expect(acquired).toBe(true)
        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        yield* released(tracker, tracker.outputs)
        yield* live(tracker.service, loaded.ownedParameters)
      })))

  it.effect("interrupted read cleans its materialized canvas and leaves prefix storage reusable", () =>
    withParameters("f32", (loaded, tracker) =>
      Effect.scoped(Effect.gen(function*() {
        const execution = scopedExecutor(loaded)
        const prefix = yield* execution.prefill(prompt())
        const cache = tracker.outputs.filter((value) => !tracker.released.has(value))
        const before = yield* snapshot(cache)
        const baseline = yield* execution.read(prefix, canvas(), 1)
        const start = tracker.outputs.length
        const stopAt = tracker.calls + 3
        const paused = yield* Deferred.make<void>()
        tracker.hook.beforeExecute = (call) =>
          call === stopAt
            ? Deferred.succeed(paused, undefined).pipe(Effect.andThen(Effect.never))
            : Effect.void
        const fiber = yield* Effect.forkChild(execution.read(prefix, canvas(), 1))
        yield* Effect.raceFirst(Deferred.await(paused), Fiber.join(fiber))
        expect(tracker.calls).toBe(stopAt)
        expect(tracker.outputs.length).toBeGreaterThan(start)
        yield* Fiber.interrupt(fiber)
        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        yield* released(tracker, tracker.outputs.slice(start))
        for (const handle of cache) expect(tracker.released.has(handle)).toBe(false)
        expect(yield* snapshot(cache)).toEqual(before)
        yield* live(tracker.service, loaded.ownedParameters)
        expect(yield* execution.read(prefix, canvas(), 1)).toEqual(baseline)
      }))))
})

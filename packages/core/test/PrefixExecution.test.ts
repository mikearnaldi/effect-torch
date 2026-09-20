import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Model, Runtime, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const definition = (weight: Tensor.Any): Model.PrefixModel => ({
  vocabSize: 4,
  maxTokens: 8,
  maxReadTokens: 2,
  layerCount: 2,
  embed: (ids, phase) =>
    Effect.gen(function*() {
      const hidden = yield* Tensor.embedding(ids, { weight })
      return phase === "read" ? yield* Tensor.neg(hidden) : hidden
    }),
  prefillLayer: (layer, hidden, positions) =>
    Effect.gen(function*() {
      const offsets = yield* Tensor.reshape(yield* Tensor.cast(positions, "f32"), hidden.shape)
      const output = yield* Tensor.add(hidden, offsets)
      let keys = yield* Tensor.reshape(output, [1, 1, hidden.shape[1], 1])
      keys = layer === 0
        ? yield* Tensor.slice(keys, { start: [0, 0, hidden.shape[1] - 1, 0] })
        : yield* Tensor.concat([keys, keys], { dim: 1 })

      return { output, keys, values: yield* Tensor.neg(keys) }
    }),
  readLayer: (_layer, hidden, positions, prefix) =>
    Effect.gen(function*() {
      const offsets = yield* Tensor.reshape(yield* Tensor.cast(positions, "f32"), hidden.shape)
      const context = yield* Tensor.sum(yield* Tensor.sub(prefix.keys, prefix.values))
      return yield* Tensor.add(yield* Tensor.add(hidden, offsets), context)
    }),
  readout: (hidden, labels) =>
    Effect.gen(function*() {
      const selected = labels === undefined ? weight : yield* Tensor.embedding(labels, { weight })
      return yield* Tensor.linearRows(hidden, selected)
    })
})

const weights = Tensor.fromTypedArray(new Float32Array([0, 1, 2, 3]), [4, 1]).pipe(
  Effect.flatMap((tensor) => Tensor.compute([tensor])),
  Effect.flatMap(Tensor.clearAllScoped),
  Effect.map(([weight]) => weight)
)

onDevices("prefix execution", () => (it) => {
  it.effect("uses logical positions with heterogeneous prefixes and independent selected readouts", () =>
    Effect.scoped(Effect.gen(function*() {
      const weight = yield* weights
      const events: Array<string> = []
      const hiddenStates: Array<Tensor.Concrete> = []
      const execution = Model.executor(definition(weight), {
        optimize: false,
        observeLayer: ({ phase, layer, hidden }) =>
          Effect.sync(() => {
            events.push(phase + "." + layer)
            hiddenStates.push(hidden)
          })
      })
      const prefix = yield* Effect.acquireRelease(
        execution.prefill(Uint32Array.of(1, 2, 3)),
        Tensor.clearKvPrefix,
        { interruptible: true }
      )
      expect(prefix.tokenCount).toBe(3)
      expect(prefix.layers.map(({ keys }) => keys.shape)).toEqual([[1, 1, 1, 1], [1, 2, 3, 1]])
      expect(yield* Tensor.toNumberArray(prefix.layers[0].keys)).toEqual([5])
      expect(yield* Tensor.toNumberArray(prefix.layers[1].keys)).toEqual([1, 4, 7, 1, 4, 7])

      const full = yield* Effect.acquireRelease(
        execution.read(prefix, Uint32Array.of(0, 1), 1),
        Tensor.clear,
        { interruptible: true }
      )
      const selected = yield* Effect.acquireRelease(
        execution.read(prefix, Uint32Array.of(0, 1), 1, Uint32Array.of(3, 1, 3)),
        Tensor.clear,
        { interruptible: true }
      )
      expect(events).toEqual(["prefill.0", "prefill.1", "read.0", "read.1", "read.0", "read.1"])

      for (const hidden of hiddenStates) {
        expect((yield* Effect.flip(Tensor.toNumberArray(hidden)))._tag).toBe("TensorError")
      }

      yield* Tensor.clearKvPrefix(prefix)
      expect(yield* Tensor.toNumberArray(full)).toEqual([0, 65, 130, 195])
      yield* Tensor.clear(full)
      expect(yield* Tensor.toNumberArray(selected)).toEqual([195, 65, 195])
      expect(yield* Tensor.toNumberArray(weight)).toEqual([0, 1, 2, 3])
    })))

  for (const failure of ["observer", "readout"] as const) {
    it.effect(
      failure + " errors release read outputs and preserve the borrowed prefix",
      () =>
        Effect.scoped(Effect.gen(function*() {
          const weight = yield* weights
          const model = definition(weight)
          const prefix = yield* Effect.acquireRelease(
            Model.executor(model).prefill(Uint32Array.of(1, 2, 3)),
            Tensor.clearKvPrefix,
            { interruptible: true }
          )
          const runtime = yield* Runtime.Runtime
          const produced: Array<Tensor.Concrete> = []
          const service: Runtime.RuntimeService = {
            ...runtime,
            execute: (program, invocation) =>
              runtime.execute(program, invocation).pipe(
                Effect.tap((outputs) =>
                  Effect.sync(() => {
                    produced.push(...outputs)
                  })
                )
              )
          }
          const execution = Model.executor({
            ...model,
            readout: failure === "readout" ? () => Effect.fail(failure) : model.readout
          }, {
            observeLayer: () => failure === "observer" ? Effect.fail(failure) : Effect.void
          })
          const error = yield* Effect.flip(
            execution.read(prefix, Uint32Array.of(0, 1), 1).pipe(Effect.provideService(Runtime.Runtime, service))
          )
          expect(error).toBe(failure)
          expect(produced.length).toBeGreaterThan(1)

          for (const tensor of produced) {
            expect((yield* Effect.flip(runtime.readback(tensor))).reason).toBe("invalid-handle")
          }

          const output = yield* Effect.acquireRelease(
            Model.executor(model).read(prefix, Uint32Array.of(0, 1), 1),
            Tensor.clear,
            { interruptible: true }
          )
          expect(yield* Tensor.toNumberArray(output)).toEqual([0, 65, 130, 195])
        }))
    )
  }
})

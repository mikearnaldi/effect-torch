import { expect } from "@effect/vitest"
import { Effect, Exit, Scope } from "effect"
import { Diffusion, Tensor } from "../src/index.ts"
import { deep, onDevices } from "./utils/devices.ts"

// This replaces the old per-layer prefix executor's geometry regression with
// actual decode-specialized attention. Failure and observer-cleanup coverage
// now lives in Diffusion.test.ts and DiffusionGemmaRuntime.test.ts.
const definition = (weight: Tensor.Any): Diffusion.Definition => {
  const build = (
    parameters: ReadonlyArray<Tensor.Any>,
    tokens: Tensor.Any,
    positions: Tensor.Any,
    causal: boolean,
    prediction: Diffusion.Prediction
  ) =>
    Effect.gen(function*() {
      const rows = tokens.shape[1]!
      const embedding = yield* Tensor.embedding(tokens, { weight: parameters[0]! })

      let hidden = yield* Tensor.add(
        causal ? embedding : yield* Tensor.neg(embedding),
        yield* Tensor.reshape(yield* Tensor.cast(positions, "f32"), [1, rows, 1])
      )

      if (prediction._tag === "Refinement") {
        hidden = yield* Tensor.add(hidden, yield* Tensor.mean(prediction.logits, { dims: [2], keepdims: true }))
      }

      for (const layer of [0, 1]) {
        const heads = layer + 1
        const keys = yield* Tensor.broadcastTo(yield* Tensor.reshape(hidden, [1, 1, rows, 1]), [1, heads, rows, 1])
        const attention = { causal, layerId: layer, retentionWindow: layer === 0 ? 1 : null }

        const output = yield* Tensor.scaledDotProductAttention(
          yield* Tensor.zerosLike(keys),
          keys,
          yield* Tensor.neg(keys),
          causal ? { ...attention, window: layer === 0 ? 2 : null } : attention
        )

        const merged = yield* Tensor.reshape(yield* Tensor.mean(output, { dims: [1], keepdims: false }), [1, rows, 1])
        hidden = yield* Tensor.add(hidden, merged)
      }

      return hidden
    })

  return {
    parameterSpecs: [{ name: "weight", shape: weight.shape, initializer: { _tag: "Constant", value: 0 } }],
    vocabSize: 4,
    canvasLength: 2,
    maxPositions: 8,
    dtype: "f32",
    predictionDtype: "f32",
    encode: (parameters, tokens, positions) => build(parameters, tokens, positions, true, { _tag: "Initial" }),
    denoise: (parameters, tokens, positions, prediction) => build(parameters, tokens, positions, false, prediction),
    readout: (parameters, hidden, selection) =>
      Effect.gen(function*() {
        const rows = selection._tag === "Full" ? hidden : yield* Tensor.take(hidden, selection.rows, { dim: 1 })

        const weights = selection._tag === "Full"
          ? parameters[0]!
          : yield* Tensor.take(parameters[0]!, selection.labels)

        return yield* Tensor.linearRows(rows, weights)
      })
  }
}

onDevices("compiled prefix geometry", () => (it) => {
  it.effect("retains heterogeneous local/global state while using absolute read positions", () =>
    Effect.scoped(Effect.gen(function*() {
      const [weight] = yield* Effect.acquireRelease(
        Tensor.compute([yield* Tensor.fromTypedArray(new Float32Array([0, 1, 2, 3]), [4, 1])]),
        Tensor.clearAll,
        { interruptible: true }
      )

      const program = yield* Diffusion.compile(definition(weight!), [weight!], {
        maxTokens: 16,
        blockSize: 1,
        prefillChunks: [1],
        selectedReadouts: [{ rows: 1, labels: 3 }],
        compile: { optimize: false }
      })

      const prefixScope = yield* Scope.fork(yield* Effect.scope)

      const prefix = yield* Effect.acquireRelease(
        program.encode(Uint32Array.of(1, 2, 3)),
        (prefix) => Effect.orDie(program.release(prefix)),
        { interruptible: true }
      ).pipe(Scope.provide(prefixScope))

      const before = yield* program.inspect(prefix)
      expect(prefix.tokenCount).toBe(3)
      expect(before.layers.map(({ kvHeads, headDim, startPosition }) => ({ kvHeads, headDim, startPosition }))).toEqual(
        [
          { kvHeads: 1, headDim: 1, startPosition: 2 },
          { kvHeads: 2, headDim: 1, startPosition: 0 }
        ]
      )
      deep(before.layers[0]!.keys, [5])
      deep(before.layers[0]!.values, [-5])
      deep(before.layers[1]!.keys, [0, 0, 1, 1, 1, 1])

      const full = yield* Effect.acquireRelease(
        program.evaluate(prefix, Uint32Array.of(0, 1), { _tag: "Initial" }),
        Tensor.clear,
        { interruptible: true }
      )

      const selected = yield* Effect.acquireRelease(
        program.score(prefix, Uint32Array.of(0, 1), [1], [3, 1, 3]),
        Tensor.clear,
        { interruptible: true }
      )

      deep(yield* Tensor.toNumberArray(full), [0, -0.8, -1.6, -2.4, 0, -0.8, -1.6, -2.4])
      deep(yield* Tensor.toNumberArray(selected), [-2.4, -0.8, -2.4])
      expect(yield* program.inspect(prefix)).toEqual(before)
      yield* Scope.close(prefixScope, Exit.void)
      yield* Tensor.clear(full)
      deep(yield* Tensor.toNumberArray(selected), [-2.4, -0.8, -2.4])
      deep(yield* Tensor.toNumberArray(weight!), [0, 1, 2, 3])
    })))
})

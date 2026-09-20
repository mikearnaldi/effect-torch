import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Tensor } from "../src/index.ts"
import { DiffusionGemma } from "../src/models/index.ts"
import { onDevices } from "./utils/devices.ts"

const config = Effect.runSync(DiffusionGemma.parseConfig({
  model_type: "diffusion_gemma",
  text_config: {
    vocab_size: 7,
    hidden_size: 4,
    intermediate_size: 3,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 8,
    global_head_dim: 16,
    num_experts: 4,
    top_k_experts: 2,
    moe_intermediate_size: 2,
    sliding_window: 4
  }
}))
const prefix = "model.decoder.layers.0.router"

onDevices("DiffusionGemma routing", () => (it) => {
  for (const dtype of ["f32", "bf16"] as const) {
    for (const optimize of [false, true]) {
      it.effect(`${dtype}, optimize=${optimize}: routes tokens on-device and scales after top-k normalization`, () =>
        Effect.scoped(Effect.gen(function*() {
          const graph = (values: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
            Tensor.fromTypedArray(new Float32Array(values), shape).pipe(Effect.flatMap(Tensor.cast(dtype)))
          const [x, scale, weight, expertScale] = yield* Tensor.compute([
            yield* graph([1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0], [1, 3, 4]),
            yield* graph([1, 1, 1, 1], [4]),
            yield* graph([2, 0, 0, 0, 4, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0], [4, 4]),
            yield* graph([2, 3, 0.5, 4], [4])
          ]).pipe(Effect.flatMap(Tensor.clearAllScoped))
          const routing = yield* DiffusionGemma.routeTokens(
            config,
            {
              [`${prefix}.scale`]: scale,
              [`${prefix}.proj.weight`]: weight,
              [`${prefix}.per_expert_scale`]: expertScale
            },
            0,
            x
          )
          expect(Tensor.isLazyTensor(routing.indices)).toBe(true)
          const program = yield* Tensor.freezeProgram([routing.probabilities, routing.weights, routing.indices], {
            optimize
          })
          const [probabilities, weights, indices] = yield* Tensor.runProgram(program, []).pipe(
            Effect.flatMap(Tensor.clearAllScoped)
          )
          expect(probabilities.dtype).toBe("f32")
          expect(weights.dtype).toBe("f32")
          expect(indices.dtype).toBe("u32")
          expect(indices.shape).toEqual([3, 2])
          expect(yield* Tensor.toNumberArray(indices)).toEqual([1, 0, 2, 3, 0, 1])
          const inverse = dtype === "bf16" ? 1 : Math.fround(Math.pow(Math.fround(1 + 1e-6), -0.5))
          const scores = [[inverse, 2 * inverse, -inverse, 0], [-inverse, -2 * inverse, inverse, 0], [0, 0, 0, 0]]
          const expected = scores.flatMap((row) => {
            const total = row.reduce((sum, value) => sum + Math.exp(value), 0)
            return row.map((value) => Math.exp(value) / total)
          })
          const actual = yield* Tensor.toNumberArray(probabilities)
          actual.forEach((value, i) => expect(Math.abs(value - expected[i])).toBeLessThan(2e-6))
          const first = Math.exp(inverse) / (1 + Math.exp(inverse))
          const expectedWeights = [first * 3, (1 - first) * 2, first * 0.5, (1 - first) * 4, 1, 1.5]
          const actualWeights = yield* Tensor.toNumberArray(weights)
          actualWeights.forEach((value, i) => expect(Math.abs(value - expectedWeights[i])).toBeLessThan(2e-6))
          expect(actualWeights[4] + actualWeights[5]).toBe(2.5)
        })))
    }
  }
})

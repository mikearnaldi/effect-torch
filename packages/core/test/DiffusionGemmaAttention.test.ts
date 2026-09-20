import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Model, Tensor } from "../src/index.ts"
import * as DiffusionGemma from "../src/models/DiffusionGemma.ts"
import { onDevices } from "./utils/devices.ts"

const config = Effect.runSync(DiffusionGemma.parseConfig({
  model_type: "diffusion_gemma",
  canvas_length: 6,
  text_config: {
    vocab_size: 7,
    hidden_size: 4,
    intermediate_size: 5,
    num_hidden_layers: 2,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    head_dim: 8,
    per_layer_config: { "1": { head_dim: 16, num_key_value_heads: 1 } },
    num_experts: 3,
    top_k_experts: 2,
    moe_intermediate_size: 2,
    sliding_window: 4
  }
}))
type Dtype = "f32" | "bf16"
const bf16 = (value: number) => {
  const view = new DataView(new ArrayBuffer(4))
  view.setFloat32(0, value, true)
  const bits = view.getUint32(0, true)
  view.setUint32(0, (bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff0000, true)
  return view.getFloat32(0, true)
}
const round = (dtype: Dtype) => dtype === "bf16" ? bf16 : Math.fround
const tensor = (values: ReadonlyArray<number>, shape: ReadonlyArray<number>, dtype: Dtype) =>
  Tensor.fromTypedArray(Float32Array.from(values), shape).pipe(Effect.flatMap(Tensor.cast(dtype)))
const positions = (start: number, sequence: number, batch = 1) =>
  Tensor.fromTypedArray(Uint32Array.from({ length: batch * sequence }, (_, i) => start + i), [batch, sequence])
const close = (actual: ReadonlyArray<number>, expected: ReadonlyArray<number>, dtype: Dtype) => {
  expect(actual).toHaveLength(expected.length)
  actual.forEach((value, i) => {
    expect(Number.isFinite(value)).toBe(true)
    const tolerance = dtype === "bf16"
      ? Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(expected[i]))) - 7))
      : 3e-6
    expect(Math.abs(value - expected[i]), "element " + i + ": " + value + " != " + expected[i]).toBeLessThanOrEqual(
      tolerance
    )
  })
}
const weights = (layer: number, dtype: Dtype, uniform = true) =>
  Effect.gen(function*() {
    const { head_dim: d, num_key_value_heads: kv } = config.text_config.per_layer_config[String(layer)]
    const prefix = "model.decoder.layers." + layer + ".self_attn"
    const basis = (rows: number, grouped: boolean) =>
      Array.from({ length: rows * 4 }, (_, i) =>
        i % 4 === Math.floor(i / 4) % 4 ? (grouped && Math.floor(i / (4 * d)) % 2 === 1 ? -1 : 1) : 0)
    const result = {
      [prefix + ".q_proj.weight"]: yield* tensor(basis(4 * d, false), [4 * d, 4], dtype),
      [prefix + ".k_proj.weight"]: yield* tensor(basis(kv * d, true), [kv * d, 4], dtype),
      [prefix + ".q_norm.weight"]: yield* tensor(Array(d).fill(uniform ? 0 : 1), [d], dtype),
      [prefix + ".k_norm.weight"]: yield* tensor(Array(d).fill(2), [d], dtype),
      [prefix + ".o_proj.weight"]: yield* tensor(
        Array.from({ length: 4 * 4 * d }, (_, i) =>
          i % (4 * d) === Math.floor(i / (4 * d)) * d ? 1 : 0),
        [4, 4 * d],
        dtype
      )
    }
    if (layer === 0) {
      result[prefix + ".v_proj.weight"] = yield* tensor(basis(kv * d, true), [kv * d, 4], dtype)
    }
    return result
  })
const rows = [1, 2, 0, -1, 2, 1, 3, 0, -1, 2, 1, 3, 3, -2, 0, 1, 2, 3, -1, 1, -2, 1, 3, 2]
const normalizedFirst = (data: ReadonlyArray<number>, dtype: Dtype) =>
  Array.from({ length: data.length / 4 }, (_, row) => {
    const values = data.slice(row * 4, row * 4 + 4)
    return round(dtype)(values[0] / Math.sqrt(values.reduce((sum, x) => sum + x * x, 0) / 4 + 1e-6))
  })
const average = (values: ReadonlyArray<number>, dtype: Dtype) => {
  const p = round(dtype)(1 / values.length)
  return round(dtype)(values.reduce((sum, v) => Math.fround(sum + Math.fround(p * v)), 0))
}

onDevices("DiffusionGemma attention", () => (it) => {
  for (const dtype of ["f32", "bf16"] as const) {
    for (const layer of [0, 1]) {
      it.effect(
        dtype + " layer " + layer + ": causal/local mask includes self and retains the correct prefix rows",
        () =>
          Effect.scoped(Effect.gen(function*() {
            const data = [...rows, ...rows.map((x) => -x)]
            const input = yield* tensor(data, [2, 6, 4], dtype)
            const result = yield* DiffusionGemma.prefillAttention(
              config,
              yield* weights(layer, dtype),
              layer,
              input,
              yield* positions(40, 6, 2)
            )
            expect(Tensor.isLazyTensor(result.output)).toBe(true)
            const [output, keys, values] = yield* Tensor.compute([result.output, result.keys, result.values]).pipe(
              Effect.flatMap(Tensor.clearAllScoped)
            )
            const { head_dim: d, num_key_value_heads: kv } = config.text_config.per_layer_config[String(layer)]
            const retained = layer === 0 ? 3 : 6
            expect(keys.shape).toEqual([2, kv, retained, d])
            expect(values.shape).toEqual(keys.shape)
            const normalized = normalizedFirst(data, dtype)
            const expected = Array.from({ length: 12 }, (_, i) => {
              const b = Math.floor(i / 6) * 6
              const first = layer === 0 ? Math.max(b, i - 3) : b
              const value = average(normalized.slice(first, i + 1), dtype)
              return layer === 0 ? [value, value, -value, -value] : Array(4).fill(value)
            }).flat()
            close(yield* Tensor.toNumberArray(output), expected, dtype)
            const v = yield* Tensor.toNumberArray(values)
            for (let b = 0; b < 2; b++) {
              for (let h = 0; h < kv; h++) {
                for (let s = 0; s < retained; s++) {
                  close([v[((b * kv + h) * retained + s) * d]], [
                    normalized[b * 6 + 6 - retained + s] * (h === 0 ? 1 : -1)
                  ], dtype)
                }
              }
            }
          }))
      )

      it.effect(
        dtype + " layer " + layer + ": decoder attends to the entire canvas and repeated reads preserve prefix owners",
        () =>
          Effect.scoped(Effect.gen(function*() {
            const w = yield* weights(layer, dtype)
            const prefill = yield* DiffusionGemma.prefillAttention(
              config,
              w,
              layer,
              yield* tensor(rows, [1, 6, 4], dtype),
              yield* positions(6, 6)
            )
            const [keys, values] = yield* Tensor.compute([prefill.keys, prefill.values]).pipe(
              Effect.flatMap(Tensor.clearAllScoped)
            )
            const beforeKeys = yield* Tensor.toNumberArray(keys)
            const beforeValues = yield* Tensor.toNumberArray(values)
            const prefix = Object.freeze({ keys, values })
            const canvas = [...rows.slice(0, 20), 5, 0, 0, 0]
            const pos = yield* positions(12, 6)
            const a = yield* DiffusionGemma.readAttention(
              config,
              w,
              layer,
              yield* tensor(rows, [1, 6, 4], dtype),
              pos,
              prefix
            )
            const b = yield* DiffusionGemma.readAttention(
              config,
              w,
              layer,
              yield* tensor(canvas, [1, 6, 4], dtype),
              pos,
              prefix
            )
            const [outA, outB, combinedKeys, combinedValues] = yield* Tensor.compute([
              a.output,
              b.output,
              b.keys,
              b.values
            ]).pipe(Effect.flatMap(Tensor.clearAllScoped))
            expect(combinedKeys.shape[2]).toBe(keys.shape[2] + 6)
            expect(combinedValues.shape).toEqual(combinedKeys.shape)
            const stored = normalizedFirst(rows, dtype).slice(layer === 0 ? 3 : 0)
            const expected = average([...stored, ...normalizedFirst(canvas, dtype)], dtype)
            close(
              yield* Tensor.toNumberArray(outB),
              Array.from(
                { length: 6 },
                () => layer === 0 ? [expected, expected, -expected, -expected] : Array(4).fill(expected)
              ).flat(),
              dtype
            )
            expect((yield* Tensor.toNumberArray(outA))[0]).not.toBe((yield* Tensor.toNumberArray(outB))[0])
            expect(yield* Tensor.toNumberArray(keys)).toEqual(beforeKeys)
            expect(yield* Tensor.toNumberArray(values)).toEqual(beforeValues)
          }))
      )
    }

    it.effect(
      dtype + ": global V precedes learned K norm and RoPE, with absolute canvas positions",
      () =>
        Effect.scoped(Effect.gen(function*() {
          const w = yield* weights(1, dtype, false)
          expect(w["model.decoder.layers.1.self_attn.v_proj.weight"]).toBeUndefined()
          const input = yield* tensor(rows.slice(0, 8), [1, 2, 4], dtype)
          const projected = yield* DiffusionGemma.attentionProjections(config, w, 1, input, yield* positions(29, 2))
          const empty = yield* tensor([], [1, 1, 0, 16], dtype)
          const a = yield* DiffusionGemma.readAttention(config, w, 1, input, yield* positions(29, 2), {
            keys: empty,
            values: empty
          })
          const b = yield* DiffusionGemma.readAttention(config, w, 1, input, yield* positions(0, 2), {
            keys: empty,
            values: empty
          })
          const [value, key, readKey, wrongKey] = yield* Tensor.compute([
            projected.values,
            projected.keys,
            a.keys,
            b.keys
          ]).pipe(Effect.flatMap(Tensor.clearAllScoped))
          const expected = rows.slice(0, 8).flatMap((_, i, data) =>
            i % 4 === 0
              ? Array.from({ length: 16 }, (_, j) =>
                round(dtype)(
                  data[i + j % 4] / Math.sqrt(data.slice(i, i + 4).reduce((sum, x) => sum + x * x, 0) / 4 + 1e-6)
                )) :
              []
          )
          close(yield* Tensor.toNumberArray(value), expected, dtype)
          expect(yield* Tensor.toNumberArray(key)).toEqual(yield* Tensor.toNumberArray(readKey))
          expect(yield* Tensor.toNumberArray(readKey)).not.toEqual(yield* Tensor.toNumberArray(wrongKey))
          expect(yield* Tensor.toNumberArray(value)).not.toEqual(yield* Tensor.toNumberArray(key))
        }))
    )
  }

  it.effect("rejects empty rows, missing/malformed/dtype-mismatched weights and invalid prefixes with ModelError", () =>
    Effect.gen(function*() {
      const input = yield* tensor(rows, [1, 6, 4], "f32")
      const pos = yield* positions(12, 6)
      const w = yield* weights(0, "f32")
      const key = "model.decoder.layers.0.self_attn.v_proj.weight"
      const { [key]: omitted, ...withoutV } = w
      expect(omitted).toBeDefined()
      const invalid = [
        DiffusionGemma.prefillAttention(config, withoutV, 0, input, pos),
        DiffusionGemma.prefillAttention(config, { ...w, [key]: yield* tensor([1], [1], "f32") }, 0, input, pos),
        DiffusionGemma.prefillAttention(config, { ...w, [key]: yield* Tensor.cast(w[key], "bf16") }, 0, input, pos),
        DiffusionGemma.prefillAttention(config, w, -1, input, pos),
        DiffusionGemma.prefillAttention(config, w, 0, yield* tensor([], [1, 0, 4], "f32"), pos),
        DiffusionGemma.prefillAttention(config, w, 0, yield* tensor([], [0, 6, 4], "f32"), pos)
      ]
      const full = yield* tensor(Array(64).fill(0), [1, 2, 4, 8], "f32")
      const short = yield* tensor(Array(32).fill(0), [1, 2, 2, 8], "f32")
      invalid.push(DiffusionGemma.readAttention(config, w, 0, input, pos, { keys: full, values: full }))
      invalid.push(DiffusionGemma.readAttention(config, w, 0, input, pos, { keys: short, values: full }))
      invalid.push(
        DiffusionGemma.readAttention(config, w, 0, input, pos, {
          keys: short,
          values: yield* Tensor.cast(short, "bf16")
        })
      )
      for (const effect of invalid) expect(yield* Effect.flip(effect)).toBeInstanceOf(Model.ModelError)
    }))
})

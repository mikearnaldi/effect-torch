import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Runtime, Tensor } from "../src/index.ts"
import { DiffusionGemma } from "../src/models/index.ts"
import { onDevices } from "./utils/devices.ts"

const config = Effect.runSync(DiffusionGemma.parseConfig({
  model_type: "diffusion_gemma",
  canvas_length: 3,
  text_config: {
    vocab_size: 7,
    hidden_size: 6,
    intermediate_size: 5,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 8,
    global_head_dim: 16,
    num_experts: 3,
    top_k_experts: 2,
    moe_intermediate_size: 2,
    sliding_window: 4
  }
}))
const embeddingName = "model.decoder.embed_tokens.weight"
const normName = "model.decoder.norm.weight"
const mlpPrefix = "model.decoder.layers.0.mlp"
const f32 = Math.fround

// Round finite F32 fixtures to BF16, ties to even, independently of the runtime.
const bf16 = (value: number): number => {
  const view = new DataView(new ArrayBuffer(4))
  view.setFloat32(0, value, true)
  const bits = view.getUint32(0, true)
  view.setUint32(0, (bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff0000, true)
  return view.getFloat32(0, true)
}
const tensor = (values: ReadonlyArray<number>, shape: ReadonlyArray<number>, dtype: "f32" | "bf16") =>
  Tensor.fromTypedArray(new Float32Array(values), shape).pipe(Effect.flatMap(Tensor.cast(dtype)))
const close = (actual: ReadonlyArray<number>, expected: ReadonlyArray<number>, tolerance = 3e-6) => {
  expect(actual).toHaveLength(expected.length)
  actual.forEach((value, i) =>
    expect(Math.abs(value - expected[i]), `element ${i}: ${value} != ${expected[i]}`).toBeLessThanOrEqual(tolerance)
  )
}
const rms = (
  input: ReadonlyArray<number>,
  width: number,
  round: (n: number) => number,
  weight?: ReadonlyArray<number>
) =>
  input.map((value, i) => {
    const row = input.slice(Math.floor(i / width) * width, (Math.floor(i / width) + 1) * width)
    const mean = f32(row.reduce((sum, x) => f32(sum + f32(x * x)), 0) / width)
    const inv = f32(Math.pow(f32(mean + f32(1e-6)), -0.5))
    const normalized = f32(value * inv)
    return round(weight === undefined ? normalized : f32(normalized * weight[i % width]))
  })
const project = (
  input: ReadonlyArray<number>,
  weight: ReadonlyArray<number>,
  width: number,
  round: (n: number) => number
) => {
  const outputs: Array<number> = []
  for (let row = 0; row < input.length / width; row++) {
    for (let output = 0; output < weight.length / width; output++) {
      let sum = 0
      for (let k = 0; k < width; k++) sum = f32(sum + f32(input[row * width + k] * weight[output * width + k]))
      outputs.push(round(sum))
    }
  }
  return outputs
}
const gelu = (x: number) => f32(0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))))

onDevices("DiffusionGemma math", () => (it) => {
  for (const dtype of ["f32", "bf16"] as const) {
    const round = dtype === "f32" ? f32 : bf16

    it.effect(`${dtype}: embedding scale is dtype-rounded and zero self-conditioning still post-normalizes`, () =>
      Effect.scoped(Effect.gen(function*() {
        const weights = Array.from({ length: 42 }, (_, i) => round((i % 17 - 8) / 8))
        const [weight] = yield* Tensor.compute([yield* tensor(weights, [7, 6], dtype)]).pipe(
          Effect.flatMap(Tensor.clearAllScoped)
        )
        const ids = yield* Tensor.fromTypedArray(new Uint32Array([6, 2, 0]), [1, 3])
        const runtime = yield* Runtime.Runtime
        let compiles = 0
        const service: Runtime.RuntimeService = {
          ...runtime,
          compile: (request) => Effect.sync(() => compiles++).pipe(Effect.andThen(runtime.compile(request)))
        }
        const embedded = yield* DiffusionGemma.embedTokens(config, { [embeddingName]: weight }, ids).pipe(
          Effect.provideService(Runtime.Runtime, service)
        )
        const conditioned = yield* DiffusionGemma.initialSelfConditioning(config, embedded).pipe(
          Effect.provideService(Runtime.Runtime, service)
        )
        expect(Tensor.isLazyTensor(embedded)).toBe(true)
        expect(Tensor.isLazyTensor(conditioned)).toBe(true)
        expect(compiles).toBe(0)
        const scale = round(f32(Math.sqrt(6)))
        const expected = [6, 2, 0].flatMap((id) =>
          weights.slice(id * 6, id * 6 + 6).map((value) => round(value * scale))
        )
        const [a, b] = yield* Tensor.compute([embedded, conditioned]).pipe(Effect.flatMap(Tensor.clearAllScoped))
        expect(a.dtype).toBe(dtype)
        expect(b.dtype).toBe(dtype)
        close(yield* Tensor.toNumberArray(a), expected, dtype === "bf16" ? 0 : 1e-6)
        close(yield* Tensor.toNumberArray(b), rms(expected, 6, round), dtype === "bf16" ? 0 : 3e-6)
        expect(yield* Tensor.toNumberArray(b)).not.toEqual(expected)
        // The graph borrows the embedding owner, including across repeated reads.
        expect(yield* Tensor.toNumberArray(weight)).toEqual(weights)
      })))

    it.effect(`${dtype}: learned and unlearned RMS retain the F32 epsilon and single output cast`, () =>
      Effect.scoped(Effect.gen(function*() {
        const input = [0, 0, 0, 0, 0, 0, -0.0002, -0.0001, 0.0003, 0.0004, 0.0001, -0.0004, -3, -0.5, 0.25, 1, 2, 4]
          .map(round)
        const weight = [0, 0.5, 0.75, 1, 1.25, 2].map(round)
        const x = yield* tensor(input, [3, 6], dtype)
        const w = yield* tensor(weight, [6], dtype)
        const [learned, unlearned] = yield* Tensor.compute([
          yield* Tensor.rmsNorm(x, w, config.text_config.rms_norm_eps),
          yield* Tensor.rmsNorm(x, undefined, config.text_config.rms_norm_eps)
        ]).pipe(Effect.flatMap(Tensor.clearAllScoped))
        close(yield* Tensor.toNumberArray(learned), rms(input, 6, round, weight), dtype === "bf16" ? 0 : 3e-6)
        close(yield* Tensor.toNumberArray(unlearned), rms(input, 6, round), dtype === "bf16" ? 0 : 3e-6)
        expect((yield* Tensor.toNumberArray(learned))[12]).toBe(-0)
      })))

    it.effect(`${dtype}: local/global RoPE uses explicit batched offsets and full-half pairing`, () =>
      Effect.scoped(Effect.gen(function*() {
        const positions = [0, 19, 1023, 4, 17, 91]
        const ids = yield* Tensor.fromTypedArray(new Uint32Array(positions), [2, 3])
        for (const layer of [0, 1]) {
          const d = layer === 0 ? 8 : 16
          const pairs = layer === 0 ? 4 : 2
          const theta = layer === 0 ? 10000 : 1000000
          const input = Array.from({ length: 2 * 2 * 3 * d }, (_, i) => round(((i * 11) % 53 - 26) / 8))
          const expected = input.map((value, i) => {
            const j = i % d
            const position = positions[Math.floor(i / (6 * d)) * 3 + Math.floor(i / d) % 3]
            const frequency = j % (d / 2) < pairs ? f32(1 / f32(Math.pow(theta, f32(2 * (j % (d / 2)) / d)))) : 0
            const angle = f32(position * frequency)
            const c = round(f32(Math.cos(angle)))
            const s = round(f32(Math.sin(angle)))
            const partner = j < d / 2 ? -input[i + d / 2] : input[i - d / 2]
            return round(round(value * c) + round(partner * s))
          })
          const output = yield* DiffusionGemma.rotaryEmbedding(
            config,
            layer,
            yield* tensor(input, [2, 2, 3, d], dtype),
            ids
          )
          const [actual] = yield* Tensor.compute([output]).pipe(Effect.flatMap(Tensor.clearAllScoped))
          close(yield* Tensor.toNumberArray(actual), expected, dtype === "bf16" ? 0 : 2e-6)
          if (layer === 1) {
            const values = yield* Tensor.toNumberArray(actual)
            input.forEach((value, i) => {
              if (i % 8 >= 2) expect(values[i]).toBe(value)
            })
          }
        }
      })))

    it.effect(`${dtype}: shared dense GELU-tanh MLP preserves projection and gate rounding`, () =>
      Effect.scoped(Effect.gen(function*() {
        const input = Array.from({ length: 12 }, (_, i) => round((i % 7 - 3) / 4))
        const gate = Array.from({ length: 30 }, (_, i) => round((i % 11 - 5) / 8))
        const up = Array.from({ length: 30 }, (_, i) => round((i % 9 - 4) / 8))
        const down = Array.from({ length: 30 }, (_, i) => round((i % 13 - 6) / 8))
        const tensors = {
          [`${mlpPrefix}.gate_proj.weight`]: yield* tensor(gate, [5, 6], dtype),
          [`${mlpPrefix}.up_proj.weight`]: yield* tensor(up, [5, 6], dtype),
          [`${mlpPrefix}.down_proj.weight`]: yield* tensor(down, [6, 5], dtype)
        }
        const activated = project(input, gate, 6, round).map((x) => round(gelu(x)))
        const projectedUp = project(input, up, 6, round)
        const product = activated.map((x, i) => round(x * projectedUp[i]))
        const expected = project(product, down, 5, round)
        const output = yield* DiffusionGemma.denseMlp(config, tensors, 0, yield* tensor(input, [1, 2, 6], dtype))
        const [actual] = yield* Tensor.compute([output]).pipe(Effect.flatMap(Tensor.clearAllScoped))
        expect(actual.dtype).toBe(dtype)
        close(yield* Tensor.toNumberArray(actual), expected, dtype === "bf16" ? 0 : 3e-6)
      })))

    it.effect(`${dtype}: tied readout normalizes first, rounds projection, and softcaps in F32`, () =>
      Effect.scoped(Effect.gen(function*() {
        const input = [-1.25, 0.3, 0.9, 2.25, -0.1, 1.125, 0.2, 0.25, -0.125, 1.75, -2, 0.4].map(round)
        const weight = Array.from({ length: 42 }, (_, i) => round((i % 13 - 6) * 3.11))
        const norm = [0.75, 1.125, 0.5, 1.25, 0.9, 1.3].map(round)
        const params = yield* Tensor.compute([yield* tensor(weight, [7, 6], dtype), yield* tensor(norm, [6], dtype)])
          .pipe(Effect.flatMap(Tensor.clearAllScoped))
        const tensors = { [embeddingName]: params[0], [normName]: params[1] }
        const x = yield* tensor(input, [1, 2, 6], dtype)
        const selectedIds = [6, 0, 3, 6]
        const ids = yield* Tensor.fromTypedArray(new Uint32Array(selectedIds))
        const full = yield* DiffusionGemma.readout(config, tensors, x)
        const selected = yield* DiffusionGemma.readout(config, tensors, x, ids)
        expect(full.dtype).toBe("f32")
        expect(selected.shape).toEqual([1, 2, 4])
        const [all, restricted] = yield* Tensor.compute([full, selected]).pipe(Effect.flatMap(Tensor.clearAllScoped))
        const expected = project(rms(input, 6, round, norm), weight, 6, round).map((x) =>
          f32(f32(Math.tanh(f32(x / 30))) * 30)
        )
        // Materialized readouts remain valid after their borrowed weights are released.
        yield* Tensor.clearAll(params)
        const values = yield* Tensor.toNumberArray(all)
        close(values, expected, 1e-5)
        close(
          yield* Tensor.toNumberArray(restricted),
          [0, 1].flatMap((row) => selectedIds.map((id) => values[row * 7 + id])),
          1e-5
        )
        expect(values.every((x) => Math.abs(x) < 30)).toBe(true)
      })))
  }

  it.effect("matches pinned Torch BF16 RoPE probe excerpts with dtype-cast frequency buffers", () =>
    Effect.scoped(Effect.gen(function*() {
      // M0 tiny-bf16.json, seed 20260919, probes.apply_rotary_pos_emb#0/#1,
      // first head at position 1023. Copied values keep this test self-contained.
      const fixtures = [
        {
          d: 8,
          frequencies: [1, 0.10009765625, 0.010009765625, 0.00099945068359375],
          input: [
            -0.09521484375,
            -0.015869140625,
            0.0634765625,
            0.142578125,
            0.22265625,
            0.30078125,
            0.380859375,
            0.4609375
          ],
          expected: [
            0.166015625,
            -0.283203125,
            0.2333984375,
            -0.318359375,
            0.1767578125,
            -0.10302734375,
            -0.30859375,
            0.361328125
          ]
        },
        {
          d: 16,
          frequencies: [1, 0.177734375, 0, 0, 0, 0, 0, 0],
          input: [
            -0.1103515625,
            -0.07080078125,
            -0.031494140625,
            0.00787353515625,
            0.04736328125,
            0.08642578125,
            0.1259765625,
            0.1650390625,
            0.205078125,
            0.244140625,
            0.283203125,
            0.322265625,
            0.361328125,
            0.40234375,
            0.44140625,
            0.48046875
          ],
          expected: [
            0.14453125,
            0.02734375,
            -0.031494140625,
            0.00787353515625,
            0.04736328125,
            0.08642578125,
            0.1259765625,
            0.1650390625,
            0.18359375,
            0.251953125,
            0.283203125,
            0.322265625,
            0.361328125,
            0.40234375,
            0.44140625,
            0.48046875
          ]
        }
      ]
      for (const [layer, fixture] of fixtures.entries()) {
        const output = yield* DiffusionGemma.rotaryEmbedding(
          config,
          layer,
          yield* tensor(fixture.input, [1, 1, 1, fixture.d], "bf16"),
          yield* Tensor.fromTypedArray(new BigInt64Array([1023n]), [1, 1]),
          yield* tensor(fixture.frequencies, [fixture.d / 2], "bf16")
        )
        const [actual] = yield* Tensor.compute([output]).pipe(Effect.flatMap(Tensor.clearAllScoped))
        expect(yield* Tensor.toNumberArray(actual)).toEqual(fixture.expected)
      }
    })))

  it.effect("rejects malformed math inputs without compiling", () =>
    Effect.gen(function*() {
      const x = yield* Tensor.ones([1, 2, 6])
      const q = yield* Tensor.ones([1, 2, 3, 8])
      const positions = yield* Tensor.arange(3, undefined, { dtype: "u32" }).pipe(
        Effect.flatMap(Tensor.reshape([1, 3]))
      )
      const record = { [embeddingName]: yield* Tensor.ones([7, 6]), [normName]: yield* Tensor.ones([6]) }
      const cases = [
        DiffusionGemma.embedTokens(config, {}, positions),
        DiffusionGemma.embedTokens(config, record, x),
        DiffusionGemma.rotaryEmbedding(config, -1, q, positions),
        DiffusionGemma.rotaryEmbedding(config, 1, q, positions),
        DiffusionGemma.rotaryEmbedding(config, 0, q, yield* Tensor.ones([1, 3])),
        DiffusionGemma.rotaryEmbedding(config, 0, q, positions, yield* Tensor.ones([3])),
        DiffusionGemma.denseMlp(config, {}, 0, x),
        DiffusionGemma.denseMlp(config, {}, 2, x),
        DiffusionGemma.initialSelfConditioning(config, q),
        DiffusionGemma.readout(config, record, x, yield* Tensor.zeros([0], { dtype: "u32" }))
      ]
      for (const failure of cases) {
        const error = yield* Effect.flip(failure)
        expect(["ModelError", "TensorError"]).toContain(error._tag)
      }
    }))
})

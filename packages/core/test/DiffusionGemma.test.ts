import { describe, expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber } from "effect"
import type * as Schema from "effect/Schema"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Runtime, Tensor } from "../src/index.ts"
import { DiffusionGemma } from "../src/models/index.ts"
import { onDevices } from "./utils/devices.ts"

// Text fields from config.json at f7f5b7f5fa82ffc52addd066915886d497f5517b.
const official = {
  model_type: "diffusion_gemma",
  dtype: "bfloat16",
  canvas_length: 256,
  tie_word_embeddings: true,
  text_config: {
    model_type: "diffusion_gemma_text",
    dtype: "bfloat16",
    vocab_size: 262144,
    hidden_size: 2816,
    intermediate_size: 2112,
    num_hidden_layers: 30,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    num_global_key_value_heads: 2,
    head_dim: 256,
    global_head_dim: 512,
    num_experts: 128,
    top_k_experts: 8,
    moe_intermediate_size: 704,
    max_position_embeddings: 262144,
    sliding_window: 1024,
    use_bidirectional_attention: "vision",
    layer_types: Array.from({ length: 30 }, (_, i) => (i + 1) % 6 === 0 ? "full_attention" : "sliding_attention"),
    rope_parameters: {
      sliding_attention: { rope_type: "default", rope_theta: 10000 },
      full_attention: { rope_type: "proportional", rope_theta: 1000000, partial_rotary_factor: 0.25 }
    }
  }
}
const tiny = {
  model_type: "diffusion_gemma",
  canvas_length: 4,
  text_config: {
    vocab_size: 8,
    hidden_size: 4,
    intermediate_size: 6,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 2,
    num_global_key_value_heads: 1,
    head_dim: 2,
    global_head_dim: 8,
    num_experts: 3,
    top_k_experts: 2,
    moe_intermediate_size: 2,
    max_position_embeddings: 16,
    sliding_window: 4
  }
}
const embedding = "model.decoder.embed_tokens.weight"
const encoderEmbedding = "model.encoder.language_model.embed_tokens.weight"

describe("DiffusionGemma config and catalog", () => {
  it.effect("matches the official text geometry, buffers, ties, and direct normalization scales", () =>
    Effect.gen(function*() {
      const catalog = yield* DiffusionGemma.parameterCatalog(official)
      const { config, parameterSpecs, aliases } = catalog
      const specs = new Map(parameterSpecs.map((spec) => [spec.name, spec]))
      expect(parameterSpecs).toHaveLength(691)
      expect(Object.keys(aliases)).toHaveLength(628)
      expect(config.text_config.layer_types.filter((type) => type === "full_attention")).toHaveLength(5)
      expect(config.text_config.rms_norm_eps).toBe(1e-6)
      expect(config.text_config.final_logit_softcapping).toBe(30)
      expect(config.text_config.rope_parameters).toEqual(official.text_config.rope_parameters)
      expect(specs.get(embedding)?.shape).toEqual([262144, 2816])
      expect(specs.get("model.decoder.layers.0.self_attn.q_proj.weight")?.shape).toEqual([4096, 2816])
      expect(specs.get("model.decoder.layers.0.self_attn.v_proj.weight")?.shape).toEqual([2048, 2816])
      expect(specs.get("model.decoder.layers.5.self_attn.q_proj.weight")?.shape).toEqual([8192, 2816])
      expect(specs.get("model.decoder.layers.5.self_attn.k_proj.weight")?.shape).toEqual([1024, 2816])
      expect(specs.has("model.decoder.layers.5.self_attn.v_proj.weight")).toBe(false)
      expect(specs.has("model.decoder.layers.0.self_attn.v_norm.weight")).toBe(false)
      expect(specs.has("model.decoder.self_conditioning.post_norm.weight")).toBe(false)
      expect(specs.get("model.decoder.layers.0.experts.gate_up_proj")?.shape).toEqual([128, 1408, 2816])
      expect(specs.get("model.decoder.layers.0.experts.down_proj")?.shape).toEqual([128, 2816, 704])
      expect(specs.get("model.decoder.self_conditioning.down_proj.weight")?.shape).toEqual([2816, 2112])
      expect(specs.get("model.decoder.layers.0.self_attn.q_norm.weight")?.initializer).toEqual({
        _tag: "Constant",
        value: 1
      })
      expect(aliases[encoderEmbedding]).toBe(embedding)
      expect(aliases["lm_head.weight"]).toBe(embedding)
      for (let layer = 0; layer < 30; layer++) {
        const encoder = `model.encoder.language_model.layers.${layer}`
        const decoder = `model.decoder.layers.${layer}`
        expect(aliases[`${encoder}.experts.gate_up_proj`]).toBe(`${decoder}.experts.gate_up_proj`)
        expect(aliases[`${encoder}.router.per_expert_scale`]).toBe(`${decoder}.router.per_expert_scale`)
        expect(aliases[`${encoder}.layer_scalar`]).toBeUndefined()
        expect(specs.get(`${encoder}.layer_scalar`)?.shape).toEqual([1])
        expect(specs.get(`${decoder}.layer_scalar`)?.shape).toEqual([1])
      }
      expect(new Set(parameterSpecs.map((spec) => spec.name)).size).toBe(parameterSpecs.length)
    }))

  it.effect("uses upstream defaults and snapshots serialized per-layer overrides", () =>
    Effect.gen(function*() {
      const defaults = yield* DiffusionGemma.parseConfig({
        model_type: "diffusion_gemma",
        text_config: { num_experts: 3, top_k_experts: 2, moe_intermediate_size: 2 }
      })
      expect(defaults.text_config.hidden_size).toBe(2304)
      expect(defaults.text_config.intermediate_size).toBe(9216)
      expect(defaults.text_config.sliding_window).toBe(512)
      expect(defaults.text_config.per_layer_config["5"]).toEqual({ head_dim: 512, num_key_value_heads: 4 })
      const input = {
        ...tiny,
        text_config: {
          ...tiny.text_config,
          global_head_dim: 16,
          layer_types: ["sliding_attention", "sliding_attention"],
          rope_parameters: structuredClone(official.text_config.rope_parameters),
          per_layer_config: { "1": { head_dim: 8, num_key_value_heads: 1 } }
        }
      }
      const config = yield* DiffusionGemma.parseConfig(input)
      input.text_config.per_layer_config["1"].head_dim = 32
      input.text_config.layer_types[0] = "full_attention"
      input.text_config.rope_parameters.full_attention.rope_theta = 99
      expect(config.text_config.per_layer_config["1"]).toEqual({ head_dim: 8, num_key_value_heads: 1 })
      expect(config.text_config.layer_types).toEqual(["sliding_attention", "full_attention"])
      expect(config.text_config.rope_parameters.full_attention.rope_theta).toBe(1000000)
      expect((yield* DiffusionGemma.parameterCatalog(config)).config).toEqual(config)
      const explicitEmpty = yield* DiffusionGemma.parseConfig({
        ...tiny,
        text_config: { ...tiny.text_config, head_dim: 8, per_layer_config: {} }
      })
      expect(explicitEmpty.text_config.per_layer_config["1"]).toEqual({ head_dim: 8, num_key_value_heads: 2 })
      const explicitNull = yield* DiffusionGemma.parseConfig({
        ...tiny,
        text_config: { ...tiny.text_config, head_dim: 8, per_layer_config: null }
      })
      expect(explicitNull).toEqual(explicitEmpty)
      const serialized = yield* DiffusionGemma.parseConfig({
        ...official,
        text_config: {
          ...official.text_config,
          per_layer_config: Object.fromEntries(["05", "11", "17", "23", "29"].map((layer) => [
            layer,
            { head_dim: 512, num_key_value_heads: 2 }
          ]))
        }
      })
      expect(serialized).toEqual(yield* DiffusionGemma.parseConfig(official))
    }))

  it.effect("rejects unsupported contracts and malformed dimensions before I/O", () =>
    Effect.gen(function*() {
      const cases: ReadonlyArray<readonly [Schema.Json, string]> = [
        [{ ...tiny, model_type: "gemma4" }, "model_type"],
        [{ ...tiny, tie_word_embeddings: false }, "tie_word_embeddings"],
        [{ ...tiny, quantization_config: { quant_method: "nvfp4" } }, "quantization_config"],
        [{ ...tiny, dtype: "float16" }, "dtype"],
        [{ ...tiny, dtype: "bfloat16", text_config: { ...tiny.text_config, dtype: "float32" } }, "conflicting dtype"],
        ...[
          ["hidden_size", 0],
          ["num_hidden_layers", 1025],
          ["head_dim", 3],
          ["top_k_experts", 4],
          ["num_key_value_heads", 3],
          ["num_experts", null],
          ["moe_intermediate_size", 1.5],
          ["sliding_window", 17],
          ["hidden_activation", "silu"],
          ["attention_bias", true],
          ["use_bidirectional_attention", "all"],
          ["tie_word_embeddings", false],
          ["layer_types", ["full_attention"]],
          ["per_layer_config", { "2": { head_dim: 8 } }],
          ["per_layer_config", { "1": { hidden_size: 16 } }],
          ["per_layer_config", { "0": {}, "00": {} }],
          ["quantization_config", { quant_method: "gptq" }]
        ].map(([key, value]): readonly [Schema.Json, string] => [
          { ...tiny, text_config: { ...tiny.text_config, [String(key)]: value } },
          String(key)
        ])
      ]
      for (const [input, field] of cases) {
        const error = yield* Effect.flip(DiffusionGemma.parseConfig(input))
        expect(error._tag).toBe("ModelError")
        expect(error.message).toContain(field)
      }
      const overflow = yield* Effect.flip(DiffusionGemma.parameterCatalog({
        ...tiny,
        text_config: { ...tiny.text_config, hidden_size: Number.MAX_SAFE_INTEGER }
      }))
      expect(overflow.message).toContain("safe F32 byte geometry")
      const heterogeneous = yield* Effect.flip(DiffusionGemma.parseConfig({
        ...tiny,
        text_config: {
          ...tiny.text_config,
          num_hidden_layers: 3,
          head_dim: 8,
          per_layer_config: { "0": { head_dim: 4 } }
        }
      }))
      expect(heterogeneous.message).toContain("homogeneous within sliding_attention")
    }))
})

interface Entry {
  readonly name: string
  readonly dtype: "BF16" | "F32" | "U8"
  readonly shape: ReadonlyArray<number>
  readonly value: number
}

// Independent tiny catalog transcribed from the pinned module constructors.
const fixtureEntries = (): Array<Entry> => {
  const entries: Array<Entry> = []
  const add = (name: string, shape: ReadonlyArray<number>, value = 0) =>
    entries.push({ name, shape, dtype: "BF16", value })
  add(embedding, [8, 4], 0.5)
  add("model.decoder.norm.weight", [4], 1)
  add("model.decoder.self_conditioning.pre_norm.weight", [4], 1)
  add("model.decoder.self_conditioning.gate_proj.weight", [6, 4])
  add("model.decoder.self_conditioning.up_proj.weight", [6, 4])
  add("model.decoder.self_conditioning.down_proj.weight", [4, 6])
  for (let layer = 0; layer < 2; layer++) {
    const prefix = `model.decoder.layers.${layer}`
    for (
      const name of [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
        "pre_feedforward_layernorm_2"
      ]
    ) add(`${prefix}.${name}.weight`, [4], 1)
    add(`${prefix}.self_attn.q_proj.weight`, [layer === 0 ? 4 : 16, 4])
    add(`${prefix}.self_attn.k_proj.weight`, [layer === 0 ? 4 : 8, 4])
    if (layer === 0) add(`${prefix}.self_attn.v_proj.weight`, [4, 4])
    add(`${prefix}.self_attn.o_proj.weight`, [4, layer === 0 ? 4 : 16])
    add(`${prefix}.self_attn.q_norm.weight`, [layer === 0 ? 2 : 8], 1)
    add(`${prefix}.self_attn.k_norm.weight`, [layer === 0 ? 2 : 8], 1)
    add(`${prefix}.mlp.gate_proj.weight`, [6, 4])
    add(`${prefix}.mlp.up_proj.weight`, [6, 4])
    add(`${prefix}.mlp.down_proj.weight`, [4, 6])
    add(`${prefix}.router.proj.weight`, [3, 4])
    add(`${prefix}.router.scale`, [4], 1)
    add(`${prefix}.router.per_expert_scale`, [3], 1)
    add(`${prefix}.experts.gate_up_proj`, [3, 4, 4])
    add(`${prefix}.experts.down_proj`, [3, 4, 2])
    entries.push(
      { name: `${prefix}.layer_scalar`, dtype: "F32", shape: [1], value: 2 + layer * 2 },
      {
        name: `model.encoder.language_model.layers.${layer}.layer_scalar`,
        dtype: "F32",
        shape: [1],
        value: 3 + layer * 2
      }
    )
  }
  return entries
}

const writeArchive = (file: string, entries: ReadonlyArray<Entry>) => {
  let offset = 0
  const payloads = entries.map((entry) => {
    const size = entry.dtype === "F32" ? 4 : entry.dtype === "BF16" ? 2 : 1
    const bytes = Buffer.alloc(entry.shape.reduce((n, dim) => n * dim, size))
    for (let i = 0; i < bytes.length; i += size) {
      if (entry.dtype === "F32") bytes.writeFloatLE(entry.value, i)
      else if (entry.dtype === "U8") bytes[i] = entry.value
      else {
        const f32 = Buffer.alloc(4)
        f32.writeFloatLE(entry.value)
        bytes.writeUInt16LE(f32.readUInt32LE() >>> 16, i)
      }
    }
    return bytes
  })
  const header = {
    __metadata__: { format: "pt", fixture: "diffusion-gemma" },
    ...Object.fromEntries(entries.map((entry, i) => {
      const start = offset
      offset += payloads[i].length
      return [entry.name, { dtype: entry.dtype, shape: entry.shape, data_offsets: [start, offset] }]
    }))
  }
  const json = Buffer.from(JSON.stringify(header))
  const padded = Buffer.alloc(Math.ceil(json.length / 8) * 8, " ")
  json.copy(padded)
  const length = Buffer.alloc(8)
  length.writeBigUInt64LE(BigInt(padded.length))
  fs.writeFileSync(file, Buffer.concat([length, padded, ...payloads]))
}

const withDirectory = <A, E, R>(use: (directory: string) => Effect.Effect<A, E, R>) =>
  Effect.acquireUseRelease(
    Effect.sync(() => fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-diffusion-gemma-"))),
    use,
    (directory) => Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true }))
  )

onDevices("DiffusionGemma selected loading", () => (it) => {
  it.effect("loads canonical shards once, skips vision payloads, and keeps scalar owners independent", () =>
    withDirectory((directory) =>
      Effect.gen(function*() {
        const entries = fixtureEntries().reverse()
        const vision = {
          name: "model.encoder.vision_tower.unused.weight",
          dtype: "U8",
          shape: [7],
          value: 7
        } satisfies Entry
        writeArchive(path.join(directory, "first.safetensors"), entries.slice(0, 20))
        writeArchive(path.join(directory, "second.safetensors"), entries.slice(20))
        writeArchive(path.join(directory, "vision.safetensors"), [vision])
        const file = path.join(directory, "model.safetensors.index.json")
        fs.writeFileSync(
          file,
          JSON.stringify({
            weight_map: Object.fromEntries([
              ...entries.map((entry, i) => [entry.name, i < 20 ? "first.safetensors" : "second.safetensors"]),
              [vision.name, "vision.safetensors"]
            ])
          })
        )
        const runtime = yield* Runtime.Runtime
        const before = yield* runtime.extensions.diagnostics.externalMemoryBytes
        const released: Array<Tensor.Concrete> = []
        const selected: Array<ReadonlyArray<string> | undefined> = []
        const service: Runtime.RuntimeService = {
          ...runtime,
          extensions: {
            ...runtime.extensions,
            pathSafetensors: {
              ...runtime.extensions.pathSafetensors,
              load: (file, options) =>
                Effect.gen(function*() {
                  selected.push(options?.names)
                  // Inspection needed the header, but selected loading must not reopen this shard.
                  fs.rmSync(path.join(directory, "vision.safetensors"))
                  return yield* runtime.extensions.pathSafetensors.load(file, options)
                })
            }
          },
          release: (tensor) => Effect.sync(() => released.push(tensor)).pipe(Effect.andThen(runtime.release(tensor)))
        }
        const loaded = yield* DiffusionGemma.loadParameters(file, tiny).pipe(
          Effect.provideService(Runtime.Runtime, service)
        )
        const { tensors, ownedParameters, parameterSpecs, aliases } = loaded
        expect(selected).toEqual([parameterSpecs.map((spec) => spec.name)])
        expect(ownedParameters).toHaveLength(51)
        expect(new Set(ownedParameters).size).toBe(51)
        expect(ownedParameters).toEqual(parameterSpecs.map((spec) => tensors[spec.name]))
        expect(tensors[vision.name]).toBeUndefined()
        for (const [alias, canonical] of Object.entries(aliases)) expect(tensors[alias]).toBe(tensors[canonical])
        expect(tensors[embedding].dtype).toBe("bf16")
        expect(yield* Tensor.toNumberArray(tensors[embedding])).toEqual(Array.from({ length: 32 }, () => 0.5))
        expect(yield* Tensor.toNumberArray(tensors["model.decoder.norm.weight"])).toEqual([1, 1, 1, 1])
        for (let layer = 0; layer < 2; layer++) {
          const decoder = tensors[`model.decoder.layers.${layer}.layer_scalar`]
          const encoder = tensors[`model.encoder.language_model.layers.${layer}.layer_scalar`]
          expect(decoder).not.toBe(encoder)
          expect(decoder.dtype).toBe("f32")
          expect(yield* Tensor.toNumberArray(decoder)).toEqual([2 + layer * 2])
          expect(yield* Tensor.toNumberArray(encoder)).toEqual([3 + layer * 2])
        }
        expect(loaded.metadata).toEqual({ format: "pt", fixture: "diffusion-gemma" })
        yield* Tensor.clearAll(ownedParameters).pipe(Effect.provideService(Runtime.Runtime, service))
        expect(released).toEqual(ownedParameters)
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
      })
    ))

  it.effect("loads a tiny F32 checkpoint without converting weights", () =>
    withDirectory((directory) =>
      Effect.gen(function*() {
        const file = path.join(directory, "model.safetensors")
        writeArchive(file, fixtureEntries().map((entry) => ({ ...entry, dtype: "F32" })))
        const loaded = yield* DiffusionGemma.loadParameters(file, { ...tiny, dtype: "float32" })
        expect(loaded.config.dtype).toBe("float32")
        expect(loaded.ownedParameters.every((tensor) => tensor.dtype === "f32")).toBe(true)
        yield* Tensor.clearAll(loaded.ownedParameters)
      })
    ))

  it.effect("rejects missing, wrong-shaped, unsupported, duplicate-alias, and extra text weights before loading", () =>
    withDirectory((directory) =>
      Effect.gen(function*() {
        const entries = fixtureEntries()
        const alias = { ...entries[0], name: "lm_head.weight" }
        const cases: ReadonlyArray<readonly [ReadonlyArray<Entry>, string]> = [
          [entries.slice(1), "missing required parameter"],
          [
            entries.filter((entry) => entry.name !== "model.encoder.language_model.layers.1.layer_scalar"),
            "layer_scalar"
          ],
          [[{ ...entries[0], shape: [4, 8] }, ...entries.slice(1)], "expected shape [8,4]"],
          [[{ ...entries[0], dtype: "U8" }, ...entries.slice(1)], "unsupported dtype u8"],
          [[...entries, alias], "equality is unproven"],
          [[...entries, { ...alias, shape: [4, 8] }], "expected shape [8,4]"],
          [[...entries, { ...alias, dtype: "F32" }], "alias dtype differs"],
          [[...entries, { ...alias, dtype: "U8" }], "unsupported dtype u8"],
          [[...entries, { ...entries[0], name: encoderEmbedding }], "duplicate alias storage"],
          [
            [...entries, { ...entries[1], name: "model.encoder.language_model.norm.weight" }],
            "duplicate alias storage"
          ],
          [
            [...entries, { ...alias, name: "model.decoder.layers.1.self_attn.v_proj.weight" }],
            "unexpected text parameter"
          ],
          [[...entries, { ...alias, name: "model.decoder.layers.0.experts.weight_scale" }], "quantized format"]
        ]
        const runtime = yield* Runtime.Runtime
        const before = yield* runtime.extensions.diagnostics.externalMemoryBytes
        let loads = 0
        const service: Runtime.RuntimeService = {
          ...runtime,
          extensions: {
            ...runtime.extensions,
            pathSafetensors: {
              ...runtime.extensions.pathSafetensors,
              load: (file, options) =>
                Effect.sync(() => loads++).pipe(Effect.andThen(runtime.extensions.pathSafetensors.load(file, options)))
            }
          }
        }
        for (const [weights, message] of cases) {
          const file = path.join(directory, "invalid.safetensors")
          writeArchive(file, weights)
          const error = yield* Effect.flip(
            DiffusionGemma.loadParameters(file, tiny).pipe(Effect.provideService(Runtime.Runtime, service))
          )
          expect(error._tag).toBe("ModelError")
          expect(error.message).toContain(message)
        }
        expect(loads).toBe(0)
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
      })
    ))

  it.effect("releases every acquired owner in catalog order when headers change after inspection", () =>
    withDirectory((directory) =>
      Effect.gen(function*() {
        const file = path.join(directory, "changed.safetensors")
        const entries = fixtureEntries()
        writeArchive(file, entries)
        const runtime = yield* Runtime.Runtime
        const before = yield* runtime.extensions.diagnostics.externalMemoryBytes
        const released: Array<Tensor.Concrete> = []
        let owners: ReadonlyArray<Tensor.Concrete> = []
        const service: Runtime.RuntimeService = {
          ...runtime,
          extensions: {
            ...runtime.extensions,
            pathSafetensors: {
              ...runtime.extensions.pathSafetensors,
              load: (file, options) =>
                Effect.gen(function*() {
                  writeArchive(file, [{ ...entries[0], dtype: "F32" }, ...entries.slice(1)])
                  const archive = yield* runtime.extensions.pathSafetensors.load(file, options)
                  const byName = new Map(archive.entries.map((entry) => [entry.name, entry.tensor]))
                  owners = options!.names!.map((name) => byName.get(name)!)
                  return archive
                })
            }
          },
          release: (tensor) =>
            Effect.gen(function*() {
              released.push(tensor)
              yield* runtime.release(tensor)
              // One release error must not prevent later owners from being released.
              if (released.length === 1) {
                return yield* new Runtime.BackendError({
                  reason: "execution-failed",
                  backend: "test",
                  operation: "release",
                  phase: "execute",
                  message: "test release failure"
                })
              }
            })
        }
        const error = yield* Effect.flip(
          DiffusionGemma.loadParameters(file, tiny).pipe(Effect.provideService(Runtime.Runtime, service))
        )
        expect(error.message).toContain("dtype changed after inspection")
        expect(released).toHaveLength(51)
        expect(released).toEqual(owners)
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
      })
    ))

  it.effect("keeps selected loading interruptible and delegates partial cleanup to safetensors", () =>
    withDirectory((directory) =>
      Effect.gen(function*() {
        const file = path.join(directory, "interrupted.safetensors")
        writeArchive(file, fixtureEntries())
        const runtime = yield* Runtime.Runtime
        const before = yield* runtime.extensions.diagnostics.externalMemoryBytes
        const started = yield* Deferred.make<void>()
        let released = 0
        const service: Runtime.RuntimeService = {
          ...runtime,
          extensions: {
            ...runtime.extensions,
            pathSafetensors: {
              ...runtime.extensions.pathSafetensors,
              load: (file, options) =>
                Effect.acquireUseRelease(
                  runtime.extensions.pathSafetensors.load(file, options),
                  () => Deferred.succeed(started, undefined).pipe(Effect.andThen(Effect.never)),
                  (archive) =>
                    Effect.forEach(
                      archive.entries,
                      ({ tensor }) => runtime.release(tensor).pipe(Effect.andThen(Effect.sync(() => released++))),
                      { discard: true }
                    )
                )
            }
          }
        }
        const fiber = yield* Effect.forkChild(
          DiffusionGemma.loadParameters(file, tiny).pipe(Effect.provideService(Runtime.Runtime, service))
        )
        yield* Deferred.await(started)
        yield* Fiber.interrupt(fiber)
        expect(released).toBe(51)
        expect(yield* runtime.extensions.diagnostics.externalMemoryBytes).toBe(before)
      })
    ))
})

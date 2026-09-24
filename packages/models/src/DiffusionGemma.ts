/**
 * Text configuration, selected safetensors loading, and graph builders for
 * DiffusionGemma. Builders borrow inputs and parameters and return lazy tensors.
 *
 * The catalog follows Transformers revision
 * `93ebf6b11127967f2725cf4d012aae55c3654f5a` and the canonical decoder names in
 * `google/diffusiongemma-26B-A4B-it` revision
 * `f7f5b7f5fa82ffc52addd066915886d497f5517b`. Normalization weights are direct
 * multipliers initialized to one. Global attention has no V projection weight.
 *
 * @since 0.1.0
 */
import { Decision, type Diffusion, Model, type Runtime, Safetensors, Tensor } from "@effect-torch/core"
import { Model as ServeModel } from "@effect-torch/serve"
import type * as DecisionPlanner from "@effect-torch/serve/DecisionPlanner"
import { canonicalJson } from "@effect-torch/serve/DecisionPlanner"
import type * as DecisionScaffold from "@effect-torch/serve/DecisionScaffold"
import type * as Tokenizers from "@effect-torch/tokenizers"
import { Effect, Exit, Option, Predicate, type Scope } from "effect"
import * as Schema from "effect/Schema"
import { createHash, randomInt } from "node:crypto"
import { readFile } from "node:fs/promises"
import { join } from "node:path"

const PositiveInt = Schema.Int.check(Schema.isGreaterThan(0), Schema.isLessThanOrEqualTo(Number.MAX_SAFE_INTEGER))

const PositiveFinite = Schema.Finite.check(Schema.isGreaterThan(0))

const count = (value: number) => PositiveInt.pipe(Schema.withDecodingDefaultKey(Effect.succeed(value)))

const positive = (value: number) => PositiveFinite.pipe(Schema.withDecodingDefaultKey(Effect.succeed(value)))

const Tied = Schema.Literal(true).pipe(Schema.withDecodingDefaultKey(Effect.succeed(true)))

const Dtype = Schema.Literals(["bfloat16", "float32"])

const LayerType = Schema.Literals(["sliding_attention", "full_attention"])

const LayerOverride = Schema.Struct({
  head_dim: Schema.optionalKey(PositiveInt),
  num_key_value_heads: Schema.optionalKey(PositiveInt)
}).annotate({ parseOptions: { onExcessProperty: "error" } })

const Rope = Schema.Struct({
  sliding_attention: Schema.Struct({
    rope_type: Schema.Literal("default"),
    rope_theta: PositiveFinite
  }).annotate({ parseOptions: { onExcessProperty: "error" } }),
  full_attention: Schema.Struct({
    rope_type: Schema.Literal("proportional"),
    rope_theta: PositiveFinite,
    partial_rotary_factor: PositiveFinite.check(Schema.isLessThanOrEqualTo(1))
  }).annotate({ parseOptions: { onExcessProperty: "error" } })
}).annotate({ parseOptions: { onExcessProperty: "error" } })

const defaultRope: Schema.Schema.Type<typeof Rope> = {
  sliding_attention: {
    rope_type: "default",
    rope_theta: 10_000
  },
  full_attention: {
    rope_type: "proportional",
    rope_theta: 1_000_000,
    partial_rotary_factor: 0.25
  }
}

const TextInput = Schema.Struct({
  model_type: Schema.Literal("diffusion_gemma_text").pipe(
    Schema.withDecodingDefaultKey(Effect.succeed("diffusion_gemma_text"))
  ),
  vocab_size: count(262_144),
  hidden_size: count(2304),
  intermediate_size: count(9216),
  num_hidden_layers: count(30).check(Schema.isLessThanOrEqualTo(1024)),
  num_attention_heads: count(8),
  num_key_value_heads: count(4),
  head_dim: count(256),
  global_head_dim: count(512),
  num_global_key_value_heads: Schema.optionalKey(Schema.NullOr(PositiveInt)),
  hidden_activation: Schema.Literal("gelu_pytorch_tanh").pipe(
    Schema.withDecodingDefaultKey(Effect.succeed("gelu_pytorch_tanh"))
  ),
  max_position_embeddings: count(131_072),
  initializer_range: positive(0.02),
  rms_norm_eps: positive(1e-6),
  tie_word_embeddings: Tied,
  attention_bias: Schema.Literal(false).pipe(Schema.withDecodingDefaultKey(Effect.succeed(false))),
  attention_dropout: Schema.Literal(0).pipe(Schema.withDecodingDefaultKey(Effect.succeed(0))),
  sliding_window: count(512),
  layer_types: Schema.optionalKey(Schema.NullOr(Schema.Array(LayerType))),
  final_logit_softcapping: positive(30),
  use_bidirectional_attention: Schema.NullOr(Schema.Literal("vision")).pipe(
    Schema.withDecodingDefaultKey(Effect.succeed(null))
  ),
  num_experts: PositiveInt,
  top_k_experts: PositiveInt,
  moe_intermediate_size: PositiveInt,
  rope_parameters: Schema.optionalKey(Schema.NullOr(Rope)),
  per_layer_config: Schema.optionalKey(Schema.NullOr(Schema.Record(Schema.String, LayerOverride))),
  dtype: Schema.optionalKey(Dtype),
  torch_dtype: Schema.optionalKey(Dtype),
  quantization_config: Schema.optionalKey(Schema.Null),
  attention_k_eq_v: Tied
})

const ConfigInput = Schema.Struct({
  model_type: Schema.Literal("diffusion_gemma"),
  text_config: TextInput,
  canvas_length: count(256),
  tie_word_embeddings: Tied,
  dtype: Schema.optionalKey(Dtype),
  torch_dtype: Schema.optionalKey(Dtype),
  quantization_config: Schema.optionalKey(Schema.Null)
})

/**
 * Effective per-layer attention geometry, including serialized HF overrides.
 *
 * @since 0.1.0
 * @category models
 */
export type LayerConfig = {
  readonly head_dim: number
  readonly num_key_value_heads: number
}

/**
 * Text-only configuration. `per_layer_config` contains the effective
 * geometry for every zero-based layer. `dtype` is the checkpoint's declared
 * preference; loading preserves each tensor's actual BF16 or F32 dtype.
 *
 * @since 0.1.0
 * @category models
 */
export type Config = {
  readonly model_type: "diffusion_gemma"
  readonly canvas_length: number
  readonly tie_word_embeddings: true
  readonly dtype: "bfloat16" | "float32"

  readonly text_config:
    & Omit<
      Schema.Schema.Type<typeof TextInput>,
      | "global_head_dim"
      | "num_global_key_value_heads"
      | "layer_types"
      | "rope_parameters"
      | "per_layer_config"
      | "dtype"
      | "torch_dtype"
      | "quantization_config"
    >
    & {
      readonly layer_types: ReadonlyArray<Schema.Schema.Type<typeof LayerType>>
      readonly rope_parameters: Schema.Schema.Type<typeof Rope>
      readonly per_layer_config: Readonly<Record<string, LayerConfig>>
    }
}

const configError = (message: string) =>
  new Model.ModelError({ op: "parseConfig", message: `DiffusionGemma config: ${message}` })

/**
 * Validates parsed HF config JSON without I/O or runtime access. Defaults come
 * from the pinned Transformers configuration, not the 26B checkpoint. The MoE
 * expert count, top-k, and expert width are required because their upstream
 * defaults are null and cannot instantiate this model.
 *
 * Omitted layer types use the upstream 5:1 pattern; the final layer is forced
 * to full attention as upstream does. Explicit `per_layer_config` replaces
 * synthesized global overrides; null means no overrides. Zero-padded serialized
 * layer indices are accepted. Head width and KV-head overrides must be
 * homogeneous within each attention type, as required by upstream's shared
 * rotary buffers. Tied, bias-free, GELU-tanh text models with causal encoders and
 * default/local plus proportional/global RoPE are supported. Quantization
 * configurations and dtypes other than BF16/F32 fail validation.
 *
 * Unknown top-level and text metadata, including vision configuration, is
 * discarded. The result snapshots all retained arrays and records.
 *
 * @since 0.1.0
 * @category constructors
 */
export const parseConfig = (input: Schema.Json): Effect.Effect<Config, Model.ModelError> =>
  Effect.gen(function*() {
    const decoded = yield* Schema.decodeUnknownEffect(ConfigInput)(input).pipe(
      Effect.mapError((error) => configError(error.message))
    )

    const {
      global_head_dim,
      num_global_key_value_heads,
      layer_types,
      rope_parameters,
      per_layer_config,
      dtype,
      torch_dtype,
      quantization_config: _quantization,
      ...text
    } = decoded.text_config

    const declaredDtypes = [decoded.dtype, decoded.torch_dtype, dtype, torch_dtype].filter((value) =>
      value !== undefined
    )

    if (new Set(declaredDtypes).size > 1) {
      return yield* configError("conflicting dtype declarations")
    }

    if (text.top_k_experts > text.num_experts) {
      return yield* configError("top_k_experts must not exceed num_experts")
    }

    if (text.sliding_window > text.max_position_embeddings) {
      return yield* configError("sliding_window must not exceed max_position_embeddings")
    }

    const types = layer_types === undefined || layer_types === null
      ? Array.from(
        { length: text.num_hidden_layers },
        (_, layer): Schema.Schema.Type<typeof LayerType> =>
          (layer + 1) % 6 === 0 ? "full_attention" : "sliding_attention"
      )
      : Array.from(layer_types)

    if (types.length !== text.num_hidden_layers) {
      return yield* configError("layer_types must have num_hidden_layers entries")
    }

    types[types.length - 1] = "full_attention"

    const rope = rope_parameters ?? defaultRope

    const overrides = per_layer_config === undefined
      ? Object.fromEntries(types.flatMap((type, layer) =>
        type === "full_attention"
          ? [[String(layer), {
            head_dim: global_head_dim,
            num_key_value_heads: num_global_key_value_heads ?? text.num_key_value_heads
          }]]
          : []
      ))
      : per_layer_config ?? {}

    const indexed = new Map<number, Schema.Schema.Type<typeof LayerOverride>>()

    for (const [key, override] of Object.entries(overrides)) {
      const index = Number(key)

      if (!/^[0-9]+$/.test(key) || !Number.isSafeInteger(index) || index >= text.num_hidden_layers) {
        return yield* configError(`per_layer_config key ${JSON.stringify(key)} must be a layer index`)
      }

      if (indexed.has(index)) {
        return yield* configError(`per_layer_config has duplicate layer index ${index}`)
      }

      indexed.set(index, override)
    }

    const layers: Record<string, LayerConfig> = Object.create(null)
    const byType = new Map<string, LayerConfig>()

    for (let layer = 0; layer < types.length; layer++) {
      const override = indexed.get(layer)
      const head_dim = override?.head_dim ?? text.head_dim
      const num_key_value_heads = override?.num_key_value_heads ?? text.num_key_value_heads

      if (text.num_attention_heads % num_key_value_heads !== 0) {
        return yield* configError(`layer ${layer}: num_attention_heads must be divisible by num_key_value_heads`)
      }

      const rotaryWidth = types[layer] === "full_attention"
        ? head_dim * rope.full_attention.partial_rotary_factor
        : head_dim

      if (head_dim % 2 !== 0 || !Number.isSafeInteger(rotaryWidth) || rotaryWidth < 2 || rotaryWidth % 2 !== 0) {
        return yield* configError(`layer ${layer}: head_dim and active rotary width must be positive even integers`)
      }

      const reference = byType.get(types[layer])

      if (
        reference !== undefined &&
        (reference.head_dim !== head_dim || reference.num_key_value_heads !== num_key_value_heads)
      ) {
        return yield* configError(`per_layer_config must be homogeneous within ${types[layer]}`)
      }

      const geometry = { head_dim, num_key_value_heads }
      layers[String(layer)] = geometry
      byType.set(types[layer], geometry)
    }

    return {
      model_type: decoded.model_type,
      canvas_length: decoded.canvas_length,
      tie_word_embeddings: decoded.tie_word_embeddings,
      dtype: declaredDtypes[0] ?? "bfloat16",
      text_config: {
        ...text,
        layer_types: types,
        per_layer_config: layers,
        rope_parameters: {
          sliding_attention: { ...rope.sliding_attention },
          full_attention: { ...rope.full_attention }
        }
      }
    }
  })

/**
 * Canonical storage specs and exact tied-name aliases. Specs include both
 * encoder and decoder layer-scalar buffers. Aliases never own storage.
 *
 * @since 0.1.0
 * @category models
 */
export interface ParameterCatalog {
  readonly config: Config
  readonly parameterSpecs: ReadonlyArray<Model.ParameterSpec>

  /** Maps each tied name to its canonical decoder name. */
  readonly aliases: Readonly<Record<string, string>>
}

/**
 * Validates configuration and builds the deterministic text tensor catalog.
 * Order is embedding, final norm, four self-conditioning parameters, then
 * each decoder layer followed by its independent encoder scalar. Vision
 * tensors and unweighted normalization operations have no catalog entries.
 *
 * @since 0.1.0
 * @category constructors
 */
export const parameterCatalog = (input: Schema.Json): Effect.Effect<ParameterCatalog, Model.ModelError> =>
  Effect.gen(function*() {
    const config = yield* parseConfig(input)
    const text = config.text_config
    const hiddenSize = text.hidden_size
    const intermediateSize = text.intermediate_size
    const expertCount = text.num_experts
    const expertSize = text.moe_intermediate_size

    const one: Model.ParameterInitializer = { _tag: "Constant", value: 1 }
    const normal: Model.ParameterInitializer = { _tag: "Normal", scale: text.initializer_range }

    const specs: Array<Model.ParameterSpec> = []
    const aliases: Record<string, string> = Object.create(null)

    const add = (
      name: string,
      shape: ReadonlyArray<number>,
      initializer: Model.ParameterInitializer,
      tied: boolean
    ) => {
      specs.push({ name, shape, initializer })

      if (tied) {
        aliases[name.replace("model.decoder.", "model.encoder.language_model.")] = name
      }
    }

    const embedding = "model.decoder.embed_tokens.weight"

    add(embedding, [text.vocab_size, hiddenSize], normal, true)
    aliases["lm_head.weight"] = embedding

    add("model.decoder.norm.weight", [hiddenSize], one, true)

    add("model.decoder.self_conditioning.pre_norm.weight", [hiddenSize], one, false)
    add("model.decoder.self_conditioning.gate_proj.weight", [intermediateSize, hiddenSize], normal, false)
    add("model.decoder.self_conditioning.up_proj.weight", [intermediateSize, hiddenSize], normal, false)
    add("model.decoder.self_conditioning.down_proj.weight", [hiddenSize, intermediateSize], normal, false)

    for (let layer = 0; layer < text.num_hidden_layers; layer++) {
      const prefix = `model.decoder.layers.${layer}`
      const geometry = text.per_layer_config[String(layer)]
      const queryWidth = text.num_attention_heads * geometry.head_dim
      const keyValueWidth = geometry.num_key_value_heads * geometry.head_dim

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
      ) {
        add(`${prefix}.${name}.weight`, [hiddenSize], one, true)
      }

      add(`${prefix}.self_attn.q_proj.weight`, [queryWidth, hiddenSize], normal, true)
      add(`${prefix}.self_attn.k_proj.weight`, [keyValueWidth, hiddenSize], normal, true)

      if (text.layer_types[layer] === "sliding_attention") {
        add(`${prefix}.self_attn.v_proj.weight`, [keyValueWidth, hiddenSize], normal, true)
      }

      add(`${prefix}.self_attn.o_proj.weight`, [hiddenSize, queryWidth], normal, true)
      add(`${prefix}.self_attn.q_norm.weight`, [geometry.head_dim], one, true)
      add(`${prefix}.self_attn.k_norm.weight`, [geometry.head_dim], one, true)

      add(`${prefix}.mlp.gate_proj.weight`, [intermediateSize, hiddenSize], normal, true)
      add(`${prefix}.mlp.up_proj.weight`, [intermediateSize, hiddenSize], normal, true)
      add(`${prefix}.mlp.down_proj.weight`, [hiddenSize, intermediateSize], normal, true)

      add(`${prefix}.router.proj.weight`, [expertCount, hiddenSize], normal, true)
      add(`${prefix}.router.scale`, [hiddenSize], one, true)
      add(`${prefix}.router.per_expert_scale`, [expertCount], one, true)

      add(`${prefix}.experts.gate_up_proj`, [expertCount, 2 * expertSize, hiddenSize], normal, true)
      add(`${prefix}.experts.down_proj`, [expertCount, hiddenSize, expertSize], normal, true)

      add(`${prefix}.layer_scalar`, [1], one, false)
      add(`model.encoder.language_model.layers.${layer}.layer_scalar`, [1], one, false)
    }

    for (const spec of specs) {
      if (!Number.isSafeInteger(spec.shape.reduce((bytes, dim) => bytes * dim, 4))) {
        return yield* configError(`tensor ${spec.name} exceeds safe F32 byte geometry`)
      }
    }

    return { config, parameterSpecs: specs, aliases }
  })

/**
 * Loaded text tensors. `tensors` includes canonical and tied names, which
 * reference the same handles. The caller owns each entry in `ownedParameters`
 * exactly once, in parameter-spec order. Release that array with
 * `Tensor.clearAll(loaded.ownedParameters)` when finished. Do not separately
 * release aliases. Releasing an owner invalidates all its names.
 *
 * @since 0.1.0
 * @category models
 */
export interface LoadedParameters extends ParameterCatalog {
  readonly tensors: Readonly<Record<string, Tensor.Concrete>>
  readonly ownedParameters: ReadonlyArray<Tensor.Concrete>
  readonly metadata: Readonly<Record<string, string>>
}

const loadError = (message: string) =>
  new Model.ModelError({ op: "loadParameters", message: `DiffusionGemma: ${message}` })

/**
 * Inspects a safetensors file or HF shard index, validates required text
 * names/shapes/dtypes, then reads only canonical text tensors and independent
 * scalar buffers. Every stored tied alias is checked for shape and dtype, then
 * rejected: this loader cannot prove equality of separately stored copies.
 * Canonical decoder storage is required. Unexpected text tensors fail closed;
 * only the vision tower and vision embedding projection are ignored.
 *
 * BF16 and F32 storage are preserved without casts or normalization offsets.
 * Safetensors owns partial and late-result cleanup. If model validation fails
 * after loading, every acquired owner receives an independent release attempt.
 *
 * @since 0.1.0
 * @category loading
 */
export const loadParameters = (
  path: string,
  input: Schema.Json
): Effect.Effect<LoadedParameters, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const catalog = yield* parameterCatalog(input)
    const inspection = yield* Safetensors.inspectArchive(path)
    const entries = new Map(inspection.entries.map((entry) => [entry.name, entry]))
    const specs = new Map(catalog.parameterSpecs.map((spec) => [spec.name, spec]))

    const validate = (
      name: string,
      shape: ReadonlyArray<number>,
      dtype: Tensor.DType,
      expected: Model.ParameterSpec
    ) => {
      if (shape.length !== expected.shape.length || shape.some((dim, index) => dim !== expected.shape[index])) {
        return loadError(`${name}: expected shape [${expected.shape}], got [${shape}]`)
      }

      if (dtype !== "bf16" && dtype !== "f32") {
        return loadError(
          `${name}: unsupported dtype ${dtype}; expected dense BF16 or F32, quantized formats are unsupported`
        )
      }

      return Effect.void
    }

    for (const spec of catalog.parameterSpecs) {
      const entry = entries.get(spec.name)

      if (entry === undefined) {
        return yield* loadError(`missing required parameter ${spec.name}`)
      }

      yield* validate(entry.name, entry.shape, entry.dtype, spec)
    }

    for (const entry of inspection.entries) {
      const canonical = catalog.aliases[entry.name]

      if (canonical !== undefined) {
        yield* validate(entry.name, entry.shape, entry.dtype, specs.get(canonical)!)

        if (entry.dtype !== entries.get(canonical)!.dtype) {
          return yield* loadError(`${entry.name}: alias dtype differs from ${canonical}`)
        }

        return yield* loadError(
          `separately stored tied alias ${entry.name} duplicates ${canonical}; equality is unproven and duplicate alias storage is unsupported`
        )
      }

      if (specs.has(entry.name)) {
        continue
      }

      if (
        entry.name.startsWith("model.encoder.vision_tower.") || entry.name.startsWith("model.encoder.embed_vision.")
      ) {
        continue
      }

      return yield* loadError(
        `unexpected text parameter ${entry.name}; unsupported checkpoint layout or quantized format`
      )
    }

    const names = catalog.parameterSpecs.map((spec) => spec.name)
    let ownedParameters: ReadonlyArray<Tensor.Concrete> = []

    return yield* Effect.onExit(
      Effect.gen(function*() {
        const archive = yield* Safetensors.loadArchive(path, { names })
        ownedParameters = names.map((name) => archive.tensors[name])

        // Catch checkpoint changes between inspection and loading.
        for (const spec of catalog.parameterSpecs) {
          const tensor = archive.tensors[spec.name]
          yield* validate(spec.name, tensor.shape, tensor.dtype, spec)

          if (tensor.dtype !== entries.get(spec.name)!.dtype) {
            return yield* loadError(spec.name + ": dtype changed after inspection")
          }
        }

        const tensors = { ...archive.tensors }

        for (const [alias, canonical] of Object.entries(catalog.aliases)) {
          tensors[alias] = tensors[canonical]
        }

        return { ...catalog, tensors, ownedParameters, metadata: archive.metadata }
      }),
      (exit) => Exit.isFailure(exit) ? Tensor.clearAll(ownedParameters) : Effect.void
    )
  })

/**
 * Reads `config.json` and the sharded safetensors index from a local checkpoint
 * directory. The caller owns `ownedParameters`.
 *
 * @since 0.1.0
 * @category loading
 */
export const loadCheckpoint = (
  directory: string
): Effect.Effect<LoadedParameters, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const json = yield* Effect.tryPromise({
      try: () => readFile(join(directory, "config.json"), "utf8"),
      catch: (error) => new Model.ModelError({ op: "loadCheckpoint", message: String(error) })
    })

    const config = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Schema.Json))(json).pipe(
      Effect.mapError((error) => new Model.ModelError({ op: "loadCheckpoint", message: error.message }))
    )

    return yield* loadParameters(join(directory, "model.safetensors.index.json"), config)
  })

const mathError = (op: string, message: string) =>
  new Model.ModelError({ op, message: `DiffusionGemma.${op}: ${message}` })

const denseFloat = (op: string, value: Tensor.Any): Effect.Effect<void, Model.ModelError> =>
  value.storage !== undefined || (value.dtype !== "f32" && value.dtype !== "bf16")
    ? mathError(op, "expected dense F32 or BF16 tensors")
    : Effect.void

const parameter = (
  op: string,
  tensors: Readonly<Record<string, Tensor.Any>>,
  name: string,
  shape: ReadonlyArray<number>,
  dtype?: Tensor.DType
): Effect.Effect<Tensor.Any, Model.ModelError> =>
  Effect.gen(function*() {
    const value = tensors[name]

    if (value === undefined) {
      return yield* mathError(op, `missing parameter ${name}`)
    }

    yield* denseFloat(op, value)

    if (dtype !== undefined && value.dtype !== dtype) {
      return yield* mathError(op, `${name}: expected ${dtype} weight`)
    }

    if (value.shape.length !== shape.length || value.shape.some((dim, i) => dim !== shape[i])) {
      return yield* mathError(op, `${name}: expected shape [${shape}], got [${value.shape}]`)
    }

    return value
  })

const hiddenRows = (op: string, config: Config, input: Tensor.Any): Effect.Effect<void, Model.ModelError> =>
  Effect.gen(function*() {
    yield* denseFloat(op, input)

    if (input.shape.length < 2 || input.shape.at(-1) !== config.text_config.hidden_size) {
      return yield* mathError(op, `expected input [..., ${config.text_config.hidden_size}], got [${input.shape}]`)
    }
  })

/**
 * Looks up tied embedding rows and multiplies by `sqrt(hidden_size)`. The
 * scale is initialized in F32 and cast to the embedding dtype before the
 * multiplication, matching `DiffusionGemmaTextScaledWordEmbedding`.
 * `ids` is an i64/u32 tensor of any shape; the result appends the hidden axis.
 * Parameters are borrowed, and only selected embedding rows enter the graph.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const embedTokens = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  ids: Tensor.Any
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const weight = yield* parameter(
      "embedTokens",
      tensors,
      "model.decoder.embed_tokens.weight",
      [config.text_config.vocab_size, config.text_config.hidden_size]
    )

    const embedded = yield* Tensor.embedding(ids, { weight })

    const scale = yield* Tensor.cast(
      yield* Tensor.constant(Math.sqrt(config.text_config.hidden_size), { dtype: "f32" }),
      embedded.dtype
    )

    return yield* Tensor.mul(embedded, scale)
  })

/**
 * Applies local or proportional global RoPE to `[B, H, S, D]` using explicit
 * i64/u32 positions `[B, S]` or `[1, S]`. Positions may start at a nonzero cache
 * offset. Every pair spans the full half-width: channel `j` pairs with
 * `j + D/2`, including for partially rotated global heads.
 *
 * Global inverse frequencies are `theta ** (-2*j/D)` for the first
 * `floor(partial_rotary_factor * D/2)` pairs and zero for the remaining pairs.
 * Default frequencies follow upstream's F32 constructor. `inverseFrequencies`
 * can supply a `[D/2]` F32/BF16 buffer, including the BF16 buffers produced by
 * upstream `model.to(dtype)`. Positions and frequencies are converted to F32
 * for angles and cos/sin, then cos/sin are cast to the input dtype. Each
 * multiplication and the final addition retains that dtype's rounding boundary.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const rotaryEmbedding = (
  config: Config,
  layer: number,
  input: Tensor.Any,
  positions: Tensor.Any,
  inverseFrequencies?: Tensor.Any
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* denseFloat("rotaryEmbedding", input)

    const geometry = config.text_config.per_layer_config[String(layer)]

    if (!Number.isInteger(layer) || geometry === undefined) {
      return yield* mathError("rotaryEmbedding", `invalid layer ${layer}`)
    }

    const headDim = geometry.head_dim

    if (input.shape.length !== 4 || input.shape[3] !== headDim) {
      return yield* mathError("rotaryEmbedding", `expected input [B, H, S, ${headDim}], got [${input.shape}]`)
    }

    const type = config.text_config.layer_types[layer]
    const rope = config.text_config.rope_parameters[type]
    let frequencies = inverseFrequencies

    if (frequencies === undefined) {
      const pairs = type === "full_attention"
        ? Math.floor(config.text_config.rope_parameters.full_attention.partial_rotary_factor * headDim / 2)
        : headDim / 2

      const values = new Float32Array(headDim / 2)

      for (let index = 0; index < pairs; index++) {
        values[index] = 1 / Math.fround(Math.pow(Math.fround(rope.rope_theta), Math.fround(2 * index / headDim)))
      }

      frequencies = yield* Tensor.fromTypedArray(values)
    }

    return yield* Tensor.rotaryEmbedding(input, input.shape[2], rope.rope_theta, {
      positions,
      inverseFrequencies: frequencies
    })
  })

/**
 * The shared dense MLP of one text layer: GELU-tanh gate, elementwise gate/up
 * product, then down projection. Inputs and all three borrowed weights must
 * have the same F32/BF16 dtype. Projection, activation, product, and output
 * each retain that dtype; this does not apply residuals or layer norms.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const denseMlp = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  input: Tensor.Any
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* hiddenRows("denseMlp", config, input)

    if (!Number.isInteger(layer) || layer < 0 || layer >= config.text_config.num_hidden_layers) {
      return yield* mathError("denseMlp", `invalid layer ${layer}`)
    }

    const prefix = `model.decoder.layers.${layer}.mlp`

    const hiddenSize = config.text_config.hidden_size
    const intermediateSize = config.text_config.intermediate_size

    const gate = yield* parameter("denseMlp", tensors, `${prefix}.gate_proj.weight`, [intermediateSize, hiddenSize])
    const up = yield* parameter("denseMlp", tensors, `${prefix}.up_proj.weight`, [intermediateSize, hiddenSize])
    const down = yield* parameter("denseMlp", tensors, `${prefix}.down_proj.weight`, [hiddenSize, intermediateSize])

    const gateOutput = yield* Tensor.linearRows(input, gate)
    const upOutput = yield* Tensor.linearRows(input, up)

    const activated = yield* Tensor.gelu(gateOutput, { approximate: "tanh" })
    const gated = yield* Tensor.mul(activated, upOutput)

    return yield* Tensor.linearRows(gated, down)
  })

/**
 * First-step self-conditioning for a zero signal. With finite weights and
 * bias-free linears the conditioning MLP contributes zero, but its unweighted
 * post-norm still normalizes the scaled embeddings. Later denoising steps with
 * nonzero self-conditioning require the full conditioning MLP.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const initialSelfConditioning = (
  config: Config,
  embeddings: Tensor.Any
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* hiddenRows("initialSelfConditioning", config, embeddings)

    return yield* Tensor.rmsNorm(embeddings, undefined, config.text_config.rms_norm_eps)
  })

/**
 * Conditions canvas embeddings on the preceding full-vocabulary predictions.
 * Softmax accumulates in F32, then probabilities, soft embeddings, and the
 * conditioning MLP use the embedding dtype and its rounding boundaries.
 * Inputs and parameters remain borrowed. Use initialSelfConditioning for the
 * first refinement step; zero logits would produce a uniform, nonzero signal.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const selfConditioning = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  embeddings: Tensor.Any,
  logits: Tensor.Any
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* hiddenRows("selfConditioning", config, embeddings)
    yield* denseFloat("selfConditioning", logits)

    const text = config.text_config

    if (
      logits.shape.length !== embeddings.shape.length ||
      logits.shape[logits.shape.length - 1] !== text.vocab_size ||
      logits.shape.slice(0, -1).some((dimension, index) => dimension !== embeddings.shape[index])
    ) {
      return yield* mathError("selfConditioning", "logits must match the canvas rows and full vocabulary")
    }

    const weight = yield* parameter(
      "selfConditioning",
      tensors,
      "model.decoder.embed_tokens.weight",
      [text.vocab_size, text.hidden_size],
      embeddings.dtype
    )

    const probabilities = yield* Tensor.cast(yield* Tensor.softmax(yield* Tensor.cast(logits, "f32")), weight.dtype)
    const softEmbeddings = yield* Tensor.matmul(probabilities, weight)

    const scale = yield* Tensor.cast(
      yield* Tensor.constant(Math.sqrt(text.hidden_size), { dtype: "f32" }),
      weight.dtype
    )

    const signal = yield* Tensor.mul(softEmbeddings, scale)
    const prefix = "model.decoder.self_conditioning"

    const preNorm = yield* parameter(
      "selfConditioning",
      tensors,
      prefix + ".pre_norm.weight",
      [text.hidden_size],
      weight.dtype
    )

    const gate = yield* parameter("selfConditioning", tensors, prefix + ".gate_proj.weight", [
      text.intermediate_size,
      text.hidden_size
    ], weight.dtype)

    const up = yield* parameter("selfConditioning", tensors, prefix + ".up_proj.weight", [
      text.intermediate_size,
      text.hidden_size
    ], weight.dtype)

    const down = yield* parameter("selfConditioning", tensors, prefix + ".down_proj.weight", [
      text.hidden_size,
      text.intermediate_size
    ], weight.dtype)

    const normalized = yield* Tensor.rmsNorm(signal, preNorm, text.rms_norm_eps)
    const activated = yield* Tensor.gelu(yield* Tensor.linearRows(normalized, gate), { approximate: "tanh" })
    const gated = yield* Tensor.mul(activated, yield* Tensor.linearRows(normalized, up))
    const conditioned = yield* Tensor.add(embeddings, yield* Tensor.linearRows(gated, down))

    return yield* Tensor.rmsNorm(conditioned, undefined, text.rms_norm_eps)
  })

/**
 * Final learned RMS norm and tied-embedding readout, followed by F32
 * `softcap * tanh(logits / softcap)`. The projection executes in the hidden
 * dtype, including its BF16 output rounding, before conversion to F32. The
 * configured scalar divisor is represented as an F32 reciprocal multiplication,
 * matching the reference scalar operation. Its rounding affects carried BF16
 * predictions even when the F32 logit difference is below readout tolerance.
 *
 * Input is the pre-final-norm hidden tensor `[..., hidden_size]`. Omitting
 * `tokenIds` projects the full vocabulary. A rank-one i64/u32 `tokenIds` tensor
 * selects embedding rows before projection and preserves their supplied order,
 * returning `[..., tokenIds.length]`. No embedding scaling or softmax is applied.
 * Parameters remain borrowed; no full-weight F32 cast is constructed.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const readout = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  input: Tensor.Any,
  tokenIds?: Tensor.Any
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* hiddenRows("readout", config, input)

    const hiddenSize = config.text_config.hidden_size
    const norm = yield* parameter("readout", tensors, "model.decoder.norm.weight", [hiddenSize])

    let weight = yield* parameter("readout", tensors, "model.decoder.embed_tokens.weight", [
      config.text_config.vocab_size,
      hiddenSize
    ])

    if (tokenIds !== undefined) {
      if (tokenIds.shape.length !== 1 || tokenIds.shape[0] === 0) {
        return yield* mathError("readout", "tokenIds must have nonempty shape [K]")
      }

      weight = yield* Tensor.embedding(tokenIds, { weight })
    }

    const hidden = yield* Tensor.rmsNorm(input, norm, config.text_config.rms_norm_eps)

    const projected = yield* Tensor.linearRows(hidden, weight)
    const logits = yield* Tensor.cast(projected, "f32")

    const cap = yield* Tensor.constantLike(logits, config.text_config.final_logit_softcapping)
    const reciprocal = yield* Tensor.constantLike(logits, 1 / Math.fround(config.text_config.final_logit_softcapping))
    const scaled = yield* Tensor.mul(logits, reciprocal)
    const capped = yield* Tensor.tanh(scaled)

    return yield* Tensor.mul(capped, cap)
  })

/**
 * Device-resident routing results, flattened to one row per input token.
 * Probabilities and selected weights are F32; expert indices are U32.
 *
 * @since 0.1.0
 * @category models
 */
export interface Routing {
  readonly probabilities: Tensor.Lazy
  readonly weights: Tensor.Lazy
  readonly indices: Tensor.Lazy
}

/**
 * Builds the router of one text layer. Input normalization, learned input
 * scaling, root-size scaling, and projection preserve their F32/BF16 rounding
 * boundaries. Softmax and top-k renormalization use F32. Selected per-expert
 * scales are applied after renormalization, so the final weights need not sum
 * to one. Results have shapes `[tokens, experts]` and `[tokens, top_k_experts]`.
 * Tied scores use the runtime's deterministic ascending-index order.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const routeTokens = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  input: Tensor.Any
): Effect.Effect<Routing, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* hiddenRows("routeTokens", config, input)

    if (!Number.isInteger(layer) || layer < 0 || layer >= config.text_config.num_hidden_layers) {
      return yield* mathError("routeTokens", `invalid layer ${layer}`)
    }

    const hiddenSize = config.text_config.hidden_size
    const experts = config.text_config.num_experts
    const topK = config.text_config.top_k_experts
    const prefix = `model.decoder.layers.${layer}.router`
    const scale = yield* parameter("routeTokens", tensors, `${prefix}.scale`, [hiddenSize])
    const weight = yield* parameter("routeTokens", tensors, `${prefix}.proj.weight`, [experts, hiddenSize])
    const expertScale = yield* parameter("routeTokens", tensors, `${prefix}.per_expert_scale`, [experts])

    const rows = input.shape.slice(0, -1).reduce((size, dim) => size * dim, 1)
    const flat = yield* Tensor.reshape(input, [rows, hiddenSize])
    const normalized = yield* Tensor.rmsNorm(flat, undefined, config.text_config.rms_norm_eps)

    // PyTorch's Python scalar is applied in opmath F32, then the BF16 result
    // is rounded. A BF16 tensor constant would round the scalar too early.
    const learnedScale = yield* Tensor.mul(normalized, scale)
    const scaledFloat = yield* Tensor.cast(learnedScale, "f32")
    const rootSize = yield* Tensor.constantLike(scaledFloat, hiddenSize ** -0.5)
    const scaledInput = yield* Tensor.mul(scaledFloat, rootSize)
    const scaled = yield* Tensor.cast(scaledInput, input.dtype)

    const projected = yield* Tensor.linearRows(scaled, weight)
    const scores = yield* Tensor.cast(projected, "f32")
    const probabilities = yield* Tensor.softmax(scores)
    const indices = yield* Tensor.topKIndices(probabilities, topK)

    const selected = yield* Tensor.gather(probabilities, indices, { dim: -1 })
    const totalWeight = yield* Tensor.sum(selected, { dims: [-1], keepdims: true })
    const normalizedWeights = yield* Tensor.div(selected, totalWeight)

    const flatIndices = yield* Tensor.reshape(indices, [rows * topK])
    const scales = yield* Tensor.take(expertScale, flatIndices)
    const selectedScale = yield* Tensor.reshape(scales, [rows, topK])
    const scalesFloat = yield* Tensor.cast(selectedScale, "f32")
    const weights = yield* Tensor.mul(normalizedWeights, scalesFloat)

    return {
      probabilities,
      weights,
      indices
    }
  })

/**
 * Executes the selected experts on `[tokens, hidden_size]` rows. Indices and
 * F32 routing weights have shape `[tokens, top_k_experts]`. Expert projections
 * read the original rank-three weight banks directly. Only activations are
 * expanded per selected expert; weight banks are never gathered or transposed.
 *
 * Gate/up and down projections use stable, exact-size expert groups with the
 * backend's ordinary matrix-product numerics, including native BF16 reduction.
 * Gate/up projection, GELU-tanh, product, down projection, and each weighted
 * contribution preserve the activation dtype. Contributions accumulate in
 * ascending expert-index order, with original route order breaking duplicates,
 * matching the official eager expert loop's F32/BF16 rounding boundaries.
 * Input is already normalized by `pre_feedforward_layernorm_2`; this function
 * applies neither that norm nor `post_feedforward_layernorm_2`.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const routedExperts = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  input: Tensor.Any,
  indices: Tensor.Any,
  routingWeights: Tensor.Any
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* hiddenRows("routedExperts", config, input)

    if (!Number.isInteger(layer) || layer < 0 || layer >= config.text_config.num_hidden_layers) {
      return yield* mathError("routedExperts", `invalid layer ${layer}`)
    }

    const hiddenSize = config.text_config.hidden_size
    const expertSize = config.text_config.moe_intermediate_size
    const expertCount = config.text_config.num_experts
    const topK = config.text_config.top_k_experts

    if (indices.shape[1] !== topK) {
      return yield* mathError("routedExperts", `expected ${topK} routes per token`)
    }

    const prefix = `model.decoder.layers.${layer}.experts`

    const gateUp = yield* parameter("routedExperts", tensors, `${prefix}.gate_up_proj`, [
      expertCount,
      2 * expertSize,
      hiddenSize
    ])

    const down = yield* parameter("routedExperts", tensors, `${prefix}.down_proj`, [
      expertCount,
      hiddenSize,
      expertSize
    ])

    return yield* Tensor.gatedExperts(
      input,
      gateUp,
      down,
      indices,
      routingWeights,
      (gate) => Tensor.gelu(gate, { approximate: "tanh" }),
      Tensor.groupedExpertLinearRows
    )
  })

/**
 * Shared dense MLP plus routed experts, including the feed-forward residual
 * and the selected encoder/decoder layer scalar. `input` is the residual
 * after attention and its post-attention norm. Routing consumes that residual
 * directly; the dense MLP and experts have separate input/output RMS norms.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const feedForward = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  input: Tensor.Any,
  mode: "encoder" | "decoder"
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* hiddenRows("feedForward", config, input)

    const hiddenSize = config.text_config.hidden_size
    const prefix = `model.decoder.layers.${layer}`

    const norm = (value: Tensor.Any, name: string) =>
      Effect.gen(function*() {
        const weight = yield* parameter("feedForward", tensors, `${prefix}.${name}.weight`, [hiddenSize])

        return yield* Tensor.rmsNorm(value, weight, config.text_config.rms_norm_eps)
      })

    const routing = yield* routeTokens(config, tensors, layer, input)

    const denseInput = yield* norm(input, "pre_feedforward_layernorm")
    const denseOutput = yield* denseMlp(config, tensors, layer, denseInput)
    const dense = yield* norm(denseOutput, "post_feedforward_layernorm_1")

    const expertNorm = yield* norm(input, "pre_feedforward_layernorm_2")
    const expertInput = yield* Tensor.reshape(expertNorm, [routing.indices.shape[0], hiddenSize])
    const experts = yield* routedExperts(config, tensors, layer, expertInput, routing.indices, routing.weights)
    const expertRows = yield* Tensor.reshape(experts, input.shape)
    const expertOutput = yield* norm(expertRows, "post_feedforward_layernorm_2")

    const summed = yield* Tensor.add(dense, expertOutput)
    const combined = yield* norm(summed, "post_feedforward_layernorm")

    const scalarPrefix = mode === "encoder" ? "model.encoder.language_model" : "model.decoder"
    const scalar = yield* parameter("feedForward", tensors, `${scalarPrefix}.layers.${layer}.layer_scalar`, [1])
    const residual = yield* Tensor.add(input, combined)

    return yield* Tensor.mul(residual, scalar)
  })

/**
 * Projected queries `[B, queryHeads, S, D]` and keys/values `[B, kvHeads, S, D]`.
 * Q/K include learned RMS scales and RoPE; V has only unweighted RMS norm.
 *
 * @since 0.1.0
 * @category models
 */
export interface Projections extends Tensor.KvPair<Tensor.Lazy> {
  readonly queries: Tensor.Lazy
}

/**
 * Lazy attention output `[B, S, hidden_size]` and KV tensors. Prefill returns
 * the prefix rows to retain; canvas attention returns temporary prefix-plus-canvas rows.
 * Materialization transfers ordinary tensor ownership to the caller. Building
 * these graphs neither changes nor releases any input or prefix owner.
 *
 * @since 0.1.0
 * @category models
 */
export interface AttentionResult extends Tensor.KvPair<Tensor.Lazy> {
  readonly output: Tensor.Lazy
}

const geometry = (config: Config, layer: number, input: Tensor.Any) =>
  Effect.gen(function*() {
    const value = config.text_config.per_layer_config[String(layer)]

    if (!Number.isInteger(layer) || value === undefined) {
      return yield* mathError("attention", `invalid layer ${layer}`)
    }

    if (input.storage !== undefined || (input.dtype !== "f32" && input.dtype !== "bf16")) {
      return yield* mathError("attention", "input must be dense F32 or BF16")
    }

    if (
      input.shape.length !== 3 || input.shape[0] === 0 || input.shape[1] === 0 ||
      input.shape[2] !== config.text_config.hidden_size
    ) {
      return yield* mathError(
        "attention",
        `expected normalized input [B > 0, S > 0, ${config.text_config.hidden_size}], got [${input.shape}]`
      )
    }

    return value
  })

/**
 * Projects already-normalized hidden rows into attention Q/K/V. Global V uses
 * the raw K projection, before learned K normalization and RoPE; it never
 * reads a global `v_proj` weight. Each projection retains the input dtype.
 *
 * `positions` is i64/u32 `[B, S]` or `[1, S]` with absolute token positions.
 * Captured frequency buffers use the named tensor
 * `model.decoder.rotary_emb.<layer_type>_inv_freq` and follow
 * {@link rotaryEmbedding}'s F32/BF16 contract. Absent buffers use the configured
 * RoPE frequencies.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const attentionProjections = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  input: Tensor.Any,
  positions: Tensor.Any
): Effect.Effect<Projections, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const { head_dim: headDim, num_key_value_heads: kvHeads } = yield* geometry(config, layer, input)
    const inverseFrequencies = tensors[`model.decoder.rotary_emb.${config.text_config.layer_types[layer]}_inv_freq`]
    const [batch, sequence, hidden] = input.shape
    const heads = config.text_config.num_attention_heads
    const prefix = `model.decoder.layers.${layer}.self_attn`

    const qWeight = yield* parameter(
      "attention",
      tensors,
      `${prefix}.q_proj.weight`,
      [heads * headDim, hidden],
      input.dtype
    )

    const kWeight = yield* parameter(
      "attention",
      tensors,
      `${prefix}.k_proj.weight`,
      [kvHeads * headDim, hidden],
      input.dtype
    )

    const qScale = yield* parameter("attention", tensors, `${prefix}.q_norm.weight`, [headDim], input.dtype)
    const kScale = yield* parameter("attention", tensors, `${prefix}.k_norm.weight`, [headDim], input.dtype)

    const headsFirst = (value: Tensor.Any, count: number) =>
      Effect.gen(function*() {
        const split = yield* Tensor.reshape(value, [batch, sequence, count, headDim])

        return yield* Tensor.transpose(split, [0, 2, 1, 3])
      })

    const qProjection = yield* Tensor.linearRows(input, qWeight)
    const kProjection = yield* Tensor.linearRows(input, kWeight)
    const rawQ = yield* headsFirst(qProjection, heads)
    const rawK = yield* headsFirst(kProjection, kvHeads)

    let rawV = rawK

    if (config.text_config.layer_types[layer] === "sliding_attention") {
      const vWeight = yield* parameter(
        "attention",
        tensors,
        `${prefix}.v_proj.weight`,
        [kvHeads * headDim, hidden],
        input.dtype
      )

      const vProjection = yield* Tensor.linearRows(input, vWeight)
      rawV = yield* headsFirst(vProjection, kvHeads)
    }

    const normalizedQ = yield* Tensor.rmsNorm(rawQ, qScale, config.text_config.rms_norm_eps)
    const normalizedK = yield* Tensor.rmsNorm(rawK, kScale, config.text_config.rms_norm_eps)
    const queries = yield* rotaryEmbedding(config, layer, normalizedQ, positions, inverseFrequencies)
    const keys = yield* rotaryEmbedding(config, layer, normalizedK, positions, inverseFrequencies)
    const values = yield* Tensor.rmsNorm(rawV, undefined, config.text_config.rms_norm_eps)

    return { queries, keys, values }
  })

const attend = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  projected: Projections,
  causal: boolean
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const [batch, heads, sequence, headDim] = projected.queries.shape
    const local = config.text_config.layer_types[layer] === "sliding_attention"

    let options: Tensor.ScaledDotProductAttentionOptions = {
      scale: 1,
      causal,
      rounding: "stepwise",
      layerId: layer,
      retentionWindow: local ? config.text_config.sliding_window - 1 : null
    }

    if (causal) {
      options = { ...options, window: local ? config.text_config.sliding_window : null }
    }

    const context = yield* Tensor.scaledDotProductAttention(
      projected.queries,
      projected.keys,
      projected.values,
      options
    )

    const tokensFirst = yield* Tensor.transpose(context, [0, 2, 1, 3])
    const merged = yield* Tensor.reshape(tokensFirst, [batch, sequence, heads * headDim])

    const outputWeight = yield* parameter(
      "attention",
      tensors,
      `model.decoder.layers.${layer}.self_attn.o_proj.weight`,
      [
        config.text_config.hidden_size,
        heads * headDim
      ],
      context.dtype
    )

    return yield* Tensor.linearRows(merged, outputWeight)
  })

const normalizeLayer = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  input: Tensor.Any,
  name: string
) =>
  Effect.gen(function*() {
    const scale = yield* parameter(
      "layerNorm",
      tensors,
      "model.decoder.layers." + layer + "." + name + ".weight",
      [config.text_config.hidden_size]
    )

    return yield* Tensor.rmsNorm(input, scale, config.text_config.rms_norm_eps)
  })

/**
 * Builds the complete causal encoder graph. Decode compilation binds its
 * semantic attention operations to an appendable sequence; ordinary evaluation
 * computes a complete causal context without persistent state.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const encode = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  tokens: Tensor.Any,
  positions: Tensor.Any
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    let hidden = yield* embedTokens(config, tensors, tokens)

    for (let layer = 0; layer < config.text_config.num_hidden_layers; layer++) {
      const normalized = yield* normalizeLayer(config, tensors, layer, hidden, "input_layernorm")
      const projections = yield* attentionProjections(config, tensors, layer, normalized, positions)
      const attention = yield* attend(config, tensors, layer, projections, true)
      const attentionOutput = yield* normalizeLayer(config, tensors, layer, attention, "post_attention_layernorm")
      const residual = yield* Tensor.add(hidden, attentionOutput)
      hidden = yield* feedForward(config, tensors, layer, residual, "encoder")
      hidden = yield* Tensor.expose(hidden, Model.hiddenExposure(layer))
    }

    return hidden
  })

/**
 * Builds a complete canvas graph with explicit initial or carried predictions.
 * Read-only decode compilation supplies the immutable encoder prefix. Every
 * current canvas row is visible, including in sliding-attention layers.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const denoise = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  tokens: Tensor.Any,
  positions: Tensor.Any,
  prediction: Diffusion.Prediction
): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const embeddings = yield* embedTokens(config, tensors, tokens)

    let hidden = prediction._tag === "Initial"
      ? yield* initialSelfConditioning(config, embeddings)
      : yield* selfConditioning(config, tensors, embeddings, prediction.logits)

    for (let layer = 0; layer < config.text_config.num_hidden_layers; layer++) {
      const normalized = yield* normalizeLayer(config, tensors, layer, hidden, "input_layernorm")
      const projections = yield* attentionProjections(config, tensors, layer, normalized, positions)
      const attention = yield* attend(config, tensors, layer, projections, false)
      const attentionOutput = yield* normalizeLayer(config, tensors, layer, attention, "post_attention_layernorm")
      const residual = yield* Tensor.add(hidden, attentionOutput)
      hidden = yield* feedForward(config, tensors, layer, residual, "decoder")
      hidden = yield* Tensor.expose(hidden, Model.hiddenExposure(layer))
    }

    return hidden
  })

/**
 * Causal prefill of a complete known sequence. Query row `i` attends to `j`
 * exactly when `j <= i`, and local layers additionally require
 * `i - j < sliding_window`. RoPE uses explicit absolute positions, independent
 * of the mask's row indices. Attention scaling is 1.0.
 *
 * Returned local K/V retain the last `min(S, sliding_window - 1)` rows; global
 * K/V retain all S rows. Attention itself sees the full supplied sequence.
 * This graph performs no persistent cache writes.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const prefillAttention = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  input: Tensor.Any,
  positions: Tensor.Any
): Effect.Effect<AttentionResult, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const projected = yield* attentionProjections(config, tensors, layer, input, positions)
    const output = yield* attend(config, tensors, layer, projected, true)

    if (config.text_config.layer_types[layer] === "full_attention") {
      return { output, keys: projected.keys, values: projected.values }
    }

    const start = Math.max(0, input.shape[1] - config.text_config.sliding_window + 1)

    return {
      output,
      keys: yield* Tensor.slice(projected.keys, { start: [0, 0, start, 0] }),
      values: yield* Tensor.slice(projected.values, { start: [0, 0, start, 0] })
    }
  })

/**
 * Bidirectional canvas attention over a borrowed immutable prefix and every
 * canvas token. Local prefixes must already be trimmed to at most
 * `sliding_window - 1` rows. There is no decoder window mask: even the first
 * canvas row can attend to the last canvas row.
 *
 * Supply absolute canvas positions starting at the full logical prefix length,
 * not its trimmed storage length. Returned K/V concatenate the prefix and
 * current canvas in temporary graph storage. They are not a prefix update.
 *
 * @since 0.1.0
 * @category graph builders
 */
export const readAttention = (
  config: Config,
  tensors: Readonly<Record<string, Tensor.Any>>,
  layer: number,
  input: Tensor.Any,
  positions: Tensor.Any,
  prefix: Tensor.KvPair
): Effect.Effect<AttentionResult, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const { head_dim: headDim, num_key_value_heads: kvHeads } = yield* geometry(config, layer, input)
    const shape = prefix.keys.shape

    for (const [name, value] of [["keys", prefix.keys], ["values", prefix.values]] as const) {
      if (value.dtype !== input.dtype || value.storage !== undefined) {
        return yield* mathError("attention", `prefix ${name} must be dense ${input.dtype}`)
      }

      if (
        value.shape.length !== 4 || value.shape[0] !== input.shape[0] || value.shape[1] !== kvHeads ||
        value.shape[2] !== shape[2] || value.shape[3] !== headDim
      ) {
        return yield* mathError(
          "attention",
          `prefix ${name} must have shape [${input.shape[0]}, ${kvHeads}, P, ${headDim}] with matching P`
        )
      }
    }

    if (
      config.text_config.layer_types[layer] === "sliding_attention" && shape[2] >= config.text_config.sliding_window
    ) {
      return yield* mathError("attention", "local prefix must retain at most sliding_window - 1 rows")
    }

    const projected = yield* attentionProjections(config, tensors, layer, input, positions)
    const keys = shape[2] === 0 ? projected.keys : yield* Tensor.concat([prefix.keys, projected.keys], { dim: 2 })

    const values = shape[2] === 0
      ? projected.values
      : yield* Tensor.concat([prefix.values, projected.values], { dim: 2 })

    return {
      output: yield* attend(config, tensors, layer, { queries: projected.queries, keys, values }, false),
      keys,
      values
    }
  })

/**
 * Defines the encoder, denoiser, and readout graphs from a validated parameter
 * catalog. The returned definition contains no tensor values or compiled state.
 *
 * @since 0.1.0
 * @category constructors
 */
export const define = (
  catalog: Pick<ParameterCatalog, "config" | "parameterSpecs" | "aliases">
): Diffusion.Definition => {
  const { aliases, config, parameterSpecs } = catalog
  const names = parameterSpecs.map((parameter) => parameter.name)
  const dtype = config.dtype === "bfloat16" ? "bf16" : "f32"

  const bind = (parameters: ReadonlyArray<Tensor.Any>): Readonly<Record<string, Tensor.Any>> => {
    const tensors = Object.fromEntries(names.map((name, index) => [name, parameters[index]!]))

    for (const [alias, canonical] of Object.entries(aliases)) {
      tensors[alias] = tensors[canonical]!
    }

    return tensors
  }

  return {
    parameterSpecs,
    vocabSize: config.text_config.vocab_size,
    canvasLength: config.canvas_length,
    maxPositions: config.text_config.max_position_embeddings,
    dtype,
    predictionDtype: dtype,
    encode: (parameters, tokens, positions) => encode(config, bind(parameters), tokens, positions),
    denoise: (parameters, tokens, positions, prediction) =>
      denoise(config, bind(parameters), tokens, positions, prediction),
    readout: (parameters, hidden, selection) =>
      Effect.gen(function*() {
        const weights = bind(parameters)

        if (selection._tag === "Full") {
          return yield* readout(config, weights, hidden)
        }

        const rows = yield* Tensor.take(hidden, selection.rows, { dim: 1 })

        return yield* readout(config, weights, rows, selection.labels)
      })
  }
}

/**
 * Pairs a validated catalog with concrete tensors. Extra named buffers become
 * parameters; tied aliases continue to share their canonical value.
 */
export const fromTensors = (
  catalog: Pick<ParameterCatalog, "config" | "parameterSpecs" | "aliases">,
  tensors: Readonly<Record<string, Tensor.Any>>
): Model.Loaded<Diffusion.Definition> => {
  const names = new Set(catalog.parameterSpecs.map((parameter) => parameter.name))
  const aliases = new Set(
    Object.entries(catalog.aliases).filter(([alias, canonical]) => alias !== canonical).map(([alias]) => alias)
  )
  const extras: Array<Model.ParameterSpec> = []

  for (const [name, tensor] of Object.entries(tensors)) {
    if (!names.has(name) && !aliases.has(name)) {
      extras.push({ name, shape: tensor.shape, initializer: { _tag: "Constant", value: 0 } })
    }
  }

  const parameterSpecs = [...catalog.parameterSpecs, ...extras]

  return {
    definition: define({ ...catalog, parameterSpecs }),
    parameters: parameterSpecs.map((parameter) => tensors[parameter.name]!)
  }
}

/** Pairs a loaded checkpoint with its pure definition. */
export const fromLoaded = (loaded: LoadedParameters): Model.Loaded<Diffusion.Definition> =>
  fromTensors(loaded, loaded.tensors)

/**
 * Entropy-bound generation settings from the pinned Transformers reference.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerationSamplerConfig {
  readonly canvasLength: number
  readonly maxSteps: number
  readonly entropyBound: number
  readonly stabilityThreshold: number
  readonly confidenceThreshold: number
  readonly eosTokenIds: ReadonlyArray<number>
  readonly padTokenId: number | undefined
}

/**
 * All statistics use temperature-processed F32 logits, before feedback casting.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerationPrediction {
  readonly sampledTokens: Uint32Array
  readonly argmaxTokens: Uint32Array
  readonly tokenEntropy: Float32Array
  /** Ascending torch.sort order. A permutation of the canvas positions. */
  readonly entropyOrder: Uint32Array
  readonly meanEntropy: number
}

/**
 * @since 0.1.0
 * @category generation
 */
export interface GenerationSamplerState {
  readonly history: ReadonlyArray<Uint32Array>
}

/**
 * @since 0.1.0
 * @category generation
 */
export interface GenerationSample {
  readonly acceptedMask: Uint8Array
  readonly acceptedCanvas: Uint32Array
  readonly canvas: Uint32Array
}

/**
 * cur_step is an int32 tensor in the reference, so its arithmetic is F32. The
 * last evaluated step is one, not zero; the schedule never actually reaches min.
 *
 * @since 0.1.0
 * @category generation
 */
export const generationTemperature = (min: number, max: number, maxSteps: number, remaining: number): number =>
  Math.fround(Math.fround(min) + Math.fround(Math.fround(max - min) * Math.fround(remaining / maxSteps)))

/**
 * Acceptance resets every step. In sorted order, accept when cumulative entropy
 * minus the current entropy is <= the bound. Even a completed refinement draws
 * an entire random canvas before stopping.
 *
 * @since 0.1.0
 * @category generation
 */
export const sampleGenerationCanvas = (
  current: Uint32Array,
  prediction: GenerationPrediction,
  randomCanvas: Uint32Array,
  entropyBound: number
): GenerationSample => {
  const acceptedMask = new Uint8Array(current.length)
  let cumulative = 0

  for (const position of prediction.entropyOrder) {
    const entropy = prediction.tokenEntropy[position]
    cumulative += entropy
    acceptedMask[position] = Math.fround(Math.fround(cumulative) - entropy) <= Math.fround(entropyBound) ? 1 : 0
  }

  const acceptedCanvas = current.slice()
  const canvas = randomCanvas.slice()

  for (let position = 0; position < current.length; position++) {
    if (acceptedMask[position] !== 0) {
      acceptedCanvas[position] = prediction.sampledTokens[position]
      canvas[position] = prediction.sampledTokens[position]
    }
  }

  return { acceptedMask, acceptedCanvas, canvas }
}

/**
 * Stability compares argmax predictions, not the sampled or accepted canvas.
 *
 * @since 0.1.0
 * @category generation
 */
export const stopGeneration = (
  state: GenerationSamplerState,
  prediction: GenerationPrediction,
  stabilityThreshold: number,
  confidenceThreshold: number
) => {
  const stable = state.history.length === stabilityThreshold &&
    state.history.every((previous) => previous.every((token, index) => token === prediction.argmaxTokens[index]))

  const history = stabilityThreshold === 0
    ? []
    : [...state.history, prediction.argmaxTokens.slice()].slice(-stabilityThreshold)

  return {
    state: { history },
    done: stable && prediction.meanEntropy < Math.fround(confidenceThreshold)
  }
}

/**
 * Preserve the first EOS and replace every later token by pad when configured.
 *
 * @since 0.1.0
 * @category generation
 */
export const finishGenerationCanvas = (
  draft: Uint32Array,
  eosTokenIds: ReadonlyArray<number>,
  padTokenId: number | undefined
): Diffusion.CompletedBlock => {
  const tokens = draft.slice()
  const eos = tokens.findIndex((token) => eosTokenIds.includes(token))

  if (eos !== -1 && padTokenId !== undefined) tokens.fill(padTokenId, eos + 1)

  return { tokens, stop: eos !== -1 }
}

/**
 * Constructs the pinned entropy-bound generation policy.
 *
 * @since 0.1.0
 * @category generation
 */
export const generationPolicy = <E, R>(
  config: GenerationSamplerConfig,
  randomCanvas: () => Effect.Effect<Uint32Array, E, R>
): Diffusion.Policy<GenerationSamplerState, GenerationPrediction, E, R> => ({
  canvasLength: config.canvasLength,
  maxSteps: config.maxSteps,
  start: () => ({ history: [] }),
  refine: ({ canvas, prediction, state }) =>
    Effect.map(randomCanvas(), (noise) => {
      const sampled = sampleGenerationCanvas(canvas, prediction, noise, config.entropyBound)
      const stopped = stopGeneration(state, prediction, config.stabilityThreshold, config.confidenceThreshold)

      return { ...stopped, canvas: sampled.canvas, draft: prediction.argmaxTokens.slice() }
    }),
  finish: (draft) => finishGenerationCanvas(draft, config.eosTokenIds, config.padTokenId)
})

/**
 * generation_config.json at checkpoint f7f5b7f5fa82ffc52addd066915886d497f5517b.
 *
 * @since 0.1.0
 * @category generation
 */
export const generationDefaults = {
  maxNewTokens: 256,
  maxSteps: 48,
  entropyBound: 0.1,
  minTemperature: 0.4,
  maxTemperature: 0.8,
  stabilityThreshold: 1,
  confidenceThreshold: 0.005,
  eosTokenIds: [1, 106, 50],
  padTokenId: 0
} as const

/**
 * Explicit random inputs for replay. Canvas draws are uniform vocabulary IDs;
 * exponentials are positive F32 Exp(1) draws in row-major canvas/vocabulary order.
 * A shared seed across frameworks does not imply the same draws.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerationRandom {
  readonly canvas: (length: number, vocabSize: number) => Uint32Array
  readonly exponentials: (length: number) => Float32Array
}

/**
 * Request-local Mulberry32 random stream. Deterministic within this API; use
 * explicit recorded draws for PyTorch comparisons. The seed must be a u32.
 *
 * @since 0.1.0
 * @category generation
 */
export const generationRandom = (seed: number): Effect.Effect<GenerationRandom, Model.ModelError> =>
  Effect.sync(() => Number.isInteger(seed) && seed >= 0 && seed <= 0xffff_ffff).pipe(Effect.flatMap((valid) => {
    if (!valid) return new Model.ModelError({ op: "DiffusionGemma.generate", message: "seed must be a u32" })

    let state = seed

    const next = () => {
      state = (state + 0x6d2b79f5) >>> 0
      let value = Math.imul(state ^ (state >>> 15), state | 1)
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61)

      return (value ^ (value >>> 14)) >>> 0
    }

    return Effect.succeed({
      canvas: (length: number, vocabSize: number) => {
        if (!Number.isInteger(vocabSize) || vocabSize < 1 || vocabSize > 0xffff_ffff) {
          throw new RangeError("vocabSize must be a positive u32")
        }

        const limit = 0x1_0000_0000 - 0x1_0000_0000 % vocabSize

        return Uint32Array.from({ length }, () => {
          let value = next()

          while (value >= limit) value = next()

          return value % vocabSize
        })
      },
      exponentials: (length: number) => Float32Array.from({ length }, () => -Math.log((next() + 0.5) / 0x1_0000_0000))
    })
  }))

/**
 * On-device sampler graph. Outputs are feedback, sampled IDs, argmax IDs, token
 * entropy, ascending entropy order, and mean entropy, in that order. Only the
 * five reduced statistics need host readback. Feedback is cast exactly once to
 * predictionDtype; full readout and all scoring arithmetic remain F32.
 *
 * @since 0.1.0
 * @category generation
 */
export const generationStatistics = (
  logits: Tensor.Any,
  exponentials: Tensor.Any,
  temperature: Tensor.Any,
  predictionDtype: Tensor.DType
): Effect.Effect<ReadonlyArray<Tensor.Any>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const processed = yield* Tensor.div(logits, temperature)
    const probabilities = yield* Tensor.softmax(processed)
    const sampled = yield* Tensor.cast(yield* Tensor.argmax(yield* Tensor.div(probabilities, exponentials), 2), "u32")
    const argmax = yield* Tensor.cast(yield* Tensor.argmax(processed, 2), "u32")
    // Categorical first normalizes logits with logsumexp, then softmaxes those
    // normalized logits for entropy. Keep both F32 rounding boundaries.
    const normalized = yield* Tensor.sub(processed, yield* Tensor.logsumexp(processed, { dims: [2], keepdims: true }))
    const clamped = yield* Tensor.clamp(normalized, { min: -3.4028234663852886e38 })

    const entropy = yield* Tensor.neg(
      yield* Tensor.sum(
        yield* Tensor.mul(clamped, yield* Tensor.softmax(normalized)),
        { dims: [2] }
      )
    )

    const order = yield* Tensor.topKIndices(yield* Tensor.neg(entropy), logits.shape[1]!)
    const mean = yield* Tensor.mean(entropy)

    return [yield* Tensor.cast(processed, predictionDtype), sampled, argmax, entropy, order, mean]
  })

const generationStatisticsWithDeviceNoise = (
  logits: Tensor.Any,
  temperature: Tensor.Any,
  predictionDtype: Tensor.DType
): Effect.Effect<ReadonlyArray<Tensor.Any>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const processed = yield* Tensor.div(logits, temperature)
    const uniform = yield* Tensor.uniform(logits.shape, { dtype: "f32" })
    const gumbel = yield* Tensor.neg(yield* Tensor.log(yield* Tensor.neg(yield* Tensor.log(uniform))))
    const sampled = yield* Tensor.cast(yield* Tensor.argmax(yield* Tensor.add(processed, gumbel), 2), "u32")
    const argmax = yield* Tensor.cast(yield* Tensor.argmax(processed, 2), "u32")
    const normalized = yield* Tensor.sub(processed, yield* Tensor.logsumexp(processed, { dims: [2], keepdims: true }))
    const clamped = yield* Tensor.clamp(normalized, { min: -3.4028234663852886e38 })
    const entropy = yield* Tensor.neg(
      yield* Tensor.sum(
        yield* Tensor.mul(clamped, yield* Tensor.softmax(normalized)),
        { dims: [2] }
      )
    )
    const order = yield* Tensor.topKIndices(yield* Tensor.neg(entropy), logits.shape[1]!)
    const mean = yield* Tensor.mean(entropy)

    return [yield* Tensor.cast(processed, predictionDtype), sampled, argmax, entropy, order, mean]
  })

/**
 * Model-specific generation controls. Defaults come from the pinned checkpoint.
 * whole-block output matches Transformers; exact clips the last published page.
 * Supply random to replay an oracle, or seed for a reproducible local stream.
 *
 * @since 0.1.0
 * @category generation
 */
export interface GenerateOptions<E = never, R = never> {
  readonly maxNewTokens?: number
  readonly maxSteps?: number
  readonly entropyBound?: number
  readonly minTemperature?: number
  readonly maxTemperature?: number
  readonly stabilityThreshold?: number
  readonly confidenceThreshold?: number
  readonly eosTokenIds?: ReadonlyArray<number>
  readonly padTokenId?: number | null
  readonly outputLimit?: "exact" | "whole-block"
  readonly seed?: number
  readonly random?: GenerationRandom
  readonly initialCanvas?: Uint32Array
  readonly compile?: Tensor.CompileOptions
  readonly onPage?: (tokens: Uint32Array, block: Diffusion.Block) => Effect.Effect<void, E, R>
  readonly onProgress?: (progress: Diffusion.Progress) => Effect.Effect<void, E, R>
}

/**
 * Generation results and measured sampler costs. Explicit replay random inputs
 * upload canvasLength * vocabSize * 4 bytes each step. Default sampling draws
 * exponentials on the device. Full logits never cross to host.
 * randomMilliseconds includes generation and validation of the random inputs.
 * processingMilliseconds includes graph compilation, invocation and readback.
 *
 * @since 0.1.0
 * @category generation
 */
export interface Generated extends Diffusion.GenerationResult {
  readonly tokens: Uint32Array
  readonly randomMilliseconds: number
  readonly processingMilliseconds: number
  readonly randomInputBytes: number
  readonly statisticsReadbackBytes: number
}

/**
 * Generates through the artifact-owned encode/refine/commit scheduler. The
 * sampler is compiled once per request and reused across steps and blocks.
 * Explicit replay draws remain on the host and are reported separately.
 *
 * @since 0.1.0
 * @category generation
 */
export const generate = <E = never, R = never>(
  program: Diffusion.Artifact,
  prompt: Uint32Array,
  options: GenerateOptions<E, R> = {}
): Effect.Effect<
  Generated,
  E | Model.ModelError | Tensor.TensorError | Diffusion.InferenceError | Diffusion.DiffusionGenerationError,
  R | Runtime.Runtime
> =>
  Effect.gen(function*() {
    const settings = { ...generationDefaults, ...options }

    const {
      maxNewTokens,
      maxSteps,
      entropyBound,
      minTemperature,
      maxTemperature,
      stabilityThreshold,
      confidenceThreshold
    } = settings

    const invalid = (message: string) => new Model.ModelError({ op: "DiffusionGemma.generate", message })

    if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 0 || !Number.isSafeInteger(maxSteps) || maxSteps < 1) {
      return yield* invalid("maxNewTokens must be nonnegative and maxSteps must be positive integers")
    }

    if (
      !Number.isFinite(entropyBound) || entropyBound <= 0 || !Number.isFinite(confidenceThreshold) ||
      confidenceThreshold <= 0 ||
      !Number.isSafeInteger(stabilityThreshold) || stabilityThreshold < 0 ||
      !Number.isFinite(minTemperature) || minTemperature < 0 || !Number.isFinite(maxTemperature) ||
      maxTemperature <= minTemperature
    ) return yield* invalid("invalid entropy, stability, confidence, or temperature settings")

    if (prompt.length + Math.ceil(maxNewTokens / program.canvasLength) * program.canvasLength > program.maxPositions) {
      return yield* invalid("full generation canvases exceed the model position limit")
    }

    const padTokenId = settings.padTokenId === null ? undefined : settings.padTokenId

    for (const token of [...settings.eosTokenIds, ...padTokenId === undefined ? [] : [padTokenId]]) {
      if (!Number.isInteger(token) || token < 0 || token > 0xffff_ffff) {
        return yield* invalid("invalid special token ID")
      }
    }

    const explicitRandom = options.random
    const random = explicitRandom ??
      (yield* generationRandom(options.seed ?? Math.floor(Math.random() * 0x1_0000_0000)))

    let randomMilliseconds = 0
    let processingMilliseconds = 0
    let randomInputBytes = 0
    let statisticsReadbackBytes = 0

    const noise = () =>
      Effect.gen(function*() {
        const start = performance.now()
        const tokens = random.canvas(program.canvasLength, program.vocabSize)

        if (tokens.length !== program.canvasLength || tokens.some((token) => token >= program.vocabSize)) {
          return yield* invalid("random canvas must contain canvasLength in-vocabulary IDs")
        }

        randomMilliseconds += performance.now() - start

        return tokens
      })

    const policy = generationPolicy({
      canvasLength: program.canvasLength,
      maxSteps,
      entropyBound,
      stabilityThreshold,
      confidenceThreshold,
      eosTokenIds: settings.eosTokenIds,
      padTokenId
    }, noise)

    const sampler = yield* Tensor.compile(
      explicitRandom === undefined
        ? ([logits, temperature]) => generationStatisticsWithDeviceNoise(logits!, temperature!, program.predictionDtype)
        : ([logits, exponentials, temperature]) =>
          generationStatistics(logits!, exponentials!, temperature!, program.predictionDtype),
      options.compile
    )

    const pages: Array<Uint32Array> = []

    const result = yield* program.generate<
      GenerationPrediction,
      GenerationSamplerState,
      E | Model.ModelError | Tensor.TensorError,
      R
    >({
      prompt,
      maxNewTokens,
      outputLimit: options.outputLimit ?? "whole-block",
      policy,
      initialize: (block) =>
        Effect.gen(function*() {
          // Python evaluates the random default even when decoder_input_ids is supplied.
          const drawn = yield* noise()

          const canvas = block.index === 0 && options.initialCanvas !== undefined
            ? options.initialCanvas.slice()
            : drawn

          return { value: { canvas, feedback: { _tag: "Initial" as const } }, release: Effect.void }
        }),
      process: (logits, _block, step) =>
        Effect.suspend(() => {
          let acquired: ReadonlyArray<Tensor.Concrete> = []

          return Effect.scoped(Effect.gen(function*() {
            const inputs: Array<Tensor.Any> = [logits]
            let exponentials: Float32Array | undefined

            if (explicitRandom !== undefined) {
              const randomStart = performance.now()
              exponentials = explicitRandom.exponentials(program.canvasLength * program.vocabSize)

              if (
                exponentials.length !== program.canvasLength * program.vocabSize ||
                exponentials.some((value) => !Number.isFinite(value) || value <= 0)
              ) {
                return yield* invalid(
                  "exponential draws must be positive finite F32 values for the full canvas/vocabulary"
                )
              }

              randomMilliseconds += performance.now() - randomStart
              randomInputBytes += exponentials.byteLength
            }

            const started = performance.now()

            if (exponentials !== undefined) inputs.push(yield* Tensor.fromTypedArray(exponentials, logits.shape))

            inputs.push(
              yield* Tensor.full(
                [],
                generationTemperature(minTemperature, maxTemperature, maxSteps, step.remaining)
              )
            )

            const outputs = yield* Effect.acquireRelease(
              sampler.call(inputs).pipe(Effect.onExit((exit) =>
                Effect.sync(() => {
                  if (Exit.isSuccess(exit)) acquired = exit.value
                })
              )),
              (outputs) => Tensor.clearAll(outputs.slice(1)),
              { interruptible: true }
            )

            const [sampled, argmax, entropy, order, mean] = yield* Effect.forEach(
              outputs.slice(1),
              Tensor.toNumberArray
            )

            statisticsReadbackBytes += (program.canvasLength * 4 + 1) * 4
            processingMilliseconds += performance.now() - started

            return {
              value: {
                prediction: {
                  sampledTokens: Uint32Array.from(sampled!),
                  argmaxTokens: Uint32Array.from(argmax!),
                  tokenEntropy: Float32Array.from(entropy!),
                  entropyOrder: Uint32Array.from(order!),
                  meanEntropy: mean![0]!
                },
                feedback: outputs[0]!
              },
              release: Tensor.clear(outputs[0]!)
            }
          })).pipe(
            // Feedback transfers only after the statistics finalizers complete.
            Effect.onExit((exit) => Exit.isFailure(exit) ? Tensor.clearAll(acquired) : Effect.void)
          )
        }),
      onProgress: (progress) => options.onProgress === undefined ? Effect.void : options.onProgress(progress),
      onPage: (tokens, block) =>
        Effect.gen(function*() {
          pages.push(tokens.slice())

          if (options.onPage !== undefined) yield* options.onPage(tokens, block)
        })
    })

    const tokens = new Uint32Array(result.generatedTokens)
    let offset = 0

    for (const page of pages) {
      tokens.set(page, offset)
      offset += page.length
    }

    return { ...result, tokens, randomMilliseconds, processingMilliseconds, randomInputBytes, statisticsReadbackBytes }
  })

/**
 * Asset, scaffold-boundary, or label validation failure.
 *
 * @since 0.1.0
 * @category models
 */
export class ScaffoldError extends Schema.TaggedErrorClass<ScaffoldError>()("ScaffoldError", {
  message: Schema.String
}) {}

/**
 * Pins audited by test/fixtures/tokenization.json. No model weights are loaded.
 *
 * @since 0.1.0
 * @category models
 */
export const pins = Object.freeze({
  model: "google/diffusiongemma-26B-A4B-it",
  revision: "f7f5b7f5fa82ffc52addd066915886d497f5517b",
  tokenizerSha256: "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
  chatTemplateSha256: "9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf",
  vocabularySize: 262144
})

// First 255 Unicode letters in U+0041..U+052F accepted by the pinned
// full-scaffold audit. Do not normalize these code points. Tokenization is
// verified, but model quality with these labels has not been evaluated.
const choiceLabels = Object.freeze(Array.from(
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzµÀÁÂÃÄÅÆÇÈÉÊÍÎÏÐÑÓÔÖØÚÜÝÞßàáâäåæçèéêìíîïðñòóôõöøùúûüýþ" +
    "ĀāĂăĆćĉČčďĐđēĝħīĮįİıķļľŁłŌőŒœŘřŚśŝŞşŠšūźŻżŽžſƏƒƙƯưȘșțɔɖəɛʻʼΆΈΌΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ" +
    "άέήαβγδεζηθικλμνξοπρστυφχψωόύώϒϕЁЄІЈЎАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ"
))

const labelPolicy = "noul-no-yes/score-1-9-or-A-J/choice-unicode-255/v1"

const thought = "<|channel>thought\n<channel|>"

const canvasText = (label: string) => thought + "answer: " + label + "<turn|>"

const sha256 = (bytes: string | Uint8Array): string => createHash("sha256").update(bytes).digest("hex")

/**
 * The methods required from the native tokenizer; useful for focused boundary tests.
 *
 * @since 0.1.0
 * @category models
 */
export type Tokenizer = Pick<
  Tokenizers.Tokenizer,
  "vocabSize" | "encode" | "applyChatTemplate" | "tokenToId" | "idToToken"
>

/**
 * An injected tokenizer and its serialized-file identity. loadPinned verifies
 * the bytes before constructing the tokenizer. Other adapters must supply their
 * own accurate SHA-256 and revision; their results are not pinned-model evidence.
 *
 * @since 0.1.0
 * @category models
 */
export interface ModelTemplate {
  readonly tokenizer: Tokenizer
  readonly tokenizerSha256: string
  readonly modelRevision: string
  readonly chatTemplate: string
}

/**
 * Load only the two local pinned assets. Missing files and hash mismatches fail.
 *
 * @since 0.1.0
 * @category models
 */
export const loadPinned = (directory: string): Effect.Effect<
  ModelTemplate & { readonly tokenizer: Tokenizers.Tokenizer },
  ScaffoldError | Tokenizers.TokenizerError
> =>
  Effect.gen(function*() {
    const [json, template] = yield* Effect.tryPromise({
      try: () =>
        Promise.all([readFile(join(directory, "tokenizer.json")), readFile(join(directory, "chat_template.jinja"))]),
      catch: (error) => new ScaffoldError({ message: "Cannot read tokenizer assets: " + String(error) })
    })

    if (sha256(json) !== pins.tokenizerSha256 || sha256(template) !== pins.chatTemplateSha256) {
      return yield* new ScaffoldError({ message: "Pinned tokenizer/template SHA-256 mismatch" })
    }

    const native = yield* Effect.tryPromise({
      try: () => import("@effect-torch/tokenizers"),
      catch: (error) => new ScaffoldError({ message: "Cannot load tokenizer addon: " + String(error) })
    })

    // Construct from the exact verified bytes, avoiding a second file read.
    const tokenizer = yield* native.fromJson(
      json.toString("utf8"),
      Object.freeze({
        padding: Object.freeze({ _tag: "None" as const }),
        truncation: Object.freeze({ _tag: "None" as const }),
        specialTokens: "Always"
      })
    )

    if (tokenizer.vocabSize !== pins.vocabularySize) {
      return yield* new ScaffoldError({ message: "Pinned tokenizer vocabulary size mismatch" })
    }

    return Object.freeze({
      tokenizer,
      tokenizerSha256: pins.tokenizerSha256,
      modelRevision: pins.revision,
      chatTemplate: template.toString("utf8")
    })
  })

const BuildOptions = Schema.Struct({
  canvasLength: Schema.Number.check(Schema.isInt(), Schema.isGreaterThanOrEqualTo(1)).pipe(
    Schema.withDecodingDefaultKey(Effect.succeed(16))
  )
})

const display = (value: Schema.Json): string => Predicate.isString(value) ? value : canonicalJson(value)

/**
 * Upstream single-question lines layout, with the fixed internal ID "answer".
 * Rich JSON descriptions extend its plain descriptions through canonical JSON.
 * Score indices stay host-side; only each level's description follows its label.
 */
const messagesFor = (
  decision: DecisionPlanner.QuestionPlan,
  labels: ReadonlyArray<string>
): ReadonlyArray<Tokenizers.ChatMessage> => {
  const instructions = decision.question.instructions === null
    ? "Choose the matching option."
    : display(decision.question.instructions)

  const lines = decision.options.map((option, index) => {
    const head = "  " + labels[index]

    switch (decision.question.type) {
      case "noul":
        return head + (option.description === null ? "" : ": " + display(option.description)) + "\n"
      case "score":
        return head + ": " + display(option.description) + "\n"
      case "choice":
        return head + ": " + option.name +
          (option.description === null ? "" : " (" + display(option.description) + ")") + "\n"
    }
  }).join("")

  const system = "Answer a fixed set of questions about the state the user provides. " +
    "Each question lists its allowed answers; reply with exactly one label per question.\n" +
    "\nQuestion answer: " + instructions + "\n" + lines +
    "\nReply with one line per question, in this order, formatted as \"id: label\"."

  return [{ role: "system", content: system }, { role: "user", content: display(decision.state) }]
}

/**
 * Render an isolated, validated plan and verify EVERY allowed label in the full
 * prompt plus canvas. The key describes actual model input, pinned identities,
 * label policy and canvas size, independently of Planner's provisional text key.
 * This prepares a one-step read; it performs no model inference.
 *
 * @since 0.1.0
 * @category models
 */
export const build = (
  decision: DecisionPlanner.QuestionPlan,
  modelTemplate: ModelTemplate,
  options: { readonly canvasLength?: number } = {}
): Effect.Effect<DecisionScaffold.Scaffold, ScaffoldError | Tokenizers.TokenizerError> =>
  Effect.gen(function*() {
    const config = yield* Schema.decodeUnknownEffect(BuildOptions, { onExcessProperty: "error" })(options).pipe(
      Effect.mapError((error) => new ScaffoldError({ message: error.message }))
    )

    const { tokenizer } = modelTemplate

    if (
      !/^[a-f0-9]{64}$/.test(modelTemplate.tokenizerSha256) || modelTemplate.modelRevision.length === 0 ||
      decision.model.length === 0
    ) {
      return yield* new ScaffoldError({ message: "Model and tokenizer identities are required" })
    }

    if (!Number.isSafeInteger(tokenizer.vocabSize) || tokenizer.vocabSize < 1 || tokenizer.vocabSize > 2 ** 32) {
      return yield* new ScaffoldError({ message: "Invalid tokenizer vocabulary size" })
    }

    const encode = (text: string) =>
      tokenizer.encode(text, { addSpecialTokens: false }).pipe(Effect.flatMap((ids) => {
        if (
          !(ids.data instanceof Uint32Array) || ids.dtype !== "u32" || ids.shape.length !== 1 ||
          ids.shape[0] !== ids.data.length || ids.data.some((id) => id >= tokenizer.vocabSize)
        ) {
          return Effect.fail(new ScaffoldError({ message: "Invalid tokenizer ID buffer" }))
        }

        return Effect.succeed(ids.data.slice())
      }))

    for (
      const [token, id] of [["<pad>", 0], ["<bos>", 2], ["<|channel>", 100], ["<channel|>", 101], ["<|turn>", 105], [
        "<turn|>",
        106
      ]] as const
    ) {
      const found = tokenizer.tokenToId(token)
      const inverse = tokenizer.idToToken(id)
      const encoded = yield* encode(token)

      if (
        Option.isNone(found) || found.value !== id || Option.isNone(inverse) || inverse.value !== token ||
        encoded.length !== 1 || encoded[0] !== id
      ) {
        return yield* new ScaffoldError({ message: "Unexpected special token: " + token })
      }
    }

    const count = decision.options.length

    const codes = decision.question.type === "noul" ?
      ["no", "yes"]
      : decision.question.type === "score" && count <= 9 ?
      Array.from({ length: count }, (_, index) => String(index + 1))
      : choiceLabels.slice(0, count)

    if (count < 1 || count > 255 || codes.length !== count) {
      return yield* new ScaffoldError({ message: "Question has no valid complete label set" })
    }

    const prompt = yield* tokenizer.applyChatTemplate(modelTemplate.chatTemplate, messagesFor(decision, codes), {
      addGenerationPrompt: true,
      variables: { bos_token: "<bos>", enable_thinking: false }
    })

    const prefix = yield* encode(prompt)

    if (prefix.length === 0 || prefix[0] !== 2) {
      return yield* new ScaffoldError({ message: "Chat template must produce the BOS prefix" })
    }

    const encodeCandidate = (code: string) =>
      Effect.gen(function*() {
        const canvas = yield* encode(canvasText(code))
        const full = yield* encode(prompt + canvasText(code))

        if (
          full.length !== prefix.length + canvas.length || prefix.some((id, index) => full[index] !== id) ||
          canvas.some((id, index) => full[prefix.length + index] !== id)
        ) {
          return yield* new ScaffoldError({ message: "Prompt/canvas boundary retokenizes for label " + code })
        }

        if (canvas[canvas.length - 1] !== 106) {
          return yield* new ScaffoldError({ message: "Canvas must end with the turn-close token" })
        }

        return { canvas, full }
      })

    const baseline = yield* encodeCandidate(codes[0])
    // The singleton Choice still needs a second candidate to discover its slot.
    // That candidate is a probe only and is never an allowed output token.
    const alternative = yield* encodeCandidate(codes[1] ?? "B")
    const changed = Array.from(baseline.full).flatMap((id, index) => id === alternative.full[index] ? [] : [index])

    if (baseline.full.length !== alternative.full.length || changed.length !== 1) {
      return yield* new ScaffoldError({ message: "Labels must differ at exactly one full-scaffold token" })
    }

    const fullSlot = changed[0]
    const slot = fullSlot - prefix.length

    if (slot < 0 || slot >= baseline.canvas.length - 1) {
      return yield* new ScaffoldError({ message: "Answer slot is outside the canvas body" })
    }

    const tokenIds: Array<number> = []

    for (const [index, code] of codes.entries()) {
      const candidate = index === 0 ? baseline : index === 1 ? alternative : yield* encodeCandidate(code)

      if (
        candidate.full.length !== baseline.full.length ||
        candidate.full.some((id, i) => i !== fullSlot && id !== baseline.full[i])
      ) {
        return yield* new ScaffoldError({ message: "Label changes multiple scaffold tokens: " + code })
      }

      tokenIds.push(candidate.full[fullSlot])
    }

    if (new Set(tokenIds).size !== count) {
      return yield* new ScaffoldError({ message: "Label normalization collides at the answer slot" })
    }

    if (config.canvasLength < baseline.canvas.length) {
      return yield* new ScaffoldError({ message: "Canvas is too short for the answer scaffold and turn-close token" })
    }

    const canvas = yield* Effect.try({
      try: () => {
        const ids = new Uint32Array(config.canvasLength)
        ids.set(baseline.canvas)

        return ids
      },
      catch: (error) => new ScaffoldError({ message: "Cannot allocate canvas: " + String(error) })
    })

    const chatTemplateSha256 = sha256(modelTemplate.chatTemplate)

    const semanticKey = sha256(canonicalJson({
      format: "decision-model/upstream-lines-scaffold/v1",
      model: decision.model,
      revision: modelTemplate.modelRevision,
      tokenizerSha256: modelTemplate.tokenizerSha256,
      chatTemplateSha256,
      labelPolicy,
      questionType: decision.question.type,
      prompt,
      prefix: Array.from(prefix),
      canvas: Array.from(canvas),
      slot,
      codes,
      tokenIds
    }))

    const allowed = Uint32Array.from(tokenIds)

    return Object.freeze({
      semanticKey,
      model: decision.model,
      prompt,
      tokenizerSha256: modelTemplate.tokenizerSha256,
      chatTemplateSha256,
      vocabularySize: tokenizer.vocabSize,
      slot,
      contentLength: baseline.canvas.length,
      labels: Object.freeze(codes.map((code, index) =>
        Object.freeze({ code, optionName: decision.options[index].name, tokenId: tokenIds[index] })
      )),
      get prefixIds() {
        return prefix.slice()
      },
      get canvasIds() {
        return canvas.slice()
      },
      get allowedTokenIds() {
        return allowed.slice()
      }
    })
  })

/** @since 0.1.0 @category loading */
export interface LoadOptions {
  readonly directory: string
  readonly id: string
  readonly maxTokens?: number
}

type ServeLoadOptions = ServeModel.DiffusionLoadOptions<
  Diffusion.Prefix,
  Decision.DiffusionInput,
  Decision.DecisionError | Diffusion.InferenceError | Tensor.TensorError,
  ScaffoldError | Tokenizers.TokenizerError,
  Model.ModelError | Tensor.TensorError | Diffusion.InferenceError | Diffusion.DiffusionGenerationError
>

/**
 * Loads checkpoint and tokenizer assets, compiles one shared diffusion
 * artifact, and registers generation and decision capabilities. The active
 * scope owns the loaded tensors and registration resources.
 *
 * @since 0.1.0
 * @category loading
 */
export const load = (
  options: LoadOptions
): Effect.Effect<
  ServeModel.Registration,
  Model.ModelError | Tensor.TensorError | ScaffoldError | Tokenizers.TokenizerError | ServeModel.ModelError,
  Runtime.Runtime | Scope.Scope
> =>
  Effect.gen(function*() {
    const loaded = yield* Effect.acquireRelease(
      loadCheckpoint(options.directory),
      (checkpoint) => Tensor.clearAll(checkpoint.ownedParameters),
      { interruptible: true }
    )
    const template = yield* loadPinned(options.directory)

    const loadOptions: ServeLoadOptions = {
      family: "Diffusion" as const,
      id: options.id,
      definition: define(loaded),
      parameters: loaded.ownedParameters,
      compile: {
        maxTokens: options.maxTokens ?? 16384,
        blockSize: 16,
        prefillChunks: [16, 64, 256],
        canvasLengths: [loaded.config.canvas_length],
        selectedReadouts: Array.from({ length: 255 }, (_, index) => ({ rows: 1, labels: index + 1 }))
      } satisfies Diffusion.CompileOptions,
      generation: (artifact: Diffusion.Artifact) => ({
        tokenizer: template.tokenizer,
        template: template.chatTemplate,
        variables: { bos_token: "<bos>", enable_thinking: false },
        eosTokens: generationDefaults.eosTokenIds,
        chatHeaderEnd: 101,
        validateGeneration: (request: ServeModel.GenerationRequest) =>
          request.temperature !== undefined || request.topP !== undefined
            ? Effect.fail(
              new ServeModel.ModelError({
                message: "DiffusionGemma does not support temperature or top_p overrides",
                invalidRequest: true
              })
            )
            : Effect.void,
        generator: {
          run: ({ prompt, onPage, settings }) =>
            generate<ServeModel.ModelError>(artifact, prompt, {
              maxNewTokens: settings.maxTokens,
              outputLimit: "exact",
              seed: settings.seed ?? randomInt(0x1_0000_0000),
              onPage: (tokens) => onPage({ tokens })
            }).pipe(Effect.map((result) => result.stop === "length" ? "maxTokens" as const : "stop" as const))
        }
      }),
      decision: (artifact: Diffusion.Artifact) => ({
        scorer: Decision.fromDiffusion(artifact),
        prepare: (plan: Parameters<typeof build>[0]) => build(plan, template, { canvasLength: artifact.canvasLength }),
        prefill: (ids: Uint32Array) =>
          Effect.acquireRelease(
            artifact.encode(ids),
            (prefix) => Effect.orDie(artifact.release(prefix)),
            { interruptible: true }
          ),
        input: (prefix: Diffusion.Prefix, canvas: Uint32Array, slot: number) =>
          Effect.succeed({ input: { prefix, canvas }, row: slot }),
        mapError: (error: { readonly message: string; readonly _tag?: string }) =>
          new ServeModel.ModelError({
            message: error.message,
            invalidRequest: error._tag === "ScaffoldError"
          })
      })
    }

    return yield* ServeModel.load(loadOptions)
  })

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
import { Effect, Exit } from "effect"
import * as Schema from "effect/Schema"
import * as Model from "../Model.ts"
import type * as Runtime from "../Runtime.ts"
import * as Safetensors from "../Safetensors.ts"
import * as Tensor from "../Tensor.ts"

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
 * Final learned RMS norm and tied-embedding readout, followed by F32
 * `softcap * tanh(logits / softcap)`. The projection executes in the hidden
 * dtype, including its BF16 output rounding, before conversion to F32.
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
    const scaled = yield* Tensor.div(logits, cap)
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

    return yield* Tensor.gatedExperts(input, gateUp, down, indices, routingWeights, (gate) =>
      Tensor.gelu(gate, { approximate: "tanh" }))
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
      rounding: "stepwise"
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
 * Defines the text graphs for a shared Model.executor. The definition borrows
 * parameters and builds embeddings, attention/feed-forward layers, and readout.
 * Named rotary buffers, when present, supply the captured inverse frequencies.
 *
 * @since 0.1.0
 * @category constructors
 */
export const make = (
  loaded: Pick<LoadedParameters, "config" | "tensors">
): Model.PrefixModel => {
  const { config, tensors } = loaded
  const text = config.text_config
  const normalize = (
    layer: number,
    input: Tensor.Any,
    name: string
  ): Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime> =>
    Effect.gen(function*() {
      const scale = yield* parameter("layerNorm", tensors, `model.decoder.layers.${layer}.${name}.weight`, [
        text.hidden_size
      ])

      return yield* Tensor.rmsNorm(input, scale, text.rms_norm_eps)
    })

  return {
    vocabSize: text.vocab_size,
    maxTokens: text.max_position_embeddings,
    maxReadTokens: config.canvas_length,
    layerCount: text.num_hidden_layers,
    embed: (ids, phase) =>
      Effect.gen(function*() {
        const embeddings = yield* embedTokens(config, tensors, ids)

        return phase === "read" ? yield* initialSelfConditioning(config, embeddings) : embeddings
      }),
    prefillLayer: (layer, hidden, positions) =>
      Effect.gen(function*() {
        const normalized = yield* normalize(layer, hidden, "input_layernorm")
        const attention = yield* prefillAttention(config, tensors, layer, normalized, positions)
        const attentionOutput = yield* normalize(layer, attention.output, "post_attention_layernorm")
        const residual = yield* Tensor.add(hidden, attentionOutput)
        const output = yield* feedForward(config, tensors, layer, residual, "encoder")

        return { output, keys: attention.keys, values: attention.values }
      }),
    readLayer: (layer, hidden, positions, prefix) =>
      Effect.gen(function*() {
        const normalized = yield* normalize(layer, hidden, "input_layernorm")
        const attention = yield* readAttention(config, tensors, layer, normalized, positions, prefix)
        const attentionOutput = yield* normalize(layer, attention.output, "post_attention_layernorm")
        const residual = yield* Tensor.add(hidden, attentionOutput)

        return yield* feedForward(config, tensors, layer, residual, "decoder")
      }),
    readout: (hidden, labels) => readout(config, tensors, hidden, labels)
  }
}

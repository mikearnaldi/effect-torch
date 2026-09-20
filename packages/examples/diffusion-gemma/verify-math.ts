/** Compare DiffusionGemma graph builders against the pinned official tiny fixtures. */
import * as BackendAppleNative from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import * as BackendCuda from "@effect-torch/backend-cuda"
import { type Model, Runtime, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/core/models"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"

const Reference = Schema.Struct({ tensor: Schema.String })
const Dimension = Schema.Int.check(
  Schema.isGreaterThanOrEqualTo(0),
  Schema.isLessThanOrEqualTo(Number.MAX_SAFE_INTEGER)
)
const FixtureTensor = Schema.Struct({
  dtype: Schema.Literals(["F32", "BF16", "I64", "BOOL"]),
  shape: Schema.Array(Dimension),
  values: Schema.Array(Schema.Union([Schema.Finite, Schema.String]))
})
const Call = Schema.Struct({
  implementation: Schema.String,
  target: Schema.String,
  args: Schema.Array(Schema.Json),
  kwargs: Schema.Record(Schema.String, Schema.Json),
  output: Schema.Json
})
const Fixture = Schema.Struct({
  schema_version: Schema.Literal(1),
  dtype: Schema.Literals(["F32", "BF16"]),
  config: Schema.Json,
  tensors: Schema.Record(Schema.String, FixtureTensor),
  weights: Schema.Record(Schema.String, Reference),
  buffers: Schema.Record(Schema.String, Reference),
  outputs: Schema.Struct({ logits: Reference }),
  calls: Schema.Record(Schema.String, Call)
})
type FloatTensor = {
  readonly dtype: "F32" | "BF16"
  readonly shape: ReadonlyArray<number>
  readonly values: ReadonlyArray<number>
}
type Graph = Effect.Effect<Tensor.Lazy, Model.ModelError | Tensor.TensorError, Runtime.Runtime>
interface Check {
  readonly call: string
  readonly inputs: ReadonlyArray<string>
  readonly expected: FloatTensor
  readonly build: (inputs: ReadonlyArray<Tensor.Lazy>) => Graph
  /** Raw oracle projection before F32 softcap, used to propagate its BF16 rounding allowance. */
  readonly projection?: FloatTensor
}
interface Failure {
  readonly index: number
  readonly actual: number
  readonly expected: number
  readonly absoluteError: number
  readonly tolerance: number
}
interface Result {
  readonly dtype: "f32" | "bf16"
  readonly outputDtype: Tensor.DType
  readonly optimized: boolean
  readonly call: string
  readonly elements: number
  readonly maxAbsoluteError: number
  readonly exact: number
  readonly failedElements: number
  readonly firstFailures: ReadonlyArray<Failure>
}

const sameShape = (left: ReadonlyArray<number>, right: ReadonlyArray<number>) =>
  left.length === right.length && left.every((dim, i) => dim === right[i])
const float = (value: typeof FixtureTensor.Type): FloatTensor => {
  if (value.dtype !== "F32" && value.dtype !== "BF16") throw new Error(`expected a float tensor, got ${value.dtype}`)
  return {
    dtype: value.dtype,
    shape: value.shape,
    values: Schema.decodeUnknownSync(Schema.Array(Schema.Finite))(value.values)
  }
}
const arity = (name: string, call: typeof Call.Type, count: number) => {
  if (call.args.length !== count) {
    throw new Error(`${name}: expected ${count} positional arguments, got ${call.args.length}`)
  }
}
const bf16Ulp = (value: number) => Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(value))) - 7))
const f32Tolerance = (value: number) => 2e-6 + 2e-5 * Math.abs(value)

const prepare = (fixture: typeof Fixture.Type, config: DiffusionGemma.Config) => {
  const data = new Map(Object.entries(fixture.tensors))
  const get = (name: string) => {
    const value = data.get(name)
    if (value === undefined) throw new Error(`missing tensor ${name}`)
    const size = value.shape.reduce((count, dim) => count * dim, 1)
    if (!Number.isSafeInteger(size) || size !== value.values.length) {
      throw new Error(`${name}: shape and element count differ`)
    }
    return value
  }
  const ref = (reference: Schema.Json) => {
    const name = Schema.decodeUnknownSync(Reference)(reference).tensor
    get(name)
    return name
  }
  const weight = (name: string) => {
    const reference = fixture.weights[name]
    if (reference === undefined) throw new Error(`missing weight ${name}`)
    return ref(reference)
  }
  const recorded = (name: string, implementation: string) => {
    const call = fixture.calls[name]
    if (call === undefined || call.implementation !== implementation) {
      throw new Error(`missing ${implementation} call ${name}`)
    }
    return call
  }
  const checks: Array<Check> = []
  const embedding = "model.decoder.embed_tokens.weight"
  const norm = "model.decoder.norm.weight"
  let mathCalls = 0
  let rotaryCalls = 0
  const rotaryPhases = { prefill: 0, probes: 0, read: 0 }
  for (const [name, call] of Object.entries(fixture.calls)) {
    const add = (inputs: ReadonlyArray<string>, build: Check["build"]) => {
      checks.push({ call: name, inputs, expected: float(get(ref(call.output))), build })
    }
    if (call.implementation === "DiffusionGemmaRMSNorm") {
      arity(name, call, 1)
      const unweighted = /[.](?:self_attn[.]v_norm|router[.]norm|self_conditioning[.]post_norm)$/.test(call.target)
      add(
        unweighted ? [ref(call.args[0])] : [ref(call.args[0]), weight(`${call.target}.weight`)],
        ([input, scale]) => Tensor.rmsNorm(input, scale, config.text_config.rms_norm_eps)
      )
      mathCalls++
    } else if (call.implementation === "DiffusionGemmaTextScaledWordEmbedding") {
      arity(name, call, 1)
      add(
        [ref(call.args[0]), weight(embedding)],
        ([ids, scale]) => DiffusionGemma.embedTokens(config, { [embedding]: scale }, ids)
      )
      mathCalls++
    } else if (call.implementation === "DiffusionGemmaText4MLP") {
      arity(name, call, 1)
      const match = /^model[.](?:decoder|encoder[.]language_model)[.]layers[.]([0-9]+)[.]mlp$/.exec(call.target)
      if (match === null) throw new Error(`${name}: unexpected dense MLP target ${call.target}`)
      const layer = Number(match[1])
      const prefix = `model.decoder.layers.${layer}.mlp`
      const names = ["gate_proj", "up_proj", "down_proj"].map((part) => `${prefix}.${part}.weight`)
      add([ref(call.args[0]), ...names.map(weight)], ([input, gate, up, down]) =>
        DiffusionGemma.denseMlp(
          config,
          { [names[0]]: gate, [names[1]]: up, [names[2]]: down },
          layer,
          input
        ))
      mathCalls++
    } else if (call.implementation === "DiffusionGemmaSelfConditioning") {
      arity(name, call, 2)
      const input = float(get(ref(call.args[0])))
      const signal = float(get(ref(call.args[1])))
      if (
        input.dtype !== signal.dtype || !sameShape(input.shape, signal.shape) ||
        signal.values.some((value) => value !== 0)
      ) {
        throw new Error(`${name}: expected zero self-conditioning with the same dtype and shape as embeddings`)
      }
      add([ref(call.args[0])], ([input]) => DiffusionGemma.initialSelfConditioning(config, input))
      mathCalls++
    } else if (call.implementation === "official.apply_rotary_pos_emb") {
      arity(name, call, 3)
      const match = /^(prefill|probes|read)[.]apply_rotary_pos_emb#([0-9]+)$/.exec(name)
      if (match === null) throw new Error(`unexpected rotary call ${name}`)
      const phase = match[1]
      const index = Number(match[2])
      const layer = phase === "probes" ? index : Math.floor(index / 2)
      if (phase === "prefill") rotaryPhases.prefill++
      else if (phase === "probes") rotaryPhases.probes++
      else rotaryPhases.read++
      const type = config.text_config.layer_types[layer]
      if (type === undefined) throw new Error(`${name}: invalid rotary layer ${layer}`)
      const rotaryTarget = phase === "prefill" ? "model.encoder.language_model.rotary_emb" : "model.decoder.rotary_emb"
      const generators = Object.entries(fixture.calls).filter(([key, candidate]) =>
        key.startsWith(`${phase}.`) && candidate.target === rotaryTarget &&
        candidate.implementation === "DiffusionGemmaTextRotaryEmbedding" && candidate.args[2] === type
      )
      if (generators.length !== 1) throw new Error(`${name}: expected one ${type} frequency generator in ${phase}`)
      const [generatorName, generator] = generators[0]
      arity(generatorName, generator, 3)
      const buffer = fixture.buffers[`${rotaryTarget}.${type}_inv_freq`]
      if (buffer === undefined) throw new Error(`${name}: missing captured inverse-frequency buffer`)
      const axis = call.kwargs.unsqueeze_dim ?? 1
      if (axis !== 1 && axis !== 2) throw new Error(`${name}: unsupported unsqueeze_dim ${JSON.stringify(axis)}`)
      const input = float(get(ref(call.args[0])))
      const cos = float(get(ref(call.args[1])))
      const sin = float(get(ref(call.args[2])))
      const positions = get(ref(generator.args[1]))
      const sequence = input.shape[axis === 2 ? 1 : 2]
      if (
        input.shape.length !== 4 || input.shape[3] !== config.text_config.per_layer_config[String(layer)].head_dim ||
        !sameShape(cos.shape, sin.shape) || cos.shape.length !== 3 || cos.shape[2] !== input.shape[3] ||
        cos.shape[1] !== sequence || !sameShape(positions.shape, cos.shape.slice(0, 2)) ||
        cos.dtype !== input.dtype || sin.dtype !== input.dtype
      ) throw new Error(`${name}: rotary inputs, captured cos/sin, and positions have inconsistent metadata`)
      add(
        [ref(call.args[0]), ref(generator.args[1]), ref(buffer)],
        ([input, positions, frequencies]) =>
          Effect.gen(function*() {
            const headsFirst = axis === 2 ? yield* Tensor.transpose(input, [0, 2, 1, 3]) : input
            const output = yield* DiffusionGemma.rotaryEmbedding(config, layer, headsFirst, positions, frequencies)
            return axis === 2 ? yield* Tensor.transpose(output, [0, 2, 1, 3]) : output
          })
      )
      rotaryCalls++
    }
  }
  if (
    mathCalls !== 57 || rotaryCalls !== 10 || rotaryPhases.prefill !== 4 || rotaryPhases.probes !== 2 ||
    rotaryPhases.read !== 4
  ) {
    throw new Error(
      `expected 57 math and 10 rotary calls, found ${mathCalls} and ${rotaryCalls}: ${JSON.stringify(rotaryPhases)}`
    )
  }
  const finalNorm = recorded("read.model.decoder.norm#0", "DiffusionGemmaRMSNorm")
  const head = recorded("read.lm_head#0", "Linear")
  arity("read.model.decoder.norm#0", finalNorm, 1)
  arity("read.lm_head#0", head, 1)
  const normalized = float(get(ref(finalNorm.output)))
  const headInput = float(get(ref(head.args[0])))
  if (
    !sameShape(normalized.shape, headInput.shape) || normalized.dtype !== headInput.dtype ||
    normalized.values.some((value, i) => value !== headInput.values[i])
  ) {
    throw new Error("recorded lm_head input differs from the final normalization output")
  }
  const logits = float(get(ref(fixture.outputs.logits)))
  const projection = float(get(ref(head.output)))
  if (logits.dtype !== "F32" || projection.dtype !== fixture.dtype || !sameShape(logits.shape, projection.shape)) {
    throw new Error("final projection and F32 softcap metadata differ")
  }
  const selectedIds = [0, 5, 36]
  if (config.text_config.vocab_size !== 37 || logits.shape.at(-1) !== 37) {
    throw new Error("expected the pinned 37-token tiny vocabulary")
  }
  const selectedName = "verify-math.selected_ids"
  if (data.has(selectedName)) throw new Error(`reserved fixture name ${selectedName}`)
  data.set(selectedName, { dtype: "I64", shape: [3], values: selectedIds })
  const selected = (value: FloatTensor): FloatTensor => ({
    dtype: value.dtype,
    shape: [...value.shape.slice(0, -1), selectedIds.length],
    values: Array.from(
      { length: value.values.length / 37 },
      (_, row) => selectedIds.map((id) => value.values[row * 37 + id])
    ).flat()
  })
  checks.push(
    {
      call: "readout.full",
      inputs: [ref(finalNorm.args[0]), weight(embedding), weight(norm)],
      expected: logits,
      projection,
      build: ([input, embeddingWeight, normWeight]) =>
        DiffusionGemma.readout(config, { [embedding]: embeddingWeight, [norm]: normWeight }, input)
    },
    {
      call: "readout.restricted",
      inputs: [ref(finalNorm.args[0]), weight(embedding), weight(norm), selectedName],
      expected: selected(logits),
      projection: selected(projection),
      build: ([input, embeddingWeight, normWeight, ids]) =>
        DiffusionGemma.readout(config, { [embedding]: embeddingWeight, [norm]: normWeight }, input, ids)
    },
    {
      call: "readout.restricted_projection_from_oracle_norm",
      inputs: [ref(head.args[0]), weight(embedding), selectedName],
      expected: selected(projection),
      build: ([input, embeddingWeight, ids]) =>
        Effect.gen(function*() {
          const rows = yield* Tensor.embedding(ids, { weight: embeddingWeight })
          return yield* Tensor.linearRows(input, rows)
        })
    }
  )
  return { get, checks }
}

const graph = (value: typeof FixtureTensor.Type) =>
  Effect.gen(function*() {
    if (value.dtype === "I64") {
      const integers = value.values.map((value) => {
        const integer = Schema.decodeUnknownSync(Schema.Union([
          Schema.Int.check(Schema.isBetween({ minimum: Number.MIN_SAFE_INTEGER, maximum: Number.MAX_SAFE_INTEGER })),
          Schema.String.check(Schema.isPattern(/^-?[0-9]+$/))
        ]))(value)
        const result = BigInt(integer)
        if (result < -(2n ** 63n) || result >= 2n ** 63n) throw new Error("I64 fixture value is out of range")
        return result
      })
      return yield* Tensor.fromTypedArray(BigInt64Array.from(integers), value.shape)
    }
    const values = float(value)
    const root = yield* Tensor.fromTypedArray(Float32Array.from(values.values), values.shape)
    return values.dtype === "BF16" ? yield* Tensor.cast(root, "bf16") : root
  })

const main = Effect.gen(function*() {
  const [device, directory, reportPath, extra] = process.argv.slice(2)
  if (directory === undefined || reportPath === undefined || extra !== undefined) {
    throw new Error("usage: verify-math.ts <cpu|metal|cuda> <numerical-fixtures-directory> <output-report.json>")
  }
  if (device !== "cpu" && device !== "metal" && device !== "cuda") throw new Error("select cpu, metal or cuda")
  if (device === "cuda" && !(yield* BackendCuda.isAvailable)) throw new Error("CUDA is required for this gate")
  if (device === "metal" && !(yield* BackendAppleNative.isAvailable)) throw new Error("Metal is required for this gate")
  const backend = device === "cpu"
    ? BackendCpu.layer
    : device === "metal"
    ? BackendAppleNative.layer()
    : BackendCuda.layer()
  const report = yield* Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const results: Array<Result> = []
    const diagnostics: Array<
      { file: string; uniqueInputTensors: number; externalMemoryBytesBefore: number; externalMemoryBytesAfter: number }
    > = []
    for (const [file, expectedDtype] of [["tiny-f32.json", "F32"], ["tiny-bf16.json", "BF16"]] as const) {
      const fixture = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Fixture))(
        readFileSync(join(directory, file), "utf8")
      )
      if (fixture.dtype !== expectedDtype) throw new Error(`${file}: expected ${expectedDtype} fixture`)
      const config = yield* DiffusionGemma.parseConfig(fixture.config)
      const { checks, get } = prepare(fixture, config)
      const names = Array.from(new Set(checks.flatMap((check) => check.inputs)))
      const externalMemoryBytesBefore = yield* runtime.extensions.diagnostics.externalMemoryBytes
      const roots = yield* Effect.forEach(names, (name) => graph(get(name)))
      yield* Effect.acquireUseRelease(
        Tensor.compute(roots),
        (owners) =>
          Effect.gen(function*() {
            if (owners.length !== names.length) throw new Error("fixture input materialization arity differs")
            const byName = new Map(names.map((name, i) => [name, owners[i]]))
            for (const optimized of [false, true]) {
              for (const check of checks) {
                const bindings = check.inputs.map((name) => {
                  const owner = byName.get(name)
                  if (owner === undefined) throw new Error(`missing input owner ${name}`)
                  return owner
                })
                const inputs = yield* Effect.forEach(bindings, (value, slot) => Tensor.makeInput(slot, value))
                const root = yield* check.build(inputs)
                const program = yield* Tensor.freezeProgram([root], { optimize: optimized, constantWeights: false })
                yield* Effect.acquireUseRelease(
                  Tensor.runProgram(program, bindings),
                  (outputs) =>
                    Effect.gen(function*() {
                      if (outputs.length !== 1) {
                        throw new Error(`${check.call}: expected one result, got ${outputs.length}`)
                      }
                      const actual = outputs[0]
                      const outputDtype = check.expected.dtype === "BF16" ? "bf16" : "f32"
                      if (actual.dtype !== outputDtype || !sameShape(actual.shape, check.expected.shape)) {
                        throw new Error(`${check.call}: output dtype/rank/shape differs from the oracle`)
                      }
                      const values = yield* Tensor.toNumberArray(actual)
                      if (values.length !== check.expected.values.length) {
                        throw new Error(`${check.call}: output arity differs from the oracle`)
                      }
                      let maxAbsoluteError = 0
                      let exact = 0
                      let failedElements = 0
                      const firstFailures: Array<Failure> = []
                      for (let i = 0; i < values.length; i++) {
                        const expected = check.expected.values[i]
                        if (!Number.isFinite(values[i]) || !Number.isFinite(expected)) {
                          throw new Error(`${check.call}[${i}]: nonfinite output or oracle value`)
                        }
                        const absoluteError = Math.abs(values[i] - expected)
                        maxAbsoluteError = Math.max(maxAbsoluteError, absoluteError)
                        if (values[i] === expected) exact++
                        // Softcap is 1-Lipschitz, so one BF16 projection ULP bounds
                        // its propagated error. Add only the fixed F32 allowance.
                        const tolerance = outputDtype === "bf16" ?
                          bf16Ulp(expected) :
                          f32Tolerance(expected) +
                          (check.projection?.dtype === "BF16" ? bf16Ulp(check.projection.values[i]) : 0)
                        if (absoluteError > tolerance) {
                          failedElements++
                          if (firstFailures.length < 8) {
                            firstFailures.push({ index: i, actual: values[i], expected, absoluteError, tolerance })
                          }
                        }
                      }
                      results.push({
                        dtype: fixture.dtype === "BF16" ? "bf16" : "f32",
                        outputDtype,
                        optimized,
                        call: check.call,
                        elements: values.length,
                        maxAbsoluteError,
                        exact,
                        failedElements,
                        firstFailures
                      })
                    }),
                  Tensor.clearAll
                )
              }
            }
          }),
        Tensor.clearAll
      )
      diagnostics.push({
        file,
        uniqueInputTensors: names.length,
        externalMemoryBytesBefore,
        externalMemoryBytesAfter: yield* runtime.extensions.diagnostics.externalMemoryBytes
      })
    }
    return {
      device,
      gate: "official tiny DiffusionGemma math",
      expectedChecksPerDtypeAndMode: { capturedMath: 57, capturedRotary: 10, readout: 3 },
      selectedTokenIds: [0, 5, 36],
      optimizationModes: [false, true],
      tolerance: {
        f32: { atol: 2e-6, rtol: 2e-5 },
        bf16: "one output ULP",
        softcap: "fixed F32 allowance plus one raw BF16 projection ULP"
      },
      diagnostics,
      results
    }
  }).pipe(Effect.provide(backend))
  mkdirSync(dirname(reportPath), { recursive: true })
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n")
  const failures = report.results.filter((result) => result.failedElements > 0)
  if (failures.length > 0) {
    const first = failures[0]
    throw new Error(
      `${device}: ${failures.length} comparisons failed; ${first.dtype} optimize=${first.optimized} ${first.call}: ${
        JSON.stringify(first.firstFailures[0])
      }; report ${reportPath}`
    )
  }
  console.log(`${device}: ${report.results.length} official math comparisons passed; report ${reportPath}`)
})

NodeRuntime.runMain(main)

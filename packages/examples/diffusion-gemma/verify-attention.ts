/** Verify complete attention graphs against recorded official Transformers calls. */
import * as BackendAppleNative from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import * as BackendCuda from "@effect-torch/backend-cuda"
import { type Model, Runtime, Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import * as DG from "../../core/src/models/DiffusionGemma.ts"

const Reference = Schema.Struct({ tensor: Schema.String })
const Dimension = Schema.Int.check(Schema.isBetween({ minimum: 0, maximum: Number.MAX_SAFE_INTEGER }))
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
  calls: Schema.Record(Schema.String, Call),
  cache: Schema.Array(Schema.Struct({ keys: Reference, values: Reference, length: Dimension }))
})
interface FloatTensor {
  readonly dtype: "F32" | "BF16"
  readonly shape: ReadonlyArray<number>
  readonly values: ReadonlyArray<number>
}
interface Comparison {
  readonly part: string
  readonly expected: FloatTensor
}
interface Check {
  readonly call: string
  readonly inputs: ReadonlyArray<string>
  readonly comparisons: ReadonlyArray<Comparison>
  readonly build: (
    get: (name: string) => Tensor.Lazy
  ) => Effect.Effect<ReadonlyArray<Tensor.Lazy>, Model.ModelError | Tensor.TensorError, Runtime.Runtime>
}
interface Failure {
  readonly index: number
  readonly actual: number
  readonly expected: number
  readonly absoluteError: number
  readonly tolerance: number
}
interface Result {
  readonly dtype: "F32" | "BF16"
  readonly optimized: boolean
  readonly call: string
  readonly part: string
  readonly elements: number
  readonly exact: number
  readonly maxAbsoluteError: number
  readonly maxToleranceRatio: number
  readonly failedElements: number
  readonly firstFailures: ReadonlyArray<Failure>
}
const sameShape = (a: ReadonlyArray<number>, b: ReadonlyArray<number>) =>
  a.length === b.length && a.every((d, i) => d === b[i])
const float = (tensor: typeof FixtureTensor.Type): FloatTensor => {
  if (tensor.dtype !== "F32" && tensor.dtype !== "BF16") throw new Error("expected F32/BF16 tensor")
  return {
    dtype: tensor.dtype,
    shape: tensor.shape,
    values: Schema.decodeUnknownSync(Schema.Array(Schema.Finite))(tensor.values)
  }
}
const sameTensor = (a: FloatTensor, b: FloatTensor) =>
  a.dtype === b.dtype && sameShape(a.shape, b.shape) && a.values.every((v, i) => v === b.values[i])
const sequenceSlice = (value: FloatTensor, start: number, length: number): FloatTensor => {
  if (value.shape.length !== 4 || start < 0 || start + length > value.shape[2]) {
    throw new Error("invalid KV sequence slice")
  }
  const [b, h, s, d] = value.shape
  return {
    dtype: value.dtype,
    shape: [b, h, length, d],
    values: Array.from(
      { length: b * h },
      (_, i) => value.values.slice((i * s + start) * d, (i * s + start + length) * d)
    ).flat()
  }
}
const outputRefs = (call: typeof Call.Type) =>
  Schema.decodeUnknownSync(Schema.Tuple([Reference, Reference]))(call.output)
const bf16Ulp = (value: number) => Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(value))) - 7))

const prepare = (fixture: typeof Fixture.Type, config: DG.Config) => {
  const get = (name: string) => {
    const value = fixture.tensors[name]
    if (value === undefined) throw new Error("missing tensor " + name)
    const size = value.shape.reduce((a, b) => a * b, 1)
    if (!Number.isSafeInteger(size) || size !== value.values.length) {
      throw new Error(name + ": shape/element count differs")
    }
    return value
  }
  const ref = (reference: Schema.Json) => {
    const name = Schema.decodeUnknownSync(Reference)(reference).tensor
    get(name)
    return name
  }
  const expected = (reference: Schema.Json) => float(get(ref(reference)))
  const recorded = (name: string, implementation: string) => {
    const call = fixture.calls[name]
    if (call?.implementation !== implementation) throw new Error("missing " + implementation + " call " + name)
    return call
  }
  if (config.text_config.num_hidden_layers !== 2 || fixture.cache.length !== 2) {
    throw new Error("expected pinned two-layer fixture")
  }
  const checks: Array<Check> = []
  for (const phase of ["prefill", "read"] as const) {
    for (const layer of [0, 1]) {
      const encoder = phase === "prefill"
      const target = (encoder ? "model.encoder.language_model" : "model.decoder") + ".layers." + layer + ".self_attn"
      const name = phase + "." + target + "#0"
      const call = recorded(name, encoder ? "DiffusionGemmaEncoderTextAttention" : "DiffusionGemmaDecoderTextAttention")
      if (call.args.length !== 0 || call.target !== target) throw new Error(name + ": unexpected call arity/target")
      const eager = recorded(phase + ".eager_attention_forward." + target + "#0", "official.eager_attention_forward")
      if (
        eager.args.length !== 5 || eager.kwargs.scaling !== 1 || eager.kwargs.dropout !== 0 ||
        eager.kwargs.is_causal !== encoder
      ) throw new Error(name + ": unsupported eager arguments")
      const inputs = new Set<string>()
      const take = (reference: Schema.Json) => {
        const name = ref(reference)
        inputs.add(name)
        return name
      }
      const hidden = take(call.kwargs.hidden_states)
      const shape = get(hidden).shape
      if (shape.length !== 3 || shape[0] === 0 || shape[1] === 0 || shape[2] !== config.text_config.hidden_size) {
        throw new Error(name + ": invalid hidden rows")
      }
      const type = config.text_config.layer_types[layer]
      const rotaryTarget = (encoder ? "model.encoder.language_model" : "model.decoder") + ".rotary_emb"
      const rotaryCalls = Object.entries(fixture.calls).filter(([key, c]) =>
        key.startsWith(phase + ".") && c.target === rotaryTarget &&
        c.implementation === "DiffusionGemmaTextRotaryEmbedding" && c.args[2] === type
      )
      if (rotaryCalls.length !== 1 || rotaryCalls[0][1].args.length !== 3) {
        throw new Error(name + ": expected one rotary generator")
      }
      const positionIds = take(rotaryCalls[0][1].args[1])
      const actualPositions = get(positionIds)
      const callPositions = get(ref(call.kwargs.position_ids))
      if (
        !sameShape(actualPositions.shape, [shape[0], shape[1]]) ||
        !sameShape(actualPositions.shape, callPositions.shape) || actualPositions.values.some((v, i) =>
          v !== callPositions.values[i]
        )
      ) throw new Error(name + ": rotary and attention positions differ")
      const inverseFrequencies = take(fixture.buffers[rotaryTarget + "." + type + "_inv_freq"])
      const prefix = "model.decoder.layers." + layer + ".self_attn"
      const names = [
        "q_proj",
        "k_proj",
        "q_norm",
        "k_norm",
        "o_proj",
        ...(type === "sliding_attention" ? ["v_proj"] : [])
      ].map((part) => prefix + "." + part + ".weight")
      const weights = names.map((name) => take(fixture.weights[name]))
      const cache = fixture.cache[layer]
      const cachedKeys = expected(cache.keys)
      const cachedValues = expected(cache.values)
      const retained = type === "sliding_attention"
        ? Math.min(cache.length, config.text_config.sliding_window - 1)
        : cache.length
      if (cachedKeys.shape[2] !== retained || !sameShape(cachedKeys.shape, cachedValues.shape)) {
        throw new Error(name + ": incorrect cache retention")
      }
      const q = take(eager.args[1])
      const k = take(eager.args[2])
      const v = take(eager.args[3])
      const expectedKeys = float(get(k))
      const expectedValues = float(get(v))
      const cacheStart = encoder ? shape[1] - retained : 0
      if (
        !sameTensor(cachedKeys, sequenceSlice(expectedKeys, cacheStart, retained)) ||
        !sameTensor(cachedValues, sequenceSlice(expectedValues, cacheStart, retained))
      ) throw new Error(name + ": eager KV and immutable prefix differ")
      if (
        !encoder &&
        actualPositions.values.some((position, i) => BigInt(position) !== BigInt(cache.length + i % shape[1]))
      ) throw new Error(name + ": canvas positions must start at full logical prefix length")
      const prefixKeys = encoder ? undefined : take(cache.keys)
      const prefixValues = encoder ? undefined : take(cache.values)
      const mask = encoder ? take(eager.args[4]) : undefined
      if (!encoder && (eager.args[4] !== null || call.kwargs.attention_mask !== null)) {
        throw new Error(name + ": decoder must be unmasked")
      }
      const outputs = outputRefs(call)
      const eagerOutputs = outputRefs(eager)
      const oracleContext = take(eagerOutputs[0])
      const currentStart = encoder ? 0 : retained
      const comparisons: Array<Comparison> = [
        { part: "output", expected: expected(outputs[0]) },
        { part: "keys", expected: encoder ? cachedKeys : expectedKeys },
        { part: "values", expected: encoder ? cachedValues : expectedValues },
        { part: "projected_queries", expected: float(get(q)) },
        { part: "projected_keys", expected: sequenceSlice(expectedKeys, currentStart, shape[1]) },
        { part: "projected_values", expected: sequenceSlice(expectedValues, currentStart, shape[1]) },
        { part: "probabilities_from_oracle_qkv", expected: expected(eagerOutputs[1]) },
        { part: "context_from_oracle_qkv", expected: expected(eagerOutputs[0]) },
        { part: "output_from_oracle_context", expected: expected(outputs[0]) }
      ]
      if (comparisons.some((c) => c.expected.dtype !== fixture.dtype)) {
        throw new Error(name + ": attention output dtype differs from fixture")
      }
      checks.push({
        call: name,
        inputs: Array.from(inputs),
        comparisons,
        build: (get) =>
          Effect.gen(function*() {
            const tensors = Object.fromEntries(names.map((name, i) => [name, get(weights[i])]))
            tensors["model.decoder.rotary_emb." + type + "_inv_freq"] = get(inverseFrequencies)
            const result = prefixKeys === undefined || prefixValues === undefined
              ? yield* DG.prefillAttention(config, tensors, layer, get(hidden), get(positionIds))
              : yield* DG.readAttention(config, tensors, layer, get(hidden), get(positionIds), {
                keys: get(prefixKeys),
                values: get(prefixValues)
              })
            const projected = yield* DG.attentionProjections(
              config,
              tensors,
              layer,
              get(hidden),
              get(positionIds)
            )
            // Independent eager-stage diagnostics borrow official projected Q/K/V.
            const [batch, heads, sequence, d] = get(q).shape
            const [, kv, length] = get(k).shape
            const repeat = (value: Tensor.Any) =>
              Effect.gen(function*() {
                return yield* Tensor.reshape(
                  yield* Tensor.broadcastTo(yield* Tensor.reshape(value, [batch, kv, 1, length, d]), [
                    batch,
                    kv,
                    heads / kv,
                    length,
                    d
                  ]),
                  [batch, heads, length, d]
                )
              })
            let scores = yield* Tensor.matmul(get(q), yield* Tensor.transpose(yield* repeat(get(k)), [0, 1, 3, 2]))
            if (mask !== undefined) scores = yield* Tensor.add(scores, get(mask))
            const probabilities = yield* Tensor.cast(
              yield* Tensor.softmax(yield* Tensor.cast(scores, "f32")),
              get(q).dtype
            )
            const context = yield* Tensor.transpose(yield* Tensor.matmul(probabilities, yield* repeat(get(v))), [
              0,
              2,
              1,
              3
            ])
            const oracleOutput = yield* Tensor.linearRows(
              yield* Tensor.reshape(get(oracleContext), [batch, sequence, heads * d]),
              tensors[prefix + ".o_proj.weight"]
            )
            return [
              result.output,
              result.keys,
              result.values,
              projected.queries,
              projected.keys,
              projected.values,
              probabilities,
              context,
              oracleOutput
            ]
          })
      })
    }
  }
  const captured = Object.values(fixture.calls).filter((call) =>
    call.implementation === "DiffusionGemmaEncoderTextAttention" ||
    call.implementation === "DiffusionGemmaDecoderTextAttention"
  )
  if (captured.length !== checks.length) throw new Error("attention call coverage differs")
  return { get, checks }
}

const graph = (value: typeof FixtureTensor.Type) =>
  Effect.gen(function*() {
    if (value.dtype === "I64") {
      const integers = value.values.map((value) => {
        const parsed = Schema.decodeUnknownSync(Schema.Union([
          Schema.Int.check(Schema.isBetween({ minimum: Number.MIN_SAFE_INTEGER, maximum: Number.MAX_SAFE_INTEGER })),
          Schema.String.check(Schema.isPattern(/^-?[0-9]+$/))
        ]))(value)
        const integer = BigInt(parsed)
        if (integer < -(2n ** 63n) || integer >= 2n ** 63n) throw new Error("I64 fixture value out of range")
        return integer
      })
      return yield* Tensor.fromTypedArray(BigInt64Array.from(integers), value.shape)
    }
    const data = float(value)
    return yield* Tensor.cast(
      yield* Tensor.fromTypedArray(Float32Array.from(data.values), data.shape),
      data.dtype === "BF16" ? "bf16" : "f32"
    )
  })
const compare = (actual: ReadonlyArray<number>, expected: FloatTensor) => {
  if (actual.length !== expected.values.length) throw new Error("output element count differs")
  let exact = 0
  let maxAbsoluteError = 0
  let maxToleranceRatio = 0
  let failedElements = 0
  const firstFailures: Array<Failure> = []
  for (let i = 0; i < actual.length; i++) {
    const value = expected.values[i]
    if (!Number.isFinite(actual[i]) || !Number.isFinite(value)) throw new Error("nonfinite output/oracle at " + i)
    const absoluteError = Math.abs(actual[i] - value)
    const tolerance = expected.dtype === "BF16" ? bf16Ulp(value) : 2e-6 + 2e-5 * Math.abs(value)
    if (actual[i] === value) exact++
    maxAbsoluteError = Math.max(maxAbsoluteError, absoluteError)
    maxToleranceRatio = Math.max(maxToleranceRatio, absoluteError / tolerance)
    if (absoluteError > tolerance) {
      failedElements++
      if (firstFailures.length < 8) {
        firstFailures.push({ index: i, actual: actual[i], expected: value, absoluteError, tolerance })
      }
    }
  }
  return { elements: actual.length, exact, maxAbsoluteError, maxToleranceRatio, failedElements, firstFailures }
}

const main = Effect.gen(function*() {
  const [device, directory, reportPath, extra] = process.argv.slice(2)
  if (directory === undefined || reportPath === undefined || extra !== undefined) {
    throw new Error("usage: verify-attention.ts <cpu|metal|cuda> <numerical-fixtures-directory> <output-report.json>")
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
    for (const [file, dtype] of [["tiny-f32.json", "F32"], ["tiny-bf16.json", "BF16"]] as const) {
      const fixture = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Fixture))(
        readFileSync(join(directory, file), "utf8")
      )
      if (fixture.dtype !== dtype) throw new Error(file + ": dtype differs")
      const config = yield* DG.parseConfig(fixture.config)
      const { get, checks } = prepare(fixture, config)
      const names = Array.from(new Set(checks.flatMap((check) => check.inputs)))
      const externalMemoryBytesBefore = yield* runtime.extensions.diagnostics.externalMemoryBytes
      const roots = yield* Effect.forEach(names, (name) => graph(get(name)))
      yield* Effect.acquireUseRelease(Tensor.compute(roots), (owners) =>
        Effect.gen(function*() {
          if (owners.length !== names.length) throw new Error("input materialization arity differs")
          const byName = new Map(names.map((name, i) => [name, owners[i]]))
          for (const optimized of [false, true]) {
            for (const check of checks) {
              const bindings = check.inputs.map((name) => {
                const tensor = byName.get(name)
                if (tensor === undefined) throw new Error("missing owner " + name)
                return tensor
              })
              const inputs = yield* Effect.forEach(bindings, (value, slot) => Tensor.makeInput(slot, value))
              const byInput = new Map(check.inputs.map((name, i) => [name, inputs[i]]))
              const outputs = yield* check.build((name) => {
                const tensor = byInput.get(name)
                if (tensor === undefined) throw new Error("missing binding " + name)
                return tensor
              })
              const program = yield* Tensor.freezeProgram(outputs, { optimize: optimized, constantWeights: false })
              yield* Effect.acquireUseRelease(Tensor.runProgram(program, bindings), (actual) =>
                Effect.gen(function*() {
                  if (actual.length !== check.comparisons.length) throw new Error(check.call + ": result arity differs")
                  for (let i = 0; i < actual.length; i++) {
                    const { part, expected } = check.comparisons[i]
                    if (
                      actual[i].dtype !== (expected.dtype === "BF16" ? "bf16" : "f32") ||
                      !sameShape(actual[i].shape, expected.shape)
                    ) throw new Error(check.call + " " + part + ": output dtype/rank/shape differs")
                    results.push({
                      dtype,
                      optimized,
                      call: check.call,
                      part,
                      ...compare(yield* Tensor.toNumberArray(actual[i]), expected)
                    })
                  }
                }), Tensor.clearAll)
            }
          }
        }), Tensor.clearAll)
      diagnostics.push({
        file,
        uniqueInputTensors: names.length,
        externalMemoryBytesBefore,
        externalMemoryBytesAfter: yield* runtime.extensions.diagnostics.externalMemoryBytes
      })
    }
    return {
      device,
      gate: "official tiny DiffusionGemma attention",
      attentionCallsPerDtype: 4,
      optimizationModes: [false, true],
      tolerance: { f32: { atol: 2e-6, rtol: 2e-5 }, bf16: "one output ULP, including composed outputs" },
      diagnostics,
      results
    }
  }).pipe(Effect.provide(backend))
  mkdirSync(dirname(reportPath), { recursive: true })
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n")
  const failures = report.results.filter((result) => result.failedElements > 0)
  if (failures.length > 0) {
    throw new Error(
      device + ": " + failures.length + " attention comparisons failed; " + JSON.stringify(failures[0]) + "; report " +
        reportPath
    )
  }
  console.log(device + ": " + report.results.length + " official attention comparisons passed; report " + reportPath)
})
NodeRuntime.runMain(main)

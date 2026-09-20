/** Compare recorded official routers, experts, and feed-forward residuals. */
import * as BackendAppleNative from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import * as BackendCuda from "@effect-torch/backend-cuda"
import { type Model, Runtime, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/core/models"
import { NodeRuntime } from "@effect/platform-node"
import { Cause, Effect, Schema } from "effect"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"

const Reference = Schema.Struct({ tensor: Schema.String })
const FixtureTensor = Schema.Struct({
  dtype: Schema.Literals(["F32", "BF16", "I64", "BOOL"]),
  shape: Schema.Array(Schema.Int.check(Schema.isBetween({ minimum: 0, maximum: Number.MAX_SAFE_INTEGER }))),
  values: Schema.Array(Schema.Union([Schema.Finite, Schema.String]))
})
const Call = Schema.Struct({
  implementation: Schema.String,
  target: Schema.String,
  args: Schema.Array(Schema.Json),
  output: Schema.Json
})
const Fixture = Schema.Struct({
  schema_version: Schema.Literal(1),
  dtype: Schema.Literals(["F32", "BF16"]),
  config: Schema.Json,
  tensors: Schema.Record(Schema.String, FixtureTensor),
  weights: Schema.Record(Schema.String, Reference),
  calls: Schema.Record(Schema.String, Call)
})
interface Data {
  readonly dtype: "F32" | "BF16" | "U32"
  readonly shape: ReadonlyArray<number>
  readonly values: ReadonlyArray<number>
}
interface Check {
  readonly call: string
  readonly kind: "router" | "experts" | "feedForward"
  readonly inputs: ReadonlyArray<string>
  readonly expected: ReadonlyArray<{ readonly name: string; readonly data: Data }>
  readonly build: (inputs: ReadonlyArray<Tensor.Lazy>) => Effect.Effect<
    ReadonlyArray<Tensor.Lazy>,
    Model.ModelError | Tensor.TensorError,
    Runtime.Runtime
  >
}
interface Failure {
  readonly index: number
  readonly actual: number
  readonly expected: number
  readonly absoluteError: number
  readonly tolerance: number
}
interface Result {
  readonly fixture: string
  readonly optimized: boolean
  readonly call: string
  readonly output: string
  readonly dtype: Tensor.DType
  readonly elements: number
  readonly exact: number
  readonly maxAbsoluteError: number
  readonly maxToleranceRatio: number
  readonly failedElements: number
  readonly firstFailures: ReadonlyArray<Failure>
}

const sameShape = (a: ReadonlyArray<number>, b: ReadonlyArray<number>) =>
  a.length === b.length && a.every((value, i) => value === b[i])
const bf16Ulp = (value: number) => Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(value))) - 7))
const dtypeOf = (value: Data): "f32" | "bf16" | "u32" =>
  value.dtype === "BF16" ? "bf16" : value.dtype === "U32" ? "u32" : "f32"

const prepare = (fixture: typeof Fixture.Type, config: DiffusionGemma.Config) => {
  const get = (name: string): Data => {
    const tensor = fixture.tensors[name]
    if (tensor === undefined) throw new Error("missing tensor " + name)
    const size = tensor.shape.reduce((n, dim) => n * dim, 1)
    if (!Number.isSafeInteger(size) || size !== tensor.values.length) throw new Error(name + ": shape/size mismatch")
    if (tensor.dtype === "BOOL") throw new Error(name + ": expected floating values or expert indices")
    const values = tensor.dtype === "I64" ?
      tensor.values.map((value) => {
        const integer = Schema.decodeUnknownSync(Schema.Union([
          Schema.Int.check(Schema.isBetween({ minimum: 0, maximum: 0xffffffff })),
          Schema.String.check(Schema.isPattern(/^[0-9]+$/))
        ]))(value)
        const exact = BigInt(integer)
        if (exact > 0xffffffffn) throw new Error(name + ": expert index does not fit U32")
        return Number(exact)
      }) :
      Schema.decodeUnknownSync(Schema.Array(Schema.Finite))(tensor.values)
    return { dtype: tensor.dtype === "I64" ? "U32" : tensor.dtype, shape: tensor.shape, values }
  }
  const ref = (value: Schema.Json): string => {
    const name = Schema.decodeUnknownSync(Reference)(value).tensor
    get(name)
    return name
  }
  const weight = (name: string): string => {
    const value = fixture.weights[name]
    if (value === undefined) throw new Error("missing weight " + name)
    return ref(value)
  }
  const layerOf = (target: string) => {
    const match = /^model[.](decoder|encoder[.]language_model)[.]layers[.]([0-9]+)(?:[.](router|experts))?$/.exec(
      target
    )
    if (match === null) throw new Error("unexpected layer target " + target)
    return { layer: Number(match[2]), mode: match[1] === "decoder" ? "decoder" as const : "encoder" as const }
  }
  const arity = (name: string, call: typeof Call.Type, expected: number) => {
    if (call.args.length !== expected) throw new Error(name + ": unexpected argument count")
  }
  const parameters = (layer: number, mode: "encoder" | "decoder", suffixes: ReadonlyArray<string>) => {
    const prefix = "model.decoder.layers." + layer
    const names = suffixes.map((suffix) => prefix + "." + suffix)
    // Encoder projections/norms are tied to canonical decoder weights. Scalars
    // are deliberately excluded from this alias check and supplied separately.
    if (mode === "encoder") {
      for (const name of names) {
        const alias = name.replace("model.decoder", "model.encoder.language_model")
        if (weight(alias) !== weight(name)) throw new Error("encoder weight is not the captured tied alias: " + alias)
      }
    }
    return names
  }
  const routingParts = ["router.scale", "router.proj.weight", "router.per_expert_scale"]
  const expertParts = ["experts.gate_up_proj", "experts.down_proj"]
  const ffParts = [
    ...routingParts,
    ...expertParts,
    ...["gate_proj", "up_proj", "down_proj"].map((name) => "mlp." + name + ".weight"),
    ...[
      "pre_feedforward_layernorm",
      "pre_feedforward_layernorm_2",
      "post_feedforward_layernorm_1",
      "post_feedforward_layernorm_2",
      "post_feedforward_layernorm"
    ].map((name) => name + ".weight")
  ]
  const bind = (names: ReadonlyArray<string>, values: ReadonlyArray<Tensor.Lazy>) =>
    Object.fromEntries(names.map((name, index) => [name, values[index]]))
  const checks: Array<Check> = []
  const routingDiagnostics = []
  for (const [name, call] of Object.entries(fixture.calls)) {
    if (call.implementation === "DiffusionGemmaTextRouter") {
      arity(name, call, 1)
      const { layer, mode } = layerOf(call.target)
      const names = parameters(layer, mode, routingParts)
      const outputs = Schema.decodeUnknownSync(Schema.Array(Reference))(call.output)
      if (outputs.length !== 3) throw new Error(name + ": expected probabilities, weights, indices")
      const expected = outputs.map((value, i) => ({
        name: ["probabilities", "weights", "indices"][i],
        data: get(ref(value))
      }))
      if (expected[0].data.dtype !== "F32" || expected[1].data.dtype !== "F32" || expected[2].data.dtype !== "U32") {
        throw new Error(name + ": unexpected routing dtypes")
      }
      checks.push({
        call: name,
        kind: "router",
        inputs: [ref(call.args[0]), ...names.map(weight)],
        expected,
        build: ([input, ...weights]) =>
          DiffusionGemma.routeTokens(config, bind(names, weights), layer, input).pipe(
            Effect.map((routing) => [routing.probabilities, routing.weights, routing.indices])
          )
      })
      const projection = fixture.calls[name.replace(".router#", ".router.proj#")]
      if (projection?.implementation !== "Linear") throw new Error(name + ": missing captured router projection")
      const logits = get(ref(projection.output))
      const probabilities = expected[0].data
      const indices = expected[2].data
      const e = config.text_config.num_experts
      const k = config.text_config.top_k_experts
      if (!sameShape(logits.shape, probabilities.shape) || logits.shape[1] !== e || indices.shape[1] !== k) {
        throw new Error(name + ": invalid routing shapes")
      }
      const rows = Array.from({ length: logits.shape[0] }, (_, row) => {
        const values = logits.values.slice(row * e, (row + 1) * e)
        const selected = indices.values.slice(row * k, (row + 1) * k)
        if (selected.some((index) => index >= e)) throw new Error(name + ": captured expert index out of range")
        const sorted = values.map((value, index) => ({ value, index })).sort((a, b) =>
          b.value - a.value || a.index - b.index
        )
        return {
          row,
          selectedIndices: selected,
          selectedLogits: selected.map((index) => values[index]),
          minLogit: Math.min(...values),
          maxLogit: Math.max(...values),
          minSelectedGap: Math.min(...sorted.slice(0, k - 1).map((value, i) => value.value - sorted[i + 1].value)),
          boundaryLogitGap: k < e ? sorted[k - 1].value - sorted[k].value : null,
          boundaryProbabilityGap: k < e
            ? probabilities.values[row * e + sorted[k - 1].index] - probabilities.values[row * e + sorted[k].index]
            : null
        }
      })
      routingDiagnostics.push({ call: name, rows })
    } else if (call.implementation === "DiffusionGemmaTextExperts") {
      arity(name, call, 3)
      const { layer, mode } = layerOf(call.target)
      const names = parameters(layer, mode, expertParts)
      const inputNames = call.args.map(ref)
      const [input, indices, weights] = inputNames.map(get)
      if (
        input.shape.length !== 2 || indices.shape.length !== 2 || indices.dtype !== "U32" || weights.dtype !== "F32" ||
        !sameShape(indices.shape, weights.shape) || indices.shape[0] !== input.shape[0]
      ) {
        throw new Error(name + ": invalid recorded expert arguments")
      }
      checks.push({
        call: name,
        kind: "experts",
        inputs: [...inputNames, ...names.map(weight)],
        expected: [{ name: "output", data: get(ref(call.output)) }],
        build: ([x, ids, scales, ...weights]) =>
          Effect.gen(function*() {
            // The all-expert probe calls the official expert module with k=1.
            const probeConfig = yield* DiffusionGemma.parseConfig({
              ...config,
              text_config: { ...config.text_config, top_k_experts: indices.shape[1] }
            })
            return [yield* DiffusionGemma.routedExperts(probeConfig, bind(names, weights), layer, x, ids, scales)]
          })
      })
    } else if (
      call.implementation === "DiffusionGemmaEncoderTextLayer" ||
      call.implementation === "DiffusionGemmaDecoderTextLayer"
    ) {
      const { layer, mode } = layerOf(call.target)
      const router = fixture.calls[name.replace("#", ".router#")]
      if (router?.implementation !== "DiffusionGemmaTextRouter") {
        throw new Error(name + ": missing raw post-attention router input")
      }
      arity(name, router, 1)
      const expected = get(ref(call.output))
      const residual = get(ref(router.args[0]))
      if (
        residual.shape.length !== 2 || expected.shape.length !== 3 ||
        !sameShape(residual.shape, [expected.shape[0] * expected.shape[1], expected.shape[2]])
      ) {
        throw new Error(name + ": router residual cannot reshape to layer output")
      }
      const scalar = call.target + ".layer_scalar"
      const names = [...parameters(layer, mode, ffParts), scalar]
      checks.push({
        call: name + ".feedForward",
        kind: "feedForward",
        inputs: [ref(router.args[0]), ...names.map(weight)],
        expected: [{ name: "output", data: expected }],
        build: ([input, ...weights]) =>
          Effect.gen(function*() {
            const residual = yield* Tensor.reshape(input, expected.shape)
            return [yield* DiffusionGemma.feedForward(config, bind(names, weights), layer, residual, mode)]
          })
      })
    }
  }
  const counts = { router: 0, experts: 0, feedForward: 0 }
  for (const check of checks) counts[check.kind]++
  if (counts.router !== 4 || counts.experts !== 5 || counts.feedForward !== 4) {
    throw new Error("incomplete fixture call coverage: " + JSON.stringify(counts))
  }
  // Compare the lower-level boundaries before complete feed-forward results.
  checks.sort((a, b) =>
    ["router", "experts", "feedForward"].indexOf(a.kind) - ["router", "experts", "feedForward"].indexOf(b.kind)
  )
  return { get, checks, counts, routingDiagnostics }
}

const graph = (data: Data) =>
  Effect.gen(function*() {
    if (data.dtype === "U32") return yield* Tensor.fromTypedArray(Uint32Array.from(data.values), data.shape)
    const root = yield* Tensor.fromTypedArray(Float32Array.from(data.values), data.shape)
    return data.dtype === "BF16" ? yield* Tensor.cast(root, "bf16") : root
  })

const main = Effect.gen(function*() {
  const [device, directory, reportPath, extra] = process.argv.slice(2)
  if (device === "--help" && directory === undefined) {
    console.log("usage: verify-moe.ts <cpu|metal|cuda> <numerical-fixtures-directory> <output-report.json>")
    return
  }
  if (
    (device !== "cpu" && device !== "metal" && device !== "cuda") || directory === undefined ||
    reportPath === undefined || extra !== undefined
  ) {
    throw new Error("usage: verify-moe.ts <cpu|metal|cuda> <numerical-fixtures-directory> <output-report.json>")
  }
  if (device === "cuda" && !(yield* BackendCuda.isAvailable)) throw new Error("CUDA is required for this gate")
  if (device === "metal" && !(yield* BackendAppleNative.isAvailable)) throw new Error("Metal is required for this gate")
  const backend = device === "cpu"
    ? BackendCpu.layer
    : device === "metal"
    ? BackendAppleNative.layer()
    : BackendCuda.layer()
  const results: Array<Result> = []
  const diagnostics: Array<Schema.Json> = []
  let current = "fixture setup"
  let runtimeFailure: string | null = null
  yield* Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    for (const [file, dtype] of [["tiny-f32.json", "F32"], ["tiny-bf16.json", "BF16"]] as const) {
      current = file + ": fixture validation"
      const fixture = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Fixture))(
        readFileSync(join(directory, file), "utf8")
      )
      if (fixture.dtype !== dtype) throw new Error(file + ": fixture dtype mismatch")
      const config = yield* DiffusionGemma.parseConfig(fixture.config)
      const prepared = prepare(fixture, config)
      const names = Array.from(new Set(prepared.checks.flatMap((check) => check.inputs)))
      const before = yield* runtime.extensions.diagnostics.externalMemoryBytes
      diagnostics.push({
        fixture: file,
        counts: prepared.counts,
        uniqueInputTensors: names.length,
        externalMemoryBytesBefore: before,
        routing: prepared.routingDiagnostics
      })
      const roots = yield* Effect.forEach(names, (name) => graph(prepared.get(name)))
      yield* Effect.acquireUseRelease(Tensor.compute(roots), (owners) =>
        Effect.gen(function*() {
          const byName = new Map(names.map((name, i) => [name, owners[i]]))
          for (const optimized of [false, true]) {
            for (const check of prepared.checks) {
              current = file + " optimize=" + optimized + " " + check.call
              const bindings = check.inputs.map((name) => {
                const value = byName.get(name)
                if (value === undefined) throw new Error("missing fixture input " + name)
                return value
              })
              const inputs = yield* Effect.forEach(bindings, (input, slot) => Tensor.makeInput(slot, input))
              const outputs = yield* check.build(inputs)
              const program = yield* Tensor.freezeProgram(outputs, { optimize: optimized, constantWeights: false })
              yield* Effect.acquireUseRelease(Tensor.runProgram(program, bindings), (actual) =>
                Effect.gen(function*() {
                  if (actual.length !== check.expected.length) throw new Error(current + ": result count mismatch")
                  for (const [index, expected] of check.expected.entries()) {
                    const output = actual[index]
                    if (output.dtype !== dtypeOf(expected.data) || !sameShape(output.shape, expected.data.shape)) {
                      throw new Error(current + ": " + expected.name + " metadata mismatch")
                    }
                    const values = yield* Tensor.toNumberArray(output)
                    if (values.length !== expected.data.values.length) {
                      throw new Error(current + ": output element count mismatch")
                    }
                    let maxAbsoluteError = 0
                    let maxToleranceRatio = 0
                    let exact = 0
                    let failedElements = 0
                    const firstFailures: Array<Failure> = []
                    for (let i = 0; i < values.length; i++) {
                      const reference = expected.data.values[i]
                      if (!Number.isFinite(values[i])) throw new Error(current + ": nonfinite result at " + i)
                      const error = Math.abs(values[i] - reference)
                      const tolerance = output.dtype === "u32"
                        ? 0
                        : output.dtype === "bf16"
                        ? bf16Ulp(reference)
                        : 2e-6 + 2e-5 * Math.abs(reference)
                      maxAbsoluteError = Math.max(maxAbsoluteError, error)
                      if (tolerance > 0) maxToleranceRatio = Math.max(maxToleranceRatio, error / tolerance)
                      if (error === 0) exact++
                      if (error > tolerance) {
                        failedElements++
                        if (firstFailures.length < 8) {
                          firstFailures.push({
                            index: i,
                            actual: values[i],
                            expected: reference,
                            absoluteError: error,
                            tolerance
                          })
                        }
                      }
                    }
                    results.push({
                      fixture: file,
                      optimized,
                      call: check.call,
                      output: expected.name,
                      dtype: output.dtype,
                      elements: values.length,
                      exact,
                      maxAbsoluteError,
                      maxToleranceRatio,
                      failedElements,
                      firstFailures
                    })
                  }
                }), Tensor.clearAll)
            }
          }
        }), Tensor.clearAll)
      diagnostics.push({
        fixture: file,
        externalMemoryBytesAfter: yield* runtime.extensions.diagnostics.externalMemoryBytes
      })
    }
  }).pipe(
    Effect.provide(backend),
    Effect.catchCause((cause) =>
      Effect.sync(() => {
        runtimeFailure = current + ": " + Cause.pretty(cause)
      })
    )
  )
  const failures = results.filter((result) => result.failedElements > 0)
  const report = {
    device,
    gate: "official tiny DiffusionGemma MoE",
    optimizationModes: [false, true],
    expectedBuffersPerDtypeAndMode: { router: 12, experts: 5, feedForward: 4 },
    tolerance: { f32: { atol: 2e-6, rtol: 2e-5 }, bf16: "one output ULP", indices: "exact I64-to-U32 values" },
    tiePolicy:
      "Runtime ties use ascending expert indices. Upstream top-k tie ordering is unspecified; captured indices are compared exactly without remapping.",
    diagnostics,
    results,
    firstFailure: failures[0] ?? null,
    runtimeFailure
  }
  mkdirSync(dirname(reportPath), { recursive: true })
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n")
  if (runtimeFailure !== null || failures.length > 0) {
    throw new Error(
      "MoE verification failed: " + (runtimeFailure ?? JSON.stringify(failures[0])) + "; report " + reportPath
    )
  }
  console.log(device + ": " + results.length + " official MoE buffer comparisons passed; report " + reportPath)
})

NodeRuntime.runMain(main)

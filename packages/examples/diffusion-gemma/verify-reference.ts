/** Compare recorded official Transformers calls on an explicitly selected backend. */
import * as BackendAppleNative from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Tensor } from "@effect-torch/core"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"

const Reference = Schema.Struct({ tensor: Schema.String })
const FixtureTensor = Schema.Struct({
  dtype: Schema.Literals(["F32", "BF16", "I64", "BOOL"]),
  shape: Schema.Array(Schema.Number),
  values: Schema.Array(Schema.Union([Schema.Number, Schema.String]))
})
const Fixture = Schema.Struct({
  dtype: Schema.String,
  tensors: Schema.Record(Schema.String, FixtureTensor),
  weights: Schema.Record(Schema.String, Reference),
  calls: Schema.Record(
    Schema.String,
    Schema.Struct({
      implementation: Schema.String,
      target: Schema.String,
      args: Schema.Array(Schema.Json),
      output: Schema.Json
    })
  )
})

const main = Effect.gen(function*() {
  const [device, directory, reportPath] = process.argv.slice(2)
  if (directory === undefined || reportPath === undefined) {
    throw new Error("usage: verify-reference.ts <cpu|metal|cuda> <numerical-fixtures> <report.json>")
  }
  if (device !== "cpu" && device !== "metal" && device !== "cuda") throw new Error("select cpu, metal or cuda")
  if (device === "cuda" && !(yield* BackendCuda.isAvailable)) throw new Error("CUDA is required for this gate")
  const backend = device === "cpu"
    ? BackendCpu.layer
    : device === "metal"
    ? BackendAppleNative.layer()
    : BackendCuda.layer()
  const report = yield* Effect.gen(function*() {
    const results: Array<{ dtype: string; call: string; elements: number; maxAbsoluteError: number; exact: number }> =
      []
    for (const file of ["tiny-f32.json", "tiny-bf16.json"]) {
      const fixture = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Fixture))(
        readFileSync(join(directory, file), "utf8")
      )
      const tensor = (reference: Schema.Json) => {
        const name = Schema.decodeUnknownSync(Reference)(reference).tensor
        const value = fixture.tensors[name]
        if (value === undefined) throw new Error(`missing tensor ${name}`)
        return value
      }
      const graph = (value: typeof FixtureTensor.Type) =>
        Effect.gen(function*() {
          if (value.dtype !== "F32" && value.dtype !== "BF16") throw new Error(`unexpected linear dtype ${value.dtype}`)
          const dense = yield* Tensor.fromTypedArray(Float32Array.from(value.values, Number), value.shape)
          return value.dtype === "BF16" ? yield* Tensor.cast(dense, "bf16") : dense
        })
      let count = 0
      for (const [name, call] of Object.entries(fixture.calls)) {
        if (call.implementation !== "Linear") continue
        const input = tensor(call.args[0])
        const reference = tensor(call.output)
        const weightRef = fixture.weights[`${call.target}.weight`]
        if (weightRef === undefined) throw new Error(`${name}: missing weight`)
        const weight = tensor(weightRef)
        yield* Effect.acquireUseRelease(
          Tensor.compute([yield* graph(input), yield* graph(weight)]),
          ([x, w]) =>
            Effect.gen(function*() {
              const root = yield* Tensor.linearRows(yield* Tensor.makeInput(0, x), yield* Tensor.makeInput(1, w))
              const program = yield* Tensor.freezeProgram([root])
              yield* Effect.acquireUseRelease(
                Tensor.runProgram(program, [x, w]),
                ([actual]) =>
                  Effect.gen(function*() {
                    const values = yield* Tensor.toNumberArray(actual)
                    const dtype = reference.dtype === "BF16" ? "bf16" : "f32"
                    if (
                      actual.dtype !== dtype || actual.shape.length !== reference.shape.length ||
                      actual.shape.some((n, i) => n !== reference.shape[i])
                    ) {
                      throw new Error(`${name}: output metadata differs from the oracle`)
                    }
                    if (values.length !== reference.values.length) throw new Error(`${name}: output length differs`)
                    let maxAbsoluteError = 0
                    let exact = 0
                    for (let i = 0; i < values.length; i++) {
                      const expected = Number(reference.values[i])
                      if (!Number.isFinite(expected)) throw new Error(`${name}[${i}]: nonfinite oracle value`)
                      const error = Math.abs(values[i] - expected)
                      maxAbsoluteError = Math.max(maxAbsoluteError, error)
                      if (values[i] === expected) exact++
                      // BF16 allows one output ULP; F32 allows different FMA/reduction orders.
                      const tolerance = dtype === "bf16"
                        ? Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(expected)))) / 128)
                        : 2e-6 + 2e-5 * Math.abs(expected)
                      if (!Number.isFinite(values[i]) || error > tolerance) {
                        throw new Error(`${file} ${name}[${i}]: ${values[i]} != ${expected}; tolerance ${tolerance}`)
                      }
                    }
                    results.push({ dtype, call: name, elements: values.length, maxAbsoluteError, exact })
                  }),
                Tensor.clearAll
              )
            }),
          Tensor.clearAll
        )
        count++
      }
      if (count !== 34) throw new Error(`${file}: expected 34 official linear calls, found ${count}`)
    }
    return { device, gate: "official tiny F32/BF16 linear calls", results }
  }).pipe(Effect.provide(backend))
  mkdirSync(dirname(reportPath), { recursive: true })
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n")
  console.log(`${device}: ${report.results.length} official linear calls passed`)
})

NodeRuntime.runMain(main)

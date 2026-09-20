/**
 * Mandatory CUDA checkpoint/load gate. Fails when CUDA is unavailable.
 *
 * pnpm --filter @effect-torch/examples exec tsx diffusion-gemma/load-cuda.ts <checkpoint-directory> <report.json>
 */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Runtime, Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/core/models"
import { NodeRuntime } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { execFileSync } from "node:child_process"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"

const memory = () => ({
  gpu: execFileSync("nvidia-smi", [
    "--query-gpu=uuid,name,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits"
  ], { encoding: "utf8" }).trim(),
  host: process.memoryUsage(),
  hostMaxRssKiB: process.resourceUsage().maxRSS
})

const main = Effect.gen(function*() {
  if (!(yield* BackendCuda.isAvailable)) throw new Error("CUDA is required for this gate")
  const [directory, reportPath] = process.argv.slice(2)
  if (directory === undefined || reportPath === undefined) {
    throw new Error("usage: load-cuda.ts <checkpoint-directory> <report.json>")
  }
  const config = yield* Schema.decodeUnknownEffect(Schema.fromJsonString(Schema.Json))(
    readFileSync(join(directory, "config.json"), "utf8")
  )
  const before = memory()
  const runtime = yield* Runtime.Runtime
  const started = performance.now()
  const report = yield* Effect.acquireUseRelease(
    DiffusionGemma.loadParameters(join(directory, "model.safetensors.index.json"), config),
    (loaded) =>
      Effect.gen(function*() {
        const loadMs = performance.now() - started
        const loadedMemory = memory()
        const bytes = loaded.ownedParameters.reduce((total, tensor) => {
          if (tensor.dtype !== "bf16") throw new Error("pinned text checkpoint must preserve BF16")
          return total + tensor.shape.reduce((size, dimension) => size * dimension, 2)
        }, 0)
        if (bytes !== 50501973624 || loaded.ownedParameters.length !== 691) {
          throw new Error(`pinned text catalog mismatch: ${bytes} bytes, ${loaded.ownedParameters.length} tensors`)
        }
        for (const [alias, canonical] of Object.entries(loaded.aliases)) {
          if (loaded.tensors[alias] !== loaded.tensors[canonical]) {
            throw new Error(`${alias}: duplicated storage handle`)
          }
        }
        const linears: Array<{
          name: string
          rows: number
          outputShape: ReadonlyArray<number>
          diagnostics: Runtime.ExecutableDiagnostics
        }> = []
        // Real local/global query, shared MLP and self-conditioning weights.
        // One-hot rows must reproduce exact BF16 weight columns.
        for (
          const name of [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.5.self_attn.q_proj.weight",
            "model.decoder.layers.0.mlp.gate_proj.weight",
            "model.decoder.self_conditioning.up_proj.weight"
          ]
        ) {
          const weight = loaded.tensors[name]
          if (weight === undefined) throw new Error(`missing probe weight ${name}`)
          const width = weight.shape[1]
          for (const rows of [1, 64]) {
            const columns = Uint32Array.from({ length: rows }, (_, row) => (row * 137) % width)
            const values = new Float32Array(rows * width)
            for (let row = 0; row < rows; row++) values[row * width + columns[row]] = 1
            const inputGraph = yield* Tensor.cast(yield* Tensor.fromTypedArray(values, [rows, width]), "bf16")
            yield* Effect.acquireUseRelease(
              Tensor.compute([inputGraph]),
              ([input]) =>
                Effect.gen(function*() {
                  const x = yield* Tensor.makeInput(0, input)
                  const w = yield* Tensor.makeInput(1, weight)
                  const output = yield* Tensor.linearRows(x, w)
                  const program = yield* Tensor.freezeProgram([output])
                  const diagnostics = program.handle.diagnostics
                  if (!diagnostics.instructions.some((entry) => entry.kind === "cublas_bf16_gemm_f32_accum")) {
                    throw new Error(`${name}: BF16 cuBLAS was not selected`)
                  }
                  if (diagnostics.legalization?.materializedConversionBytes !== 0) {
                    throw new Error(`${name}: unexpected materialized dtype conversions`)
                  }
                  const indices = yield* Tensor.fromTypedArray(columns, [rows])
                  const expected = yield* Tensor.transpose(yield* Tensor.take(weight, indices, { dim: 1 }), [1, 0])
                  yield* Effect.acquireUseRelease(
                    Tensor.runProgram(program, [input, weight]),
                    ([actual]) =>
                      Effect.acquireUseRelease(
                        Tensor.compute([expected]),
                        ([reference]) =>
                          Effect.gen(function*() {
                            const values = yield* Tensor.toNumberArray(actual)
                            const expectedValues = yield* Tensor.toNumberArray(reference)
                            if (
                              values.length !== expectedValues.length ||
                              values.some((value, index) => !Number.isFinite(value) || value !== expectedValues[index])
                            ) throw new Error(`${name}: one-hot GEMM disagrees with stored BF16 columns`)
                            linears.push({ name, rows, outputShape: actual.shape, diagnostics })
                          }),
                        Tensor.clearAll
                      ),
                    Tensor.clearAll
                  )
                }),
              Tensor.clearAll
            )
          }
        }
        return {
          gate: "DiffusionGemma M1 selected BF16 load and row linears",
          modelRevision: "f7f5b7f5fa82ffc52addd066915886d497f5517b",
          pid: process.pid,
          loadMs,
          selectedTensorBytes: bytes,
          selectedTensorCount: loaded.ownedParameters.length,
          aliases: Object.keys(loaded.aliases).length,
          before,
          loaded: loadedMemory,
          afterLinears: memory(),
          linears
        }
      }),
    (loaded) =>
      Effect.gen(function*() {
        yield* Tensor.clearAll(loaded.ownedParameters)
        for (const tensor of loaded.ownedParameters) {
          yield* runtime.readback(tensor).pipe(Effect.match({
            onFailure: (error) => {
              if (error.reason !== "invalid-handle") throw new Error(`unexpected release error: ${error.message}`)
            },
            onSuccess: () => {
              throw new Error("released parameter remained readable")
            }
          }))
        }
      })
  )
  // CUDA frees are stream-ordered. Complete a small subsequent invocation
  // before observing driver memory after all parameter releases.
  yield* Effect.acquireUseRelease(
    Tensor.compute([yield* Tensor.zeros([1])]),
    ([barrier]) => Tensor.toNumberArray(barrier),
    Tensor.clearAll
  )
  const result = { ...report, afterRelease: memory() }
  mkdirSync(dirname(reportPath), { recursive: true })
  writeFileSync(reportPath, JSON.stringify(result, null, 2) + "\n")
  console.log(JSON.stringify(result, null, 2))
}).pipe(Effect.provide(BackendCuda.layer()))

NodeRuntime.runMain(main)

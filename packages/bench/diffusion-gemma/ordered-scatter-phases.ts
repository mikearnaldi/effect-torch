/** Exact ordered-scatter component benchmark; timings exclude uploads and readback. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import assert from "node:assert/strict"
import { performance } from "node:perf_hooks"

const rows = 256
const routes = 8
const width = 2816
const shape = [rows, routes, width]
const values = Float32Array.from({ length: rows * routes * width }, (_, index) => {
  const route = Math.floor(index / width) % routes
  const feature = index % width
  return [256, 1, -256, 0.5, -0.5, 0.00390625, -0.00390625, 1][route]! * (1 + feature % 17 / 128)
})
const ranks = Uint32Array.from({ length: rows * routes }, (_, index) => {
  const row = Math.floor(index / routes)
  const route = index % routes
  return row % 13 === 0 ? Math.floor(route / 2) : (route * 3 + row) % routes
})

const build = (weighted: Tensor.Any, compactRanks: Tensor.Any) =>
  Effect.gen(function*() {
    const indexes = yield* Tensor.broadcastTo(yield* Tensor.unsqueeze(compactRanks, -1), shape)
    const sorted = yield* Tensor.scatterAdd(
      yield* Tensor.zeros(shape, { dtype: "bf16" }),
      indexes,
      weighted,
      { dim: 1 }
    )
    let output: Tensor.Any = yield* Tensor.zeros([rows, width], { dtype: "bf16" })
    for (let route = 0; route < routes; route++) {
      const selected = yield* Tensor.slice(sorted, {
        start: [0, route, 0],
        end: [rows, route + 1, width]
      })
      output = yield* Tensor.add(output, yield* Tensor.reshape(selected, [rows, width]))
    }
    return [output]
  })

const flag = "EFFECT_TORCH_CUDA_ORDERED_SCATTER_FUSION"
const previous = process.env[flag]
try {
  await Effect.runPromise(Effect.provide(
    Effect.scoped(Effect.gen(function*() {
      const inputs = yield* Effect.acquireRelease(
        Tensor.compute([
          yield* Tensor.cast(yield* Tensor.fromTypedArray(values, shape), "bf16"),
          yield* Tensor.fromTypedArray(ranks, [rows, routes])
        ]),
        Tensor.clearAll,
        { interruptible: true }
      )
      const placeholders = [yield* Tensor.makeInput(0, inputs[0]!), yield* Tensor.makeInput(1, inputs[1]!)]
      const roots = yield* build(placeholders[0]!, placeholders[1]!)
      let reference: Uint8Array | undefined
      for (const mode of ["materialized", "original", "fused"] as const) {
        process.env[flag] = mode === "fused" ? "1" : "0"
        const program = yield* Tensor.freezeProgram(roots, { optimize: mode !== "materialized" })
        const diagnostics = program.handle.diagnostics
        const selected = diagnostics.instructions.some((instruction) =>
          instruction.kind === "et_ordered_scatter_reduce"
        )
        assert.equal(selected, mode === "fused", "benchmark must select the requested implementation")
        process.stdout.write(JSON.stringify({ mode, shape, diagnostics }) + "\n")
        const durations: Array<number> = []
        const count = mode === "materialized" ? 1 : 13
        for (let run = 0; run < count; run++) {
          yield* Effect.scoped(Effect.gen(function*() {
            const started = performance.now()
            const output = yield* Effect.acquireRelease(
              Tensor.runProgram(program, inputs),
              Tensor.clearAll,
              { interruptible: true }
            )
            // CUDA invocation completion precedes output publication. Readback
            // and exact comparison below deliberately occur after the timer.
            const milliseconds = performance.now() - started
            const actual = yield* Tensor.toTypedArray(output[0]!)
            const bytes = new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength)
            if (reference === undefined) reference = bytes.slice()
            else assert.deepEqual(bytes, reference, `${mode} run ${run} must match materialized BF16 bytes`)
            if (mode !== "materialized" && run >= 3) {
              durations.push(milliseconds)
              process.stdout.write(JSON.stringify({ mode, run: run - 3, milliseconds, exact: true, shape }) + "\n")
            }
          }))
        }
        if (durations.length > 0) {
          const sorted = durations.slice().sort((a, b) => a - b)
          const medianMs = (sorted[4]! + sorted[5]!) / 2
          process.stdout.write(JSON.stringify({ mode, medianMs, runs: durations.length, exact: true, shape }) + "\n")
        }
      }
    })),
    BackendCuda.layer()
  ))
} finally {
  if (previous === undefined) delete process.env[flag]
  else process.env[flag] = previous
}

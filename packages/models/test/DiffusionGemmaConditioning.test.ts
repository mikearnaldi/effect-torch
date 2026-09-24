import { Tensor } from "@effect-torch/core"
import * as DG from "@effect-torch/models/DiffusionGemma"
import { expect } from "@effect/vitest"
import { Effect, type Schema } from "effect"
import { readFileSync } from "node:fs"
import { onDevices } from "./utils/devices.ts"

interface RecordedTensor {
  readonly dtype: "float32" | "bfloat16"
  readonly shape: ReadonlyArray<number>
  readonly values: ReadonlyArray<number>
}

interface Run {
  readonly dtype: "float32" | "bfloat16"
  readonly model_config: Schema.Json
  readonly conditioning_weights: Readonly<Record<string, RecordedTensor>>
  readonly blocks: ReadonlyArray<{
    readonly steps: ReadonlyArray<{
      readonly feedback_in: RecordedTensor | null
      readonly conditioning: {
        readonly embeddings: RecordedTensor
        readonly signal: RecordedTensor
        readonly output: RecordedTensor
      }
    }>
  }>
}

const fixture: { readonly runs: ReadonlyArray<Run> } = JSON.parse(
  readFileSync(new URL("./fixtures/diffusion-gemma-generation.json", import.meta.url), "utf8")
)

const graph = (value: RecordedTensor) =>
  Effect.gen(function*() {
    const tensor = yield* Tensor.fromTypedArray(Float32Array.from(value.values), value.shape)

    return value.dtype === "bfloat16" ? yield* Tensor.cast(tensor, "bf16") : tensor
  })

onDevices("DiffusionGemma carried self-conditioning", () => (it) => {
  for (const run of fixture.runs) {
    for (const optimize of [false, true]) {
      it.effect(
        run.dtype + " optimize=" + optimize + " matches pinned nonzero feedback in both blocks",
        () =>
          Effect.scoped(Effect.gen(function*() {
            const config = yield* DG.parseConfig(run.model_config)
            const entries = Object.entries(run.conditioning_weights)
            const roots = yield* Effect.forEach(entries, ([, value]) => graph(value))

            const weights = yield* Effect.acquireRelease(Tensor.compute(roots), Tensor.clearAll, {
              interruptible: true
            })

            const parameters = Object.fromEntries(
              entries.map(([name], index) => ["model.decoder." + name, weights[index]!])
            )

            let carried = 0

            for (const block of run.blocks) {
              for (const step of block.steps) {
                const embeddings = yield* graph(step.conditioning.embeddings)

                const output = step.feedback_in === null
                  ? yield* DG.initialSelfConditioning(config, embeddings)
                  : yield* DG.selfConditioning(config, parameters, embeddings, yield* graph(step.feedback_in))

                const [actual] = yield* Effect.acquireRelease(
                  Tensor.compute([output], { optimize }),
                  Tensor.clearAll,
                  { interruptible: true }
                )

                const values = yield* Tensor.toNumberArray(actual)
                const expected = step.conditioning.output.values
                expect(values).toHaveLength(expected.length)
                values.forEach((value, index) => {
                  const target = expected[index]!

                  const tolerance = run.dtype === "bfloat16"
                    ? Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(target))) - 7))
                    : 2e-6 + 2e-5 * Math.abs(target)

                  expect(Math.abs(value - target), "conditioning element " + index).toBeLessThanOrEqual(tolerance)
                })

                if (step.feedback_in !== null) {
                  expect(step.conditioning.signal.values.some((value) => value !== 0)).toBe(true)
                  carried++
                }
              }
            }

            expect(carried).toBe(4)
          }))
      )
    }
  }
})

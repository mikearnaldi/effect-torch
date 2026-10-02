import { Diffusion, Runtime, Tensor } from "@effect-torch/core"
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
  readonly weights: Readonly<Record<string, RecordedTensor>>
  readonly buffers: Readonly<Record<string, RecordedTensor>>
  readonly weight_aliases: Readonly<Record<string, string>>
  readonly buffer_aliases: Readonly<Record<string, string>>
  readonly prompt: ReadonlyArray<number>
  readonly sequences: ReadonlyArray<number>
  readonly canvas_length: number
  readonly vocab_size: number
  readonly encoder_calls: ReadonlyArray<{
    readonly tokens: ReadonlyArray<number>
    readonly prefix: ReadonlyArray<{
      readonly keys: RecordedTensor
      readonly values: RecordedTensor
    }>
  }>
  readonly blocks: ReadonlyArray<{
    readonly initial_canvas: ReadonlyArray<number>
    readonly steps: ReadonlyArray<{
      readonly canvas: ReadonlyArray<number>
      readonly random_canvas: ReadonlyArray<number>
      readonly exponentials: ReadonlyArray<number>
      readonly done: boolean
      readonly remaining: number
      readonly feedback_in: RecordedTensor | null
      readonly raw_logits: RecordedTensor
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

const close = (actual: ReadonlyArray<number>, expected: ReadonlyArray<number>, dtype: Run["dtype"], name: string) => {
  expect(actual, name + " length").toHaveLength(expected.length)
  actual.forEach((value, index) => {
    const target = expected[index]!

    const tolerance = dtype === "bfloat16"
      ? Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(target))) - 7))
      : 2e-5 + 2e-5 * Math.abs(target)

    expect(Math.abs(value - target), name + " element " + index).toBeLessThanOrEqual(tolerance)
  })
}

const tokensFirst = (tensor: RecordedTensor) => {
  const [, heads, tokens, width] = tensor.shape

  return Array.from({ length: tensor.values.length }, (_, index) => {
    const row = Math.floor(index / (heads! * width!))
    const head = Math.floor(index / width!) % heads!
    const column = index % width!

    return tensor.values[(head * tokens! + row) * width! + column]!
  })
}

onDevices("compiled DiffusionGemma reference generation", () => (it) => {
  for (const run of fixture.runs) {
    for (const optimize of [false, true]) {
      it.effect(
        run.dtype + " optimize=" + optimize + " replays both blocks and all carried predictions",
        () =>
          Effect.scoped(Effect.gen(function*() {
            const runtime = yield* Runtime.Runtime
            const catalog = yield* DG.parameterCatalog(run.model_config)
            const entries = Object.entries({ ...run.weights, ...run.buffers })
            const roots = yield* Effect.forEach(entries, ([, value]) => graph(value))
            const owned = yield* Effect.acquireRelease(Tensor.compute(roots), Tensor.clearAll, { interruptible: true })
            const tensors = Object.fromEntries(entries.map(([name], index) => [name, owned[index]!]))

            for (
              const [alias, canonical] of Object.entries({
                ...catalog.aliases,
                ...run.weight_aliases,
                ...run.buffer_aliases
              })
            ) {
              if (tensors[canonical] !== undefined) tensors[alias] = tensors[canonical]!
            }

            const model = DG.fromTensors({
              ...catalog,
              aliases: {
                ...catalog.aliases,
                ...run.weight_aliases,
                ...run.buffer_aliases
              }
            }, tensors)

            const program = yield* Diffusion.compile(model.definition, model.parameters, {
              maxTokens: 64,
              blockSize: 4,
              prefillChunks: [1, 3, run.prompt.length],
              canvasLengths: [run.canvas_length],
              selectedReadouts: [{ rows: 2, labels: 3 }],
              compile: { optimize }
            })

            yield* Tensor.clearAll(owned)
            const release = (prefix: Diffusion.Prefix) => Effect.orDie(program.release(prefix))

            let prefix = yield* Effect.acquireRelease(program.encode(Uint32Array.from(run.prompt)), release, {
              interruptible: true
            })

            for (const [blockIndex, block] of run.blocks.entries()) {
              const before = yield* program.inspect(prefix)
              const expectedPrefix = run.encoder_calls[blockIndex]!.prefix
              expect(before.layers).toHaveLength(expectedPrefix.length)

              for (const layer of before.layers) {
                const reference = expectedPrefix[layer.layerId]!
                close(layer.keys, tokensFirst(reference.keys), run.dtype, "prefix keys")
                close(layer.values, tokensFirst(reference.values), run.dtype, "prefix values")
              }

              for (const [stepIndex, step] of block.steps.entries()) {
                yield* Effect.scoped(Effect.gen(function*() {
                  const prediction: Diffusion.Prediction = step.feedback_in === null
                    ? { _tag: "Initial" }
                    : { _tag: "Refinement", logits: yield* graph(step.feedback_in) }

                  const logits = yield* Effect.acquireRelease(
                    program.evaluate(prefix, Uint32Array.from(step.canvas), prediction),
                    Tensor.clear,
                    { interruptible: true }
                  )

                  close(yield* Tensor.toNumberArray(logits), step.raw_logits.values, run.dtype, "decoder logits")

                  if (stepIndex === 0) {
                    const rows = [run.canvas_length - 1, 0]
                    const labels = [run.vocab_size - 1, 0, run.vocab_size - 1]

                    const selected = yield* Effect.acquireRelease(
                      program.score(prefix, Uint32Array.from(step.canvas), rows, labels),
                      Tensor.clear,
                      { interruptible: true }
                    )

                    close(
                      yield* Tensor.toNumberArray(selected),
                      rows.flatMap((row) =>
                        labels.map((label) => step.raw_logits.values[row * run.vocab_size + label]!)
                      ),
                      run.dtype,
                      "selected logits"
                    )
                  }
                }))
              }

              const after = yield* program.inspect(prefix)
              expect(after).toEqual(before)

              if (blockIndex + 1 < run.blocks.length) {
                const original = prefix
                prefix = yield* Effect.acquireRelease(
                  program.commit(prefix, Uint32Array.from(run.encoder_calls[blockIndex + 1]!.tokens)),
                  release,
                  { interruptible: true }
                )
                expect((yield* program.inspect(original)).layers).toEqual(before.layers)
                expect(prefix.tokenCount).toBe(original.tokenCount + run.encoder_calls[blockIndex + 1]!.tokens.length)
                expect((yield* program.inspect(prefix)).sharedBytes).toBeGreaterThan(0)
              }
            }

            const canvases = run.blocks.flatMap((block) => [
              block.initial_canvas,
              ...block.steps.map((step) => step.random_canvas)
            ])

            const steps = run.blocks.flatMap((block) => block.steps)

            for (const outputLimit of ["whole-block", "exact"] as const) {
              let canvasIndex = 0
              let stepIndex = 0
              let evaluationIndex = 0
              const committed: Array<ReadonlyArray<number>> = []
              const pages: Array<ReadonlyArray<number>> = []
              const progress: Array<Diffusion.Progress> = []

              const observed: Runtime.RuntimeService = {
                ...runtime,
                execute: (executable, invocation) =>
                  Effect.gen(function*() {
                    if (invocation.state?.access === "Append") {
                      committed.push(invocation.state.tokens[0]!)
                    } else if (invocation.state?.access === "ReadOnly") {
                      const reference = steps[evaluationIndex++]!
                      const canvas = new Uint32Array(yield* runtime.readback(invocation.bindings[0]!))
                      expect(Array.from(canvas)).toEqual(reference.canvas)

                      if (reference.feedback_in === null) {
                        expect(invocation.bindings).toHaveLength(2)
                      } else {
                        const feedback = invocation.bindings[2]!
                        expect(feedback.dtype).toBe(program.predictionDtype)
                        const actual = Array.from(new Float32Array(yield* runtime.readback(feedback)))

                        if (run.dtype === "bfloat16") {
                          close(actual, reference.feedback_in.values, run.dtype, "generated BF16 feedback")
                        } else {
                          // The complete-model absolute tolerance is in raw-logit units.
                          // Undo temperature scaling before applying that same bound.
                          const previous = steps[evaluationIndex - 2]!
                          const temperature = DG.generationTemperature(0.4, 0.8, 3, previous.remaining)
                          close(
                            actual.map((value) => value * temperature),
                            previous.raw_logits.values,
                            run.dtype,
                            "generated feedback in raw-logit units"
                          )
                        }
                      }
                    }

                    return yield* runtime.execute(executable, invocation)
                  })
              }

              const generated = yield* DG.generate(program, Uint32Array.from(run.prompt), {
                maxNewTokens: 5,
                maxSteps: 3,
                confidenceThreshold: 1e-8,
                eosTokenIds: [],
                outputLimit,
                compile: { optimize },
                random: {
                  canvas: () => Uint32Array.from(canvases[canvasIndex++]!),
                  exponentials: () => Float32Array.from(steps[stepIndex++]!.exponentials)
                },
                onPage: (tokens) =>
                  Effect.sync(() => {
                    expect(tokens.length).toBeGreaterThan(0)
                    pages.push(Array.from(tokens))
                  }),
                onProgress: (step) =>
                  Effect.sync(() => {
                    progress.push(step)
                  })
              }).pipe(Effect.provideService(Runtime.Runtime, observed))

              const expected = run.sequences.slice(
                run.prompt.length,
                outputLimit === "exact"
                  ? run.prompt.length + 5
                  : undefined
              )

              expect(Array.from(generated.tokens)).toEqual(expected)
              expect(pages.flat()).toEqual(expected)
              expect(pages.map((page) => page.length)).toEqual([3, outputLimit === "exact" ? 2 : 3])
              expect(generated.blocks).toBe(2)
              expect(generated.refinements).toBe(6)
              expect(generated.stop).toBe("length")
              expect(progress.map((step) => step.done)).toEqual(steps.map((step) => step.done || step.remaining === 1))
              expect(canvasIndex).toBe(canvases.length)
              expect(stepIndex).toBe(steps.length)
              expect(evaluationIndex).toBe(steps.length)
              expect(committed).toEqual(run.encoder_calls.map((call) => call.tokens))
            }
          }))
      )
    }
  }
})

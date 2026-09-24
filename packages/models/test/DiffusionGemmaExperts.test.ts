import { Tensor } from "@effect-torch/core"
import { DiffusionGemma } from "@effect-torch/models"
import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { onDevices } from "./utils/devices.ts"

const prefix = "model.decoder.layers.0.experts"

const configFor = (hidden: number, intermediate: number, k: number, experts = 3) =>
  DiffusionGemma.parseConfig({
    model_type: "diffusion_gemma",
    text_config: {
      vocab_size: 7,
      hidden_size: hidden,
      intermediate_size: 3,
      num_hidden_layers: 2,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      head_dim: 8,
      global_head_dim: 16,
      num_experts: experts,
      top_k_experts: k,
      moe_intermediate_size: intermediate,
      sliding_window: 4
    }
  })

// Independent finite BF16 conversion with round-to-nearest, ties-to-even.
const bf16 = (value: number): number => {
  const view = new DataView(new ArrayBuffer(4))
  view.setFloat32(0, value, true)
  const bits = view.getUint32(0, true)
  view.setUint32(0, (bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff0000, true)

  return view.getFloat32(0, true)
}

const ulp = (value: number) => Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(value))) - 7))

const f32 = Math.fround

const gelu = (x: number) => f32(0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))))

interface Case {
  readonly hidden: number
  readonly intermediate: number
  readonly k: number
  readonly input: ReadonlyArray<number>
  readonly gateUp: ReadonlyArray<number>
  readonly down: ReadonlyArray<number>
  readonly indices: ReadonlyArray<number>
  readonly weights: ReadonlyArray<number>
}

// Scalar expert-major eager evaluation, independent of the graph's rank/scatter
// implementation. Repeated routes for one expert retain their original order.
const reference = (fixture: Case, dtype: "f32" | "bf16", order: "expert" | "route" = "expert") => {
  const round = dtype === "bf16" ? bf16 : f32
  const { hidden: h, intermediate: m, k } = fixture
  const input = fixture.input.map(round)
  const gateUp = fixture.gateUp.map(round)
  const down = fixture.down.map(round)
  const weights = fixture.weights.map(f32)

  const linear = (x: ReadonlyArray<number>, bank: ReadonlyArray<number>, expert: number, columns: number) =>
    Array.from({ length: columns }, (_, column) => {
      let sum = 0

      for (let i = 0; i < x.length; i++) sum = f32(sum + f32(x[i] * bank[(expert * columns + column) * x.length + i]))

      return round(sum)
    })

  return Array.from({ length: input.length / h }, (_, row) => {
    const result = Array<number>(h).fill(0)
    const routes = Array.from({ length: k }, (_, route) => ({ expert: fixture.indices[row * k + route], route }))

    if (order === "expert") routes.sort((a, b) => a.expert - b.expert || a.route - b.route)

    for (const { expert, route } of routes) {
      const projected = linear(input.slice(row * h, (row + 1) * h), gateUp, expert, 2 * m)
      const activation = projected.slice(0, m).map((gate, i) => round(round(gelu(gate)) * projected[m + i]))
      const output = linear(activation, down, expert, h)

      for (let i = 0; i < h; i++) {
        const contribution = round(f32(output[i] * weights[row * k + route]))
        result[i] = round(result[i] + contribution)
      }
    }

    return result
  }).flat()
}

const ordinary: Case = {
  hidden: 2,
  intermediate: 2,
  k: 2,
  input: [1, 1, -1, 2, 2, -0.5, -2, -1],
  gateUp: [
    0.5,
    -0.25,
    1,
    0.5,
    0.25,
    1,
    -1,
    0.5,
    -1,
    0.5,
    0.75,
    -0.25,
    0.5,
    0.125,
    0.375,
    -0.5,
    1.5,
    -0.5,
    -0.75,
    0.5,
    0.25,
    -1,
    0.5,
    0.5
  ],
  down: [2, -1, -0.5, 0.25, -4, 0.5, 0.75, -1, 0.25, 1.5, -2, 0.5],
  indices: [2, 0, 1, 2, 2, 2, 0, 1],
  weights: [0.75, 0.25, 2, -0.5, 1, 1, 0.8, -0.25]
}

const cancellation: Case = {
  hidden: 2,
  intermediate: 1,
  k: 3,
  input: [1, 1],
  gateUp: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
  down: [2 ** 26, -(2 ** 26), -(2 ** 26), 2 ** 26, 1, -1],
  indices: [2, 0, 1],
  weights: [1, 1, 1]
}

const repeated: Case = {
  ...cancellation,
  down: [1, -1, 1, -1, 1, -1],
  indices: [2, 2, 2],
  weights: [2 ** 26, 1, -(2 ** 26)]
}

const graph = (values: ReadonlyArray<number>, shape: ReadonlyArray<number>, dtype: "f32" | "bf16") =>
  Tensor.fromTypedArray(Float32Array.from(values), shape).pipe(Effect.flatMap(Tensor.cast(dtype)))

const setup = (fixture: Case, dtype: "f32" | "bf16") =>
  Effect.gen(function*() {
    const rows = fixture.input.length / fixture.hidden

    const roots = [
      yield* graph(fixture.input, [rows, fixture.hidden], dtype),
      yield* graph(fixture.gateUp, [3, 2 * fixture.intermediate, fixture.hidden], dtype),
      yield* graph(fixture.down, [3, fixture.hidden, fixture.intermediate], dtype),
      yield* Tensor.fromTypedArray(Uint32Array.from(fixture.indices), [rows, fixture.k]),
      yield* Tensor.fromTypedArray(Float32Array.from(fixture.weights), [rows, fixture.k])
    ]

    return yield* Tensor.compute(roots).pipe(Effect.flatMap(Tensor.clearAllScoped))
  })

const build = (config: DiffusionGemma.Config, values: ReadonlyArray<Tensor.Any>) =>
  DiffusionGemma.routedExperts(
    config,
    {
      [prefix + ".gate_up_proj"]: values[1],
      [prefix + ".down_proj"]: values[2]
    },
    0,
    values[0],
    values[3],
    values[4]
  )

const close = (actual: ReadonlyArray<number>, expected: ReadonlyArray<number>, dtype: "f32" | "bf16") => {
  expect(actual).toHaveLength(expected.length)
  actual.forEach((value, i) => {
    expect(Number.isFinite(value)).toBe(true)
    const tolerance = dtype === "bf16" ? ulp(expected[i]) : 2e-6 + 2e-5 * Math.abs(expected[i])
    expect(Math.abs(value - expected[i]), "element " + i + ": " + value + " != " + expected[i]).toBeLessThanOrEqual(
      tolerance
    )
  })
}

onDevices("DiffusionGemma experts", (device) => (it) => {
  it.effect("default projector retains strict dots and the callback receives route-major rows", () =>
    Effect.scoped(Effect.gen(function*() {
      for (const dtype of ["f32", "bf16"] as const) {
        for (const fixture of [ordinary, cancellation, repeated]) {
          const bindings = yield* setup(fixture, dtype)
          const inputs = yield* Effect.forEach(bindings, (input, slot) => Tensor.makeInput(slot, input))

          const projections: Array<{
            input: Tensor.Any
            weight: Tensor.Any
            indices: Tensor.Any
          }> = []

          const activation = (gate: Tensor.Any) => Tensor.gelu(gate, { approximate: "tanh" })

          const defaultRoot = yield* Tensor.gatedExperts(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            activation
          )

          const callbackRoot = yield* Tensor.gatedExperts(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            activation,
            (input, weight, indices) => {
              projections.push({ input, weight, indices })

              return Tensor.expertLinearRows(input, weight, indices)
            }
          )

          expect(projections).toHaveLength(2)
          expect(projections[0].weight).toBe(inputs[1])
          expect(projections[1].weight).toBe(inputs[2])
          const rows = fixture.input.length / fixture.hidden
          const expectedInput = Array.from({ length: fixture.k }, () => fixture.input).flat()

          const expectedIndices = Array.from(
            { length: fixture.k },
            (_, route) => Array.from({ length: rows }, (_, row) => fixture.indices[row * fixture.k + route])
          ).flat()

          for (const optimize of [false, true]) {
            const program = yield* Tensor.freezeProgram([
              defaultRoot,
              callbackRoot,
              projections[0].input,
              projections[0].indices,
              projections[1].indices
            ], { optimize })

            const outputs = yield* Tensor.runProgram(program, bindings).pipe(Effect.flatMap(Tensor.clearAllScoped))
            const expected = reference(fixture, dtype)
            close(yield* Tensor.toNumberArray(outputs[0]), expected, dtype)
            expect(yield* Tensor.toNumberArray(outputs[1])).toEqual(yield* Tensor.toNumberArray(outputs[0]))
            expect(yield* Tensor.toNumberArray(outputs[2])).toEqual(expectedInput.map(dtype === "bf16" ? bf16 : f32))
            expect(yield* Tensor.toNumberArray(outputs[3])).toEqual(expectedIndices)
            expect(yield* Tensor.toNumberArray(outputs[4])).toEqual(expectedIndices)
          }
        }
      }
    })))

  for (const dtype of ["f32", "bf16"] as const) {
    for (const optimize of [false, true]) {
      for (
        const [name, fixture] of [["selected negative outputs and duplicate indices", ordinary], [
          "expert-index cancellation order",
          cancellation
        ], ["stable repeated-expert route order", repeated]] as const
      ) {
        it.effect(dtype + " optimize=" + optimize + ": " + name, () =>
          Effect.scoped(Effect.gen(function*() {
            const config = yield* configFor(fixture.hidden, fixture.intermediate, fixture.k)
            const bindings = yield* setup(fixture, dtype)
            const inputs = yield* Effect.forEach(bindings, (input, slot) => Tensor.makeInput(slot, input))
            const root = yield* build(config, inputs)
            expect(root.dtype).toBe(dtype)
            expect(root.shape).toEqual([fixture.input.length / fixture.hidden, fixture.hidden])
            const program = yield* Tensor.freezeProgram([root], { optimize, constantWeights: false })
            const [output] = yield* Tensor.runProgram(program, bindings).pipe(Effect.flatMap(Tensor.clearAllScoped))
            expect(output.dtype).toBe(dtype)
            const expected = reference(fixture, dtype)
            close(yield* Tensor.toNumberArray(output), expected, dtype)

            if (fixture === ordinary) expect(expected.some((value) => value < 0)).toBe(true)

            if (fixture === cancellation) {
              // The small final expert survives only in ascending expert order.
              expect(expected).not.toEqual(reference(fixture, dtype, "route"))
              expect(Math.abs(expected[0])).toBeGreaterThan(0.5)
            }

            if (fixture === repeated) {
              const reordered = { ...fixture, weights: [2 ** 26, -(2 ** 26), 1] }
              expect(expected).not.toEqual(reference(reordered, dtype))
            }

            const round = dtype === "bf16" ? bf16 : f32
            expect(yield* Tensor.toNumberArray(bindings[1])).toEqual(fixture.gateUp.map(round))
            expect(yield* Tensor.toNumberArray(bindings[2])).toEqual(fixture.down.map(round))
          })))
      }
    }
  }

  it.effect("rejects wrong k/route dtype/weights shape before native execution", () =>
    Effect.gen(function*() {
      const config = yield* configFor(2, 2, 2)
      const x = yield* Tensor.ones([2, 2])

      const bank = {
        [prefix + ".gate_up_proj"]: yield* Tensor.ones([3, 4, 2]),
        [prefix + ".down_proj"]: yield* Tensor.ones([3, 2, 2])
      }

      const ids = yield* Tensor.zeros([2, 2], { dtype: "u32" })
      const weights = yield* Tensor.ones([2, 2])

      const failures = [
        DiffusionGemma.routedExperts(config, bank, 0, x, yield* Tensor.zeros([2, 1], { dtype: "u32" }), weights),
        DiffusionGemma.routedExperts(config, bank, 0, x, yield* Tensor.zeros([2, 2], { dtype: "i64" }), weights),
        DiffusionGemma.routedExperts(config, bank, 0, x, ids, yield* Tensor.ones([2, 1])),
        DiffusionGemma.routedExperts(config, bank, 0, x, ids, yield* Tensor.ones([2, 2], { dtype: "bf16" })),
        DiffusionGemma.routedExperts(config, bank, 0, yield* Tensor.ones([1, 2, 2]), ids, weights),
        DiffusionGemma.routedExperts(config, bank, -1, x, ids, weights),
        DiffusionGemma.routedExperts(config, {}, 0, x, ids, weights)
      ]

      const expected = [
        "ModelError",
        "TensorError",
        "TensorError",
        "TensorError",
        "TensorError",
        "ModelError",
        "ModelError"
      ]

      for (let index = 0; index < failures.length; index++) {
        expect((yield* Effect.flip(failures[index]))._tag).toBe(expected[index])
      }
    }))

  it.effect("out-of-range experts fail without poisoning the compiled combine", () =>
    Effect.scoped(Effect.gen(function*() {
      const config = yield* configFor(2, 2, 2)
      const bindings = yield* setup(ordinary, "bf16")
      const inputs = yield* Effect.forEach(bindings, (input, slot) => Tensor.makeInput(slot, input))
      const program = yield* Tensor.freezeProgram([yield* build(config, inputs)], { constantWeights: false })

      for (const bad of [3, 0xffffffff]) {
        const indices = [...ordinary.indices]
        indices[0] = bad

        const [invalid] = yield* Tensor.compute([yield* Tensor.fromTypedArray(Uint32Array.from(indices), [4, 2])]).pipe(
          Effect.flatMap(Tensor.clearAllScoped)
        )

        const args = [...bindings]
        args[3] = invalid
        const error = yield* Effect.flip(Tensor.runProgram(program, args))
        expect(error._tag).toBe("TensorError")
        expect(error.message).toContain("index is out of range")
      }

      const [recovered] = yield* Tensor.runProgram(program, bindings).pipe(Effect.flatMap(Tensor.clearAllScoped))
      close(yield* Tensor.toNumberArray(recovered), reference(ordinary, "bf16"), "bf16")
    })))

  it.effect("keeps large BF16 expert banks external without whole-bank conversion workspace", () =>
    Effect.gen(function*() {
      const h = 512
      const m = 256

      // Measure ordinary GEMM workspace independently of any expert bank. CUDA
      // lowering names this fixed 32 MiB resource cublas_workspace. The public
      // diagnostics report total capacity, so use this isolated plan as a control.
      for (const optimize of [false, true]) {
        let gemmWorkspace = 0

        if (device === "cuda") {
          const x = yield* Tensor.makeInput(0, yield* Tensor.zeros([2, h], { dtype: "bf16" }))
          const weight = yield* Tensor.makeInput(1, yield* Tensor.zeros([2 * m, h], { dtype: "bf16" }))
          const gemm = yield* Tensor.linearRows(x, weight)
          const control = yield* Tensor.freezeProgram([gemm], { optimize, constantWeights: false })
          gemmWorkspace = control.handle.diagnostics.memory.workspaceBytes
          expect(gemmWorkspace).toBe(32 * 1024 * 1024)
        }

        const workspaces: Array<number> = []

        for (const experts of [3, 24]) {
          const config = yield* configFor(h, m, 2, experts)

          // Metadata-only inputs. No multi-megabyte bank is materialized or uploaded.
          const values = [
            yield* Tensor.zeros([1, h], { dtype: "bf16" }),
            yield* Tensor.zeros([experts, 2 * m, h], { dtype: "bf16" }),
            yield* Tensor.zeros([experts, h, m], { dtype: "bf16" }),
            yield* Tensor.zeros([1, 2], { dtype: "u32" }),
            yield* Tensor.zeros([1, 2], { dtype: "f32" })
          ]

          const inputs = yield* Effect.forEach(values, (value, slot) => Tensor.makeInput(slot, value))

          const program = yield* Tensor.freezeProgram([yield* build(config, inputs)], {
            optimize,
            constantWeights: false
          })

          const diagnostics = program.handle.diagnostics
          const bankBytes = experts * (2 * m * h + h * m) * 2
          expect(diagnostics.memory.externalBytes).toBe(bankBytes + h * 2 + 2 * 4 + 2 * 4)
          expect(diagnostics.memory.persistentBytes).toBeLessThan(64 * 1024)
          // Keep the original small-bank bound on variable scratch even when
          // the borrowed bank grows eightfold. Fixed GEMM capacity cannot hide a copy.
          const smallBankBytes = 3 * (2 * m * h + h * m) * 2
          expect(diagnostics.memory.workspaceBytes - gemmWorkspace).toBeLessThan(smallBankBytes / 4)
          workspaces.push(diagnostics.memory.workspaceBytes)
          // Even one converted expert matrix exceeds this activation-sized bound.
          expect(diagnostics.legalization?.materializedConversionBytes).toBeLessThan(64 * 1024)
        }

        // Extra experts need only bounded group-control metadata, not matrix storage.
        expect(Math.abs(workspaces[1] - workspaces[0])).toBeLessThan(64 * 1024)
      }
    }))
})

import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Model, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

type ModelDtype = "f16" | "bf16"

const fixtures = [
  { name: "embedding", shape: [4, 2], values: [0.5, -0.75, 1.25, 0.25, -0.5, 1, 0.75, 0.5] },
  { name: "norm", shape: [2], values: [1, 0.75] },
  { name: "query", shape: [2, 2], values: [0.5, -0.25, 0.25, 0.75] },
  { name: "key", shape: [2, 2], values: [0.75, 0.25, -0.5, 0.5] },
  { name: "value", shape: [2, 2], values: [0.5, 0.25, -0.25, 0.75] },
  { name: "projection", shape: [2, 3], values: [0.75, -0.5, 0.25, 0.25, 0.5, -0.75] },
  { name: "bias", shape: [3], values: [0.125, -0.25, 0.375] }
] as const
const tokenBatches = [[0, 1, 2], [2, 3, 1]] as const
const eps = 1e-5

const attentionModel = Model.define({
  parameterSpecs: fixtures.map(({ name, shape }): Model.ParameterSpec => ({
    name,
    shape,
    initializer: { _tag: "Constant", value: 0 }
  })),
  forward: ([table, norm, qw, kw, vw, projection, bias], tokens) =>
    Effect.gen(function*() {
      expect(tokens.dtype).toBe("u32")
      const embedded = yield* Tensor.embedding(tokens, { weight: table })
      const normalized = yield* Tensor.rmsNorm(embedded, norm, eps)
      const query = yield* Tensor.matmul(normalized, qw)
      const key = yield* Tensor.matmul(normalized, kw)
      const value = yield* Tensor.matmul(normalized, vw)
      const rotatedQuery = yield* Tensor.rotaryEmbedding(query, 3, 10000)
      const rotatedKey = yield* Tensor.rotaryEmbedding(key, 3, 10000)
      const attended = yield* Tensor.scaledDotProductAttention(rotatedQuery, rotatedKey, value, { causal: true })
      const residual = yield* Tensor.add(embedded, attended)
      const activated = yield* Tensor.silu(residual)
      const output = yield* Tensor.linear(activated, projection, bias)
      for (
        const tensor of [
          table,
          norm,
          qw,
          kw,
          vw,
          projection,
          bias,
          embedded,
          normalized,
          query,
          key,
          value,
          rotatedQuery,
          rotatedKey,
          attended,
          residual,
          activated,
          output
        ]
      ) {
        expect(tensor.dtype).toBe(table.dtype)
        expect(tensor.storage).toBeUndefined()
      }
      return output
    })
})

// The fixture stays finite and normal in both formats. Round F32 to the nearest
// representable value, ties to even, at each semantic operation boundary.
const narrow = (dtype: ModelDtype, value: number): number => {
  const x = Math.fround(value)
  if (x === 0) return x
  const step = 2 ** (Math.floor(Math.log2(Math.abs(x))) - (dtype === "f16" ? 10 : 7))
  const units = Math.abs(x) / step
  const lower = Math.floor(units)
  const rounded = units - lower === 0.5 ? lower + lower % 2 : Math.round(units)
  return Math.sign(x) * rounded * step
}

const reference = (dtype: ModelDtype, tokens: ReadonlyArray<number>): Array<number> => {
  const f32 = Math.fround
  const round = (x: number) => narrow(dtype, x)
  const [table, norm, qw, kw, vw, projection, bias] = fixtures.map(({ values }) => values.map(round))
  const dot = (a: ReadonlyArray<number>, b: ReadonlyArray<number>) =>
    a.reduce((sum, x, i) => f32(sum + f32(x * b[i])), 0)
  const project = (rows: ReadonlyArray<ReadonlyArray<number>>, weight: ReadonlyArray<number>, columns = 2) =>
    rows.map((row) =>
      Array.from({ length: columns }, (_, column) => round(dot(row, row.map((_, i) => weight[i * columns + column]))))
    )
  const embedded = tokens.map((token) => table.slice(token * 2, token * 2 + 2))
  const normalized = embedded.map((row) => {
    const inverse = f32(1 / f32(Math.sqrt(f32(f32(dot(row, row) / 2) + f32(eps)))))
    return row.map((x, i) => round(f32(f32(x * inverse) * norm[i])))
  })
  const rotate = (rows: ReadonlyArray<ReadonlyArray<number>>) =>
    rows.map(([x, y], position) => {
      const c = f32(Math.cos(position))
      const s = f32(Math.sin(position))
      return [round(f32(f32(x * c) - f32(y * s))), round(f32(f32(x * s) + f32(y * c)))]
    })
  const query = rotate(project(normalized, qw))
  const key = rotate(project(normalized, kw))
  const value = project(normalized, vw)
  const attended = query.map((row, position) => {
    const scores = key.slice(0, position + 1).map((k) => f32(dot(row, k) * f32(1 / Math.sqrt(2))))
    const maximum = Math.max(...scores)
    const weights = scores.map((score) => f32(Math.exp(f32(score - maximum))))
    const total = weights.reduce((sum, weight) => f32(sum + weight), 0)
    return row.map((_, column) => round(f32(dot(weights, value.map((v) => v[column])) / total)))
  })
  const activated = embedded.map((row, position) =>
    row.map((x, column) => {
      const residual = round(x + attended[position][column])
      // Tensor.silu is x * sigmoid(x); sigmoid expands to div, tanh, div, add.
      const sigmoid = round(round(round(Math.tanh(round(residual / 2))) / 2) + 0.5)
      return round(residual * sigmoid)
    })
  )
  // Linear adds bias before its single semantic result rounding.
  return activated.flatMap((row) =>
    bias.map((b, column) => round(f32(dot(row, row.map((_, i) => projection[i * 3 + column])) + b)))
  )
}

onDevices("Same-definition dtype model", () => (it) => {
  for (const dtype of ["f16", "bf16"] as const) {
    it.effect(`${dtype} attention inference preserves semantic precision and reuses bindings`, () =>
      Effect.scoped(Effect.gen(function*() {
        const model = yield* attentionModel
        yield* Effect.addFinalizer(() => model.clear)
        const parameterGraphs = yield* Effect.forEach(fixtures, ({ shape, values }) =>
          Effect.gen(function*() {
            const source = yield* Tensor.fromTypedArray(new Float32Array(values), shape)
            expect(source.dtype).toBe("f32")
            const cast = yield* Tensor.cast(source, dtype)
            expect(cast.dtype).toBe(dtype)
            return cast
          }))
        const params = yield* Tensor.compute(parameterGraphs).pipe(Effect.flatMap(Tensor.clearAllScoped))
        const tokenGraphs = yield* Effect.forEach(
          tokenBatches,
          (tokens) => Tensor.fromTypedArray(new Uint32Array(tokens), [1, 3])
        )
        const inputs = yield* Tensor.compute(tokenGraphs).pipe(Effect.flatMap(Tensor.clearAllScoped))
        for (const parameter of params) expect(parameter.dtype).toBe(dtype)
        for (const input of [...tokenGraphs, ...inputs]) expect(input.dtype).toBe("u32")
        const parameters = yield* Effect.forEach(params, (parameter, slot) => Tensor.makeInput(slot, parameter))
        const tokens = yield* Tensor.makeInput(params.length, inputs[0])
        const root = yield* model.forward(parameters, tokens)
        expect(root.dtype).toBe(dtype)
        expect(root.shape).toEqual([1, 3, 3])
        const programs = yield* Effect.forEach([false, true], (optimize) => Tensor.freezeProgram([root], { optimize }))
        for (const program of programs) expect(program.outputs[0].dtype).toBe(dtype)
        const expected = tokenBatches.map((batch) => reference(dtype, batch))
        expect(expected[0]).not.toEqual(expected[1])
        for (let index = 0; index < inputs.length; index++) {
          for (const program of programs) {
            const [output] = yield* Tensor.runProgram(program, [...params, inputs[index]]).pipe(
              Effect.flatMap(Tensor.clearAllScoped)
            )
            expect(output.dtype).toBe(dtype)
            expect(output.shape).toEqual([1, 3, 3])
            expect(yield* Tensor.toNumberArray(output)).toEqual(expected[index])
          }
          const cached = yield* model.execute(params, inputs[index]).pipe(Effect.flatMap(Tensor.clearScoped))
          expect(cached.dtype).toBe(dtype)
          expect(yield* Tensor.toNumberArray(cached)).toEqual(expected[index])
          expect(yield* model.stats).toEqual({ cached: 1, compiled: 1 })
        }
      })))
  }
})

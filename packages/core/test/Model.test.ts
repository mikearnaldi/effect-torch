import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Gradient, Loss, Model, Optimizer, Tensor } from "../src/index.ts"
import { deep, floats, onDevices, type TestDevice } from "./utils/devices.ts"

const tmpdir = Effect.sync(() => fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-")))

const values = (t: Tensor.GenericTensor) => Tensor.toNumberArray(t)

const mlp = Effect.gen(function* () {
  return yield* Model.chain(
    yield* Model.linear("fc1", 2, 8),
    yield* Model.tanh,
    yield* Model.linear("fc2", 8, 1),
    yield* Model.sigmoid
  )
})

const handForward = (
  [w1, b1, w2, b2]: ReadonlyArray<Tensor.GenericTensor>,
  x: Tensor.GenericTensor
) =>
  Effect.gen(function* () {
    const h = yield* Tensor.tanh(yield* Tensor.add(yield* Tensor.matmul(x, w1), b1))
    return yield* Tensor.sigmoid(yield* Tensor.add(yield* Tensor.matmul(h, w2), b2))
  })

onDevices("Model", (device: TestDevice) => (it) => {
  describe("validation", () => {
    it.effect("linear rejects an empty name", () =>
      Effect.gen(function* () {
        const error = yield* Effect.flip(Model.linear("", 2, 8))
        expect(error._tag).toBe("ModelError")
        expect(error.op).toBe("linear")
        expect(error.message).toContain("name")
      })
    )

    it.effect("linear rejects non-positive feature counts", () =>
      Effect.gen(function* () {
        for (const [inF, outF] of [[0, 8], [-1, 8], [2.5, 8], [2, 0]] as const) {
          const error = yield* Effect.flip(Model.linear("fc", inF, outF))
          expect(error._tag).toBe("ModelError")
          expect(error.op).toBe("linear")
        }
      })
    )

    it.effect("chain fails on duplicate names", () =>
      Effect.gen(function* () {
        const error = yield* Effect.flip(
          Model.chain(
            yield* Model.linear("fc", 2, 2),
            yield* Model.relu,
            yield* Model.linear("fc", 2, 2)
          )
        )
        expect(error._tag).toBe("ModelError")
        expect(error.op).toBe("chain")
        expect(error.message).toContain("fc.weight")
      })
    )

    it.effect("chain fails when empty", () =>
      Effect.gen(function* () {
        const error = yield* Effect.flip(Model.chain())
        expect(error._tag).toBe("ModelError")
        expect(error.message).toContain("at least one model")
      })
    )
  })

  describe("names", () => {
    it.effect("concatenates names in order and reports arity", () =>
      Effect.gen(function* () {
        const model = yield* mlp
        expect(model.names).toEqual(["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])
        const params = yield* model.init
        expect(params.length).toBe(model.names.length)
      })
    )
  })

  describe("types", () => {
    it.effect("chain infers the concatenated parameter tuple", () =>
      Effect.gen(function* () {
        const params: readonly [
          Tensor.GenericTensor,
          Tensor.GenericTensor,
          Tensor.GenericTensor,
          Tensor.GenericTensor
        ] = yield* (yield* mlp).init
        expect(params.length).toBe(4)
      })
    )

    it.effect("nested chains flatten to one tuple", () =>
      Effect.gen(function* () {
        const nested = yield* Model.chain(
          yield* Model.chain(yield* Model.linear("a", 2, 3), yield* Model.relu),
          yield* Model.chain(yield* Model.relu, yield* Model.linear("b", 3, 1))
        )
        expect(nested.names).toEqual(["a.weight", "a.bias", "b.weight", "b.bias"])
        const params: readonly [
          Tensor.GenericTensor,
          Tensor.GenericTensor,
          Tensor.GenericTensor,
          Tensor.GenericTensor
        ] = yield* nested.init
        expect(params.length).toBe(4)
      })
    )
  })

  describe("composition", () => {
    it.effect("chained forward matches the hand-written forward on the same parameters", () =>
      Effect.gen(function* () {
        const model = yield* mlp
        const params = yield* Tensor.evaluate(yield* model.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
        const viaModel = yield* Tensor.evaluate([yield* model.forward(params, x)])
        const byHand = yield* Tensor.evaluate([yield* handForward(params, x)])
        deep(yield* values(viaModel[0]), yield* values(byHand[0]))
      })
    )

    it.effect("chained gradients match the hand-written gradients", () =>
      Effect.gen(function* () {
        const model = yield* mlp
        const params = yield* Tensor.evaluate(yield* model.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
        const lossModel = yield* Tensor.sum(yield* model.forward(params, x))
        const lossHand = yield* Tensor.sum(yield* handForward(params, x))
        const gradsModel = yield* Tensor.evaluate(yield* Gradient.grad(lossModel, params))
        const gradsHand = yield* Tensor.evaluate(yield* Gradient.grad(lossHand, params))
        for (let i = 0; i < gradsModel.length; i++) {
          deep(yield* values(gradsModel[i]), yield* values(gradsHand[i]))
        }
      })
    )

    it.effect("slices the parameter tuple across mixed parameterless and parameterised stages", () =>
      Effect.gen(function* () {
        const model = yield* Model.chain(
          yield* Model.linear("a", 2, 3),
          yield* Model.relu,
          yield* Model.relu,
          yield* Model.linear("b", 3, 1)
        )
        const [wa, ba, wb, bb] = yield* Tensor.evaluate(yield* model.init)
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        const manual = Effect.gen(function* () {
          const h1 = yield* Tensor.relu(yield* Tensor.add(yield* Tensor.matmul(x, wa), ba))
          const h2 = yield* Tensor.relu(h1)
          return yield* Tensor.add(yield* Tensor.matmul(h2, wb), bb)
        })
        const [viaModel] = yield* Tensor.evaluate([yield* model.forward([wa, ba, wb, bb], x)])
        const [byHand] = yield* Tensor.evaluate([yield* manual])
        deep(yield* values(viaModel), yield* values(byHand))
      })
    )
  })

  describe("serialization", () => {
    it.effect("save/load round-trips values and order", () =>
      Effect.gen(function* () {
        const dir = yield* tmpdir
        const file = path.join(dir, "mlp.safetensors")
        const model = yield* mlp
        const params = yield* Tensor.evaluate(yield* model.init)
        yield* Model.save(model, params, file)
        const loaded = yield* Model.load(model, file)
        expect(loaded.length).toBe(model.names.length)
        for (let i = 0; i < params.length; i++) {
          expect(loaded[i].shape).toEqual(params[i].shape)
          deep(yield* values(loaded[i]), yield* values(params[i]))
        }
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const [before] = yield* Tensor.evaluate([yield* model.forward(params, x)])
        const [after] = yield* Tensor.evaluate([yield* model.forward(loaded, x)])
        deep(yield* values(after), yield* values(before))
      })
    )

    it.effect("save fails with ModelError on an arity mismatch", () =>
      Effect.gen(function* () {
        const dir = yield* tmpdir
        const model = yield* mlp
        const params = yield* Tensor.evaluate(yield* model.init)
        const error = yield* Effect.flip(Model.save(model, params.slice(0, 3), path.join(dir, "x.safetensors")))
        expect(error._tag).toBe("ModelError")
        expect(error.op).toBe("save")
        expect(error.message).toContain("4 parameters, got 3")
      })
    )

    it.effect("load fails with ModelError on missing keys", () =>
      Effect.gen(function* () {
        const dir = yield* tmpdir
        const file = path.join(dir, "partial.safetensors")
        const small = yield* Model.linear("fc1", 2, 8)
        const params = yield* Tensor.evaluate(yield* small.init)
        yield* Model.save(small, params, file)
        const error = yield* Effect.flip(Model.load(yield* mlp, file))
        expect(error._tag).toBe("ModelError")
        expect(error.op).toBe("load")
        expect(error.message).toContain("fc2.weight")
      })
    )

    it.effect("params from a different architecture fail at graph-build time", () =>
      Effect.gen(function* () {
        const dir = yield* tmpdir
        const file = path.join(dir, "wide.safetensors")
        const wide = yield* Model.chain(
          yield* Model.linear("fc1", 3, 8),
          yield* Model.tanh,
          yield* Model.linear("fc2", 8, 1)
        )
        yield* Model.save(wide, yield* Tensor.evaluate(yield* wide.init), file)
        const narrow = yield* Model.chain(
          yield* Model.linear("fc1", 2, 8),
          yield* Model.tanh,
          yield* Model.linear("fc2", 8, 1)
        )
        const params = yield* Model.load(narrow, file)
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const error = yield* Effect.flip(narrow.forward(params, x))
        expect(error._tag).toBe("TensorError")
        expect(error.op).toBe("matmul")
      })
    )
  })

  describe("stop policy", () => {
    it.effect("stops on a loss target", () =>
      Effect.gen(function* () {
        const model = yield* mlp
        const x = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
        const y = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [4, 1])
        let steps = 0
        const { loss } = yield* Model.train(model, {
          optimizer: Optimizer.adam({ lr: 0.1 }),
          loss: Loss.mse,
          data: { input: x, target: y },
          stop: ({ step, loss }) => loss < 0.2 || step >= 2500,
          onStep: () => Effect.sync(() => steps++)
        })
        expect(loss).toBeLessThan(0.2)
        expect(steps).toBeLessThan(2500)
      })
    )

    it.effect("stops on any condition — a step count, a loss target, or external state", () =>
      Effect.gen(function* () {
        const model = yield* mlp
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const y = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        let patience = 3
        const { loss } = yield* Model.train(model, {
          optimizer: Optimizer.sgd({ lr: 0.1 }),
          loss: Loss.mse,
          data: { input: x, target: y },
          stop: () => --patience === 0
        })
        expect(patience).toBe(0)
        expect(Number.isFinite(loss)).toBe(true)
      })
    )
  })

  describe("train", () => {
    it.effect("trains a chained MLP on xor to convergence", () =>
      Effect.gen(function* () {
        const model = yield* mlp
        const x = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
        const y = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [4, 1])
        const { params, loss } = yield* Model.train(model, {
          optimizer: Optimizer.adam({ lr: 0.1 }),
          loss: Loss.mse,
          data: { input: x, target: y },
          stop: ({ step }) => step >= 2500
        })
        expect(loss).toBeLessThan(0.05)
        const [pred] = yield* Tensor.evaluate([yield* model.forward(params, x)])
        expect((yield* values(pred)).map((v) => (v > 0.5 ? 1 : 0))).toEqual([0, 1, 1, 0])
      })
    )

    it.effect("reports every step to onStep in order", () =>
      Effect.gen(function* () {
        const model = yield* mlp
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const y = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
        const seen: Array<Model.TrainStep> = []
        yield* Model.train(model, {
          optimizer: Optimizer.sgd({ lr: 0.1 }),
          loss: Loss.mse,
          data: { input: x, target: y },
          stop: ({ step }) => step >= 10,
          onStep: (info) => Effect.sync(() => seen.push(info))
        })
        expect(seen.map(({ step }) => step)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expect(seen.every(({ loss }) => Number.isFinite(loss))).toBe(true)
      })
    )

    it.effect("trains from explicit initial parameters", () =>
      Effect.gen(function* () {
        const model = yield* mlp
        const x = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
        const y = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [4, 1])
        const initial = yield* Tensor.evaluate(yield* model.init)
        const lossOf = (params: Model.Params<typeof model>) =>
          Effect.gen(function* () {
            const [value] = yield* Tensor.evaluate([
              yield* Loss.mse(yield* model.forward(params, x), y)
            ])
            return (yield* values(value))[0]
          })
        const before = yield* lossOf(initial)
        const { params, loss } = yield* Model.train(model, {
          optimizer: Optimizer.adam({ lr: 0.1 }),
          loss: Loss.mse,
          data: { input: x, target: y },
          stop: ({ step }) => step >= 200,
          params: initial
        })
        expect(loss).toBeLessThan(before)
        expect(yield* lossOf(params)).toBeLessThan(before)
      })
    )
  })
})

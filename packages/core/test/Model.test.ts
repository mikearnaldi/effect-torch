import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Gradient, Model, type Runtime, Tensor } from "../src/index.ts"
import { deep, floats, onDevices } from "./utils/devices.ts"

const tmpdir = Effect.sync(() => fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-")))

const values = (t: Tensor.Any) => Tensor.toNumberArray(t)

const identityForward: Model.Definition["forward"] = (_, input) => Effect.succeed(input as Tensor.Lazy)

const mlp = Effect.gen(function*() {
  return yield* Model.chain(
    yield* Model.linear("fc1", 2, 8),
    yield* Model.tanh,
    yield* Model.linear("fc2", 8, 1),
    yield* Model.sigmoid
  )
})

const handForward = (
  [w1, b1, w2, b2]: ReadonlyArray<Tensor.Any>,
  x: Tensor.Any
) =>
  Effect.gen(function*() {
    const h = yield* Tensor.tanh(yield* Tensor.add(yield* Tensor.matmul(x, w1), b1))
    return yield* Tensor.sigmoid(yield* Tensor.add(yield* Tensor.matmul(h, w2), b2))
  })

onDevices("Model", () => (it) => {
  describe("validation", () => {
    it.effect("define validates names and logical shapes", () =>
      Effect.gen(function*() {
        const model = yield* Model.define({
          parameters: [
            { name: "scalar", shape: [] },
            { name: "empty", shape: [2, 0, 3] }
          ],
          init: Effect.succeed([]),
          forward: identityForward
        })
        expect(model.names).toEqual(["scalar", "empty"])
        expect(model.names).toEqual(model.parameters.map((parameter) => parameter.name))
        expect(model.parameters.map((parameter) => parameter.shape)).toEqual([[], [2, 0, 3]])

        for (
          const parameters of [
            [{ name: "", shape: [1] }],
            [{ name: "x", shape: [1] }, { name: "x", shape: [2] }],
            [{ name: "x", shape: [-1] }],
            [{ name: "x", shape: [1.5] }],
            [{ name: "x", shape: [Number.MAX_SAFE_INTEGER + 1] }]
          ]
        ) {
          const error = yield* Effect.flip(Model.define({ parameters, forward: identityForward }))
          expect(error._tag).toBe("ModelError")
          expect(error.op).toBe("define")
        }
      }))

    it.effect("define without init creates a load-only model", () =>
      Effect.gen(function*() {
        const model = yield* Model.define({
          parameters: [{ name: "weight", shape: [2, 2] }],
          forward: identityForward
        })
        const error = yield* Effect.flip(model.init)
        expect(error._tag).toBe("ModelError")
        expect(error.op).toBe("init")
      }))

    it.effect("linear rejects an empty name", () =>
      Effect.gen(function*() {
        const error = yield* Effect.flip(Model.linear("", 2, 8))
        expect(error._tag).toBe("ModelError")
        expect(error.op).toBe("linear")
        expect(error.message).toContain("name")
      }))

    it.effect("linear rejects non-positive feature counts", () =>
      Effect.gen(function*() {
        for (const [inF, outF] of [[0, 8], [-1, 8], [2.5, 8], [2, 0]] as const) {
          const error = yield* Effect.flip(Model.linear("fc", inF, outF))
          expect(error._tag).toBe("ModelError")
          expect(error.op).toBe("linear")
        }
      }))

    it.effect("chain fails on duplicate names", () =>
      Effect.gen(function*() {
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
      }))

    it.effect("chain fails when empty", () =>
      Effect.gen(function*() {
        const error = yield* Effect.flip(Model.chain())
        expect(error._tag).toBe("ModelError")
        expect(error.message).toContain("at least one model")
      }))
  })

  describe("names", () => {
    it.effect("concatenates names in order and reports arity", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        expect(model.names).toEqual(["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])
        expect(model.parameters).toEqual([
          { name: "fc1.weight", shape: [2, 8] },
          { name: "fc1.bias", shape: [1, 8] },
          { name: "fc2.weight", shape: [8, 1] },
          { name: "fc2.bias", shape: [1, 1] }
        ])
        const params = yield* model.init
        expect(params.length).toBe(model.names.length)
      }))
  })

  describe("arity", () => {
    it.effect("init produces one parameter per name", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const params = yield* model.init
        expect(params.length).toBe(model.names.length)
      }))

    it.effect("forward fails with ModelError on the wrong parameter count", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const params = yield* model.init
        const x = yield* Tensor.fromTypedArray(floats([0, 1]), [1, 2])
        const error = yield* Effect.flip(model.forward(params.slice(0, 3), x))
        expect(error._tag).toBe("ModelError")
        if (error._tag === "ModelError") {
          expect(error.op).toBe("forward")
          expect(error.message).toContain("4 parameters")
          expect(error.message).toContain("got 3")
        }
      }))

    it.effect("a layer called directly checks its own arity", () =>
      Effect.gen(function*() {
        const fc = yield* Model.linear("fc1", 2, 8)
        const x = yield* Tensor.fromTypedArray(floats([0, 1]), [1, 2])
        const error = yield* Effect.flip(fc.forward([], x))
        expect(error._tag).toBe("ModelError")
        if (error._tag === "ModelError") {
          expect(error.message).toContain("fc1")
          expect(error.message).toContain("[fc1.weight, fc1.bias]")
        }
      }))

    it.effect("nested chains flatten to one array", () =>
      Effect.gen(function*() {
        const nested = yield* Model.chain(
          yield* Model.chain(yield* Model.linear("a", 2, 3), yield* Model.relu),
          yield* Model.chain(yield* Model.relu, yield* Model.linear("b", 3, 1))
        )
        expect(nested.names).toEqual(["a.weight", "a.bias", "b.weight", "b.bias"])
        const params = yield* nested.init
        expect(params.length).toBe(4)
      }))
  })

  describe("composition", () => {
    it.effect("chained forward matches the hand-written forward on the same parameters", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const params = yield* Tensor.compute(yield* model.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
        const viaModel = yield* Tensor.compute([yield* model.forward(params, x)])
        const byHand = yield* Tensor.compute([yield* handForward(params, x)])
        deep(yield* values(viaModel[0]), yield* values(byHand[0]))
      }))

    it.effect("chained gradients match the hand-written gradients", () =>
      Effect.gen(function*() {
        const model = yield* mlp
        const params = yield* Tensor.compute(yield* model.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 0, 0, 1, 1, 0, 1, 1]), [4, 2])
        const lossModel = yield* Tensor.sum(yield* model.forward(params, x))
        const lossHand = yield* Tensor.sum(yield* handForward(params, x))
        const gradsModel = yield* Tensor.compute(yield* Gradient.grad(lossModel, params))
        const gradsHand = yield* Tensor.compute(yield* Gradient.grad(lossHand, params))
        for (let i = 0; i < gradsModel.length; i++) {
          deep(yield* values(gradsModel[i]), yield* values(gradsHand[i]))
        }
      }))

    it.effect("slices the parameter array across mixed parameterless and parameterised stages", () =>
      Effect.gen(function*() {
        const model = yield* Model.chain(
          yield* Model.linear("a", 2, 3),
          yield* Model.relu,
          yield* Model.relu,
          yield* Model.linear("b", 3, 1)
        )
        const [wa, ba, wb, bb] = yield* Tensor.compute(yield* model.init)
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        const manual = Effect.gen(function*() {
          const h1 = yield* Tensor.relu(yield* Tensor.add(yield* Tensor.matmul(x, wa), ba))
          const h2 = yield* Tensor.relu(h1)
          return yield* Tensor.add(yield* Tensor.matmul(h2, wb), bb)
        })
        const [viaModel] = yield* Tensor.compute([yield* model.forward([wa, ba, wb, bb], x)])
        const [byHand] = yield* Tensor.compute([yield* manual])
        deep(yield* values(viaModel), yield* values(byHand))
      }))

    it.effect("add sums the branches' outputs over a shared input", () =>
      Effect.gen(function*() {
        const model = yield* Model.add(
          yield* Model.linear("a", 2, 2),
          yield* Model.linear("b", 2, 2),
          yield* Model.relu
        )
        const params = yield* Tensor.compute(yield* model.init)
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4]), [2, 2])
        const manual = Effect.gen(function*() {
          const [wa, ba, wb, bb] = params
          const a = yield* Tensor.add(yield* Tensor.matmul(x, wa), ba)
          const b = yield* Tensor.add(yield* Tensor.matmul(x, wb), bb)
          return yield* Tensor.add(yield* Tensor.add(a, b), yield* Tensor.relu(x))
        })
        const [viaModel] = yield* Tensor.compute([yield* model.forward(params, x)])
        const [byHand] = yield* Tensor.compute([yield* manual])
        deep(yield* values(viaModel), yield* values(byHand))
      }))

    it.effect("add fails on an empty chain and on colliding parameter names", () =>
      Effect.gen(function*() {
        const empty = yield* Effect.flip(Model.add())
        expect(empty._tag).toBe("ModelError")
        const colliding = yield* Effect.flip(Model.add(
          yield* Model.linear("a", 2, 2),
          yield* Model.linear("a", 2, 2)
        ))
        expect(colliding._tag).toBe("ModelError")
      }))
  })

  describe("layers", () => {
    it.effect("conv2d forward matches a manual conv + bias", () =>
      Effect.gen(function*() {
        const model = yield* Model.conv2d("c", 2, 4, 3, { padding: 1 })
        expect(model.names).toEqual(["c.weight", "c.bias"])
        const [w, b] = yield* Tensor.compute(yield* model.init)
        expect(w.shape).toEqual([4, 2, 3, 3])
        expect(b.shape).toEqual([4])
        const x = yield* Tensor.fromTypedArray(floats(Array.from({ length: 32 }, (_, i) => i / 32)), [1, 2, 4, 4])
        const [viaModel] = yield* Tensor.compute([yield* model.forward([w, b], x)])
        const [byHand] = yield* Tensor.compute([
          yield* Tensor.add(yield* Tensor.conv2d(x, w, { padding: 1 }), yield* Tensor.reshape(b, [1, 4, 1, 1]))
        ])
        expect(viaModel.shape).toEqual([1, 4, 4, 4])
        deep(yield* values(viaModel), yield* values(byHand))
      }))

    it.effect("conv2d supports grouped convolutions", () =>
      Effect.gen(function*() {
        const model = yield* Model.conv2d("c", 4, 4, [3, 3], { groups: 2 })
        const [w, b] = yield* Tensor.compute(yield* model.init)
        expect(w.shape).toEqual([4, 2, 3, 3])
        const x = yield* Tensor.fromTypedArray(floats(Array.from({ length: 4 * 5 * 5 }, (_, i) => i / 100)), [
          1,
          4,
          5,
          5
        ])
        const [out] = yield* Tensor.compute([yield* model.forward([w, b], x)])
        expect(out.shape).toEqual([1, 4, 3, 3])
      }))

    it.effect("conv2d rejects invalid configuration", () =>
      Effect.gen(function*() {
        expect((yield* Effect.flip(Model.conv2d("", 2, 4, 3))).op).toBe("conv2d")
        expect((yield* Effect.flip(Model.conv2d("c", 0, 4, 3))).op).toBe("conv2d")
        expect((yield* Effect.flip(Model.conv2d("c", 2, 4, 0))).op).toBe("conv2d")
        const grouped = yield* Effect.flip(Model.conv2d("c", 3, 4, 3, { groups: 2 }))
        expect(grouped.message).toContain("groups")
      }))

    it.effect("conv1d forward matches a manual conv + bias", () =>
      Effect.gen(function*() {
        const model = yield* Model.conv1d("c", 2, 3, 3, { stride: 2 })
        expect(model.names).toEqual(["c.weight", "c.bias"])
        const [w, b] = yield* Tensor.compute(yield* model.init)
        expect(w.shape).toEqual([3, 2, 3])
        const x = yield* Tensor.fromTypedArray(floats(Array.from({ length: 16 }, (_, i) => i / 16)), [1, 2, 8])
        const [viaModel] = yield* Tensor.compute([yield* model.forward([w, b], x)])
        const [byHand] = yield* Tensor.compute([
          yield* Tensor.add(yield* Tensor.conv1d(x, w, { stride: 2 }), yield* Tensor.reshape(b, [1, 3, 1]))
        ])
        expect(viaModel.shape).toEqual([1, 3, 3])
        deep(yield* values(viaModel), yield* values(byHand))
      }))

    it.effect("embedding looks up rows and accumulates gradients", () =>
      Effect.gen(function*() {
        const model = yield* Model.embedding("emb", 3, 2)
        expect(model.names).toEqual(["emb.weight"])
        const [w] = yield* Tensor.compute(yield* model.init)
        expect(w.shape).toEqual([3, 2])
        const indexes = yield* Tensor.fromTypedArray(new BigInt64Array([0n, 1n, 0n]), [3])
        const [viaModel] = yield* Tensor.compute([yield* model.forward([w], indexes)])
        const [byHand] = yield* Tensor.compute([yield* Tensor.embedding(indexes, { weight: w })])
        deep(yield* values(viaModel), yield* values(byHand))
        const loss = yield* Tensor.sum(yield* model.forward([w], indexes))
        const [grad] = yield* Tensor.compute(yield* Gradient.grad(loss, [w]))
        deep(yield* values(grad), [2, 2, 1, 1, 0, 0])
      }))

    it.effect("embedding rejects an out-of-range paddingIndex", () =>
      Effect.gen(function*() {
        const error = yield* Effect.flip(Model.embedding("emb", 3, 2, { paddingIndex: 3 }))
        expect(error.message).toContain("paddingIndex")
      }))

    it.effect("positionEmbedding looks up rows 0..t-1 and ignores the input values", () =>
      Effect.gen(function*() {
        const model = yield* Model.positionEmbedding("wpe", 4, 2)
        expect(model.names).toEqual(["wpe.weight"])
        const [w] = yield* Tensor.compute(yield* model.init)
        expect(w.shape).toEqual([4, 2])
        const ids = yield* Tensor.fromTypedArray(new BigInt64Array([9n, 9n, 9n]), [1, 3])
        const [out] = yield* Tensor.compute([yield* model.forward([w], ids)])
        expect(out.shape).toEqual([3, 2])
        const [byHand] = yield* Tensor.compute([
          yield* Tensor.embedding(yield* Tensor.fromTypedArray(new BigInt64Array([0n, 1n, 2n]), [3]), {
            weight: w
          })
        ])
        deep(yield* values(out), yield* values(byHand))
        const loss = yield* Tensor.sum(yield* model.forward([w], ids))
        const [grad] = yield* Tensor.compute(yield* Gradient.grad(loss, [w]))
        deep(yield* values(grad), [1, 1, 1, 1, 1, 1, 0, 0])
        const tooLong = yield* Tensor.fromTypedArray(new BigInt64Array([0n, 0n, 0n, 0n, 0n]), [5])
        const error = yield* Effect.flip(model.forward([w], tooLong))
        expect(error._tag).toBe("ModelError")
        expect(error.message).toContain("exceeds maxPositions")
      }))

    it.effect("layerNorm normalizes the trailing dimensions to unit statistics", () =>
      Effect.gen(function*() {
        const model = yield* Model.layerNorm("ln", 4)
        expect(model.names).toEqual(["ln.weight", "ln.bias"])
        const params = yield* Tensor.compute(yield* model.init)
        expect(params[0].shape).toEqual([4])
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3, 10, -5, 0, 7, 3]), [2, 4])
        const [out] = yield* Tensor.compute([yield* model.forward(params, x)])
        const means = yield* values(yield* Tensor.mean(out, { dims: [-1] }))
        const variances = yield* values(yield* Tensor.variance(out, { dims: [-1], correction: 0 }))
        deep(means, [0, 0])
        deep(variances, [1, 1])
      }))

    it.effect("layerNorm normalizes multi-dimension shapes and applies weight and bias", () =>
      Effect.gen(function*() {
        const model = yield* Model.layerNorm("ln", [2, 3])
        const [w, b] = yield* Tensor.compute(yield* model.init)
        expect(w.shape).toEqual([2, 3])
        const x = yield* Tensor.fromTypedArray(floats(Array.from({ length: 12 }, (_, i) => i - 5)), [2, 2, 3])
        const [out] = yield* Tensor.compute([yield* model.forward([w, b], x)])
        const means = yield* values(yield* Tensor.mean(out, { dims: [-2, -1] }))
        deep(means, [0, 0])
        const two = yield* Tensor.fromTypedArray(floats([2, 2, 2, 2, 2, 2]), [2, 3])
        const one = yield* Tensor.fromTypedArray(floats([1, 1, 1, 1, 1, 1]), [2, 3])
        const [scaled] = yield* Tensor.compute([yield* model.forward([two, one], x)])
        deep(yield* values(scaled), (yield* values(out)).map((v) => v * 2 + 1))
      }))

    it.effect("layerNorm rejects invalid configuration", () =>
      Effect.gen(function*() {
        expect((yield* Effect.flip(Model.layerNorm("ln", []))).message).toContain("empty")
        expect((yield* Effect.flip(Model.layerNorm("ln", 0))).message).toContain("positive")
        expect((yield* Effect.flip(Model.layerNorm("ln", 4, { eps: 0 }))).message).toContain("eps")
      }))

    it.effect("multiHeadAttention matches a manual head-split composition (values and gradients)", () =>
      Effect.gen(function*() {
        const embedDim = 8
        const numHeads = 2
        const headDim = embedDim / numHeads
        const model = yield* Model.multiHeadAttention("attn", embedDim, numHeads)
        expect(model.names).toEqual([
          "attn.qkv.weight",
          "attn.qkv.bias",
          "attn.wo.weight",
          "attn.wo.bias"
        ])
        const params = yield* Tensor.compute(yield* model.init)
        const x = yield* Tensor.fromTypedArray(
          floats(Array.from({ length: 2 * 3 * 8 }, (_, i) => ((i * 5 + 1) % 11 - 5) / 5)),
          [2, 3, 8]
        )
        const manual = Effect.gen(function*() {
          const split = (t: Tensor.Any) =>
            Effect.gen(function*() {
              const r = yield* Tensor.reshape(t, [2, 3, numHeads, headDim])
              return yield* Tensor.transpose(r, [0, 2, 1, 3])
            })
          const qkv = yield* Tensor.add(yield* Tensor.matmul(x, params[0]), params[1])
          const q = yield* split(yield* Tensor.slice(qkv, { start: [0, 0, 0], end: [2, 3, embedDim] }))
          const k = yield* split(yield* Tensor.slice(qkv, { start: [0, 0, embedDim], end: [2, 3, 2 * embedDim] }))
          const v = yield* split(yield* Tensor.slice(qkv, { start: [0, 0, 2 * embedDim], end: [2, 3, 3 * embedDim] }))
          const attended = yield* Tensor.scaledDotProductAttention(q, k, v)
          const merged = yield* Tensor.reshape(yield* Tensor.transpose(attended, [0, 2, 1, 3]), [2, 3, embedDim])
          return yield* Tensor.add(yield* Tensor.matmul(merged, params[2]), params[3])
        })
        const [viaModel] = yield* Tensor.compute([yield* model.forward(params, x)])
        expect(viaModel.shape).toEqual([2, 3, 8])
        const [byHand] = yield* Tensor.compute([yield* manual])
        deep(yield* values(viaModel), yield* values(byHand))
        const lossModel = yield* Tensor.sum(yield* model.forward(params, x))
        const lossManual = yield* Tensor.sum(yield* manual)
        const gradsModel = yield* Tensor.compute(yield* Gradient.grad(lossModel, params))
        const gradsManual = yield* Tensor.compute(yield* Gradient.grad(lossManual, params))
        for (let i = 0; i < params.length; i++) {
          deep(yield* values(gradsModel[i]), yield* values(gradsManual[i]))
        }
      }))

    it.effect("multiHeadAttention with causal masking changes the output and trains", () =>
      Effect.gen(function*() {
        const plain = yield* Model.multiHeadAttention("attn", 8, 2)
        const causal = yield* Model.multiHeadAttention("attn", 8, 2, { causal: true })
        expect(causal.names).toEqual(plain.names)
        const params = yield* Tensor.compute(yield* plain.init)
        const x = yield* Tensor.fromTypedArray(
          floats(Array.from({ length: 2 * 3 * 8 }, (_, i) => ((i * 5 + 1) % 11 - 5) / 5)),
          [2, 3, 8]
        )
        const [a] = yield* Tensor.compute([yield* plain.forward(params, x)])
        const [b] = yield* Tensor.compute([yield* causal.forward(params, x)])
        expect(yield* values(a)).not.toEqual(yield* values(b))
        const loss = yield* Tensor.sum(yield* causal.forward(params, x))
        const grads = yield* Tensor.compute(yield* Gradient.grad(loss, params))
        expect(grads.length).toBe(4)
      }))

    it.effect("multiHeadAttention rejects invalid configuration and arity", () =>
      Effect.gen(function*() {
        expect((yield* Effect.flip(Model.multiHeadAttention("", 8, 2))).op).toBe("multiHeadAttention")
        expect((yield* Effect.flip(Model.multiHeadAttention("a", 0, 2))).op).toBe("multiHeadAttention")
        expect((yield* Effect.flip(Model.multiHeadAttention("a", 8, 0))).op).toBe("multiHeadAttention")
        expect((yield* Effect.flip(Model.multiHeadAttention("a", 7, 2))).message).toContain("divisible")
        const model = yield* Model.multiHeadAttention("a", 8, 2)
        const x = yield* Tensor.fromTypedArray(floats(Array.from({ length: 8 }, (_, i) => i / 8)), [1, 1, 8])
        const error = yield* Effect.flip(model.forward([], x))
        expect(error._tag).toBe("ModelError")
      }))

    it.effect("activation models apply the corresponding tensor operations", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([-2, -0.5, 0, 0.5, 2]), [5])
        const cases: Array<
          [Model.Model, (x: Tensor.Any) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime>]
        > = [
          [yield* Model.gelu(), (x) => Tensor.gelu(x)],
          [yield* Model.gelu({ approximate: "tanh" }), (x) => Tensor.gelu(x, { approximate: "tanh" })],
          [yield* Model.silu, Tensor.silu],
          [yield* Model.mish, Tensor.mish],
          [yield* Model.softplus, Tensor.softplus],
          [yield* Model.elu(), (x) => Tensor.elu(x)],
          [yield* Model.elu({ alpha: 2 }), (x) => Tensor.elu(x, { alpha: 2 })],
          [yield* Model.leakyRelu(), (x) => Tensor.leakyRelu(x)],
          [yield* Model.leakyRelu({ negativeSlope: 0.2 }), (x) => Tensor.leakyRelu(x, { negativeSlope: 0.2 })],
          [yield* Model.softmax(), (x) => Tensor.softmax(x)],
          [yield* Model.logSoftmax(), (x) => Tensor.logSoftmax(x)]
        ]
        for (const [model, op] of cases) {
          const [viaModel] = yield* Tensor.compute([yield* model.forward([], x)])
          const [byHand] = yield* Tensor.compute([yield* op(x)])
          deep(yield* values(viaModel), yield* values(byHand))
        }
      }))

    it.effect("flatten collapses all but the batch dimension by default", () =>
      Effect.gen(function*() {
        const model = yield* Model.flatten()
        const x = yield* Tensor.fromTypedArray(floats(Array.from({ length: 120 }, (_, i) => i)), [2, 3, 4, 5])
        const [out] = yield* Tensor.compute([yield* model.forward([], x)])
        expect(out.shape).toEqual([2, 60])
        const full = yield* Model.flatten({ startDim: 0 })
        const [vector] = yield* Tensor.compute([yield* full.forward([], x)])
        expect(vector.shape).toEqual([120])
      }))

    it.effect("dropout with p = 0 is the identity and validates p", () =>
      Effect.gen(function*() {
        const model = yield* Model.dropout({ p: 0 })
        const x = yield* Tensor.fromTypedArray(floats([1, 2, 3]), [3])
        const [out] = yield* Tensor.compute([yield* model.forward([], x)])
        deep(yield* values(out), [1, 2, 3])
        expect((yield* Effect.flip(Model.dropout({ p: 1 }))).message).toContain("[0, 1)")
        expect((yield* Effect.flip(Model.dropout({ p: -0.1 }))).message).toContain("[0, 1)")
      }))

    it.effect("pool models apply the corresponding tensor operations", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats(Array.from({ length: 16 }, (_, i) => i)), [1, 1, 4, 4])
        const maxPool = yield* Model.maxPool2d({ kernelSize: 2 })
        const [viaMax] = yield* Tensor.compute([yield* maxPool.forward([], x)])
        const [handMax] = yield* Tensor.compute([yield* Tensor.maxPool2d(x, { kernelSize: 2 })])
        deep(yield* values(viaMax), yield* values(handMax))
        const avgPool = yield* Model.avgPool2d({ kernelSize: [2, 2], stride: 2 })
        const [viaAvg] = yield* Tensor.compute([yield* avgPool.forward([], x)])
        const [handAvg] = yield* Tensor.compute([yield* Tensor.avgPool2d(x, { kernelSize: [2, 2], stride: 2 })])
        deep(yield* values(viaAvg), yield* values(handAvg))
        expect((yield* Effect.flip(Model.maxPool2d({ kernelSize: 0 }))).message).toContain("kernelSize")
        expect((yield* Effect.flip(Model.avgPool2d({ kernelSize: 2, stride: 0 }))).message).toContain("stride")
        expect((yield* Effect.flip(Model.maxPool2d({ kernelSize: 2, padding: -1 }))).message).toContain("padding")
      }))

    it.effect("a conv net chain flows end to end", () =>
      Effect.gen(function*() {
        const model = yield* Model.chain(
          yield* Model.conv2d("conv", 3, 4, 3, { padding: 1 }),
          yield* Model.relu,
          yield* Model.maxPool2d({ kernelSize: 2 }),
          yield* Model.flatten(),
          yield* Model.linear("fc", 64, 10)
        )
        expect(model.names).toEqual(["conv.weight", "conv.bias", "fc.weight", "fc.bias"])
        const params = yield* model.init
        const x = yield* Tensor.fromTypedArray(floats(Array.from({ length: 2 * 3 * 8 * 8 }, (_, i) => i / 384)), [
          2,
          3,
          8,
          8
        ])
        const [out] = yield* Tensor.compute([yield* model.forward(params, x)])
        expect(out.shape).toEqual([2, 10])
      }))

    it.effect("a dropout stage drops out of the eval chain with the same parameter array", () =>
      Effect.gen(function*() {
        const trainNet = yield* Model.chain(
          yield* Model.linear("fc1", 2, 4),
          yield* Model.relu,
          yield* Model.dropout({ p: 0 }),
          yield* Model.linear("fc2", 4, 1)
        )
        const evalNet = yield* Model.chain(
          yield* Model.linear("fc1", 2, 4),
          yield* Model.relu,
          yield* Model.linear("fc2", 4, 1)
        )
        const params = yield* Tensor.compute(yield* trainNet.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const [viaTrain] = yield* Tensor.compute([yield* trainNet.forward(params, x)])
        const [viaEval] = yield* Tensor.compute([yield* evalNet.forward(params, x)])
        deep(yield* values(viaTrain), yield* values(viaEval))
      }))

    it.effect("checkpoint preserves names, outputs, and gradients", () =>
      Effect.gen(function*() {
        const block = Model.chain(yield* Model.linear("fc", 2, 8), yield* Model.tanh)
        const plain = yield* Model.chain(yield* block, yield* Model.linear("head", 8, 1))
        const wrapped = yield* Model.chain(
          yield* Model.checkpoint(yield* block),
          yield* Model.linear("head", 8, 1)
        )
        expect(wrapped.names).toEqual(plain.names)
        const params = yield* Tensor.compute(yield* plain.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0, 1, 1, 0, 0]), [4, 2])
        const [outPlain] = yield* Tensor.compute([yield* plain.forward(params, x)])
        const [outWrapped] = yield* Tensor.compute([yield* wrapped.forward(params, x)])
        deep(yield* values(outWrapped), yield* values(outPlain))
        const lossPlain = yield* Tensor.sum(yield* plain.forward(params, x))
        const lossWrapped = yield* Tensor.sum(yield* wrapped.forward(params, x))
        const gradsPlain = yield* Tensor.compute(yield* Gradient.grad(lossPlain, params))
        const gradsWrapped = yield* Tensor.compute(yield* Gradient.grad(lossWrapped, params))
        for (let i = 0; i < gradsPlain.length; i++) {
          deep(yield* values(gradsWrapped[i]), yield* values(gradsPlain[i]))
        }
      }))

    it.effect("a checkpointed sub-model still checks its own arity", () =>
      Effect.gen(function*() {
        const wrapped = yield* Model.checkpoint(yield* Model.linear("fc1", 2, 8))
        const x = yield* Tensor.fromTypedArray(floats([0, 1]), [1, 2])
        const error = yield* Effect.flip(wrapped.forward([], x))
        expect(error._tag).toBe("ModelError")
        if (error._tag === "ModelError") {
          expect(error.message).toContain("fc1")
        }
      }))

    it.effect("residual adds the block input to its output (values and gradients)", () =>
      Effect.gen(function*() {
        const block = Model.chain(yield* Model.linear("fc", 2, 2), yield* Model.tanh)
        const plain = yield* block
        const skip = yield* Model.residual(yield* block)
        expect(skip.names).toEqual(plain.names)
        const params = yield* Tensor.compute(yield* plain.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0, 1, 1]), [3, 2])
        const [inner] = yield* Tensor.compute([yield* plain.forward(params, x)])
        const [out] = yield* Tensor.compute([yield* skip.forward(params, x)])
        const xv = yield* values(x)
        deep(yield* values(out), (yield* values(inner)).map((v, i) => v + xv[i]))
        const lossSkip = yield* Tensor.sum(yield* skip.forward(params, x))
        const grads = yield* Tensor.compute(yield* Gradient.grad(lossSkip, params))
        // d/dw sum(x + f(x)) = d/dw sum(f(x)): the skip path adds no
        // parameter gradient; compare against the plain block's grads
        const lossPlain = yield* Tensor.sum(yield* plain.forward(params, x))
        const gradsPlain = yield* Tensor.compute(yield* Gradient.grad(lossPlain, params))
        for (let i = 0; i < grads.length; i++) {
          deep(yield* values(grads[i]), yield* values(gradsPlain[i]))
        }
      }))

    it.effect("residual composes with checkpoint inside a chain", () =>
      Effect.gen(function*() {
        const block = Model.chain(yield* Model.linear("fc", 2, 2), yield* Model.relu)
        const net = yield* Model.chain(
          yield* Model.checkpoint(yield* Model.residual(yield* block)),
          yield* Model.linear("head", 2, 1)
        )
        const params = yield* Tensor.compute(yield* net.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const [out] = yield* Tensor.compute([yield* net.forward(params, x)])
        expect(out.shape).toEqual([2, 1])
        const loss = yield* Tensor.sum(yield* net.forward(params, x))
        const grads = yield* Tensor.compute(yield* Gradient.grad(loss, params))
        expect(grads.length).toBe(net.names.length)
      }))

    it.effect("merge fans one input into several models and combines outputs (values and gradients)", () =>
      Effect.gen(function*() {
        const a = yield* Model.linear("a", 2, 2)
        const b = yield* Model.linear("b", 2, 2)
        const merged = yield* Model.merge([a, b], (x, y) => Tensor.add(x, y))
        expect(merged.names).toEqual(["a.weight", "a.bias", "b.weight", "b.bias"])
        const params = yield* Tensor.compute(yield* merged.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const hand = Effect.gen(function*() {
          return yield* Tensor.add(
            yield* a.forward(params.slice(0, 2), x),
            yield* b.forward(params.slice(2), x)
          )
        })
        const [viaMerge] = yield* Tensor.compute([yield* merged.forward(params, x)])
        const [byHand] = yield* Tensor.compute([yield* hand])
        deep(yield* values(viaMerge), yield* values(byHand))
        const lossMerged = yield* Tensor.sum(yield* merged.forward(params, x))
        const grads = yield* Tensor.compute(yield* Gradient.grad(lossMerged, params))
        const lossHand = yield* Tensor.sum(yield* hand)
        const gradsHand = yield* Tensor.compute(yield* Gradient.grad(lossHand, params))
        for (let i = 0; i < grads.length; i++) {
          deep(yield* values(grads[i]), yield* values(gradsHand[i]))
        }
        const error = yield* Effect.flip(merged.forward(params.slice(0, 3), x))
        expect(error._tag).toBe("ModelError")
      }))

    it.effect("merge is variadic: three models merge with per-model combiner arguments", () =>
      Effect.gen(function*() {
        const a = yield* Model.linear("a", 2, 2)
        const b = yield* Model.linear("b", 2, 2)
        const c = yield* Model.linear("c", 2, 2)
        const merged = yield* Model.merge([a, b, c], (x, y, z) =>
          Effect.gen(function*() {
            return yield* Tensor.add(x, yield* Tensor.add(y, z))
          }))
        expect(merged.names).toEqual(["a.weight", "a.bias", "b.weight", "b.bias", "c.weight", "c.bias"])
        const params = yield* Tensor.compute(yield* merged.init)
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const [viaMerge] = yield* Tensor.compute([yield* merged.forward(params, x)])
        const [byHand] = yield* Tensor.compute([
          yield* Tensor.add(
            yield* a.forward(params.slice(0, 2), x),
            yield* Tensor.add(
              yield* b.forward(params.slice(2, 4), x),
              yield* c.forward(params.slice(4), x)
            )
          )
        ])
        deep(yield* values(viaMerge), yield* values(byHand))
      }))

    it.effect("merge rejects an empty array and duplicate parameter names", () =>
      Effect.gen(function*() {
        const empty = yield* Effect.flip(Model.merge([], () => Effect.succeed(null as never)))
        expect(empty.message).toContain("at least one")
        const error = yield* Effect.flip(
          Model.merge([yield* Model.linear("fc", 2, 2), yield* Model.linear("fc", 2, 2)], (x, y) => Tensor.add(x, y))
        )
        expect(error._tag).toBe("ModelError")
        expect(error.message).toContain("duplicate")
      }))

    it.effect("mapInput transforms the input before the sub-model", () =>
      Effect.gen(function*() {
        const emb = yield* Model.embedding("pos", 4, 2)
        const positioned = yield* Model.mapInput(emb, (idx) =>
          Tensor.arange(idx.shape[idx.shape.length - 1], undefined, { dtype: "i64" }))
        expect(positioned.names).toEqual(["pos.weight"])
        const [w] = yield* Tensor.compute(yield* positioned.init)
        const ids = yield* Tensor.fromTypedArray(new BigInt64Array([2n, 3n]), [1, 2])
        const [viaMapped] = yield* Tensor.compute([yield* positioned.forward([w], ids)])
        // the mapped input is arange(2) = [0, 1]: rows 0 and 1 of the table
        const [byHand] = yield* Tensor.compute([
          yield* emb.forward(
            [w],
            yield* Tensor.fromTypedArray(new BigInt64Array([0n, 1n]), [2])
          )
        ])
        deep(yield* values(viaMapped), yield* values(byHand))
      }))
  })

  describe("serialization", () => {
    it.effect("save/load round-trips values and order", () =>
      Effect.gen(function*() {
        const dir = yield* tmpdir
        const file = path.join(dir, "mlp.safetensors")
        const model = yield* mlp
        const params = yield* Tensor.compute(yield* model.init)
        yield* Model.save(model, params, file)
        const loaded = yield* Model.load(model, file)
        expect(loaded.length).toBe(model.names.length)
        for (let i = 0; i < params.length; i++) {
          expect(loaded[i].shape).toEqual(params[i].shape)
          deep(yield* values(loaded[i]), yield* values(params[i]))
        }
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const [before] = yield* Tensor.compute([yield* model.forward(params, x)])
        const [after] = yield* Tensor.compute([yield* model.forward(loaded, x)])
        deep(yield* values(after), yield* values(before))
      }))

    it.effect("save fails with ModelError on an arity mismatch", () =>
      Effect.gen(function*() {
        const dir = yield* tmpdir
        const model = yield* mlp
        const params = yield* Tensor.compute(yield* model.init)
        const error = yield* Effect.flip(Model.save(model, params.slice(0, 3), path.join(dir, "x.safetensors")))
        expect(error._tag).toBe("ModelError")
        expect(error.op).toBe("save")
        expect(error.message).toContain("4 parameters, got 3")
      }))

    it.effect("load fails with ModelError on missing keys", () =>
      Effect.gen(function*() {
        const dir = yield* tmpdir
        const file = path.join(dir, "partial.safetensors")
        const small = yield* Model.linear("fc1", 2, 8)
        const params = yield* Tensor.compute(yield* small.init)
        yield* Model.save(small, params, file)
        const error = yield* Effect.flip(Model.load(yield* mlp, file))
        expect(error._tag).toBe("ModelError")
        expect(error.op).toBe("load")
        expect(error.message).toContain("fc2.weight")
      }))

    it.effect("params from a different architecture fail at graph-build time", () =>
      Effect.gen(function*() {
        const dir = yield* tmpdir
        const file = path.join(dir, "wide.safetensors")
        const wide = yield* Model.chain(
          yield* Model.linear("fc1", 3, 8),
          yield* Model.tanh,
          yield* Model.linear("fc2", 8, 1)
        )
        yield* Model.save(wide, yield* Tensor.compute(yield* wide.init), file)
        const narrow = yield* Model.chain(
          yield* Model.linear("fc1", 2, 8),
          yield* Model.tanh,
          yield* Model.linear("fc2", 8, 1)
        )
        const params = yield* Model.load(narrow, file)
        const x = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
        const error = yield* Effect.flip(narrow.forward(params, x))
        expect(error._tag).toBe("TensorError")
        expect(error.op).toBe("linear")
      }))
  })
})

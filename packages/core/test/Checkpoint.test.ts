import { expect } from "@effect/vitest"
import { Effect } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Checkpoint, Gradient, LearningRate, Loss, Model, Optimizer, Sampler, Tensor, Trainer } from "../src/index.ts"
import { floats, onDevices } from "./utils/devices.ts"

const tmpdir = Effect.sync(() => fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-")))

const values = (t: Tensor.Any) =>
  Effect.map(Tensor.toTypedArray(t), (arr) => Array.from<number | bigint>(arr).map(Number))

onDevices("Checkpoint", () => (it) => {
  it.effect("round-trips tensors of every dtype", () =>
    Effect.gen(function*() {
      const dir = yield* tmpdir
      const file = path.join(dir, "model.safetensors")
      // every dtype that runs on every device (f64 is CPU-only hardware-wise)
      yield* Tensor.save(file, {
        "w.f32": yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4]), [2, 2]),
        "w.f16": yield* Tensor.cast(yield* Tensor.fromTypedArray(new Float32Array([5, 6]), [2]), "f16"),
        "w.i64": yield* Tensor.fromTypedArray(new BigInt64Array([7n, 8n, 9n]), [3]),
        "w.u8": yield* Tensor.fromTypedArray(new Uint8Array([10, 11]), [2]),
        "w.u32": yield* Tensor.fromTypedArray(new Uint32Array([12]), [1])
      })
      const loaded = yield* Tensor.load(file)
      expect(Object.keys(loaded).sort()).toEqual(["w.f16", "w.f32", "w.i64", "w.u32", "w.u8"])
      expect(loaded["w.f32"].dtype).toBe("f32")
      expect(loaded["w.f32"].shape).toEqual([2, 2])
      expect(loaded["w.f16"].dtype).toBe("f16")
      expect(loaded["w.i64"].dtype).toBe("i64")
      expect(loaded["w.u8"].dtype).toBe("u8")
      expect(loaded["w.u32"].dtype).toBe("u32")
      expect(yield* values(loaded["w.f32"])).toEqual([1, 2, 3, 4])
      expect(yield* values(loaded["w.f16"])).toEqual([5, 6])
      expect(yield* values(loaded["w.i64"])).toEqual([7, 8, 9])
      expect(yield* values(loaded["w.u8"])).toEqual([10, 11])
      expect(yield* values(loaded["w.u32"])).toEqual([12])
    }))

  it.effect("evaluates lazy graphs during save", () =>
    Effect.gen(function*() {
      const dir = yield* tmpdir
      const file = path.join(dir, "lazy.safetensors")
      const x = yield* Tensor.fromTypedArray(floats([1, 2, 3]), [3])
      const y = yield* Tensor.fromTypedArray(floats([4, 5, 6]), [3])
      yield* Tensor.save(file, {
        sum: yield* Tensor.add(x, y),
        product: yield* Tensor.mul(x, y)
      })
      const loaded = yield* Tensor.load(file)
      expect(yield* values(loaded["sum"])).toEqual([5, 7, 9])
      expect(yield* values(loaded["product"])).toEqual([4, 10, 18])
    }))

  it.effect("round-trips non-contiguous views", () =>
    Effect.gen(function*() {
      const dir = yield* tmpdir
      const file = path.join(dir, "views.safetensors")
      const source = yield* Tensor.fromTypedArray(floats([1, 2, 3, 4, 5, 6]), [2, 3])
      yield* Tensor.save(file, { transposed: yield* Tensor.transpose(source, [1, 0]) })
      const loaded = yield* Tensor.load(file)
      expect(loaded.transposed.shape).toEqual([3, 2])
      expect(yield* values(loaded.transposed)).toEqual([1, 4, 2, 5, 3, 6])
    }))

  it.effect("loaded tensors are ordinary materialized tensors", () =>
    Effect.gen(function*() {
      const dir = yield* tmpdir
      const file = path.join(dir, "ops.safetensors")
      yield* Tensor.save(file, {
        x: yield* Tensor.fromTypedArray(floats([1, 2]), [2])
      })
      const loaded = yield* Tensor.load(file)
      const doubled = yield* Tensor.add(loaded["x"], loaded["x"])
      expect(yield* values(doubled)).toEqual([2, 4])
    }))

  it.effect("round-trips optimizer state", () =>
    Effect.gen(function*() {
      const dir = yield* tmpdir
      const file = path.join(dir, "state.safetensors")
      const optimizer = yield* Optimizer.adam()
      const p = yield* Tensor.fromTypedArray(floats([1, -1]), [2])
      const state = yield* optimizer.init([p])
      const loss = yield* Tensor.sum(yield* Tensor.mul(p, p))
      const [gp] = yield* Gradient.grad(loss, [p])
      const lr = yield* Tensor.full([], 0.1, { dtype: "f32" })
      const next = yield* optimizer.step([p], [gp], state, lr)
      const [m, v] = yield* Tensor.compute(next.stateRoots)
      yield* Tensor.save(file, { "m.0": m, "v.0": v })
      const loaded = yield* Tensor.load(file)
      expect(yield* values(loaded["m.0"])).toEqual(yield* values(m))
      expect(yield* values(loaded["v.0"])).toEqual(yield* values(v))
    }))

  it.effect("trainer checkpoint resumes bit-exactly (params, state, step)", () =>
    Effect.gen(function*() {
      const dir = yield* tmpdir
      const file = path.join(dir, "trainer.safetensors")
      const model = yield* Model.chain(
        yield* Model.linear("fc1", 2, 8),
        yield* Model.tanh,
        yield* Model.linear("fc2", 8, 1),
        yield* Model.sigmoid
      )
      const input = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
      const target = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
      const optimizer = yield* Optimizer.adam()
      const initial = yield* Tensor.compute(yield* model.init)
      const makeTrainer = (stopStep: number) =>
        Effect.gen(function*() {
          const config: Trainer.TrainConfig<Optimizer.AdamState, Tensor.TensorError> = {
            optimizer,
            lr: LearningRate.constant(0.05),
            loss: Loss.mse,
            data: { input, target },
            stop: ({ step }) => step >= stopStep
          }
          return yield* Trainer.make(model, config)
        })
      const trainer = yield* makeTrainer(8)
      const uninterrupted = yield* (yield* makeTrainer(20)).train(initial)
      const first = yield* trainer.train(initial)
      expect(first.step).toBe(8)
      yield* Checkpoint.save(file, trainer, first)
      const checkpoint = yield* Checkpoint.load(file, trainer)
      expect(checkpoint.resume.step).toBe(8)
      const resumed = yield* (yield* makeTrainer(20)).train(checkpoint.params, checkpoint.resume)
      expect(resumed.step).toBe(20)
      expect(resumed.loss).toBe(uninterrupted.loss)
      for (let i = 0; i < uninterrupted.params.length; i++) {
        expect(yield* values(resumed.params[i])).toEqual(yield* values(uninterrupted.params[i]))
      }
    }))

  it.effect("sampler state round-trips: resume continues the permutation", () =>
    Effect.gen(function*() {
      const dir = yield* tmpdir
      const file = path.join(dir, "trainer-sampler.safetensors")
      const model = yield* Model.chain(yield* Model.linear("fc1", 2, 4), yield* Model.linear("fc2", 4, 1))
      const input = yield* Tensor.fromTypedArray(floats([0, 1, 1, 0]), [2, 2])
      const target = yield* Tensor.fromTypedArray(floats([1, 0]), [2, 1])
      const optimizer = yield* Optimizer.sgd()
      const config: Trainer.TrainConfig<Optimizer.SgdState, Tensor.TensorError> = {
        optimizer,
        lr: LearningRate.constant(0.1),
        loss: Loss.mse,
        data: { input, target },
        stop: ({ step }) => step >= 2
      }
      const trainer = yield* Trainer.make(model, config)
      const samplerConfig = { length: 4 * 8 + 1, block: 8, batch: 2 }
      const sampler = yield* Sampler.make(samplerConfig)
      sampler.next()
      const trained = yield* trainer.train()
      yield* Checkpoint.saveWithSampler(file, trainer, trained, sampler)
      const expected = sampler.next()
      const checkpoint = yield* Checkpoint.loadWithSampler(file, trainer)
      const restored = yield* Sampler.restore(samplerConfig, checkpoint.sampler)
      expect(restored.next()).toEqual(expected)
      expect(checkpoint.resume.step).toBe(2)
      expect(checkpoint.sampler._tag).toBe("SamplerState")
      expect(checkpoint.sampler.config).toEqual(samplerConfig)
    }))

  it.effect("rejects missing or malformed sampler metadata", () =>
    Effect.gen(function*() {
      const dir = yield* tmpdir
      const base = path.join(dir, "sampler-base.safetensors")
      const model = yield* Model.linear("fc", 2, 1)
      const input = yield* Tensor.fromTypedArray(floats([0, 1]), [1, 2])
      const target = yield* Tensor.fromTypedArray(floats([1]), [1, 1])
      const optimizer = yield* Optimizer.sgd()
      const trainer = yield* Trainer.make(model, {
        optimizer,
        lr: LearningRate.constant(0.1),
        loss: Loss.mse,
        data: { input, target },
        stop: ({ step }) => step >= 1
      })
      const trained = yield* trainer.train()
      const samplerConfig = { length: 4 * 8 + 1, block: 8, batch: 2 }
      const sampler = yield* Sampler.make(samplerConfig)
      sampler.next()
      yield* Checkpoint.saveWithSampler(base, trainer, trained, sampler)
      const corrupt = (name: string, mutate: (entries: Record<string, Tensor.Any>) => void) =>
        Effect.gen(function*() {
          const entries: Record<string, Tensor.Any> = { ...yield* Tensor.load(base) }
          mutate(entries)
          const file = path.join(dir, name)
          yield* Tensor.save(file, entries)
          return file
        })
      const expectCheckpointError = (file: string, text: string) =>
        Effect.gen(function*() {
          const error = yield* Effect.flip(Checkpoint.loadWithSampler(file, trainer))
          expect(error._tag).toBe("CheckpointError")
          expect(error.message).toContain(text)
        })

      const missingVersion = yield* corrupt("missing-version.safetensors", (entries) => {
        delete entries["sampler:version"]
      })
      yield* expectCheckpointError(missingVersion, "missing sampler:version")

      const floatVersion = yield* Tensor.full([], 1, { dtype: "f32" })
      const malformedVersion = yield* corrupt("version-dtype.safetensors", (entries) => {
        entries["sampler:version"] = floatVersion
      })
      yield* expectCheckpointError(malformedVersion, "expected a u32 scalar")

      const futureVersion = yield* Tensor.full([], 2, { dtype: "u32" })
      const unsupportedVersion = yield* corrupt("future-version.safetensors", (entries) => {
        entries["sampler:version"] = futureVersion
      })
      yield* expectCheckpointError(unsupportedVersion, "unsupported sampler version 2")

      const vectorLength = yield* Tensor.fromTypedArray(new Uint32Array([33]), [1])
      const malformedLength = yield* corrupt("length-rank.safetensors", (entries) => {
        entries["sampler:length"] = vectorLength
      })
      yield* expectCheckpointError(malformedLength, "expected a u32 scalar")

      const matrixOrder = yield* Tensor.fromTypedArray(new Uint32Array([0, 1, 2, 3]), [2, 2])
      const malformedOrder = yield* corrupt("order-rank.safetensors", (entries) => {
        entries["sampler:order"] = matrixOrder
      })
      yield* expectCheckpointError(malformedOrder, "expected a u32 vector")

      const floatStep = yield* Tensor.full([], 1, { dtype: "f32" })
      const malformedStep = yield* corrupt("step-dtype.safetensors", (entries) => {
        entries["meta:step"] = floatStep
      })
      yield* expectCheckpointError(malformedStep, "invalid meta:step")
    }))

  it.effect("optimizer state retains the parameter placement", () =>
    Effect.gen(function*() {
      const optimizer = yield* Optimizer.adamW()
      const p = yield* Tensor.fromTypedArray(floats([1, -1]), [2])
      const state = yield* optimizer.init([p])
      for (const root of optimizer.stateRoots(state)) {
        expect(root.placement.id).toBe(p.placement.id)
      }
    }))

  it.effect("fails with TensorError on a missing file", () =>
    Effect.gen(function*() {
      const error = yield* Effect.flip(Tensor.load("/nonexistent/model.safetensors"))
      expect(error._tag).toBe("TensorError")
      expect(error.op).toBe("load")
    }))

  it.effect("fails with TensorError on an unwritable path", () =>
    Effect.gen(function*() {
      const error = yield* Effect.flip(
        Tensor.save("/nonexistent/dir/model.safetensors", {
          x: yield* Tensor.fromTypedArray(floats([1]), [1])
        })
      )
      expect(error._tag).toBe("TensorError")
      expect(error.op).toBe("save")
    }))
})

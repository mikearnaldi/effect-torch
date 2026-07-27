import { expect, layer } from "@effect/vitest"
import { Effect } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Device, Optimizer, Tensor } from "../src/index.ts"

const tmpdir = Effect.sync(() => fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-")))

const values = (t: Tensor.GenericTensor) =>
  Effect.map(Tensor.toTypedArray(t), (arr) => Array.from<number | bigint>(arr).map(Number))

layer(Device.Cpu)("Checkpoint", (it) => {
  it.effect("round-trips tensors of every dtype", () =>
    Effect.gen(function* () {
      const dir = yield* tmpdir
      const file = path.join(dir, "model.safetensors")
      yield* Tensor.save(file, {
        "w.f32": yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4]), [2, 2]),
        "w.f64": yield* Tensor.fromTypedArray(new Float64Array([5, 6]), [2]),
        "w.i64": yield* Tensor.fromTypedArray(new BigInt64Array([7n, 8n, 9n]), [3]),
        "w.u8": yield* Tensor.fromTypedArray(new Uint8Array([10, 11]), [2]),
        "w.u32": yield* Tensor.fromTypedArray(new Uint32Array([12]), [1])
      })
      const loaded = yield* Tensor.load(file)
      expect(Object.keys(loaded).sort()).toEqual(["w.f32", "w.f64", "w.i64", "w.u32", "w.u8"])
      expect(loaded["w.f32"].dtype).toBe("f32")
      expect(loaded["w.f32"].shape).toEqual([2, 2])
      expect(loaded["w.f64"].dtype).toBe("f64")
      expect(loaded["w.i64"].dtype).toBe("i64")
      expect(loaded["w.u8"].dtype).toBe("u8")
      expect(loaded["w.u32"].dtype).toBe("u32")
      expect(yield* values(loaded["w.f32"])).toEqual([1, 2, 3, 4])
      expect(yield* values(loaded["w.f64"])).toEqual([5, 6])
      expect(yield* values(loaded["w.i64"])).toEqual([7, 8, 9])
      expect(yield* values(loaded["w.u8"])).toEqual([10, 11])
      expect(yield* values(loaded["w.u32"])).toEqual([12])
    })
  )

  it.effect("evaluates lazy graphs during save", () =>
    Effect.gen(function* () {
      const dir = yield* tmpdir
      const file = path.join(dir, "lazy.safetensors")
      const x = yield* Tensor.fromTypedArray(new Float64Array([1, 2, 3]), [3])
      const y = yield* Tensor.fromTypedArray(new Float64Array([4, 5, 6]), [3])
      yield* Tensor.save(file, {
        sum: yield* Tensor.add(x, y),
        product: yield* Tensor.mul(x, y)
      })
      const loaded = yield* Tensor.load(file)
      expect(yield* values(loaded["sum"])).toEqual([5, 7, 9])
      expect(yield* values(loaded["product"])).toEqual([4, 10, 18])
    })
  )

  it.effect("loaded tensors are ordinary materialized tensors", () =>
    Effect.gen(function* () {
      const dir = yield* tmpdir
      const file = path.join(dir, "ops.safetensors")
      yield* Tensor.save(file, {
        x: yield* Tensor.fromTypedArray(new Float64Array([1, 2]), [2])
      })
      const loaded = yield* Tensor.load(file)
      const doubled = yield* Tensor.add(loaded["x"], loaded["x"])
      expect(yield* values(doubled)).toEqual([2, 4])
    })
  )

  it.effect("round-trips optimizer state", () =>
    Effect.gen(function* () {
      const dir = yield* tmpdir
      const file = path.join(dir, "state.safetensors")
      const optimizer = Optimizer.adam({ lr: 0.1 })
      const p = yield* Tensor.fromTypedArray(new Float64Array([1, -1]), [2])
      const state = yield* optimizer.init([p])
      const loss = yield* Tensor.sum(yield* Tensor.mul(p, p))
      const [gp] = yield* Tensor.grad(loss, [p])
      const next = yield* optimizer.step([p], [gp], state)
      const [m, v] = yield* Tensor.evaluate(next.stateRoots)
      yield* Tensor.save(file, { "m.0": m, "v.0": v })
      const loaded = yield* Tensor.load(file)
      expect(yield* values(loaded["m.0"])).toEqual(yield* values(m))
      expect(yield* values(loaded["v.0"])).toEqual(yield* values(v))
    })
  )

  it.effect("fails with TensorError on a missing file", () =>
    Effect.gen(function* () {
      const error = yield* Effect.flip(Tensor.load("/nonexistent/model.safetensors"))
      expect(error._tag).toBe("TensorError")
      expect(error.op).toBe("load")
    })
  )

  it.effect("fails with TensorError on an unwritable path", () =>
    Effect.gen(function* () {
      const error = yield* Effect.flip(
        Tensor.save("/nonexistent/dir/model.safetensors", {
          x: yield* Tensor.fromTypedArray(new Float64Array([1]), [1])
        })
      )
      expect(error._tag).toBe("TensorError")
      expect(error.op).toBe("save")
    })
  )
})

import { Runtime, Tensor } from "@effect-torch/core"
import { expect, it } from "@effect/vitest"
import { Effect } from "effect"
import { vi } from "vitest"
import { createRuntimeAdapter } from "../src/internal/adapter.ts"
import type { NativeAddon } from "../src/internal/native-addon.js"

it.effect("combined99 batches canvas and scalar literal upload with independently retained and released outputs", () =>
  Effect.gen(function*() {
    const old = process.env.EFFECT_TORCH_CUDA_LITERAL89
    yield* Effect.acquireRelease(
      Effect.sync(() => {
        process.env.EFFECT_TORCH_CUDA_LITERAL89 = "1"
      }),
      () =>
        Effect.sync(() => {
          if (old === undefined) delete process.env.EFFECT_TORCH_CUDA_LITERAL89
          else process.env.EFFECT_TORCH_CUDA_LITERAL89 = old
        })
    )
    class Graph {
      readonly device = "cuda:0"
      readonly storage = { representation: "dense" }
      constructor(readonly shape: Array<number>, readonly dtype: string, readonly bytes: Buffer) {}
    }
    class Value extends Graph {
      readonly clear = vi.fn()
    }
    class Token {
      cancel() {}
    }
    const returned: Array<Array<Value>> = []
    const materialize = vi.fn((roots: Array<Graph>): Promise<Array<Value>> => {
      const values = roots.map((root) => new Value(root.shape, root.dtype, Buffer.from(root.bytes)))
      returned.push(values)
      return Promise.resolve(values)
    })
    const compile = vi.fn(() => {
      throw new Error("literal batch must bypass ordinary compilation")
    })
    class Backend {
      readonly compile = compile
      readonly materializeLiterals89 = materialize
      fromBytes(bytes: Buffer, shape: Array<number>, dtype: string) {
        return new Graph(shape, dtype, Buffer.from(bytes))
      }
      full(shape: Array<number>, value: number, dtype: string) {
        return new Graph(shape, dtype, Buffer.from(new Float32Array([value]).buffer))
      }
      fromMaterialized(value: Value) {
        return value
      }
    }
    // SAFETY: only bounded literal construction, dispatch, wrapping and release are exercised.
    // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- Native double intentionally omits unrelated methods.
    const addon = { CudaRuntime: Backend, CancellationToken: Token } as unknown as NativeAddon
    const runtime = createRuntimeAdapter(addon, 0)
    yield* Effect.gen(function*() {
      const source = Uint32Array.of(17, 29)
      const inputs = [yield* Tensor.fromTypedArray(source, [1, 2]), yield* Tensor.full([], 0.5)]
      source.fill(0)
      const first = yield* Tensor.compute(inputs)
      const second = yield* Tensor.compute(inputs)
      expect(materialize).toHaveBeenCalledTimes(2)
      expect(compile).not.toHaveBeenCalled()
      expect(first.map((x) => [x.dtype, x.shape])).toEqual([["u32", [1, 2]], ["f32", []]])
      expect(returned[0]![0]!.bytes).toEqual(Buffer.from(Uint32Array.of(17, 29).buffer))
      expect(returned[0]![1]!.bytes).toEqual(Buffer.from(new Float32Array([0.5]).buffer))
      yield* Tensor.clearAll(second)
      for (const value of returned[0]!) expect(value.clear).not.toHaveBeenCalled()
      yield* Tensor.clearAll(first)
      yield* Tensor.clearAll(first)
      for (const values of returned) for (const value of values) expect(value.clear).toHaveBeenCalledTimes(1)
    }).pipe(Effect.provideService(Runtime.Runtime, runtime))
  }).pipe(Effect.scoped))

import * as BackendCpu from "@effect-torch/backend-cpu"
import { describe, expect, layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { setFlagsFromString } from "node:v8"
import { runInNewContext } from "node:vm"
import { Runtime, Tensor } from "../src/index.ts"

// These CPU-only diagnostics distinguish explicit handle release from V8
// finalization. Forced GC makes objects eligible; native finalizers remain async.
setFlagsFromString("--expose-gc")
const collectGarbage: () => void = runInNewContext("gc")

const externalMemoryBytes = Effect.gen(function*() {
  const runtime = yield* Runtime.Runtime
  const diagnostics = runtime.extensions.diagnostics
  return yield* diagnostics.externalMemoryBytes
})

layer(BackendCpu.layer)("Memory", (it) => {
  // Exact external-byte deltas are measured around live native handles, not RSS;
  // allocator caching and process-global runtime memory are outside this counter.
  describe("external memory accounting", () => {
    it.effect("clear releases the bytes immediately, without GC", () =>
      Effect.gen(function*() {
        const bytes = 4096 * 4096 * 4
        collectGarbage()
        const before = yield* externalMemoryBytes
        const [t] = yield* Tensor.compute([yield* Tensor.zeros([4096, 4096])])
        assert.strictEqual((yield* externalMemoryBytes) - before, bytes)
        yield* Tensor.clear(t)
        assert.strictEqual(yield* externalMemoryBytes, before)
      }))

    it.effect("clear is idempotent while cleared handles remain invalid", () =>
      Effect.gen(function*() {
        const [tensor] = yield* Tensor.compute([yield* Tensor.ones([4])])
        yield* Tensor.clear(tensor)
        yield* Tensor.clear(tensor)
        const error = yield* Effect.flip(Tensor.toNumberArray(tensor))
        expect(error.message).toContain("cleared")
      }))

    it.effect("clearAll releases every tensor immediately", () =>
      Effect.gen(function*() {
        const bytes = 2 * 1024 * 1024 * 4
        collectGarbage()
        const before = yield* externalMemoryBytes
        const tensors = yield* Tensor.compute([
          yield* Tensor.zeros([1024, 1024]),
          yield* Tensor.ones([1024, 1024])
        ])
        assert.strictEqual((yield* externalMemoryBytes) - before, bytes)
        yield* Tensor.clearAll(tensors)
        assert.strictEqual(yield* externalMemoryBytes, before)
      }))

    it.effect("clearAll accepts iterable ownership collections", () =>
      Effect.gen(function*() {
        const tensors = yield* Tensor.compute([yield* Tensor.ones([4]), yield* Tensor.zeros([4])])
        yield* Tensor.clearAll(new Set(tensors))
        const error = yield* Effect.flip(Tensor.toNumberArray(tensors[0]))
        expect(error.message).toContain("cleared")
      }))

    it.effect("clearAll accepts duplicate and already-cleared handles", () =>
      Effect.gen(function*() {
        const [tensor] = yield* Tensor.compute([yield* Tensor.ones([4])])
        yield* Tensor.clearAll([tensor, tensor])
        yield* Tensor.clearAll([tensor, tensor])
        const error = yield* Effect.flip(Tensor.toNumberArray(tensor))
        expect(error.message).toContain("cleared")
      }))

    it.effect("clearScoped releases an already-owned handle when its scope closes", () =>
      Effect.gen(function*() {
        const tensor = yield* Effect.scoped(
          Effect.gen(function*() {
            const [owned] = yield* Tensor.compute([yield* Tensor.ones([4])])
            expect(yield* Tensor.clearScoped(owned)).toBe(owned)
            assert.deepStrictEqual(yield* Tensor.toNumberArray(owned), [1, 1, 1, 1])
            return owned
          })
        )
        const error = yield* Effect.flip(Tensor.toNumberArray(tensor))
        expect(error.message).toContain("cleared")
      }))

    it.effect("clearAllScoped snapshots iterable identities at registration", () =>
      Effect.gen(function*() {
        const [first, second] = yield* Effect.scoped(
          Effect.gen(function*() {
            const [first, second] = yield* Tensor.compute([
              yield* Tensor.ones([4]),
              yield* Tensor.zeros([4])
            ])
            const owned = [first]
            expect(yield* Tensor.clearAllScoped(owned)).toBe(owned)
            owned[0] = second
            return [first, second] as const
          })
        )
        const firstError = yield* Effect.flip(Tensor.toNumberArray(first))
        expect(firstError.message).toContain("cleared")
        assert.deepStrictEqual(yield* Tensor.toNumberArray(second), [0, 0, 0, 0])
        yield* Tensor.clear(second)
      }))

    it.effect("use after clear is a typed error, through the handle and the graph", () =>
      Effect.gen(function*() {
        const [t] = yield* Tensor.compute([yield* Tensor.zeros([4])])
        const downstream = yield* Tensor.add(t, yield* Tensor.constant(1))
        yield* Tensor.clear(t)
        const direct = yield* Effect.flip(Tensor.toNumberArray(t))
        assert.strictEqual(direct._tag, "TensorError")
        expect(direct.message).toMatch(/cleared/)
        const viaGraph = yield* Effect.flip(Tensor.toNumberArray(downstream))
        assert.strictEqual(viaGraph._tag, "TensorError")
        expect(viaGraph.message).toMatch(/cleared/)
      }))

    it.effect("clearing a computed concrete root does not consume the source", () =>
      Effect.gen(function*() {
        const [source] = yield* Tensor.compute([yield* Tensor.ones([4])])
        const [copy] = yield* Tensor.compute([source])
        yield* Tensor.clear(copy)
        assert.deepStrictEqual(yield* Tensor.toNumberArray(source), [1, 1, 1, 1])
        yield* Tensor.clear(source)
      }))

    it.effect("lazy readback releases its private materialization", () =>
      Effect.gen(function*() {
        const before = yield* externalMemoryBytes
        const values = yield* Tensor.toTypedArray(yield* Tensor.ones([4]))
        assert.deepStrictEqual(Array.from<number | bigint>(values).map(Number), [1, 1, 1, 1])
        assert.strictEqual(yield* externalMemoryBytes, before)
      }))

    it.effect("runtime tensor bytes are reported on compute and released on GC", () =>
      Effect.gen(function*() {
        const bytes = 4096 * 4096 * 4

        const allocate = Effect.gen(function*() {
          const [t] = yield* Tensor.compute([yield* Tensor.zeros([4096, 4096])])
          return t.shape
        })

        collectGarbage()
        const before = yield* externalMemoryBytes
        assert.deepStrictEqual(yield* allocate, [4096, 4096])
        assert.strictEqual((yield* externalMemoryBytes) - before, bytes)
        // Backend finalizers run on a later event-loop turn after the handle
        // becomes unreachable; pump the loop until the bytes come back
        const waitTurn = Effect.promise(() => new Promise((resolve) => setTimeout(resolve, 100)))
        yield* Effect.gen(function*() {
          for (let i = 0; i < 30; i++) {
            if ((yield* externalMemoryBytes) === before) return
            yield* waitTurn
            collectGarbage()
          }
        })
        assert.strictEqual(yield* externalMemoryBytes, before)
      }), 20000)
  })

  describe("early free during evaluation", () => {
    // A 512x512 f32 intermediate is 1 MiB. Retaining all 2,000 chain nodes would
    // exceed 2 GiB; 512 MiB leaves headroom for runtime/JIT and allocator noise.
    it.effect("long chains free intermediates instead of holding the whole walk", () =>
      Effect.gen(function*() {
        const chain = Effect.gen(function*() {
          let x = yield* Tensor.ones([512, 512])
          for (let i = 0; i < 2000; i++) {
            x = yield* Tensor.add(x, yield* Tensor.constantLike(x, 1))
          }
          const [result] = yield* Tensor.compute([x])
          return result
        })

        collectGarbage()
        const before = process.memoryUsage().rss
        const result = yield* chain
        const grown = process.memoryUsage().rss - before
        assert.strictEqual(result.shape.length, 2)
        assert.assertTrue(
          grown < 512 * 1024 * 1024,
          `expected bounded peak memory, RSS grew by ${grown} bytes`
        )
      }), 20000)
  })
})

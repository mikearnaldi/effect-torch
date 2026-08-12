import * as BackendCpu from "@effect-torch/backend-cpu"
import { describe, expect, layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { setFlagsFromString } from "node:v8"
import { runInNewContext } from "node:vm"
import { Runtime, Tensor } from "../src/index.ts"

setFlagsFromString("--expose-gc")
const collectGarbage = runInNewContext("gc") as () => void

const externalMemoryBytes = Effect.gen(function*() {
  const runtime = yield* Runtime.Runtime
  const diagnostics = runtime.extensions.diagnostics
  if (diagnostics === undefined) {
    return yield* Effect.die(new Error("runtime does not provide memory diagnostics"))
  }
  return yield* diagnostics.externalMemoryBytes
})

layer(BackendCpu.layer)("Memory", (it) => {
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

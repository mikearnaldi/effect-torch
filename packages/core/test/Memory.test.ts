import { describe, layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { setFlagsFromString } from "node:v8"
import { runInNewContext } from "node:vm"
import native from "@effect-torch/native"
import { Device, Gradient, Tensor } from "../src/index.ts"

setFlagsFromString("--expose-gc")
const collectGarbage = runInNewContext("gc") as () => void

layer(Device.Cpu)("Memory", (it) => {
  describe("external memory accounting", () => {
    it.effect("native tensor bytes are reported on evaluate and released on GC", () =>
      Effect.gen(function* () {
        const bytes = 4096 * 4096 * 4

        const allocate = Effect.gen(function* () {
          const [t] = yield* Tensor.evaluate([yield* Tensor.zeros([4096, 4096])])
          return t.shape
        })

        collectGarbage()
        const before = native.externalMemoryBytes()
        assert.deepStrictEqual(yield* allocate, [4096, 4096])
        assert.strictEqual(native.externalMemoryBytes() - before, bytes)
        // native finalizers run on a later event-loop turn after the handle
        // becomes unreachable; pump the loop until the bytes come back
        const waitTurn = Effect.promise(() => new Promise((resolve) => setTimeout(resolve, 100)))
        yield* Effect.gen(function* () {
          for (let i = 0; i < 30; i++) {
            if (native.externalMemoryBytes() === before) return
            yield* waitTurn
            collectGarbage()
          }
        })
        assert.strictEqual(native.externalMemoryBytes(), before)
      }),
      20000
    )
  })

  describe("early free during evaluation", () => {
    it.effect("long chains free intermediates instead of holding the whole walk", () =>
      Effect.gen(function* () {
        const chain = Effect.gen(function* () {
          let x = yield* Tensor.ones([512, 512])
          for (let i = 0; i < 2000; i++) {
            x = yield* Tensor.add(x, 1)
          }
          const [result] = yield* Tensor.evaluate([x])
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
      })
    )
  })
})

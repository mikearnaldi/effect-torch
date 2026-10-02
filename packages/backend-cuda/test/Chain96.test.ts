import type { Runtime } from "@effect-torch/core"
import { expect, it } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { vi } from "vitest"
import { createRuntimeAdapter } from "../src/internal/adapter.ts"
import { type Chain96Request, execute96 } from "../src/internal/chain96.ts"
import type { Executable, NativeAddon, NativeKvPrefix, NativeTensor } from "../src/internal/native-addon.js"

class Graph {
  readonly device = "cuda:0"
  readonly storage = { representation: "dense" }
  constructor(readonly shape: Array<number> = [1], readonly dtype = "f32") {}
}
class Value extends Graph {
  readonly clear = vi.fn()
}
class Token {
  cancelled = false
  cancel() {
    this.cancelled = true
  }
}
const fixture = () =>
  Effect.gen(function*() {
    const borrowed = new Value()
    const chain = vi.fn((
      _head: Executable | null | undefined,
      _sampler: Executable,
      _bindings: Array<NativeTensor>,
      _prefixes: Array<NativeKvPrefix>,
      _slots: Array<number>,
      _active: Array<boolean>,
      _lengths: Array<number>,
      _temperature: number,
      _token?: Token
    ): Promise<Array<Value>> => Promise.resolve([new Value(), new Value()]))
    class Backend {
      full(shape: Array<number>, _value: number, dtype: string) {
        return new Graph(shape, dtype)
      }
      fromMaterialized(value: Value) {
        return new Graph(value.shape, value.dtype)
      }
      compile() {
        return {
          kvLayers: [],
          layers: 0,
          kvHeads: 0,
          headDim: 0,
          kdaLayers: 0,
          kdaHeads: 0,
          kdaHeadDim: 0,
          kdaValueDim: 0,
          convLayers: 0,
          convChannels: 0,
          convKernel: 0,
          allowsWindowEviction: false,
          diagnostics: { instructions: [], memory: {}, legalization: {}, compilePhases: [] },
          execute: () => Promise.resolve([borrowed]),
          executeChain96: chain
        }
      }
    }
    // SAFETY: only compile metadata, tensor wrapping, and chain dispatch are exercised.
    // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- Deliberately bounded native boundary mock.
    const addon = { CudaRuntime: Backend, CancellationToken: Token } as unknown as NativeAddon
    const runtime = createRuntimeAdapter(addon, 0)
    const root = yield* runtime.node({ op: "full", inputs: [], attributes: { shape: [1], value: 0, dtype: "f32" } })
    const body = yield* runtime.compile({
      roots: [root],
      state: { access: "ReadOnly", maxTokens: 16, blockSize: 16, batch: 1, kvDtype: "bf16" }
    })
    const head = yield* runtime.compile({ roots: [root], options: { optimize: true } })
    const sampler = yield* runtime.compile({ roots: [root, root], options: { optimize: true } })
    const [binding] = yield* runtime.execute(head, { bindings: [], scalars: [], runtimeValues: {} })
    const request: Chain96Request = {
      body,
      sampler,
      bindings: [binding!],
      temperature: 0.5,
      state: { access: "ReadOnly", prefixes: [], slots: [], activeMask: [], validLengths: [] }
    }
    return { runtime, request, chain, binding: binding!, head, borrowed }
  })

it.effect("chain96 rejects wrong-runtime and foreign executable or binding handles before dispatch", () =>
  Effect.gen(function*() {
    const own = yield* fixture()
    const other = yield* fixture()
    const invalid: Array<readonly [Runtime.RuntimeService, Chain96Request]> = [
      [{ ...own.runtime, identity: {} }, own.request],
      [other.runtime, own.request],
      [own.runtime, { ...own.request, sampler: other.request.sampler }],
      [own.runtime, { ...own.request, bindings: [other.binding] }],
      [own.runtime, { ...own.request, head: other.head }]
    ]
    for (const [runtime, request] of invalid) {
      expect(Exit.isFailure(yield* Effect.exit(execute96(runtime, request)))).toBe(true)
    }
    expect(own.chain).not.toHaveBeenCalled()
    expect(other.chain).not.toHaveBeenCalled()
    yield* own.runtime.release(own.binding)
    yield* other.runtime.release(other.binding)
  }))

it.effect("chain96 transfers independent outputs and borrows inputs through completion", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const values = [new Value(), new Value()]
    f.chain.mockResolvedValue(values)
    const outputs = yield* execute96(f.runtime, { ...f.request, head: f.head })
    expect(outputs).toHaveLength(2)
    expect(values.every((x) => x.clear.mock.calls.length === 0)).toBe(true)
    yield* Effect.forEach(outputs, f.runtime.release, { discard: true })
    expect(values.every((x) => x.clear.mock.calls.length === 1)).toBe(true)
    expect(f.chain.mock.calls[0]?.[7]).toBe(0.5)
    expect(f.borrowed.clear).not.toHaveBeenCalled()
    yield* f.runtime.release(f.binding)
  }))

it.effect("chain96 rejects duplicate ownership and clears each unique native output once", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const value = new Value()
    f.chain.mockResolvedValue([value, value])
    expect(Exit.isFailure(yield* Effect.exit(execute96(f.runtime, f.request)))).toBe(true)
    expect(value.clear).toHaveBeenCalledTimes(1)
    expect(f.borrowed.clear).not.toHaveBeenCalled()
    yield* f.runtime.release(f.binding)
  }))

it.effect("chain96 clears all outputs when later metadata wrapping fails", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const values = [new Value(), new Value([2])]
    f.chain.mockResolvedValue(values)
    expect(Exit.isFailure(yield* Effect.exit(execute96(f.runtime, f.request)))).toBe(true)
    expect(values.every((x) => x.clear.mock.calls.length === 1)).toBe(true)
    expect(f.borrowed.clear).not.toHaveBeenCalled()
    yield* f.runtime.release(f.binding)
  }))

it.effect("chain96 cancellation clears each unpublished late output exactly once", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const started = yield* Deferred.make<void>()
    let resolve!: (values: Array<Value>) => void
    let token: Token | undefined
    f.chain.mockImplementation(
      (_head, _sampler, _bindings, _prefixes, _slots, _active, _lengths, _temperature, cancellation) => {
        token = cancellation
        Effect.runSync(Deferred.succeed(started, undefined))
        return new Promise((done) => resolve = done)
      }
    )
    const fiber = yield* Effect.forkChild(execute96(f.runtime, f.request))
    yield* Deferred.await(started)
    yield* Fiber.interrupt(fiber)
    expect(token?.cancelled).toBe(true)
    const values = [new Value(), new Value()]
    values[0]!.clear.mockImplementation(() => {
      throw new Error("clear failed")
    })
    resolve(values)
    yield* Effect.promise(() => Promise.resolve())
    expect(values.every((x) => x.clear.mock.calls.length === 1)).toBe(true)
    expect(f.borrowed.clear).not.toHaveBeenCalled()
    yield* f.runtime.release(f.binding)
  }))

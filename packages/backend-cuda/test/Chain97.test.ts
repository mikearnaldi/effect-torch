import type { Runtime } from "@effect-torch/core"
import { expect, it } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { vi } from "vitest"
import { createRuntimeAdapter } from "../src/internal/adapter.ts"
import { type Chain97Request, execute97 } from "../src/internal/chain97.ts"
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
interface Result {
  readonly feedback: Value
  readonly statistics: Buffer
}
const result = (): Result => ({
  feedback: new Value(),
  statistics: Buffer.from(new Float32Array([1, 2, 3, 4, 5]).buffer)
})
const fixture = () =>
  Effect.gen(function*() {
    const borrowed = new Value()
    const supports = vi.fn(() => true)
    const chain = vi.fn((
      _head: Executable | null | undefined,
      _sampler: Executable,
      _canvas: Uint32Array,
      _bindings: Array<NativeTensor>,
      _prefixes: Array<NativeKvPrefix>,
      _slots: Array<number>,
      _active: Array<boolean>,
      _lengths: Array<number>,
      _temperature: number,
      _token?: Token
    ): Promise<Result> => Promise.resolve(result()))
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
          supportsChain97: supports,
          executeChain97: chain
        }
      }
    }
    // SAFETY: only compilation metadata, tensor wrapping and private chain97 dispatch are exercised.
    // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- Deliberately bounded native boundary mock.
    const addon = { CudaRuntime: Backend, CancellationToken: Token } as unknown as NativeAddon
    const runtime = createRuntimeAdapter(addon, 0)
    const root = yield* runtime.node({ op: "full", inputs: [], attributes: { shape: [1], value: 0, dtype: "f32" } })
    const stats = yield* runtime.node({ op: "full", inputs: [], attributes: { shape: [5], value: 0, dtype: "f32" } })
    const body = yield* runtime.compile({
      roots: [root],
      state: { access: "ReadOnly", maxTokens: 16, blockSize: 16, batch: 1, kvDtype: "bf16" }
    })
    const head = yield* runtime.compile({ roots: [root], options: { optimize: true } })
    const sampler = yield* runtime.compile({ roots: [root, stats], options: { optimize: true } })
    const [binding] = yield* runtime.execute(head, { bindings: [], scalars: [], runtimeValues: {} })
    const request: Chain97Request = {
      body,
      sampler,
      canvas: Uint32Array.of(17),
      bindingsWithoutCanvas: [binding!],
      temperature: 0.5,
      state: { access: "ReadOnly", prefixes: [], slots: [], activeMask: [], validLengths: [] }
    }
    return { runtime, request, chain, supports, head, borrowed, binding: binding! }
  })

it.effect("chain97 snapshots host canvas and copies offset statistics into independent host storage", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const started = yield* Deferred.make<void>()
    let finish!: (value: Result) => void
    let snapshot: Uint32Array | undefined
    f.chain.mockImplementation((_head, _sampler, canvas) => {
      snapshot = canvas
      return new Promise((resolve) => {
        finish = resolve
        Effect.runSync(Deferred.succeed(started, undefined))
      })
    })
    const fiber = yield* Effect.forkChild(execute97(f.runtime, f.request))
    yield* Deferred.await(started)
    f.request.canvas[0] = 99
    expect(snapshot).toEqual(Uint32Array.of(17))
    const native = result()
    const offset = Buffer.alloc(22)
    native.statistics.copy(offset, 1)
    const bytes = offset.subarray(1, 21)
    finish({ feedback: native.feedback, statistics: bytes })
    const output = yield* Fiber.join(fiber)
    expect(Array.from(output.statistics)).toEqual([1, 2, 3, 4, 5])
    bytes.fill(0)
    expect(Array.from(output.statistics)).toEqual([1, 2, 3, 4, 5])
    expect(native.feedback.clear).not.toHaveBeenCalled()
    expect(f.borrowed.clear).not.toHaveBeenCalled()
    yield* f.runtime.release(output.feedback)
    expect(native.feedback.clear).toHaveBeenCalledTimes(1)
    yield* f.runtime.release(f.binding)
  }))

it.effect("chain97 rejects foreign capabilities and invalid canvas before dispatch", () =>
  Effect.gen(function*() {
    const own = yield* fixture()
    const other = yield* fixture()
    const invalid: Array<readonly [Runtime.RuntimeService, Chain97Request]> = [
      [{ ...own.runtime, identity: {} }, own.request],
      [other.runtime, own.request],
      [own.runtime, { ...own.request, head: other.head }],
      [own.runtime, { ...own.request, sampler: other.request.sampler }],
      [own.runtime, { ...own.request, bindingsWithoutCanvas: [other.binding] }],
      [own.runtime, { ...own.request, canvas: new Uint32Array(0) }]
    ]
    for (const [runtime, request] of invalid) {
      expect(Exit.isFailure(yield* Effect.exit(execute97(runtime, request)))).toBe(true)
    }
    expect(own.chain).not.toHaveBeenCalled()
    expect(other.chain).not.toHaveBeenCalled()
    yield* own.runtime.release(own.binding)
    yield* other.runtime.release(other.binding)
  }))

it.effect("chain97 clears feedback on malformed statistics or wrapping failure", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    for (
      const native of [{ feedback: new Value(), statistics: Buffer.alloc(16) }, {
        feedback: new Value([2]),
        statistics: Buffer.alloc(20)
      }]
    ) {
      f.chain.mockResolvedValue(native)
      expect(Exit.isFailure(yield* Effect.exit(execute97(f.runtime, f.request)))).toBe(true)
      expect(native.feedback.clear).toHaveBeenCalledTimes(1)
      expect(f.borrowed.clear).not.toHaveBeenCalled()
    }
    yield* f.runtime.release(f.binding)
  }))

it.effect("chain97 cancellation clears unpublished late feedback exactly once", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const started = yield* Deferred.make<void>()
    let finish!: (value: Result) => void
    let token: Token | undefined
    f.chain.mockImplementation(
      (_head, _sampler, _canvas, _bindings, _prefixes, _slots, _active, _lengths, _temperature, cancellation) => {
        token = cancellation
        Effect.runSync(Deferred.succeed(started, undefined))
        return new Promise((resolve) => finish = resolve)
      }
    )
    const fiber = yield* Effect.forkChild(execute97(f.runtime, f.request))
    yield* Deferred.await(started)
    yield* Fiber.interrupt(fiber)
    expect(token?.cancelled).toBe(true)
    const native = result()
    finish(native)
    yield* Effect.promise(() => Promise.resolve())
    expect(native.feedback.clear).toHaveBeenCalledTimes(1)
    expect(f.borrowed.clear).not.toHaveBeenCalled()
    yield* f.runtime.release(f.binding)
  }))

it.effect("processed pipeline admission is pure and execution transfers native ownership", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const prepare = f.runtime.extensions.decode.prepareProcessedReadOnly!
    const descriptor = { body: f.request.body, processor: f.request.sampler, hostShape: [1, 1] }
    const other = yield* fixture()
    expect(Exit.isFailure(yield* Effect.exit(prepare({ ...descriptor, processor: other.request.sampler })))).toBe(true)
    expect(yield* prepare({ ...descriptor, hostShape: [1, 257] })).toBeUndefined()
    expect(f.supports).not.toHaveBeenCalled()
    yield* other.runtime.release(other.binding)
    f.supports.mockReturnValue(false)
    expect(yield* prepare(descriptor)).toBeUndefined()
    expect(f.chain).not.toHaveBeenCalled()
    f.supports.mockReturnValue(true)
    const prepared = yield* prepare(descriptor)
    expect(prepared).toBeDefined()
    expect(f.chain).not.toHaveBeenCalled()
    const native = result()
    f.chain.mockResolvedValue(native)
    const output = yield* prepared!.execute({
      hostInput: f.request.canvas,
      bindings: f.request.bindingsWithoutCanvas,
      state: f.request.state,
      scalar: 0.5
    })
    expect(Array.from(output.host)).toEqual([1, 2, 3, 4, 5])
    yield* f.runtime.release(output.device)
    expect(native.feedback.clear).toHaveBeenCalledTimes(1)
    yield* f.runtime.release(f.binding)
  }))

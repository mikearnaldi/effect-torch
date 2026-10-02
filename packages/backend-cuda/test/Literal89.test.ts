import type { Runtime } from "@effect-torch/core"
import { expect, it } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { afterEach, beforeEach, vi } from "vitest"
import { createRuntimeAdapter } from "../src/internal/adapter.ts"
import type { NativeAddon } from "../src/internal/native-addon.js"

beforeEach(() => vi.stubEnv("EFFECT_TORCH_CUDA_LITERAL89", "1"))
afterEach(() => vi.unstubAllEnvs())

class Literal {
  readonly device = "cuda:0"
  readonly storage = { representation: "dense" }
  constructor(readonly bytes: Uint8Array, readonly shape: Array<number>, readonly dtype: string) {}
}
class Value extends Literal {
  cleared = false
  clear() {
    this.cleared = true
  }
  readback() {
    if (this.cleared) throw new Error("cleared")
    return Promise.resolve(Buffer.from(this.bytes))
  }
}
class Token {
  cancelled = false
  cancel() {
    this.cancelled = true
  }
}
const mock = () => {
  const compile = vi.fn(() => {
    throw new Error("ordinary compile fallback")
  })
  const materialize = vi.fn((roots: Array<Literal>, token?: Token): Promise<Array<Value>> => {
    if (token?.cancelled) return Promise.reject(new Error("operation aborted"))
    return Promise.resolve(roots.map((root) => new Value(root.bytes.slice(), root.shape, root.dtype)))
  })
  class Backend {
    fromBytes(bytes: Uint8Array, shape: Array<number>, dtype: string) {
      return new Literal(bytes.slice(), shape, dtype)
    }
    full(shape: Array<number>, value: number, dtype: string) {
      const bytes = new Uint8Array(4)
      new DataView(bytes.buffer).setFloat32(0, value, true)
      return new Literal(bytes, shape, dtype)
    }
    constant(value: number, dtype: string) {
      return this.full([], value, dtype)
    }
    fromMaterialized(value: Value) {
      if (value.cleared) throw new Error("cleared")
      return new Literal(value.bytes, value.shape, value.dtype)
    }
    compile = compile
    materializeLiterals89 = materialize
  }
  // SAFETY: these tests exercise only the literal adapter boundary; unimplemented APIs must never be called.
  // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- Only literal materialization and cancellation boundaries are exercised.
  const addon = { CudaRuntime: Backend, CancellationToken: Token } as unknown as NativeAddon
  return { runtime: createRuntimeAdapter(addon, 0), addon, compile, materialize }
}
const invocation = { bindings: [], scalars: [], runtimeValues: {} }
const source = (runtime: Runtime.RuntimeService, data: Uint8Array) =>
  runtime.node({
    op: "fromBytes",
    inputs: [],
    attributes: { data, shape: [data.byteLength / 4], dtype: "u32" }
  })

it.effect("literal89 preserves native snapshots, duplicate ownership and retained outputs without compilation", () =>
  Effect.gen(function*() {
    const { runtime, compile } = mock()
    const data = new Uint8Array([7, 0, 0, 0, 19, 0, 0, 0])
    const root = yield* source(runtime, data.subarray(4))
    data.fill(0)
    const temperature = yield* runtime.node({
      op: "full",
      inputs: [],
      attributes: { shape: [], value: -0, dtype: "f32" }
    })
    const artifact = yield* runtime.compile({ roots: [root, root, temperature], options: {} })
    expect(artifact.diagnostics.instructions[0]?.kind).toBe("literal_materialize89")
    expect(artifact.diagnostics.memory.outputBytes).toBe(8)
    const first = yield* runtime.execute(artifact, invocation)
    const second = yield* runtime.execute(artifact, invocation)
    expect(first[0]).not.toBe(first[1])
    yield* runtime.release(first[0]!)
    yield* Effect.forEach(second, (value) => runtime.release(value), { discard: true })
    expect(Array.from(new Uint8Array(yield* runtime.readback(first[1]!)))).toEqual([19, 0, 0, 0])
    expect(Array.from(new Uint8Array(yield* runtime.readback(first[2]!)))).toEqual([0, 0, 0, 128])
    expect(compile).not.toHaveBeenCalled()
    yield* runtime.release(first[1]!)
    yield* runtime.release(first[2]!)
  }))

it.effect("literal89 falls back for options, operations, disabled flags, oversized roots and concrete identities", () =>
  Effect.gen(function*() {
    const { runtime, compile } = mock()
    const root = yield* source(runtime, new Uint8Array(4))
    const artifact = yield* runtime.compile({ roots: [root] })
    const [concrete] = yield* runtime.execute(artifact, invocation)
    const vector = yield* runtime.node({ op: "full", inputs: [], attributes: { shape: [1], value: 1, dtype: "f32" } })
    const big = yield* source(runtime, new Uint8Array(65540))
    for (const roots of [[concrete!], [vector], [big], Array(9).fill(root)]) {
      expect(Exit.isFailure(yield* Effect.exit(runtime.compile({ roots })))).toBe(true)
    }
    expect(Exit.isFailure(yield* Effect.exit(runtime.compile({ roots: [root], options: { optimize: true } })))).toBe(
      true
    )
    expect(Exit.isFailure(
      yield* Effect.exit(runtime.compile({
        roots: [root],
        state: {
          access: "ReadOnly",
          maxTokens: 16,
          blockSize: 16,
          kvDtype: "bf16",
          batch: 1
        }
      }))
    )).toBe(true)
    vi.stubEnv("EFFECT_TORCH_CUDA_LITERAL89", "0")
    expect(Exit.isFailure(yield* Effect.exit(runtime.compile({ roots: [root] })))).toBe(true)
    expect(compile).toHaveBeenCalledTimes(7)
    yield* runtime.release(concrete!)
  }))

it.effect("literal89 validates runtime ownership and rejects invocation inputs before materialization", () =>
  Effect.gen(function*() {
    const { runtime, addon, materialize } = mock()
    const root = yield* source(runtime, new Uint8Array(4))
    const foreign = createRuntimeAdapter(addon, 0)
    expect(Exit.isFailure(yield* Effect.exit(foreign.compile({ roots: [root] })))).toBe(true)
    const artifact = yield* runtime.compile({ roots: [root] })
    expect(Exit.isFailure(yield* Effect.exit(runtime.execute(artifact, { ...invocation, scalars: [1] })))).toBe(true)
    expect(materialize).not.toHaveBeenCalled()
  }))

it.effect("literal89 cancels native work and clears unpublished late outputs", () =>
  Effect.gen(function*() {
    const { runtime, materialize } = mock()
    const started = yield* Deferred.make<void>()
    let resolve!: (values: Array<Value>) => void
    let token: Token | undefined
    materialize.mockImplementation((_roots, cancellation) => {
      token = cancellation
      Effect.runSync(Deferred.succeed(started, undefined))
      return new Promise((done) => {
        resolve = done
      })
    })
    const root = yield* source(runtime, new Uint8Array(4))
    const artifact = yield* runtime.compile({ roots: [root] })
    const fiber = yield* Effect.forkChild(runtime.execute(artifact, invocation))
    yield* Deferred.await(started)
    yield* Fiber.interrupt(fiber)
    expect(token?.cancelled).toBe(true)
    const output = new Value(new Uint8Array(4), [1], "u32")
    resolve([output])
    yield* Effect.promise(() => Promise.resolve())
    expect(output.cleared).toBe(true)
  }))

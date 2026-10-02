import type { Runtime } from "@effect-torch/core"
import { expect, it } from "@effect/vitest"
import { Effect, Exit } from "effect"
import { vi } from "vitest"
import { createRuntimeAdapter } from "../src/internal/adapter.ts"
import type { NativeAddon } from "../src/internal/native-addon.js"
import { forkRequestRng99 } from "../src/internal/requestRng99.ts"

class Graph {
  readonly device = "cuda:0"
  readonly storage = { representation: "dense" }
  constructor(readonly shape: Array<number>, readonly dtype: string) {}
}
const fixture = () =>
  Effect.gen(function*() {
    class Program {
      readonly kvLayers = []
      readonly layers = 0
      readonly kvHeads = 0
      readonly headDim = 0
      readonly kdaLayers = 0
      readonly kdaHeads = 0
      readonly kdaHeadDim = 0
      readonly kdaValueDim = 0
      readonly convLayers = 0
      readonly convChannels = 0
      readonly convKernel = 0
      readonly allowsWindowEviction = false
      readonly diagnostics = { instructions: [], memory: {}, legalization: {}, compilePhases: [] }
      readonly forkRequestRng99 = vi.fn((
        _peer: Program,
        _seed: number
      ): Array<Program> => [new Program(), new Program()])
    }
    const programs: Array<Program> = []
    class Backend {
      full(shape: Array<number>, _value: number, dtype: string) {
        return new Graph(shape, dtype)
      }
      compile() {
        const program = new Program()
        programs.push(program)
        return program
      }
    }
    // SAFETY: this bounded mock exercises compilation metadata and private executable fork ownership only.
    // oxlint-disable-next-line anti-slop/no-chained-type-assertions -- Native test double intentionally omits unrelated methods.
    const addon = { CudaRuntime: Backend } as unknown as NativeAddon
    const runtime = createRuntimeAdapter(addon, 0)
    const root = yield* runtime.node({ op: "full", inputs: [], attributes: { shape: [1], value: 0, dtype: "f32" } })
    const compile = () =>
      runtime.compile({
        roots: [root],
        state: { access: "ReadOnly", maxTokens: 16, blockSize: 16, batch: 1, kvDtype: "bf16" }
      })
    const initial = yield* compile()
    const refinement = yield* compile()
    const stateless = yield* runtime.compile({ roots: [root] })
    return { runtime, initial, refinement, stateless, programs }
  })

it.effect("request RNG99 forks preserve metadata and register distinct runtime-owned executable capabilities", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const [initial, refinement] = yield* forkRequestRng99(f.runtime, {
      initial: f.initial,
      refinement: f.refinement,
      seed: 42
    })
    expect(f.programs[0]!.forkRequestRng99).toHaveBeenCalledWith(f.programs[1], 42)
    expect(initial).not.toBe(f.initial)
    expect(refinement).not.toBe(f.refinement)
    expect(initial).not.toBe(refinement)
    expect(initial.diagnostics).toBe(f.initial.diagnostics)
    expect(initial.state).toBe(f.initial.state)
    // A further fork proves both returned capabilities were registered and point
    // to their own native wrappers, without consuming the original templates.
    const again = yield* forkRequestRng99(f.runtime, { initial, refinement, seed: 7 })
    expect(again[0]).not.toBe(initial)
    expect(f.programs[0]!.forkRequestRng99).toHaveBeenCalledTimes(1)
  }))

it.effect("request RNG99 rejects wrong domains, non-readonly templates and invalid seeds before native fork", () =>
  Effect.gen(function*() {
    const f = yield* fixture()
    const other = yield* fixture()
    const requests = [
      { initial: other.initial, refinement: f.refinement, seed: 1 },
      { initial: f.initial, refinement: other.refinement, seed: 1 },
      { initial: f.stateless, refinement: f.refinement, seed: 1 },
      { initial: f.initial, refinement: f.refinement, seed: -1 },
      { initial: f.initial, refinement: f.refinement, seed: 0x1_0000_0000 },
      { initial: f.initial, refinement: f.refinement, seed: 1.5 }
    ]
    for (const request of requests) {
      expect(Exit.isFailure(yield* Effect.exit(forkRequestRng99(f.runtime, request)))).toBe(true)
    }
    const foreign: Runtime.RuntimeService = { ...f.runtime, identity: {} }
    expect(Exit.isFailure(yield* Effect.exit(forkRequestRng99(foreign, requests[3]!)))).toBe(true)
    expect(f.programs[0]!.forkRequestRng99).not.toHaveBeenCalled()
  }))

import { expect } from "@effect/vitest"
import { Effect } from "effect"
import { Sampler } from "../src/index.ts"
import { it } from "@effect/vitest"

const config = { length: 4 * 8 + 1, block: 8, batch: 2 }

it.effect("every window exactly once per epoch, reshuffled at the boundary", () =>
  Effect.gen(function* () {
    const sampler = yield* Sampler.make(config)
    const first: Array<number> = []
    for (let i = 0; i < 2; i++) first.push(...sampler.next())
    expect(first.slice().sort((a, b) => a - b)).toEqual([0, 8, 16, 24])
    // the epoch boundary: the next draw starts a new permutation
    const second = sampler.next()
    expect(second.every((start) => start % 8 === 0 && start < 32)).toBe(true)
  })
)

it.effect("restore continues the permutation exactly where it stopped", () =>
  Effect.gen(function* () {
    const sampler = yield* Sampler.make(config)
    const before = sampler.next()
    const state = sampler.state()
    const expected = sampler.next()
    const restored = yield* Sampler.restore(config, { order: state.order, cursor: state.cursor, epoch: state.epoch })
    expect(restored.next()).toEqual(expected)
    expect(before).not.toEqual(expected)
    expect(restored.state().epoch).toBe(state.epoch)
  })
)

it.effect("rejects impossible configs and mismatched states", () =>
  Effect.gen(function* () {
    const bad = yield* Effect.flip(Sampler.make({ length: 4, block: 8, batch: 1 }))
    expect(bad._tag).toBe("SamplerError")
    const sampler = yield* Sampler.make(config)
    const mismatched = yield* Effect.flip(
      Sampler.restore({ ...config, block: 4 }, { ...sampler.state(), order: new Uint32Array(7) })
    )
    expect(mismatched._tag).toBe("SamplerError")
  })
)

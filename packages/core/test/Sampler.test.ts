import { expect, it } from "@effect/vitest"
import { Effect } from "effect"
import { Sampler } from "../src/index.ts"

const config = { length: 4 * 8 + 1, block: 8, batch: 2 }

it.effect("every window exactly once in a divisible epoch, reshuffled at the boundary", () =>
  Effect.gen(function*() {
    const sampler = yield* Sampler.make(config)
    const first: Array<number> = []
    for (let i = 0; i < 2; i++) first.push(...sampler.next())
    expect(first.slice().sort((a, b) => a - b)).toEqual([0, 8, 16, 24])
    // the epoch boundary: the next draw starts a new permutation
    const second = sampler.next()
    expect(second.every((start) => start % 8 === 0 && start < 32)).toBe(true)
  }))

it.effect("skips a trailing partial batch consistently across legacy epoch migration", () =>
  Effect.gen(function*() {
    const partialConfig = { length: 5 * 8 + 1, block: 8, batch: 2 }
    const sampler = yield* Sampler.make(partialConfig)
    const complete = [...sampler.next(), ...sampler.next()]
    expect(new Set(complete).size).toBe(4)
    const boundary = sampler.state()
    expect(boundary).toMatchObject({ cursor: 4, epoch: 1 })

    const restored = yield* Sampler.restoreCheckpoint(partialConfig, {
      _tag: "LegacySamplerState",
      order: boundary.order,
      cursor: boundary.cursor,
      epoch: boundary.epoch
    }, 2)
    expect(restored.state()).toMatchObject({ cursor: 4, epoch: 1 })
    expect(restored.next()).toHaveLength(2)
    expect(restored.state()).toMatchObject({ cursor: 2, epoch: 2 })
  }))

it.effect("restore continues the permutation exactly where it stopped", () =>
  Effect.gen(function*() {
    const sampler = yield* Sampler.make(config)
    const before = sampler.next()
    const state = sampler.state()
    const expected = sampler.next()
    const restored = yield* Sampler.restore(config, state)
    expect(restored.next()).toEqual(expected)
    expect(before).not.toEqual(expected)
    expect(restored.state().epoch).toBe(state.epoch)
  }))

it.effect("rejects impossible configs and mismatched states", () =>
  Effect.gen(function*() {
    const bad = yield* Effect.flip(Sampler.make({ length: 4, block: 8, batch: 1 }))
    expect(bad._tag).toBe("SamplerError")
    for (
      const malformedConfig of [
        { ...config, length: Number.NaN },
        { ...config, block: 1.5 },
        { ...config, batch: 0x1_0000_0000 }
      ]
    ) {
      const malformed = yield* Effect.flip(Sampler.make(malformedConfig))
      expect(malformed._tag).toBe("SamplerError")
    }
    const sampler = yield* Sampler.make(config)
    const mismatched = yield* Effect.flip(
      Sampler.restore({ ...config, block: 4 }, { ...sampler.state(), order: new Uint32Array(7) })
    )
    expect(mismatched._tag).toBe("SamplerError")
  }))

it.effect("rejects corrupt state before restoring", () =>
  Effect.gen(function*() {
    const sampler = yield* Sampler.make(config)
    const state = sampler.state()
    const duplicate = state.order.slice()
    duplicate[1] = duplicate[0]
    for (
      const corrupt of [
        { ...state, order: duplicate },
        { ...state, cursor: 1 },
        { ...state, cursor: state.order.length + state.config.batch },
        { ...state, epoch: 0 }
      ]
    ) {
      const error = yield* Effect.flip(Sampler.restore(config, corrupt))
      expect(error._tag).toBe("SamplerError")
    }
  }))

it.effect("state snapshots do not alias mutable sampler storage", () =>
  Effect.gen(function*() {
    const sampler = yield* Sampler.make(config)
    const snapshot = sampler.state()
    const order = snapshot.order.slice()
    snapshot.order.fill(0)
    Object.assign(snapshot.config, { block: 1 })
    expect(sampler.state().order).toEqual(order)
    expect(sampler.state().config).toEqual(config)

    const restoreState = sampler.state()
    const restored = yield* Sampler.restore(config, restoreState)
    const restoredOrder = restored.state().order
    restoreState.order.fill(0)
    expect(restored.state().order).toEqual(restoredOrder)
  }))

it.effect("sampler construction snapshots config and restore inputs", () =>
  Effect.gen(function*() {
    const mutableConfig = { ...config }
    const make = Sampler.make(mutableConfig)
    mutableConfig.batch = 1
    const sampler = yield* make
    expect(sampler.next()).toHaveLength(config.batch)

    const restoreConfig = { ...config }
    const restoreState = sampler.state()
    const restore = Sampler.restore(restoreConfig, restoreState)
    restoreConfig.batch = 1
    Object.assign(restoreState.config, { batch: 1 })
    restoreState.order.fill(0)
    const restored = yield* restore
    expect(restored.state().config).toEqual(config)
    expect(new Set(restored.state().order).size).toBe(restored.state().order.length)
  }))

it.effect("checkpoint restore rejects batch changes and migrates matching legacy state", () =>
  Effect.gen(function*() {
    const sampler = yield* Sampler.make(config)
    sampler.next()
    const state = sampler.state()
    const changed = yield* Effect.flip(
      Sampler.restoreCheckpoint({ ...config, batch: 1 }, state, 1)
    )
    expect(changed._tag).toBe("SamplerError")

    const legacy: Sampler.LegacySamplerState = {
      _tag: "LegacySamplerState",
      order: state.order,
      cursor: state.cursor,
      epoch: state.epoch
    }
    const restored = yield* Sampler.restoreCheckpoint(config, legacy, 1)
    expect(restored.state().config).toEqual(config)
    const legacyChanged = yield* Effect.flip(
      Sampler.restoreCheckpoint({ ...config, batch: 1 }, legacy, 1)
    )
    expect(legacyChanged._tag).toBe("SamplerError")
  }))

it.effect("migrates matching legacy state after the first epoch", () =>
  Effect.gen(function*() {
    const sampler = yield* Sampler.make(config)
    sampler.next()
    sampler.next()
    const boundary = sampler.state()
    expect(boundary).toMatchObject({ cursor: 4, epoch: 1 })
    const boundaryRestore = yield* Sampler.restoreCheckpoint(config, {
      _tag: "LegacySamplerState",
      order: boundary.order,
      cursor: boundary.cursor,
      epoch: boundary.epoch
    }, 2)
    boundaryRestore.next()
    expect(boundaryRestore.state()).toMatchObject({ cursor: 2, epoch: 2 })

    sampler.next()
    const state = sampler.state()
    expect(state).toMatchObject({ cursor: 2, epoch: 2 })
    const restored = yield* Sampler.restoreCheckpoint(config, {
      _tag: "LegacySamplerState",
      order: state.order,
      cursor: state.cursor,
      epoch: state.epoch
    }, 3)
    expect(restored.state()).toMatchObject({ cursor: 2, epoch: 2, config })
  }))

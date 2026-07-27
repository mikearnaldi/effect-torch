import { expect, layer } from "@effect/vitest"
import { Effect } from "effect"
import { Device } from "../src/index.ts"

layer(Device.Best)("Device", (it) => {
  it.effect("Best provides an available device", () =>
    Effect.gen(function* () {
      const device = yield* Device.CurrentDevice
      expect(yield* Device.isAvailable(device)).toBe(true)
    })
  )
})

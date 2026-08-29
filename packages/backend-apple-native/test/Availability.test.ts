import { Runtime } from "@effect-torch/core"
import { Effect } from "effect"
import { expect, it } from "vitest"

it("imports safely and defers the unsupported-platform error", async () => {
  // Import after overriding the platform so the backend uses the test setting.
  // Calling isAvailable then checks deferred native-addon selection.
  const platform = Object.getOwnPropertyDescriptor(process, "platform")!
  try {
    Object.defineProperty(process, "platform", { value: "linux" })
    const backend = await import("../src/index.ts")

    expect(await Effect.runPromise(backend.isAvailable)).toBe(false)
    await expect(
      Effect.runPromise(Runtime.Runtime.pipe(Effect.provide(backend.layer())))
    ).rejects.toThrow(/supports only platform "darwin"/)
  } finally {
    Object.defineProperty(process, "platform", platform)
  }
})

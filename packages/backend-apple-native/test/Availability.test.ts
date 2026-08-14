import { Effect } from "effect"
import { expect, it } from "vitest"

it("is import-safe on unsupported platforms", async () => {
  // The dynamic import obtains the backend API after the platform override;
  // invoking isAvailable then exercises deferred native-loader selection.
  const platform = Object.getOwnPropertyDescriptor(process, "platform")!
  try {
    Object.defineProperty(process, "platform", { value: "linux" })
    const backend = await import("../src/index.ts")

    expect(await Effect.runPromise(backend.isAvailable)).toBe(false)
    expect(() => backend.makeRuntime()).toThrow(/supports only platform "darwin"/)
  } finally {
    Object.defineProperty(process, "platform", platform)
  }
})

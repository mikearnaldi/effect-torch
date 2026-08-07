import { Runtime } from "@effect-torch/core"
import { Effect, Layer } from "effect"
import { makeRuntime as makeRuntimeAdapter } from "./internal/adapter.js"
import { loadNative } from "./internal/native.js"

export const isAvailable: Effect.Effect<boolean> = Effect.sync(() => {
  try {
    return loadNative().isAvailable()
  } catch {
    return false
  }
})

let runtime: Runtime.RuntimeService | undefined

export const makeRuntime = (): Runtime.RuntimeService => runtime ??= makeRuntimeAdapter(loadNative())

export const layer: Layer.Layer<Runtime.Runtime> = Layer.effect(Runtime.Runtime, Effect.sync(makeRuntime))

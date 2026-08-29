/**
 * Apple Metal package entry point. Importing this module does not select or load
 * a native addon. Selection occurs when {@link isAvailable} runs or when
 * {@link layer} is built.
 */
import { Runtime } from "@effect-torch/core"
import { Effect, Layer } from "effect"
import { createRuntimeAdapter } from "./internal/adapter.js"
import { loadNative } from "./internal/native.js"

/**
 * Checks whether the Apple native backend is available.
 *
 * Each run loads and caches the native addon if needed, then calls its
 * availability probe. It returns `false` if addon selection or loading throws,
 * if the probe throws, or if the probe reports that Metal is unavailable. Once
 * loaded, the addon remains cached even when the probe returns `false` or
 * throws.
 *
 * The Effect reruns the native probe on every execution. `true` means that an
 * enumerated Metal device could create a command queue and shared event at that
 * time. Later runtime operations can still fail.
 *
 * @since 0.1.0
 * @category utilities
 */
export const isAvailable: Effect.Effect<boolean> = Effect.sync(() => {
  try {
    return loadNative().isAvailable()
  } catch {
    return false
  }
})

let runtime: Runtime.RuntimeService | undefined

/**
 * Provides the cached Apple native runtime singleton as a reusable Layer.
 *
 * The Layer waits until it is built to load the addon and construct the runtime.
 * Successful builds share the same service. If loading or construction throws,
 * `Effect.sync` reports the exception as a defect and does not cache the failed
 * construction. A later build can retry. Building the Layer does not run
 * {@link isAvailable}.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer: Layer.Layer<Runtime.Runtime> = Layer.effect(
  Runtime.Runtime,
  Effect.sync(() => runtime ??= createRuntimeAdapter(loadNative()))
)

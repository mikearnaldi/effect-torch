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

const runtimes = new Map<number, Runtime.RuntimeService>()

/**
 * Selects the Metal device used by a runtime layer.
 *
 * @since 0.1.0
 * @category models
 */
export interface LayerOptions {
  /** Zero-based Metal device ordinal. Defaults to `0`. */
  readonly device?: number
}

/**
 * Provides a cached Apple native runtime for one Metal device.
 *
 * The Layer waits until it is built to load the addon, validate the requested
 * device, and construct the runtime. Successful builds for the same ordinal
 * share one service. Omitted options select `metal:0`. Failed construction is
 * not cached, so a later build can retry.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer = (options: LayerOptions = {}): Layer.Layer<Runtime.Runtime> =>
  Layer.effect(
    Runtime.Runtime,
    Effect.sync(() => {
      const device = options.device ?? 0
      if (!Number.isSafeInteger(device) || device < 0 || device > 0xffff_ffff) {
        throw new Error(`Metal device ordinal must be an integer in [0, 4294967295]; received ${device}`)
      }
      const cached = runtimes.get(device)
      if (cached !== undefined) return cached
      const native = loadNative()
      if (!native.isDeviceAvailable(device)) {
        throw new Error(`Metal device metal:${device} is unavailable`)
      }
      const runtime = createRuntimeAdapter(native, device)
      runtimes.set(device, runtime)
      return runtime
    })
  )

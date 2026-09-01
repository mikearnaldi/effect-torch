/**
 * CUDA package entry point. Importing this module does not load the native
 * addon; selection occurs when {@link isAvailable} runs or {@link layer} builds.
 */
import { Runtime } from "@effect-torch/core"
import { Effect, Layer } from "effect"
import { createRuntimeAdapter } from "./internal/adapter.js"
import { loadNative } from "./internal/native.js"

/** Checks whether device zero and the CUDA 12.9 NVRTC path are usable. */
export const isAvailable: Effect.Effect<boolean> = Effect.sync(() => {
  try {
    return loadNative().isAvailable()
  } catch {
    return false
  }
})

const runtimes = new Map<number, Runtime.RuntimeService>()

/** CUDA device selection for {@link layer}. */
export interface LayerOptions {
  /** Zero-based CUDA device ordinal. Defaults to `0`. */
  readonly device?: number
}

/** Provides a cached CUDA runtime for one device ordinal. */
export const layer = (options: LayerOptions = {}): Layer.Layer<Runtime.Runtime> =>
  Layer.effect(
    Runtime.Runtime,
    Effect.sync(() => {
      const device = options.device ?? 0
      if (!Number.isSafeInteger(device) || device < 0 || device > 0xffff_ffff) {
        throw new Error(`CUDA device ordinal must be an integer in [0, 4294967295]; received ${device}`)
      }
      const cached = runtimes.get(device)
      if (cached !== undefined) return cached
      const native = loadNative()
      const runtime = createRuntimeAdapter(native, device)
      runtimes.set(device, runtime)
      return runtime
    })
  )

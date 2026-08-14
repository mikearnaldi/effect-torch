/**
 * Apple Metal package entry point. Importing this module is platform-safe:
 * native selection stays deferred until {@link isAvailable}, {@link makeRuntime},
 * or {@link layer} is evaluated.
 */
import { Runtime } from "@effect-torch/core"
import { Effect, Layer } from "effect"
import { makeRuntime as makeRuntimeAdapter } from "./internal/adapter.js"
import { loadNative } from "./internal/native.js"

/**
 * An availability probe for the Apple native backend.
 *
 * Running this Effect attempts to load and memoize the native addon as a side
 * effect, then invokes its availability probe. It returns `false` when addon
 * selection or loading throws, when the native probe throws, or when the probe
 * itself reports that Metal is unavailable. A successfully loaded addon remains
 * cached even if the probe subsequently returns `false` or throws.
 * Each Effect execution reruns the native probe; `true` means that, at that
 * moment, an enumerated Metal device could create a command queue and shared
 * event. It does not guarantee that later runtime operations will succeed.
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
 * Synchronously creates and memoizes the Apple native runtime singleton for
 * this package module.
 *
 * The first call loads the native addon on demand and constructs the runtime
 * adapter; successful calls thereafter return the same service. This function
 * can throw synchronously when the host is unsupported, the addon cannot be
 * loaded, or runtime construction fails. A failed construction is not memoized,
 * so a later call can retry. This function does not run {@link isAvailable} or
 * initialize a Metal device; probe first when availability must be established.
 *
 * @since 0.1.0
 * @category constructors
 */
export const makeRuntime = (): Runtime.RuntimeService => runtime ??= makeRuntimeAdapter(loadNative())

/**
 * A reusable Layer that provides the memoized Apple native runtime singleton.
 *
 * Addon loading and runtime construction are deferred until the Layer is built.
 * Every successful build installs the same service returned by
 * {@link makeRuntime}. If loading or construction throws, `Effect.sync` reports
 * the exception as a defect and no runtime is memoized, so a later independent
 * build can retry. Building the Layer does not run {@link isAvailable}.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer: Layer.Layer<Runtime.Runtime> = Layer.effect(Runtime.Runtime, Effect.sync(makeRuntime))

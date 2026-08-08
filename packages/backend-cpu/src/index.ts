import { Runtime } from "@effect-torch/core"
import { Effect, Layer } from "effect"
import { makeRuntime as makeRuntimeAdapter } from "./internal/adapter.js"
import native from "./internal/native.js"

let runtime: Runtime.RuntimeService | undefined

/**
 * Returns the memoized CPU runtime singleton for this package module.
 *
 * Importing `@effect-torch/backend-cpu` eagerly selects the native binary for
 * the current platform, architecture, and Linux C library, then loads that
 * addon. Unsupported hosts and addon-loading failures therefore throw during
 * module evaluation, before this function can be called. Runtime adapter
 * construction is synchronous on the first call and is cached only after it
 * succeeds; later calls return the same service, while a failed construction
 * can be retried.
 *
 * @since 0.1.0
 * @category constructors
 */
export const makeRuntime = (): Runtime.RuntimeService => runtime ??= makeRuntimeAdapter(native)

/**
 * A reusable Layer that provides the memoized CPU runtime singleton.
 *
 * Runtime adapter construction is deferred until the Layer is built, and
 * successful builds all install the same service returned by {@link makeRuntime}.
 * Native binary loading has already happened at package import time. If runtime
 * construction throws during a build, `Effect.sync` reports the exception as a
 * defect; no runtime is memoized, so a later independent build can retry.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer: Layer.Layer<Runtime.Runtime> = Layer.effect(Runtime.Runtime, Effect.sync(makeRuntime))

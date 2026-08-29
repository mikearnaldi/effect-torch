/**
 * CPU package entry point. Importing this module loads the native addon for the
 * current host. {@link layer} provides its cached adapter as an Effect
 * dependency.
 */
import { Runtime } from "@effect-torch/core"
import { Effect, Layer } from "effect"
import { createRuntimeAdapter } from "./internal/adapter.js"
import native from "./internal/native.js"

let runtime: Runtime.RuntimeService | undefined

/**
 * Provides the cached CPU runtime as a reusable Layer.
 *
 * The Layer defers runtime adapter construction until it is built. Successful
 * builds share the same service. The package import has already loaded the
 * native binary. If construction throws, `Effect.sync` reports the exception as
 * a defect. The failed runtime is not cached, so a later build can retry.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer: Layer.Layer<Runtime.Runtime> = Layer.effect(
  Runtime.Runtime,
  Effect.sync(() => runtime ??= createRuntimeAdapter(native))
)

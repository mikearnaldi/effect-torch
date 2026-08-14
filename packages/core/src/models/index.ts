/**
 * Built-in model architectures exported by `@effect-torch/core/models`.
 *
 * Architecture modules define configuration-to-model translation only: they
 * neither load artifacts nor mutate a registry. `Gguf.load` resolves a source-
 * qualified identifier in the active `Registry`, invokes the matching
 * architecture, validates its exact parameter catalog against the artifact,
 * and returns loaded parameters in model order. `Registry.layer` owns default
 * registration of the architectures exported here.
 *
 * @since 0.1.0
 */
import * as MuseGlimmer from "./MuseGlimmer.ts"

/**
 * The Muse-Glimmer namespace, containing its exact GGUF registry identifier and
 * load-only architecture definition. It is registered by the default
 * `Registry.layer` as `gguf:muse-glimmer`; importing this namespace alone does
 * not register it.
 *
 * @since 0.1.0
 * @category models
 */
export { MuseGlimmer }

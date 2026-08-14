/**
 * Built-in model architectures and target-coupled checkpoint artifacts exported
 * by `@effect-torch/core/models`.
 *
 * Ordinary architecture modules are resolved by `Gguf.load` through the active
 * `Registry`. Target-coupled modules such as DFlash use a dedicated loader and
 * return proposer artifacts instead of pretending to be one-input models.
 *
 * @since 0.1.0
 */
import * as DFlash from "./DFlash.ts"
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
export { DFlash, MuseGlimmer }

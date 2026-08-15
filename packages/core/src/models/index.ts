/**
 * Built-in model architectures and target-coupled checkpoint artifacts exported
 * by `@effect-torch/core/models`.
 *
 * Each module owns its exact GGUF architecture check and exposes a dedicated
 * `loadGGUF` function. Target-coupled modules such as DFlash return proposer
 * artifacts instead of pretending to be one-input models.
 *
 * @since 0.1.0
 */
import * as DFlash from "./DFlash.ts"
import * as MuseGlimmer from "./MuseGlimmer.ts"

/**
 * The Muse-Glimmer namespace, containing its exact GGUF architecture value,
 * model factory, and dedicated loader.
 *
 * @since 0.1.0
 * @category models
 */
export { DFlash, MuseGlimmer }

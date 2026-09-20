/**
 * Built-in model architectures exported by `@effect-torch/core/models`.
 *
 * Each module owns its architecture validation and checkpoint loading.
 *
 * @since 0.1.0
 */
import * as DiffusionGemma from "./DiffusionGemma.ts"
import * as MuseGlimmer from "./MuseGlimmer.ts"

/**
 * DiffusionGemma configuration, loading, graph builders, and inference.
 *
 * @since 0.1.0
 * @category models
 */
export { DiffusionGemma }

/**
 * The Muse-Glimmer namespace, containing its exact GGUF architecture value,
 * model factory, and dedicated loader.
 *
 * @since 0.1.0
 * @category models
 */
export { MuseGlimmer }

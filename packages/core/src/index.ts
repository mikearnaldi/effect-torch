/**
 * Safetensors checkpoint persistence for parameters, positional optimizer
 * tensor state, global training step, and optional sampler continuation state.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Checkpoint from "./Checkpoint.ts"

/**
 * Streaming conversational inference over structured chat messages.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Chat from "./Chat.ts"

/**
 * Reverse-mode automatic differentiation and lazy graph-to-graph transforms.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Gradient from "./Gradient.ts"

/**
 * Native GGUF v3 inspection, registry resolution, validation, and owned tensor
 * loading.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Gguf from "./Gguf.ts"

/**
 * Synchronous composable learning-rate schedules over global training steps.
 *
 * @since 0.1.0
 * @category modules
 */
export * as LearningRate from "./LearningRate.ts"

/**
 * Lazy loss graph builders, reduction policies, and validation boundaries.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Loss from "./Loss.ts"

/**
 * Functional model construction, composition, parameter management, and execution.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Model from "./Model.ts"

/**
 * Effect-environment model-architecture registration, exact-key lookup, and
 * default Layers.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Registry from "./Registry.ts"

/**
 * Pure optimizer update graphs, explicit tensor-state schemas, and gradient
 * clipping.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Optimizer from "./Optimizer.ts"

/**
 * Backend-neutral runtime services, handles, and execution contracts.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Runtime from "./Runtime.ts"

/**
 * Restorable shuffled epoch sampling of next-token training windows.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Sampler from "./Sampler.ts"

/**
 * Lazy tensor graph construction, evaluation, compilation, transfer, and persistence.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Tensor from "./Tensor.ts"

/**
 * Stateful compiled and uncompiled training loops with explicit resume and
 * cache semantics.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Trainer from "./Trainer.ts"

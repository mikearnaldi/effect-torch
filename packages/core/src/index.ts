/**
 * Checkpoint persistence for parameters, optimizer state, training progress,
 * and sampler state.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Checkpoint from "./Checkpoint.ts"

/**
 * Reverse-mode automatic differentiation and graph-gradient transforms.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Gradient from "./Gradient.ts"

/**
 * Composable learning-rate schedules for training steps.
 *
 * @since 0.1.0
 * @category modules
 */
export * as LearningRate from "./LearningRate.ts"

/**
 * Loss functions and reduction policies for lazy tensor graphs.
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
 * Pure optimizer graph transforms and optimizer-state management.
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
 * Restorable shuffled epoch sampling for token-sequence training.
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
 * Compiled and uncompiled training-loop construction and execution.
 *
 * @since 0.1.0
 * @category modules
 */
export * as Trainer from "./Trainer.ts"

/**
 * Learning-rate schedules. A schedule is a plain function from the step
 * number (0-based) to the learning rate for that step. The training loop
 * ({@link Trainer.train}) evaluates it every step and lifts the value to a
 * 0-d tensor that flows into the optimizer update as graph data — the
 * rate is never baked into the optimizer or its graph, so one optimizer
 * (and one compiled step) serves the whole schedule:
 *
 * ```ts
 * const trainer = yield* Trainer.make(model, {
 *   optimizer: yield* Optimizer.adam(),
 *   lr: LearningRate.withWarmup(LearningRate.cosine(1e-3, { totalSteps }), 100),
 *   ...
 * })
 * yield* trainer.train()
 * ```
 *
 * @since 0.1.0
 */

/**
 * A learning-rate schedule: maps the 0-based step number to a rate.
 *
 * @since 0.1.0
 * @category models
 */
export type LearningRate = (step: number) => number

/**
 * A constant rate.
 *
 * @since 0.1.0
 * @category constructors
 */
export const constant = (lr: number): LearningRate => () => lr

/**
 * Exponential decay: `initial * decayRate ^ (step / decaySteps)`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const exponential = (
  initial: number,
  options: { readonly decayRate: number; readonly decaySteps: number }
): LearningRate => {
  if (initial <= 0 || options.decayRate <= 0 || options.decaySteps < 1) {
    throw new Error(
      `exponential: expected initial > 0, decayRate > 0, decaySteps >= 1, got ${initial}, ${options.decayRate}, ${options.decaySteps}`
    )
  }
  return (step) => initial * Math.pow(options.decayRate, step / options.decaySteps)
}

/**
 * Step decay: `initial * dropFactor ^ floor(step / dropEvery)`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const stepwise = (
  initial: number,
  options: { readonly dropFactor: number; readonly dropEvery: number }
): LearningRate => {
  if (initial <= 0 || options.dropFactor <= 0 || options.dropEvery < 1) {
    throw new Error(
      `stepwise: expected initial > 0, dropFactor > 0, dropEvery >= 1, got ${initial}, ${options.dropFactor}, ${options.dropEvery}`
    )
  }
  return (step) => initial * Math.pow(options.dropFactor, Math.floor(step / options.dropEvery))
}

/**
 * Cosine annealing from `initial` to `minLr` (default `0`) over
 * `totalSteps`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const cosine = (
  initial: number,
  options: { readonly totalSteps: number; readonly minLr?: number }
): LearningRate => {
  const minLr = options.minLr ?? 0
  if (initial <= 0 || minLr < 0 || minLr >= initial || options.totalSteps < 1) {
    throw new Error(
      `cosine: expected initial > minLr >= 0 and totalSteps >= 1, got ${initial}, ${minLr}, ${options.totalSteps}`
    )
  }
  return (step) => {
    const progress = Math.min(step / options.totalSteps, 1)
    return minLr + (initial - minLr) * (1 + Math.cos(Math.PI * progress)) / 2
  }
}

/**
 * Linear warmup from `0` to the base schedule's rate over `warmupSteps`,
 * then the base schedule (re-indexed so its step 0 starts after warmup).
 *
 * @since 0.1.0
 * @category combinators
 */
export const withWarmup = (base: LearningRate, warmupSteps: number): LearningRate => {
  if (!Number.isInteger(warmupSteps) || warmupSteps < 1) {
    throw new Error(`withWarmup: warmupSteps must be a positive integer, got ${warmupSteps}`)
  }
  return (step) => {
    if (step < warmupSteps) {
      return base(0) * ((step + 1) / warmupSteps)
    }
    return base(step - warmupSteps)
  }
}

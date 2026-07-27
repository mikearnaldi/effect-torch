/**
 * Learning-rate schedules. A schedule is a plain function from the step
 * number (0-based) to the learning rate for that step; since the rate is a
 * scalar constant embedded in the graph at step-construction time, no
 * framework machinery is needed — evaluate the schedule in JS and pass the
 * result to the optimizer factory:
 *
 * ```ts
 * const lr = Schedule.withWarmup(Schedule.cosine(1e-3, { totalSteps }), 100)
 * for (let t = 0; t < totalSteps; t++) {
 *   const optimizer = Optimizer.adam({ lr: lr(t) })
 *   ...
 * }
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
export type Schedule = (step: number) => number

/**
 * A constant rate.
 *
 * @since 0.1.0
 * @category constructors
 */
export const constant = (lr: number): Schedule => () => lr

/**
 * Exponential decay: `initial * decayRate ^ (step / decaySteps)`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const exponential = (
  initial: number,
  options: { readonly decayRate: number; readonly decaySteps: number }
): Schedule => {
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
): Schedule => {
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
): Schedule => {
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
export const withWarmup = (base: Schedule, warmupSteps: number): Schedule => {
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

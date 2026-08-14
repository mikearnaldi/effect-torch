/**
 * Pure learning-rate schedule constructors. A schedule is a synchronous plain
 * function from the 0-based global step to that step's learning rate. The
 * training loop (`Trainer.train`) evaluates it every step and binds the value
 * as a 0-d tensor/scalar input to the optimizer update. The rate is not baked
 * into optimizer configuration or a compiled trainer program, so one optimizer
 * and one compiled signature serve the whole schedule:
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
 * A fresh trainer run calls schedules with consecutive non-negative integer
 * steps; a caller-supplied resume step is not validated. A `LearningRate` is an
 * ordinary function and does not enforce its domain, catch exceptions, or
 * validate returned rates. Constructor validation is synchronous and limited
 * to the comparisons documented below; except for `warmupSteps`, values are not
 * separately checked for finiteness or integrality. The guards use direct
 * JavaScript comparisons, so `NaN` is not rejected.
 *
 * @since 0.1.0
 */

/**
 * A synchronous learning-rate schedule mapping a 0-based step number to a
 * rate. Fresh trainer runs supply consecutive non-negative integers; resumed
 * runs continue from the unvalidated `Resume.step` supplied by the caller.
 * Direct callers are responsible for that domain and for returning a finite
 * rate suitable for their optimizer. A thrown exception becomes a defect when
 * called by the trainer, not a typed training failure.
 *
 * @since 0.1.0
 * @category models
 */
export type LearningRate = (step: number) => number

/**
 * A constant rate. Neither `lr` nor the call's step is validated.
 *
 * @since 0.1.0
 * @category constructors
 */
export const constant = (lr: number): LearningRate => () => lr

/**
 * Exponential decay: `initial * decayRate ^ (step / decaySteps)`.
 * `initial` and `decayRate` must be greater than `0`; `decaySteps` must be at
 * least `1` but need not be an integer. Violating one of those comparisons
 * throws synchronously. Schedule calls are not validated.
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
 * `initial` and `dropFactor` must be greater than `0`; `dropEvery` must be at
 * least `1` but need not be an integer. Violating one of those comparisons
 * throws synchronously. Schedule calls are not validated.
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
 * `totalSteps`. It uses `min(step / totalSteps, 1)`, so with finite
 * configuration and finite non-negative steps, calls at or beyond
 * `totalSteps` stay at `minLr`. Positive infinity is also upper-clamped;
 * negative steps and `NaN` are not normalized. Requires
 * `initial > minLr >= 0` and `totalSteps >= 1`, without requiring an integral
 * `totalSteps`. Violating one of those comparisons throws synchronously, but
 * values are not separately checked for finiteness.
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
 * Prepends `warmupSteps` linearly scaled calls to `base(0)`. For
 * `warmupSteps = N`, steps `0..N-1` return
 * `base(0) / N, 2 * base(0) / N, ..., base(0)`; step `N` then starts the
 * re-indexed base schedule at `base(0)`, followed by `base(1)`, and so on.
 * Thus the sequence does not emit zero and emits `base(0)` twice at the
 * boundary. `warmupSteps` must be a positive integer or construction throws
 * synchronously. `base(0)` is called afresh for every warmup step rather than
 * memoized, so callers should supply a pure schedule. The base schedule's
 * outputs and call steps are not validated.
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

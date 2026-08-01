/**
 * Loss functions. Every loss takes the prediction and the target and
 * returns a lazy graph value; with the default `reduction: "mean"` (or
 * `"sum"`) the result is a scalar ready for {@link Gradient.grad}, while
 * `reduction: "none"` returns the unreduced per-element loss for custom
 * weighting or masking.
 *
 * @since 0.1.0
 */
import { Effect } from "effect"
import { dual } from "effect/Function"
import type { CurrentDevice } from "./Device.ts"
import * as Tensor from "./Tensor.ts"

/**
 * How a loss aggregates its per-element values: `"mean"` (default) or
 * `"sum"` over all dimensions, or `"none"` to keep the elementwise loss.
 *
 * @since 0.1.0
 * @category models
 */
export type Reduction = "mean" | "sum" | "none"

/**
 * Common options for all losses.
 *
 * @since 0.1.0
 * @category models
 */
export interface LossOptions {
  readonly reduction?: Reduction
}

const applyReduction = (
  self: Tensor.Any,
  reduction: Reduction
): Effect.Effect<Tensor.Lazy, Tensor.TensorError> => {
  switch (reduction) {
    case "mean":
      return Tensor.mean(self)
    case "sum":
      return Tensor.sum(self)
    case "none":
      return Effect.flatMap(Tensor.constantLike(self, 0), (zero) => Tensor.add(self, zero))
  }
}

const isTarget = (value: unknown): boolean =>
  value !== undefined && value !== null && typeof value === "object" && "_tag" in value

const dualLoss = <T, O, R = never>(
  impl: (
    pred: Tensor.Any,
    target: T,
    options: O | undefined
  ) => Effect.Effect<Tensor.Lazy, Tensor.TensorError, R>
): {
  (target: T, options?: O): (pred: Tensor.Any) => Effect.Effect<
    Tensor.Lazy,
    Tensor.TensorError,
    R
  >
  (
    pred: Tensor.Any,
    target: T,
    options?: O
  ): Effect.Effect<Tensor.Lazy, Tensor.TensorError, R>
} =>
  dual(
    (args) => args.length === 3 || (args.length === 2 && isTarget(args[1])),
    impl
  ) as never

/**
 * Mean squared error: `(pred - target)^2`.
 *
 * @since 0.1.0
 * @category losses
 */
export const mse = dualLoss<Tensor.Any, LossOptions>((pred, target, options) =>
  Effect.gen(function* () {
    const err = yield* Tensor.sub(pred, target)
    return yield* applyReduction(yield* Tensor.square(err), options?.reduction ?? "mean")
  })
)

/**
 * L1 loss: `|pred - target|`.
 *
 * @since 0.1.0
 * @category losses
 */
export const l1 = dualLoss<Tensor.Any, LossOptions>((pred, target, options) =>
  Effect.gen(function* () {
    const err = yield* Tensor.sub(pred, target)
    return yield* applyReduction(yield* Tensor.abs(err), options?.reduction ?? "mean")
  })
)

/**
 * Options for {@link huber}. `delta` is the point where the loss switches
 * from quadratic to linear, default `1`.
 *
 * @since 0.1.0
 * @category models
 */
export interface HuberOptions extends LossOptions {
  readonly delta?: number
}

/**
 * Huber loss: quadratic for `|pred - target| <= delta`, linear beyond it.
 * Smooth like MSE near zero, robust to outliers like L1.
 *
 * @since 0.1.0
 * @category losses
 */
export const huber = dualLoss<Tensor.Any, HuberOptions>((pred, target, options) =>
  Effect.gen(function* () {
    const delta = options?.delta ?? 1
    if (delta <= 0) {
      return yield* new Tensor.TensorError({ op: "huber", message: `huber: delta must be positive, got ${delta}` })
    }
    const e = yield* Tensor.abs(yield* Tensor.sub(pred, target))
    const quad = yield* Tensor.minimum(e, yield* Tensor.constantLike(e, delta))
    const lin = yield* Tensor.sub(e, quad)
    const loss = yield* Tensor.add(
      yield* Tensor.mul(yield* Tensor.square(quad), yield* Tensor.constantLike(quad, 0.5)),
      yield* Tensor.mul(lin, yield* Tensor.constantLike(lin, delta))
    )
    return yield* applyReduction(loss, options?.reduction ?? "mean")
  })
)

/**
 * Options for {@link binaryCrossEntropy}. `fromLogits` applies the
 * numerically stable logits form (`max(x, 0) - x * y + log1p(exp(-|x|))`)
 * instead of taking probabilities.
 *
 * @since 0.1.0
 * @category models
 */
export interface BinaryCrossEntropyOptions extends LossOptions {
  readonly fromLogits?: boolean
}

/**
 * Binary cross entropy between probabilities (or logits with
 * `fromLogits`) and 0/1 targets. Probabilities are clamped to
 * `[1e-12, 1 - 1e-12]` to keep the logs finite.
 *
 * @since 0.1.0
 * @category losses
 */
export const binaryCrossEntropy = dualLoss<Tensor.Any, BinaryCrossEntropyOptions>(
  (pred, target, options) =>
    Effect.gen(function* () {
      if (options?.fromLogits === true) {
        const head = yield* Tensor.relu(pred)
        const mid = yield* Tensor.mul(pred, target)
        const tail = yield* Tensor.log1p(yield* Tensor.exp(yield* Tensor.neg(yield* Tensor.abs(pred))))
        const loss = yield* Tensor.sub(yield* Tensor.add(head, tail), mid)
        return yield* applyReduction(loss, options?.reduction ?? "mean")
      }
      const p = yield* Tensor.clamp(pred, { min: 1e-12, max: 1 - 1e-12 })
      const oneMinusP = yield* Tensor.add(yield* Tensor.neg(p), yield* Tensor.constantLike(p, 1))
      const oneMinusY = yield* Tensor.add(yield* Tensor.neg(target), yield* Tensor.constantLike(target, 1))
      const pos = yield* Tensor.mul(yield* Tensor.log(p), target)
      const neg = yield* Tensor.mul(yield* Tensor.log(oneMinusP), oneMinusY)
      const loss = yield* Tensor.neg(yield* Tensor.add(pos, neg))
      return yield* applyReduction(loss, options?.reduction ?? "mean")
    })
)

const checkClassTargets = (
  op: string,
  input: Tensor.Any,
  targets: Tensor.Any
): Effect.Effect<number, Tensor.TensorError> =>
  Effect.gen(function* () {
    if (input.shape.length < 1) {
      return yield* new Tensor.TensorError({ op, message: `${op}: expected rank >= 1, got rank ${input.shape.length}` })
    }
    if (targets.dtype !== "i64" && targets.dtype !== "u32") {
      return yield* new Tensor.TensorError({ op, message: `${op}: targets must be i64 or u32 class indexes, got ${targets.dtype}` })
    }
    const expected = input.shape.slice(0, -1)
    if (expected.length !== targets.shape.length || expected.some((d, i) => d !== targets.shape[i])) {
      return yield* new Tensor.TensorError({
        op,
        message: `${op}: targets shape [${targets.shape}] does not match input shape [${input.shape}] minus the class dimension`
      })
    }
    return input.shape[input.shape.length - 1]
  })

/**
 * Cross entropy between class logits and `i64` class-index targets:
 * `nll(logSoftmax(logits), targets)`. The class dimension is the last one.
 * The default `mean` reduction delegates to {@link Tensor.crossEntropy}
 * (whose backward is not second-order differentiable); `sum` and `none`
 * are computed from log-softmax directly.
 *
 * @since 0.1.0
 * @category losses
 */
export const crossEntropy = dualLoss<Tensor.Any, LossOptions, CurrentDevice>((logits, targets, options) =>
  Effect.gen(function* () {
    const depth = yield* checkClassTargets("crossEntropy", logits, targets)
    if (logits.dtype !== "f32" && logits.dtype !== "f64") {
      return yield* new Tensor.TensorError({
        op: "crossEntropy",
        message: `crossEntropy: logits must be f32 or f64, got ${logits.dtype}`
      })
    }
    if ((options?.reduction ?? "mean") === "mean") {
      return yield* Tensor.crossEntropy(logits, { target: targets })
    }
    const oneHot = yield* Tensor.oneHot(targets, depth, { dtype: logits.dtype })
    const logProbs = yield* Tensor.logSoftmax(logits, { dims: [-1] })
    const nll = yield* Tensor.neg(yield* Tensor.sum(yield* Tensor.mul(oneHot, logProbs), { dims: [-1] }))
    return yield* applyReduction(nll, options?.reduction ?? "mean")
  })
)

/**
 * Negative log likelihood between log-probabilities and `i64` class-index
 * targets. The class dimension is the last one.
 *
 * @since 0.1.0
 * @category losses
 */
export const nll = dualLoss<Tensor.Any, LossOptions, CurrentDevice>((logProbs, targets, options) =>
  Effect.gen(function* () {
    const depth = yield* checkClassTargets("nll", logProbs, targets)
    if (logProbs.dtype !== "f32" && logProbs.dtype !== "f64") {
      return yield* new Tensor.TensorError({
        op: "nll",
        message: `nll: log-probabilities must be f32 or f64, got ${logProbs.dtype}`
      })
    }
    const oneHot = yield* Tensor.oneHot(targets, depth, { dtype: logProbs.dtype })
    const picked = yield* Tensor.neg(yield* Tensor.sum(yield* Tensor.mul(oneHot, logProbs), { dims: [-1] }))
    return yield* applyReduction(picked, options?.reduction ?? "mean")
  })
)

/**
 * Kullback-Leibler divergence `sum(target * (log(target) - log(pred)))`,
 * with `pred` given as log-probabilities. Elements where `target` is `0`
 * contribute `0`.
 *
 * @since 0.1.0
 * @category losses
 */
export const klDiv = dualLoss<Tensor.Any, LossOptions>((logPred, target, options) =>
  Effect.gen(function* () {
    const zero = yield* Tensor.constantLike(target, 0)
    const elements = yield* Tensor.where(
      yield* Tensor.gt(target, zero),
      yield* Tensor.mul(target, yield* Tensor.sub(yield* Tensor.log(target), logPred)),
      zero
    )
    return yield* applyReduction(elements, options?.reduction ?? "mean")
  })
)

/**
 * Hinge loss for ±1 targets: `max(0, 1 - target * pred)`.
 *
 * @since 0.1.0
 * @category losses
 */
export const hinge = dualLoss<Tensor.Any, LossOptions>((pred, target, options) =>
  Effect.gen(function* () {
    const margin = yield* Tensor.add(
      yield* Tensor.neg(yield* Tensor.mul(pred, target)),
      yield* Tensor.constantLike(pred, 1)
    )
    return yield* applyReduction(
      yield* Tensor.maximum(margin, yield* Tensor.constantLike(margin, 0)),
      options?.reduction ?? "mean"
    )
  })
)

/**
 * Options for {@link cosineEmbedding}. `margin` is the value below which
 * negative (−1 target) pairs are pushed, default `0`.
 *
 * @since 0.1.0
 * @category models
 */
export interface CosineEmbeddingOptions extends LossOptions {
  readonly margin?: number
}

/**
 * Cosine embedding loss between two tensors over their last dimension:
 * `1 - cos(a, b)` for `+1` targets, `max(0, cos(a, b) - margin)` for `−1`
 * targets. `targets` must be a tensor of ±1 broadcastable against the
 * leading dimensions.
 *
 * @since 0.1.0
 * @category losses
 */
export const cosineEmbeddingLoss = (
  a: Tensor.Any,
  b: Tensor.Any,
  targets: Tensor.Any,
  options: CosineEmbeddingOptions = {}
): Effect.Effect<Tensor.Lazy, Tensor.TensorError> =>
  Effect.gen(function* () {
    const margin = options.margin ?? 0
    const dot = yield* Tensor.sum(yield* Tensor.mul(a, b), { dims: [-1] })
    const na = yield* Tensor.sqrt(yield* Tensor.sum(yield* Tensor.square(a), { dims: [-1] }))
    const nb = yield* Tensor.sqrt(yield* Tensor.sum(yield* Tensor.square(b), { dims: [-1] }))
    const cos = yield* Tensor.div(dot, yield* Tensor.add(yield* Tensor.mul(na, nb), yield* Tensor.constantLike(dot, 1e-12)))
    const positive = yield* Tensor.add(yield* Tensor.neg(cos), yield* Tensor.constantLike(cos, 1))
    const negative = yield* Tensor.maximum(
      yield* Tensor.add(cos, yield* Tensor.constantLike(cos, -margin)),
      yield* Tensor.constantLike(cos, 0)
    )
    const loss = yield* Tensor.where(yield* Tensor.gt(targets, yield* Tensor.constantLike(targets, 0)), positive, negative)
    return yield* applyReduction(loss, options.reduction ?? "mean")
  })

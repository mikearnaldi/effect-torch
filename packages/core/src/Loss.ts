/**
 * Loss graph builders. They borrow predictions and targets and return lazy
 * graph values without evaluating or clearing inputs. Shape, dtype, placement,
 * and backend constraints are checked by each wrapper and its tensor
 * operations. Failures use the returned `Tensor.TensorError` channel.
 * Value-domain checks are limited, and failures that require evaluation remain
 * deferred until compute or execution.
 *
 * `"mean"` and `"sum"` reduce every dimension of each function's
 * loss-specific unreduced result to a scalar suitable for `Gradient.grad`;
 * `"none"` preserves that result, which may be elementwise or
 * per-example/per-position depending on the loss.
 *
 * @since 0.1.0
 */
import { Effect } from "effect"
import { dual } from "effect/Function"
import type * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * How a loss aggregates its unreduced values: `"mean"` (default) or `"sum"`
 * over all resulting dimensions, or `"none"` to preserve the loss-specific
 * unreduced shape. This is not a batch-only or KL `batchmean` reduction.
 *
 * @since 0.1.0
 * @category models
 */
export type Reduction = "mean" | "sum" | "none"

/**
 * Common options for loss graph builders. An unsupported runtime value is not
 * normalized; use one of the three declared reductions.
 *
 * @since 0.1.0
 * @category models
 */
export interface LossOptions {
  /** Aggregation applied to the unreduced loss; defaults to `"mean"`. */
  readonly reduction?: Reduction
}

const applyReduction = (
  self: Tensor.Any,
  reduction: Reduction
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> => {
  switch (reduction) {
    case "mean":
      return Tensor.mean(self)
    case "sum":
      return Tensor.sum(self)
    case "none":
      return Effect.flatMap(Tensor.constantLike(self, 0), (zero) => Tensor.add(self, zero))
  }
}

const isTarget = (value: unknown): value is { readonly _tag: unknown } =>
  value !== undefined && value !== null && typeof value === "object" && "_tag" in value

const dualLoss = <T, O, R = Runtime.Runtime>(
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
  // SAFETY: Effect.dual implements these overloads; the predicate selects by arity and target position.
  dual(
    (args) => args.length === 3 || (args.length === 2 && isTarget(args[1])),
    impl
  ) as never

/**
 * Squared error `(pred - target)^2`, followed by the requested reduction.
 * With `"none"`, the result has the broadcast shape of `pred` and `target`.
 * Dtype and placement compatibility follow the underlying tensor operations.
 *
 * @since 0.1.0
 * @category losses
 */
export const mse = dualLoss<Tensor.Any, LossOptions>((pred, target, options) =>
  Effect.gen(function*() {
    const err = yield* Tensor.sub(pred, target)
    return yield* applyReduction(yield* Tensor.square(err), options?.reduction ?? "mean")
  })
)

/**
 * Absolute error `|pred - target|`, followed by the requested reduction.
 * With `"none"`, the result has the broadcast shape of `pred` and `target`.
 * Dtype and placement compatibility follow the underlying tensor operations.
 *
 * @since 0.1.0
 * @category losses
 */
export const l1 = dualLoss<Tensor.Any, LossOptions>((pred, target, options) =>
  Effect.gen(function*() {
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
  /**
   * Positive transition point between quadratic and linear regions; defaults
   * to `1`. Values `<= 0` fail. Finiteness is not checked: `NaN` or positive
   * infinity is accepted and may produce non-finite results.
   */
  readonly delta?: number
}

/**
 * Huber loss: `0.5 * e^2` for `e = |pred - target| <= delta`, and
 * `delta * (e - 0.5 * delta)` beyond it, followed by the requested reduction.
 * `delta <= 0` fails with {@link Tensor.TensorError} when the effect runs;
 * the composed operations perform other tensor compatibility checks.
 *
 * @since 0.1.0
 * @category losses
 */
export const huber = dualLoss<Tensor.Any, HuberOptions>((pred, target, options) =>
  Effect.gen(function*() {
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
  /**
   * Treat `pred` as logits and use the stable logits formula; defaults to
   * `false`.
   */
  readonly fromLogits?: boolean
}

/**
 * Binary cross entropy between probabilities, or logits when `fromLogits` is
 * true, and broadcast-compatible targets. Tensor operations check prediction
 * and target dtype and placement compatibility. Target values
 * are not checked to be in `[0, 1]`. The probability path requests a clamp to
 * `[1e-12, 1 - 1e-12]`, but those bounds are represented in the tensor dtype
 * and can round to `0` or `1`; finite logarithms are therefore not guaranteed,
 * especially at reduced precision. Prefer the stable logits form when
 * possible.
 *
 * @since 0.1.0
 * @category losses
 */
export const binaryCrossEntropy = dualLoss<Tensor.Any, BinaryCrossEntropyOptions>(
  (pred, target, options) =>
    Effect.gen(function*() {
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
  Effect.gen(function*() {
    if (input.shape.length < 1) {
      return yield* new Tensor.TensorError({ op, message: `${op}: expected rank >= 1, got rank ${input.shape.length}` })
    }
    if (targets.dtype !== "i64" && targets.dtype !== "u32") {
      return yield* new Tensor.TensorError({
        op,
        message: `${op}: targets must be i64 or u32 class indexes, got ${targets.dtype}`
      })
    }
    const expected = input.shape.slice(0, -1)
    if (expected.length !== targets.shape.length || expected.some((d, i) => d !== targets.shape[i])) {
      return yield* new Tensor.TensorError({
        op,
        message:
          `${op}: targets shape [${targets.shape}] does not match input shape [${input.shape}] minus the class dimension`
      })
    }
    return input.shape[input.shape.length - 1]
  })

/**
 * Cross entropy between class logits and `i64` or `u32` class-index targets.
 * The class dimension is last and the target shape must equal the leading
 * logits shape. The default `mean` delegates to {@link Tensor.crossEntropy}:
 * `i64` target `-100` is excluded, an empty active set or other out-of-range
 * active index fails during evaluation, and the backward is not second-order
 * differentiable. `sum` and `none` instead use one-hot log-softmax; `none`
 * returns the target shape, and any index unmatched by `0..classes-1`
 * contributes zero because that path does not validate index values. Logit
 * dtype, target dtype/shape, and placement mismatches fail while building the
 * graph; active target values on the fused mean path are validated only during
 * execution.
 *
 * @since 0.1.0
 * @category losses
 */
export const crossEntropy = dualLoss<Tensor.Any, LossOptions>((logits, targets, options) =>
  Effect.gen(function*() {
    const depth = yield* checkClassTargets("crossEntropy", logits, targets)
    if (logits.dtype !== "f32" && logits.dtype !== "f64" && logits.dtype !== "bf16") {
      return yield* new Tensor.TensorError({
        op: "crossEntropy",
        message: `crossEntropy: logits must be f32, f64 or bf16, got ${logits.dtype}`
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
 * Negative log likelihood between `f32` or `f64` log-probabilities and `i64`
 * or `u32` class-index targets. The class dimension is last and `"none"`
 * returns the target shape. Index values are not range-checked. An index
 * unmatched by `0..classes-1` produces zero loss. This path materializes a
 * one-hot graph and has no ignore-index convention.
 *
 * @since 0.1.0
 * @category losses
 */
export const nll = dualLoss<Tensor.Any, LossOptions>((logProbs, targets, options) =>
  Effect.gen(function*() {
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
 * Elementwise Kullback-Leibler terms
 * `target * (log(target) - logPred)`, with predictions already expressed as
 * log-probabilities. Non-positive target elements contribute zero, including
 * negative values; target validity is not otherwise checked. The default is
 * the mean over every broadcast result element, not the conventional summed
 * KL or batch mean. Select `"sum"` explicitly for a total divergence.
 *
 * @since 0.1.0
 * @category losses
 */
export const klDiv = dualLoss<Tensor.Any, LossOptions>((logPred, target, options) =>
  Effect.gen(function*() {
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
 * Hinge expression `max(0, 1 - target * pred)`, intended for targets `-1` and
 * `1`. Target values are not validated.
 *
 * @since 0.1.0
 * @category losses
 */
export const hinge = dualLoss<Tensor.Any, LossOptions>((pred, target, options) =>
  Effect.gen(function*() {
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
 * Options for {@link cosineEmbeddingLoss}. `margin` is the maximum cosine at
 * which the non-positive-target branch has zero loss, default `0`.
 *
 * @since 0.1.0
 * @category models
 */
export interface CosineEmbeddingOptions extends LossOptions {
  /** Threshold used by the non-positive-target branch; defaults to `0` and is not validated. */
  readonly margin?: number
}

/**
 * Cosine embedding loss over the last dimension; inputs must therefore have a
 * last dimension, with rank/shape/dtype/placement compatibility delegated to
 * the composed tensor operations. The cosine denominator is
 * `norm(a) * norm(b) + 1e-12`, with the epsilon represented in the input
 * dtype. In dtypes that represent it, zero vectors produce a cosine of zero;
 * in low-precision dtypes it may round to zero and permit non-finite results.
 * Every result is otherwise slightly biased rather than using a clamped norm.
 * Any positive target selects `1 - cosine`; zero or a negative target selects
 * `max(0, cosine - margin)`. Values are not restricted to `-1` and `1`. Inputs
 * and targets follow the underlying broadcasting rules; this wrapper does not
 * independently validate equal feature widths or target shape.
 *
 * @since 0.1.0
 * @category losses
 */
export const cosineEmbeddingLoss = (
  a: Tensor.Any,
  b: Tensor.Any,
  targets: Tensor.Any,
  options: CosineEmbeddingOptions = {}
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const margin = options.margin ?? 0
    const dot = yield* Tensor.sum(yield* Tensor.mul(a, b), { dims: [-1] })
    const na = yield* Tensor.sqrt(yield* Tensor.sum(yield* Tensor.square(a), { dims: [-1] }))
    const nb = yield* Tensor.sqrt(yield* Tensor.sum(yield* Tensor.square(b), { dims: [-1] }))
    const cos = yield* Tensor.div(
      dot,
      yield* Tensor.add(yield* Tensor.mul(na, nb), yield* Tensor.constantLike(dot, 1e-12))
    )
    const positive = yield* Tensor.add(yield* Tensor.neg(cos), yield* Tensor.constantLike(cos, 1))
    const negative = yield* Tensor.maximum(
      yield* Tensor.add(cos, yield* Tensor.constantLike(cos, -margin)),
      yield* Tensor.constantLike(cos, 0)
    )
    const loss = yield* Tensor.where(
      yield* Tensor.gt(targets, yield* Tensor.constantLike(targets, 0)),
      positive,
      negative
    )
    return yield* applyReduction(loss, options.reduction ?? "mean")
  })

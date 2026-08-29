/**
 * Reverse-mode autodiff and graph-to-graph transforms. The backward transform
 * runs natively on an existing lazy graph; there is no JavaScript function
 * tracing or function transformation. Adjoint expressions are ordinary graph
 * nodes and remain lazy until computed or compiled. Higher-order derivatives
 * can be requested by applying {@link grad} again, but succeed only when the
 * native autodiff transform defines adjoints for every forward and backward
 * node involved.
 *
 * @since 0.1.0
 */
import { Data, Effect } from "effect"
import * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * Error raised by {@link grad} for wrapper-level loss or target validation.
 * Unsupported or non-differentiable nodes, incompatible runtimes or
 * placements, and invalid backend results are reported as
 * {@link Tensor.TensorError}; the `"not-differentiable"` reason is reserved and
 * is not currently emitted.
 *
 * @since 0.1.0
 * @category errors
 */
export class GradError extends Data.TaggedError("GradError")<{
  /**
   * Validation class for a loss or differentiation target that cannot enter
   * autodiff.
   */
  readonly reason: "non-scalar-output" | "non-float-dtype" | "not-differentiable"
  /**
   * Human-readable explanation, including the offending shape, dtype, or
   * operation.
   */
  readonly detail: string
}> {}

const isFloatDtype = (dtype: string): boolean =>
  dtype === "f32" || dtype === "f64" || dtype === "f16" || dtype === "bf16"

const fromBackend = <A>(
  op: string,
  effect: Effect.Effect<A, Runtime.BackendError>
): Effect.Effect<A, Tensor.TensorError> =>
  effect.pipe(
    Effect.mapError((error) => new Tensor.TensorError({ op, message: error.message, backend: error }))
  )

const validateResult = (
  op: string,
  runtime: Runtime.RuntimeService,
  value: Runtime.LazyTensorHandle,
  expected: {
    readonly shape: ReadonlyArray<number>
    readonly dtype: Tensor.DType
    readonly placement: Runtime.Placement
  }
): Effect.Effect<Tensor.Lazy, Tensor.TensorError> => {
  const candidate = value
  const placement = candidate.placement
  if (
    candidate._tag !== "LazyTensor" || candidate.dtype !== expected.dtype || !Array.isArray(candidate.shape) ||
    candidate.shape.length !== expected.shape.length ||
    !candidate.shape.every((dimension, index) => dimension === expected.shape[index]) ||
    placement === undefined || candidate.device !== placement.deviceType || placement.id !== expected.placement.id ||
    placement.deviceType !== expected.placement.deviceType ||
    placement.description !== expected.placement.description ||
    placement.ordinal !== expected.placement.ordinal || placement.memorySpace !== expected.placement.memorySpace ||
    placement.id !== runtime.placement.id || placement.deviceType !== runtime.placement.deviceType
  ) {
    return new Tensor.TensorError({
      op,
      message:
        `${op}: backend returned invalid lazy tensor metadata; expected ${expected.dtype} [${expected.shape}], got ${
          String(candidate.dtype)
        } [${Array.isArray(candidate.shape) ? candidate.shape : "invalid shape"}]`
    })
  }
  return Effect.succeed(value)
}

/**
 * Computes the gradients of a scalar loss with respect to the given tensors.
 * The loss is an ordinary lazy graph value. The backward transform runs
 * natively on the graph itself, without tracing or function transformation.
 * Adjoints use the same node vocabulary as the forward pass. Applying `grad`
 * to a derivative graph can produce higher-order derivatives only where all
 * participating forward ops and generated adjoint ops are themselves
 * differentiable. Dedicated semantic
 * backward nodes for cross entropy, RoPE, layer normalization, attention,
 * KDA, short convolution, and convolution do not currently provide a
 * second-order path. Unsupported or non-differentiable nodes in the native
 * autodiff transform are reported as {@link Tensor.TensorError} rather than
 * `GradError`.
 *
 * The loss and every `wrt` tensor must use `f32`, `f64`, `f16`, or `bf16`, and
 * that dtype must be supported by the active backend. Returned gradients are
 * lazy tensors with each target's shape, dtype, and placement. They capture and
 * borrow the forward graph; this function does not evaluate or clear inputs. A
 * `wrt` tensor that does not influence the loss yields a zero gradient.
 * Because the loss and its gradients share the forward graph, evaluate them
 * together with {@link Tensor.compute}. Evaluating them separately recomputes
 * the forward pass. If the graph contains `randn`, the separate evaluations use
 * different random draws.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const grad = (
  loss: Tensor.Any,
  wrt: ReadonlyArray<Tensor.Any>
): Effect.Effect<Array<Tensor.Lazy>, GradError | Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    if (loss.shape.length !== 0) {
      return yield* new GradError({
        reason: "non-scalar-output",
        detail: `grad: expected a scalar (0-d) loss, got shape [${loss.shape}], reduce it first (e.g. with sum or mean)`
      })
    }
    if (!isFloatDtype(loss.dtype)) {
      return yield* new GradError({
        reason: "non-float-dtype",
        detail: `grad: loss dtype must be a float dtype, got ${loss.dtype}`
      })
    }
    for (const target of wrt) {
      if (!isFloatDtype(target.dtype)) {
        return yield* new GradError({
          reason: "non-float-dtype",
          detail:
            `grad: cannot differentiate with respect to ${target.dtype} tensor, only f32 and f64 are differentiable`
        })
      }
    }
    const grads = yield* fromBackend("grad", runtime.grad(loss, wrt))
    if (grads.length !== wrt.length) {
      return yield* new Tensor.TensorError({
        op: "grad",
        message: `grad: backend returned ${grads.length} tensors for ${wrt.length} targets`
      })
    }
    return yield* Effect.forEach(grads, (value, index) => validateResult("grad", runtime, value, wrt[index]))
  })

/**
 * Stops gradient flow by adding a lazy identity node. The returned tensor has
 * the same metadata and value as the input, but the backward walk does not
 * continue past it, so ancestors of the input receive no gradient through this
 * path. No tensor is evaluated or transferred.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const stopGradient = (
  self: Tensor.Any
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const handle = yield* fromBackend(
      "stopGradient",
      runtime.node({ op: "stopGradient", inputs: [self] })
    )
    return yield* validateResult("stopGradient", runtime, handle, self)
  })

/**
 * Gradient checkpointing adds a lazy identity node with the input's metadata
 * and value. When a backward walk crosses it, the transform rebuilds the
 * checkpointed region with fresh node identities so its forward intermediates
 * can be recomputed instead of retained. This trades recomputation for lower
 * retained-intermediate pressure; the exact memory benefit is compiler and
 * backend dependent. Region inputs (nodes also reachable from outside the
 * checkpoint) and constructor leaves (including `randn` draws) are shared, so
 * recomputation uses the same leaves as the forward pass.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const checkpoint = (
  self: Tensor.Any
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const handle = yield* fromBackend("checkpoint", runtime.node({ op: "checkpoint", inputs: [self] }))
    return yield* validateResult("checkpoint", runtime, handle, self)
  })

const checkSameShapeDtype = (
  op: string,
  a: Tensor.Any,
  b: Tensor.Any,
  bName: string
): Effect.Effect<void, Tensor.TensorError> =>
  Effect.gen(function*() {
    if (a.shape.length !== b.shape.length || !a.shape.every((d, i) => d === b.shape[i])) {
      return yield* new Tensor.TensorError({
        op,
        message: `${op}: ${bName} shape [${b.shape}] does not match [${a.shape}]`
      })
    }
    if (a.dtype !== b.dtype) {
      return yield* new Tensor.TensorError({
        op,
        message: `${op}: ${bName} dtype ${b.dtype} does not match ${a.dtype}`
      })
    }
  })

/**
 * Vector-Jacobian product, or reverse-mode pullback. Given an output graph
 * `y` (built from `x` however you like), the primal `x`, and a cotangent
 * `v` with `y`'s shape, returns `J(x)^T v`, the gradient of `sum(y * v)`
 * with respect to `x`. `v` must exactly match `y`'s shape and dtype and use a
 * compatible placement; `x` and the derived loss must satisfy {@link grad}'s
 * floating-dtype contract. `v` is stopped before differentiation. A
 * disconnected `x` produces zeros.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const vjp = (
  y: Tensor.Any,
  x: Tensor.Any,
  v: Tensor.Any
): Effect.Effect<Tensor.Lazy, Tensor.TensorError | GradError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* checkSameShapeDtype("vjp", y, v, "cotangent")
    const loss = yield* Tensor.sum(yield* Tensor.mul(y, yield* stopGradient(v)))
    const [pullback] = yield* grad(loss, [x])
    return pullback
  })

/**
 * Jacobian-vector product, using a forward-mode pushforward through
 * forward-over-reverse. Given an output graph `y` built from `x`, the
 * primal `x`, and a tangent `v` with `x`'s shape, returns `J(x) v`. This
 * construction uses second-order adjoints and therefore fails for operations
 * whose backward graph is not differentiable. `v` must exactly match `x`'s
 * shape and dtype and use a compatible placement; `x` and the derived losses
 * must satisfy {@link grad}'s floating-dtype contract. The tangent is stopped
 * before the outer reverse pass. A disconnected `x` produces zeros.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const jvp = (
  y: Tensor.Any,
  x: Tensor.Any,
  v: Tensor.Any
): Effect.Effect<Tensor.Lazy, Tensor.TensorError | GradError, Runtime.Runtime> =>
  Effect.gen(function*() {
    yield* checkSameShapeDtype("jvp", x, v, "tangent")
    // u is a free linearization point. g(u) = J(x)^T u is linear in u, and
    // its own vjp at u = 0 with cotangent v is J(x) v.
    const u = yield* Tensor.zerosLike(y)
    const loss1 = yield* Tensor.sum(yield* Tensor.mul(y, u))
    const [gradX] = yield* grad(loss1, [x])
    const loss2 = yield* Tensor.sum(yield* Tensor.mul(gradX, yield* stopGradient(v)))
    const [tangent] = yield* grad(loss2, [u])
    return tangent
  })

/**
 * Maps the function implicit in a graph over a batch dimension. Given an output
 * graph `y` built from the unbatched input `x`, and `batchedX`
 * equal to `x` with a batch dimension inserted at `dim`, returns the graph
 * of `y` applied elementwise along that dimension. `y` must depend on `x`.
 * The output batch axis is inserted at `min(dim, y.rank)`. This is a native
 * graph rewrite with per-op batching rules, not a slice-and-restack loop. Graph
 * size remains linear in the source graph and independent of batch length;
 * some indexing rules add reshape and broadcast helpers. Elementwise ops and
 * matmul batch by broadcasting; reductions and shape ops shift their metadata;
 * `randn`/`uniform` nodes in the mapped graph draw per batch element.
 * Shared-index `indexSelect`/`take`, `gather`, and supported `scatterAdd` forms
 * have batching rules; indexes that depend on `x` and unsupported fused,
 * quantized, convolutional, recurrent, or decode operations fail with
 * {@link Tensor.TensorError}. The transform is lazy and does not evaluate or
 * clear any tensor.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const vmap = (
  y: Tensor.Any,
  x: Tensor.Any,
  batchedX: Tensor.Any,
  options: { readonly dim?: number } = {}
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const { dim, outShape } = yield* Effect.try({
      try: () => {
        const dim = options.dim ?? 0
        if (batchedX.shape.length !== x.shape.length + 1 || dim < 0 || dim >= batchedX.shape.length) {
          throw new Error(
            `vmap: batched input shape [${batchedX.shape}] must be the input shape [${x.shape}] with one dimension inserted`
          )
        }
        for (let i = 0; i < x.shape.length; i++) {
          const at = i < dim ? i : i + 1
          if (batchedX.shape[at] !== x.shape[i]) {
            throw new Error(
              `vmap: batched input shape [${batchedX.shape}] does not match input shape [${x.shape}] outside dim ${dim}`
            )
          }
        }
        if (batchedX.dtype !== x.dtype) {
          throw new Error(`vmap: dtype mismatch, got ${batchedX.dtype} and ${x.dtype}`)
        }
        const outShape = [...y.shape]
        outShape.splice(Math.min(dim, outShape.length), 0, batchedX.shape[dim])
        return { dim, outShape }
      },
      catch: (error) =>
        new Tensor.TensorError({
          op: "vmap",
          message: error instanceof Error ? error.message : String(error)
        })
    })
    const handle = yield* fromBackend(
      "vmap",
      runtime.node({
        op: "vmap",
        inputs: [y, x, batchedX],
        attributes: { dim }
      })
    )
    return yield* validateResult("vmap", runtime, handle, {
      shape: outShape,
      dtype: y.dtype,
      placement: y.placement
    })
  })

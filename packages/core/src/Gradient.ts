/**
 * Reverse-mode autodiff and graph transforms. The backward transform runs
 * natively on the graph itself — there is no tracing and no function
 * transformation: the loss is an ordinary lazy graph value and adjoints are
 * expressed in the same node vocabulary as the forward pass, so
 * higher-order derivatives work by applying {@link grad} again.
 *
 * @since 0.1.0
 */
import { Data, Effect } from "effect"
import native from "@effect-torch/native"
import type { CurrentDevice } from "./Device.ts"
import * as Tensor from "./Tensor.ts"

/**
 * Error raised by {@link grad} when the graph violates the autodiff
 * contract.
 *
 * @since 0.1.0
 * @category errors
 */
export class GradError extends Data.TaggedError("GradError")<{
  readonly reason: "non-scalar-output" | "non-float-dtype" | "not-differentiable"
  readonly detail: string
}> {}

const isFloatDtype = (dtype: string): boolean => dtype === "f32" || dtype === "f64"

const toGradError = (error: unknown): GradError => {
  const detail = error instanceof Error ? error.message : String(error)
  return new GradError({
    // the scalar and float-dtype contracts are validated above, so a native
    // error here means the graph contains a non-differentiable construct
    reason: "not-differentiable",
    detail
  })
}

/**
 * Computes the gradients of a scalar loss with respect to the given tensors.
 * The loss is an ordinary lazy graph value — there is no tracing and no
 * function transformation, the backward transform runs natively on the
 * graph itself: one walk, with adjoints expressed in the same node
 * vocabulary as the forward pass, so higher-order derivatives work by
 * applying `grad` again.
 *
 * Gradients are lazy tensors sharing the forward graph; a `wrt` tensor that
 * does not influence the loss yields a zero gradient. Because the loss and
 * its gradients share the forward graph, evaluate them together with
 * {@link Tensor.compute}: evaluating them separately recomputes the forward
 * pass and, if the graph contains `randn`, produces values from different
 * random draws.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const grad = (
  loss: Tensor.GenericTensor,
  wrt: ReadonlyArray<Tensor.GenericTensor>
): Effect.Effect<Array<Tensor.LazyTensor>, GradError> =>
  Effect.gen(function* () {
    if (loss.shape.length !== 0) {
      return yield* new GradError({
        reason: "non-scalar-output",
        detail: `grad: expected a scalar (0-d) loss, got shape [${loss.shape}], reduce it first (e.g. with sum or mean)`
      })
    }
    if (!isFloatDtype(loss.dtype)) {
      return yield* new GradError({
        reason: "non-float-dtype",
        detail: `grad: loss dtype must be f32 or f64, got ${loss.dtype}`
      })
    }
    for (const target of wrt) {
      if (!isFloatDtype(target.dtype)) {
        return yield* new GradError({
          reason: "non-float-dtype",
          detail: `grad: cannot differentiate with respect to ${target.dtype} tensor, only f32 and f64 are differentiable`
        })
      }
    }
    const grads = yield* Effect.try({
      try: () => native.grad(loss.lazy, wrt.map((target) => target.lazy)),
      catch: toGradError
    })
    return grads.map((handle, i) => Tensor.makeLazy(handle, wrt[i].shape, wrt[i].dtype, wrt[i].device))
  })

/**
 * Stops gradient flow: the returned tensor has the same value as the input,
 * but the backward walk does not continue past it, so ancestors of the input
 * receive no gradient through this path.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const stopGradient = (
  self: Tensor.GenericTensor
): Effect.Effect<Tensor.LazyTensor, Tensor.TensorError> =>
  Effect.try({
    try: () => Tensor.makeLazy(self.lazy.stopGradient(), self.shape, self.dtype, self.device),
    catch: (error) =>
      new Tensor.TensorError({
        op: "stopGradient",
        message: error instanceof Error ? error.message : String(error)
      })
  })

/**
 * Gradient checkpointing: the returned tensor has the same value as the
 * input, but during the backward pass the forward intermediates of the
 * subgraph that produced it are recomputed from a fresh copy instead of
 * being retained — trading one extra forward evaluation of the region for
 * its peak memory. Region inputs (nodes also reachable from outside the
 * checkpoint) and constructor leaves (including `randn` draws) are shared,
 * so recomputation is consistent with the forward pass.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const checkpoint = (
  self: Tensor.GenericTensor
): Effect.Effect<Tensor.LazyTensor, Tensor.TensorError> =>
  Effect.try({
    try: () => Tensor.makeLazy(self.lazy.checkpoint(), self.shape, self.dtype, self.device),
    catch: (error) =>
      new Tensor.TensorError({
        op: "checkpoint",
        message: error instanceof Error ? error.message : String(error)
      })
  })

const checkSameShapeDtype = (
  op: string,
  a: Tensor.GenericTensor,
  b: Tensor.GenericTensor,
  bName: string
): Effect.Effect<void, Tensor.TensorError> =>
  Effect.gen(function* () {
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
 * Vector-Jacobian product (reverse-mode pullback): given an output graph
 * `y` (built from `x` however you like), the primal `x`, and a cotangent
 * `v` with `y`'s shape, returns `J(x)ᵀ v` — the gradient of `sum(y * v)`
 * with respect to `x`.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const vjp = (
  y: Tensor.GenericTensor,
  x: Tensor.GenericTensor,
  v: Tensor.GenericTensor
): Effect.Effect<Tensor.LazyTensor, Tensor.TensorError | GradError> =>
  Effect.gen(function* () {
    yield* checkSameShapeDtype("vjp", y, v, "cotangent")
    const loss = yield* Tensor.sum(yield* Tensor.mul(y, yield* stopGradient(v)))
    const [pullback] = yield* grad(loss, [x])
    return pullback
  })

/**
 * Jacobian-vector product (forward-mode pushforward via
 * forward-over-reverse): given an output graph `y` built from `x`, the
 * primal `x`, and a tangent `v` with `x`'s shape, returns `J(x) v`. Uses
 * second-order adjoints, so the graph must be twice differentiable through
 * the ops it uses.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const jvp = (
  y: Tensor.GenericTensor,
  x: Tensor.GenericTensor,
  v: Tensor.GenericTensor
): Effect.Effect<Tensor.LazyTensor, Tensor.TensorError | GradError, CurrentDevice> =>
  Effect.gen(function* () {
    yield* checkSameShapeDtype("jvp", x, v, "tangent")
    // u is a free linearization point: g(u) = J(x)ᵀ u is linear in u, and
    // its own vjp at u = 0 with cotangent v is J(x) v
    const u = yield* Tensor.zerosLike(y)
    const loss1 = yield* Tensor.sum(yield* Tensor.mul(y, u))
    const [gradX] = yield* grad(loss1, [x])
    const loss2 = yield* Tensor.sum(yield* Tensor.mul(gradX, yield* stopGradient(v)))
    const [tangent] = yield* grad(loss2, [u])
    return tangent
  })

/**
 * Maps the function implicit in a graph over a batch dimension: given an
 * output graph `y` built from the unbatched input `x`, and `batchedX`
 * equal to `x` with a batch dimension inserted at `dim`, returns the graph
 * of `y` applied elementwise along that dimension (the output carries the
 * batch at `dim` too). This is a native graph rewrite with per-op batching
 * rules — not a slice-and-restack loop — so the batched graph is the same
 * size as the original. Elementwise ops and matmul batch by broadcasting;
 * reductions and shape ops shift their metadata; `randn`/`uniform` draw
 * per batch element; indexing with data-dependent indexes, `gather`,
 * `scatterAdd`, and rank-2 linalg are not supported.
 *
 * @since 0.1.0
 * @category autodiff
 */
export const vmap = (
  y: Tensor.GenericTensor,
  x: Tensor.GenericTensor,
  batchedX: Tensor.GenericTensor,
  options: { readonly dim?: number } = {}
): Effect.Effect<Tensor.LazyTensor, Tensor.TensorError> =>
  Effect.try({
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
      if (batchedX.device !== x.device) {
        throw new Error(`vmap: device mismatch, got ${batchedX.device} and ${x.device}`)
      }
      const batch = batchedX.shape[dim]
      const outShape = [...y.shape]
      outShape.splice(Math.min(dim, outShape.length), 0, batch)
      return Tensor.makeLazy(y.lazy.vmap(x.lazy, batchedX.lazy, dim), outShape, y.dtype, y.device)
    },
    catch: (error) =>
      new Tensor.TensorError({
        op: "vmap",
        message: error instanceof Error ? error.message : String(error)
      })
  })

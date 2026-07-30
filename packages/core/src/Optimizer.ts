/**
 * Optimizers expressed as pure graph transforms. An optimizer's `step`
 * takes the current parameters, their gradients, and optimizer state, and
 * returns updated parameters and updated state as lazy graph values —
 * nothing is mutated and nothing is materialized inside the optimizer.
 * Because gradients share the loss's forward graph and the updates extend
 * the same graphs further, the loss, the updated parameters, and the
 * updated state can all be roots of a single {@link Tensor.compute} walk:
 * one training step costs exactly one forward and one backward pass, and
 * intermediate gradient tensors are freed as soon as their update consumes
 * them.
 *
 * State is re-materialized into graph leaves at every step (that is what
 * {@link step} does), so the graph depth stays O(model depth) no matter how
 * many steps run.
 *
 * Update formulas match PyTorch / candle-nn exactly:
 *
 * - SGD: `g += weightDecay * p`; `v = momentum * v + (1 - dampening) * g`
 *   (with `v = g` on the first step); `p -= lr * g'` where
 *   `g' = g + momentum * v` when `nesterov`, else `g' = v`. With no
 *   momentum, `p -= lr * g`.
 * - Adam / AdamW: `m = beta1 * m + (1 - beta1) * g`,
 *   `v = beta2 * v + (1 - beta2) * g^2`, bias-corrected
 *   `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)`,
 *   `p = p * (1 - lr * weightDecay) - lr * m_hat / (sqrt(v_hat) + eps)`.
 *   The decay term is decoupled (AdamW) and zero for plain Adam.
 *
 * @since 0.1.0
 */
import { Effect } from "effect"
import type { CurrentDevice } from "./Device.ts"
import * as Gradient from "./Gradient.ts"
import * as Tensor from "./Tensor.ts"

/**
 * Configuration for stochastic gradient descent. `fused` (default `true`)
 * computes each momentum update as a single native node instead of several
 * graph nodes — identical numerics, smaller graphs.
 *
 * @since 0.1.0
 * @category models
 */
export interface SgdConfig {
  readonly lr: number
  readonly momentum?: number
  readonly dampening?: number
  readonly nesterov?: boolean
  readonly weightDecay?: number
  readonly fused?: boolean
}

/**
 * State carried between SGD steps: one velocity tensor per parameter when
 * momentum is enabled, `null` otherwise.
 *
 * @since 0.1.0
 * @category models
 */
export interface SgdState {
  readonly velocity: ReadonlyArray<Tensor.Any> | null
}

/**
 * Configuration for Adam. All fields default to the standard values
 * (`lr = 1e-3`, `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`). `fused`
 * (default `true`) computes each parameter's update as a single native
 * node instead of ~10 graph nodes — identical numerics, smaller graphs.
 *
 * @since 0.1.0
 * @category models
 */
export interface AdamConfig {
  readonly lr?: number
  readonly beta1?: number
  readonly beta2?: number
  readonly eps?: number
  readonly fused?: boolean
}

/**
 * Configuration for AdamW. `weightDecay` defaults to `0.01` and is applied
 * decoupled from the gradient, `p *= (1 - lr * weightDecay)`.
 *
 * @since 0.1.0
 * @category models
 */
export interface AdamWConfig extends AdamConfig {
  readonly weightDecay?: number
}

/**
 * State carried between Adam-family steps: first and second moment
 * estimates, one per parameter, plus the step count `t` used for bias
 * correction.
 *
 * @since 0.1.0
 * @category models
 */
export interface AdamState {
  readonly m: ReadonlyArray<Tensor.Any>
  readonly v: ReadonlyArray<Tensor.Any>
  readonly t: number
}

/**
 * The result of {@link Optimizer.step}: updated parameters and updated
 * state as lazy graph values, in the same order as the input parameters,
 * plus everything needed to materialize the state for the next step:
 *
 * - `stateRoots` lists the tensors inside `state` that must be evaluated
 *   before the state is fed into another `step` call (state is always
 *   re-materialized into graph leaves between steps, so graph depth stays
 *   O(model depth) no matter how many steps run).
 * - `rebuildState` repacks the evaluated `stateRoots` (in the same order,
 *   always materialized) into a new state value.
 *
 * User-land optimizers implement the same contract: return your new state
 * alongside the list of tensors it contains and a function that rebuilds
 * it from their materialized counterparts.
 *
 * @since 0.1.0
 * @category models
 */
export interface OptimizerUpdate<S> {
  readonly params: Array<Tensor.Lazy>
  readonly state: S
  readonly stateRoots: ReadonlyArray<Tensor.Any>
  readonly rebuildState: (evaluated: ReadonlyArray<Tensor.Concrete>) => S
}

/**
 * A stateful optimizer as a pure graph transform. `init` validates the
 * parameters and builds zero-initialized state; `step` extends the graph
 * with the update arithmetic. Neither evaluates anything.
 *
 * @since 0.1.0
 * @category models
 */
export interface Optimizer<S> {
  readonly init: (
    params: ReadonlyArray<Tensor.Any>
  ) => Effect.Effect<S, Tensor.TensorError, CurrentDevice>
  readonly step: (
    params: ReadonlyArray<Tensor.Any>,
    grads: ReadonlyArray<Tensor.Any>,
    state: S
  ) => Effect.Effect<OptimizerUpdate<S>, Tensor.TensorError>
}

const isFloat = (dtype: Tensor.DType): boolean => dtype === "f32" || dtype === "f64"

const checkParams = (
  op: string,
  params: ReadonlyArray<Tensor.Any>
): Effect.Effect<void, Tensor.TensorError> => {
  for (const param of params) {
    if (!isFloat(param.dtype)) {
      return new Tensor.TensorError({
        op,
        message: `${op}: parameters must be f32 or f64, got ${param.dtype}`
      })
    }
  }
  return Effect.void
}

const sameShape = (a: ReadonlyArray<number>, b: ReadonlyArray<number>): boolean =>
  a.length === b.length && a.every((dim, i) => dim === b[i])

const checkGrads = (
  op: string,
  params: ReadonlyArray<Tensor.Any>,
  grads: ReadonlyArray<Tensor.Any>
): Effect.Effect<void, Tensor.TensorError> => {
  if (params.length !== grads.length) {
    return new Tensor.TensorError({
      op,
      message: `${op}: expected ${params.length} gradients, got ${grads.length}`
    })
  }
  for (let i = 0; i < params.length; i++) {
    if (params[i].dtype !== grads[i].dtype) {
      return new Tensor.TensorError({
        op,
        message: `${op}: gradient ${i} has dtype ${grads[i].dtype}, expected ${params[i].dtype}`
      })
    }
    if (!sameShape(params[i].shape, grads[i].shape)) {
      return new Tensor.TensorError({
        op,
        message: `${op}: gradient ${i} has shape [${grads[i].shape}], expected [${params[i].shape}]`
      })
    }
  }
  return Effect.void
}

const checkStateLength = (
  op: string,
  kind: string,
  state: ReadonlyArray<Tensor.Any>,
  params: ReadonlyArray<Tensor.Any>
): Effect.Effect<void, Tensor.TensorError> => {
  if (state.length !== params.length) {
    return new Tensor.TensorError({
      op,
      message: `${op}: state holds ${state.length} ${kind} tensors for ${params.length} parameters, use init for these parameters`
    })
  }
  for (let i = 0; i < params.length; i++) {
    if (state[i].dtype !== params[i].dtype || !sameShape(state[i].shape, params[i].shape)) {
      return new Tensor.TensorError({
        op,
        message: `${op}: ${kind} ${i} does not match parameter shape/dtype, use init for these parameters`
      })
    }
  }
  return Effect.void
}

/**
 * Creates a stochastic gradient descent optimizer with optional momentum,
 * dampening, nesterov, and coupled (L2) weight decay. With no momentum the
 * update is `p -= lr * g`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const sgd = (config: SgdConfig): Optimizer<SgdState> => {
  const lr = config.lr
  const momentum = config.momentum ?? 0
  const dampening = config.dampening ?? 0
  const nesterov = config.nesterov ?? false
  const weightDecay = config.weightDecay ?? 0
  const fused = config.fused ?? true
  if (!Number.isFinite(lr) || lr <= 0) {
    throw new Error(`sgd: lr must be positive, got ${lr}`)
  }
  if (!Number.isFinite(momentum) || momentum < 0) {
    throw new Error(`sgd: momentum must be non-negative, got ${momentum}`)
  }
  if (nesterov && (momentum <= 0 || dampening !== 0)) {
    throw new Error("sgd: nesterov requires momentum > 0 and dampening = 0")
  }

  const updateParam = (
    param: Tensor.Any,
    grad: Tensor.Any,
    velocity: Tensor.Any | null
  ): Effect.Effect<
    { readonly param: Tensor.Lazy; readonly velocity: Tensor.Any | null },
    Tensor.TensorError
  > =>
    Effect.gen(function* () {
      let g: Tensor.Any = grad
      if (weightDecay !== 0) {
        g = yield* Tensor.add(g, yield* Tensor.mul(param, weightDecay))
      }
      if (momentum === 0) {
        return { param: yield* Tensor.sub(param, yield* Tensor.mul(g, lr)), velocity: null }
      }
      if (fused) {
        const step = yield* Effect.try({
          try: () =>
            param.lazy.sgdStep(
              grad.lazy,
              (velocity ?? param).lazy,
              velocity === null,
              lr,
              momentum,
              dampening,
              nesterov,
              weightDecay
            ),
          catch: (error) =>
            new Tensor.TensorError({
              op: "sgd",
              message: error instanceof Error ? error.message : String(error)
            })
        })
        const makeOut = (index: number): Tensor.Lazy => {
          const handle = step.sgdOut(index)
          return Tensor.makeLazy(handle, param.shape, param.dtype, param.device)
        }
        return { param: makeOut(0), velocity: makeOut(1) }
      }
      const nextVelocity = velocity === null
        ? g
        : yield* Tensor.add(yield* Tensor.mul(velocity, momentum), yield* Tensor.mul(g, 1 - dampening))
      const used = nesterov
        ? yield* Tensor.add(g, yield* Tensor.mul(nextVelocity, momentum))
        : nextVelocity
      return { param: yield* Tensor.sub(param, yield* Tensor.mul(used, lr)), velocity: nextVelocity }
    })

  return {
    init: (params) =>
      Effect.map(checkParams("sgd", params), (): SgdState => ({ velocity: null })),
    step: (params, grads, state) =>
      Effect.gen(function* () {
        yield* checkParams("sgd", params)
        yield* checkGrads("sgd", params, grads)
        if (momentum !== 0 && state.velocity !== null) {
          yield* checkStateLength("sgd", "velocity", state.velocity, params)
        }
        const updates: Array<Tensor.Lazy> = []
        const velocities: Array<Tensor.Any> = []
        for (let i = 0; i < params.length; i++) {
          const update = yield* updateParam(params[i], grads[i], state.velocity?.[i] ?? null)
          updates.push(update.param)
          if (update.velocity !== null) velocities.push(update.velocity)
        }
        const roots = momentum === 0 ? [] : velocities
        return {
          params: updates,
          state: { velocity: momentum === 0 ? null : roots },
          stateRoots: roots,
          rebuildState: (evaluated): SgdState => ({
            velocity: momentum === 0 ? null : [...evaluated]
          })
        }
      })
  }
}

interface ResolvedAdamConfig {
  readonly lr: number
  readonly beta1: number
  readonly beta2: number
  readonly eps: number
  readonly weightDecay: number
  readonly fused: boolean
}

const makeAdam = (op: string, config: ResolvedAdamConfig): Optimizer<AdamState> => {
  const { lr, beta1, beta2, eps, weightDecay, fused } = config
  if (!Number.isFinite(lr) || lr <= 0) {
    throw new Error(`${op}: lr must be positive, got ${lr}`)
  }
  if (
    !Number.isFinite(beta1) || !Number.isFinite(beta2) || beta1 < 0 || beta1 >= 1 || beta2 < 0 ||
    beta2 >= 1
  ) {
    throw new Error(`${op}: beta1 and beta2 must be in [0, 1), got ${beta1} and ${beta2}`)
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error(`${op}: eps must be positive, got ${eps}`)
  }

  const updateParam = (
    param: Tensor.Any,
    grad: Tensor.Any,
    m: Tensor.Any,
    v: Tensor.Any,
    t: number
  ): Effect.Effect<
    {
      readonly param: Tensor.Lazy
      readonly m: Tensor.Lazy
      readonly v: Tensor.Lazy
    },
    Tensor.TensorError
  > =>
    Effect.gen(function* () {
      if (fused) {
        const step = yield* Effect.try({
          try: () =>
            param.lazy.adamwStep(grad.lazy, m.lazy, v.lazy, lr, beta1, beta2, eps, weightDecay, t),
          catch: (error) =>
            new Tensor.TensorError({
              op,
              message: error instanceof Error ? error.message : String(error)
            })
        })
        const makeOut = (index: number): Tensor.Lazy => {
          const handle = step.adamwOut(index)
          return Tensor.makeLazy(handle, param.shape, param.dtype, param.device)
        }
        return { param: makeOut(0), m: makeOut(1), v: makeOut(2) }
      }
      const nextM = yield* Tensor.add(yield* Tensor.mul(m, beta1), yield* Tensor.mul(grad, 1 - beta1))
      const nextV = yield* Tensor.add(
        yield* Tensor.mul(v, beta2),
        yield* Tensor.mul(yield* Tensor.mul(grad, grad), 1 - beta2)
      )
      const mHat = yield* Tensor.mul(nextM, 1 / (1 - Math.pow(beta1, t)))
      const vHat = yield* Tensor.mul(nextV, 1 / (1 - Math.pow(beta2, t)))
      const denom = yield* Tensor.add(yield* Tensor.sqrt(vHat), eps)
      const adjusted = yield* Tensor.mul(yield* Tensor.div(mHat, denom), lr)
      const base = weightDecay === 0 ? param : yield* Tensor.mul(param, 1 - lr * weightDecay)
      return { param: yield* Tensor.sub(base, adjusted), m: nextM, v: nextV }
    })

  return {
    init: (params) =>
      Effect.gen(function* () {
        yield* checkParams(op, params)
        const m: Array<Tensor.Any> = []
        const v: Array<Tensor.Any> = []
        for (const param of params) {
          m.push(yield* Tensor.zeros(param.shape, { dtype: param.dtype }))
          v.push(yield* Tensor.zeros(param.shape, { dtype: param.dtype }))
        }
        return { m, v, t: 0 } satisfies AdamState
      }),
    step: (params, grads, state) =>
      Effect.gen(function* () {
        yield* checkParams(op, params)
        yield* checkGrads(op, params, grads)
        yield* checkStateLength(op, "first-moment", state.m, params)
        yield* checkStateLength(op, "second-moment", state.v, params)
        const t = state.t + 1
        const updates: Array<Tensor.Lazy> = []
        const m: Array<Tensor.Lazy> = []
        const v: Array<Tensor.Lazy> = []
        for (let i = 0; i < params.length; i++) {
          const update = yield* updateParam(params[i], grads[i], state.m[i], state.v[i], t)
          updates.push(update.param)
          m.push(update.m)
          v.push(update.v)
        }
        return {
          params: updates,
          state: { m, v, t },
          stateRoots: [...m, ...v],
          rebuildState: (evaluated): AdamState => ({
            m: evaluated.slice(0, m.length),
            v: evaluated.slice(m.length),
            t
          })
        }
      })
  }
}

/**
 * Creates an Adam optimizer with the standard defaults (`lr = 1e-3`,
 * `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`).
 *
 * @since 0.1.0
 * @category constructors
 */
export const adam = (config: AdamConfig = {}): Optimizer<AdamState> =>
  makeAdam("adam", {
    lr: config.lr ?? 1e-3,
    beta1: config.beta1 ?? 0.9,
    beta2: config.beta2 ?? 0.999,
    eps: config.eps ?? 1e-8,
    weightDecay: 0,
    fused: config.fused ?? true
  })

/**
 * Creates an AdamW optimizer: Adam with decoupled weight decay (default
 * `0.01`), applied as `p *= (1 - lr * weightDecay)` before the adaptive
 * update.
 *
 * @since 0.1.0
 * @category constructors
 */
export const adamW = (config: AdamWConfig = {}): Optimizer<AdamState> =>
  makeAdam("adamW", {
    lr: config.lr ?? 1e-3,
    beta1: config.beta1 ?? 0.9,
    beta2: config.beta2 ?? 0.999,
    eps: config.eps ?? 1e-8,
    weightDecay: config.weightDecay ?? 0.01,
    fused: config.fused ?? true
  })

/**
 * Clips every gradient elementwise into `[min, max]`. A pure graph
 * transform, applied between {@link Gradient.grad} and
 * {@link Optimizer.step}.
 *
 * @since 0.1.0
 * @category transforms
 */
export const clipByValue = (
  grads: ReadonlyArray<Tensor.Any>,
  options: { readonly min?: number; readonly max?: number }
): Effect.Effect<Array<Tensor.Lazy>, Tensor.TensorError> =>
  Effect.gen(function* () {
    if (options.min === undefined && options.max === undefined) {
      return yield* new Tensor.TensorError({
        op: "clipByValue",
        message: "clipByValue: at least one of min and max is required"
      })
    }
    const out: Array<Tensor.Lazy> = []
    for (const g of grads) {
      out.push(yield* Tensor.clamp(g, options))
    }
    return out
  })

/**
 * Clips gradients by global norm (PyTorch semantics): the total norm is the
 * square root of the sum of squares over *all* gradients, and every
 * gradient is scaled by `maxNorm / (totalNorm + 1e-6)` when that factor is
 * below `1`. A pure graph transform, applied between
 * {@link Gradient.grad} and {@link Optimizer.step}.
 *
 * @since 0.1.0
 * @category transforms
 */
export const clipByGlobalNorm = (
  grads: ReadonlyArray<Tensor.Any>,
  maxNorm: number
): Effect.Effect<Array<Tensor.Lazy>, Tensor.TensorError> =>
  Effect.gen(function* () {
    if (maxNorm <= 0) {
      return yield* new Tensor.TensorError({
        op: "clipByGlobalNorm",
        message: `clipByGlobalNorm: maxNorm must be positive, got ${maxNorm}`
      })
    }
    if (grads.length === 0) {
      return []
    }
    let total: Tensor.Any = yield* Tensor.sum(yield* Tensor.square(grads[0]))
    for (const g of grads.slice(1)) {
      total = yield* Tensor.add(total, yield* Tensor.sum(yield* Tensor.square(g)))
    }
    const norm = yield* Tensor.sqrt(total)
    const scale = yield* Tensor.minimum(yield* Tensor.mul(yield* Tensor.reciprocal(yield* Tensor.add(norm, 1e-6)), maxNorm), 1)
    const out: Array<Tensor.Lazy> = []
    for (const g of grads) {
      out.push(yield* Tensor.mul(g, scale))
    }
    return out
  })

/**
 * Maps a parameter tuple/array to the same structure with materialized
 * tensors — tuple in, tuple out.
 *
 * @since 0.1.0
 * @category models
 */
export type Materialized<P extends ReadonlyArray<Tensor.Any>> = {
  readonly [K in keyof P]: Tensor.Concrete
}

/**
 * Runs one full training step: computes the gradients of a scalar loss
 * with respect to the parameters, extends the graph with the optimizer
 * update, and evaluates loss, updated parameters, and every tensor inside
 * the updated state in a single walk — one forward pass, one backward
 * pass, one async boundary. The returned parameters are materialized
 * tensors with the same length, order, shapes, and dtypes as the input
 * parameters (a tuple in, the same tuple out), and the state is rebuilt
 * from materialized tensors via the update's `rebuildState`: both are
 * plain leaves of the next step's graph, so graph depth stays
 * O(model depth) no matter how many steps run.
 *
 * @since 0.1.0
 * @category destructors
 */
export const step = <S, P extends ReadonlyArray<Tensor.Any>>(
  optimizer: Optimizer<S>,
  loss: Tensor.Any,
  params: P,
  state: S
): Effect.Effect<
  { readonly loss: Tensor.Concrete; readonly params: Materialized<P>; readonly state: S },
  Gradient.GradError | Tensor.TensorError
> =>
  Effect.gen(function* () {
    const grads = yield* Gradient.grad(loss, params)
    const next = yield* optimizer.step(params, grads, state)
    const evaluated = yield* Tensor.compute([loss, ...next.params, ...next.stateRoots])
    const [evaluatedLoss, ...rest] = evaluated
    return {
      loss: evaluatedLoss,
      params: rest.slice(0, next.params.length) as Materialized<P>,
      state: next.rebuildState(rest.slice(next.params.length))
    }
  })

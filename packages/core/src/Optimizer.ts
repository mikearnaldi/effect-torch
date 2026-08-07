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
 * many steps run. Optimizer state is *all tensors* — the Adam step count
 * `t` is a 0-d tensor, not a JS number — so every piece of step-varying
 * data flows through the graph and a frozen graph (a compiled step) never
 * replays a stale count, flag, or rate. The learning rate is not part of
 * the configuration: it is a per-step input to {@link Optimizer.step}, so
 * learning-rate schedules are ordinary data flowing through the same
 * graph instead of a reason to rebuild the optimizer (or its graph) every
 * step.
 *
 * Update formulas match PyTorch / candle-nn exactly:
 *
 * - SGD: `g += weightDecay * p`; `v = momentum * v + (1 - dampening) * g`
 *   (with `v = g` on the first step, selected by the 0-d `first` flag in
 *   the state); `p -= lr * g'` where `g' = g + momentum * v` when
 *   `nesterov`, else `g' = v`. With no momentum, `p -= lr * g`.
 * - Adam / AdamW: `m = beta1 * m + (1 - beta1) * g`,
 *   `v = beta2 * v + (1 - beta2) * g^2`, bias-corrected
 *   `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)`,
 *   `p = p * (1 - lr * weightDecay) - lr * m_hat / (sqrt(v_hat) + eps)`.
 *   The decay term is decoupled (AdamW) and zero for plain Adam. The
 *   correction denominators `1 - beta1^t` / `1 - beta2^t` are computed as
 *   tensor ops from the state's `t`, once per step, shared by every
 *   parameter's update node.
 *
 * @since 0.1.0
 */
import { Effect } from "effect"
import * as Gradient from "./Gradient.ts"
import * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * Configuration for stochastic gradient descent. The learning rate is a
 * per-step input to {@link Optimizer.step}, not configuration.
 *
 * @since 0.1.0
 * @category models
 */
export interface SgdConfig {
  readonly momentum?: number
  readonly dampening?: number
  readonly nesterov?: boolean
  readonly weightDecay?: number
}

/**
 * State carried between SGD steps: one velocity tensor per parameter
 * (empty when momentum is disabled) and the 0-d `first` flag (1 on the
 * first step, 0 after) that selects `v = g` over the momentum recurrence.
 *
 * @since 0.1.0
 * @category models
 */
export interface SgdState {
  readonly velocity: ReadonlyArray<Tensor.Any>
  readonly first: Tensor.Any
}

/**
 * Configuration for Adam (`beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8` by
 * default). The learning rate is a per-step input to
 * {@link Optimizer.step}, not configuration. For f32 state, initialization
 * rejects betas whose rounded first-step bias correction differs from the
 * configured value by more than 1%.
 *
 * @since 0.1.0
 * @category models
 */
export interface AdamConfig {
  readonly beta1?: number
  readonly beta2?: number
  readonly eps?: number
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
 * correction. Every field is a tensor: `t` is a 0-d tensor matching the
 * first parameter's dtype and placement, so the count flows through the
 * graph like any other state.
 *
 * @since 0.1.0
 * @category models
 */
export interface AdamState {
  readonly m: ReadonlyArray<Tensor.Any>
  readonly v: ReadonlyArray<Tensor.Any>
  readonly t: Tensor.Any
}

/**
 * The result of {@link Optimizer.step}: updated parameters and updated
 * state as lazy graph values, in the same order as the input parameters,
 * plus `stateRoots` listing the tensors inside `state` that must be
 * evaluated before the state is fed into another `step` call (state is
 * always re-materialized into graph leaves between steps, so graph depth
 * stays O(model depth) no matter how many steps run). Repack the
 * evaluated roots into a new state value with
 * `optimizer.rebuildState(state, evaluated)`.
 *
 * User-land optimizers implement the same contract: return your new state
 * alongside the list of tensors it contains.
 *
 * @since 0.1.0
 * @category models
 */
export interface OptimizerUpdate<S> {
  readonly params: Array<Tensor.Lazy>
  readonly state: S
  readonly stateRoots: ReadonlyArray<Tensor.Any>
}

/**
 * A stateful optimizer as a pure graph transform. `init` validates the
 * parameters and builds zero-initialized state; `step` extends the graph
 * with the update arithmetic. Neither evaluates anything.
 *
 * The learning rate is a per-step input: a 0-d float tensor on the same
 * device as the parameters. Pass a different value every step (a
 * `LearningRate` schedule evaluated by the training loop) without
 * rebuilding the optimizer — the rate flows through the graph as data.
 *
 * {@link Optimizer.stateRoots} / {@link Optimizer.rebuildState} are the
 * canonical extraction and injection of a state's tensor leaves, in one
 * stable order — the boundary a compiled training step rebinds per call.
 *
 * @since 0.1.0
 * @category models
 */
export interface Optimizer<S> {
  readonly init: (
    params: ReadonlyArray<Tensor.Any>
  ) => Effect.Effect<S, Tensor.TensorError, Runtime.Runtime>
  readonly step: (
    params: ReadonlyArray<Tensor.Any>,
    grads: ReadonlyArray<Tensor.Any>,
    state: S,
    lr: Tensor.Any
  ) => Effect.Effect<OptimizerUpdate<S>, Tensor.TensorError, Runtime.Runtime>
  readonly stateRoots: (state: S) => ReadonlyArray<Tensor.Any>
  readonly rebuildState: (state: S, roots: ReadonlyArray<Tensor.Any>) => S
}

const isFloat = (dtype: Tensor.DType): boolean => dtype === "f32" || dtype === "f64"

// The step count lives on the ambient device like every other state
// tensor — never a hidden device override. Its dtype follows the
// params' float width; f32 counts exactly to 2^24 (~16.7M steps),
// beyond which bias correction is a no-op anyway.
const stepCount = (
  value: number,
  params: ReadonlyArray<Tensor.Any>
): Effect.Effect<Tensor.Lazy, Tensor.TensorError, Runtime.Runtime> =>
  params[0] === undefined
    ? new Tensor.TensorError({ op: "stepCount", message: "stepCount: expected at least one parameter" })
    : Tensor.constantLike(params[0], value)

const checkLr = (op: string, lr: Tensor.Any): Effect.Effect<void, Tensor.TensorError> => {
  if (lr.shape.length !== 0 || !isFloat(lr.dtype)) {
    return new Tensor.TensorError({
      op,
      message: `${op}: lr must be a 0-d float tensor, got shape [${lr.shape}] ${lr.dtype}`
    })
  }
  return Effect.void
}

const checkStateScalar = (
  op: string,
  name: string,
  value: Tensor.Any
): Effect.Effect<void, Tensor.TensorError> => {
  if (value.shape.length !== 0 || !isFloat(value.dtype)) {
    return new Tensor.TensorError({ op, message: `${op}: ${name} must be a 0-d float tensor` })
  }
  return Effect.void
}

const checkParams = (
  op: string,
  params: ReadonlyArray<Tensor.Any>
): Effect.Effect<void, Tensor.TensorError> => {
  const first = params[0]
  if (first === undefined) {
    return new Tensor.TensorError({ op, message: `${op}: expected at least one parameter` })
  }
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
      message:
        `${op}: state holds ${state.length} ${kind} tensors for ${params.length} parameters, use init for these parameters`
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
  expected: Tensor.Any
): Tensor.Lazy => {
  const candidate = value as Partial<Runtime.LazyTensorHandle>
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
    throw new Tensor.TensorError({ op, message: `${op}: backend returned invalid lazy tensor metadata` })
  }
  return value
}

/**
 * Creates a stochastic gradient descent optimizer with optional momentum,
 * dampening, nesterov, and coupled (L2) weight decay. With no momentum the
 * update is `p -= lr * g`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const sgd = (config: SgdConfig = {}): Effect.Effect<Optimizer<SgdState>> => {
  const momentum = config.momentum ?? 0
  const dampening = config.dampening ?? 0
  const nesterov = config.nesterov ?? false
  const weightDecay = config.weightDecay ?? 0
  if (!Number.isFinite(momentum) || momentum < 0) {
    throw new Error(`sgd: momentum must be non-negative, got ${momentum}`)
  }
  if (nesterov && (momentum <= 0 || dampening !== 0)) {
    throw new Error("sgd: nesterov requires momentum > 0 and dampening = 0")
  }

  return Effect.succeed({
    init: (params) =>
      Effect.gen(function*() {
        yield* checkParams("sgd", params)
        const velocity: Array<Tensor.Any> = []
        if (momentum !== 0) {
          for (const param of params) {
            velocity.push(yield* Tensor.zerosLike(param))
          }
        }
        const first = yield* Tensor.constantLike(params[0], 1)
        return { velocity, first } satisfies SgdState
      }),
    step: (params, grads, state, lr) =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        yield* checkParams("sgd", params)
        yield* checkGrads("sgd", params, grads)
        yield* checkLr("sgd", lr)
        yield* checkStateScalar("sgd", "first-step flag", state.first)
        if (momentum !== 0) {
          yield* checkStateLength("sgd", "velocity", state.velocity, params)
        }
        const updates: Array<Tensor.Lazy> = []
        const velocities: Array<Tensor.Any> = []
        for (let i = 0; i < params.length; i++) {
          if (momentum === 0) {
            let g: Tensor.Any = grads[i]
            if (weightDecay !== 0) {
              g = yield* Tensor.add(g, yield* Tensor.mul(params[i], yield* Tensor.constantLike(params[i], weightDecay)))
            }
            updates.push(yield* Tensor.sub(params[i], yield* Tensor.mul(g, lr)))
            continue
          }
          const rawStep = yield* fromBackend(
            "sgd",
            runtime.node({
              op: "sgdStep",
              inputs: [params[i], grads[i], state.velocity[i], state.first, lr],
              attributes: { momentum, dampening, nesterov, weightDecay }
            })
          )
          const step = yield* Effect.try({
            try: () => validateResult("sgd", runtime, rawStep, params[i]),
            catch: (error) => error as Tensor.TensorError
          })
          const makeOut = (index: number): Effect.Effect<Tensor.Lazy, Tensor.TensorError> =>
            Effect.flatMap(
              fromBackend("sgd", runtime.node({ op: "sgdOut", inputs: [step], attributes: { index } })),
              (value) =>
                Effect.try({
                  try: () => validateResult("sgd", runtime, value, params[i]),
                  catch: (error) => error as Tensor.TensorError
                })
            )
          updates.push(yield* makeOut(0))
          velocities.push(yield* makeOut(1))
        }
        if (momentum === 0) {
          return {
            params: updates,
            state: { velocity: [], first: state.first },
            stateRoots: []
          }
        }
        const first = yield* Tensor.mul(state.first, yield* Tensor.constantLike(state.first, 0))
        return {
          params: updates,
          state: { velocity: velocities, first },
          stateRoots: [...velocities, first]
        }
      }),
    stateRoots: (state) => momentum === 0 ? [] : [...state.velocity, state.first],
    rebuildState: (state, roots) =>
      momentum === 0
        ? state
        : { velocity: roots.slice(0, state.velocity.length), first: roots[state.velocity.length] }
  })
}

interface ResolvedAdamConfig {
  readonly beta1: number
  readonly beta2: number
  readonly eps: number
  readonly weightDecay: number
}

const makeAdam = (op: string, config: ResolvedAdamConfig): Effect.Effect<Optimizer<AdamState>> => {
  const { beta1, beta2, eps, weightDecay } = config
  if (
    !Number.isFinite(beta1) || !Number.isFinite(beta2) || beta1 < 0 || beta1 >= 1 || beta2 < 0 ||
    beta2 >= 1
  ) {
    throw new Error(`${op}: beta1 and beta2 must be in [0, 1), got ${beta1} and ${beta2}`)
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error(`${op}: eps must be positive, got ${eps}`)
  }
  const f32CorrectionError = (beta: number): number => {
    const exact = 1 - beta
    const rounded = 1 - Math.fround(Math.exp(Math.fround(Math.log(beta))))
    return Math.abs(rounded - exact) / exact
  }
  const checkBetaPrecision = (dtype: Tensor.DType): Effect.Effect<void, Tensor.TensorError> =>
    dtype === "f32" && (f32CorrectionError(beta1) > 0.01 || f32CorrectionError(beta2) > 0.01)
      ? new Tensor.TensorError({
        op,
        message:
          `${op}: beta1 and beta2 must keep bias-correction error below 1% at f32 precision, got ${beta1} and ${beta2}`
      })
      : Effect.void

  return Effect.succeed({
    init: (params) =>
      Effect.gen(function*() {
        yield* checkParams(op, params)
        yield* checkBetaPrecision(params[0]!.dtype)
        const m: Array<Tensor.Any> = []
        const v: Array<Tensor.Any> = []
        for (const param of params) {
          m.push(yield* Tensor.zerosLike(param))
          v.push(yield* Tensor.zerosLike(param))
        }
        const t = yield* stepCount(0, params)
        return { m, v, t } satisfies AdamState
      }),
    step: (params, grads, state, lr) =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        yield* checkParams(op, params)
        yield* checkGrads(op, params, grads)
        yield* checkLr(op, lr)
        yield* checkStateScalar(op, "step count", state.t)
        yield* checkBetaPrecision(state.t.dtype)
        yield* checkStateLength(op, "first-moment", state.m, params)
        yield* checkStateLength(op, "second-moment", state.v, params)
        // The bias corrections 1 - beta^t are tensor ops over the state's
        // step count, built once per step and shared by every parameter's
        // update node (the evaluator dedups them into one kernel each).
        const t = yield* Tensor.add(state.t, yield* Tensor.constantLike(state.t, 1))
        const one = yield* Tensor.constantLike(t, 1)
        const c1 = yield* Tensor.add(
          yield* Tensor.neg(
            yield* Tensor.exp(yield* Tensor.mul(t, yield* Tensor.constantLike(t, Math.log(beta1))))
          ),
          one
        )
        const c2 = yield* Tensor.add(
          yield* Tensor.neg(
            yield* Tensor.exp(yield* Tensor.mul(t, yield* Tensor.constantLike(t, Math.log(beta2))))
          ),
          one
        )
        const updates: Array<Tensor.Lazy> = []
        const m: Array<Tensor.Lazy> = []
        const v: Array<Tensor.Lazy> = []
        for (let i = 0; i < params.length; i++) {
          const rawStep = yield* fromBackend(
            op,
            runtime.node({
              op: "adamwStep",
              inputs: [
                params[i],
                grads[i],
                state.m[i],
                state.v[i],
                lr,
                c1,
                c2
              ],
              attributes: { beta1, beta2, eps, weightDecay }
            })
          )
          const step = yield* Effect.try({
            try: () => validateResult(op, runtime, rawStep, params[i]),
            catch: (error) => error as Tensor.TensorError
          })
          const makeOut = (index: number): Effect.Effect<Tensor.Lazy, Tensor.TensorError> =>
            Effect.flatMap(
              fromBackend(op, runtime.node({ op: "adamwOut", inputs: [step], attributes: { index } })),
              (value) =>
                Effect.try({
                  try: () => validateResult(op, runtime, value, params[i]),
                  catch: (error) => error as Tensor.TensorError
                })
            )
          updates.push(yield* makeOut(0))
          m.push(yield* makeOut(1))
          v.push(yield* makeOut(2))
        }
        return {
          params: updates,
          state: { m, v, t },
          stateRoots: [...m, ...v, t]
        }
      }),
    stateRoots: (state) => [...state.m, ...state.v, state.t],
    rebuildState: (state, roots) => ({
      m: roots.slice(0, state.m.length),
      v: roots.slice(state.m.length, state.m.length * 2),
      t: roots[state.m.length * 2]
    })
  })
}

/**
 * Creates an Adam optimizer with the standard defaults (`beta1 = 0.9`,
 * `beta2 = 0.999`, `eps = 1e-8`). The learning rate is a per-step input
 * to `step`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const adam = (config: AdamConfig = {}): Effect.Effect<Optimizer<AdamState>> =>
  makeAdam("adam", {
    beta1: config.beta1 ?? 0.9,
    beta2: config.beta2 ?? 0.999,
    eps: config.eps ?? 1e-8,
    weightDecay: 0
  })

/**
 * Creates an AdamW optimizer: Adam with decoupled weight decay (default
 * `0.01`), applied as `p *= (1 - lr * weightDecay)` before the adaptive
 * update. The learning rate is a per-step input to `step`.
 *
 * @since 0.1.0
 * @category constructors
 */
export const adamW = (config: AdamWConfig = {}): Effect.Effect<Optimizer<AdamState>> =>
  makeAdam("adamW", {
    beta1: config.beta1 ?? 0.9,
    beta2: config.beta2 ?? 0.999,
    eps: config.eps ?? 1e-8,
    weightDecay: config.weightDecay ?? 0.01
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
): Effect.Effect<Array<Tensor.Lazy>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
): Effect.Effect<Array<Tensor.Lazy>, Tensor.TensorError, Runtime.Runtime> =>
  Effect.gen(function*() {
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
    const scale = yield* Tensor.minimum(
      yield* Tensor.mul(
        yield* Tensor.reciprocal(yield* Tensor.add(norm, yield* Tensor.constantLike(norm, 1e-6))),
        yield* Tensor.constantLike(norm, maxNorm)
      ),
      yield* Tensor.constantLike(norm, 1)
    )
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
 * from materialized tensors via `optimizer.rebuildState`: both are
 * plain leaves of the next step's graph, so graph depth stays
 * O(model depth) no matter how many steps run.
 *
 * `lr` is the step's learning rate as a 0-d float tensor on the
 * parameters' device — lift the step's scheduled value with
 * `Tensor.full([], schedule(step), { dtype: params[0].dtype })`.
 *
 * @since 0.1.0
 * @category destructors
 */
export const step = <S, P extends ReadonlyArray<Tensor.Any>>(
  optimizer: Optimizer<S>,
  loss: Tensor.Any,
  params: P,
  state: S,
  lr: Tensor.Any
): Effect.Effect<
  { readonly loss: Tensor.Concrete; readonly params: Materialized<P>; readonly state: S },
  Gradient.GradError | Tensor.TensorError,
  Runtime.Runtime
> =>
  Effect.gen(function*() {
    const grads = yield* Gradient.grad(loss, params)
    const next = yield* optimizer.step(params, grads, state, lr)
    const evaluated = yield* Tensor.compute([loss, ...next.params, ...next.stateRoots])
    const [evaluatedLoss, ...rest] = evaluated
    return {
      loss: evaluatedLoss,
      params: rest.slice(0, next.params.length) as Materialized<P>,
      state: optimizer.rebuildState(next.state, rest.slice(next.params.length))
    }
  })

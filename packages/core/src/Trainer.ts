/**
 * A {@link Trainer} pairs a model with an optimizer, learning-rate schedule,
 * loss, data source, and stop policy.
 * Its {@link Trainer.train} method repeatedly builds or runs
 * `loss(forward(params, input), target)`, differentiates it, applies the
 * optimizer, and materializes loss, parameters, and state together.
 *
 * {@link make} creates the compiled trainer. The first step for an input
 * signature traces and freezes the forward, loss, backward, and update graph;
 * later steps reuse that program. Parameters, optimizer state roots, input, and
 * target are tensor inputs, while the scheduled learning rate is a runtime
 * scalar. Signatures include runtime identity and every tensor input's shape,
 * dtype, storage encoding, and placement; values and learning-rate scalar
 * values are excluded. New signatures trace automatically into a 32-ready-entry
 * LRU cache, with concurrent misses for one signature sharing a trace.
 * {@link makeUncompiled} creates the reference trainer that constructs and
 * evaluates the full graph every step.
 *
 * Tracing executes the effects returned by `model.forward`, `config.loss`,
 * and `optimizer.step` only on a cache miss. Those effects must describe a
 * graph and must not be used for per-step application side effects, because
 * cache hits do not run them. The data sampler and `onStep` effect, and the
 * learning-rate and stop functions, still run on every step. Frozen random
 * nodes such as `randn` and dropout draw fresh values on each program call.
 * Compiled and uncompiled trainers share the same graph definition;
 * floating-point results remain subject to backend, dtype, and fusion rounding
 * behavior. Optimizer root count/order/metadata must remain stable because the
 * same schema is used for compiled bindings and checkpoint restoration.
 *
 * @since 0.1.0
 */
import { Clock, Duration, Effect, Exit, Predicate } from "effect"
import * as Gradient from "./Gradient.ts"
import type { LearningRate } from "./LearningRate.ts"
import * as Model from "./Model.ts"
import * as Optimizer from "./Optimizer.ts"
import * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * The training data for {@link Trainer.train}: one input batch and its
 * target. The trainer reads but never clears these tensors.
 *
 * @since 0.1.0
 * @category models
 */
export interface TrainData {
  /** Model input; its shape, dtype, storage, and placement participate in the compiled cache key. */
  readonly input: Tensor.Any
  /** Target passed unchanged to the configured loss; it also participates in the cache key. */
  readonly target: Tensor.Any
}

/**
 * The batches {@link Trainer.train} consumes: either a fixed `(input, target)`
 * pair (full-batch, the same handles every step) or a sampler called
 * with the 1-based step number to produce that step's batch (mini-batch
 * training). The trainer borrows each returned pair and never clears it.
 *
 * @since 0.1.0
 * @category models
 */
export type TrainDataSource<E = never, R = never> =
  | TrainData
  | ((step: number) => Effect.Effect<TrainData, E, R>)

/**
 * Per-step progress reported to {@link TrainConfig.onStep} and
 * {@link TrainConfig.stop}: the 1-based step number, the step's loss
 * value, and the time elapsed since the run began. The clock starts at
 * each `train` invocation unless {@link Resume.startedAt} carries an
 * earlier anchor; chunked runs that want one continuous clock pass
 * the anchor through.
 *
 * @since 0.1.0
 * @category models
 */
export interface TrainStep {
  /** Global, 1-based completed-step number, including any resumed steps. */
  readonly step: number
  /** Materialized scalar loss for this completed step. */
  readonly loss: number
  /** Duration since this invocation began or since the resume anchor. */
  readonly elapsed: Duration.Duration
}

/**
 * The trainer's parameter-cast policy. `"f32"` adds no casts; actual
 * parameter, data, forward, loss, and optimizer dtypes remain those supplied
 * by the model and configuration. With the usual f32 parameters, optimizer
 * state and updates therefore remain f32; built-in optimizers still require
 * f32 or f64 master parameters.
 *
 * `"mixedBf16"` casts parameters to bf16 only at the model-forward boundary.
 * Gradients flow through those casts to the master parameters, whose original
 * dtype is retained for optimizer state and updates (normally f32). Inputs
 * and targets are not cast, and every selected model/loss operation must
 * support its resulting dtypes. This mode requires the runtime's
 * `mixed-bf16` feature, currently provided by the bundled Metal runtime;
 * unsupported runtimes fail when training starts. It is a parameter-cast
 * policy, not general AMP or autocast. It performs no automatic activation or
 * loss casting beyond the model graph, loss scaling, finite-gradient detection,
 * overflow skipping, or dynamic
 * scale adjustment.
 *
 * @since 0.1.0
 * @category models
 */
export type Precision = "f32" | "mixedBf16"

/**
 * Configuration for {@link make} and {@link makeUncompiled}. `loss` accepts any
 * function shaped like `Loss.mse`,
 * `(prediction, target) => Effect<Lazy>`. Exports from the `Loss` module can be
 * passed directly. `lr` is a learning-rate schedule from the `LearningRate`
 * module. The trainer evaluates it with the 0-based step number and passes the
 * value to the update as a 0-d tensor. `LearningRate.constant(0.1)` provides a
 * fixed rate. `onStep` runs after every step with the loss value. Throttle
 * inside the callback.
 *
 * `stop` decides when training ends. The trainer checks it after every step, so
 * at least one step always runs. A policy is a plain function. For example,
 * `({ step }) => step >= 3000` stops on a step count,
 * `({ loss }) => loss < 0.01` stops on a loss target,
 * `({ elapsed }) => Duration.toSeconds(elapsed) > 60` stops on a
 * wall-clock budget. Combine conditions with `||`, or close over other state.
 *
 * `data` is either a fixed `(input, target)` pair used every step
 * (full-batch) or a sampler producing each step's batch (mini-batch).
 *
 * Each effectful field keeps its own error and requirement channels:
 * `EL`/`RL` for `loss`, `ED`/`RD` for `data`, and `EO`/`RO` for `onStep`. The
 * call site infers them, so the fields do not need a shared environment in
 * advance. `ModelError`, `TensorError`, and
 * `GradError` are added by the trainer itself. The synchronous `lr` and `stop`
 * functions have no typed error channel; thrown exceptions are defects.
 *
 * @since 0.1.0
 * @category models
 */
export interface TrainConfig<
  S,
  EL = never,
  RL = never,
  ED = never,
  RD = never,
  EO = never,
  RO = never
> {
  /** Optimizer used for every step; its state type is `S`. */
  readonly optimizer: Optimizer.Optimizer<S>
  /**
   * Schedule called every step with the 0-based global step (`step - 1`).
   * Its numeric result is not validated before being bound as a runtime scalar.
   */
  readonly lr: LearningRate
  /**
   * Builds a scalar loss from the model prediction and target. Scalar shape is
   * enforced later by autodiff. In a compiled
   * trainer its effect runs while tracing a cache miss, not on cache hits, so
   * it must be a graph-building effect rather than a per-step callback.
   */
  readonly loss: (
    pred: Tensor.Any,
    target: Tensor.Any
  ) => Effect.Effect<Tensor.Lazy, EL | Tensor.TensorError, Runtime.Runtime | RL>
  /** Fixed training data or an effectful sampler invoked with each 1-based global step. */
  readonly data: TrainDataSource<ED, RD>
  /** Synchronous stop policy, checked after `onStep`; at least one step always runs. */
  readonly stop: (info: TrainStep) => boolean
  /** Effect run after each successful step and before `stop` is checked. */
  readonly onStep?: (info: TrainStep) => Effect.Effect<void, EO, RO>
  /**
   * Parameter-cast policy. Defaults to `"f32"`; see {@link Precision}.
   */
  readonly precision?: Precision
}

/**
 * The result of {@link Trainer.train}. It contains the trained parameters as
 * materialized leaves ready for `forward`, `save`, or more training, the final
 * optimizer state, and the final step's loss.
 *
 * @since 0.1.0
 * @category models
 */
export interface Trained<S> {
  /** Final materialized parameters in model order; ownership transfers to the caller. */
  readonly params: ReadonlyArray<Tensor.Concrete>
  /**
   * Final optimizer state. Every concrete tensor returned by
   * `optimizer.stateRoots(state)` is caller-owned. Inert non-root fields remain
   * as supplied by the optimizer.
   */
  readonly state: S
  /** Final step's scalar loss. */
  readonly loss: number
  /** Global, 1-based number of the final completed step. */
  readonly step: number
}

/**
 * A resumable training position with optimizer state and global step count from
 * a previous {@link Trained} value. Resume it only together with that value's
 * matching `params`, using the same model, training configuration, and
 * parameter order. The trainer does not validate that relationship up front;
 * mismatched state is rejected only where tensor metadata can be checked and
 * otherwise resumes incorrectly. Carrying `startedAt`
 * also preserves one elapsed-time clock across calls.
 *
 * @since 0.1.0
 * @category models
 */
export interface Resume<S> {
  /** Optimizer state paired with the matching parameters from the same completed step. */
  readonly state: S
  /** Prior non-negative integer step count; the next sample uses `step + 1`. Not validated here. */
  readonly step: number
  /**
   * Millisecond anchor in the active Effect `Clock.currentTimeMillis` time base;
   * with the live clock this is Unix-epoch time. When set,
   * {@link TrainStep.elapsed} measures from this point instead of the
   * current `train` invocation, so a run chunked into several `train`
   * calls keeps one continuous clock. When absent, the clock starts now. The
   * trainer does not validate this value, so an incompatible or future anchor
   * can yield meaningless elapsed time.
   */
  readonly startedAt?: number
}

/**
 * A model and training configuration with a reusable training loop.
 * Trainers from {@link make} execute frozen programs keyed by runtime identity
 * and each parameter, state, input, and target tensor's placement, shape,
 * dtype, and encoded-storage metadata. Tensor values and the learning-rate
 * value do not affect the key. Trainers from {@link makeUncompiled} expose the
 * same API but build each step graph.
 *
 * @since 0.1.0
 * @category models
 */
export interface Trainer<S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never> {
  /** Model whose parameter order and forward graph define the training step. */
  readonly model: Model.Model
  /**
   * Configuration retained by reference. Mutating it through an alias does not
   * invalidate compiled cache entries; treat the model, optimizer, and graph
   * building fields as immutable for the trainer's lifetime.
   */
  readonly config: TrainConfig<S, EL, RL, ED, RD, EO, RO>
  /**
   * Runs at least one step from `params` and calls `onStep` before checking
   * `stop`. A `resume` must be accompanied by its matching `params`; omitting
   * `resume` initializes fresh optimizer state.
   *
   * The loop owns every parameter/state generation it materializes and releases
   * each superseded generation after the next step consumes it. Supplied
   * parameters, resume roots, and data tensors remain caller-owned. Failure or
   * interruption releases the current loop-owned generation; on success, final
   * parameters and concrete state roots transfer atomically to the caller.
   *
   * Fails in the typed channel for model arity or mixed-precision support,
   * tensor/gradient/backend errors, or configured loss, data, and callback
   * errors. Exceptions thrown by the schedule or synchronous policies are
   * defects rather than typed failures.
   */
  readonly train: (
    params: Model.Params,
    resume?: Resume<S>
  ) => Effect.Effect<
    Trained<S>,
    Model.ModelError | Tensor.TensorError | Gradient.GradError | EL | ED | EO,
    Runtime.Runtime | RL | RD | RO
  >
  /**
   * Current JavaScript ready-or-pending cached-program count and cumulative
   * trace-attempt count, including failures and retraces. Pending traces can
   * temporarily exceed the ready-entry capacity. This does not count native
   * cold compilations. Uncompiled trainers always report zero for both values.
   */
  readonly stats: Effect.Effect<Tensor.CompileStats>
  /**
   * Drops JavaScript cached-program references and signature history; it does
   * not clear tensors, native structural or pipeline caches, or the cumulative
   * trace count. A later step retraces, although an already in-flight trace may
   * repopulate the cache after clearing. This is a no-op when uncompiled.
   */
  readonly clear: Effect.Effect<void>
}

/**
 * Creates a compiled trainer for `model` without requiring a runtime or tracing
 * a graph yet. Programs are traced lazily and retained in a trainer-owned
 * 32-ready-entry LRU cache across `train` calls. Cache eviction or
 * {@link Trainer.clear} drops JavaScript program references but not native
 * structural/pipeline cache entries; dropping the trainer makes its remaining
 * JavaScript cache collectable. The model and configuration are retained by
 * reference rather than cloned or frozen; mutating graph-building behavior
 * requires clearing the cache and remains unsupported while a trace is active.
 *
 * @since 0.1.0
 * @category constructors
 */
export const make = <S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<Trainer<S, EL, RL, ED, RD, EO, RO>> =>
  Effect.sync(() => {
    const cache = Tensor.makeProgramCache(undefined)
    return {
      model,
      config,
      train: (params, resume) => trainLoop(model, config, params, resume, cache),
      get stats() {
        return cache.stats
      },
      get clear() {
        return cache.clear
      }
    }
  })

/**
 * Creates the uncompiled reference trainer without requiring a runtime. Every
 * step constructs and evaluates the full forward, loss, backward, and update
 * graph. Its
 * {@link Trainer.stats} values remain zero and {@link Trainer.clear} is a
 * no-op.
 *
 * @since 0.1.0
 * @category constructors
 */
export const makeUncompiled = <S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<Trainer<S, EL, RL, ED, RD, EO, RO>> =>
  Effect.succeed({
    model,
    config,
    train: (params, resume) => trainLoop(model, config, params, resume, undefined),
    stats: Effect.succeed({ cached: 0, compiled: 0 }),
    clear: Effect.void
  })

const uncompiledStep = <S, EL, RL, ED, RD, EO, RO>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>,
  params: Model.Params,
  state: S,
  data: TrainData,
  step: number
): Effect.Effect<
  { readonly loss: number; readonly params: ReadonlyArray<Tensor.Concrete>; readonly state: S },
  Model.ModelError | Tensor.TensorError | Gradient.GradError | EL,
  Runtime.Runtime | RL
> =>
  Effect.suspend(() => {
    const owned: Array<Tensor.Concrete> = []
    return Effect.onExit(
      Effect.gen(function*() {
        const forwardParams = config.precision === "mixedBf16"
          ? yield* Effect.all(params.map((param) => Tensor.cast(param, "bf16")))
          : params
        const prediction = yield* model.forward(forwardParams, data.input)
        const lossTensor = yield* config.loss(prediction, data.target)
        const lr = yield* Tensor.constantLike(params[0], config.lr(step - 1))
        const result = yield* Optimizer.step(config.optimizer, lossTensor, params, state, lr)
        owned.push(
          result.loss,
          ...result.params,
          ...config.optimizer.stateRoots(result.state).filter(Tensor.isTensor)
        )
        const loss = (yield* Tensor.toNumberArray(result.loss))[0]
        yield* Tensor.clear(result.loss)
        return { loss, params: result.params, state: result.state }
      }),
      (exit) =>
        Exit.isFailure(exit)
          ? Tensor.clearAll(owned)
          : Effect.void
    )
  })

// Traces the step graph against placeholder leaves: parameter, state-root,
// input, and target tensor slots, then one scalar slot for the learning
// rate. The roots are [loss, ...nextParams, ...nextStateRoots]. This is the
// graph transform that the uncompiled step computes, differentiated at trace
// time. The placeholders take their signatures from the current step's
// tensors, so the trace is valid for exactly one cache-key signature.
const traceStep = <S, EL, RL, ED, RD, EO, RO>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>,
  params: Model.Params,
  stateRoots: ReadonlyArray<Tensor.Any>,
  state: S,
  data: TrainData
): Effect.Effect<
  Tensor.CompiledProgram,
  Model.ModelError | Tensor.TensorError | Gradient.GradError | EL,
  Runtime.Runtime | RL
> =>
  Effect.gen(function*() {
    const optimizer = config.optimizer
    const paramCount = params.length
    const stateCount = stateRoots.length
    const paramPlaceholders: Array<Tensor.Lazy> = []
    for (let i = 0; i < paramCount; i++) {
      paramPlaceholders.push(yield* Tensor.makeInput(i, params[i]))
    }
    const statePlaceholders: Array<Tensor.Lazy> = []
    for (let i = 0; i < stateCount; i++) {
      statePlaceholders.push(yield* Tensor.makeInput(paramCount + i, stateRoots[i]))
    }
    const input = yield* Tensor.makeInput(paramCount + stateCount, data.input)
    const target = yield* Tensor.makeInput(paramCount + stateCount + 1, data.target)
    // The learning rate is the step's only runtime scalar, declared with
    // the parameters' dtype. This matches the lifting that the uncompiled loop
    // applies with Tensor.constant.
    const lr = yield* Tensor.makeScalarInput(
      paramCount + stateCount + 2,
      params[0]?.dtype ?? "f32"
    )
    const placeholderState = optimizer.rebuildState(state, statePlaceholders)
    const forwardParams = config.precision === "mixedBf16"
      ? yield* Effect.all(paramPlaceholders.map((param) => Tensor.cast(param, "bf16")))
      : paramPlaceholders
    const prediction = yield* model.forward(forwardParams, input)
    const lossTensor = yield* config.loss(prediction, target)
    const grads = yield* Gradient.grad(lossTensor, paramPlaceholders)
    const next = yield* optimizer.step(paramPlaceholders, grads, placeholderState, lr)
    return yield* Tensor.freezeProgram([lossTensor, ...next.params, ...next.stateRoots])
  })

const compiledStep = <S, EL, RL, ED, RD, EO, RO>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>,
  cache: Tensor.ProgramCache,
  params: Model.Params,
  state: S,
  data: TrainData,
  step: number
): Effect.Effect<
  { readonly loss: number; readonly params: ReadonlyArray<Tensor.Concrete>; readonly state: S },
  Model.ModelError | Tensor.TensorError | Gradient.GradError | EL,
  Runtime.Runtime | RL
> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    const optimizer = config.optimizer
    const stateRoots = optimizer.stateRoots(state)
    const inputs = [...params, ...stateRoots, data.input, data.target]
    const program = yield* Tensor.cachedProgram(
      cache,
      Tensor.signatureOf(inputs, runtime),
      () => traceStep(model, config, params, stateRoots, state, data)
    )
    const owned: Array<Tensor.Concrete> = []
    return yield* Effect.onExit(
      Effect.gen(function*() {
        const outputs = yield* Tensor.runProgram(program, inputs, [config.lr(step - 1)])
        owned.push(...outputs)
        const loss = (yield* Tensor.toNumberArray(outputs[0]))[0]
        const trained = outputs.slice(1, 1 + params.length)
        const nextState = optimizer.rebuildState(state, outputs.slice(1 + params.length))
        yield* Tensor.clear(outputs[0])
        return { loss, params: trained, state: nextState }
      }),
      (exit) =>
        Exit.isFailure(exit)
          ? Tensor.clearAll(owned)
          : Effect.void
    )
  })

// Both forms start with the supplied parameters, call `onStep` after each step,
// and continue until `stop` says otherwise. At least one step always runs.
// Without a cache each step builds and evaluates the full step graph;
// with one each step is a single frozen-program call. The current generation
// is released on replacement and retained for failure cleanup until ownership
// transfers to the next generation.
const trainLoop = <S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>,
  initial: Model.Params,
  resume: Resume<S> | undefined,
  cache: Tensor.ProgramCache | undefined
): Effect.Effect<
  Trained<S>,
  Model.ModelError | Tensor.TensorError | Gradient.GradError | EL | ED | EO,
  Runtime.Runtime | RL | RD | RO
> =>
  Effect.suspend(() => {
    let owned: ReadonlyArray<Tensor.Concrete> = []
    const releaseOwned = (tensors: ReadonlyArray<Tensor.Concrete>) => Tensor.clearAll(tensors)
    return Effect.onExit(
      Effect.gen(function*() {
        let params: Model.Params = initial
        const runtime = yield* Runtime.Runtime
        if (config.precision === "mixedBf16" && !runtime.capabilities.features.includes("mixed-bf16")) {
          return yield* new Model.ModelError({
            op: "train",
            message:
              `mixedBf16 precision is not supported by ${runtime.backend.name} on ${runtime.placement.description}`
          })
        }
        let state: S
        if (resume !== undefined) {
          state = resume.state
        } else {
          state = yield* config.optimizer.init(params)
          const roots = config.optimizer.stateRoots(state).filter(Tensor.isTensor)
          const callerOwnedParams = new Set(params.filter(Tensor.isTensor))
          const stateOwned = roots.filter((root) => !callerOwnedParams.has(root))
          owned = [...owned, ...stateOwned]
        }
        let step = resume?.step ?? 0
        let loss = Number.NaN
        let trained: ReadonlyArray<Tensor.Concrete>
        const started = resume?.startedAt ?? (yield* Clock.currentTimeMillis)
        if (cache !== undefined) {
          // Program inputs must be materialized buffers. The initial
          // parameters and state are lazy graph values, so evaluate them
          // once up front; every later step returns materialized values.
          const roots = [...params, ...config.optimizer.stateRoots(state)]
          const previousOwned = owned
          const materialized = yield* Tensor.compute(roots)
          owned = [...previousOwned, ...materialized]
          params = materialized.slice(0, params.length)
          state = config.optimizer.rebuildState(state, materialized.slice(params.length))
          yield* releaseOwned(previousOwned)
          owned = materialized
        }
        do {
          step++
          const data: TrainData = Predicate.isFunction(config.data)
            ? yield* config.data(step)
            : config.data
          const previousOwned = owned
          const result = yield* (cache !== undefined
            ? compiledStep(model, config, cache, params, state, data, step)
            : uncompiledStep(model, config, params, state, data, step))
          const nextOwned = [
            ...result.params,
            ...config.optimizer.stateRoots(result.state).filter(Tensor.isTensor)
          ]
          owned = [...previousOwned, ...nextOwned]
          yield* releaseOwned(previousOwned)
          owned = nextOwned
          loss = result.loss
          trained = result.params
          params = result.params
          state = result.state
          const info: TrainStep = {
            step,
            loss,
            elapsed: Duration.millis((yield* Clock.currentTimeMillis) - started)
          }
          if (config.onStep !== undefined) {
            yield* config.onStep(info)
          }
          if (config.stop(info)) break
        } while (true)
        return { params: trained, state, loss, step }
      }),
      (exit) => Exit.isFailure(exit) ? releaseOwned(owned) : Effect.void
    )
  })

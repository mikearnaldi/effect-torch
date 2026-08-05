/**
 * Training as an encapsulated value. A {@link Trainer} pairs a model
 * with a training configuration — optimizer, learning-rate schedule,
 * loss, data source, stop policy — and its {@link Trainer.train} method
 * runs the loop: initialize with the given parameters (or `model.init`),
 * then repeatedly build `loss(forward(params, input), target)`,
 * differentiate it, extend the graph with the optimizer update, and
 * compute loss, parameters, and state in a single walk — one forward
 * pass, one backward pass, one async boundary per step, with graph depth
 * staying O(model depth).
 *
 * {@link make} builds the uncompiled trainer: every step constructs the
 * full forward+backward+update graph. {@link compile} returns a trainer
 * that freezes the step graph into a native program the first time a
 * batch signature is seen and reuses it afterwards — a compiled trainer
 * is still a `Trainer`, trained by the same `train` method: one step is
 * one program call with parameters, optimizer state roots, batch, and
 * the scheduled learning rate in, and loss, updated parameters, and
 * updated state roots out. A batch whose signature (shapes, dtypes,
 * device) differs from every cached program traces a new one, so shape
 * changes (a partial last batch, a different eval batch size) recompile
 * automatically up to the cache capacity.
 *
 * Since there is one semantic definition of a step — the compiled trace
 * is exactly the uncompiled step's graph transform — compiled and
 * uncompiled loops agree step-for-step on deterministic graphs and in
 * distribution on stochastic ones (`randn`/`dropout` draw fresh per step
 * in both).
 *
 * @since 0.1.0
 */
import { Duration, Effect } from "effect"
import { CurrentDevice, type DeviceKind } from "./Device.ts"
import * as Gradient from "./Gradient.ts"
import type { LearningRate } from "./LearningRate.ts"
import * as Model from "./Model.ts"
import * as Optimizer from "./Optimizer.ts"
import * as Tensor from "./Tensor.ts"

/**
 * The training data for { Trainer.train}: a full-batch input and its target.
 * (A `Dataset` module with batching is future work.)
 *
 * @since 0.1.0
 * @category models
 */
export interface TrainData {
  readonly input: Tensor.Any
  readonly target: Tensor.Any
}

/**
 * The batches { Trainer.train} consumes: either a fixed `(input, target)`
 * pair (full-batch — the same tensors every step) or a sampler called
 * with the 1-based step number to produce that step's batch (mini-batch
 * training).
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
 * value, and the time elapsed since this `train` run began — each
 * invocation of `train` starts its own clock, so continued training and
 * re-runs never share a start time.
 *
 * @since 0.1.0
 * @category models
 */
export interface TrainStep {
  readonly step: number
  readonly loss: number
  readonly elapsed: Duration.Duration
}

/**
 * The numeric precision of the training loop. `"f32"` keeps parameters,
 * forward, backward, and optimizer state in f32 end to end. `"mixedBf16"`
 * runs forward/backward in bf16 while the optimizer owns f32 master
 * weights: the step graph casts masters to bf16 at the forward boundary
 * and gradients flow back through the casts, so the update arithmetic
 * stays f32. Metal only.
 *
 * @since 0.1.0
 * @category models
 */
export type Precision = "f32" | "mixedBf16"

/**
 * Configuration for {@link make}. `loss` is any loss function in the
 * shape of {@link Loss.mse} — `(prediction, target) => Effect<Lazy>` —
 * so the `Loss` module's exports slot in directly. `lr` is the
 * learning-rate schedule (see the `LearningRate` module): it is evaluated
 * with the 0-based step number on every step and the value flows into the
 * update as a 0-d tensor — `LearningRate.constant(0.1)` is the fixed-rate
 * case. `onStep` runs after every step with the step's loss value —
 * throttle inside the callback.
 *
 * `stop` decides when training ends; it is checked after every step (at
 * least one step always runs), so any policy is a plain function:
 * `({ step }) => step >= 3000` stops on a step count,
 * `({ loss }) => loss < 0.01` stops on a loss target,
 * `({ elapsed }) => Duration.toSeconds(elapsed) > 60` stops on a
 * wall-clock budget, and the three compose with `||` — or close over any
 * other state you track.
 *
 * `data` is either a fixed `(input, target)` pair used every step
 * (full-batch) or a sampler producing each step's batch (mini-batch).
 *
 * The effectful fields carry their own error and requirement channels
 * (`EL`/`RL` for `loss`, `ED`/`RD` for `data`, `EO`/`RO` for `onStep`),
 * inferred at the call site, so a loss needing the current device, a
 * sampler hitting a dataset, and a logging `onStep` compose without
 * being pre-widened to a common environment.
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
  readonly optimizer: Optimizer.Optimizer<S>
  readonly lr: LearningRate
  readonly loss: (
    pred: Tensor.Any,
    target: Tensor.Any
  ) => Effect.Effect<Tensor.Lazy, EL | Tensor.TensorError, CurrentDevice | RL>
  readonly data: TrainDataSource<ED, RD>
  readonly stop: (info: TrainStep) => boolean
  readonly onStep?: (info: TrainStep) => Effect.Effect<void, EO, RO>
  /**
   * Defaults to `"f32"`; see {@link Precision}.
   */
  readonly precision?: Precision
}

/**
 * The result of { Trainer.train}: the trained parameters (materialized
 * leaves, ready for `forward`, `save`, or more training), the final
 * optimizer state, and the final step's loss.
 *
 * @since 0.1.0
 * @category models
 */
export interface Trained<S> {
  readonly params: ReadonlyArray<Tensor.Concrete>
  readonly state: S
  readonly loss: number
  readonly step: number
}

/**
 * A resumable training position: the optimizer state and global step
 * count exactly as a previous {@link Trained} returned them. Passing it
 * back to `train` continues the run as if it had never stopped — same
 * optimizer moments, same step numbering for `stop`/`onStep`.
 *
 * @since 0.1.0
 * @category models
 */
export interface Resume<S> {
  readonly state: S
  readonly step: number
}

/**
 * A trainer: a model plus an encapsulated training configuration, with
 * the training loop as a method. {@link make} creates the uncompiled
 * form (every step builds the step graph); {@link compile} returns a
 * {@link CompiledTrainer} whose steps run as frozen native programs —
 * still a `Trainer`, with the same `train` method.
 *
 * @since 0.1.0
 * @category models
 */
export interface Trainer<S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never> {
  readonly model: Model.Model
  readonly config: TrainConfig<S, EL, RL, ED, RD, EO, RO>
  /**
   * Runs the training loop: initialize with `params` (or `model.init`
   * when omitted — continued training and fine-tuning pass a checkpoint's
   * parameters), then step until `stop` says otherwise (at least one
   * step always runs), calling `onStep` after every step. With an
   * uncompiled trainer each step builds and evaluates the full step
   * graph; with a compiled one each step is a single frozen-program
   * call. Both share the loop's semantics — same data sampling, same
   * schedule, same stop policy — and agree step-for-step on
   * deterministic graphs.
   */
  readonly train: (
    params?: Model.Params,
    resume?: Resume<S>
  ) => Effect.Effect<
    Trained<S>,
    Model.ModelError | Tensor.TensorError | Gradient.GradError | EL | ED | EO,
    CurrentDevice | RL | RD | RO
  >
}

const CompiledTypeId: unique symbol = Symbol.for("@effect-torch/core/Trainer/Compiled")

/**
 * @since 0.1.0
 * @category symbols
 */
export type CompiledTypeId = typeof CompiledTypeId

/**
 * A trainer whose steps run as frozen native programs (see
 * {@link compile}): still a {@link Trainer} in every respect, with the
 * shape-keyed program cache's diagnostics and release added as required
 * members.
 *
 * @since 0.1.0
 * @category models
 */
export interface CompiledTrainer<S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never>
  extends Trainer<S, EL, RL, ED, RD, EO, RO>
{
  readonly [CompiledTypeId]: CompiledTypeId
  /**
   * Shape-cache diagnostics: programs cached, traces performed.
   */
  readonly stats: () => Tensor.CompileStats
  /**
   * Clears the cached programs early (they are otherwise collected by GC).
   */
  readonly clear: () => Effect.Effect<void>
}

/**
 * Returns `true` if the trainer was produced by {@link compile} and
 * narrows it to {@link CompiledTrainer}.
 *
 * @since 0.1.0
 * @category refinements
 */
export const isCompiled = <S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never>(
  trainer: Trainer<S, EL, RL, ED, RD, EO, RO>
): trainer is CompiledTrainer<S, EL, RL, ED, RD, EO, RO> => CompiledTypeId in trainer

/**
 * Creates a trainer for `model` from a training configuration. The
 * returned trainer runs the uncompiled loop: every step constructs the
 * forward, backward, and update graphs.
 *
 * @since 0.1.0
 * @category constructors
 */
export const make = <S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>
): Effect.Effect<Trainer<S, EL, RL, ED, RD, EO, RO>> =>
  Effect.succeed({
    model,
    config,
    train: (params, resume) => trainLoop(model, config, params, resume, undefined)
  })

/**
 * Options for {@link compile}.
 *
 * @since 0.1.0
 * @category compilation
 */
export interface CompileOptions {
  /**
   * Shape-cache capacity in programs. The first step with a new batch
   * signature traces and freezes a step program; later steps with the
   * same signature reuse it. Defaults to 32.
   */
  readonly cacheCapacity?: number
}

/**
 * Returns a compiled trainer: same configuration, but each step runs as a
 * frozen native program — parameters, optimizer state roots, input, and
 * target in (plus the step's scheduled learning rate as a runtime
 * scalar), loss, updated parameters, and updated state roots out, in one
 * native call per step. The step graph is traced from the same
 * definitions the uncompiled loop uses (`model.forward`, `config.loss`,
 * `Gradient.grad`, `optimizer.step`), so the two agree step-for-step on
 * deterministic graphs.
 *
 * Recompilation is automatic and shape-keyed: a step whose batch (or
 * parameter, or state) signature differs from every cached program traces
 * a new one, up to `cacheCapacity` programs with least-recently-used
 * eviction. The compiled trainer is still a {@link Trainer} — its
 * `train` method runs the same loop as the uncompiled form.
 *
 * @since 0.1.0
 * @category compilation
 */
export const compile = <S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never>(
  trainer: Trainer<S, EL, RL, ED, RD, EO, RO>,
  options: CompileOptions = {}
): Effect.Effect<CompiledTrainer<S, EL, RL, ED, RD, EO, RO>> =>
  Effect.sync(() => {
    const cache = Tensor.makeProgramCache(options.cacheCapacity)
    return {
      [CompiledTypeId]: CompiledTypeId,
      ...trainer,
      train: (params, resume) => trainLoop(trainer.model, trainer.config, params, resume, cache),
      stats: cache.stats,
      clear: cache.clear
    }
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
  CurrentDevice | RL
> =>
  Effect.gen(function* () {
    const forwardParams = config.precision === "mixedBf16"
      ? yield* Effect.all(params.map((param) => Tensor.cast(param, "bf16")))
      : params
    const prediction = yield* model.forward(forwardParams, data.input)
    const lossTensor = yield* config.loss(prediction, data.target)
    const lr = yield* Tensor.constant(config.lr(step - 1), { dtype: params[0].dtype })
    const result = yield* Optimizer.step(config.optimizer, lossTensor, params, state, lr)
    return {
      loss: (yield* Tensor.toNumberArray(result.loss))[0],
      params: result.params,
      state: result.state
    }
  })

// Traces the step graph against placeholder leaves: parameter, state-root,
// input, and target tensor slots, then one scalar slot for the learning
// rate. The roots are [loss, ...nextParams, ...nextStateRoots] — the same
// graph transform the uncompiled step computes, differentiated at trace
// time. The placeholders take their signatures from the current step's
// tensors, so the trace is valid for exactly one cache-key signature.
const traceStep = <S, EL, RL, ED, RD, EO, RO>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>,
  params: Model.Params,
  stateRoots: ReadonlyArray<Tensor.Any>,
  state: S,
  data: TrainData,
  device: DeviceKind
): Effect.Effect<
  Tensor.NativeCompiledProgram,
  Model.ModelError | Tensor.TensorError | Gradient.GradError | EL,
  CurrentDevice | RL
> =>
  Effect.gen(function* () {
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
    // the parameters' dtype — the same lifting the uncompiled loop
    // applies with Tensor.constant.
    const lr = yield* Tensor.makeScalarInput(
      paramCount + stateCount + 2,
      params[0]?.dtype ?? "f32",
      device
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
  CurrentDevice | RL
> =>
  Effect.gen(function* () {
    const device = yield* CurrentDevice
    const optimizer = config.optimizer
    const stateRoots = optimizer.stateRoots(state)
    const inputs = [...params, ...stateRoots, data.input, data.target]
    const program = yield* Tensor.cachedProgram(
      cache,
      Tensor.signatureOf(inputs),
      traceStep(model, config, params, stateRoots, state, data, device)
    )
    const outputs = yield* Tensor.runProgram(program, inputs, [config.lr(step - 1)])
    const loss = (yield* Tensor.toNumberArray(outputs[0]))[0]
    const trained = outputs.slice(1, 1 + params.length)
    return {
      loss,
      params: trained,
      state: optimizer.rebuildState(state, outputs.slice(1 + params.length))
    }
  })

// The training loop shared by both forms: initialize with `initial` (or
// `model.init` when omitted), then step until `stop` says otherwise (at
// least one step always runs), calling `onStep` after every step.
// Without a cache each step builds and evaluates the full step graph;
// with one each step is a single frozen-program call.
const trainLoop = <S, EL = never, RL = never, ED = never, RD = never, EO = never, RO = never>(
  model: Model.Model,
  config: TrainConfig<S, EL, RL, ED, RD, EO, RO>,
  initial: Model.Params | undefined,
  resume: Resume<S> | undefined,
  cache: Tensor.ProgramCache | undefined
): Effect.Effect<
  Trained<S>,
  Model.ModelError | Tensor.TensorError | Gradient.GradError | EL | ED | EO,
  CurrentDevice | RL | RD | RO
> =>
  Effect.gen(function* () {
    if (config.precision === "mixedBf16" && (yield* CurrentDevice) !== "metal") {
      return yield* new Model.ModelError({
        op: "train",
        message: "mixedBf16 precision requires the metal device (bf16 kernels)"
      })
    }
    let params: Model.Params = initial !== undefined
      ? initial
      : yield* model.init
    let state = resume !== undefined ? resume.state : yield* config.optimizer.init(params)
    if (cache !== undefined) {
      // Program inputs must be materialized buffers: the initial
      // parameters and state are lazy graph values, so evaluate them
      // once up front; every later step returns materialized values.
      const roots = [...params, ...config.optimizer.stateRoots(state)]
      const materialized = yield* Tensor.compute(roots)
      params = materialized.slice(0, params.length)
      state = config.optimizer.rebuildState(state, materialized.slice(params.length))
    }
    let step = resume?.step ?? 0
    let loss = Number.NaN
    let trained: ReadonlyArray<Tensor.Concrete>
    const started = yield* Effect.sync(() => Date.now())
    do {
      step++
      const data: TrainData = typeof config.data === "function"
        ? yield* config.data(step)
        : config.data
      const result = cache !== undefined
        ? yield* compiledStep(model, config, cache, params, state, data, step)
        : yield* uncompiledStep(model, config, params, state, data, step)
      loss = result.loss
      trained = result.params
      params = result.params
      state = result.state
      const info: TrainStep = {
        step,
        loss,
        elapsed: Duration.millis(yield* Effect.sync(() => Date.now() - started))
      }
      if (config.onStep !== undefined) {
        yield* config.onStep(info)
      }
      if (config.stop(info)) break
    } while (true)
    return { params: trained, state, loss, step }
  })

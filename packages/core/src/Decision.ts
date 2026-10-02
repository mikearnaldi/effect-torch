/**
 * Selected-answer scoring over prepared inputs and independent probability reads.
 *
 * @since 0.1.0
 */
import { Data, Effect, Exit } from "effect"
import type * as AutoRegressive from "./AutoRegressive.ts"
import type * as Diffusion from "./Diffusion.ts"
import type * as Model from "./Model.ts"
import type * as Runtime from "./Runtime.ts"
import * as Tensor from "./Tensor.ts"

/**
 * Invalid selection, read policy, or incomplete/non-finite scoring result.
 *
 * @since 0.1.0
 * @category errors
 */
export class DecisionError extends Data.TaggedError("DecisionError")<{ readonly message: string }> {}

/**
 * Ordered output positions and verified single-token answer codes. Duplicates
 * remain separate entries; neither list is sorted or deduplicated. Positions
 * refer to the evaluator's output rows, not flattened batch offsets.
 *
 * @since 0.1.0
 * @category models
 */
export interface Selection {
  readonly rows: ReadonlyArray<number>
  readonly labels: ReadonlyArray<number>
}

/**
 * Required independent batch evaluation. Each input is a complete prepared
 * read. Batch lanes must be isolated through every model layer. Implementations
 * may borrow immutable prefixes and parameters, but must not carry feedback or
 * sequence changes from one input to another.
 *
 * The result is a caller-owned floating tensor shaped
 * `[inputs.length, selection.rows.length, selection.labels.length]`. Later
 * calls cannot invalidate it. The caller clears it with `Tensor.clear`. Inputs
 * remain borrowed. Acquisition owns partial and late native results on failure
 * or interruption, and finishes native borrowing before the effect exits.
 *
 * @since 0.1.0
 * @category models
 */
export interface Scorer<Input, E = never, R = never> {
  readonly score: (
    inputs: ReadonlyArray<Input>,
    selection: Selection
  ) => Effect.Effect<Tensor.Concrete, E, R>
}

/**
 * Bind a required evaluator, including a diffusion canvas evaluator, without
 * changing its prepared-input, error, or service types. The evaluator implements
 * the full {@link Scorer} contract, including independent initial prediction
 * state per input. Iterative chains must be prepared explicitly by the caller.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromEvaluator = <Input, E, R>(score: Scorer<Input, E, R>["score"]): Scorer<Input, E, R> => ({ score })

/**
 * An independently initialized canvas borrowing an encoded immutable prefix.
 * The caller keeps the prefix alive through every concurrent scoring call.
 *
 * @since 0.1.0
 * @category models
 */
export interface DiffusionInput {
  readonly prefix: Diffusion.Prefix
  readonly canvas: Uint32Array
}

/**
 * Score supplied canvases with the artifact's selected initial-state evaluator.
 * Each input gets a fresh independent evaluation against its borrowed prefix.
 * The family currently executes single canvases, so a batch evaluates them in
 * order and joins their owned outputs. `scoreIndependently` bounds concurrent
 * batches. A single-input call returns the artifact's tensor directly.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromDiffusion = (
  program: Diffusion.Artifact
): Scorer<DiffusionInput, DecisionError | Diffusion.InferenceError | Tensor.TensorError, Runtime.Runtime> => ({
  score: (inputs, selection) =>
    Effect.gen(function*() {
      yield* validateSelection(selection)

      if (inputs.length === 0) return yield* new DecisionError({ message: "At least one input is required" })

      const evaluate = (input: DiffusionInput) =>
        program.score(input.prefix, input.canvas, selection.rows, selection.labels)

      if (inputs.length === 1) return yield* evaluate(inputs[0]!)

      let acquired: ReadonlyArray<Tensor.Concrete> = []

      return yield* Effect.scoped(Effect.gen(function*() {
        const outputs = yield* Effect.forEach(inputs, (input) =>
          Effect.acquireRelease(evaluate(input), Tensor.clear, { interruptible: true }))

        const combined = yield* Tensor.concat([outputs[0]!, outputs[1]!, ...outputs.slice(2)])

        const [result] = yield* Tensor.compute([combined]).pipe(Effect.onExit((exit) =>
          Effect.sync(() => {
            if (Exit.isSuccess(exit)) {
              acquired = exit.value
            }
          })
        ))

        return result
      })).pipe(Effect.onExit((exit) =>
        Exit.isFailure(exit) ? Tensor.clearAll(acquired) : Effect.void
      ))
    })
})

const validateSelection = (selection: Selection): Effect.Effect<void, DecisionError> =>
  Effect.gen(function*() {
    for (const [name, values] of [["rows", selection.rows], ["labels", selection.labels]] as const) {
      if (values.length === 0) return yield* new DecisionError({ message: `${name} must be nonempty` })

      for (const value of values) {
        if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
          return yield* new DecisionError({ message: `${name} must contain unsigned 32-bit integers` })
        }
      }
    }
  })

/**
 * Score the next position after each caller-supplied nonempty `[1, T]` prompt.
 * Row `0` is the next position; repeated zeros repeat that row. Uses the
 * artifact's caller-token execution session without sampling or generation.
 * Batch size must fit the artifact's configured lanes. Sessions and full logits
 * are released after selection; the selected tensor remains caller-owned.
 *
 * @since 0.1.0
 * @category constructors
 */
export const fromAutoRegressive = (
  program: AutoRegressive.Artifact
): Scorer<
  Tensor.Any,
  DecisionError | AutoRegressive.InferenceError | Model.ModelError | Tensor.TensorError,
  Runtime.Runtime
> => ({
  score: (inputs, selection) =>
    Effect.suspend(() => {
      let acquired: ReadonlyArray<Tensor.Concrete> = []

      return Effect.scoped(Effect.gen(function*() {
        yield* validateSelection(selection)

        if (inputs.length === 0) return yield* new DecisionError({ message: "At least one input is required" })

        if (selection.rows.some((row) => row !== 0)) {
          return yield* new DecisionError({ message: "Autoregressive scoring exposes only next-position row 0" })
        }

        const session = yield* Effect.acquireRelease(
          program.execution(),
          (session) => Effect.orDie(session.close()),
          { interruptible: true }
        )

        const outputs = yield* Effect.acquireRelease(
          session.add(inputs),
          (outputs) => Tensor.clearAll(outputs.map(({ logits }) => logits)),
          { interruptible: true }
        )

        if (outputs.length !== inputs.length) {
          return yield* new DecisionError({ message: "Execution returned the wrong input count" })
        }

        const indexes = yield* Tensor.fromTypedArray(new Uint32Array(selection.labels), [selection.labels.length])

        const selected = yield* Effect.forEach(outputs, ({ logits }) =>
          Effect.gen(function*() {
            if (logits.shape.length !== 1 || selection.labels.some((label) => label >= logits.shape[0]!)) {
              return yield* new DecisionError({ message: "Answer label is outside the next-position vocabulary" })
            }

            const labels = yield* Tensor.take(logits, indexes)

            return yield* Tensor.broadcastTo(
              yield* Tensor.reshape(labels, [1, selection.labels.length]),
              [selection.rows.length, selection.labels.length]
            )
          }))

        const batch = selected.length === 1
          ? yield* Tensor.reshape(selected[0]!, [1, selection.rows.length, selection.labels.length])
          : yield* Tensor.stack([selected[0]!, selected[1]!, ...selected.slice(2)])

        const [result] = yield* Tensor.compute([batch]).pipe(Effect.onExit((exit) =>
          Effect.sync(() => {
            if (Exit.isSuccess(exit)) acquired = exit.value
          })
        ))

        return result
      })).pipe(Effect.onExit((exit) => Exit.isFailure(exit) ? Tensor.clearAll(acquired) : Effect.void))
    })
})

// Compensated summation retains small contributions in complete distributions.
const sum = (values: ReadonlyArray<number>): number => {
  let total = 0
  let correction = 0

  for (const value of values) {
    const next = total + value
    correction += Math.abs(total) >= Math.abs(value) ? (total - next) + value : (value - next) + total
    total = next
  }

  return total + correction
}

const validateVector = (values: ReadonlyArray<number>, count: number): Effect.Effect<void, DecisionError> =>
  Effect.gen(function*() {
    if (!Number.isSafeInteger(count) || count < 1 || !Array.isArray(values) || values.length !== count) {
      return yield* new DecisionError({ message: "Values must match a positive answer count" })
    }

    for (const value of values) {
      if (!Number.isFinite(value)) return yield* new DecisionError({ message: "Values must be finite and complete" })
    }
  })

/**
 * Absolute tolerance for validating sums of host probability vectors.
 *
 * @since 0.1.0
 * @category readouts
 */
export const probabilityTolerance = 1e-12

const probabilities = (values: ReadonlyArray<number>, count: number) =>
  Effect.gen(function*() {
    yield* validateVector(values, count)

    if (values.some((value) => value < 0 || value > 1)) {
      return yield* new DecisionError({ message: "Probabilities must be in [0, 1]" })
    }

    const total = sum(values)

    if (total <= 0 || Math.abs(total - 1) > probabilityTolerance) {
      return yield* new DecisionError({ message: "Probabilities must sum to one" })
    }

    return values.map((value) => value / total)
  })

/**
 * Normalize all selected logits using a maximum-shifted softmax. Zero logits
 * are valid. Missing and non-finite logits fail instead of producing a fallback.
 * Duplicate answer codes occupy separate entries in the distribution.
 *
 * @since 0.1.0
 * @category readouts
 */
export const restrictedSoftmax = (
  logits: ReadonlyArray<number>,
  answerCount: number = logits.length
): Effect.Effect<ReadonlyArray<number>, DecisionError> =>
  Effect.gen(function*() {
    yield* validateVector(logits, answerCount)
    const maximum = logits.reduce((maximum, logit) => Math.max(maximum, logit), -Infinity)
    const weights = logits.map((logit) => Math.exp(logit - maximum))
    const total = sum(weights)

    return weights.map((weight) => weight / total)
  })

/**
 * Average complete probability vectors from independent reads.
 *
 * @since 0.1.0
 * @category readouts
 */
export const meanProbabilities = (
  reads: ReadonlyArray<ReadonlyArray<number>>,
  answerCount: number
): Effect.Effect<ReadonlyArray<number>, DecisionError> =>
  Effect.gen(function*() {
    if (!Array.isArray(reads) || reads.length === 0) {
      return yield* new DecisionError({ message: "At least one probability read is required" })
    }

    const checked = yield* Effect.forEach(reads, (read) => probabilities(read, answerCount))

    return Array.from({ length: answerCount }, (_, index) => sum(checked.map((read) => read[index]!)) / checked.length)
  })

/**
 * Softmax each independent read, then average probabilities. This policy never
 * averages logits or carries a prediction state between reads.
 *
 * @since 0.1.0
 * @category readouts
 */
export const independentProbabilities = (
  reads: ReadonlyArray<ReadonlyArray<number>>,
  answerCount: number
): Effect.Effect<ReadonlyArray<number>, DecisionError> =>
  Effect.gen(function*() {
    if (!Array.isArray(reads) || reads.length === 0) {
      return yield* new DecisionError({ message: "At least one logit read is required" })
    }

    return yield* meanProbabilities(
      yield* Effect.forEach(reads, (read) => restrictedSoftmax(read, answerCount)),
      answerCount
    )
  })

/**
 * Probability of the positive outcome in a binary distribution. The default
 * order is false, true; `positiveIndex` explicitly supports the reverse order.
 *
 * @since 0.1.0
 * @category readouts
 */
export const binaryProbability = (
  values: ReadonlyArray<number>,
  positiveIndex: 0 | 1 = 1
): Effect.Effect<number, DecisionError> =>
  Effect.gen(function*() {
    const p = yield* probabilities(values, 2)

    return p[positiveIndex]!
  })

/**
 * Categorical result retaining the caller's ordered outcomes and distribution.
 *
 * @since 0.1.0
 * @category readouts
 */
export interface Category<A> {
  readonly index: number
  readonly value: A
  readonly probabilities: ReadonlyArray<number>
}

/**
 * Choose the first maximum-probability outcome, preserving ties and duplicates.
 *
 * @since 0.1.0
 * @category readouts
 */
export const categorical = <A>(
  values: ReadonlyArray<number>,
  outcomes: ReadonlyArray<A>
): Effect.Effect<Category<A>, DecisionError> =>
  Effect.gen(function*() {
    const p = yield* probabilities(values, outcomes.length)
    let index = 0

    for (let candidate = 1; candidate < p.length; candidate++) {
      if (p[candidate]! > p[index]!) index = candidate
    }

    return { index, value: outcomes[index]!, probabilities: p }
  })

/**
 * Expected value of explicitly ordered finite outcomes.
 *
 * @since 0.1.0
 * @category readouts
 */
export const expectedValue = (
  values: ReadonlyArray<number>,
  outcomes: ReadonlyArray<number>
): Effect.Effect<number, DecisionError> =>
  Effect.gen(function*() {
    const p = yield* probabilities(values, outcomes.length)
    yield* validateVector(outcomes, p.length)
    const result = sum(p.map((probability, index) => probability * outcomes[index]!))

    if (!Number.isFinite(result)) return yield* new DecisionError({ message: "Expected value is not finite" })

    return result
  })

/**
 * One decision with caller-prepared independent reads sharing an answer order.
 * Input preparation, noise seeds, and question routing remain application-owned.
 *
 * @since 0.1.0
 * @category models
 */
export interface IndependentReads<Input> {
  readonly inputs: ReadonlyArray<Input>
  readonly selection: Selection
}

/**
 * Bounds batches and simultaneous evaluator calls for one scoring request.
 * `batchSize` must fit the chosen artifact's capacity.
 *
 * @since 0.1.0
 * @category models
 */
export interface ScoringOptions {
  readonly batchSize: number
  readonly concurrency: number
}

/**
 * Score independent decisions, returning `[decision][row][label]` probabilities
 * in caller order. Every read is normalized before averaging. Batching never
 * concatenates question contexts or merges duplicate inputs. Only compatible
 * reads within one decision enter a batch; concurrency is bounded globally
 * across the request. All temporary logits are cleared on success, failure,
 * and interruption, after readback has finished borrowing them.
 *
 * @since 0.1.0
 * @category execution
 */
export const scoreIndependently = <Input, E, R>(
  scorer: Scorer<Input, E, R>,
  decisions: ReadonlyArray<IndependentReads<Input>>,
  options: ScoringOptions
): Effect.Effect<
  ReadonlyArray<ReadonlyArray<ReadonlyArray<number>>>,
  E | DecisionError | Tensor.TensorError,
  R | Runtime.Runtime
> =>
  Effect.gen(function*() {
    for (const value of [options.batchSize, options.concurrency]) {
      if (!Number.isSafeInteger(value) || value < 1) {
        return yield* new DecisionError({ message: "Batch size and concurrency must be positive safe integers" })
      }
    }

    const batches: Array<
      {
        readonly decision: number
        readonly inputs: ReadonlyArray<Input>
        readonly selection: Selection
      }
    > = []

    for (const [decision, read] of decisions.entries()) {
      yield* validateSelection(read.selection)

      if (read.inputs.length === 0) {
        return yield* new DecisionError({ message: "At least one independent read is required" })
      }

      for (let offset = 0; offset < read.inputs.length; offset += options.batchSize) {
        batches.push({
          decision,
          inputs: read.inputs.slice(offset, offset + options.batchSize),
          selection: read.selection
        })
      }
    }

    const results = yield* Effect.forEach(batches, ({ inputs, selection }) =>
      Effect.scoped(Effect.gen(function*() {
        const logits = yield* Effect.acquireRelease(
          Effect.suspend(() => scorer.score(inputs, selection)),
          Tensor.clear,
          { interruptible: true }
        )

        const shape = [inputs.length, selection.rows.length, selection.labels.length]

        if (logits.shape.length !== 3 || logits.shape.some((size, index) => size !== shape[index])) {
          return yield* new DecisionError({ message: `Scorer returned shape [${logits.shape}], expected [${shape}]` })
        }

        if (!["f16", "bf16", "f32", "f64"].includes(logits.dtype)) {
          return yield* new DecisionError({ message: "Scorer must return floating logits" })
        }

        const values = yield* Tensor.toNumberArray(logits)
        const count = selection.labels.length

        return yield* Effect.forEach(
          Array.from({ length: inputs.length * selection.rows.length }, (_, row) => row),
          (row) => restrictedSoftmax(values.slice(row * count, (row + 1) * count), count)
        )
      })), { concurrency: options.concurrency })

    const reads = decisions.map(({ selection }) => selection.rows.map(() => Array<ReadonlyArray<number>>()))

    for (const [index, batch] of batches.entries()) {
      for (const [row, probabilities] of results[index]!.entries()) {
        reads[batch.decision]![row % batch.selection.rows.length]!.push(probabilities)
      }
    }

    return yield* Effect.forEach(reads, (rows, decision) =>
      Effect.forEach(rows, (row) =>
        meanProbabilities(row, decisions[decision]!.selection.labels.length)))
  })

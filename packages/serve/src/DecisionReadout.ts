import { Decision } from "@effect-torch/core"
import { Effect, Schema } from "effect"
import * as Contract from "./Decision.ts"
import { optionsFor, type RequestPlan } from "./DecisionPlanner.ts"

/**
 * Invalid or incomplete model output. Never replaced by a floor or uniform fallback.
 *
 * @since 0.1.0
 * @category models
 */
export class ReadoutError extends Schema.TaggedErrorClass<ReadoutError>()(
  "ReadoutError",
  { message: Schema.String }
) {}

/**
 * Absolute sum tolerance for floating-point distributions, not service rounding.
 *
 * @since 0.1.0
 * @category models
 */
export const probabilityTolerance = Decision.probabilityTolerance

// Compensated summation avoids artificial nonzero confidence at uniformity.
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

const validateVector = (values: ReadonlyArray<number>, count: number) =>
  Effect.gen(function*() {
    if (!Number.isSafeInteger(count) || count < 1) {
      return yield* new ReadoutError({ message: "Option count must be a positive safe integer" })
    }

    if (!Array.isArray(values) || values.length !== count) {
      return yield* new ReadoutError({ message: "Missing values or wrong option count" })
    }

    for (let index = 0; index < count; index++) {
      if (!Number.isFinite(values[index])) {
        return yield* new ReadoutError({ message: "Missing or non-finite value at option " + index })
      }
    }
  })

const probabilities = (values: ReadonlyArray<number>, count: number) =>
  Effect.gen(function*() {
    yield* validateVector(values, count)

    if (values.some((value) => value < 0 || value > 1)) {
      return yield* new ReadoutError({ message: "Probabilities must be in [0, 1]" })
    }

    const total = sum(values)

    if (total <= 0 || Math.abs(total - 1) > probabilityTolerance) {
      return yield* new ReadoutError({ message: "Probabilities must sum to one" })
    }

    return values.map((value) => value / total)
  })

/**
 * Stable softmax over every allowed option, with no top-k truncation. All-zero logits are valid.
 *
 * @since 0.1.0
 * @category models
 */
export const restrictedSoftmax = (logits: ReadonlyArray<number>, optionCount: number): Effect.Effect<
  ReadonlyArray<number>,
  ReadoutError
> =>
  Effect.gen(function*() {
    yield* validateVector(logits, optionCount)

    return yield* Decision.restrictedSoftmax(logits, optionCount).pipe(
      Effect.mapError((error) => new ReadoutError({ message: error.message }))
    )
  })

/**
 * Average complete probability vectors. Empty reads and malformed distributions fail.
 *
 * @since 0.1.0
 * @category models
 */
export const meanProbabilities = (
  reads: ReadonlyArray<ReadonlyArray<number>>,
  optionCount: number
): Effect.Effect<ReadonlyArray<number>, ReadoutError> =>
  Effect.gen(function*() {
    if (!Array.isArray(reads) || reads.length === 0) {
      return yield* new ReadoutError({ message: "At least one probability read is required" })
    }

    yield* Effect.forEach(reads, (read) => validateVector(read, optionCount), { discard: true })

    return yield* Decision.meanProbabilities(reads, optionCount).pipe(
      Effect.mapError((error) => new ReadoutError({ message: error.message }))
    )
  })

// First mode wins ties, matching Python max(range(...), key=...).
const mode = (values: ReadonlyArray<number>): number => {
  let best = 0

  for (let index = 1; index < values.length; index++) {
    if (values[index] > values[best]) best = index
  }

  return best
}

// Official adapter fb52b1030b7fc1f4f1cf39910afa5da54f9835e3,
// src/system_one_adapter/_utils/confidence_metrics.py. Preserve the formulas;
// reject invalid distributions instead of its zero-sum uniform fallback.
const choiceConcentration = (p: ReadonlyArray<number>): number => {
  if (p.length === 1) return 1

  const uniform = 1 / p.length

  return (p[mode(p)] - uniform) / (1 - uniform)
}

const scoreConcentration = (p: ReadonlyArray<number>): number => {
  if (p.length === 1) return 1

  const modalIndex = mode(p)
  const distance = sum(p.map((value, index) => value * Math.abs(index - modalIndex)))
  const center = (p.length - 1) / 2
  const uniformDistance = sum(p.map((_, index) => Math.abs(index - center))) / p.length

  return Math.max(0, 1 - distance / uniformDistance)
}

/**
 * Official adapter Choice confidence, without display rounding.
 *
 * @since 0.1.0
 * @category models
 */
export const choiceConfidence = (p: ReadonlyArray<number>): Effect.Effect<number, ReadoutError> =>
  probabilities(p, p?.length).pipe(Effect.map(choiceConcentration))

/**
 * Official adapter first-mode Score confidence, without display rounding.
 *
 * @since 0.1.0
 * @category models
 */
export const scoreConfidence = (p: ReadonlyArray<number>): Effect.Effect<number, ReadoutError> =>
  probabilities(p, p?.length).pipe(Effect.map(scoreConcentration))

/**
 * Decode a complete normalized distribution in optionsFor order. Ties choose
 * the first option. Score returns the expected zero-based level and the original
 * rich legend. Noul returns P(true) without a separate confidence field.
 *
 * @since 0.1.0
 * @category models
 */
export const answerFromProbabilities = (
  question: Contract.Question,
  values: ReadonlyArray<number>
): Effect.Effect<Contract.Answer, ReadoutError> =>
  Effect.gen(function*() {
    const options = optionsFor(question)
    const p = yield* probabilities(values, options.length)

    if (question.type === "noul") return { type: "noul", noul: p[1] }

    const distribution = Object.fromEntries(options.map((option, index) => [option.name, p[index]]))

    if (question.type === "choice") {
      return {
        type: "choice",
        choice: options[mode(p)].name,
        probabilities: distribution,
        confidence: choiceConcentration(p)
      }
    }

    return {
      type: "score",
      score: sum(p.map((value, index) => index * value)),
      legend: Object.fromEntries(options.map((option) => [option.name, structuredClone(option.description)])),
      probabilities: distribution,
      confidence: scoreConcentration(p)
    }
  })

/**
 * Softmax each independent read, then average probabilities rather than logits.
 *
 * @since 0.1.0
 * @category models
 */
export const answerFromLogits = (
  question: Contract.Question,
  reads: ReadonlyArray<ReadonlyArray<number>>
): Effect.Effect<Contract.Answer, ReadoutError> =>
  Effect.gen(function*() {
    if (!Array.isArray(reads) || reads.length === 0) {
      return yield* new ReadoutError({ message: "At least one logit read is required" })
    }

    const count = optionsFor(question).length
    const distributions = yield* Effect.forEach(reads, (read) => restrictedSoftmax(read, count))
    const average = yield* meanProbabilities(distributions, count)

    return yield* answerFromProbabilities(question, average)
  })

/**
 * Route completed semantic-keyed results back to caller IDs, independent of
 * scheduler completion order. Usage is supplied by the caller's real accounting.
 *
 * @since 0.1.0
 * @category models
 */
export const responseFromLogits = (
  plan: RequestPlan,
  readsByKey: ReadonlyMap<string, ReadonlyArray<ReadonlyArray<number>>>,
  usage: Contract.Usage
): Effect.Effect<Contract.Response, ReadoutError> =>
  Effect.gen(function*() {
    const counts = yield* Schema.decodeUnknownEffect(Contract.Usage, { onExcessProperty: "error" })(usage).pipe(
      Effect.mapError((error) => new ReadoutError({ message: error.message }))
    )

    const answers = yield* Effect.forEach(plan.routes, (route) =>
      Effect.gen(function*() {
        const reads = readsByKey.get(route.decision.semanticKey)

        if (reads === undefined) return yield* new ReadoutError({ message: "Missing question result" })

        const answer = yield* answerFromLogits(route.decision.question, reads)

        return [route.questionId, answer] as const
      }))

    return { model: plan.model, answers: Object.fromEntries(answers), usage: counts }
  })

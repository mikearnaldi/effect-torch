import { Effect, Predicate, Schema } from "effect"
import { createHash } from "node:crypto"
import * as Contract from "./Decision.ts"

const isJsonArray = (value: Schema.Json): value is Schema.JsonArray => Array.isArray(value)

/**
 * Stable serialization of validated JSON; object keys sort, arrays keep order.
 *
 * @since 0.1.0
 * @category models
 */
export const canonicalJson = (value: Schema.Json): string => {
  if (value === null || Predicate.isString(value) || Predicate.isNumber(value) || Predicate.isBoolean(value)) {
    return JSON.stringify(value)
  }

  if (isJsonArray(value)) return "[" + value.map(canonicalJson).join(",") + "]"

  return "{" + Object.keys(value).sort().map((key) => JSON.stringify(key) + ":" + canonicalJson(value[key])).join(",") +
    "}"
}

/**
 * An option name is semantic data, unlike a caller's question ID.
 *
 * @since 0.1.0
 * @category models
 */
export interface Option {
  readonly name: string
  readonly description: Contract.Description
}

/**
 * Fixed false/true Noul order; original Choice and Score order otherwise.
 *
 * @since 0.1.0
 * @category models
 */
export const optionsFor = (question: Contract.Question): ReadonlyArray<Option> => {
  switch (question.type) {
    case "noul":
      return [
        { name: "false", description: question.criteria.false },
        { name: "true", description: question.criteria.true }
      ]
    case "choice":
      // JS integer-like object keys enumerate numerically. This is the order
      // available after JSON parsing; textual wire order cannot be recovered.
      return Object.entries(question.criteria).map(([name, description]) => ({ name, description }))
    case "score":
      return question.criteria.map((description, index) => ({ name: String(index), description }))
  }
}

/**
 * Isolated text plan only. Tokenizer/scaffold validation belongs to model integration.
 *
 * @since 0.1.0
 * @category models
 */
export interface QuestionPlan {
  readonly model: string
  readonly state: Contract.Content
  readonly semanticKey: string
  readonly prompt: string
  readonly question: Contract.Question
  readonly options: ReadonlyArray<Option>
}

/**
 * Caller IDs exist only in host-side output routes.
 *
 * @since 0.1.0
 * @category models
 */
export interface RequestPlan {
  readonly model: string
  readonly routes: ReadonlyArray<{
    readonly questionId: string
    readonly decision: QuestionPlan
  }>
}

// This local single-question text format is versioned separately from the
// upstream tokenizer scaffold. It makes no claim of model/template parity.
const template = "decision-model/isolated-question/v1"

const freeze = <A>(value: A): A => {
  if (value !== null && (Predicate.isObject(value) || Array.isArray(value))) {
    for (const child of Object.values(value)) freeze(child)

    Object.freeze(value)
  }

  return value
}

/**
 * Validate and snapshot a request, then render each question independently.
 * IDs, sibling questions, request order, and scheduler state never enter the
 * prompt or semantic hash. Semantically duplicate questions share a key.
 * Config must name the actual local model; the request must match it exactly.
 *
 * @since 0.1.0
 * @category models
 */
export const plan = <Input>(input: Input, config: Contract.ModelConfig): Effect.Effect<
  RequestPlan,
  Contract.ValidationError
> =>
  Effect.gen(function*() {
    const local = yield* Schema.decodeUnknownEffect(Contract.ModelConfig, { onExcessProperty: "error" })(config).pipe(
      Effect.mapError((error) => new Contract.ValidationError({ message: error.message }))
    )

    const decoded = yield* Contract.decodeRequest(input)

    if (decoded.model !== local.modelId) {
      return yield* new Contract.ValidationError({ message: "Unsupported model: " + decoded.model })
    }

    const request = structuredClone(decoded)

    const routes = Object.entries(request.questions).map(([questionId, question]) => {
      const options = optionsFor(question)

      // Convert Choice maps to ordered arrays BEFORE sorting other JSON keys.
      const body = canonicalJson({
        state: request.state,
        question: {
          type: question.type,
          instructions: question.instructions,
          options: options.map(
            (option) => ({ name: option.name, description: option.description })
          )
        }
      })

      const prompt = template + "\nEvaluate this question using the supplied state.\n" + body
      const semanticKey = createHash("sha256").update(canonicalJson({ model: local.modelId, prompt })).digest("hex")

      return {
        questionId,
        decision: { model: local.modelId, state: request.state, semanticKey, prompt, question, options }
      }
    })

    return freeze({ model: local.modelId, routes })
  })

import { Effect, Predicate, Schema } from "effect"

/**
 * Request validation failure, including unsupported local model IDs.
 *
 * @since 0.1.0
 * @category models
 */
export class ValidationError extends Schema.TaggedErrorClass<ValidationError>()(
  "ValidationError",
  { message: Schema.String }
) {}

// Schema.Json validates finite values and rejects cycles, but its array check
// skips holes. Host callers must supply dense JSON arrays, just like JSON.parse.
const hasDenseArrays = (value: Schema.Json): boolean => {
  if (value === null || Predicate.isString(value) || Predicate.isNumber(value) || Predicate.isBoolean(value)) {
    return true
  }

  if (Array.isArray(value)) {
    return Array.from(value).every((item) => item !== undefined && hasDenseArrays(item))
  }

  return Object.values(value).every(hasDenseArrays)
}

const Json = Schema.Json.check(Schema.makeFilter(hasDenseArrays, { expected: "JSON without sparse arrays" }))

/**
 * JSON text, object, or array. Scalar numbers, booleans, and root null are excluded.
 *
 * @since 0.1.0
 * @category models
 */
export const Content = Schema.Union([
  Schema.String,
  Schema.Record(Schema.String, Json),
  Schema.Array(Json)
])

/**
 * JSON text, object or array accepted as decision state or descriptive content.
 *
 * @since 0.1.0
 * @category models
 */
export type Content = typeof Content.Type

/**
 * Descriptions retain nested JSON and allow null, including Score legends.
 *
 * @since 0.1.0
 * @category models
 */
export const Description = Schema.NullOr(Content)

/**
 * Optional descriptive content for instructions, criteria and score legends.
 *
 * @since 0.1.0
 * @category models
 */
export type Description = typeof Description.Type

// SDK compatibility choice: omitted and null instructions both mean no extra
// instructions. Explicit undefined is not JSON and is rejected. HTTP docs are
// narrower; this normalization does not assert private endpoint equivalence.
const Instructions = Description.pipe(Schema.withDecodingDefaultKey(Effect.succeed(null)))

const OptionalDescription = Description.pipe(Schema.withDecodingDefaultKey(Effect.succeed(null)))

const NoulCriteria = Schema.Struct({ true: OptionalDescription, false: OptionalDescription })

/**
 * Noul distributions use the fixed order false, true.
 *
 * @since 0.1.0
 * @category models
 */
export const NoulQuestion = Schema.Struct({
  type: Schema.Literal("noul"),
  instructions: Instructions,
  criteria: NoulCriteria.pipe(Schema.withDecodingDefaultKey(Effect.succeed({})))
})

/**
 * Choice option names and their enumeration order are model-visible.
 *
 * @since 0.1.0
 * @category models
 */
export const ChoiceQuestion = Schema.Struct({
  type: Schema.Literal("choice"),
  instructions: Instructions,
  criteria: Schema.Record(Schema.String, Description).check(
    Schema.isMinProperties(1),
    Schema.isMaxProperties(255)
  )
})

/**
 * Score levels retain their original order and rich descriptions.
 *
 * @since 0.1.0
 * @category models
 */
export const ScoreQuestion = Schema.Struct({
  type: Schema.Literal("score"),
  instructions: Instructions,
  criteria: Schema.Array(Description).check(Schema.isMinLength(2), Schema.isMaxLength(10))
})

/**
 * Supported independent question types.
 *
 * @since 0.1.0
 * @category models
 */
export const Question = Schema.Union([NoulQuestion, ChoiceQuestion, ScoreQuestion])

/**
 * A decoded binary, categorical or ordered-score question.
 *
 * @since 0.1.0
 * @category models
 */
export type Question = typeof Question.Type

const ModelId = Schema.String.check(Schema.isMinLength(1))

/**
 * Required, application-owned model identity; no implicit Jev alias.
 *
 * @since 0.1.0
 * @category models
 */
export const ModelConfig = Schema.Struct({ modelId: ModelId })

/**
 * The application-owned identity of the model used for decision inference.
 *
 * @since 0.1.0
 * @category models
 */
export type ModelConfig = typeof ModelConfig.Type

/**
 * Local Jev-shaped request. Empty question maps are rejected. Omitted/null
 * instructions normalize to null; omitted Noul descriptions normalize to null.
 * Choice has 1–255 options, Score has 2–10 levels. No service quotas are inferred.
 * Use decodeRequest to reject excess properties on contract structs. Arbitrary
 * JSON keys within state, descriptions, and question/option maps remain valid.
 *
 * @since 0.1.0
 * @category models
 */
export const Request = Schema.Struct({
  model: ModelId,
  state: Content,
  questions: Schema.Record(Schema.String, Question).check(Schema.isMinProperties(1))
})

/**
 * A validated decision request with normalized instructions and criteria.
 *
 * @since 0.1.0
 * @category models
 */
export type Request = typeof Request.Type

const Probability = Schema.Number.check(Schema.isFinite(), Schema.isBetween({ minimum: 0, maximum: 1 }))

const TokenCount = Schema.Number.check(Schema.isInt(), Schema.isGreaterThanOrEqualTo(0))

/**
 * Counts supplied by the tokenizer and runtime.
 *
 * @since 0.1.0
 * @category models
 */
export const Usage = Schema.Struct({ input_tokens: TokenCount, output_tokens: TokenCount })

/**
 * Logical input-token and evaluated-canvas-token counts for a decision response.
 *
 * @since 0.1.0
 * @category models
 */
export type Usage = typeof Usage.Type

/**
 * Decoded answer and complete selected-label probabilities.
 *
 * @since 0.1.0
 * @category models
 */
export const Answer = Schema.Union([
  Schema.Struct({ type: Schema.Literal("noul"), noul: Probability }),
  Schema.Struct({
    type: Schema.Literal("choice"),
    choice: Schema.String,
    probabilities: Schema.Record(Schema.String, Probability),
    confidence: Probability
  }),
  Schema.Struct({
    type: Schema.Literal("score"),
    score: Schema.Number.check(Schema.isFinite(), Schema.isBetween({ minimum: 0, maximum: 9 })),
    legend: Schema.Record(Schema.String, Description),
    probabilities: Schema.Record(Schema.String, Probability),
    confidence: Probability
  })
])

/**
 * A decoded decision answer with its task-specific distribution and confidence.
 *
 * @since 0.1.0
 * @category models
 */
export type Answer = typeof Answer.Type

/**
 * Answers keyed by caller question ID with logical-work token counts.
 *
 * @since 0.1.0
 * @category models
 */
export const Response = Schema.Struct({
  model: ModelId,
  answers: Schema.Record(Schema.String, Answer),
  usage: Usage
})

/**
 * Decision answers keyed by caller question ID, with model identity and usage.
 *
 * @since 0.1.0
 * @category models
 */
export type Response = typeof Response.Type

/**
 * Decode wire input with strict excess-property errors and typed failures.
 *
 * @since 0.1.0
 * @category models
 */
export const decodeRequest = <Input>(input: Input): Effect.Effect<Request, ValidationError> =>
  Schema.decodeUnknownEffect(Json)(input).pipe(
    Effect.flatMap(Schema.decodeUnknownEffect(Request, { onExcessProperty: "error" })),
    Effect.mapError((error) => new ValidationError({ message: error.message }))
  )

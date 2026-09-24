import { Schema } from "effect"

/**
 * OpenAI error envelope, also used by decision endpoints.
 *
 * @since 0.1.0
 * @category schemas
 */
export const ErrorResponse = Schema.Struct({
  error: Schema.Struct({
    message: Schema.String,
    type: Schema.String,
    param: Schema.Null,
    code: Schema.String
  })
})

const PositiveInt = Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 65536 }))

const Common = {
  model: Schema.String.check(Schema.isMinLength(1)),
  max_tokens: Schema.optionalKey(PositiveInt),
  temperature: Schema.optionalKey(Schema.Number.check(Schema.isBetween({ minimum: 0, maximum: 2 }))),
  top_p: Schema.optionalKey(Schema.Number.check(Schema.isGreaterThan(0), Schema.isLessThanOrEqualTo(1))),
  seed: Schema.optionalKey(Schema.Int.check(Schema.isBetween({ minimum: 0, maximum: 0xffff_ffff }))),
  n: Schema.optionalKey(Schema.Literal(1)),
  stream: Schema.optionalKey(Schema.Boolean),
  stream_options: Schema.optionalKey(Schema.Struct({ include_usage: Schema.optionalKey(Schema.Boolean) })),
  user: Schema.optionalKey(Schema.String)
}

/**
 * Text-only OpenAI chat request. Unsupported fields fail validation.
 *
 * @since 0.1.0
 * @category schemas
 */
export const ChatRequest = Schema.Struct({
  ...Common,
  messages: Schema.Array(Schema.Struct({
    role: Schema.Literals(["system", "user", "assistant"]),
    content: Schema.String
  })).check(Schema.isMinLength(1)),
  max_completion_tokens: Schema.optionalKey(PositiveInt)
})

/**
 * Single-prompt OpenAI completion request.
 *
 * @since 0.1.0
 * @category schemas
 */
export const CompletionRequest = Schema.Struct({ ...Common, prompt: Schema.String.check(Schema.isMinLength(1)) })

/**
 * Token counts include model control tokens and exclude padding after EOS.
 *
 * @since 0.1.0
 * @category schemas
 */
export const Usage = Schema.Struct({
  prompt_tokens: Schema.Int,
  completion_tokens: Schema.Int,
  total_tokens: Schema.Int
})

/**
 * OpenAI non-streaming text response.
 *
 * @since 0.1.0
 * @category schemas
 */
export const Completion = Schema.Struct({
  id: Schema.String,
  object: Schema.Literal("text_completion"),
  created: Schema.Int,
  model: Schema.String,
  choices: Schema.Array(
    Schema.Struct({
      index: Schema.Int,
      text: Schema.String,
      logprobs: Schema.Null,
      finish_reason: Schema.Literals(["stop", "length"])
    })
  ),
  usage: Usage
})

/**
 * OpenAI non-streaming chat response.
 *
 * @since 0.1.0
 * @category schemas
 */
export const ChatCompletion = Schema.Struct({
  id: Schema.String,
  object: Schema.Literal("chat.completion"),
  created: Schema.Int,
  model: Schema.String,
  choices: Schema.Array(
    Schema.Struct({
      index: Schema.Int,
      message: Schema.Struct({ role: Schema.Literal("assistant"), content: Schema.String }),
      finish_reason: Schema.Literals(["stop", "length"])
    })
  ),
  usage: Usage
})

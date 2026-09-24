import { Schema } from "effect"
import { HttpApi, HttpApiEndpoint, HttpApiGroup, HttpApiSchema } from "effect/unstable/httpapi"
import * as Decision from "./Decision.ts"
import * as OpenAi from "./OpenAi.ts"

const errors = [400, 404, 429, 500].map((status) => OpenAi.ErrorResponse.pipe(HttpApiSchema.status(status)))

/**
 * Independent decision reads and semantic noise seed are request-local.
 *
 * @since 0.1.0
 * @category schemas
 */
export const DecisionRequest = Schema.Struct({
  ...Decision.Request.fields,
  reads: Schema.optionalKey(Schema.Literals([1, 4])),
  seed: Schema.optionalKey(Schema.String)
})

/**
 * Effect HttpApi contract for the model server.
 *
 * @since 0.1.0
 * @category api
 */
export const Api = HttpApi.make("effect-torch").add(
  HttpApiGroup.make("models").add(
    HttpApiEndpoint.get("list", "/v1/models", {
      success: Schema.Struct({
        object: Schema.Literal("list"),
        data: Schema.Array(
          Schema.Struct({
            id: Schema.String,
            object: Schema.Literal("model"),
            created: Schema.Int,
            owned_by: Schema.String
          })
        )
      })
    }),
    HttpApiEndpoint.post("complete", "/v1/completions", {
      payload: OpenAi.CompletionRequest,
      error: errors,
      success: [OpenAi.Completion, HttpApiSchema.StreamUint8Array({ contentType: "text/event-stream" })]
    }),
    HttpApiEndpoint.post("chat", "/v1/chat/completions", {
      payload: OpenAi.ChatRequest,
      error: errors,
      success: [OpenAi.ChatCompletion, HttpApiSchema.StreamUint8Array({ contentType: "text/event-stream" })]
    }),
    HttpApiEndpoint.post("decide", "/v1/decisions", {
      payload: DecisionRequest,
      success: Decision.Response,
      error: errors
    })
  )
)

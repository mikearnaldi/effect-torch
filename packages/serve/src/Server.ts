import { Effect, Fiber, type FileSystem, Layer, type Path, Queue, Schema, Semaphore, Stream } from "effect"
import { type Etag, type HttpPlatform, HttpRouter, HttpServer, HttpServerResponse } from "effect/unstable/http"
import { HttpApiBuilder } from "effect/unstable/httpapi"
import { randomUUID } from "node:crypto"
import { Api, DecisionRequest } from "./Api.ts"
import type * as Model from "./Model.ts"
import * as OpenAi from "./OpenAi.ts"

class RequestError extends Schema.TaggedErrorClass<RequestError>()("ServeRequestError", {
  message: Schema.String,
  status: Schema.Number,
  code: Schema.String
}) {}

const failure = (message: string, status = 400, code = "invalid_request_error") =>
  new RequestError({ message, status, code })

const errorBody = (message: string, code: string) => ({ error: { message, type: code, param: null, code } })

const encoder = new TextEncoder()

const sse = (value: Schema.Json) => encoder.encode("data: " + JSON.stringify(value) + String.fromCharCode(10, 10))

/**
 * Admission bounds apply to both task types and streamed response lifetimes.
 *
 * @since 0.1.0
 * @category models
 */
export interface Options {
  readonly models: ReadonlyArray<Model.Registration>
  readonly maxConcurrentRequests?: number
  readonly maxPendingRequests?: number
  readonly defaultMaxTokens?: number
}

/**
 * Compose these routes with any Effect HttpServer layer.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer = (options: Options): Layer.Layer<
  never,
  RequestError,
  FileSystem.FileSystem | Path.Path | Etag.Generator | HttpPlatform.HttpPlatform | HttpRouter.HttpRouter
> => {
  const handlers = HttpApiBuilder.group(Api, "models", (handlers) =>
    Effect.gen(function*() {
      const maximum = options.maxConcurrentRequests ?? 1
      const capacity = options.maxPendingRequests ?? 16
      const defaultMaxTokens = options.defaultMaxTokens ?? 256

      for (const value of [maximum, capacity, defaultMaxTokens]) {
        if (!Number.isSafeInteger(value) || value < 1) {
          return yield* failure("Server limits must be positive safe integers")
        }
      }

      const models = new Map(options.models.map((model) => [model.id, model]))

      if (models.size !== options.models.length || options.models.some((model) => model.id.length === 0)) {
        return yield* failure("Model IDs must be nonempty and unique")
      }

      const created = Math.floor(Date.now() / 1000)
      const permits = Semaphore.makeUnsafe(maximum)
      const lifetime = yield* Effect.scope

      // The server owns in-flight model calls; a disconnected caller also interrupts its call.
      const run = <A, E, R>(effect: Effect.Effect<A, E, R>) =>
        Effect.uninterruptibleMask((restore) =>
          Effect.gen(function*() {
            const fiber = yield* Effect.forkIn(permits.withPermit(effect), lifetime)

            return yield* restore(Fiber.join(fiber)).pipe(Effect.onInterrupt(() => Fiber.interrupt(fiber)))
          })
        )

      let pending = 0

      const admitted = Effect.acquireRelease(
        Effect.suspend(() => {
          if (pending >= capacity) {
            return Effect.fail(failure("Server request capacity reached", 429, "rate_limit_error"))
          }

          pending++

          return Effect.void
        }),
        () =>
          Effect.sync(() => {
            pending--
          })
      )

      const lookup = (id: string) => {
        const model = models.get(id)

        return model === undefined
          ? Effect.fail(failure("Unknown model: " + id, 404, "model_not_found"))
          : Effect.succeed(model)
      }

      const decode = <S extends Schema.Top, E>(schema: S, json: Effect.Effect<unknown, E>) =>
        json.pipe(
          Effect.flatMap(Schema.decodeUnknownEffect(schema, { onExcessProperty: "error" })),
          Effect.mapError((error) => failure(String(error)))
        )

      const recover = <A, R>(effect: Effect.Effect<A, RequestError | Model.ModelError, R>) =>
        effect.pipe(
          Effect.catchTag("ServeRequestError", (error) =>
            Effect.succeed(
              HttpServerResponse.jsonUnsafe(errorBody(error.message, error.code), { status: error.status })
            )),
          Effect.catchTag("ServeModelError", (error) =>
            Effect.succeed(HttpServerResponse.jsonUnsafe(
              errorBody(
                error.invalidRequest ? error.message : "Model execution failed",
                error.invalidRequest ? "invalid_request_error" : "server_error"
              ),
              { status: error.invalidRequest ? 400 : 500 }
            )))
        )

      const generation = (payload: typeof OpenAi.ChatRequest.Type | typeof OpenAi.CompletionRequest.Type) =>
        Effect.gen(function*() {
          const model = yield* lookup(payload.model)

          if (model.generate === undefined) return yield* failure("Model does not support generation")

          const chat = "messages" in payload

          if (chat && payload.max_tokens !== undefined && payload.max_completion_tokens !== undefined) {
            return yield* failure("Supply only one generation token limit")
          }

          const request: Model.GenerationRequest = {
            input: "messages" in payload ? { messages: payload.messages } : { prompt: payload.prompt },
            maxTokens: (chat ? payload.max_completion_tokens : undefined) ?? payload.max_tokens ?? defaultMaxTokens,
            temperature: payload.temperature,
            topP: payload.top_p,
            seed: payload.seed
          }

          if (model.validateGeneration !== undefined) yield* model.validateGeneration(request)

          const id = (chat ? "chatcmpl-" : "cmpl-") + randomUUID()
          const base = { id, created: Math.floor(Date.now() / 1000), model: model.id }
          const source = model.generate(request)

          const usage = (event: Extract<Model.GenerationEvent, { _tag: "Done" }>) => ({
            prompt_tokens: event.promptTokens,
            completion_tokens: event.completionTokens,
            total_tokens: event.promptTokens + event.completionTokens
          })

          if (payload.stream === true) {
            yield* admitted

            const body = Stream.callback<Uint8Array, Model.ModelError>((queue) =>
              run(Effect.gen(function*() {
                let finished = false

                const chunk = (choices: Schema.Json, extra: Schema.JsonObject = {}) =>
                  sse({ ...base, object: chat ? "chat.completion.chunk" : "text_completion", choices, ...extra })

                if (chat) {
                  yield* Queue.offer(
                    queue,
                    chunk([{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }])
                  )
                }

                yield* Stream.runForEach(source, (event) =>
                  Effect.gen(function*() {
                    if (finished) return yield* Effect.die("Generation emitted after Done")

                    if (event._tag === "Delta") {
                      yield* Queue.offer(
                        queue,
                        chunk([
                          chat
                            ? { index: 0, delta: { content: event.text }, finish_reason: null }
                            : { index: 0, text: event.text, logprobs: null, finish_reason: null }
                        ])
                      )
                    } else {
                      finished = true
                      yield* Queue.offer(
                        queue,
                        chunk([
                          chat
                            ? { index: 0, delta: {}, finish_reason: event.finishReason }
                            : { index: 0, text: "", logprobs: null, finish_reason: event.finishReason }
                        ])
                      )

                      if (payload.stream_options?.include_usage === true) {
                        yield* Queue.offer(queue, chunk([], { usage: usage(event) }))
                      }
                    }
                  }))

                if (!finished) return yield* Effect.die("Generation ended without Done")

                yield* Queue.offer(queue, encoder.encode("data: [DONE]" + String.fromCharCode(10, 10)))
                yield* Queue.end(queue)
              })).pipe(Effect.catchCause((cause) => Queue.failCause(queue, cause))), { bufferSize: 8 }).pipe(
                Stream.catchCause(() => Stream.make(sse(errorBody("Model execution failed", "server_error"))))
              )

            return HttpServerResponse.stream(body, {
              contentType: "text/event-stream",
              headers: { "cache-control": "no-cache", "x-accel-buffering": "no" }
            })
          }

          yield* admitted

          return yield* run(Effect.gen(function*() {
            let text = ""
            let done: Extract<Model.GenerationEvent, { _tag: "Done" }> | undefined
            yield* Stream.runForEach(source, (event) =>
              Effect.sync(() => {
                if (event._tag === "Delta") text += event.text
                else done = event
              }))

            if (done === undefined) return yield* failure("Generation ended without Done", 500, "server_error")

            return HttpServerResponse.jsonUnsafe({
              ...base,
              object: chat ? "chat.completion" : "text_completion",
              choices: [
                chat
                  ? { index: 0, message: { role: "assistant", content: text }, finish_reason: done.finishReason }
                  : { index: 0, text, logprobs: null, finish_reason: done.finishReason }
              ],
              usage: usage(done)
            })
          }))
        })

      return handlers
        .handle("list", () =>
          Effect.succeed({
            object: "list",
            data: [...models.keys()].map((id) => ({ id, object: "model" as const, created, owned_by: "local" }))
          }))
        .handleRaw(
          "chat",
          ({ request }) => recover(decode(OpenAi.ChatRequest, request.json).pipe(Effect.flatMap(generation))),
          { uninterruptible: false }
        )
        .handleRaw(
          "complete",
          ({ request }) => recover(decode(OpenAi.CompletionRequest, request.json).pipe(Effect.flatMap(generation))),
          { uninterruptible: false }
        )
        .handleRaw("decide", ({ request }) =>
          recover(Effect.gen(function*() {
            const { seed, reads, ...payload } = yield* decode(DecisionRequest, request.json)
            const model = yield* lookup(payload.model)

            if (model.decide === undefined) return yield* failure("Model does not support decisions")

            yield* admitted

            return HttpServerResponse.jsonUnsafe(
              yield* run(model.decide(payload, { seed: seed ?? "0", reads: reads ?? 1 }))
            )
          })), { uninterruptible: false })
    }))

  return HttpApiBuilder.layer(Api, { openapiPath: "/openapi.json" }).pipe(Layer.provide(handlers))
}

/**
 * Fetch-compatible handler with AbortSignal cancellation and explicit disposal.
 *
 * @since 0.1.0
 * @category constructors
 */
export const toWebHandler = (options: Options): WebHandler =>
  HttpRouter.toWebHandler(layer(options).pipe(Layer.provide(HttpServer.layerServices)))

/**
 * Fetch-compatible request handling and asynchronous model-call shutdown.
 *
 * @since 0.1.0
 * @category models
 */
export interface WebHandler {
  readonly handler: (request: Request) => Promise<Response>
  readonly dispose: () => Promise<void>
}

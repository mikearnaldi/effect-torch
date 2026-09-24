import { describe, expect, it } from "@effect/vitest"
import { Deferred, Effect, Fiber, Queue, Schema, Stream } from "effect"
import { Model, OpenAi, Server } from "../src/index.ts"

const request = (path: string, body: Schema.Json, signal?: AbortSignal) =>
  new Request("http://localhost" + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: signal ?? null
  })

const model: Model.Registration = {
  id: "shared",
  generate: () =>
    Stream.make<ReadonlyArray<Model.GenerationEvent>>(
      { _tag: "Delta", text: "Hello " },
      { _tag: "Delta", text: "世界" },
      { _tag: "Done", finishReason: "stop", promptTokens: 4, completionTokens: 3 }
    ),
  decide: (request) =>
    Effect.succeed({
      model: request.model,
      answers: { answer: { type: "noul", noul: 0.75 } },
      usage: { input_tokens: 5, output_tokens: 256 }
    })
}

const server = (models = [model], options: Partial<Server.Options> = {}) =>
  Effect.acquireRelease(
    Effect.sync(() => Server.toWebHandler({ models, ...options })),
    (server) => Effect.promise(() => server.dispose())
  )

const json = <S extends Schema.Top>(schema: S, response: Response) =>
  Effect.promise(() => response.json()).pipe(Effect.flatMap(Schema.decodeUnknownEffect(schema)))

describe("model HttpApi", () => {
  it.effect("disposal interrupts active streams and awaits their finalizers", () =>
    Effect.gen(function*() {
      const entered = yield* Deferred.make<void>()
      const released = yield* Deferred.make<void>()

      const owned: Model.Registration = {
        ...model,
        generate: () =>
          Stream.callback<Model.GenerationEvent>((queue) =>
            Effect.gen(function*() {
              yield* Effect.addFinalizer(() => Deferred.succeed(released, undefined))
              yield* Deferred.succeed(entered, undefined)
              yield* Queue.offer(queue, { _tag: "Delta", text: "started" })
              yield* Effect.never
            })
          )
      }

      const web = yield* server([owned])

      const response = yield* Effect.promise(() =>
        web.handler(request("/v1/completions", { model: "shared", prompt: "Hi", stream: true }))
      )

      const reader = response.body!.getReader()
      yield* Effect.promise(() => reader.read())
      yield* Deferred.await(entered)
      yield* Effect.promise(() => web.dispose())
      expect(yield* Deferred.isDone(released)).toBe(true)
      yield* Effect.promise(() => reader.cancel())
    }))
  it.effect("serves models and generated OpenAPI", () =>
    Effect.gen(function*() {
      const { handler } = yield* server()
      const response = yield* Effect.promise(() => handler(new Request("http://localhost/v1/models")))
      expect(response.status).toBe(200)
      expect(yield* Effect.promise(() => response.json())).toEqual({
        object: "list",
        data: [{ id: "shared", object: "model", owned_by: "local", created: expect.any(Number) }]
      })
      const spec = yield* Effect.promise(() => handler(new Request("http://localhost/openapi.json")))
      const text = yield* Effect.promise(() => spec.text())
      expect(text).toContain("/v1/decisions")
      expect(text).toContain("/v1/chat/completions")
    }))

  it.effect("shares a model ID across chat, completion and decision routes", () =>
    Effect.gen(function*() {
      const { handler } = yield* server()

      const complete = yield* Effect.promise(() =>
        handler(request("/v1/completions", { model: "shared", prompt: "Hi" }))
      )

      const completion = yield* json(OpenAi.Completion, complete)
      expect(completion.choices[0]?.text).toBe("Hello 世界")
      expect(completion.usage.total_tokens).toBe(7)

      const chat = yield* Effect.promise(() =>
        handler(request("/v1/chat/completions", { model: "shared", messages: [{ role: "user", content: "Hi" }] }))
      )

      expect((yield* json(OpenAi.ChatCompletion, chat)).choices[0]?.message.content).toBe("Hello 世界")

      const decision = yield* Effect.promise(() =>
        handler(
          request("/v1/decisions", {
            model: "shared",
            state: "Hi",
            questions: { answer: { type: "noul" } },
            reads: 4,
            seed: "test"
          })
        )
      )

      expect(decision.status).toBe(200)
      expect(yield* Effect.promise(() => decision.json())).toMatchObject({ answers: { answer: { noul: 0.75 } } })
    }))

  it.effect("streams deltas, finish reason, usage and DONE in OpenAI order", () =>
    Effect.gen(function*() {
      const { handler } = yield* server()

      for (const chat of [true, false]) {
        const payload = chat ? { messages: [{ role: "user", content: "Hi" }] } : { prompt: "Hi" }

        const response = yield* Effect.promise(() =>
          handler(request(chat ? "/v1/chat/completions" : "/v1/completions", {
            ...payload,
            model: "shared",
            stream: true,
            stream_options: { include_usage: true }
          }))
        )

        expect(response.headers.get("content-type")).toContain("text/event-stream")
        const text = yield* Effect.promise(() => response.text())
        const frames = text.trim().split(String.fromCharCode(10, 10))
        expect(frames.at(-1)).toBe("data: [DONE]")
        expect(text).toContain("\"finish_reason\":\"stop\"")
        expect(text).toContain("\"total_tokens\":7")
        expect(text).toContain("世界")

        if (chat) expect(frames[0]).toContain("\"role\":\"assistant\"")
      }
    }))

  it.effect("rejects malformed, unknown, unsupported and ambiguous requests before inference", () =>
    Effect.gen(function*() {
      const { handler } = yield* server()

      for (
        const body of [
          { model: "shared", prompt: "Hi", n: 2 },
          { model: "shared", prompt: "Hi", tools: [] },
          { model: "shared", prompt: "Hi", max_tokens: -1 },
          { model: "shared", prompt: "Hi", stop: "end" }
        ]
      ) {
        const response = yield* Effect.promise(() => handler(request("/v1/completions", body)))
        expect(response.status).toBe(400)
        expect(yield* Effect.promise(() => response.text())).toContain("invalid_request_error")
      }

      const unknown = yield* Effect.promise(() =>
        handler(request("/v1/completions", { model: "missing", prompt: "Hi" }))
      )

      expect(unknown.status).toBe(404)

      const both = yield* Effect.promise(() =>
        handler(request("/v1/chat/completions", {
          model: "shared",
          messages: [{ role: "user", content: "Hi" }],
          max_tokens: 10,
          max_completion_tokens: 10
        }))
      )

      expect(both.status).toBe(400)

      const malformed = yield* Effect.promise(() =>
        handler(
          new Request("http://localhost/v1/completions", {
            method: "POST",
            body: "{",
            headers: { "content-type": "application/json" }
          })
        )
      )

      expect(malformed.status).toBe(400)
    }))

  it.effect("reports model failures without inventing a successful stream completion", () =>
    Effect.gen(function*() {
      const { handler } = yield* server([{
        id: "shared",
        generate: () => Stream.fail(new Model.ModelError({ message: "private detail", invalidRequest: false }))
      }])

      const response = yield* Effect.promise(() =>
        handler(request("/v1/completions", { model: "shared", prompt: "Hi" }))
      )

      expect(response.status).toBe(500)
      expect(yield* Effect.promise(() => response.text())).not.toContain("private detail")

      const stream = yield* Effect.promise(() =>
        handler(request("/v1/completions", { model: "shared", prompt: "Hi", stream: true }))
      )

      const text = yield* Effect.promise(() => stream.text())
      expect(text).toContain("server_error")
      expect(text).not.toContain("[DONE]")
    }))

  it.effect("bounds both task types and releases admission after abort", () =>
    Effect.gen(function*() {
      const entered = yield* Deferred.make<void>()
      const released = yield* Deferred.make<void>()

      const blocking: Model.Registration = {
        ...model,
        generate: () =>
          Stream.callback<Model.GenerationEvent>((queue) =>
            Effect.gen(function*() {
              yield* Effect.addFinalizer(() => Deferred.succeed(released, undefined))
              yield* Deferred.succeed(entered, undefined)
              yield* Queue.offer(queue, { _tag: "Delta", text: "started" })
              yield* Effect.never
            })
          )
      }

      const { handler } = yield* server([blocking], { maxPendingRequests: 1 })
      const controller = new AbortController()

      const first = yield* Effect.forkChild(
        Effect.promise(() => handler(request("/v1/completions", { model: "shared", prompt: "Hi" }, controller.signal)))
      )

      yield* Deferred.await(entered)

      const overloaded = yield* Effect.promise(() =>
        handler(request("/v1/decisions", { model: "shared", state: "Hi", questions: { answer: { type: "noul" } } }))
      )

      expect(overloaded.status).toBe(429)
      controller.abort()
      yield* Deferred.await(released)
      yield* Fiber.await(first)

      const next = yield* Effect.promise(() =>
        handler(request("/v1/decisions", { model: "shared", state: "Hi", questions: { answer: { type: "noul" } } }))
      )

      expect(next.status).toBe(200)
    }))

  it.effect("cancels an SSE producer when the response reader disconnects", () =>
    Effect.gen(function*() {
      const released = yield* Deferred.make<void>()

      const blocking: Model.Registration = {
        ...model,
        generate: () =>
          Stream.callback<Model.GenerationEvent>((queue) =>
            Effect.gen(function*() {
              yield* Effect.addFinalizer(() => Deferred.succeed(released, undefined))
              yield* Queue.offer(queue, { _tag: "Delta", text: "started" })
              yield* Effect.never
            })
          )
      }

      const { handler } = yield* server([blocking], { maxPendingRequests: 1 })

      const response = yield* Effect.promise(() =>
        handler(request("/v1/completions", { model: "shared", prompt: "Hi", stream: true }))
      )

      const reader = response.body!.getReader()
      yield* Effect.promise(() => reader.read())
      yield* Effect.promise(() => reader.cancel())
      yield* Deferred.await(released)

      const next = yield* Effect.promise(() =>
        handler(request("/v1/decisions", { model: "shared", state: "Hi", questions: { answer: { type: "noul" } } }))
      )

      expect(next.status).toBe(200)
    }))
})

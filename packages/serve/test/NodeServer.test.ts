import { NodeHttpServer } from "@effect/platform-node"
import { expect, it } from "@effect/vitest"
import { Context, Deferred, Effect, Layer, Queue, Stream } from "effect"
import { HttpRouter, HttpServer } from "effect/unstable/http"
import { createServer } from "node:http"
import { type Model, Server } from "../src/index.ts"

it.effect("serves through Node HttpApi and releases a model call on socket disconnect", () =>
  Effect.gen(function*() {
    const released = yield* Deferred.make<void>()

    const model: Model.Registration = {
      id: "node-test",
      generate: () =>
        Stream.callback<Model.GenerationEvent>((queue) =>
          Effect.gen(function*() {
            yield* Effect.addFinalizer(() => Deferred.succeed(released, undefined))
            yield* Queue.offer(queue, { _tag: "Delta", text: "first" })
            yield* Effect.never
          })
        )
    }

    const context = yield* Layer.build(
      HttpRouter.serve(Server.layer({ models: [model] })).pipe(
        Layer.provideMerge(NodeHttpServer.layer(createServer, { host: "127.0.0.1", port: 0 }))
      )
    )

    const address = Context.get(context, HttpServer.HttpServer).address

    if (address._tag !== "TcpAddress") return yield* Effect.die("Expected TCP server")

    const base = "http://127.0.0.1:" + address.port
    const models = yield* Effect.promise(() => fetch(base + "/v1/models"))
    expect(models.status).toBe(200)
    expect(yield* Effect.promise(() => models.text())).toContain("node-test")

    const response = yield* Effect.promise(() =>
      fetch(base + "/v1/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "node-test", prompt: "hello", stream: true })
      })
    )

    const reader = response.body!.getReader()
    expect(new TextDecoder().decode((yield* Effect.promise(() => reader.read())).value)).toContain("first")
    yield* Effect.promise(() => reader.cancel())
    yield* Deferred.await(released)
  }))

it.effect("layer-owned model resources outlive active calls and close after server shutdown", () =>
  Effect.gen(function*() {
    const events: Array<string> = []
    const application = Layer.unwrap(Effect.gen(function*() {
      yield* Effect.acquireRelease(
        Effect.sync(() => {
          events.push("model acquired")
        }),
        () =>
          Effect.sync(() => {
            events.push("model released")
          })
      )

      const model: Model.Registration = {
        id: "owned",
        generate: () =>
          Stream.callback<Model.GenerationEvent>((queue) =>
            Effect.gen(function*() {
              yield* Effect.addFinalizer(() =>
                Effect.sync(() => {
                  events.push("call released")
                })
              )
              yield* Queue.offer(queue, { _tag: "Delta", text: "first" })
              yield* Effect.never
            })
          )
      }

      return Server.layer({ models: [model] })
    }))

    const web = HttpRouter.toWebHandler(application.pipe(Layer.provide(HttpServer.layerServices)))
    yield* Effect.addFinalizer(() => Effect.promise(() => web.dispose()))
    const response = yield* Effect.promise(() =>
      web.handler(
        new Request("http://localhost/v1/completions", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model: "owned", prompt: "hello", stream: true })
        })
      )
    )
    const reader = response.body!.getReader()
    expect(new TextDecoder().decode((yield* Effect.promise(() => reader.read())).value)).toContain("first")
    expect(events).toEqual(["model acquired"])

    yield* Effect.promise(() => web.dispose())
    expect(events).toEqual(["model acquired", "call released", "model released"])
    yield* Effect.promise(() => reader.cancel())
  }))

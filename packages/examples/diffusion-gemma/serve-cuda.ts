/**
 * pnpm --filter @effect-torch/examples serve-cuda <checkpoint-directory> [port]
 */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { DiffusionGemma } from "@effect-torch/models"
import { Server } from "@effect-torch/serve"
import { NodeHttpServer, NodeRuntime } from "@effect/platform-node"
import { Effect, Layer } from "effect"
import { HttpRouter } from "effect/unstable/http"
import { createServer } from "node:http"

const application = Layer.unwrap(Effect.gen(function*() {
  const [directory, portText = "8080"] = process.argv.slice(2)
  const port = Number(portText)
  const maxConcurrentRequests = Number(process.env.MAX_CONCURRENT_REQUESTS ?? "1")
  const maxTokens = Number(process.env.MAX_TOKENS ?? "16384")

  if (
    directory === undefined || !Number.isInteger(port) || port < 0 || port > 65535 ||
    !Number.isInteger(maxConcurrentRequests) || maxConcurrentRequests <= 0 ||
    !Number.isInteger(maxTokens) || maxTokens <= 0
  ) {
    return yield* Effect.die("usage: serve-cuda <checkpoint-directory> [port]")
  }

  const model = yield* DiffusionGemma.load({ directory, id: "diffusiongemma", maxTokens })

  yield* Effect.log("Loaded diffusiongemma for generation and decisions")

  return HttpRouter.serve(Server.layer({ models: [model], maxConcurrentRequests, maxPendingRequests: 16 })).pipe(
    Layer.provide(NodeHttpServer.layer(createServer, { host: "127.0.0.1", port }))
  )
})).pipe(Layer.provide(BackendCuda.layer()))

NodeRuntime.runMain(Layer.launch(application))

/**
 * pnpm --filter @effect-torch/examples serve-cuda <checkpoint-directory> [port]
 */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Safetensors, Tensor } from "@effect-torch/core"
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
  const maxTokens = Number(process.env.MAX_TOKENS ?? "8192")
  const initializedStatePath = process.env.INITIALIZED_STATE_PATH

  if (
    directory === undefined || !Number.isInteger(port) || port < 0 || port > 65535 ||
    !Number.isInteger(maxConcurrentRequests) || maxConcurrentRequests <= 0 ||
    !Number.isInteger(maxTokens) || maxTokens <= 0 ||
    initializedStatePath === undefined || initializedStatePath.length === 0
  ) {
    return yield* Effect.die(
      "usage: INITIALIZED_STATE_PATH=<initialized-rope.safetensors> serve-cuda <checkpoint-directory> [port]"
    )
  }

  const initializedState = yield* Effect.acquireRelease(
    Safetensors.load(initializedStatePath),
    (tensors) => Tensor.clearAll(Object.values(tensors)),
    { interruptible: true }
  )
  const expectedInitializedState = {
    "model.decoder.rotary_emb.full_attention_inv_freq": [256],
    "model.decoder.rotary_emb.sliding_attention_inv_freq": [128]
  } as const
  if (
    Object.keys(initializedState).length !== Object.keys(expectedInitializedState).length ||
    Object.entries(expectedInitializedState).some(([name, shape]) => {
      const tensor = initializedState[name]
      return tensor === undefined || tensor.dtype !== "f32" || tensor.shape.join(",") !== shape.join(",")
    })
  ) return yield* Effect.die("initialized state tensor metadata differs")

  const model = yield* DiffusionGemma.load({
    directory,
    id: "diffusiongemma",
    maxTokens,
    initializedState
  })

  yield* Effect.log("Loaded diffusiongemma for generation and decisions")

  return HttpRouter.serve(Server.layer({ models: [model], maxConcurrentRequests, maxPendingRequests: 16 })).pipe(
    Layer.provide(NodeHttpServer.layer(createServer, { host: "127.0.0.1", port }))
  )
})).pipe(Layer.provide(BackendCuda.layer()))

NodeRuntime.runMain(Layer.launch(application))

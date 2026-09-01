import * as BackendCuda from "@effect-torch/backend-cuda"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { inference } from "./model.ts"

const prompt = process.argv.slice(2).join(" ")
if (prompt.length === 0) {
  process.stderr.write("usage: pnpm muse-glimmer:cuda <prompt>\n")
} else {
  NodeRuntime.runMain(inference(prompt).pipe(Effect.provide(BackendCuda.layer())))
}

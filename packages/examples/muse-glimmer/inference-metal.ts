import * as BackendApple from "@effect-torch/backend-apple-native"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { inference } from "./model.ts"

const prompt = process.argv.slice(2).join(" ")
if (prompt.length === 0) {
  process.stderr.write("usage: pnpm muse-glimmer:metal <prompt>\n")
} else {
  NodeRuntime.runMain(inference(prompt).pipe(Effect.provide(BackendApple.layer())))
}

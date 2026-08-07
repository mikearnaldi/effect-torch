import { Runtime } from "@effect-torch/core"
import { Effect, Layer } from "effect"
import { makeRuntime as makeRuntimeAdapter } from "./internal/adapter.js"
import native from "./internal/native.js"

let runtime: Runtime.RuntimeService | undefined

export const makeRuntime = (): Runtime.RuntimeService => runtime ??= makeRuntimeAdapter(native)

export const layer: Layer.Layer<Runtime.Runtime> = Layer.effect(Runtime.Runtime, Effect.sync(makeRuntime))

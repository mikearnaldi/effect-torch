import { Context, Effect, Layer } from "effect"
import native from "@effect-torch/native"

export type DeviceKind = "cpu" | "metal" | "cuda"

export class CurrentDevice extends Context.Service<CurrentDevice, DeviceKind>()(
  "@effect-torch/core/CurrentDevice"
) {}

export const layer = (device: DeviceKind): Layer.Layer<CurrentDevice> =>
  Layer.succeed(CurrentDevice, device)

export const Cpu: Layer.Layer<CurrentDevice> = layer("cpu")

export const Metal: Layer.Layer<CurrentDevice> = layer("metal")

export const Cuda: Layer.Layer<CurrentDevice> = layer("cuda")

export const isAvailable = (device: DeviceKind): Effect.Effect<boolean> =>
  Effect.sync(() => native.isDeviceAvailable(device))

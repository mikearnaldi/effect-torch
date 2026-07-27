import { Context, Effect, Layer } from "effect"
import native from "@effect-torch/native"

/**
 * Devices supported by the native backend.
 *
 * @since 0.1.0
 * @category models
 */
export type DeviceKind = "cpu" | "metal" | "cuda"

/**
 * The device on which new tensors are created. Every tensor constructor
 * requires this service, so device selection is explicit and tracked at the
 * type level.
 *
 * @since 0.1.0
 * @category services
 */
export class CurrentDevice extends Context.Service<CurrentDevice, DeviceKind>()(
  "@effect-torch/core/CurrentDevice"
) {}

/**
 * Creates a layer that provides {@link CurrentDevice} with the given device.
 *
 * @since 0.1.0
 * @category layers
 */
export const layer = (device: DeviceKind): Layer.Layer<CurrentDevice> =>
  Layer.succeed(CurrentDevice, device)

/**
 * Provides {@link CurrentDevice} with the CPU device.
 *
 * @since 0.1.0
 * @category layers
 */
export const Cpu: Layer.Layer<CurrentDevice> = layer("cpu")

/**
 * Provides {@link CurrentDevice} with the Metal device (macOS only).
 *
 * @since 0.1.0
 * @category layers
 */
export const Metal: Layer.Layer<CurrentDevice> = layer("metal")

/**
 * Provides {@link CurrentDevice} with the CUDA device (requires a build with
 * the `cuda` feature).
 *
 * @since 0.1.0
 * @category layers
 */
export const Cuda: Layer.Layer<CurrentDevice> = layer("cuda")

/**
 * Checks whether a device is available on this machine and build.
 *
 * @since 0.1.0
 * @category utilities
 */
export const isAvailable = (device: DeviceKind): Effect.Effect<boolean> =>
  Effect.sync(() => native.isDeviceAvailable(device))

/**
 * Provides {@link CurrentDevice} with the best available device, probing in
 * priority order: CUDA, then Metal, falling back to CPU.
 *
 * @since 0.1.0
 * @category layers
 */
export const Best: Layer.Layer<CurrentDevice> = Layer.effect(
  CurrentDevice,
  Effect.gen(function* () {
    if (yield* isAvailable("cuda")) return "cuda" as const
    if (yield* isAvailable("metal")) return "metal" as const
    return "cpu" as const
  })
)

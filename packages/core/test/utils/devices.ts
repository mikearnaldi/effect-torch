import { layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import native from "@effect-torch/native"
import { Device } from "../../src/index.ts"

export type TestDevice = "cpu" | "metal"

export const metalAvailable: boolean = (() => {
  try {
    return native.isDeviceAvailable("metal")
  } catch {
    return false
  }
})()

/** Device-dependent float array: f64 on CPU, f32 on Metal (no f64 there). */
export const floats = (device: TestDevice, values: ReadonlyArray<number>): Float32Array | Float64Array =>
  device === "metal" ? new Float32Array(values) : new Float64Array(values)

/** The dtype that {@link floats} produces, for explicit dtype options. */
export const floatDtype = (device: TestDevice): "f32" | "f64" => (device === "metal" ? "f32" : "f64")

const eps = (device: TestDevice): number => (device === "metal" ? 1e-4 : 1e-12)

const closeEnough = (device: TestDevice, a: number, b: number): boolean =>
  a === b || (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= eps(device)

/**
 * deepStrictEqual with a device-dependent tolerance for numeric content:
 * exact for shapes, dtypes and strings; elementwise-close for numbers.
 */
export const deep = (device: TestDevice, actual: unknown, expected: unknown): void => {
  if (typeof actual === "number" && typeof expected === "number") {
    assert.assertTrue(closeEnough(device, actual, expected), `${actual} != ${expected}`)
    return
  }
  if (Array.isArray(actual) && Array.isArray(expected)) {
    const numeric = (v: ReadonlyArray<unknown>): v is ReadonlyArray<number> =>
      v.every((x) => typeof x === "number")
    if (numeric(actual) && numeric(expected)) {
      assert.deepStrictEqual(actual.length, expected.length)
      actual.forEach((v, i) => {
        assert.assertTrue(closeEnough(device, v, expected[i]), `[${i}]: ${v} != ${expected[i]}`)
      })
      return
    }
  }
  assert.deepStrictEqual(actual, expected)
}

type SuiteFn = Parameters<ReturnType<typeof layer<Device.CurrentDevice, never>>>[1]

/**
 * Registers the same suite once per device: always on CPU, and on Metal
 * when the machine has one. The suite body receives the device and can
 * pick dtypes/tolerances with {@link floats}/{@link floatDtype} and skip
 * unsupported sections with a plain `if (device === "cpu")`.
 */
export const onDevices = (name: string, make: (device: TestDevice) => SuiteFn): void => {
  layer(Device.Cpu)(`${name} (cpu)`, make("cpu"))
  if (metalAvailable) {
    layer(Device.Metal)(`${name} (metal)`, make("metal"))
  }
}

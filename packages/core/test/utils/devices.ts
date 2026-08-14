import * as BackendApple from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import { layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import type { Runtime } from "../../src/index.ts"

export type TestDevice = "cpu" | "metal"

// Availability is sampled once during suite registration. An unavailable Metal
// backend omits that matrix entry rather than falling back to CPU under its name.
export const metalAvailable: boolean = Effect.runSync(BackendApple.isAvailable)

/** Encodes common numerical fixtures as f32, the shared CPU/Metal dtype. */
export const floats = (values: ReadonlyArray<number>): Float32Array => new Float32Array(values)

export const floatDtype = "f32" as const

/** Default absolute f32 tolerance; magnitude-sensitive suites scale it explicitly. */
export const TOL = 1e-4

/** Finite-difference step and tolerance for gradient checks in f32:
 * large enough that f(x±eps) clears f32 rounding, small enough that the
 * central-difference truncation stays well below the tolerance. */
export const GRADCHECK_EPS = 2e-3
export const GRADCHECK_TOL = 2e-2

const closeEnough = (a: number, b: number): boolean =>
  a === b || (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= TOL

/**
 * deepStrictEqual with the f32 tolerance for numeric content: exact for
 * shapes, dtypes and strings; elementwise-close for numbers.
 */
export const deep = (actual: unknown, expected: unknown): void => {
  if (typeof actual === "number" && typeof expected === "number") {
    assert.assertTrue(closeEnough(actual, expected), `${actual} != ${expected}`)
    return
  }
  if (Array.isArray(actual) && Array.isArray(expected)) {
    const numeric = (v: ReadonlyArray<unknown>): v is ReadonlyArray<number> => v.every((x) => typeof x === "number")
    if (numeric(actual) && numeric(expected)) {
      assert.deepStrictEqual(actual.length, expected.length)
      actual.forEach((v, i) => {
        assert.assertTrue(closeEnough(v, expected[i]), `[${i}]: ${v} != ${expected[i]}`)
      })
      return
    }
  }
  assert.deepStrictEqual(actual, expected)
}

type SuiteFn = Parameters<ReturnType<typeof layer<Runtime.Runtime, never>>>[1]

/**
 * Registers the same suite with its real backend layer: always CPU, plus Metal
 * when available. This matrix never emulates unsupported device behavior.
 */
export const onDevices = (name: string, make: (device: TestDevice) => SuiteFn): void => {
  layer(BackendCpu.layer)(`${name} (cpu)`, make("cpu"))
  if (metalAvailable) {
    layer(BackendApple.layer)(`${name} (metal)`, make("metal"))
  }
}

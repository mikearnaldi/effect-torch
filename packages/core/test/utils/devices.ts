import * as BackendApple from "@effect-torch/backend-apple-native"
import * as BackendCpu from "@effect-torch/backend-cpu"
import * as BackendCuda from "@effect-torch/backend-cuda"
import { layer } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect, Predicate } from "effect"
import type { Runtime } from "../../src/index.ts"

/** Backends included in the shared numerical test matrix. */
export type TestDevice = "cpu" | "metal" | "cuda"

/**
 * Whether Metal was available when the test suite was registered. An
 * unavailable backend is omitted rather than silently replaced with CPU.
 */
export const metalAvailable: boolean = Effect.runSync(BackendApple.isAvailable)

/** Whether CUDA was available when the test suite was registered. */
export const cudaAvailable: boolean = Effect.runSync(BackendCuda.isAvailable)

/** Encodes numerical fixtures as f32, the shared CPU/Metal dtype. */
export const floats = (values: ReadonlyArray<number>): Float32Array => new Float32Array(values)

/** Shared floating-point dtype used by CPU and Metal fixtures. */
export const floatDtype = "f32" as const

/** Default absolute f32 tolerance; magnitude-sensitive suites scale it explicitly. */
export const TOL = 1e-4

/**
 * Finite-difference step for f32 gradient checks. It clears f32 rounding while
 * keeping central-difference truncation below the matching tolerance.
 */
export const GRADCHECK_EPS = 2e-3

/** Absolute tolerance used with {@link GRADCHECK_EPS}. */
export const GRADCHECK_TOL = 2e-2

const closeEnough = (a: number, b: number): boolean =>
  a === b || (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= TOL

/**
 * Compares structures exactly except for numeric arrays, which use the shared
 * f32 tolerance.
 */
export const deep = <A>(actual: A, expected: A): void => {
  if (Predicate.isNumber(actual) && Predicate.isNumber(expected)) {
    assert.assertTrue(closeEnough(actual, expected), `${actual} != ${expected}`)
    return
  }
  if (Array.isArray(actual) && Array.isArray(expected)) {
    const numeric = (v: ReadonlyArray<unknown>): v is ReadonlyArray<number> => v.every(Predicate.isNumber)
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
 * Registers the same suite with its real backend layer. CPU always runs, and
 * Metal runs when available. This matrix never emulates unsupported devices.
 */
export const onDevices = (name: string, make: (device: TestDevice) => SuiteFn): void => {
  layer(BackendCpu.layer)(`${name} (cpu)`, make("cpu"))
  if (metalAvailable) {
    layer(BackendApple.layer())(`${name} (metal)`, make("metal"))
  }
  if (cudaAvailable) {
    layer(BackendCuda.layer())(`${name} (cuda)`, make("cuda"))
  }
}

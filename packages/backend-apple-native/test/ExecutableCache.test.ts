import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { createRuntimeAdapter, normalizedStructure, structuralCacheKey } from "../src/internal/adapter.ts"
import type { NativeAddon, NativeDType } from "../src/internal/native-addon.js"

const numberFromBits = (high: number, low: number): number => {
  const view = new DataView(new ArrayBuffer(8))
  view.setUint32(0, high, false)
  view.setUint32(4, low, false)
  return view.getFloat64(0, false)
}

// Cache keys encode exceptional IEEE-754 values and -0 as raw bits. Ordinary
// finite values remain numbers, so equal attributes share cache entries.
describe("Metal executable cache keys", () => {
  it("uses the selected ordinal in placement identity", () => {
    // SAFETY: Adapter construction does not call the addon; this test only needs its identity.
    const runtime = createRuntimeAdapter({} as NativeAddon, 2)
    expect(runtime.placement).toEqual({
      id: "metal:2",
      deviceType: "metal",
      description: "Apple Metal device 2",
      ordinal: 2
    })
  })

  it("passes the selected ordinal into graph leaves", () => {
    let receivedOrdinal: number | undefined
    class LazyTensorDouble {
      static zeros(_shape: Array<number>, _dtype?: NativeDType | null, deviceOrdinal?: number) {
        receivedOrdinal = deviceOrdinal
        return new LazyTensorDouble()
      }

      metadata(): [Array<number>, string] {
        return [[2], "f32"]
      }
    }
    // SAFETY: This test invokes only LazyTensor.zeros and metadata through the adapter.
    const native = { LazyTensor: LazyTensorDouble } as NativeAddon
    const runtime = createRuntimeAdapter(native, 2)

    Effect.runSync(runtime.node({
      op: "zeros",
      inputs: [],
      attributes: { shape: [2], dtype: "f32" }
    }))

    expect(receivedOrdinal).toBe(2)
  })

  it("distinguishes special numbers by their IEEE-754 bits", () => {
    const values = [
      numberFromBits(0x7ff8_0000, 0x0000_0001),
      numberFromBits(0x7ff8_0000, 0x0000_0002),
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
      0,
      -0
    ]
    const keys = values.map((value) => structuralCacheKey({ attributes: { value } }))

    expect(new Set(keys).size).toBe(values.length)
    expect(normalizedStructure(values[0])).toEqual({ $number: "7ff8000000000001" })
    expect(normalizedStructure(values[1])).toEqual({ $number: "7ff8000000000002" })
    expect(normalizedStructure(Number.POSITIVE_INFINITY)).toEqual({ $number: "7ff0000000000000" })
    expect(normalizedStructure(Number.NEGATIVE_INFINITY)).toEqual({ $number: "fff0000000000000" })
    expect(normalizedStructure(0)).toBe(0)
    expect(normalizedStructure(-0)).toEqual({ $number: "8000000000000000" })
  })

  it("normalizes finite values and attributes deterministically", () => {
    const nextAfterOne = numberFromBits(0x3ff0_0000, 0x0000_0001)
    expect(normalizedStructure(1)).toBe(1)
    expect(normalizedStructure(nextAfterOne)).toBe(nextAfterOne)
    expect(structuralCacheKey({ value: 1 })).not.toBe(structuralCacheKey({ value: nextAfterOne }))
    expect(normalizedStructure(new Uint8Array([0, 255]))).toEqual([0, 255])
    expect(structuralCacheKey({ value: 1, data: new Uint8Array([0, 255]) })).toBe(
      structuralCacheKey({ data: [0, 255], value: 1 })
    )
  })
})

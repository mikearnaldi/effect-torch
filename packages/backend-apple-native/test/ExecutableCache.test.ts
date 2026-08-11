import { describe, expect, it } from "vitest"
import { normalizedStructure, structuralCacheKey } from "../src/internal/adapter.ts"

const numberFromBits = (high: number, low: number): number => {
  const view = new DataView(new ArrayBuffer(8))
  view.setUint32(0, high, false)
  view.setUint32(4, low, false)
  return view.getFloat64(0, false)
}

describe("Metal executable structural cache", () => {
  it("keeps every representable special-number key distinct", () => {
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

  it("keeps finite keys deterministic and preserves attribute normalization", () => {
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

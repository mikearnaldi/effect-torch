import { describe, expect, it } from "@effect/vitest"
import { Runtime } from "../src/index.ts"

describe("storage representations", () => {
  it("validates canonical byte geometry for each accepted K-quant", () => {
    const formats = [["Q2_K", 84], ["Q3_K", 110], ["Q4_K", 144], ["Q5_K", 176], ["Q6_K", 210]] as const
    for (const [encoding, bytes] of formats) {
      expect(Runtime.encodedStorageGeometry(encoding, [2, 3, 512])).toEqual({
        physicalShape: [6, bytes * 2],
        byteLength: 12 * bytes
      })
      expect(Runtime.validEncodedStorage([2, 3, 512], {
        encoding,
        physicalShape: [6, bytes * 2],
        physicalDtype: "u8"
      })).toBe(true)
      expect(Runtime.validEncodedStorage([2, 3, 512], {
        encoding,
        physicalShape: [6, bytes * 2 + 1],
        physicalDtype: "u8"
      })).toBe(false)
    }
  })

  it("rejects unknown formats instead of inheriting Q6_K geometry", () => {
    for (const encoding of ["Q2_K_XL", "NVFP4", "q4_k", "", null, 14]) {
      expect(Runtime.isTensorStorageEncoding(encoding)).toBe(false)
    }
    // Runtime metadata is untrusted even when a TypeScript caller forges its type.
    // @ts-expect-error exercise an unknown representation at the runtime boundary
    expect(Runtime.encodedStorageGeometry("NVFP4", [2, 256])).toBeUndefined()
  })

  it("rejects partial blocks, invalid dimensions and overflowing logical sizes", () => {
    for (const shape of [[], [255], [0, 256], [-1, 256], [1.5, 256], [Infinity, 256], [2 ** 40, 2 ** 20]]) {
      expect(Runtime.encodedStorageGeometry("Q4_K", shape)).toBeUndefined()
    }
    expect(Runtime.encodedStorageGeometry("Q4_K", [256])).toEqual({ physicalShape: [1, 144], byteLength: 144 })
  })
})

import { expect } from "@effect/vitest"
import { Effect } from "effect"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
import { Safetensors, Tensor } from "../src/index.ts"
import { onDevices } from "./utils/devices.ts"

const halfTensor = (dtype: "f16" | "bf16", values: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
  Effect.gen(function*() {
    const source = yield* Tensor.fromTypedArray(new Float32Array(values), shape)
    const [value] = yield* Tensor.compute([yield* Tensor.cast(source, dtype)])
    return value
  })

const halfArchive = (dtype: "F16" | "BF16", bits: ReadonlyArray<number>) => {
  const payload = Buffer.alloc(bits.length * 2)
  bits.forEach((value, index) => payload.writeUInt16LE(value, index * 2))
  const json = JSON.stringify({ weight: { dtype, shape: [bits.length], data_offsets: [0, payload.length] } })
  const header = Buffer.from(json.padEnd(Math.ceil(json.length / 8) * 8, " "))
  const length = Buffer.alloc(8)
  length.writeBigUInt64LE(BigInt(header.length))
  return Buffer.concat([length, header, payload])
}

onDevices("Dtype legalization", (device) => (it) => {
  it.effect("integer narrowing keeps low bits without floating conversion", () =>
    Effect.gen(function*() {
      const source = yield* Tensor.fromTypedArray(
        new BigInt64Array([-1n, (1n << 53n) + 1n, (1n << 63n) - 1n, -(1n << 63n)]),
        [4]
      )
      for (const dtype of ["u8", "u32"] as const) {
        const [result] = yield* Tensor.compute([yield* Tensor.cast(source, dtype)])
        const maximum = dtype === "u8" ? 255 : 4294967295
        expect(yield* Tensor.toNumberArray(result)).toEqual([maximum, 1, maximum, 0])
        yield* Tensor.clear(result)
      }
    }))

  it.effect("floating casts to integers truncate and saturate with defined nonfinite results", () =>
    Effect.gen(function*() {
      const source = yield* Tensor.fromTypedArray(
        new Float32Array([NaN, Infinity, -Infinity, 1.75, -1.75, 2 ** 64, -(2 ** 64)]),
        [7]
      )
      for (const dtype of ["i64", "u32", "u8"] as const) {
        const [result] = yield* Tensor.compute([yield* Tensor.cast(source, dtype)])
        const expected = dtype === "i64"
          ? [0n, (1n << 63n) - 1n, -(1n << 63n), 1n, -1n, (1n << 63n) - 1n, -(1n << 63n)]
          : [0, dtype === "u32" ? 4294967295 : 255, 0, 1, 0, dtype === "u32" ? 4294967295 : 255, 0]
        expect(Array.from<number | bigint>(yield* Tensor.toTypedArray(result))).toEqual(expected)
        yield* Tensor.clear(result)
      }
    }))

  it.effect("integer-to-F16 rounding overflows to signed infinity", () =>
    Effect.gen(function*() {
      const source = yield* Tensor.fromTypedArray(
        new BigInt64Array([65519n, 65520n, -65519n, -65520n, (1n << 63n) - 1n, -(1n << 63n)]),
        [6]
      )
      const [result] = yield* Tensor.compute([yield* Tensor.cast(source, "f16")])
      expect(yield* Tensor.toNumberArray(result)).toEqual([65504, Infinity, -65504, -Infinity, Infinity, -Infinity])
      yield* Tensor.clear(result)
    }))

  it.effect("integer-to-float casts round directly from the exact integer", () =>
    Effect.gen(function*() {
      for (const [dtype, fractionBits] of [["f32", 23], ["bf16", 7]] as const) {
        const base = 1n << 62n
        const halfUlp = 1n << BigInt(61 - fractionBits)
        const source = yield* Tensor.fromTypedArray(
          new BigInt64Array([base + halfUlp - 1n, base + halfUlp, base + halfUlp + 1n, -base - halfUlp - 1n]),
          [4]
        )
        const roundedUp = 2 ** 62 + 2 ** (62 - fractionBits)
        const [result] = yield* Tensor.compute([yield* Tensor.cast(source, dtype)])
        expect(yield* Tensor.toNumberArray(result)).toEqual([2 ** 62, 2 ** 62, roundedUp, -roundedUp])
        yield* Tensor.clear(result)
      }
    }))

  it.effect("arg reductions reject empty axes before reading storage", () =>
    Effect.gen(function*() {
      for (const shape of [[0], [0, 2]]) {
        const source = yield* Tensor.zeros(shape)
        for (const reduce of [Tensor.argmax, Tensor.argmin]) {
          const result = yield* Effect.flip(Effect.gen(function*() {
            const root = yield* reduce(source, 0)
            return yield* Tensor.freezeProgram([root])
          }))
          expect(result.message).toMatch(/empty|zero/i)
        }
      }
    }))

  if (device !== "metal") {
    it.effect("explicit F64 to half casts avoid double rounding through F32", () =>
      Effect.gen(function*() {
        for (const [dtype, fractionBits] of [["f16", 10], ["bf16", 7]] as const) {
          const halfway = 1 + 2 ** -(fractionBits + 1)
          const source = yield* Tensor.fromTypedArray(new Float64Array([halfway + 2 ** -40]), [1])
          const [result] = yield* Tensor.compute([yield* Tensor.cast(source, dtype)])
          expect(yield* Tensor.toNumberArray(result)).toEqual([1 + 2 ** -fractionBits])
          yield* Tensor.clear(result)
        }
      }))
  }

  for (const dtype of ["f16", "bf16"] as const) {
    it.effect(`${dtype} casts preserve subnormals and arithmetic retains target F32 behavior`, () =>
      Effect.gen(function*() {
        const smallest = 2 ** (dtype === "f16" ? -24 : -133)
        const source = yield* Tensor.fromTypedArray(new Float32Array([smallest, -smallest]), [2])
        const referenceProgram = yield* Tensor.freezeProgram([yield* Tensor.add(source, source)], { optimize: false })
        const [reference] = yield* Tensor.runProgram(referenceProgram, [])
        // RFC 0025 preserves each target's existing arithmetic treatment of
        // subnormals. Dense conversions must preserve their represented bits.
        const expectedSum = yield* Tensor.toNumberArray(reference)
        const value = yield* halfTensor(dtype, [smallest, -smallest], [2])
        const cast = yield* Tensor.cast(value, "f32")
        const sum = yield* Tensor.add(value, value)
        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram([cast, sum], { optimize })
          const [widened, doubled] = yield* Tensor.runProgram(program, [])
          expect(yield* Tensor.toNumberArray(widened)).toEqual([smallest, -smallest])
          expect(yield* Tensor.toNumberArray(doubled)).toEqual(expectedSum)
          yield* Tensor.clearAll([widened, doubled])
        }
        yield* Tensor.clearAll([value, reference])
      }))

    it.effect(`${dtype} comparisons coerce mixed scalars before comparing`, () =>
      Effect.gen(function*() {
        const value = yield* halfTensor(dtype, [1], [1])
        const scalar = yield* Tensor.constant(1 + 2 ** -(dtype === "bf16" ? 8 : 11), { dtype: "f32" })
        const roots = [
          yield* Tensor.eq(value, scalar),
          yield* Tensor.eq(scalar, value),
          yield* Tensor.lt(value, scalar)
        ]
        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram(roots, { optimize })
          const outputs = yield* Tensor.runProgram(program, [])
          expect(yield* Tensor.toNumberArray(outputs[0])).toEqual([1])
          expect(yield* Tensor.toNumberArray(outputs[1])).toEqual([1])
          expect(yield* Tensor.toNumberArray(outputs[2])).toEqual([0])
          yield* Tensor.clearAll(outputs)
        }
        yield* Tensor.clear(value)
      }))

    it.effect(`${dtype} scalar coercion rounds before arithmetic`, () =>
      Effect.gen(function*() {
        const value = yield* halfTensor(dtype, [3], [1])
        const scalar = yield* Tensor.constant(1 + 2 ** -(dtype === "bf16" ? 8 : 11), { dtype: "f32" })
        const root = yield* Tensor.mul(value, scalar)
        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram([root], { optimize })
          const [result] = yield* Tensor.runProgram(program, [])
          expect(yield* Tensor.toNumberArray(result)).toEqual([3])
          yield* Tensor.clear(result)
        }
        yield* Tensor.clear(value)
      }))

    it.effect(`${dtype} reductions and linear bias use F32 accumulation`, () =>
      Effect.gen(function*() {
        const large = dtype === "bf16" ? 256 : 2048
        const x = yield* halfTensor(dtype, [large, 1], [1, 2])
        const weight = yield* halfTensor(dtype, [1, 1], [2, 1])
        const bias = yield* halfTensor(dtype, [-large], [1])
        const reductionInput = yield* halfTensor(dtype, [large, 1, -large], [3])
        const root = yield* Tensor.linear(x, weight, bias)
        const sum = yield* Tensor.sum(reductionInput)
        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram([root, sum], { optimize })
          const [linear, reduction] = yield* Tensor.runProgram(program, [])
          expect(linear.dtype).toBe(dtype)
          expect(reduction.dtype).toBe(dtype)
          expect(yield* Tensor.toNumberArray(linear)).toEqual([1])
          expect(yield* Tensor.toNumberArray(reduction)).toEqual([1])
          yield* Tensor.clearAll([linear, reduction])
        }
        yield* Tensor.clearAll([x, weight, bias, reductionInput])
      }))

    it.effect(`${dtype} attention retains half signatures with F32 computation`, () =>
      Effect.gen(function*() {
        const q = yield* halfTensor(dtype, [0, 0, 0, 0], [1, 2, 2])
        const k = yield* halfTensor(dtype, [0, 0, 0, 0], [1, 2, 2])
        const v = yield* halfTensor(dtype, [1, 3, 5, 7], [1, 2, 2])
        const root = yield* Tensor.scaledDotProductAttention(q, k, v)
        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram([root], { optimize })
          const [result] = yield* Tensor.runProgram(program, [])
          expect(result.dtype).toBe(dtype)
          expect(yield* Tensor.toNumberArray(result)).toEqual([3, 5, 3, 5])
          yield* Tensor.clear(result)
        }
        yield* Tensor.clearAll([q, k, v])
      }))

    it.effect(`${dtype} matmul preserves model signatures across optimization modes`, () =>
      Effect.gen(function*() {
        const a = yield* halfTensor(dtype, [1, 2, 3, 4, 5, 6], [2, 3])
        const b = yield* halfTensor(dtype, [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [3, 4])
        const left = yield* Tensor.makeInput(0, a)
        const right = yield* Tensor.makeInput(1, b)
        const root = yield* Tensor.matmul(left, right)
        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram([root], { optimize })
          const [result] = yield* Tensor.runProgram(program, [a, b])
          expect(root.dtype).toBe(dtype)
          expect(program.outputs[0].dtype).toBe(dtype)
          expect(result.dtype).toBe(dtype)
          const legalization = program.handle.diagnostics.legalization
          expect(legalization).toBeDefined()
          expect(legalization!.policyRevision).toBeGreaterThan(0)
          expect(legalization!.targetArchitecture.length).toBeGreaterThan(0)
          if (
            device === "cuda" && dtype === "bf16" &&
            Number(legalization!.targetArchitecture.replace("sm_", "")) >= 80
          ) {
            expect(program.handle.diagnostics.instructions).toContainEqual({
              kind: "cublas_bf16_gemm_f32_accum",
              count: 1
            })
            expect(legalization!.legalizedLoweringUnits).toBe(0)
            expect(legalization!.materializedConversions).toBe(0)
            expect(legalization!.materializedConversionBytes).toBe(0)
            expect(program.handle.diagnostics.memory.externalBytes).toBe(
              (a.shape[0] * a.shape[1] + b.shape[0] * b.shape[1]) * 2
            )
          } else if (device === "cpu" || device === "cuda") {
            expect(program.handle.diagnostics.memory.workspaceBytes).toBeGreaterThan(0)
            expect(legalization!.legalizedLoweringUnits).toBeGreaterThan(0)
            expect(legalization!.materializedConversions).toBeGreaterThanOrEqual(3)
            expect(legalization!.materializedConversionBytes).toBeGreaterThan(0)
          }
          expect(yield* Tensor.toNumberArray(result)).toEqual([1, 5, 5, 1, 4, 11, 11, 4])
          yield* Tensor.clear(result)
        }
        yield* Tensor.clear(a)
        yield* Tensor.clear(b)
      }))

    it.effect(`${dtype} retains intermediate rounding in fused chains`, () =>
      Effect.gen(function*() {
        const large = dtype === "bf16" ? 256 : 2048
        const a = yield* halfTensor(dtype, [large, large / 2], [2])
        const b = yield* halfTensor(dtype, [1, 0.5], [2])
        const left = yield* Tensor.makeInput(0, a)
        const right = yield* Tensor.makeInput(1, b)
        const root = yield* Tensor.sub(yield* Tensor.add(left, right), left)
        for (const optimize of [false, true]) {
          const program = yield* Tensor.freezeProgram([root], { optimize })
          const [result] = yield* Tensor.runProgram(program, [a, b])
          expect(result.dtype).toBe(dtype)
          // Both additions land halfway between representable values and round
          // back to the even input. Keeping a wide intermediate would yield b.
          expect(yield* Tensor.toNumberArray(result)).toEqual([0, 0])
          yield* Tensor.clear(result)
        }
        yield* Tensor.clear(a)
        yield* Tensor.clear(b)
      }))

    it.effect(`${dtype} archives retain two-byte storage and special values`, () =>
      Effect.acquireUseRelease(
        Effect.sync(() => fs.mkdtempSync(path.join(os.tmpdir(), "effect-torch-dtype-"))),
        (directory) =>
          Effect.gen(function*() {
            const filename = path.join(directory, "weight.safetensors")
            const bits = dtype === "bf16"
              ? [0x3f80, 0x8000, 0x7f80, 0xff80, 0x7fd5, 0x0001]
              : [0x3c00, 0x8000, 0x7c00, 0xfc00, 0x7e35, 0x0001]
            yield* Effect.sync(() => fs.writeFileSync(filename, halfArchive(dtype === "bf16" ? "BF16" : "F16", bits)))
            const loaded = yield* Safetensors.load(filename)
            const weight = loaded.weight
            expect(weight.dtype).toBe(dtype)
            const values = yield* Tensor.toNumberArray(weight)
            expect(values[0]).toBe(1)
            expect(Object.is(values[1], -0)).toBe(true)
            expect(values[2]).toBe(Infinity)
            expect(values[3]).toBe(-Infinity)
            expect(Number.isNaN(values[4])).toBe(true)
            expect(values[5]).toBe(dtype === "bf16" ? 2 ** -133 : 2 ** -24)
            const program = yield* Tensor.freezeProgram([weight], { constantWeights: true })
            expect(program.handle.diagnostics.memory.persistentBytes).toBe(bits.length * 2)
            const saved = path.join(directory, "saved.safetensors")
            yield* Safetensors.save(saved, { weight })
            const output = yield* Effect.sync(() => fs.readFileSync(saved))
            const payload = output.subarray(8 + Number(output.readBigUInt64LE()))
            expect(Array.from({ length: bits.length }, (_, index) => payload.readUInt16LE(index * 2))).toEqual(bits)
            yield* Tensor.clear(weight)
            const [retained] = yield* Tensor.runProgram(program, [])
            expect(yield* Tensor.toNumberArray(retained)).toEqual(values)
            yield* Tensor.clear(retained)
          }),
        (directory) => Effect.sync(() => fs.rmSync(directory, { recursive: true, force: true }))
      ))
  }

  it.effect("dense constant memory matches each supported dtype's element size", () =>
    Effect.gen(function*() {
      const formats: ReadonlyArray<readonly [Tensor.DType, number]> = [
        ["f32", 4],
        ["f16", 2],
        ["bf16", 2],
        ["u8", 1],
        ["u32", 4],
        ["i64", 8],
        ...(device === "metal" ? [] : [["f64", 8] as const])
      ]
      for (const [dtype, bytes] of formats) {
        const root = yield* Tensor.zeros([17], { dtype })
        const program = yield* Tensor.freezeProgram([root])
        expect(program.handle.diagnostics.memory.persistentBytes).toBe(17 * bytes)
      }
    }))
})

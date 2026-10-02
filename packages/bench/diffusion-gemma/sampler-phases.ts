/** Isolated CUDA sampler geometry; timings are component costs, not model latency. */
import * as BackendCuda from "@effect-torch/backend-cuda"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import assert from "node:assert/strict"
import { performance } from "node:perf_hooks"

const shape = [1, 256, 262144]
const values = new Float32Array(shape.reduce((left, right) => left * right, 1))
for (let index = 0; index < values.length; index++) values[index] = (index % 997 - 498) / 16

const entropy = (processed: Tensor.Any) =>
  Effect.gen(function*() {
    const normalized = yield* Tensor.sub(processed, yield* Tensor.logsumexp(processed, { dims: [2], keepdims: true }))
    const clamped = yield* Tensor.clamp(normalized, { min: -3.4028234663852886e38 })
    const tokens = yield* Tensor.neg(
      yield* Tensor.sum(
        yield* Tensor.mul(clamped, yield* Tensor.softmax(normalized)),
        { dims: [2] }
      )
    )
    return [tokens, yield* Tensor.topKIndices(yield* Tensor.neg(tokens), 256), yield* Tensor.mean(tokens)]
  })

const sample = (processed: Tensor.Any) =>
  Effect.gen(function*() {
    const uniform = yield* Tensor.uniform(shape, { dtype: "f32" })
    const gumbel = yield* Tensor.neg(yield* Tensor.log(yield* Tensor.neg(yield* Tensor.log(uniform))))
    return [
      yield* Tensor.cast(yield* Tensor.argmax(yield* Tensor.add(processed, gumbel), 2), "u32"),
      yield* Tensor.cast(yield* Tensor.argmax(processed, 2), "u32")
    ]
  })

await Effect.runPromise(Effect.provide(
  Effect.scoped(Effect.gen(function*() {
    // Validate both the power-of-two row shift and ordinary row division
    // against host F32 arithmetic before measuring the large sampler canvas.
    for (const width of [7, 8]) {
      const input = Float32Array.from({ length: 6 * width }, (_, index) => (index - 17) / 32)
      const divisor = Float32Array.from({ length: 6 }, (_, index) => (index + 1) / 8)
      yield* Effect.scoped(Effect.gen(function*() {
        const operands = yield* Effect.acquireRelease(
          Tensor.compute([
            yield* Tensor.fromTypedArray(input, [2, 3, width]),
            yield* Tensor.fromTypedArray(divisor, [2, 3, 1])
          ]),
          Tensor.clearAll,
          { interruptible: true }
        )
        const output = yield* Effect.acquireRelease(
          Tensor.compute([yield* Tensor.div(operands[0]!, operands[1]!)]),
          Tensor.clearAll,
          { interruptible: true }
        )
        const actual = yield* Tensor.toTypedArray(output[0]!)
        for (let index = 0; index < input.length; index++) {
          assert.equal(actual[index], Math.fround(input[index]! / divisor[Math.floor(index / width)]!))
        }
      }))
    }
    for (const dtype of ["f16", "bf16"] as const) {
      yield* Effect.scoped(Effect.gen(function*() {
        const scalar = 1 + 2 ** (dtype === "f16" ? -11 : -8)
        const operands = yield* Effect.acquireRelease(
          Tensor.compute([
            yield* Tensor.cast(yield* Tensor.fromTypedArray(new Float32Array([3, -5]), [2]), dtype),
            yield* Tensor.full([], scalar)
          ]),
          Tensor.clearAll,
          { interruptible: true }
        )
        for (const reverse of [false, true]) {
          const output = yield* Effect.acquireRelease(
            Tensor.compute([
              yield* Tensor.cast(
                yield* Tensor.mul(operands[reverse ? 1 : 0]!, operands[reverse ? 0 : 1]!),
                "f32"
              )
            ]),
            Tensor.clearAll,
            { interruptible: true }
          )
          const actual = yield* Tensor.toTypedArray(output[0]!)
          assert(actual instanceof Float32Array)
          assert.deepEqual(Array.from(actual), [3, -5])
        }
      }))
    }
    process.stdout.write(JSON.stringify({ binaryBroadcastAndScalarCoercion: "passed" }) + "\n")
    const [logits, temperature, broadcastTemperature] = yield* Effect.acquireRelease(
      Tensor.compute([
        yield* Tensor.fromTypedArray(values, shape),
        yield* Tensor.full([], 0.8),
        yield* Tensor.full([1, 1, 1], 0.8)
      ]),
      Tensor.clearAll,
      { interruptible: true }
    )
    const stages = {
      uniformAndRowSum: ([input]: ReadonlyArray<Tensor.Any>) =>
        Effect.gen(function*() {
          return [yield* Tensor.sum(yield* Tensor.uniform(input!.shape, { dtype: "f32" }), { dims: [2] })]
        }),
      sampling: ([input]: ReadonlyArray<Tensor.Any>) => sample(input!),
      entropy: ([input]: ReadonlyArray<Tensor.Any>) => entropy(input!),
      full: ([input, temp]: ReadonlyArray<Tensor.Any>) =>
        Effect.gen(function*() {
          const processed = yield* Tensor.div(input!, temp!)
          return [yield* Tensor.cast(processed, "bf16"), ...yield* sample(processed), ...yield* entropy(processed)]
        })
    }
    for (const [stage, build] of Object.entries(stages)) {
      for (const geometry of stage === "full" ? ["scalar", "rank3"] : ["scalar"]) {
        const program = yield* Tensor.compile(build, { randomSeed: 42 })
        const inputs = stage === "full"
          ? [logits!, geometry === "scalar" ? temperature! : broadcastTemperature!]
          : [logits!]
        for (let run = -2; run < 5; run++) {
          const milliseconds = yield* Effect.scoped(Effect.gen(function*() {
            const started = performance.now()
            const outputs = yield* Effect.acquireRelease(program.call(inputs), Tensor.clearAll, { interruptible: true })
            // Reduced output readback waits for the entire program.
            yield* Tensor.toTypedArray(outputs[outputs.length - 1]!)
            return performance.now() - started
          }))
          if (run >= 0) {
            process.stdout.write(
              JSON.stringify({ stage, geometry, run, milliseconds, shape, syntheticLogits: true }) + "\n"
            )
          }
        }
        yield* program.clear
      }
    }
  })),
  BackendCuda.layer()
))

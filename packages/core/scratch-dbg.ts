import { Effect } from "effect"
import { Device, Tensor } from "./src/index.ts"
const program = Effect.gen(function* () {
  const x = yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3])
  const w = yield* Tensor.fromTypedArray(new Float32Array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]), [3, 4])
  const b = yield* Tensor.fromTypedArray(new Float32Array([10, 20, 30, 40]), [1, 4])
  const fused = yield* Tensor.linear(x, w, b)
  const manual = yield* Tensor.add(yield* Tensor.matmul(x, w), b)
  const [f, m] = yield* Tensor.compute([fused, manual])
  const fv = yield* Tensor.toNumberArray(f)
  const mv = yield* Tensor.toNumberArray(m)
  console.log("max diff:", fv.reduce((mx, v, i) => Math.max(mx, Math.abs(v - mv[i]!)), 0))
  console.log("fused:", fv.map((v) => v.toFixed(2)).join(" "))
})
Effect.runPromiseExit(program.pipe(Effect.provide(Device.Metal))).then((e) => {
  if (e._tag !== "Success") console.log(e.cause.toString())
  process.exit(0)
})

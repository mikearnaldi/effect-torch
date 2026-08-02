import { Effect } from "effect"
import { Device, Gradient, Tensor } from "./src/index.ts"
const floats = (v: Array<number>) => new Float32Array(v)
const build = Effect.gen(function* () {
  const x = yield* Tensor.fromTypedArray(
    floats([1, -2, 3, -4, 0.5, -0.5, 2, -1, 0.1, 0.2, 0.3, 0.4, -3, 2.5, 1.5, -0.7]),
    [4, 4]
  )
  const m = yield* Tensor.max(x, { dims: [1], keepdims: true })
  const y = yield* Tensor.tanh(yield* Tensor.exp(yield* Tensor.sub(x, m)))
  const [gx] = yield* Gradient.grad(yield* Tensor.sum(y), [x])
  return yield* Tensor.toNumberArray(gx)
})
const program = Effect.gen(function* () {
  const prev = process.env.EFFECT_TORCH_NO_FUSION
  delete process.env.EFFECT_TORCH_NO_FUSION
  const fused = yield* build.pipe(Effect.provide(Device.Cpu))
  process.env.EFFECT_TORCH_NO_FUSION = "1"
  const unfused = yield* build.pipe(Effect.provide(Device.Cpu))
  if (prev === undefined) delete process.env.EFFECT_TORCH_NO_FUSION
  else process.env.EFFECT_TORCH_NO_FUSION = prev
  fused.forEach((v, i) => { if (Math.abs(v - unfused[i]!) > 1e-4) console.log(`[${i}] fused=${v.toFixed(5)} unfused=${unfused[i]!.toFixed(5)}`) })
})
Effect.runPromiseExit(program).then(() => process.exit(0))

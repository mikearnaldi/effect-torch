import { Effect } from "effect"
import { Device, Model, Tensor } from "./src/index.ts"

const run = Effect.gen(function* () {
  const x = yield* Tensor.fromTypedArray(new Float32Array(Array.from({ length: 32 }, (_, i) => Math.sin(i) )), [1, 4, 8])
  const attn = yield* Model.multiHeadAttention("attn", 8, 2, { causal: true, rope: 10000 })
  const params = yield* Tensor.compute(yield* attn.init)
  const out = yield* attn.forward(params, x)
  return yield* Tensor.toNumberArray((yield* Tensor.compute([out]))[0])
})

const program = Effect.gen(function* () {
  const cpu = yield* run.pipe(Effect.provide(Device.Cpu))
  const metal = yield* run.pipe(Effect.provide(Device.Metal))
  const maxDiff = cpu.reduce((m, v, i) => Math.max(m, Math.abs(v - metal[i]!)), 0)
  console.log("max |cpu - metal|:", maxDiff)
  console.log("cpu[0..4]:", cpu.slice(0, 4).map((v) => v.toFixed(4)).join(" "))
  console.log("met[0..4]:", metal.slice(0, 4).map((v) => v.toFixed(4)).join(" "))
})
Effect.runPromiseExit(program).then((e) => {
  if (e._tag !== "Success") console.log(e.cause.toString())
  process.exit(0)
})

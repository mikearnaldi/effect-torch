import { describe, expect, layer } from "@effect/vitest"
import { Effect } from "effect"
import { Device, Loss, Optimizer, Tensor } from "../src/index.ts"

const f64 = (data: ReadonlyArray<number>, shape?: ReadonlyArray<number>) =>
  Tensor.fromTypedArray(new Float64Array(data), shape)

const values = (t: Tensor.GenericTensor) => Tensor.toNumberArray(t)

const scalar = (t: Tensor.GenericTensor) => Effect.map(values(t), (v) => v[0])

const runStep = <S>(
  optimizer: Optimizer.Optimizer<S>,
  params: ReadonlyArray<Tensor.GenericTensor>,
  grads: ReadonlyArray<Tensor.GenericTensor>,
  state: S
) =>
  Effect.gen(function* () {
    const next = yield* optimizer.step(params, grads, state)
    const evaluated = yield* Tensor.evaluate([...next.params, ...next.stateRoots])
    return {
      params: evaluated.slice(0, params.length),
      state: next.rebuildState(evaluated.slice(params.length))
    }
  })

const closeTo = (actual: Array<number>, expected: ReadonlyArray<number>, tol = 1e-9) => {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < actual.length; i++) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThan(tol)
  }
}

layer(Device.Cpu)("Optimizer", (it) => {
  describe("sgd", () => {
    it.effect("plain update matches hand computation", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.sgd({ lr: 0.1 })
        const p = yield* f64([1, 2])
        const g = yield* f64([0.5, -0.5])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state)
        closeTo(yield* values(step1.params[0]), [0.95, 2.05])
        expect(step1.state.velocity).toBeNull()
      })
    )

    it.effect("momentum recurrence matches hand computation over 3 steps", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.sgd({ lr: 0.1, momentum: 0.9 })
        const g = yield* f64([0.5, -0.5])
        let params = [yield* f64([1, 2])] as ReadonlyArray<Tensor.GenericTensor>
        let state = yield* optimizer.init(params)
        const expected = [
          [0.95, 2.05],
          [0.855, 2.145],
          [0.7195, 2.2805]
        ]
        for (const wanted of expected) {
          const next = yield* runStep(optimizer, params, [g], state)
          closeTo(yield* values(next.params[0]), wanted)
          params = next.params
          state = next.state
        }
      })
    )

    it.effect("weight decay adds coupled L2 to the gradient", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.sgd({ lr: 0.1, weightDecay: 1 })
        const p = yield* f64([1, 2])
        const g = yield* f64([0.5, -0.5])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state)
        closeTo(yield* values(step1.params[0]), [0.85, 1.85])
      })
    )

    it.effect("nesterov uses the lookahead velocity", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.sgd({ lr: 0.1, momentum: 0.9, nesterov: true })
        const p = yield* f64([1])
        const g = yield* f64([0.5])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state)
        closeTo(yield* values(step1.params[0]), [0.905])
      })
    )
  })

  describe("adam", () => {
    it.effect("first step matches the reference formula", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.adam({ lr: 0.1 })
        const p = yield* f64([1, -1])
        const g = yield* f64([0.1, 0.2])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state)
        closeTo(yield* values(step1.params[0]), [0.90000001, -1.099999995], 1e-6)
        expect(step1.state.t).toBe(1)
      })
    )

    it.effect("bias correction uses the step count", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.adam({ lr: 0.1 })
        const g = yield* f64([0.1])
        let params = [yield* f64([1])] as ReadonlyArray<Tensor.GenericTensor>
        let state = yield* optimizer.init(params)
        for (let i = 0; i < 2; i++) {
          const next = yield* runStep(optimizer, params, [g], state)
          params = next.params
          state = next.state
        }
        const m2 = 0.9 * 0.01 + 0.1 * 0.1
        const v2 = 0.999 * 0.00001 + 0.001 * 0.01
        const mHat = m2 / (1 - 0.9 * 0.9)
        const vHat = v2 / (1 - 0.999 * 0.999)
        const expected = 0.90000001 - (0.1 * mHat) / (Math.sqrt(vHat) + 1e-8)
        closeTo(yield* values(params[0]), [expected], 1e-6)
      })
    )

    it.effect("zero gradients decay the moments while the parameter follows m_hat", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.adam({ lr: 0.1 })
        const p = yield* f64([1])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [yield* f64([0.5])], state)
        const step2 = yield* runStep(optimizer, step1.params, [yield* f64([0])], step1.state)
        const mHat = (0.9 * 0.05) / (1 - 0.81)
        const vHat = (0.999 * 0.00025) / (1 - 0.998001)
        const expected = 0.90000001 - (0.1 * mHat) / (Math.sqrt(vHat) + 1e-8)
        closeTo(yield* values(step2.params[0]), [expected], 1e-6)
        closeTo(yield* values(step2.state.m[0]), [0.9 * 0.05])
        closeTo(yield* values(step2.state.v[0]), [0.999 * 0.00025])
      })
    )
  })

  describe("adamW", () => {
    it.effect("applies decoupled weight decay", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.adamW({ lr: 0.1, weightDecay: 0.01 })
        const p = yield* f64([1, -1])
        const g = yield* f64([0.1, 0.2])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state)
        closeTo(yield* values(step1.params[0]), [0.89900001, -1.098999995], 1e-6)
      })
    )

    it.effect("weight decay scales with lr", () =>
      Effect.gen(function* () {
        const plain = Optimizer.adam({ lr: 0.1 })
        const decayed = Optimizer.adamW({ lr: 0.1, weightDecay: 1 })
        const p1 = yield* f64([1])
        const p2 = yield* f64([1])
        const g1 = yield* f64([0.5])
        const g2 = yield* f64([0.5])
        const s1 = yield* plain.init([p1])
        const s2 = yield* decayed.init([p2])
        const r1 = yield* runStep(plain, [p1], [g1], s1)
        const r2 = yield* runStep(decayed, [p2], [g2], s2)
        const [a] = yield* values(r1.params[0])
        const [b] = yield* values(r2.params[0])
        expect(Math.abs(a - b - 0.1)).toBeLessThan(1e-9)
      })
    )
  })

  describe("step", () => {
    it.effect("fits a linear model and drives the loss down", () =>
      Effect.gen(function* () {
        const x = yield* f64([1, 2, 2, 5, 4, 3, 5, 8], [4, 2])
        const y = yield* f64([7, 18, 16, 33], [4, 1])
        const optimizer = Optimizer.adam({ lr: 0.1 })
        const lossOf = (w: Tensor.GenericTensor, b: Tensor.GenericTensor) =>
          Effect.gen(function* () {
            const pred = yield* Tensor.add(yield* Tensor.matmul(x, w), b)
            return yield* Loss.mse(pred, y)
          })
        let params: ReadonlyArray<Tensor.GenericTensor> = [yield* f64([0, 0], [2, 1]), yield* f64([0], [1, 1])]
        let state = yield* optimizer.init(params)
        let first = 0
        let last = 0
        for (let i = 0; i < 1000; i++) {
          const loss = yield* lossOf(params[0], params[1])
          const result = yield* Optimizer.step(optimizer, loss, params, state)
          const value = yield* scalar(result.loss)
          if (i === 0) first = value
          last = value
          params = result.params
          state = result.state
        }
        expect(last).toBeLessThan(first * 1e-4)
        closeTo(yield* values(params[0]), [2, 3], 5e-2)
        closeTo(yield* values(params[1]), [-1], 5e-2)
      })
    )

    it.effect("the joint walk computes the same loss as a loss-only walk", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.sgd({ lr: 0.01 })
        const p = yield* f64([1, 2, 3])
        const state = yield* optimizer.init([p])
        const loss = yield* Tensor.sum(yield* Tensor.mul(p, p))
        const result = yield* Optimizer.step(optimizer, loss, [p], state)
        const jointLoss = yield* scalar(result.loss)
        const aloneLoss = yield* scalar(loss)
        expect(jointLoss).toBe(aloneLoss)
      })
    )

    it.effect("returned params and state tensors are materialized leaves, across steps", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.adam({ lr: 0.1 })
        const p = yield* f64([1, 2])
        let params: ReadonlyArray<Tensor.GenericTensor> = [p]
        let state = yield* optimizer.init(params)
        for (let i = 0; i < 3; i++) {
          const loss = yield* Tensor.sum(yield* Tensor.mul(params[0], params[0]))
          const result = yield* Optimizer.step(optimizer, loss, params, state)
          for (const param of result.params) {
            expect(Tensor.isTensor(param)).toBe(true)
          }
          for (const t of [...result.state.m, ...result.state.v]) {
            expect(Tensor.isTensor(t)).toBe(true)
          }
          params = result.params
          state = result.state
        }
      })
    )
  })

  describe("user-land optimizers", () => {
    it.effect("a custom optimizer with tensor state implements the same contract", () =>
      Effect.gen(function* () {
        interface AvgState {
          readonly prev: ReadonlyArray<Tensor.GenericTensor>
        }
        const avgGradSgd = (lr: number): Optimizer.Optimizer<AvgState> => ({
          init: (params) =>
            Effect.map(
              Effect.forEach(params, (p) => Tensor.zeros(p.shape, { dtype: p.dtype })),
              (prev): AvgState => ({ prev })
            ),
          step: (params, grads, state) =>
            Effect.gen(function* () {
              const updates: Array<Tensor.LazyTensor> = []
              const used: Array<Tensor.LazyTensor> = []
              for (let i = 0; i < params.length; i++) {
                const g = yield* Tensor.mul(yield* Tensor.add(grads[i], state.prev[i]), 0.5)
                updates.push(yield* Tensor.sub(params[i], yield* Tensor.mul(g, lr)))
                used.push(g)
              }
              return {
                params: updates,
                state: { prev: used },
                stateRoots: used,
                rebuildState: (evaluated): AvgState => ({ prev: [...evaluated] })
              }
            })
        })

        const optimizer = avgGradSgd(0.1)
        const g = yield* f64([0.5])
        let params: ReadonlyArray<Tensor.GenericTensor> = [yield* f64([1])]
        let state = yield* optimizer.init(params)
        const expected = [0.975, 0.9375]
        for (const wanted of expected) {
          const next = yield* runStep(optimizer, params, [g], state)
          closeTo(yield* values(next.params[0]), [wanted])
          params = next.params
          state = next.state
        }
      })
    )
  })

  describe("validation", () => {
    it.effect("rejects non-float parameters at init", () =>
      Effect.gen(function* () {
        const p = yield* Tensor.fromTypedArray(new BigInt64Array([1n]))
        const optimizer = Optimizer.sgd({ lr: 0.1 })
        const error = yield* Effect.flip(optimizer.init([p]))
        expect(error.message).toContain("f32 or f64")
      })
    )

    it.effect("rejects mismatched params/grads lengths", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.sgd({ lr: 0.1 })
        const p = yield* f64([1])
        const g = yield* f64([1])
        const state = yield* optimizer.init([p])
        const error = yield* Effect.flip(optimizer.step([p, p], [g], state))
        expect(error.message).toContain("expected 2 gradients, got 1")
      })
    )

    it.effect("rejects state built for different parameters", () =>
      Effect.gen(function* () {
        const optimizer = Optimizer.adam()
        const a = yield* f64([1])
        const b = yield* f64([1, 2])
        const g = yield* f64([0.5, 0.5])
        const state = yield* optimizer.init([a])
        const error = yield* Effect.flip(optimizer.step([b], [g], state))
        expect(error.message).toContain("use init for these parameters")
      })
    )

    it.effect("rejects invalid configuration", () =>
      Effect.sync(() => {
        expect(() => Optimizer.sgd({ lr: 0 })).toThrow("lr must be positive")
        expect(() => Optimizer.sgd({ lr: 0.1, momentum: 0.9, nesterov: true, dampening: 0.1 })).toThrow(
          "nesterov"
        )
        expect(() => Optimizer.adam({ beta1: 1.5 })).toThrow("beta1 and beta2")
        expect(() => Optimizer.adam({ eps: 0 })).toThrow("eps must be positive")
      })
    )
  })
})

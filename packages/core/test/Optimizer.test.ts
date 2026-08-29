import { describe, expect } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Gradient, LearningRate, Loss, Optimizer, Tensor } from "../src/index.ts"
import { floats, onDevices, TOL } from "./utils/devices.ts"

const values = (t: Tensor.Any) => Tensor.toNumberArray(t)

const scalar = (t: Tensor.Any) => Effect.map(values(t), (v) => v[0])

// Optimizers return lazy params plus ordered state roots. Materializing them
// together executes shared work once and rebuilds state with concrete roots.
const runStep = <S>(
  optimizer: Optimizer.Optimizer<S>,
  params: ReadonlyArray<Tensor.Any>,
  grads: ReadonlyArray<Tensor.Any>,
  state: S,
  lr: number
) =>
  Effect.gen(function*() {
    const lrT = yield* Tensor.full([], lr, { dtype: params[0].dtype })
    const next = yield* optimizer.step(params, grads, state, lrT)
    const evaluated = yield* Tensor.compute([...next.params, ...next.stateRoots])
    return {
      params: evaluated.slice(0, params.length),
      state: optimizer.rebuildState(next.state, evaluated.slice(params.length))
    }
  })

onDevices("Optimizer", () => (it) => {
  const f32 = (data: ReadonlyArray<number>, shape?: ReadonlyArray<number>) => Tensor.fromTypedArray(floats(data), shape)
  const closeTo = (actual: Array<number>, expected: ReadonlyArray<number>, t = TOL) => {
    expect(actual.length).toBe(expected.length)
    for (let i = 0; i < actual.length; i++) {
      expect(Math.abs(actual[i] - expected[i])).toBeLessThan(t)
    }
  }
  describe("sgd", () => {
    it.effect("plain update matches hand computation", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.sgd()
        const p = yield* f32([1, 2])
        const g = yield* f32([0.5, -0.5])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state, 0.1)
        closeTo(yield* values(step1.params[0]), [0.95, 2.05])
        expect(step1.state.velocity).toEqual([])
      }))

    it.effect("momentum recurrence matches hand computation over 3 steps", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.sgd({ momentum: 0.9 })
        const g = yield* f32([0.5, -0.5])
        let params: ReadonlyArray<Tensor.Any> = [yield* f32([1, 2])]
        let state = yield* optimizer.init(params)
        const expected = [
          [0.95, 2.05],
          [0.855, 2.145],
          [0.7195, 2.2805]
        ]
        for (const wanted of expected) {
          const next = yield* runStep(optimizer, params, [g], state, 0.1)
          closeTo(yield* values(next.params[0]), wanted)
          params = next.params
          state = next.state
        }
      }))

    it.effect("weight decay adds coupled L2 to the gradient", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.sgd({ weightDecay: 1 })
        const p = yield* f32([1, 2])
        const g = yield* f32([0.5, -0.5])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state, 0.1)
        closeTo(yield* values(step1.params[0]), [0.85, 1.85])
      }))

    it.effect("nesterov uses the lookahead velocity", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.sgd({ momentum: 0.9, nesterov: true })
        const p = yield* f32([1])
        const g = yield* f32([0.5])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state, 0.1)
        closeTo(yield* values(step1.params[0]), [0.905])
      }))
  })

  describe("adam", () => {
    it.effect("first step matches the reference formula", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.adam()
        const p = yield* f32([1, -1])
        const g = yield* f32([0.1, 0.2])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state, 0.1)
        closeTo(yield* values(step1.params[0]), [0.90000001, -1.099999995], 1e-5)
        expect(yield* scalar(step1.state.t)).toBe(1)
      }))

    it.effect("bias correction uses the step count", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.adam()
        const g = yield* f32([0.1])
        let params: ReadonlyArray<Tensor.Any> = [yield* f32([1])]
        let state = yield* optimizer.init(params)
        for (let i = 0; i < 2; i++) {
          const next = yield* runStep(optimizer, params, [g], state, 0.1)
          params = next.params
          state = next.state
        }
        const m2 = 0.9 * 0.01 + 0.1 * 0.1
        const v2 = 0.999 * 0.00001 + 0.001 * 0.01
        const mHat = m2 / (1 - 0.9 * 0.9)
        const vHat = v2 / (1 - 0.999 * 0.999)
        const expected = 0.90000001 - (0.1 * mHat) / (Math.sqrt(vHat) + 1e-8)
        closeTo(yield* values(params[0]), [expected], 1e-5)
      }))

    it.effect("zero gradients decay the moments while the parameter follows m_hat", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.adam()
        const p = yield* f32([1])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [yield* f32([0.5])], state, 0.1)
        const step2 = yield* runStep(optimizer, step1.params, [yield* f32([0])], step1.state, 0.1)
        const mHat = (0.9 * 0.05) / (1 - 0.81)
        const vHat = (0.999 * 0.00025) / (1 - 0.998001)
        const expected = 0.90000001 - (0.1 * mHat) / (Math.sqrt(vHat) + 1e-8)
        closeTo(yield* values(step2.params[0]), [expected], 1e-5)
        closeTo(yield* values(step2.state.m[0]), [0.9 * 0.05])
        closeTo(yield* values(step2.state.v[0]), [0.999 * 0.00025])
      }))
  })

  describe("adamW", () => {
    it.effect("applies decoupled weight decay", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.adamW({ weightDecay: 0.01 })
        const p = yield* f32([1, -1])
        const g = yield* f32([0.1, 0.2])
        const state = yield* optimizer.init([p])
        const step1 = yield* runStep(optimizer, [p], [g], state, 0.1)
        closeTo(yield* values(step1.params[0]), [0.89900001, -1.098999995], 1e-5)
      }))

    it.effect("weight decay scales with lr", () =>
      Effect.gen(function*() {
        const plain = yield* Optimizer.adam()
        const decayed = yield* Optimizer.adamW({ weightDecay: 1 })
        const p1 = yield* f32([1])
        const p2 = yield* f32([1])
        const g1 = yield* f32([0.5])
        const g2 = yield* f32([0.5])
        const s1 = yield* plain.init([p1])
        const s2 = yield* decayed.init([p2])
        const r1 = yield* runStep(plain, [p1], [g1], s1, 0.1)
        const r2 = yield* runStep(decayed, [p2], [g2], s2, 0.1)
        const [a] = yield* values(r1.params[0])
        const [b] = yield* values(r2.params[0])
        expect(Math.abs(a - b - 0.1)).toBeLessThan(TOL)
      }))
  })

  describe("step", () => {
    it.effect("fits a linear model and drives the loss down", () =>
      Effect.gen(function*() {
        const x = yield* f32([1, 2, 2, 5, 4, 3, 5, 8], [4, 2])
        const y = yield* f32([7, 18, 16, 33], [4, 1])
        const optimizer = yield* Optimizer.adam()
        const lossOf = (w: Tensor.Any, b: Tensor.Any) =>
          Effect.gen(function*() {
            const pred = yield* Tensor.add(yield* Tensor.matmul(x, w), b)
            return yield* Loss.mse(pred, y)
          })
        let params: ReadonlyArray<Tensor.Any> = [yield* f32([0, 0], [2, 1]), yield* f32([0], [1, 1])]
        let state = yield* optimizer.init(params)
        const lr = yield* Tensor.full([], 0.1, { dtype: "f32" })
        let first = 0
        let last = 0
        for (let i = 0; i < 1000; i++) {
          const loss = yield* lossOf(params[0], params[1])
          const result = yield* Optimizer.step(optimizer, loss, params, state, lr)
          const value = yield* scalar(result.loss)
          if (i === 0) first = value
          last = value
          params = result.params
          state = result.state
        }
        expect(last).toBeLessThan(first * 1e-4)
        closeTo(yield* values(params[0]), [2, 3], 5e-2)
        closeTo(yield* values(params[1]), [-1], 5e-2)
      }))

    it.effect("the joint walk computes the same loss as a loss-only walk", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.sgd()
        const p = yield* f32([1, 2, 3])
        const state = yield* optimizer.init([p])
        const loss = yield* Tensor.sum(yield* Tensor.mul(p, p))
        const lr = yield* Tensor.full([], 0.01, { dtype: "f32" })
        const result = yield* Optimizer.step(optimizer, loss, [p], state, lr)
        const jointLoss = yield* scalar(result.loss)
        const aloneLoss = yield* scalar(loss)
        expect(jointLoss).toBe(aloneLoss)
      }))

    it.effect("returned params and state tensors are materialized leaves, across steps", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.adam()
        const p = yield* f32([1, 2])
        let params: ReadonlyArray<Tensor.Any> = [p]
        let state = yield* optimizer.init(params)
        for (let i = 0; i < 3; i++) {
          const loss = yield* Tensor.sum(yield* Tensor.mul(params[0], params[0]))
          const lr = yield* Tensor.full([], 0.1, { dtype: "f32" })
          const result = yield* Optimizer.step(optimizer, loss, params, state, lr)
          for (const param of result.params) {
            expect(Tensor.isTensor(param)).toBe(true)
          }
          for (const t of [...result.state.m, ...result.state.v, result.state.t]) {
            expect(Tensor.isTensor(t)).toBe(true)
          }
          params = result.params
          state = result.state
        }
      }))
  })

  describe("custom optimizers", () => {
    it.effect("a custom optimizer with tensor state implements the same contract", () =>
      Effect.gen(function*() {
        interface AvgState {
          readonly prev: ReadonlyArray<Tensor.Any>
        }
        const avgGradSgd = (): Optimizer.Optimizer<AvgState> => ({
          init: (params) =>
            Effect.map(
              Effect.forEach(params, Tensor.zerosLike),
              (prev): AvgState => ({ prev })
            ),
          step: (params, grads, state, lr) =>
            Effect.gen(function*() {
              const updates: Array<Tensor.Lazy> = []
              const used: Array<Tensor.Lazy> = []
              for (let i = 0; i < params.length; i++) {
                const g = yield* Tensor.mul(
                  yield* Tensor.add(grads[i], state.prev[i]),
                  yield* Tensor.constantLike(grads[i], 0.5)
                )
                updates.push(yield* Tensor.sub(params[i], yield* Tensor.mul(g, lr)))
                used.push(g)
              }
              return {
                params: updates,
                state: { prev: used },
                stateRoots: used
              }
            }),
          stateRoots: (state) => state.prev,
          rebuildState: (_, roots): AvgState => ({ prev: [...roots] })
        })

        const optimizer = avgGradSgd()
        const g = yield* f32([0.5])
        let params: ReadonlyArray<Tensor.Any> = [yield* f32([1])]
        let state = yield* optimizer.init(params)
        const expected = [0.975, 0.9375]
        for (const wanted of expected) {
          const next = yield* runStep(optimizer, params, [g], state, 0.1)
          closeTo(yield* values(next.params[0]), [wanted])
          params = next.params
          state = next.state
        }
      }))
  })

  describe("validation", () => {
    it.effect("rejects non-float parameters at init", () =>
      Effect.gen(function*() {
        const p = yield* Tensor.fromTypedArray(new BigInt64Array([1n]))
        const optimizer = yield* Optimizer.sgd()
        const error = yield* Effect.flip(optimizer.init([p]))
        expect(error.message).toContain("f32 or f64")
      }))

    it.effect("rejects mismatched params/grads lengths", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.sgd()
        const p = yield* f32([1])
        const g = yield* f32([1])
        const state = yield* optimizer.init([p])
        const lr = yield* Tensor.full([], 0.1, { dtype: "f32" })
        const error = yield* Effect.flip(optimizer.step([p, p], [g], state, lr))
        expect(error.message).toContain("expected 2 gradients, got 1")
      }))

    it.effect("rejects state built for different parameters", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.adam()
        const a = yield* f32([1])
        const b = yield* f32([1, 2])
        const g = yield* f32([0.5, 0.5])
        const state = yield* optimizer.init([a])
        const lr = yield* Tensor.full([], 0.1, { dtype: "f32" })
        const error = yield* Effect.flip(optimizer.step([b], [g], state, lr))
        expect(error.message).toContain("use init for these parameters")
      }))

    it.effect("rejects invalid configuration", () =>
      Effect.sync(() => {
        expect(() => Optimizer.sgd({ momentum: Number.NaN })).toThrow("momentum")
        expect(() => Optimizer.sgd({ momentum: 0.9, nesterov: true, dampening: 0.1 })).toThrow(
          "nesterov"
        )
        expect(() => Optimizer.adam({ beta1: 1.5 })).toThrow("beta1 and beta2")
        expect(() => Optimizer.adam({ beta1: Number.NaN })).toThrow("beta1 and beta2")
        expect(() => Optimizer.adam({ beta2: Number.NaN })).toThrow("beta1 and beta2")
        expect(() => Optimizer.adam({ eps: 0 })).toThrow("eps must be positive")
        expect(() => Optimizer.adam({ eps: Number.NaN })).toThrow("eps must be positive")
      }))

    it.effect("rejects numerically unstable f32 bias correction", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.adam({ beta2: 0.9999999 })
        const p = yield* f32([1])
        const error = yield* Effect.flip(optimizer.init([p]))
        expect(error.message).toContain("bias-correction error below 1%")
      }))

    it.effect("rejects a non-scalar learning rate at step", () =>
      Effect.gen(function*() {
        const optimizer = yield* Optimizer.adam()
        const p = yield* f32([1])
        const g = yield* f32([0.5])
        const state = yield* optimizer.init([p])
        const lr = yield* Tensor.full([1], 0.1, { dtype: "f32" })
        const error = yield* Effect.flip(optimizer.step([p], [g], state, lr))
        expect(error.message).toContain("0-d float tensor")
      }))

    it.effect("sgd matches a reference update composed from tensor ops", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 1, 2, 1, 3, 1, 4, 1]), [4, 2])
        const y = yield* Tensor.fromTypedArray(floats([2, 3, 4, 5]), [4, 1])
        const lr = 0.05
        const momentum = 0.9
        const run = (config: { dampening?: number; nesterov?: boolean; weightDecay?: number }) =>
          Effect.gen(function*() {
            let params: ReadonlyArray<Tensor.Any> = [
              yield* Tensor.fromTypedArray(floats([0.5, -0.5]))
            ]
            const optimizer = yield* Optimizer.sgd({ momentum, ...config })
            let state = yield* optimizer.init(params)
            const lrT = yield* Tensor.full([], lr, { dtype: "f32" })
            for (let i = 0; i < 20; i++) {
              const pred = yield* Tensor.matmul(x, yield* Tensor.reshape(params[0], [2, 1]))
              const loss = yield* Loss.mse(pred, y)
              const next = yield* Optimizer.step(optimizer, loss, params, state, lrT)
              params = next.params
              state = next.state
            }
            return {
              w: yield* values(params[0]),
              velocity: yield* values(state.velocity[0])
            }
          })
        const reference = (config: { dampening?: number; nesterov?: boolean; weightDecay?: number }) =>
          Effect.gen(function*() {
            const dampening = config.dampening ?? 0
            const nesterov = config.nesterov ?? false
            const weightDecay = config.weightDecay ?? 0
            let params: ReadonlyArray<Tensor.Any> = [
              yield* Tensor.fromTypedArray(floats([0.5, -0.5]))
            ]
            let velocity: Tensor.Any | null = null
            for (let i = 0; i < 20; i++) {
              const pred = yield* Tensor.matmul(x, yield* Tensor.reshape(params[0], [2, 1]))
              const loss = yield* Loss.mse(pred, y)
              const [grad] = yield* Gradient.grad(loss, params)
              let g: Tensor.Any = grad
              if (weightDecay !== 0) {
                g = yield* Tensor.add(
                  g,
                  yield* Tensor.mul(params[0], yield* Tensor.constantLike(params[0], weightDecay))
                )
              }
              const nextVelocity: Tensor.Any = velocity === null
                ? g
                : yield* Tensor.add(
                  yield* Tensor.mul(velocity, yield* Tensor.constantLike(velocity, momentum)),
                  yield* Tensor.mul(g, yield* Tensor.constantLike(g, 1 - dampening))
                )
              const used: Tensor.Any = nesterov
                ? yield* Tensor.add(
                  g,
                  yield* Tensor.mul(nextVelocity, yield* Tensor.constantLike(nextVelocity, momentum))
                )
                : nextVelocity
              const [nextParam, v] = yield* Tensor.compute([
                yield* Tensor.sub(params[0], yield* Tensor.mul(used, yield* Tensor.constantLike(used, lr))),
                nextVelocity
              ])
              params = [nextParam]
              velocity = v
            }
            return {
              w: yield* values(params[0]),
              velocity: yield* values(velocity!)
            }
          })
        for (
          const config of [
            {},
            { dampening: 0.3 },
            { nesterov: true },
            { weightDecay: 0.01 },
            { dampening: 0.1, weightDecay: 0.01 }
          ]
        ) {
          const optimizerRun = yield* run(config)
          const referenceRun = yield* reference(config)
          for (let i = 0; i < 2; i++) {
            expect(Math.abs(optimizerRun.w[i] - referenceRun.w[i])).toBeLessThan(TOL)
            expect(Math.abs(optimizerRun.velocity[i] - referenceRun.velocity[i])).toBeLessThan(TOL)
          }
        }
      }))

    it.effect("adamW matches a reference update composed from tensor ops", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(floats([1, 1, 2, 1, 3, 1, 4, 1]), [4, 2])
        const y = yield* Tensor.fromTypedArray(floats([2, 3, 4, 5]), [4, 1])
        const run = () =>
          Effect.gen(function*() {
            let params: ReadonlyArray<Tensor.Any> = [
              yield* Tensor.fromTypedArray(floats([0, 0]))
            ]
            const optimizer = yield* Optimizer.adamW()
            let state = yield* optimizer.init(params)
            const lr = yield* Tensor.full([], 0.05, { dtype: "f32" })
            for (let i = 0; i < 20; i++) {
              const pred = yield* Tensor.matmul(x, yield* Tensor.reshape(params[0], [2, 1]))
              const loss = yield* Loss.mse(pred, y)
              const next = yield* Optimizer.step(optimizer, loss, params, state, lr)
              params = next.params
              state = next.state
            }
            return {
              w: yield* values(params[0]),
              m: yield* values(state.m[0]),
              v: yield* values(state.v[0])
            }
          })
        const reference = () =>
          Effect.gen(function*() {
            const lr = 0.05
            const beta1 = 0.9
            const beta2 = 0.999
            const eps = 1e-8
            const weightDecay = 0.01
            let params: ReadonlyArray<Tensor.Any> = [
              yield* Tensor.fromTypedArray(floats([0, 0]))
            ]
            let m: Tensor.Any = yield* Tensor.zeros([2])
            let v: Tensor.Any = yield* Tensor.zeros([2])
            for (let t = 1; t <= 20; t++) {
              const pred = yield* Tensor.matmul(x, yield* Tensor.reshape(params[0], [2, 1]))
              const loss = yield* Loss.mse(pred, y)
              const [grad] = yield* Gradient.grad(loss, params)
              const nextM = yield* Tensor.add(
                yield* Tensor.mul(m, yield* Tensor.constantLike(m, beta1)),
                yield* Tensor.mul(grad, yield* Tensor.constantLike(grad, 1 - beta1))
              )
              const nextV = yield* Tensor.add(
                yield* Tensor.mul(v, yield* Tensor.constantLike(v, beta2)),
                yield* Tensor.mul(yield* Tensor.mul(grad, grad), yield* Tensor.constantLike(grad, 1 - beta2))
              )
              const mHat = yield* Tensor.mul(nextM, yield* Tensor.constantLike(nextM, 1 / (1 - Math.pow(beta1, t))))
              const vHat = yield* Tensor.mul(nextV, yield* Tensor.constantLike(nextV, 1 / (1 - Math.pow(beta2, t))))
              const denom = yield* Tensor.add(yield* Tensor.sqrt(vHat), yield* Tensor.constantLike(vHat, eps))
              const adjusted = yield* Tensor.mul(yield* Tensor.div(mHat, denom), yield* Tensor.constantLike(mHat, lr))
              const base = yield* Tensor.mul(params[0], yield* Tensor.constantLike(params[0], 1 - lr * weightDecay))
              const [nextParam, mOut, vOut] = yield* Tensor.compute([
                yield* Tensor.sub(base, adjusted),
                nextM,
                nextV
              ])
              params = [nextParam]
              m = mOut
              v = vOut
            }
            return {
              w: yield* values(params[0]),
              m: yield* values(m),
              v: yield* values(v)
            }
          })
        const optimizerRun = yield* run()
        const referenceRun = yield* reference()
        for (let i = 0; i < 2; i++) {
          expect(Math.abs(optimizerRun.w[i] - referenceRun.w[i])).toBeLessThan(TOL)
          expect(Math.abs(optimizerRun.m[i] - referenceRun.m[i])).toBeLessThan(TOL)
          expect(Math.abs(optimizerRun.v[i] - referenceRun.v[i])).toBeLessThan(TOL)
        }
      }))
  })

  describe("gradient clipping", () => {
    it.effect("clipByValue clamps elementwise", () =>
      Effect.gen(function*() {
        const g = yield* Tensor.fromTypedArray(floats([-5, 0.5, 10]))
        const [clipped] = yield* Optimizer.clipByValue([g], { min: -1, max: 1 })
        assert.deepStrictEqual(yield* values(clipped), [-1, 0.5, 1])
      }))

    it.effect("clipByGlobalNorm scales down only above the norm", () =>
      Effect.gen(function*() {
        const g1 = yield* Tensor.fromTypedArray(floats([3, 4]))
        const g2 = yield* Tensor.fromTypedArray(floats([0, 12]))
        // total norm = 13, maxNorm 6.5 -> scale 0.5
        const [c1, c2] = yield* Optimizer.clipByGlobalNorm([g1, g2], 6.5)
        const v1 = yield* values(c1)
        const v2 = yield* values(c2)
        for (let i = 0; i < 2; i++) {
          expect(Math.abs(v1[i] - [1.5, 2][i])).toBeLessThan(TOL)
        }
        expect(Math.abs(v2[0])).toBeLessThan(TOL)
        expect(Math.abs(v2[1] - 6)).toBeLessThan(TOL)
        const [u1] = yield* Optimizer.clipByGlobalNorm([g1], 100)
        assert.deepStrictEqual(yield* values(u1), [3, 4])
      }))
  })

  describe("schedules", () => {
    it.effect("schedule values", () =>
      Effect.sync(() => {
        expect(LearningRate.constant(0.1)(42)).toBe(0.1)
        const exp = LearningRate.exponential(1, { decayRate: 0.5, decaySteps: 10 })
        expect(exp(0)).toBe(1)
        expect(Math.abs(exp(10) - 0.5)).toBeLessThan(1e-12)
        expect(Math.abs(exp(20) - 0.25)).toBeLessThan(1e-12)
        const step = LearningRate.stepwise(1, { dropFactor: 0.1, dropEvery: 5 })
        expect(step(4)).toBe(1)
        expect(Math.abs(step(5) - 0.1)).toBeLessThan(1e-12)
        const cos = LearningRate.cosine(1, { totalSteps: 100 })
        expect(Math.abs(cos(0) - 1)).toBeLessThan(1e-12)
        expect(Math.abs(cos(50) - 0.5)).toBeLessThan(1e-12)
        expect(Math.abs(cos(100))).toBeLessThan(1e-12)
        expect(Math.abs(cos(200))).toBeLessThan(1e-12)
        const warm = LearningRate.withWarmup(LearningRate.constant(0.5), 10)
        expect(Math.abs(warm(0) - 0.05)).toBeLessThan(1e-12)
        expect(Math.abs(warm(9) - 0.5)).toBeLessThan(1e-12)
        expect(warm(10)).toBe(0.5)
        expect(() => LearningRate.cosine(0, { totalSteps: 10 })).toThrow()
      }))

    it.effect("a scheduled adam converges", () =>
      Effect.gen(function*() {
        const schedule = LearningRate.withWarmup(LearningRate.cosine(0.1, { totalSteps: 200 }), 20)
        const w = yield* Tensor.fromTypedArray(floats([0, 0]))
        const x = yield* Tensor.fromTypedArray(floats([1, 1, 2, 1, 3, 1, 4, 1]), [4, 2])
        const y = yield* Tensor.fromTypedArray(floats([2, 3, 4, 5]), [4, 1])
        const optimizer = yield* Optimizer.adam()
        let params: ReadonlyArray<Tensor.Any> = [w]
        let state = yield* optimizer.init(params)
        for (let t = 0; t < 200; t++) {
          const lr = yield* Tensor.full([], schedule(t), { dtype: "f32" })
          const pred = yield* Tensor.matmul(x, yield* Tensor.reshape(params[0], [2, 1]))
          const loss = yield* Loss.mse(pred, y)
          const next = yield* Optimizer.step(optimizer, loss, params, state, lr)
          params = next.params
          state = next.state
        }
        const finalLoss = yield* values(
          yield* Loss.mse(yield* Tensor.matmul(x, yield* Tensor.reshape(params[0], [2, 1])), y)
        )
        expect(finalLoss[0]).toBeLessThan(1e-3)
      }))
  })
})

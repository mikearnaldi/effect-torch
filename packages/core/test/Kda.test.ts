import { describe, expect } from "@effect/vitest"
import * as assert from "@effect/vitest/utils"
import { Effect } from "effect"
import { Gradient, LearningRate, Loss, Model, Optimizer, Speculation, Tensor, Trainer } from "../src/index.ts"
import { floats, GRADCHECK_EPS, GRADCHECK_TOL, onDevices } from "./utils/devices.ts"

const values = (t: Tensor.Any) => Tensor.toNumberArray(t)

// f32 accumulation orders differ between the composed chunk kernels and
// the per-token oracle; the 129-step fixture stays within this relative bound.
const TOL = 2e-3
const close = (a: number, b: number): boolean => Math.abs(a - b) <= TOL * Math.max(1, Math.abs(b))

// deterministic, asymmetric values in (-1, 1)
const pattern = (n: number): Array<number> => Array.from({ length: n }, (_, i) => ((i * 7 + 3) % 13 - 6) / 5)

// Per-token gated delta-rule recurrence, the ground truth for the chunked
// semantic node:
//   S_t = (I - beta_t k_t k_tᵀ) Diag(exp(g_t)) S_{t-1} + beta_t k_t v_tᵀ
//   o_t = scale · S_tᵀ q_t
const naiveKda = (
  q: Tensor.Any,
  k: Tensor.Any,
  v: Tensor.Any,
  logDecay: Tensor.Any,
  beta: Tensor.Any,
  scale: number
) =>
  Effect.gen(function*() {
    const [b, h, t, dk] = q.shape
    const dv = v.shape[3]
    const swap = [0, 1, 3, 2]
    let state = yield* Tensor.zeros([b, h, dk, dv])
    const outs: Array<Tensor.Any> = []
    for (let i = 0; i < t; i++) {
      const at = (x: Tensor.Any, width: number) => Tensor.slice(x, { start: [0, 0, i, 0], end: [b, h, i + 1, width] })
      const qT = yield* at(q, dk)
      const kT = yield* at(k, dk)
      const vT = yield* at(v, dv)
      const alphaT = yield* Tensor.exp(yield* at(logDecay, dk))
      const bT = yield* at(beta, 1)
      const decayed = yield* Tensor.mul(state, yield* Tensor.transpose(alphaT, swap))
      const kvMem = yield* Tensor.matmul(
        yield* Tensor.transpose(decayed, swap),
        yield* Tensor.transpose(kT, swap)
      )
      const delta = yield* Tensor.mul(
        yield* Tensor.sub(yield* Tensor.transpose(vT, swap), kvMem),
        yield* Tensor.transpose(bT, swap)
      )
      state = yield* Tensor.add(
        decayed,
        yield* Tensor.matmul(yield* Tensor.transpose(kT, swap), yield* Tensor.transpose(delta, swap))
      )
      const o = yield* Tensor.mul(
        yield* Tensor.matmul(yield* Tensor.transpose(state, swap), yield* Tensor.transpose(qT, swap)),
        yield* Tensor.constantLike(state, scale)
      )
      outs.push(yield* Tensor.transpose(o, swap))
    }
    const [first, second, ...rest] = outs
    if (first === undefined) throw new Error("KDA reference must produce at least one row")
    if (second === undefined) return first
    return yield* Tensor.concat([first, second, ...rest], { dim: 2 })
  })

const inputs = (t: number, dk: number, dv: number, seed: number) =>
  Effect.gen(function*() {
    const b = 1
    const h = 2
    const f32 = (data: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
      Tensor.fromTypedArray(floats(data), shape)
    const n = b * h * t * dk
    const q = yield* f32(pattern(n).map((x) => x * 0.8 + seed * 0.01), [b, h, t, dk])
    const k = yield* f32(pattern(n).map((x) => x * 0.7 - 0.1), [b, h, t, dk])
    const v = yield* f32(pattern(b * h * t * dv).map((x) => x * -0.5 + 0.2), [b, h, t, dv])
    const logDecay = yield* f32(pattern(n).map((x) => -(Math.abs(x) + 0.05) * 1.5), [b, h, t, dk])
    const beta = yield* f32(pattern(b * h * t).map((x) => Math.abs(x) * 0.9 + 0.05), [b, h, t, 1])
    return { beta, k, logDecay, q, v }
  })

onDevices("Kda", (device) => (it) => {
  describe("kdaChunk", () => {
    const parity = (name: string, t: number, dk: number, dv: number, seed: number) =>
      it.effect(name, () =>
        Effect.gen(function*() {
          const { beta, k, logDecay, q, v } = yield* inputs(t, dk, dv, seed)
          const scale = 1 / Math.sqrt(dk)
          const out = yield* Tensor.kdaChunk(q, k, v, logDecay, beta)
          expect(out.shape).toEqual([1, 2, t, dv])
          const expected = yield* naiveKda(q, k, v, logDecay, beta, scale)
          const [a, b] = [yield* values(out), yield* values(expected)]
          a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}]: ${x} != ${b[i]}`))
        }))

    parity("matches the per-token recurrence within a sub-chunk", 13, 8, 8, 1)
    parity("matches the per-token recurrence across sub-chunks", 40, 8, 16, 2)
    parity("matches the per-token recurrence across a ragged chunk tail", 70, 16, 8, 3)
    parity("matches the per-token recurrence over multiple chunks", 129, 16, 16, 4)
    parity("matches the per-token recurrence for a single token", 1, 8, 8, 5)

    it.effect("respects a custom scale", () =>
      Effect.gen(function*() {
        const { beta, k, logDecay, q, v } = yield* inputs(20, 8, 8, 6)
        const out = yield* Tensor.kdaChunk(q, k, v, logDecay, beta, { scale: 0.5 })
        const expected = yield* naiveKda(q, k, v, logDecay, beta, 0.5)
        const [a, b] = [yield* values(out), yield* values(expected)]
        a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}]: ${x} != ${b[i]}`))
      }))

    it.effect("rejects a mismatched beta shape", () =>
      Effect.gen(function*() {
        const { k, logDecay, q, v } = yield* inputs(8, 8, 8, 7)
        const beta = yield* Tensor.fromTypedArray(floats(pattern(1 * 2 * 8 * 2)), [1, 2, 8, 2])
        const result = yield* Tensor.kdaChunk(q, k, v, logDecay, beta).pipe(Effect.flip)
        assert.assertTrue(String(result.message).includes("beta must have shape"))
      }))

    it.effect("numeric gradcheck of all five operands", () =>
      Effect.gen(function*() {
        const t = 8
        const dk = 4
        const dv = 6
        const base = yield* inputs(t, dk, dv, 9)
        const scale = 1 / Math.sqrt(dk)
        const w = yield* Tensor.fromTypedArray(floats(pattern(1 * 2 * t * dv)), [1, 2, t, dv])
        const build = (vals: ReadonlyArray<number>, which: number) =>
          Effect.gen(function*() {
            const parts: Array<Tensor.Any> = []
            const shapes = [base.q.shape, base.k.shape, base.v.shape, base.logDecay.shape, base.beta.shape]
            const current = [base.q, base.k, base.v, base.logDecay, base.beta]
            for (let j = 0; j < 5; j++) {
              parts.push(
                j === which
                  ? yield* Tensor.fromTypedArray(floats(vals), shapes[j])
                  : current[j]
              )
            }
            const out = yield* Tensor.kdaChunk(parts[0], parts[1], parts[2], parts[3], parts[4], { scale })
            return yield* Tensor.sum(yield* Tensor.mul(out, w))
          })
        const operands = [base.q, base.k, base.v, base.logDecay, base.beta]
        const loss = yield* Effect.gen(function*() {
          const out = yield* Tensor.kdaChunk(base.q, base.k, base.v, base.logDecay, base.beta, { scale })
          return yield* Tensor.sum(yield* Tensor.mul(out, w))
        })
        const analytic = yield* Tensor.compute(yield* Gradient.grad(loss, operands))
        for (let which = 0; which < 5; which++) {
          const vals = yield* values(operands[which])
          const gradVals = yield* values(analytic[which])
          // Probe boundaries and interior positions in every operand; exhaustive
          // finite differences would rebuild the recurrent graph per element.
          const probes = [...new Set([0, 3, 7, 13, 19, Math.floor(vals.length / 2), vals.length - 1])]
            .filter((i) => i < vals.length)
          for (const i of probes) {
            const plus = vals.map((x, j) => (j === i ? x + GRADCHECK_EPS : x))
            const minus = vals.map((x, j) => (j === i ? x - GRADCHECK_EPS : x))
            const fp = yield* values(yield* build(plus, which))
            const fm = yield* values(yield* build(minus, which))
            const numeric = (fp[0] - fm[0]) / (2 * GRADCHECK_EPS)
            expect(Math.abs(gradVals[i] - numeric)).toBeLessThan(
              GRADCHECK_TOL * Math.max(1, Math.abs(gradVals[i]))
            )
          }
        }
      }))
  })

  describe("kimiDeltaAttention", () => {
    it.effect("runs a finite forward pass at the declared shapes", () =>
      Effect.gen(function*() {
        const model = yield* Model.kimiDeltaAttention("kda", 32, 4)
        expect(model.parameterSpecs.map(({ name }) => name)).toEqual([
          "kda.qkv.weight",
          "kda.qkv.bias",
          "kda.convqkv.weight",
          "kda.fa.weight",
          "kda.fb.weight",
          "kda.alog",
          "kda.dtbias",
          "kda.b.weight",
          "kda.ga.weight",
          "kda.gb.weight",
          "kda.norm.weight",
          "kda.wo.weight",
          "kda.wo.bias"
        ])
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const x = yield* Tensor.fromTypedArray(
          floats(Array.from({ length: 2 * 10 * 32 }, (_, i) => ((i * 5 + 1) % 11 - 5) / 5)),
          [2, 10, 32]
        )
        const [out] = yield* Tensor.compute([yield* model.forward(params, x)])
        expect(out.shape).toEqual([2, 10, 32])
        const vs = yield* values(out)
        vs.forEach((x, i) => assert.assertTrue(Number.isFinite(x), `out[${i}] not finite`))
      }))

    it.effect("rejects invalid configurations", () =>
      Effect.gen(function*() {
        expect((yield* Effect.flip(Model.kimiDeltaAttention("", 8, 2))).op).toBe("kimiDeltaAttention")
        expect((yield* Effect.flip(Model.kimiDeltaAttention("a", 0, 2))).op).toBe("kimiDeltaAttention")
        expect((yield* Effect.flip(Model.kimiDeltaAttention("a", 7, 2))).message).toContain("divisible")
      }))

    it.effect("gradients flow to every parameter", () =>
      Effect.gen(function*() {
        const model = yield* Model.kimiDeltaAttention("kda", 16, 4)
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const x = yield* Tensor.fromTypedArray(
          floats(Array.from({ length: 2 * 9 * 16 }, (_, i) => ((i * 5 + 1) % 11 - 5) / 5)),
          [2, 9, 16]
        )
        const out = yield* model.forward(params, x)
        const loss = yield* Tensor.sum(yield* Tensor.square(out))
        const grads = yield* Tensor.compute(yield* Gradient.grad(loss, params))
        assert.strictEqual(grads.length, params.length)
        for (const [i, g] of grads.entries()) {
          const gv = yield* values(g)
          assert.assertTrue(
            gv.every(Number.isFinite) && gv.some((x) => x !== 0),
            `param ${model.parameterSpecs.map(({ name }) => name)[i]} has a degenerate gradient`
          )
        }
      }))

    it.effect("trains in mixedBf16 on Metal and reports a typed error elsewhere", () =>
      Effect.gen(function*() {
        const model = yield* Model.kimiDeltaAttention("kda", 32, 4)
        const raw = yield* Tensor.fromTypedArray(
          floats(Array.from({ length: 2 * 8 * 32 }, (_, i) => ((i * 5 + 1) % 11 - 5) / 5)),
          [2, 8, 32]
        )
        const data = {
          input: yield* Tensor.cast(raw, "bf16"),
          target: yield* Tensor.cast(raw, "bf16")
        }
        const makeTrainer = Effect.gen(function*() {
          return yield* Trainer.make(model, {
            optimizer: yield* Optimizer.adamW(),
            lr: LearningRate.constant(1e-3),
            loss: Loss.mse,
            data,
            stop: ({ step }) => step >= 4,
            precision: "mixedBf16"
          })
        })
        if (device !== "metal") {
          const error = yield* Effect.flip(
            Effect.flatMap(makeTrainer, (trainer) => Effect.flatMap(Model.initialize(model), trainer.train))
          )
          expect(error._tag).toBe("ModelError")
          return
        }
        const trainer = yield* makeTrainer
        const { params, step } = yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(step).toBe(4)
        expect(params.every((p) => p.dtype === "f32")).toBe(true)
        const forwardParams = yield* Effect.all(params.map((param) => Tensor.cast(param, "bf16")))
        const [out] = yield* Tensor.compute([yield* model.forward(forwardParams, data.input)])
        ;(yield* values(out)).forEach((x) => assert.assertTrue(Number.isFinite(x)))
      }))
  })

  describe("shortConv1d", () => {
    it.effect("matches the pad + depthwise conv1d composition", () =>
      Effect.gen(function*() {
        const x = yield* Tensor.fromTypedArray(
          floats(Array.from({ length: 2 * 7 * 6 }, (_, i) => ((i * 5 + 1) % 11 - 5) / 5)),
          [2, 7, 6]
        )
        const w = yield* Tensor.fromTypedArray(
          floats(Array.from({ length: 6 * 4 }, (_, i) => ((i * 3 + 2) % 7 - 3) / 3)),
          [6, 4]
        )
        const viaNode = yield* Tensor.shortConv1d(x, w)
        const manual = Effect.gen(function*() {
          const transposed = yield* Tensor.transpose(x, [0, 2, 1])
          const padded = yield* Tensor.pad(transposed, [[0, 0], [0, 0], [3, 0]])
          const w4 = yield* Tensor.reshape(w, [6, 1, 4])
          const convolved = yield* Tensor.conv1d(padded, w4, { groups: 6 })
          return yield* Tensor.transpose(convolved, [0, 2, 1])
        })
        const [a, b] = [yield* values(viaNode), yield* values(yield* manual)]
        a.forEach((x, i) => assert.assertTrue(close(x, b[i]), `out[${i}]: ${x} != ${b[i]}`))
      }))

    it.effect("numeric gradcheck of input and weight", () =>
      Effect.gen(function*() {
        const xVals = Array.from({ length: 2 * 5 * 4 }, (_, i) => ((i * 5 + 1) % 11 - 5) / 5)
        const wVals = Array.from({ length: 4 * 4 }, (_, i) => ((i * 3 + 2) % 7 - 3) / 3)
        const gw = yield* Tensor.fromTypedArray(floats(pattern(2 * 5 * 4)), [2, 5, 4])
        const build = (xv: ReadonlyArray<number>, wv: ReadonlyArray<number>) =>
          Effect.gen(function*() {
            const x = yield* Tensor.fromTypedArray(floats(xv), [2, 5, 4])
            const w = yield* Tensor.fromTypedArray(floats(wv), [4, 4])
            return yield* Tensor.sum(yield* Tensor.mul(yield* Tensor.shortConv1d(x, w), gw))
          })
        const x = yield* Tensor.fromTypedArray(floats(xVals), [2, 5, 4])
        const w = yield* Tensor.fromTypedArray(floats(wVals), [4, 4])
        const loss = yield* Tensor.sum(yield* Tensor.mul(yield* Tensor.shortConv1d(x, w), gw))
        const [dx, dw] = yield* Tensor.compute(yield* Gradient.grad(loss, [x, w]))
        const analytic = [yield* values(dx), yield* values(dw)]
        const bases = [xVals, wVals]
        for (const which of [0, 1]) {
          for (let i = 0; i < bases[which].length; i++) {
            const plus = bases[which].map((v, j) => (j === i ? v + GRADCHECK_EPS : v))
            const minus = bases[which].map((v, j) => (j === i ? v - GRADCHECK_EPS : v))
            const fp = which === 0 ? yield* values(yield* build(plus, wVals)) : yield* values(yield* build(xVals, plus))
            const fm = which === 0
              ? yield* values(yield* build(minus, wVals))
              : yield* values(yield* build(xVals, minus))
            const numeric = (fp[0] - fm[0]) / (2 * GRADCHECK_EPS)
            expect(Math.abs(analytic[which][i] - numeric)).toBeLessThan(
              GRADCHECK_TOL * Math.max(1, Math.abs(analytic[which][i]))
            )
          }
        }
      }))
  })

  describe("recurrent generation (RFC 0018)", () => {
    const VOCAB = 12
    const EMBED = 8
    const HEADS = 2

    // The K3-style hybrid has a KDA layer and a causal full-attention layer,
    // with no positional encoding in either one.
    const makeHybrid = Effect.gen(function*() {
      const wte = yield* Model.embedding("wte", VOCAB, EMBED)
      const kda = yield* Model.kimiDeltaAttention("kda", EMBED, HEADS)
      const attn = yield* Model.multiHeadAttention("attn", EMBED, HEADS, { causal: true })
      const head = yield* Model.linear("head", EMBED, VOCAB)
      return yield* Model.chain(wte, kda, attn, head)
    })

    const makePureKda = Effect.gen(function*() {
      const wte = yield* Model.embedding("wte", VOCAB, EMBED)
      const kda = yield* Model.kimiDeltaAttention("kda", EMBED, HEADS)
      const head = yield* Model.linear("head", EMBED, VOCAB)
      return yield* Model.chain(wte, kda, head)
    })

    const ids = (tokens: ReadonlyArray<number>) => Tensor.fromTypedArray(new Uint32Array(tokens), [1, tokens.length])

    const argmaxOf = (logits: Tensor.Any) =>
      Effect.gen(function*() {
        const values = yield* Tensor.toNumberArray(logits)
        if (Tensor.isTensor(logits)) yield* Tensor.clear(logits)
        return values.reduce((best, value, index) => (value > values[best] ? index : best), 0)
      })

    it.effect("rejects speculative configuration for recurrent target state", () =>
      Effect.gen(function*() {
        const model = yield* makeHybrid
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 2 })
        const error = yield* Effect.flip(Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          speculation: { proposer, maxDraftTokens: 2 }
        }))
        expect(error.message).toMatch(/KV-only/)
      }))

    // The reference performs greedy generation through the ordinary forward
    // graph and recomputes the whole context at every step.
    const naiveGenerate = (model: Model.Model, params: Model.Params, prompt: ReadonlyArray<number>, steps: number) =>
      Effect.gen(function*() {
        const context = [...prompt]
        for (let i = 0; i < steps; i++) {
          const input = yield* ids(context)
          const output = yield* model.forward(params, input)
          const t = input.shape[1]
          const [logits] = yield* Tensor.compute([
            yield* Tensor.reshape(
              yield* Tensor.slice(output, { start: [0, t - 1, 0], end: [1, t, VOCAB] }),
              [VOCAB]
            )
          ])
          context.push(yield* argmaxOf(logits))
        }
        return context
      })

    const cachedGenerate = (program: Model.InferenceProgram, prompt: ReadonlyArray<number>, steps: number) =>
      Effect.gen(function*() {
        const gen = yield* program.execution()
        const context = [...prompt]
        const entry = (yield* gen.add([yield* ids(prompt)]))[0]!
        let logits = entry.logits
        for (let i = 0; i < steps; i++) {
          const next = yield* argmaxOf(logits)
          context.push(next)
          if (i < steps - 1) {
            const [nextLogits] = yield* gen.step([{ seq: entry.seq, token: next }])
            logits = nextLogits
          }
        }
        return context
      })

    it.effect("hybrid KDA + NoPE attention matches naive greedy generation token-for-token", () =>
      Effect.gen(function*() {
        const model = yield* makeHybrid
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 5, 3]
        const steps = 9
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4]
        })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
        expect(cached.length).toBe(prompt.length + steps)
      }))

    it.effect("hybrid KDA multi-chunk prefill matches the naive reference", () =>
      Effect.gen(function*() {
        const model = yield* makeHybrid
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 5, 3, 8, 2, 11, 4, 7, 6]
        const steps = 4
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4]
        })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      }))

    it.effect("recurrent pools restore shared prefixes across sequences", () =>
      Effect.gen(function*() {
        const model = yield* makeHybrid
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 5, 3, 8, 2]
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4]
        })
        const gen = yield* program.execution()
        const first = (yield* gen.add([yield* ids(prompt)]))[0]!
        const second = (yield* gen.add([yield* ids(prompt)]))[0]!
        const firstValues = yield* Tensor.toNumberArray(first.logits)
        const secondValues = yield* Tensor.toNumberArray(second.logits)
        secondValues.forEach((value, index) => assert.assertTrue(close(value, firstValues[index]!)))
        expect(yield* first.seq.cursor()).toBe(prompt.length)
        expect(yield* second.seq.cursor()).toBe(prompt.length)
        yield* Tensor.clearAll([first.logits, second.logits])
        yield* gen.close()
      }))

    it.effect("prefillMatch restores a recurrent snapshot at a block boundary", () =>
      Effect.gen(function*() {
        const model = yield* makeHybrid
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const exemplar = yield* Tensor.zeros([1, 4], { dtype: "u32" })
        const placeholders: Array<Tensor.Lazy> = []
        for (let index = 0; index < params.length; index++) {
          placeholders.push(yield* Tensor.makeInput(index, params[index]!))
        }
        placeholders.push(yield* Tensor.makeInput(params.length, exemplar))
        const output = yield* model.forward(placeholders.slice(0, -1), placeholders[placeholders.length - 1]!)
        const program = yield* Tensor.compileDecodeProgram([output], {
          maxTokens: 16,
          blockSize: 4,
          kvDtype: "f32",
          batch: 1
        })
        const pool = yield* Tensor.makeKvPool(
          program.layers,
          program.kvHeads,
          program.headDim,
          program.maxTokens,
          program.blockSize,
          program.kvDtype,
          {
            kdaLayers: program.kdaLayers,
            kdaHeads: program.kdaHeads,
            kdaHeadDim: program.kdaHeadDim,
            kdaValueDim: program.kdaValueDim,
            convLayers: program.convLayers,
            convChannels: program.convChannels,
            convKernel: program.convKernel
          }
        )
        const first = yield* Tensor.makeKvSequence(pool)
        const prefix = [1, 5, 3, 8]
        const [firstOutput] = yield* Tensor.runDecodeProgram(
          program,
          [...params, yield* ids(prefix)],
          first,
          prefix
        )
        yield* Tensor.clear(firstOutput)
        yield* Tensor.releaseKvSequence(first)

        const resumed = yield* Tensor.makeKvSequence(pool)
        const prompt = [...prefix, 2]
        expect(yield* Tensor.kvPrefillMatch(resumed, prompt)).toBe(4)
        const [suffixOutput] = yield* Tensor.runDecodeProgram(
          program,
          [...params, yield* ids([2, 0, 0, 0])],
          resumed,
          [2]
        )
        const [actual] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(suffixOutput, { start: [0, 0, 0], end: [1, 1, VOCAB] }),
            [VOCAB]
          )
        ])
        const fullOutput = yield* model.forward(params, yield* ids(prompt))
        const [expected] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(fullOutput, { start: [0, 4, 0], end: [1, 5, VOCAB] }),
            [VOCAB]
          )
        ])
        const actualValues = yield* Tensor.toNumberArray(actual)
        const expectedValues = yield* Tensor.toNumberArray(expected)
        actualValues.forEach((value, index) => assert.assertTrue(close(value, expectedValues[index]!)))
        yield* Tensor.clearAll([suffixOutput, actual, expected])
        yield* Tensor.releaseKvSequence(resumed)
      }))

    it.effect("a pure-KDA stack (zero KV layers) generates through the decode programs", () =>
      Effect.gen(function*() {
        const model = yield* makePureKda
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [2, 4]
        const steps = 7
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4]
        })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      }))

    it.effect("batched decode steps two hybrid sequences independently", () =>
      Effect.gen(function*() {
        const model = yield* makeHybrid
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 4
        })
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids([1, 5, 3])]))[0]!
        const b = (yield* gen.add([yield* ids([2, 7])]))[0]!
        const naiveA = yield* naiveGenerate(model, params, [1, 5, 3], 5)
        const naiveB = yield* naiveGenerate(model, params, [2, 7], 5)
        const contextA = [1, 5, 3]
        const contextB = [2, 7]
        let logitsA = a.logits
        let logitsB = b.logits
        for (let i = 0; i < 5; i++) {
          const nextA = yield* argmaxOf(logitsA)
          const nextB = yield* argmaxOf(logitsB)
          contextA.push(nextA)
          contextB.push(nextB)
          if (i < 4) {
            const [nextLogitsA, nextLogitsB] = yield* gen.step([
              { seq: a.seq, token: nextA },
              { seq: b.seq, token: nextB }
            ])
            logitsA = nextLogitsA
            logitsB = nextLogitsB
          }
        }
        expect(contextA).toEqual(naiveA)
        expect(contextB).toEqual(naiveB)
      }))

    it.effect("hybrid recurrent state stays physical across sparse refill and reversed requests", () =>
      Effect.gen(function*() {
        const model = yield* makeHybrid
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2
        })
        const referenceB = yield* program.execution()
        const bRef = (yield* referenceB.add([yield* ids([2, 7])]))[0]!
        yield* Tensor.clear(bRef.logits)
        const [bRefFirst] = yield* referenceB.step([{ seq: bRef.seq, token: 3 }])
        const [bRefSecond] = yield* referenceB.step([{ seq: bRef.seq, token: 4 }])
        const referenceC = yield* program.execution()
        const cRef = (yield* referenceC.add([yield* ids([4, 1, 6])]))[0]!
        yield* Tensor.clear(cRef.logits)
        const [cRefNext] = yield* referenceC.step([{ seq: cRef.seq, token: 5 }])

        const gen = yield* program.execution()
        const entries = yield* gen.add([yield* ids([1, 5, 3]), yield* ids([2, 7])])
        yield* Tensor.clearAll(entries.map((entry) => entry.logits))
        yield* entries[0]!.seq.finish()
        const [bFirst] = yield* gen.step([{ seq: entries[1]!.seq, token: 3 }])
        const [c] = yield* gen.add([yield* ids([4, 1, 6])])
        yield* Tensor.clear(c!.logits)
        const [bSecond, cNext] = yield* gen.step([
          { seq: entries[1]!.seq, token: 4 },
          { seq: c!.seq, token: 5 }
        ])
        for (const [actual, expected] of [[bFirst, bRefFirst], [bSecond, bRefSecond], [cNext, cRefNext]] as const) {
          const actualValues = yield* Tensor.toNumberArray(actual!)
          const expectedValues = yield* Tensor.toNumberArray(expected!)
          actualValues.forEach((value, index) => assert.assertTrue(close(value, expectedValues[index]!)))
        }
        yield* Tensor.clearAll([bFirst!, bRefFirst!, bSecond!, bRefSecond!, cNext!, cRefNext!])
        yield* gen.close()
        yield* referenceB.close()
        yield* referenceC.close()
      }))
  })
})

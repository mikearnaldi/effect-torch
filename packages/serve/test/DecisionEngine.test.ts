import { describe, expect, it } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber, Schema, Scope } from "effect"
import { TestClock } from "effect/testing"
import {
  DecisionEngine as Engine,
  DecisionNoise as Noise,
  DecisionPlanner as Planner,
  type DecisionScaffold as Scaffold
} from "../src/index.ts"

const modelId = "scheduler-test-model"

const settings = { seed: "independent-read-test", reads: 1 } as const

const question = (instructions = "pick") => ({ type: "choice", instructions, criteria: { left: null, right: null } })

const request = (instructions = "pick") => ({
  model: modelId,
  state: "state",
  questions: { q: question(instructions) }
})

// No tokenizer/native assets: this compiler stands in for Scaffold.build. Its
// key deliberately differs from the planner key, and every buffer is a copy.
const scaffoldFor = (
  plan: Planner.QuestionPlan,
  ids = Uint32Array.from([2, ...Buffer.from(plan.semanticKey, "hex")])
): Scaffold.Scaffold => {
  const canvas = Uint32Array.from([100, 101, 200, 106, 0, 0, 0, 0])

  const labels = plan.options.map((option, index) =>
    Object.freeze({ code: String(index), optionName: option.name, tokenId: 200 + index })
  )

  return Object.freeze({
    semanticKey: "compiled-scaffold:" + plan.semanticKey,
    model: plan.model,
    prompt: plan.prompt,
    tokenizerSha256: "test-tokenizer",
    chatTemplateSha256: "test-template",
    vocabularySize: 4096,
    labels: Object.freeze(labels),
    slot: 2,
    contentLength: 4,
    get prefixIds() {
      return ids.slice()
    },
    get canvasIds() {
      return canvas.slice()
    },
    get allowedTokenIds() {
      return Uint32Array.from(labels.map((label) => label.tokenId))
    }
  })
}

class BackendError extends Schema.TaggedErrorClass<BackendError>()("BackendError", { message: Schema.String }) {}

interface Prefix {
  readonly serial: number
  readonly ids: Uint32Array
  dead: boolean
  readers: number
}

interface ReadCall {
  readonly prefix: Prefix
  readonly canvas: Uint32Array
  readonly slot: number
  readonly labels: Uint32Array
  readonly logits: ReadonlyArray<number>
}

const mock = (hooks: {
  readonly prefill?: (prefix: Prefix) => Effect.Effect<void, BackendError>
  readonly read?: (call: ReadCall, index: number) => Effect.Effect<void, BackendError>
} = {}) => {
  const prefixes: Array<Prefix> = []
  const reads: Array<ReadCall> = []
  const events: Array<string> = []
  let resident = 0
  let peakResident = 0
  let activeReads = 0
  let peakReads = 0

  const runtime: Engine.Runtime<Prefix, BackendError> = {
    prefill: (ids) =>
      Effect.gen(function*() {
        const prefix = yield* Effect.acquireRelease(
          Effect.sync(() => {
            const prefix: Prefix = { serial: prefixes.length, ids: ids.slice(), dead: false, readers: 0 }
            prefixes.push(prefix)
            peakResident = Math.max(peakResident, ++resident)
            events.push("prefill:" + prefix.serial)

            return prefix
          }),
          (prefix) =>
            Effect.sync(() => {
              expect(prefix.dead).toBe(false)
              expect(prefix.readers).toBe(0)
              prefix.dead = true
              resident--
              events.push("release:" + prefix.serial)
            })
        )

        if (hooks.prefill !== undefined) yield* hooks.prefill(prefix)

        return prefix
      }),
    read: (prefix, canvas, slot, labels) =>
      Effect.acquireUseRelease(
        Effect.sync(() => {
          expect(prefix.dead).toBe(false)
          prefix.readers++
          peakReads = Math.max(peakReads, ++activeReads)

          const call = {
            prefix,
            canvas,
            slot,
            labels,
            logits: Array.from(labels, (_, i) => (canvas[slot] % (17 + 7 * i) - 10) * (i + 1) / 3)
          }

          const index = reads.push(call) - 1
          events.push("read:" + index)

          return { call, index }
        }),
        ({ call, index }) =>
          Effect.gen(function*() {
            if (hooks.read !== undefined) yield* hooks.read(call, index)

            expect(prefix.dead).toBe(false)

            return call.logits
          }),
        ({ index }) =>
          Effect.sync(() => {
            expect(prefix.dead).toBe(false)
            prefix.readers--
            activeReads--
            events.push("read-end:" + index)
          })
      )
  }

  return {
    runtime,
    prefixes,
    reads,
    events,
    get resident() {
      return resident
    },
    get peakResident() {
      return peakResident
    },
    get activeReads() {
      return activeReads
    },
    get peakReads() {
      return peakReads
    }
  }
}

const optionsFor = (backend: ReturnType<typeof mock>): Engine.Options<Prefix, BackendError, never> => ({
  modelId,
  runtime: backend.runtime,
  prepare: (plan) => Effect.succeed(scaffoldFor(plan)),
  maxActiveQuestions: 2,
  maxCachedPrefixes: 2,
  maxPrefixTokens: 128,
  cacheIdleMs: 100
})

const signal = (deferred: Deferred.Deferred<void>) => Deferred.succeed(deferred, undefined)

const leftProbability = (logits: ReadonlyArray<number>) => 1 / (1 + Math.exp(logits[1] - logits[0]))

describe("bounded independent-read engine", () => {
  it.effect("uses actual scaffold noise, averages per-read probabilities, and accounts logical canvas work", () =>
    Effect.scoped(Effect.gen(function*() {
      const backend = mock()
      const engine = yield* Engine.make(optionsFor(backend))
      const one = yield* engine.run(request(), settings)
      const four = yield* engine.run(request(), { ...settings, reads: 4 })
      const plan = yield* Planner.plan(request(), { modelId })
      const scaffold = scaffoldFor(plan.routes[0].decision)
      expect(scaffold.semanticKey).not.toBe(plan.routes[0].decision.semanticKey)
      const calls = backend.reads.slice(1)
      expect(calls[0].canvas).toEqual(backend.reads[0].canvas)

      for (const [readIndex, call] of calls.entries()) {
        const noise = yield* Noise.make({
          semanticKey: scaffold.semanticKey,
          seed: settings.seed,
          readIndex,
          length: 1,
          vocabularySize: 4096
        })

        expect(call.canvas[2]).toBe(noise[0])
        expect(Array.from(call.canvas).filter((_, i) => i !== 2)).toEqual([100, 101, 106, 0, 0, 0, 0])
        expect(call.labels).toEqual(scaffold.allowedTokenIds)
        expect(call.prefix).toBe(backend.prefixes[0])
      }

      expect(new Set(calls.map((call) => call.canvas)).size).toBe(4)
      expect(new Set(calls.map((call) => call.canvas[2])).size).toBe(4)
      const answer = four.answers.q
      expect(answer.type).toBe("choice")

      if (answer.type !== "choice") return

      const mean = calls.reduce((sum, call) => sum + leftProbability(call.logits), 0) / 4
      expect(answer.probabilities.left).toBeCloseTo(mean, 14)
      const meanLogits = [0, 1].map((i) => calls.reduce((sum, call) => sum + call.logits[i], 0) / 4)
      expect(Math.abs(mean - leftProbability(meanLogits))).toBeGreaterThan(0.01)
      expect(one.usage).toEqual({ input_tokens: 33, output_tokens: 8 })
      expect(four.usage).toEqual({ input_tokens: 33, output_tokens: 32 })
      expect(yield* engine.statistics).toEqual({ prefills: 1, livePrefixes: 1, reads: 5, cachedPrefixes: 1 })
    })))

  it.effect("keeps answers invariant under ID renaming, siblings and reordering; duplicate routes execute once", () =>
    Effect.scoped(Effect.gen(function*() {
      const backend = mock()
      const engine = yield* Engine.make(optionsFor(backend))
      const baseline = yield* engine.run(request(), { ...settings, reads: 4 })

      const first = yield* engine.run({
        ...request(),
        questions: { sibling: question("other"), renamed: question(), duplicate: question() }
      }, { ...settings, reads: 4 })

      const second = yield* engine.run({
        ...request(),
        questions: { anotherId: question(), sibling: question("other") }
      }, { ...settings, reads: 4 })

      expect(first.answers.renamed).toEqual(baseline.answers.q)
      expect(first.answers.duplicate).toEqual(baseline.answers.q)
      expect(second.answers.anotherId).toEqual(baseline.answers.q)
      expect(second.answers.sibling).toEqual(first.answers.sibling)
      expect(first.usage).toEqual({ input_tokens: 66, output_tokens: 64 })
      expect(yield* engine.statistics).toEqual({ prefills: 2, livePrefixes: 2, reads: 20, cachedPrefixes: 2 })
    })))

  it.effect("keys whole prefixes by exact token arrays rather than semantic keys or ambiguous concatenation", () =>
    Effect.scoped(Effect.gen(function*() {
      const backend = mock()

      const engine = yield* Engine.make({
        ...optionsFor(backend),
        prepare: (plan) =>
          Effect.succeed(
            scaffoldFor(plan, Uint32Array.from(plan.question.instructions === "third" ? [12, 3] : [1, 23]))
          )
      })

      yield* engine.run(request("first"), settings)
      yield* engine.run(request("second"), settings)
      expect(backend.prefixes).toHaveLength(1)
      yield* engine.run(request("third"), settings)
      expect(backend.prefixes).toHaveLength(2)
      expect(backend.reads[0].prefix).toBe(backend.reads[1].prefix)
      expect(backend.reads[2].prefix).not.toBe(backend.reads[0].prefix)
    })))

  it.effect("rejects configuration, request, seed and read-count errors before backend work", () =>
    Effect.scoped(Effect.gen(function*() {
      const backend = mock()

      for (
        const bad of [{ maxActiveQuestions: 0 }, { maxCachedPrefixes: 1 }, { maxPrefixTokens: 1.5 }, {
          cacheIdleMs: Infinity
        }]
      ) {
        expect(yield* Engine.make({ ...optionsFor(backend), ...bad }).pipe(Effect.flip)).toBeInstanceOf(
          Engine.EngineError
        )
      }

      let prepared = 0

      const engine = yield* Engine.make({
        ...optionsFor(backend),
        prepare: (plan) =>
          Effect.sync(() => {
            prepared++

            return scaffoldFor(plan)
          })
      })

      expect(yield* engine.run({ ...request(), model: "wrong" }, settings).pipe(Effect.flip)).toMatchObject({
        _tag: "ValidationError"
      })
      expect(yield* engine.run({ ...request(), questions: {} }, settings).pipe(Effect.flip)).toMatchObject({
        _tag: "ValidationError"
      })
      // @ts-expect-error Untrusted JS callers still receive a typed failure.
      expect(yield* engine.run(request(), { seed: 123, reads: 1 }).pipe(Effect.flip)).toBeInstanceOf(Engine.EngineError)
      // @ts-expect-error Only one or four independent reads are supported.
      expect(yield* engine.run(request(), { seed: "seed", reads: 2 }).pipe(Effect.flip)).toBeInstanceOf(
        Engine.EngineError
      )
      expect(prepared).toBe(0)
      expect(backend.prefixes).toHaveLength(0)
      expect(backend.reads).toHaveLength(0)
    })))

  it.effect("rejects empty/overlong prefixes and a mismatched prepared model before prefill", () =>
    Effect.scoped(Effect.gen(function*() {
      const backend = mock()

      for (const length of [0, 129]) {
        const engine = yield* Engine.make({
          ...optionsFor(backend),
          prepare: (plan) => Effect.succeed(scaffoldFor(plan, new Uint32Array(length)))
        })

        expect(yield* engine.run(request(), settings).pipe(Effect.flip)).toBeInstanceOf(Engine.EngineError)
      }

      const wrong = yield* Engine.make({
        ...optionsFor(backend),
        prepare: (plan) => Effect.succeed(scaffoldFor({ ...plan, model: "other" }))
      })

      expect(yield* wrong.run(request(), settings).pipe(Effect.flip)).toBeInstanceOf(Engine.EngineError)
      expect(backend.prefixes).toHaveLength(0)
    })))

  it.effect("evicts the least-recently-used idle prefix before allocating a replacement", () =>
    Effect.scoped(Effect.gen(function*() {
      const backend = mock()
      const engine = yield* Engine.make(optionsFor(backend))

      for (const name of ["A", "B", "A", "C"]) yield* engine.run(request(name), settings)

      expect(backend.prefixes.map((prefix) => prefix.dead)).toEqual([false, true, false])
      expect(backend.peakResident).toBe(2)
      expect(backend.events.indexOf("release:1")).toBeLessThan(backend.events.indexOf("prefill:2"))
      yield* engine.run(request("B"), settings)
      expect(backend.prefixes.map((prefix) => prefix.dead)).toEqual([true, true, false, false])
      expect((yield* engine.statistics).cachedPrefixes).toBe(2)
    })))

  it.effect("skips an older active prefix when evicting an idle entry", () =>
    Effect.scoped(Effect.gen(function*() {
      const started = yield* Deferred.make<void>()
      const finish = yield* Deferred.make<void>()

      const backend = mock({
        read: (_, index) => index === 0 ? signal(started).pipe(Effect.andThen(Deferred.await(finish))) : Effect.void
      })

      const engine = yield* Engine.make(optionsFor(backend))
      const active = yield* Effect.forkScoped(engine.run(request("A"), settings))
      yield* Deferred.await(started)
      yield* engine.run(request("B"), settings)
      yield* engine.run(request("C"), settings)
      expect(backend.prefixes.map((prefix) => prefix.dead)).toEqual([false, true, false])
      expect(backend.prefixes[0].readers).toBe(1)
      expect(backend.peakResident).toBe(2)
      yield* signal(finish)
      yield* Fiber.join(active)
    })))

  it.effect("shares an in-flight prefill across callers and survives cancellation of its first borrower", () =>
    Effect.scoped(Effect.gen(function*() {
      const started = yield* Deferred.make<void>()
      const finish = yield* Deferred.make<void>()
      const backend = mock({ prefill: () => signal(started).pipe(Effect.andThen(Deferred.await(finish))) })
      const engine = yield* Engine.make(optionsFor(backend))
      const first = yield* Effect.forkScoped(engine.run(request(), settings))
      yield* Deferred.await(started)
      const second = yield* Effect.forkScoped(engine.run(request(), settings))
      yield* TestClock.adjust(1)
      yield* Fiber.interrupt(first)
      expect(backend.prefixes).toHaveLength(1)
      expect(backend.prefixes[0].dead).toBe(false)
      expect((yield* engine.statistics).cachedPrefixes).toBe(1)
      yield* signal(finish)
      yield* Fiber.join(second)
      yield* engine.run(request(), settings)
      expect(backend.prefixes).toHaveLength(1)
      expect(backend.reads).toHaveLength(2)
    })))

  it.effect("releases abandoned partial prefills immediately and permits a fresh attempt", () =>
    Effect.scoped(Effect.gen(function*() {
      const started = yield* Deferred.make<void>()

      const backend = mock({
        prefill: (prefix) => prefix.serial === 0 ? signal(started).pipe(Effect.andThen(Effect.never)) : Effect.void
      })

      const engine = yield* Engine.make(optionsFor(backend))
      const first = yield* Effect.forkScoped(engine.run(request(), settings))
      yield* Deferred.await(started)
      yield* Fiber.interrupt(first)
      expect(backend.prefixes[0].dead).toBe(true)
      expect((yield* engine.statistics).cachedPrefixes).toBe(0)
      yield* engine.run(request(), settings)
      expect(backend.prefixes).toHaveLength(2)
      expect(backend.resident).toBe(1)
    })))

  it.effect("does not cache prefill failures, cleans partial resources, and leaves another caller active", () =>
    Effect.scoped(Effect.gen(function*() {
      const readStarted = yield* Deferred.make<void>()
      const readFinish = yield* Deferred.make<void>()
      const failure = new BackendError({ message: "partial prefill" })

      const backend = mock({
        prefill: (prefix) => prefix.serial === 1 ? Effect.fail(failure) : Effect.void,
        read: (_, index) =>
          index === 0 ? signal(readStarted).pipe(Effect.andThen(Deferred.await(readFinish))) : Effect.void
      })

      const engine = yield* Engine.make(optionsFor(backend))
      const active = yield* Effect.forkScoped(engine.run(request("active"), settings))
      yield* Deferred.await(readStarted)
      expect(yield* engine.run(request("retry"), settings).pipe(Effect.flip)).toBe(failure)
      expect(backend.prefixes[1].dead).toBe(true)
      expect(backend.prefixes[0].readers).toBe(1)
      yield* engine.run(request("retry"), settings)
      expect(backend.prefixes).toHaveLength(3)
      expect(backend.peakResident).toBe(2)
      yield* signal(readFinish)
      yield* Fiber.join(active)
    })))

  it.effect("delivers one failed prefill to both waiting borrowers and retries after cleanup", () =>
    Effect.scoped(Effect.gen(function*() {
      const started = yield* Deferred.make<void>()
      const finish = yield* Deferred.make<void>()
      const failure = new BackendError({ message: "shared failure" })

      const backend = mock({
        prefill: (prefix) =>
          prefix.serial === 0 ?
            signal(started).pipe(
              Effect.andThen(Deferred.await(finish)),
              Effect.andThen(Effect.fail(failure))
            ) :
            Effect.void
      })

      const engine = yield* Engine.make(optionsFor(backend))
      const first = yield* Effect.forkScoped(engine.run(request(), settings).pipe(Effect.flip))
      yield* Deferred.await(started)
      const second = yield* Effect.forkScoped(engine.run(request(), settings).pipe(Effect.flip))
      yield* TestClock.adjust(1)
      yield* signal(finish)
      expect(yield* Fiber.join(first)).toBe(failure)
      expect(yield* Fiber.join(second)).toBe(failure)
      expect(backend.prefixes).toHaveLength(1)
      expect(backend.resident).toBe(0)
      expect((yield* engine.statistics).cachedPrefixes).toBe(0)
      yield* engine.run(request(), settings)
      expect(backend.prefixes).toHaveLength(2)
    })))

  it.effect("a read failure cancels its sibling reads but preserves a different caller sharing the prefix", () =>
    Effect.scoped(Effect.gen(function*() {
      const starts = yield* Effect.forEach(Array.from({ length: 5 }), () => Deferred.make<void>())
      const finishFirst = yield* Deferred.make<void>()
      const failRead = yield* Deferred.make<void>()
      const failure = new BackendError({ message: "read failed" })

      const backend = mock({
        read: (_, index) =>
          index > 4 ? Effect.void : signal(starts[index]).pipe(Effect.andThen(
            index === 0 ?
              Deferred.await(finishFirst)
              : index === 1
              ? Deferred.await(failRead).pipe(Effect.andThen(Effect.fail(failure)))
              : Effect.never
          ))
      })

      const engine = yield* Engine.make(optionsFor(backend))
      const first = yield* Effect.forkScoped(engine.run(request(), settings))
      yield* Deferred.await(starts[0])
      const failing = yield* Effect.forkScoped(engine.run(request(), { ...settings, reads: 4 }).pipe(Effect.flip))
      yield* Deferred.await(starts[4])
      yield* signal(failRead)
      expect(yield* Fiber.join(failing)).toBe(failure)
      expect(backend.prefixes[0].readers).toBe(1)
      expect(backend.prefixes[0].dead).toBe(false)
      yield* engine.run(request(), settings)
      expect(backend.prefixes).toHaveLength(1)
      yield* signal(finishFirst)
      yield* Fiber.join(first)
      expect(backend.activeReads).toBe(0)
    })))

  it.effect("cancels one read without releasing another caller's lease, then serves a real cache hit", () =>
    Effect.scoped(Effect.gen(function*() {
      const starts = yield* Effect.forEach([0, 1], () => Deferred.make<void>())
      const finish = yield* Deferred.make<void>()

      const backend = mock({
        read: (_, index) => index < 2 ? signal(starts[index]).pipe(Effect.andThen(Deferred.await(finish))) : Effect.void
      })

      const engine = yield* Engine.make(optionsFor(backend))
      const first = yield* Effect.forkScoped(engine.run(request(), settings))
      yield* Deferred.await(starts[0])
      const second = yield* Effect.forkScoped(engine.run(request(), settings))
      yield* Deferred.await(starts[1])
      yield* Fiber.interrupt(first)
      expect(backend.prefixes[0].dead).toBe(false)
      expect(backend.prefixes[0].readers).toBe(1)
      yield* engine.run(request(), settings)
      expect(backend.prefixes).toHaveLength(1)
      yield* signal(finish)
      yield* Fiber.join(second)
      expect(backend.reads).toHaveLength(3)
      expect((yield* engine.statistics).reads).toBe(2)
    })))

  it.effect("bounds questions globally across requests, including in-flight entries, while four reads run concurrently", () =>
    Effect.scoped(Effect.gen(function*() {
      const starts = yield* Effect.forEach(Array.from({ length: 8 }), () => Deferred.make<void>())
      const finish = yield* Deferred.make<void>()

      const backend = mock({
        read: (_, index) => index < 8 ? signal(starts[index]).pipe(Effect.andThen(Deferred.await(finish))) : Effect.void
      })

      let prepared = 0

      const engine = yield* Engine.make({
        ...optionsFor(backend),
        prepare: (plan) =>
          Effect.sync(() => {
            prepared++

            return scaffoldFor(plan)
          })
      })

      const first = yield* Effect.forkScoped(engine.run(request("first"), { ...settings, reads: 4 }))
      yield* Deferred.await(starts[3])
      const second = yield* Effect.forkScoped(engine.run(request("second"), { ...settings, reads: 4 }))
      yield* Deferred.await(starts[7])
      const queued = yield* Effect.forkScoped(engine.run(request("third"), settings))
      yield* TestClock.adjust(1)
      expect(prepared).toBe(2)
      expect(backend.activeReads).toBe(8)
      expect((yield* engine.statistics).cachedPrefixes).toBe(2)
      yield* signal(finish)
      yield* Fiber.joinAll([first, second, queued])
      expect(prepared).toBe(3)
      expect(backend.peakResident).toBe(2)
      expect(backend.peakReads).toBe(8)
    })))

  it.effect("counts incomplete prefills against capacity and closes their partial resources on shutdown", () =>
    Effect.scoped(Effect.gen(function*() {
      const starts = yield* Effect.forEach([0, 1], () => Deferred.make<void>())
      const backend = mock({ prefill: (prefix) => signal(starts[prefix.serial]).pipe(Effect.andThen(Effect.never)) })
      const engineScope = yield* Scope.make()
      yield* Effect.addFinalizer((exit) => Scope.close(engineScope, exit))
      let prepared = 0

      const engine = yield* Scope.provide(
        Engine.make({
          ...optionsFor(backend),
          prepare: (plan) =>
            Effect.sync(() => {
              prepared++

              return scaffoldFor(plan)
            })
        }),
        engineScope
      )

      const first = yield* Effect.forkScoped(engine.run(request("first"), settings))
      yield* Deferred.await(starts[0])
      const second = yield* Effect.forkScoped(engine.run(request("second"), settings))
      yield* Deferred.await(starts[1])
      const queued = yield* Effect.forkScoped(engine.run(request("third"), settings))
      yield* TestClock.adjust(1)
      expect(prepared).toBe(2)
      expect(yield* engine.statistics).toEqual({ prefills: 0, livePrefixes: 0, reads: 0, cachedPrefixes: 2 })
      yield* Scope.close(engineScope, Exit.void)

      for (const fiber of [first, second, queued]) expect(Exit.hasInterrupts(yield* Fiber.await(fiber))).toBe(true)

      expect(backend.resident).toBe(0)
      expect(backend.peakResident).toBe(2)
      expect((yield* engine.statistics).cachedPrefixes).toBe(0)
    })))

  it.effect("starts TTL at last release, refreshes idle hits, and never expires an active lease", () =>
    Effect.scoped(Effect.gen(function*() {
      const started = yield* Deferred.make<void>()
      const finish = yield* Deferred.make<void>()

      const backend = mock({
        read: (_, index) => index === 2 ? signal(started).pipe(Effect.andThen(Deferred.await(finish))) : Effect.void
      })

      const engine = yield* Engine.make(optionsFor(backend))
      yield* engine.run(request(), settings)
      yield* TestClock.adjust(90)
      yield* engine.run(request(), settings)
      yield* TestClock.adjust(90)
      expect(backend.prefixes[0].dead).toBe(false)
      const active = yield* Effect.forkScoped(engine.run(request(), settings))
      yield* Deferred.await(started)
      yield* TestClock.adjust(1000)
      expect(backend.prefixes[0].dead).toBe(false)
      yield* signal(finish)
      yield* Fiber.join(active)
      yield* TestClock.adjust(99)
      expect(backend.prefixes[0].dead).toBe(false)
      yield* TestClock.adjust(1)
      expect(backend.prefixes[0].dead).toBe(true)
      expect((yield* engine.statistics).cachedPrefixes).toBe(0)
      yield* engine.run(request(), settings)
      expect(backend.prefixes).toHaveLength(2)
    })))

  it.effect("scope shutdown drains active and queued calls before releasing prefixes and rejects later calls", () =>
    Effect.scoped(Effect.gen(function*() {
      const started = yield* Deferred.make<void>()
      const interrupted = yield* Deferred.make<void>()
      const cleanup = yield* Deferred.make<void>()

      const backend = mock({
        read: (_, index) =>
          index === 1 ?
            signal(started).pipe(
              Effect.andThen(Effect.never),
              Effect.onInterrupt(() => signal(interrupted).pipe(Effect.andThen(Deferred.await(cleanup))))
            ) :
            Effect.void
      })

      const engineScope = yield* Scope.make()
      yield* Effect.addFinalizer((exit) => Scope.close(engineScope, exit))
      let prepared = 0

      const engine = yield* Scope.provide(
        Engine.make({
          ...optionsFor(backend),
          maxActiveQuestions: 1,
          prepare: (plan) =>
            Effect.sync(() => {
              prepared++

              return scaffoldFor(plan)
            })
        }),
        engineScope
      )

      yield* engine.run(request("idle"), settings)
      const active = yield* Effect.forkScoped(engine.run(request("active"), settings))
      yield* Deferred.await(started)
      const queued = yield* Effect.forkScoped(engine.run(request("queued"), settings))
      const closing = yield* Effect.forkScoped(Scope.close(engineScope, Exit.void))
      yield* Deferred.await(interrupted)
      expect(backend.prefixes.every((prefix) => !prefix.dead)).toBe(true)
      expect(yield* engine.run(request(), settings).pipe(Effect.flip)).toBeInstanceOf(Engine.EngineError)
      yield* signal(cleanup)
      yield* Fiber.join(closing)
      expect(Exit.hasInterrupts(yield* Fiber.await(active))).toBe(true)
      expect(Exit.hasInterrupts(yield* Fiber.await(queued))).toBe(true)
      expect(prepared).toBe(2)
      expect(backend.resident).toBe(0)
      expect(backend.events.indexOf("read-end:1")).toBeLessThan(backend.events.indexOf("release:1"))
      expect(yield* engine.statistics).toEqual({ prefills: 2, livePrefixes: 0, reads: 1, cachedPrefixes: 0 })
    })))
})

import { Deferred, Effect, Exit, Fiber, Schema, Scope, Semaphore } from "effect"
import type * as Contract from "./Decision.ts"
import type * as Noise from "./DecisionNoise.ts"
import * as Planner from "./DecisionPlanner.ts"
import * as Readout from "./DecisionReadout.ts"
import * as Scaffold from "./DecisionScaffold.ts"

/**
 * Invalid cache configuration or engine lifecycle failure.
 *
 * @since 0.1.0
 * @category models
 */
export class EngineError extends Schema.TaggedErrorClass<EngineError>()("EngineError", { message: Schema.String }) {}

/**
 * Model-specific execution, with prefix ownership attached to the acquisition scope.
 *
 * @since 0.1.0
 * @category models
 */
export interface Runtime<Prefix, Error> {
  readonly prefill: (ids: Uint32Array) => Effect.Effect<Prefix, Error, Scope.Scope>
  /**
   * Read the complete canvas without modifying prefix state; return all selected
   * logits in order. Concurrent reads share a prefix. Interruption must finish
   * native use of that prefix before the read effect exits.
   */
  readonly read: (
    prefix: Prefix,
    canvas: Uint32Array,
    slot: number,
    labels: Uint32Array
  ) => Effect.Effect<ReadonlyArray<number>, Error>
}

/**
 * Runtime, scaffold preparation and cache capacity for one model.
 *
 * @since 0.1.0
 * @category models
 */
export interface Options<Prefix, Error, PrepareError> {
  /** Fixed model/checkpoint identity for this engine and its injected runtime. */
  readonly modelId: string
  readonly runtime: Runtime<Prefix, Error>
  readonly prepare: (plan: Planner.QuestionPlan) => Effect.Effect<Scaffold.Scaffold, PrepareError>
  readonly maxActiveQuestions: number
  readonly maxCachedPrefixes: number
  readonly maxPrefixTokens: number
  readonly cacheIdleMs: number
}

const Settings = Schema.Struct({ seed: Schema.String, reads: Schema.Literals([1, 4]) })

/**
 * Completed backend operations and current immutable-prefix cache residency.
 *
 * @since 0.1.0
 * @category models
 */
export interface Statistics {
  readonly prefills: number
  readonly livePrefixes: number
  readonly reads: number
  readonly cachedPrefixes: number
}

/**
 * A scoped decision engine with caller-linked cancellation and independent reads.
 *
 * @since 0.1.0
 * @category models
 */
export interface Engine<E> {
  readonly run: <Input>(input: Input, settings: typeof Settings.Type) => Effect.Effect<
    Contract.Response,
    E | EngineError | Noise.NoiseError | Readout.ReadoutError | Contract.ValidationError
  >
  readonly statistics: Effect.Effect<Statistics>
}

/**
 * Schedule independent one-step reads against an injected model runtime. Exact
 * whole-prompt token arrays share a prefix within this fixed model instance.
 * Capacity includes pending prefills; eviction is least-recently-used and idle
 * only. TTL starts when the last lease returns. A failed or abandoned prefill
 * releases partial resources and is never retained as an idle cache entry.
 *
 * The engine owns active call fibers. Closing its scope rejects new calls,
 * interrupts and awaits existing calls, then releases idle prefixes. Canceling
 * one caller releases only its leases; other callers can finish using theirs.
 *
 * @since 0.1.0
 * @category models
 */
export const make = <Prefix, Error, PrepareError>(
  options: Options<Prefix, Error, PrepareError>
): Effect.Effect<Engine<Error | PrepareError>, EngineError, Scope.Scope> =>
  Effect.gen(function*() {
    const { modelId, runtime, prepare, maxActiveQuestions, maxCachedPrefixes, maxPrefixTokens, cacheIdleMs } = options

    for (
      const [name, value] of Object.entries({ maxActiveQuestions, maxCachedPrefixes, maxPrefixTokens, cacheIdleMs })
    ) {
      if (!Number.isSafeInteger(value) || value < 1) {
        return yield* new EngineError({ message: name + " must be a positive safe integer" })
      }
    }

    if (maxActiveQuestions > maxCachedPrefixes) {
      return yield* new EngineError({ message: "Active questions must fit in the prefix cache" })
    }

    return yield* Effect.uninterruptible(Effect.gen(function*() {
      const calls = yield* Scope.make("parallel")
      const background = yield* Scope.make("parallel")
      const slots = Semaphore.makeUnsafe(maxActiveQuestions)
      const mutex = Semaphore.makeUnsafe(1)
      let prefills = 0
      let livePrefixes = 0
      let reads = 0
      let closed = false

      interface Entry {
        readonly key: string
        readonly resources: Scope.Closeable
        readonly result: Deferred.Deferred<Prefix, Error>
        leases: number
        state: "loading" | "ready" | "failed"
        worker: Fiber.Fiber<void> | undefined
        timer: Fiber.Fiber<void> | undefined
      }
      // Map insertion order is LRU order. Move entries to the end on acquisition
      // and last release. All pool operations, including cleanup, hold the mutex.
      const entries = new Map<string, Entry>()

      const cancelTimer = (entry: Entry) =>
        Effect.gen(function*() {
          const timer = entry.timer
          entry.timer = undefined

          if (timer !== undefined) yield* Fiber.interrupt(timer)
        })

      const dispose = (entry: Entry) =>
        Effect.gen(function*() {
          yield* cancelTimer(entry)

          if (entry.worker !== undefined) yield* Fiber.interrupt(entry.worker)

          yield* Scope.close(entry.resources, Exit.void).pipe(Effect.ensuring(Effect.sync(() => {
            if (entry.state === "ready") livePrefixes--

            entries.delete(entry.key)
          })))
        })

      const acquire = (ids: Uint32Array) =>
        mutex.withPermit(Effect.uninterruptible(Effect.gen(function*() {
          if (closed) return yield* new EngineError({ message: "Engine scope is closed" })

          const key = JSON.stringify(Array.from(ids))
          const cached = entries.get(key)

          if (cached !== undefined) {
            yield* cancelTimer(cached)
            cached.leases++
            entries.delete(key)
            entries.set(key, cached)

            return cached
          }

          if (entries.size === maxCachedPrefixes) {
            const idle = [...entries.values()].find((entry) => entry.leases === 0)

            // Each active question has at most one lease, and this question already
            // holds a slot. maxActiveQuestions <= capacity guarantees an idle entry.
            if (idle === undefined) return yield* new EngineError({ message: "No idle prefix cache slot" })

            yield* dispose(idle)
          }

          const entry: Entry = {
            key,
            resources: yield* Scope.make(),
            result: yield* Deferred.make<Prefix, Error>(),
            leases: 1,
            state: "loading",
            worker: undefined,
            timer: undefined
          }

          entries.set(key, entry)
          // Acquisition belongs to the pool, not the first borrower. A borrower
          // cancellation can interrupt this worker only after the last lease leaves.
          entry.worker = yield* Effect.forkIn(
            Effect.uninterruptibleMask((restore) =>
              Effect.gen(function*() {
                const exit = yield* Effect.exit(
                  restore(Scope.provide(Effect.suspend(() => runtime.prefill(ids)), entry.resources)).pipe(
                    Effect.onExit((exit) => Exit.isFailure(exit) ? Scope.close(entry.resources, exit) : Effect.void)
                  )
                )

                if (Exit.isFailure(exit)) {
                  entry.state = "failed"
                } else {
                  entry.state = "ready"
                  prefills++
                  livePrefixes++
                }

                yield* Deferred.done(entry.result, exit)
              })
            ),
            background
          )

          return entry
        })))

      const release = (entry: Entry) =>
        mutex.withPermit(Effect.uninterruptible(Effect.gen(function*() {
          entry.leases--

          if (entry.leases !== 0) return

          // Shutdown drains all call fibers before disposing the remaining pool.
          if (closed) return

          if (entry.state !== "ready") return yield* dispose(entry)

          entries.delete(entry.key)
          entries.set(entry.key, entry)
          entry.timer = yield* Effect.forkIn(
            Effect.sleep(cacheIdleMs).pipe(Effect.andThen(
              mutex.withPermit(Effect.uninterruptible(Effect.gen(function*() {
                // A hit cancels this timer under the mutex before adding its lease.
                entry.timer = undefined
                yield* dispose(entry)
              })))
            )),
            background
          )
        })))

      yield* Effect.addFinalizer((exit) =>
        Effect.gen(function*() {
          closed = true
          yield* Scope.close(calls, exit).pipe(
            Effect.ensuring(Scope.close(background, exit)),
            Effect.ensuring(
              mutex.withPermit(Effect.suspend(() => Effect.forEach([...entries.values()], dispose, { discard: true })))
            )
          )
        })
      )

      const execute = <Input>(input: Input, settings: typeof Settings.Type) =>
        Effect.gen(function*() {
          const config = yield* Schema.decodeUnknownEffect(Settings, { onExcessProperty: "error" })(settings).pipe(
            Effect.mapError((error) => new EngineError({ message: error.message }))
          )

          const plan = yield* Planner.plan(input, { modelId })
          // Caller IDs route outputs; duplicate semantic questions execute once.
          const unique = new Map(plan.routes.map((route) => [route.decision.semanticKey, route.decision]))

          const results = yield* Effect.forEach([...unique.values()], (question) =>
            slots.withPermit(Effect.scoped(Effect.gen(function*() {
              const scaffold = yield* prepare(question)

              if (scaffold.model !== plan.model) {
                return yield* new EngineError({ message: "Prepared scaffold uses a different model" })
              }

              const prefixIds = scaffold.prefixIds

              if (prefixIds.length === 0 || prefixIds.length > maxPrefixTokens) {
                return yield* new EngineError({ message: "Prompt exceeds the configured token bound" })
              }

              const entry = yield* Effect.acquireRelease(acquire(prefixIds), release)
              const prefix = yield* Deferred.await(entry.result)

              const logits = yield* Effect.forEach(
                Array.from({ length: config.reads }, (_, index) => index),
                (readIndex) =>
                  Effect.gen(function*() {
                    // prepareRead keys noise from actual scaffold identity, independently
                    // of the provisional planner key used for host-side result routing.
                    const prepared = yield* Scaffold.prepareRead(scaffold, { seed: config.seed, readIndex })

                    const result = yield* runtime.read(
                      prefix,
                      prepared.canvasIds,
                      prepared.slot,
                      prepared.allowedTokenIds
                    )

                    reads++

                    return result
                  }),
                { concurrency: config.reads }
              )

              return {
                key: question.semanticKey,
                logits,
                inputTokens: prefixIds.length,
                canvasTokens: scaffold.canvasIds.length * config.reads
              }
            }))), { concurrency: maxActiveQuestions })

          const usage: Contract.Usage = {
            // Logical work per distinct semantic decision, including cache hits.
            // Output counts evaluated canvas rows, not generated text tokens.
            input_tokens: results.reduce((total, result) =>
              total + result.inputTokens, 0),
            output_tokens: results.reduce((total, result) =>
              total + result.canvasTokens, 0)
          }

          return yield* Readout.responseFromLogits(
            plan,
            new Map(results.map((result) => [result.key, result.logits])),
            usage
          )
        })

      const run = <Input>(input: Input, settings: typeof Settings.Type) =>
        Effect.uninterruptibleMask((restore) =>
          Effect.gen(function*() {
            if (closed) return yield* new EngineError({ message: "Engine scope is closed" })

            const fiber = yield* Effect.forkIn(execute(input, settings), calls)

            // forkIn gives the engine ownership; this link also propagates caller
            // interruption and awaits lease cleanup before the caller can exit.
            return yield* restore(Fiber.join(fiber)).pipe(Effect.onInterrupt(() => Fiber.interrupt(fiber)))
          })
        )

      return Object.freeze({
        run,
        /** Successful backend operations and current pool residency, including pending entries. */
        statistics: Effect.sync(() => Object.freeze({ prefills, livePrefixes, reads, cachedPrefixes: entries.size }))
      })
    }))
  })

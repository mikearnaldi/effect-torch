/** Private host-boundary diagnostic; absent from the public Runtime API. */
import { Runtime } from "@effect-torch/core"
import { Effect } from "effect"

export interface Chain97Request {
  readonly body: Runtime.ExecutableHandle
  readonly head?: Runtime.ExecutableHandle | undefined
  readonly sampler: Runtime.ExecutableHandle
  readonly canvas: Uint32Array
  readonly bindingsWithoutCanvas: ReadonlyArray<Runtime.ConcreteTensorHandle>
  readonly state: Runtime.ReadOnlyStateInvocation
  readonly temperature: number
}

export interface Chain97Result {
  readonly feedback: Runtime.ConcreteTensorHandle
  readonly statistics: Float32Array
}

type Execute97 = (request: Chain97Request) => Effect.Effect<Chain97Result, Runtime.BackendError>
const implementations = new WeakMap<object, Execute97>()

export const register97 = (identity: Runtime.RuntimeService["identity"], execute: Execute97): void => {
  implementations.set(identity, execute)
}

export const execute97 = (
  runtime: Runtime.RuntimeService,
  request: Chain97Request
): Effect.Effect<Chain97Result, Runtime.BackendError> =>
  Effect.suspend(() => {
    const execute = implementations.get(runtime.identity)
    return execute === undefined
      ? Effect.fail(
        new Runtime.BackendError({
          reason: "unsupported-operation",
          backend: "@effect-torch/backend-cuda",
          operation: "executeChain97",
          phase: "execute",
          message: "Private chain97 bridge is unavailable for this runtime"
        })
      )
      : execute(request)
  })

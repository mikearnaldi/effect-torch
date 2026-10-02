/** Private diagnostic bridge; deliberately absent from the public Runtime API. */
import { Runtime } from "@effect-torch/core"
import { Effect } from "effect"

export interface Chain96Request {
  readonly body: Runtime.ExecutableHandle
  readonly head?: Runtime.ExecutableHandle | undefined
  readonly sampler: Runtime.ExecutableHandle
  readonly bindings: ReadonlyArray<Runtime.ConcreteTensorHandle>
  readonly state: Runtime.ReadOnlyStateInvocation
  readonly temperature: number
}

type Execute96 = (
  request: Chain96Request
) => Effect.Effect<Array<Runtime.ConcreteTensorHandle>, Runtime.BackendError>

const implementations = new WeakMap<object, Execute96>()

export const register96 = (identity: Runtime.RuntimeService["identity"], execute: Execute96): void => {
  implementations.set(identity, execute)
}

export const execute96 = (
  runtime: Runtime.RuntimeService,
  request: Chain96Request
): Effect.Effect<Array<Runtime.ConcreteTensorHandle>, Runtime.BackendError> =>
  Effect.suspend(() => {
    const execute = implementations.get(runtime.identity)
    return execute === undefined
      ? Effect.fail(
        new Runtime.BackendError({
          reason: "unsupported-operation",
          backend: "@effect-torch/backend-cuda",
          operation: "executeChain96",
          phase: "execute",
          message: "Private chain96 bridge is unavailable for this runtime"
        })
      )
      : execute(request)
  })

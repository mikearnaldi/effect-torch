/** Private combined-program diagnostic; absent from the public Runtime API. */
import { Runtime } from "@effect-torch/core"
import { Effect } from "effect"

export interface RequestRng99 {
  readonly initial: Runtime.ExecutableHandle
  readonly refinement: Runtime.ExecutableHandle
  readonly seed: number
}
type Fork99 = (
  request: RequestRng99
) => Effect.Effect<readonly [Runtime.ExecutableHandle, Runtime.ExecutableHandle], Runtime.BackendError>
const implementations = new WeakMap<object, Fork99>()
export const registerRequestRng99 = (identity: Runtime.RuntimeService["identity"], fork: Fork99): void => {
  implementations.set(identity, fork)
}
export const forkRequestRng99 = (runtime: Runtime.RuntimeService, request: RequestRng99): ReturnType<Fork99> =>
  Effect.suspend(() => {
    const fork = implementations.get(runtime.identity)
    return fork === undefined
      ? Effect.fail(
        new Runtime.BackendError({
          reason: "unsupported-operation",
          backend: "@effect-torch/backend-cuda",
          operation: "forkRequestRng99",
          phase: "execute",
          message: "Private request RNG99 bridge is unavailable for this runtime"
        })
      )
      : fork(request)
  })

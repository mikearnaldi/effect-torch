import { Runtime, Tensor } from "@effect-torch/core"
import { expect, it } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { vi } from "vitest"
import { isAvailable, layer } from "../src/index.ts"
import { createRuntimeAdapter } from "../src/internal/adapter.ts"
import type {
  NativeAddon,
  NativeCompileOptions,
  NativeCurrentBlockAttention,
  NativeDecodeOutputSelection
} from "../src/internal/native-addon.js"

it.effect("uploads, adds, reads back, and releases f32 tensors on CUDA", () =>
  Effect.flatMap(isAvailable, (available) => {
    if (!available) return Effect.void
    return Effect.gen(function*() {
      const a = yield* Tensor.fromTypedArray(new Float32Array([1, 2, 3, 4]), [2, 2])
      const b = yield* Tensor.fromTypedArray(new Float32Array([10, 20, 30, 40]), [2, 2])
      const sum = yield* Tensor.add(a, b)
      const [output] = yield* Tensor.compute([sum])

      expect(output.placement.id).toBe("cuda:0")
      const values = yield* Tensor.toTypedArray(output)
      expect(Array.from<number | bigint>(values).map(Number)).toEqual([11, 22, 33, 44])
      const downstream = yield* Tensor.add(output, output)

      yield* Tensor.clear(output)
      expect(Array.from<number | bigint>(values).map(Number)).toEqual([11, 22, 33, 44])
      expect(Exit.isFailure(yield* Effect.exit(Tensor.toTypedArray(output)))).toBe(true)
      expect(Exit.isFailure(yield* Effect.exit(Tensor.compute([downstream])))).toBe(true)

      const empty = yield* Tensor.ones([0])
      expect((yield* Tensor.toTypedArray(empty)).length).toBe(0)

      const filled = yield* Tensor.full([3], 2.5)
      expect(Array.from<number | bigint>(yield* Tensor.toTypedArray(filled)).map(Number)).toEqual([2.5, 2.5, 2.5])
    }).pipe(Effect.provide(layer()))
  }))

it.effect("supports logical dtypes, broadcasting, and multiplication", () =>
  Effect.flatMap(isAvailable, (available) => {
    if (!available) return Effect.void
    return Effect.gen(function*() {
      const precise = yield* Tensor.ones([2], { dtype: "f64" })
      expect(Array.from<number | bigint>(yield* Tensor.toTypedArray(precise)).map(Number)).toEqual([1, 1])

      const a = yield* Tensor.ones([2, 1])
      const b = yield* Tensor.ones([1, 2])
      const broadcast = yield* Tensor.add(a, b)
      expect(Array.from<number | bigint>(yield* Tensor.toTypedArray(broadcast)).map(Number)).toEqual([2, 2, 2, 2])

      const product = yield* Tensor.mul(yield* Tensor.full([2], 2), yield* Tensor.full([2], 3))
      expect(Array.from<number | bigint>(yield* Tensor.toTypedArray(product)).map(Number)).toEqual([6, 6])
    }).pipe(Effect.provide(layer()))
  }))

it.effect("validates device ordinals before loading the addon", () =>
  Effect.gen(function*() {
    const exit = yield* Effect.exit(Runtime.Runtime.pipe(Effect.provide(layer({ device: -1 }))))
    expect(Exit.isFailure(exit)).toBe(true)
  }))

it.effect("clears unpublished native outputs after interruption", () =>
  Effect.gen(function*() {
    let started = yield* Deferred.make<void>()
    const clear = vi.fn()
    class CancellationTokenDouble {
      cancelled = false

      cancel() {
        this.cancelled = true
      }
    }
    class LazyTensorDouble {
      readonly shape = [1]
      readonly dtype = "f32"
      readonly device = "cuda:0"

      exposures() {
        return []
      }
    }
    class NativeTensorDouble extends LazyTensorDouble {
      constructor(private readonly onClear: () => void = () => {}) {
        super()
      }

      clear() {
        this.onClear()
      }

      writeBytes() {}

      readback() {
        return Promise.resolve(Buffer.alloc(0))
      }

      sample() {
        return Promise.resolve(0)
      }
    }
    class NativeExposureDouble {
      readonly name = "test"
      readonly tensor = new LazyTensorDouble()
    }
    let resolve!: (values: Array<NativeTensorDouble>) => void
    class ExecutableDouble {
      readonly stateful = false
      readonly batch = 0
      readonly allowsWindowEviction = false
      readonly layers = 0
      readonly kvHeads = 0
      readonly headDim = 0
      readonly kdaLayers = 0
      readonly kdaHeads = 0
      readonly kdaHeadDim = 0
      readonly kdaValueDim = 0
      readonly convLayers = 0
      readonly convChannels = 0
      readonly convKernel = 0
      readonly device = "cuda:0"
      readonly instructionCount = 1
      readonly diagnostics = {
        semanticNodesBeforeOptimization: 1,
        semanticNodesAfterOptimization: 1,
        instructions: [{ kind: "value", count: 1 }],
        pipelineCount: 0,
        commandCount: 0,
        synchronizationCount: 1,
        memory: {
          externalBytes: 0,
          persistentBytes: 0,
          stateBytes: 0,
          outputBytes: 0,
          workspaceBytes: 0,
          transactionBytes: 0,
          peakLiveBytes: 0,
          packingOverheadBytes: 0
        },
        compilePhases: [{ phase: "graph_index", nanoseconds: 1 }]
      }

      execute() {
        Effect.runSync(Deferred.succeed(started, undefined))
        return new Promise<Array<NativeTensorDouble>>((resume) => resolve = resume)
      }

      executeStateful() {
        return Promise.resolve<Array<NativeTensorDouble>>([])
      }

      executeSampled() {
        return Promise.resolve<Array<number>>([])
      }

      executeSampledSteps() {
        return Promise.resolve<Array<Array<number>>>([])
      }

      executeTargetMatching() {
        return Promise.resolve({ pages: [], accepted: [], outputs: [] })
      }
    }
    class NativeKvSequenceDouble {
      readonly cursor = 0

      fork() {
        return new NativeKvSequenceDouble()
      }

      release() {}
      prefillMatch() {
        return 0
      }
    }
    class NativeKvPoolDouble {
      readonly capacity = 1
      readonly freeBlocks = 1
      readonly cachedBlocks = 0

      makeSequence() {
        return new NativeKvSequenceDouble()
      }
    }
    class NativeTargetMatchingOutputDouble {
      readonly pages: Array<Array<number>> = []
      readonly accepted: Array<number> = []
      readonly outputs: Array<NativeTensorDouble> = []
    }
    const executable = new ExecutableDouble()
    let compiledOptions: NativeCompileOptions | undefined
    class CudaRuntimeDouble {
      readonly device = "cuda:0"

      constant() {
        return new LazyTensorDouble()
      }

      zeros() {
        return new LazyTensorDouble()
      }

      ones() {
        return new LazyTensorDouble()
      }

      full() {
        return new LazyTensorDouble()
      }

      fromBytes() {
        return new LazyTensorDouble()
      }

      uploadBytes() {
        return new NativeTensorDouble()
      }

      fromMaterialized() {
        return new LazyTensorDouble()
      }

      add() {
        return new LazyTensorDouble()
      }

      graphNode() {
        return new LazyTensorDouble()
      }

      compile(_roots: Array<LazyTensorDouble>, options?: NativeCompileOptions) {
        compiledOptions = options
        return executable
      }
    }
    // SAFETY: these test doubles reproduce the generated N-API string enums exactly.
    const native: NativeAddon = {
      CancellationToken: CancellationTokenDouble,
      CudaRuntime: CudaRuntimeDouble,
      Executable: ExecutableDouble,
      LazyTensor: LazyTensorDouble,
      NativeCurrentBlockAttention: {
        Causal: "Causal" as NativeCurrentBlockAttention.Causal,
        Bidirectional: "Bidirectional" as NativeCurrentBlockAttention.Bidirectional
      },
      NativeDecodeOutputSelection: {
        AllRows: "AllRows" as NativeDecodeOutputSelection.AllRows,
        SplitLastTokenRow: "SplitLastTokenRow" as NativeDecodeOutputSelection.SplitLastTokenRow,
        BatchedLastTokenRow: "BatchedLastTokenRow" as NativeDecodeOutputSelection.BatchedLastTokenRow
      },
      NativeExposure: NativeExposureDouble,
      NativeKvPool: NativeKvPoolDouble,
      NativeKvSequence: NativeKvSequenceDouble,
      NativeTargetMatchingOutput: NativeTargetMatchingOutputDouble,
      NativeTensor: NativeTensorDouble,
      deviceCount: () => 1,
      grad: () => [new LazyTensorDouble()],
      isAvailable: () => true,
      inspectGguf: () => Promise.resolve({ metadata: [], tensors: [] }),
      loadGgufForDevice: () => Promise.resolve({ entries: [] }),
      loadTensors: () => Promise.resolve({ entries: [], metadata: {} }),
      saveTensors: () => Promise.resolve()
    }
    const runtime = createRuntimeAdapter(native, 0)
    const root = yield* runtime.node({ op: "zeros", inputs: [], attributes: { shape: [1], dtype: "f32" } })
    const program = yield* runtime.compile({
      roots: [root],
      options: { optimize: false, constantWeights: true }
    })
    expect(compiledOptions).toEqual({ optimize: false, constantWeights: true })
    expect(program.diagnostics.instructions).toEqual([{ kind: "value", count: 1 }])
    expect(program.diagnostics.compilePhases).toEqual([{ phase: "graph_index", nanoseconds: 1 }])
    const fiber = yield* runtime.execute(program, {
      bindings: [],
      scalars: [],
      runtimeValues: {}
    }).pipe(Effect.forkChild({ startImmediately: true }))
    yield* Deferred.await(started)
    const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
    yield* Effect.sync(() => resolve([new NativeTensorDouble(clear)]))
    yield* Fiber.join(interruption)
    yield* Effect.promise(() => Promise.resolve())

    expect(clear).toHaveBeenCalledTimes(1)

    started = yield* Deferred.make<void>()
    const duplicateClear = vi.fn()
    const duplicateProgram = yield* runtime.compile({ roots: [root, root] })
    const duplicateFiber = yield* runtime.execute(duplicateProgram, {
      bindings: [],
      scalars: [],
      runtimeValues: {}
    }).pipe(Effect.forkChild({ startImmediately: true }))
    yield* Deferred.await(started)
    const duplicate = new NativeTensorDouble(duplicateClear)
    yield* Effect.sync(() => resolve([duplicate, duplicate]))
    expect(Exit.isFailure(yield* Fiber.await(duplicateFiber))).toBe(true)
    expect(duplicateClear).toHaveBeenCalledTimes(1)
  }))

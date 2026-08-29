// JSONL benchmark of native executable compilation and structural-cache lookup.
// Each cold/warm pair builds the same graph structure twice with a unique salt.
// The first compile misses the process cache. The second can reuse its structural
// artifact. Graph construction is not timed, and the executables never run.
// Native pipeline preparation remains part of the cold sample. Pairs run
// serially. The output reports medians and diagnostics for the cold artifact.
// `compilePhases` come from the cold compile, not the warm lookup.
//
// The compiler snapshots EFFECT_TORCH_* switches in the compile options and
// cache identity. The benchmark rejects EFFECT_TORCH_NO_EXECUTABLE_CACHE whenever
// it is present, even with an empty or "0" value. It also rejects constant
// weights. Either setting invalidates the warm-cache measurement.

import { Gradient, Optimizer, Runtime, Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import { arch, cpus, platform, release, totalmem } from "node:os"
import { performance } from "node:perf_hooks"

const workloadNames = ["elementwise", "wide", "training", "decode"] as const
type WorkloadName = (typeof workloadNames)[number]
type RuntimeName = "cpu" | "metal"

interface Config {
  readonly workloads: ReadonlyArray<WorkloadName>
  readonly runtimes: ReadonlyArray<RuntimeName>
  readonly iterations: number
  readonly size: number | undefined
  readonly options: Runtime.ExecutableCompileOptions
}

interface GraphSpec {
  readonly roots: ReadonlyArray<Tensor.Any>
  readonly state?: Runtime.DecodeStateRequest
}

const defaultSizes = {
  elementwise: 64,
  wide: 24,
  training: 3,
  decode: 4
} satisfies Readonly<Record<WorkloadName, number>>

const positiveInteger = (value: string, name: string): number => {
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer, got ${JSON.stringify(value)}`)
  }
  return parsed
}

const commaList = <A extends string>(
  value: string,
  all: ReadonlyArray<A>,
  name: string
): Array<A> => {
  const output: Array<A> = []
  for (const item of value.split(",")) {
    if (item === "all") {
      output.push(...all)
    } else {
      const match = all.find((candidate) => candidate === item)
      if (match !== undefined) {
        output.push(match)
        continue
      }
      throw new Error(`unknown ${name} ${JSON.stringify(item)}; expected ${all.join(", ")}, or all`)
    }
  }
  return output
}

const parseBoolean = (value: string, name: string): boolean => {
  if (value === "true") return true
  if (value === "false") return false
  throw new Error(`${name} must be true or false, got ${JSON.stringify(value)}`)
}

const printHelp = (): void => {
  process.stdout.write(
    `effect-torch native compile benchmark\n\n` +
      `Usage: pnpm --filter @effect-torch/bench bench:compile -- [OPTIONS]\n\n` +
      `Options:\n` +
      `  --workload NAME[,NAME]  elementwise, wide, training, decode, or all\n` +
      `  --runtime NAME[,NAME]   cpu, metal, or all (default: all)\n` +
      `  --size N                Override the selected workload size\n` +
      `  --iterations N          Cold/warm sample pairs (default: 5)\n` +
      `  --optimize BOOL         Compile with optimization enabled or disabled\n` +
      `  --constant-weights      Rejected: constant weights disable the structural cache\n` +
      `  -h, --help              Print this help\n\n` +
      `Measurement protocol:\n` +
      `  cold_native_compile is a full runtime.compile cache miss.\n` +
      `  warm_structural_cache_lookup recompiles an identical graph through the executable cache.\n` +
      `  EFFECT_TORCH_NO_EXECUTABLE_CACHE and constant weights are rejected in this mode.\n` +
      `  Compile phase timings are native diagnostics from the cold compile; pipeline_preparation,\n` +
      `  when reported, is not a third wall-clock benchmark measurement.\n`
  )
}

const parseConfig = (args: ReadonlyArray<string>): Config | undefined => {
  let workloads: Array<WorkloadName> = []
  let runtimes: Array<RuntimeName> = []
  let iterations = 5
  let size: number | undefined
  let optimize: boolean | undefined
  for (let index = 0; index < args.length; index++) {
    const argument = args[index]
    const value = (): string => {
      const next = args[++index]
      if (next === undefined) throw new Error(`${argument} requires a value`)
      return next
    }
    switch (argument) {
      case "--":
        break
      case "--workload":
      case "-w":
        workloads.push(...commaList(value(), workloadNames, "workload"))
        break
      case "--runtime":
      case "-r":
        runtimes.push(...commaList(value(), ["cpu", "metal"] as const, "runtime"))
        break
      case "--size":
      case "-s":
        size = positiveInteger(value(), "size")
        break
      case "--iterations":
      case "-n":
        iterations = positiveInteger(value(), "iterations")
        break
      case "--optimize":
        optimize = parseBoolean(value(), "optimize")
        break
      case "--constant-weights":
        throw new Error(
          "--constant-weights cannot be used by the cold/warm benchmark because it disables structural caching"
        )
      case "--help":
      case "-h":
        printHelp()
        return undefined
      default:
        throw new Error(`unknown argument ${JSON.stringify(argument)}; use --help`)
    }
  }
  if (workloads.length === 0) workloads = [...workloadNames]
  if (runtimes.length === 0) runtimes = ["cpu", "metal"]
  return {
    workloads,
    runtimes,
    iterations,
    size,
    options: optimize === undefined ? {} : { optimize }
  }
}

const buildElementwise = (
  size: number,
  salt: number
): Effect.Effect<GraphSpec, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    let current: Tensor.Any = yield* Tensor.zeros([1024])
    const other = yield* Tensor.full([1024], 0.5)
    current = yield* Tensor.add(current, yield* Tensor.full([], salt))
    for (let index = 0; index < size; index++) {
      switch (index % 5) {
        case 0:
          current = yield* Tensor.add(current, other)
          break
        case 1:
          current = yield* Tensor.tanh(current)
          break
        case 2:
          current = yield* Tensor.mul(current, other)
          break
        case 3:
          current = yield* Tensor.sin(current)
          break
        default:
          current = yield* Tensor.relu(current)
      }
    }
    return { roots: [current] }
  })

const buildWide = (
  size: number,
  salt: number
): Effect.Effect<GraphSpec, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    const x = yield* Tensor.zeros([256])
    const y = yield* Tensor.full([256], 0.25)
    let prefix: Tensor.Any = yield* Tensor.tanh(yield* Tensor.add(x, y))
    prefix = yield* Tensor.add(prefix, yield* Tensor.full([], salt))
    const roots: Array<Tensor.Any> = [prefix]
    for (let index = 0; index < size; index++) {
      const coefficient = yield* Tensor.full([], (index + 1) / (size + 1))
      let branch: Tensor.Any = yield* Tensor.mul(prefix, coefficient)
      branch = index % 2 === 0 ? yield* Tensor.sin(branch) : yield* Tensor.tanh(branch)
      roots.push(yield* Tensor.add(branch, y))
    }
    return { roots }
  })

const buildTraining = (
  size: number,
  salt: number
): Effect.Effect<GraphSpec, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    let hidden: Tensor.Any = yield* Tensor.zeros([16, 64])
    const parameters: Array<Tensor.Any> = []
    for (let index = 0; index < size; index++) {
      const weight = yield* Tensor.full([64, 64], (index + 1) / 1024)
      const bias = yield* Tensor.zeros([64])
      hidden = yield* Tensor.gelu(
        yield* Tensor.linear(hidden, weight, bias),
        { approximate: "tanh" }
      )
      parameters.push(weight, bias)
    }
    let loss: Tensor.Any = yield* Tensor.mean(hidden)
    loss = yield* Tensor.add(loss, yield* Tensor.full([], salt))
    const gradients = yield* Gradient.grad(loss, parameters)
    const optimizer = yield* Optimizer.adamW()
    const state = yield* optimizer.init(parameters)
    const update = yield* optimizer.step(
      parameters,
      gradients,
      state,
      yield* Tensor.full([], 1e-3)
    )
    return { roots: [loss, ...update.params, ...update.stateRoots] }
  })

const buildDecode = (
  size: number,
  salt: number
): Effect.Effect<GraphSpec, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    const roots: Array<Tensor.Any> = []
    for (let index = 0; index < size; index++) {
      const q = yield* Tensor.rotaryEmbedding(
        yield* Tensor.full([1, 4, 1, 32], (index + 1) / 128),
        1,
        10_000
      )
      const k = yield* Tensor.rotaryEmbedding(
        yield* Tensor.full([1, 4, 1, 32], (index + 2) / 128),
        1,
        10_000
      )
      const v = yield* Tensor.full([1, 4, 1, 32], (index + 3) / 128)
      const attention = yield* Tensor.scaledDotProductAttention(q, k, v, { causal: true })
      const kdaQ = yield* Tensor.full([1, 4, 1, 32], (index + 4) / 128)
      const recurrence = yield* Tensor.kdaChunk(
        kdaQ,
        yield* Tensor.full([1, 4, 1, 32], (index + 5) / 128),
        yield* Tensor.full([1, 4, 1, 32], (index + 6) / 128),
        yield* Tensor.full([1, 4, 1, 32], -0.01),
        yield* Tensor.full([1, 4, 1, 1], 0.5)
      )
      let combined: Tensor.Any = yield* Tensor.add(attention, recurrence)
      if (index === 0) combined = yield* Tensor.add(combined, yield* Tensor.full([], salt))
      roots.push(combined)
      roots.push(
        yield* Tensor.shortConv1d(
          yield* Tensor.full([1, 1, 128], (index + 1) / 64),
          yield* Tensor.full([128, 3], 1 / 3)
        )
      )
    }
    return {
      roots,
      state: {
        maxTokens: 256,
        blockSize: 16,
        kvDtype: "f32",
        window: 128,
        batch: 1
      }
    }
  })

const buildWorkload = (
  workload: WorkloadName,
  size: number,
  salt: number
): Effect.Effect<GraphSpec, unknown, Runtime.Runtime> => {
  switch (workload) {
    case "elementwise":
      return buildElementwise(size, salt)
    case "wide":
      return buildWide(size, salt)
    case "training":
      return buildTraining(size, salt)
    case "decode":
      return buildDecode(size, salt)
  }
}

const median = (input: ReadonlyArray<number>): number => {
  const values = [...input].sort((left, right) => left - right)
  const middle = Math.floor(values.length / 2)
  return values.length % 2 === 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2
}

const rounded = (value: number): number => Number(value.toFixed(6))

const stableDiagnostics = (diagnostics: Runtime.ExecutableDiagnostics) => ({
  semanticNodesBeforeOptimization: diagnostics.semanticNodesBeforeOptimization,
  semanticNodesAfterOptimization: diagnostics.semanticNodesAfterOptimization,
  instructions: diagnostics.instructions,
  pipelineCount: diagnostics.pipelineCount,
  commandCount: diagnostics.commandCount,
  synchronizationCount: diagnostics.synchronizationCount,
  memory: diagnostics.memory
})

const benchmarkWorkload = (
  runtime: Runtime.RuntimeService,
  workload: WorkloadName,
  workloadIndex: number,
  size: number,
  config: Config
): Effect.Effect<unknown, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    const coldMilliseconds: Array<number> = []
    const warmMilliseconds: Array<number> = []
    const phaseMilliseconds = new Map<string, Array<number>>()
    let diagnostics: ReturnType<typeof stableDiagnostics> | undefined
    for (let iteration = 0; iteration < config.iterations; iteration++) {
      const salt = (workloadIndex + 1) * 1_000_000 + iteration
      const coldGraph = yield* buildWorkload(workload, size, salt)
      const coldStarted = performance.now()
      const cold = yield* runtime.compile({
        roots: coldGraph.roots,
        options: config.options,
        state: coldGraph.state
      })
      coldMilliseconds.push(performance.now() - coldStarted)

      const stable = stableDiagnostics(cold.diagnostics)
      if (diagnostics === undefined) diagnostics = stable
      else if (JSON.stringify(diagnostics) !== JSON.stringify(stable)) {
        throw new Error(`${workload} produced non-deterministic executable diagnostics`)
      }
      for (const phase of cold.diagnostics.compilePhases ?? []) {
        const samples = phaseMilliseconds.get(phase.phase) ?? []
        samples.push(phase.nanoseconds / 1_000_000)
        phaseMilliseconds.set(phase.phase, samples)
      }

      const warmGraph = yield* buildWorkload(workload, size, salt)
      const warmStarted = performance.now()
      yield* runtime.compile({
        roots: warmGraph.roots,
        options: config.options,
        state: warmGraph.state
      })
      warmMilliseconds.push(performance.now() - warmStarted)
    }
    const compilePhaseDiagnosticsMedianMs = Object.fromEntries(
      [...phaseMilliseconds.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([phase, samples]) => [phase, rounded(median(samples))])
    )
    return {
      kind: "result",
      runtime: {
        backend: runtime.backend.name,
        placement: runtime.placement,
        capabilities: runtime.capabilities
      },
      options: config.options,
      workload: { name: workload, size },
      iterations: config.iterations,
      measurements: {
        coldNativeCompile: {
          mode: "cold_native_compile",
          medianMs: rounded(median(coldMilliseconds))
        },
        warmStructuralCacheLookup: {
          mode: "warm_structural_cache_lookup",
          medianMs: rounded(median(warmMilliseconds))
        }
      },
      compilePhaseDiagnostics: {
        sourceMode: "cold_native_compile",
        medianMs: compilePhaseDiagnosticsMedianMs
      },
      executableDiagnostics: diagnostics
    }
  })

const writeJson = (json: string): void => {
  process.stdout.write(`${json}\n`)
}

const suite = (config: Config): Effect.Effect<void, unknown, Runtime.Runtime> =>
  Effect.gen(function*() {
    const runtime = yield* Runtime.Runtime
    for (let index = 0; index < config.workloads.length; index++) {
      const workload = config.workloads[index]
      const result = yield* benchmarkWorkload(
        runtime,
        workload,
        index,
        config.size ?? defaultSizes[workload],
        config
      )
      yield* Effect.sync(() => writeJson(JSON.stringify(result) ?? "undefined"))
    }
  })

const main = async (): Promise<void> => {
  const config = parseConfig(process.argv.slice(2))
  if (config === undefined) return
  if (process.env.EFFECT_TORCH_NO_EXECUTABLE_CACHE !== undefined) {
    throw new Error(
      "EFFECT_TORCH_NO_EXECUTABLE_CACHE must be unset when measuring warm structural cache lookups"
    )
  }
  const cpuInfo = cpus()
  writeJson(
    JSON.stringify({
      kind: "metadata",
      schema: "effect-torch.compile-benchmark.v2",
      machine: {
        platform: platform(),
        arch: arch(),
        release: release(),
        cpuModel: cpuInfo[0]?.model ?? "unknown",
        logicalCpus: cpuInfo.length,
        totalMemoryBytes: totalmem(),
        node: process.version
      },
      selection: {
        mode: "cold_native_compile_and_warm_structural_cache_lookup",
        runtimes: config.runtimes,
        workloads: config.workloads,
        iterations: config.iterations,
        ...(config.size === undefined ? { sizes: defaultSizes } : { size: config.size })
      },
      options: config.options
    }) ?? "undefined"
  )

  if (config.runtimes.includes("cpu")) {
    const BackendCpu = await import("@effect-torch/backend-cpu")
    await Effect.runPromise(Effect.provide(suite(config), BackendCpu.layer))
  }
  if (config.runtimes.includes("metal")) {
    const BackendApple = await import("@effect-torch/backend-apple-native")
    if (await Effect.runPromise(BackendApple.isAvailable)) {
      await Effect.runPromise(Effect.provide(suite(config), BackendApple.layer()))
    } else {
      writeJson(JSON.stringify({ kind: "skipped", runtime: "metal", reason: "unavailable" }) ?? "undefined")
    }
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`)
  process.exitCode = 1
})

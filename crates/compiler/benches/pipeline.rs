//! Compiler graph-analysis benchmark (invoked via `cargo bench --bench pipeline`).
//!
//! Measures only the compiler's frontend: `GraphIndex` construction plus
//! `OptimizationPlan` selection, over parameterized workloads (elementwise
//! chains, wide fan-out, training with autodiff + optimizer steps, decode
//! specialization, grouped optimizers, and 50k/100k-node stack-safety
//! stress). No lowering, memory planning, or backend work is performed.
//!
//! Beyond timing, each sample asserts the structural invariants the rest of
//! the pipeline depends on: exactly one graph-index build per program,
//! zero semantic nodes rebuilt by optimization, and identical structural
//! metrics across iterations (determinism). Run with `--help` for options.

use effect_torch_autodiff::grad;
use effect_torch_compiler::{specialize_decode, CompileOptions, GraphIndex, OptimizationPlan};
use effect_torch_graph::{AttentionWindow, Device, Node, NodeKind, PositionOffset, RotaryLayout};
use effect_torch_runtime::DType;
use std::env;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_ITERATIONS: usize = 7;

/// The parameterized workloads; `Stack50k`/`Stack100k` are stress-only and
/// run on a small-stack thread to prove the analysis is iterative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Workload {
    Elementwise,
    Wide,
    Training,
    Decode,
    Optimizer,
    Stack50k,
    Stack100k,
}

impl Workload {
    const DEFAULTS: [Self; 5] = [
        Self::Elementwise,
        Self::Wide,
        Self::Training,
        Self::Decode,
        Self::Optimizer,
    ];

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "elementwise" => Ok(Self::Elementwise),
            "wide" => Ok(Self::Wide),
            "training" => Ok(Self::Training),
            "decode" => Ok(Self::Decode),
            "optimizer" => Ok(Self::Optimizer),
            "stack-50k" => Ok(Self::Stack50k),
            "stack-100k" => Ok(Self::Stack100k),
            _ => Err(format!(
                "unknown workload {value:?}; use --list to see valid workloads"
            )),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::Wide => "wide",
            Self::Training => "training",
            Self::Decode => "decode",
            Self::Optimizer => "optimizer",
            Self::Stack50k => "stack-50k",
            Self::Stack100k => "stack-100k",
        }
    }

    const fn default_size(self) -> usize {
        match self {
            Self::Elementwise => 2_000,
            Self::Wide => 24,
            Self::Training => 8,
            Self::Decode => 8,
            Self::Optimizer => 128,
            Self::Stack50k => 50_000,
            Self::Stack100k => 100_000,
        }
    }

    const fn is_stress(self) -> bool {
        matches!(self, Self::Stack50k | Self::Stack100k)
    }
}

struct Config {
    workloads: Vec<Workload>,
    size: Option<usize>,
    iterations: usize,
}

impl Config {
    fn parse() -> Result<Option<Self>, String> {
        let mut args = env::args().skip(1);
        let mut workloads = Vec::new();
        let mut size = None;
        let mut iterations = DEFAULT_ITERATIONS;
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--bench" => {}
                "--workload" | "-w" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{argument} requires a value"))?;
                    add_workloads(&mut workloads, &value)?;
                }
                "--size" | "-s" => {
                    size = Some(parse_positive(
                        &args
                            .next()
                            .ok_or_else(|| format!("{argument} requires a value"))?,
                        "size",
                    )?);
                }
                "--iterations" | "-n" => {
                    iterations = parse_positive(
                        &args
                            .next()
                            .ok_or_else(|| format!("{argument} requires a value"))?,
                        "iterations",
                    )?;
                }
                "--list" => {
                    println!(
                        "elementwise\nwide\ntraining\ndecode\noptimizer\nstack-50k\nstack-100k"
                    );
                    return Ok(None);
                }
                "--help" | "-h" => {
                    print_help();
                    return Ok(None);
                }
                _ => return Err(format!("unknown argument {argument:?}; use --help")),
            }
        }
        if workloads.is_empty() {
            workloads.extend(Workload::DEFAULTS);
        }
        Ok(Some(Self {
            workloads,
            size,
            iterations,
        }))
    }
}

fn add_workloads(output: &mut Vec<Workload>, value: &str) -> Result<(), String> {
    for name in value.split(',') {
        if name == "default" {
            output.extend(Workload::DEFAULTS);
            continue;
        }
        if name == "all" {
            output.extend(Workload::DEFAULTS);
            output.extend([Workload::Stack50k, Workload::Stack100k]);
            continue;
        }
        if name == "stress" {
            output.extend([Workload::Stack50k, Workload::Stack100k]);
            continue;
        }
        output.push(Workload::parse(name)?);
    }
    Ok(())
}

fn parse_positive(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{name} must be a positive integer, got {value:?}"))
}

fn print_help() {
    println!(
        "effect-torch compiler graph-analysis benchmark\n\n\
Usage: cargo bench -p effect-torch-compiler --bench pipeline -- [OPTIONS]\n\n\
Options:\n  \
-w, --workload NAME[,NAME]  Workload, default, stress, or all (repeatable)\n  \
-s, --size N                Override the workload size (not stack-50k/stack-100k)\n  \
-n, --iterations N          Samples per workload (default: {DEFAULT_ITERATIONS})\n  \
    --list                  List workload names\n  \
-h, --help                  Print this help\n\n\
Measurement scope:\n  \
mode=graph_analysis measures GraphIndex construction plus OptimizationPlan selection.\n  \
graph_build_ms is reported separately and is excluded from analysis_ms.\n  \
No lowering, memory/physical planning, pipeline preparation, shader work, or backend compile is performed.\n  \
planned_lowering_units is plan structure, not a count of lowered instructions.\n\n\
default excludes the 50k/100k stack workloads; stress selects them; all includes them."
    );
}

/// Structural counters captured per sample; compared across iterations to
/// detect any nondeterminism in graph analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Metrics {
    semantic_nodes: usize,
    graph_index_builds: usize,
    graph_semantic_node_visits: usize,
    graph_edge_visits: usize,
    optimization_semantic_node_scans: usize,
    semantic_nodes_rebuilt: usize,
    fusion_candidates: usize,
    multi_output_work_items: usize,
    multi_output_dependency_edges: usize,
    multi_output_dependency_passes: usize,
    multi_output_dependency_edge_visits: usize,
    multi_output_dependency_queries: usize,
    region_table_merges: usize,
    selected_regions: usize,
    planned_lowering_units: usize,
}

/// One measured run: graph build time is kept separate from the analysis
/// phases it feeds, and `analysis_ns` is the scope reported as the headline
/// number (index + optimization).
struct Sample {
    graph_build_ns: u128,
    index_ns: u128,
    optimization_ns: u128,
    analysis_ns: u128,
    metrics: Metrics,
}

fn node(kind: NodeKind) -> Arc<Node> {
    Node::new(kind).expect("benchmark workload must construct a valid graph")
}

fn input(next_slot: &mut u32, shape: &[usize]) -> Arc<Node> {
    let slot = *next_slot;
    *next_slot = next_slot
        .checked_add(1)
        .expect("benchmark input slot overflow");
    node(NodeKind::Input {
        slot,
        shape: shape.to_vec(),
        dtype: DType::F32,
        device: Device::Cpu,
    })
}

fn full(shape: &[usize], value: f64) -> Arc<Node> {
    node(NodeKind::Full {
        shape: shape.to_vec(),
        value,
        dtype: DType::F32,
        device: Device::Cpu,
    })
}

fn build_elementwise(size: usize) -> Vec<Arc<Node>> {
    let mut slot = 0;
    let mut current = input(&mut slot, &[1024]);
    let other = input(&mut slot, &[1024]);
    for index in 0..size {
        current = match index % 6 {
            0 => node(NodeKind::Add {
                a: current,
                b: other.clone(),
            }),
            1 => node(NodeKind::Tanh { a: current }),
            2 => node(NodeKind::Mul {
                a: current,
                b: other.clone(),
            }),
            3 => node(NodeKind::Relu { a: current }),
            4 => node(NodeKind::Sin { a: current }),
            _ => node(NodeKind::Neg { a: current }),
        };
    }
    vec![current]
}

fn build_wide(size: usize) -> Vec<Arc<Node>> {
    let mut slot = 0;
    let x = input(&mut slot, &[256]);
    let y = input(&mut slot, &[256]);
    let z = input(&mut slot, &[256]);
    let sum = node(NodeKind::Add { a: x, b: y });
    let prefix = node(NodeKind::Tanh { a: sum });
    let mut roots = Vec::with_capacity(size + 1);
    roots.push(prefix.clone());
    for index in 0..size {
        let branch = match index % 4 {
            0 => node(NodeKind::Neg { a: prefix.clone() }),
            1 => node(NodeKind::Exp { a: prefix.clone() }),
            2 => node(NodeKind::Sin { a: prefix.clone() }),
            _ => node(NodeKind::Cos { a: prefix.clone() }),
        };
        let branch = node(NodeKind::Add {
            a: branch,
            b: z.clone(),
        });
        roots.push(node(NodeKind::Tanh { a: branch }));
    }
    roots
}

fn build_training(size: usize) -> Vec<Arc<Node>> {
    let mut slot = 0;
    let mut hidden = input(&mut slot, &[16, 64]);
    let mut parameters = Vec::with_capacity(size * 2);
    for _ in 0..size {
        let weight = input(&mut slot, &[64, 64]);
        let bias = input(&mut slot, &[64]);
        let linear = node(NodeKind::Linear {
            x: hidden.clone(),
            weight: weight.clone(),
            bias: bias.clone(),
        });
        hidden = node(NodeKind::Gelu {
            a: linear,
            approximate: true,
        });
        parameters.extend([weight, bias]);
    }
    let loss = node(NodeKind::Mean {
        a: hidden,
        dims: vec![0, 1],
        keepdims: false,
    });
    let gradients = grad(&loss, &parameters).expect("training workload autodiff must succeed");
    let lr = full(&[], 1e-3);
    let c1 = full(&[], 0.1);
    let c2 = full(&[], 0.001);
    let mut roots = Vec::with_capacity(1 + parameters.len() * 3);
    roots.push(loss);
    for (parameter, gradient) in parameters.into_iter().zip(gradients) {
        let m = input(&mut slot, &parameter.shape);
        let v = input(&mut slot, &parameter.shape);
        let step = node(NodeKind::AdamWStep {
            param: parameter,
            grad: gradient,
            m,
            v,
            lr: lr.clone(),
            c1: c1.clone(),
            c2: c2.clone(),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        });
        for index in 0..3 {
            roots.push(node(NodeKind::AdamWOut {
                step: step.clone(),
                index,
            }));
        }
    }
    roots
}

fn build_decode(size: usize) -> Result<Vec<Arc<Node>>, String> {
    let mut slot = 0;
    let mut roots = Vec::with_capacity(size * 2);
    for _ in 0..size {
        let q = input(&mut slot, &[1, 4, 1, 32]);
        let k = input(&mut slot, &[1, 4, 1, 32]);
        let v = input(&mut slot, &[1, 4, 1, 32]);
        let q = node(NodeKind::RotaryEmbedding {
            x: q,
            seq_len: 1,
            theta: 10_000.0,
            offset: PositionOffset::Absolute,
            layout: RotaryLayout::HalfSplit,
        });
        let k = node(NodeKind::RotaryEmbedding {
            x: k,
            seq_len: 1,
            theta: 10_000.0,
            offset: PositionOffset::Absolute,
            layout: RotaryLayout::HalfSplit,
        });
        let attention = node(NodeKind::Sdpa {
            q,
            k,
            v,
            scale: 32.0f64.sqrt().recip(),
            causal: true,
            window: AttentionWindow::Inherit,
        });

        let kda_q = input(&mut slot, &[1, 4, 1, 32]);
        let recurrence = node(NodeKind::KdaChunk {
            q: kda_q.clone(),
            k: input(&mut slot, &[1, 4, 1, 32]),
            v: input(&mut slot, &[1, 4, 1, 32]),
            log_decay: input(&mut slot, &[1, 4, 1, 32]),
            beta: input(&mut slot, &[1, 4, 1, 1]),
            scale: 32.0f64.sqrt().recip(),
        });
        roots.push(node(NodeKind::Add {
            a: attention,
            b: recurrence,
        }));
        roots.push(node(NodeKind::ShortConv1d {
            x: input(&mut slot, &[1, 1, 128]),
            weight: input(&mut slot, &[128, 3]),
        }));
    }
    specialize_decode(&roots, Some(128), 1, false).map(|(specialized, _)| specialized)
}

fn build_optimizer(size: usize) -> Vec<Arc<Node>> {
    let mut slot = 0;
    let lr = full(&[], 1e-3);
    let c1 = full(&[], 0.1);
    let c2 = full(&[], 0.001);
    let mut roots = Vec::with_capacity(size * 3);
    for _ in 0..size {
        let step = node(NodeKind::AdamWStep {
            param: input(&mut slot, &[1024]),
            grad: input(&mut slot, &[1024]),
            m: input(&mut slot, &[1024]),
            v: input(&mut slot, &[1024]),
            lr: lr.clone(),
            c1: c1.clone(),
            c2: c2.clone(),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        });
        for index in 0..3 {
            roots.push(node(NodeKind::AdamWOut {
                step: step.clone(),
                index,
            }));
        }
    }
    roots
}

fn build_stack(size: usize) -> Vec<Arc<Node>> {
    let mut slot = 0;
    let mut root = input(&mut slot, &[1]);
    for _ in 1..size {
        root = node(NodeKind::Neg { a: root });
    }
    vec![root]
}

fn build_workload(workload: Workload, size: usize) -> Result<Vec<Arc<Node>>, String> {
    match workload {
        Workload::Elementwise => Ok(build_elementwise(size)),
        Workload::Wide => Ok(build_wide(size)),
        Workload::Training => Ok(build_training(size)),
        Workload::Decode => build_decode(size),
        Workload::Optimizer => Ok(build_optimizer(size)),
        Workload::Stack50k | Workload::Stack100k => Ok(build_stack(size)),
    }
}

fn measure_once(workload: Workload, size: usize) -> Result<Sample, String> {
    let graph_started = Instant::now();
    let mut roots = build_workload(workload, size)?;
    let graph_build_ns = graph_started.elapsed().as_nanos();

    let analysis_started = Instant::now();
    let index_started = Instant::now();
    let index = GraphIndex::new(&roots)?;
    let index_ns = index_started.elapsed().as_nanos();
    let optimization_started = Instant::now();
    let plan = OptimizationPlan::select(&index, &CompileOptions::default())?;
    let optimization_ns = optimization_started.elapsed().as_nanos();
    let analysis_ns = analysis_started.elapsed().as_nanos();

    if index.work.graph_index_builds != 1 || plan.work.graph_index_builds != 1 {
        return Err(format!(
            "{} built {} graph indexes; expected exactly one",
            workload.name(),
            index
                .work
                .graph_index_builds
                .max(plan.work.graph_index_builds)
        ));
    }
    if plan.work.semantic_nodes_rebuilt != 0 {
        return Err(format!(
            "{} rebuilt {} semantic nodes; expected zero",
            workload.name(),
            plan.work.semantic_nodes_rebuilt
        ));
    }
    let metrics = Metrics {
        semantic_nodes: index.order.len(),
        graph_index_builds: index.work.graph_index_builds,
        graph_semantic_node_visits: index.work.semantic_node_visits,
        graph_edge_visits: index.work.graph_edge_visits,
        optimization_semantic_node_scans: plan.work.semantic_nodes_scanned,
        semantic_nodes_rebuilt: plan.work.semantic_nodes_rebuilt,
        fusion_candidates: plan.work.fusion_candidates,
        multi_output_work_items: plan.work.multi_output_work_items,
        multi_output_dependency_edges: plan.work.multi_output_dependency_edges,
        multi_output_dependency_passes: plan.work.multi_output_dependency_passes,
        multi_output_dependency_edge_visits: plan.work.multi_output_dependency_edge_visits,
        multi_output_dependency_queries: plan.work.multi_output_dependency_queries,
        region_table_merges: plan.work.region_table_merges,
        selected_regions: plan.work.selected_regions,
        planned_lowering_units: plan.lowering_order.len(),
    };

    drop(plan);
    roots.clear();
    drop(index);
    Ok(Sample {
        graph_build_ns,
        index_ns,
        optimization_ns,
        analysis_ns,
        metrics,
    })
}

fn measure(workload: Workload, size: usize) -> Result<Sample, String> {
    if !workload.is_stress() {
        return measure_once(workload, size);
    }
    std::thread::Builder::new()
        .name(format!("pipeline-{}", workload.name()))
        .stack_size(256 * 1024)
        .spawn(move || measure_once(workload, size))
        .map_err(|error| format!("failed to spawn stack-safety sample: {error}"))?
        .join()
        .map_err(|_| format!("{} panicked on a 256 KiB stack", workload.name()))?
}

fn median(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        values[middle - 1] / 2
            + values[middle] / 2
            + (values[middle - 1] % 2 + values[middle] % 2) / 2
    }
}

fn milliseconds(nanoseconds: u128) -> f64 {
    nanoseconds as f64 / 1_000_000.0
}

fn run() -> Result<(), String> {
    let Some(config) = Config::parse()? else {
        return Ok(());
    };
    println!(
        "mode\tworkload\tsize\titerations\tgraph_build_ms\tindex_ms\toptimization_ms\tanalysis_ms\tsemantic_nodes\tgraph_index_builds\tgraph_semantic_node_visits\tgraph_edge_visits\toptimization_semantic_node_scans\tsemantic_nodes_rebuilt\tfusion_candidates\tmulti_output_work_items\tmulti_output_dependency_edges\tmulti_output_dependency_passes\tmulti_output_dependency_edge_visits\tmulti_output_dependency_queries\tregion_table_merges\tselected_regions\tplanned_lowering_units"
    );
    for workload in config.workloads {
        let size = if workload.is_stress() {
            workload.default_size()
        } else {
            config.size.unwrap_or_else(|| workload.default_size())
        };
        let mut samples = Vec::with_capacity(config.iterations);
        for _ in 0..config.iterations {
            samples.push(measure(workload, size)?);
        }
        let expected = samples[0].metrics;
        if samples.iter().any(|sample| sample.metrics != expected) {
            return Err(format!(
                "{} produced non-deterministic structural metrics",
                workload.name()
            ));
        }
        println!(
            "graph_analysis\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            workload.name(),
            size,
            config.iterations,
            milliseconds(median(
                samples.iter().map(|sample| sample.graph_build_ns).collect()
            )),
            milliseconds(median(
                samples.iter().map(|sample| sample.index_ns).collect()
            )),
            milliseconds(median(
                samples
                    .iter()
                    .map(|sample| sample.optimization_ns)
                    .collect()
            )),
            milliseconds(median(
                samples.iter().map(|sample| sample.analysis_ns).collect()
            )),
            expected.semantic_nodes,
            expected.graph_index_builds,
            expected.graph_semantic_node_visits,
            expected.graph_edge_visits,
            expected.optimization_semantic_node_scans,
            expected.semantic_nodes_rebuilt,
            expected.fusion_candidates,
            expected.multi_output_work_items,
            expected.multi_output_dependency_edges,
            expected.multi_output_dependency_passes,
            expected.multi_output_dependency_edge_visits,
            expected.multi_output_dependency_queries,
            expected.region_table_merges,
            expected.selected_regions,
            expected.planned_lowering_units,
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("graph-analysis benchmark: {error}");
        std::process::exit(2);
    }
}

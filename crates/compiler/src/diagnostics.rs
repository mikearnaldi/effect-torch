//! Compiler observability: structural work reports and executable
//! diagnostics assembled from the pipeline's authoritative artifacts.
//!
//! Two distinct families of numbers live here, with a hard boundary between
//! them:
//!
//! - **Structural counters** ([`CompilerWorkReport`], [`PassScanCount`]):
//!   exact, deterministic tallies of what the compiler did — nodes indexed,
//!   edges visited, passes scanned, regions selected, values and
//!   instructions lowered. These are combined only from the authoritative
//!   artifacts (graph index, optimization plan, lowered program), never
//!   re-measured.
//! - **Observational timings** (`CompilePhaseTiming` lists): wall-clock
//!   measurements carried alongside the structural data. They must never
//!   influence cache identity, equality, or hashing of any artifact.
//!
//! Executable diagnostics additionally present instruction-kind histograms
//! in lexical order so reports are stable regardless of backend map
//! implementations or insertion order.

use crate::{GraphIndex, LoweredProgram, OptimizationWork};
use effect_torch_runtime::{
    CompilePhaseTiming, ExecutableDiagnostics, InstructionCount, MemoryPlan,
};
use std::collections::BTreeMap;

/// Counts measured by the backend driver outside the compiler's structural
/// artifacts: pipeline/command/synchronization totals plus the phase timings
/// to attach to the report.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiagnosticsInput {
    pub pipeline_count: usize,
    pub command_count: usize,
    pub synchronization_count: usize,
    pub compile_phases: Vec<CompilePhaseTiming>,
}

/// Semantic-node visits attributed to one named pass or pass group.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PassScanCount {
    pub pass: String,
    /// Semantic-node visits performed by this pass or pass group.
    pub count: usize,
}

impl PassScanCount {
    pub fn new(pass: impl Into<String>, count: usize) -> Self {
        Self {
            pass: pass.into(),
            count,
        }
    }
}

/// Structural compiler work plus observational phase timings.
///
/// This type intentionally does not implement `Hash`: wall-clock timings are
/// diagnostics and must never become part of executable cache identity.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompilerWorkReport {
    pub semantic_nodes: usize,
    pub graph_index_builds: usize,
    pub graph_edge_visits: usize,
    pub semantic_nodes_rebuilt: usize,
    pub pass_scans: Box<[PassScanCount]>,
    pub fusion_candidates: usize,
    pub multi_output_dependency_edges: usize,
    pub multi_output_dependency_passes: usize,
    pub multi_output_dependency_edge_visits: usize,
    pub multi_output_dependency_queries: usize,
    pub selected_regions: usize,
    pub lowered_values: usize,
    pub lowered_instructions: usize,
    pub compile_phases: Box<[CompilePhaseTiming]>,
}

impl CompilerWorkReport {
    /// Combines structural counters from the compiler's authoritative artifacts.
    pub fn from_artifacts<K, M, V>(
        index: &GraphIndex,
        optimization: &OptimizationWork,
        lowered: &LoweredProgram<K, M, V>,
    ) -> Self {
        let pass_scans = (optimization.semantic_nodes_scanned != 0)
            .then(|| PassScanCount::new("optimization", optimization.semantic_nodes_scanned))
            .into_iter()
            .collect();
        Self {
            semantic_nodes: index.order.len(),
            graph_index_builds: index
                .work
                .graph_index_builds
                .max(optimization.graph_index_builds),
            graph_edge_visits: index.work.graph_edge_visits,
            semantic_nodes_rebuilt: optimization.semantic_nodes_rebuilt,
            pass_scans,
            fusion_candidates: optimization.fusion_candidates,
            multi_output_dependency_edges: optimization.multi_output_dependency_edges,
            multi_output_dependency_passes: optimization.multi_output_dependency_passes,
            multi_output_dependency_edge_visits: optimization.multi_output_dependency_edge_visits,
            multi_output_dependency_queries: optimization.multi_output_dependency_queries,
            selected_regions: optimization.selected_regions,
            lowered_values: lowered.values.len(),
            lowered_instructions: lowered.instructions.len(),
            compile_phases: Box::new([]),
        }
    }

    /// Overrides the pass-scan list (used when a backend attributes scans
    /// more finely than the default single "optimization" entry).
    pub fn with_pass_scans(mut self, pass_scans: impl Into<Box<[PassScanCount]>>) -> Self {
        self.pass_scans = pass_scans.into();
        self
    }

    /// Attaches observational phase timings. These travel with the report
    /// but are excluded from its `Eq` semantics' intent: they are
    /// diagnostics, never identity.
    pub fn with_compile_phases(
        mut self,
        compile_phases: impl Into<Box<[CompilePhaseTiming]>>,
    ) -> Self {
        self.compile_phases = compile_phases.into();
        self
    }
}

/// Builds diagnostics with instruction kinds in lexical order, independent of
/// backend map implementations or insertion order.
pub fn build_executable_diagnostics<K, M, V, F, S>(
    schedule: &LoweredProgram<K, M, V>,
    memory: &MemoryPlan<M>,
    index: &GraphIndex,
    input: DiagnosticsInput,
    kind_name: F,
) -> ExecutableDiagnostics
where
    F: Fn(&K) -> S,
    S: AsRef<str>,
{
    let mut counts = BTreeMap::<String, usize>::new();
    for instruction in &schedule.instructions {
        let name = kind_name(&instruction.kind).as_ref().to_owned();
        *counts.entry(name).or_default() += 1;
    }
    let instructions = counts
        .into_iter()
        .map(|(kind, count)| InstructionCount { kind, count })
        .collect();

    ExecutableDiagnostics {
        semantic_nodes_before_optimization: index.order.len(),
        semantic_nodes_after_optimization: index.order.len(),
        instructions,
        pipeline_count: input.pipeline_count,
        command_count: input.command_count,
        synchronization_count: input.synchronization_count,
        memory: memory.report.clone(),
        compile_phases: input.compile_phases.into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LoweredInstruction;
    use effect_torch_runtime::InstructionId;

    #[test]
    fn instruction_counts_have_stable_lexical_order() {
        let schedule = LoweredProgram::<&str, ()>::new(
            Vec::new(),
            vec![
                LoweredInstruction::new(InstructionId::new(0), "zeta", Vec::new(), Vec::new()),
                LoweredInstruction::new(InstructionId::new(1), "alpha", Vec::new(), Vec::new()),
                LoweredInstruction::new(InstructionId::new(2), "zeta", Vec::new(), Vec::new()),
            ],
            Vec::new(),
        );
        let diagnostics = build_executable_diagnostics(
            &schedule,
            &MemoryPlan::default(),
            &GraphIndex::new(&[]).unwrap(),
            DiagnosticsInput::default(),
            |kind| *kind,
        );
        assert_eq!(
            diagnostics.instructions.as_ref(),
            [
                InstructionCount {
                    kind: "alpha".into(),
                    count: 1,
                },
                InstructionCount {
                    kind: "zeta".into(),
                    count: 2,
                },
            ]
        );
    }

    #[test]
    fn compiler_work_combines_structural_artifacts_without_timings() {
        let index = GraphIndex::new(&[]).unwrap();
        let optimization = OptimizationWork {
            graph_index_builds: 1,
            semantic_nodes_scanned: 7,
            semantic_nodes_rebuilt: 2,
            fusion_candidates: 3,
            multi_output_dependency_edges: 11,
            multi_output_dependency_passes: 2,
            multi_output_dependency_edge_visits: 19,
            multi_output_dependency_queries: 5,
            selected_regions: 1,
            ..OptimizationWork::default()
        };
        let lowered = LoweredProgram::<&str, ()>::new(Vec::new(), Vec::new(), Vec::new());

        let report = CompilerWorkReport::from_artifacts(&index, &optimization, &lowered);
        assert_eq!(report.graph_index_builds, 1);
        assert_eq!(report.semantic_nodes_rebuilt, 2);
        assert_eq!(
            report.pass_scans.as_ref(),
            [PassScanCount::new("optimization", 7)]
        );
        assert_eq!(report.fusion_candidates, 3);
        assert_eq!(report.multi_output_dependency_edges, 11);
        assert_eq!(report.multi_output_dependency_passes, 2);
        assert_eq!(report.multi_output_dependency_edge_visits, 19);
        assert_eq!(report.multi_output_dependency_queries, 5);
        assert_eq!(report.selected_regions, 1);
        assert!(report.compile_phases.is_empty());
    }
}

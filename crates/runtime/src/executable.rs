//! Diagnostics from compiling a program into an executable.
//!
//! These data structures contain no behavior. Backends fill them during
//! compilation and do not update them afterward. Callers use the snapshots
//! for profiling, logging, and regression tests.

use crate::MemoryReport;

/// Number of emitted instructions of one kind (e.g. `"matmul"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstructionCount {
    pub kind: String,
    pub count: usize,
}

/// Wall-clock time spent in one named compilation phase.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompilePhaseTiming {
    pub phase: String,
    pub nanoseconds: u64,
}

/// Target identity summary and structural dtype-legalization work for an executable.
///
/// The compiler fills this immutable snapshot from its selected target and plan.
/// Runtime does not depend on compiler types. Counts describe the compiled plan,
/// not invocation activity; timing remains in [`CompilePhaseTiming`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DTypeLegalizationDiagnostics {
    pub target_backend: String,
    pub target_architecture: String,
    pub lowering_abi_revision: u64,
    pub policy_revision: u64,
    pub capability_queries: usize,
    pub native_lowering_units: usize,
    pub legalized_lowering_units: usize,
    pub kernel_local_legalizations: usize,
    pub materialized_conversions: usize,
    /// Sum of declared conversion-value bytes, rather than peak live bytes.
    pub materialized_conversion_bytes: usize,
    pub decompositions: usize,
    pub rejected_region_candidates: usize,
}

/// Statistics for one compiled executable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ExecutableDiagnostics {
    pub semantic_nodes_before_optimization: usize,
    pub semantic_nodes_after_optimization: usize,
    pub instructions: Box<[InstructionCount]>,
    pub pipeline_count: usize,
    pub command_count: usize,
    pub synchronization_count: usize,
    pub memory: MemoryReport,
    pub legalization: DTypeLegalizationDiagnostics,
    pub compile_phases: Box<[CompilePhaseTiming]>,
}

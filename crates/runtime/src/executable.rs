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
    pub compile_phases: Box<[CompilePhaseTiming]>,
}

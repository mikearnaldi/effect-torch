//! Diagnostics produced when a program is compiled into an executable.
//!
//! These are plain data structures (no behavior): backends fill them in
//! during compilation and callers use them for profiling, logging and
//! regression testing of the compilation pipeline itself. All fields are
//! snapshots taken at compile time and never mutate afterwards.

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

/// Aggregate statistics describing one compiled executable.
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

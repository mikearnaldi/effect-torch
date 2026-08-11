use crate::MemoryReport;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstructionCount {
    pub kind: String,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompilePhaseTiming {
    pub phase: String,
    pub nanoseconds: u64,
}

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

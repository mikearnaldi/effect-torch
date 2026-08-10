use crate::LoweredSchedule;
use effect_torch_runtime::{
    CompilePhaseTiming, ExecutableDiagnostics, InstructionCount, MemoryPlan,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DiagnosticsInput {
    pub semantic_nodes_before_optimization: usize,
    pub semantic_nodes_after_optimization: usize,
    pub pipeline_count: usize,
    pub command_count: usize,
    pub synchronization_count: usize,
    pub output_capacity: usize,
    pub compile_phases: Vec<CompilePhaseTiming>,
}

/// Builds diagnostics with instruction kinds in lexical order, independent of
/// backend map implementations or insertion order.
pub fn build_executable_diagnostics<K, M, F, S>(
    schedule: &LoweredSchedule<K, M>,
    memory: &MemoryPlan<M>,
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
        semantic_nodes_before_optimization: input.semantic_nodes_before_optimization,
        semantic_nodes_after_optimization: input.semantic_nodes_after_optimization,
        instructions,
        pipeline_count: input.pipeline_count,
        command_count: input.command_count,
        synchronization_count: input.synchronization_count,
        output_capacity: input.output_capacity,
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
        let schedule = LoweredSchedule::<&str, ()>::new(
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
}

use crate::request::{derive_graph_signature, request_parts, signature_with_contract};
use crate::{
    CompilerWorkReport, GraphIndex, LoweredProgram, LoweredValue, LoweringUnit,
    MemoryPlannerConfig, OptimizationPlan, PlannerError, PreparedProgram, ProgramRequest,
};
use effect_torch_runtime::{CompilePhaseTiming, MemoryPlan};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const GRAPH_INDEX_PHASE: &str = "graph_index";
pub const OPTIMIZATION_PHASE: &str = "optimization";
pub const LOWERING_PHASE: &str = "lowering";
pub const LOWERED_PROGRAM_VALIDATION_PHASE: &str = "lowered_program_validation";
pub const MEMORY_PLANNING_PHASE: &str = "memory_planning";
pub const PHYSICAL_PLANNING_PHASE: &str = "physical_planning";
pub const PIPELINE_PREPARATION_PHASE: &str = "pipeline_preparation";
pub const ARTIFACT_ASSEMBLY_PHASE: &str = "artifact_assembly";
pub const COMPILE_SUBMISSION_PHASE: &str = "compile_submission";
pub const PUBLICATION_PHASE: &str = "publication";

fn timing(phase: &str, elapsed: Duration) -> CompilePhaseTiming {
    CompilePhaseTiming {
        phase: phase.to_string(),
        nanoseconds: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
    }
}

/// Prepares one semantic graph generation and records its only graph-index build.
pub fn prepare_program(request: ProgramRequest) -> Result<PreparedProgram, String> {
    let (roots, bindings, invocation, options, state_cursor, derive_contract) =
        request_parts(request);
    let Some(first_root) = roots.first() else {
        return Err("compile: expected at least one root".to_string());
    };
    options.validate()?;
    let expected_device = first_root.device.clone();
    let started = Instant::now();
    let index = GraphIndex::new(&roots);
    let graph_index_timing = timing(GRAPH_INDEX_PHASE, started.elapsed());
    let index = index?;
    if index
        .order
        .iter()
        .any(|node| !node.device.same_device(&expected_device))
    {
        return Err("compile: graph contains multiple devices".to_string());
    }
    let signature = if derive_contract {
        derive_graph_signature(&index, state_cursor)?
    } else {
        signature_with_contract(&index, bindings, invocation)
    };
    Ok(PreparedProgram {
        roots: roots.into_boxed_slice(),
        index: Arc::new(index),
        signature,
        options,
        state_cursor,
        preparation_phases: vec![graph_index_timing].into_boxed_slice(),
    })
}

/// Shared, statically-dispatched sequencing for one prepared cache miss.
pub struct CompilerDriver<'a> {
    prepared: &'a PreparedProgram,
    optimization: OptimizationPlan,
    compile_phases: Vec<CompilePhaseTiming>,
}

impl<'a> CompilerDriver<'a> {
    pub fn new(prepared: &'a PreparedProgram) -> Result<Self, String> {
        let mut compile_phases = prepared.preparation_phases.to_vec();
        let started = Instant::now();
        let optimization = OptimizationPlan::from_prepared(prepared);
        compile_phases.push(timing(OPTIMIZATION_PHASE, started.elapsed()));
        Ok(Self {
            prepared,
            optimization: optimization?,
            compile_phases,
        })
    }

    pub fn prepared(&self) -> &'a PreparedProgram {
        self.prepared
    }

    pub fn optimization(&self) -> &OptimizationPlan {
        &self.optimization
    }

    /// Iterates the plan's deterministic lowering units through a backend callback.
    pub fn lower(
        &mut self,
        mut lower: impl FnMut(LoweringUnit, &GraphIndex, &OptimizationPlan) -> Result<(), String>,
    ) -> Result<(), String> {
        let started = Instant::now();
        let result = self
            .optimization
            .lowering_order
            .iter()
            .copied()
            .try_for_each(|unit| lower(unit, &self.prepared.index, &self.optimization));
        self.compile_phases
            .push(timing(LOWERING_PHASE, started.elapsed()));
        result
    }

    /// Times a named backend completion phase without introducing a backend trait object.
    pub fn phase<T, E>(
        &mut self,
        phase: &'static str,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let started = Instant::now();
        let result = operation();
        self.compile_phases.push(timing(phase, started.elapsed()));
        result
    }

    pub fn record_phase(&mut self, phase: &'static str, elapsed: Duration) {
        self.compile_phases.push(timing(phase, elapsed));
    }

    /// Validates the lowered tables before timing logical memory planning.
    pub fn plan_memory<K, M, V>(
        &mut self,
        lowered: &LoweredProgram<K, M, V>,
        config: &MemoryPlannerConfig<M>,
    ) -> Result<MemoryPlan<M>, PlannerError>
    where
        M: Clone + Eq,
        V: LoweredValue<M>,
    {
        let liveness = self.phase(LOWERED_PROGRAM_VALIDATION_PHASE, || {
            crate::analyze_liveness(lowered)
        })?;
        self.phase(MEMORY_PLANNING_PHASE, || {
            crate::planner::plan_memory_with_liveness(lowered, config, &liveness)
        })
    }

    pub fn finish<K, M, V>(self, lowered: &LoweredProgram<K, M, V>) -> CompilerWorkReport
    where
        V: LoweredValue<M>,
    {
        CompilerWorkReport::from_artifacts(&self.prepared.index, &self.optimization.work, lowered)
            .with_compile_phases(self.compile_phases.into_boxed_slice())
    }

    pub fn finish_with_phase<K, M, V>(
        mut self,
        lowered: &LoweredProgram<K, M, V>,
        phase: &'static str,
        started: Instant,
    ) -> CompilerWorkReport
    where
        V: LoweredValue<M>,
    {
        let report = CompilerWorkReport::from_artifacts(
            &self.prepared.index,
            &self.optimization.work,
            lowered,
        );
        self.record_phase(phase, started.elapsed());
        report.with_compile_phases(self.compile_phases.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompileOptions, LoweredInstruction, ValueDecl};
    use effect_torch_graph::{Device, Node, NodeKind};
    use effect_torch_runtime::{DType, InstructionId, NativeMemorySpace};

    #[test]
    fn driver_owns_index_optimization_lowering_and_work_reporting() {
        let root = Node::new(NodeKind::Zeros {
            shape: vec![1],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let prepared = ProgramRequest::from_roots(vec![root], CompileOptions::default())
            .prepare()
            .unwrap();
        let mut driver = CompilerDriver::new(&prepared).unwrap();
        let mut units = Vec::new();
        driver
            .lower(|unit, _, _| {
                units.push(unit);
                Ok(())
            })
            .unwrap();
        let lowered = LoweredProgram::<&str, NativeMemorySpace, ValueDecl<NativeMemorySpace>>::new(
            Vec::new(),
            vec![LoweredInstruction::new(
                InstructionId::new(0),
                "zero",
                Vec::new(),
                Vec::new(),
            )],
            Vec::new(),
        );
        driver
            .plan_memory(
                &lowered,
                &crate::MemoryPlannerConfig::uniform(NativeMemorySpace::Cpu, usize::MAX / 2, 1, 1),
            )
            .unwrap();
        driver
            .phase(PHYSICAL_PLANNING_PHASE, || Ok::<_, String>(()))
            .unwrap();
        let report = driver.finish(&lowered);

        assert_eq!(units.len(), 1);
        assert_eq!(report.graph_index_builds, 1);
        assert_eq!(report.semantic_nodes_rebuilt, 0);
        assert_eq!(
            report
                .compile_phases
                .iter()
                .map(|phase| phase.phase.as_str())
                .collect::<Vec<_>>(),
            [
                GRAPH_INDEX_PHASE,
                OPTIMIZATION_PHASE,
                LOWERING_PHASE,
                LOWERED_PROGRAM_VALIDATION_PHASE,
                MEMORY_PLANNING_PHASE,
                PHYSICAL_PLANNING_PHASE,
            ]
        );
    }

    #[test]
    fn preparation_rejects_invalid_compile_options_before_lowering() {
        let root = Node::new(NodeKind::Zeros {
            shape: vec![1],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let mut options = CompileOptions::default();
        options.environment.ce_chunk_size = 0;

        let error = ProgramRequest::from_roots(vec![root], options)
            .prepare()
            .err()
            .unwrap();

        assert_eq!(error, "compile: CE chunk size must be positive");
    }
}

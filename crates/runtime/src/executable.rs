use crate::{
    BackendError, BackendResult, Capabilities, ErasedBuffer, Invocation, MemoryPlan, MemoryReport,
    NativeMemorySpace, ProgramSignature, RuntimeId, RuntimeIdentity,
};
use std::fmt;

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

#[derive(Debug, Clone)]
pub struct NativeExecutable<C, P, M = NativeMemorySpace> {
    pub runtime_id: RuntimeId,
    pub signature: ProgramSignature,
    pub pipelines: Box<[P]>,
    pub commands: Box<[C]>,
    pub memory: MemoryPlan<M>,
    pub diagnostics: ExecutableDiagnostics,
}

pub trait Executable: fmt::Debug + Send + Sync {
    fn runtime_id(&self) -> RuntimeId;
    fn signature(&self) -> &ProgramSignature;
    fn diagnostics(&self) -> &ExecutableDiagnostics;

    fn validate_owner(&self, expected: RuntimeId) -> BackendResult<()> {
        let actual = self.runtime_id();
        if actual == expected {
            Ok(())
        } else {
            Err(BackendError::invalid_handle(expected, actual))
        }
    }
}

impl<C, P, M> Executable for NativeExecutable<C, P, M>
where
    C: fmt::Debug + Send + Sync,
    P: fmt::Debug + Send + Sync,
    M: fmt::Debug + Send + Sync,
{
    fn runtime_id(&self) -> RuntimeId {
        self.runtime_id
    }

    fn signature(&self) -> &ProgramSignature {
        &self.signature
    }

    fn diagnostics(&self) -> &ExecutableDiagnostics {
        &self.diagnostics
    }
}

pub trait ExecutableBackend<Request>: Send + Sync {
    type Executable: Executable;

    fn identity(&self) -> &RuntimeIdentity;
    fn capabilities(&self) -> &Capabilities;
    fn compile(&self, request: &Request) -> BackendResult<Self::Executable>;
    fn execute(
        &self,
        executable: &Self::Executable,
        invocation: &Invocation,
    ) -> BackendResult<Vec<ErasedBuffer>>;

    fn synchronize(&self) -> BackendResult<()> {
        Ok(())
    }
}

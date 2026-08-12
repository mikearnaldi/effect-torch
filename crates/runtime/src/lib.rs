mod backend;
mod cancellation;
mod dense;
mod dtype;
mod error;
mod executable;
mod gguf;
mod layout;
mod memory;
mod program;
mod workspace;

pub use backend::{
    Buffer, Capabilities, Capability, DeviceId, ErasedBuffer, Placement, RuntimeId, RuntimeIdentity,
};
pub use cancellation::CancellationFlag;
pub use dense::{
    BindingId, CommandId, DenseId, InstructionId, LocationId, OutputId, PipelineId, RuntimeValueId,
    ScalarId, SegmentId, TableRange, ValueId,
};
pub use dtype::DType;
pub use error::{BackendError, BackendResult};
pub use executable::{CompilePhaseTiming, ExecutableDiagnostics, InstructionCount};
pub use gguf::{
    parse_gguf, read_gguf_tensor_into, GgmlKQuant, GgufFile, GgufMetadataArray, GgufMetadataEntry,
    GgufMetadataValue, GgufParseError, GgufTensorDescriptor, GgufTensorFormat,
};
pub use layout::{broadcast_shape, Layout};
pub use memory::{
    AllocationReport, InvocationMemoryReport, Location, MemoryPlan, MemoryPlanError, MemoryReport,
    NativeMemorySpace, OutputSlot, ReuseEdge, SegmentDecl, SegmentOwnership, StorageClass,
};
pub use program::{
    BindingAliasing, BindingDecl, BindingLayoutPolicy, Invocation, InvocationError,
    InvocationSignature, LayoutConstraint, OutputSignature, ProgramSignature, RngDecl,
    RngInvocation, RuntimeValue, RuntimeValueDecl, RuntimeValueError, RuntimeValueKind, ScalarDecl,
    ScalarType, ScalarValue,
};
pub use workspace::{
    LeasedWorkspace, WorkspaceAllocation, WorkspaceAllocator, WorkspaceLease, WorkspacePool,
    WorkspacePoolError, WorkspacePoolStats, WorkspaceRequest,
};

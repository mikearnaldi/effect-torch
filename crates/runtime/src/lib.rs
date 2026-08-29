//! Backend-agnostic core types for the effect-torch runtime.
//!
//! This crate defines the types shared by concrete backends and the compiler.
//! It does not contain an execution engine.
//!
//! - `backend` defines runtime identity and ownership. Each runtime gets a
//!   unique [`RuntimeId`], and its [`Buffer`] handles carry that id.
//!   [`ErasedBuffer::validate_owner`] rejects handles from other runtimes.
//! - `dtype` and `layout` define element types and shape, stride, and offset
//!   geometry. [`Layout`] provides checked and panicking arithmetic methods.
//! - `program` defines [`ProgramSignature`] and [`Invocation`]. Validation
//!   checks counts, dtypes, placements, shapes, layout policies, bounded runtime
//!   values, and RNG state before execution.
//! - `memory` defines compiler-produced [`MemoryPlan`] values with typed
//!   segments, value locations, output slots, reuse edges, structural
//!   validation, and accounting reports.
//! - `workspace` provides a thread-safe [`WorkspacePool`] with deterministic
//!   best-fit reuse and an LRU limit on idle memory.
//! - `gguf` provides the resource-limited, cancellable [`parse_gguf`] parser
//!   for GGUF v3 files.
//! - `dense` provides compact `u32` ids such as [`ValueId`] for the flat
//!   tables shared by the compiler and runtime.
//! - `cancellation` provides the cooperative [`CancellationFlag`] used by
//!   long-running operations.
//! - `error` and `executable` provide [`BackendError`] and
//!   [`ExecutableDiagnostics`].
//!
//! The crate contains no `unsafe` code. Types, validation, and boundary
//! assertions enforce its invariants.

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
mod sampling;
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
pub use sampling::{
    effective_probabilities, purpose_counter, random_unit, random_unit_at, rejection_sample,
    sample_logits, sample_probabilities, sample_probabilities_at, sampling_coordinate,
    target_sample_match, RejectionResult, SamplingCoordinate, SamplingOptions, SamplingPurpose,
    TargetSampleMatchResult, MAX_SAMPLING_VOCABULARY,
};
pub use workspace::{
    LeasedWorkspace, WorkspaceAllocation, WorkspaceAllocator, WorkspaceLease, WorkspacePool,
    WorkspacePoolError, WorkspacePoolStats, WorkspaceRequest,
};

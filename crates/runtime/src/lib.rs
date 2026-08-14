//! Backend-agnostic core types for the effect-torch runtime.
//!
//! This crate defines the vocabulary that concrete backends (CPU, Metal,
//! ...) implement and that the compiler targets; it contains no execution
//! engine itself. The main pieces are:
//!
//! - `backend`: runtime identity and ownership. Every runtime instance
//!   mints a unique [`RuntimeId`]; [`Buffer`] handles carry the id of the
//!   runtime that allocated them, and [`ErasedBuffer::validate_owner`]
//!   rejects handles from foreign runtimes so buffers can never cross
//!   runtime boundaries unchecked.
//! - `dtype` and `layout`: the element-type and shape/stride/offset
//!   model shared by every backend. Layout arithmetic is available in both
//!   panicking and checked (overflow-detecting) forms.
//! - `program`: declared program signatures ([`ProgramSignature`]) and
//!   per-call arguments ([`Invocation`]) with full validation of counts,
//!   dtypes, placements, shapes, layout policies, bounded runtime values
//!   and RNG state before any execution.
//! - `memory`: the compiler-produced [`MemoryPlan`] — typed segments,
//!   value locations, output slots and buffer-reuse edges — together with
//!   its structural validation rules and accounting reports.
//! - `workspace`: a generic, thread-safe [`WorkspacePool`] that recycles
//!   transient execution buffers with deterministic best-fit reuse and
//!   LRU-bounded idle memory.
//! - `gguf`: a strict, resource-limited, cancellable parser for GGUF v3
//!   model files (see [`parse_gguf`]).
//! - `dense`: compact `u32`-based identifiers (e.g. [`ValueId`]) used to
//!   index the flat tables exchanged between compiler and runtime.
//! - `cancellation`: the cooperative [`CancellationFlag`] polled by
//!   long-running operations.
//! - `error` and `executable`: the backend error taxonomy
//!   ([`BackendError`]) and the diagnostics emitted by compiled
//!   executables ([`ExecutableDiagnostics`]).
//!
//! The crate is intentionally free of `unsafe` code; all invariants are
//! enforced through types, validation and assertion at the boundaries.

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
pub use sampling::{sample_logits, SamplingOptions, MAX_SAMPLING_VOCABULARY};
pub use workspace::{
    LeasedWorkspace, WorkspaceAllocation, WorkspaceAllocator, WorkspaceLease, WorkspacePool,
    WorkspacePoolError, WorkspacePoolStats, WorkspaceRequest,
};

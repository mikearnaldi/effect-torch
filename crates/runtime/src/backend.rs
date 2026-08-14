//! Runtime identity, device placement and buffer ownership.
//!
//! # Ownership model
//!
//! Every live runtime instance owns a process-unique [`RuntimeId`] minted
//! from a global counter (`RuntimeId::new`). All buffers allocated by that
//! runtime carry its id, and entry points that accept buffers validate the
//! id before use (see [`ErasedBuffer::validate_owner`] and
//! [`ProgramSignature::validate_invocation`](crate::ProgramSignature::validate_invocation)).
//! This turns "buffer used with the wrong runtime" — a common source of
//! undefined behavior in GPU stacks — into an ordinary, recoverable
//! [`BackendError::InvalidHandle`].
//!
//! [`RuntimeIdentity`] couples a [`RuntimeId`] with a human-readable backend
//! name; [`DeviceId`] names one device and [`Placement`] pins a buffer to a
//! device plus an optional memory space. [`Capabilities`] advertises the
//! dtypes and features a backend supports so compilation can fail early
//! instead of producing executables that cannot run.
//!
//! # Buffer trait
//!
//! [`Buffer`] is the minimal object-safe view every backend buffer must
//! expose: owner id, placement, dtype and layout, plus an [`Any`] downcast
//! escape hatch. [`ErasedBuffer`] is the `Arc`-shared, type-erased handle
//! passed across crate boundaries; it re-delegates the trait methods and
//! provides checked downcasting via [`ErasedBuffer::downcast_ref`].

use crate::{BackendError, BackendResult, DType, Layout};
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

/// Process-unique identifier of one runtime instance.
///
/// Ids are allocated from a monotonically increasing global counter and
/// are never reused within a process, so equality of two ids implies the
/// same owning runtime. Ids start at 1; 0 is never issued.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeId(u64);

impl RuntimeId {
    /// Mints a fresh, never-before-issued runtime id.
    pub fn new() -> Self {
        Self(NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for RuntimeId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for RuntimeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RuntimeId").field(&self.0).finish()
    }
}

impl fmt::Display for RuntimeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Identity of one runtime instance: a unique [`RuntimeId`] plus the name
/// of the backend that created it (e.g. `"cpu"`, `"metal"`).
///
/// Two identities created with the same backend name are still distinct —
/// the id, not the name, is the ownership token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeIdentity {
    id: RuntimeId,
    backend: Arc<str>,
}

impl RuntimeIdentity {
    /// Creates a new identity for the named backend, minting a fresh id.
    pub fn new(backend: impl Into<Arc<str>>) -> Self {
        Self {
            id: RuntimeId::new(),
            backend: backend.into(),
        }
    }

    /// The unique id that all buffers of this runtime carry.
    pub fn id(&self) -> RuntimeId {
        self.id
    }

    /// The backend name supplied at construction.
    pub fn backend(&self) -> &str {
        &self.backend
    }
}

/// Opaque, backend-assigned name of one device (e.g. `"cpu:0"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceId(Arc<str>);

impl DeviceId {
    pub fn new(id: impl Into<Arc<str>>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where a buffer lives: a device plus an optional named memory space
/// (e.g. a Metal shared vs. private heap).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Placement {
    device: DeviceId,
    memory_space: Option<Arc<str>>,
}

impl Placement {
    /// Placement on `device` in the backend's default memory space.
    pub fn new(device: DeviceId) -> Self {
        Self {
            device,
            memory_space: None,
        }
    }

    /// Placement on `device` in an explicitly named memory space.
    pub fn with_memory_space(device: DeviceId, memory_space: impl Into<Arc<str>>) -> Self {
        Self {
            device,
            memory_space: Some(memory_space.into()),
        }
    }

    pub fn device(&self) -> &DeviceId {
        &self.device
    }

    /// The explicit memory space, if one was named.
    pub fn memory_space(&self) -> Option<&str> {
        self.memory_space.as_deref()
    }
}

impl fmt::Display for Placement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.device.fmt(f)?;
        if let Some(memory_space) = &self.memory_space {
            write!(f, "/{memory_space}")?;
        }
        Ok(())
    }
}

/// An optional execution feature a backend may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// The backend can compile programs into executables.
    Compilation,
    /// The backend can execute asynchronously (non-blocking launches).
    AsyncExecution,
    /// Host and device share one address space, making transfers free.
    UnifiedMemory,
}

/// The set of dtypes and [`Capability`] features a backend supports.
///
/// Lists are kept as ordered vectors; duplicates are meaningless but
/// harmless since all queries are membership tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Capabilities {
    dtypes: Vec<DType>,
    features: Vec<Capability>,
}

impl Capabilities {
    pub fn new(dtypes: Vec<DType>, features: Vec<Capability>) -> Self {
        Self { dtypes, features }
    }

    pub fn dtypes(&self) -> &[DType] {
        &self.dtypes
    }

    pub fn features(&self) -> &[Capability] {
        &self.features
    }

    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.dtypes.contains(&dtype)
    }

    pub fn supports(&self, capability: Capability) -> bool {
        self.features.contains(&capability)
    }
}

/// Minimal object-safe view of a backend-allocated buffer.
///
/// Implementors must guarantee that [`runtime_id`](Buffer::runtime_id)
/// returns the id of the runtime that allocated the buffer and that
/// [`layout`](Buffer::layout) and [`dtype`](Buffer::dtype) describe the
/// buffer's current contents. `Send + Sync` is required because buffers
/// cross thread boundaries inside execution pools.
pub trait Buffer: Any + fmt::Debug + Send + Sync {
    fn runtime_id(&self) -> RuntimeId;
    fn placement(&self) -> &Placement;
    fn dtype(&self) -> DType;
    fn layout(&self) -> &Layout;
    fn as_any(&self) -> &dyn Any;
}

/// Reference-counted, type-erased buffer handle.
///
/// Cloning is cheap (an `Arc` bump) and all clones share the same
/// underlying buffer. The concrete backend type can be recovered with
/// [`downcast_ref`](Self::downcast_ref).
#[derive(Clone)]
pub struct ErasedBuffer(Arc<dyn Buffer>);

impl ErasedBuffer {
    pub fn new(buffer: impl Buffer) -> Self {
        Self(Arc::new(buffer))
    }

    /// Id of the runtime that owns the underlying buffer.
    pub fn runtime_id(&self) -> RuntimeId {
        self.0.runtime_id()
    }

    pub fn placement(&self) -> &Placement {
        self.0.placement()
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype()
    }

    pub fn layout(&self) -> &Layout {
        self.0.layout()
    }

    /// Downcasts to the concrete buffer type, or `None` if the underlying
    /// buffer is of a different type.
    pub fn downcast_ref<B: Buffer>(&self) -> Option<&B> {
        self.0.as_any().downcast_ref()
    }

    /// Fails with [`BackendError::InvalidHandle`] unless the buffer is
    /// owned by the runtime with id `expected`.
    pub fn validate_owner(&self, expected: RuntimeId) -> BackendResult<()> {
        let actual = self.runtime_id();
        if actual == expected {
            Ok(())
        } else {
            Err(BackendError::invalid_handle(expected, actual))
        }
    }
}

impl fmt::Debug for ErasedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ErasedBuffer")
            .field("runtime_id", &self.runtime_id())
            .field("placement", &self.placement())
            .field("dtype", &self.dtype())
            .field("layout", &self.layout())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestBuffer {
        runtime_id: RuntimeId,
        placement: Placement,
        layout: Layout,
    }

    impl Buffer for TestBuffer {
        fn runtime_id(&self) -> RuntimeId {
            self.runtime_id
        }

        fn placement(&self) -> &Placement {
            &self.placement
        }

        fn dtype(&self) -> DType {
            DType::F32
        }

        fn layout(&self) -> &Layout {
            &self.layout
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn runtime_identities_do_not_alias() {
        let first = RuntimeIdentity::new("cpu");
        let second = RuntimeIdentity::new("cpu");
        assert_ne!(first.id(), second.id());
    }

    #[test]
    fn erased_buffers_reject_foreign_runtime_owners() {
        let owner = RuntimeIdentity::new("cpu");
        let foreign = RuntimeIdentity::new("cpu");
        let buffer = ErasedBuffer::new(TestBuffer {
            runtime_id: owner.id(),
            placement: Placement::new(DeviceId::new("cpu:0")),
            layout: Layout::contiguous(vec![2]),
        });

        assert!(buffer.validate_owner(owner.id()).is_ok());
        assert_eq!(
            buffer.validate_owner(foreign.id()),
            Err(BackendError::InvalidHandle {
                expected_runtime: foreign.id(),
                actual_runtime: Some(owner.id()),
            })
        );
        assert!(buffer.downcast_ref::<TestBuffer>().is_some());
    }
}

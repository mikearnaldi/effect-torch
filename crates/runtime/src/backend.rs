//! Runtime identity, device placement and buffer ownership.
//!
//! Every live runtime owns a process-unique [`RuntimeId`] from a global
//! counter ([`RuntimeId::new`]). Its buffers carry that id. Entry points
//! validate the id before using a buffer. See [`ErasedBuffer::validate_owner`]
//! and [`ProgramSignature::validate_invocation`](crate::ProgramSignature::validate_invocation).
//! Using a buffer with the wrong runtime returns
//! [`BackendError::InvalidHandle`] instead of risking undefined behavior
//! in the GPU stack.
//!
//! [`RuntimeIdentity`] contains a [`RuntimeId`] and human-readable backend
//! name. [`DeviceId`] names one device. [`Placement`] assigns a buffer to a
//! device and optional memory space. [`Capabilities`] lists the dtypes and
//! features a backend supports, so compilation can reject programs that could
//! not run.
//!
//! [`Buffer`] exposes a backend buffer's owner id, placement, dtype, layout,
//! and [`Any`] representation. [`ErasedBuffer`] is an `Arc`-shared,
//! type-erased handle with checked downcasting through
//! [`ErasedBuffer::downcast_ref`].

use crate::{BackendError, BackendResult, DType, Layout};
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

/// Process-unique identifier of one runtime instance.
///
/// A monotonically increasing global counter allocates ids. It starts at 1
/// and never issues 0 or reuses an id within a process. Equal ids therefore
/// identify the same owning runtime.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeId(u64);

impl RuntimeId {
    /// Returns a runtime id not previously issued in this process.
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
/// Two identities with the same backend name remain distinct. The id is the
/// ownership token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeIdentity {
    id: RuntimeId,
    backend: Arc<str>,
}

impl RuntimeIdentity {
    /// Creates an identity with a new id for the named backend.
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

/// A buffer's device and optional named memory space, such as a Metal shared
/// or private heap.
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
    /// Host and device share one address space, so transfers require no copy.
    UnifiedMemory,
}

/// The set of dtypes and [`Capability`] features a backend supports.
///
/// The lists are ordered vectors. Duplicates do not affect membership tests.
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
/// buffer's current contents. Buffers must implement `Send + Sync` because
/// execution pools move them across thread boundaries.
pub trait Buffer: Any + fmt::Debug + Send + Sync {
    fn runtime_id(&self) -> RuntimeId;
    fn placement(&self) -> &Placement;
    fn dtype(&self) -> DType;
    fn layout(&self) -> &Layout;
    fn as_any(&self) -> &dyn Any;
}

/// Reference-counted, type-erased buffer handle.
///
/// Cloning only increments an `Arc`, and all clones share the same buffer.
/// Recover the concrete backend type with
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

    /// Returns [`BackendError::InvalidHandle`] if runtime `expected` does
    /// not own the buffer.
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

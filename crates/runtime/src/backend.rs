use crate::{BackendError, BackendResult, DType, Layout};
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeId(u64);

impl RuntimeId {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeIdentity {
    id: RuntimeId,
    backend: Arc<str>,
}

impl RuntimeIdentity {
    pub fn new(backend: impl Into<Arc<str>>) -> Self {
        Self {
            id: RuntimeId::new(),
            backend: backend.into(),
        }
    }

    pub fn id(&self) -> RuntimeId {
        self.id
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Placement {
    device: DeviceId,
    memory_space: Option<Arc<str>>,
}

impl Placement {
    pub fn new(device: DeviceId) -> Self {
        Self {
            device,
            memory_space: None,
        }
    }

    pub fn with_memory_space(device: DeviceId, memory_space: impl Into<Arc<str>>) -> Self {
        Self {
            device,
            memory_space: Some(memory_space.into()),
        }
    }

    pub fn device(&self) -> &DeviceId {
        &self.device
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    Compilation,
    AsyncExecution,
    UnifiedMemory,
}

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

pub trait Buffer: Any + fmt::Debug + Send + Sync {
    fn runtime_id(&self) -> RuntimeId;
    fn placement(&self) -> &Placement;
    fn dtype(&self) -> DType;
    fn layout(&self) -> &Layout;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone)]
pub struct ErasedBuffer(Arc<dyn Buffer>);

impl ErasedBuffer {
    pub fn new(buffer: impl Buffer) -> Self {
        Self(Arc::new(buffer))
    }

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

    pub fn downcast_ref<B: Buffer>(&self) -> Option<&B> {
        self.0.as_any().downcast_ref()
    }

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

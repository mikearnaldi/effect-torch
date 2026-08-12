pub mod composed;
pub mod conv;
pub mod executable;
pub mod fusion;
pub mod indexing;
pub mod linalg;
pub mod matmul;
#[cfg(feature = "napi-addon")]
pub mod napi;
pub mod ops;
pub mod pool;
mod quantized;
pub mod random;
pub mod reduce;
pub mod storage;
pub mod tensor;
pub mod value;
mod workspace;

pub use storage::{
    CpuElement, CpuSegment, CpuStorage, CpuStorageError, ExecutableAllocationGuard,
    CPU_STORAGE_ALIGNMENT,
};
pub use tensor::{
    CpuBuffer, CpuDestination, CpuOperationRequirements, CpuTensorRequirement, Elem, Tensor,
};
pub use value::Value;

use effect_torch_runtime::{Buffer, DType, DeviceId, Placement, RuntimeIdentity};
use std::any::Any;
use std::sync::OnceLock;

fn identity() -> &'static RuntimeIdentity {
    static IDENTITY: OnceLock<RuntimeIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| RuntimeIdentity::new("cpu"))
}

fn placement() -> &'static Placement {
    static PLACEMENT: OnceLock<Placement> = OnceLock::new();
    PLACEMENT.get_or_init(|| Placement::new(DeviceId::new("cpu:0")))
}

impl Buffer for Tensor {
    fn runtime_id(&self) -> effect_torch_runtime::RuntimeId {
        identity().id()
    }

    fn placement(&self) -> &Placement {
        placement()
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn layout(&self) -> &effect_torch_runtime::Layout {
        &self.layout
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

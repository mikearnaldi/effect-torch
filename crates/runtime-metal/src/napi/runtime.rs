pub mod dtype {
    pub use effect_torch_runtime::DType;
}

pub mod layout {
    pub use effect_torch_runtime::Layout;
}

pub mod metal {
    pub use crate::{device, run};
}

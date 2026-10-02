//! Persistent attention storage contracts shared by the compiler and backends.

use crate::DType;

/// Access granted to a stateful executable. An absent state schema is stateless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateAccessMode {
    Append,
    ReadOnly,
}

/// One stable attention layer in a persistent K/V schema.
///
/// Both K and V use logical `[batch, kv_heads, tokens, head_dim]` layout and
/// the declared storage dtype. `retention` limits retained prefix rows, not
/// query visibility or absolute sequence positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvLayerDescriptor {
    pub layer_id: u32,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub retention: Option<usize>,
}

impl KvLayerDescriptor {
    pub fn row_bytes(self) -> Option<usize> {
        self.kv_heads
            .checked_mul(self.head_dim)?
            .checked_mul(self.dtype.size_in_bytes())?
            .checked_mul(2)
    }
}

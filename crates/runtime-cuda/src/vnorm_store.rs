//! Private V normalization retained inside the transactional KV store.
use crate::executable::CudaStateLayout;
use effect_torch_runtime::DType;

#[cfg(test)]
thread_local! { static TEST_POLICY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(crate) fn with_test_policy<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_POLICY.with(|p| p.set(self.0));
        }
    }
    let _restore = Restore(TEST_POLICY.with(|p| p.replace(Some(enabled))));
    run()
}
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_POLICY.with(|p| p.get()) {
        return value;
    }
    std::env::var("EFFECT_TORCH_CUDA_VNORM_STORE").as_deref() == Ok("1")
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VNormStorePlan {
    pub(crate) tokens: usize,
    pub(crate) heads: usize,
    pub(crate) dim: usize,
    pub(crate) eps: f64,
}

impl VNormStorePlan {
    pub(crate) fn for_layout(
        layout: Option<&CudaStateLayout>,
        layer: u32,
        shape: &[usize],
        eps: f64,
    ) -> Option<Self> {
        let [1, heads, tokens, dim] = shape else {
            return None;
        };
        if !matches!((*heads, *dim), (8, 256) | (2, 512))
            || !matches!(*tokens, 64 | 256)
            || eps.to_bits() != 1e-6_f64.to_bits()
        {
            return None;
        }
        let layout = layout?;
        if layout.validate().is_err()
            || layout.slots != 1
            || layout.packed_rows_per_sequence.unwrap_or(1) != 1
            || layout.dtype != DType::BF16
            || !layout.kv_layers.iter().any(|d| {
                d.layer_id == layer
                    && d.dtype == DType::BF16
                    && d.kv_heads == *heads
                    && d.head_dim == *dim
            })
        {
            return None;
        }
        let positions = (layout.capacity as usize).checked_add(*tokens)?;
        let table_bytes = positions.checked_add(1)?.checked_mul(32)?;
        let score_bytes = 16_usize
            .checked_mul(*tokens)?
            .checked_mul(positions)?
            .checked_mul(4)?;
        crate::kv_matmul::KvMatmulWorkspace::new(
            table_bytes.checked_add(score_bytes)?,
            16,
            *tokens,
            positions,
            *dim,
        )?;
        Some(Self {
            tokens: *tokens,
            heads: *heads,
            dim: *dim,
            eps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_compiler::{CompileOptions, CompilerDriver, NativeRegion, ProgramRequest};
    use effect_torch_graph::{AttentionRounding, Device, KvAttentionMode, Node, NodeKind};
    use effect_torch_runtime::{KvLayerDescriptor, StateAccessMode, StorageMetadata};
    use std::sync::Arc;
    fn input(slot: u32, shape: &[usize]) -> Arc<Node> {
        Node::new(NodeKind::Input {
            slot,
            shape: shape.to_vec(),
            dtype: DType::BF16,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    }
    #[test]
    fn private_vnorm_region_selection_and_semantic_outputs() {
        for (heads, dim) in [(8, 256), (2, 512)] {
            for tokens in [64, 256] {
                for escape in 0..4 {
                    for enabled in [false, true] {
                        let raw = input(2, &[1, tokens, heads, dim]);
                        let view = Node::new(NodeKind::Permute {
                            a: raw.clone(),
                            dims: vec![0, 2, 1, 3],
                        })
                        .unwrap();
                        let normalized = Node::new(NodeKind::RmsNorm {
                            x: view,
                            weight: None,
                            eps: 1e-6,
                        })
                        .unwrap();
                        let attention = Node::new(NodeKind::KvAttention {
                            q: input(0, &[1, 16, tokens, dim]),
                            k: input(1, &[1, heads, tokens, dim]),
                            v: normalized.clone(),
                            scale: 1.,
                            layer: 0,
                            window: None,
                            mode: KvAttentionMode::Causal,
                            rounding: AttentionRounding::Stepwise,
                        })
                        .unwrap();
                        let mut roots = vec![attention.clone()];
                        if escape == 1 {
                            roots.push(normalized.clone());
                        }
                        if escape == 2 {
                            roots.push(
                                Node::new(NodeKind::Neg {
                                    a: normalized.clone(),
                                })
                                .unwrap(),
                            );
                        }
                        if escape == 3 {
                            roots.push(raw);
                        }
                        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                            .prepare()
                            .unwrap();
                        let capabilities = with_test_policy(enabled, || {
                            crate::capabilities::CudaCapabilities::new(0, 12, 0)
                        });
                        let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
                        let regions = driver
                            .optimization()
                            .regions
                            .iter()
                            .filter(|r| matches!(r, NativeRegion::VNormKvAttention(_)))
                            .collect::<Vec<_>>();
                        assert_eq!(
                            regions.len(),
                            usize::from(enabled && (escape == 0 || escape == 3))
                        );
                        if let Some(region) = regions.first() {
                            assert_eq!(region.output_count(), 1);
                            assert_eq!(region.semantic_outputs().len(), 1);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn vnorm_store_layout_preflight_is_narrow() {
        let base = CudaStateLayout {
            capacity: 1024,
            dtype: DType::BF16,
            slots: 1,
            packed_rows_per_sequence: None,
            access: StateAccessMode::Append,
            kv_layers: vec![KvLayerDescriptor {
                layer_id: 0,
                kv_heads: 8,
                head_dim: 256,
                dtype: DType::BF16,
                retention: None,
            }],
        };
        assert!(VNormStorePlan::for_layout(Some(&base), 0, &[1, 8, 64, 256], 1e-6).is_some());
        for shape in [
            [2, 8, 64, 256],
            [1, 8, 32, 256],
            [1, 4, 64, 256],
            [1, 8, 64, 512],
        ] {
            assert!(VNormStorePlan::for_layout(Some(&base), 0, &shape, 1e-6).is_none());
        }
        assert!(VNormStorePlan::for_layout(None, 0, &[1, 8, 64, 256], 1e-6).is_none());
        assert!(VNormStorePlan::for_layout(Some(&base), 1, &[1, 8, 64, 256], 1e-6).is_none());
        assert!(VNormStorePlan::for_layout(Some(&base), 0, &[1, 8, 64, 256], 1e-5).is_none());
        let mut other = base.clone();
        other.dtype = DType::F32;
        assert!(VNormStorePlan::for_layout(Some(&other), 0, &[1, 8, 64, 256], 1e-6).is_none());
        let mut other = base;
        other.packed_rows_per_sequence = Some(2);
        assert!(VNormStorePlan::for_layout(Some(&other), 0, &[1, 8, 64, 256], 1e-6).is_none());
    }
}

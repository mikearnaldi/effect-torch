//! Pure CUDA dtype policy for the typed reference-kernel ABI and native
//! row-major BF16 GEMM.
use crate::cublas::{plan_row_bf16_gemm, RowGemmKind, BF16_GEMM_MIN_MAJOR};
use effect_torch_compiler::{
    DTypeDisposition, DTypeRequirement, ExecutionRealization, LayoutConstraintSpec, NativeRegion,
    OperationDTypeSpec, RegionDTypeSpec, StorageSupport, TargetBackend, TargetDTypeCapabilities,
    TargetFingerprint, UnsupportedDType, ValueRole, ValueSpec,
};
use effect_torch_graph::{Device, NodeKind};
use effect_torch_runtime::{DType, PackedFormat, StorageRepresentation};

#[derive(Clone, Debug)]
pub(crate) struct CudaCapabilities {
    device: Device,
    fingerprint: TargetFingerprint,
    bf16_gemm: bool,
    wide_sum_fusion: bool,
    wide_arg_fusion: bool,
    ordered_scatter_fusion: bool,
    ordered_scatter_weighted: bool,
    shared_rms: bool,
    norm_rope: bool,
    bf16_softmax: bool,
    entropy_recompute: bool,
    entropy_relaxed81: bool,
    small_softmax: bool,
    grouped_retained_layout: bool,
    ffn_tail: bool,
    ffn_next_norm63: bool,
    rms_residual: bool,
    attention_ffn_entrance: bool,
    vnorm_store: bool,
    router_tail: bool,
    dual_argmax: bool,
    expert_route_rank: bool,
    expert_finalize: bool,
    expert_pair_graph61: bool,
}
impl CudaCapabilities {
    #[cfg(test)]
    pub(crate) fn enable_norm98_fixture_regions(&mut self) {
        self.attention_ffn_entrance = true;
        self.ffn_next_norm63 = true;
    }

    pub(crate) fn new(ordinal: u32, major: i32, minor: i32) -> Self {
        let mut fingerprint =
            TargetFingerprint::new(TargetBackend::Cuda, format!("sm_{major}{minor}"), 6);
        let typed_binary = std::env::var("EFFECT_TORCH_CUDA_TYPED_BINARY").as_deref() == Ok("1");
        let expert_route_rank =
            std::env::var("EFFECT_TORCH_CUDA_EXPERT_ROUTE_RANK").as_deref() == Ok("1");
        let expert_finalize = std::env::var("EFFECT_TORCH_CUDA_EXPERT_FINALIZE").as_deref()
            == Ok("1")
            || crate::fused_moe75::enabled();
        let expert_pair_graph61 = crate::executable::expert_pair61::enabled();
        let bf16_gemm = major >= BF16_GEMM_MIN_MAJOR;
        let dual_argmax = std::env::var("EFFECT_TORCH_CUDA_DUAL_ARGMAX").as_deref() == Ok("1")
            || crate::rng_arg80::enabled();
        let router_tail = std::env::var("EFFECT_TORCH_CUDA_ROUTER_TAIL").as_deref() == Ok("1");
        let vnorm_store = crate::vnorm_store::enabled();
        let attention_ffn_entrance =
            std::env::var("EFFECT_TORCH_CUDA_ATTN_FFN_ENTRANCE").as_deref() == Ok("1");
        let rms_residual = std::env::var("EFFECT_TORCH_CUDA_RMS_RESIDUAL").as_deref() == Ok("1");
        let ffn_next_norm63 =
            std::env::var("EFFECT_TORCH_CUDA_FFN_NEXT_NORM63").as_deref() == Ok("1");
        let ffn_tail = std::env::var("EFFECT_TORCH_CUDA_FFN_TAIL").as_deref() == Ok("1");
        let grouped_retained_layout =
            std::env::var("EFFECT_TORCH_CUDA_GROUPED_RETAINED_LAYOUT").as_deref() == Ok("1");
        let ordered_scatter_fusion =
            std::env::var("EFFECT_TORCH_CUDA_ORDERED_SCATTER_FUSION").as_deref() == Ok("1");
        let ordered_scatter_weighted =
            std::env::var("EFFECT_TORCH_CUDA_ORDERED_SCATTER_WEIGHTED").as_deref() == Ok("1");
        let wide_arg_fusion =
            std::env::var("EFFECT_TORCH_CUDA_WIDE_ARG_FUSION").as_deref() == Ok("1");
        let wide_sum_fusion =
            std::env::var("EFFECT_TORCH_CUDA_WIDE_SUM_FUSION").as_deref() == Ok("1");
        let router_bitonic =
            std::env::var("EFFECT_TORCH_CUDA_ROUTER_BITONIC").as_deref() == Ok("1");
        let norm_rope = std::env::var("EFFECT_TORCH_CUDA_NORM_ROPE").as_deref() == Ok("1");
        let norm_rope_mask =
            std::env::var("EFFECT_TORCH_CUDA_NORM_ROPE_MASK").as_deref() == Ok("1");
        let shared_rms = std::env::var("EFFECT_TORCH_CUDA_SHARED_RMS").as_deref() == Ok("1");
        let entropy_recompute =
            std::env::var("EFFECT_TORCH_CUDA_ENTROPY_RECOMPUTE").as_deref() == Ok("1");
        let entropy_relaxed81 = crate::entropy81::enabled();
        let small_softmax = std::env::var("EFFECT_TORCH_CUDA_SMALL_SOFTMAX").as_deref() == Ok("1");
        let bf16_softmax = std::env::var("EFFECT_TORCH_CUDA_BF16_SOFTMAX").as_deref() == Ok("1");
        let ordinary_k16 =
            std::env::var(crate::cublas::ordinary_k16::PATH_ENV).is_ok_and(|path| !path.is_empty());
        let mut features = vec!["typed-reference-kernels-v1".into()];
        features.push(format!(
            "triton-normrope101-proven-full-or-half-tables-fixed-ptx-v2-{:?}",
            std::env::var_os(crate::triton_normrope101::DIRECTORY_ENV)
        ));
        features.push(format!(
            "triton-norm98-paired-bf16-rho-fixed-ptx-v1-{:?}",
            std::env::var_os(crate::triton_norm98::DIRECTORY_ENV)
        ));
        features.push(format!(
            "online-attention75-relaxed-bf16-{:?}",
            std::env::var_os(crate::attention75::DIRECTORY_ENV)
        ));
        features.push(format!(
            "fused-moe75-relaxed-bf16-{}-{:?}-{:?}-{:?}",
            crate::fused_moe75::enabled(),
            std::env::var_os("EFFECT_TORCH_CUDA_FUSED_MOE75_SO"),
            std::env::var_os("EFFECT_TORCH_CUDA_FUSED_MOE75_GEMM1"),
            std::env::var_os("EFFECT_TORCH_CUDA_FUSED_MOE75_GEMM2")
        ));
        features.push(format!(
            "kv-pair-independent-k16-v1-{}",
            crate::kv_pair::enabled()
        ));
        let mean256 = std::env::var("EFFECT_TORCH_CUDA_MEAN256").as_deref() == Ok("1");
        features.push(format!("mean256-contiguous-f32-sequential-v1-{mean256}"));
        let kv_bf16_inputs =
            std::env::var("EFFECT_TORCH_CUDA_KV_BF16_INPUTS").as_deref() == Ok("1");
        features.push(format!("kv-bf16-inputs-f32-output-v1-{kv_bf16_inputs}"));
        features.push(format!(
            "attention82-bf16-io-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_ATTENTION82_BF16_IO").as_deref() == Ok("1")
        ));
        features.push(format!(
            "dual-argmax-external-uniform-f32-i64-v1-{dual_argmax}"
        ));
        features.push(format!(
            "router-tail-probability-top8-exact-v1-{router_tail}"
        ));
        features.push(format!(
            "attention-ffn-entrance-four-output-v1-{attention_ffn_entrance}"
        ));
        features.push(format!("rms-residual-bf16-v1-{rms_residual}"));
        features.push(format!("vnorm-private-kv-store-v1-{vnorm_store}"));
        if crate::lowering::rotary_reuse66::enabled() {
            features.push(crate::lowering::rotary_reuse66::FEATURE.into());
        }
        if ffn_next_norm63 {
            features.push("ffn-next-norm63-two-output-v1".into());
        }
        if expert_pair_graph61 {
            features.push("expert-pair-explicit-graph61-true".into());
        }
        features.push(format!(
            "expert-device59-{}",
            crate::expert_device::enabled()
        ));
        features.push(format!(
            "typed-binary-static-op-dtype-dense-scalar-v1-{typed_binary}"
        ));
        features.push(format!(
            "expert-route-rank-u32-explicit-positions-v1-{expert_route_rank}"
        ));
        features.push(format!(
            "expert-finalize-private-inverse-bf16-v1-{expert_finalize}"
        ));
        features.push(format!("ffn-tail-private-bf16-width2816-v1-{ffn_tail}"));
        features.push(format!(
            "packed-projection77-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_PACKED_PROJECTION77").as_deref() == Ok("1")
        ));
        features.push(format!(
            "relaxed-full-cta-rms76-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_RELAXED_NORM76").as_deref() == Ok("1")
        ));
        features.push(format!(
            "grouped-retained-layout-v1-{grouped_retained_layout}"
        ));
        features.push(format!(
            "grouped-stable-block-rows-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_GROUPED_ROWS_BLOCK").as_deref() == Ok("1")
        ));
        features.push(format!(
            "grouped-vector-copy-bf16-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_GROUPED_VECTOR_COPY").as_deref() == Ok("1")
        ));
        features.push(format!(
            "fused-static-storage-dtypes-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_FUSED_STATIC_DTYPES").as_deref() == Ok("1")
        ));
        features.push(format!(
            "grouped-status-summary-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_GROUPED_STATUS_SUMMARY").as_deref() == Ok("1")
        ));
        features.push(format!(
            "expert-merged-upload128-v2-{}",
            std::env::var("EFFECT_TORCH_CUDA_EXPERT_UPLOAD128").as_deref() == Ok("1")
        ));
        features.push(format!("shared-rms-f32-products-v1-{shared_rms}"));
        features.push(format!("norm-rope-private-bf16-v1-{norm_rope}"));
        features.push(format!("norm-rope-power-two-mask-v1-{norm_rope_mask}"));
        features.push(format!(
            "entropy-recompute-exact-trees-v1-{entropy_recompute}"
        ));
        features.push(format!(
            "entropy81-experimental-f32-moment-v1-{entropy_relaxed81}"
        ));
        features.push(format!(
            "sampler83-private-rng80-entropy81-v1-{}",
            crate::sampler83::enabled()
        ));
        features.push(format!("small-softmax-exact-warp-v1-{small_softmax}"));
        features.push(format!("bf16-softmax-exact-reductions-v1-{bf16_softmax}"));
        features.push(format!(
            "ordinary-k16-m256-n2048-2112-k2816-v1-{ordinary_k16}"
        ));
        features.push(format!(
            "div-f32-feedback-bf16-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_DIV_FEEDBACK").as_deref() == Ok("1")
        ));
        features.push(format!(
            "f32-bf16-bits-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_F32_BF16_BITS").as_deref() == Ok("1")
        ));
        features.push(format!("ordered-scatter-bf16-v1-{ordered_scatter_fusion}"));
        features.push(format!(
            "ordered-scatter-weighted-bf16-v1-{ordered_scatter_weighted}"
        ));
        features.push(format!("router-bitonic-width128-v1-{router_bitonic}"));
        features.push(format!(
            "rms-vector-loads-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_RMS_VECTOR_LOADS").as_deref() == Ok("1")
        ));
        features.push(format!("fused-wide-f32-arg-v1-{wide_arg_fusion}"));
        features.push(format!("rng-dualarg80-v1-{}", crate::rng_arg80::enabled()));
        features.push(format!(
            "expert-gemv-dedicated-stream-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_EXPERT_GEMV_DEDICATED_STREAM").as_deref() == Ok("1")
        ));
        features.push(format!(
            "expert-gemv-overlap-independent-descriptors-v1-{}",
            crate::device::expert_gemv_overlap::enabled()
        ));
        features.push("grouped-expert-stable-exact-rows-host-control-v1".into());
        features.push("grouped-expert-routing-reuse-v1".into());
        features.push(format!(
            "expert-merged-register-splits-v1-{}",
            std::env::var("EFFECT_TORCH_CUDA_EXPERT_MERGED_PTX").is_ok_and(|path| !path.is_empty())
        ));
        features.push("grouped-expert-compact-routed-input-v1".into());
        features.push("rms-f32-warp-four-partials-mean-factor-rsqrt-v1".into());
        features.push("rms-f32-wide-block-output-v1".into());
        features.push("rms-f32-row-permute-view-v1".into());
        features.push("rotary-half-reindex-v1".into());
        features.push("sum-f32-warp-or-block1024-vector4-v2".into());
        if wide_sum_fusion {
            features.push("fused-wide-f32-sum-v1".into());
        }
        features.push("fused-elementwise-nvrtc-v1".into());
        features.push("fused-elementwise-scalar-inputs-v1".into());
        features.push("top-k-full-width256-stable-sort-v1".into());
        features.push("binary-contiguous-scalar-row-broadcast-indexing-v1".into());
        features.push("fused-elementwise-repeat-view-v1".into());
        features.push("fused-elementwise-strided-views-v2".into());
        features.push("scatter-add-compact-inner-index-v1".into());
        features.push("kv-sequence-major-output-v1".into());
        if bf16_gemm {
            features.push("stepwise-bf16-kv-f32-gemm-active-rows-v1".into());
            features.push("cublas-bf16-row-major-f32-accum-v3".into());
            features.push("cublas-bf16-default-reduction".into());
            features.push("cublas-f32-output-no-reduced-precision-reduction".into());
            features.push("cublas-status-free-bias-epilogue-v1".into());
            features.push(format!(
                "cublas-expert-streams-{}",
                crate::cublas::EXPERT_BLAS_STREAMS
            ));
            features.push(format!(
                "cublas-workspace-bytes-{}",
                crate::cublas::CUBLAS_WORKSPACE_BYTES
            ));
        }
        features.sort();
        features.dedup();
        fingerprint.features = features.into_boxed_slice();
        Self {
            device: Device::Cuda(ordinal),
            fingerprint,
            bf16_gemm,
            wide_sum_fusion,
            wide_arg_fusion,
            ordered_scatter_fusion,
            ordered_scatter_weighted,
            shared_rms,
            norm_rope,
            bf16_softmax,
            entropy_recompute,
            entropy_relaxed81,
            small_softmax,
            grouped_retained_layout,
            ffn_tail,
            ffn_next_norm63,
            rms_residual,
            attention_ffn_entrance,
            vnorm_store,
            router_tail,
            dual_argmax,
            expert_route_rank,
            expert_finalize,
            expert_pair_graph61,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_rotary_reuse66(mut self, enabled: bool) -> Self {
        let feature = crate::lowering::rotary_reuse66::FEATURE;
        let mut features = self.fingerprint.features.into_vec();
        features.retain(|f| f != feature);
        if enabled {
            features.push(feature.into());
        }
        features.sort();
        self.fingerprint.features = features.into_boxed_slice();
        self
    }

    pub(crate) fn for_device(device: &crate::CudaDevice) -> Result<Self, String> {
        let (major, minor) = device
            .stream
            .context()
            .compute_capability()
            .map_err(|e| e.to_string())?;
        let mut driver = 0;
        unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver) }
            .result()
            .map_err(|e| e.to_string())?;
        let (mut nvrtc_major, mut nvrtc_minor) = (0, 0);
        unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) }
            .result()
            .map_err(|e| e.to_string())?;
        let mut capabilities = Self::new(device.ordinal, major, minor);
        capabilities.fingerprint.libraries = vec![
            format!("cublas-{}", device.cublas.version),
            format!("cuda-driver-{driver}"),
            format!("nvrtc-{nvrtc_major}.{nvrtc_minor}"),
        ]
        .into_boxed_slice();
        Ok(capabilities)
    }

    /// Native row-major BF16 GEMM recipe when hardware and shapes support it.
    fn native_bf16_gemm(
        &self,
        spec: &OperationDTypeSpec<'_>,
    ) -> Option<effect_torch_compiler::DTypeExecution> {
        if !self.bf16_gemm {
            return None;
        }
        let kind = match spec.operation {
            NodeKind::Linear { .. } => RowGemmKind::Linear,
            NodeKind::Matmul { .. } => RowGemmKind::Matmul,
            _ => return None,
        };
        if spec.required_numerics.compute_dtype != Some(DType::BF16) {
            return None;
        }
        if spec
            .operands
            .iter()
            .chain(spec.results.iter())
            .any(|operand| {
                operand.value.semantic_dtype != DType::BF16
                    || operand.value.storage.representation != StorageRepresentation::Dense
            })
        {
            return None;
        }
        let activation = spec.operands.first()?;
        let weight = spec.operands.get(1)?;
        let result = spec.results.first()?;
        plan_row_bf16_gemm(
            kind,
            activation.value.logical_shape,
            weight.value.logical_shape,
            result.value.logical_shape,
        )?;
        Some(spec.native_execution())
    }
}
fn unsupported(requirement: DTypeRequirement, reason: impl Into<String>) -> DTypeDisposition {
    DTypeDisposition::Unsupported(UnsupportedDType::new(requirement, reason))
}
impl TargetDTypeCapabilities for CudaCapabilities {
    fn device(&self) -> &Device {
        &self.device
    }
    fn fingerprint(&self) -> &TargetFingerprint {
        &self.fingerprint
    }
    fn policy_revision(&self) -> u64 {
        // 61: Opt-in explicit expert-pair graph; 60 remains the eager policy.
        if self
            .fingerprint
            .features
            .iter()
            .any(|f| f == crate::lowering::rotary_reuse66::FEATURE)
        {
            66
        } else if self.ffn_next_norm63 {
            63
        } else if self.expert_pair_graph61 {
            61
        } else {
            60
        }
    }

    fn storage_support(&self, value: &ValueSpec<'_>) -> StorageSupport {
        let invalid = |reason| {
            StorageSupport::Unsupported(UnsupportedDType::new(DTypeRequirement::Storage, reason))
        };
        if matches!(
            value.storage.layout_constraint,
            LayoutConstraintSpec::DenseStrided { .. }
        ) {
            return invalid("CUDA binding storage must be canonical contiguous");
        }
        let elements = match crate::value::element_count(value.logical_shape) {
            Ok(n) => n,
            Err(_) => return invalid("CUDA shape overflows usize"),
        };
        match value.storage.representation {
            StorageRepresentation::Dense => {
                if elements
                    .checked_mul(value.semantic_dtype.size_in_bytes())
                    .is_none()
                {
                    invalid("CUDA dense byte size overflows usize")
                } else {
                    StorageSupport::Supported
                }
            }
            StorageRepresentation::Packed(PackedFormat::GgmlKQuant(codec)) => {
                if value.semantic_dtype != DType::F32 || value.logical_shape.len() != 2 {
                    return invalid("CUDA packed weights require logical F32 matrices");
                }
                match codec
                    .encoded_row_bytes(value.logical_shape[1])
                    .and_then(|bytes| bytes.checked_mul(value.logical_shape[0]))
                {
                    Some(_) => StorageSupport::Supported,
                    None => invalid("invalid CUDA packed block geometry"),
                }
            }
        }
    }
    fn classify_node(&self, spec: &OperationDTypeSpec<'_>) -> DTypeDisposition {
        if matches!(
            spec.operation,
            NodeKind::SdpaConfigured { .. } | NodeKind::RotaryEmbeddingExplicit { .. }
        ) {
            return unsupported(
                DTypeRequirement::Realization,
                "semantic operation requires native semantic preparation",
            );
        }
        if spec.placement != &self.device {
            return unsupported(
                DTypeRequirement::Layout,
                "CUDA device placement differs from the target",
            );
        }
        if let NodeKind::Argmax { a, dim } | NodeKind::Argmin { a, dim } = spec.operation {
            if a.shape.get(*dim).is_none_or(|extent| *extent == 0) {
                return unsupported(
                    DTypeRequirement::Layout,
                    "CUDA argmax/argmin requires a nonempty reduction axis",
                );
            }
        }
        if let NodeKind::TopKIndices { a, .. } = spec.operation {
            let rows = a.shape[..a.shape.len() - 1].iter().product::<usize>();
            if rows > i32::MAX as usize {
                return unsupported(
                    DTypeRequirement::Layout,
                    "CUDA topKIndices row grid exceeds i32::MAX",
                );
            }
        }
        for value in spec.operands.iter().chain(spec.results.iter()) {
            if let StorageSupport::Unsupported(error) = self.storage_support(&value.value) {
                return DTypeDisposition::Unsupported(error);
            }
        }
        for operand in &spec.operands {
            if operand.value.storage.representation != StorageRepresentation::Dense
                && !(operand.role == ValueRole::Weight
                    && matches!(
                        spec.operation,
                        NodeKind::QuantizedLinear { .. } | NodeKind::QuantizedEmbedding { .. }
                    ))
            {
                return unsupported(
                    DTypeRequirement::Operand(operand.role),
                    "packed CUDA values require canonical packed linear or embedding access",
                );
            }
        }
        if !matches!(spec.operation, NodeKind::Leaf(_) | NodeKind::Input { .. })
            && spec
                .results
                .iter()
                .any(|r| r.value.storage.representation != StorageRepresentation::Dense)
        {
            return unsupported(
                DTypeRequirement::Representation,
                "CUDA operation results must be dense",
            );
        }
        for result in spec.operands.iter().chain(spec.results.iter()) {
            // Storage-only sources do not launch the u32 reference kernels.
            // Expert dots have bounded grid-stride launches and u64 offsets,
            // including for banks with more than 2^32 stored elements.
            if matches!(
                spec.operation,
                NodeKind::Input { .. }
                    | NodeKind::Leaf(_)
                    | NodeKind::FromBytes { .. }
                    | NodeKind::ExpertLinearRows { .. }
                    | NodeKind::GroupedExpertLinearRows { .. }
            ) {
                continue;
            }
            let count = match crate::value::element_count(result.value.logical_shape) {
                Ok(count) => count,
                Err(error) => return unsupported(DTypeRequirement::Storage, error),
            };
            if count.div_ceil(256) > i32::MAX as usize {
                return unsupported(
                    DTypeRequirement::Realization,
                    "CUDA typed kernel grid exceeds the device grid limit",
                );
            }
            if result.value.logical_shape.len() > 64
                || count > u32::MAX as usize
                || result
                    .value
                    .logical_shape
                    .iter()
                    .any(|d| *d > u32::MAX as usize)
            {
                return unsupported(
                    DTypeRequirement::Layout,
                    "CUDA reference kernels require rank <=64 and u32 element counts and dimensions",
                );
            }
        }
        if spec
            .required_numerics
            .compute_dtype
            .is_some_and(|dtype| !dtype.is_float())
            && !matches!(
                spec.operation,
                NodeKind::Add { .. }
                    | NodeKind::Sub { .. }
                    | NodeKind::Mul { .. }
                    | NodeKind::Div { .. }
                    | NodeKind::Maximum { .. }
                    | NodeKind::Minimum { .. }
                    | NodeKind::Eq { .. }
                    | NodeKind::Gt { .. }
                    | NodeKind::Lt { .. }
                    | NodeKind::Ge { .. }
                    | NodeKind::Le { .. }
                    | NodeKind::Neg { .. }
                    | NodeKind::Abs { .. }
                    | NodeKind::Relu { .. }
                    | NodeKind::Sign { .. }
                    | NodeKind::Floor { .. }
                    | NodeKind::Ceil { .. }
                    | NodeKind::Round { .. }
                    | NodeKind::Sum { .. }
                    | NodeKind::Prod { .. }
                    | NodeKind::Max { .. }
                    | NodeKind::Min { .. }
                    | NodeKind::Mean { .. }
                    | NodeKind::Argmax { .. }
                    | NodeKind::Argmin { .. }
                    | NodeKind::Cumsum { .. }
                    | NodeKind::IndexSelect { .. }
                    | NodeKind::Gather { .. }
                    | NodeKind::ScatterAdd { .. }
                    | NodeKind::Where { .. }
                    | NodeKind::Arange { .. }
                    | NodeKind::Eye { .. }
                    | NodeKind::LastTokenRow { .. }
                    | NodeKind::PositionEmbedding { .. }
            )
        {
            return unsupported(
                DTypeRequirement::Compute,
                "CUDA operation has no exact integer kernel",
            );
        }
        if let NodeKind::GroupedExpertLinearRows { x, weight, .. } = spec.operation {
            if [x.shape[0], x.shape[1], weight.shape[1]]
                .iter()
                .any(|&n| n > i32::MAX as usize)
                || weight.shape[0] > (u32::MAX - 2) as usize
            {
                return unsupported(
                    DTypeRequirement::Layout,
                    "CUDA grouped expert dimensions exceed bounded control or GEMM limits",
                );
            }
            if x.dtype == DType::BF16
                && !self.bf16_gemm
                && x.shape.iter().all(|&n| n != 0)
                && weight.shape[1] != 0
            {
                return unsupported(
                    DTypeRequirement::Compute,
                    "CUDA grouped BF16 GEMM requires compute capability >= 8.0",
                );
            }
            return DTypeDisposition::Native(spec.native_execution());
        }
        if let Some(execution) = self.native_bf16_gemm(spec) {
            return DTypeDisposition::Native(execution);
        }
        let execution = spec.native_execution();
        if spec.required_numerics.permits_f32_compute {
            return DTypeDisposition::Legalize(
                execution.promote_half(ExecutionRealization::MaterializedTransforms),
            );
        }
        if matches!(
            spec.required_numerics.compute_dtype,
            Some(DType::F16 | DType::BF16)
        ) && !matches!(
            spec.operation,
            NodeKind::Argmax { .. }
                | NodeKind::Argmin { .. }
                | NodeKind::IndexSelect { .. }
                | NodeKind::Gather { .. }
                | NodeKind::ScatterAdd { .. }
                | NodeKind::LastTokenRow { .. }
                | NodeKind::PositionEmbedding { .. }
        ) {
            return unsupported(
                DTypeRequirement::Compute,
                "this half operation has no approved F32 execution contract",
            );
        }
        DTypeDisposition::Native(execution)
    }
    fn classify_region(&self, spec: &RegionDTypeSpec<'_>) -> DTypeDisposition {
        if let NativeRegion::DualArgmax(_) = spec.region {
            if !self.dual_argmax
                || spec.boundary_inputs.len() != 2
                || spec.boundary_inputs.iter().any(|v| {
                    v.value.semantic_dtype != DType::F32
                        || v.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "dual argmax requires enabled dense F32 inputs",
                );
            }
            return DTypeDisposition::Native(spec.native_execution());
        }
        if let NativeRegion::RouterTail(_) = spec.region {
            if !self.router_tail
                || spec.boundary_inputs.len() != 2
                || spec.boundary_inputs[0].value.semantic_dtype != DType::F32
                || !matches!(
                    spec.boundary_inputs[1].value.semantic_dtype,
                    DType::BF16 | DType::F32
                )
                || spec
                    .boundary_inputs
                    .iter()
                    .any(|v| v.value.storage.representation != StorageRepresentation::Dense)
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "router tail requires enabled dense F32 scores and BF16/F32 scales",
                );
            }
            return DTypeDisposition::Native(spec.native_execution());
        }
        if let NativeRegion::VNormKvAttention(_) = spec.region {
            if !self.vnorm_store
                || !self.bf16_gemm
                || spec.boundary_inputs.iter().any(|v| {
                    v.value.semantic_dtype != DType::BF16
                        || v.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "private V norm/store requires enabled BF16 KV GEMM",
                );
            }
            let mut execution = spec.native_execution();
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) | DTypeDisposition::Legalize(plan) => {
                        *recipe = plan.operations[0].clone()
                    }
                }
            }
            execution.realization = ExecutionRealization::KernelLocal;
            return DTypeDisposition::Legalize(execution);
        }
        if let NativeRegion::AttentionFfnEntrance(region) = spec.region {
            if !self.attention_ffn_entrance
                || !matches!(region.rows, 64 | 256)
                || spec.boundary_inputs.len() != 7
                || spec
                    .boundary_inputs
                    .iter()
                    .enumerate()
                    .any(|(slot, value)| {
                        value.value.semantic_dtype
                            != if slot == 6 { DType::F32 } else { DType::BF16 }
                            || value.value.storage.representation != StorageRepresentation::Dense
                    })
                || spec.boundary_results.iter().any(|value| {
                    value.value.semantic_dtype != DType::BF16
                        || value.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "attention FFN entrance requires enabled dense BF16 width2816 and original F32 scale",
                );
            }
            let mut execution = spec.native_execution();
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) | DTypeDisposition::Legalize(plan) => {
                        *recipe = plan.operations[0].clone()
                    }
                }
            }
            execution.realization = ExecutionRealization::KernelLocal;
            return DTypeDisposition::Legalize(execution);
        }
        if let NativeRegion::RmsResidual(region) = spec.region {
            if !self.rms_residual
                || !matches!(region.rows, 64 | 256)
                || spec.boundary_inputs.len() != 3
                || spec.boundary_inputs.iter().any(|value| {
                    value.value.semantic_dtype != DType::BF16
                        || value.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "RMS residual requires enabled dense width2816 rows",
                );
            }
            let mut execution = spec.native_execution();
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) | DTypeDisposition::Legalize(plan) => {
                        *recipe = plan.operations[0].clone()
                    }
                }
            }
            execution.realization = ExecutionRealization::KernelLocal;
            return DTypeDisposition::Legalize(execution);
        }
        if let NativeRegion::FfnNextNorm(region) = spec.region {
            if !self.ffn_next_norm63
                || !matches!(region.rows, 64 | 256)
                || spec.boundary_inputs.len() != 8
                || spec.boundary_results.iter().any(|value| {
                    value.value.semantic_dtype != DType::BF16
                        || value.value.storage.representation != StorageRepresentation::Dense
                })
                || spec.boundary_inputs.iter().any(|value| {
                    value.value.semantic_dtype != DType::BF16
                        || value.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "FFN next norm requires enabled dense BF16 width2816 rows",
                );
            }
            let mut execution = spec.native_execution();
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) | DTypeDisposition::Legalize(plan) => {
                        *recipe = plan.operations[0].clone()
                    }
                }
            }
            execution.realization = ExecutionRealization::KernelLocal;
            return DTypeDisposition::Legalize(execution);
        }
        if let NativeRegion::FfnTail(region) = spec.region {
            if !self.ffn_tail
                || !(1..=65535).contains(&region.rows)
                || spec.boundary_inputs.len() != 7
                || spec.boundary_inputs.iter().any(|value| {
                    value.value.semantic_dtype != DType::BF16
                        || value.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "FFN tail requires enabled dense BF16 width2816 rows",
                );
            }
            let mut execution = spec.native_execution();
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) | DTypeDisposition::Legalize(plan) => {
                        *recipe = plan.operations[0].clone()
                    }
                }
            }
            execution.realization = ExecutionRealization::KernelLocal;
            return DTypeDisposition::Legalize(execution);
        }
        if let NativeRegion::GroupedExpertGated(region) = spec.region {
            if region.finalizer.is_some()
                && (!self.expert_finalize
                    || !self.ordered_scatter_fusion
                    || !self.ordered_scatter_weighted
                    || spec.boundary_inputs.len() != 6)
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "private expert finalization requires retained layout and weighted ordered scatter",
                );
            }
            if !self.grouped_retained_layout
                || !self.bf16_gemm
                || spec
                    .boundary_inputs
                    .iter()
                    .any(|value| value.value.storage.representation != StorageRepresentation::Dense)
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "grouped retained layout requires enabled dense BF16 GEMM",
                );
            }
            let mut execution = spec.native_execution();
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) | DTypeDisposition::Legalize(plan) => {
                        *recipe = plan.operations[0].clone()
                    }
                }
            }
            execution.realization = ExecutionRealization::KernelLocal;
            return DTypeDisposition::Legalize(execution);
        }
        if let NativeRegion::SmallSoftmax(region) = spec.region {
            if !self.small_softmax
                || region.width != 128
                || spec.boundary_inputs.len() != 1
                || spec.boundary_inputs[0].value.semantic_dtype != DType::F32
                || spec.boundary_inputs[0].value.storage.representation
                    != StorageRepresentation::Dense
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "small softmax requires enabled dense width-128 F32 rows",
                );
            }
            return DTypeDisposition::Native(spec.native_execution());
        }
        if let NativeRegion::Entropy(region) = spec.region {
            let rows = region
                .shape
                .iter()
                .try_fold(1usize, |elements, &extent| elements.checked_mul(extent))
                .and_then(|elements| elements.checked_div(region.width));
            if (!self.entropy_recompute && !self.entropy_relaxed81)
                || region.width < 4096
                || !rows.is_some_and(|rows| rows > 0 && rows <= 65535)
                || spec.boundary_inputs.len() != 1
                || spec.boundary_inputs[0].value.semantic_dtype != DType::F32
                || spec.boundary_inputs[0].value.storage.representation
                    != StorageRepresentation::Dense
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "entropy recompute requires enabled dense wide F32 rows",
                );
            }
            return DTypeDisposition::Native(spec.native_execution());
        }
        if let NativeRegion::Bf16Softmax(region) = spec.region {
            if !self.bf16_softmax
                || region.width < 4096
                || spec.boundary_inputs.len() != 1
                || spec.boundary_inputs[0].value.semantic_dtype != DType::BF16
                || spec.boundary_inputs[0].value.storage.representation
                    != StorageRepresentation::Dense
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "BF16 softmax requires enabled dense wide rows",
                );
            }
            let mut execution = spec.native_execution();
            let mut transformed = false;
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) => *recipe = plan.operations[0].clone(),
                    DTypeDisposition::Legalize(plan) => {
                        transformed = true;
                        *recipe = plan.operations[0].clone();
                    }
                }
            }
            return if transformed {
                execution.realization = ExecutionRealization::KernelLocal;
                DTypeDisposition::Legalize(execution)
            } else {
                DTypeDisposition::Native(execution)
            };
        }
        if let NativeRegion::NormRope(region) = spec.region {
            if !self.norm_rope
                || region.shape.len() != 4
                || region.shape[0] != 1
                || !matches!(region.shape[1], 8 | 16)
                || region.shape[2] != 256
                || !matches!(region.shape[3], 256 | 512)
                || spec.boundary_inputs.iter().any(|v| {
                    v.value.semantic_dtype != DType::BF16
                        || v.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "norm/RoPE requires enabled private BF16 measured shapes",
                );
            }
            let mut execution = spec.native_execution();
            let mut transformed = false;
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) => *recipe = plan.operations[0].clone(),
                    DTypeDisposition::Legalize(plan) => {
                        transformed = true;
                        *recipe = plan.operations[0].clone();
                    }
                }
            }
            return if transformed {
                execution.realization = ExecutionRealization::KernelLocal;
                DTypeDisposition::Legalize(execution)
            } else {
                DTypeDisposition::Native(execution)
            };
        }
        if let NativeRegion::SharedRmsNorm(region) = spec.region {
            if !self.shared_rms
                || region.width < 1024
                || !(2..=3).contains(&region.nodes.len())
                || !matches!(region.dtype, DType::BF16 | DType::F32)
                || spec.boundary_inputs.iter().any(|v| {
                    v.value.semantic_dtype != region.dtype
                        || v.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "shared RMS requires enabled dense F32 or BF16 wide rows",
                );
            }
            let mut execution = spec.native_execution();
            let mut transformed = false;
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) => *recipe = plan.operations[0].clone(),
                    DTypeDisposition::Legalize(plan) => {
                        transformed = true;
                        *recipe = plan.operations[0].clone();
                    }
                }
            }
            return if transformed {
                execution.realization = ExecutionRealization::KernelLocal;
                DTypeDisposition::Legalize(execution)
            } else {
                DTypeDisposition::Native(execution)
            };
        }
        if let NativeRegion::ExpertRouteRank(region) = spec.region {
            if !self.expert_route_rank
                || region.rows == 0
                || !(1..=32).contains(&region.routes)
                || spec.boundary_inputs.len() != 2
                || spec.boundary_inputs.iter().any(|v| {
                    v.value.semantic_dtype != DType::U32
                        || v.value.storage.representation != StorageRepresentation::Dense
                })
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "route ranks require enabled dense U32 experts and positions",
                );
            }
            let mut execution = spec.native_execution();
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) | DTypeDisposition::Legalize(plan) => {
                        *recipe = plan.operations[0].clone()
                    }
                }
            }
            execution.realization = ExecutionRealization::DirectKernel;
            return DTypeDisposition::Native(execution);
        }
        if let NativeRegion::OrderedScatterReduce(region) = spec.region {
            if !self.ordered_scatter_fusion
                || region.dtype != DType::BF16
                || !(1..=32).contains(&region.routes)
                || region.shape.iter().any(|&d| d == 0)
                || spec.boundary_inputs.len() != if region.weighted_source { 3 } else { 2 }
                || (region.weighted_source
                    && (!self.ordered_scatter_weighted
                        || spec.boundary_inputs[2].value.semantic_dtype != DType::F32))
                || spec.boundary_inputs[0].value.semantic_dtype != DType::BF16
                || !matches!(
                    spec.boundary_inputs[1].value.semantic_dtype,
                    DType::U32 | DType::I64
                )
                || spec
                    .boundary_inputs
                    .iter()
                    .any(|v| v.value.storage.representation != StorageRepresentation::Dense)
            {
                return unsupported(
                    DTypeRequirement::Region,
                    "ordered scatter fusion requires nonempty BF16 rows and compact integer indexes",
                );
            }
            let mut execution = spec.native_execution();
            for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
                match self.classify_node(operation) {
                    DTypeDisposition::Unsupported(error) => {
                        return DTypeDisposition::Unsupported(error);
                    }
                    DTypeDisposition::Native(plan) | DTypeDisposition::Legalize(plan) => {
                        *recipe = plan.operations[0].clone()
                    }
                }
            }
            execution.realization = ExecutionRealization::KernelLocal;
            return DTypeDisposition::Legalize(execution);
        }
        let inputs = match spec.region {
            NativeRegion::Elementwise(region) => region.inputs.len(),
            NativeRegion::ElementwiseArgReduce(region)
                if self.wide_arg_fusion
                    && region.dtype == DType::I64
                    && !region.input_shape.is_empty()
                    && region.dim == region.input_shape.len() - 1
                    && (4096..=u32::MAX as usize).contains(region.input_shape.last().unwrap())
                    && spec
                        .boundary_inputs
                        .iter()
                        .all(|v| v.value.semantic_dtype == DType::F32)
                    && spec.operations.iter().all(|op| {
                        op.results.iter().all(|v| {
                            v.value.semantic_dtype == DType::F32
                                || (op.node == region.output
                                    && v.value.semantic_dtype == DType::I64)
                        })
                    }) =>
            {
                region.inputs.len()
            }
            NativeRegion::ElementwiseReduce(region)
                if self.wide_sum_fusion
                    && region.dtype == DType::F32
                    && region.op == effect_torch_compiler::ReduceOp::Sum
                    && !region.input_shape.is_empty()
                    && region.dims.as_ref() == [region.input_shape.len() - 1]
                    && region.input_shape.last().copied().unwrap_or(0) >= 4096
                    && spec
                        .boundary_inputs
                        .iter()
                        .all(|input| input.value.semantic_dtype == DType::F32)
                    && spec.operations.iter().all(|op| {
                        op.results
                            .iter()
                            .all(|v| v.value.semantic_dtype == DType::F32)
                    }) =>
            {
                region.inputs.len()
            }
            _ => {
                return unsupported(
                    DTypeRequirement::Region,
                    "unsupported CUDA optimization region",
                );
            }
        };
        if inputs > 8 {
            return unsupported(
                DTypeRequirement::Region,
                "CUDA fusion supports at most eight inputs",
            );
        }
        let Some(first) = spec.boundary_results.first() else {
            return unsupported(DTypeRequirement::Region, "region has no results");
        };
        let dtype = first.value.semantic_dtype;
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
            && !matches!(spec.region, NativeRegion::ElementwiseArgReduce(_))
        {
            return unsupported(
                DTypeRequirement::Region,
                "CUDA fusion requires F32, F16, or BF16 storage",
            );
        }
        if spec.boundary_inputs.iter().any(|input| {
            !matches!(
                input.value.semantic_dtype,
                DType::F32 | DType::F16 | DType::BF16
            ) || input.value.storage.representation != StorageRepresentation::Dense
        }) {
            return unsupported(
                DTypeRequirement::Region,
                "CUDA fusion requires floating dense inputs",
            );
        }
        let mut execution = spec.native_execution();
        let mut transformed = false;
        for (operation, recipe) in spec.operations.iter().zip(execution.operations.iter_mut()) {
            match self.classify_node(operation) {
                DTypeDisposition::Unsupported(error) => {
                    return DTypeDisposition::Unsupported(error);
                }
                DTypeDisposition::Native(plan) => *recipe = plan.operations[0].clone(),
                DTypeDisposition::Legalize(plan) => {
                    transformed = true;
                    *recipe = plan.operations[0].clone();
                }
            }
        }
        if transformed {
            execution.realization = ExecutionRealization::KernelLocal;
            DTypeDisposition::Legalize(execution)
        } else {
            DTypeDisposition::Native(execution)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_compiler::{GraphIndex, OperationDTypeSpec};
    use effect_torch_graph::Node;
    use std::sync::Arc;

    fn bf16_softmax_chain(width: usize, source_dtype: DType) -> Vec<Arc<Node>> {
        let make = |kind| Node::new(kind).unwrap();
        let source = make(NodeKind::Input {
            slot: 0,
            shape: vec![9, width],
            dtype: source_dtype,
            device: Device::Cuda(0),
            storage: effect_torch_runtime::StorageMetadata::dense(),
        });
        let floating = make(NodeKind::Cast {
            a: source,
            dtype: DType::F32,
        });
        let maximum = make(NodeKind::Max {
            a: floating.clone(),
            dims: vec![1],
            keepdims: true,
        });
        let shifted = make(NodeKind::Sub {
            a: floating.clone(),
            b: maximum.clone(),
        });
        let exponential = make(NodeKind::Exp { a: shifted.clone() });
        let total = make(NodeKind::Sum {
            a: exponential.clone(),
            dims: vec![1],
            keepdims: true,
        });
        let quotient = make(NodeKind::Div {
            a: exponential.clone(),
            b: total.clone(),
        });
        let output = make(NodeKind::Cast {
            a: quotient.clone(),
            dtype: DType::BF16,
        });
        vec![
            output,
            floating,
            maximum,
            shifted,
            exponential,
            total,
            quotient,
        ]
    }

    #[test]
    fn bf16_softmax_selection_requires_exact_pattern_opt_in_and_private_intermediates() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for (enabled, width, dtype, escape, expected) in [
            (true, 4096, DType::BF16, 0, true),
            (true, 262144, DType::BF16, 0, true),
            (false, 4096, DType::BF16, 0, false),
            (true, 4095, DType::BF16, 0, false),
            (true, 4096, DType::F16, 0, false),
            (true, 4096, DType::F32, 0, false),
            (true, 4096, DType::BF16, 1, false),
            (true, 4096, DType::BF16, 2, false),
            (true, 4096, DType::BF16, 3, false),
            (true, 4096, DType::BF16, 4, false),
            (true, 4096, DType::BF16, 5, false),
            (true, 4096, DType::BF16, 6, false),
        ] {
            let nodes = bf16_softmax_chain(width, dtype);
            let mut roots = vec![nodes[0].clone()];
            if escape != 0 {
                roots.push(nodes[escape].clone());
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut capabilities = CudaCapabilities::new(0, 12, 0);
            capabilities.bf16_softmax = enabled;
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            let selected = driver
                .optimization()
                .regions
                .iter()
                .any(|r| matches!(r, NativeRegion::Bf16Softmax(_)));
            assert_eq!(
                selected, expected,
                "{enabled} {width} {dtype:?} escape={escape}"
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA and EFFECT_TORCH_CUDA_BF16_SOFTMAX=1"]
    fn bf16_softmax_exact_wide_reductions_special_values_and_retained_outputs() {
        use effect_torch_compiler::CompileOptions;
        use effect_torch_runtime::CancellationFlag;
        assert_eq!(
            std::env::var("EFFECT_TORCH_CUDA_BF16_SOFTMAX").unwrap(),
            "1"
        );
        for width in [4096, 4097, 16385, 262144] {
            let roots = vec![bf16_softmax_chain(width, DType::BF16)[0].clone()];
            let optimized =
                crate::compile_with_options(roots.clone(), 0, CompileOptions::default()).unwrap();
            assert!(optimized
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "et_bf16_softmax_prepare" && i.count == 1));
            assert!(optimized
                .diagnostics()
                .instructions
                .iter()
                .any(|i| i.kind == "et_bf16_softmax_store" && i.count == 1));
            let reference = crate::compile_with_options(
                roots,
                0,
                CompileOptions {
                    optimize: false,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut retained = Vec::new();
            for seed in [17usize, 29] {
                let data = (0..9 * width)
                    .map(|i| {
                        let col = i % width;
                        match i / width {
                            0 => [-0.0, 0.0][col % 2],
                            1 => f32::from_bits(1 + col as u32 % 17) as f64,
                            2 => f64::NEG_INFINITY,
                            3 => {
                                if col == 0 {
                                    f64::NAN
                                } else {
                                    -1.0
                                }
                            }
                            4 => {
                                if col == width - 1 {
                                    f64::INFINITY
                                } else {
                                    2.0
                                }
                            }
                            5 => {
                                if col == width - 1 {
                                    -1e30
                                } else {
                                    1e30
                                }
                            }
                            6 => -(col as f64 % 200.0),
                            _ => (((i * 101 + seed) % 65521) as f64 - 32760.0) / 1024.0,
                        }
                    })
                    .collect::<Vec<_>>();
                let bindings = [crate::CudaValue::from_host(
                    crate::CudaDevice::get(0).unwrap(),
                    vec![9, width],
                    DType::BF16,
                    &data,
                )
                .unwrap()];
                let bytes = bindings[0].read_storage_bytes().unwrap();
                let cancelled = CancellationFlag::new();
                cancelled.cancel();
                assert!(optimized.execute(&bindings, &[], &cancelled).is_err());
                let expected = reference
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap()[0]
                    .read_storage_bytes()
                    .unwrap();
                let actual = optimized
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                assert_eq!(
                    actual[0].read_storage_bytes().unwrap(),
                    expected,
                    "width={width} seed={seed}"
                );
                assert_eq!(bindings[0].read_storage_bytes().unwrap(), bytes);
                retained.push((actual[0].clone(), expected));
            }
            drop(optimized);
            drop(reference);
            for (output, expected) in retained {
                assert_eq!(output.read_storage_bytes().unwrap(), expected);
            }
        }
    }

    #[test]
    fn shared_rms_selection_checks_views_epsilon_dtype_and_opt_in() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for (enabled, dtype, width, eps, expected) in [
            (true, DType::BF16, 2816, 1e-6, true),
            (true, DType::F32, 1025, 1e-6, true),
            (false, DType::BF16, 2816, 1e-6, false),
            (true, DType::F64, 2816, 1e-6, false),
            (true, DType::BF16, 512, 1e-6, false),
            (true, DType::BF16, 2816, 1e-5, false),
        ] {
            let input = |slot, shape| {
                Node::new(NodeKind::Input {
                    slot,
                    shape,
                    dtype,
                    device: Device::Cuda(0),
                    storage: effect_torch_runtime::StorageMetadata::dense(),
                })
                .unwrap()
            };
            let x = input(0, vec![1, 3, width]);
            let view = Node::new(NodeKind::Reshape {
                a: x.clone(),
                shape: vec![3, width],
            })
            .unwrap();
            let first = Node::new(NodeKind::RmsNorm {
                x: view.clone(),
                weight: None,
                eps: 1e-6,
            })
            .unwrap();
            let second = Node::new(NodeKind::RmsNorm {
                x,
                weight: Some(input(1, vec![width])),
                eps,
            })
            .unwrap();
            // The escaped view must still materialize and preserve its root.
            let prepared =
                ProgramRequest::from_roots(vec![first, second, view], CompileOptions::default())
                    .prepare()
                    .unwrap();
            let mut capabilities = CudaCapabilities::new(0, 12, 0);
            capabilities.shared_rms = enabled;
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            let regions: Vec<_> = driver
                .optimization()
                .regions
                .iter()
                .filter(|r| matches!(r, NativeRegion::SharedRmsNorm(_)))
                .collect();
            assert_eq!(
                regions.len(),
                usize::from(expected),
                "{enabled} {dtype:?} {width} {eps}"
            );
            driver.optimization().validate(&prepared.index).unwrap();
        }
    }

    #[test]
    fn shared_rms_schedules_late_views_and_rejects_dependent_weights() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        let make = |kind| Node::new(kind).unwrap();
        for dependent_weight in [false, true] {
            let x = make(NodeKind::Input {
                slot: 0,
                shape: vec![1, 1, 2816],
                dtype: DType::BF16,
                device: Device::Cuda(0),
                storage: effect_torch_runtime::StorageMetadata::dense(),
            });
            let first = make(NodeKind::RmsNorm {
                x: x.clone(),
                weight: None,
                eps: 1e-6,
            });
            let consumer = make(NodeKind::Neg { a: first.clone() });
            // Root traversal encounters this reshape after the first norm and
            // its consumer; fusion must move the view before both outputs.
            let late_view = make(NodeKind::Reshape {
                a: x,
                shape: vec![1, 2816],
            });
            let weight = dependent_weight.then(|| {
                make(NodeKind::Reshape {
                    a: first,
                    shape: vec![2816],
                })
            });
            let second = make(NodeKind::RmsNorm {
                x: late_view.clone(),
                weight,
                eps: 1e-6,
            });
            let prepared = ProgramRequest::from_roots(
                vec![consumer, second, late_view],
                CompileOptions::default(),
            )
            .prepare()
            .unwrap();
            let mut capabilities = CudaCapabilities::new(0, 12, 0);
            capabilities.shared_rms = true;
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            let count = driver
                .optimization()
                .regions
                .iter()
                .filter(|r| matches!(r, NativeRegion::SharedRmsNorm(_)))
                .count();
            assert_eq!(count, usize::from(!dependent_weight));
            driver.optimization().validate(&prepared.index).unwrap();
        }
    }

    fn ordered_scatter_chain(
        routes: usize,
        width: usize,
        dtype: DType,
        index_dtype: DType,
        reverse: bool,
    ) -> (Arc<Node>, Arc<Node>) {
        ordered_scatter_chain_with_weighting(routes, width, dtype, index_dtype, reverse, false)
    }

    fn ordered_scatter_chain_with_weighting(
        routes: usize,
        width: usize,
        dtype: DType,
        index_dtype: DType,
        reverse: bool,
        weighted: bool,
    ) -> (Arc<Node>, Arc<Node>) {
        let node = |kind| Node::new(kind).unwrap();
        let input = |slot, shape, dtype| {
            node(NodeKind::Input {
                slot,
                shape,
                dtype,
                device: Device::Cuda(0),
                storage: effect_torch_runtime::StorageMetadata::dense(),
            })
        };
        let indexes = input(1, vec![2, routes], index_dtype);
        let expanded = node(NodeKind::Reshape {
            a: indexes,
            shape: vec![2, routes, 1],
        });
        let indexes = node(NodeKind::BroadcastTo {
            a: expanded,
            shape: vec![2, routes, width],
        });
        let zero = |shape| {
            node(NodeKind::Zeros {
                shape,
                dtype,
                device: Device::Cuda(0),
            })
        };
        let source = if weighted {
            let route_major = node(NodeKind::Reshape {
                a: input(0, vec![2 * routes, width], dtype),
                shape: vec![routes, 2, width],
            });
            let token_major = node(NodeKind::Permute {
                a: route_major,
                dims: vec![1, 0, 2],
            });
            let floating = node(NodeKind::Cast {
                a: token_major,
                dtype: DType::F32,
            });
            let product = node(NodeKind::Mul {
                a: floating,
                b: input(2, vec![2, routes, 1], DType::F32),
            });
            node(NodeKind::Cast { a: product, dtype })
        } else {
            input(0, vec![2, routes, width], dtype)
        };
        let scatter = node(NodeKind::ScatterAdd {
            a: zero(vec![2, routes, width]),
            dim: 1,
            indexes,
            src: source,
        });
        let mut output = zero(vec![2, width]);
        for position in 0..routes {
            let route = if reverse {
                routes - position - 1
            } else {
                position
            };
            let sliced = node(NodeKind::Slice {
                a: scatter.clone(),
                ranges: vec![(0, 2, 1), (route, route + 1, 1), (0, width, 1)],
            });
            let selected = node(NodeKind::Reshape {
                a: sliced,
                shape: vec![2, width],
            });
            output = node(NodeKind::Add {
                a: output,
                b: selected,
            });
        }
        (output, scatter)
    }

    #[test]
    fn ordered_scatter_region_checks_dtype_order_shared_outputs_and_opt_in() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for (enabled, dtype, reverse, shared, expected) in [
            (false, DType::BF16, false, false, false),
            (true, DType::BF16, false, false, true),
            (true, DType::F32, false, false, false),
            (true, DType::BF16, true, false, false),
            (true, DType::BF16, false, true, false),
        ] {
            let (root, scatter) = ordered_scatter_chain(8, 7, dtype, DType::U32, reverse);
            let roots = if shared {
                vec![root, scatter]
            } else {
                vec![root]
            };
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut capabilities = CudaCapabilities::new(0, 12, 0);
            capabilities.ordered_scatter_fusion = enabled;
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            assert_eq!(
                driver
                    .optimization()
                    .regions
                    .iter()
                    .any(|r| matches!(r, NativeRegion::OrderedScatterReduce(_))),
                expected
            );
        }
    }

    #[test]
    fn weighted_scatter_selection_preserves_opt_out_and_shared_source() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for (enabled, shared, expected) in [
            (false, false, false),
            (true, false, true),
            (true, true, false),
        ] {
            let (root, scatter) =
                ordered_scatter_chain_with_weighting(8, 2816, DType::BF16, DType::U32, false, true);
            let NodeKind::ScatterAdd { src, .. } = &scatter.kind else {
                unreachable!()
            };
            let roots = if shared {
                vec![root, src.clone()]
            } else {
                vec![root]
            };
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut capabilities = CudaCapabilities::new(0, 12, 0);
            capabilities.ordered_scatter_fusion = true;
            capabilities.ordered_scatter_weighted = enabled;
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            let regions: Vec<_> = driver
                .optimization()
                .regions
                .iter()
                .filter_map(|region| {
                    if let NativeRegion::OrderedScatterReduce(region) = region {
                        Some(region)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(
                regions.len(),
                1,
                "base scatter fusion must survive candidate rejection"
            );
            assert_eq!(regions[0].weighted_source, expected);
            assert_eq!(regions[0].inputs.len(), if expected { 3 } else { 2 });
        }
    }

    #[test]
    #[ignore = "requires CUDA, ORDERED_SCATTER_FUSION=1 and ORDERED_SCATTER_WEIGHTED=1"]
    fn weighted_scatter_matches_materialized_rounds_and_duplicates() {
        use effect_torch_compiler::CompileOptions;
        use effect_torch_runtime::CancellationFlag;
        assert!(CudaCapabilities::new(0, 12, 0).ordered_scatter_weighted);
        let device = crate::CudaDevice::get(0).unwrap();
        for (routes, width) in [(1, 7), (3, 7), (8, 2816), (32, 256)] {
            for index_dtype in [DType::U32, DType::I64] {
                let (root, _) = ordered_scatter_chain_with_weighting(
                    routes,
                    width,
                    DType::BF16,
                    index_dtype,
                    false,
                    true,
                );
                let optimized =
                    crate::compile_with_options(vec![root.clone()], 0, CompileOptions::default())
                        .unwrap();
                let diagnostics = optimized.diagnostics();
                // A checked kernel also owns status preparation/check commands.
                assert!(diagnostics
                    .instructions
                    .iter()
                    .any(|i| i.kind == "et_ordered_scatter_reduce" && i.count == 1));
                assert!(
                    diagnostics.instructions.iter().all(|i| matches!(
                        i.kind.as_str(),
                        "et_ordered_scatter_reduce" | "input" | "prepare" | "status_check"
                    )),
                    "routes={routes} width={width}: {:?}",
                    diagnostics.instructions
                );
                let reference = crate::compile_with_options(
                    vec![root],
                    0,
                    CompileOptions {
                        optimize: false,
                        ..Default::default()
                    },
                )
                .unwrap();
                for witness in 0..5 {
                    let values: Vec<_> = (0..2 * routes * width)
                        .map(|i| match witness {
                            0 => [256.0, 1.0, -256.0, 0.5, -0.5, 0.00390625, -0.00390625]
                                [(i / width + i % width) % 7],
                            1 => {
                                if i % 2 == 0 {
                                    -0.0
                                } else {
                                    0.0
                                }
                            }
                            2 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0][i % 4],
                            3 => [
                                f32::from_bits(0x10000) as f64,
                                -(f32::from_bits(0x10000) as f64),
                                1.0,
                                -1.0,
                            ][i % 4],
                            _ => ((i * 101 % 4093) as f64 - 2046.0) / 128.0,
                        })
                        .collect();
                    let weights: Vec<_> = (0..2 * routes)
                        .map(|i| match witness {
                            1 => {
                                if i % 2 == 0 {
                                    -1.0
                                } else {
                                    1.0
                                }
                            }
                            2 => [0.0, f64::NAN, f64::INFINITY, -1.0][i % 4],
                            _ => [
                                1.00390625,
                                -0.3333333432674408,
                                0.0000001,
                                -0.0,
                                1.00001,
                                -1.00001,
                            ][i % 6],
                        })
                        .collect();
                    let ranks: Vec<_> = (0..2 * routes)
                        .map(|i| {
                            if i < routes {
                                (routes - i - 1) as f64
                            } else {
                                (i % routes.div_ceil(2)) as f64
                            }
                        })
                        .collect();
                    let bindings = [
                        crate::CudaValue::from_host(
                            device.clone(),
                            vec![2 * routes, width],
                            DType::BF16,
                            &values,
                        )
                        .unwrap(),
                        crate::CudaValue::from_host(
                            device.clone(),
                            vec![2, routes],
                            index_dtype,
                            &ranks,
                        )
                        .unwrap(),
                        crate::CudaValue::from_host(
                            device.clone(),
                            vec![2, routes, 1],
                            DType::F32,
                            &weights,
                        )
                        .unwrap(),
                    ];
                    let actual = optimized
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    let expected = reference
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    assert_eq!(
                        actual[0].read_storage_bytes().unwrap(),
                        expected[0].read_storage_bytes().unwrap(),
                        "routes={routes} width={width} witness={witness}"
                    );
                    if witness == 0 {
                        let mut invalid = ranks.clone();
                        invalid[0] = if index_dtype == DType::I64 {
                            -1.0
                        } else {
                            routes as f64
                        };
                        let bad = crate::CudaValue::from_host(
                            device.clone(),
                            vec![2, routes],
                            index_dtype,
                            &invalid,
                        )
                        .unwrap();
                        assert!(optimized
                            .execute(
                                &[bindings[0].clone(), bad, bindings[2].clone()],
                                &[],
                                &CancellationFlag::new()
                            )
                            .is_err());
                        let recovered = optimized
                            .execute(&bindings, &[], &CancellationFlag::new())
                            .unwrap();
                        assert_eq!(
                            recovered[0].read_storage_bytes().unwrap(),
                            expected[0].read_storage_bytes().unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA and EFFECT_TORCH_CUDA_ORDERED_SCATTER_FUSION=1"]
    fn ordered_scatter_fusion_preserves_bf16_rounds_duplicates_and_errors() {
        use effect_torch_compiler::CompileOptions;
        use effect_torch_runtime::CancellationFlag;
        assert!(CudaCapabilities::new(0, 12, 0).ordered_scatter_fusion);
        let device = crate::CudaDevice::get(0).unwrap();
        for (routes, width) in [
            (1, 7),
            (3, 7),
            (8, 2816),
            (32, 33),
            (1, 256),
            (3, 256),
            (32, 256),
        ] {
            for index_dtype in [DType::U32, DType::I64] {
                let (root, _) =
                    ordered_scatter_chain(routes, width, DType::BF16, index_dtype, false);
                let optimized =
                    crate::compile_with_options(vec![root.clone()], 0, CompileOptions::default())
                        .unwrap();
                assert!(optimized
                    .diagnostics()
                    .instructions
                    .iter()
                    .any(|i| i.kind == "et_ordered_scatter_reduce"));
                let reference = crate::compile_with_options(
                    vec![root],
                    0,
                    CompileOptions {
                        optimize: false,
                        ..Default::default()
                    },
                )
                .unwrap();
                for witness in 0..4 {
                    let values: Vec<_> = (0..2 * routes * width)
                        .map(|i| {
                            let route = i / width % routes;
                            match witness {
                                0 => [256.0, 1.0, -256.0, 0.5, -0.5, 0.00390625, -0.00390625, 1.0]
                                    [route % 8],
                                1 => {
                                    if i % 2 == 0 {
                                        -0.0
                                    } else {
                                        0.0
                                    }
                                }
                                2 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0][i % 4],
                                _ => [
                                    f32::from_bits(0x00010000) as f64,
                                    -(f32::from_bits(0x00010000) as f64),
                                    1.0,
                                    -1.0,
                                ][i % 4],
                            }
                        })
                        .collect();
                    // One row is a permutation; the second has duplicates and
                    // missing destinations, requiring zero-add boundaries.
                    let ranks: Vec<_> = (0..2 * routes)
                        .map(|i| {
                            if i < routes {
                                (routes - i - 1) as f64
                            } else {
                                (i % routes.div_ceil(2)) as f64
                            }
                        })
                        .collect();
                    let source = crate::CudaValue::from_host(
                        device.clone(),
                        vec![2, routes, width],
                        DType::BF16,
                        &values,
                    )
                    .unwrap();
                    let indexes = crate::CudaValue::from_host(
                        device.clone(),
                        vec![2, routes],
                        index_dtype,
                        &ranks,
                    )
                    .unwrap();
                    let bindings = [source, indexes];
                    let actual = optimized
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    let expected = reference
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap();
                    assert_eq!(
                        actual[0].read_storage_bytes().unwrap(),
                        expected[0].read_storage_bytes().unwrap(),
                        "routes={routes} width={width} witness={witness}"
                    );
                    if witness == 0 {
                        let mut invalid = ranks.clone();
                        invalid[0] = if index_dtype == DType::I64 {
                            -1.0
                        } else {
                            routes as f64
                        };
                        let bad = crate::CudaValue::from_host(
                            device.clone(),
                            vec![2, routes],
                            index_dtype,
                            &invalid,
                        )
                        .unwrap();
                        for executable in [&optimized, &reference] {
                            assert!(executable
                                .execute(
                                    &[bindings[0].clone(), bad.clone()],
                                    &[],
                                    &CancellationFlag::new()
                                )
                                .is_err());
                        }
                        // A failing invocation cannot poison the next status lease.
                        let recovered = optimized
                            .execute(&bindings, &[], &CancellationFlag::new())
                            .unwrap();
                        assert_eq!(
                            recovered[0].read_storage_bytes().unwrap(),
                            expected[0].read_storage_bytes().unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ordered_scatter_route_map_requires_complete_single_row_tiles() {
        use crate::executable::{ordered_scatter_route_map_eligible, CudaKernelArgs};
        let mut args = CudaKernelArgs {
            elements: 2 * 2816,
            ..Default::default()
        };
        args.integers[0] = 8;
        args.integers[1] = 2816;
        assert!(ordered_scatter_route_map_eligible(
            "et_ordered_scatter_reduce",
            &args
        ));
        assert!(!ordered_scatter_route_map_eligible("et_scatter", &args));
        for routes in [0, 33] {
            args.integers[0] = routes;
            assert!(!ordered_scatter_route_map_eligible(
                "et_ordered_scatter_reduce",
                &args
            ));
        }
        args.integers[0] = 8;
        for width in [0, 7, 255, 257, 2817] {
            args.integers[1] = width;
            assert!(!ordered_scatter_route_map_eligible(
                "et_ordered_scatter_reduce",
                &args
            ));
        }
        args.integers[1] = 2816;
        for elements in [0, 2815, 2817, 2 * 2816 - 1] {
            args.elements = elements;
            assert!(!ordered_scatter_route_map_eligible(
                "et_ordered_scatter_reduce",
                &args
            ));
        }
    }

    #[test]
    #[ignore = "requires CUDA, ORDERED_SCATTER_FUSION=1 and ORDERED_SCATTER_ROUTE_MAP=1"]
    fn ordered_scatter_route_map_preserves_permutations_duplicates_and_errors() {
        assert_eq!(
            std::env::var("EFFECT_TORCH_CUDA_ORDERED_SCATTER_ROUTE_MAP").unwrap(),
            "1"
        );
        ordered_scatter_fusion_preserves_bf16_rounds_duplicates_and_errors();
    }

    fn arg_chain(width: usize, maximum: bool) -> (Arc<Node>, Arc<Node>) {
        let input = |slot| {
            Node::new(NodeKind::Input {
                slot,
                shape: vec![3, width],
                dtype: DType::F32,
                device: Device::Cuda(0),
                storage: effect_torch_runtime::StorageMetadata::dense(),
            })
            .unwrap()
        };
        let log = Node::new(NodeKind::Log { a: input(1) }).unwrap();
        let neg = Node::new(NodeKind::Neg { a: log }).unwrap();
        let log = Node::new(NodeKind::Log { a: neg }).unwrap();
        let noise = Node::new(NodeKind::Neg { a: log }).unwrap();
        let noisy = Node::new(NodeKind::Add {
            a: input(0),
            b: noise,
        })
        .unwrap();
        let output = Node::new(if maximum {
            NodeKind::Argmax {
                a: noisy.clone(),
                dim: 1,
            }
        } else {
            NodeKind::Argmin {
                a: noisy.clone(),
                dim: 1,
            }
        })
        .unwrap();
        (output, noisy)
    }

    #[test]
    fn wide_arg_fusion_selection_preserves_shared_roots() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for (enabled, shared, width, expected) in [
            (false, false, 4096, false),
            (true, false, 4095, false),
            (true, false, 4096, true),
            (true, true, 4096, false),
        ] {
            for maximum in [false, true] {
                let (root, noisy) = arg_chain(width, maximum);
                let roots = if shared {
                    vec![root, noisy]
                } else {
                    vec![root]
                };
                let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                    .prepare()
                    .unwrap();
                let mut capabilities = CudaCapabilities::new(0, 12, 0);
                capabilities.wide_arg_fusion = enabled;
                let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
                if !expected {
                    assert!(
                        driver
                            .optimization()
                            .regions
                            .iter()
                            .any(|r| matches!(r, NativeRegion::Elementwise(_))),
                        "rejected index reduction must preserve elementwise fusion"
                    );
                }
                assert_eq!(
                    driver
                        .optimization()
                        .regions
                        .iter()
                        .any(|r| matches!(r, NativeRegion::ElementwiseArgReduce(_))),
                    expected
                );
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA and EFFECT_TORCH_CUDA_WIDE_ARG_FUSION=1"]
    fn wide_arg_fusion_matches_materialized_gumbel_special_values() {
        use effect_torch_compiler::CompileOptions;
        use effect_torch_runtime::CancellationFlag;
        assert!(CudaCapabilities::new(0, 12, 0).wide_arg_fusion);
        let device = crate::CudaDevice::get(0).unwrap();
        for width in [4096, 4101, 262144] {
            for maximum in [false, true] {
                for witness in 0..5 {
                    let values: Vec<_> = (0..3 * width)
                        .map(|i| match witness {
                            0 => ((i * 101 % 4093) as f64 - 2046.0) / 128.0,
                            1 => {
                                if i % width == 0 {
                                    f64::NAN
                                } else {
                                    1.0
                                }
                            }
                            2 => {
                                if i % width == 1025 {
                                    f64::NAN
                                } else {
                                    (i % 4) as f64
                                }
                            }
                            3 => [f64::INFINITY, f64::NEG_INFINITY, 0.0][i / width],
                            _ => {
                                if i % 2 == 0 {
                                    0.0
                                } else {
                                    -0.0
                                }
                            }
                        })
                        .collect();
                    let uniforms: Vec<_> = (0..3 * width)
                        .map(|i| {
                            if witness == 0 {
                                ((i * 73 % 8191) + 1) as f64 / 8192.0
                            } else {
                                0.5
                            }
                        })
                        .collect();
                    let bindings = [
                        crate::CudaValue::from_host(
                            device.clone(),
                            vec![3, width],
                            DType::F32,
                            &values,
                        )
                        .unwrap(),
                        crate::CudaValue::from_host(
                            device.clone(),
                            vec![3, width],
                            DType::F32,
                            &uniforms,
                        )
                        .unwrap(),
                    ];
                    let (root, _) = arg_chain(width, maximum);
                    let mut observed = Vec::new();
                    for optimize in [false, true] {
                        let executable = crate::compile_with_options(
                            vec![root.clone()],
                            0,
                            CompileOptions {
                                optimize,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        if optimize {
                            assert_eq!(executable.diagnostics().command_count, 1);
                        }
                        let outputs = executable
                            .execute(&bindings, &[], &CancellationFlag::new())
                            .unwrap();
                        observed.push(outputs[0].read_storage_bytes().unwrap());
                    }
                    assert_eq!(
                        observed[0], observed[1],
                        "width={width} maximum={maximum} witness={witness}"
                    );
                }
            }
        }
    }

    fn wide_chain(width: usize, dtype: DType) -> Arc<Node> {
        let input = |slot, shape| {
            Node::new(NodeKind::Input {
                slot,
                shape,
                dtype,
                device: Device::Cuda(0),
                storage: effect_torch_runtime::StorageMetadata::dense(),
            })
            .unwrap()
        };
        let shifted = Node::new(NodeKind::Sub {
            a: input(0, vec![3, width]),
            b: input(1, vec![3, 1]),
        })
        .unwrap();
        let exp = Node::new(NodeKind::Exp { a: shifted }).unwrap();
        Node::new(NodeKind::Sum {
            a: exp,
            dims: vec![1],
            keepdims: true,
        })
        .unwrap()
    }

    #[test]
    fn wide_sum_fusion_requires_opt_in_f32_and_wide_axis() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for (enabled, width, dtype, expected) in [
            (false, 4096, DType::F32, false),
            (true, 4095, DType::F32, false),
            (true, 4096, DType::BF16, false),
            (true, 4096, DType::F32, true),
            (true, 4101, DType::F32, true),
        ] {
            let prepared = ProgramRequest::from_roots(
                vec![wide_chain(width, dtype)],
                CompileOptions::default(),
            )
            .prepare()
            .unwrap();
            let mut capabilities = CudaCapabilities::new(0, 12, 0);
            capabilities.wide_sum_fusion = enabled;
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            assert_eq!(
                driver
                    .optimization()
                    .regions
                    .iter()
                    .any(|r| matches!(r, NativeRegion::ElementwiseReduce(_))),
                expected,
                "{enabled} {width} {dtype:?}"
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA and EFFECT_TORCH_CUDA_WIDE_SUM_FUSION=1"]
    fn wide_sum_fusion_matches_materialized_f32_bits() {
        use effect_torch_compiler::CompileOptions;
        use effect_torch_runtime::CancellationFlag;
        assert!(CudaCapabilities::new(0, 12, 0).wide_sum_fusion);
        let device = crate::CudaDevice::get(0).unwrap();
        for (width, exponential, witness) in [4096, 4101, 8193, 262144]
            .into_iter()
            .flat_map(|w| [(w, true), (w, false)])
            .flat_map(|(w, e)| (0..5).map(move |witness| (w, e, witness)))
        {
            // Each exceptional category gets a separate invocation: NaNs must
            // not mask finite underflow or cancellation mismatches.
            let values: Vec<_> = (0..3 * width)
                .map(|i| match witness {
                    0 => ((i * 101 % 4093) as f64 - 2046.0) / 128.0,
                    1 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][i / width],
                    2 => {
                        if i % 2 == 0 {
                            0.0
                        } else {
                            -0.0
                        }
                    }
                    3 => {
                        if exponential {
                            [-90.0, -103.0, -104.0, -87.25][i % 4]
                        } else {
                            [
                                f32::from_bits(1) as f64,
                                -(f32::from_bits(7) as f64),
                                f32::MIN_POSITIVE as f64,
                                -(f32::MIN_POSITIVE as f64),
                            ][i % 4]
                        }
                    }
                    _ => [16.0, -16.0, 0.00000095367431640625, -0.00000095367431640625][i % 4],
                })
                .collect();
            let bindings = [
                crate::CudaValue::from_host(device.clone(), vec![3, width], DType::F32, &values)
                    .unwrap(),
                crate::CudaValue::from_host(
                    device.clone(),
                    vec![3, 1],
                    DType::F32,
                    &[0.03125, -0.125, 1.0625],
                )
                .unwrap(),
            ];
            let root = if exponential {
                wide_chain(width, DType::F32)
            } else {
                let input = |slot, shape| {
                    Node::new(NodeKind::Input {
                        slot,
                        shape,
                        dtype: DType::F32,
                        device: Device::Cuda(0),
                        storage: effect_torch_runtime::StorageMetadata::dense(),
                    })
                    .unwrap()
                };
                let a = input(0, vec![3, width]);
                let b = input(1, vec![3, 1]);
                let quotient = Node::new(NodeKind::Div {
                    a: a.clone(),
                    b: b.clone(),
                })
                .unwrap();
                let clamped = Node::new(NodeKind::Maximum { a, b }).unwrap();
                let product = Node::new(NodeKind::Mul {
                    a: clamped,
                    b: quotient,
                })
                .unwrap();
                Node::new(NodeKind::Sum {
                    a: product,
                    dims: vec![1],
                    keepdims: true,
                })
                .unwrap()
            };
            let mut observed = Vec::new();
            for optimize in [false, true] {
                let executable = crate::compile_with_options(
                    vec![root.clone()],
                    0,
                    CompileOptions {
                        optimize,
                        ..Default::default()
                    },
                )
                .unwrap();
                if optimize {
                    // One launch proves that the reduction was absorbed;
                    // elementwise-only fusion plus a materialized Sum needs two.
                    assert_eq!(executable.diagnostics().command_count, 1);
                }
                let outputs = executable
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                observed.push(outputs[0].read_storage_bytes().unwrap());
            }
            assert_eq!(
                observed[0], observed[1],
                "width={width}, exponential={exponential}, witness={witness}"
            );
        }
    }

    fn scalar_chain(dtype: DType) -> Arc<Node> {
        let tensor = |slot, shape, dtype| {
            Node::new(NodeKind::Input {
                slot,
                shape,
                dtype,
                device: Device::Cuda(0),
                storage: effect_torch_runtime::StorageMetadata::dense(),
            })
            .unwrap()
        };
        let quotient = Node::new(NodeKind::Div {
            a: tensor(0, vec![2, 17], dtype),
            b: tensor(1, vec![], DType::F32),
        })
        .unwrap();
        let shifted = Node::new(NodeKind::Add {
            a: quotient,
            b: tensor(2, vec![], DType::F32),
        })
        .unwrap();
        Node::new(NodeKind::Mul {
            a: shifted,
            b: tensor(3, vec![], DType::F32),
        })
        .unwrap()
    }

    #[test]
    fn floating_scalar_input_lanes_are_eligible_for_elementwise_fusion() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for dtype in [DType::F32, DType::BF16] {
            let prepared =
                ProgramRequest::from_roots(vec![scalar_chain(dtype)], CompileOptions::default())
                    .prepare()
                    .unwrap();
            let capabilities = CudaCapabilities::new(0, 12, 0);
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            let regions: Vec<_> = driver
                .optimization()
                .regions
                .iter()
                .filter_map(|region| match region {
                    NativeRegion::Elementwise(region) => Some(region),
                    _ => None,
                })
                .collect();
            assert_eq!(regions.len(), 1);
            let region = regions[0];
            assert_eq!(region.dtype, dtype);
            assert_eq!(region.inputs.len(), 4);
            assert_eq!(region.lane_strides[0].as_ref(), [17, 1]);
            for lane in &region.lane_strides[1..] {
                assert_eq!(lane.as_ref(), [0, 0]);
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn fused_scalar_lanes_preserve_half_coercion_and_intermediate_rounding() {
        use effect_torch_compiler::CompileOptions;
        use effect_torch_runtime::CancellationFlag;
        for dtype in [DType::F32, DType::BF16] {
            let device = crate::CudaDevice::get(0).unwrap();
            // Scalar values straddle BF16 ties. Multiple divisions/additions
            // distinguish per-operation BF16 rounding from one final cast.
            let values: Vec<_> = (0..34)
                .map(|i| (i as f64 - 17.0) * 0.0311279296875)
                .collect();
            let bindings = [
                crate::CudaValue::from_host(device.clone(), vec![2, 17], dtype, &values).unwrap(),
                crate::CudaValue::from_host(device.clone(), vec![], DType::F32, &[0.80078125])
                    .unwrap(),
                crate::CudaValue::from_host(device.clone(), vec![], DType::F32, &[0.001953125])
                    .unwrap(),
                crate::CudaValue::from_host(device, vec![], DType::F32, &[1.00390625]).unwrap(),
            ];
            let root = scalar_chain(dtype);
            let mut observed = Vec::new();
            for optimize in [false, true] {
                let executable = crate::compile_with_options(
                    vec![root.clone()],
                    0,
                    CompileOptions {
                        optimize,
                        ..Default::default()
                    },
                )
                .unwrap();
                let outputs = executable
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                observed.push(outputs[0].read_storage_bytes().unwrap());
            }
            assert_eq!(observed[0], observed[1], "{dtype:?}");
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn canvas_full_sort_preserves_stable_float_order_and_nan_errors() {
        use effect_torch_runtime::CancellationFlag;
        let device = crate::CudaDevice::get(0).unwrap();
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![4, 256],
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: effect_torch_runtime::StorageMetadata::dense(),
        })
        .unwrap();
        let root = Node::new(NodeKind::TopKIndices { a: input, k: 256 }).unwrap();
        let executable = crate::compile(vec![root], 0).unwrap();
        let mut values = vec![0.0; 4 * 256];
        for index in 0..256 {
            values[index] = index as f64;
            values[256 + index] = (255 - index) as f64;
            values[512 + index] = (index % 7) as f64 - 3.0;
            values[768 + index] = if index % 2 == 0 { -0.0 } else { 0.0 };
        }
        values[512] = f64::NEG_INFINITY;
        values[513] = f64::INFINITY;
        values[514] = f32::from_bits(1) as f64;
        let binding =
            crate::CudaValue::from_host(device.clone(), vec![4, 256], DType::F32, &values).unwrap();
        let outputs = executable
            .execute(&[binding], &[], &CancellationFlag::new())
            .unwrap();
        let observed = outputs[0].readback().unwrap();
        // Width 257 keeps the existing serial insertion branch. A final -Inf
        // sentinel loses ties to every original index and is outside top 256.
        let padded: Vec<_> = values
            .chunks(256)
            .flat_map(|row| {
                row.iter()
                    .copied()
                    .chain(std::iter::once(f64::NEG_INFINITY))
            })
            .collect();
        let serial_input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![4, 257],
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: effect_torch_runtime::StorageMetadata::dense(),
        })
        .unwrap();
        let serial = crate::compile(
            vec![Node::new(NodeKind::TopKIndices {
                a: serial_input,
                k: 256,
            })
            .unwrap()],
            0,
        )
        .unwrap();
        let serial_binding =
            crate::CudaValue::from_host(device.clone(), vec![4, 257], DType::F32, &padded).unwrap();
        let serial_outputs = serial
            .execute(&[serial_binding], &[], &CancellationFlag::new())
            .unwrap();
        assert_eq!(observed, serial_outputs[0].readback().unwrap());
        for row in 0..4 {
            let mut expected: Vec<usize> = (0..256).collect();
            expected.sort_by(|left, right| {
                values[row * 256 + *right]
                    .partial_cmp(&values[row * 256 + *left])
                    .unwrap()
            });
            assert_eq!(
                observed[row * 256..(row + 1) * 256],
                expected
                    .iter()
                    .map(|index| *index as f64)
                    .collect::<Vec<_>>(),
                "row {row}"
            );
        }
        values[100] = f64::NAN;
        let binding =
            crate::CudaValue::from_host(device, vec![4, 256], DType::F32, &values).unwrap();
        let error = match executable.execute(&[binding], &[], &CancellationFlag::new()) {
            Ok(_) => panic!("NaN sort input must fail"),
            Err(error) => error,
        };
        assert!(error.contains("NaN input"), "{error}");
    }

    #[test]
    fn empty_arg_reductions_fail_before_lowering() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for shape in [vec![0], vec![0, 2]] {
            for minimum in [false, true] {
                let a = Node::new(NodeKind::Zeros {
                    shape: shape.clone(),
                    dtype: DType::F32,
                    device: Device::Cuda(0),
                })
                .unwrap();
                let root = Node::new(if minimum {
                    NodeKind::Argmin { a, dim: 0 }
                } else {
                    NodeKind::Argmax { a, dim: 0 }
                })
                .unwrap();
                let prepared = ProgramRequest::from_roots(vec![root], CompileOptions::default())
                    .prepare()
                    .unwrap();
                let capabilities = CudaCapabilities::new(0, 12, 0);
                let error = match CompilerDriver::new(&prepared, &capabilities) {
                    Ok(_) => panic!("empty arg reduction must fail before lowering"),
                    Err(error) => error,
                };
                assert!(error.contains("nonempty reduction axis"), "{error}");
            }
        }
    }

    fn bf16_linear(dtype: DType, transposed: bool) -> Arc<Node> {
        let x = Node::new(NodeKind::Zeros {
            shape: vec![2, 3],
            dtype,
            device: Device::Cuda(0),
        })
        .unwrap();
        let weight = Node::new(NodeKind::Zeros {
            shape: if transposed { vec![2, 3] } else { vec![3, 2] },
            dtype,
            device: Device::Cuda(0),
        })
        .unwrap();
        let weight = if transposed {
            Node::new(NodeKind::Permute {
                a: weight,
                dims: vec![1, 0],
            })
            .unwrap()
        } else {
            weight
        };
        let bias = Node::new(NodeKind::Zeros {
            shape: vec![2],
            dtype,
            device: Device::Cuda(0),
        })
        .unwrap();
        Node::new(NodeKind::Linear { x, weight, bias }).unwrap()
    }

    fn classify(node: &Arc<Node>, major: i32) -> DTypeDisposition {
        let index = GraphIndex::new(std::slice::from_ref(node)).unwrap();
        let spec = OperationDTypeSpec::new(&index, index.roots[0]).unwrap();
        CudaCapabilities::new(0, major, 0).classify_node(&spec)
    }

    #[test]
    fn expert_rows_are_native_f32_dots_with_one_boundary_rounding() {
        for dtype in [DType::F32, DType::BF16] {
            let tensor = |slot, shape: Vec<usize>, dtype| {
                Node::new(NodeKind::Input {
                    slot,
                    shape,
                    dtype,
                    device: Device::Cuda(0),
                    storage: effect_torch_runtime::StorageMetadata::dense(),
                })
                .unwrap()
            };
            let node = Node::new(NodeKind::ExpertLinearRows {
                x: tensor(0, vec![16, 13], dtype),
                weight: tensor(1, vec![3, 64, 13], dtype),
                indexes: tensor(2, vec![16], DType::U32),
            })
            .unwrap();
            for major in [7, 8, 12] {
                let DTypeDisposition::Native(execution) = classify(&node, major) else {
                    panic!("expert rows must read native storage on all CUDA targets");
                };
                assert_eq!(execution.realization, ExecutionRealization::DirectKernel);
                let op = &execution.operations[0];
                assert_eq!(op.compute_dtype, Some(DType::F32));
                assert_eq!(op.accumulation.unwrap().dtype, DType::F32);
                assert_eq!(op.rounding_boundaries.len(), 1);
                assert_eq!(op.rounding_boundaries[0].dtype, dtype);
            }
        }
    }

    #[test]
    fn semantic_preparation_legalizes_half_attention_and_explicit_rotary_on_cuda() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        use effect_torch_graph::{AttentionRounding, AttentionWindow, RotaryLayout};
        for dtype in [DType::F16, DType::BF16] {
            let input = |slot, shape: Vec<usize>, dtype| {
                Node::new(NodeKind::Input {
                    slot,
                    shape,
                    dtype,
                    device: Device::Cuda(0),
                    storage: effect_torch_runtime::StorageMetadata::dense(),
                })
                .unwrap()
            };
            let q = input(0, vec![1, 4, 2, 8], dtype);
            let attention = Node::new(NodeKind::SdpaConfigured {
                q: q.clone(),
                k: input(1, vec![1, 2, 4, 8], dtype),
                v: input(2, vec![1, 2, 4, 8], dtype),
                scale: 0.3,
                causal: true,
                window: AttentionWindow::Local(2),
                rounding: AttentionRounding::Stepwise,
                layer_id: Some(7),
                retention: AttentionWindow::Local(0),
            })
            .unwrap();
            let rotary = Node::new(NodeKind::RotaryEmbeddingExplicit {
                x: q,
                positions: input(3, vec![1, 2], DType::U32),
                inverse_frequencies: input(4, vec![4], DType::F32),
                layout: RotaryLayout::InterleavedPairs,
            })
            .unwrap();
            for optimize in [false, true] {
                let prepared = ProgramRequest::from_roots(
                    vec![attention.clone(), rotary.clone()],
                    CompileOptions {
                        optimize,
                        ..Default::default()
                    },
                )
                .prepare()
                .unwrap();
                for major in [7, 8] {
                    let target = CudaCapabilities::new(0, major, 0);
                    CompilerDriver::new(&prepared, &target).unwrap();
                }
                assert!(prepared
                    .roots
                    .iter()
                    .all(|root| root.dtype == dtype && root.device == Device::Cuda(0)));
                assert!(prepared.index.order.iter().any(|node| matches!(
                    node.kind,
                    NodeKind::Cast {
                        dtype: DType::F32,
                        ..
                    }
                )));
            }
        }
    }

    #[test]
    fn bf16_linear_is_native_on_ampere_and_newer() {
        for transposed in [false, true] {
            let node = bf16_linear(DType::BF16, transposed);
            match classify(&node, 12) {
                DTypeDisposition::Native(execution) => {
                    assert_eq!(execution.operations[0].compute_dtype, Some(DType::BF16));
                    assert_eq!(
                        execution.operations[0].accumulation.unwrap().dtype,
                        DType::F32
                    );
                    assert_eq!(execution.realization, ExecutionRealization::DirectKernel);
                }
                other => panic!("expected native BF16 linear, got {other:?}"),
            }
            assert!(matches!(classify(&node, 7), DTypeDisposition::Legalize(_)));
        }
    }

    #[test]
    fn half_scatter_preserves_typed_update_rounding_without_materialized_widening() {
        for dtype in [DType::F16, DType::BF16] {
            let tensor = |shape: Vec<usize>, dtype| {
                Node::new(NodeKind::Zeros {
                    shape,
                    dtype,
                    device: Device::Cuda(0),
                })
                .unwrap()
            };
            let node = Node::new(NodeKind::ScatterAdd {
                a: tensor(vec![3], dtype),
                indexes: tensor(vec![4], DType::U32),
                src: tensor(vec![4], dtype),
                dim: 0,
            })
            .unwrap();
            let DTypeDisposition::Native(execution) = classify(&node, 12) else {
                panic!("half scatter must use the directly rounded native kernel");
            };
            assert_eq!(execution.realization, ExecutionRealization::DirectKernel);
            assert_eq!(execution.operations[0].compute_dtype, Some(dtype));
        }
    }

    #[test]
    fn f16_linear_keeps_materialized_f32_legalization() {
        let node = bf16_linear(DType::F16, false);
        assert!(matches!(classify(&node, 12), DTypeDisposition::Legalize(_)));
    }

    #[test]
    fn bf16_matmul_partial_broadcast_is_not_native() {
        let a = Node::new(NodeKind::Zeros {
            shape: vec![2, 1, 3, 4],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        })
        .unwrap();
        let b = Node::new(NodeKind::Zeros {
            shape: vec![1, 5, 4, 6],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        })
        .unwrap();
        let node = Node::new(NodeKind::Matmul { a, b }).unwrap();
        assert!(matches!(classify(&node, 12), DTypeDisposition::Legalize(_)));
    }

    #[test]
    fn target_fingerprint_records_native_bf16_realization() {
        let capabilities = CudaCapabilities::new(0, 12, 0);
        assert_eq!(capabilities.policy_revision(), 60);
        assert!(capabilities
            .fingerprint()
            .features
            .iter()
            .any(|feature| feature == "sum-f32-warp-or-block1024-vector4-v2"));
        assert!(capabilities
            .fingerprint()
            .features
            .iter()
            .any(|feature| feature == "cublas-bf16-row-major-f32-accum-v3"));
        assert!(capabilities
            .fingerprint()
            .features
            .iter()
            .any(|feature| feature == "cublas-expert-streams-32"));
        assert_eq!(capabilities.fingerprint().architecture, "sm_120");
        let off = CudaCapabilities::new(0, 7, 0);
        assert!(capabilities
            .fingerprint()
            .features
            .iter()
            .any(|feature| feature == "stepwise-bf16-kv-f32-gemm-active-rows-v1"));
        assert!(!off
            .fingerprint()
            .features
            .iter()
            .any(|feature| feature == "stepwise-bf16-kv-f32-gemm-active-rows-v1"));
        for target in [&capabilities, &off] {
            assert!(target
                .fingerprint()
                .features
                .iter()
                .any(|feature| feature == "rms-f32-warp-four-partials-mean-factor-rsqrt-v1"));
        }
        assert!(!off
            .fingerprint()
            .features
            .iter()
            .any(|feature| feature == "cublas-bf16-row-major-f32-accum-v3"));
    }

    #[test]
    fn half_arithmetic_records_materialized_boundaries() {
        let x = Node::new(NodeKind::Zeros {
            shape: vec![3],
            dtype: DType::BF16,
            device: Device::Cuda(0),
        })
        .unwrap();
        let root = Node::new(NodeKind::Add { a: x.clone(), b: x }).unwrap();
        let index = GraphIndex::new(&[root]).unwrap();
        let spec = OperationDTypeSpec::new(&index, index.roots[0]).unwrap();
        let DTypeDisposition::Legalize(execution) =
            CudaCapabilities::new(0, 8, 0).classify_node(&spec)
        else {
            panic!("half add must legalize")
        };
        assert_eq!(
            execution.realization,
            ExecutionRealization::MaterializedTransforms
        );
        assert_eq!(execution.operations[0].compute_dtype, Some(DType::F32));
        assert!(matches!(
            execution.operations[0].results[0].completion,
            effect_torch_compiler::ResultCompletion::ConvertToBoundary(_)
        ));
    }
    #[test]
    fn integer_arithmetic_does_not_convert_to_float() {
        let x = Node::new(NodeKind::Zeros {
            shape: vec![3],
            dtype: DType::I64,
            device: Device::Cuda(0),
        })
        .unwrap();
        let root = Node::new(NodeKind::Mul { a: x.clone(), b: x }).unwrap();
        let index = GraphIndex::new(&[root]).unwrap();
        let spec = OperationDTypeSpec::new(&index, index.roots[0]).unwrap();
        let DTypeDisposition::Native(execution) =
            CudaCapabilities::new(0, 8, 0).classify_node(&spec)
        else {
            panic!("integer multiply must remain typed")
        };
        assert_eq!(execution.operations[0].compute_dtype, Some(DType::I64));
    }
    fn norm_rope_chain(
        heads: usize,
        sequence: usize,
        width: usize,
        dtype: DType,
        strided: bool,
        repeated_tables: bool,
    ) -> Vec<Arc<Node>> {
        let make = |kind| Node::new(kind).unwrap();
        let input = |slot, shape| {
            make(NodeKind::Input {
                slot,
                shape,
                dtype,
                device: Device::Cuda(0),
                storage: effect_torch_runtime::StorageMetadata::dense(),
            })
        };
        let raw = if strided {
            make(NodeKind::Permute {
                a: input(0, vec![1, sequence, heads, width]),
                dims: vec![0, 2, 1, 3],
            })
        } else {
            input(0, vec![1, heads, sequence, width])
        };
        let norm = make(NodeKind::RmsNorm {
            x: raw,
            weight: Some(input(1, vec![width])),
            eps: 1e-6,
        });
        let table = |slot| {
            if repeated_tables {
                let half = input(slot, vec![1, 1, sequence, width / 2]);
                make(NodeKind::Concat {
                    a: half.clone(),
                    b: half,
                    dim: 3,
                })
            } else {
                input(slot, vec![1, 1, sequence, width])
            }
        };
        let cosine = table(2);
        let sine = table(3);
        let first = make(NodeKind::Slice {
            a: norm.clone(),
            ranges: vec![
                (0, 1, 1),
                (0, heads, 1),
                (0, sequence, 1),
                (0, width / 2, 1),
            ],
        });
        let second = make(NodeKind::Slice {
            a: norm.clone(),
            ranges: vec![
                (0, 1, 1),
                (0, heads, 1),
                (0, sequence, 1),
                (width / 2, width, 1),
            ],
        });
        let neg = make(NodeKind::Neg { a: second.clone() });
        let rotated = make(NodeKind::Concat {
            a: neg.clone(),
            b: first.clone(),
            dim: 3,
        });
        let direct = make(NodeKind::Mul {
            a: norm.clone(),
            b: cosine.clone(),
        });
        let cross = make(NodeKind::Mul {
            a: rotated.clone(),
            b: sine.clone(),
        });
        let result = make(NodeKind::Add {
            a: direct.clone(),
            b: cross.clone(),
        });
        vec![
            result, norm, first, second, neg, rotated, direct, cross, cosine, sine,
        ]
    }

    #[test]
    fn norm_rope_selection_preserves_private_intermediates_and_measured_shapes() {
        use effect_torch_compiler::{CompileOptions, CompilerDriver, ProgramRequest};
        for (enabled, heads, sequence, width, dtype, escaped, expected) in [
            (true, 8, 256, 256, DType::BF16, 0, true),
            (true, 16, 256, 512, DType::BF16, 0, true),
            (false, 8, 256, 256, DType::BF16, 0, false),
            (true, 2, 256, 512, DType::BF16, 0, false),
            (true, 8, 64, 256, DType::BF16, 0, false),
            (true, 8, 256, 128, DType::BF16, 0, false),
            (true, 8, 256, 256, DType::F32, 0, false),
            (true, 8, 256, 256, DType::BF16, 1, false),
            (true, 8, 256, 256, DType::BF16, 2, false),
            (true, 8, 256, 256, DType::BF16, 3, false),
            (true, 8, 256, 256, DType::BF16, 4, false),
            (true, 8, 256, 256, DType::BF16, 5, false),
            (true, 8, 256, 256, DType::BF16, 6, false),
            (true, 8, 256, 256, DType::BF16, 7, false),
        ] {
            let nodes = norm_rope_chain(heads, sequence, width, dtype, true, true);
            let mut roots = vec![nodes[0].clone()];
            if escaped != 0 {
                roots.push(nodes[escaped].clone());
            }
            let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
                .prepare()
                .unwrap();
            let mut capabilities = CudaCapabilities::new(0, 12, 0);
            capabilities.norm_rope = enabled;
            let driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
            let actual = driver
                .optimization()
                .regions
                .iter()
                .filter(|r| matches!(r, NativeRegion::NormRope(_)))
                .count();
            assert_eq!(
                actual,
                usize::from(expected),
                "enabled={enabled} H={heads} S={sequence} D={width} dtype={dtype:?} escape={escaped}"
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA and EFFECT_TORCH_CUDA_NORM_ROPE=1"]
    fn norm_rope_matches_generated_elementwise_bytes_views_specials_and_retained_outputs() {
        use crate::{CudaDevice, CudaValue};
        use effect_torch_runtime::CancellationFlag;
        assert_eq!(std::env::var("EFFECT_TORCH_CUDA_NORM_ROPE").unwrap(), "1");
        let device = CudaDevice::get(0).unwrap();
        for width in [256, 512] {
            for heads in [8, 16] {
                for strided in [false, true] {
                    for repeated in [false, true] {
                        let nodes =
                            norm_rope_chain(heads, 256, width, DType::BF16, strided, repeated);
                        let fused = crate::compile(vec![nodes[0].clone()], 0).unwrap();
                        // Escaping normalization retains the generated elementwise RoPE.
                        // Generic elementwise lowering cannot broadcast a repeated-table
                        // physical view, so materialize only the baseline tables as roots.
                        // The candidate must still consume the repeated physical views.
                        let mut baseline_roots = vec![nodes[0].clone(), nodes[1].clone()];
                        if repeated {
                            baseline_roots.extend([nodes[8].clone(), nodes[9].clone()]);
                        }
                        let baseline = crate::compile(baseline_roots, 0).unwrap();
                        assert!(fused
                            .diagnostics()
                            .instructions
                            .iter()
                            .any(|i| i.kind == "et_norm_rope_bf16"));
                        assert!(!baseline
                            .diagnostics()
                            .instructions
                            .iter()
                            .any(|i| i.kind == "et_norm_rope_bf16"));
                        if repeated {
                            assert!(
                                fused
                                    .diagnostics()
                                    .instructions
                                    .iter()
                                    .all(|i| !i.kind.contains("concat")),
                                "repeated cosine/sine tables must retain their physical views"
                            );
                        }
                        let shape = if strided {
                            vec![1, 256, heads, width]
                        } else {
                            vec![1, heads, 256, width]
                        };
                        let count = 256 * heads * width;
                        let table_width = if repeated { width / 2 } else { width };
                        let mut retained = Vec::new();
                        for pattern in 0..3 {
                            let values: Vec<f64> = (0..count)
                                .map(|i| {
                                    let bits = match pattern {
                                        1 => [0u16, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080]
                                            [(i / width) % 8],
                                        2 => [
                                            0x7f80, 0xff80, 0x7fc1, 0xffc1, 0x7f81, 0xff81, 0x7f7f,
                                            0xff7f,
                                        ][(i / width) % 8],
                                        _ => ((i as u16).wrapping_mul(31) & 0x807f) | (123 << 7),
                                    };
                                    half::bf16::from_bits(bits).to_f64()
                                })
                                .collect();
                            let weight: Vec<f64> =
                                (0..width).map(|i| (i % 13 + 1) as f64 / 8.).collect();
                            let cosine: Vec<f64> = (0..256 * table_width)
                                .map(|i| ((i % 17) as f64 - 8.) / 8.)
                                .collect();
                            let sine: Vec<f64> = (0..256 * table_width)
                                .map(|i| ((i % 19) as f64 - 9.) / 8.)
                                .collect();
                            let bindings = vec![
                                CudaValue::from_host(
                                    device.clone(),
                                    shape.clone(),
                                    DType::BF16,
                                    &values,
                                )
                                .unwrap(),
                                CudaValue::from_host(
                                    device.clone(),
                                    vec![width],
                                    DType::BF16,
                                    &weight,
                                )
                                .unwrap(),
                                CudaValue::from_host(
                                    device.clone(),
                                    vec![1, 1, 256, table_width],
                                    DType::BF16,
                                    &cosine,
                                )
                                .unwrap(),
                                CudaValue::from_host(
                                    device.clone(),
                                    vec![1, 1, 256, table_width],
                                    DType::BF16,
                                    &sine,
                                )
                                .unwrap(),
                            ];
                            let expected = baseline
                                .execute(&bindings, &[], &CancellationFlag::new())
                                .unwrap()[0]
                                .read_storage_bytes()
                                .unwrap();
                            let actual = fused
                                .execute(&bindings, &[], &CancellationFlag::new())
                                .unwrap()
                                .remove(0);
                            assert_eq!(
                                actual.read_storage_bytes().unwrap(),
                                expected,
                                "D={width} H={heads} strided={strided} repeated={repeated} pattern={pattern}"
                            );
                            retained.push((actual, expected));
                            let cancelled = CancellationFlag::new();
                            cancelled.cancel();
                            assert!(fused.execute(&bindings, &[], &cancelled).is_err());
                        }
                        for (value, bytes) in retained {
                            assert_eq!(value.read_storage_bytes().unwrap(), bytes);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "entropy_tests.rs"]
mod entropy_tests;

#[cfg(test)]
#[path = "small_softmax_tests.rs"]
mod small_softmax_tests;

#[cfg(test)]
#[path = "grouped_retained_tests.rs"]
mod grouped_retained_tests;

#[cfg(test)]
#[path = "ffn_tail_tests.rs"]
mod ffn_tail_tests;

#[cfg(test)]
#[path = "expert_route_rank_tests.rs"]
mod expert_route_rank_tests;

#[cfg(test)]
#[path = "expert_finalize_tests.rs"]
mod expert_finalize_tests;

#[cfg(test)]
#[path = "rms_residual_tests.rs"]
mod rms_residual_tests;

#[cfg(test)]
#[path = "router_tail_tests.rs"]
mod router_tail_tests;

#[cfg(test)]
#[path = "dual_argmax_tests.rs"]
mod dual_argmax_tests;

#[cfg(test)]
#[path = "attention_ffn_entrance_tests.rs"]
mod attention_ffn_entrance_tests;

#[cfg(test)]
#[path = "ffn_next_norm63_tests.rs"]
mod ffn_next_norm63_tests;

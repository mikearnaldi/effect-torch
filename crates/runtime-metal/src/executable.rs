use crate::value::Value;
use crate::{
    device, flash, fusion, kda, layer_norm, linear, loss, ops as metal_ops, quantized, rotary,
    shortconv,
};
#[cfg(test)]
use effect_torch_compiler::ProgramRequest;
use effect_torch_compiler::{
    build_executable_diagnostics, CompileOptions, CompilerDriver, CompilerWorkReport, DenseNodeId,
    DiagnosticsInput, EnvironmentOptions, Expr, GeneratedBinding, GraphIndex, InstructionEffects,
    LoweredInstruction, LoweredProgram, LoweredValue, LoweringUnit, MemoryPlannerConfig,
    NativeRegion, OptimizationPlan, OutputDecl, PreparedProgram, ProgramSlot, RegionId,
    StateCursorSlot, ValueDecl, ValueStorage, ValueUse, ARTIFACT_ASSEMBLY_PHASE,
    COMPILE_SUBMISSION_PHASE, PHYSICAL_PLANNING_PHASE, PIPELINE_PREPARATION_PHASE,
    PUBLICATION_PHASE,
};
use effect_torch_graph::{node_children, CrossEntropyReduction, PositionOffset, RotaryLayout};
use effect_torch_runtime::{
    Buffer, CancellationFlag, DType, ExecutableDiagnostics, GgmlKQuant, InstructionId,
    InvocationMemoryReport, Location, MemoryPlan, NativeMemorySpace, ProgramSignature,
    SegmentOwnership, StorageClass, ValueId,
};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

static INVOCATION_NONCE: AtomicU64 = AtomicU64::new(0);

pub(crate) type Node = effect_torch_graph::Node;
pub(crate) type NodeKind = effect_torch_graph::NodeKind;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) struct KdaGeometry {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) struct ConvGeometry {
    pub layers: usize,
    pub channels: usize,
    pub kernel: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct KvStateSchema {
    pub max_tokens: usize,
    pub block_size: usize,
    pub kv_dtype: DType,
    pub window: Option<usize>,
    pub batch: usize,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kda: KdaGeometry,
    pub conv: ConvGeometry,
    pub cursor_slot: u32,
    pub cursor_tensor: bool,
}

impl KvStateSchema {
    fn validate(&self) -> Result<(), String> {
        if self.max_tokens == 0
            || self.block_size == 0
            || !self.max_tokens.is_multiple_of(self.block_size)
        {
            return Err(
                "compile: KV max_tokens must be positive and divisible by block_size".to_string(),
            );
        }
        if self.batch == 0 {
            return Err("compile: KV batch must be positive".to_string());
        }
        if !matches!(
            self.kv_dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::U8
        ) {
            return Err(format!(
                "compile: unsupported KV state dtype {}",
                self.kv_dtype
            ));
        }
        if self.layers == 0 && (self.kv_heads != 0 || self.head_dim != 0) {
            return Err("compile: KV heads and head_dim require at least one layer".to_string());
        }
        if self.layers > 0 && (self.kv_heads == 0 || self.head_dim == 0) {
            return Err("compile: KV layers require positive heads and head_dim".to_string());
        }
        if self
            .window
            .is_some_and(|window| window == 0 || window > self.max_tokens)
        {
            return Err("compile: KV window must be in 1..=max_tokens".to_string());
        }
        Ok(())
    }

    fn max_blocks(&self) -> usize {
        self.max_tokens / self.block_size
    }

    fn referenced_state_bytes(&self) -> Result<usize, String> {
        let checked = |values: &[usize], label: &str| {
            values
                .iter()
                .try_fold(1usize, |total, value| total.checked_mul(*value))
                .ok_or_else(|| format!("compile: {label} byte size overflow"))
        };
        let kv_elements = checked(
            &[
                self.layers,
                self.max_tokens,
                self.kv_heads,
                self.head_dim,
                2,
            ],
            "KV slab",
        )?;
        let mut bytes = kv_elements
            .checked_mul(self.kv_dtype.size_in_bytes())
            .ok_or_else(|| "compile: KV slab byte size overflow".to_string())?;
        if self.kv_dtype == DType::U8 {
            bytes = bytes
                .checked_add(
                    checked(
                        &[self.layers, self.max_tokens, self.kv_heads, 2],
                        "KV scale",
                    )?
                    .checked_mul(DType::F32.size_in_bytes())
                    .ok_or_else(|| "compile: KV scale byte size overflow".to_string())?,
                )
                .ok_or_else(|| "compile: KV state byte size overflow".to_string())?;
        }
        let recurrent_elements = checked(
            &[
                self.batch,
                self.kda.layers,
                self.kda.heads,
                self.kda.head_dim,
                self.kda.value_dim,
            ],
            "KDA state",
        )?
        .checked_add(checked(
            &[
                self.batch,
                self.conv.layers,
                self.conv.kernel.saturating_sub(1),
                self.conv.channels,
            ],
            "convolution state",
        )?)
        .ok_or_else(|| "compile: recurrent state size overflow".to_string())?;
        bytes
            .checked_add(
                recurrent_elements
                    .checked_mul(DType::F32.size_in_bytes())
                    .ok_or_else(|| "compile: recurrent state byte size overflow".to_string())?,
            )
            .ok_or_else(|| "compile: total state byte size overflow".to_string())
    }
}

pub(crate) struct SeqState {
    /// Compact live block table; `head` is the absolute logical index of `blocks[0]`.
    pub blocks: Vec<u32>,
    pub head: usize,
    pub cursor: usize,
    pub advance: usize,
    pub last_hash: u64,
    pub pending: Vec<u32>,
    pub kda_states: Vec<crate::run::MetalTensor>,
    pub conv_states: Vec<crate::run::MetalTensor>,
}

pub(crate) trait MetalDecodeContext {
    fn schema(&self) -> &KvStateSchema;
    fn slots(&self) -> &[Arc<Mutex<SeqState>>];
    fn active_batch(&self) -> usize {
        self.slots().len()
    }
    fn prepare_state(&self, cursor: &crate::run::MetalTensor) -> Result<(), String>;
    fn prepare_kv_attention(
        &self,
        layer: u32,
        plan: &KvAttentionPlan,
        staging: &[crate::run::MetalTensor],
    ) -> Result<(), String>;
    #[allow(clippy::too_many_arguments)]
    fn kv_attention_into(
        &self,
        layer: u32,
        q: &crate::run::MetalTensor,
        k: &crate::run::MetalTensor,
        v: &crate::run::MetalTensor,
        scale: f64,
        window: Option<usize>,
        output: &crate::run::MetalTensor,
        staging: &[crate::run::MetalTensor],
    ) -> Result<(), String>;
    fn evict_before(&self, state: &mut SeqState, start: usize);
    fn commit_slot(&self, index: usize, state: &mut SeqState);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetalBindingSource {
    Declared(u32),
    Generated(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MetalBinding {
    pub value: ValueId,
    pub source: MetalBindingSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalScalarBinding {
    value: ValueId,
    source: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalPaddedBinding {
    value: ValueId,
    source: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalDeclaredSource {
    Tensor(u32),
    Scalar(u32),
    StateCursor,
}

#[derive(Debug, Clone)]
struct MetalValueMetadata {
    pub shape: Box<[usize]>,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
pub(super) struct MetalLoweredValue {
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub layout: effect_torch_runtime::Layout,
    pub declaration: ValueDecl<NativeMemorySpace>,
}

impl LoweredValue<NativeMemorySpace> for MetalLoweredValue {
    fn value_decl(&self) -> &ValueDecl<NativeMemorySpace> {
        &self.declaration
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalValueStorage {
    External,
    Constant,
    Dynamic,
    ProvisionalOutput,
    Alias { source: ValueId, byte_offset: usize },
    Workspace,
    InvocationStaging,
    DeviceStatus,
    StateTransaction,
}

fn storage_root(storage: &[MetalValueStorage], mut value: ValueId) -> ValueId {
    while let MetalValueStorage::Alias { source, .. } = storage[value.index()] {
        value = source;
    }
    value
}

#[derive(Clone)]
struct MetalConstant {
    value: ValueId,
    payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetalUnaryOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Relu,
    Erf,
    Gelu { approximate: bool },
    Floor,
    Ceil,
    Round,
    Sign,
    Pow { exponent_bits: u64 },
    Cast { dtype: DType },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetalBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Gt,
    Lt,
    Ge,
    Le,
    Maximum,
    Minimum,
    Concat { dim: usize },
    Matmul,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetalReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetalImplementation {
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OptimizerImplementation {
    Fused,
    Generic,
}

#[derive(Debug, Clone)]
pub(super) enum MetalOp {
    PrepareState,
    Randn {
        shape: Box<[usize]>,
        dtype: DType,
    },
    Uniform {
        lo: f64,
        hi: f64,
        shape: Box<[usize]>,
        dtype: DType,
    },
    Unary(MetalUnaryOp),
    Binary(MetalBinaryOp),
    Where,
    Argmax {
        dim: usize,
    },
    Argmin {
        dim: usize,
    },
    Cumsum {
        dim: usize,
    },
    ScatterAdd {
        dim: usize,
    },
    Gather {
        dim: usize,
    },
    IndexSelect {
        dim: usize,
    },
    CrossEntropy {
        ignore_index: i64,
        reduction: CrossEntropyReduction,
        implementation: MetalImplementation,
    },
    CrossEntropyBackward {
        ignore_index: i64,
        reduction: CrossEntropyReduction,
        implementation: MetalImplementation,
    },
    ChunkedHeadCe {
        ignore_index: i64,
        chunk_size: usize,
    },
    ChunkedHeadCeBackward {
        ignore_index: i64,
        chunk_size: usize,
    },
    Sdpa {
        scale: f64,
        causal: bool,
        window: Option<usize>,
        implementation: MetalImplementation,
    },
    SdpaBackward {
        scale: f64,
        causal: bool,
        window: Option<usize>,
        implementation: MetalImplementation,
    },
    KdaChunk {
        scale: f64,
    },
    KdaRecurrence {
        scale: f64,
        layer: u32,
    },
    KdaBackward {
        scale: f64,
    },
    ShortConv1d,
    ShortConv1dBackwardX,
    ShortConv1dBackwardW,
    ConvState {
        layer: u32,
    },
    LastTokenRow,
    PositionEmbedding {
        seq_len: usize,
    },
    KvAttention {
        scale: f64,
        layer: u32,
        window: Option<usize>,
    },
    RotaryEmbedding {
        theta: f64,
        cursor_offset: bool,
        layout: RotaryLayout,
        implementation: MetalImplementation,
    },
    RotaryEmbeddingBackward {
        theta: f64,
        layout: RotaryLayout,
        implementation: MetalImplementation,
    },
    Linear {
        implementation: MetalImplementation,
    },
    QuantizedLinear {
        codec: GgmlKQuant,
        weight_shape: [usize; 2],
    },
    QuantizedEmbedding {
        codec: GgmlKQuant,
        weight_shape: [usize; 2],
    },
    LinearResidual {
        implementation: MetalImplementation,
    },
    LinearGelu {
        approximate: bool,
        dual: bool,
        implementation: MetalImplementation,
    },
    LayerNorm {
        eps: f64,
        implementation: MetalImplementation,
    },
    RmsNorm {
        eps: f64,
        implementation: MetalImplementation,
    },
    LayerNormBackward {
        eps: f64,
        implementation: MetalImplementation,
    },
    Conv1d {
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv2d {
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose1d {
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose2d {
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv1dBackwardW {
        kernel: usize,
        out_channels: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv2dBackwardW {
        kernel: [usize; 2],
        out_channels: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Reduce {
        op: MetalReduceOp,
        dims: Box<[usize]>,
        keepdims: bool,
    },
    Reshape {
        shape: Box<[usize]>,
    },
    Permute {
        dims: Box<[usize]>,
    },
    Slice {
        ranges: Box<[(usize, usize, usize)]>,
    },
    BroadcastTo {
        shape: Box<[usize]>,
    },
    PackOptimizerScalars,
    AdamW {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        implementation: OptimizerImplementation,
        exprs: Box<[Expr]>,
    },
    AdamWGroup {
        parameters: usize,
        exprs: Box<[Expr]>,
    },
    Sgd {
        momentum: f64,
        dampening: f64,
        nesterov: bool,
        weight_decay: f64,
        implementation: OptimizerImplementation,
        exprs: Box<[Expr]>,
    },
    FusedElementwise {
        strides: Box<[Vec<usize>]>,
        shape: Box<[usize]>,
        exprs: Box<[Expr]>,
    },
    FusedReduce {
        strides: Box<[Vec<usize>]>,
        in_shape: Box<[usize]>,
        expr: Expr,
        op: effect_torch_compiler::ReduceOp,
        dims: Box<[usize]>,
        keepdims: bool,
        shape: Box<[usize]>,
    },
}

impl MetalOp {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::PrepareState => "prepare_state",
            Self::Randn { .. } => "randn",
            Self::Uniform { .. } => "uniform",
            Self::Unary(_) => "unary",
            Self::Binary(_) => "binary",
            Self::Where => "where",
            Self::Argmax { .. } | Self::Argmin { .. } => "arg_reduce",
            Self::Cumsum { .. } => "cumsum",
            Self::ScatterAdd { .. } => "scatter_add",
            Self::Gather { .. } => "gather",
            Self::IndexSelect { .. } => "index_select",
            Self::CrossEntropy { .. } => "cross_entropy_native",
            Self::CrossEntropyBackward { .. } => "cross_entropy_backward_native",
            Self::ChunkedHeadCe { .. } => "chunked_head_ce",
            Self::ChunkedHeadCeBackward { .. } => "chunked_head_ce_backward",
            Self::Sdpa { .. } => "sdpa_flash",
            Self::SdpaBackward { .. } => "sdpa_backward_flash",
            Self::KdaChunk { .. } => "kda_chunk",
            Self::KdaRecurrence { .. } => "kda_recurrence",
            Self::KdaBackward { .. } => "kda_backward",
            Self::ShortConv1d => "short_conv1d",
            Self::ShortConv1dBackwardX => "short_conv1d_backward_x",
            Self::ShortConv1dBackwardW => "short_conv1d_backward_w",
            Self::ConvState { .. } => "conv_state",
            Self::LastTokenRow => "last_token_row",
            Self::PositionEmbedding { .. } => "position_embedding",
            Self::KvAttention { .. } => "kv_attention",
            Self::RotaryEmbedding { .. } | Self::RotaryEmbeddingBackward { .. } => "rotary_native",
            Self::Linear { .. } | Self::LinearResidual { .. } | Self::LinearGelu { .. } => {
                "linear_native"
            }
            Self::QuantizedLinear { .. } => "quantized_linear",
            Self::QuantizedEmbedding { .. } => "quantized_embedding",
            Self::LayerNorm { .. } | Self::LayerNormBackward { .. } => "layer_norm_native",
            Self::RmsNorm { .. } => "rms_norm_native",
            Self::Conv1d { .. } => "conv1d",
            Self::Conv2d { .. } => "conv2d",
            Self::ConvTranspose1d { .. } => "conv_transpose1d",
            Self::ConvTranspose2d { .. } => "conv_transpose2d",
            Self::Conv1dBackwardW { .. } => "conv1d_backward_w",
            Self::Conv2dBackwardW { .. } => "conv2d_backward_w",
            Self::Reduce { .. } => "reduce",
            Self::Reshape { .. } => "reshape",
            Self::Permute { .. } => "permute",
            Self::Slice { .. } => "slice",
            Self::BroadcastTo { .. } => "broadcast_to",
            Self::PackOptimizerScalars => "pack_optimizer_scalars",
            Self::AdamW { implementation, .. } => match implementation {
                OptimizerImplementation::Fused => "adamw_fused",
                OptimizerImplementation::Generic => "adamw_generic",
            },
            Self::AdamWGroup { .. } => "adamw_group_fused",
            Self::Sgd { implementation, .. } => match implementation {
                OptimizerImplementation::Fused => "sgd_fused",
                OptimizerImplementation::Generic => "sgd_generic",
            },
            Self::FusedElementwise { .. } => "fused_elementwise",
            Self::FusedReduce { .. } => "fused_reduce",
        }
    }

    fn profile_name(&self, plan: &MetalCommandPlan) -> &'static str {
        match (self, plan) {
            (
                Self::QuantizedLinear { codec, .. },
                MetalCommandPlan::QuantizedLinear(requirements),
            ) => quantized_linear_profile_name(*codec, requirements.vectors),
            _ => self.name(),
        }
    }
}

fn quantized_linear_profile_name(codec: GgmlKQuant, vectors: usize) -> &'static str {
    match (codec, vectors) {
        (GgmlKQuant::Q2K, 1) => "quantized_linear_q2_k_m1",
        (GgmlKQuant::Q2K, 2..=8) => "quantized_linear_q2_k_m2_8",
        (GgmlKQuant::Q2K, _) => "quantized_linear_q2_k_m9_plus",
        (GgmlKQuant::Q3K, 1) => "quantized_linear_q3_k_m1",
        (GgmlKQuant::Q3K, 2..=8) => "quantized_linear_q3_k_m2_8",
        (GgmlKQuant::Q3K, _) => "quantized_linear_q3_k_m9_plus",
        (GgmlKQuant::Q4K, 1) => "quantized_linear_q4_k_m1",
        (GgmlKQuant::Q4K, 2..=8) => "quantized_linear_q4_k_m2_8",
        (GgmlKQuant::Q4K, _) => "quantized_linear_q4_k_m9_plus",
        (GgmlKQuant::Q5K, 1) => "quantized_linear_q5_k_m1",
        (GgmlKQuant::Q5K, 2..=8) => "quantized_linear_q5_k_m2_8",
        (GgmlKQuant::Q5K, _) => "quantized_linear_q5_k_m9_plus",
        (GgmlKQuant::Q6K, 1) => "quantized_linear_q6_k_m1",
        (GgmlKQuant::Q6K, 2..=8) => "quantized_linear_q6_k_m2_8",
        (GgmlKQuant::Q6K, _) => "quantized_linear_q6_k_m9_plus",
    }
}

#[derive(Debug, Clone)]
pub(super) enum MetalInstruction {
    PrepareInvocation,
    Operation {
        op: MetalOp,
        plan: MetalCommandPlan,
        release: Box<[ValueId]>,
        /// Legacy Metal randomness is keyed by physical lowered-command order.
        random_seed_token: u64,
    },
    FinalizeInvocation,
}

impl MetalInstruction {
    fn name(&self) -> &'static str {
        match self {
            Self::PrepareInvocation => "prepare_invocation",
            Self::Operation { op, .. } => op.name(),
            Self::FinalizeInvocation => "publish_outputs",
        }
    }

    pub(super) fn operation(&self) -> Option<(&MetalOp, &MetalCommandPlan)> {
        match self {
            Self::Operation { op, plan, .. } => Some((op, plan)),
            Self::PrepareInvocation | Self::FinalizeInvocation => None,
        }
    }

    fn operation_mut(
        &mut self,
    ) -> Option<(
        &mut MetalOp,
        &mut MetalCommandPlan,
        &mut Box<[ValueId]>,
        &mut u64,
    )> {
        match self {
            Self::Operation {
                op,
                plan,
                release,
                random_seed_token,
            } => Some((op, plan, release, random_seed_token)),
            Self::PrepareInvocation | Self::FinalizeInvocation => None,
        }
    }
}

pub(super) type MetalCommand = LoweredInstruction<MetalInstruction>;

trait MetalResourceId {
    fn index(&self) -> usize;
}

impl MetalResourceId for ValueUse {
    fn index(&self) -> usize {
        self.value.index()
    }
}

impl MetalResourceId for OutputDecl {
    fn index(&self) -> usize {
        self.value.index()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetalPhysicalCommand {
    Encode(InstructionId),
    StatusGate(InstructionId),
    Commit,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MetalPreparedArtifacts {
    pub pipeline_count: usize,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // Exact requirement records are also consumed by precompile.
pub(super) enum MetalCommandPlan {
    #[default]
    Direct,
    Gemm(crate::gemm::GemmRequirements),
    Linear(crate::linear::LinearRequirements),
    Conv(crate::conv::ConvRequirements),
    Indexing(crate::indexing::IndexingRequirements),
    CeForward(crate::loss::CeForwardRequirements),
    CeBackward(crate::loss::CeBackwardRequirements),
    ChunkedHeadForward(ChunkedHeadForwardPlan),
    ChunkedHeadBackward(ChunkedHeadBackwardPlan),
    SdpaForward(crate::flash::SdpaForwardRequirements),
    SdpaBackward(crate::flash::SdpaBackwardRequirements),
    LayerNormForward(crate::layer_norm::LayerNormForwardRequirements),
    LayerNormBackward(crate::layer_norm::LayerNormBackwardRequirements),
    KdaForward(crate::kda::ForwardRequirements),
    KdaDecode(crate::kda::DecodeRequirements),
    KdaBackward(crate::kda::BackwardRequirements),
    ShortConvForward(crate::shortconv::ForwardRequirements),
    ShortConvState(crate::shortconv::ForwardRequirements),
    ShortConvBackwardX(crate::shortconv::BackwardXRequirements),
    ShortConvBackwardW(crate::shortconv::BackwardWRequirements),
    Rotary(crate::rotary::RotaryRequirements),
    QuantizedLinear(crate::quantized::LinearRequirements),
    QuantizedEmbedding(crate::quantized::EmbeddingRequirements),
    KvAttention(KvAttentionPlan),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KvAttentionPlan {
    pub(crate) batch: usize,
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) time: usize,
    pub(crate) head_dim: usize,
}

#[derive(Debug, Clone)]
struct ChunkedHeadForwardVariant {
    rows: usize,
    head: crate::gemm::GemmRequirements,
    ce: crate::loss::CeForwardRequirements,
}

#[derive(Debug, Clone)]
pub(super) struct ChunkedHeadForwardPlan {
    rows: usize,
    inner: usize,
    vocab: usize,
    chunk_len: usize,
    full: ChunkedHeadForwardVariant,
    tail: Option<ChunkedHeadForwardVariant>,
    split_k_elements: usize,
}

#[derive(Debug, Clone)]
struct ChunkedHeadBackwardVariant {
    rows: usize,
    head: crate::gemm::GemmRequirements,
    dx: crate::gemm::GemmRequirements,
    dw: crate::gemm::GemmRequirements,
}

#[derive(Debug, Clone)]
pub(super) struct ChunkedHeadBackwardPlan {
    rows: usize,
    inner: usize,
    vocab: usize,
    chunk_len: usize,
    full: ChunkedHeadBackwardVariant,
    tail: Option<ChunkedHeadBackwardVariant>,
    split_k_elements: usize,
}

fn chunked_head_geometry(
    x: &MetalValueMetadata,
    weight: &MetalValueMetadata,
    target: &MetalValueMetadata,
    chunk_size: usize,
) -> Result<(usize, usize, usize, usize), String> {
    if x.shape.is_empty() || weight.shape.len() != 2 {
        return Err(
            "compile: chunked head CE requires rank >= 1 input and rank-2 weight".to_string(),
        );
    }
    let rows = x.shape[..x.shape.len() - 1]
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| "compile: chunked head CE row count overflow".to_string())?;
    let inner = weight.shape[0];
    let vocab = weight.shape[1];
    if rows == 0
        || inner != x.shape[x.shape.len() - 1]
        || target.shape.iter().product::<usize>() != rows
    {
        return Err("compile: chunked head CE input geometry is inconsistent".to_string());
    }
    let chunks = rows
        .checked_mul(vocab)
        .map(|elements| (elements / chunk_size).clamp(2, 64).min(rows))
        .ok_or_else(|| "compile: chunked head CE element count overflow".to_string())?;
    Ok((rows, inner, vocab, rows.div_ceil(chunks.max(1))))
}

fn split_k_elements(requirements: &crate::gemm::GemmRequirements) -> usize {
    requirements
        .split_k_scratch
        .map_or(0, |requirement| requirement.elements)
}

#[derive(Debug)]
struct ResourceSpec {
    name: String,
    shape: Vec<usize>,
    dtype: DType,
    storage: MetalValueStorage,
}

#[derive(Debug, Default)]
struct CommandResources {
    scratch: Vec<ResourceSpec>,
    staging: Vec<ResourceSpec>,
    status: Vec<ResourceSpec>,
    plan: MetalCommandPlan,
}

fn tensor_bytes(shape: &[usize], dtype: DType, label: &str) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
        .map(|bytes| bytes.max(1))
        .ok_or_else(|| format!("compile: {label} byte size overflow"))
}

fn scratch(name: impl Into<String>, shape: &[usize], dtype: DType) -> ResourceSpec {
    ResourceSpec {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
        storage: MetalValueStorage::Workspace,
    }
}

fn staging(name: impl Into<String>, shape: &[usize], dtype: DType) -> ResourceSpec {
    ResourceSpec {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
        storage: MetalValueStorage::InvocationStaging,
    }
}

fn status(name: impl Into<String>, shape: &[usize], dtype: DType) -> ResourceSpec {
    ResourceSpec {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
        storage: MetalValueStorage::DeviceStatus,
    }
}

fn transaction(name: impl Into<String>, shape: &[usize], dtype: DType) -> ResourceSpec {
    ResourceSpec {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
        storage: MetalValueStorage::StateTransaction,
    }
}

fn ops_scratch(
    prefix: &str,
    requirements: impl IntoIterator<Item = crate::ops::ScratchRequirement>,
) -> Vec<ResourceSpec> {
    requirements
        .into_iter()
        .enumerate()
        .map(|(index, requirement)| {
            scratch(
                format!("{prefix}_scratch_{index}"),
                &requirement.shape,
                requirement.dtype,
            )
        })
        .collect()
}

fn declaration_layout(value: &MetalValueMetadata) -> effect_torch_runtime::Layout {
    effect_torch_runtime::Layout::contiguous(value.shape.to_vec())
}

fn plan_command_resources(
    command: &MetalCommand,
    values: &[MetalValueMetadata],
    environment: MetalEnvironment,
    state_schema: Option<&KvStateSchema>,
) -> Result<CommandResources, String> {
    let (op, _) = command
        .kind
        .operation()
        .ok_or_else(|| "compile: boundary instruction has no command resources".to_string())?;
    let input = |index: usize| -> Result<&MetalValueMetadata, String> {
        let value = command
            .inputs
            .get(index)
            .map(|use_| use_.value)
            .ok_or_else(|| format!("compile: {} is missing input {index}", op.name()))?;
        values
            .get(value.index())
            .ok_or_else(|| format!("compile: command references unknown value {value}"))
    };
    let output = |index: usize| -> Result<&MetalValueMetadata, String> {
        let value = command
            .outputs
            .get(index)
            .map(|output| output.value)
            .ok_or_else(|| format!("compile: {} is missing output {index}", op.name()))?;
        values
            .get(value.index())
            .ok_or_else(|| format!("compile: command references unknown value {value}"))
    };
    let mut resources = CommandResources::default();
    match op {
        MetalOp::PrepareState => {}
        MetalOp::Randn { shape, dtype } | MetalOp::Uniform { shape, dtype, .. } => {
            if *dtype != DType::F32 {
                resources
                    .scratch
                    .push(scratch("random_f32", shape, DType::F32));
            }
        }
        MetalOp::Unary(kind) => {
            let source = input(0)?;
            let requirements = match kind {
                MetalUnaryOp::Cast { .. } => Vec::new(),
                MetalUnaryOp::Relu => match source.dtype {
                    DType::F16 | DType::BF16 => vec![crate::ops::ScratchRequirement {
                        shape: source.shape.to_vec(),
                        dtype: DType::F32,
                    }],
                    _ => Vec::new(),
                },
                MetalUnaryOp::Pow { .. } if source.dtype != DType::F32 => {
                    vec![crate::ops::ScratchRequirement {
                        shape: source.shape.to_vec(),
                        dtype: DType::F32,
                    }]
                }
                MetalUnaryOp::Pow { .. } => Vec::new(),
                _ if source.dtype != DType::F32 => vec![crate::ops::ScratchRequirement {
                    shape: source.shape.to_vec(),
                    dtype: DType::F32,
                }],
                _ => Vec::new(),
            };
            resources.scratch = ops_scratch("unary", requirements);
        }
        MetalOp::Binary(MetalBinaryOp::Matmul) => {
            let a = input(0)?;
            let b = input(1)?;
            let requirements = crate::gemm::matmul_requirements(
                device::MetalDevice::get(),
                &a.shape,
                &b.shape,
                a.dtype,
                environment.mma,
            )?;
            if let Some(requirement) = requirements.split_k_scratch {
                resources.scratch.push(scratch(
                    "matmul_split_k",
                    &requirement.shape,
                    requirement.dtype,
                ));
            }
            resources.plan = MetalCommandPlan::Gemm(requirements);
        }
        MetalOp::Binary(MetalBinaryOp::Concat { .. }) => {}
        MetalOp::Binary(kind) => {
            let a = input(0)?;
            let b = input(1)?;
            let compare = matches!(
                kind,
                MetalBinaryOp::Eq
                    | MetalBinaryOp::Gt
                    | MetalBinaryOp::Lt
                    | MetalBinaryOp::Ge
                    | MetalBinaryOp::Le
            );
            let requirements = if compare {
                let mut requirements = Vec::new();
                if a.dtype != DType::F32 {
                    requirements.push(crate::ops::ScratchRequirement {
                        shape: a.shape.to_vec(),
                        dtype: DType::F32,
                    });
                }
                if b.dtype != DType::F32 {
                    requirements.push(crate::ops::ScratchRequirement {
                        shape: b.shape.to_vec(),
                        dtype: DType::F32,
                    });
                }
                requirements.push(crate::ops::ScratchRequirement {
                    shape: output(0)?.shape.to_vec(),
                    dtype: DType::F32,
                });
                requirements
            } else {
                let mut requirements = Vec::new();
                let mut a_dtype = a.dtype;
                let mut b_dtype = b.dtype;
                if a_dtype != b_dtype
                    && a_dtype.is_float()
                    && b_dtype.is_float()
                    && a.shape.is_empty()
                    && !b.shape.is_empty()
                {
                    requirements.push(crate::ops::ScratchRequirement {
                        shape: a.shape.to_vec(),
                        dtype: b_dtype,
                    });
                    a_dtype = b_dtype;
                } else if a_dtype != b_dtype
                    && a_dtype.is_float()
                    && b_dtype.is_float()
                    && b.shape.is_empty()
                    && !a.shape.is_empty()
                {
                    requirements.push(crate::ops::ScratchRequirement {
                        shape: b.shape.to_vec(),
                        dtype: a_dtype,
                    });
                    b_dtype = a_dtype;
                }
                if !(a_dtype == b_dtype && matches!(a_dtype, DType::F32 | DType::BF16)) {
                    if a_dtype != DType::F32 {
                        requirements.push(crate::ops::ScratchRequirement {
                            shape: a.shape.to_vec(),
                            dtype: DType::F32,
                        });
                    }
                    if b_dtype != DType::F32 {
                        requirements.push(crate::ops::ScratchRequirement {
                            shape: b.shape.to_vec(),
                            dtype: DType::F32,
                        });
                    }
                    let output_dtype = if a.dtype != b.dtype
                        && a.dtype.is_float()
                        && b.dtype.is_float()
                        && a.shape.is_empty()
                        && !b.shape.is_empty()
                    {
                        b.dtype
                    } else {
                        a.dtype
                    };
                    if output_dtype != DType::F32 {
                        requirements.push(crate::ops::ScratchRequirement {
                            shape: output(0)?.shape.to_vec(),
                            dtype: DType::F32,
                        });
                    }
                }
                requirements
            };
            resources.scratch = ops_scratch("binary", requirements);
        }
        MetalOp::Where => {
            let condition = input(0)?;
            let a = input(1)?;
            let b = input(2)?;
            if a.dtype != b.dtype {
                return Err("compile: where branch dtypes must match".to_string());
            }
            if condition.dtype != a.dtype {
                resources
                    .scratch
                    .push(scratch("where_condition", &condition.shape, a.dtype));
            }
        }
        MetalOp::Argmax { dim } | MetalOp::Argmin { dim } => {
            let mut keepdim_shape = input(0)?.shape.to_vec();
            keepdim_shape[*dim] = 1;
            resources
                .scratch
                .push(scratch("arg_reduce_u32", &keepdim_shape, DType::U32));
        }
        MetalOp::Cumsum { .. } => {}
        MetalOp::IndexSelect { dim } => {
            let source = input(0)?;
            let ids = input(1)?;
            let requirements = crate::indexing::index_select_requirements(
                &declaration_layout(source),
                source.dtype,
                *dim,
                &declaration_layout(ids),
                ids.dtype,
            )?;
            if let Some(requirement) = &requirements.ids {
                resources.scratch.push(scratch(
                    "index_select_ids",
                    &requirement.shape,
                    requirement.dtype,
                ));
            }
            resources.plan = MetalCommandPlan::Indexing(requirements);
        }
        MetalOp::Gather { dim } => {
            let source = input(0)?;
            let ids = input(1)?;
            let requirements = crate::indexing::gather_requirements(
                &declaration_layout(source),
                source.dtype,
                *dim,
                &declaration_layout(ids),
                ids.dtype,
                &ids.shape,
            )?;
            if let Some(requirement) = &requirements.ids {
                resources.scratch.push(scratch(
                    "gather_ids",
                    &requirement.shape,
                    requirement.dtype,
                ));
            }
            resources.plan = MetalCommandPlan::Indexing(requirements);
        }
        MetalOp::ScatterAdd { dim } => {
            let source = input(0)?;
            let ids = input(1)?;
            let update = input(2)?;
            let requirements = crate::indexing::scatter_add_requirements(
                &declaration_layout(source),
                source.dtype,
                *dim,
                &declaration_layout(ids),
                ids.dtype,
                &declaration_layout(update),
                update.dtype,
            )?;
            for (name, requirement) in [
                ("scatter_ids", &requirements.ids),
                ("scatter_accumulator", &requirements.accumulator),
                ("scatter_source_cast", &requirements.source_cast),
            ] {
                if let Some(requirement) = requirement {
                    resources
                        .scratch
                        .push(scratch(name, &requirement.shape, requirement.dtype));
                }
            }
            resources.plan = MetalCommandPlan::Indexing(requirements);
        }
        MetalOp::CrossEntropy {
            reduction,
            implementation: MetalImplementation::Native,
            ..
        } => {
            let logits = input(0)?;
            let target = input(1)?;
            let requirements = loss::ce_forward_requirements(
                &logits.shape,
                logits.dtype,
                &target.shape,
                target.dtype,
                *reduction,
            )?;
            if output(0)?.dtype != DType::F32 {
                resources.scratch.push(scratch(
                    "ce_loss_f32",
                    &requirements.loss.shape,
                    DType::F32,
                ));
            }
            resources.scratch.push(scratch(
                "ce_nll",
                &requirements.nll_scratch.shape,
                requirements.nll_scratch.dtype,
            ));
            resources.scratch.push(scratch(
                "ce_flags",
                &requirements.flags_scratch.shape,
                requirements.flags_scratch.dtype,
            ));
            resources.status.push(status(
                "ce_status",
                &requirements.status.shape,
                requirements.status.dtype,
            ));
            resources.plan = MetalCommandPlan::CeForward(requirements);
        }
        MetalOp::CrossEntropyBackward {
            reduction,
            implementation: MetalImplementation::Native,
            ..
        } => {
            let logits = input(0)?;
            let target = input(1)?;
            let requirements = loss::ce_backward_requirements(
                &logits.shape,
                logits.dtype,
                &target.shape,
                target.dtype,
                *reduction,
            )?;
            if let Some(requirement) = &requirements.count_status {
                resources.status.push(status(
                    "ce_count_status",
                    &requirement.shape,
                    requirement.dtype,
                ));
            }
            resources.plan = MetalCommandPlan::CeBackward(requirements);
        }
        MetalOp::ChunkedHeadCe { chunk_size, .. } => {
            let x = input(0)?;
            let weight = input(1)?;
            let bias = input(2)?;
            let target = input(3)?;
            let (rows, inner, vocab, chunk_len) =
                chunked_head_geometry(x, weight, target, *chunk_size)?;
            if weight.dtype != x.dtype || bias.dtype != x.dtype || bias.shape.as_ref() != [vocab] {
                return Err("compile: chunked head CE weight and bias must match the input dtype and geometry".to_string());
            }
            let make_variant = |length: usize| -> Result<ChunkedHeadForwardVariant, String> {
                Ok(ChunkedHeadForwardVariant {
                    rows: length,
                    head: crate::gemm::gemm_requirements(
                        device::MetalDevice::get(),
                        x.dtype,
                        true,
                        crate::gemm::Epilogue::None,
                        1,
                        length,
                        vocab,
                        inner,
                        environment.mma,
                    )?,
                    ce: loss::ce_forward_requirements(
                        &[length, vocab],
                        x.dtype,
                        &[length],
                        target.dtype,
                        CrossEntropyReduction::Sum,
                    )?,
                })
            };
            let full = make_variant(chunk_len)?;
            let tail_len = rows % chunk_len;
            let tail = (tail_len != 0)
                .then(|| make_variant(tail_len))
                .transpose()?;
            let split_k_elements = std::iter::once(&full)
                .chain(tail.iter())
                .map(|variant| split_k_elements(&variant.head))
                .max()
                .unwrap_or(0);
            resources
                .scratch
                .push(scratch("chunked_head_logits", &[chunk_len, vocab], x.dtype));
            resources
                .scratch
                .push(scratch("chunked_head_status", &[3], DType::F32));
            resources
                .scratch
                .push(scratch("chunked_head_nll", &[chunk_len], DType::F32));
            resources
                .scratch
                .push(scratch("chunked_head_flags", &[chunk_len], DType::U32));
            if split_k_elements != 0 {
                resources.scratch.push(scratch(
                    "chunked_head_split_k",
                    &[split_k_elements],
                    DType::F32,
                ));
            }
            if output(0)?.dtype != DType::F32 {
                resources
                    .scratch
                    .push(scratch("chunked_head_loss_f32", &[], DType::F32));
            }
            resources
                .status
                .push(status("chunked_head_aggregate", &[3], DType::F32));
            resources.plan = MetalCommandPlan::ChunkedHeadForward(ChunkedHeadForwardPlan {
                rows,
                inner,
                vocab,
                chunk_len,
                full,
                tail,
                split_k_elements,
            });
        }
        MetalOp::ChunkedHeadCeBackward { chunk_size, .. } => {
            let x = input(0)?;
            let weight = input(1)?;
            let bias = input(2)?;
            let target = input(3)?;
            let gradient = input(4)?;
            let (rows, inner, vocab, chunk_len) =
                chunked_head_geometry(x, weight, target, *chunk_size)?;
            if weight.dtype != x.dtype || bias.dtype != x.dtype || bias.shape.as_ref() != [vocab] {
                return Err("compile: chunked head CE backward weight and bias must match the input dtype and geometry".to_string());
            }
            if gradient.shape.iter().product::<usize>() != 1 {
                return Err("compile: chunked head CE backward gradient must be scalar".to_string());
            }
            let make_variant = |length: usize| -> Result<ChunkedHeadBackwardVariant, String> {
                Ok(ChunkedHeadBackwardVariant {
                    rows: length,
                    head: crate::gemm::gemm_requirements(
                        device::MetalDevice::get(),
                        x.dtype,
                        true,
                        crate::gemm::Epilogue::None,
                        1,
                        length,
                        vocab,
                        inner,
                        environment.mma,
                    )?,
                    dx: crate::gemm::gemm_requirements(
                        device::MetalDevice::get(),
                        DType::F32,
                        false,
                        crate::gemm::Epilogue::None,
                        1,
                        length,
                        inner,
                        vocab,
                        environment.mma,
                    )?,
                    dw: crate::gemm::gemm_requirements(
                        device::MetalDevice::get(),
                        DType::F32,
                        false,
                        crate::gemm::Epilogue::None,
                        1,
                        inner,
                        vocab,
                        length,
                        environment.mma,
                    )?,
                })
            };
            let full = make_variant(chunk_len)?;
            let tail_len = rows % chunk_len;
            let tail = (tail_len != 0)
                .then(|| make_variant(tail_len))
                .transpose()?;
            let split_k_elements = std::iter::once(&full)
                .chain(tail.iter())
                .flat_map(|variant| [&variant.head, &variant.dx, &variant.dw])
                .map(split_k_elements)
                .max()
                .unwrap_or(0);
            resources.scratch.extend([
                scratch("chunked_head_logits", &[chunk_len, vocab], x.dtype),
                scratch(
                    "chunked_head_grad_logits_f32",
                    &[chunk_len, vocab],
                    DType::F32,
                ),
            ]);
            resources
                .scratch
                .push(scratch("chunked_head_scale", &[1], DType::F32));
            if gradient.dtype != DType::F32 {
                resources
                    .scratch
                    .push(scratch("chunked_head_gradient_f32", &[], DType::F32));
            }
            if weight.dtype != DType::F32 {
                resources.scratch.push(scratch(
                    "chunked_head_weight_f32",
                    &weight.shape,
                    DType::F32,
                ));
            }
            resources.scratch.extend([scratch(
                "chunked_head_weight_t",
                &[vocab, inner],
                DType::F32,
            )]);
            if x.dtype != DType::F32 {
                resources.scratch.push(scratch(
                    "chunked_head_x_f32",
                    &[chunk_len, inner],
                    DType::F32,
                ));
            }
            resources
                .scratch
                .push(scratch("chunked_head_x_t", &[inner, chunk_len], DType::F32));
            if x.dtype != DType::F32 {
                resources.scratch.push(scratch(
                    "chunked_head_dx_f32",
                    &[chunk_len, inner],
                    DType::F32,
                ));
            }
            resources.scratch.extend([
                scratch("chunked_head_dw", &[inner, vocab], DType::F32),
                scratch("chunked_head_dw_acc", &[inner, vocab], DType::F32),
                scratch("chunked_head_db", &[1, vocab], DType::F32),
                scratch("chunked_head_db_acc", &[1, vocab], DType::F32),
            ]);
            if split_k_elements != 0 {
                resources.scratch.push(scratch(
                    "chunked_head_split_k",
                    &[split_k_elements],
                    DType::F32,
                ));
            }
            resources
                .status
                .push(status("chunked_head_aggregate", &[3], DType::F32));
            resources.plan = MetalCommandPlan::ChunkedHeadBackward(ChunkedHeadBackwardPlan {
                rows,
                inner,
                vocab,
                chunk_len,
                full,
                tail,
                split_k_elements,
            });
        }
        MetalOp::Sdpa {
            implementation: MetalImplementation::Native,
            ..
        } => {
            let q = input(0)?;
            let k = input(1)?;
            let v = input(2)?;
            resources.plan = MetalCommandPlan::SdpaForward(flash::forward_requirements(
                &q.shape, &k.shape, &v.shape, q.dtype,
            )?);
        }
        MetalOp::QuantizedLinear {
            codec,
            weight_shape,
        } => {
            let x = input(0)?;
            let weight = input(1)?;
            let bias = if command.inputs.len() == 3 {
                let bias = input(2)?;
                Some((bias.shape.as_ref(), bias.dtype))
            } else {
                None
            };
            let output = output(0)?;
            resources.plan = MetalCommandPlan::QuantizedLinear(quantized::linear_requirements(
                &x.shape,
                x.dtype,
                &weight.shape,
                weight.dtype,
                bias,
                &output.shape,
                output.dtype,
                *codec,
                *weight_shape,
            )?);
        }
        MetalOp::QuantizedEmbedding {
            codec,
            weight_shape,
        } => {
            let indexes = input(0)?;
            let weight = input(1)?;
            let output = output(0)?;
            resources.plan =
                MetalCommandPlan::QuantizedEmbedding(quantized::embedding_requirements(
                    &indexes.shape,
                    indexes.dtype,
                    &weight.shape,
                    weight.dtype,
                    &output.shape,
                    output.dtype,
                    *codec,
                    *weight_shape,
                )?);
            resources
                .status
                .push(status("quantized_embedding_status", &[1], DType::U32));
        }
        MetalOp::SdpaBackward {
            implementation: MetalImplementation::Native,
            ..
        } => {
            let q = input(0)?;
            let k = input(1)?;
            let v = input(2)?;
            let requirements = flash::backward_requirements(&q.shape, &k.shape, &v.shape, q.dtype)?;
            resources.scratch.push(scratch(
                "sdpa_d_vec",
                &requirements.d_vec_scratch.shape,
                requirements.d_vec_scratch.dtype,
            ));
            resources.plan = MetalCommandPlan::SdpaBackward(requirements);
        }
        MetalOp::Linear {
            implementation: MetalImplementation::Native,
        }
        | MetalOp::LinearResidual {
            implementation: MetalImplementation::Native,
        }
        | MetalOp::LinearGelu {
            implementation: MetalImplementation::Native,
            ..
        } => {
            let x = input(0)?;
            let weight = input(1)?;
            let (residual, gelu) = match op {
                MetalOp::Linear { .. } => (false, None),
                MetalOp::LinearResidual { .. } => (true, None),
                MetalOp::LinearGelu {
                    approximate, dual, ..
                } => (false, Some((*approximate, *dual))),
                _ => unreachable!(),
            };
            let requirements = linear::linear_forward_fused_requirements(
                device::MetalDevice::get(),
                &x.shape,
                &weight.shape,
                x.dtype,
                residual,
                gelu,
                environment.mma,
            )?;
            if let Some(requirement) = requirements.split_k_scratch {
                resources.scratch.push(scratch(
                    "linear_split_k",
                    &requirement.shape,
                    requirement.dtype,
                ));
            }
            resources.plan = MetalCommandPlan::Linear(requirements);
        }
        MetalOp::LayerNorm {
            implementation: MetalImplementation::Native,
            ..
        } => {
            let x = input(0)?;
            let weight = input(1)?;
            let bias = input(2)?;
            resources.plan =
                MetalCommandPlan::LayerNormForward(layer_norm::ln_forward_requirements(
                    &x.shape,
                    x.dtype,
                    &weight.shape,
                    weight.dtype,
                    &bias.shape,
                    bias.dtype,
                )?);
        }
        MetalOp::RmsNorm {
            implementation: MetalImplementation::Native,
            ..
        } => {
            let x = input(0)?;
            let weight = if command.inputs.len() == 2 {
                let weight = input(1)?;
                Some((weight.shape.as_ref(), weight.dtype))
            } else {
                None
            };
            resources.plan = MetalCommandPlan::LayerNormForward(
                layer_norm::rms_forward_requirements(&x.shape, x.dtype, weight)?,
            );
        }
        MetalOp::LayerNormBackward {
            implementation: MetalImplementation::Native,
            ..
        } => {
            let x = input(0)?;
            let weight = input(1)?;
            let gradient = input(2)?;
            let requirements = layer_norm::ln_backward_requirements(
                &x.shape,
                x.dtype,
                &weight.shape,
                weight.dtype,
                &gradient.shape,
                gradient.dtype,
            )?;
            resources.scratch.push(scratch(
                "layer_norm_normalized",
                &requirements.normalized.shape,
                requirements.normalized.dtype,
            ));
            resources
                .scratch
                .push(scratch("layer_norm_dw_product", &x.shape, x.dtype));
            resources.plan = MetalCommandPlan::LayerNormBackward(requirements);
        }
        MetalOp::KdaChunk { scale } => {
            let q = input(0)?;
            let v = input(2)?;
            let rank = q.shape.len();
            resources.plan = MetalCommandPlan::KdaForward(kda::forward_requirements(
                q.dtype,
                q.shape[..rank - 2].iter().product(),
                q.shape[rank - 2],
                q.shape[rank - 1],
                v.shape[rank - 1],
                *scale,
                false,
            )?);
        }
        MetalOp::KdaRecurrence { scale, layer } => {
            let q = input(0)?;
            let v = input(2)?;
            let rank = q.shape.len();
            if rank < 3 || v.shape.len() != rank {
                return Err(
                    "compile: KDA recurrence inputs must have matching rank >= 3".to_string(),
                );
            }
            let heads = q.shape[rank - 3];
            let time = q.shape[rank - 2];
            let dk = q.shape[rank - 1];
            let dv = v.shape[rank - 1];
            let batch = q.shape[..rank - 3]
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| "compile: KDA recurrence batch size overflow".to_string())?;
            let schema = state_schema.ok_or_else(|| {
                "compile: KDA recurrence requires an explicit state schema".to_string()
            })?;
            if batch != schema.batch
                || (*layer as usize) >= schema.kda.layers
                || heads != schema.kda.heads
                || dk != schema.kda.head_dim
                || dv != schema.kda.value_dim
            {
                return Err("compile: KDA recurrence does not match the state schema".to_string());
            }
            resources.scratch.push(transaction(
                "kda_state_next",
                &[batch, heads, dk, dv],
                DType::F32,
            ));
            if time == 1 {
                resources.plan = MetalCommandPlan::KdaDecode(kda::decode_requirements(
                    q.dtype, heads, dk, dv, *scale,
                )?);
            } else {
                resources
                    .staging
                    .push(staging("kda_advance_mask", &[batch, time, 1], q.dtype));
                resources
                    .scratch
                    .push(scratch("kda_masked_decay", &[heads, time, dk], q.dtype));
                resources
                    .scratch
                    .push(scratch("kda_masked_beta", &[heads, time, 1], q.dtype));
                resources.plan = MetalCommandPlan::KdaForward(kda::forward_requirements(
                    q.dtype, heads, time, dk, dv, *scale, true,
                )?);
            }
        }
        MetalOp::KdaBackward { scale } => {
            let q = input(0)?;
            let v = input(2)?;
            let rank = q.shape.len();
            let requirements = kda::backward_requirements(
                q.dtype,
                q.shape[..rank - 2].iter().product(),
                q.shape[rank - 2],
                q.shape[rank - 1],
                v.shape[rank - 1],
                *scale,
            )?;
            resources.scratch.push(scratch(
                "kda_backward",
                &[requirements.scratch_bytes / DType::F32.size_in_bytes()],
                DType::F32,
            ));
            resources.plan = MetalCommandPlan::KdaBackward(requirements);
        }
        MetalOp::ShortConv1d => {
            let x = input(0)?;
            let weight = input(1)?;
            let rank = x.shape.len();
            resources.plan = MetalCommandPlan::ShortConvForward(shortconv::forward_requirements(
                x.dtype,
                x.shape[..rank - 2].iter().product(),
                x.shape[rank - 2],
                x.shape[rank - 1],
                weight.shape[1],
                false,
            )?);
        }
        MetalOp::ShortConv1dBackwardX => {
            let x = input(0)?;
            let rank = x.shape.len();
            resources.plan =
                MetalCommandPlan::ShortConvBackwardX(shortconv::backward_x_requirements(
                    x.dtype,
                    x.shape[..rank - 2].iter().product(),
                    x.shape[rank - 2],
                    x.shape[rank - 1],
                )?);
        }
        MetalOp::ShortConv1dBackwardW => {
            let x = input(0)?;
            let weight = input(1)?;
            resources.plan = MetalCommandPlan::ShortConvBackwardW(
                shortconv::backward_w_requirements(x.dtype, weight.shape[0], weight.shape[1])?,
            );
        }
        MetalOp::ConvState { layer } => {
            let x = input(0)?;
            let weight = input(1)?;
            let rank = x.shape.len();
            if rank < 2 || weight.shape.len() != 2 {
                return Err(
                    "compile: convolution state requires rank >= 2 input and rank-2 weight"
                        .to_string(),
                );
            }
            let time = x.shape[rank - 2];
            let channels = x.shape[rank - 1];
            let kernel = weight.shape[1];
            let batch = x.shape[..rank - 2]
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| "compile: convolution state batch size overflow".to_string())?;
            let schema = state_schema.ok_or_else(|| {
                "compile: convolution state requires an explicit state schema".to_string()
            })?;
            if batch != schema.batch
                || (*layer as usize) >= schema.conv.layers
                || channels != schema.conv.channels
                || kernel != schema.conv.kernel
            {
                return Err(
                    "compile: convolution state does not match the state schema".to_string()
                );
            }
            let requirements = shortconv::state_requirements(x.dtype, time, channels, kernel)?;
            resources.scratch.push(transaction(
                "conv_state_next",
                &[batch, kernel.saturating_sub(1), channels],
                DType::F32,
            ));
            resources.plan = MetalCommandPlan::ShortConvState(requirements);
        }
        MetalOp::RotaryEmbedding { .. } | MetalOp::RotaryEmbeddingBackward { .. } => {
            let source = input(0)?;
            let requirements = rotary::rotary_requirements(source.dtype, &source.shape)?;
            let batch = if source.shape.len() == 2 {
                1
            } else {
                source.shape[0]
            };
            resources
                .staging
                .push(staging("rotary_offsets", &[batch], DType::F32));
            resources.plan = MetalCommandPlan::Rotary(requirements);
        }
        MetalOp::Reduce { .. } => {
            let source = input(0)?;
            if !matches!(source.dtype, DType::F32 | DType::BF16) {
                resources
                    .scratch
                    .push(scratch("reduce_input_f32", &source.shape, DType::F32));
                resources
                    .scratch
                    .push(scratch("reduce_output_f32", &output(0)?.shape, DType::F32));
            }
        }
        MetalOp::PackOptimizerScalars => {
            for (index, value) in command.inputs.iter().map(|use_| use_.value).enumerate() {
                let scalar = values
                    .get(value.index())
                    .ok_or_else(|| format!("compile: command references unknown value {value}"))?;
                if scalar.dtype != DType::F32 {
                    resources.scratch.push(scratch(
                        format!("optimizer_scalar_{index}_f32"),
                        &scalar.shape,
                        DType::F32,
                    ));
                }
            }
        }
        MetalOp::AdamW { .. } | MetalOp::AdamWGroup { .. } | MetalOp::Sgd { .. } => {}
        MetalOp::KvAttention { layer, .. } => {
            let q = input(0)?;
            let k = input(1)?;
            if q.shape.len() < 3 || q.dtype != DType::F32 {
                return Err(
                    "compile: paged KV attention requires rank >= 3 f32 queries".to_string()
                );
            }
            let rank = q.shape.len();
            let plan = KvAttentionPlan {
                batch: q.shape[..rank - 3]
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .ok_or_else(|| "compile: KV attention batch size overflow".to_string())?,
                query_heads: q.shape[rank - 3],
                kv_heads: k.shape[rank - 3],
                time: q.shape[rank - 2],
                head_dim: q.shape[rank - 1],
            };
            if plan.time == 0 || plan.head_dim > 128 {
                return Err(format!(
                    "compile: paged KV attention does not support time {} and head dimension {}",
                    plan.time, plan.head_dim
                ));
            }
            let schema = state_schema.ok_or_else(|| {
                "compile: paged KV attention requires an explicit state schema".to_string()
            })?;
            if plan.batch != schema.batch
                || plan.kv_heads != schema.kv_heads
                || plan.head_dim != schema.head_dim
                || (*layer as usize) >= schema.layers
            {
                return Err(format!(
                    "compile: KV attention geometry [{}, query heads {}, KV heads {}, {}, layer {}] does not match schema [{}, KV heads {}, {}, layers {}]",
                    plan.batch,
                    plan.query_heads,
                    plan.kv_heads,
                    plan.head_dim,
                    layer,
                    schema.batch,
                    schema.kv_heads,
                    schema.head_dim,
                    schema.layers
                ));
            }
            resources.staging.extend([
                staging(
                    "kv_block_table",
                    &[schema.batch, schema.max_blocks()],
                    DType::U32,
                ),
                staging("kv_context_lengths", &[schema.batch], DType::U32),
                staging("kv_block_bases", &[schema.batch], DType::U32),
                staging("kv_token_advances", &[schema.batch], DType::U32),
                staging("kv_padding", &[schema.batch], DType::U32),
            ]);
            resources.plan = MetalCommandPlan::KvAttention(plan);
        }
        MetalOp::Conv1d {
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = input(0)?;
            let weight = input(1)?;
            resources.plan = MetalCommandPlan::Conv(crate::conv::conv1d_requirements(
                &effect_torch_runtime::Layout::contiguous(x.shape.to_vec()),
                x.dtype,
                &effect_torch_runtime::Layout::contiguous(weight.shape.to_vec()),
                weight.dtype,
                *stride,
                *padding,
                *dilation,
                *groups,
            )?);
        }
        MetalOp::Conv2d {
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = input(0)?;
            let weight = input(1)?;
            resources.plan = MetalCommandPlan::Conv(crate::conv::conv2d_requirements(
                &effect_torch_runtime::Layout::contiguous(x.shape.to_vec()),
                x.dtype,
                &effect_torch_runtime::Layout::contiguous(weight.shape.to_vec()),
                weight.dtype,
                *stride,
                *padding,
                *dilation,
                *groups,
            )?);
        }
        MetalOp::ConvTranspose1d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => {
            let x = input(0)?;
            let weight = input(1)?;
            resources.plan = MetalCommandPlan::Conv(crate::conv::conv_transpose1d_requirements(
                &effect_torch_runtime::Layout::contiguous(x.shape.to_vec()),
                x.dtype,
                &effect_torch_runtime::Layout::contiguous(weight.shape.to_vec()),
                weight.dtype,
                *stride,
                *padding,
                *output_padding,
                *dilation,
                *groups,
            )?);
        }
        MetalOp::ConvTranspose2d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => {
            let x = input(0)?;
            let weight = input(1)?;
            resources.plan = MetalCommandPlan::Conv(crate::conv::conv_transpose2d_requirements(
                &effect_torch_runtime::Layout::contiguous(x.shape.to_vec()),
                x.dtype,
                &effect_torch_runtime::Layout::contiguous(weight.shape.to_vec()),
                weight.dtype,
                *stride,
                *padding,
                *output_padding,
                *dilation,
                *groups,
            )?);
        }
        MetalOp::Conv1dBackwardW {
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = input(0)?;
            let gradient = input(1)?;
            resources.plan = MetalCommandPlan::Conv(crate::conv::conv1d_backward_w_requirements(
                &effect_torch_runtime::Layout::contiguous(x.shape.to_vec()),
                x.dtype,
                &effect_torch_runtime::Layout::contiguous(gradient.shape.to_vec()),
                gradient.dtype,
                *kernel,
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
            )?);
        }
        MetalOp::Conv2dBackwardW {
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = input(0)?;
            let gradient = input(1)?;
            resources.plan = MetalCommandPlan::Conv(crate::conv::conv2d_backward_w_requirements(
                &effect_torch_runtime::Layout::contiguous(x.shape.to_vec()),
                x.dtype,
                &effect_torch_runtime::Layout::contiguous(gradient.shape.to_vec()),
                gradient.dtype,
                *kernel,
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
            )?);
        }
        MetalOp::Reshape { .. }
        | MetalOp::Permute { .. }
        | MetalOp::Slice { .. }
        | MetalOp::BroadcastTo { .. }
        | MetalOp::PositionEmbedding { .. }
        | MetalOp::LastTokenRow
        | MetalOp::FusedElementwise { .. }
        | MetalOp::FusedReduce { .. } => {}
    }
    Ok(resources)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MetalEnvironment {
    pub private_intermediates: bool,
    pub mma: bool,
}

impl From<EnvironmentOptions> for MetalEnvironment {
    fn from(environment: EnvironmentOptions) -> Self {
        Self {
            private_intermediates: environment.metal_private_intermediates,
            mma: environment.metal_mma,
        }
    }
}

pub(super) struct MetalExecutable {
    pub signature: ProgramSignature,
    pub program: Arc<LoweredProgram<MetalInstruction, NativeMemorySpace, MetalLoweredValue>>,
    pub physical: Box<[MetalPhysicalCommand]>,
    pub prepared: MetalPreparedArtifacts,
    pub bindings: Box<[MetalBinding]>,
    scalar_bindings: Box<[MetalScalarBinding]>,
    padded_bindings: Box<[MetalPaddedBinding]>,
    constants: Box<[MetalConstant]>,
    pub options: CompileOptions,
    pub environment: MetalEnvironment,
    pub state_schema: Option<KvStateSchema>,
    pub memory: MemoryPlan<NativeMemorySpace>,
    pub diagnostics: ExecutableDiagnostics,
    pub compiler_work: CompilerWorkReport,
    pub last_invocation_memory: Mutex<Option<InvocationMemoryReport>>,
    state_cursor: Option<ValueId>,
}

fn program_instruction<'a>(
    program: &'a LoweredProgram<MetalInstruction, NativeMemorySpace, MetalLoweredValue>,
    id: InstructionId,
) -> Option<&'a MetalCommand> {
    program
        .instructions
        .get(id.index())
        .filter(|instruction| instruction.id == id)
}

impl fmt::Debug for MetalExecutable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalExecutable")
            .field("signature", &self.signature)
            .field("program", &self.program)
            .field("physical", &self.physical)
            .field("prepared", &self.prepared)
            .field("bindings", &self.bindings)
            .field("scalar_bindings", &self.scalar_bindings)
            .field("padded_bindings", &self.padded_bindings)
            .field("constant_count", &self.constants.len())
            .field("options", &self.options)
            .field("environment", &self.environment)
            .field("state_schema", &self.state_schema)
            .field("memory", &self.memory)
            .field("diagnostics", &self.diagnostics)
            .field("compiler_work", &self.compiler_work)
            .field("state_cursor", &self.state_cursor)
            .field(
                "last_invocation_memory",
                &self
                    .last_invocation_memory
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()),
            )
            .finish()
    }
}

impl MetalExecutable {
    pub(super) fn instruction(&self, id: InstructionId) -> Option<&MetalCommand> {
        program_instruction(&self.program, id)
    }

    #[cfg(test)]
    pub(super) fn commands(&self) -> Vec<&MetalCommand> {
        self.physical
            .iter()
            .filter_map(|physical| match *physical {
                MetalPhysicalCommand::Encode(id) => self.instruction(id),
                MetalPhysicalCommand::StatusGate(_)
                | MetalPhysicalCommand::Commit
                | MetalPhysicalCommand::Complete => None,
            })
            .collect()
    }
}

pub(super) struct MetalCompilation {
    pub executable: Arc<MetalExecutable>,
    pub slots: Vec<ProgramSlot>,
    pub generated_bindings: Vec<Value>,
    pub generated_order: Vec<usize>,
}

struct Lowerer<'a> {
    values: Vec<MetalValueMetadata>,
    storage: Vec<MetalValueStorage>,
    names: Vec<String>,
    bindings: Vec<MetalBinding>,
    scalar_bindings: Vec<MetalScalarBinding>,
    padded_bindings: Vec<MetalPaddedBinding>,
    declared_sources: HashMap<u32, MetalDeclaredSource>,
    constants: Vec<MetalConstant>,
    instructions: Vec<MetalCommand>,
    node_values: HashMap<u64, Box<[ValueId]>>,
    declared_values: HashMap<u32, ValueId>,
    generated: Vec<Value>,
    prepared_generated: &'a [Value],
    generated_by_node: HashMap<u64, usize>,
    generated_order: Vec<usize>,
    ce_chunk_size: usize,
    options: CompileOptions,
    environment: MetalEnvironment,
    state_schema: Option<KvStateSchema>,
    state_cursor: Option<ValueId>,
    padded_slot: Option<u32>,
    optimizer_scalar_packs: HashMap<Vec<ValueId>, ValueId>,
}

impl<'a> Lowerer<'a> {
    fn new(
        index: &GraphIndex,
        prepared_generated: &'a [Value],
        options: CompileOptions,
        ce_chunk_size: usize,
        environment: MetalEnvironment,
        state_schema: Option<KvStateSchema>,
        slots: &[ProgramSlot],
    ) -> Self {
        let padded_slot = state_schema.and_then(|schema| {
            slots
                .iter()
                .enumerate()
                .rev()
                .find(|(slot, declaration)| {
                    !declaration.scalar && *slot as u32 != schema.cursor_slot
                })
                .map(|(slot, _)| slot as u32)
        });
        let mut tensor = 0u32;
        let mut scalar = 0u32;
        let declared_sources = slots
            .iter()
            .enumerate()
            .map(|(slot, declaration)| {
                let slot = u32::try_from(slot).expect("program slots fit u32");
                let source = if state_schema.is_some_and(|schema| schema.cursor_slot == slot) {
                    MetalDeclaredSource::StateCursor
                } else if declaration.scalar {
                    let source = MetalDeclaredSource::Scalar(scalar);
                    scalar = scalar
                        .checked_add(1)
                        .expect("scalar binding count overflow");
                    source
                } else {
                    let source = MetalDeclaredSource::Tensor(tensor);
                    tensor = tensor
                        .checked_add(1)
                        .expect("tensor binding count overflow");
                    source
                };
                (slot, source)
            })
            .collect();
        Self {
            values: Vec::new(),
            storage: Vec::new(),
            names: Vec::new(),
            bindings: Vec::new(),
            scalar_bindings: Vec::new(),
            padded_bindings: Vec::new(),
            declared_sources,
            constants: Vec::new(),
            instructions: Vec::new(),
            node_values: HashMap::new(),
            declared_values: HashMap::new(),
            generated: Vec::new(),
            prepared_generated,
            generated_by_node: index
                .leaves
                .iter()
                .enumerate()
                .map(|(position, binding)| (binding.node_id, position))
                .collect(),
            generated_order: Vec::new(),
            ce_chunk_size,
            options,
            environment,
            state_schema,
            state_cursor: None,
            padded_slot,
            optimizer_scalar_packs: HashMap::new(),
        }
    }

    fn value(
        &mut self,
        shape: &[usize],
        dtype: DType,
        storage: MetalValueStorage,
        name: impl Into<String>,
    ) -> Result<ValueId, String> {
        shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| "compile: value byte size overflow".to_string())?;
        let id = ValueId::from_index(self.values.len())
            .ok_or_else(|| "compile: too many Metal values".to_string())?;
        self.values.push(MetalValueMetadata {
            shape: shape.to_vec().into_boxed_slice(),
            dtype,
        });
        self.storage.push(storage);
        self.names.push(name.into());
        Ok(id)
    }

    fn dynamic_value(
        &mut self,
        shape: &[usize],
        dtype: DType,
        name: &str,
    ) -> Result<ValueId, String> {
        self.value(shape, dtype, MetalValueStorage::Dynamic, name)
    }

    fn alias_value(
        &mut self,
        source: ValueId,
        shape: &[usize],
        name: impl Into<String>,
    ) -> Result<ValueId, String> {
        let source_decl = self
            .values
            .get(source.index())
            .ok_or_else(|| format!("compile: alias source {source} is missing"))?;
        let source_elements = source_decl
            .shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
        let target_elements = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
        if source_elements.is_none() || source_elements != target_elements {
            return Err("compile: reshape alias element count mismatch".to_string());
        }
        self.value(
            shape,
            source_decl.dtype,
            MetalValueStorage::Alias {
                source,
                byte_offset: 0,
            },
            name,
        )
    }

    fn child_value(&self, child: &Arc<Node>) -> Result<ValueId, String> {
        self.node_values
            .get(&child.id)
            .and_then(|values| values.first())
            .copied()
            .ok_or_else(|| format!("compile: child {} was not lowered", child.id))
    }

    fn dense_value(&self, index: &GraphIndex, node: DenseNodeId) -> Result<ValueId, String> {
        let node = index
            .node(node)
            .ok_or_else(|| format!("compile: dense node {node} is out of range"))?;
        self.child_value(node)
    }

    fn dense_values(
        &self,
        index: &GraphIndex,
        nodes: &[DenseNodeId],
    ) -> Result<Vec<ValueId>, String> {
        nodes
            .iter()
            .map(|node| self.dense_value(index, *node))
            .collect()
    }

    fn child_output(&self, child: &Arc<Node>, index: usize) -> Result<ValueId, String> {
        self.node_values
            .get(&child.id)
            .and_then(|values| values.get(index))
            .copied()
            .ok_or_else(|| format!("compile: child {} has no output {index}", child.id))
    }

    fn command(
        &mut self,
        op: MetalOp,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> Result<(), String> {
        let id = InstructionId::from_index(self.instructions.len())
            .ok_or_else(|| "compile: too many Metal instructions".to_string())?;
        self.instructions.push(
            LoweredInstruction::new(
                id,
                MetalInstruction::Operation {
                    op,
                    plan: MetalCommandPlan::Direct,
                    release: Box::new([]),
                    random_seed_token: 0,
                },
                inputs.into_iter().map(ValueUse::read).collect::<Vec<_>>(),
                outputs.into_iter().map(OutputDecl::new).collect::<Vec<_>>(),
            )
            .with_effects(InstructionEffects {
                may_fail: true,
                has_side_effects: false,
            }),
        );
        Ok(())
    }

    fn operation_command(
        &mut self,
        op: MetalOp,
        mut inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> Result<(), String> {
        match &op {
            MetalOp::AdamW { .. } => {
                if inputs.len() != 7 {
                    return Err(
                        "compile: AdamW requires four tensors and three scalars".to_string()
                    );
                }
                let packed = self.optimizer_scalar_pack(&inputs[4..7])?;
                inputs.truncate(4);
                inputs.push(packed);
            }
            MetalOp::AdamWGroup { parameters, .. } => {
                let scalar_start = parameters
                    .checked_mul(4)
                    .ok_or_else(|| "compile: grouped AdamW input count overflow".to_string())?;
                if inputs.len() != scalar_start + 3 {
                    return Err(
                        "compile: grouped AdamW requires four tensors per parameter and three scalars"
                            .to_string(),
                    );
                }
                let packed = self.optimizer_scalar_pack(&inputs[scalar_start..])?;
                inputs.truncate(scalar_start);
                inputs.push(packed);
            }
            MetalOp::Sgd { .. } => {
                if inputs.len() != 5 {
                    return Err("compile: SGD requires three tensors and two scalars".to_string());
                }
                let packed = self.optimizer_scalar_pack(&[inputs[4], inputs[3]])?;
                inputs.truncate(3);
                inputs.push(packed);
            }
            _ => {}
        }
        self.command(op, inputs, outputs)
    }

    fn optimizer_scalar_pack(&mut self, scalars: &[ValueId]) -> Result<ValueId, String> {
        if let Some(value) = self.optimizer_scalar_packs.get(scalars).copied() {
            return Ok(value);
        }
        let packed = self.dynamic_value(
            &[scalars.len()],
            DType::F32,
            &format!(
                "optimizer_scalar_pack_{}",
                self.optimizer_scalar_packs.len()
            ),
        )?;
        self.command(
            MetalOp::PackOptimizerScalars,
            scalars.to_vec(),
            vec![packed],
        )?;
        self.optimizer_scalar_packs.insert(scalars.to_vec(), packed);
        Ok(packed)
    }

    fn constant(&mut self, node: &Arc<Node>, payload: Value) -> Result<(), String> {
        let value = self.value(
            &node.shape,
            node.dtype,
            MetalValueStorage::Constant,
            "constant",
        )?;
        self.constants.push(MetalConstant { value, payload });
        self.node_values.insert(node.id, Box::new([value]));
        Ok(())
    }

    fn selector(&mut self, node: &Arc<Node>, of: &Arc<Node>, index: usize) -> Result<(), String> {
        let value = self.child_output(of, index).map_err(|_| {
            format!(
                "compile: selector {} references missing output {index} of {}",
                node.id, of.id
            )
        })?;
        let declaration = &self.values[value.index()];
        if declaration.shape.as_ref() != node.shape || declaration.dtype != node.dtype {
            return Err(format!(
                "compile: selector {} metadata does not match producer output {index}",
                node.id
            ));
        }
        self.node_values.insert(node.id, Box::new([value]));
        Ok(())
    }

    fn bind_region_outputs(
        &mut self,
        index: &GraphIndex,
        plan: &OptimizationPlan,
        region_id: RegionId,
        region: &NativeRegion,
        outputs: &[ValueId],
    ) -> Result<(), String> {
        for output in region.semantic_outputs() {
            let route = plan
                .outputs
                .get(output.semantic_node.index())
                .copied()
                .flatten()
                .ok_or_else(|| {
                    format!(
                        "compile: region {region_id} semantic output {} has no physical route",
                        output.semantic_node
                    )
                })?;
            if route.region != region_id || route.index != output.index {
                return Err(format!(
                    "compile: region {region_id} semantic output {} has an inconsistent route",
                    output.semantic_node
                ));
            }
            let semantic = index.node(output.semantic_node).ok_or_else(|| {
                format!(
                    "compile: optimization output {} is out of range",
                    output.semantic_node
                )
            })?;
            let value = outputs.get(output.index as usize).copied().ok_or_else(|| {
                format!(
                    "compile: region {region_id} has no physical output {} for semantic node {}",
                    output.index, semantic.id
                )
            })?;
            let declaration = &self.values[value.index()];
            if declaration.shape.as_ref() != semantic.shape || declaration.dtype != semantic.dtype {
                return Err(format!(
                    "compile: region {region_id} output {} metadata does not match semantic node {}",
                    output.index, semantic.id
                ));
            }
            match self.node_values.insert(semantic.id, Box::new([value])) {
                Some(previous) if previous.as_ref() != [value] => {
                    return Err(format!(
                        "compile: semantic node {} was routed to multiple physical values",
                        semantic.id
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn region_outputs(
        &mut self,
        region: RegionId,
        count: usize,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Vec<ValueId>, String> {
        (0..count)
            .map(|output| {
                self.dynamic_value(shape, dtype, &format!("region_{region}_output_{output}"))
            })
            .collect()
    }

    fn lower_region(
        &mut self,
        index: &GraphIndex,
        plan: &OptimizationPlan,
        region_id: RegionId,
    ) -> Result<(), String> {
        let region = plan
            .regions
            .get(region_id.index())
            .ok_or_else(|| format!("compile: optimization region {region_id} is out of range"))?;
        let (op, inputs, outputs) = match region {
            NativeRegion::Elementwise(region) => {
                if !region.device.is_metal() {
                    return Err(format!(
                        "compile: impossible Metal optimization plan contains an elementwise region for {}",
                        region.device.name()
                    ));
                }
                if region.inputs.is_empty() {
                    return Err(
                        "compile: fused Metal operation requires at least one input".to_string()
                    );
                }
                if !matches!(region.dtype, DType::F32 | DType::BF16) {
                    return Err(format!(
                        "compile: fused operation does not support Metal dtype {}",
                        region.dtype
                    ));
                }
                let inputs = self.dense_values(index, &region.inputs)?;
                let outputs = self.region_outputs(region_id, 1, &region.shape, region.dtype)?;
                (
                    MetalOp::FusedElementwise {
                        strides: region
                            .lane_strides
                            .iter()
                            .map(|strides| strides.to_vec())
                            .collect(),
                        shape: region.shape.clone(),
                        exprs: vec![region.output.expression.clone()].into_boxed_slice(),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::ElementwiseReduce(region) => {
                if !region.device.is_metal() {
                    return Err(format!(
                        "compile: impossible Metal optimization plan contains an elementwise-reduce region for {}",
                        region.device.name()
                    ));
                }
                if region.inputs.is_empty() {
                    return Err(
                        "compile: fused Metal operation requires at least one input".to_string()
                    );
                }
                if !matches!(region.dtype, DType::F32 | DType::BF16) {
                    return Err(format!(
                        "compile: fused operation does not support Metal dtype {}",
                        region.dtype
                    ));
                }
                let inputs = self.dense_values(index, &region.inputs)?;
                let outputs = self.region_outputs(region_id, 1, &region.shape, region.dtype)?;
                (
                    MetalOp::FusedReduce {
                        strides: region
                            .lane_strides
                            .iter()
                            .map(|strides| strides.to_vec())
                            .collect(),
                        in_shape: region.input_shape.clone(),
                        expr: region.expression.clone(),
                        op: region.op,
                        dims: region.dims.clone(),
                        keepdims: region.keepdims,
                        shape: region.shape.clone(),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::MultiOutput(region) => {
                if !region.device.is_metal() {
                    return Err(format!(
                        "compile: impossible Metal optimization plan contains a multi-output region for {}",
                        region.device.name()
                    ));
                }
                if region.inputs.is_empty() {
                    return Err(
                        "compile: fused Metal operation requires at least one input".to_string()
                    );
                }
                if !matches!(region.dtype, DType::F32 | DType::BF16) {
                    return Err(format!(
                        "compile: fused operation does not support Metal dtype {}",
                        region.dtype
                    ));
                }
                let inputs = self.dense_values(index, &region.inputs)?;
                let outputs = self.region_outputs(
                    region_id,
                    region.outputs.len(),
                    &region.shape,
                    region.dtype,
                )?;
                (
                    MetalOp::FusedElementwise {
                        strides: region
                            .lane_strides
                            .iter()
                            .map(|strides| strides.to_vec())
                            .collect(),
                        shape: region.shape.clone(),
                        exprs: region
                            .outputs
                            .iter()
                            .map(|output| output.expression.clone())
                            .collect(),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::LinearResidual(region) => {
                if !region.device.is_metal() {
                    return Err(format!(
                        "compile: impossible Metal optimization plan contains a linear-residual region for {}",
                        region.device.name()
                    ));
                }
                let inputs = self.dense_values(index, &region.inputs)?;
                let outputs = self.region_outputs(region_id, 1, &region.shape, region.dtype)?;
                (
                    MetalOp::LinearResidual {
                        implementation: MetalImplementation::Native,
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::LinearGelu(region) => {
                if !region.device.is_metal() {
                    return Err(format!(
                        "compile: impossible Metal optimization plan contains a linear-GELU region for {}",
                        region.device.name()
                    ));
                }
                let inputs = self.dense_values(index, &region.inputs)?;
                let outputs = self.region_outputs(
                    region_id,
                    1 + usize::from(region.dual),
                    &region.shape,
                    region.dtype,
                )?;
                (
                    MetalOp::LinearGelu {
                        approximate: region.approximate,
                        dual: region.dual,
                        implementation: MetalImplementation::Native,
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::AdamW(region) => {
                if !region.device.is_metal() {
                    return Err(format!(
                        "compile: impossible Metal optimization plan contains an AdamW region for {}",
                        region.device.name()
                    ));
                }
                let inputs = self.dense_values(index, &region.inputs)?;
                let outputs = self.region_outputs(
                    region_id,
                    region.expressions.len(),
                    &region.shape,
                    region.dtype,
                )?;
                (
                    MetalOp::AdamW {
                        beta1: region.options.beta1,
                        beta2: region.options.beta2,
                        eps: region.options.eps,
                        weight_decay: region.options.weight_decay,
                        implementation: OptimizerImplementation::Fused,
                        exprs: region.expressions.clone(),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::AdamWGroup(region) => {
                if !region.device.is_metal() {
                    return Err(format!(
                        "compile: impossible Metal optimization plan contains an AdamW-group region for {}",
                        region.device.name()
                    ));
                }
                if region.parameter_inputs.is_empty() {
                    return Err(
                        "compile: grouped AdamW requires at least one parameter".to_string()
                    );
                }
                let inputs = self.dense_values(index, &region.inputs)?;
                let outputs = self.region_outputs(
                    region_id,
                    region.expressions.len(),
                    &region.shape,
                    region.dtype,
                )?;
                (
                    MetalOp::AdamWGroup {
                        parameters: region.parameter_inputs.len(),
                        exprs: region.expressions.clone(),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::Sgd(region) => {
                if !region.device.is_metal() {
                    return Err(format!(
                        "compile: impossible Metal optimization plan contains an SGD region for {}",
                        region.device.name()
                    ));
                }
                // MetalOp::Sgd retains the legacy [param, grad, velocity, first, lr] input ABI.
                let dense_inputs = [
                    region.tensor_inputs[0],
                    region.tensor_inputs[1],
                    region.tensor_inputs[2],
                    region.scalar_inputs[1],
                    region.scalar_inputs[0],
                ];
                let inputs = self.dense_values(index, &dense_inputs)?;
                let outputs = self.region_outputs(
                    region_id,
                    region.expressions.len(),
                    &region.shape,
                    region.dtype,
                )?;
                (
                    MetalOp::Sgd {
                        momentum: region.options.momentum,
                        dampening: region.options.dampening,
                        nesterov: region.options.nesterov,
                        weight_decay: region.options.weight_decay,
                        implementation: OptimizerImplementation::Fused,
                        exprs: region.expressions.clone(),
                    },
                    inputs,
                    outputs,
                )
            }
        };
        self.operation_command(op, inputs, outputs.clone())?;
        self.bind_region_outputs(index, plan, region_id, region, &outputs)
    }

    fn lower(&mut self, node: &Arc<Node>) -> Result<(), String> {
        if !node.device.is_metal() {
            return Err(format!(
                "compile: Metal lowering does not support device {}",
                node.device.name()
            ));
        }
        node.shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .and_then(|elements| elements.checked_mul(node.dtype.size_in_bytes()))
            .ok_or_else(|| {
                format!(
                    "compile: node {} has an unsupported overflowing layout",
                    node.id
                )
            })?;
        validate_metal_support(node)?;

        match &node.kind {
            NodeKind::Leaf(_) => {
                let semantic_position =
                    self.generated_by_node
                        .get(&node.id)
                        .copied()
                        .ok_or_else(|| {
                            format!(
                                "compile: generated binding {} is missing from the prepared index",
                                node.id
                            )
                        })?;
                let payload = self
                    .prepared_generated
                    .get(semantic_position)
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "compile: generated binding {} has no prepared value",
                            node.id
                        )
                    })?;
                if self.options.constant_weights() {
                    let tensor = payload.as_metal()?;
                    device::MetalDevice::get().synchronize_buffer_producer(&tensor.buffer)?;
                    self.constant(node, payload)?;
                    return Ok(());
                }
                let generated = u32::try_from(self.generated.len())
                    .map_err(|_| "compile: too many generated bindings".to_string())?;
                let value = self.value(
                    &node.shape,
                    node.dtype,
                    MetalValueStorage::External,
                    format!("generated_binding_{generated}"),
                )?;
                self.generated.push(payload);
                self.generated_order.push(semantic_position);
                self.bindings.push(MetalBinding {
                    value,
                    source: MetalBindingSource::Generated(generated),
                });
                self.node_values.insert(node.id, Box::new([value]));
                return Ok(());
            }
            NodeKind::Input { slot, .. } | NodeKind::ScalarInput { slot, .. } => {
                let value = match self.declared_values.get(slot).copied() {
                    Some(value) => value,
                    None => {
                        let source = self.declared_sources.get(slot).copied().ok_or_else(|| {
                            format!("compile: slot {slot} has no invocation declaration")
                        })?;
                        let is_state_cursor = source == MetalDeclaredSource::StateCursor;
                        if is_state_cursor {
                            let schema = self.state_schema.expect("state cursor has a schema");
                            let expected_shape = if schema.cursor_tensor {
                                vec![schema.batch]
                            } else {
                                Vec::new()
                            };
                            if node.dtype != DType::I64 || node.shape != expected_shape {
                                return Err(format!(
                                    "compile: cursor slot {slot} must be i64 with shape {expected_shape:?}"
                                ));
                            }
                        }
                        let padded = self.padded_slot == Some(*slot)
                            && matches!(source, MetalDeclaredSource::Tensor(_))
                            && self
                                .state_schema
                                .is_some_and(|schema| node.shape.first() == Some(&schema.batch));
                        let value = self.value(
                            &node.shape,
                            node.dtype,
                            if matches!(
                                source,
                                MetalDeclaredSource::StateCursor | MetalDeclaredSource::Scalar(_)
                            ) || padded
                            {
                                MetalValueStorage::InvocationStaging
                            } else {
                                MetalValueStorage::External
                            },
                            if is_state_cursor {
                                "decode_cursor".to_string()
                            } else {
                                format!("binding_{slot}")
                            },
                        )?;
                        self.declared_values.insert(*slot, value);
                        if is_state_cursor {
                            self.state_cursor = Some(value);
                        } else if let MetalDeclaredSource::Tensor(source) = source {
                            if padded {
                                self.padded_bindings
                                    .push(MetalPaddedBinding { value, source });
                            } else {
                                self.bindings.push(MetalBinding {
                                    value,
                                    source: MetalBindingSource::Declared(source),
                                });
                            }
                        } else if let MetalDeclaredSource::Scalar(source) = source {
                            self.scalar_bindings
                                .push(MetalScalarBinding { value, source });
                        }
                        value
                    }
                };
                let declaration = &self.values[value.index()];
                if declaration.shape.as_ref() != node.shape || declaration.dtype != node.dtype {
                    return Err(format!(
                        "compile: slot {slot} is used with conflicting signatures"
                    ));
                }
                self.node_values.insert(node.id, Box::new([value]));
                return Ok(());
            }
            NodeKind::FromBytes {
                data, shape, dtype, ..
            } => {
                return self.constant(node, crate::value::value_from_bytes(data, shape, *dtype)?);
            }
            NodeKind::Zeros { shape, dtype, .. } => {
                return self.constant(node, Value(metal_ops::fill(shape, 0.0, *dtype)?));
            }
            NodeKind::Ones { shape, dtype, .. } => {
                return self.constant(node, Value(metal_ops::fill(shape, 1.0, *dtype)?));
            }
            NodeKind::Full {
                shape,
                value,
                dtype,
                ..
            } => {
                return self.constant(node, Value(metal_ops::fill(shape, *value, *dtype)?));
            }
            NodeKind::Arange {
                start,
                end,
                step,
                dtype,
                ..
            } => {
                return self.constant(node, Value(metal_ops::arange(*start, *end, *step, *dtype)?));
            }
            NodeKind::Eye { n, dtype, .. } => {
                return self.constant(node, Value(metal_ops::eye(*n, *dtype)?));
            }
            NodeKind::SdpaBackwardOut { of, index }
            | NodeKind::ChunkedHeadCeBackwardOut { of, index }
            | NodeKind::KdaBackwardOut { of, index }
            | NodeKind::AdamWOut { step: of, index }
            | NodeKind::SgdOut { step: of, index } => {
                return self.selector(node, of, *index as usize);
            }
            NodeKind::LayerNormBackwardOut { of, index } => {
                return self.selector(node, of, *index as usize);
            }
            NodeKind::StopGradient { a } | NodeKind::Checkpoint { a } => {
                let value = self.child_value(a)?;
                self.node_values.insert(node.id, Box::new([value]));
                return Ok(());
            }
            NodeKind::Reshape { a, shape } => {
                let source = self.child_value(a)?;
                let value = self.alias_value(source, shape, format!("{}_reshape", node.id))?;
                self.node_values.insert(node.id, Box::new([value]));
                return Ok(());
            }
            _ => {}
        }

        let mut inputs = node_children(&node.kind)
            .iter()
            .map(|child| self.child_value(child))
            .collect::<Result<Vec<_>, _>>()?;
        let native_sdpa = |dtype| matches!(dtype, DType::F32 | DType::F16 | DType::BF16);
        if let NodeKind::SdpaBackward { fwd, q, .. } = &node.kind {
            if native_sdpa(q.dtype) {
                inputs.push(self.child_output(fwd, 1)?);
            }
        }

        let output_metadata: Vec<(Vec<usize>, DType)> = match &node.kind {
            NodeKind::ChunkedHeadCeBackward {
                x, weight, bias, ..
            } => vec![
                (x.shape.clone(), x.dtype),
                (weight.shape.clone(), weight.dtype),
                (bias.shape.clone(), bias.dtype),
            ],
            NodeKind::Sdpa { q, .. } if native_sdpa(q.dtype) => {
                let mut l_shape = q.shape[..q.shape.len() - 1].to_vec();
                vec![
                    (node.shape.clone(), node.dtype),
                    (std::mem::take(&mut l_shape), DType::F32),
                ]
            }
            NodeKind::SdpaBackward { q, k, v, .. } => vec![
                (q.shape.clone(), q.dtype),
                (k.shape.clone(), k.dtype),
                (v.shape.clone(), v.dtype),
            ],
            NodeKind::KdaBackward {
                q,
                k,
                v,
                log_decay,
                beta,
                ..
            } => vec![
                (q.shape.clone(), q.dtype),
                (k.shape.clone(), k.dtype),
                (v.shape.clone(), v.dtype),
                (log_decay.shape.clone(), log_decay.dtype),
                (beta.shape.clone(), beta.dtype),
            ],
            NodeKind::LayerNormBackward { x, weight, .. } => vec![
                (x.shape.clone(), x.dtype),
                (weight.shape.clone(), weight.dtype),
                (weight.shape.clone(), weight.dtype),
            ],
            NodeKind::AdamWStep { param, m, v, .. } => vec![
                (param.shape.clone(), param.dtype),
                (m.shape.clone(), m.dtype),
                (v.shape.clone(), v.dtype),
            ],
            NodeKind::SgdStep {
                param, velocity, ..
            } => vec![
                (param.shape.clone(), param.dtype),
                (velocity.shape.clone(), velocity.dtype),
            ],
            _ => vec![(node.shape.clone(), node.dtype)],
        };
        let outputs = output_metadata
            .iter()
            .enumerate()
            .map(|(index, (shape, dtype))| {
                self.dynamic_value(shape, *dtype, &format!("{}_output_{index}", node.id))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let implementation = |_dtype: DType| MetalImplementation::Native;
        let optimizer = |dtype: DType| {
            if self.options.optimize
                && self.options.environment.fusion
                && matches!(dtype, DType::F32 | DType::BF16)
            {
                OptimizerImplementation::Fused
            } else {
                OptimizerImplementation::Generic
            }
        };
        let op = match &node.kind {
            NodeKind::Randn { shape, dtype, .. } => MetalOp::Randn {
                shape: shape.clone().into_boxed_slice(),
                dtype: *dtype,
            },
            NodeKind::Uniform {
                lo,
                hi,
                shape,
                dtype,
                ..
            } => MetalOp::Uniform {
                lo: *lo,
                hi: *hi,
                shape: shape.clone().into_boxed_slice(),
                dtype: *dtype,
            },
            NodeKind::Add { .. } => MetalOp::Binary(MetalBinaryOp::Add),
            NodeKind::Sub { .. } => MetalOp::Binary(MetalBinaryOp::Sub),
            NodeKind::Mul { .. } => MetalOp::Binary(MetalBinaryOp::Mul),
            NodeKind::Div { .. } => MetalOp::Binary(MetalBinaryOp::Div),
            NodeKind::Eq { .. } => MetalOp::Binary(MetalBinaryOp::Eq),
            NodeKind::Gt { .. } => MetalOp::Binary(MetalBinaryOp::Gt),
            NodeKind::Lt { .. } => MetalOp::Binary(MetalBinaryOp::Lt),
            NodeKind::Ge { .. } => MetalOp::Binary(MetalBinaryOp::Ge),
            NodeKind::Le { .. } => MetalOp::Binary(MetalBinaryOp::Le),
            NodeKind::Maximum { .. } => MetalOp::Binary(MetalBinaryOp::Maximum),
            NodeKind::Minimum { .. } => MetalOp::Binary(MetalBinaryOp::Minimum),
            NodeKind::Neg { .. } => MetalOp::Unary(MetalUnaryOp::Neg),
            NodeKind::Abs { .. } => MetalOp::Unary(MetalUnaryOp::Abs),
            NodeKind::Sqrt { .. } => MetalOp::Unary(MetalUnaryOp::Sqrt),
            NodeKind::Exp { .. } => MetalOp::Unary(MetalUnaryOp::Exp),
            NodeKind::Log { .. } => MetalOp::Unary(MetalUnaryOp::Log),
            NodeKind::Sin { .. } => MetalOp::Unary(MetalUnaryOp::Sin),
            NodeKind::Cos { .. } => MetalOp::Unary(MetalUnaryOp::Cos),
            NodeKind::Tanh { .. } => MetalOp::Unary(MetalUnaryOp::Tanh),
            NodeKind::Relu { .. } => MetalOp::Unary(MetalUnaryOp::Relu),
            NodeKind::Erf { .. } => MetalOp::Unary(MetalUnaryOp::Erf),
            NodeKind::Gelu { approximate, .. } => MetalOp::Unary(MetalUnaryOp::Gelu {
                approximate: *approximate,
            }),
            NodeKind::Floor { .. } => MetalOp::Unary(MetalUnaryOp::Floor),
            NodeKind::Ceil { .. } => MetalOp::Unary(MetalUnaryOp::Ceil),
            NodeKind::Round { .. } => MetalOp::Unary(MetalUnaryOp::Round),
            NodeKind::Sign { .. } => MetalOp::Unary(MetalUnaryOp::Sign),
            NodeKind::Where { .. } => MetalOp::Where,
            NodeKind::Argmax { dim, .. } => MetalOp::Argmax { dim: *dim },
            NodeKind::Argmin { dim, .. } => MetalOp::Argmin { dim: *dim },
            NodeKind::Cumsum { dim, .. } => MetalOp::Cumsum { dim: *dim },
            NodeKind::ScatterAdd { dim, .. } => MetalOp::ScatterAdd { dim: *dim },
            NodeKind::Gather { dim, .. } => MetalOp::Gather { dim: *dim },
            NodeKind::IndexSelect { dim, .. } => MetalOp::IndexSelect { dim: *dim },
            NodeKind::CrossEntropy {
                logits,
                ignore_index,
                reduction,
                ..
            } => MetalOp::CrossEntropy {
                ignore_index: *ignore_index,
                reduction: *reduction,
                implementation: implementation(logits.dtype),
            },
            NodeKind::CrossEntropyBackward {
                logits,
                ignore_index,
                reduction,
                ..
            } => MetalOp::CrossEntropyBackward {
                ignore_index: *ignore_index,
                reduction: *reduction,
                implementation: implementation(logits.dtype),
            },
            NodeKind::ChunkedHeadCe { ignore_index, .. } => MetalOp::ChunkedHeadCe {
                ignore_index: *ignore_index,
                chunk_size: self.ce_chunk_size,
            },
            NodeKind::ChunkedHeadCeBackward { ignore_index, .. } => {
                MetalOp::ChunkedHeadCeBackward {
                    ignore_index: *ignore_index,
                    chunk_size: self.ce_chunk_size,
                }
            }
            NodeKind::Sdpa {
                q,
                scale,
                causal,
                window,
                ..
            } => MetalOp::Sdpa {
                scale: *scale,
                causal: *causal,
                window: window.local(),
                implementation: implementation(q.dtype),
            },
            NodeKind::SdpaBackward {
                q,
                scale,
                causal,
                window,
                ..
            } => MetalOp::SdpaBackward {
                scale: *scale,
                causal: *causal,
                window: window.local(),
                implementation: implementation(q.dtype),
            },
            NodeKind::KdaChunk { scale, .. } => MetalOp::KdaChunk { scale: *scale },
            NodeKind::KdaRecurrence { scale, layer, .. } => MetalOp::KdaRecurrence {
                scale: *scale,
                layer: *layer,
            },
            NodeKind::KdaBackward { scale, .. } => MetalOp::KdaBackward { scale: *scale },
            NodeKind::ShortConv1d { .. } => MetalOp::ShortConv1d,
            NodeKind::ShortConv1dBackwardX { .. } => MetalOp::ShortConv1dBackwardX,
            NodeKind::ShortConv1dBackwardW { .. } => MetalOp::ShortConv1dBackwardW,
            NodeKind::ConvState { layer, .. } => MetalOp::ConvState { layer: *layer },
            NodeKind::LastTokenRow { .. } => MetalOp::LastTokenRow,
            NodeKind::PositionEmbedding { seq_len, .. } => {
                MetalOp::PositionEmbedding { seq_len: *seq_len }
            }
            NodeKind::KvAttention {
                scale,
                layer,
                window,
                ..
            } => MetalOp::KvAttention {
                scale: *scale,
                layer: *layer,
                window: *window,
            },
            NodeKind::RotaryEmbedding {
                x,
                theta,
                offset,
                layout,
                ..
            } => MetalOp::RotaryEmbedding {
                theta: *theta,
                cursor_offset: matches!(offset, PositionOffset::Cursor),
                layout: *layout,
                implementation: implementation(x.dtype),
            },
            NodeKind::RotaryEmbeddingBackward {
                g, theta, layout, ..
            } => MetalOp::RotaryEmbeddingBackward {
                theta: *theta,
                layout: *layout,
                implementation: implementation(g.dtype),
            },
            NodeKind::Linear { x, .. } => MetalOp::Linear {
                implementation: implementation(x.dtype),
            },
            NodeKind::QuantizedLinear {
                codec,
                weight_shape,
                ..
            } => MetalOp::QuantizedLinear {
                codec: *codec,
                weight_shape: *weight_shape,
            },
            NodeKind::QuantizedEmbedding {
                codec,
                weight_shape,
                ..
            } => MetalOp::QuantizedEmbedding {
                codec: *codec,
                weight_shape: *weight_shape,
            },
            NodeKind::LayerNorm { x, eps, .. } => MetalOp::LayerNorm {
                eps: *eps,
                implementation: implementation(x.dtype),
            },
            NodeKind::RmsNorm { x, eps, .. } => MetalOp::RmsNorm {
                eps: *eps,
                implementation: implementation(x.dtype),
            },
            NodeKind::LayerNormBackward { x, eps, .. } => MetalOp::LayerNormBackward {
                eps: *eps,
                implementation: implementation(x.dtype),
            },
            NodeKind::Conv1d {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => MetalOp::Conv1d {
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
                groups: *groups,
            },
            NodeKind::Conv2d {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => MetalOp::Conv2d {
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
                groups: *groups,
            },
            NodeKind::ConvTranspose1d {
                stride,
                padding,
                output_padding,
                dilation,
                groups,
                ..
            } => MetalOp::ConvTranspose1d {
                stride: *stride,
                padding: *padding,
                output_padding: *output_padding,
                dilation: *dilation,
                groups: *groups,
            },
            NodeKind::ConvTranspose2d {
                stride,
                padding,
                output_padding,
                dilation,
                groups,
                ..
            } => MetalOp::ConvTranspose2d {
                stride: *stride,
                padding: *padding,
                output_padding: *output_padding,
                dilation: *dilation,
                groups: *groups,
            },
            NodeKind::Conv1dBackwardW {
                kernel,
                out_channels,
                stride,
                padding,
                dilation,
                groups,
                ..
            } => MetalOp::Conv1dBackwardW {
                kernel: *kernel,
                out_channels: *out_channels,
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
                groups: *groups,
            },
            NodeKind::Conv2dBackwardW {
                kernel,
                out_channels,
                stride,
                padding,
                dilation,
                groups,
                ..
            } => MetalOp::Conv2dBackwardW {
                kernel: *kernel,
                out_channels: *out_channels,
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
                groups: *groups,
            },
            NodeKind::Pow { exp, .. } => MetalOp::Unary(MetalUnaryOp::Pow {
                exponent_bits: exp.to_bits(),
            }),
            NodeKind::Cast { dtype, .. } => MetalOp::Unary(MetalUnaryOp::Cast { dtype: *dtype }),
            NodeKind::Sum { dims, keepdims, .. } => MetalOp::Reduce {
                op: MetalReduceOp::Sum,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Mean { dims, keepdims, .. } => MetalOp::Reduce {
                op: MetalReduceOp::Mean,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Max { dims, keepdims, .. } => MetalOp::Reduce {
                op: MetalReduceOp::Max,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Min { dims, keepdims, .. } => MetalOp::Reduce {
                op: MetalReduceOp::Min,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Prod { dims, keepdims, .. } => MetalOp::Reduce {
                op: MetalReduceOp::Prod,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Reshape { shape, .. } => MetalOp::Reshape {
                shape: shape.clone().into_boxed_slice(),
            },
            NodeKind::Permute { dims, .. } => MetalOp::Permute {
                dims: dims.clone().into_boxed_slice(),
            },
            NodeKind::Slice { ranges, .. } => MetalOp::Slice {
                ranges: ranges.clone().into_boxed_slice(),
            },
            NodeKind::Concat { dim, .. } => MetalOp::Binary(MetalBinaryOp::Concat { dim: *dim }),
            NodeKind::BroadcastTo { shape, .. } => MetalOp::BroadcastTo {
                shape: shape.clone().into_boxed_slice(),
            },
            NodeKind::Matmul { .. } => MetalOp::Binary(MetalBinaryOp::Matmul),
            NodeKind::AdamWStep {
                beta1,
                beta2,
                eps,
                weight_decay,
                param,
                ..
            } => {
                let implementation = optimizer(param.dtype);
                MetalOp::AdamW {
                    beta1: *beta1,
                    beta2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    implementation,
                    exprs: fusion::adamw_exprs(*beta1, *beta2, *eps, *weight_decay)
                        .into_iter()
                        .collect(),
                }
            }
            NodeKind::SgdStep {
                momentum,
                dampening,
                nesterov,
                weight_decay,
                param,
                ..
            } => {
                let implementation = optimizer(param.dtype);
                MetalOp::Sgd {
                    momentum: *momentum,
                    dampening: *dampening,
                    nesterov: *nesterov,
                    weight_decay: *weight_decay,
                    implementation,
                    exprs: fusion::sgd_exprs(*momentum, *dampening, *nesterov, *weight_decay)
                        .into_iter()
                        .collect(),
                }
            }
            NodeKind::Inverse { .. } | NodeKind::Det { .. } | NodeKind::Solve { .. } => {
                unreachable!("unsupported Metal linalg is rejected during validation")
            }
            NodeKind::Leaf(_)
            | NodeKind::Input { .. }
            | NodeKind::ScalarInput { .. }
            | NodeKind::FromBytes { .. }
            | NodeKind::Zeros { .. }
            | NodeKind::Ones { .. }
            | NodeKind::Full { .. }
            | NodeKind::Arange { .. }
            | NodeKind::Eye { .. }
            | NodeKind::SdpaBackwardOut { .. }
            | NodeKind::ChunkedHeadCeBackwardOut { .. }
            | NodeKind::KdaBackwardOut { .. }
            | NodeKind::LayerNormBackwardOut { .. }
            | NodeKind::AdamWOut { .. }
            | NodeKind::SgdOut { .. }
            | NodeKind::StopGradient { .. }
            | NodeKind::Checkpoint { .. } => {
                unreachable!("zero-command nodes return before operation lowering")
            }
        };
        self.operation_command(op, inputs, outputs.clone())?;
        self.node_values.insert(node.id, outputs.into_boxed_slice());
        Ok(())
    }

    fn finish(
        mut self,
        outputs: Vec<ValueId>,
        driver: &mut CompilerDriver<'_>,
    ) -> Result<(MetalExecutable, Vec<Value>, Vec<usize>), String> {
        if let Some(cursor) = self.state_cursor {
            self.instructions.insert(
                0,
                LoweredInstruction::new(
                    InstructionId::new(0),
                    MetalInstruction::Operation {
                        op: MetalOp::PrepareState,
                        plan: MetalCommandPlan::Direct,
                        release: Box::new([]),
                        random_seed_token: 0,
                    },
                    Vec::new(),
                    vec![OutputDecl::new(cursor)],
                )
                .with_effects(InstructionEffects {
                    may_fail: true,
                    has_side_effects: true,
                }),
            );
        }
        for (token, instruction) in self.instructions.iter_mut().enumerate() {
            let (_, _, _, random_seed_token) = instruction
                .kind
                .operation_mut()
                .expect("lowerer retains only operation instructions before finalization");
            *random_seed_token = token as u64;
        }
        let output_roots = outputs
            .iter()
            .copied()
            .map(|value| storage_root(&self.storage, value))
            .collect::<HashSet<_>>();
        for root in &output_roots {
            if matches!(
                self.storage[root.index()],
                MetalValueStorage::Dynamic | MetalValueStorage::InvocationStaging
            ) {
                self.storage[root.index()] = MetalValueStorage::ProvisionalOutput;
            }
        }
        let protected = outputs
            .iter()
            .copied()
            .chain(output_roots.iter().copied())
            .collect::<HashSet<_>>();
        let mut last_use = vec![None::<usize>; self.values.len()];
        for (command_index, command) in self.instructions.iter().enumerate() {
            for output in &command.outputs {
                last_use[storage_root(&self.storage, output.value).index()]
                    .get_or_insert(command_index);
            }
            for input in &command.inputs {
                last_use[storage_root(&self.storage, input.value).index()] = Some(command_index);
            }
        }
        for (command_index, command) in self.instructions.iter_mut().enumerate() {
            let (_, _, release, _) = command
                .kind
                .operation_mut()
                .expect("lowerer retains only operation instructions before finalization");
            *release = last_use
                .iter()
                .enumerate()
                .filter_map(|(index, last)| {
                    let value = ValueId::from_index(index)?;
                    (*last == Some(command_index) && !protected.contains(&value)).then_some(value)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
        }

        let planned_resources = self
            .instructions
            .iter()
            .map(|command| {
                plan_command_resources(
                    command,
                    &self.values,
                    self.environment,
                    self.state_schema.as_ref(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (command_index, mut resources) in planned_resources.into_iter().enumerate() {
            let mut append = |resource: ResourceSpec| {
                self.value(
                    &resource.shape,
                    resource.dtype,
                    resource.storage,
                    format!("{}_{}", command_index, resource.name),
                )
            };
            let scratch = resources
                .scratch
                .drain(..)
                .map(&mut append)
                .collect::<Result<Vec<_>, _>>()?;
            let staging = resources
                .staging
                .drain(..)
                .map(&mut append)
                .collect::<Result<Vec<_>, _>>()?;
            let status = resources
                .status
                .drain(..)
                .map(&mut append)
                .collect::<Result<Vec<_>, _>>()?;
            let (scratch, state): (Vec<_>, Vec<_>) = scratch.into_iter().partition(|value| {
                self.storage[value.index()] != MetalValueStorage::StateTransaction
            });
            let command = &mut self.instructions[command_index];
            command.scratch = scratch
                .into_iter()
                .map(ValueUse::read_write)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            command.staging = staging
                .into_iter()
                .map(ValueUse::read_write)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            command.status = status
                .into_iter()
                .map(ValueUse::write)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            command.state = state
                .into_iter()
                .map(ValueUse::write)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let (op, plan, _, _) = command
                .kind
                .operation_mut()
                .expect("lowerer retains only operation instructions before finalization");
            *plan = resources.plan;
            command.effects.has_side_effects = matches!(
                op,
                MetalOp::PrepareState
                    | MetalOp::KdaRecurrence { .. }
                    | MetalOp::ConvState { .. }
                    | MetalOp::KvAttention { .. }
            );
        }

        let mut constant_slots = 0u32;
        let values = self
            .values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let id = ValueId::from_index(index)
                    .ok_or_else(|| "compile: too many Metal values".to_string())?;
                let bytes = value
                    .shape
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .and_then(|elements| elements.checked_mul(value.dtype.size_in_bytes()))
                    .ok_or_else(|| "compile: value byte size overflow".to_string())?
                    .max(1);
                let storage = match self.storage[index] {
                    MetalValueStorage::External => ValueStorage::Fixed {
                        class: StorageClass::ExternalInput,
                        location: Location::External { slot: id.get() },
                    },
                    MetalValueStorage::Constant => {
                        let slot = constant_slots;
                        constant_slots = constant_slots
                            .checked_add(1)
                            .ok_or_else(|| "compile: too many constants".to_string())?;
                        ValueStorage::Fixed {
                            class: StorageClass::PersistentConstant,
                            location: Location::Persistent { slot },
                        }
                    }
                    MetalValueStorage::Dynamic => ValueStorage::Planned {
                        class: StorageClass::Workspace,
                        alignment: 256,
                        memory_space: NativeMemorySpace::MetalShared,
                        ownership: SegmentOwnership::Workspace,
                    },
                    MetalValueStorage::ProvisionalOutput => ValueStorage::Planned {
                        class: StorageClass::EscapingOutput,
                        alignment: 256,
                        memory_space: NativeMemorySpace::MetalShared,
                        ownership: SegmentOwnership::ProvisionalOutput,
                    },
                    MetalValueStorage::Alias {
                        source,
                        byte_offset,
                    } => ValueStorage::Alias {
                        source,
                        byte_offset,
                    },
                    MetalValueStorage::Workspace => ValueStorage::Planned {
                        class: StorageClass::Workspace,
                        alignment: 256,
                        memory_space: NativeMemorySpace::MetalShared,
                        ownership: SegmentOwnership::Workspace,
                    },
                    MetalValueStorage::InvocationStaging => ValueStorage::Planned {
                        class: StorageClass::Workspace,
                        alignment: 256,
                        memory_space: NativeMemorySpace::MetalShared,
                        ownership: SegmentOwnership::InvocationStaging,
                    },
                    MetalValueStorage::DeviceStatus => ValueStorage::Planned {
                        class: StorageClass::DeviceStatus,
                        alignment: 256,
                        memory_space: NativeMemorySpace::MetalShared,
                        ownership: SegmentOwnership::Workspace,
                    },
                    MetalValueStorage::StateTransaction => ValueStorage::Planned {
                        class: StorageClass::PersistentState,
                        alignment: 256,
                        memory_space: NativeMemorySpace::MetalShared,
                        ownership: SegmentOwnership::StateTransaction,
                    },
                };
                let layout = effect_torch_runtime::Layout::contiguous(value.shape.to_vec());
                Ok(MetalLoweredValue {
                    shape: value.shape.clone(),
                    dtype: value.dtype,
                    layout,
                    declaration: ValueDecl {
                        id,
                        name: self.names[index].clone(),
                        bytes,
                        storage,
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        use objc2_metal::MTLDevice;
        let device_max = usize::try_from(device::MetalDevice::get().raw().maxBufferLength())
            .map_err(|_| "compile: Metal maxBufferLength does not fit usize".to_string())?;
        let segment_max = device_max - (device_max % 256);
        if segment_max == 0 {
            return Err("compile: Metal maxBufferLength is below required alignment".to_string());
        }
        let invocation_values = self
            .scalar_bindings
            .iter()
            .map(|binding| binding.value)
            .chain(self.padded_bindings.iter().map(|binding| binding.value))
            .collect::<Vec<_>>();
        let mut instructions = Vec::with_capacity(
            self.instructions.len() + usize::from(!invocation_values.is_empty()) + 1,
        );
        if !invocation_values.is_empty() {
            instructions.push(
                LoweredInstruction::new(
                    InstructionId::from_index(0)
                        .ok_or_else(|| "compile: too many Metal instructions".to_string())?,
                    MetalInstruction::PrepareInvocation,
                    Vec::new(),
                    invocation_values
                        .iter()
                        .copied()
                        .map(OutputDecl::new)
                        .collect::<Vec<_>>(),
                )
                .with_effects(InstructionEffects {
                    may_fail: true,
                    has_side_effects: false,
                }),
            );
        }
        for mut instruction in self.instructions {
            let id = InstructionId::from_index(instructions.len())
                .ok_or_else(|| "compile: too many Metal instructions".to_string())?;
            instruction.id = id;
            instructions.push(instruction);
        }
        let mut publication_inputs = outputs
            .iter()
            .copied()
            .map(ValueUse::read)
            .collect::<Vec<_>>();
        publication_inputs.extend(invocation_values.iter().copied().map(ValueUse::read));
        publication_inputs.extend(self.state_cursor.map(ValueUse::read));
        for command in &instructions {
            publication_inputs.extend(
                command
                    .staging
                    .iter()
                    .map(|use_| ValueUse::read(use_.value)),
            );
            publication_inputs.extend(command.state.iter().map(|use_| ValueUse::read(use_.value)));
            if !command.status.is_empty() {
                publication_inputs.extend(
                    command
                        .scratch
                        .iter()
                        .chain(command.status.iter())
                        .map(|use_| ValueUse::read(use_.value)),
                );
            }
        }
        let stateful = self.state_schema.is_some()
            || instructions.iter().any(|instruction| {
                instruction.effects.has_side_effects || !instruction.state.is_empty()
            });
        instructions.push(
            LoweredInstruction::new(
                InstructionId::from_index(instructions.len())
                    .ok_or_else(|| "compile: too many Metal instructions".to_string())?,
                MetalInstruction::FinalizeInvocation,
                publication_inputs,
                Vec::new(),
            )
            .with_effects(InstructionEffects {
                may_fail: stateful,
                has_side_effects: stateful,
            }),
        );
        let program = Arc::new(LoweredProgram::new(values, instructions, outputs.clone()));
        let mut memory = driver
            .plan_memory(
                &program,
                &MemoryPlannerConfig::uniform(
                    NativeMemorySpace::MetalShared,
                    segment_max,
                    256,
                    256,
                ),
            )
            .map_err(|error| format!("compile: Metal memory planning failed: {error}"))?;
        if let Some(schema) = &self.state_schema {
            memory.report.state_bytes = schema.referenced_state_bytes()?;
        }

        let physical = driver.phase(PHYSICAL_PLANNING_PHASE, || {
            let mut physical = Vec::with_capacity(program.instructions.len() * 3 + 1);
            for instruction in &program.instructions {
                if instruction.kind.operation().is_none() {
                    continue;
                }
                physical.push(MetalPhysicalCommand::Encode(instruction.id));
                if !instruction.status.is_empty() {
                    physical.push(MetalPhysicalCommand::StatusGate(instruction.id));
                    physical.push(MetalPhysicalCommand::Commit);
                }
            }
            physical.push(MetalPhysicalCommand::Complete);
            Ok::<_, String>(physical)
        })?;

        let pipeline_count = driver.phase(PIPELINE_PREPARATION_PHASE, || {
            let mut pipeline_count = 0usize;
            for physical_command in &physical {
                let MetalPhysicalCommand::Encode(id) = *physical_command else {
                    continue;
                };
                let command = program_instruction(&program, id).ok_or_else(|| {
                    format!("compile: physical Metal instruction {id} is out of range")
                })?;
                let Some((op, command_plan)) = command.kind.operation() else {
                    return Err(format!(
                        "compile: physical Metal instruction {id} is not an operation"
                    ));
                };
                let dtype = command
                    .inputs
                    .first()
                    .map(|value| self.values[value.index()].dtype)
                    .or_else(|| {
                        command
                            .outputs
                            .first()
                            .map(|value| self.values[value.index()].dtype)
                    })
                    .unwrap_or(DType::F32);
                match op {
                    MetalOp::ChunkedHeadCe { .. } => {
                        let MetalCommandPlan::ChunkedHeadForward(plan) = command_plan else {
                            return Err("compile: chunked head CE plan is missing".to_string());
                        };
                        for variant in std::iter::once(&plan.full).chain(plan.tail.iter()) {
                            pipeline_count += crate::gemm::precompile_gemm_fused(
                                device::MetalDevice::get(),
                                &variant.head,
                            )?;
                            loss::warm_forward_exact(&variant.ce)?;
                            pipeline_count += 2;
                        }
                        crate::kernels::warm_fill(&[3], 0.0, DType::F32)?;
                        crate::ops::warm_binary(
                            &[3],
                            DType::F32,
                            &[3],
                            DType::F32,
                            metal_ops::BinOp::Add,
                            false,
                        )?;
                        crate::ops::warm_binary(
                            &[1],
                            DType::F32,
                            &[1],
                            DType::F32,
                            metal_ops::BinOp::Div,
                            false,
                        )?;
                        let destination = &self.values[command.outputs[0].index()];
                        if destination.dtype != DType::F32 {
                            crate::kernels::warm_cast(
                                &destination.shape,
                                DType::F32,
                                destination.dtype,
                            )?;
                        }
                        pipeline_count += 3;
                    }
                    MetalOp::ChunkedHeadCeBackward { .. } => {
                        let MetalCommandPlan::ChunkedHeadBackward(plan) = command_plan else {
                            return Err(
                                "compile: chunked head CE backward plan is missing".to_string()
                            );
                        };
                        let x = &self.values[command.inputs[0].index()];
                        let weight = &self.values[command.inputs[1].index()];
                        let target = &self.values[command.inputs[3].index()];
                        let gradient = &self.values[command.inputs[4].index()];
                        if gradient.dtype != DType::F32 {
                            crate::kernels::warm_cast(&gradient.shape, gradient.dtype, DType::F32)?;
                        }
                        if weight.dtype != DType::F32 {
                            crate::kernels::warm_cast(&weight.shape, weight.dtype, DType::F32)?;
                        }
                        let transposed_weight =
                            effect_torch_runtime::Layout::contiguous(weight.shape.to_vec())
                                .permute(&[1, 0]);
                        crate::kernels::warm_copy_layout(&transposed_weight, DType::F32)?;
                        crate::kernels::warm_fill(&[plan.inner, plan.vocab], 0.0, DType::F32)?;
                        crate::kernels::warm_fill(&[1, plan.vocab], 0.0, DType::F32)?;
                        loss::warm_target_status(x.dtype, target.dtype)?;
                        loss::warm_backward_scaled_f32(x.dtype, target.dtype)?;
                        crate::ops::warm_binary(
                            &[1],
                            DType::F32,
                            &[1],
                            DType::F32,
                            metal_ops::BinOp::Div,
                            false,
                        )?;
                        for variant in std::iter::once(&plan.full).chain(plan.tail.iter()) {
                            pipeline_count += crate::gemm::precompile_gemm_fused(
                                device::MetalDevice::get(),
                                &variant.head,
                            )?;
                            pipeline_count += crate::gemm::precompile_gemm_fused(
                                device::MetalDevice::get(),
                                &variant.dx,
                            )?;
                            pipeline_count += crate::gemm::precompile_gemm_fused(
                                device::MetalDevice::get(),
                                &variant.dw,
                            )?;
                            if x.dtype != DType::F32 {
                                crate::kernels::warm_cast(
                                    &[variant.rows, plan.inner],
                                    x.dtype,
                                    DType::F32,
                                )?;
                                crate::kernels::warm_cast(
                                    &[variant.rows, plan.inner],
                                    DType::F32,
                                    x.dtype,
                                )?;
                            }
                            let transposed_x = effect_torch_runtime::Layout::contiguous(vec![
                                variant.rows,
                                plan.inner,
                            ])
                            .permute(&[1, 0]);
                            crate::kernels::warm_copy_layout(&transposed_x, DType::F32)?;
                            crate::ops::warm_reduce(
                                &[variant.rows, plan.vocab],
                                DType::F32,
                                &[0],
                                true,
                                fusion::ReduceOp::Sum,
                            )?;
                        }
                        crate::ops::warm_binary(
                            &[plan.inner, plan.vocab],
                            DType::F32,
                            &[plan.inner, plan.vocab],
                            DType::F32,
                            metal_ops::BinOp::Add,
                            false,
                        )?;
                        crate::ops::warm_binary(
                            &[1, plan.vocab],
                            DType::F32,
                            &[1, plan.vocab],
                            DType::F32,
                            metal_ops::BinOp::Add,
                            false,
                        )?;
                        for output in &command.outputs[1..] {
                            let destination = &self.values[output.index()];
                            if destination.dtype == DType::F32 {
                                crate::kernels::warm_copy(&destination.shape, DType::F32)?;
                            } else {
                                crate::kernels::warm_cast(
                                    &destination.shape,
                                    DType::F32,
                                    destination.dtype,
                                )?;
                            }
                        }
                        pipeline_count += 10;
                    }
                    MetalOp::Randn { shape, .. } => {
                        crate::kernels::warm_randn(shape)?;
                        crate::kernels::warm_cast(shape, DType::F32, dtype)?;
                        pipeline_count += 1;
                    }
                    MetalOp::QuantizedLinear { .. } => {
                        let MetalCommandPlan::QuantizedLinear(requirements) = command_plan else {
                            return Err("compile: quantized linear plan is missing".to_string());
                        };
                        quantized::warm_linear_exact(requirements)?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    MetalOp::QuantizedEmbedding { .. } => {
                        let MetalCommandPlan::QuantizedEmbedding(requirements) = command_plan
                        else {
                            return Err("compile: quantized embedding plan is missing".to_string());
                        };
                        quantized::warm_embedding_exact(requirements)?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    MetalOp::Uniform { lo, hi, shape, .. } => {
                        crate::kernels::warm_uniform(*lo, *hi, shape)?;
                        crate::kernels::warm_cast(shape, DType::F32, dtype)?;
                        pipeline_count += 1;
                    }
                    MetalOp::FusedElementwise {
                        strides,
                        shape,
                        exprs,
                    } => {
                        crate::run::warm_elementwise(
                            device::MetalDevice::get(),
                            exprs,
                            strides,
                            shape,
                            shape.iter().product(),
                            0,
                            dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::FusedReduce {
                        strides,
                        in_shape,
                        expr,
                        op,
                        dims,
                        keepdims,
                        shape,
                    } => {
                        crate::run::warm_reduce(
                            device::MetalDevice::get(),
                            *op,
                            expr,
                            strides,
                            in_shape,
                            dims,
                            *keepdims,
                            shape,
                            dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::PackOptimizerScalars => {
                        for input in &command.inputs {
                            let scalar = &self.values[input.index()];
                            if scalar.dtype != DType::F32 {
                                crate::kernels::warm_cast(&scalar.shape, scalar.dtype, DType::F32)?;
                            }
                        }
                        let shapes = vec![[1usize]; command.inputs.len()];
                        let shapes = shapes
                            .iter()
                            .map(|shape| shape.as_slice())
                            .collect::<Vec<_>>();
                        crate::indexing::warm_cat(&shapes, DType::F32, 0)?;
                        pipeline_count += 1;
                    }
                    MetalOp::AdamW {
                        implementation: OptimizerImplementation::Fused,
                        exprs,
                        ..
                    } => {
                        let shape = &self.values[command.inputs[0].index()].shape;
                        let strides = vec![fusion::contiguous_strides(shape); 4];
                        crate::run::warm_elementwise(
                            device::MetalDevice::get(),
                            exprs,
                            &strides,
                            shape,
                            shape.iter().product(),
                            3,
                            dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::AdamW {
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        implementation: OptimizerImplementation::Generic,
                        ..
                    } => {
                        let shape = &self.values[command.inputs[0].index()].shape;
                        let strides = vec![fusion::contiguous_strides(shape); 4];
                        crate::run::warm_elementwise(
                            device::MetalDevice::get(),
                            &fusion::adamw_exprs(*beta1, *beta2, *eps, *weight_decay),
                            &strides,
                            shape,
                            shape.iter().product(),
                            3,
                            dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::AdamWGroup { parameters, exprs } => {
                        let shape = &self.values[command.inputs[0].index()].shape;
                        let strides = vec![fusion::contiguous_strides(shape); parameters * 4];
                        crate::run::warm_elementwise(
                            device::MetalDevice::get(),
                            exprs,
                            &strides,
                            shape,
                            shape.iter().product(),
                            3,
                            dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::Sgd {
                        implementation: OptimizerImplementation::Fused,
                        exprs,
                        ..
                    } => {
                        let shape = &self.values[command.inputs[0].index()].shape;
                        let strides = vec![fusion::contiguous_strides(shape); 3];
                        crate::run::warm_elementwise(
                            device::MetalDevice::get(),
                            exprs,
                            &strides,
                            shape,
                            shape.iter().product(),
                            2,
                            dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::Sgd {
                        momentum,
                        dampening,
                        nesterov,
                        weight_decay,
                        implementation: OptimizerImplementation::Generic,
                        ..
                    } => {
                        let shape = &self.values[command.inputs[0].index()].shape;
                        let strides = vec![fusion::contiguous_strides(shape); 3];
                        crate::run::warm_elementwise(
                            device::MetalDevice::get(),
                            &fusion::sgd_exprs(*momentum, *dampening, *nesterov, *weight_decay),
                            &strides,
                            shape,
                            shape.iter().product(),
                            2,
                            dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::CrossEntropy {
                        implementation: MetalImplementation::Native,
                        ..
                    } => {
                        let target = self.values[command.inputs[1].index()].dtype;
                        loss::warm_forward(dtype, target)?;
                        crate::kernels::warm_cast(&[], DType::F32, dtype)?;
                        pipeline_count += 2;
                    }
                    MetalOp::CrossEntropyBackward {
                        reduction,
                        implementation: MetalImplementation::Native,
                        ..
                    } => {
                        let target = self.values[command.inputs[1].index()].dtype;
                        loss::warm_backward(dtype, target, *reduction)?;
                        pipeline_count += if *reduction == CrossEntropyReduction::Mean {
                            2
                        } else {
                            1
                        };
                    }
                    MetalOp::Sdpa {
                        scale,
                        causal,
                        window,
                        implementation: MetalImplementation::Native,
                    } => {
                        flash::warm_forward(
                            &self.values[command.inputs[0].index()].shape,
                            &self.values[command.inputs[1].index()].shape,
                            &self.values[command.inputs[2].index()].shape,
                            *scale,
                            *causal,
                            *window,
                            dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::SdpaBackward {
                        scale,
                        causal,
                        window,
                        implementation: MetalImplementation::Native,
                    } => {
                        flash::warm_backward(
                            &self.values[command.inputs[0].index()].shape,
                            &self.values[command.inputs[1].index()].shape,
                            &self.values[command.inputs[2].index()].shape,
                            *scale,
                            *causal,
                            *window,
                            dtype,
                        )?;
                        let q = &self.values[command.inputs[0].index()];
                        let v = &self.values[command.inputs[2].index()];
                        let rank = q.shape.len();
                        let flat_shape = vec![
                            q.shape[..rank - 2].iter().product(),
                            q.shape[rank - 2],
                            v.shape[rank - 1],
                        ];
                        crate::ops::warm_binary(
                            &flat_shape,
                            dtype,
                            &flat_shape,
                            dtype,
                            metal_ops::BinOp::Mul,
                            false,
                        )?;
                        crate::ops::warm_reduce(
                            &flat_shape,
                            dtype,
                            &[2],
                            true,
                            fusion::ReduceOp::Sum,
                        )?;
                        crate::kernels::warm_cast(
                            &[flat_shape[0], flat_shape[1], 1],
                            dtype,
                            DType::F32,
                        )?;
                        pipeline_count += 2;
                    }
                    MetalOp::KdaChunk { scale } => {
                        let q = &self.values[command.inputs[0].index()];
                        let v = &self.values[command.inputs[2].index()];
                        kda::warm_forward(
                            dtype,
                            q.shape[q.shape.len() - 1],
                            v.shape[v.shape.len() - 1],
                            *scale,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::KdaRecurrence { .. } => {
                        let q = &self.values[command.inputs[0].index()];
                        let rank = q.shape.len();
                        let dk = q.shape[q.shape.len() - 1];
                        let heads = q.shape[rank - 3];
                        let time = q.shape[rank - 2];
                        match command_plan {
                            MetalCommandPlan::KdaDecode(requirements) => {
                                kda::warm_decode_exact(requirements)?;
                            }
                            MetalCommandPlan::KdaForward(requirements) => {
                                kda::warm_forward_exact(requirements)?;
                                crate::ops::warm_binary(
                                    &[heads, time, dk],
                                    dtype,
                                    &[1, time, 1],
                                    dtype,
                                    metal_ops::BinOp::Mul,
                                    false,
                                )?;
                                crate::ops::warm_binary(
                                    &[heads, time, 1],
                                    dtype,
                                    &[1, time, 1],
                                    dtype,
                                    metal_ops::BinOp::Mul,
                                    false,
                                )?;
                            }
                            _ => return Err("compile: KDA recurrence plan is missing".to_string()),
                        }
                        pipeline_count += 1;
                    }
                    MetalOp::KdaBackward { scale } => {
                        let q = &self.values[command.inputs[0].index()];
                        let v = &self.values[command.inputs[2].index()];
                        kda::warm_backward(
                            dtype,
                            q.shape[q.shape.len() - 1],
                            v.shape[v.shape.len() - 1],
                            *scale,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::ShortConv1d
                    | MetalOp::ShortConv1dBackwardX
                    | MetalOp::ShortConv1dBackwardW => {
                        shortconv::warm_all(dtype)?;
                        pipeline_count += 3;
                    }
                    MetalOp::ConvState { .. } => {
                        let MetalCommandPlan::ShortConvState(requirements) = command_plan else {
                            return Err("compile: convolution state plan is missing".to_string());
                        };
                        shortconv::warm_forward_exact(requirements)?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    MetalOp::KvAttention { scale, .. } => {
                        let query = &self.values[command.inputs[0].index()];
                        let key = &self.values[command.inputs[1].index()];
                        let rank = query.shape.len();
                        pipeline_count += crate::paged::warm_all(
                            query.shape[rank - 1],
                            query.shape[rank - 3],
                            key.shape[rank - 3],
                            *scale,
                        )?;
                    }
                    MetalOp::RotaryEmbedding {
                        layout,
                        implementation: MetalImplementation::Native,
                        ..
                    }
                    | MetalOp::RotaryEmbeddingBackward {
                        layout,
                        implementation: MetalImplementation::Native,
                        ..
                    } => {
                        rotary::warm(dtype, *layout)?;
                        pipeline_count += 1;
                    }
                    MetalOp::LayerNorm {
                        implementation: MetalImplementation::Native,
                        ..
                    } => {
                        layer_norm::warm_forward(dtype)?;
                        pipeline_count += 1;
                    }
                    MetalOp::RmsNorm {
                        implementation: MetalImplementation::Native,
                        ..
                    } => {
                        let MetalCommandPlan::LayerNormForward(requirements) = command_plan else {
                            return Err("compile: RMS norm plan is missing".to_string());
                        };
                        layer_norm::warm_rms_exact(requirements)?;
                        pipeline_count += 1;
                    }
                    MetalOp::LayerNormBackward {
                        implementation: MetalImplementation::Native,
                        ..
                    } => {
                        layer_norm::warm_backward(dtype)?;
                        let x = &self.values[command.inputs[0].index()];
                        let weight = &self.values[command.inputs[1].index()];
                        let gradient = &self.values[command.inputs[2].index()];
                        crate::ops::warm_binary(
                            &gradient.shape,
                            gradient.dtype,
                            &x.shape,
                            x.dtype,
                            metal_ops::BinOp::Mul,
                            false,
                        )?;
                        let dimensions =
                            (0..x.shape.len() - weight.shape.len()).collect::<Vec<_>>();
                        crate::ops::warm_reduce(
                            &x.shape,
                            x.dtype,
                            &dimensions,
                            false,
                            fusion::ReduceOp::Sum,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::Unary(kind) => {
                        let shape = &self.values[command.inputs[0].index()].shape;
                        let input_dtype = self.values[command.inputs[0].index()].dtype;
                        match kind {
                            MetalUnaryOp::Relu if dtype.is_float() => {
                                crate::kernels::warm_cast(shape, input_dtype, DType::F32)?;
                                crate::ops::warm_relu(shape, dtype)?;
                                crate::kernels::warm_cast(shape, DType::F32, input_dtype)?;
                            }
                            MetalUnaryOp::Relu => crate::kernels::warm_relu_i64(shape)?,
                            MetalUnaryOp::Pow { exponent_bits } => {
                                crate::kernels::warm_cast(shape, input_dtype, DType::F32)?;
                                crate::ops::warm_pow(shape, *exponent_bits)?;
                                crate::kernels::warm_cast(shape, DType::F32, input_dtype)?;
                            }
                            MetalUnaryOp::Cast { dtype: destination } => {
                                crate::kernels::warm_cast(shape, input_dtype, *destination)?;
                            }
                            kind => crate::ops::warm_unary(
                                shape,
                                dtype,
                                match kind {
                                    MetalUnaryOp::Neg => metal_ops::UnOp::Neg,
                                    MetalUnaryOp::Abs => metal_ops::UnOp::Abs,
                                    MetalUnaryOp::Sqrt => metal_ops::UnOp::Sqrt,
                                    MetalUnaryOp::Exp => metal_ops::UnOp::Exp,
                                    MetalUnaryOp::Log => metal_ops::UnOp::Log,
                                    MetalUnaryOp::Sin => metal_ops::UnOp::Sin,
                                    MetalUnaryOp::Cos => metal_ops::UnOp::Cos,
                                    MetalUnaryOp::Tanh => metal_ops::UnOp::Tanh,
                                    MetalUnaryOp::Erf => metal_ops::UnOp::Erf,
                                    MetalUnaryOp::Gelu { approximate } => {
                                        if *approximate {
                                            metal_ops::UnOp::GeluTanh
                                        } else {
                                            metal_ops::UnOp::Gelu
                                        }
                                    }
                                    MetalUnaryOp::Floor => metal_ops::UnOp::Floor,
                                    MetalUnaryOp::Ceil => metal_ops::UnOp::Ceil,
                                    MetalUnaryOp::Round => metal_ops::UnOp::Round,
                                    MetalUnaryOp::Sign => metal_ops::UnOp::Sign,
                                    MetalUnaryOp::Relu
                                    | MetalUnaryOp::Pow { .. }
                                    | MetalUnaryOp::Cast { .. } => unreachable!(),
                                },
                            )?,
                        }
                        if !matches!(
                            kind,
                            MetalUnaryOp::Cast { .. }
                                | MetalUnaryOp::Relu
                                | MetalUnaryOp::Pow { .. }
                        ) {
                            crate::kernels::warm_cast(shape, input_dtype, DType::F32)?;
                            crate::kernels::warm_cast(shape, DType::F32, input_dtype)?;
                        }
                        pipeline_count += 1;
                    }
                    MetalOp::Binary(kind)
                        if !matches!(
                            kind,
                            MetalBinaryOp::Concat { .. } | MetalBinaryOp::Matmul
                        ) =>
                    {
                        let a = &self.values[command.inputs[0].index()];
                        let b = &self.values[command.inputs[1].index()];
                        let (operation, compare) = match kind {
                            MetalBinaryOp::Add => (metal_ops::BinOp::Add, false),
                            MetalBinaryOp::Sub => (metal_ops::BinOp::Sub, false),
                            MetalBinaryOp::Mul => (metal_ops::BinOp::Mul, false),
                            MetalBinaryOp::Div => (metal_ops::BinOp::Div, false),
                            MetalBinaryOp::Eq => (metal_ops::BinOp::Eq, true),
                            MetalBinaryOp::Gt => (metal_ops::BinOp::Gt, true),
                            MetalBinaryOp::Lt => (metal_ops::BinOp::Lt, true),
                            MetalBinaryOp::Ge => (metal_ops::BinOp::Ge, true),
                            MetalBinaryOp::Le => (metal_ops::BinOp::Le, true),
                            MetalBinaryOp::Maximum => (metal_ops::BinOp::Max, false),
                            MetalBinaryOp::Minimum => (metal_ops::BinOp::Min, false),
                            MetalBinaryOp::Concat { .. } | MetalBinaryOp::Matmul => unreachable!(),
                        };
                        crate::ops::warm_binary(
                            &a.shape, a.dtype, &b.shape, b.dtype, operation, compare,
                        )?;
                        if compare {
                            crate::kernels::warm_cast(&a.shape, a.dtype, DType::F32)?;
                            crate::kernels::warm_cast(&b.shape, b.dtype, DType::F32)?;
                            crate::kernels::warm_cast(
                                &self.values[command.outputs[0].index()].shape,
                                DType::F32,
                                DType::U8,
                            )?;
                        } else {
                            let mut a_dtype = a.dtype;
                            let mut b_dtype = b.dtype;
                            if a_dtype != b_dtype
                                && a_dtype.is_float()
                                && b_dtype.is_float()
                                && a.shape.is_empty()
                                && !b.shape.is_empty()
                            {
                                crate::kernels::warm_cast(&a.shape, a_dtype, b_dtype)?;
                                a_dtype = b_dtype;
                            } else if a_dtype != b_dtype
                                && a_dtype.is_float()
                                && b_dtype.is_float()
                                && b.shape.is_empty()
                                && !a.shape.is_empty()
                            {
                                crate::kernels::warm_cast(&b.shape, b_dtype, a_dtype)?;
                                b_dtype = a_dtype;
                            }
                            if a_dtype != b_dtype || !matches!(a_dtype, DType::F32 | DType::BF16) {
                                crate::kernels::warm_cast(&a.shape, a_dtype, DType::F32)?;
                                crate::kernels::warm_cast(&b.shape, b_dtype, DType::F32)?;
                                crate::kernels::warm_cast(
                                    &self.values[command.outputs[0].index()].shape,
                                    DType::F32,
                                    a_dtype,
                                )?;
                            }
                        }
                        pipeline_count += 1;
                    }
                    MetalOp::Binary(MetalBinaryOp::Matmul) => {
                        let a = &self.values[command.inputs[0].index()];
                        let b = &self.values[command.inputs[1].index()];
                        let ar = a.shape.len();
                        let br = b.shape.len();
                        let batch_a = a.shape[..ar - 2].iter().product::<usize>();
                        let batch_b = b.shape[..br - 2].iter().product::<usize>();
                        pipeline_count += crate::gemm::warm_gemm_fused(
                            device::MetalDevice::get(),
                            a.dtype,
                            false,
                            crate::gemm::Epilogue::None,
                            batch_a.max(batch_b),
                            a.shape[ar - 2],
                            b.shape[br - 1],
                            a.shape[ar - 1],
                            self.environment.mma,
                        )?;
                    }
                    MetalOp::Linear {
                        implementation: MetalImplementation::Native,
                    }
                    | MetalOp::LinearResidual {
                        implementation: MetalImplementation::Native,
                    }
                    | MetalOp::LinearGelu {
                        implementation: MetalImplementation::Native,
                        ..
                    } => {
                        let x = &self.values[command.inputs[0].index()];
                        let weight = &self.values[command.inputs[1].index()];
                        let rank = x.shape.len();
                        let epilogue = match op {
                            MetalOp::Linear { .. } => crate::gemm::Epilogue::None,
                            MetalOp::LinearResidual { .. } => crate::gemm::Epilogue::Residual,
                            MetalOp::LinearGelu {
                                approximate: false,
                                dual: false,
                                ..
                            } => crate::gemm::Epilogue::GeluErf,
                            MetalOp::LinearGelu {
                                approximate: true,
                                dual: false,
                                ..
                            } => crate::gemm::Epilogue::GeluTanh,
                            MetalOp::LinearGelu {
                                approximate: false,
                                dual: true,
                                ..
                            } => crate::gemm::Epilogue::GeluErfDual,
                            MetalOp::LinearGelu {
                                approximate: true,
                                dual: true,
                                ..
                            } => crate::gemm::Epilogue::GeluTanhDual,
                            _ => unreachable!(),
                        };
                        pipeline_count += crate::gemm::warm_gemm_fused(
                            device::MetalDevice::get(),
                            x.dtype,
                            true,
                            epilogue,
                            x.shape[..rank - 2].iter().product(),
                            x.shape[rank - 2],
                            weight.shape[1],
                            weight.shape[0],
                            self.environment.mma,
                        )?;
                    }
                    MetalOp::Where => {
                        let condition = &self.values[command.inputs[0].index()];
                        let branch = &self.values[command.inputs[1].index()];
                        crate::kernels::warm_cast(&condition.shape, condition.dtype, branch.dtype)?;
                        crate::ops::warm_where(
                            &self.values[command.inputs[0].index()].shape,
                            &self.values[command.inputs[1].index()].shape,
                            &self.values[command.inputs[2].index()].shape,
                            self.values[command.inputs[1].index()].dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::Argmax { dim } | MetalOp::Argmin { dim } => {
                        let input = &self.values[command.inputs[0].index()];
                        crate::kernels::warm_argreduce(
                            &input.shape,
                            input.dtype,
                            *dim,
                            matches!(op, MetalOp::Argmax { .. }),
                        )?;
                        crate::kernels::warm_cast(
                            &self.values[command.outputs[0].index()].shape,
                            DType::U32,
                            DType::I64,
                        )?;
                        pipeline_count += 2;
                    }
                    MetalOp::Cumsum { dim } => {
                        let input = &self.values[command.inputs[0].index()];
                        crate::kernels::warm_cumsum(&input.shape, input.dtype, *dim)?;
                        pipeline_count += 1;
                    }
                    MetalOp::Gather { dim } => {
                        let input = &self.values[command.inputs[0].index()];
                        let indexes = &self.values[command.inputs[1].index()];
                        crate::indexing::warm_gather_layout(
                            &declaration_layout(input),
                            input.dtype,
                            *dim,
                            &declaration_layout(indexes),
                            indexes.dtype,
                            &indexes.shape,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::IndexSelect { dim } => {
                        let input = &self.values[command.inputs[0].index()];
                        let indexes = &self.values[command.inputs[1].index()];
                        crate::indexing::warm_index_select_layout_exact(
                            &declaration_layout(input),
                            input.dtype,
                            *dim,
                            &declaration_layout(indexes),
                            indexes.dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::ScatterAdd { dim } => {
                        let input = &self.values[command.inputs[0].index()];
                        let indexes = &self.values[command.inputs[1].index()];
                        let source = &self.values[command.inputs[2].index()];
                        crate::indexing::warm_scatter_add_layouts(
                            &declaration_layout(input),
                            input.dtype,
                            *dim,
                            &declaration_layout(indexes),
                            indexes.dtype,
                            &declaration_layout(source),
                            source.dtype,
                        )?;
                        pipeline_count += 1;
                    }
                    MetalOp::Binary(MetalBinaryOp::Concat { dim }) => {
                        let a = &self.values[command.inputs[0].index()];
                        let b = &self.values[command.inputs[1].index()];
                        crate::indexing::warm_cat(&[&a.shape, &b.shape], a.dtype, *dim)?;
                        pipeline_count += 1;
                    }
                    MetalOp::Reshape { shape } => {
                        let input = &self.values[command.inputs[0].index()];
                        crate::kernels::warm_copy(shape, input.dtype)?;
                        pipeline_count += usize::from(shape.iter().product::<usize>() != 0);
                    }
                    MetalOp::PositionEmbedding { seq_len } => {
                        let input = &self.values[command.inputs[0].index()];
                        let layout = effect_torch_runtime::Layout::contiguous(input.shape.to_vec())
                            .narrow(0, 0, *seq_len);
                        crate::kernels::warm_copy_layout(&layout, input.dtype)?;
                        pipeline_count +=
                            usize::from(layout.shape().iter().product::<usize>() != 0);
                    }
                    MetalOp::Permute { dims } => {
                        let input = &self.values[command.inputs[0].index()];
                        let layout = effect_torch_runtime::Layout::contiguous(input.shape.to_vec())
                            .permute(dims);
                        crate::kernels::warm_copy_layout(&layout, input.dtype)?;
                        pipeline_count += 1;
                    }
                    MetalOp::BroadcastTo { shape } => {
                        let input = &self.values[command.inputs[0].index()];
                        let layout = effect_torch_runtime::Layout::contiguous(input.shape.to_vec())
                            .broadcast_to(shape);
                        crate::kernels::warm_copy_layout(&layout, input.dtype)?;
                        pipeline_count += 1;
                    }
                    MetalOp::Slice { ranges } => {
                        let input = &self.values[command.inputs[0].index()];
                        let mut layout =
                            effect_torch_runtime::Layout::contiguous(input.shape.to_vec());
                        let mut empty = false;
                        for (dimension, &(start, stop, stride)) in ranges.iter().enumerate() {
                            let length = stop.saturating_sub(start).div_ceil(stride);
                            if length == 0 {
                                empty = true;
                                break;
                            }
                            layout = layout.narrow(dimension, start, (length - 1) * stride + 1);
                            if stride > 1 {
                                let mut shape = layout.shape().to_vec();
                                let mut strides = layout.strides().to_vec();
                                strides[dimension] = strides[dimension]
                                    .checked_mul(stride)
                                    .ok_or_else(|| "compile: slice stride overflow".to_string())?;
                                shape[dimension] = length;
                                layout = effect_torch_runtime::Layout::new(
                                    shape,
                                    strides,
                                    layout.offset(),
                                );
                            }
                        }
                        if !empty {
                            crate::kernels::warm_copy_layout(&layout, input.dtype)?;
                            pipeline_count += usize::from(layout.numel() != 0);
                        }
                    }
                    MetalOp::LastTokenRow => {
                        let input = &self.values[command.inputs[0].index()];
                        if input.shape.len() == 3 {
                            let layout =
                                effect_torch_runtime::Layout::contiguous(vec![input.shape[2]]);
                            crate::kernels::warm_copy_layout(&layout, input.dtype)?;
                            pipeline_count += usize::from(layout.numel() != 0);
                        }
                    }
                    MetalOp::Reduce { op, dims, keepdims } => {
                        let input = &self.values[command.inputs[0].index()];
                        let output = &self.values[command.outputs[0].index()];
                        crate::kernels::warm_cast(&input.shape, input.dtype, DType::F32)?;
                        crate::ops::warm_reduce(
                            &input.shape,
                            input.dtype,
                            dims,
                            *keepdims,
                            match op {
                                MetalReduceOp::Sum => fusion::ReduceOp::Sum,
                                MetalReduceOp::Mean => fusion::ReduceOp::Mean,
                                MetalReduceOp::Max => fusion::ReduceOp::Max,
                                MetalReduceOp::Min => fusion::ReduceOp::Min,
                                MetalReduceOp::Prod => fusion::ReduceOp::Prod,
                            },
                        )?;
                        crate::kernels::warm_cast(&output.shape, DType::F32, output.dtype)?;
                        pipeline_count += 1;
                    }
                    MetalOp::Conv1d {
                        stride,
                        padding,
                        dilation,
                        groups,
                    } => {
                        let requirements = conv_plan(command_plan)?;
                        crate::conv::warm_conv1d(
                            &self.values[command.inputs[0].index()].shape,
                            &self.values[command.inputs[1].index()].shape,
                            *stride,
                            *padding,
                            *dilation,
                            *groups,
                        )?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    MetalOp::Conv2d {
                        stride,
                        padding,
                        dilation,
                        groups,
                    } => {
                        let requirements = conv_plan(command_plan)?;
                        crate::conv::warm_conv2d(
                            &self.values[command.inputs[0].index()].shape,
                            &self.values[command.inputs[1].index()].shape,
                            *stride,
                            *padding,
                            *dilation,
                            *groups,
                        )?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    MetalOp::ConvTranspose1d {
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups,
                    } => {
                        let requirements = conv_plan(command_plan)?;
                        crate::conv::warm_conv_transpose1d(
                            &self.values[command.inputs[0].index()].shape,
                            &self.values[command.inputs[1].index()].shape,
                            *stride,
                            *padding,
                            *output_padding,
                            *dilation,
                            *groups,
                        )?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    MetalOp::ConvTranspose2d {
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups,
                    } => {
                        let requirements = conv_plan(command_plan)?;
                        crate::conv::warm_conv_transpose2d(
                            &self.values[command.inputs[0].index()].shape,
                            &self.values[command.inputs[1].index()].shape,
                            *stride,
                            *padding,
                            *output_padding,
                            *dilation,
                            *groups,
                        )?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    MetalOp::Conv1dBackwardW {
                        kernel,
                        out_channels,
                        stride,
                        padding,
                        dilation,
                        groups,
                    } => {
                        let requirements = conv_plan(command_plan)?;
                        let mut x = self.values[command.inputs[0].index()].shape.to_vec();
                        let mut gradient = self.values[command.inputs[1].index()].shape.to_vec();
                        x.push(1);
                        gradient.push(1);
                        crate::conv::warm_conv2d_backward_w(
                            &x,
                            &gradient,
                            [*kernel, 1],
                            *out_channels,
                            *stride,
                            *padding,
                            *dilation,
                            *groups,
                        )?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    MetalOp::Conv2dBackwardW {
                        kernel,
                        out_channels,
                        stride,
                        padding,
                        dilation,
                        groups,
                    } => {
                        let requirements = conv_plan(command_plan)?;
                        crate::conv::warm_conv2d_backward_w(
                            &self.values[command.inputs[0].index()].shape,
                            &self.values[command.inputs[1].index()].shape,
                            *kernel,
                            *out_channels,
                            *stride,
                            *padding,
                            *dilation,
                            *groups,
                        )?;
                        pipeline_count += requirements.pipeline_count;
                    }
                    _ => {}
                }
            }
            Ok(pipeline_count)
        })?;
        let command_count = physical
            .iter()
            .filter(|command| matches!(command, MetalPhysicalCommand::Encode(_)))
            .count();
        let synchronization_count = physical
            .iter()
            .filter(|command| matches!(command, MetalPhysicalCommand::Complete))
            .count();
        let diagnostics = build_executable_diagnostics(
            &program,
            &memory,
            &driver.prepared().index,
            DiagnosticsInput {
                pipeline_count,
                command_count,
                synchronization_count,
                ..DiagnosticsInput::default()
            },
            MetalInstruction::name,
        );
        Ok((
            MetalExecutable {
                signature: driver.prepared().signature.clone(),
                program,
                physical: physical.into_boxed_slice(),
                prepared: MetalPreparedArtifacts { pipeline_count },
                bindings: self.bindings.into_boxed_slice(),
                scalar_bindings: self.scalar_bindings.into_boxed_slice(),
                padded_bindings: self.padded_bindings.into_boxed_slice(),
                constants: self.constants.into_boxed_slice(),
                options: self.options,
                environment: self.environment,
                state_schema: self.state_schema,
                memory,
                diagnostics,
                compiler_work: CompilerWorkReport::default(),
                last_invocation_memory: Mutex::new(None),
                state_cursor: self.state_cursor,
            },
            self.generated,
            self.generated_order,
        ))
    }
}

fn validate_generated_binding_metadata<'a>(
    index: &'a GraphIndex,
    binding: &GeneratedBinding,
) -> Result<&'a Arc<Node>, String> {
    let node = index.node(binding.node).ok_or_else(|| {
        format!(
            "compile: generated binding {} references an unknown dense node",
            binding.node_id
        )
    })?;
    if node.id != binding.node_id || index.dense_id(binding.node_id) != Some(binding.node) {
        return Err(format!(
            "compile: generated binding {} has inconsistent node identity",
            binding.node_id
        ));
    }
    let NodeKind::Leaf(slot) = &node.kind else {
        return Err(format!(
            "compile: generated binding {} does not reference a leaf",
            binding.node_id
        ));
    };
    if Arc::as_ptr(slot) as usize != binding.slot_identity {
        return Err(format!(
            "compile: generated binding {} has inconsistent slot identity",
            binding.node_id
        ));
    }
    if node.shape != binding.shape
        || node.dtype != binding.dtype
        || !node.device.same_device(&binding.device)
        || !binding.device.is_metal()
    {
        return Err(format!(
            "compile: generated binding {} has inconsistent indexed metadata",
            binding.node_id
        ));
    }
    Ok(node)
}

fn validate_generated_binding_value(
    binding: &GeneratedBinding,
    payload: &Value,
) -> Result<(), String> {
    if payload.shape() != binding.shape
        || payload.dtype() != binding.dtype
        || !payload.device().same_device(&binding.device)
        || !payload.device().is_metal()
    {
        return Err(format!(
            "compile: generated binding {} changed shape, dtype, or device",
            binding.node_id
        ));
    }
    let tensor = payload.as_metal().map_err(|error| {
        format!(
            "compile: generated binding {} is not a Metal value: {error}",
            binding.node_id
        )
    })?;
    if !tensor.layout.is_contiguous() || tensor.layout.offset() != 0 {
        return Err(format!(
            "compile: generated binding {} must be zero-offset contiguous",
            binding.node_id
        ));
    }
    let bytes = tensor
        .layout
        .checked_byte_size(payload.dtype())
        .ok_or_else(|| {
            format!(
                "compile: generated binding {} has an overflowing layout",
                binding.node_id
            )
        })?;
    if bytes > tensor.buffer.size {
        return Err(format!(
            "compile: generated binding {} exceeds its Metal buffer",
            binding.node_id
        ));
    }
    Ok(())
}

fn validate_prepared_generated_bindings(
    index: &GraphIndex,
    generated_bindings: &[Value],
) -> Result<(), String> {
    if generated_bindings.len() != index.leaves.len() {
        return Err(format!(
            "compile: prepared index declares {} generated bindings, got {} values",
            index.leaves.len(),
            generated_bindings.len()
        ));
    }
    for (binding, payload) in index.leaves.iter().zip(generated_bindings) {
        validate_generated_binding_metadata(index, binding)?;
        validate_generated_binding_value(binding, payload)?;
    }
    Ok(())
}

/// Loads one immutable value snapshot for each generated leaf in semantic order.
pub(super) fn load_generated_bindings(index: &GraphIndex) -> Result<Vec<Value>, String> {
    if index.order.iter().any(|node| !node.device.is_metal())
        || index.slots.iter().any(|slot| !slot.device.is_metal())
    {
        return Err("compile: graph contains an unsupported device".to_string());
    }
    index
        .leaves
        .iter()
        .map(|binding| {
            let node = validate_generated_binding_metadata(index, binding)?;
            let NodeKind::Leaf(slot) = &node.kind else {
                unreachable!("generated binding metadata validation requires a leaf")
            };
            let payload = slot.get::<Value>().map_err(|error| {
                format!("compile: generated binding {}: {error}", binding.node_id)
            })?;
            validate_generated_binding_value(binding, &payload)?;
            Ok(payload)
        })
        .collect()
}

fn validate_metal_support(node: &Node) -> Result<(), String> {
    match &node.kind {
        NodeKind::Inverse { .. } => Err("compile: inverse is not supported on Metal".to_string()),
        NodeKind::Det { .. } => Err("compile: det is not supported on Metal".to_string()),
        NodeKind::Solve { .. } => Err("compile: solve is not supported on Metal".to_string()),
        NodeKind::Arange { step, .. } if *step == 0.0 => {
            Err("compile: arange step must not be zero".to_string())
        }
        NodeKind::KdaChunk { q, .. }
        | NodeKind::KdaRecurrence { q, .. }
        | NodeKind::KdaBackward { q, .. }
            if !matches!(q.dtype, DType::F32 | DType::BF16) =>
        {
            Err(format!(
                "compile: KDA does not support Metal dtype {}",
                q.dtype
            ))
        }
        NodeKind::ShortConv1d { x, .. }
        | NodeKind::ShortConv1dBackwardX { x, .. }
        | NodeKind::ShortConv1dBackwardW { x, .. }
        | NodeKind::ConvState { x, .. }
            if !matches!(x.dtype, DType::F32 | DType::BF16) =>
        {
            Err(format!(
                "compile: short convolution does not support Metal dtype {}",
                x.dtype
            ))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
pub(super) fn compile(
    roots: &[Arc<Node>],
    options: CompileOptions,
    ce_chunk_size: usize,
    environment: MetalEnvironment,
) -> Result<MetalCompilation, String> {
    compile_with_state(roots, options, ce_chunk_size, environment, None)
}

#[cfg(test)]
pub(super) fn compile_with_state(
    roots: &[Arc<Node>],
    mut options: CompileOptions,
    ce_chunk_size: usize,
    environment: MetalEnvironment,
    state_schema: Option<KvStateSchema>,
) -> Result<MetalCompilation, String> {
    options.environment.ce_chunk_size = ce_chunk_size;
    options.environment.metal_private_intermediates = environment.private_intermediates;
    options.environment.metal_mma = environment.mma;
    let mut request = ProgramRequest::from_roots(roots.to_vec(), options);
    if let Some(schema) = state_schema {
        request = request.with_state_cursor(StateCursorSlot::new(
            schema.cursor_slot,
            schema.cursor_tensor,
        ));
    }
    let program = request.prepare()?;
    let generated_bindings = load_generated_bindings(&program.index)?;
    compile_prepared_with_state(&program, &generated_bindings, state_schema)
}

pub(super) fn compile_prepared_with_state(
    program: &PreparedProgram,
    generated_bindings: &[Value],
    state_schema: Option<KvStateSchema>,
) -> Result<MetalCompilation, String> {
    device::snapshot_global_environment();
    let index = &program.index;
    let options = program.options.clone();
    let ce_chunk_size = options.environment.ce_chunk_size;
    let environment = options.environment.into();
    if index.roots.is_empty() {
        return Err("compile: expected at least one root".to_string());
    }
    if ce_chunk_size == 0 {
        return Err("compile: CE chunk size must be positive".to_string());
    }
    let state_cursor =
        state_schema.map(|schema| StateCursorSlot::new(schema.cursor_slot, schema.cursor_tensor));
    if program.state_cursor != state_cursor {
        return Err("compile: Metal state cursor does not match the prepared program".to_string());
    }
    if let Some(schema) = &state_schema {
        schema.validate()?;
    }
    validate_prepared_generated_bindings(index, generated_bindings)?;
    let mut driver = CompilerDriver::new(program)?;
    let slots = index.slots.to_vec();
    if index.order.iter().any(|node| !node.device.is_metal())
        || slots.iter().any(|slot| !slot.device.is_metal())
    {
        return Err("compile: graph contains an unsupported device".to_string());
    }
    // Constructor constants lowered below dispatch Metal kernels. Their
    // submission is owned and drained before the artifact can be published or
    // moved to another NAPI worker.
    let metal = device::MetalDevice::get();
    let compile_submission_started = Instant::now();
    let _compile_submission = metal.begin_submission()?;
    let mut lowerer = Lowerer::new(
        index,
        generated_bindings,
        options,
        ce_chunk_size,
        environment,
        state_schema,
        &slots,
    );
    let lowering_result = (|| {
        driver.lower(|unit, index, plan| {
            match unit {
                LoweringUnit::Node(node) => {
                    let semantic = index
                        .node(node)
                        .ok_or_else(|| format!("compile: dense node {node} is out of range"))?;
                    lowerer.lower(semantic)?;
                }
                LoweringUnit::Region(region) => lowerer.lower_region(index, plan, region)?,
            }
            Ok(())
        })?;
        let outputs = index
            .roots
            .iter()
            .map(|root| lowerer.dense_value(index, *root))
            .collect::<Result<Vec<_>, _>>()?;
        let artifact_assembly_started = Instant::now();
        let artifact = lowerer.finish(outputs, &mut driver);
        driver.record_phase(ARTIFACT_ASSEMBLY_PHASE, artifact_assembly_started.elapsed());
        artifact
    })();
    let gpu_result = metal.synchronize();
    driver.record_phase(
        COMPILE_SUBMISSION_PHASE,
        compile_submission_started.elapsed(),
    );
    gpu_result?;
    let (executable, generated_bindings, generated_order) = lowering_result?;
    let publication_started = Instant::now();
    let mut compilation = MetalCompilation {
        executable: Arc::new(executable),
        slots,
        generated_bindings,
        generated_order,
    };
    let compiler_work = driver.finish_with_phase(
        &compilation.executable.program,
        PUBLICATION_PHASE,
        publication_started,
    );
    let executable = Arc::get_mut(&mut compilation.executable)
        .expect("newly published Metal executable must be uniquely owned");
    executable.diagnostics.compile_phases = compiler_work.compile_phases.clone();
    executable.compiler_work = compiler_work;
    Ok(compilation)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CeCheck {
    ForwardMean,
    ForwardSum,
    BackwardMean,
}

struct DeferredCeCheck {
    buffer: Value,
    kind: CeCheck,
    classes: usize,
}

struct DeferredQuantizedEmbeddingCheck {
    buffer: Value,
    rows: usize,
}

fn run_ce_checks(checks: &[DeferredCeCheck]) -> Result<(), String> {
    for check in checks {
        let tensor = check.buffer.as_metal()?;
        if tensor.dtype != DType::F32 || !tensor.layout.is_contiguous() {
            return Err("cross_entropy: deferred status must be contiguous f32".to_string());
        }
        let values = tensor
            .buffer
            .read_f32(tensor.layout.offset(), tensor.numel());
        match check.kind {
            CeCheck::ForwardMean | CeCheck::ForwardSum => {
                let (active, invalid) = (values[1] as usize, values[2] as usize);
                if check.kind == CeCheck::ForwardMean && active == 0 {
                    return Err(
                        "cross_entropy: no active targets (all positions are ignored)".to_string(),
                    );
                }
                if invalid > 0 {
                    return Err(format!(
                        "cross_entropy: target out of range [0, {}) at an active position",
                        check.classes
                    ));
                }
            }
            CeCheck::BackwardMean if values[0] == 0.0 => {
                return Err(
                    "cross_entropy: no active targets (all positions are ignored)".to_string(),
                );
            }
            CeCheck::BackwardMean => {}
        }
    }
    Ok(())
}

fn run_quantized_embedding_checks(
    checks: &[DeferredQuantizedEmbeddingCheck],
) -> Result<(), String> {
    for check in checks {
        let tensor = check.buffer.as_metal()?;
        if tensor.dtype != DType::U32 || !tensor.layout.is_contiguous() || tensor.numel() != 1 {
            return Err(
                "quantized_embedding: deferred status must be one contiguous u32".to_string(),
            );
        }
        if tensor.to_u32_vec()?[0] != 0 {
            return Err(format!(
                "quantized_embedding: index is outside 0..{}",
                check.rows
            ));
        }
    }
    Ok(())
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(error) = payload.downcast_ref::<device::ForbiddenExecutableAllocation>() {
        error.to_string()
    } else {
        "Metal executable command panicked".to_string()
    }
}

fn resolved_values(
    ids: impl IntoIterator<Item = ValueId>,
    resolved: &[Option<Value>],
) -> Result<Vec<Value>, String> {
    ids.into_iter()
        .map(|value| {
            resolved
                .get(value.index())
                .and_then(Option::as_ref)
                .cloned()
                .ok_or_else(|| format!("Metal executable value {value} was not resolved"))
        })
        .collect()
}

fn resolve_physical_instruction<'a>(
    executable: &'a MetalExecutable,
    id: InstructionId,
) -> Result<&'a MetalCommand, String> {
    let instruction = executable
        .instruction(id)
        .ok_or_else(|| format!("execute: physical instruction {id} is out of range"))?;
    if instruction.kind.operation().is_none() {
        return Err(format!(
            "execute: physical instruction {id} does not resolve to an operation"
        ));
    }
    Ok(instruction)
}

fn pack_optimizer_scalars(
    inputs: &[Value],
    scratch: &[&crate::run::MetalTensor],
    destination: &Value,
) -> Result<crate::run::MetalTensor, String> {
    let destination = destination.as_metal()?;
    let mut scratch_index = 0usize;
    let mut scalars = Vec::with_capacity(inputs.len());
    for (index, value) in inputs.iter().enumerate() {
        let source = value.as_metal()?;
        if source.numel() != 1 {
            return Err(format!(
                "optimizer scalar input {index} has {} elements",
                source.numel()
            ));
        }
        let scalar = if source.dtype == DType::F32 {
            source.clone()
        } else {
            let target = scratch
                .get(scratch_index)
                .copied()
                .ok_or_else(|| format!("optimizer scalar {index} cast scratch is missing"))?;
            scratch_index += 1;
            metal_ops::cast_into(source, target)?;
            target.clone()
        };
        scalars.push(crate::run::MetalTensor {
            buffer: scalar.buffer,
            layout: effect_torch_runtime::Layout::contiguous(vec![1]),
            dtype: DType::F32,
        });
    }
    if scratch_index != scratch.len() {
        return Err(format!(
            "optimizer received {} unused scalar scratch tensor(s)",
            scratch.len() - scratch_index
        ));
    }
    let references = scalars.iter().collect::<Vec<_>>();
    metal_ops::cat_into(&references, 0, destination)?;
    Ok(destination.clone())
}

fn write_advance_mask(mask: &crate::run::MetalTensor, advance: usize) -> Result<(), String> {
    let shape = mask.layout.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 1 || advance > shape[1] {
        return Err(format!(
            "KDA advance mask {:?} cannot represent advance {advance}",
            shape
        ));
    }
    let offset = mask.layout.offset();
    unsafe {
        match mask.dtype {
            DType::F32 => {
                let destination = mask.buffer.contents_ptr().cast::<f32>().add(offset);
                for index in 0..shape[1] {
                    destination
                        .add(index)
                        .write(usize::from(index < advance) as f32);
                }
            }
            DType::BF16 => {
                let destination = mask.buffer.contents_ptr().cast::<u16>().add(offset);
                for index in 0..shape[1] {
                    destination
                        .add(index)
                        .write(half::bf16::from_f32(usize::from(index < advance) as f32).to_bits());
                }
            }
            dtype => return Err(format!("KDA advance mask does not support dtype {dtype}")),
        }
    }
    Ok(())
}

fn contiguous_tensor_view(
    tensor: &crate::run::MetalTensor,
    shape: &[usize],
    element_offset: usize,
) -> Result<crate::run::MetalTensor, String> {
    let elements = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| "Metal tensor view element count overflow".to_string())?;
    let offset = tensor
        .layout
        .offset()
        .checked_add(element_offset)
        .ok_or_else(|| "Metal tensor view offset overflow".to_string())?;
    let end = offset
        .checked_add(elements)
        .and_then(|elements| elements.checked_mul(tensor.dtype.size_in_bytes()))
        .ok_or_else(|| "Metal tensor view byte range overflow".to_string())?;
    if end > tensor.buffer.size {
        return Err(format!(
            "Metal tensor view needs {end} bytes from a {} byte buffer",
            tensor.buffer.size
        ));
    }
    let contiguous = effect_torch_runtime::Layout::contiguous(shape.to_vec());
    Ok(crate::run::MetalTensor {
        buffer: tensor.buffer.clone(),
        layout: effect_torch_runtime::Layout::new(
            shape.to_vec(),
            contiguous.strides().to_vec(),
            offset,
        ),
        dtype: tensor.dtype,
    })
}

fn gemm_scratch_view(
    root: Option<&crate::run::MetalTensor>,
    requirements: &crate::gemm::GemmRequirements,
) -> Result<Option<crate::run::MetalTensor>, String> {
    match requirements.split_k_scratch {
        Some(requirement) => Ok(Some(contiguous_tensor_view(
            root.ok_or_else(|| "split-K workspace is missing".to_string())?,
            &requirement.shape,
            0,
        )?)),
        None => Ok(None),
    }
}

fn conv_plan(plan: &MetalCommandPlan) -> Result<&crate::conv::ConvRequirements, String> {
    let MetalCommandPlan::Conv(requirements) = plan else {
        return Err("convolution is missing exact requirements".to_string());
    };
    Ok(requirements)
}

fn validate_conv_destination(
    plan: &MetalCommandPlan,
    destination: &crate::run::MetalTensor,
) -> Result<(), String> {
    let requirements = conv_plan(plan)?;
    if destination.layout.shape() != requirements.output.shape
        || destination.dtype != requirements.output.dtype
    {
        return Err("convolution destination does not match its exact requirements".to_string());
    }
    destination.validate_destination(
        "convolution",
        &requirements.output.shape,
        requirements.output.dtype,
    )
}

fn commit_state_transactions(
    executable: &MetalExecutable,
    resolved: &[Option<Value>],
    context: Option<&dyn MetalDecodeContext>,
) -> Result<(), String> {
    let context = context.ok_or_else(|| "state commit requires a decode context".to_string())?;
    let mut states = context
        .slots()
        .iter()
        .map(|slot| {
            slot.lock()
                .map_err(|error| format!("state commit sequence lock poisoned: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut copies = Vec::new();
    let mut eviction_starts = vec![0usize; states.len()];
    for physical in &executable.physical {
        let MetalPhysicalCommand::Encode(id) = *physical else {
            continue;
        };
        let command = resolve_physical_instruction(executable, id)?;
        let (op, _) = command
            .kind
            .operation()
            .expect("resolved physical instruction is an operation");
        match op {
            MetalOp::KdaRecurrence { layer, .. } => {
                let root = resolved
                    .get(command.state[0].index())
                    .and_then(Option::as_ref)
                    .ok_or_else(|| "KDA state transaction is unresolved".to_string())?
                    .as_metal()?;
                let geometry = context.schema().kda;
                let elements = geometry
                    .heads
                    .checked_mul(geometry.head_dim)
                    .and_then(|value| value.checked_mul(geometry.value_dim))
                    .ok_or_else(|| "KDA state transaction size overflow".to_string())?;
                for (batch_index, state) in states.iter().take(context.active_batch()).enumerate() {
                    let destination = state
                        .kda_states
                        .get(*layer as usize)
                        .ok_or_else(|| "KDA state commit destination is missing".to_string())?;
                    let source = contiguous_tensor_view(
                        root,
                        &[geometry.heads, geometry.head_dim, geometry.value_dim],
                        batch_index
                            .checked_mul(elements)
                            .ok_or_else(|| "KDA state transaction offset overflow".to_string())?,
                    )?;
                    copies.push(transaction_copy(&source, destination)?);
                }
            }
            MetalOp::ConvState { layer } => {
                let root = resolved
                    .get(command.state[0].index())
                    .and_then(Option::as_ref)
                    .ok_or_else(|| "conv state transaction is unresolved".to_string())?
                    .as_metal()?;
                let geometry = context.schema().conv;
                let elements = geometry
                    .kernel
                    .saturating_sub(1)
                    .checked_mul(geometry.channels)
                    .ok_or_else(|| "conv state transaction size overflow".to_string())?;
                for (batch_index, state) in states.iter().take(context.active_batch()).enumerate() {
                    let destination = state
                        .conv_states
                        .get(*layer as usize)
                        .ok_or_else(|| "conv state commit destination is missing".to_string())?;
                    let source = contiguous_tensor_view(
                        root,
                        &[geometry.kernel.saturating_sub(1), geometry.channels],
                        batch_index
                            .checked_mul(elements)
                            .ok_or_else(|| "conv state transaction offset overflow".to_string())?,
                    )?;
                    copies.push(transaction_copy(&source, destination)?);
                }
            }
            MetalOp::KvAttention { .. } => {
                for (index, state) in states.iter().take(context.active_batch()).enumerate() {
                    let frontier = state
                        .cursor
                        .checked_add(state.advance)
                        .ok_or_else(|| "KV state commit cursor overflow".to_string())?;
                    eviction_starts[index] = eviction_starts[index].max(
                        context
                            .schema()
                            .window
                            .map_or(0, |limit| frontier.saturating_sub(limit)),
                    );
                }
            }
            _ => {}
        }
    }
    for copy in copies {
        unsafe {
            std::ptr::copy_nonoverlapping(copy.source, copy.destination, copy.bytes);
        }
    }
    for (index, state) in states.iter_mut().take(context.active_batch()).enumerate() {
        context.commit_slot(index, state);
    }
    for (state, start) in states
        .iter_mut()
        .take(context.active_batch())
        .zip(eviction_starts)
    {
        context.evict_before(state, start);
    }
    Ok(())
}

struct TransactionCopy {
    source: *const u8,
    destination: *mut u8,
    bytes: usize,
}

fn transaction_copy(
    source: &crate::run::MetalTensor,
    destination: &crate::run::MetalTensor,
) -> Result<TransactionCopy, String> {
    if source.dtype != destination.dtype
        || source.layout.shape() != destination.layout.shape()
        || !source.layout.is_contiguous()
        || !destination.layout.is_contiguous()
    {
        return Err("state transaction source and destination are incompatible".to_string());
    }
    let element_bytes = source.dtype.size_in_bytes();
    let bytes = source
        .layout
        .shape()
        .iter()
        .try_fold(element_bytes, |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| "state transaction byte size overflow".to_string())?;
    let source_offset = source
        .layout
        .offset()
        .checked_mul(element_bytes)
        .ok_or_else(|| "state transaction source offset overflow".to_string())?;
    let destination_offset = destination
        .layout
        .offset()
        .checked_mul(element_bytes)
        .ok_or_else(|| "state transaction destination offset overflow".to_string())?;
    if source_offset.saturating_add(bytes) > source.buffer.size
        || destination_offset.saturating_add(bytes) > destination.buffer.size
    {
        return Err("state transaction copy exceeds its buffer".to_string());
    }
    Ok(TransactionCopy {
        source: unsafe { source.buffer.contents_ptr().cast::<u8>().add(source_offset) },
        destination: unsafe {
            destination
                .buffer
                .contents_ptr()
                .cast::<u8>()
                .add(destination_offset)
        },
        bytes,
    })
}

fn write_scalar_binding(tensor: &crate::run::MetalTensor, value: f64) -> Result<(), String> {
    if !tensor.layout.shape().is_empty() || tensor.layout.offset() != 0 {
        return Err("scalar binding destination must be a zero-offset scalar".to_string());
    }
    if tensor.buffer.size < tensor.dtype.size_in_bytes() {
        return Err("scalar binding destination is too small".to_string());
    }
    let destination = tensor.buffer.contents_ptr().cast::<u8>();
    macro_rules! write {
        ($value:expr) => {{
            let value = $value;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const value).cast::<u8>(),
                    destination,
                    std::mem::size_of_val(&value),
                );
            }
        }};
    }
    match tensor.dtype {
        DType::F32 => write!(value as f32),
        DType::F64 => write!(value),
        DType::F16 => write!(half::f16::from_f64(value).to_bits()),
        DType::BF16 => write!(half::bf16::from_f64(value).to_bits()),
        DType::U8 => write!(value as u8),
        DType::U32 => write!(value as u32),
        DType::I64 => write!(value as i64),
    }
    Ok(())
}

fn write_padded_binding(
    destination: &crate::run::MetalTensor,
    source: &crate::run::MetalTensor,
) -> Result<(), String> {
    let destination_shape = destination.layout.shape();
    let source_shape = source.layout.shape();
    if destination.dtype != source.dtype
        || destination_shape.len() != source_shape.len()
        || source_shape.first().is_none_or(|active| {
            *active == 0
                || *active > destination_shape[0]
                || source_shape[1..] != destination_shape[1..]
        })
        || !source.layout.is_contiguous()
        || source.layout.offset() != 0
        || destination.layout.offset() != 0
    {
        return Err("padded input does not match its bounded signature".to_string());
    }
    let destination_bytes = tensor_bytes(destination_shape, destination.dtype, "padded input")?;
    let source_bytes = tensor_bytes(source_shape, source.dtype, "active padded input")?;
    if destination_bytes > destination.buffer.size || source_bytes > source.buffer.size {
        return Err("padded input exceeds its backing buffer".to_string());
    }
    unsafe {
        std::ptr::write_bytes(
            destination.buffer.contents_ptr().cast::<u8>(),
            0,
            destination_bytes,
        );
        std::ptr::copy_nonoverlapping(
            source.buffer.contents_ptr().cast::<u8>(),
            destination.buffer.contents_ptr().cast::<u8>(),
            source_bytes,
        );
    }
    Ok(())
}

fn random_seed(nonce: u64, provenance: u64) -> u64 {
    let mut value = nonce ^ provenance.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Executes one already-lowered program, including invocation-owned Metal
/// submission, synchronization, deferred statuses, and state publication.
#[cfg(test)]
pub(super) fn execute(
    executable: &MetalExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    cancelled: &CancellationFlag,
    kv: Option<&dyn MetalDecodeContext>,
) -> Result<Vec<Value>, String> {
    execute_with_commit(
        executable,
        declared_bindings,
        generated_bindings,
        &[],
        cancelled,
        kv,
        None,
    )
}

pub(super) fn execute_with_scalars(
    executable: &MetalExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    scalar_bindings: &[f64],
    cancelled: &CancellationFlag,
) -> Result<Vec<Value>, String> {
    execute_with_commit(
        executable,
        declared_bindings,
        generated_bindings,
        scalar_bindings,
        cancelled,
        None,
        None,
    )
}

pub(super) fn execute_stateful(
    executable: &MetalExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    cancelled: &CancellationFlag,
    kv: &dyn MetalDecodeContext,
    commit_allowed: &dyn Fn() -> bool,
) -> Result<Vec<Value>, String> {
    execute_with_commit(
        executable,
        declared_bindings,
        generated_bindings,
        &[],
        cancelled,
        Some(kv),
        Some(commit_allowed),
    )
}

fn execute_with_commit(
    executable: &MetalExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    scalar_bindings: &[f64],
    cancelled: &CancellationFlag,
    kv: Option<&dyn MetalDecodeContext>,
    commit_allowed: Option<&dyn Fn() -> bool>,
) -> Result<Vec<Value>, String> {
    if cancelled.load(Ordering::Relaxed) {
        return Err("operation aborted".to_string());
    }
    match (&executable.state_schema, kv) {
        (Some(schema), Some(context)) if context.schema() == schema => {}
        (Some(_), Some(_)) => {
            return Err("decode state does not match the executable schema".to_string())
        }
        (Some(_), None) => return Err("decode executable requires state bindings".to_string()),
        (None, Some(_)) => return Err("stateless executable received decode state".to_string()),
        (None, None) => {}
    }
    executable
        .signature
        .validate_invocation_counts(
            declared_bindings.len(),
            scalar_bindings.len(),
            usize::from(executable.state_cursor.is_some() && kv.is_some()),
            None,
        )
        .map_err(|error| format!("execute: {error}"))?;
    for (index, value) in declared_bindings.iter().enumerate() {
        if executable
            .padded_bindings
            .iter()
            .any(|binding| binding.source as usize == index)
        {
            continue;
        }
        let tensor = value.as_metal()?;
        executable
            .signature
            .validate_binding_metadata(index, value.dtype(), tensor.placement(), &tensor.layout)
            .map_err(|error| format!("execute: {error}"))?;
    }
    let mut values: Vec<Option<Value>> = std::iter::repeat_with(|| None)
        .take(executable.program.values.len())
        .collect();
    for constant in &executable.constants {
        values[constant.value.index()] = Some(constant.payload.clone());
    }
    for binding in &executable.bindings {
        let source = match binding.source {
            MetalBindingSource::Declared(slot) => declared_bindings
                .get(slot as usize)
                .ok_or_else(|| format!("input slot {slot} is unbound"))?,
            MetalBindingSource::Generated(slot) => generated_bindings
                .get(slot as usize)
                .ok_or_else(|| format!("generated input slot {slot} is unbound"))?,
        };
        let declaration = &executable.program.values[binding.value.index()];
        if source.shape() != declaration.shape.as_ref() || source.dtype() != declaration.dtype {
            return Err(format!(
                "binding for value {} does not match its compiled signature",
                binding.value
            ));
        }
        let tensor = source.as_metal()?;
        if !tensor.layout.is_contiguous() || tensor.layout.offset() != 0 {
            return Err(format!(
                "binding for value {} must be zero-offset contiguous for its compiled executable",
                binding.value
            ));
        }
        device::MetalDevice::get().synchronize_buffer_producer(&tensor.buffer)?;
        values[binding.value.index()] = Some(source.clone());
    }

    if std::env::var_os("EFFECT_TORCH_MEMORY_PLAN_TRACE").is_some() {
        eprintln!("[metal-plan] report: {:?}", executable.memory.report);
        eprintln!(
            "[metal-plan] segments: {:?}",
            executable
                .memory
                .segments
                .iter()
                .map(|segment| (segment.ownership, segment.bytes))
                .collect::<Vec<_>>()
        );
    }
    let resources = crate::workspace::acquire(&executable.memory.segments)?;
    *executable
        .last_invocation_memory
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = Some(InvocationMemoryReport {
        logical: executable.memory.report.clone(),
        leased_workspace_bytes: resources.actual_workspace_bytes,
        opaque_headroom_bytes: 0,
    });

    let mut resolved = values;
    for (index, location) in executable.memory.locations.iter().enumerate() {
        let declaration = executable
            .program
            .values
            .get(index)
            .ok_or_else(|| format!("Metal location {index} has no value declaration"))?;
        resolved[index] = match location {
            Location::External { .. } | Location::Persistent { .. } => resolved[index].clone(),
            Location::Segment {
                segment,
                offset,
                bytes,
            } => {
                let root = resources
                    .segments
                    .get(segment.index())
                    .ok_or_else(|| format!("Metal segment {segment} was not acquired"))?;
                Some(Value(crate::run::MetalTensor {
                    buffer: Arc::new(device::Buffer::suballoc_with_retention(
                        root,
                        *offset,
                        *bytes,
                        resources.retentions[segment.index()].clone(),
                    )),
                    layout: declaration.layout.clone(),
                    dtype: declaration.dtype,
                }))
            }
            Location::Alias { root, byte_offset } => {
                let source = resolved
                    .get(root.index())
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("Metal alias root {root} is unresolved"))?
                    .as_metal()?;
                let bytes = tensor_bytes(&declaration.shape, declaration.dtype, "alias")?;
                Some(Value(crate::run::MetalTensor {
                    buffer: Arc::new(device::Buffer::suballoc(
                        &source.buffer,
                        *byte_offset,
                        bytes,
                    )),
                    layout: declaration.layout.clone(),
                    dtype: declaration.dtype,
                }))
            }
            Location::Output { .. } | Location::State { .. } | Location::InlineScalar { .. } => {
                return Err(format!(
                    "unsupported Metal executable location at index {index}: {location:?}"
                ))
            }
        };
        if resolved[index].is_none() {
            return Err(format!(
                "Metal executable location {index} was not resolved"
            ));
        }
    }
    for binding in &executable.scalar_bindings {
        let scalar = scalar_bindings
            .get(binding.source as usize)
            .copied()
            .ok_or_else(|| format!("scalar binding {} is unbound", binding.source))?;
        let destination = resolved
            .get(binding.value.index())
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("scalar binding value {} is unresolved", binding.value))?
            .as_metal()?;
        write_scalar_binding(destination, scalar)?;
    }
    for binding in &executable.padded_bindings {
        let source = declared_bindings
            .get(binding.source as usize)
            .ok_or_else(|| format!("padded input {} is unbound", binding.source))?
            .as_metal()?;
        let destination = resolved
            .get(binding.value.index())
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("padded input value {} is unresolved", binding.value))?
            .as_metal()?;
        write_padded_binding(destination, source)?;
    }

    let mut ce_checks = Vec::new();
    let mut quantized_embedding_checks = Vec::new();
    let metal = device::MetalDevice::get();
    let _submission = metal.begin_submission()?;
    let invocation_nonce = INVOCATION_NONCE.fetch_add(1, Ordering::AcqRel);
    let dispatch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _dispatch_guard = metal.begin_executable_dispatch()?;
        device::with_execution_environment(
            executable.environment.private_intermediates,
            executable.environment.mma,
            || {
                let mut last_encoded = None;
                let mut pending_status_gate = None;
                let mut completed = false;
                for physical in &executable.physical {
                    if completed {
                        return Err("physical completion is not final".to_string());
                    }
                    match *physical {
                        MetalPhysicalCommand::Encode(id) => {
                            if pending_status_gate.is_some() {
                                return Err(
                                    "physical status gate is missing its command-buffer commit"
                                        .to_string(),
                                );
                            }
                            if cancelled.load(Ordering::Relaxed) {
                                return Err("operation aborted".to_string());
                            }
                            let command = resolve_physical_instruction(executable, id)?;
                            let MetalInstruction::Operation {
                                op,
                                plan,
                                random_seed_token,
                                ..
                            } = &command.kind
                            else {
                                unreachable!("physical resolution requires an operation")
                            };
                            let _profile = metal.profile_region(op.profile_name(plan));
                            let inputs = resolved_values(
                                command.inputs.iter().map(|use_| use_.value),
                                &resolved,
                            )?;
                            let outputs = resolved_values(
                                command.outputs.iter().map(|output| output.value),
                                &resolved,
                            )?;
                            let scratch = resolved_values(
                                command.scratch.iter().map(|use_| use_.value),
                                &resolved,
                            )?;
                            let staging = resolved_values(
                                command.staging.iter().map(|use_| use_.value),
                                &resolved,
                            )?;
                            let status = resolved_values(
                                command.status.iter().map(|use_| use_.value),
                                &resolved,
                            )?;
                            let state = resolved_values(
                                command.state.iter().map(|use_| use_.value),
                                &resolved,
                            )?;
                            if let MetalOp::KvAttention { layer, .. } = op {
                                let context = kv.ok_or_else(|| {
                                    "KV attention requires executable decode state".to_string()
                                })?;
                                let MetalCommandPlan::KvAttention(plan) = plan else {
                                    return Err(
                                        "KV attention is missing exact requirements".to_string()
                                    );
                                };
                                let staging_tensors = staging
                                    .iter()
                                    .map(|value| value.as_metal().cloned())
                                    .collect::<Result<Vec<_>, _>>()?;
                                context.prepare_kv_attention(*layer, plan, &staging_tensors)?;
                            }
                            execute_op_into(
                                op,
                                plan,
                                &inputs,
                                &outputs,
                                &scratch,
                                &staging,
                                &status,
                                &state,
                                kv,
                                &mut ce_checks,
                                &mut quantized_embedding_checks,
                                random_seed(invocation_nonce, *random_seed_token),
                            )
                            .map_err(|error| format!("{}: {error}", op.name()))?;
                            last_encoded = Some(id);
                        }
                        MetalPhysicalCommand::StatusGate(id) => {
                            let command = resolve_physical_instruction(executable, id)?;
                            if last_encoded != Some(id) || command.status.is_empty() {
                                return Err(format!(
                                    "physical status gate {id} does not follow its status instruction"
                                ));
                            }
                            pending_status_gate = Some(id);
                        }
                        MetalPhysicalCommand::Commit => {
                            pending_status_gate.take().ok_or_else(|| {
                                "physical command-buffer commit has no status gate".to_string()
                            })?;
                            metal.commit_executable_command();
                        }
                        MetalPhysicalCommand::Complete => {
                            if pending_status_gate.is_some() || completed {
                                return Err("invalid physical completion ordering".to_string());
                            }
                            completed = true;
                        }
                    }
                }
                if !completed {
                    return Err("physical program has no final completion".to_string());
                }
                Ok(())
            },
        )
    }));
    // Synchronize unconditionally before any segment or output owner can be
    // released. Backend submission failures take precedence over host errors,
    // panics, and cancellation.
    let gpu_result = metal.synchronize();
    gpu_result?;
    // Status command buffers retain their boundaries, but one final fence is
    // sufficient for every deferred host check in command order.
    run_ce_checks(&ce_checks)?;
    run_quantized_embedding_checks(&quantized_embedding_checks)?;
    match dispatch_result {
        Ok(result) => result?,
        Err(payload) => return Err(panic_message(payload)),
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err("operation aborted".to_string());
    }
    if commit_allowed.is_some_and(|allowed| !allowed()) {
        return Err("operation aborted".to_string());
    }
    if kv.is_some() {
        commit_state_transactions(executable, &resolved, kv)?;
    }
    executable
        .program
        .outputs
        .iter()
        .map(|value| {
            resolved[value.index()]
                .as_ref()
                .cloned()
                .ok_or_else(|| format!("internal error: output value {value} is unavailable"))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn execute_op_into(
    op: &MetalOp,
    plan: &MetalCommandPlan,
    inputs: &[Value],
    outputs: &[Value],
    scratch: &[Value],
    staging: &[Value],
    status: &[Value],
    state: &[Value],
    kv: Option<&dyn MetalDecodeContext>,
    ce_checks: &mut Vec<DeferredCeCheck>,
    quantized_embedding_checks: &mut Vec<DeferredQuantizedEmbeddingCheck>,
    random_seed: u64,
) -> Result<(), String> {
    let input = |index: usize| {
        inputs
            .get(index)
            .ok_or_else(|| format!("{}: missing input {index}", op.name()))
    };
    let output = |index: usize| {
        outputs
            .get(index)
            .ok_or_else(|| format!("{}: missing output {index}", op.name()))
    };
    let scratch_tensors = scratch
        .iter()
        .map(Value::as_metal)
        .collect::<Result<Vec<_>, _>>()?;
    let state_tensors = state
        .iter()
        .map(Value::as_metal)
        .collect::<Result<Vec<_>, _>>()?;
    match op {
        MetalOp::PrepareState => kv
            .ok_or_else(|| "state preparation requires a decode context".to_string())?
            .prepare_state(output(0)?.as_metal()?),
        MetalOp::Randn { shape, dtype } => {
            let destination = output(0)?.as_metal()?;
            let random = if *dtype == DType::F32 {
                destination
            } else {
                scratch_tensors[0]
            };
            metal_ops::randn_into(random, random_seed ^ 299_792_458)?;
            if *dtype != DType::F32 {
                metal_ops::cast_into(random, destination)?;
            }
            debug_assert_eq!(destination.layout.shape(), shape.as_ref());
            Ok(())
        }
        MetalOp::Uniform {
            lo,
            hi,
            shape,
            dtype,
        } => {
            let destination = output(0)?.as_metal()?;
            let random = if *dtype == DType::F32 {
                destination
            } else {
                scratch_tensors[0]
            };
            metal_ops::uniform_into(*lo, *hi, random, random_seed ^ 78_778_899)?;
            if *dtype != DType::F32 {
                metal_ops::cast_into(random, destination)?;
            }
            debug_assert_eq!(destination.layout.shape(), shape.as_ref());
            Ok(())
        }
        MetalOp::Unary(kind) => {
            let tensor = input(0)?.as_metal()?;
            let destination = output(0)?.as_metal()?;
            match kind {
                MetalUnaryOp::Relu => metal_ops::relu_into(tensor, destination, &scratch_tensors),
                MetalUnaryOp::Pow { exponent_bits } => {
                    if tensor.dtype == DType::F32 {
                        metal_ops::powf_into(tensor, f64::from_bits(*exponent_bits), destination)
                    } else {
                        let temporary = scratch_tensors[0];
                        metal_ops::cast_into(tensor, temporary)?;
                        metal_ops::powf_into(temporary, f64::from_bits(*exponent_bits), temporary)?;
                        metal_ops::cast_into(temporary, destination)
                    }
                }
                MetalUnaryOp::Cast { .. } => metal_ops::cast_into(tensor, destination),
                kind => metal_ops::unary_promote_into(
                    tensor,
                    match kind {
                        MetalUnaryOp::Neg => metal_ops::UnOp::Neg,
                        MetalUnaryOp::Abs => metal_ops::UnOp::Abs,
                        MetalUnaryOp::Sqrt => metal_ops::UnOp::Sqrt,
                        MetalUnaryOp::Exp => metal_ops::UnOp::Exp,
                        MetalUnaryOp::Log => metal_ops::UnOp::Log,
                        MetalUnaryOp::Sin => metal_ops::UnOp::Sin,
                        MetalUnaryOp::Cos => metal_ops::UnOp::Cos,
                        MetalUnaryOp::Tanh => metal_ops::UnOp::Tanh,
                        MetalUnaryOp::Erf => metal_ops::UnOp::Erf,
                        MetalUnaryOp::Gelu { approximate: true } => metal_ops::UnOp::GeluTanh,
                        MetalUnaryOp::Gelu { approximate: false } => metal_ops::UnOp::Gelu,
                        MetalUnaryOp::Floor => metal_ops::UnOp::Floor,
                        MetalUnaryOp::Ceil => metal_ops::UnOp::Ceil,
                        MetalUnaryOp::Round => metal_ops::UnOp::Round,
                        MetalUnaryOp::Sign => metal_ops::UnOp::Sign,
                        MetalUnaryOp::Relu
                        | MetalUnaryOp::Pow { .. }
                        | MetalUnaryOp::Cast { .. } => unreachable!(),
                    },
                    destination,
                    &scratch_tensors,
                ),
            }
        }
        MetalOp::Binary(kind) => {
            let a = input(0)?.as_metal()?;
            let b = input(1)?.as_metal()?;
            let destination = output(0)?.as_metal()?;
            match kind {
                MetalBinaryOp::Add
                | MetalBinaryOp::Sub
                | MetalBinaryOp::Mul
                | MetalBinaryOp::Div
                | MetalBinaryOp::Maximum
                | MetalBinaryOp::Minimum => {
                    let operation = match kind {
                        MetalBinaryOp::Add => metal_ops::BinOp::Add,
                        MetalBinaryOp::Sub => metal_ops::BinOp::Sub,
                        MetalBinaryOp::Mul => metal_ops::BinOp::Mul,
                        MetalBinaryOp::Div => metal_ops::BinOp::Div,
                        MetalBinaryOp::Maximum => metal_ops::BinOp::Max,
                        MetalBinaryOp::Minimum => metal_ops::BinOp::Min,
                        _ => unreachable!(),
                    };
                    metal_ops::binary_promote_into(a, b, operation, destination, &scratch_tensors)
                }
                MetalBinaryOp::Eq
                | MetalBinaryOp::Gt
                | MetalBinaryOp::Lt
                | MetalBinaryOp::Ge
                | MetalBinaryOp::Le => {
                    let operation = match kind {
                        MetalBinaryOp::Eq => metal_ops::BinOp::Eq,
                        MetalBinaryOp::Gt => metal_ops::BinOp::Gt,
                        MetalBinaryOp::Lt => metal_ops::BinOp::Lt,
                        MetalBinaryOp::Ge => metal_ops::BinOp::Ge,
                        MetalBinaryOp::Le => metal_ops::BinOp::Le,
                        _ => unreachable!(),
                    };
                    metal_ops::compare_into(a, b, operation, destination, &scratch_tensors)
                }
                MetalBinaryOp::Concat { dim } => metal_ops::cat_into(&[a, b], *dim, destination),
                MetalBinaryOp::Matmul => {
                    let MetalCommandPlan::Gemm(requirements) = plan else {
                        return Err("matmul is missing exact GEMM requirements".to_string());
                    };
                    metal_ops::matmul_into(
                        a,
                        b,
                        destination,
                        scratch_tensors.first().copied(),
                        requirements,
                    )
                }
            }
        }
        MetalOp::Where => metal_ops::where_into(
            input(0)?.as_metal()?,
            input(1)?.as_metal()?,
            input(2)?.as_metal()?,
            output(0)?.as_metal()?,
            &scratch_tensors,
        ),
        MetalOp::Argmax { dim } | MetalOp::Argmin { dim } => {
            metal_ops::argreduce_into(
                input(0)?.as_metal()?,
                *dim,
                matches!(op, MetalOp::Argmax { .. }),
                scratch_tensors[0],
            )?;
            metal_ops::cast_into(
                &contiguous_tensor_view(scratch_tensors[0], &output(0)?.shape(), 0)?,
                output(0)?.as_metal()?,
            )
        }
        MetalOp::Cumsum { dim } => {
            metal_ops::cumsum_into(input(0)?.as_metal()?, *dim, output(0)?.as_metal()?)
        }
        MetalOp::ScatterAdd { dim } => {
            let MetalCommandPlan::Indexing(requirements) = plan else {
                return Err("scatter_add is missing exact indexing requirements".to_string());
            };
            let mut index = 0usize;
            let ids_scratch = requirements.ids.as_ref().map(|_| {
                let value = scratch_tensors[index];
                index += 1;
                value
            });
            let accumulator = requirements.accumulator.as_ref().map(|_| {
                let value = scratch_tensors[index];
                index += 1;
                value
            });
            let source_cast = requirements
                .source_cast
                .as_ref()
                .map(|_| scratch_tensors[index]);
            metal_ops::scatter_add_into(
                input(0)?.as_metal()?,
                *dim,
                input(1)?.as_metal()?,
                input(2)?.as_metal()?,
                output(0)?.as_metal()?,
                ids_scratch,
                accumulator,
                source_cast,
            )
        }
        MetalOp::Gather { dim } => {
            let MetalCommandPlan::Indexing(requirements) = plan else {
                return Err("gather is missing exact indexing requirements".to_string());
            };
            metal_ops::gather_into(
                input(0)?.as_metal()?,
                *dim,
                input(1)?.as_metal()?,
                &input(1)?.shape(),
                output(0)?.as_metal()?,
                requirements.ids.as_ref().map(|_| scratch_tensors[0]),
            )
        }
        MetalOp::IndexSelect { dim } => {
            let MetalCommandPlan::Indexing(requirements) = plan else {
                return Err("index_select is missing exact indexing requirements".to_string());
            };
            metal_ops::index_select_into(
                input(0)?.as_metal()?,
                *dim,
                input(1)?.as_metal()?,
                output(0)?.as_metal()?,
                requirements.ids.as_ref().map(|_| scratch_tensors[0]),
            )
        }
        MetalOp::CrossEntropy {
            ignore_index,
            reduction,
            implementation,
        } => match implementation {
            MetalImplementation::Native => {
                let logits = input(0)?.as_metal()?;
                let MetalCommandPlan::CeForward(requirements) = plan else {
                    return Err("cross_entropy is missing exact requirements".to_string());
                };
                let mut scratch_index = 0usize;
                let destination = output(0)?.as_metal()?;
                let loss_destination = if destination.dtype == DType::F32 {
                    destination
                } else {
                    let temporary = scratch_tensors[scratch_index];
                    scratch_index += 1;
                    temporary
                };
                loss::ce_forward_into(
                    logits,
                    input(1)?.as_metal()?,
                    *ignore_index,
                    *reduction,
                    loss_destination,
                    status[0].as_metal()?,
                    scratch_tensors[scratch_index],
                    scratch_tensors[scratch_index + 1],
                )?;
                if destination.dtype != DType::F32 {
                    metal_ops::cast_into(loss_destination, destination)?;
                }
                ce_checks.push(DeferredCeCheck {
                    buffer: status[0].clone(),
                    kind: match reduction {
                        CrossEntropyReduction::Mean => CeCheck::ForwardMean,
                        CrossEntropyReduction::Sum => CeCheck::ForwardSum,
                    },
                    classes: logits.layout.shape()[logits.layout.shape().len() - 1],
                });
                debug_assert_eq!(
                    requirements.classes,
                    logits.layout.shape()[logits.layout.rank() - 1]
                );
                Ok(())
            }
        },
        MetalOp::CrossEntropyBackward {
            ignore_index,
            reduction,
            implementation,
        } => match implementation {
            MetalImplementation::Native => {
                let MetalCommandPlan::CeBackward(_) = plan else {
                    return Err("cross_entropy backward is missing exact requirements".to_string());
                };
                loss::ce_backward_into(
                    input(0)?.as_metal()?,
                    input(1)?.as_metal()?,
                    *ignore_index,
                    *reduction,
                    output(0)?.as_metal()?,
                    status.first().map(Value::as_metal).transpose()?,
                )?;
                if *reduction == CrossEntropyReduction::Mean {
                    ce_checks.push(DeferredCeCheck {
                        buffer: status[0].clone(),
                        kind: CeCheck::BackwardMean,
                        classes: 0,
                    });
                }
                Ok(())
            }
        },
        MetalOp::ChunkedHeadCe {
            ignore_index,
            chunk_size: _,
        } => {
            let MetalCommandPlan::ChunkedHeadForward(plan) = plan else {
                return Err("chunked head CE is missing its fixed destination plan".to_string());
            };
            let logits_root = scratch_tensors[0];
            let chunk_status = scratch_tensors[1];
            let nll_root = scratch_tensors[2];
            let flags_root = scratch_tensors[3];
            let mut scratch_index = 4usize;
            let split_k = if plan.split_k_elements != 0 {
                let value = scratch_tensors[scratch_index];
                scratch_index += 1;
                Some(value)
            } else {
                None
            };
            let destination = output(0)?.as_metal()?;
            let loss_f32 = if destination.dtype == DType::F32 {
                destination
            } else {
                let value = scratch_tensors[scratch_index];
                scratch_index += 1;
                value
            };
            if scratch_index != scratch_tensors.len() {
                return Err("chunked head CE received unexpected scratch views".to_string());
            }
            let aggregate = status
                .first()
                .ok_or_else(|| "chunked head CE aggregate status is missing".to_string())?
                .as_metal()?;
            metal_ops::fill_into(0.0, aggregate)?;
            let x = input(0)?.as_metal()?;
            let weight = input(1)?.as_metal()?;
            let bias = input(2)?.as_metal()?;
            let target = input(3)?.as_metal()?;
            let mut offset = 0usize;
            while offset < plan.rows {
                let length = plan.chunk_len.min(plan.rows - offset);
                let variant = if length == plan.full.rows {
                    &plan.full
                } else {
                    plan.tail
                        .as_ref()
                        .ok_or_else(|| "chunked head CE tail plan is missing".to_string())?
                };
                let x_chunk = contiguous_tensor_view(
                    x,
                    &[length, plan.inner],
                    offset
                        .checked_mul(plan.inner)
                        .ok_or_else(|| "chunked head CE input offset overflow".to_string())?,
                )?;
                let target_chunk = contiguous_tensor_view(target, &[length], offset)?;
                let logits = contiguous_tensor_view(logits_root, &[length, plan.vocab], 0)?;
                let gemm_scratch = gemm_scratch_view(split_k, &variant.head)?;
                crate::gemm::gemm_fused_into(
                    device::MetalDevice::get(),
                    &x_chunk,
                    weight,
                    Some(bias),
                    None,
                    &logits,
                    None,
                    gemm_scratch.as_ref(),
                    length
                        .checked_mul(plan.inner)
                        .ok_or_else(|| "chunked head CE GEMM stride overflow".to_string())?,
                    0,
                    &variant.head,
                )?;
                let chunk_loss = contiguous_tensor_view(chunk_status, &[], 0)?;
                let nll = contiguous_tensor_view(nll_root, &[length], 0)?;
                let flags = contiguous_tensor_view(flags_root, &[length], 0)?;
                loss::ce_forward_into(
                    &logits,
                    &target_chunk,
                    *ignore_index,
                    CrossEntropyReduction::Sum,
                    &chunk_loss,
                    chunk_status,
                    &nll,
                    &flags,
                )?;
                metal_ops::binary_into(aggregate, chunk_status, metal_ops::BinOp::Add, aggregate)?;
                offset += length;
            }
            let total = contiguous_tensor_view(aggregate, &[1], 0)?;
            let count = contiguous_tensor_view(aggregate, &[1], 1)?;
            let loss = contiguous_tensor_view(loss_f32, &[1], 0)?;
            metal_ops::binary_into(&total, &count, metal_ops::BinOp::Div, &loss)?;
            if destination.dtype != DType::F32 {
                metal_ops::cast_into(loss_f32, destination)?;
            }
            ce_checks.push(DeferredCeCheck {
                buffer: status[0].clone(),
                kind: CeCheck::ForwardMean,
                classes: plan.vocab,
            });
            Ok(())
        }
        MetalOp::ChunkedHeadCeBackward {
            ignore_index,
            chunk_size: _,
        } => {
            let MetalCommandPlan::ChunkedHeadBackward(plan) = plan else {
                return Err(
                    "chunked head CE backward is missing its fixed destination plan".to_string(),
                );
            };
            let mut index = 0usize;
            let logits_root = scratch_tensors[index];
            index += 1;
            let grad_f32_root = scratch_tensors[index];
            index += 1;
            let scale = scratch_tensors[index];
            index += 1;
            let gradient_f32 = (input(4)?.dtype() != DType::F32).then(|| {
                let value = scratch_tensors[index];
                index += 1;
                value
            });
            let weight_f32 = (input(1)?.dtype() != DType::F32).then(|| {
                let value = scratch_tensors[index];
                index += 1;
                value
            });
            let weight_t = scratch_tensors[index];
            index += 1;
            let x_f32_root = (input(0)?.dtype() != DType::F32).then(|| {
                let value = scratch_tensors[index];
                index += 1;
                value
            });
            let x_t_root = scratch_tensors[index];
            index += 1;
            let dx_f32_root = (output(0)?.dtype() != DType::F32).then(|| {
                let value = scratch_tensors[index];
                index += 1;
                value
            });
            let dw = scratch_tensors[index];
            index += 1;
            let dw_acc = scratch_tensors[index];
            index += 1;
            let db = scratch_tensors[index];
            index += 1;
            let db_acc = scratch_tensors[index];
            index += 1;
            let split_k = if plan.split_k_elements != 0 {
                let value = scratch_tensors[index];
                index += 1;
                Some(value)
            } else {
                None
            };
            if index != scratch_tensors.len() {
                return Err(
                    "chunked head CE backward received unexpected scratch views".to_string()
                );
            }
            let aggregate = status
                .first()
                .ok_or_else(|| "chunked head CE backward status is missing".to_string())?
                .as_metal()?;
            metal_ops::fill_into(0.0, dw_acc)?;
            metal_ops::fill_into(0.0, db_acc)?;
            let x = input(0)?.as_metal()?;
            let weight = input(1)?.as_metal()?;
            let bias = input(2)?.as_metal()?;
            let target = input(3)?.as_metal()?;
            loss::ce_target_status_into(target, x.dtype, plan.vocab, *ignore_index, aggregate)?;

            let gradient = input(4)?.as_metal()?;
            let gradient = if let Some(gradient_f32) = gradient_f32 {
                metal_ops::cast_into(gradient, gradient_f32)?;
                gradient_f32
            } else {
                gradient
            };
            metal_ops::binary_into(
                &contiguous_tensor_view(gradient, &[1], 0)?,
                &contiguous_tensor_view(aggregate, &[1], 1)?,
                metal_ops::BinOp::Div,
                scale,
            )?;
            let weight_f32 = if let Some(weight_f32) = weight_f32 {
                metal_ops::cast_into(weight, weight_f32)?;
                weight_f32
            } else {
                weight
            };
            metal_ops::permute_into(weight_f32, &[1, 0], weight_t)?;

            // Recompute each logits chunk once and accumulate all destinations.
            let mut offset = 0usize;
            while offset < plan.rows {
                let length = plan.chunk_len.min(plan.rows - offset);
                let variant = if length == plan.full.rows {
                    &plan.full
                } else {
                    plan.tail.as_ref().ok_or_else(|| {
                        "chunked head CE backward tail plan is missing".to_string()
                    })?
                };
                let x_chunk = contiguous_tensor_view(
                    x,
                    &[length, plan.inner],
                    offset
                        .checked_mul(plan.inner)
                        .ok_or_else(|| "chunked head CE input offset overflow".to_string())?,
                )?;
                let target_chunk = contiguous_tensor_view(target, &[length], offset)?;
                let logits = contiguous_tensor_view(logits_root, &[length, plan.vocab], 0)?;
                let head_scratch = gemm_scratch_view(split_k, &variant.head)?;
                crate::gemm::gemm_fused_into(
                    device::MetalDevice::get(),
                    &x_chunk,
                    weight,
                    Some(bias),
                    None,
                    &logits,
                    None,
                    head_scratch.as_ref(),
                    length
                        .checked_mul(plan.inner)
                        .ok_or_else(|| "chunked head CE GEMM stride overflow".to_string())?,
                    0,
                    &variant.head,
                )?;
                let grad_f32 = contiguous_tensor_view(grad_f32_root, &[length, plan.vocab], 0)?;
                loss::ce_backward_scaled_f32_into(
                    &logits,
                    &target_chunk,
                    *ignore_index,
                    scale,
                    &grad_f32,
                )?;

                let dx_destination = contiguous_tensor_view(
                    output(0)?.as_metal()?,
                    &[length, plan.inner],
                    offset
                        .checked_mul(plan.inner)
                        .ok_or_else(|| "chunked head CE dx offset overflow".to_string())?,
                )?;
                let dx_f32 = if let Some(root) = dx_f32_root {
                    contiguous_tensor_view(root, &[length, plan.inner], 0)?
                } else {
                    dx_destination.clone()
                };
                let dx_scratch = gemm_scratch_view(split_k, &variant.dx)?;
                crate::gemm::gemm_fused_into(
                    device::MetalDevice::get(),
                    &grad_f32,
                    weight_t,
                    None,
                    None,
                    &dx_f32,
                    None,
                    dx_scratch.as_ref(),
                    length
                        .checked_mul(plan.vocab)
                        .ok_or_else(|| "chunked head CE dx stride overflow".to_string())?,
                    0,
                    &variant.dx,
                )?;
                if dx_destination.dtype != DType::F32 {
                    metal_ops::cast_into(&dx_f32, &dx_destination)?;
                }

                let x_f32 = if let Some(root) = x_f32_root {
                    let value = contiguous_tensor_view(root, &[length, plan.inner], 0)?;
                    metal_ops::cast_into(&x_chunk, &value)?;
                    value
                } else {
                    x_chunk
                };
                let x_t = contiguous_tensor_view(x_t_root, &[plan.inner, length], 0)?;
                metal_ops::permute_into(&x_f32, &[1, 0], &x_t)?;
                let dw_scratch = gemm_scratch_view(split_k, &variant.dw)?;
                crate::gemm::gemm_fused_into(
                    device::MetalDevice::get(),
                    &x_t,
                    &grad_f32,
                    None,
                    None,
                    dw,
                    None,
                    dw_scratch.as_ref(),
                    plan.inner
                        .checked_mul(length)
                        .ok_or_else(|| "chunked head CE dw stride overflow".to_string())?,
                    length
                        .checked_mul(plan.vocab)
                        .ok_or_else(|| "chunked head CE dw stride overflow".to_string())?,
                    &variant.dw,
                )?;
                metal_ops::binary_into(dw_acc, dw, metal_ops::BinOp::Add, dw_acc)?;
                metal_ops::reduce_into(&grad_f32, &[0], true, fusion::ReduceOp::Sum, db)?;
                metal_ops::binary_into(db_acc, db, metal_ops::BinOp::Add, db_acc)?;
                offset += length;
            }
            let dw_destination = output(1)?.as_metal()?;
            if dw_destination.dtype == DType::F32 {
                crate::kernels::copy_into(device::MetalDevice::get(), dw_acc, dw_destination)?;
            } else {
                metal_ops::cast_into(dw_acc, dw_destination)?;
            }
            let db_acc_flat = contiguous_tensor_view(db_acc, &[plan.vocab], 0)?;
            let db_destination = output(2)?.as_metal()?;
            if db_destination.dtype == DType::F32 {
                crate::kernels::copy_into(
                    device::MetalDevice::get(),
                    &db_acc_flat,
                    db_destination,
                )?;
            } else {
                metal_ops::cast_into(&db_acc_flat, db_destination)?;
            }
            ce_checks.push(DeferredCeCheck {
                buffer: status[0].clone(),
                kind: CeCheck::ForwardMean,
                classes: plan.vocab,
            });
            Ok(())
        }
        MetalOp::Sdpa {
            scale,
            causal,
            window,
            implementation,
        } => match implementation {
            MetalImplementation::Native => {
                let MetalCommandPlan::SdpaForward(_) = plan else {
                    return Err("sdpa is missing exact requirements".to_string());
                };
                flash::forward_into(
                    input(0)?.as_metal()?,
                    input(1)?.as_metal()?,
                    input(2)?.as_metal()?,
                    *scale,
                    *causal,
                    *window,
                    output(0)?.as_metal()?,
                    output(1)?.as_metal()?,
                )?;
                Ok(())
            }
        },
        MetalOp::SdpaBackward {
            scale,
            causal,
            window,
            implementation,
        } => match implementation {
            MetalImplementation::Native => {
                let MetalCommandPlan::SdpaBackward(_) = plan else {
                    return Err("sdpa backward is missing exact requirements".to_string());
                };
                flash::backward_fused_into(
                    input(0)?.as_metal()?,
                    input(1)?.as_metal()?,
                    input(2)?.as_metal()?,
                    input(4)?.as_metal()?,
                    input(5)?.as_metal()?,
                    input(3)?.as_metal()?,
                    *scale,
                    *causal,
                    *window,
                    output(0)?.as_metal()?,
                    output(1)?.as_metal()?,
                    output(2)?.as_metal()?,
                    scratch_tensors[0],
                )?;
                Ok(())
            }
        },
        MetalOp::KdaChunk { scale } => {
            let q = input(0)?.as_metal()?;
            let dims = q.layout.shape();
            let rank = dims.len();
            let (time, dk) = (dims[rank - 2], dims[rank - 1]);
            let dv = input(2)?.as_metal()?.layout.shape()[rank - 1];
            let bh = dims[..rank - 2].iter().product();
            let flat = |value: &Value, width: usize| -> Result<crate::run::MetalTensor, String> {
                Ok(crate::run::MetalTensor {
                    buffer: value.as_metal()?.buffer.clone(),
                    layout: effect_torch_runtime::Layout::contiguous(vec![bh, time, width]),
                    dtype: value.dtype(),
                })
            };
            let destination = crate::run::MetalTensor {
                buffer: output(0)?.as_metal()?.buffer.clone(),
                layout: effect_torch_runtime::Layout::contiguous(vec![bh, time, dv]),
                dtype: q.dtype,
            };
            let MetalCommandPlan::KdaForward(_) = plan else {
                return Err("kda forward is missing exact requirements".to_string());
            };
            kda::forward_into(
                &flat(input(0)?, dk)?,
                &flat(input(1)?, dk)?,
                &flat(input(2)?, dv)?,
                &flat(input(3)?, dk)?,
                &flat(input(4)?, 1)?,
                *scale,
                None,
                &destination,
                None,
                kda::IntoResources::empty(),
            )?;
            Ok(())
        }
        MetalOp::KdaRecurrence { scale, layer } => {
            let context = kv.ok_or_else(|| {
                "kda recurrence requires an executable decode context".to_string()
            })?;
            let geometry = context.schema().kda;
            if geometry.layers == 0 || (*layer as usize) >= geometry.layers {
                return Err("kda recurrence layer is outside the decode geometry".to_string());
            }
            let q = input(0)?.as_metal()?;
            let rank = q.layout.shape().len();
            let time = q.layout.shape()[rank - 2];
            let batch = q.layout.shape()[..rank - 3].iter().product::<usize>();
            if batch != context.slots().len() || time == 0 {
                return Err(format!(
                    "kda recurrence shape has batch {batch} and time {time} for {} decode slots",
                    context.slots().len()
                ));
            }
            let view = |value: &Value,
                        batch_index: usize,
                        width: usize|
             -> Result<crate::run::MetalTensor, String> {
                let tensor = value.as_metal()?;
                let shape = vec![geometry.heads, time, width];
                let per_batch = shape
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .ok_or_else(|| "kda recurrence view size overflow".to_string())?;
                let offset = batch_index
                    .checked_mul(per_batch)
                    .and_then(|value| value.checked_add(tensor.layout.offset()))
                    .ok_or_else(|| "kda recurrence view offset overflow".to_string())?;
                let contiguous = effect_torch_runtime::Layout::contiguous(shape.clone());
                Ok(crate::run::MetalTensor {
                    buffer: tensor.buffer.clone(),
                    layout: effect_torch_runtime::Layout::new(
                        shape,
                        contiguous.strides().to_vec(),
                        offset,
                    ),
                    dtype: tensor.dtype,
                })
            };
            let state_next_root = state_tensors[0];
            let mask_root = staging.first().map(Value::as_metal).transpose()?;
            for (batch_index, slot) in context.slots().iter().enumerate() {
                let state = slot
                    .lock()
                    .map_err(|error| format!("kda recurrence sequence lock poisoned: {error}"))?;
                let initial = state
                    .kda_states
                    .get(*layer as usize)
                    .ok_or_else(|| "kda recurrence persistent state is missing".to_string())?;
                let q = view(input(0)?, batch_index, geometry.head_dim)?;
                let k = view(input(1)?, batch_index, geometry.head_dim)?;
                let v = view(input(2)?, batch_index, geometry.value_dim)?;
                let decay = view(input(3)?, batch_index, geometry.head_dim)?;
                let beta = view(input(4)?, batch_index, 1)?;
                let destination = view(output(0)?, batch_index, geometry.value_dim)?;
                let state_elements = geometry
                    .heads
                    .checked_mul(geometry.head_dim)
                    .and_then(|value| value.checked_mul(geometry.value_dim))
                    .ok_or_else(|| "kda recurrence state size overflow".to_string())?;
                let state_next = contiguous_tensor_view(
                    state_next_root,
                    &[geometry.heads, geometry.head_dim, geometry.value_dim],
                    batch_index
                        .checked_mul(state_elements)
                        .ok_or_else(|| "kda recurrence state offset overflow".to_string())?,
                )?;
                if time == 1 {
                    let MetalCommandPlan::KdaDecode(_) = plan else {
                        return Err(
                            "kda recurrence is missing exact decode requirements".to_string()
                        );
                    };
                    let flatten = |tensor: &crate::run::MetalTensor,
                                   width: usize|
                     -> crate::run::MetalTensor {
                        crate::run::MetalTensor {
                            buffer: tensor.buffer.clone(),
                            layout: effect_torch_runtime::Layout::new(
                                vec![geometry.heads, width],
                                vec![width, 1],
                                tensor.layout.offset(),
                            ),
                            dtype: tensor.dtype,
                        }
                    };
                    kda::decode_into(
                        &flatten(&q, geometry.head_dim),
                        &flatten(&k, geometry.head_dim),
                        &flatten(&v, geometry.value_dim),
                        &flatten(&decay, geometry.head_dim),
                        &flatten(&beta, 1),
                        initial,
                        *scale,
                        &flatten(&destination, geometry.value_dim),
                        &state_next,
                        kda::IntoResources::empty(),
                    )?;
                } else {
                    let MetalCommandPlan::KdaForward(_) = plan else {
                        return Err(
                            "kda recurrence is missing exact forward requirements".to_string()
                        );
                    };
                    let mask_root = mask_root
                        .ok_or_else(|| "kda recurrence advance staging is missing".to_string())?;
                    let mask = contiguous_tensor_view(
                        mask_root,
                        &[1, time, 1],
                        batch_index
                            .checked_mul(time)
                            .ok_or_else(|| "kda recurrence mask offset overflow".to_string())?,
                    )?;
                    write_advance_mask(&mask, state.advance)?;
                    metal_ops::binary_into(
                        &decay,
                        &mask,
                        metal_ops::BinOp::Mul,
                        scratch_tensors[0],
                    )?;
                    metal_ops::binary_into(
                        &beta,
                        &mask,
                        metal_ops::BinOp::Mul,
                        scratch_tensors[1],
                    )?;
                    kda::forward_into(
                        &q,
                        &k,
                        &v,
                        scratch_tensors[0],
                        scratch_tensors[1],
                        *scale,
                        Some(initial),
                        &destination,
                        Some(&state_next),
                        kda::IntoResources::empty(),
                    )?;
                }
            }
            Ok(())
        }
        MetalOp::KdaBackward { scale } => {
            let q = input(0)?.as_metal()?;
            let dims = q.layout.shape();
            let rank = dims.len();
            let (time, dk_width) = (dims[rank - 2], dims[rank - 1]);
            let dv_width = input(2)?.as_metal()?.layout.shape()[rank - 1];
            let bh = dims[..rank - 2].iter().product();
            let flat = |value: &Value, width: usize| -> Result<crate::run::MetalTensor, String> {
                Ok(crate::run::MetalTensor {
                    buffer: value.as_metal()?.buffer.clone(),
                    layout: effect_torch_runtime::Layout::contiguous(vec![bh, time, width]),
                    dtype: value.dtype(),
                })
            };
            let flat_output =
                |index: usize, width: usize| -> Result<crate::run::MetalTensor, String> {
                    Ok(crate::run::MetalTensor {
                        buffer: output(index)?.as_metal()?.buffer.clone(),
                        layout: effect_torch_runtime::Layout::contiguous(vec![bh, time, width]),
                        dtype: q.dtype,
                    })
                };
            let resource_tensors = scratch_tensors
                .iter()
                .map(|tensor| (*tensor).clone())
                .collect::<Vec<_>>();
            let MetalCommandPlan::KdaBackward(_) = plan else {
                return Err("kda backward is missing exact requirements".to_string());
            };
            kda::backward_into(
                &flat(input(0)?, dk_width)?,
                &flat(input(1)?, dk_width)?,
                &flat(input(2)?, dv_width)?,
                &flat(input(3)?, dk_width)?,
                &flat(input(4)?, 1)?,
                &flat(input(5)?, dv_width)?,
                *scale,
                &flat_output(0, dk_width)?,
                &flat_output(1, dk_width)?,
                &flat_output(2, dv_width)?,
                &flat_output(3, dk_width)?,
                &flat_output(4, 1)?,
                kda::IntoResources {
                    staging: &[],
                    status: &[],
                    scratch: &resource_tensors,
                },
            )?;
            Ok(())
        }
        MetalOp::ShortConv1d => {
            let x = input(0)?.as_metal()?;
            let time = x.layout.shape()[x.layout.shape().len() - 2];
            let MetalCommandPlan::ShortConvForward(_) = plan else {
                return Err("shortconv forward is missing exact requirements".to_string());
            };
            shortconv::forward_into(
                x,
                input(1)?.as_metal()?,
                None,
                time,
                output(0)?.as_metal()?,
                None,
                shortconv::IntoResources::empty(),
            )
        }
        MetalOp::ShortConv1dBackwardX => shortconv::backward_x_into(
            input(2)?.as_metal()?,
            input(1)?.as_metal()?,
            output(0)?.as_metal()?,
            shortconv::IntoResources::empty(),
        ),
        MetalOp::ShortConv1dBackwardW => shortconv::backward_w_into(
            input(0)?.as_metal()?,
            input(2)?.as_metal()?,
            input(1)?.as_metal()?.layout.shape()[1],
            output(0)?.as_metal()?,
            shortconv::IntoResources::empty(),
        ),
        MetalOp::ConvState { layer } => {
            let context =
                kv.ok_or_else(|| "conv state requires an executable decode context".to_string())?;
            let geometry = context.schema().conv;
            if geometry.layers == 0 || (*layer as usize) >= geometry.layers {
                return Err("conv state layer is outside the decode geometry".to_string());
            }
            let source = input(0)?.as_metal()?;
            let rank = source.layout.shape().len();
            let time = source.layout.shape()[rank - 2];
            let batch = source.layout.shape()[..rank - 2].iter().product::<usize>();
            if batch != context.slots().len() || time == 0 {
                return Err(format!(
                    "conv state shape has batch {batch} and time {time} for {} decode slots",
                    context.slots().len()
                ));
            }
            let MetalCommandPlan::ShortConvState(_) = plan else {
                return Err("conv state is missing exact requirements".to_string());
            };
            let view =
                |value: &Value, batch_index: usize| -> Result<crate::run::MetalTensor, String> {
                    let tensor = value.as_metal()?;
                    let per_batch = time
                        .checked_mul(geometry.channels)
                        .ok_or_else(|| "conv state view size overflow".to_string())?;
                    let offset = batch_index
                        .checked_mul(per_batch)
                        .and_then(|value| value.checked_add(tensor.layout.offset()))
                        .ok_or_else(|| "conv state view offset overflow".to_string())?;
                    Ok(crate::run::MetalTensor {
                        buffer: tensor.buffer.clone(),
                        layout: effect_torch_runtime::Layout::new(
                            vec![time, geometry.channels],
                            vec![geometry.channels, 1],
                            offset,
                        ),
                        dtype: tensor.dtype,
                    })
                };
            let state_next_root = state_tensors[0];
            for (batch_index, slot) in context.slots().iter().enumerate() {
                let state = slot
                    .lock()
                    .map_err(|error| format!("conv state sequence lock poisoned: {error}"))?;
                let initial = state
                    .conv_states
                    .get(*layer as usize)
                    .ok_or_else(|| "conv persistent state is missing".to_string())?;
                let state_elements = geometry
                    .kernel
                    .saturating_sub(1)
                    .checked_mul(geometry.channels)
                    .ok_or_else(|| "conv state size overflow".to_string())?;
                let state_next = contiguous_tensor_view(
                    state_next_root,
                    &[geometry.kernel.saturating_sub(1), geometry.channels],
                    batch_index
                        .checked_mul(state_elements)
                        .ok_or_else(|| "conv state offset overflow".to_string())?,
                )?;
                shortconv::state_into(
                    &view(input(0)?, batch_index)?,
                    input(1)?.as_metal()?,
                    Some(initial),
                    state.advance,
                    &view(output(0)?, batch_index)?,
                    &state_next,
                    shortconv::IntoResources::empty(),
                )?;
            }
            Ok(())
        }
        MetalOp::PositionEmbedding { seq_len } => {
            let weight = input(0)?.as_metal()?;
            crate::kernels::copy_into(
                device::MetalDevice::get(),
                &crate::run::MetalTensor {
                    buffer: weight.buffer.clone(),
                    layout: weight.layout.narrow(0, 0, *seq_len),
                    dtype: weight.dtype,
                },
                output(0)?.as_metal()?,
            )
        }
        MetalOp::KvAttention {
            scale,
            layer,
            window,
        } => {
            let MetalCommandPlan::KvAttention(requirements) = plan else {
                return Err("KV attention is missing exact paged requirements".to_string());
            };
            let query = input(0)?.as_metal()?;
            if query.layout.shape()[query.layout.shape().len() - 3] != requirements.query_heads
                || query.layout.shape()[query.layout.shape().len() - 2] != requirements.time
                || query.layout.shape()[query.layout.shape().len() - 1] != requirements.head_dim
                || query.layout.shape()[..query.layout.shape().len() - 3]
                    .iter()
                    .product::<usize>()
                    != requirements.batch
            {
                return Err("KV attention invocation does not match its fixed plan".to_string());
            }
            let context = kv.ok_or_else(|| "KV attention requires a decode context".to_string())?;
            let staging_tensors = staging
                .iter()
                .map(|value| value.as_metal().cloned())
                .collect::<Result<Vec<_>, _>>()?;
            context.kv_attention_into(
                *layer,
                query,
                input(1)?.as_metal()?,
                input(2)?.as_metal()?,
                *scale,
                *window,
                output(0)?.as_metal()?,
                &staging_tensors,
            )
        }
        MetalOp::RotaryEmbedding {
            theta,
            cursor_offset,
            layout,
            implementation: _,
        } => {
            let offsets = if *cursor_offset {
                kv.ok_or_else(|| {
                    "rotary embedding: cursor offset requires a decode context".to_string()
                })?
                .slots()
                .iter()
                .map(|slot| {
                    slot.lock().map(|state| state.cursor).map_err(|error| {
                        format!("rotary embedding: sequence lock poisoned: {error}")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![0]
            };
            let x = input(0)?.as_metal()?;
            let MetalCommandPlan::Rotary(_) = plan else {
                return Err("rotary is missing exact requirements".to_string());
            };
            let staging_tensors = staging
                .iter()
                .map(|value| value.as_metal().cloned())
                .collect::<Result<Vec<_>, _>>()?;
            rotary::rotary_into(
                x,
                &offsets,
                *theta,
                1.0,
                *layout,
                output(0)?.as_metal()?,
                rotary::IntoResources {
                    staging: &staging_tensors,
                    status: &[],
                    scratch: &[],
                },
            )
        }
        MetalOp::RotaryEmbeddingBackward {
            theta,
            layout,
            implementation: _,
        } => {
            let gradient = input(0)?.as_metal()?;
            let MetalCommandPlan::Rotary(_) = plan else {
                return Err("rotary backward is missing exact requirements".to_string());
            };
            let staging_tensors = staging
                .iter()
                .map(|value| value.as_metal().cloned())
                .collect::<Result<Vec<_>, _>>()?;
            rotary::rotary_into(
                gradient,
                &[0],
                *theta,
                -1.0,
                *layout,
                output(0)?.as_metal()?,
                rotary::IntoResources {
                    staging: &staging_tensors,
                    status: &[],
                    scratch: &[],
                },
            )
        }
        MetalOp::Linear { implementation } => {
            let x = input(0)?.as_metal()?;
            match implementation {
                MetalImplementation::Native => {
                    let MetalCommandPlan::Linear(requirements) = plan else {
                        return Err("linear is missing exact requirements".to_string());
                    };
                    linear::linear_forward_into(
                        device::MetalDevice::get(),
                        x,
                        input(1)?.as_metal()?,
                        input(2)?.as_metal()?,
                        output(0)?.as_metal()?,
                        scratch_tensors.first().copied(),
                        requirements,
                    )
                }
            }
        }
        MetalOp::QuantizedLinear { .. } => {
            let MetalCommandPlan::QuantizedLinear(requirements) = plan else {
                return Err("quantized linear is missing exact requirements".to_string());
            };
            quantized::linear_into(
                input(0)?.as_metal()?,
                input(1)?.as_metal()?,
                inputs.get(2).map(Value::as_metal).transpose()?,
                output(0)?.as_metal()?,
                requirements,
            )
        }
        MetalOp::QuantizedEmbedding { .. } => {
            let MetalCommandPlan::QuantizedEmbedding(requirements) = plan else {
                return Err("quantized embedding is missing exact requirements".to_string());
            };
            quantized::embedding_into(
                input(0)?.as_metal()?,
                input(1)?.as_metal()?,
                output(0)?.as_metal()?,
                status[0].as_metal()?,
                requirements,
            )?;
            quantized_embedding_checks.push(DeferredQuantizedEmbeddingCheck {
                buffer: status[0].clone(),
                rows: requirements.rows,
            });
            Ok(())
        }
        MetalOp::LinearResidual { implementation } => {
            let x = input(0)?.as_metal()?;
            match implementation {
                MetalImplementation::Native => {
                    let MetalCommandPlan::Linear(requirements) = plan else {
                        return Err("linear residual is missing exact requirements".to_string());
                    };
                    linear::linear_forward_fused_into(
                        device::MetalDevice::get(),
                        x,
                        input(1)?.as_metal()?,
                        input(2)?.as_metal()?,
                        Some(input(3)?.as_metal()?),
                        output(0)?.as_metal()?,
                        None,
                        scratch_tensors.first().copied(),
                        requirements,
                    )
                }
            }
        }
        MetalOp::LinearGelu {
            approximate: _,
            dual,
            implementation,
        } => {
            let x = input(0)?.as_metal()?;
            match implementation {
                MetalImplementation::Native => {
                    let MetalCommandPlan::Linear(requirements) = plan else {
                        return Err("linear gelu is missing exact requirements".to_string());
                    };
                    let second = if *dual {
                        Some(output(1)?.as_metal()?)
                    } else {
                        None
                    };
                    linear::linear_forward_fused_into(
                        device::MetalDevice::get(),
                        x,
                        input(1)?.as_metal()?,
                        input(2)?.as_metal()?,
                        None,
                        output(0)?.as_metal()?,
                        second,
                        scratch_tensors.first().copied(),
                        requirements,
                    )
                }
            }
        }
        MetalOp::LayerNorm {
            eps,
            implementation,
        } => {
            let x = input(0)?.as_metal()?;
            match implementation {
                MetalImplementation::Native => {
                    let MetalCommandPlan::LayerNormForward(_) = plan else {
                        return Err("layer_norm is missing exact requirements".to_string());
                    };
                    layer_norm::ln_forward_into(
                        x,
                        input(1)?.as_metal()?,
                        input(2)?.as_metal()?,
                        *eps,
                        output(0)?.as_metal()?,
                    )
                }
            }
        }
        MetalOp::RmsNorm {
            eps,
            implementation: MetalImplementation::Native,
        } => {
            let MetalCommandPlan::LayerNormForward(requirements) = plan else {
                return Err("rms_norm is missing exact requirements".to_string());
            };
            layer_norm::rms_forward_into(
                input(0)?.as_metal()?,
                inputs.get(1).map(Value::as_metal).transpose()?,
                *eps,
                output(0)?.as_metal()?,
                requirements,
            )
        }
        MetalOp::LayerNormBackward {
            eps,
            implementation,
        } => {
            let x = input(0)?.as_metal()?;
            let weight = input(1)?.as_metal()?;
            let gradient = input(2)?.as_metal()?;
            match implementation {
                MetalImplementation::Native => {
                    let MetalCommandPlan::LayerNormBackward(_) = plan else {
                        return Err("layer_norm backward is missing exact requirements".to_string());
                    };
                    layer_norm::ln_backward_into(
                        x,
                        weight,
                        gradient,
                        *eps,
                        output(0)?.as_metal()?,
                        scratch_tensors[0],
                    )?;
                    let dimensions = (0..x.layout.shape().len() - weight.layout.shape().len())
                        .collect::<Vec<_>>();
                    metal_ops::binary_into(
                        gradient,
                        scratch_tensors[0],
                        metal_ops::BinOp::Mul,
                        scratch_tensors[1],
                    )?;
                    metal_ops::reduce_into(
                        scratch_tensors[1],
                        &dimensions,
                        false,
                        fusion::ReduceOp::Sum,
                        output(1)?.as_metal()?,
                    )?;
                    metal_ops::reduce_into(
                        gradient,
                        &dimensions,
                        false,
                        fusion::ReduceOp::Sum,
                        output(2)?.as_metal()?,
                    )
                }
            }
        }
        MetalOp::Conv1d {
            stride,
            padding,
            dilation,
            groups,
        } => {
            let destination = output(0)?.as_metal()?;
            validate_conv_destination(plan, destination)?;
            crate::conv::conv1d_into(
                device::MetalDevice::get(),
                input(0)?.as_metal()?,
                input(1)?.as_metal()?,
                *stride,
                *padding,
                *dilation,
                *groups,
                destination,
            )
        }
        MetalOp::Conv2d {
            stride,
            padding,
            dilation,
            groups,
        } => {
            let destination = output(0)?.as_metal()?;
            validate_conv_destination(plan, destination)?;
            crate::conv::conv2d_into(
                device::MetalDevice::get(),
                input(0)?.as_metal()?,
                input(1)?.as_metal()?,
                *stride,
                *padding,
                *dilation,
                *groups,
                destination,
            )
        }
        MetalOp::ConvTranspose1d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => {
            let destination = output(0)?.as_metal()?;
            validate_conv_destination(plan, destination)?;
            crate::conv::conv_transpose1d_into(
                device::MetalDevice::get(),
                input(0)?.as_metal()?,
                input(1)?.as_metal()?,
                *stride,
                *padding,
                *output_padding,
                *dilation,
                *groups,
                destination,
            )
        }
        MetalOp::ConvTranspose2d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => {
            let destination = output(0)?.as_metal()?;
            validate_conv_destination(plan, destination)?;
            crate::conv::conv_transpose2d_into(
                device::MetalDevice::get(),
                input(0)?.as_metal()?,
                input(1)?.as_metal()?,
                *stride,
                *padding,
                *output_padding,
                *dilation,
                *groups,
                destination,
            )
        }
        MetalOp::Conv1dBackwardW {
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let destination = output(0)?.as_metal()?;
            validate_conv_destination(plan, destination)?;
            crate::conv::conv1d_backward_w_into(
                device::MetalDevice::get(),
                input(0)?.as_metal()?,
                input(1)?.as_metal()?,
                *kernel,
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
                destination,
            )
        }
        MetalOp::Conv2dBackwardW {
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let destination = output(0)?.as_metal()?;
            validate_conv_destination(plan, destination)?;
            crate::conv::conv2d_backward_w_into(
                device::MetalDevice::get(),
                input(0)?.as_metal()?,
                input(1)?.as_metal()?,
                *kernel,
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
                destination,
            )
        }
        MetalOp::Reduce { op, dims, keepdims } => {
            let source = input(0)?.as_metal()?;
            let operation = match op {
                MetalReduceOp::Sum => fusion::ReduceOp::Sum,
                MetalReduceOp::Mean => fusion::ReduceOp::Mean,
                MetalReduceOp::Max => fusion::ReduceOp::Max,
                MetalReduceOp::Min => fusion::ReduceOp::Min,
                MetalReduceOp::Prod => fusion::ReduceOp::Prod,
            };
            if matches!(source.dtype, DType::F32 | DType::BF16) {
                metal_ops::reduce_into(source, dims, *keepdims, operation, output(0)?.as_metal()?)
            } else {
                metal_ops::cast_into(source, scratch_tensors[0])?;
                metal_ops::reduce_into(
                    scratch_tensors[0],
                    dims,
                    *keepdims,
                    operation,
                    scratch_tensors[1],
                )?;
                metal_ops::cast_into(scratch_tensors[1], output(0)?.as_metal()?)
            }
        }
        MetalOp::Reshape { shape } => {
            let source = input(0)?.as_metal()?;
            let reshaped = crate::run::MetalTensor {
                buffer: source.buffer.clone(),
                layout: effect_torch_runtime::Layout::contiguous(shape.to_vec()),
                dtype: source.dtype,
            };
            crate::kernels::copy_into(
                device::MetalDevice::get(),
                &reshaped,
                output(0)?.as_metal()?,
            )
        }
        MetalOp::Permute { dims } => {
            metal_ops::permute_into(input(0)?.as_metal()?, dims, output(0)?.as_metal()?)
        }
        MetalOp::Slice { ranges } => {
            let source = input(0)?.as_metal()?;
            let mut layout = source.layout.clone();
            for (dimension, &(start, stop, stride)) in ranges.iter().enumerate() {
                let length = stop.saturating_sub(start).div_ceil(stride);
                if length == 0 {
                    return crate::kernels::copy_into(
                        device::MetalDevice::get(),
                        source,
                        output(0)?.as_metal()?,
                    );
                }
                layout = layout.narrow(dimension, start, (length - 1) * stride + 1);
                if stride > 1 {
                    let mut strides = layout.strides().to_vec();
                    strides[dimension] = strides[dimension]
                        .checked_mul(stride)
                        .ok_or_else(|| "slice stride overflow".to_string())?;
                    let mut shape = layout.shape().to_vec();
                    shape[dimension] = length;
                    layout = effect_torch_runtime::Layout::new(shape, strides, layout.offset());
                }
            }
            crate::kernels::copy_into(
                device::MetalDevice::get(),
                &crate::run::MetalTensor {
                    buffer: source.buffer.clone(),
                    layout,
                    dtype: source.dtype,
                },
                output(0)?.as_metal()?,
            )
        }
        MetalOp::BroadcastTo { shape } => {
            metal_ops::broadcast_to_into(input(0)?.as_metal()?, shape, output(0)?.as_metal()?)
        }
        MetalOp::LastTokenRow => {
            let context = kv.ok_or_else(|| {
                "last token row requires an executable decode context".to_string()
            })?;
            let source = input(0)?.as_metal()?;
            let shape = source.layout.shape();
            if shape.len() != 3 || shape[0] != 1 {
                return Err(format!(
                    "last token row: expected source [1, T, V], got {shape:?}"
                ));
            }
            let (time, width) = (shape[1], shape[2]);
            let advance = context
                .slots()
                .first()
                .ok_or_else(|| "last token row requires a decode slot".to_string())?
                .lock()
                .map_err(|error| format!("last token row sequence lock poisoned: {error}"))?
                .advance;
            if advance == 0 || advance > time {
                return Err(format!(
                    "last token row: token advance must be in 1..={time}, got {advance}"
                ));
            }
            let strides = source.layout.strides();
            if strides[2] != 1 {
                return Err(format!(
                    "last token row: innermost source stride must be 1, got {strides:?}"
                ));
            }
            let offset = (advance - 1)
                .checked_mul(strides[1])
                .and_then(|row| row.checked_add(source.layout.offset()))
                .ok_or_else(|| "last token row source offset overflow".to_string())?;
            let row = crate::run::MetalTensor {
                buffer: source.buffer.clone(),
                layout: effect_torch_runtime::Layout::new(vec![width], vec![1], offset),
                dtype: source.dtype,
            };
            crate::kernels::copy_into(device::MetalDevice::get(), &row, output(0)?.as_metal()?)
        }
        MetalOp::PackOptimizerScalars => {
            pack_optimizer_scalars(inputs, &scratch_tensors, output(0)?).map(|_| ())
        }
        MetalOp::AdamW { exprs, .. } => {
            let shape = input(0)?.shape();
            let packed = input(4)?.as_metal()?;
            let input_tensors = inputs[..4]
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            let output_tensors = outputs
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            let strides = vec![fusion::contiguous_strides(&shape); input_tensors.len()];
            crate::run::run_elementwise_into(
                device::MetalDevice::get(),
                exprs,
                &input_tensors,
                &strides,
                Some(&packed.buffer),
                3,
                shape.iter().product(),
                &shape,
                &output_tensors,
            )
        }
        MetalOp::AdamWGroup { parameters, exprs } => {
            let scalar_start = parameters
                .checked_mul(4)
                .ok_or_else(|| "grouped AdamW input count overflow".to_string())?;
            let shape = input(0)?.shape();
            let packed = input(scalar_start)?.as_metal()?;
            let input_tensors = inputs[..scalar_start]
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            let output_tensors = outputs
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            let strides = vec![fusion::contiguous_strides(&shape); input_tensors.len()];
            crate::run::run_elementwise_into(
                device::MetalDevice::get(),
                exprs,
                &input_tensors,
                &strides,
                Some(&packed.buffer),
                3,
                shape.iter().product(),
                &shape,
                &output_tensors,
            )
        }
        MetalOp::Sgd { exprs, .. } => {
            let shape = input(0)?.shape();
            let packed = input(3)?.as_metal()?;
            let input_tensors = inputs[..3]
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            let output_tensors = outputs
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            let strides = vec![fusion::contiguous_strides(&shape); input_tensors.len()];
            crate::run::run_elementwise_into(
                device::MetalDevice::get(),
                exprs,
                &input_tensors,
                &strides,
                Some(&packed.buffer),
                2,
                shape.iter().product(),
                &shape,
                &output_tensors,
            )
        }
        MetalOp::FusedElementwise {
            strides,
            shape,
            exprs,
        } => {
            let input_tensors = inputs
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            let output_tensors = outputs
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            crate::run::run_elementwise_into(
                device::MetalDevice::get(),
                exprs,
                &input_tensors,
                strides,
                None,
                0,
                shape.iter().product(),
                shape,
                &output_tensors,
            )
        }
        MetalOp::FusedReduce {
            strides,
            in_shape,
            expr,
            op,
            dims,
            keepdims,
            shape,
        } => {
            let input_tensors = inputs
                .iter()
                .map(Value::as_metal)
                .collect::<Result<Vec<_>, _>>()?;
            crate::run::run_reduce_into(
                device::MetalDevice::get(),
                *op,
                expr,
                &input_tensors,
                strides,
                in_shape,
                dims,
                *keepdims,
                shape,
                output(0)?.as_metal()?,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_graph::{Device, LeafSlot};

    fn leaf(values: Vec<f32>) -> Arc<Node> {
        let length = values.len();
        leaf_shape(values, vec![length])
    }

    fn leaf_shape(values: Vec<f32>, shape: Vec<usize>) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            crate::run::MetalTensor::from_f32(device::MetalDevice::get(), values, shape),
        )))))
        .unwrap()
    }

    fn leaf_u32(values: &[u32], shape: Vec<usize>) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            crate::run::MetalTensor {
                buffer: device::MetalDevice::get().alloc_with_data_u32(values),
                layout: effect_torch_runtime::Layout::contiguous(shape),
                dtype: DType::U32,
            },
        )))))
        .unwrap()
    }

    fn leaf_u8(values: &[u8], shape: Vec<usize>) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            crate::run::MetalTensor {
                buffer: device::MetalDevice::get().upload_bytes(values),
                layout: effect_torch_runtime::Layout::contiguous(shape),
                dtype: DType::U8,
            },
        )))))
        .unwrap()
    }

    fn leaf_i64(values: &[i64], shape: Vec<usize>) -> Arc<Node> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            crate::run::MetalTensor {
                buffer: device::MetalDevice::get().upload_bytes(&bytes),
                layout: effect_torch_runtime::Layout::contiguous(shape),
                dtype: DType::I64,
            },
        )))))
        .unwrap()
    }

    struct QuantizedFixture {
        codec: GgmlKQuant,
        bytes: Vec<u8>,
        group_width: usize,
        expected_groups: Vec<f32>,
    }

    impl QuantizedFixture {
        fn expected(&self) -> Vec<f32> {
            self.expected_groups
                .iter()
                .flat_map(|value| std::iter::repeat_n(*value, self.group_width))
                .collect()
        }
    }

    fn quantized_fixtures() -> Vec<QuantizedFixture> {
        let mut q2 = vec![0u8; 84];
        for (index, value) in q2[..16].iter_mut().enumerate() {
            *value = 0x10 | ((index % 15 + 1) as u8);
        }
        q2[16..80].fill(0xe4);
        q2[80..84].copy_from_slice(&[0x00, 0x3c, 0x00, 0x38]);

        let mut q3 = vec![0u8; 110];
        q3[..32].fill(0xaa);
        q3[32..96].fill(0xe4);
        q3[96..104].fill(0x11);
        q3[104..108].fill(0xaa);
        q3[108..110].copy_from_slice(&[0x00, 0x3c]);

        let scale_min = |length: usize, quant_offset: usize, high_offset: Option<usize>| {
            let mut bytes = vec![0u8; length];
            bytes[..4].copy_from_slice(&[0x00, 0x3c, 0x00, 0x38]);
            bytes[4..8].fill(1);
            bytes[8..12].fill(2);
            bytes[12..16].fill(0x21);
            if let Some(offset) = high_offset {
                bytes[offset..offset + 32].fill(0x99);
            }
            bytes[quant_offset..].fill(0xe4);
            bytes
        };

        let mut q6 = vec![0u8; 210];
        q6[..192].fill(0xe4);
        for (index, value) in q6[192..208].iter_mut().enumerate() {
            *value = (index + 1) as u8;
        }
        q6[208..210].copy_from_slice(&[0x00, 0x38]);

        vec![
            QuantizedFixture {
                codec: GgmlKQuant::Q2K,
                bytes: q2,
                group_width: 16,
                expected_groups: vec![
                    -0.5, -0.5, 2.5, 3.5, 9.5, 11.5, 20.5, 23.5, -0.5, -0.5, 10.5, 11.5, 25.5,
                    27.5, 44.5, 2.5,
                ],
            },
            QuantizedFixture {
                codec: GgmlKQuant::Q3K,
                bytes: q3,
                group_width: 16,
                expected_groups: vec![
                    -4.0, -4.0, 1.0, 1.0, -2.0, -2.0, 3.0, 3.0, -4.0, -4.0, 1.0, 1.0, -2.0, -2.0,
                    3.0, 3.0,
                ],
            },
            QuantizedFixture {
                codec: GgmlKQuant::Q4K,
                bytes: scale_min(144, 16, None),
                group_width: 32,
                expected_groups: vec![3.0, 13.0, 3.0, 13.0, 3.0, 13.0, 3.0, 13.0],
            },
            QuantizedFixture {
                codec: GgmlKQuant::Q5K,
                bytes: scale_min(176, 48, Some(16)),
                group_width: 32,
                expected_groups: vec![19.0, 13.0, 3.0, 29.0, 19.0, 13.0, 3.0, 29.0],
            },
            QuantizedFixture {
                codec: GgmlKQuant::Q6K,
                bytes: q6,
                group_width: 16,
                expected_groups: vec![
                    -14.0, -28.0, -18.0, -24.0, 35.0, 42.0, 105.0, 120.0, -126.0, -140.0, -66.0,
                    -72.0, 91.0, 98.0, 225.0, 240.0,
                ],
            },
        ]
    }

    fn assert_quantized_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1e-4 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: {actual} differs from {expected} by more than {tolerance}"
            );
        }
    }

    fn assert_quantized_embedding_has_status_validation(compilation: &MetalCompilation) {
        let commands = compilation.executable.commands();
        assert_eq!(commands.len(), 1);
        let command = commands[0];
        assert!(matches!(
            operation(command),
            (
                MetalOp::QuantizedEmbedding { .. },
                MetalCommandPlan::QuantizedEmbedding(_)
            )
        ));
        assert_eq!(command.status.len(), 1);
        assert_eq!(
            compilation.executable.physical.as_ref(),
            [
                MetalPhysicalCommand::Encode(command.id),
                MetalPhysicalCommand::StatusGate(command.id),
                MetalPhysicalCommand::Commit,
                MetalPhysicalCommand::Complete,
            ]
        );
        assert_eq!(compilation.executable.diagnostics.command_count, 1);
        assert_eq!(compilation.executable.diagnostics.synchronization_count, 1);
        assert!(compilation.executable.program.values.iter().any(|value| {
            matches!(
                &value.declaration.storage,
                ValueStorage::Fixed {
                    class: StorageClass::DeviceStatus,
                    ..
                } | ValueStorage::Planned {
                    class: StorageClass::DeviceStatus,
                    ..
                }
            )
        }));
    }

    fn scalar(value: f32) -> Arc<Node> {
        Node::new(NodeKind::Reshape {
            a: leaf(vec![value]),
            shape: Vec::new(),
        })
        .unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{actual} differs from {expected}"
            );
        }
    }

    fn options(optimize: bool) -> CompileOptions {
        CompileOptions {
            optimize,
            ..CompileOptions::default()
        }
    }

    fn compile_graph(roots: &[Arc<Node>], optimize: bool) -> MetalCompilation {
        compile(
            roots,
            options(optimize),
            1024,
            MetalEnvironment {
                private_intermediates: false,
                mma: true,
            },
        )
        .unwrap()
    }

    #[test]
    fn quantized_linear_executes_every_codec_for_single_and_multiple_vectors() {
        for fixture in quantized_fixtures() {
            let columns = 512usize;
            let decoded = fixture
                .expected()
                .into_iter()
                .cycle()
                .take(columns)
                .collect::<Vec<_>>();
            for vectors in [1usize, 3, 4, 5, 16, 17] {
                let input_values = (0..vectors * columns)
                    .map(|index| ((index * 7 + index / columns * 3) % 17) as f32 * 0.125 - 1.0)
                    .collect::<Vec<_>>();
                let packed = fixture
                    .bytes
                    .iter()
                    .copied()
                    .cycle()
                    .take(fixture.bytes.len() * 4)
                    .collect::<Vec<_>>();
                let bias = (vectors == 1).then(|| leaf_shape(vec![0.25, -0.5], vec![2]));
                let root = Node::new(NodeKind::QuantizedLinear {
                    x: leaf_shape(input_values.clone(), vec![vectors, columns]),
                    weight: leaf_u8(&packed, vec![2, fixture.bytes.len() * 2]),
                    bias,
                    codec: fixture.codec,
                    weight_shape: [2, columns],
                })
                .unwrap();
                let compilation = compile_graph(&[root], false);
                let actual = run(&compilation)[0].to_f32_vec().unwrap();
                let mut expected = Vec::with_capacity(vectors * 2);
                for vector in 0..vectors {
                    let dot = input_values[vector * columns..(vector + 1) * columns]
                        .iter()
                        .zip(&decoded)
                        .fold(0.0f32, |sum, (&input, &weight)| sum + input * weight);
                    if vectors == 1 {
                        expected.extend([dot + 0.25, dot - 0.5]);
                    } else {
                        expected.extend([dot, dot]);
                    }
                }
                assert_quantized_close(&actual, &expected);
            }
        }
    }

    #[test]
    fn quantized_embedding_executes_every_codec_for_u32_and_i64_indexes() {
        for fixture in quantized_fixtures() {
            let expected_row = fixture.expected();
            let mut packed = vec![0u8; fixture.bytes.len() * 3];
            packed[fixture.bytes.len()..fixture.bytes.len() * 2].copy_from_slice(&fixture.bytes);
            for i64_indexes in [false, true] {
                let indexes = if i64_indexes {
                    leaf_i64(&[1, 1], vec![2])
                } else {
                    leaf_u32(&[1, 1], vec![2])
                };
                let root = Node::new(NodeKind::QuantizedEmbedding {
                    indexes,
                    weight: leaf_u8(&packed, vec![3, fixture.bytes.len()]),
                    codec: fixture.codec,
                    weight_shape: [3, 256],
                    padding_index: None,
                })
                .unwrap();
                let compilation = compile_graph(&[root], false);
                assert_quantized_embedding_has_status_validation(&compilation);
                let actual = run(&compilation)[0].to_f32_vec().unwrap();
                let expected = expected_row
                    .iter()
                    .chain(&expected_row)
                    .copied()
                    .collect::<Vec<_>>();
                assert_quantized_close(&actual, &expected);
            }
        }
    }

    #[test]
    fn mixed_codec_graph_has_metal_commands_without_fallback_allocations_or_intermediate_syncs() {
        let fixtures = quantized_fixtures();
        let q2 = &fixtures[0];
        let q6 = &fixtures[4];
        let linear = Node::new(NodeKind::QuantizedLinear {
            x: leaf_shape(vec![1.0; 256], vec![1, 256]),
            weight: leaf_u8(&q2.bytes, vec![1, q2.bytes.len()]),
            bias: None,
            codec: q2.codec,
            weight_shape: [1, 256],
        })
        .unwrap();
        let embedding = Node::new(NodeKind::QuantizedEmbedding {
            indexes: leaf_u32(&[0], vec![1]),
            weight: leaf_u8(&q6.bytes, vec![1, q6.bytes.len()]),
            codec: q6.codec,
            weight_shape: [1, 256],
            padding_index: None,
        })
        .unwrap();
        let compilation = compile_graph(&[linear, embedding], false);
        assert_eq!(compilation.executable.diagnostics.command_count, 2);
        assert_eq!(compilation.executable.diagnostics.pipeline_count, 2);
        assert_eq!(compilation.executable.diagnostics.synchronization_count, 1);
        let commands = compilation.executable.commands();
        assert!(commands[0].status.is_empty());
        assert_eq!(commands[1].status.len(), 1);
        assert_eq!(
            compilation.executable.physical.as_ref(),
            [
                MetalPhysicalCommand::Encode(commands[0].id),
                MetalPhysicalCommand::Encode(commands[1].id),
                MetalPhysicalCommand::StatusGate(commands[1].id),
                MetalPhysicalCommand::Commit,
                MetalPhysicalCommand::Complete,
            ]
        );
        assert!(compilation.executable.program.values.iter().any(|value| {
            matches!(
                &value.declaration.storage,
                ValueStorage::Fixed {
                    class: StorageClass::DeviceStatus,
                    ..
                } | ValueStorage::Planned {
                    class: StorageClass::DeviceStatus,
                    ..
                }
            )
        }));
        assert!(matches!(
            operation(commands[0]),
            (
                MetalOp::QuantizedLinear {
                    codec: GgmlKQuant::Q2K,
                    ..
                },
                MetalCommandPlan::QuantizedLinear(_)
            )
        ));
        assert!(matches!(
            operation(commands[1]),
            (
                MetalOp::QuantizedEmbedding {
                    codec: GgmlKQuant::Q6K,
                    ..
                },
                MetalCommandPlan::QuantizedEmbedding(_)
            )
        ));

        let allocations = device::EXECUTABLE_ALLOCATION_ATTEMPTS.load(Ordering::Relaxed);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        let dispatches = device::DISPATCHES.load(Ordering::Relaxed);
        let synchronizations = device::SYNCS.load(Ordering::Relaxed);
        let output = run(&compilation);
        assert_eq!(
            device::EXECUTABLE_ALLOCATION_ATTEMPTS.load(Ordering::Relaxed),
            allocations
        );
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
        assert!(device::DISPATCHES.load(Ordering::Relaxed) >= dispatches + 2);
        assert!(device::SYNCS.load(Ordering::Relaxed) > synchronizations);

        let q2_expected = q2.expected().iter().sum::<f32>();
        assert_quantized_close(&output[0].to_f32_vec().unwrap(), &[q2_expected]);
        assert_quantized_close(&output[1].to_f32_vec().unwrap(), &q6.expected());
    }

    #[test]
    fn quantized_embedding_rejects_invalid_indexes_and_execution_observes_cancellation() {
        let fixture = &quantized_fixtures()[0];
        for indexes in [
            leaf_i64(&[-1], vec![1]),
            leaf_i64(&[1], vec![1]),
            leaf_u32(&[1], vec![1]),
        ] {
            let root = Node::new(NodeKind::QuantizedEmbedding {
                indexes,
                weight: leaf_u8(&fixture.bytes, vec![1, fixture.bytes.len()]),
                codec: fixture.codec,
                weight_shape: [1, 256],
                padding_index: None,
            })
            .unwrap();
            let compilation = compile_graph(&[root], false);
            assert_quantized_embedding_has_status_validation(&compilation);
            let error = match execute(
                &compilation.executable,
                &[],
                &compilation.generated_bindings,
                &CancellationFlag::new(),
                None,
            ) {
                Ok(_) => panic!("invalid embedding index unexpectedly succeeded"),
                Err(error) => error,
            };
            assert!(error.contains("index is outside 0..1"));
        }

        let root = Node::new(NodeKind::QuantizedLinear {
            x: leaf_shape(vec![1.0; 256], vec![1, 256]),
            weight: leaf_u8(&fixture.bytes, vec![1, fixture.bytes.len()]),
            bias: None,
            codec: fixture.codec,
            weight_shape: [1, 256],
        })
        .unwrap();
        let compilation = compile_graph(&[root], false);
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        assert_eq!(
            execute(
                &compilation.executable,
                &[],
                &compilation.generated_bindings,
                &cancelled,
                None,
            )
            .err()
            .unwrap(),
            "operation aborted"
        );
    }

    #[test]
    fn compile_time_constant_is_ready_for_execution_on_another_thread() {
        let constant = Node::new(NodeKind::Full {
            shape: vec![2],
            value: 2.0,
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add {
            a: leaf(vec![1.0, 3.0]),
            b: constant,
        })
        .unwrap();

        let compilation = compile_graph(&[root], false);

        let values = std::thread::spawn(move || run(&compilation)[0].to_f32_vec().unwrap())
            .join()
            .unwrap();
        assert_eq!(values, [3.0, 5.0]);
    }

    #[test]
    fn scalar_bindings_are_written_into_planned_invocation_staging() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2],
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let scalar = Node::new(NodeKind::ScalarInput {
            slot: 1,
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add {
            a: input,
            b: scalar,
        })
        .unwrap();
        let compilation = compile_graph(&[root], false);
        assert_eq!(compilation.executable.signature.bindings.len(), 1);
        assert_eq!(
            compilation.executable.signature.bindings[0].layout,
            effect_torch_runtime::BindingLayoutPolicy::Require(
                effect_torch_runtime::LayoutConstraint::ZeroOffsetContiguous
            )
        );
        assert_eq!(
            compilation.executable.signature.invocation.scalars,
            [effect_torch_runtime::ScalarDecl {
                name: "slot_1".to_string(),
                scalar_type: effect_torch_runtime::ScalarType::F32,
            }]
        );
        assert!(compilation
            .executable
            .signature
            .invocation
            .runtime_values
            .is_empty());
        let scalar_value = compilation.executable.scalar_bindings[0].value;
        let Location::Segment { segment, .. } =
            compilation.executable.memory.locations[scalar_value.index()]
        else {
            panic!("scalar binding was not assigned a planned segment");
        };
        assert_eq!(
            compilation.executable.memory.segments[segment.index()].ownership,
            SegmentOwnership::InvocationStaging
        );

        let input = Value(crate::run::MetalTensor::from_f32(
            device::MetalDevice::get(),
            vec![1.0, 2.0],
            vec![2],
        ));
        let output = execute_with_scalars(
            &compilation.executable,
            &[input],
            &compilation.generated_bindings,
            &[3.5],
            &CancellationFlag::new(),
        )
        .unwrap();
        assert_eq!(output[0].to_f32_vec().unwrap(), [4.5, 5.5]);
    }

    #[test]
    fn learned_position_decode_signatures_keep_state_cursors_internal() {
        for batch in [1usize, 3] {
            let input = |slot| {
                Node::new(NodeKind::Input {
                    slot,
                    shape: vec![batch, 1, 2, 4],
                    dtype: DType::F32,
                    device: Device::Metal,
                })
                .unwrap()
            };
            let attention = Node::new(NodeKind::Sdpa {
                q: input(0),
                k: input(1),
                v: input(2),
                scale: 0.5,
                causal: true,
                window: effect_torch_graph::AttentionWindow::Inherit,
            })
            .unwrap();
            let positions = Node::new(NodeKind::PositionEmbedding {
                weight: Node::new(NodeKind::Zeros {
                    shape: vec![128, 6],
                    dtype: DType::F32,
                    device: Device::Metal,
                })
                .unwrap(),
                seq_len: 2,
            })
            .unwrap();
            let (roots, geometry) = effect_torch_compiler::specialize_decode(
                &[attention.clone(), positions, attention],
                None,
                batch,
                false,
            )
            .unwrap();
            let compilation = compile_with_state(
                &roots,
                options(false),
                1024,
                MetalEnvironment {
                    private_intermediates: false,
                    mma: true,
                },
                Some(KvStateSchema {
                    max_tokens: 16,
                    block_size: 16,
                    kv_dtype: DType::F32,
                    window: None,
                    batch,
                    layers: geometry.layers,
                    kv_heads: geometry.kv_heads,
                    head_dim: geometry.head_dim,
                    kda: KdaGeometry::default(),
                    conv: ConvGeometry::default(),
                    cursor_slot: geometry.cursor_slot,
                    cursor_tensor: geometry.cursor_tensor,
                }),
            )
            .unwrap();
            let signature = &compilation.executable.signature;

            assert_eq!(signature.bindings.len(), 3);
            assert!(signature.bindings.iter().all(|binding| {
                binding.layout
                    == effect_torch_runtime::BindingLayoutPolicy::Require(
                        effect_torch_runtime::LayoutConstraint::ZeroOffsetContiguous,
                    )
            }));
            assert!(signature.invocation.scalars.is_empty());
            assert_eq!(signature.invocation.runtime_values.len(), 1);
            assert_eq!(
                signature.invocation.runtime_values[0].kind,
                if batch == 1 {
                    effect_torch_runtime::RuntimeValueKind::U64 {
                        min: 0,
                        max: u32::MAX as u64,
                    }
                } else {
                    effect_torch_runtime::RuntimeValueKind::U32Array {
                        max_len: batch,
                        element_min: 0,
                        element_max: u32::MAX,
                    }
                }
            );
            assert_eq!(signature.outputs.len(), 3);
            assert_eq!(signature.outputs[0], signature.outputs[2]);
        }
    }

    #[test]
    fn declared_strided_binding_is_rejected_by_the_metal_signature_contract() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2, 2],
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let root = Node::new(NodeKind::Neg { a: input }).unwrap();
        let compilation = compile_graph(&[root], false);
        assert_eq!(
            compilation.executable.signature.bindings[0].layout,
            effect_torch_runtime::BindingLayoutPolicy::Require(
                effect_torch_runtime::LayoutConstraint::ZeroOffsetContiguous
            )
        );

        let mut input = crate::run::MetalTensor::from_f32(
            device::MetalDevice::get(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        input.layout = input.layout.permute(&[1, 0]);
        let error = execute_with_scalars(
            &compilation.executable,
            &[Value(input)],
            &compilation.generated_bindings,
            &[],
            &CancellationFlag::new(),
        )
        .err()
        .unwrap();

        assert_eq!(error, "execute: binding 0 has the wrong layout");
    }

    #[test]
    fn bounded_batch_inputs_are_zero_padded_in_planned_staging() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2, 2],
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add {
            a: input,
            b: leaf_shape(vec![1.0; 4], vec![2, 2]),
        })
        .unwrap();
        let schema = KvStateSchema {
            max_tokens: 64,
            block_size: 16,
            kv_dtype: DType::F32,
            window: None,
            batch: 2,
            layers: 0,
            kv_heads: 0,
            head_dim: 0,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            cursor_slot: u32::MAX,
            cursor_tensor: false,
        };
        let compilation = compile_graph_with_state(&[root], false, schema);
        let padded = compilation.executable.padded_bindings[0].value;
        let Location::Segment { segment, .. } =
            compilation.executable.memory.locations[padded.index()]
        else {
            panic!("bounded input was not assigned a planned segment");
        };
        assert_eq!(
            compilation.executable.memory.segments[segment.index()].ownership,
            SegmentOwnership::InvocationStaging
        );
        let context = TestDecodeContext {
            schema,
            slots: Vec::new(),
        };
        let input = Value(crate::run::MetalTensor::from_f32(
            device::MetalDevice::get(),
            vec![2.0, 3.0],
            vec![1, 2],
        ));
        let output = execute_stateful(
            &compilation.executable,
            &[input],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            &context,
            &|| true,
        )
        .unwrap();
        assert_eq!(output[0].to_f32_vec().unwrap(), [3.0, 4.0, 1.0, 1.0]);
    }

    fn compile_graph_with_state(
        roots: &[Arc<Node>],
        optimize: bool,
        schema: KvStateSchema,
    ) -> MetalCompilation {
        compile_with_state(
            roots,
            options(optimize),
            1024,
            MetalEnvironment {
                private_intermediates: false,
                mma: true,
            },
            Some(schema),
        )
        .unwrap()
    }

    fn run(compilation: &MetalCompilation) -> Vec<Value> {
        execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            None,
        )
        .unwrap()
    }

    fn operation(command: &MetalCommand) -> (&MetalOp, &MetalCommandPlan) {
        command
            .kind
            .operation()
            .expect("encoded command is an operation")
    }

    fn output_ids(command: &MetalCommand) -> Vec<ValueId> {
        command.outputs.iter().map(|output| output.value).collect()
    }

    fn release_ids(command: &MetalCommand) -> &[ValueId] {
        let MetalInstruction::Operation { release, .. } = &command.kind else {
            unreachable!("encoded command is an operation")
        };
        release
    }

    fn random_seed_token(command: &MetalCommand) -> u64 {
        let MetalInstruction::Operation {
            random_seed_token, ..
        } = command.kind
        else {
            unreachable!("encoded command is an operation")
        };
        random_seed_token
    }

    #[test]
    fn command_order_is_dense_deterministic_and_shared_nodes_are_lowered_once() {
        let x = leaf(vec![1.0, 2.0]);
        let y = leaf(vec![3.0, 4.0]);
        let shared = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let left = Node::new(NodeKind::Neg { a: shared.clone() }).unwrap();
        let right = Node::new(NodeKind::Tanh { a: shared }).unwrap();
        let compilation = compile_graph(&[left, right], false);
        assert_eq!(
            compilation
                .executable
                .commands()
                .iter()
                .map(|command| operation(command).0.name())
                .collect::<Vec<_>>(),
            ["binary", "unary", "unary"]
        );
        assert!(compilation
            .executable
            .program
            .values
            .iter()
            .enumerate()
            .all(|(index, _)| ValueId::from_index(index).is_some()));
        assert_eq!(
            compilation.executable.commands()[1].inputs[0].value,
            compilation.executable.commands()[0].outputs[0].value
        );
        assert_eq!(
            compilation.executable.commands()[2].inputs[0].value,
            compilation.executable.commands()[0].outputs[0].value
        );
    }

    #[test]
    fn executable_keeps_values_but_not_graph_nodes_or_leaf_slots() {
        let leaf = leaf(vec![1.0, 2.0]);
        let leaf_slot = match &leaf.kind {
            NodeKind::Leaf(slot) => Arc::downgrade(slot),
            _ => unreachable!(),
        };
        let root = Node::new(NodeKind::Neg {
            a: Node::new(NodeKind::Neg { a: leaf }).unwrap(),
        })
        .unwrap();
        let weak_root = Arc::downgrade(&root);
        let compilation = compile_graph(std::slice::from_ref(&root), true);
        assert!(matches!(
            operation(compilation.executable.commands()[0]).0,
            MetalOp::FusedElementwise { .. }
        ));
        drop(root);
        assert!(weak_root.upgrade().is_none());
        assert!(leaf_slot.upgrade().is_none());
        assert_eq!(run(&compilation)[0].to_f32_vec().unwrap(), [1.0, 2.0]);
    }

    #[test]
    fn shared_random_command_is_shared_per_run_and_fresh_across_runs() {
        let random = Node::new(NodeKind::Randn {
            shape: vec![16],
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let compilation = compile_graph(&[random.clone(), random], false);
        assert_eq!(compilation.executable.commands().len(), 1);
        assert_eq!(
            compilation.executable.program.outputs[0],
            compilation.executable.program.outputs[1]
        );
        let first = run(&compilation);
        let second = run(&compilation);
        let first_values = first[0].to_f32_vec().unwrap();
        assert_eq!(first_values, first[1].to_f32_vec().unwrap());
        assert_eq!(
            second[0].to_f32_vec().unwrap(),
            second[1].to_f32_vec().unwrap()
        );
        assert_ne!(first_values, second[0].to_f32_vec().unwrap());
    }

    #[test]
    fn optimized_and_unoptimized_executables_have_numerical_parity() {
        let root = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add {
                a: leaf(vec![1.0, -2.0, 3.0, -4.0]),
                b: leaf(vec![0.5, 0.25, -0.5, 2.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let optimized = compile_graph(std::slice::from_ref(&root), true);
        let unoptimized = compile_graph(std::slice::from_ref(&root), false);
        assert!(matches!(root.kind, NodeKind::Tanh { .. }));
        assert_eq!(optimized.executable.commands().len(), 1);
        assert!(matches!(
            operation(optimized.executable.commands()[0]).0,
            MetalOp::FusedElementwise { .. }
        ));
        assert_eq!(
            run(&optimized)[0].to_f32_vec().unwrap(),
            run(&unoptimized)[0].to_f32_vec().unwrap()
        );
        assert!(optimized.executable.commands().len() < unoptimized.executable.commands().len());
    }

    #[test]
    fn direct_elementwise_reduce_region_preserves_topology_and_warming() {
        let root = Node::new(NodeKind::Sum {
            a: Node::new(NodeKind::Tanh {
                a: Node::new(NodeKind::Add {
                    a: leaf_shape(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]),
                    b: leaf_shape(vec![0.5, 0.25, -0.5, 2.0], vec![2, 2]),
                })
                .unwrap(),
            })
            .unwrap(),
            dims: vec![1],
            keepdims: false,
        })
        .unwrap();
        let optimized = compile_graph(std::slice::from_ref(&root), true);
        let unoptimized = compile_graph(std::slice::from_ref(&root), false);
        assert_eq!(optimized.executable.commands().len(), 1);
        assert_eq!(unoptimized.executable.commands().len(), 3);
        assert!(matches!(
            operation(optimized.executable.commands()[0]).0,
            MetalOp::FusedReduce { .. }
        ));
        assert!(matches!(
            operation(optimized.executable.commands()[0]).1,
            MetalCommandPlan::Direct
        ));
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        let actual = run(&optimized)[0].to_f32_vec().unwrap();
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
        assert_close(&actual, &run(&unoptimized)[0].to_f32_vec().unwrap());
    }

    #[test]
    fn direct_multi_output_region_routes_roots_without_selector_commands() {
        let prefix = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add {
                a: leaf(vec![1.0, -2.0, 3.0, -4.0]),
                b: leaf(vec![0.5, 0.25, -0.5, 2.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let left = Node::new(NodeKind::Exp {
            a: Node::new(NodeKind::Neg { a: prefix.clone() }).unwrap(),
        })
        .unwrap();
        let right = Node::new(NodeKind::Sin {
            a: Node::new(NodeKind::Mul {
                a: prefix,
                b: leaf(vec![2.0, 1.5, -0.5, 0.25]),
            })
            .unwrap(),
        })
        .unwrap();
        let roots = [left, right];
        let optimized = compile_graph(&roots, true);
        let unoptimized = compile_graph(&roots, false);
        assert_eq!(optimized.executable.commands().len(), 1);
        assert_eq!(optimized.executable.commands()[0].outputs.len(), 2);
        assert!(matches!(
            operation(optimized.executable.commands()[0]).0,
            MetalOp::FusedElementwise { .. }
        ));
        assert_eq!(
            optimized.executable.program.outputs.as_ref(),
            output_ids(optimized.executable.commands()[0])
        );
        for (actual, expected) in run(&optimized).iter().zip(run(&unoptimized)) {
            assert_close(
                &actual.to_f32_vec().unwrap(),
                &expected.to_f32_vec().unwrap(),
            );
        }
    }

    #[test]
    fn direct_linear_epilogues_preserve_single_and_dual_output_routing() {
        let linear = || {
            Node::new(NodeKind::Linear {
                x: leaf_shape(vec![1.0, 2.0, -1.0, 0.5], vec![1, 2, 2]),
                weight: leaf_shape(vec![0.5, -1.0, 2.0, 0.25], vec![2, 2]),
                bias: leaf(vec![0.1, -0.2]),
            })
            .unwrap()
        };

        let residual_linear = linear();
        let residual = Node::new(NodeKind::Add {
            a: residual_linear,
            b: leaf_shape(vec![0.5, 1.0, -0.5, 0.25], vec![1, 2, 2]),
        })
        .unwrap();
        let optimized_residual = compile_graph(std::slice::from_ref(&residual), true);
        let unoptimized_residual = compile_graph(std::slice::from_ref(&residual), false);
        assert!(matches!(
            operation(optimized_residual.executable.commands()[0]).0,
            MetalOp::LinearResidual { .. }
        ));
        assert_eq!(optimized_residual.executable.commands().len(), 1);
        assert_close(
            &run(&optimized_residual)[0].to_f32_vec().unwrap(),
            &run(&unoptimized_residual)[0].to_f32_vec().unwrap(),
        );

        let single_linear = linear();
        let single = Node::new(NodeKind::Gelu {
            a: single_linear,
            approximate: false,
        })
        .unwrap();
        let optimized_single = compile_graph(std::slice::from_ref(&single), true);
        assert!(matches!(
            operation(optimized_single.executable.commands()[0]).0,
            MetalOp::LinearGelu { dual: false, .. }
        ));
        let MetalCommandPlan::Linear(single_plan) =
            operation(optimized_single.executable.commands()[0]).1
        else {
            panic!("single GELU epilogue is missing its immutable requirements");
        };
        assert_eq!(single_plan.output_count, 1);

        let dual_linear = linear();
        let dual_gelu = Node::new(NodeKind::Gelu {
            a: dual_linear.clone(),
            approximate: true,
        })
        .unwrap();
        let dual_roots = [dual_linear, dual_gelu];
        let optimized_dual = compile_graph(&dual_roots, true);
        let unoptimized_dual = compile_graph(&dual_roots, false);
        assert_eq!(optimized_dual.executable.commands().len(), 1);
        assert!(matches!(
            operation(optimized_dual.executable.commands()[0]).0,
            MetalOp::LinearGelu {
                approximate: true,
                dual: true,
                ..
            }
        ));
        let MetalCommandPlan::Linear(dual_plan) =
            operation(optimized_dual.executable.commands()[0]).1
        else {
            panic!("dual GELU epilogue is missing its immutable requirements");
        };
        assert_eq!(dual_plan.output_count, 2);
        assert_eq!(
            optimized_dual.executable.program.outputs.as_ref(),
            output_ids(optimized_dual.executable.commands()[0])
        );
        for (actual, expected) in run(&optimized_dual).iter().zip(run(&unoptimized_dual)) {
            assert_close(
                &actual.to_f32_vec().unwrap(),
                &expected.to_f32_vec().unwrap(),
            );
        }
    }

    #[test]
    fn optimizer_region_selectors_route_physical_outputs_without_commands() {
        let adamw = Node::new(NodeKind::AdamWStep {
            param: leaf(vec![1.0, 2.0]),
            grad: leaf(vec![0.1, -0.2]),
            m: leaf(vec![0.0, 0.0]),
            v: leaf(vec![0.0, 0.0]),
            lr: scalar(0.01),
            c1: scalar(0.1),
            c2: scalar(0.01),
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.01,
        })
        .unwrap();
        let adamw_m = Node::new(NodeKind::AdamWOut {
            step: adamw.clone(),
            index: 1,
        })
        .unwrap();
        let adamw_v = Node::new(NodeKind::AdamWOut {
            step: adamw.clone(),
            index: 2,
        })
        .unwrap();
        let adamw_roots = [adamw, adamw_m, adamw_v];
        let optimized_adamw = compile_graph(&adamw_roots, true);
        let unoptimized_adamw = compile_graph(&adamw_roots, false);
        assert_eq!(optimized_adamw.executable.commands().len(), 2);
        assert!(matches!(
            operation(optimized_adamw.executable.commands()[0]).0,
            MetalOp::PackOptimizerScalars
        ));
        assert!(matches!(
            operation(optimized_adamw.executable.commands()[1]).0,
            MetalOp::AdamW {
                implementation: OptimizerImplementation::Fused,
                ..
            }
        ));
        assert_eq!(
            optimized_adamw.executable.program.outputs.as_ref(),
            output_ids(optimized_adamw.executable.commands()[1])
        );
        for (actual, expected) in run(&optimized_adamw).iter().zip(run(&unoptimized_adamw)) {
            assert_close(
                &actual.to_f32_vec().unwrap(),
                &expected.to_f32_vec().unwrap(),
            );
        }

        let sgd = Node::new(NodeKind::SgdStep {
            param: leaf(vec![1.0, 2.0]),
            grad: leaf(vec![0.5, -0.25]),
            velocity: leaf(vec![0.0, 0.0]),
            first: scalar(1.0),
            lr: scalar(0.1),
            momentum: 0.9,
            dampening: 0.0,
            nesterov: false,
            weight_decay: 0.0,
        })
        .unwrap();
        let velocity = Node::new(NodeKind::SgdOut {
            step: sgd.clone(),
            index: 1,
        })
        .unwrap();
        let sgd_roots = [sgd, velocity];
        let optimized_sgd = compile_graph(&sgd_roots, true);
        let unoptimized_sgd = compile_graph(&sgd_roots, false);
        assert_eq!(optimized_sgd.executable.commands().len(), 2);
        assert!(matches!(
            operation(optimized_sgd.executable.commands()[1]).0,
            MetalOp::Sgd {
                implementation: OptimizerImplementation::Fused,
                ..
            }
        ));
        assert_eq!(
            optimized_sgd.executable.program.outputs.as_ref(),
            output_ids(optimized_sgd.executable.commands()[1])
        );
        for (actual, expected) in run(&optimized_sgd).iter().zip(run(&unoptimized_sgd)) {
            assert_close(
                &actual.to_f32_vec().unwrap(),
                &expected.to_f32_vec().unwrap(),
            );
        }
    }

    #[test]
    fn grouped_adamw_region_uses_interleaved_lanes_and_routes_selectors() {
        let lr = scalar(0.01);
        let c1 = scalar(0.1);
        let c2 = scalar(0.01);
        let step = |param: Vec<f32>, grad: Vec<f32>| {
            Node::new(NodeKind::AdamWStep {
                param: leaf(param),
                grad: leaf(grad),
                m: leaf(vec![0.0, 0.0]),
                v: leaf(vec![0.0, 0.0]),
                lr: lr.clone(),
                c1: c1.clone(),
                c2: c2.clone(),
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.01,
            })
            .unwrap()
        };
        let first = step(vec![1.0, 2.0], vec![0.1, -0.2]);
        let first_m = Node::new(NodeKind::AdamWOut {
            step: first.clone(),
            index: 1,
        })
        .unwrap();
        let second = step(vec![3.0, 4.0], vec![-0.3, 0.4]);
        let second_v = Node::new(NodeKind::AdamWOut {
            step: second.clone(),
            index: 2,
        })
        .unwrap();
        let roots = [first, first_m, second, second_v];
        let mut grouped_options = options(true);
        grouped_options.environment.optimizer_groups = true;
        let grouped = compile(
            &roots,
            grouped_options,
            1024,
            MetalEnvironment {
                private_intermediates: false,
                mma: true,
            },
        )
        .unwrap();
        let independent = compile_graph(&roots, false);
        assert_eq!(grouped.executable.commands().len(), 2);
        assert!(matches!(
            operation(grouped.executable.commands()[1]).0,
            MetalOp::AdamWGroup { parameters: 2, .. }
        ));
        assert_eq!(grouped.executable.commands()[1].outputs.len(), 6);
        assert_eq!(
            grouped.executable.program.outputs.as_ref(),
            [
                grouped.executable.commands()[1].outputs[0].value,
                grouped.executable.commands()[1].outputs[1].value,
                grouped.executable.commands()[1].outputs[3].value,
                grouped.executable.commands()[1].outputs[5].value,
            ]
        );
        for (actual, expected) in run(&grouped).iter().zip(run(&independent)) {
            assert_close(
                &actual.to_f32_vec().unwrap(),
                &expected.to_f32_vec().unwrap(),
            );
        }
    }

    #[test]
    fn random_seed_identity_remains_lowered_command_derived_across_optimization() {
        let deterministic = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add {
                a: leaf(vec![1.0, 2.0]),
                b: leaf(vec![3.0, 4.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let random = Node::new(NodeKind::Randn {
            shape: vec![16],
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let roots = [deterministic, random];
        let optimized = compile_graph(&roots, true);
        let unoptimized = compile_graph(&roots, false);
        let optimized_random = optimized
            .executable
            .commands()
            .iter()
            .find(|command| matches!(operation(command).0, MetalOp::Randn { .. }))
            .copied()
            .unwrap();
        let unoptimized_random = unoptimized
            .executable
            .commands()
            .iter()
            .find(|command| matches!(operation(command).0, MetalOp::Randn { .. }))
            .copied()
            .unwrap();
        assert_eq!(random_seed_token(optimized_random), 1);
        assert_eq!(random_seed_token(unoptimized_random), 2);
        for executable in [&optimized.executable, &unoptimized.executable] {
            assert!(executable
                .commands()
                .iter()
                .enumerate()
                .all(|(index, command)| random_seed_token(command) == index as u64));
        }
    }

    #[test]
    fn optimization_plan_builds_one_index_and_rebuilds_no_semantic_nodes() {
        let root = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add {
                a: leaf(vec![1.0, 2.0]),
                b: leaf(vec![3.0, 4.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let compilation = compile_graph(&[root], true);
        let work = &compilation.executable.compiler_work;
        assert_eq!(work.graph_index_builds, 1);
        assert_eq!(work.semantic_nodes_rebuilt, 0);
        assert_eq!(work.selected_regions, 1);
        assert_eq!(
            compilation
                .executable
                .diagnostics
                .semantic_nodes_before_optimization,
            work.semantic_nodes
        );
        assert_eq!(
            compilation
                .executable
                .diagnostics
                .semantic_nodes_after_optimization,
            work.semantic_nodes
        );
    }

    #[test]
    fn prepared_compile_uses_one_index_and_one_leaf_snapshot_collection() {
        let left = leaf(vec![1.0, 2.0]);
        let right = leaf(vec![3.0, 4.0]);
        let leaf_slots = [&left, &right].map(|node| match &node.kind {
            NodeKind::Leaf(slot) => Arc::clone(slot),
            _ => unreachable!(),
        });
        let root = Node::new(NodeKind::Add { a: left, b: right }).unwrap();
        let mut compile_options = options(false);
        compile_options.environment.ce_chunk_size = 1024;
        let program = ProgramRequest::from_roots(vec![root], compile_options)
            .prepare()
            .unwrap();
        assert_eq!(program.index.work.graph_index_builds, 1);

        let generated = load_generated_bindings(&program.index).unwrap();
        assert_eq!(generated.len(), 2);
        for slot in leaf_slots {
            assert!(slot.clear());
        }

        let compilation = compile_prepared_with_state(&program, &generated, None).unwrap();
        assert_eq!(compilation.generated_bindings.len(), 2);
        assert_eq!(compilation.executable.memory.report.external_bytes, 16);
        assert_eq!(compilation.executable.memory.report.persistent_bytes, 0);
        assert_eq!(compilation.executable.signature, program.signature);
        assert_eq!(compilation.executable.compiler_work.graph_index_builds, 1);
        assert_eq!(
            compilation.executable.compiler_work.semantic_nodes_rebuilt,
            0
        );
        assert_eq!(
            compilation
                .executable
                .diagnostics
                .compile_phases
                .iter()
                .map(|timing| timing.phase.as_str())
                .collect::<Vec<_>>(),
            [
                "graph_index",
                "optimization",
                "lowering",
                "lowered_program_validation",
                "memory_planning",
                "physical_planning",
                "pipeline_preparation",
                "artifact_assembly",
                "compile_submission",
                "publication",
            ]
        );
        assert_eq!(compilation.generated_order, [0, 1]);
        assert_eq!(run(&compilation)[0].to_f32_vec().unwrap(), [4.0, 6.0]);
    }

    #[test]
    fn constant_weights_retain_leaf_storage_without_generated_bindings() {
        let slot = Arc::new(LeafSlot::new(Value(crate::run::MetalTensor::from_f32(
            device::MetalDevice::get(),
            vec![1.0, 2.0],
            vec![2],
        ))));
        let leaf = Node::new(NodeKind::Leaf(Arc::clone(&slot))).unwrap();
        let root = Node::new(NodeKind::Neg { a: leaf }).unwrap();
        let mut compile_options = options(false);
        compile_options.inference = Some(effect_torch_compiler::InferenceOptions {
            constant_weights: true,
        });
        let program = ProgramRequest::from_roots(vec![root], compile_options)
            .prepare()
            .unwrap();
        let generated = load_generated_bindings(&program.index).unwrap();
        assert!(slot.clear());

        let compilation = compile_prepared_with_state(&program, &generated, None).unwrap();

        assert!(compilation.generated_bindings.is_empty());
        assert!(compilation.generated_order.is_empty());
        assert!(compilation.executable.bindings.is_empty());
        assert_eq!(compilation.executable.constants.len(), 1);
        assert_eq!(compilation.executable.memory.report.external_bytes, 0);
        assert_eq!(compilation.executable.memory.report.persistent_bytes, 8);
        let constant = &compilation.executable.constants[0];
        let lowered = &compilation.executable.program.values[constant.value.index()];
        assert_eq!(
            lowered.layout,
            effect_torch_runtime::Layout::contiguous(vec![2])
        );
        assert!(matches!(
            lowered.declaration.storage,
            ValueStorage::Fixed {
                class: StorageClass::PersistentConstant,
                location: Location::Persistent { .. }
            }
        ));
        let output = execute(
            &compilation.executable,
            &[],
            &[],
            &CancellationFlag::new(),
            None,
        )
        .unwrap()
        .remove(0);
        assert_eq!(output.to_f32_vec().unwrap(), [-1.0, -2.0]);
    }

    #[test]
    fn explicit_false_constant_weights_remain_generated_bindings() {
        let root = leaf(vec![3.0, 4.0]);
        let mut compile_options = options(false);
        compile_options.inference = Some(effect_torch_compiler::InferenceOptions {
            constant_weights: false,
        });
        let compilation = compile(
            &[root],
            compile_options,
            1024,
            MetalEnvironment {
                private_intermediates: false,
                mma: true,
            },
        )
        .unwrap();

        assert_eq!(compilation.generated_bindings.len(), 1);
        assert!(compilation.executable.constants.is_empty());
        assert_eq!(compilation.executable.memory.report.external_bytes, 8);
        assert_eq!(compilation.executable.memory.report.persistent_bytes, 0);
        assert_eq!(run(&compilation)[0].to_f32_vec().unwrap(), [3.0, 4.0]);
    }

    #[test]
    fn prepared_generated_values_require_the_indexed_zero_offset_contract() {
        let root = leaf(vec![1.0, 2.0]);
        let program = ProgramRequest::from_roots(vec![root], options(false))
            .prepare()
            .unwrap();
        let mut generated = load_generated_bindings(&program.index).unwrap();
        generated[0].0.layout = effect_torch_runtime::Layout::new(vec![2], vec![1], 1);

        let error = compile_prepared_with_state(&program, &generated, None)
            .err()
            .unwrap();
        assert!(error.contains("must be zero-offset contiguous"));
    }

    #[test]
    fn optimized_status_commands_keep_physical_gate_order() {
        let fused = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add {
                a: leaf(vec![1.0, 2.0]),
                b: leaf(vec![3.0, 4.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let ce = Node::new(NodeKind::CrossEntropy {
            logits: leaf_shape(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]),
            target: leaf_u32(&[0, 1], vec![2]),
            ignore_index: u32::MAX as i64,
            reduction: CrossEntropyReduction::Mean,
        })
        .unwrap();
        let compilation = compile_graph(&[fused, ce], true);
        assert!(matches!(
            operation(compilation.executable.commands()[0]).0,
            MetalOp::FusedElementwise { .. }
        ));
        let status = compilation.executable.commands()[1];
        assert!(matches!(operation(status).0, MetalOp::CrossEntropy { .. }));
        assert!(!status.status.is_empty());
        assert_eq!(
            compilation.executable.physical.as_ref(),
            [
                MetalPhysicalCommand::Encode(compilation.executable.commands()[0].id),
                MetalPhysicalCommand::Encode(status.id),
                MetalPhysicalCommand::StatusGate(status.id),
                MetalPhysicalCommand::Commit,
                MetalPhysicalCommand::Complete,
            ]
        );
    }

    #[test]
    fn typed_program_physical_resources_and_diagnostics_are_authoritative() {
        fn assert_typed(
            _: &LoweredProgram<MetalInstruction, NativeMemorySpace, MetalLoweredValue>,
        ) {
        }

        let q = leaf_shape(vec![0.2, 0.4, 0.1, 0.3], vec![1, 1, 2, 2]);
        let recurrence = Node::new(NodeKind::KdaRecurrence {
            q: q.clone(),
            k: q.clone(),
            v: q,
            log_decay: leaf_shape(vec![-0.1, -0.2, -0.3, -0.4], vec![1, 1, 2, 2]),
            beta: leaf_shape(vec![0.5, 0.25], vec![1, 1, 2, 1]),
            scale: 0.5,
            layer: 0,
        })
        .unwrap();
        let ce = Node::new(NodeKind::CrossEntropy {
            logits: leaf_shape(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]),
            target: leaf_u32(&[0, 1], vec![2]),
            ignore_index: u32::MAX as i64,
            reduction: CrossEntropyReduction::Mean,
        })
        .unwrap();
        let compilation = compile_graph_with_state(
            &[recurrence, ce],
            false,
            KvStateSchema {
                max_tokens: 64,
                block_size: 16,
                kv_dtype: DType::F32,
                window: None,
                batch: 1,
                layers: 0,
                kv_heads: 0,
                head_dim: 0,
                kda: KdaGeometry {
                    layers: 1,
                    heads: 1,
                    head_dim: 2,
                    value_dim: 2,
                },
                conv: ConvGeometry::default(),
                cursor_slot: u32::MAX,
                cursor_tensor: false,
            },
        );
        let executable = &compilation.executable;
        assert_typed(&executable.program);

        for (index, value) in executable.program.values.iter().enumerate() {
            assert_eq!(value.declaration.id.index(), index);
            assert_eq!(value.layout.shape(), value.shape.as_ref());
            assert_eq!(
                value.layout,
                effect_torch_runtime::Layout::contiguous(value.shape.to_vec())
            );
        }

        let mut encoded = 0usize;
        let mut saw_staging = false;
        let mut saw_status = false;
        let mut saw_state = false;
        for physical in &executable.physical {
            let MetalPhysicalCommand::Encode(id) = *physical else {
                continue;
            };
            encoded += 1;
            let instruction = executable
                .instruction(id)
                .expect("every physical Encode ID resolves");
            assert_eq!(instruction.id, id);
            assert!(matches!(
                &instruction.kind,
                MetalInstruction::Operation { .. }
            ));

            let planner = instruction
                .resource_uses()
                .map(|use_| use_.value)
                .collect::<Vec<_>>();
            let executor = instruction
                .inputs
                .iter()
                .map(|use_| use_.value)
                .chain(instruction.outputs.iter().map(|output| output.value))
                .chain(instruction.scratch.iter().map(|use_| use_.value))
                .chain(instruction.staging.iter().map(|use_| use_.value))
                .chain(instruction.status.iter().map(|use_| use_.value))
                .chain(instruction.state.iter().map(|use_| use_.value))
                .collect::<Vec<_>>();
            assert_eq!(planner, executor);
            saw_staging |= !instruction.staging.is_empty();
            saw_status |= !instruction.status.is_empty();
            saw_state |= !instruction.state.is_empty();
        }
        assert!(saw_staging && saw_status && saw_state);
        assert_eq!(
            executable.physical.last(),
            Some(&MetalPhysicalCommand::Complete)
        );
        assert_eq!(executable.diagnostics.command_count, encoded);
        assert_eq!(executable.diagnostics.synchronization_count, 1);
        assert_eq!(
            executable.diagnostics.pipeline_count,
            executable.prepared.pipeline_count
        );

        let mut expected = std::collections::BTreeMap::<String, usize>::new();
        for instruction in &executable.program.instructions {
            *expected
                .entry(instruction.kind.name().to_string())
                .or_default() += 1;
        }
        assert_eq!(
            executable
                .diagnostics
                .instructions
                .iter()
                .map(|instruction| (instruction.kind.clone(), instruction.count))
                .collect::<Vec<_>>(),
            expected.into_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn unsupported_ops_and_invalid_compile_options_fail_before_execute() {
        let inverse = Node::new(NodeKind::Inverse {
            a: Node::new(NodeKind::Reshape {
                a: leaf(vec![1.0, 0.0, 0.0, 1.0]),
                shape: vec![2, 2],
            })
            .unwrap(),
        })
        .unwrap();
        let error = compile(
            &[inverse],
            options(false),
            1024,
            MetalEnvironment {
                private_intermediates: false,
                mma: true,
            },
        )
        .err()
        .unwrap();
        assert!(error.contains("inverse is not supported on Metal"));

        let root = leaf(vec![1.0]);
        assert!(compile(
            &[root],
            options(false),
            0,
            MetalEnvironment {
                private_intermediates: true,
                mma: true,
            },
        )
        .err()
        .unwrap()
        .contains("CE chunk size must be positive"));
    }

    #[test]
    fn last_use_release_and_allocate_return_diagnostics_are_explicit() {
        let root = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add {
                a: leaf(vec![1.0, 2.0]),
                b: leaf(vec![3.0, 4.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let compilation = compile_graph(&[root], false);
        let commands = compilation.executable.commands();
        let intermediate = commands[0].outputs[0].value;
        assert!(release_ids(commands[1]).contains(&intermediate));
        assert!(!release_ids(commands[1]).contains(&compilation.executable.program.outputs[0]));
        assert!(compilation.executable.diagnostics.memory.output_bytes > 0);
        assert!(compilation.executable.diagnostics.memory.workspace_bytes > 0);
        assert_eq!(compilation.executable.diagnostics.command_count, 2);
        assert_eq!(compilation.executable.diagnostics.synchronization_count, 1);
        assert!(compilation.executable.memory.report.workspace_bytes > 0);
    }

    #[test]
    fn composite_commands_have_complete_static_recipes() {
        let root = Node::new(NodeKind::KvAttention {
            q: leaf_shape(vec![1.0, 0.0], vec![1, 1, 1, 2]),
            k: leaf_shape(vec![1.0, 0.0], vec![1, 1, 1, 2]),
            v: leaf_shape(vec![1.0, 0.0], vec![1, 1, 1, 2]),
            scale: 1.0,
            layer: 0,
            window: None,
        })
        .unwrap();
        let error = compile(
            std::slice::from_ref(&root),
            options(false),
            1024,
            MetalEnvironment {
                private_intermediates: false,
                mma: true,
            },
        )
        .err()
        .unwrap();
        assert!(error.contains("requires an explicit state schema"));
        let schema = KvStateSchema {
            max_tokens: 64,
            block_size: 16,
            kv_dtype: DType::F16,
            window: None,
            batch: 1,
            layers: 1,
            kv_heads: 1,
            head_dim: 2,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            cursor_slot: u32::MAX,
            cursor_tensor: false,
        };
        let compilation = compile_graph_with_state(&[root], false, schema);
        let commands = compilation.executable.commands();
        let command = commands.last().unwrap();
        assert!(!command.outputs.is_empty());
        assert_eq!(command.staging.len(), 5);
        assert_eq!(
            compilation.executable.program.values[command.staging[0].index()]
                .shape
                .as_ref(),
            [1, 4]
        );
        assert_eq!(
            compilation.executable.memory.report.state_bytes,
            1 * 64 * 1 * 2 * 2 * DType::F16.size_in_bytes()
        );
    }

    #[test]
    fn decode_cursor_is_planned_invocation_staging_not_an_external_binding() {
        let cursor = Node::new(NodeKind::ScalarInput {
            slot: 0,
            dtype: DType::I64,
            device: Device::Metal,
        })
        .unwrap();
        let positions = Node::new(NodeKind::Arange {
            start: 0.0,
            end: 4.0,
            step: 1.0,
            dtype: DType::I64,
            device: Device::Metal,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add {
            a: positions,
            b: cursor,
        })
        .unwrap();
        let compilation = compile_graph_with_state(
            &[root],
            true,
            KvStateSchema {
                max_tokens: 64,
                block_size: 16,
                kv_dtype: DType::F32,
                window: None,
                batch: 1,
                layers: 0,
                kv_heads: 0,
                head_dim: 0,
                kda: KdaGeometry::default(),
                conv: ConvGeometry::default(),
                cursor_slot: 0,
                cursor_tensor: false,
            },
        );
        assert!(matches!(
            operation(compilation.executable.commands()[0]).0,
            MetalOp::PrepareState
        ));
        assert!(compilation.executable.bindings.is_empty());
        let cursor = compilation.executable.commands()[0].outputs[0].value;
        let Location::Segment { segment, bytes, .. } =
            compilation.executable.memory.locations[cursor.index()]
        else {
            panic!("decode cursor was not assigned to invocation staging");
        };
        assert_eq!(bytes, DType::I64.size_in_bytes());
        assert_eq!(
            compilation.executable.memory.segments[segment.index()].ownership,
            SegmentOwnership::InvocationStaging
        );
    }

    #[test]
    fn environment_ce_chunk_and_optimizer_strategy_are_compile_time_state() {
        let scalar = |value: f32| {
            Node::new(NodeKind::Reshape {
                a: leaf(vec![value]),
                shape: Vec::new(),
            })
            .unwrap()
        };
        let adamw = Node::new(NodeKind::AdamWStep {
            param: leaf(vec![1.0, 1.0]),
            grad: leaf(vec![0.1, 0.1]),
            m: leaf(vec![0.0, 0.0]),
            v: leaf(vec![0.0, 0.0]),
            lr: scalar(0.01),
            c1: scalar(0.1),
            c2: scalar(0.01),
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.01,
        })
        .unwrap();
        let optimized = compile_graph(std::slice::from_ref(&adamw), true);
        let unoptimized = compile_graph(std::slice::from_ref(&adamw), false);
        assert!(matches!(
            operation(optimized.executable.commands().last().unwrap()).0,
            MetalOp::AdamW {
                implementation: OptimizerImplementation::Fused,
                ..
            }
        ));
        assert!(matches!(
            operation(unoptimized.executable.commands().last().unwrap()).0,
            MetalOp::AdamW {
                implementation: OptimizerImplementation::Generic,
                ..
            }
        ));
        assert!(!optimized.executable.environment.private_intermediates);
        assert!(optimized.executable.environment.mma);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&optimized).len(), 1);
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&unoptimized).len(), 1);
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );

        let target = Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            crate::run::MetalTensor {
                buffer: device::MetalDevice::get().upload_bytes(&0i64.to_le_bytes()),
                layout: effect_torch_runtime::Layout::contiguous(vec![1]),
                dtype: DType::I64,
            },
        )))))
        .unwrap();
        let ce = Node::new(NodeKind::ChunkedHeadCe {
            x: Node::new(NodeKind::Reshape {
                a: leaf(vec![0.5, 0.5]),
                shape: vec![1, 2],
            })
            .unwrap(),
            weight: Node::new(NodeKind::Reshape {
                a: leaf(vec![0.25; 6]),
                shape: vec![2, 3],
            })
            .unwrap(),
            bias: leaf(vec![0.0; 3]),
            target,
            ignore_index: -100,
        })
        .unwrap();
        let compilation = compile(
            &[ce],
            options(false),
            17,
            MetalEnvironment {
                private_intermediates: true,
                mma: false,
            },
        )
        .unwrap();
        assert!(matches!(
            operation(compilation.executable.commands().last().unwrap()).0,
            MetalOp::ChunkedHeadCe { chunk_size: 17, .. }
        ));
        assert!(compilation.executable.environment.private_intermediates);
        assert!(!compilation.executable.environment.mma);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(
            execute(
                &compilation.executable,
                &[],
                &compilation.generated_bindings,
                &CancellationFlag::new(),
                None,
            )
            .unwrap()
            .len(),
            1
        );
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }

    #[test]
    fn optimizer_scalars_are_packed_once_per_exact_input_tuple() {
        let scalar = |value: f32| {
            Node::new(NodeKind::Reshape {
                a: leaf(vec![value]),
                shape: Vec::new(),
            })
            .unwrap()
        };
        let lr = scalar(0.01);
        let c1 = scalar(0.1);
        let c2 = scalar(0.01);
        let adamw = |value: f32| {
            Node::new(NodeKind::AdamWStep {
                param: leaf(vec![value, value]),
                grad: leaf(vec![0.1, 0.1]),
                m: leaf(vec![0.0, 0.0]),
                v: leaf(vec![0.0, 0.0]),
                lr: lr.clone(),
                c1: c1.clone(),
                c2: c2.clone(),
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.01,
            })
            .unwrap()
        };
        let compilation = compile_graph(&[adamw(1.0), adamw(2.0)], true);
        let packs = compilation
            .executable
            .commands()
            .iter()
            .filter(|command| matches!(operation(command).0, MetalOp::PackOptimizerScalars))
            .copied()
            .collect::<Vec<_>>();
        let updates = compilation
            .executable
            .commands()
            .iter()
            .filter(|command| matches!(operation(command).0, MetalOp::AdamW { .. }))
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(packs.len(), 1);
        assert_eq!(updates.len(), 2);
        let packed = packs[0].outputs[0].value;
        assert!(updates
            .iter()
            .all(|command| command.inputs.last().map(|use_| use_.value) == Some(packed)));
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&compilation).len(), 2);
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }

    #[test]
    fn contiguous_reshape_is_a_planned_alias_with_output_owned_backing() {
        let source = Node::new(NodeKind::Neg {
            a: leaf(vec![1.0, 2.0, 3.0, 4.0]),
        })
        .unwrap();
        let reshape = Node::new(NodeKind::Reshape {
            a: source,
            shape: vec![2, 2],
        })
        .unwrap();
        let compilation = compile_graph(&[reshape], true);
        assert_eq!(compilation.executable.commands().len(), 1);
        assert!(!compilation
            .executable
            .commands()
            .iter()
            .any(|command| matches!(operation(command).0, MetalOp::Reshape { .. })));
        let output = compilation.executable.program.outputs[0];
        let Location::Alias {
            root,
            byte_offset: 0,
        } = compilation.executable.memory.locations[output.index()]
        else {
            panic!("reshape output was not planned as a zero-offset alias");
        };
        let Location::Segment { segment, .. } =
            compilation.executable.memory.locations[root.index()]
        else {
            panic!("reshape root was not assigned a planned segment");
        };
        assert_eq!(
            compilation.executable.memory.segments[segment.index()].ownership,
            SegmentOwnership::ProvisionalOutput
        );
        assert!(!release_ids(compilation.executable.commands()[0]).contains(&root));
        let first = run(&compilation).remove(0);
        assert_eq!(first.shape(), &[2, 2]);
        assert_eq!(first.to_f32_vec().unwrap(), vec![-1.0, -2.0, -3.0, -4.0]);
        let _second = run(&compilation);
        assert_eq!(first.to_f32_vec().unwrap(), vec![-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn prior_command_buffer_failure_is_propagated_by_the_owning_submission() {
        let compilation = compile_graph(&[leaf(vec![1.0])], false);
        device::inject_prior_command_buffer_failure_for_test();
        let error = execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            None,
        )
        .err()
        .unwrap();
        assert!(error.contains("GPU command buffer failure"));
    }

    #[test]
    fn static_plan_is_right_sized_bounded_and_deterministic() {
        use objc2_metal::MTLDevice;

        let root = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Neg {
                a: Node::new(NodeKind::Tanh {
                    a: Node::new(NodeKind::Add {
                        a: leaf(vec![1.0, 2.0, 3.0, 4.0]),
                        b: leaf(vec![4.0, 3.0, 2.0, 1.0]),
                    })
                    .unwrap(),
                })
                .unwrap(),
            })
            .unwrap(),
        })
        .unwrap();
        let first = compile_graph(std::slice::from_ref(&root), false);
        let second = compile_graph(&[root], false);
        assert_eq!(first.executable.memory, second.executable.memory);
        let max = device::MetalDevice::get().raw().maxBufferLength() as usize;
        assert!(first
            .executable
            .memory
            .segments
            .iter()
            .all(|segment| segment.bytes <= max));
        assert!(first.executable.memory.report.workspace_bytes <= 1 << 20);
        assert!(first
            .executable
            .memory
            .segments
            .iter()
            .filter(|segment| segment.ownership == SegmentOwnership::Workspace)
            .all(|segment| segment.bytes <= 1 << 20));
        assert!(first.executable.memory.reuse_edges.len() > 0);
    }

    #[test]
    fn recurrent_next_state_uses_an_invocation_transaction_segment() {
        let q = leaf_shape(
            vec![0.2, 0.4, 0.1, 0.3, 0.5, 0.7, 0.6, 0.8],
            vec![2, 1, 2, 2],
        );
        let recurrence = Node::new(NodeKind::KdaRecurrence {
            q: q.clone(),
            k: q.clone(),
            v: q.clone(),
            log_decay: leaf_shape(
                vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
                vec![2, 1, 2, 2],
            ),
            beta: leaf_shape(vec![0.5, 0.4, 0.3, 0.2], vec![2, 1, 2, 1]),
            scale: 0.5,
            layer: 0,
        })
        .unwrap();
        let compilation = compile_graph_with_state(
            &[recurrence],
            false,
            KvStateSchema {
                max_tokens: 64,
                block_size: 16,
                kv_dtype: DType::F32,
                window: None,
                batch: 2,
                layers: 0,
                kv_heads: 0,
                head_dim: 0,
                kda: KdaGeometry {
                    layers: 1,
                    heads: 1,
                    head_dim: 2,
                    value_dim: 2,
                },
                conv: ConvGeometry::default(),
                cursor_slot: u32::MAX,
                cursor_tensor: false,
            },
        );
        assert!(compilation.executable.memory.report.transaction_bytes >= 32);
        assert!(compilation
            .executable
            .memory
            .segments
            .iter()
            .any(|segment| segment.ownership == SegmentOwnership::StateTransaction));
        let mask = compilation.executable.commands()[0].staging[0].value;
        assert_eq!(
            compilation.executable.program.values[mask.index()]
                .shape
                .as_ref(),
            [2, 2, 1]
        );
    }

    #[test]
    fn recurrent_transactions_remain_disjoint_until_publication() {
        let q = leaf_shape(
            vec![0.2, 0.4, 0.1, 0.3, 0.5, 0.7, 0.6, 0.8],
            vec![2, 1, 2, 2],
        );
        let recurrence = |layer| {
            Node::new(NodeKind::KdaRecurrence {
                q: q.clone(),
                k: q.clone(),
                v: q.clone(),
                log_decay: leaf_shape(
                    vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
                    vec![2, 1, 2, 2],
                ),
                beta: leaf_shape(vec![0.5, 0.4, 0.3, 0.2], vec![2, 1, 2, 1]),
                scale: 0.5,
                layer,
            })
            .unwrap()
        };
        let compilation = compile_graph_with_state(
            &[recurrence(0), recurrence(1)],
            false,
            KvStateSchema {
                max_tokens: 64,
                block_size: 16,
                kv_dtype: DType::F32,
                window: None,
                batch: 2,
                layers: 0,
                kv_heads: 0,
                head_dim: 0,
                kda: KdaGeometry {
                    layers: 2,
                    heads: 1,
                    head_dim: 2,
                    value_dim: 2,
                },
                conv: ConvGeometry::default(),
                cursor_slot: u32::MAX,
                cursor_tensor: false,
            },
        );
        let transactions = compilation
            .executable
            .commands()
            .iter()
            .filter_map(|command| match operation(command).0 {
                MetalOp::KdaRecurrence { .. } => command.state.first().map(|use_| use_.value),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(transactions.len(), 2);
        assert_ne!(
            compilation.executable.memory.locations[transactions[0].index()],
            compilation.executable.memory.locations[transactions[1].index()]
        );
    }

    struct TestDecodeContext {
        schema: KvStateSchema,
        slots: Vec<Arc<Mutex<SeqState>>>,
    }

    impl MetalDecodeContext for TestDecodeContext {
        fn schema(&self) -> &KvStateSchema {
            &self.schema
        }

        fn slots(&self) -> &[Arc<Mutex<SeqState>>] {
            &self.slots
        }

        fn prepare_state(&self, _cursor: &crate::run::MetalTensor) -> Result<(), String> {
            Ok(())
        }

        fn prepare_kv_attention(
            &self,
            _layer: u32,
            _plan: &KvAttentionPlan,
            _staging: &[crate::run::MetalTensor],
        ) -> Result<(), String> {
            Err("unexpected KV attention".to_string())
        }

        fn kv_attention_into(
            &self,
            _layer: u32,
            _q: &crate::run::MetalTensor,
            _k: &crate::run::MetalTensor,
            _v: &crate::run::MetalTensor,
            _scale: f64,
            _window: Option<usize>,
            _output: &crate::run::MetalTensor,
            _staging: &[crate::run::MetalTensor],
        ) -> Result<(), String> {
            Err("unexpected KV attention".to_string())
        }

        fn evict_before(&self, _state: &mut SeqState, _start: usize) {}

        fn commit_slot(&self, _index: usize, state: &mut SeqState) {
            state.cursor += state.advance;
            state.advance = 0;
        }
    }

    fn last_token_row_schema() -> KvStateSchema {
        KvStateSchema {
            max_tokens: 64,
            block_size: 16,
            kv_dtype: DType::F32,
            window: None,
            batch: 1,
            layers: 0,
            kv_heads: 0,
            head_dim: 0,
            kda: KdaGeometry::default(),
            conv: ConvGeometry::default(),
            cursor_slot: u32::MAX,
            cursor_tensor: false,
        }
    }

    fn last_token_row_context(advance: usize) -> TestDecodeContext {
        TestDecodeContext {
            schema: last_token_row_schema(),
            slots: vec![Arc::new(Mutex::new(SeqState {
                blocks: Vec::with_capacity(4),
                head: 0,
                cursor: 0,
                advance,
                last_hash: 0,
                pending: Vec::new(),
                kda_states: Vec::new(),
                conv_states: Vec::new(),
            }))],
        }
    }

    fn compile_last_token_row() -> MetalCompilation {
        let logits = Node::new(NodeKind::LastTokenRow {
            a: leaf_shape((0..12).map(|value| value as f32).collect(), vec![1, 3, 4]),
        })
        .unwrap();
        compile_graph_with_state(&[logits], false, last_token_row_schema())
    }

    #[test]
    fn last_token_row_copies_the_advanced_row() {
        let compilation = compile_last_token_row();
        let context = last_token_row_context(2);
        let outputs = execute_stateful(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            &context,
            &|| true,
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape(), &[4]);
        assert_eq!(outputs[0].to_f32_vec().unwrap(), [4.0, 5.0, 6.0, 7.0]);
        assert_eq!(context.slots[0].lock().unwrap().cursor, 2);
    }

    #[test]
    fn last_token_row_requires_a_decode_context_and_a_valid_advance() {
        let logits = Node::new(NodeKind::LastTokenRow {
            a: leaf_shape((0..12).map(|value| value as f32).collect(), vec![1, 3, 4]),
        })
        .unwrap();
        let stateless = compile_graph(&[logits], false);
        let error = execute(
            &stateless.executable,
            &[],
            &stateless.generated_bindings,
            &CancellationFlag::new(),
            None,
        )
        .err()
        .unwrap();
        assert_eq!(
            error,
            "last_token_row: last token row requires an executable decode context"
        );

        let compilation = compile_last_token_row();

        for advance in [0, 4] {
            let context = last_token_row_context(advance);
            let error = execute_stateful(
                &compilation.executable,
                &[],
                &compilation.generated_bindings,
                &CancellationFlag::new(),
                &context,
                &|| true,
            )
            .err()
            .unwrap();
            assert_eq!(
                error,
                format!(
                    "last_token_row: last token row: token advance must be in 1..=3, got {advance}"
                )
            );
            assert_eq!(context.slots[0].lock().unwrap().cursor, 0);
        }
    }

    #[test]
    fn backend_and_cancellation_failures_do_not_commit_recurrent_state() {
        let q = leaf_shape(vec![0.2, 0.4], vec![1, 1, 1, 2]);
        let recurrence = Node::new(NodeKind::KdaRecurrence {
            q: q.clone(),
            k: q.clone(),
            v: q,
            log_decay: leaf_shape(vec![-0.1, -0.2], vec![1, 1, 1, 2]),
            beta: leaf_shape(vec![0.5], vec![1, 1, 1, 1]),
            scale: 0.5,
            layer: 0,
        })
        .unwrap();
        let schema = KvStateSchema {
            max_tokens: 64,
            block_size: 16,
            kv_dtype: DType::F32,
            window: None,
            batch: 1,
            layers: 0,
            kv_heads: 0,
            head_dim: 0,
            kda: KdaGeometry {
                layers: 1,
                heads: 1,
                head_dim: 2,
                value_dim: 2,
            },
            conv: ConvGeometry::default(),
            cursor_slot: u32::MAX,
            cursor_tensor: false,
        };
        let compilation = compile_graph_with_state(&[recurrence], false, schema);
        let persistent = crate::run::MetalTensor::from_f32(
            device::MetalDevice::get(),
            vec![0.0; 4],
            vec![1, 2, 2],
        );
        let context = TestDecodeContext {
            schema,
            slots: vec![Arc::new(Mutex::new(SeqState {
                blocks: Vec::with_capacity(4),
                head: 0,
                cursor: 0,
                advance: 1,
                last_hash: 0,
                pending: Vec::new(),
                kda_states: vec![persistent.clone()],
                conv_states: Vec::new(),
            }))],
        };

        device::inject_prior_command_buffer_failure_for_test();
        let error = execute_stateful(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            &context,
            &|| true,
        )
        .err()
        .unwrap();
        assert!(error.contains("GPU command buffer failure"));
        assert_eq!(persistent.read_f32().unwrap(), vec![0.0; 4]);
        assert_eq!(context.slots[0].lock().unwrap().cursor, 0);

        let error = execute_stateful(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            &context,
            &|| false,
        )
        .err()
        .unwrap();
        assert_eq!(error, "operation aborted");
        assert_eq!(persistent.read_f32().unwrap(), vec![0.0; 4]);
        let state = context.slots[0].lock().unwrap();
        assert_eq!(state.cursor, 0);
        assert_eq!(state.advance, 1);
    }

    #[test]
    fn first_execution_uses_one_planned_set_and_compiles_no_pipeline() {
        let root = Node::new(NodeKind::Neg {
            a: leaf(vec![1.0, 2.0, 3.0, 4.0]),
        })
        .unwrap();
        let compilation = compile_graph(&[root], false);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        let output = run(&compilation);
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
        let invocation = compilation
            .executable
            .last_invocation_memory
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        let requested_workspace: usize = compilation
            .executable
            .memory
            .segments
            .iter()
            .filter(|segment| {
                matches!(
                    segment.ownership,
                    SegmentOwnership::Workspace | SegmentOwnership::InvocationStaging
                )
            })
            .map(|segment| segment.bytes)
            .sum();
        assert_eq!(invocation.logical, compilation.executable.memory.report);
        assert!(invocation.leased_workspace_bytes >= requested_workspace);
        assert_eq!(invocation.opaque_headroom_bytes, 0);
        assert_eq!(output[0].to_f32_vec().unwrap(), [-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn retained_output_backing_survives_later_invocations() {
        let random = Node::new(NodeKind::Randn {
            shape: vec![32],
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let compilation = compile_graph(&[random], false);
        let mut retained = Vec::new();
        let mut snapshots = Vec::new();
        for _ in 0..8 {
            let output = run(&compilation).remove(0);
            snapshots.push(output.to_f32_vec().unwrap());
            retained.push(output);
        }
        for (output, expected) in retained.iter().zip(snapshots) {
            assert_eq!(output.to_f32_vec().unwrap(), expected);
        }
    }

    #[test]
    fn one_plan_executes_concurrently_with_independent_submissions() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2],
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        let output = Node::new(NodeKind::Neg { a: input }).unwrap();
        let executable = compile_graph(&[output], false).executable;
        let barrier = Arc::new(std::sync::Barrier::new(9));
        let handles = (0..8)
            .map(|index| {
                let executable = executable.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let input = Value(crate::run::MetalTensor::from_f32(
                        device::MetalDevice::get(),
                        vec![index as f32, index as f32 + 1.0],
                        vec![2],
                    ));
                    barrier.wait();
                    execute(&executable, &[input], &[], &CancellationFlag::new(), None)
                        .unwrap()
                        .remove(0)
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        for (index, handle) in handles.into_iter().enumerate() {
            assert_eq!(
                handle.join().unwrap().to_f32_vec().unwrap(),
                [-(index as f32), -(index as f32 + 1.0)]
            );
        }
    }

    #[test]
    fn cancelled_invocation_does_not_acquire_or_leak_a_lease() {
        let compilation = compile_graph(&[leaf(vec![1.0])], false);
        let cancelled = CancellationFlag::new();
        cancelled.store(true, Ordering::Relaxed);
        assert!(compilation
            .executable
            .last_invocation_memory
            .lock()
            .unwrap()
            .is_none());
        let error = match execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &cancelled,
            None,
        ) {
            Ok(_) => panic!("cancelled invocation unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error, "operation aborted");
        assert!(compilation
            .executable
            .last_invocation_memory
            .lock()
            .unwrap()
            .is_none());
    }

    #[test]
    fn indexing_reduction_and_layout_families_are_fully_precompiled() {
        let matrix = leaf_shape(vec![1.0, 4.0, 3.0, 2.0], vec![2, 2]);
        let indexes = leaf_u32(&[1, 0, 0, 1], vec![2, 2]);
        let vector_indexes = leaf_u32(&[1, 0], vec![2]);
        let argmax = Node::new(NodeKind::Argmax {
            a: matrix.clone(),
            dim: 1,
        })
        .unwrap();
        let cumsum = Node::new(NodeKind::Cumsum {
            a: matrix.clone(),
            dim: 1,
        })
        .unwrap();
        let gather = Node::new(NodeKind::Gather {
            a: matrix.clone(),
            dim: 1,
            indexes,
        })
        .unwrap();
        let select = Node::new(NodeKind::IndexSelect {
            a: matrix.clone(),
            dim: 0,
            indexes: vector_indexes,
        })
        .unwrap();
        let scatter = Node::new(NodeKind::ScatterAdd {
            a: matrix.clone(),
            dim: 1,
            indexes: leaf_u32(&[0, 1, 1, 0], vec![2, 2]),
            src: leaf_shape(vec![0.5, 0.5, 1.0, 1.0], vec![2, 2]),
        })
        .unwrap();
        let permute = Node::new(NodeKind::Permute {
            a: matrix.clone(),
            dims: vec![1, 0],
        })
        .unwrap();
        let broadcast = Node::new(NodeKind::BroadcastTo {
            a: Node::new(NodeKind::Slice {
                a: matrix.clone(),
                ranges: vec![(0, 2, 1), (0, 1, 1)],
            })
            .unwrap(),
            shape: vec![2, 2],
        })
        .unwrap();
        let concat = Node::new(NodeKind::Concat {
            a: matrix.clone(),
            b: matrix,
            dim: 0,
        })
        .unwrap();
        let roots = [
            argmax, cumsum, gather, select, scatter, permute, broadcast, concat,
        ];
        let compilation = compile_graph(&roots, false);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        let outputs = run(&compilation);
        assert_eq!(outputs.len(), roots.len());
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }

    #[test]
    fn convolution_families_are_fully_precompiled() {
        let x1 = leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let w1 = leaf_shape(vec![1.0, 1.0], vec![1, 1, 2]);
        let conv1 = Node::new(NodeKind::Conv1d {
            x: x1.clone(),
            w: w1.clone(),
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        })
        .unwrap();
        let transpose1 = Node::new(NodeKind::ConvTranspose1d {
            x: x1,
            w: w1,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        })
        .unwrap();
        let x2 = leaf_shape(
            (1..=9).map(|value| value as f32).collect(),
            vec![1, 1, 3, 3],
        );
        let w2 = leaf_shape(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let conv2 = Node::new(NodeKind::Conv2d {
            x: x2.clone(),
            w: w2.clone(),
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        })
        .unwrap();
        let transpose2 = Node::new(NodeKind::ConvTranspose2d {
            x: x2,
            w: w2,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        })
        .unwrap();
        let roots = [conv1, transpose1, conv2, transpose2];
        let compilation = compile_graph(&roots, false);
        assert!(compilation
            .executable
            .commands()
            .iter()
            .all(|command| matches!(operation(command).1, MetalCommandPlan::Conv(_))));
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&compilation).len(), roots.len());
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }

    #[test]
    fn native_semantic_families_are_fully_precompiled() {
        let q = leaf_shape(vec![0.2, 0.4, 0.1, 0.3], vec![1, 1, 2, 2]);
        let k = leaf_shape(vec![0.3, 0.1, 0.4, 0.2], vec![1, 1, 2, 2]);
        let v = leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let sdpa = Node::new(NodeKind::Sdpa {
            q,
            k,
            v,
            scale: 0.5,
            causal: true,
            window: effect_torch_graph::AttentionWindow::Inherit,
        })
        .unwrap();
        let rotary = Node::new(NodeKind::RotaryEmbedding {
            x: leaf_shape(vec![0.0; 8], vec![1, 1, 2, 4]),
            seq_len: 2,
            theta: 10_000.0,
            offset: PositionOffset::Absolute,
            layout: RotaryLayout::HalfSplit,
        })
        .unwrap();
        let layer_norm = Node::new(NodeKind::LayerNorm {
            x: leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            weight: leaf(vec![1.0, 1.0]),
            bias: leaf(vec![0.0, 0.0]),
            eps: 1e-5,
        })
        .unwrap();
        let linear = Node::new(NodeKind::Linear {
            x: leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]),
            weight: leaf_shape(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            bias: leaf(vec![0.0, 0.0]),
        })
        .unwrap();
        let shortconv = Node::new(NodeKind::ShortConv1d {
            x: leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]),
            weight: leaf_shape(vec![1.0, 0.5, 1.0, 0.5], vec![2, 2]),
        })
        .unwrap();
        let ce = Node::new(NodeKind::CrossEntropy {
            logits: leaf_shape(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]),
            target: leaf_u32(&[0, 1], vec![2]),
            ignore_index: u32::MAX as i64,
            reduction: CrossEntropyReduction::Mean,
        })
        .unwrap();
        let roots = [sdpa, rotary, layer_norm, linear, shortconv, ce];
        let compilation = compile_graph(&roots, false);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&compilation).len(), roots.len());
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }

    #[test]
    fn native_backward_families_are_fully_precompiled() {
        let x = leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let weight = leaf(vec![1.0, 1.0]);
        let gradient = leaf_shape(vec![0.5, 0.5, 1.0, 1.0], vec![2, 2]);
        let layer_norm = Node::new(NodeKind::LayerNormBackward {
            x,
            weight,
            g: gradient,
            eps: 1e-5,
        })
        .unwrap();
        let layer_norm_dw = Node::new(NodeKind::LayerNormBackwardOut {
            of: layer_norm.clone(),
            index: 1,
        })
        .unwrap();
        let layer_norm_db = Node::new(NodeKind::LayerNormBackwardOut {
            of: layer_norm.clone(),
            index: 2,
        })
        .unwrap();
        let ce = Node::new(NodeKind::CrossEntropyBackward {
            logits: leaf_shape(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]),
            target: leaf_u32(&[0, 1], vec![2]),
            ignore_index: u32::MAX as i64,
            reduction: CrossEntropyReduction::Mean,
        })
        .unwrap();
        let short_x = leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let short_w = leaf_shape(vec![1.0, 0.5, 1.0, 0.5], vec![2, 2]);
        let short_g = leaf_shape(vec![1.0, 1.0, 1.0, 1.0], vec![1, 2, 2]);
        let short_dx = Node::new(NodeKind::ShortConv1dBackwardX {
            x: short_x.clone(),
            weight: short_w.clone(),
            g: short_g.clone(),
        })
        .unwrap();
        let short_dw = Node::new(NodeKind::ShortConv1dBackwardW {
            x: short_x,
            weight: short_w,
            g: short_g,
        })
        .unwrap();
        let roots = [
            layer_norm,
            layer_norm_dw,
            layer_norm_db,
            ce,
            short_dx,
            short_dw,
        ];
        let compilation = compile_graph(&roots, false);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&compilation).len(), roots.len());
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }

    #[test]
    fn attention_backward_is_fully_precompiled() {
        let q = leaf_shape(vec![0.2, 0.4, 0.1, 0.3], vec![1, 1, 2, 2]);
        let k = leaf_shape(vec![0.3, 0.1, 0.4, 0.2], vec![1, 1, 2, 2]);
        let v = leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let forward = Node::new(NodeKind::Sdpa {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            scale: 0.5,
            causal: true,
            window: effect_torch_graph::AttentionWindow::Inherit,
        })
        .unwrap();
        let backward = Node::new(NodeKind::SdpaBackward {
            q,
            k,
            v,
            g: leaf_shape(vec![1.0; 4], vec![1, 1, 2, 2]),
            fwd: forward,
            scale: 0.5,
            causal: true,
            window: effect_torch_graph::AttentionWindow::Inherit,
        })
        .unwrap();
        let dk = Node::new(NodeKind::SdpaBackwardOut {
            of: backward.clone(),
            index: 1,
        })
        .unwrap();
        let dv = Node::new(NodeKind::SdpaBackwardOut {
            of: backward.clone(),
            index: 2,
        })
        .unwrap();
        let roots = [backward, dk, dv];
        let compilation = compile_graph(&roots, false);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&compilation).len(), roots.len());
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }

    #[test]
    fn chunked_head_backward_is_fully_precompiled() {
        let backward = Node::new(NodeKind::ChunkedHeadCeBackward {
            x: leaf_shape(vec![0.5; 8], vec![4, 2]),
            weight: leaf_shape(vec![0.25; 6], vec![2, 3]),
            bias: leaf(vec![0.0; 3]),
            target: leaf_u32(&[0, 1, 2, 0], vec![4]),
            g: leaf_shape(vec![1.0], vec![]),
            ignore_index: -100,
        })
        .unwrap();
        let dw = Node::new(NodeKind::ChunkedHeadCeBackwardOut {
            of: backward.clone(),
            index: 1,
        })
        .unwrap();
        let db = Node::new(NodeKind::ChunkedHeadCeBackwardOut {
            of: backward.clone(),
            index: 2,
        })
        .unwrap();
        let roots = [backward, dw, db];
        let compilation = compile_graph(&roots, false);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&compilation).len(), roots.len());
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }

    #[test]
    fn kda_forward_and_backward_are_fully_precompiled() {
        let q = leaf_shape(vec![0.2, 0.4, 0.1, 0.3], vec![1, 1, 2, 2]);
        let k = leaf_shape(vec![0.3, 0.1, 0.4, 0.2], vec![1, 1, 2, 2]);
        let v = leaf_shape(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let decay = leaf_shape(vec![-0.1; 4], vec![1, 1, 2, 2]);
        let beta = leaf_shape(vec![0.5; 2], vec![1, 1, 2, 1]);
        let forward = Node::new(NodeKind::KdaChunk {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            log_decay: decay.clone(),
            beta: beta.clone(),
            scale: 0.5,
        })
        .unwrap();
        let backward = Node::new(NodeKind::KdaBackward {
            q,
            k,
            v,
            log_decay: decay,
            beta,
            g: leaf_shape(vec![1.0; 4], vec![1, 1, 2, 2]),
            scale: 0.5,
        })
        .unwrap();
        let dk = Node::new(NodeKind::KdaBackwardOut {
            of: backward.clone(),
            index: 1,
        })
        .unwrap();
        let dv = Node::new(NodeKind::KdaBackwardOut {
            of: backward.clone(),
            index: 2,
        })
        .unwrap();
        let dg = Node::new(NodeKind::KdaBackwardOut {
            of: backward.clone(),
            index: 3,
        })
        .unwrap();
        let db = Node::new(NodeKind::KdaBackwardOut {
            of: backward.clone(),
            index: 4,
        })
        .unwrap();
        let roots = [forward, backward, dk, dv, dg, db];
        let compilation = compile_graph(&roots, false);
        let misses = device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed);
        assert_eq!(run(&compilation).len(), roots.len());
        assert_eq!(
            device::EXECUTABLE_PIPELINE_MISS_ATTEMPTS.load(Ordering::Relaxed),
            misses
        );
    }
}

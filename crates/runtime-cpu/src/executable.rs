use crate::composed::{
    AdamWRequirements, ChunkedHeadCeBackwardRequirements, ChunkedHeadCeForwardRequirements,
    CrossEntropyBackwardRequirements, CrossEntropyForwardRequirements, KdaBackwardRequirements,
    KdaForwardRequirements, LayerNormBackwardRequirements, LayerNormForwardRequirements,
    SdpaBackwardRequirements, SdpaForwardRequirements, SgdRequirements,
    ShortConvBackwardWRequirements, ShortConvBackwardXRequirements, ShortConvForwardRequirements,
};
use crate::conv::ConvRequirements;
use crate::linalg::{DeterminantRequirements, InverseRequirements, SolveRequirements};
use crate::matmul::MatmulRequirements;
use crate::quantized::QuantizedRequirements;
use crate::storage::CpuStorageRetention;
use crate::value::Value;
use crate::workspace::{workspace_pool, workspace_request, CpuWorkspaceLease};
use crate::{
    composed, conv, fusion, quantized, CpuBuffer, CpuDestination, CpuSegment, CpuTensorRequirement,
};
use crate::{ExecutableAllocationGuard, Tensor, CPU_STORAGE_ALIGNMENT};
use effect_torch_compiler::{
    build_executable_diagnostics, CompileOptions, CompilerDriver, CompilerWorkReport, DenseNodeId,
    DiagnosticsInput, Expr, GraphIndex, InstructionEffects, LoweredInstruction, LoweredProgram,
    LoweredValue, LoweringUnit, MemoryPlannerConfig, NativeRegion, OptimizationPlan, OutputDecl,
    PreparedProgram, ProgramRequest, ProgramSlot, RegionId, StateCursorSlot, ValueDecl,
    ValueStorage, ValueUse, ARTIFACT_ASSEMBLY_PHASE, PHYSICAL_PLANNING_PHASE, PUBLICATION_PHASE,
};
use effect_torch_graph::{
    node_children, CrossEntropyReduction, Device, Node as GraphNode, NodeKind, PositionOffset,
    RotaryLayout,
};
use effect_torch_runtime::{
    Buffer, CancellationFlag, DType, ExecutableDiagnostics, GgmlKQuant, InstructionId,
    InvocationMemoryReport, Location, MemoryPlan, NativeMemorySpace, ProgramSignature,
    SegmentOwnership, StorageClass, ValueId,
};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub type Node = GraphNode;
static INVOCATION_NONCE: AtomicU64 = AtomicU64::new(0);

fn cpu_device() -> Device {
    Device::Cpu
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuBindingSource {
    Declared(u32),
    Generated(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuBinding {
    pub value: ValueId,
    pub source: CpuBindingSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CpuScalarBinding {
    value: ValueId,
    source: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CpuPaddedBinding {
    value: ValueId,
    source: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuDeclaredSource {
    Tensor(u32),
    Scalar(u32),
    StateCursor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuStatePlan {
    pub bytes: usize,
    pub cursor_slot: u32,
    pub cursor_tensor: bool,
    pub batch: usize,
}

#[derive(Debug, Clone)]
struct CpuValueMetadata {
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub layout: effect_torch_runtime::Layout,
}

#[derive(Debug, Clone)]
pub struct CpuLoweredValue {
    pub shape: Box<[usize]>,
    pub dtype: DType,
    pub layout: effect_torch_runtime::Layout,
    pub declaration: ValueDecl<NativeMemorySpace>,
}

impl LoweredValue<NativeMemorySpace> for CpuLoweredValue {
    fn value_decl(&self) -> &ValueDecl<NativeMemorySpace> {
        &self.declaration
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuValueStorage {
    External,
    Constant,
    Planned(SegmentOwnership),
    Alias { source: ValueId, byte_offset: usize },
}

#[derive(Clone)]
struct CpuConstant {
    value: ValueId,
    payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuUnaryOp {
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
    Inverse,
    Det,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuBinaryOp {
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
    Solve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerImplementation {
    Composed,
    Fused,
}

#[derive(Debug, Clone)]
pub enum CpuOp {
    Randn {
        shape: Box<[usize]>,
        dtype: DType,
        provenance: u64,
    },
    Uniform {
        lo: f64,
        hi: f64,
        shape: Box<[usize]>,
        dtype: DType,
        provenance: u64,
    },
    Unary(CpuUnaryOp),
    Binary(CpuBinaryOp),
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
    },
    CrossEntropyBackward {
        ignore_index: i64,
        reduction: CrossEntropyReduction,
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
    },
    SdpaBackward {
        scale: f64,
        causal: bool,
        window: Option<usize>,
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
    },
    RotaryEmbeddingBackward {
        theta: f64,
        layout: RotaryLayout,
    },
    Linear,
    QuantizedLinear {
        codec: GgmlKQuant,
        weight_shape: [usize; 2],
    },
    QuantizedEmbedding {
        codec: GgmlKQuant,
        weight_shape: [usize; 2],
    },
    LinearResidual,
    LinearGelu {
        approximate: bool,
        dual: bool,
    },
    LayerNorm {
        eps: f64,
    },
    RmsNorm {
        eps: f64,
    },
    LayerNormBackward {
        eps: f64,
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
        op: CpuReduceOp,
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
    AdamW {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        implementation: OptimizerImplementation,
        fused_exprs: Option<Box<[Expr]>>,
    },
    AdamWGroup {
        parameters: usize,
        fused_exprs: Box<[Expr]>,
    },
    Sgd {
        momentum: f64,
        dampening: f64,
        nesterov: bool,
        weight_decay: f64,
        implementation: OptimizerImplementation,
        fused_exprs: Option<Box<[Expr]>>,
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

impl CpuOp {
    pub fn name(&self) -> &'static str {
        match self {
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
            Self::CrossEntropy { .. } => "cross_entropy",
            Self::CrossEntropyBackward { .. } => "cross_entropy_backward",
            Self::ChunkedHeadCe { .. } => "chunked_head_ce",
            Self::ChunkedHeadCeBackward { .. } => "chunked_head_ce_backward",
            Self::Sdpa { .. } => "sdpa",
            Self::SdpaBackward { .. } => "sdpa_backward",
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
            Self::RotaryEmbedding { .. } => "rotary_embedding",
            Self::RotaryEmbeddingBackward { .. } => "rotary_embedding_backward",
            Self::Linear => "linear",
            Self::QuantizedLinear { .. } => "quantized_linear",
            Self::QuantizedEmbedding { .. } => "quantized_embedding",
            Self::LinearResidual => "linear_residual",
            Self::LinearGelu { .. } => "linear_gelu",
            Self::LayerNorm { .. } => "layer_norm",
            Self::RmsNorm { .. } => "rms_norm",
            Self::LayerNormBackward { .. } => "layer_norm_backward",
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
            Self::AdamW { implementation, .. } => match implementation {
                OptimizerImplementation::Composed => "adamw_composed",
                OptimizerImplementation::Fused => "adamw_fused",
            },
            Self::AdamWGroup { .. } => "adamw_group_fused",
            Self::Sgd { implementation, .. } => match implementation {
                OptimizerImplementation::Composed => "sgd_composed",
                OptimizerImplementation::Fused => "sgd_fused",
            },
            Self::FusedElementwise { .. } => "fused_elementwise",
            Self::FusedReduce { .. } => "fused_reduce",
        }
    }
}

#[derive(Debug, Clone)]
pub enum CpuAlgorithmPlan {
    None,
    Matmul(MatmulRequirements),
    Determinant(DeterminantRequirements),
    Inverse(InverseRequirements),
    Solve(SolveRequirements),
    Convolution(ConvRequirements),
    CrossEntropyForward(CrossEntropyForwardRequirements),
    CrossEntropyBackward(CrossEntropyBackwardRequirements),
    ChunkedHeadCeForward(ChunkedHeadCeForwardRequirements),
    ChunkedHeadCeBackward(ChunkedHeadCeBackwardRequirements),
    SdpaForward(SdpaForwardRequirements),
    SdpaBackward(SdpaBackwardRequirements),
    LayerNormForward(LayerNormForwardRequirements),
    LayerNormBackward(LayerNormBackwardRequirements),
    AdamW {
        requirements: AdamWRequirements,
        fusion: Option<effect_torch_compiler::CpuFusionProgram>,
        strides: Box<[Vec<usize>]>,
    },
    Sgd {
        requirements: SgdRequirements,
        fusion: Option<effect_torch_compiler::CpuFusionProgram>,
        strides: Box<[Vec<usize>]>,
    },
    KdaForward(KdaForwardRequirements),
    KdaBackward(KdaBackwardRequirements),
    ShortConvForward(ShortConvForwardRequirements),
    ShortConvBackwardX(ShortConvBackwardXRequirements),
    ShortConvBackwardW(ShortConvBackwardWRequirements),
    Fusion(effect_torch_compiler::CpuFusionProgram),
    UnaryFusion {
        program: effect_torch_compiler::CpuFusionProgram,
        strides: Box<[Vec<usize>]>,
    },
    MultiFusion {
        program: effect_torch_compiler::CpuFusionProgram,
        strides: Box<[Vec<usize>]>,
    },
    Linear {
        matmul: MatmulRequirements,
        gelu: Option<effect_torch_compiler::CpuFusionProgram>,
    },
    Quantized(QuantizedRequirements),
}

#[derive(Debug, Clone)]
pub enum CpuInstruction {
    PrepareInvocation,
    Operation { op: CpuOp, plan: CpuAlgorithmPlan },
    FinalizeInvocation,
}

impl CpuInstruction {
    pub fn name(&self) -> &'static str {
        match self {
            Self::PrepareInvocation => "prepare_invocation",
            Self::Operation { op, .. } => op.name(),
            Self::FinalizeInvocation => "finalize_invocation",
        }
    }

    pub fn operation(&self) -> Option<(&CpuOp, &CpuAlgorithmPlan)> {
        match self {
            Self::Operation { op, plan } => Some((op, plan)),
            Self::PrepareInvocation | Self::FinalizeInvocation => None,
        }
    }
}

pub type CpuCommand = LoweredInstruction<CpuInstruction>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuPhysicalCommand {
    Encode(InstructionId),
}

pub struct CpuExecutable {
    pub signature: ProgramSignature,
    pub program: Arc<LoweredProgram<CpuInstruction, NativeMemorySpace, CpuLoweredValue>>,
    pub physical: Box<[CpuPhysicalCommand]>,
    pub bindings: Box<[CpuBinding]>,
    scalar_bindings: Box<[CpuScalarBinding]>,
    padded_bindings: Box<[CpuPaddedBinding]>,
    constants: Box<[CpuConstant]>,
    pub options: CompileOptions,
    pub memory: MemoryPlan<NativeMemorySpace>,
    pub diagnostics: ExecutableDiagnostics,
    pub compiler_work: CompilerWorkReport,
    pub state_cursor: Option<ValueId>,
}

impl fmt::Debug for CpuExecutable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuExecutable")
            .field("signature", &self.signature)
            .field("program", &self.program)
            .field("physical", &self.physical)
            .field("bindings", &self.bindings)
            .field("scalar_bindings", &self.scalar_bindings)
            .field("padded_bindings", &self.padded_bindings)
            .field("constant_count", &self.constants.len())
            .field("options", &self.options)
            .field("memory", &self.memory)
            .field("diagnostics", &self.diagnostics)
            .field("compiler_work", &self.compiler_work)
            .field("state_cursor", &self.state_cursor)
            .finish()
    }
}

impl CpuExecutable {
    pub fn instruction(&self, id: InstructionId) -> Option<&CpuCommand> {
        self.program
            .instructions
            .get(id.index())
            .filter(|instruction| instruction.id == id)
    }
}

pub struct CpuCompilation {
    pub executable: Arc<CpuExecutable>,
    pub slots: Vec<ProgramSlot>,
    pub generated_bindings: Vec<Value>,
    pub generated_slots: Vec<usize>,
}

#[derive(Clone)]
pub struct CpuGeneratedValue {
    node: DenseNodeId,
    node_id: u64,
    slot_identity: usize,
    value: Value,
}

impl CpuGeneratedValue {
    pub fn slot_identity(&self) -> usize {
        self.slot_identity
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
}

struct Lowerer {
    values: Vec<CpuValueMetadata>,
    storage: Vec<CpuValueStorage>,
    names: Vec<String>,
    bindings: Vec<CpuBinding>,
    scalar_bindings: Vec<CpuScalarBinding>,
    padded_bindings: Vec<CpuPaddedBinding>,
    declared_sources: HashMap<u32, CpuDeclaredSource>,
    constants: Vec<CpuConstant>,
    instructions: Vec<LoweredInstruction<CpuInstruction>>,
    node_values: HashMap<u64, Box<[ValueId]>>,
    declared_values: HashMap<u32, ValueId>,
    random_provenance: HashMap<u64, u64>,
    preloaded_generated: HashMap<u64, Value>,
    generated: Vec<Value>,
    generated_slots: Vec<usize>,
    state: Option<CpuStatePlan>,
    state_cursor: Option<ValueId>,
    padded_slot: Option<u32>,
    ce_chunk_size: usize,
    options: CompileOptions,
}

impl Lowerer {
    fn new(
        options: CompileOptions,
        ce_chunk_size: usize,
        random_provenance: HashMap<u64, u64>,
        slots: &[ProgramSlot],
        state: Option<CpuStatePlan>,
        generated_values: &[CpuGeneratedValue],
    ) -> Self {
        let padded_slot = state.and_then(|state| {
            slots
                .iter()
                .enumerate()
                .rev()
                .find(|(slot, declaration)| {
                    !declaration.scalar && *slot as u32 != state.cursor_slot
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
                let source = if state.is_some_and(|state| state.cursor_slot == slot) {
                    CpuDeclaredSource::StateCursor
                } else if declaration.scalar {
                    let source = CpuDeclaredSource::Scalar(scalar);
                    scalar = scalar
                        .checked_add(1)
                        .expect("scalar binding count overflow");
                    source
                } else {
                    let source = CpuDeclaredSource::Tensor(tensor);
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
            random_provenance,
            preloaded_generated: generated_values
                .iter()
                .map(|generated| (generated.node_id, generated.value.clone()))
                .collect(),
            generated: Vec::new(),
            generated_slots: Vec::new(),
            state,
            state_cursor: None,
            padded_slot,
            ce_chunk_size,
            options,
        }
    }

    fn value(
        &mut self,
        shape: &[usize],
        dtype: DType,
        storage: CpuValueStorage,
        name: impl Into<String>,
    ) -> Result<ValueId, String> {
        self.value_with_layout(
            shape,
            dtype,
            effect_torch_runtime::Layout::contiguous(shape.to_vec()),
            storage,
            name,
        )
    }

    fn value_with_layout(
        &mut self,
        shape: &[usize],
        dtype: DType,
        layout: effect_torch_runtime::Layout,
        storage: CpuValueStorage,
        name: impl Into<String>,
    ) -> Result<ValueId, String> {
        shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| "compile: value byte size overflow".to_string())?;
        let id = ValueId::from_index(self.values.len())
            .ok_or_else(|| "compile: too many CPU values".to_string())?;
        self.values.push(CpuValueMetadata {
            shape: shape.to_vec().into_boxed_slice(),
            dtype,
            layout,
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
        self.value(
            shape,
            dtype,
            CpuValueStorage::Planned(SegmentOwnership::Workspace),
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

    fn alias_value(
        &mut self,
        source: ValueId,
        shape: &[usize],
        layout: effect_torch_runtime::Layout,
        name: impl Into<String>,
    ) -> Result<ValueId, String> {
        let source_decl = &self.values[source.index()];
        let element_offset = layout
            .offset()
            .checked_sub(source_decl.layout.offset())
            .ok_or_else(|| "compile: alias precedes its source storage".to_string())?;
        let byte_offset = element_offset
            .checked_mul(source_decl.dtype.size_in_bytes())
            .ok_or_else(|| "compile: alias byte offset overflow".to_string())?;
        self.value_with_layout(
            shape,
            source_decl.dtype,
            layout,
            CpuValueStorage::Alias {
                source,
                byte_offset,
            },
            name,
        )
    }

    fn command(
        &mut self,
        op: CpuOp,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> Result<(), String> {
        let id = InstructionId::from_index(self.instructions.len())
            .ok_or_else(|| "compile: too many CPU instructions".to_string())?;
        self.instructions.push(LoweredInstruction::new(
            id,
            CpuInstruction::Operation {
                op,
                plan: CpuAlgorithmPlan::None,
            },
            inputs.into_iter().map(ValueUse::read).collect::<Vec<_>>(),
            outputs.into_iter().map(OutputDecl::new).collect::<Vec<_>>(),
        ));
        Ok(())
    }

    fn metadata_tensor(&self, value: ValueId) -> Result<Tensor, String> {
        let declaration = &self.values[value.index()];
        let owner = CpuSegment::allocate(0, CPU_STORAGE_ALIGNMENT)
            .map_err(|error| format!("compile: metadata storage: {error}"))?;
        Ok(Tensor {
            buffer: CpuBuffer::from_segment(owner, 0, 0, declaration.dtype)?,
            layout: declaration.layout.clone(),
        })
    }

    fn requirement_value(
        &mut self,
        requirement: &CpuTensorRequirement,
        ownership: SegmentOwnership,
        name: impl Into<String>,
    ) -> Result<ValueId, String> {
        if requirement.alignment == 0 || !requirement.alignment.is_power_of_two() {
            return Err("compile: invalid CPU requirement alignment".to_string());
        }
        let value = self.value(
            &requirement.shape,
            requirement.dtype,
            CpuValueStorage::Planned(ownership),
            name,
        )?;
        let declaration = &self.values[value.index()];
        let bytes = declaration
            .shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .and_then(|elements| elements.checked_mul(declaration.dtype.size_in_bytes()))
            .ok_or_else(|| "compile: requirement byte size overflow".to_string())?;
        if bytes != requirement.bytes {
            return Err(format!(
                "compile: requirement declares {} bytes, dense value requires {bytes}",
                requirement.bytes
            ));
        }
        Ok(value)
    }

    fn prepare_last_command(&mut self) -> Result<(), String> {
        let index = self.instructions.len() - 1;
        let (op, _) = self.instructions[index]
            .kind
            .operation()
            .expect("lowerer only prepares operation instructions");
        let op = op.clone();
        let inputs = self.instructions[index]
            .inputs
            .iter()
            .map(|value_use| value_use.value)
            .collect::<Vec<_>>();
        let outputs = self.instructions[index]
            .outputs
            .iter()
            .map(|output| output.value)
            .collect::<Vec<_>>();
        let tensors = inputs
            .iter()
            .map(|value| self.metadata_tensor(*value))
            .collect::<Result<Vec<_>, _>>()?;
        let mut scratch = Vec::new();
        let mut staging = Vec::new();
        let mut state_outputs = Vec::new();
        let mut add_scratch = |this: &mut Self,
                               requirement: &CpuTensorRequirement,
                               name: &str|
         -> Result<ValueId, String> {
            let value = this.requirement_value(
                requirement,
                SegmentOwnership::Workspace,
                format!("command_{index}_{name}"),
            )?;
            scratch.push(value);
            Ok(value)
        };
        let plan = match &op {
            CpuOp::Unary(CpuUnaryOp::Gelu { approximate }) => {
                let expr = if *approximate {
                    Expr::GeluTanh(Box::new(Expr::Input(0)))
                } else {
                    Expr::Gelu(Box::new(Expr::Input(0)))
                };
                let program = fusion::prepare(&[expr]);
                let requirement = fusion::scratch_requirement(&program, tensors[0].dtype())?;
                add_scratch(self, &requirement, "fusion_scratch")?;
                CpuAlgorithmPlan::UnaryFusion {
                    program,
                    strides: vec![self.values[inputs[0].index()].layout.strides().to_vec()]
                        .into_boxed_slice(),
                }
            }
            CpuOp::Unary(CpuUnaryOp::Inverse) => {
                let requirements = tensors[0].inverse_requirements().map_err(str::to_string)?;
                for (scratch_index, requirement) in requirements.scratch.iter().enumerate() {
                    add_scratch(
                        self,
                        requirement,
                        &format!("inverse_scratch_{scratch_index}"),
                    )?;
                }
                CpuAlgorithmPlan::Inverse(requirements)
            }
            CpuOp::Unary(CpuUnaryOp::Det) => {
                let requirements = tensors[0].det_requirements().map_err(str::to_string)?;
                if self.values[outputs[0].index()].shape.as_ref()
                    != requirements.output.shape.as_slice()
                {
                    add_scratch(self, &requirements.output, "det_output")?;
                }
                for (scratch_index, requirement) in requirements.scratch.iter().enumerate() {
                    add_scratch(self, requirement, &format!("det_scratch_{scratch_index}"))?;
                }
                CpuAlgorithmPlan::Determinant(requirements)
            }
            CpuOp::Binary(CpuBinaryOp::Matmul) => {
                let requirements = tensors[0]
                    .matmul_requirements(&tensors[1])
                    .map_err(str::to_string)?;
                for (scratch_index, requirement) in requirements.scratch.iter().enumerate() {
                    add_scratch(
                        self,
                        requirement,
                        &format!("matmul_scratch_{scratch_index}"),
                    )?;
                }
                CpuAlgorithmPlan::Matmul(requirements)
            }
            CpuOp::Binary(CpuBinaryOp::Solve) => {
                let requirements = tensors[0]
                    .solve_requirements(&tensors[1])
                    .map_err(str::to_string)?;
                for (scratch_index, requirement) in requirements.scratch.iter().enumerate() {
                    add_scratch(self, requirement, &format!("solve_scratch_{scratch_index}"))?;
                }
                CpuAlgorithmPlan::Solve(requirements)
            }
            CpuOp::Binary(kind) => {
                if matches!(
                    kind,
                    CpuBinaryOp::Add
                        | CpuBinaryOp::Sub
                        | CpuBinaryOp::Mul
                        | CpuBinaryOp::Div
                        | CpuBinaryOp::Eq
                        | CpuBinaryOp::Gt
                        | CpuBinaryOp::Lt
                        | CpuBinaryOp::Ge
                        | CpuBinaryOp::Le
                        | CpuBinaryOp::Maximum
                        | CpuBinaryOp::Minimum
                ) {
                    let operand_dtype = inputs
                        .iter()
                        .map(|input| &self.values[input.index()])
                        .find(|input| !input.shape.is_empty())
                        .map_or(self.values[inputs[0].index()].dtype, |input| input.dtype);
                    for (input_index, input) in inputs.iter().enumerate() {
                        let declaration = &self.values[input.index()];
                        if declaration.dtype != operand_dtype && declaration.shape.is_empty() {
                            let requirement = CpuTensorRequirement::new(&[], operand_dtype);
                            let value = self.requirement_value(
                                &requirement,
                                SegmentOwnership::InvocationStaging,
                                format!("command_{index}_scalar_{input_index}"),
                            )?;
                            staging.push(value);
                        }
                    }
                }
                CpuAlgorithmPlan::None
            }
            CpuOp::Argmax { .. } | CpuOp::Argmin { .. } => {
                let dimension = match &op {
                    CpuOp::Argmax { dim } | CpuOp::Argmin { dim } => *dim,
                    _ => unreachable!(),
                };
                let requirement = tensors[0].argmax_output_requirements(dimension);
                add_scratch(self, &requirement, "arg_reduce_u32")?;
                add_scratch(
                    self,
                    &CpuTensorRequirement::new(&requirement.shape, DType::I64),
                    "arg_reduce_i64",
                )?;
                CpuAlgorithmPlan::None
            }
            CpuOp::CrossEntropy { reduction, .. } => {
                let requirements = composed::cross_entropy_forward_requirements(
                    &tensors[0],
                    &tensors[1],
                    *reduction,
                )?;
                add_scratch(self, &requirements.status, "ce_status")?;
                add_scratch(self, &requirements.nll_scratch, "ce_nll")?;
                add_scratch(self, &requirements.flags_scratch, "ce_flags")?;
                CpuAlgorithmPlan::CrossEntropyForward(requirements)
            }
            CpuOp::CrossEntropyBackward { reduction, .. } => {
                let requirements = composed::cross_entropy_backward_requirements(
                    &tensors[0],
                    &tensors[1],
                    *reduction,
                )?;
                if let Some(requirement) = &requirements.count_status {
                    add_scratch(self, requirement, "ce_count")?;
                }
                CpuAlgorithmPlan::CrossEntropyBackward(requirements)
            }
            CpuOp::ChunkedHeadCe { chunk_size, .. } => {
                let requirements = composed::chunked_head_ce_forward_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                    &tensors[3],
                    *chunk_size,
                )?;
                for (requirement, name) in [
                    (&requirements.status, "head_ce_status"),
                    (&requirements.logits_scratch, "head_ce_logits"),
                    (&requirements.nll_scratch, "head_ce_nll"),
                    (&requirements.flags_scratch, "head_ce_flags"),
                ] {
                    add_scratch(self, requirement, name)?;
                }
                CpuAlgorithmPlan::ChunkedHeadCeForward(requirements)
            }
            CpuOp::ChunkedHeadCeBackward { chunk_size, .. } => {
                let requirements = composed::chunked_head_ce_backward_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                    &tensors[3],
                    *chunk_size,
                )?;
                for (requirement, name) in [
                    (&requirements.status, "head_ce_status"),
                    (&requirements.logits_scratch, "head_ce_logits"),
                    (&requirements.grad_logits_scratch, "head_ce_grad_logits"),
                    (&requirements.dweight_scratch, "head_ce_dweight"),
                    (&requirements.dbias_scratch, "head_ce_dbias"),
                ] {
                    add_scratch(self, requirement, name)?;
                }
                CpuAlgorithmPlan::ChunkedHeadCeBackward(requirements)
            }
            CpuOp::Sdpa { .. } => {
                let requirements =
                    composed::sdpa_forward_requirements(&tensors[0], &tensors[1], &tensors[2])?;
                add_scratch(self, &requirements.logsumexp, "sdpa_logsumexp")?;
                CpuAlgorithmPlan::SdpaForward(requirements)
            }
            CpuOp::SdpaBackward { .. } => {
                let forward =
                    composed::sdpa_forward_requirements(&tensors[0], &tensors[1], &tensors[2])?;
                let requirements =
                    composed::sdpa_backward_requirements(&tensors[0], &tensors[1], &tensors[2])?;
                add_scratch(self, &forward.logsumexp, "sdpa_logsumexp")?;
                add_scratch(self, &requirements.d_vec_scratch, "sdpa_row_dot")?;
                CpuAlgorithmPlan::SdpaBackward(requirements)
            }
            CpuOp::LayerNorm { .. } => CpuAlgorithmPlan::LayerNormForward(
                composed::layer_norm_forward_requirements(&tensors[0], &tensors[1], &tensors[2])?,
            ),
            CpuOp::RmsNorm { .. } => CpuAlgorithmPlan::LayerNormForward(
                composed::rms_norm_forward_requirements(&tensors[0], tensors.get(1))?,
            ),
            CpuOp::LayerNormBackward { .. } => {
                let requirements = composed::layer_norm_backward_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                )?;
                add_scratch(
                    self,
                    &requirements.normalized_scratch,
                    "layer_norm_normalized",
                )?;
                CpuAlgorithmPlan::LayerNormBackward(requirements)
            }
            CpuOp::KdaChunk { .. } => {
                let requirements = composed::kda_chunk_forward_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                    &tensors[3],
                    &tensors[4],
                    false,
                )?;
                add_scratch(
                    self,
                    requirements
                        .state_scratch
                        .as_ref()
                        .expect("non-state KDA has state scratch"),
                    "kda_state",
                )?;
                CpuAlgorithmPlan::KdaForward(requirements)
            }
            CpuOp::KdaBackward { .. } => {
                let requirements = composed::kda_chunk_backward_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                    &tensors[3],
                    &tensors[4],
                )?;
                add_scratch(self, &requirements.scratch, "kda_backward")?;
                CpuAlgorithmPlan::KdaBackward(requirements)
            }
            CpuOp::ShortConv1d => CpuAlgorithmPlan::ShortConvForward(
                composed::short_conv1d_forward_requirements(&tensors[0], &tensors[1], false)?,
            ),
            CpuOp::ShortConv1dBackwardX => CpuAlgorithmPlan::ShortConvBackwardX(
                composed::short_conv1d_backward_x_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                )?,
            ),
            CpuOp::ShortConv1dBackwardW => CpuAlgorithmPlan::ShortConvBackwardW(
                composed::short_conv1d_backward_w_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                )?,
            ),
            CpuOp::Conv1d {
                stride,
                padding,
                dilation,
                groups,
            } => CpuAlgorithmPlan::Convolution(conv::conv1d_requirements(
                &tensors[0],
                &tensors[1],
                *stride,
                *padding,
                *dilation,
                *groups,
            )?),
            CpuOp::Conv2d {
                stride,
                padding,
                dilation,
                groups,
            } => CpuAlgorithmPlan::Convolution(conv::conv2d_requirements(
                &tensors[0],
                &tensors[1],
                *stride,
                *padding,
                *dilation,
                *groups,
            )?),
            CpuOp::ConvTranspose1d {
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            } => CpuAlgorithmPlan::Convolution(conv::conv_transpose1d_requirements(
                &tensors[0],
                &tensors[1],
                *stride,
                *padding,
                *output_padding,
                *dilation,
                *groups,
            )?),
            CpuOp::ConvTranspose2d {
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            } => CpuAlgorithmPlan::Convolution(conv::conv_transpose2d_requirements(
                &tensors[0],
                &tensors[1],
                *stride,
                *padding,
                *output_padding,
                *dilation,
                *groups,
            )?),
            CpuOp::Conv1dBackwardW {
                kernel,
                out_channels,
                stride,
                padding,
                dilation,
                groups,
            } => CpuAlgorithmPlan::Convolution(conv::conv1d_backward_w_requirements(
                &tensors[0],
                &tensors[1],
                *kernel,
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
            )?),
            CpuOp::Conv2dBackwardW {
                kernel,
                out_channels,
                stride,
                padding,
                dilation,
                groups,
            } => CpuAlgorithmPlan::Convolution(conv::conv2d_backward_w_requirements(
                &tensors[0],
                &tensors[1],
                *kernel,
                *out_channels,
                *stride,
                *padding,
                *dilation,
                *groups,
            )?),
            CpuOp::Reduce { dims, keepdims, .. } if !keepdims => {
                let mut shape = tensors[0].shape().to_vec();
                for dimension in dims.iter().copied() {
                    shape[dimension] = 1;
                }
                let requirement = CpuTensorRequirement::new(&shape, tensors[0].dtype());
                add_scratch(self, &requirement, "kept_reduce")?;
                CpuAlgorithmPlan::None
            }
            CpuOp::AdamW {
                implementation,
                fused_exprs,
                ..
            } => {
                let requirements = composed::adamw_step_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                    &tensors[3],
                )?;
                for input_index in 4..7 {
                    let requirement = CpuTensorRequirement::new(&[], tensors[0].dtype());
                    let value = self.requirement_value(
                        &requirement,
                        SegmentOwnership::InvocationStaging,
                        format!("command_{index}_optimizer_scalar_{input_index}"),
                    )?;
                    staging.push(value);
                }
                let fusion = match implementation {
                    OptimizerImplementation::Composed => None,
                    OptimizerImplementation::Fused => {
                        let program = fusion::prepare(fused_exprs.as_deref().ok_or_else(|| {
                            "compile: fused AdamW has no expression program".to_string()
                        })?);
                        let requirement =
                            fusion::scratch_requirement(&program, tensors[0].dtype())?;
                        add_scratch(self, &requirement, "optimizer_fusion")?;
                        Some(program)
                    }
                };
                CpuAlgorithmPlan::AdamW {
                    requirements,
                    fusion,
                    strides: inputs[..4]
                        .iter()
                        .map(|value| self.values[value.index()].layout.strides().to_vec())
                        .collect(),
                }
            }
            CpuOp::Sgd {
                implementation,
                fused_exprs,
                ..
            } => {
                let requirements =
                    composed::sgd_step_requirements(&tensors[0], &tensors[1], &tensors[2])?;
                for input_index in [4usize, 3] {
                    let requirement = CpuTensorRequirement::new(&[], tensors[0].dtype());
                    let value = self.requirement_value(
                        &requirement,
                        SegmentOwnership::InvocationStaging,
                        format!("command_{index}_optimizer_scalar_{input_index}"),
                    )?;
                    staging.push(value);
                }
                let fusion = match implementation {
                    OptimizerImplementation::Composed => None,
                    OptimizerImplementation::Fused => {
                        let program = fusion::prepare(fused_exprs.as_deref().ok_or_else(|| {
                            "compile: fused SGD has no expression program".to_string()
                        })?);
                        let requirement =
                            fusion::scratch_requirement(&program, tensors[0].dtype())?;
                        add_scratch(self, &requirement, "optimizer_fusion")?;
                        Some(program)
                    }
                };
                CpuAlgorithmPlan::Sgd {
                    requirements,
                    fusion,
                    strides: inputs[..3]
                        .iter()
                        .map(|value| self.values[value.index()].layout.strides().to_vec())
                        .collect(),
                }
            }
            CpuOp::AdamWGroup {
                parameters,
                fused_exprs,
            } => {
                let program = fusion::prepare(fused_exprs);
                let first = &tensors[0];
                for scalar_index in 0..3 {
                    let requirement = CpuTensorRequirement::new(&[], first.dtype());
                    let value = self.requirement_value(
                        &requirement,
                        SegmentOwnership::InvocationStaging,
                        format!("command_{index}_group_scalar_{scalar_index}"),
                    )?;
                    staging.push(value);
                }
                let requirement = fusion::scratch_requirement(&program, first.dtype())?;
                add_scratch(self, &requirement, "group_fusion")?;
                let mut strides = Vec::with_capacity(*parameters * 4);
                for parameter in 0..*parameters {
                    for input in [
                        parameter,
                        *parameters + parameter,
                        *parameters * 2 + parameter,
                        *parameters * 3 + parameter,
                    ] {
                        strides.push(self.values[inputs[input].index()].layout.strides().to_vec());
                    }
                }
                CpuAlgorithmPlan::MultiFusion {
                    program,
                    strides: strides.into_boxed_slice(),
                }
            }
            CpuOp::FusedElementwise { exprs, .. } => {
                let program = fusion::prepare(exprs);
                let requirement = fusion::scratch_requirement(&program, tensors[0].dtype())?;
                add_scratch(self, &requirement, "fusion")?;
                CpuAlgorithmPlan::Fusion(program)
            }
            CpuOp::FusedReduce { expr, .. } => {
                let program = fusion::prepare(std::slice::from_ref(expr));
                let requirement = fusion::scratch_requirement(&program, tensors[0].dtype())?;
                add_scratch(self, &requirement, "fusion_reduce")?;
                CpuAlgorithmPlan::Fusion(program)
            }
            CpuOp::Linear | CpuOp::LinearResidual | CpuOp::LinearGelu { .. } => {
                let matmul = tensors[0]
                    .matmul_requirements(&tensors[1])
                    .map_err(str::to_string)?;
                add_scratch(self, &matmul.output, "linear_matmul")?;
                if matches!(&op, CpuOp::LinearResidual) {
                    let requirement = CpuTensorRequirement::new(
                        &self.values[outputs[0].index()].shape,
                        self.values[outputs[0].index()].dtype,
                    );
                    add_scratch(self, &requirement, "linear_bias")?;
                }
                let gelu = if let CpuOp::LinearGelu { approximate, dual } = &op {
                    if !*dual {
                        let requirement = CpuTensorRequirement::new(
                            &self.values[outputs[0].index()].shape,
                            self.values[outputs[0].index()].dtype,
                        );
                        add_scratch(self, &requirement, "linear_activation")?;
                    }
                    let expr = if *approximate {
                        Expr::GeluTanh(Box::new(Expr::Input(0)))
                    } else {
                        Expr::Gelu(Box::new(Expr::Input(0)))
                    };
                    let program = fusion::prepare(&[expr]);
                    let requirement = fusion::scratch_requirement(&program, tensors[0].dtype())?;
                    add_scratch(self, &requirement, "linear_gelu")?;
                    Some(program)
                } else {
                    None
                };
                CpuAlgorithmPlan::Linear { matmul, gelu }
            }
            CpuOp::QuantizedLinear {
                codec,
                weight_shape,
            } => {
                let requirements = quantized::linear_requirements(
                    &tensors[0],
                    &tensors[1],
                    tensors.get(2),
                    *codec,
                    *weight_shape,
                )?;
                let output = &self.values[outputs[0].index()];
                if output.shape.as_ref() != requirements.output.shape
                    || output.dtype != requirements.output.dtype
                {
                    return Err(
                        "compile: quantized_linear output does not match logical node shape"
                            .to_string(),
                    );
                }
                CpuAlgorithmPlan::Quantized(requirements)
            }
            CpuOp::QuantizedEmbedding {
                codec,
                weight_shape,
            } => {
                let requirements = quantized::embedding_requirements(
                    &tensors[0],
                    &tensors[1],
                    *codec,
                    *weight_shape,
                )?;
                let output = &self.values[outputs[0].index()];
                if output.shape.as_ref() != requirements.output.shape
                    || output.dtype != requirements.output.dtype
                {
                    return Err(
                        "compile: quantized_embedding output does not match logical node shape"
                            .to_string(),
                    );
                }
                CpuAlgorithmPlan::Quantized(requirements)
            }
            CpuOp::KdaRecurrence { .. } => {
                let requirements = composed::kda_chunk_forward_requirements(
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                    &tensors[3],
                    &tensors[4],
                    true,
                )?;
                let state = requirements
                    .state_next
                    .as_ref()
                    .expect("stateful KDA declares next state");
                for (requirement, name) in [
                    (state, "kda_initial_state"),
                    (
                        &CpuTensorRequirement::new(tensors[3].shape(), tensors[3].dtype()),
                        "kda_masked_decay",
                    ),
                    (
                        &CpuTensorRequirement::new(tensors[4].shape(), tensors[4].dtype()),
                        "kda_masked_beta",
                    ),
                ] {
                    staging.push(self.requirement_value(
                        requirement,
                        SegmentOwnership::InvocationStaging,
                        format!("command_{index}_{name}"),
                    )?);
                }
                state_outputs.push(self.requirement_value(
                    state,
                    SegmentOwnership::StateTransaction,
                    format!("command_{index}_kda_next_state"),
                )?);
                CpuAlgorithmPlan::KdaForward(requirements)
            }
            CpuOp::ConvState { .. } => {
                let requirements =
                    composed::short_conv1d_forward_requirements(&tensors[0], &tensors[1], false)?;
                let state = CpuTensorRequirement::new(
                    &[
                        requirements.batch,
                        requirements.kernel - 1,
                        requirements.channels,
                    ],
                    DType::F32,
                );
                staging.push(self.requirement_value(
                    &state,
                    SegmentOwnership::InvocationStaging,
                    format!("command_{index}_conv_initial_state"),
                )?);
                state_outputs.push(self.requirement_value(
                    &state,
                    SegmentOwnership::StateTransaction,
                    format!("command_{index}_conv_next_state"),
                )?);
                CpuAlgorithmPlan::ShortConvForward(requirements)
            }
            CpuOp::KvAttention { .. } => CpuAlgorithmPlan::None,
            _ => CpuAlgorithmPlan::None,
        };
        let instruction = &mut self.instructions[index];
        instruction.scratch = scratch
            .into_iter()
            .map(ValueUse::read_write)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        instruction.staging = staging
            .into_iter()
            .map(ValueUse::read_write)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        instruction.state = state_outputs
            .into_iter()
            .map(ValueUse::write)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        instruction.effects = InstructionEffects {
            may_fail: true,
            has_side_effects: matches!(
                op,
                CpuOp::KdaRecurrence { .. } | CpuOp::ConvState { .. } | CpuOp::KvAttention { .. }
            ),
        };
        let CpuInstruction::Operation {
            plan: instruction_plan,
            ..
        } = &mut instruction.kind
        else {
            unreachable!("lowerer only prepares operation instructions")
        };
        *instruction_plan = plan;
        Ok(())
    }

    fn constant(&mut self, node: &Arc<Node>, payload: Value) -> Result<(), String> {
        let value = self.value_with_layout(
            &node.shape,
            node.dtype,
            payload.tensor().layout.clone(),
            CpuValueStorage::Constant,
            "constant",
        )?;
        self.constants.push(CpuConstant { value, payload });
        self.node_values.insert(node.id, Box::new([value]));
        Ok(())
    }

    fn selector(&mut self, node: &Arc<Node>, of: &Arc<Node>, index: usize) -> Result<(), String> {
        let value = self
            .node_values
            .get(&of.id)
            .and_then(|values| values.get(index))
            .copied()
            .ok_or_else(|| {
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
                if !region.device.is_cpu() {
                    return Err(format!(
                        "compile: impossible CPU optimization plan contains an elementwise region for {}",
                        region.device.name()
                    ));
                }
                ensure_fusion_dtype(region.dtype)?;
                let inputs = self.dense_values(index, &region.inputs)?;
                let strides = inputs
                    .iter()
                    .map(|value| {
                        self.values[value.index()]
                            .layout
                            .broadcast_to(&region.shape)
                            .strides()
                            .to_vec()
                    })
                    .collect();
                let outputs = self.region_outputs(region_id, 1, &region.shape, region.dtype)?;
                (
                    CpuOp::FusedElementwise {
                        strides,
                        shape: region.shape.clone(),
                        exprs: vec![region.output.expression.clone()].into_boxed_slice(),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::ElementwiseReduce(region) => {
                if !region.device.is_cpu() {
                    return Err(format!(
                        "compile: impossible CPU optimization plan contains an elementwise-reduce region for {}",
                        region.device.name()
                    ));
                }
                ensure_fusion_dtype(region.dtype)?;
                let inputs = self.dense_values(index, &region.inputs)?;
                let strides = inputs
                    .iter()
                    .map(|value| {
                        self.values[value.index()]
                            .layout
                            .broadcast_to(&region.input_shape)
                            .strides()
                            .to_vec()
                    })
                    .collect();
                let outputs = self.region_outputs(region_id, 1, &region.shape, region.dtype)?;
                (
                    CpuOp::FusedReduce {
                        strides,
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
                if !region.device.is_cpu() {
                    return Err(format!(
                        "compile: impossible CPU optimization plan contains a multi-output region for {}",
                        region.device.name()
                    ));
                }
                ensure_fusion_dtype(region.dtype)?;
                let inputs = self.dense_values(index, &region.inputs)?;
                let strides = inputs
                    .iter()
                    .map(|value| {
                        self.values[value.index()]
                            .layout
                            .broadcast_to(&region.shape)
                            .strides()
                            .to_vec()
                    })
                    .collect();
                let outputs = self.region_outputs(
                    region_id,
                    region.outputs.len(),
                    &region.shape,
                    region.dtype,
                )?;
                (
                    CpuOp::FusedElementwise {
                        strides,
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
            NativeRegion::AdamW(region) => {
                if !region.device.is_cpu() {
                    return Err(format!(
                        "compile: impossible CPU optimization plan contains an AdamW region for {}",
                        region.device.name()
                    ));
                }
                ensure_fusion_dtype(region.dtype)?;
                let inputs = self.dense_values(index, &region.inputs)?;
                let outputs = self.region_outputs(
                    region_id,
                    region.expressions.len(),
                    &region.shape,
                    region.dtype,
                )?;
                (
                    CpuOp::AdamW {
                        beta1: region.options.beta1,
                        beta2: region.options.beta2,
                        eps: region.options.eps,
                        weight_decay: region.options.weight_decay,
                        implementation: OptimizerImplementation::Fused,
                        fused_exprs: Some(region.expressions.clone()),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::AdamWGroup(region) => {
                if !region.device.is_cpu() {
                    return Err(format!(
                        "compile: impossible CPU optimization plan contains an AdamW-group region for {}",
                        region.device.name()
                    ));
                }
                ensure_fusion_dtype(region.dtype)?;
                // CpuOp::AdamWGroup uses blocked command inputs and interleaves
                // these lanes immediately before fusion execution.
                let mut dense_inputs = Vec::with_capacity(region.inputs.len());
                for lane in 0..4 {
                    dense_inputs.extend(
                        region
                            .parameter_inputs
                            .iter()
                            .map(|parameter| parameter[lane]),
                    );
                }
                dense_inputs.extend(region.scalar_inputs);
                let inputs = self.dense_values(index, &dense_inputs)?;
                let outputs = self.region_outputs(
                    region_id,
                    region.expressions.len(),
                    &region.shape,
                    region.dtype,
                )?;
                (
                    CpuOp::AdamWGroup {
                        parameters: region.parameter_inputs.len(),
                        fused_exprs: region.expressions.clone(),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::Sgd(region) => {
                if !region.device.is_cpu() {
                    return Err(format!(
                        "compile: impossible CPU optimization plan contains an SGD region for {}",
                        region.device.name()
                    ));
                }
                ensure_fusion_dtype(region.dtype)?;
                // CpuOp::Sgd expects [param, grad, velocity, first, lr].
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
                    CpuOp::Sgd {
                        momentum: region.options.momentum,
                        dampening: region.options.dampening,
                        nesterov: region.options.nesterov,
                        weight_decay: region.options.weight_decay,
                        implementation: OptimizerImplementation::Fused,
                        fused_exprs: Some(region.expressions.clone()),
                    },
                    inputs,
                    outputs,
                )
            }
            NativeRegion::LinearResidual(region) => {
                return Err(format!(
                    "compile: impossible CPU optimization plan contains Metal-only linear-residual region for {}",
                    region.device.name()
                ));
            }
            NativeRegion::LinearGelu(region) => {
                return Err(format!(
                    "compile: impossible CPU optimization plan contains Metal-only linear-GELU region for {}",
                    region.device.name()
                ));
            }
        };
        self.command(op, inputs, outputs.clone())?;
        self.prepare_last_command()?;
        self.bind_region_outputs(index, plan, region_id, region, &outputs)
    }

    fn lower(&mut self, node: &Arc<Node>) -> Result<(), String> {
        if !node.device.is_cpu() {
            return Err(format!(
                "compile: CPU lowering does not support device {}",
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
        validate_cpu_support(node)?;

        match &node.kind {
            NodeKind::Leaf(slot) => {
                let payload = self.preloaded_generated.remove(&node.id).ok_or_else(|| {
                    format!(
                        "compile: generated binding {} has no prepared value",
                        node.id
                    )
                })?;
                if payload.shape() != node.shape || payload.dtype() != node.dtype {
                    return Err(format!(
                        "compile: generated binding {} changed shape or dtype",
                        node.id
                    ));
                }
                let layout = &payload.tensor().layout;
                if layout.checked_max_index().is_none()
                    || layout.max_index() > payload.tensor().buffer.len()
                {
                    return Err(format!(
                        "compile: generated binding {} has an unsupported layout",
                        node.id
                    ));
                }
                if self.options.constant_weights() {
                    self.constant(node, payload)?;
                    return Ok(());
                }
                let generated = u32::try_from(self.generated.len())
                    .map_err(|_| "compile: too many generated bindings".to_string())?;
                let value = self.value_with_layout(
                    &node.shape,
                    node.dtype,
                    payload.tensor().layout.clone(),
                    CpuValueStorage::External,
                    format!("generated_binding_{generated}"),
                )?;
                self.generated.push(payload);
                self.generated_slots.push(Arc::as_ptr(slot) as usize);
                self.bindings.push(CpuBinding {
                    value,
                    source: CpuBindingSource::Generated(generated),
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
                        if source == CpuDeclaredSource::StateCursor {
                            let state = self.state.expect("state cursor has a state plan");
                            let expected_shape = if state.cursor_tensor {
                                vec![state.batch]
                            } else {
                                Vec::new()
                            };
                            if node.dtype != DType::I64 || expected_shape != node.shape {
                                return Err(format!(
                                    "compile: cursor slot {slot} must be i64 with shape {expected_shape:?}"
                                ));
                            }
                        }
                        let padded = self.padded_slot == Some(*slot)
                            && matches!(source, CpuDeclaredSource::Tensor(_))
                            && self
                                .state
                                .is_some_and(|state| node.shape.first() == Some(&state.batch));
                        let value = self.value(
                            &node.shape,
                            node.dtype,
                            if matches!(
                                source,
                                CpuDeclaredSource::Scalar(_) | CpuDeclaredSource::StateCursor
                            ) || padded
                            {
                                CpuValueStorage::Planned(SegmentOwnership::InvocationStaging)
                            } else {
                                CpuValueStorage::External
                            },
                            if source == CpuDeclaredSource::StateCursor {
                                "decode_cursor".to_string()
                            } else {
                                format!("binding_{slot}")
                            },
                        )?;
                        self.declared_values.insert(*slot, value);
                        match source {
                            CpuDeclaredSource::Tensor(source) => {
                                if padded {
                                    self.padded_bindings
                                        .push(CpuPaddedBinding { value, source });
                                } else {
                                    self.bindings.push(CpuBinding {
                                        value,
                                        source: CpuBindingSource::Declared(source),
                                    });
                                }
                            }
                            CpuDeclaredSource::Scalar(source) => {
                                self.scalar_bindings
                                    .push(CpuScalarBinding { value, source });
                            }
                            CpuDeclaredSource::StateCursor => self.state_cursor = Some(value),
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
                let payload = value_from_bytes(data, shape, *dtype)
                    .map_err(|error| format!("compile: from_bytes: {error}"))?;
                return self.constant(node, payload);
            }
            NodeKind::Zeros { shape, dtype, .. } => {
                return self.constant(node, Value(Tensor::zeros(shape, *dtype)));
            }
            NodeKind::Ones { shape, dtype, .. } => {
                return self.constant(node, Value(Tensor::ones(shape, *dtype)));
            }
            NodeKind::Full {
                shape,
                value,
                dtype,
                ..
            } => {
                return self.constant(node, Value(Tensor::full(shape, *value, *dtype)));
            }
            NodeKind::Arange {
                start,
                end,
                step,
                dtype,
                ..
            } => {
                return self.constant(node, Value(Tensor::arange(*start, *end, *step, *dtype)));
            }
            NodeKind::Eye { n, dtype, .. } => {
                return self.constant(node, Value(Tensor::eye(*n, *dtype)));
            }
            NodeKind::SdpaBackwardOut { of, index }
            | NodeKind::ChunkedHeadCeBackwardOut { of, index }
            | NodeKind::KdaBackwardOut { of, index }
            | NodeKind::AdamWOut { step: of, index }
            | NodeKind::SgdOut { step: of, index } => {
                return self.selector(node, of, *index as usize)
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
                let source_layout = &self.values[source.index()].layout;
                if source_layout.is_contiguous() {
                    let layout = effect_torch_runtime::Layout::new(
                        shape.clone(),
                        effect_torch_runtime::Layout::contiguous(shape.clone())
                            .strides()
                            .to_vec(),
                        source_layout.offset(),
                    );
                    let value =
                        self.alias_value(source, shape, layout, format!("{}_reshape", node.id))?;
                    self.node_values.insert(node.id, Box::new([value]));
                    return Ok(());
                }
            }
            NodeKind::Permute { a, dims } => {
                let source = self.child_value(a)?;
                let layout = self.values[source.index()].layout.permute(dims);
                let value =
                    self.alias_value(source, &node.shape, layout, format!("{}_permute", node.id))?;
                self.node_values.insert(node.id, Box::new([value]));
                return Ok(());
            }
            NodeKind::Slice { a, ranges } => {
                let source = self.child_value(a)?;
                let mut layout = self.values[source.index()].layout.clone();
                for (dimension, &(start, stop, stride)) in ranges.iter().enumerate() {
                    let length = stop.saturating_sub(start).div_ceil(stride);
                    if length == 0 {
                        return Err("compile: zero-length CPU slices are unsupported".to_string());
                    }
                    layout = layout.narrow(dimension, start, (length - 1) * stride + 1);
                    let mut strides = layout.strides().to_vec();
                    strides[dimension] = strides[dimension]
                        .checked_mul(stride)
                        .ok_or_else(|| "compile: slice stride overflow".to_string())?;
                    let mut shape = layout.shape().to_vec();
                    shape[dimension] = length;
                    layout = effect_torch_runtime::Layout::new(shape, strides, layout.offset());
                }
                if layout.shape() != node.shape {
                    return Err("compile: slice layout does not match inferred shape".to_string());
                }
                let value =
                    self.alias_value(source, &node.shape, layout, format!("{}_slice", node.id))?;
                self.node_values.insert(node.id, Box::new([value]));
                return Ok(());
            }
            NodeKind::BroadcastTo { a, shape } => {
                let source = self.child_value(a)?;
                let layout = self.values[source.index()].layout.broadcast_to(shape);
                let value = self.alias_value(
                    source,
                    &node.shape,
                    layout,
                    format!("{}_broadcast", node.id),
                )?;
                self.node_values.insert(node.id, Box::new([value]));
                return Ok(());
            }
            NodeKind::PositionEmbedding { weight, seq_len } => {
                let source = self.child_value(weight)?;
                let layout = self.values[source.index()].layout.narrow(0, 0, *seq_len);
                let value = self.alias_value(
                    source,
                    &node.shape,
                    layout,
                    format!("{}_position_embedding", node.id),
                )?;
                self.node_values.insert(node.id, Box::new([value]));
                return Ok(());
            }
            _ => {}
        }

        let inputs = node_children(&node.kind)
            .iter()
            .map(|child| self.child_value(child))
            .collect::<Result<Vec<_>, _>>()?;
        let output_metadata: Vec<(Vec<usize>, DType)> = match &node.kind {
            NodeKind::ChunkedHeadCeBackward {
                x, weight, bias, ..
            } => vec![
                (x.shape.clone(), x.dtype),
                (weight.shape.clone(), weight.dtype),
                (bias.shape.clone(), bias.dtype),
            ],
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

        let optimizer_implementation = |dtype: DType| {
            if self.options.optimize
                && self.options.environment.fusion
                && fusion::is_supported(&cpu_device(), dtype)
            {
                OptimizerImplementation::Fused
            } else {
                OptimizerImplementation::Composed
            }
        };
        let op = match &node.kind {
            NodeKind::Randn { shape, dtype, .. } => CpuOp::Randn {
                shape: shape.clone().into_boxed_slice(),
                dtype: *dtype,
                provenance: *self.random_provenance.get(&node.id).ok_or_else(|| {
                    "compile: random node lost its semantic provenance".to_string()
                })?,
            },
            NodeKind::Uniform {
                lo,
                hi,
                shape,
                dtype,
                ..
            } => CpuOp::Uniform {
                lo: *lo,
                hi: *hi,
                shape: shape.clone().into_boxed_slice(),
                dtype: *dtype,
                provenance: *self.random_provenance.get(&node.id).ok_or_else(|| {
                    "compile: random node lost its semantic provenance".to_string()
                })?,
            },
            NodeKind::Add { .. } => CpuOp::Binary(CpuBinaryOp::Add),
            NodeKind::Sub { .. } => CpuOp::Binary(CpuBinaryOp::Sub),
            NodeKind::Mul { .. } => CpuOp::Binary(CpuBinaryOp::Mul),
            NodeKind::Div { .. } => CpuOp::Binary(CpuBinaryOp::Div),
            NodeKind::Eq { .. } => CpuOp::Binary(CpuBinaryOp::Eq),
            NodeKind::Gt { .. } => CpuOp::Binary(CpuBinaryOp::Gt),
            NodeKind::Lt { .. } => CpuOp::Binary(CpuBinaryOp::Lt),
            NodeKind::Ge { .. } => CpuOp::Binary(CpuBinaryOp::Ge),
            NodeKind::Le { .. } => CpuOp::Binary(CpuBinaryOp::Le),
            NodeKind::Maximum { .. } => CpuOp::Binary(CpuBinaryOp::Maximum),
            NodeKind::Minimum { .. } => CpuOp::Binary(CpuBinaryOp::Minimum),
            NodeKind::Neg { .. } => CpuOp::Unary(CpuUnaryOp::Neg),
            NodeKind::Abs { .. } => CpuOp::Unary(CpuUnaryOp::Abs),
            NodeKind::Sqrt { .. } => CpuOp::Unary(CpuUnaryOp::Sqrt),
            NodeKind::Exp { .. } => CpuOp::Unary(CpuUnaryOp::Exp),
            NodeKind::Log { .. } => CpuOp::Unary(CpuUnaryOp::Log),
            NodeKind::Sin { .. } => CpuOp::Unary(CpuUnaryOp::Sin),
            NodeKind::Cos { .. } => CpuOp::Unary(CpuUnaryOp::Cos),
            NodeKind::Tanh { .. } => CpuOp::Unary(CpuUnaryOp::Tanh),
            NodeKind::Relu { .. } => CpuOp::Unary(CpuUnaryOp::Relu),
            NodeKind::Erf { .. } => CpuOp::Unary(CpuUnaryOp::Erf),
            NodeKind::Gelu { approximate, .. } => CpuOp::Unary(CpuUnaryOp::Gelu {
                approximate: *approximate,
            }),
            NodeKind::Floor { .. } => CpuOp::Unary(CpuUnaryOp::Floor),
            NodeKind::Ceil { .. } => CpuOp::Unary(CpuUnaryOp::Ceil),
            NodeKind::Round { .. } => CpuOp::Unary(CpuUnaryOp::Round),
            NodeKind::Sign { .. } => CpuOp::Unary(CpuUnaryOp::Sign),
            NodeKind::Where { .. } => CpuOp::Where,
            NodeKind::Argmax { dim, .. } => CpuOp::Argmax { dim: *dim },
            NodeKind::Argmin { dim, .. } => CpuOp::Argmin { dim: *dim },
            NodeKind::Cumsum { dim, .. } => CpuOp::Cumsum { dim: *dim },
            NodeKind::ScatterAdd { dim, .. } => CpuOp::ScatterAdd { dim: *dim },
            NodeKind::Gather { dim, .. } => CpuOp::Gather { dim: *dim },
            NodeKind::IndexSelect { dim, .. } => CpuOp::IndexSelect { dim: *dim },
            NodeKind::CrossEntropy {
                ignore_index,
                reduction,
                ..
            } => CpuOp::CrossEntropy {
                ignore_index: *ignore_index,
                reduction: *reduction,
            },
            NodeKind::CrossEntropyBackward {
                ignore_index,
                reduction,
                ..
            } => CpuOp::CrossEntropyBackward {
                ignore_index: *ignore_index,
                reduction: *reduction,
            },
            NodeKind::ChunkedHeadCe { ignore_index, .. } => CpuOp::ChunkedHeadCe {
                ignore_index: *ignore_index,
                chunk_size: self.ce_chunk_size,
            },
            NodeKind::ChunkedHeadCeBackward { ignore_index, .. } => CpuOp::ChunkedHeadCeBackward {
                ignore_index: *ignore_index,
                chunk_size: self.ce_chunk_size,
            },
            NodeKind::Sdpa {
                scale,
                causal,
                window,
                ..
            } => CpuOp::Sdpa {
                scale: *scale,
                causal: *causal,
                window: window.local(),
            },
            NodeKind::SdpaBackward {
                scale,
                causal,
                window,
                ..
            } => CpuOp::SdpaBackward {
                scale: *scale,
                causal: *causal,
                window: window.local(),
            },
            NodeKind::KdaChunk { scale, .. } => CpuOp::KdaChunk { scale: *scale },
            NodeKind::KdaRecurrence { scale, layer, .. } => CpuOp::KdaRecurrence {
                scale: *scale,
                layer: *layer,
            },
            NodeKind::KdaBackward { scale, .. } => CpuOp::KdaBackward { scale: *scale },
            NodeKind::ShortConv1d { .. } => CpuOp::ShortConv1d,
            NodeKind::ShortConv1dBackwardX { .. } => CpuOp::ShortConv1dBackwardX,
            NodeKind::ShortConv1dBackwardW { .. } => CpuOp::ShortConv1dBackwardW,
            NodeKind::ConvState { layer, .. } => CpuOp::ConvState { layer: *layer },
            NodeKind::LastTokenRow { .. } => CpuOp::LastTokenRow,
            NodeKind::PositionEmbedding { seq_len, .. } => {
                CpuOp::PositionEmbedding { seq_len: *seq_len }
            }
            NodeKind::KvAttention {
                scale,
                layer,
                window,
                ..
            } => {
                if node.dtype != DType::F32 {
                    return Err(format!(
                        "compile: kv_attention does not support CPU dtype {}",
                        node.dtype
                    ));
                }
                CpuOp::KvAttention {
                    scale: *scale,
                    layer: *layer,
                    window: *window,
                }
            }
            NodeKind::RotaryEmbedding {
                theta,
                offset,
                layout,
                ..
            } => CpuOp::RotaryEmbedding {
                theta: *theta,
                cursor_offset: matches!(offset, PositionOffset::Cursor),
                layout: *layout,
            },
            NodeKind::RotaryEmbeddingBackward { theta, layout, .. } => {
                CpuOp::RotaryEmbeddingBackward {
                    theta: *theta,
                    layout: *layout,
                }
            }
            NodeKind::Linear { .. } => CpuOp::Linear,
            NodeKind::QuantizedLinear {
                codec,
                weight_shape,
                ..
            } => CpuOp::QuantizedLinear {
                codec: *codec,
                weight_shape: *weight_shape,
            },
            NodeKind::QuantizedEmbedding {
                codec,
                weight_shape,
                ..
            } => CpuOp::QuantizedEmbedding {
                codec: *codec,
                weight_shape: *weight_shape,
            },
            NodeKind::LayerNorm { eps, .. } => CpuOp::LayerNorm { eps: *eps },
            NodeKind::RmsNorm { eps, .. } => CpuOp::RmsNorm { eps: *eps },
            NodeKind::LayerNormBackward { eps, .. } => CpuOp::LayerNormBackward { eps: *eps },
            NodeKind::Conv1d {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => CpuOp::Conv1d {
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
            } => CpuOp::Conv2d {
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
            } => CpuOp::ConvTranspose1d {
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
            } => CpuOp::ConvTranspose2d {
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
            } => CpuOp::Conv1dBackwardW {
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
            } => CpuOp::Conv2dBackwardW {
                kernel: *kernel,
                out_channels: *out_channels,
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
                groups: *groups,
            },
            NodeKind::Pow { exp, .. } => CpuOp::Unary(CpuUnaryOp::Pow {
                exponent_bits: exp.to_bits(),
            }),
            NodeKind::Cast { dtype, .. } => CpuOp::Unary(CpuUnaryOp::Cast { dtype: *dtype }),
            NodeKind::Sum { dims, keepdims, .. } => CpuOp::Reduce {
                op: CpuReduceOp::Sum,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Mean { dims, keepdims, .. } => CpuOp::Reduce {
                op: CpuReduceOp::Mean,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Max { dims, keepdims, .. } => CpuOp::Reduce {
                op: CpuReduceOp::Max,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Min { dims, keepdims, .. } => CpuOp::Reduce {
                op: CpuReduceOp::Min,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Prod { dims, keepdims, .. } => CpuOp::Reduce {
                op: CpuReduceOp::Prod,
                dims: dims.clone().into_boxed_slice(),
                keepdims: *keepdims,
            },
            NodeKind::Reshape { shape, .. } => CpuOp::Reshape {
                shape: shape.clone().into_boxed_slice(),
            },
            NodeKind::Permute { dims, .. } => CpuOp::Permute {
                dims: dims.clone().into_boxed_slice(),
            },
            NodeKind::Slice { ranges, .. } => CpuOp::Slice {
                ranges: ranges.clone().into_boxed_slice(),
            },
            NodeKind::Concat { dim, .. } => CpuOp::Binary(CpuBinaryOp::Concat { dim: *dim }),
            NodeKind::BroadcastTo { shape, .. } => CpuOp::BroadcastTo {
                shape: shape.clone().into_boxed_slice(),
            },
            NodeKind::Matmul { .. } => CpuOp::Binary(CpuBinaryOp::Matmul),
            NodeKind::Inverse { .. } => CpuOp::Unary(CpuUnaryOp::Inverse),
            NodeKind::Det { .. } => CpuOp::Unary(CpuUnaryOp::Det),
            NodeKind::Solve { .. } => CpuOp::Binary(CpuBinaryOp::Solve),
            NodeKind::AdamWStep {
                beta1,
                beta2,
                eps,
                weight_decay,
                param,
                ..
            } => {
                let implementation = optimizer_implementation(param.dtype);
                let fused_exprs = (implementation == OptimizerImplementation::Fused).then(|| {
                    fusion::adamw_exprs(*beta1, *beta2, *eps, *weight_decay)
                        .into_iter()
                        .collect()
                });
                CpuOp::AdamW {
                    beta1: *beta1,
                    beta2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    implementation,
                    fused_exprs,
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
                let implementation = optimizer_implementation(param.dtype);
                let fused_exprs = (implementation == OptimizerImplementation::Fused).then(|| {
                    fusion::sgd_exprs(*momentum, *dampening, *nesterov, *weight_decay)
                        .into_iter()
                        .collect()
                });
                CpuOp::Sgd {
                    momentum: *momentum,
                    dampening: *dampening,
                    nesterov: *nesterov,
                    weight_decay: *weight_decay,
                    implementation,
                    fused_exprs,
                }
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
        self.command(op, inputs, outputs.clone())?;
        self.prepare_last_command()?;
        self.node_values.insert(node.id, outputs.into_boxed_slice());
        Ok(())
    }

    fn finish(
        mut self,
        outputs: Vec<ValueId>,
        driver: &mut CompilerDriver<'_>,
    ) -> Result<(CpuExecutable, Vec<Value>, Vec<usize>), String> {
        if !self.preloaded_generated.is_empty() {
            return Err("compile: not all prepared generated bindings were lowered".to_string());
        }
        for output in &outputs {
            let mut root = *output;
            while let CpuValueStorage::Alias { source, .. } = self.storage[root.index()] {
                root = source;
            }
            if matches!(
                self.storage[root.index()],
                CpuValueStorage::Planned(
                    SegmentOwnership::Workspace | SegmentOwnership::InvocationStaging
                )
            ) {
                self.storage[root.index()] =
                    CpuValueStorage::Planned(SegmentOwnership::ProvisionalOutput);
            }
        }

        let mut constant_slots = 0u32;
        let values =
            self.values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let id = ValueId::from_index(index)
                        .ok_or_else(|| "compile: too many CPU values".to_string())?;
                    let bytes = if matches!(self.storage[index], CpuValueStorage::Alias { .. }) {
                        value
                            .layout
                            .checked_max_index()
                            .and_then(|end| end.checked_sub(value.layout.offset()))
                            .and_then(|elements| elements.checked_mul(value.dtype.size_in_bytes()))
                            .ok_or_else(|| "compile: alias byte span overflow".to_string())?
                    } else {
                        value
                            .shape
                            .iter()
                            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                            .and_then(|elements| elements.checked_mul(value.dtype.size_in_bytes()))
                            .ok_or_else(|| "compile: value byte size overflow".to_string())?
                    };
                    let storage = match self.storage[index] {
                        CpuValueStorage::External => ValueStorage::Fixed {
                            class: StorageClass::ExternalInput,
                            location: Location::External { slot: id.get() },
                        },
                        CpuValueStorage::Constant => {
                            let slot = constant_slots;
                            constant_slots = constant_slots
                                .checked_add(1)
                                .ok_or_else(|| "compile: too many constants".to_string())?;
                            ValueStorage::Fixed {
                                class: StorageClass::PersistentConstant,
                                location: Location::Persistent { slot },
                            }
                        }
                        CpuValueStorage::Planned(ownership) => ValueStorage::Planned {
                            class: match ownership {
                                SegmentOwnership::ProvisionalOutput => StorageClass::EscapingOutput,
                                SegmentOwnership::StateTransaction => StorageClass::PersistentState,
                                SegmentOwnership::Workspace
                                | SegmentOwnership::InvocationStaging => StorageClass::Workspace,
                            },
                            alignment: CPU_STORAGE_ALIGNMENT,
                            memory_space: NativeMemorySpace::Cpu,
                            ownership,
                        },
                        CpuValueStorage::Alias {
                            source,
                            byte_offset,
                        } => ValueStorage::Alias {
                            source,
                            byte_offset,
                        },
                    };
                    Ok(CpuLoweredValue {
                        shape: value.shape.clone(),
                        dtype: value.dtype,
                        layout: value.layout.clone(),
                        declaration: ValueDecl {
                            id,
                            name: self.names[index].clone(),
                            bytes,
                            storage,
                        },
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
        let invocation_values = self
            .scalar_bindings
            .iter()
            .map(|binding| binding.value)
            .chain(self.padded_bindings.iter().map(|binding| binding.value))
            .chain(self.state_cursor)
            .collect::<Vec<_>>();
        let stateful = self.state.is_some()
            || self.instructions.iter().any(|instruction| {
                instruction.effects.has_side_effects || !instruction.state.is_empty()
            });
        let mut instructions = Vec::with_capacity(
            self.instructions.len() + 1 + usize::from(!invocation_values.is_empty()),
        );
        if !invocation_values.is_empty() {
            instructions.push(
                LoweredInstruction::new(
                    InstructionId::from_index(0)
                        .ok_or_else(|| "compile: too many CPU instructions".to_string())?,
                    CpuInstruction::PrepareInvocation,
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
        let mut operation_ids = Vec::with_capacity(self.instructions.len());
        for mut instruction in self.instructions {
            let id = InstructionId::from_index(instructions.len())
                .ok_or_else(|| "compile: too many CPU instructions".to_string())?;
            instruction.id = id;
            instructions.push(instruction);
            operation_ids.push(id);
        }
        let finalization_inputs = outputs
            .iter()
            .copied()
            .chain(
                instructions
                    .iter()
                    .flat_map(|instruction| instruction.state.iter().map(|use_| use_.value)),
            )
            .map(ValueUse::read)
            .collect::<Vec<_>>();
        instructions.push(
            LoweredInstruction::new(
                InstructionId::from_index(instructions.len())
                    .ok_or_else(|| "compile: too many CPU instructions".to_string())?,
                CpuInstruction::FinalizeInvocation,
                finalization_inputs,
                Vec::new(),
            )
            .with_effects(InstructionEffects {
                may_fail: stateful,
                has_side_effects: stateful,
            }),
        );
        let program = Arc::new(LoweredProgram::new(values, instructions, outputs));
        let memory = driver
            .plan_memory(
                &program,
                &MemoryPlannerConfig::uniform(
                    NativeMemorySpace::Cpu,
                    usize::MAX / 2,
                    CPU_STORAGE_ALIGNMENT,
                    1,
                ),
            )
            .map_err(|error| format!("compile: CPU memory planning failed: {error}"))?;
        let physical = driver.phase(PHYSICAL_PLANNING_PHASE, || {
            Ok::<_, String>(
                operation_ids
                    .into_iter()
                    .map(CpuPhysicalCommand::Encode)
                    .collect::<Vec<_>>(),
            )
        })?;
        let diagnostics = build_executable_diagnostics(
            &program,
            &memory,
            &driver.prepared().index,
            DiagnosticsInput {
                command_count: physical.len(),
                ..DiagnosticsInput::default()
            },
            CpuInstruction::name,
        );
        Ok((
            CpuExecutable {
                signature: driver.prepared().signature.clone(),
                program,
                physical: physical.into_boxed_slice(),
                bindings: self.bindings.into_boxed_slice(),
                scalar_bindings: self.scalar_bindings.into_boxed_slice(),
                padded_bindings: self.padded_bindings.into_boxed_slice(),
                constants: self.constants.into_boxed_slice(),
                options: self.options,
                memory,
                diagnostics,
                compiler_work: CompilerWorkReport::default(),
                state_cursor: self.state_cursor,
            },
            self.generated,
            self.generated_slots,
        ))
    }
}

fn value_from_bytes(bytes: &[u8], shape: &[usize], dtype: DType) -> Result<Value, String> {
    let elements = shape
        .iter()
        .try_fold(1usize, |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| "compile: constant element count overflow".to_string())?;
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| "compile: constant byte length overflow".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "compile: constant expected {expected} bytes for {dtype} {shape:?}, got {}",
            bytes.len()
        ));
    }
    macro_rules! decode {
        ($type:ty, $width:literal) => {
            Tensor::from_vec(
                bytes
                    .chunks_exact($width)
                    .map(|chunk| <$type>::from_le_bytes(chunk.try_into().expect("validated chunk")))
                    .collect(),
                shape.to_vec(),
            )
        };
    }
    let tensor = match dtype {
        DType::F32 => decode!(f32, 4),
        DType::F64 => decode!(f64, 8),
        DType::F16 => decode!(half::f16, 2),
        DType::BF16 => decode!(half::bf16, 2),
        DType::U8 => Tensor::from_vec(bytes.to_vec(), shape.to_vec()),
        DType::U32 => decode!(u32, 4),
        DType::I64 => decode!(i64, 8),
    };
    Ok(Value(tensor))
}

fn ensure_fusion_dtype(dtype: DType) -> Result<(), String> {
    if matches!(dtype, DType::F32 | DType::F64) {
        Ok(())
    } else {
        Err(format!(
            "compile: fused CPU operation does not support dtype {dtype}"
        ))
    }
}

fn validate_cpu_support(node: &Node) -> Result<(), String> {
    let same_dtype = |operation: &str, tensors: &[&Arc<Node>]| {
        if tensors
            .windows(2)
            .all(|pair| pair[0].dtype == pair[1].dtype)
        {
            Ok(())
        } else {
            Err(format!(
                "compile: {operation} does not support mixed CPU dtypes"
            ))
        }
    };
    let matmul_dtype = |operation: &str, dtype: DType| {
        if matches!(dtype, DType::F16 | DType::BF16) {
            Err(format!(
                "compile: {operation} does not support CPU dtype {dtype}"
            ))
        } else {
            Ok(())
        }
    };
    let float_dtype = |operation: &str, dtype: DType| {
        if dtype.is_float() {
            Ok(())
        } else {
            Err(format!(
                "compile: {operation} requires a floating CPU dtype, got {dtype}"
            ))
        }
    };

    match &node.kind {
        NodeKind::Arange { step, .. } if *step == 0.0 => {
            Err("compile: arange step must not be zero".to_string())
        }
        NodeKind::Neg { a } if a.dtype == DType::U8 => {
            Err("compile: neg does not support CPU dtype u8".to_string())
        }
        NodeKind::Abs { a }
        | NodeKind::Sqrt { a }
        | NodeKind::Exp { a }
        | NodeKind::Log { a }
        | NodeKind::Sin { a }
        | NodeKind::Cos { a }
        | NodeKind::Tanh { a }
        | NodeKind::Erf { a }
        | NodeKind::Gelu { a, .. }
        | NodeKind::Floor { a }
        | NodeKind::Ceil { a }
        | NodeKind::Round { a }
        | NodeKind::Sign { a }
        | NodeKind::Pow { a, .. } => float_dtype("unary operation", a.dtype),
        NodeKind::Add { a, b }
        | NodeKind::Sub { a, b }
        | NodeKind::Mul { a, b }
        | NodeKind::Div { a, b }
        | NodeKind::Maximum { a, b }
        | NodeKind::Minimum { a, b } => {
            if a.dtype == b.dtype
                || (a.dtype.is_float()
                    && b.dtype.is_float()
                    && (a.shape.is_empty() ^ b.shape.is_empty()))
            {
                Ok(())
            } else {
                Err("compile: binary operation does not support these mixed CPU dtypes".to_string())
            }
        }
        NodeKind::Eq { a, b }
        | NodeKind::Gt { a, b }
        | NodeKind::Lt { a, b }
        | NodeKind::Ge { a, b }
        | NodeKind::Le { a, b }
        | NodeKind::Concat { a, b, .. } => same_dtype("operation", &[a, b]),
        NodeKind::Matmul { a, b } => {
            same_dtype("matmul", &[a, b])?;
            matmul_dtype("matmul", a.dtype)
        }
        NodeKind::Linear { x, weight, bias } => {
            same_dtype("linear", &[x, weight, bias])?;
            matmul_dtype("linear", x.dtype)
        }
        NodeKind::Sdpa { q, .. } | NodeKind::SdpaBackward { q, .. } => {
            matmul_dtype("sdpa", q.dtype)
        }
        NodeKind::ChunkedHeadCe { x, .. } | NodeKind::ChunkedHeadCeBackward { x, .. } => {
            matmul_dtype("chunked_head_ce", x.dtype)
        }
        NodeKind::KvAttention { q, .. } if q.dtype != DType::F32 => Err(format!(
            "compile: kv_attention does not support CPU dtype {}",
            q.dtype
        )),
        NodeKind::RotaryEmbedding { x, .. } if x.dtype != DType::F32 => Err(format!(
            "compile: rotary_embedding does not support CPU dtype {}",
            x.dtype
        )),
        NodeKind::RotaryEmbeddingBackward { g, .. } if g.dtype != DType::F32 => Err(format!(
            "compile: rotary_embedding_backward does not support CPU dtype {}",
            g.dtype
        )),
        NodeKind::LayerNorm {
            x, weight, bias, ..
        } => {
            float_dtype("layer_norm", x.dtype)?;
            same_dtype("layer_norm", &[x, weight, bias])
        }
        NodeKind::RmsNorm { x, weight, .. } => {
            float_dtype("rms_norm", x.dtype)?;
            if let Some(weight) = weight {
                same_dtype("rms_norm", &[x, weight])?;
            }
            Ok(())
        }
        NodeKind::LayerNormBackward { x, weight, g, .. } => {
            float_dtype("layer_norm_backward", x.dtype)?;
            same_dtype("layer_norm_backward", &[x, weight, g])
        }
        NodeKind::ConvTranspose1d { x, w, .. } | NodeKind::ConvTranspose2d { x, w, .. } => {
            float_dtype("conv_transpose", x.dtype)?;
            same_dtype("conv_transpose", &[x, w])
        }
        _ => Ok(()),
    }
}

pub fn compile(
    roots: &[Arc<Node>],
    options: CompileOptions,
    ce_chunk_size: usize,
) -> Result<CpuCompilation, String> {
    compile_roots_internal(roots, options, ce_chunk_size, None, None)
}

pub fn compile_with_state_bytes(
    roots: &[Arc<Node>],
    options: CompileOptions,
    ce_chunk_size: usize,
    state_bytes: Option<usize>,
) -> Result<CpuCompilation, String> {
    compile_roots_internal(roots, options, ce_chunk_size, state_bytes, None)
}

pub fn compile_with_state(
    roots: &[Arc<Node>],
    options: CompileOptions,
    ce_chunk_size: usize,
    state: CpuStatePlan,
) -> Result<CpuCompilation, String> {
    compile_roots_internal(
        roots,
        options,
        ce_chunk_size,
        Some(state.bytes),
        Some(state),
    )
}

fn validate_generated_payload(node: &Arc<Node>, payload: &Value) -> Result<(), String> {
    if payload.shape() != node.shape || payload.dtype() != node.dtype {
        return Err(format!(
            "compile: generated binding {} changed shape or dtype",
            node.id
        ));
    }
    let tensor = payload.tensor();
    if tensor.layout.shape() != node.shape {
        return Err(format!(
            "compile: generated binding {} layout shape does not match its node",
            node.id
        ));
    }
    let Some(end) = tensor.layout.checked_max_index() else {
        return Err(format!(
            "compile: generated binding {} has an overflowing CPU layout",
            node.id
        ));
    };
    if end > tensor.buffer.len() {
        return Err(format!(
            "compile: generated binding {} CPU layout exceeds its buffer",
            node.id
        ));
    }
    Ok(())
}

pub fn load_generated_values(index: &GraphIndex) -> Result<Vec<CpuGeneratedValue>, String> {
    index
        .leaves
        .iter()
        .map(|leaf| {
            let node = index.node(leaf.node).ok_or_else(|| {
                format!(
                    "compile: generated binding node {} is out of range",
                    leaf.node
                )
            })?;
            if node.id != leaf.node_id
                || node.shape != leaf.shape
                || node.dtype != leaf.dtype
                || !node.device.is_cpu()
                || !leaf.device.is_cpu()
            {
                return Err(format!(
                    "compile: generated binding {} does not match its graph index metadata",
                    leaf.node_id
                ));
            }
            let NodeKind::Leaf(slot) = &node.kind else {
                return Err(format!(
                    "compile: generated binding {} is not a leaf",
                    leaf.node_id
                ));
            };
            if Arc::as_ptr(slot) as usize != leaf.slot_identity {
                return Err(format!(
                    "compile: generated binding {} changed leaf identity",
                    leaf.node_id
                ));
            }
            let value: Value = slot
                .get()
                .map_err(|error| format!("compile: generated binding {}: {error}", node.id))?;
            validate_generated_payload(node, &value)?;
            Ok(CpuGeneratedValue {
                node: leaf.node,
                node_id: leaf.node_id,
                slot_identity: leaf.slot_identity,
                value,
            })
        })
        .collect()
}

fn validate_generated_values(
    index: &GraphIndex,
    generated_values: &[CpuGeneratedValue],
) -> Result<(), String> {
    if generated_values.len() != index.leaves.len() {
        return Err(format!(
            "compile: expected {} prepared generated bindings, got {}",
            index.leaves.len(),
            generated_values.len()
        ));
    }
    for (leaf, generated) in index.leaves.iter().zip(generated_values) {
        let node = index.node(leaf.node).ok_or_else(|| {
            format!(
                "compile: generated binding node {} is out of range",
                leaf.node
            )
        })?;
        let slot_identity = match &node.kind {
            NodeKind::Leaf(slot) => Arc::as_ptr(slot) as usize,
            _ => {
                return Err(format!(
                    "compile: generated binding {} is not a leaf",
                    leaf.node_id
                ));
            }
        };
        if generated.node != leaf.node
            || generated.node_id != leaf.node_id
            || generated.slot_identity != leaf.slot_identity
            || node.id != leaf.node_id
            || slot_identity != leaf.slot_identity
            || node.shape != leaf.shape
            || node.dtype != leaf.dtype
            || !node.device.is_cpu()
            || !leaf.device.is_cpu()
        {
            return Err(format!(
                "compile: prepared generated binding {} has the wrong node identity or metadata",
                leaf.node_id
            ));
        }
        validate_generated_payload(node, &generated.value)?;
    }
    Ok(())
}

pub fn compile_prepared(
    program: &PreparedProgram,
    generated_values: &[CpuGeneratedValue],
    state: Option<CpuStatePlan>,
) -> Result<CpuCompilation, String> {
    compile_prepared_internal(
        program,
        generated_values,
        state.map(|state| state.bytes),
        state,
    )
}

fn compile_roots_internal(
    roots: &[Arc<Node>],
    options: CompileOptions,
    ce_chunk_size: usize,
    state_bytes: Option<usize>,
    state: Option<CpuStatePlan>,
) -> Result<CpuCompilation, String> {
    if roots.is_empty() {
        return Err("compile: expected at least one root".to_string());
    }
    if ce_chunk_size == 0 {
        return Err("compile: CE chunk size must be positive".to_string());
    }
    let mut options = options;
    options.environment.ce_chunk_size = ce_chunk_size;
    let mut request = ProgramRequest::from_roots(roots.to_vec(), options);
    if let Some(state) = state {
        request =
            request.with_state_cursor(StateCursorSlot::new(state.cursor_slot, state.cursor_tensor));
    }
    let program = request.prepare()?;
    let generated_values = load_generated_values(&program.index)?;
    compile_prepared_internal(&program, &generated_values, state_bytes, state)
}

fn compile_prepared_internal(
    program: &PreparedProgram,
    generated_values: &[CpuGeneratedValue],
    state_bytes: Option<usize>,
    state: Option<CpuStatePlan>,
) -> Result<CpuCompilation, String> {
    let index = &program.index;
    let options = program.options.clone();
    let ce_chunk_size = options.environment.ce_chunk_size;
    if index.roots.is_empty() {
        return Err("compile: expected at least one root".to_string());
    }
    if ce_chunk_size == 0 {
        return Err("compile: CE chunk size must be positive".to_string());
    }
    let state_cursor =
        state.map(|state| StateCursorSlot::new(state.cursor_slot, state.cursor_tensor));
    if program.state_cursor != state_cursor {
        return Err("compile: CPU state cursor does not match the prepared program".to_string());
    }
    validate_generated_values(index, generated_values)?;
    let mut driver = CompilerDriver::new(program)?;
    let slots = index.slots.to_vec();
    if slots.iter().any(|slot| !slot.device.is_cpu()) {
        return Err("compile: graph contains an unsupported device".to_string());
    }
    let random_provenance = index
        .random_source_order
        .iter()
        .map(|source| {
            let node = index
                .node(source.node)
                .ok_or_else(|| format!("compile: random source {} is out of range", source.node))?;
            Ok((node.id, source.provenance))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;
    let mut lowerer = Lowerer::new(
        options,
        ce_chunk_size,
        random_provenance,
        &slots,
        state,
        generated_values,
    );
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
    let (mut executable, generated_bindings, generated_slots) = artifact?;
    let publication_started = Instant::now();
    if let Some(state_bytes) = state_bytes {
        executable.memory.report.state_bytes = state_bytes;
        executable.diagnostics.memory.state_bytes = state_bytes;
    }
    let mut compilation = CpuCompilation {
        executable: Arc::new(executable),
        slots,
        generated_bindings,
        generated_slots,
    };
    let compiler_work = driver.finish_with_phase(
        &compilation.executable.program,
        PUBLICATION_PHASE,
        publication_started,
    );
    let executable = Arc::get_mut(&mut compilation.executable)
        .expect("newly published CPU executable must be uniquely owned");
    executable.diagnostics.compile_phases = compiler_work.compile_phases.clone();
    executable.compiler_work = compiler_work;
    Ok(compilation)
}

pub trait CpuState: Send + Sync {
    fn begin(&self, _executable: &CpuExecutable, _values: &[Value]) -> Result<(), String> {
        Ok(())
    }

    fn run_command(
        &self,
        command: &CpuCommand,
        inputs: &[Value],
        staging: &[Value],
        outputs: &mut [CpuDestination<'_>],
        scratch: &mut [CpuDestination<'_>],
        state_outputs: &mut [CpuDestination<'_>],
    ) -> Result<(), String>;

    fn commit(&self, _executable: &CpuExecutable, _values: &[Value]) -> Result<(), String> {
        Ok(())
    }

    fn rollback(&self) {}
}

#[derive(Debug)]
pub struct CpuExecution {
    pub outputs: Vec<Value>,
    pub memory: InvocationMemoryReport,
}

fn resolved_destination(value: &Value) -> CpuDestination<'_> {
    // SAFETY: the memory plan and linear command schedule exclusively own every
    // command output, scratch, staging, and transaction range while it is written.
    unsafe { CpuDestination::from_planned(value.tensor()) }
}

fn random_seed(nonce: u64, provenance: u64) -> u64 {
    let mut value = nonce ^ provenance.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn resolve_physical_instruction<'a>(
    executable: &'a CpuExecutable,
    physical: &CpuPhysicalCommand,
) -> Result<&'a CpuCommand, String> {
    let CpuPhysicalCommand::Encode(id) = *physical;
    let instruction = executable
        .program
        .instructions
        .get(id.index())
        .ok_or_else(|| format!("execute: physical instruction {id} is out of range"))?;
    if instruction.id != id || instruction.kind.operation().is_none() {
        return Err(format!(
            "execute: physical instruction {id} does not resolve to an operation"
        ));
    }
    Ok(instruction)
}

struct InvocationSegments {
    owners: Box<[Arc<CpuSegment>]>,
    retentions: Box<[Option<CpuStorageRetention>]>,
    _workspace: CpuWorkspaceLease,
    actual_workspace_bytes: usize,
}

fn acquire_segments(executable: &CpuExecutable) -> Result<InvocationSegments, String> {
    let mut workspace_indices = Vec::new();
    let mut workspace_requests = Vec::new();
    for (index, segment) in executable.memory.segments.iter().enumerate() {
        if segment.memory_space != NativeMemorySpace::Cpu {
            return Err(format!(
                "execute: CPU segment {index} has unsupported memory space {:?}",
                segment.memory_space
            ));
        }
        if !matches!(segment.ownership, SegmentOwnership::ProvisionalOutput) {
            workspace_indices.push(index);
            workspace_requests.push(workspace_request(segment.bytes, segment.alignment)?);
        }
    }
    let workspace = workspace_pool()
        .acquire_set(&workspace_requests)
        .map_err(|error| format!("execute: CPU workspace acquisition failed: {error}"))?;
    let mut owners: Vec<Option<Arc<CpuSegment>>> = std::iter::repeat_with(|| None)
        .take(executable.memory.segments.len())
        .collect();
    let mut retentions: Vec<Option<CpuStorageRetention>> = std::iter::repeat_with(|| None)
        .take(executable.memory.segments.len())
        .collect();
    let mut actual_workspace_bytes = 0usize;
    for (&index, leased) in workspace_indices.iter().zip(workspace.segments()) {
        owners[index] = Some(Arc::clone(leased.workspace()));
        if matches!(
            executable.memory.segments[index].ownership,
            SegmentOwnership::Workspace | SegmentOwnership::InvocationStaging
        ) {
            actual_workspace_bytes = actual_workspace_bytes
                .checked_add(leased.capacity())
                .ok_or_else(|| "execute: CPU workspace byte size overflow".to_string())?;
        }
    }
    for (index, segment) in executable.memory.segments.iter().enumerate() {
        if !matches!(segment.ownership, SegmentOwnership::ProvisionalOutput) {
            continue;
        }
        let request = workspace_request(segment.bytes, segment.alignment)?;
        let lease = Arc::new(
            workspace_pool()
                .acquire(std::slice::from_ref(&request))
                .map_err(|error| format!("execute: CPU output acquisition failed: {error}"))?,
        );
        let owner = Arc::clone(lease.segments()[0].workspace());
        let retention: CpuStorageRetention = lease;
        owners[index] = Some(owner);
        retentions[index] = Some(retention);
    }
    let owners = owners
        .into_iter()
        .enumerate()
        .map(|(index, owner)| {
            owner.ok_or_else(|| format!("execute: CPU segment {index} was not acquired"))
        })
        .collect::<Result<Box<[_]>, _>>()?;
    Ok(InvocationSegments {
        owners,
        retentions: retentions.into_boxed_slice(),
        _workspace: workspace,
        actual_workspace_bytes,
    })
}

fn resolve_values(
    executable: &CpuExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    scalar_bindings: &[f64],
    segments: &InvocationSegments,
) -> Result<Box<[Value]>, String> {
    let mut values: Vec<Option<Value>> = std::iter::repeat_with(|| None)
        .take(executable.program.values.len())
        .collect();
    for constant in &executable.constants {
        values[constant.value.index()] = Some(constant.payload.clone());
    }
    for binding in &executable.bindings {
        let source = match binding.source {
            CpuBindingSource::Declared(slot) => declared_bindings
                .get(slot as usize)
                .ok_or_else(|| format!("execute: input slot {slot} is unbound"))?,
            CpuBindingSource::Generated(slot) => generated_bindings
                .get(slot as usize)
                .ok_or_else(|| format!("execute: generated input slot {slot} is unbound"))?,
        };
        let declaration = &executable.program.values[binding.value.index()];
        if source.shape() != declaration.shape.as_ref()
            || source.dtype() != declaration.dtype
            || source.tensor().layout != declaration.layout
        {
            return Err(format!(
                "execute: binding for value {} does not match shape, dtype, and exact layout",
                binding.value
            ));
        }
        values[binding.value.index()] = Some(source.clone());
    }

    for (index, location) in executable.memory.locations.iter().enumerate() {
        if values[index].is_some() {
            continue;
        }
        let declaration = &executable.program.values[index];
        let value = match location {
            Location::Segment {
                segment,
                offset,
                bytes,
            } => {
                let expected = declaration
                    .shape
                    .iter()
                    .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
                    .and_then(|elements| elements.checked_mul(declaration.dtype.size_in_bytes()))
                    .ok_or_else(|| "execute: value byte size overflow".to_string())?;
                if *bytes != expected {
                    return Err(format!(
                        "execute: location {index} has {bytes} bytes, expected {expected}"
                    ));
                }
                Value(Tensor::from_segment_with_retention(
                    Arc::clone(&segments.owners[segment.index()]),
                    *offset,
                    declaration.shape.to_vec(),
                    declaration.dtype,
                    segments.retentions[segment.index()].clone(),
                )?)
            }
            Location::Alias { root, .. } => {
                let root = values[root.index()]
                    .as_ref()
                    .ok_or_else(|| format!("execute: alias {index} precedes root {root}"))?;
                Value(Tensor {
                    buffer: root.tensor().buffer.clone(),
                    layout: declaration.layout.clone(),
                })
            }
            Location::External { .. } | Location::Persistent { .. } => {
                return Err(format!("execute: unresolved fixed location {index}"));
            }
            Location::InlineScalar { .. } => {
                return Err("execute: inline scalar locations are unsupported on CPU".to_string());
            }
            Location::Output { .. } | Location::State { .. } => {
                return Err(
                    "execute: output/state locations are not valid in a native CPU plan"
                        .to_string(),
                );
            }
        };
        values[index] = Some(value);
    }
    let values: Box<[Value]> = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| value.ok_or_else(|| format!("execute: value {index} was unresolved")))
        .collect::<Result<_, _>>()?;
    for binding in &executable.padded_bindings {
        let source = declared_bindings
            .get(binding.source as usize)
            .ok_or_else(|| format!("execute: padded input {} is unbound", binding.source))?;
        let destination = &values[binding.value.index()];
        let source_shape = source.shape();
        let destination_shape = destination.shape();
        if source.dtype() != destination.dtype()
            || source_shape.len() != destination_shape.len()
            || source_shape.first().is_none_or(|active| {
                *active == 0
                    || *active > destination_shape[0]
                    || source_shape[1..] != destination_shape[1..]
            })
            || !source.tensor().layout.is_contiguous()
        {
            return Err(format!(
                "execute: padded input {} does not match its bounded signature",
                binding.source
            ));
        }
        Tensor::zeros_into(
            &destination_shape,
            destination.dtype(),
            &mut resolved_destination(destination),
        )?;
        let active = source_shape[0];
        let target = destination
            .tensor()
            .view(destination.tensor().layout.narrow(0, 0, active));
        let mut target = unsafe { CpuDestination::from_planned(&target) };
        source.tensor().copy_into(&mut target)?;
    }
    for binding in &executable.scalar_bindings {
        let scalar = scalar_bindings
            .get(binding.source as usize)
            .copied()
            .ok_or_else(|| format!("execute: scalar slot {} is unbound", binding.source))?;
        let value = &values[binding.value.index()];
        Tensor::full_into(&[], scalar, value.dtype(), &mut resolved_destination(value))?;
    }
    Ok(values)
}

fn cast_staging(source: &Value, staging: &Value) -> Result<(), String> {
    source
        .tensor()
        .cast_into(staging.dtype(), &mut resolved_destination(staging))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_command<'a>(
    nonce: u64,
    command: &CpuCommand,
    values: &'a [Value],
    inputs: &[Value],
    staging_values: &[Value],
    destinations: &mut Vec<CpuDestination<'a>>,
    scratch: &mut Vec<CpuDestination<'a>>,
    staging: &mut Vec<CpuDestination<'a>>,
    state_outputs: &mut Vec<CpuDestination<'a>>,
    lanes: &mut Vec<Value>,
    state: Option<&dyn CpuState>,
    cancelled: &CancellationFlag,
) -> Result<(), String> {
    let (op, plan) = command
        .kind
        .operation()
        .ok_or_else(|| "execute: physical command references a boundary instruction".to_string())?;
    destinations.clear();
    scratch.clear();
    staging.clear();
    state_outputs.clear();
    for value in &command.outputs {
        destinations.push(resolved_destination(&values[value.value.index()]));
    }
    for value in &command.scratch {
        scratch.push(resolved_destination(&values[value.value.index()]));
    }
    for value in &command.staging {
        staging.push(resolved_destination(&values[value.value.index()]));
    }
    for value in &command.state {
        state_outputs.push(resolved_destination(&values[value.value.index()]));
    }

    let output = |index: usize| -> Result<&Value, String> {
        command
            .outputs
            .get(index)
            .map(|output| &values[output.value.index()])
            .ok_or_else(|| format!("{}: missing output {index}", op.name()))
    };
    let scratch_value = |index: usize| -> &Value { &values[command.scratch[index].value.index()] };
    match op {
        CpuOp::Randn { provenance, .. } => {
            Tensor::randn_seeded_into(random_seed(nonce, *provenance), &mut destinations[0])
        }
        CpuOp::Uniform {
            lo, hi, provenance, ..
        } => Tensor::uniform_seeded_into(
            *lo,
            *hi,
            random_seed(nonce, *provenance),
            &mut destinations[0],
        ),
        CpuOp::Unary(kind) => {
            let source = inputs[0].tensor();
            match (kind, plan) {
                (CpuUnaryOp::Neg, _) => source.neg_into(&mut destinations[0]),
                (CpuUnaryOp::Abs, _) => source.abs_into(&mut destinations[0]),
                (CpuUnaryOp::Sqrt, _) => source.sqrt_into(&mut destinations[0]),
                (CpuUnaryOp::Exp, _) => source.exp_into(&mut destinations[0]),
                (CpuUnaryOp::Log, _) => source.log_into(&mut destinations[0]),
                (CpuUnaryOp::Sin, _) => source.sin_into(&mut destinations[0]),
                (CpuUnaryOp::Cos, _) => source.cos_into(&mut destinations[0]),
                (CpuUnaryOp::Tanh, _) => source.tanh_into(&mut destinations[0]),
                (CpuUnaryOp::Relu, _) => source.relu_into(&mut destinations[0]),
                (CpuUnaryOp::Erf, _) => source.erf_into(&mut destinations[0]),
                (CpuUnaryOp::Floor, _) => source.floor_into(&mut destinations[0]),
                (CpuUnaryOp::Ceil, _) => source.ceil_into(&mut destinations[0]),
                (CpuUnaryOp::Round, _) => source.round_into(&mut destinations[0]),
                (CpuUnaryOp::Sign, _) => source.sign_into(&mut destinations[0]),
                (CpuUnaryOp::Pow { exponent_bits }, _) => {
                    source.powf_into(f64::from_bits(*exponent_bits), &mut destinations[0])
                }
                (CpuUnaryOp::Cast { dtype }, _) => source.cast_into(*dtype, &mut destinations[0]),
                (CpuUnaryOp::Gelu { .. }, CpuAlgorithmPlan::UnaryFusion { program, strides }) => {
                    fusion::run_elementwise_into(
                        program,
                        inputs,
                        Some(strides),
                        &[],
                        output(0)?.numel(),
                        output(0)?.tensor().shape(),
                        output(0)?.dtype(),
                        &cpu_device(),
                        &mut destinations[0],
                        &mut scratch[0],
                    )
                }
                (CpuUnaryOp::Inverse, CpuAlgorithmPlan::Inverse(requirements)) => {
                    source.inverse_into(&mut destinations[0], scratch, requirements)
                }
                (CpuUnaryOp::Det, CpuAlgorithmPlan::Determinant(requirements)) => {
                    if destinations[0].shape() == requirements.output.shape {
                        source.det_into(&mut destinations[0], scratch, requirements)
                    } else {
                        let (temporary, lu) = scratch.split_at_mut(1);
                        source.det_into(&mut temporary[0], lu, requirements)?;
                        scratch_value(0)
                            .tensor()
                            .squeeze_dims_into(&[0], &mut destinations[0])
                    }
                }
                _ => Err(format!("{}: invalid immutable algorithm plan", op.name())),
            }
        }
        CpuOp::Binary(kind) => {
            lanes.clear();
            let target_dtype = inputs
                .iter()
                .find(|input| !input.tensor().shape().is_empty())
                .map_or(inputs[0].dtype(), Value::dtype);
            let mut staged = 0usize;
            for input in inputs {
                if input.dtype() != target_dtype && input.tensor().shape().is_empty() {
                    let target = staging_values.get(staged).ok_or_else(|| {
                        format!("{}: missing scalar cast staging {staged}", op.name())
                    })?;
                    cast_staging(input, target)?;
                    lanes.push(target.clone());
                    staged += 1;
                } else {
                    lanes.push(input.clone());
                }
            }
            let a = lanes[0].tensor();
            let b = lanes[1].tensor();
            match (kind, plan) {
                (CpuBinaryOp::Add, _) => a.add_into(b, &mut destinations[0]),
                (CpuBinaryOp::Sub, _) => a.sub_into(b, &mut destinations[0]),
                (CpuBinaryOp::Mul, _) => a.mul_into(b, &mut destinations[0]),
                (CpuBinaryOp::Div, _) => a.div_into(b, &mut destinations[0]),
                (CpuBinaryOp::Eq, _) => a.eq_into(b, &mut destinations[0]),
                (CpuBinaryOp::Gt, _) => a.gt_into(b, &mut destinations[0]),
                (CpuBinaryOp::Lt, _) => a.lt_into(b, &mut destinations[0]),
                (CpuBinaryOp::Ge, _) => a.ge_into(b, &mut destinations[0]),
                (CpuBinaryOp::Le, _) => a.le_into(b, &mut destinations[0]),
                (CpuBinaryOp::Maximum, _) => a.maximum_into(b, &mut destinations[0]),
                (CpuBinaryOp::Minimum, _) => a.minimum_into(b, &mut destinations[0]),
                (CpuBinaryOp::Concat { dim }, _) => {
                    Tensor::cat_into(&[a, b], *dim, &mut destinations[0])
                }
                (CpuBinaryOp::Matmul, CpuAlgorithmPlan::Matmul(requirements)) => {
                    a.matmul_into(b, &mut destinations[0], scratch, requirements)
                }
                (CpuBinaryOp::Solve, CpuAlgorithmPlan::Solve(requirements)) => {
                    a.solve_into(b, &mut destinations[0], scratch, requirements)
                }
                _ => Err(format!("{}: invalid immutable algorithm plan", op.name())),
            }
        }
        CpuOp::Where => inputs[1].tensor().where_into(
            inputs[0].tensor(),
            inputs[2].tensor(),
            &mut destinations[0],
        ),
        CpuOp::Argmax { dim } => {
            inputs[0].tensor().argmax_into(*dim, &mut scratch[0])?;
            scratch_value(0)
                .tensor()
                .cast_into(DType::I64, &mut scratch[1])?;
            scratch_value(1)
                .tensor()
                .squeeze_dims_into(&[*dim], &mut destinations[0])
        }
        CpuOp::Argmin { dim } => {
            inputs[0].tensor().argmin_into(*dim, &mut scratch[0])?;
            scratch_value(0)
                .tensor()
                .cast_into(DType::I64, &mut scratch[1])?;
            scratch_value(1)
                .tensor()
                .squeeze_dims_into(&[*dim], &mut destinations[0])
        }
        CpuOp::Cumsum { dim } => inputs[0].tensor().cumsum_into(*dim, &mut destinations[0]),
        CpuOp::ScatterAdd { dim } => inputs[0].tensor().scatter_add_into(
            *dim,
            inputs[1].tensor(),
            inputs[2].tensor(),
            &mut destinations[0],
        ),
        CpuOp::Gather { dim } => {
            inputs[0]
                .tensor()
                .gather_into(*dim, inputs[1].tensor(), &mut destinations[0])
        }
        CpuOp::IndexSelect { dim } => {
            inputs[0]
                .tensor()
                .index_select_into(*dim, inputs[1].tensor(), &mut destinations[0])
        }
        CpuOp::CrossEntropy {
            ignore_index,
            reduction,
        } => {
            let (s0, rest) = scratch.split_at_mut(1);
            let (s1, s2) = rest.split_at_mut(1);
            composed::cross_entropy_forward_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                *ignore_index,
                *reduction,
                &mut destinations[0],
                &mut s0[0],
                &mut s1[0],
                &mut s2[0],
            )
        }
        CpuOp::CrossEntropyBackward {
            ignore_index,
            reduction,
        } => composed::cross_entropy_backward_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            *ignore_index,
            *reduction,
            &mut destinations[0],
            scratch.get_mut(0),
        ),
        CpuOp::ChunkedHeadCe {
            ignore_index,
            chunk_size,
        } => {
            let (s0, rest) = scratch.split_at_mut(1);
            let (s1, rest) = rest.split_at_mut(1);
            let (s2, s3) = rest.split_at_mut(1);
            composed::chunked_head_ce_forward_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                inputs[2].tensor(),
                inputs[3].tensor(),
                *ignore_index,
                *chunk_size,
                &mut destinations[0],
                &mut s0[0],
                &mut s1[0],
                &mut s2[0],
                &mut s3[0],
            )
        }
        CpuOp::ChunkedHeadCeBackward {
            ignore_index,
            chunk_size,
        } => {
            let (outputs0, outputs12) = destinations.split_at_mut(1);
            let (outputs1, outputs2) = outputs12.split_at_mut(1);
            let (scratch0, scratch_rest) = scratch.split_at_mut(1);
            let (scratch1, scratch_rest) = scratch_rest.split_at_mut(1);
            let (scratch2, scratch_rest) = scratch_rest.split_at_mut(1);
            let (scratch3, scratch4) = scratch_rest.split_at_mut(1);
            composed::chunked_head_ce_backward_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                inputs[2].tensor(),
                inputs[3].tensor(),
                inputs[4].tensor(),
                *ignore_index,
                *chunk_size,
                &mut outputs0[0],
                &mut outputs1[0],
                &mut outputs2[0],
                &mut scratch0[0],
                &mut scratch1[0],
                &mut scratch2[0],
                &mut scratch3[0],
                &mut scratch4[0],
            )
        }
        CpuOp::Sdpa {
            scale,
            causal,
            window,
        } => composed::sdpa_forward_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            inputs[2].tensor(),
            *scale,
            *causal,
            *window,
            &mut destinations[0],
            &mut scratch[0],
        ),
        CpuOp::SdpaBackward {
            scale,
            causal,
            window,
        } => {
            let (scratch0, scratch1) = scratch.split_at_mut(1);
            composed::sdpa_logsumexp_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                inputs[2].tensor(),
                *scale,
                *causal,
                *window,
                &mut scratch0[0],
            )?;
            let (outputs0, outputs12) = destinations.split_at_mut(1);
            let (outputs1, outputs2) = outputs12.split_at_mut(1);
            composed::sdpa_backward_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                inputs[2].tensor(),
                inputs[4].tensor(),
                scratch_value(0).tensor(),
                inputs[3].tensor(),
                *scale,
                *causal,
                *window,
                &mut outputs0[0],
                &mut outputs1[0],
                &mut outputs2[0],
                &mut scratch1[0],
            )
        }
        CpuOp::KdaChunk { scale } => composed::kda_chunk_forward_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            inputs[2].tensor(),
            inputs[3].tensor(),
            inputs[4].tensor(),
            *scale,
            None,
            &mut destinations[0],
            None,
            Some(&mut scratch[0]),
        ),
        CpuOp::KdaBackward { scale } => {
            let (o0, rest) = destinations.split_at_mut(1);
            let (o1, rest) = rest.split_at_mut(1);
            let (o2, rest) = rest.split_at_mut(1);
            let (o3, o4) = rest.split_at_mut(1);
            composed::kda_backward_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                inputs[2].tensor(),
                inputs[3].tensor(),
                inputs[4].tensor(),
                inputs[5].tensor(),
                *scale,
                &mut o0[0],
                &mut o1[0],
                &mut o2[0],
                &mut o3[0],
                &mut o4[0],
                &mut scratch[0],
            )
        }
        CpuOp::ShortConv1d => composed::short_conv1d_forward_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            &mut destinations[0],
        ),
        CpuOp::ShortConv1dBackwardX => composed::short_conv1d_backward_x_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            inputs[2].tensor(),
            &mut destinations[0],
        ),
        CpuOp::ShortConv1dBackwardW => composed::short_conv1d_backward_w_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            inputs[2].tensor(),
            &mut destinations[0],
        ),
        CpuOp::KdaRecurrence { .. }
        | CpuOp::ConvState { .. }
        | CpuOp::KvAttention { .. }
        | CpuOp::LastTokenRow => state
            .ok_or_else(|| format!("{}: operation requires a state context", op.name()))?
            .run_command(
                command,
                inputs,
                staging_values,
                destinations,
                scratch,
                state_outputs,
            ),
        CpuOp::RotaryEmbedding {
            theta,
            cursor_offset,
            layout,
        } => {
            if *cursor_offset {
                state
                    .ok_or_else(|| "rotary embedding: cursor offset requires state".to_string())?
                    .run_command(
                        command,
                        inputs,
                        staging_values,
                        destinations,
                        scratch,
                        state_outputs,
                    )
            } else {
                composed::rotary_forward_into(
                    inputs[0].tensor(),
                    &[0],
                    *theta,
                    1.0,
                    *layout,
                    &mut destinations[0],
                )
            }
        }
        CpuOp::RotaryEmbeddingBackward { theta, layout } => composed::rotary_forward_into(
            inputs[0].tensor(),
            &[0],
            *theta,
            -1.0,
            *layout,
            &mut destinations[0],
        ),
        CpuOp::LayerNorm { eps } => composed::layer_norm_forward_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            inputs[2].tensor(),
            *eps,
            &mut destinations[0],
        ),
        CpuOp::RmsNorm { eps } => composed::rms_norm_forward_into(
            inputs[0].tensor(),
            inputs.get(1).map(|input| input.tensor()),
            *eps,
            &mut destinations[0],
        ),
        CpuOp::LayerNormBackward { eps } => {
            let (o0, rest) = destinations.split_at_mut(1);
            let (o1, o2) = rest.split_at_mut(1);
            composed::layer_norm_backward_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                inputs[2].tensor(),
                *eps,
                &mut o0[0],
                &mut o1[0],
                &mut o2[0],
                &mut scratch[0],
            )
        }
        CpuOp::Linear | CpuOp::LinearResidual | CpuOp::LinearGelu { .. } => {
            let CpuAlgorithmPlan::Linear { matmul, gelu } = plan else {
                return Err("linear: invalid immutable algorithm plan".to_string());
            };
            inputs[0]
                .tensor()
                .matmul_into(inputs[1].tensor(), &mut scratch[0], &mut [], matmul)?;
            match op {
                CpuOp::Linear => scratch_value(0)
                    .tensor()
                    .add_into(inputs[2].tensor(), &mut destinations[0]),
                CpuOp::LinearResidual => {
                    scratch_value(0)
                        .tensor()
                        .add_into(inputs[2].tensor(), &mut scratch[1])?;
                    scratch_value(1)
                        .tensor()
                        .add_into(inputs[3].tensor(), &mut destinations[0])
                }
                CpuOp::LinearGelu { dual, .. } => {
                    let activation_index = usize::from(!*dual);
                    if *dual {
                        scratch_value(0)
                            .tensor()
                            .add_into(inputs[2].tensor(), &mut destinations[0])?;
                    } else {
                        scratch_value(0)
                            .tensor()
                            .add_into(inputs[2].tensor(), &mut scratch[1])?;
                    }
                    let activation = if *dual {
                        output(0)?
                    } else {
                        scratch_value(activation_index)
                    };
                    let gelu_output = usize::from(*dual);
                    let fusion_scratch = scratch.len() - 1;
                    fusion::run_elementwise_into(
                        gelu.as_ref()
                            .ok_or_else(|| "linear_gelu: missing prepared program".to_string())?,
                        std::slice::from_ref(activation),
                        None,
                        &[],
                        activation.numel(),
                        activation.tensor().shape(),
                        activation.dtype(),
                        &cpu_device(),
                        &mut destinations[gelu_output],
                        &mut scratch[fusion_scratch],
                    )
                }
                _ => unreachable!(),
            }
        }
        CpuOp::QuantizedLinear { codec, .. } => {
            let CpuAlgorithmPlan::Quantized(requirements) = plan else {
                return Err("quantized_linear: invalid immutable algorithm plan".to_string());
            };
            quantized::linear_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                inputs.get(2).map(Value::tensor),
                *codec,
                &mut destinations[0],
                requirements,
                cancelled,
            )
        }
        CpuOp::QuantizedEmbedding { codec, .. } => {
            let CpuAlgorithmPlan::Quantized(requirements) = plan else {
                return Err("quantized_embedding: invalid immutable algorithm plan".to_string());
            };
            quantized::embedding_into(
                inputs[0].tensor(),
                inputs[1].tensor(),
                *codec,
                &mut destinations[0],
                requirements,
                cancelled,
            )
        }
        CpuOp::Conv1d {
            stride,
            padding,
            dilation,
            groups,
        } => conv::conv1d_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            *stride,
            *padding,
            *dilation,
            *groups,
            &mut destinations[0],
        ),
        CpuOp::Conv2d {
            stride,
            padding,
            dilation,
            groups,
        } => conv::conv2d_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            *stride,
            *padding,
            *dilation,
            *groups,
            &mut destinations[0],
        ),
        CpuOp::ConvTranspose1d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => conv::conv_transpose1d_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            *stride,
            *padding,
            *output_padding,
            *dilation,
            *groups,
            &mut destinations[0],
        ),
        CpuOp::ConvTranspose2d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => conv::conv_transpose2d_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            *stride,
            *padding,
            *output_padding,
            *dilation,
            *groups,
            &mut destinations[0],
        ),
        CpuOp::Conv1dBackwardW {
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => conv::conv1d_backward_w_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            *kernel,
            *out_channels,
            *stride,
            *padding,
            *dilation,
            *groups,
            &mut destinations[0],
        ),
        CpuOp::Conv2dBackwardW {
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => conv::conv2d_backward_w_into(
            inputs[0].tensor(),
            inputs[1].tensor(),
            *kernel,
            *out_channels,
            *stride,
            *padding,
            *dilation,
            *groups,
            &mut destinations[0],
        ),
        CpuOp::Reduce { op, dims, keepdims } => {
            if *keepdims {
                match op {
                    CpuReduceOp::Sum => inputs[0].tensor().sum_into(dims, &mut destinations[0])?,
                    CpuReduceOp::Mean => {
                        inputs[0].tensor().mean_into(dims, &mut destinations[0])?
                    }
                    CpuReduceOp::Max => inputs[0].tensor().max_into(dims, &mut destinations[0])?,
                    CpuReduceOp::Min => inputs[0].tensor().min_into(dims, &mut destinations[0])?,
                    CpuReduceOp::Prod => {
                        inputs[0].tensor().prod_into(dims, &mut destinations[0])?
                    }
                }
            } else {
                match op {
                    CpuReduceOp::Sum => inputs[0].tensor().sum_into(dims, &mut scratch[0])?,
                    CpuReduceOp::Mean => inputs[0].tensor().mean_into(dims, &mut scratch[0])?,
                    CpuReduceOp::Max => inputs[0].tensor().max_into(dims, &mut scratch[0])?,
                    CpuReduceOp::Min => inputs[0].tensor().min_into(dims, &mut scratch[0])?,
                    CpuReduceOp::Prod => inputs[0].tensor().prod_into(dims, &mut scratch[0])?,
                }
                scratch_value(0)
                    .tensor()
                    .squeeze_dims_into(dims, &mut destinations[0])?;
            }
            Ok(())
        }
        CpuOp::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
            implementation,
            ..
        } => {
            for (source, target) in [4usize, 5, 6].into_iter().zip(staging_values) {
                cast_staging(&inputs[source], target)?;
            }
            let (o0, rest) = destinations.split_at_mut(1);
            let (o1, o2) = rest.split_at_mut(1);
            match implementation {
                OptimizerImplementation::Composed => composed::adamw_step_into(
                    inputs[0].tensor(),
                    inputs[1].tensor(),
                    inputs[2].tensor(),
                    inputs[3].tensor(),
                    staging_values[0].tensor(),
                    staging_values[1].tensor(),
                    staging_values[2].tensor(),
                    *beta1,
                    *beta2,
                    *eps,
                    *weight_decay,
                    &mut o0[0],
                    &mut o1[0],
                    &mut o2[0],
                ),
                OptimizerImplementation::Fused => {
                    let CpuAlgorithmPlan::AdamW {
                        fusion: Some(program),
                        strides,
                        ..
                    } = plan
                    else {
                        return Err("adamw_fused: missing prepared program".to_string());
                    };
                    fusion::run_elementwise_multi_into(
                        program,
                        &inputs[..4],
                        Some(strides),
                        staging_values,
                        inputs[0].numel(),
                        inputs[0].tensor().shape(),
                        inputs[0].dtype(),
                        &cpu_device(),
                        destinations,
                        &mut scratch[0],
                    )
                }
            }
        }
        CpuOp::AdamWGroup { parameters, .. } => {
            let scalar_start = parameters * 4;
            for (source, target) in (scalar_start..scalar_start + 3).zip(staging_values) {
                cast_staging(&inputs[source], target)?;
            }
            lanes.clear();
            for index in 0..*parameters {
                lanes.push(inputs[index].clone());
                lanes.push(inputs[parameters + index].clone());
                lanes.push(inputs[parameters * 2 + index].clone());
                lanes.push(inputs[parameters * 3 + index].clone());
            }
            let CpuAlgorithmPlan::MultiFusion { program, strides } = plan else {
                return Err("adamw_group: missing prepared program".to_string());
            };
            fusion::run_elementwise_multi_into(
                program,
                lanes,
                Some(strides),
                staging_values,
                inputs[0].numel(),
                inputs[0].tensor().shape(),
                inputs[0].dtype(),
                &cpu_device(),
                destinations,
                &mut scratch[0],
            )
        }
        CpuOp::Sgd {
            momentum,
            dampening,
            nesterov,
            weight_decay,
            implementation,
            ..
        } => {
            cast_staging(&inputs[4], &staging_values[0])?;
            cast_staging(&inputs[3], &staging_values[1])?;
            let (o0, o1) = destinations.split_at_mut(1);
            match implementation {
                OptimizerImplementation::Composed => composed::sgd_step_into(
                    inputs[0].tensor(),
                    inputs[1].tensor(),
                    inputs[2].tensor(),
                    staging_values[0].tensor(),
                    staging_values[1].tensor(),
                    *momentum,
                    *dampening,
                    *nesterov,
                    *weight_decay,
                    &mut o0[0],
                    &mut o1[0],
                ),
                OptimizerImplementation::Fused => {
                    let CpuAlgorithmPlan::Sgd {
                        fusion: Some(program),
                        strides,
                        ..
                    } = plan
                    else {
                        return Err("sgd_fused: missing prepared program".to_string());
                    };
                    fusion::run_elementwise_multi_into(
                        program,
                        &inputs[..3],
                        Some(strides),
                        staging_values,
                        inputs[0].numel(),
                        inputs[0].tensor().shape(),
                        inputs[0].dtype(),
                        &cpu_device(),
                        destinations,
                        &mut scratch[0],
                    )
                }
            }
        }
        CpuOp::FusedElementwise { strides, shape, .. } => {
            let CpuAlgorithmPlan::Fusion(program) = plan else {
                return Err("fused_elementwise: missing prepared program".to_string());
            };
            fusion::run_elementwise_multi_into(
                program,
                inputs,
                Some(strides),
                &[],
                shape.iter().product(),
                shape,
                inputs[0].dtype(),
                &cpu_device(),
                destinations,
                &mut scratch[0],
            )
        }
        CpuOp::FusedReduce {
            strides,
            in_shape,
            op,
            dims,
            keepdims,
            shape,
            ..
        } => {
            let CpuAlgorithmPlan::Fusion(program) = plan else {
                return Err("fused_reduce: missing prepared program".to_string());
            };
            fusion::run_reduce_into(
                *op,
                program,
                inputs,
                strides,
                in_shape,
                dims,
                *keepdims,
                shape,
                inputs[0].dtype(),
                &cpu_device(),
                &mut destinations[0],
                &mut scratch[0],
            )
        }
        CpuOp::Reshape { .. } => inputs[0]
            .tensor()
            .materialize_reshaped_into(&mut destinations[0]),
        CpuOp::Permute { .. }
        | CpuOp::Slice { .. }
        | CpuOp::BroadcastTo { .. }
        | CpuOp::PositionEmbedding { .. } => {
            Err("execute: view operation was not lowered as an alias".to_string())
        }
    }
}

pub fn execute_reported(
    executable: &CpuExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    cancelled: &CancellationFlag,
    state: Option<&dyn CpuState>,
) -> Result<CpuExecution, String> {
    execute_reported_with_commit(
        executable,
        declared_bindings,
        generated_bindings,
        &[],
        cancelled,
        state,
        None,
    )
}

pub fn execute_reported_with_scalars(
    executable: &CpuExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    scalar_bindings: &[f64],
    cancelled: &CancellationFlag,
) -> Result<CpuExecution, String> {
    execute_reported_with_commit(
        executable,
        declared_bindings,
        generated_bindings,
        scalar_bindings,
        cancelled,
        None,
        None,
    )
}

fn execute_reported_with_commit(
    executable: &CpuExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    scalar_bindings: &[f64],
    cancelled: &CancellationFlag,
    state: Option<&dyn CpuState>,
    commit_allowed: Option<&dyn Fn() -> bool>,
) -> Result<CpuExecution, String> {
    if cancelled.load(Ordering::Acquire) {
        return Err("operation aborted".to_string());
    }
    executable
        .signature
        .validate_invocation_counts(
            declared_bindings.len(),
            scalar_bindings.len(),
            usize::from(executable.state_cursor.is_some() && state.is_some()),
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
        executable
            .signature
            .validate_binding_metadata(
                index,
                value.dtype(),
                value.tensor().placement(),
                &value.tensor().layout,
            )
            .map_err(|error| format!("execute: {error}"))?;
    }
    let nonce = INVOCATION_NONCE.fetch_add(1, Ordering::AcqRel);
    let segments = acquire_segments(executable)?;
    let values = {
        let _allocation_guard = ExecutableAllocationGuard::enter();
        let values = resolve_values(
            executable,
            declared_bindings,
            generated_bindings,
            scalar_bindings,
            &segments,
        )?;
        if let Some(state) = state {
            state.begin(executable, &values)?;
        }
        values
    };

    let mut max_inputs = 0;
    let mut max_outputs = 0;
    let mut max_scratch = 0;
    let mut max_staging = 0;
    let mut max_state_outputs = 0;
    for physical in &executable.physical {
        let instruction = resolve_physical_instruction(executable, physical)?;
        max_inputs = max_inputs.max(instruction.inputs.len());
        max_outputs = max_outputs.max(instruction.outputs.len());
        max_scratch = max_scratch.max(instruction.scratch.len());
        max_staging = max_staging.max(instruction.staging.len());
        max_state_outputs = max_state_outputs.max(instruction.state.len());
    }
    let mut input_values = Vec::with_capacity(max_inputs);
    let mut staging_values = Vec::with_capacity(max_staging);
    let mut destinations = Vec::with_capacity(max_outputs);
    let mut scratch = Vec::with_capacity(max_scratch);
    let mut staging_destinations = Vec::with_capacity(max_staging);
    let mut state_destinations = Vec::with_capacity(max_state_outputs);
    let mut lanes = Vec::with_capacity(max_inputs.max(max_outputs));

    let execution = (|| {
        let _allocation_guard = ExecutableAllocationGuard::enter();
        for physical in &executable.physical {
            if cancelled.load(Ordering::Acquire) {
                return Err("operation aborted".to_string());
            }
            let command = resolve_physical_instruction(executable, physical)?;
            input_values.clear();
            staging_values.clear();
            input_values.extend(
                command
                    .inputs
                    .iter()
                    .map(|value_use| values[value_use.value.index()].clone()),
            );
            staging_values.extend(
                command
                    .staging
                    .iter()
                    .map(|value_use| values[value_use.value.index()].clone()),
            );
            dispatch_command(
                nonce,
                command,
                &values,
                &input_values,
                &staging_values,
                &mut destinations,
                &mut scratch,
                &mut staging_destinations,
                &mut state_destinations,
                &mut lanes,
                state,
                cancelled,
            )?;
        }
        if cancelled.load(Ordering::Acquire) {
            return Err("operation aborted".to_string());
        }
        Ok(())
    })();
    if let Err(error) = execution {
        if let Some(state) = state {
            state.rollback();
        }
        return Err(error);
    }
    if commit_allowed.is_some_and(|allowed| !allowed()) {
        if let Some(state) = state {
            state.rollback();
        }
        return Err("operation aborted".to_string());
    }
    if let Some(state) = state {
        if let Err(error) = state.commit(executable, &values) {
            state.rollback();
            return Err(error);
        }
    }
    let outputs = executable
        .program
        .outputs
        .iter()
        .map(|value| values[value.index()].clone())
        .collect();
    Ok(CpuExecution {
        outputs,
        memory: InvocationMemoryReport {
            logical: executable.memory.report.clone(),
            leased_workspace_bytes: segments.actual_workspace_bytes,
            opaque_headroom_bytes: 0,
        },
    })
}

pub fn execute(
    executable: &CpuExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    cancelled: &CancellationFlag,
    state: Option<&dyn CpuState>,
) -> Result<Vec<Value>, String> {
    Ok(execute_reported(
        executable,
        declared_bindings,
        generated_bindings,
        cancelled,
        state,
    )?
    .outputs)
}

pub fn execute_with_scalars(
    executable: &CpuExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    scalar_bindings: &[f64],
    cancelled: &CancellationFlag,
) -> Result<Vec<Value>, String> {
    Ok(execute_reported_with_scalars(
        executable,
        declared_bindings,
        generated_bindings,
        scalar_bindings,
        cancelled,
    )?
    .outputs)
}

pub fn execute_stateful(
    executable: &CpuExecutable,
    declared_bindings: &[Value],
    generated_bindings: &[Value],
    cancelled: &CancellationFlag,
    state: &dyn CpuState,
    commit_allowed: &dyn Fn() -> bool,
) -> Result<Vec<Value>, String> {
    Ok(execute_reported_with_commit(
        executable,
        declared_bindings,
        generated_bindings,
        &[],
        cancelled,
        Some(state),
        Some(commit_allowed),
    )?
    .outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_graph::{Device, LeafSlot};
    use std::sync::atomic::AtomicBool;
    use std::sync::Barrier;

    fn leaf(values: Vec<f32>) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            Tensor::from_vec(values.clone(), vec![values.len()]),
        )))))
        .unwrap()
    }

    fn leaf_shape(values: Vec<f32>, shape: Vec<usize>) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            Tensor::from_vec(values, shape),
        )))))
        .unwrap()
    }

    fn leaf_u8_shape(values: Vec<u8>, shape: Vec<usize>) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            Tensor::from_vec(values, shape),
        )))))
        .unwrap()
    }

    fn leaf_u32_shape(values: Vec<u32>, shape: Vec<usize>) -> Arc<Node> {
        Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            Tensor::from_vec(values, shape),
        )))))
        .unwrap()
    }

    fn options(optimize: bool) -> CompileOptions {
        CompileOptions {
            optimize,
            ..CompileOptions::default()
        }
    }

    fn encoded_instructions(executable: &CpuExecutable) -> Vec<&CpuCommand> {
        executable
            .physical
            .iter()
            .map(|physical| {
                let CpuPhysicalCommand::Encode(id) = *physical;
                let instruction = executable
                    .instruction(id)
                    .expect("physical ID resolves in the authoritative program");
                assert!(instruction.kind.operation().is_some());
                instruction
            })
            .collect()
    }

    fn operation(instruction: &CpuCommand) -> &CpuOp {
        instruction
            .kind
            .operation()
            .expect("encoded instruction is an operation")
            .0
    }

    fn output_ids(instruction: &CpuCommand) -> Vec<ValueId> {
        instruction
            .outputs
            .iter()
            .map(|output| output.value)
            .collect()
    }

    #[test]
    fn physical_encodes_resolve_authoritative_operations_and_resources() {
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
        let executable = compile(&[recurrence], options(false), 1024)
            .unwrap()
            .executable;

        assert_eq!(executable.physical.len(), 1);
        for physical in &executable.physical {
            let CpuPhysicalCommand::Encode(id) = *physical;
            let instruction = executable
                .program
                .instructions
                .get(id.index())
                .expect("physical ID is in range");
            assert_eq!(instruction.id, id);
            assert!(matches!(instruction.kind, CpuInstruction::Operation { .. }));

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
            assert!(!instruction.staging.is_empty());
            assert!(!instruction.state.is_empty());
        }
    }

    #[test]
    fn quantized_linear_lowers_and_executes() {
        let fixture = crate::quantized::fixtures::fixture(GgmlKQuant::Q4K);
        let input = leaf_shape(vec![1.0; 256], vec![1, 256]);
        let mut packed = Vec::with_capacity(3 * fixture.bytes.len());
        for _ in 0..3 {
            packed.extend_from_slice(fixture.bytes);
        }
        let weight = leaf_u8_shape(packed, vec![3, fixture.bytes.len()]);
        let root = Node::new(NodeKind::QuantizedLinear {
            x: input,
            weight,
            bias: None,
            codec: GgmlKQuant::Q4K,
            weight_shape: [3, 256],
        })
        .unwrap();
        let compilation = compile(&[root], options(false), 1024).unwrap();
        assert!(encoded_instructions(&compilation.executable)
            .iter()
            .any(|instruction| matches!(
                operation(instruction),
                CpuOp::QuantizedLinear {
                    codec: GgmlKQuant::Q4K,
                    ..
                }
            )));

        let outputs = execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            None,
        )
        .unwrap();
        assert_eq!(outputs[0].to_f32_vec().unwrap(), [2048.0; 3]);
    }

    #[test]
    fn mixed_k_quant_embedding_codecs_execute_in_one_program() {
        let mut roots = Vec::new();
        for fixture in &crate::quantized::fixtures::CASES {
            roots.push(
                Node::new(NodeKind::QuantizedEmbedding {
                    indexes: leaf_u32_shape(vec![0], vec![1]),
                    weight: leaf_u8_shape(fixture.bytes.to_vec(), vec![1, fixture.bytes.len()]),
                    codec: fixture.codec,
                    weight_shape: [1, 256],
                    padding_index: Some(0),
                })
                .unwrap(),
            );
        }
        let compilation = compile(&roots, options(false), 1024).unwrap();
        let outputs = execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            None,
        )
        .unwrap();
        assert_eq!(outputs.len(), 5);
        for (output, fixture) in outputs.iter().zip(&crate::quantized::fixtures::CASES) {
            let values = output.to_f32_vec().unwrap();
            for (group, &expected) in fixture.expected_groups.iter().enumerate() {
                assert_eq!(
                    &values[group * fixture.group_width..(group + 1) * fixture.group_width],
                    vec![expected; fixture.group_width],
                    "{} group {group}",
                    fixture.codec.name()
                );
            }
        }
    }

    #[test]
    fn typed_program_is_authoritative_for_values_and_diagnostics() {
        fn assert_typed(_: &LoweredProgram<CpuInstruction, NativeMemorySpace, CpuLoweredValue>) {}

        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let scalar = Node::new(NodeKind::ScalarInput {
            slot: 1,
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add {
            a: input,
            b: scalar,
        })
        .unwrap();
        let executable = compile(&[root], options(false), 1024).unwrap().executable;
        assert_typed(&executable.program);

        assert!(matches!(
            executable
                .program
                .instructions
                .first()
                .map(|instruction| &instruction.kind),
            Some(CpuInstruction::PrepareInvocation)
        ));
        assert!(matches!(
            executable
                .program
                .instructions
                .last()
                .map(|instruction| &instruction.kind),
            Some(CpuInstruction::FinalizeInvocation)
        ));
        for (index, value) in executable.program.values.iter().enumerate() {
            assert_eq!(value.declaration.id.index(), index);
            assert_eq!(value.layout.shape(), value.shape.as_ref());
        }

        let mut expected = std::collections::BTreeMap::<String, usize>::new();
        for instruction in &executable.program.instructions {
            *expected
                .entry(instruction.kind.name().to_string())
                .or_default() += 1;
        }
        let expected = expected.into_iter().collect::<Vec<_>>();
        let actual = executable
            .diagnostics
            .instructions
            .iter()
            .map(|instruction| (instruction.kind.clone(), instruction.count))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(
            executable.diagnostics.command_count,
            executable.physical.len()
        );
    }

    #[test]
    fn prepared_compile_does_not_reload_leaf_slots() {
        let slot = Arc::new(LeafSlot::new(Value(Tensor::from_vec(
            vec![1.0f32, 2.0],
            vec![2],
        ))));
        let root = Node::new(NodeKind::Leaf(Arc::clone(&slot))).unwrap();
        let program = ProgramRequest::from_roots(vec![root], CompileOptions::default())
            .prepare()
            .unwrap();
        let generated = load_generated_values(&program.index).unwrap();
        assert_eq!(program.index.work.graph_index_builds, 1);
        assert!(slot.clear());

        let compilation = compile_prepared(&program, &generated, None).unwrap();
        assert_eq!(compilation.generated_bindings.len(), 1);
        assert_eq!(compilation.executable.memory.report.external_bytes, 8);
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
                "artifact_assembly",
                "publication",
            ]
        );
        let output = execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            None,
        )
        .unwrap()
        .remove(0);
        assert_eq!(output.to_f32_vec().unwrap(), [1.0, 2.0]);
    }

    #[test]
    fn constant_weights_retain_exact_leaf_storage_without_generated_bindings() {
        let mut tensor = Tensor::from_vec(vec![1.0f32, 99.0, 2.0, 99.0], vec![4]);
        let layout = effect_torch_runtime::Layout::new(vec![2], vec![2], 0);
        tensor.layout = layout.clone();
        let slot = Arc::new(LeafSlot::new(Value(tensor)));
        let leaf = Node::new(NodeKind::Leaf(Arc::clone(&slot))).unwrap();
        let root = Node::new(NodeKind::Neg { a: leaf }).unwrap();
        let program = ProgramRequest::from_roots(
            vec![root],
            CompileOptions {
                inference: Some(effect_torch_compiler::InferenceOptions {
                    constant_weights: true,
                }),
                ..CompileOptions::default()
            },
        )
        .prepare()
        .unwrap();
        let generated = load_generated_values(&program.index).unwrap();
        assert!(slot.clear());

        let compilation = compile_prepared(&program, &generated, None).unwrap();

        assert!(compilation.generated_bindings.is_empty());
        assert!(compilation.generated_slots.is_empty());
        assert!(compilation.executable.bindings.is_empty());
        assert_eq!(compilation.executable.constants.len(), 1);
        assert_eq!(compilation.executable.memory.report.external_bytes, 0);
        assert_eq!(compilation.executable.memory.report.persistent_bytes, 8);
        let constant = &compilation.executable.constants[0];
        let lowered = &compilation.executable.program.values[constant.value.index()];
        assert_eq!(lowered.layout, layout);
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
        let compilation = compile(
            &[root],
            CompileOptions {
                inference: Some(effect_torch_compiler::InferenceOptions {
                    constant_weights: false,
                }),
                ..CompileOptions::default()
            },
            1024,
        )
        .unwrap();

        assert_eq!(compilation.generated_bindings.len(), 1);
        assert!(compilation.executable.constants.is_empty());
        assert_eq!(compilation.executable.memory.report.external_bytes, 8);
        assert_eq!(compilation.executable.memory.report.persistent_bytes, 0);
        assert_eq!(run(&compilation)[0].to_f32_vec().unwrap(), [3.0, 4.0]);
    }

    fn run(compilation: &CpuCompilation) -> Vec<Value> {
        execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn scalar_bindings_are_written_into_planned_invocation_staging() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let scalar = Node::new(NodeKind::ScalarInput {
            slot: 1,
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add {
            a: input,
            b: scalar,
        })
        .unwrap();
        let compilation = compile(&[root], options(false), 1024).unwrap();
        assert_eq!(compilation.executable.signature.bindings.len(), 1);
        assert_eq!(
            compilation.executable.signature.bindings[0].layout,
            effect_torch_runtime::BindingLayoutPolicy::Require(
                effect_torch_runtime::LayoutConstraint::Exact(
                    effect_torch_runtime::Layout::contiguous(vec![2])
                )
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

        let output = execute_with_scalars(
            &compilation.executable,
            &[Value(Tensor::from_vec(vec![1.0f32, 2.0], vec![2]))],
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
                    device: Device::Cpu,
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
                    device: Device::Cpu,
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
                CpuStatePlan {
                    bytes: 0,
                    cursor_slot: geometry.cursor_slot,
                    cursor_tensor: geometry.cursor_tensor,
                    batch,
                },
            )
            .unwrap();
            let signature = &compilation.executable.signature;

            assert_eq!(signature.bindings.len(), 3);
            assert!(signature.bindings.iter().all(|binding| matches!(
                &binding.layout,
                effect_torch_runtime::BindingLayoutPolicy::Require(
                    effect_torch_runtime::LayoutConstraint::Exact(layout)
                ) if layout == &effect_torch_runtime::Layout::contiguous(vec![batch, 1, 2, 4])
            )));
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
    fn retained_outputs_remain_valid_across_later_invocations() {
        let random = Node::new(NodeKind::Randn {
            shape: vec![32],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let compilation = compile(&[random], options(false), 1024).unwrap();
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
    fn one_plan_executes_concurrently_with_independent_frames() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let output = Node::new(NodeKind::Neg { a: input }).unwrap();
        let executable = compile(&[output], options(false), 1024).unwrap().executable;
        let barrier = Arc::new(Barrier::new(9));
        let handles = (0..8)
            .map(|index| {
                let executable = Arc::clone(&executable);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let input = Value(Tensor::from_vec(
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
    fn bounded_batch_inputs_are_zero_padded_in_planned_staging() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2, 2],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add {
            a: input,
            b: leaf_shape(vec![1.0; 4], vec![2, 2]),
        })
        .unwrap();
        let compilation = compile_with_state(
            &[root],
            options(false),
            1024,
            CpuStatePlan {
                bytes: 0,
                cursor_slot: u32::MAX,
                cursor_tensor: false,
                batch: 2,
            },
        )
        .unwrap();
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

        let output = execute_with_scalars(
            &compilation.executable,
            &[Value(Tensor::from_vec(vec![2.0f32, 3.0], vec![1, 2]))],
            &compilation.generated_bindings,
            &[],
            &CancellationFlag::new(),
        )
        .unwrap();
        assert_eq!(output[0].to_f32_vec().unwrap(), [3.0, 4.0, 1.0, 1.0]);
    }

    #[test]
    fn command_order_is_deterministic_and_shared_nodes_are_lowered_once() {
        let x = leaf(vec![1.0, 2.0]);
        let y = leaf(vec![3.0, 4.0]);
        let shared = Node::new(NodeKind::Add { a: x, b: y }).unwrap();
        let left = Node::new(NodeKind::Neg { a: shared.clone() }).unwrap();
        let right = Node::new(NodeKind::Tanh { a: shared }).unwrap();
        let compilation = compile(&[left, right], options(false), 1024).unwrap();
        let instructions = encoded_instructions(&compilation.executable);
        assert_eq!(
            instructions
                .iter()
                .map(|instruction| operation(instruction).name())
                .collect::<Vec<_>>(),
            ["binary", "unary", "unary"]
        );
        assert_eq!(instructions[1].inputs.len(), 1);
        assert_eq!(
            instructions[1].inputs[0].value,
            instructions[0].outputs[0].value
        );
        assert_eq!(
            instructions[2].inputs[0].value,
            instructions[0].outputs[0].value
        );
    }

    #[test]
    fn executable_does_not_keep_graph_roots_alive() {
        let root = Node::new(NodeKind::Neg {
            a: Node::new(NodeKind::Neg {
                a: leaf(vec![1.0, 2.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let weak = Arc::downgrade(&root);
        let compilation = compile(std::slice::from_ref(&root), options(true), 1024).unwrap();
        assert!(matches!(
            operation(encoded_instructions(&compilation.executable)[0]),
            CpuOp::FusedElementwise { .. }
        ));
        drop(root);
        assert!(weak.upgrade().is_none());
        assert_eq!(run(&compilation)[0].to_f32_vec().unwrap(), [1.0, 2.0]);
    }

    #[test]
    fn production_optimizer_selectors_route_to_region_outputs_without_commands() {
        let step = Node::new(NodeKind::SgdStep {
            param: leaf(vec![1.0, 2.0]),
            grad: leaf(vec![0.5, -0.25]),
            velocity: leaf(vec![0.0, 0.0]),
            first: leaf_shape(vec![1.0], vec![]),
            lr: leaf_shape(vec![0.1], vec![]),
            momentum: 0.9,
            dampening: 0.0,
            nesterov: false,
            weight_decay: 0.0,
        })
        .unwrap();
        let velocity = Node::new(NodeKind::SgdOut {
            step: step.clone(),
            index: 1,
        })
        .unwrap();
        let compilation = compile(&[step, velocity], options(true), 1024).unwrap();
        let instructions = encoded_instructions(&compilation.executable);
        assert_eq!(instructions.len(), 1);
        assert!(matches!(
            operation(instructions[0]),
            CpuOp::Sgd {
                implementation: OptimizerImplementation::Fused,
                ..
            }
        ));
        assert_eq!(
            compilation.executable.program.outputs.as_ref(),
            output_ids(instructions[0])
        );
        let outputs = run(&compilation);
        assert_eq!(outputs[0].to_f32_vec().unwrap(), [0.95, 2.025]);
        assert_eq!(outputs[1].to_f32_vec().unwrap(), [0.5, -0.25]);
    }

    #[test]
    fn a_shared_random_source_is_shared_across_roots_and_fresh_per_invocation() {
        let random = Node::new(NodeKind::Randn {
            shape: vec![16],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let compilation = compile(&[random.clone(), random], options(false), 1024).unwrap();
        assert_eq!(compilation.executable.physical.len(), 1);
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
    fn independently_compiled_random_plans_receive_distinct_runtime_nonces() {
        let random = Node::new(NodeKind::Randn {
            shape: vec![16],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let executables = [
            compile(std::slice::from_ref(&random), options(false), 1024)
                .unwrap()
                .executable,
            compile(&[random], options(false), 1024).unwrap().executable,
        ];
        let barrier = Arc::new(Barrier::new(3));
        let handles = executables
            .into_iter()
            .map(|executable| {
                let executable = Arc::clone(&executable);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    execute(&executable, &[], &[], &CancellationFlag::new(), None)
                        .unwrap()
                        .remove(0)
                        .to_f32_vec()
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let outputs = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_ne!(outputs[0], outputs[1]);
    }

    #[test]
    fn random_stream_identity_is_stable_across_optimization() {
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
            device: Device::Cpu,
        })
        .unwrap();
        let random_id = random.id;
        let optimized = compile(
            &[deterministic.clone(), random.clone()],
            options(true),
            1024,
        )
        .unwrap();
        let unoptimized = compile(&[deterministic, random], options(false), 1024).unwrap();
        assert_ne!(
            optimized.executable.physical.len(),
            unoptimized.executable.physical.len()
        );
        let provenance = |compilation: &CpuCompilation| {
            encoded_instructions(&compilation.executable)
                .into_iter()
                .find_map(|instruction| match operation(instruction) {
                    CpuOp::Randn { provenance, .. } => Some(*provenance),
                    _ => None,
                })
                .expect("compiled random command")
        };
        let optimized_provenance = provenance(&optimized);
        let unoptimized_provenance = provenance(&unoptimized);
        assert_eq!(optimized_provenance, unoptimized_provenance);
        assert_eq!(optimized_provenance, random_id);

        let sample = |provenance| {
            let mut tensor = Tensor::empty(&[16], DType::F32);
            Tensor::randn_seeded_into(
                random_seed(7, provenance),
                &mut tensor.destination().unwrap(),
            )
            .unwrap();
            Value(tensor).to_f32_vec().unwrap()
        };
        assert_eq!(sample(optimized_provenance), sample(unoptimized_provenance));
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
        let roots = [root.clone(), root];
        let optimized = compile(&roots, options(true), 1024).unwrap();
        let unoptimized = compile(&roots, options(false), 1024).unwrap();
        assert_eq!(optimized.executable.physical.len(), 1);
        assert_eq!(unoptimized.executable.physical.len(), 2);
        assert!(matches!(
            operation(encoded_instructions(&optimized.executable)[0]),
            CpuOp::FusedElementwise { .. }
        ));
        assert_eq!(
            optimized.executable.program.outputs[0],
            optimized.executable.program.outputs[1]
        );
        assert_eq!(
            run(&optimized)[0].to_f32_vec().unwrap(),
            run(&unoptimized)[0].to_f32_vec().unwrap()
        );
    }

    #[test]
    fn direct_elementwise_reduce_region_matches_semantic_lowering() {
        let sum = Node::new(NodeKind::Add {
            a: leaf_shape(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]),
            b: leaf_shape(vec![0.5, 0.25, -0.5, 2.0], vec![2, 2]),
        })
        .unwrap();
        let tanh = Node::new(NodeKind::Tanh { a: sum }).unwrap();
        let root = Node::new(NodeKind::Sum {
            a: tanh,
            dims: vec![1],
            keepdims: false,
        })
        .unwrap();
        let optimized = compile(std::slice::from_ref(&root), options(true), 1024).unwrap();
        let unoptimized = compile(std::slice::from_ref(&root), options(false), 1024).unwrap();
        assert_eq!(optimized.executable.physical.len(), 1);
        assert_eq!(unoptimized.executable.physical.len(), 3);
        assert!(matches!(
            operation(encoded_instructions(&optimized.executable)[0]),
            CpuOp::FusedReduce { .. }
        ));
        assert_eq!(
            run(&optimized)[0].to_f32_vec().unwrap(),
            run(&unoptimized)[0].to_f32_vec().unwrap()
        );
    }

    #[test]
    fn direct_multi_output_region_emits_one_command_and_no_selectors() {
        let sum = Node::new(NodeKind::Add {
            a: leaf(vec![1.0, -2.0, 3.0, -4.0]),
            b: leaf(vec![0.5, 0.25, -0.5, 2.0]),
        })
        .unwrap();
        let prefix = Node::new(NodeKind::Tanh { a: sum }).unwrap();
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
        let optimized = compile(&roots, options(true), 1024).unwrap();
        let unoptimized = compile(&roots, options(false), 1024).unwrap();
        let instructions = encoded_instructions(&optimized.executable);
        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].outputs.len(), 2);
        assert_eq!(unoptimized.executable.physical.len(), 6);
        assert_eq!(
            optimized.executable.program.outputs.as_ref(),
            output_ids(instructions[0])
        );
        for (optimized, unoptimized) in run(&optimized).iter().zip(run(&unoptimized)) {
            assert_eq!(
                optimized.to_f32_vec().unwrap(),
                unoptimized.to_f32_vec().unwrap()
            );
        }
    }

    #[test]
    fn optimization_plan_reports_one_index_and_no_semantic_rebuilds() {
        let root = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add {
                a: leaf(vec![1.0, 2.0]),
                b: leaf(vec![3.0, 4.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let compilation = compile(&[root], options(true), 1024).unwrap();
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
    fn standalone_gelu_uses_compiled_view_strides() {
        let source = leaf_shape(vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], vec![2, 3]);
        let permuted = Node::new(NodeKind::Permute {
            a: source,
            dims: vec![1, 0],
        })
        .unwrap();
        let gelu = Node::new(NodeKind::Gelu {
            a: permuted,
            approximate: false,
        })
        .unwrap();
        let compilation = compile(&[gelu], options(false), 1024).unwrap();
        let actual = run(&compilation)[0].to_f32_vec().unwrap();
        let expected_inputs = [-1.0f32, 0.5, -0.5, 1.0, 0.0, 1.5];
        for (actual, input) in actual.iter().zip(expected_inputs) {
            let expected = input
                * 0.5
                * (1.0 + libm::erf(input as f64 * std::f64::consts::FRAC_1_SQRT_2) as f32);
            assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        }
    }

    #[test]
    fn multidimensional_strided_slice_is_one_alias() {
        let source = leaf_shape((0..20).map(|value| value as f32).collect(), vec![4, 5]);
        let slice = Node::new(NodeKind::Slice {
            a: source,
            ranges: vec![(1, 4, 2), (1, 5, 2)],
        })
        .unwrap();
        let compilation = compile(&[slice], options(false), 1024).unwrap();
        assert!(compilation.executable.physical.is_empty());
        assert_eq!(
            run(&compilation)[0].to_f32_vec().unwrap(),
            [6.0, 8.0, 16.0, 18.0]
        );
    }

    #[test]
    fn unsupported_device_dtype_and_layout_fail_during_compilation() {
        let metal = Node::new(NodeKind::Zeros {
            shape: vec![1],
            dtype: DType::F32,
            device: Device::Metal,
        })
        .unwrap();
        assert!(compile(&[metal], options(false), 1024)
            .err()
            .unwrap()
            .contains("does not support device metal"));

        let half = || {
            Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
                Tensor::ones(&[2, 2], DType::F16),
            )))))
            .unwrap()
        };
        let half_matmul = Node::new(NodeKind::Matmul {
            a: half(),
            b: half(),
        })
        .unwrap();
        assert!(compile(&[half_matmul], options(false), 1024)
            .err()
            .unwrap()
            .contains("matmul does not support CPU dtype f16"));

        let overflowing = Node::new(NodeKind::Zeros {
            shape: vec![usize::MAX, 2],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        assert!(compile(&[overflowing], options(false), 1024)
            .err()
            .unwrap()
            .contains("unsupported overflowing layout"));
    }

    #[test]
    fn last_use_is_planned_and_intermediates_are_workspace() {
        let root = Node::new(NodeKind::Tanh {
            a: Node::new(NodeKind::Add {
                a: leaf(vec![1.0, 2.0]),
                b: leaf(vec![3.0, 4.0]),
            })
            .unwrap(),
        })
        .unwrap();
        let compilation = compile(&[root], options(false), 1024).unwrap();
        let instructions = encoded_instructions(&compilation.executable);
        let intermediate = instructions[0].outputs[0].value;
        assert!(matches!(
            compilation.executable.memory.locations[intermediate.index()],
            Location::Segment { .. }
        ));
        assert!(compilation.executable.diagnostics.memory.output_bytes > 0);
        assert!(compilation.executable.diagnostics.memory.workspace_bytes > 0);
        assert_eq!(run(&compilation)[0].to_f32_vec().unwrap().len(), 2);
    }

    #[test]
    fn ce_chunk_size_and_optimizer_implementation_are_compile_time_choices() {
        let tensor = |shape: Vec<usize>, value: f32| {
            Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
                Tensor::full(&shape, value as f64, DType::F32),
            )))))
            .unwrap()
        };
        let scalar = |value: f32| tensor(Vec::new(), value);
        let adamw = Node::new(NodeKind::AdamWStep {
            param: tensor(vec![2], 1.0),
            grad: tensor(vec![2], 0.1),
            m: tensor(vec![2], 0.0),
            v: tensor(vec![2], 0.0),
            lr: scalar(0.01),
            c1: scalar(0.1),
            c2: scalar(0.01),
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.01,
        })
        .unwrap();
        let optimized = compile(std::slice::from_ref(&adamw), options(true), 1024).unwrap();
        let unoptimized = compile(std::slice::from_ref(&adamw), options(false), 1024).unwrap();
        assert!(matches!(
            operation(encoded_instructions(&optimized.executable)[0]),
            CpuOp::AdamW {
                implementation: OptimizerImplementation::Fused,
                fused_exprs: Some(_),
                ..
            }
        ));
        assert!(matches!(
            operation(encoded_instructions(&unoptimized.executable)[0]),
            CpuOp::AdamW {
                implementation: OptimizerImplementation::Composed,
                fused_exprs: None,
                ..
            }
        ));
        for (optimized, unoptimized) in run(&optimized)[0]
            .to_f32_vec()
            .unwrap()
            .iter()
            .zip(run(&unoptimized)[0].to_f32_vec().unwrap())
        {
            assert!((optimized - unoptimized).abs() < 1e-6);
        }

        let target = Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value(
            Tensor::from_vec(vec![0i64, 1], vec![2]),
        )))))
        .unwrap();
        let ce = Node::new(NodeKind::ChunkedHeadCe {
            x: tensor(vec![2, 2], 0.5),
            weight: tensor(vec![2, 3], 0.25),
            bias: tensor(vec![3], 0.0),
            target,
            ignore_index: -100,
        })
        .unwrap();
        let compilation = compile(&[ce], options(false), 17).unwrap();
        assert!(matches!(
            operation(encoded_instructions(&compilation.executable)[0]),
            CpuOp::ChunkedHeadCe { chunk_size: 17, .. }
        ));
    }

    #[test]
    fn grouped_adamw_region_preserves_output_routes_and_numerical_results() {
        let scalar = |value| leaf_shape(vec![value], vec![]);
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
        let grouped = compile(&roots, grouped_options, 1024).unwrap();
        let independent = compile(&roots, options(false), 1024).unwrap();

        let instructions = encoded_instructions(&grouped.executable);
        assert_eq!(instructions.len(), 1);
        assert_eq!(independent.executable.physical.len(), 2);
        assert!(matches!(
            operation(instructions[0]),
            CpuOp::AdamWGroup { parameters: 2, .. }
        ));
        assert_eq!(instructions[0].outputs.len(), 6);
        assert_eq!(
            grouped.executable.program.outputs.as_ref(),
            [
                instructions[0].outputs[0].value,
                instructions[0].outputs[1].value,
                instructions[0].outputs[3].value,
                instructions[0].outputs[5].value,
            ]
        );
        for (grouped, independent) in run(&grouped).iter().zip(run(&independent)) {
            for (grouped, independent) in grouped
                .to_f32_vec()
                .unwrap()
                .iter()
                .zip(independent.to_f32_vec().unwrap())
            {
                assert!((grouped - independent).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn grouped_adamw_rejects_different_runtime_scalars_and_preserves_results() {
        let scalar = |value| leaf_shape(vec![value], vec![]);
        let step = |param: Vec<f32>, grad: Vec<f32>, lr: f32, c1: f32, c2: f32| {
            Node::new(NodeKind::AdamWStep {
                param: leaf(param),
                grad: leaf(grad),
                m: leaf(vec![0.0, 0.0]),
                v: leaf(vec![0.0, 0.0]),
                lr: scalar(lr),
                c1: scalar(c1),
                c2: scalar(c2),
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.01,
            })
            .unwrap()
        };
        let first = step(vec![1.0, 2.0], vec![0.1, -0.2], 0.01, 0.1, 0.01);
        let second = step(vec![3.0, 4.0], vec![-0.3, 0.4], 0.02, 0.2, 0.04);
        let roots = [first, second];
        let mut grouped_options = options(true);
        grouped_options.environment.optimizer_groups = true;
        let selected = compile(&roots, grouped_options, 1024).unwrap();
        let independent = compile(&roots, options(false), 1024).unwrap();

        let instructions = encoded_instructions(&selected.executable);
        assert_eq!(instructions.len(), 2);
        assert!(instructions.iter().all(|instruction| matches!(
            operation(instruction),
            CpuOp::AdamW {
                implementation: OptimizerImplementation::Fused,
                ..
            }
        )));
        assert!(!instructions
            .iter()
            .any(|instruction| matches!(operation(instruction), CpuOp::AdamWGroup { .. })));
        for (selected, independent) in run(&selected).iter().zip(run(&independent)) {
            for (selected, independent) in selected
                .to_f32_vec()
                .unwrap()
                .iter()
                .zip(independent.to_f32_vec().unwrap())
            {
                assert!((selected - independent).abs() < 1e-6);
            }
        }
    }

    struct InjectedFailureState {
        fail_command: bool,
        committed: AtomicBool,
        rolled_back: AtomicBool,
    }

    impl CpuState for InjectedFailureState {
        fn run_command(
            &self,
            _command: &CpuCommand,
            _inputs: &[Value],
            _staging: &[Value],
            outputs: &mut [CpuDestination<'_>],
            _scratch: &mut [CpuDestination<'_>],
            state_outputs: &mut [CpuDestination<'_>],
        ) -> Result<(), String> {
            outputs[0].write_current::<f32, _>("injected output", |values| values.fill(1.0))?;
            state_outputs[0]
                .write_current::<f32, _>("injected state", |values| values.fill(2.0))?;
            if self.fail_command {
                Err("injected state command failure".to_string())
            } else {
                Ok(())
            }
        }

        fn commit(&self, _executable: &CpuExecutable, _values: &[Value]) -> Result<(), String> {
            self.committed.store(true, Ordering::Release);
            Ok(())
        }

        fn rollback(&self) {
            self.rolled_back.store(true, Ordering::Release);
        }
    }

    #[test]
    fn state_outputs_are_planned_transactions_and_failures_roll_back() {
        let q = leaf_shape(vec![0.2, 0.4, 0.1, 0.3], vec![1, 1, 2, 2]);
        let recurrence = Node::new(NodeKind::KdaRecurrence {
            q: q.clone(),
            k: q.clone(),
            v: q.clone(),
            log_decay: leaf_shape(vec![-0.1, -0.2, -0.3, -0.4], vec![1, 1, 2, 2]),
            beta: leaf_shape(vec![0.5, 0.25], vec![1, 1, 2, 1]),
            scale: 0.5,
            layer: 0,
        })
        .unwrap();
        let compilation = compile(&[recurrence], options(false), 1024).unwrap();
        assert_eq!(
            encoded_instructions(&compilation.executable)[0].state.len(),
            1
        );
        assert!(compilation.executable.memory.report.transaction_bytes > 0);
        assert!(compilation
            .executable
            .memory
            .segments
            .iter()
            .any(|segment| segment.ownership == SegmentOwnership::StateTransaction));

        let state = InjectedFailureState {
            fail_command: true,
            committed: AtomicBool::new(false),
            rolled_back: AtomicBool::new(false),
        };
        let error = execute(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            Some(&state),
        )
        .unwrap_err();
        assert!(error.contains("injected state command failure"));
        assert!(!state.committed.load(Ordering::Acquire));
        assert!(state.rolled_back.load(Ordering::Acquire));

        let cancelled = InjectedFailureState {
            fail_command: false,
            committed: AtomicBool::new(false),
            rolled_back: AtomicBool::new(false),
        };
        let error = execute_stateful(
            &compilation.executable,
            &[],
            &compilation.generated_bindings,
            &CancellationFlag::new(),
            &cancelled,
            &|| false,
        )
        .unwrap_err();
        assert_eq!(error, "operation aborted");
        assert!(!cancelled.committed.load(Ordering::Acquire));
        assert!(cancelled.rolled_back.load(Ordering::Acquire));
    }

    #[test]
    fn state_transaction_segments_are_invocation_owned_and_never_published() {
        let q = leaf_shape(vec![0.2, 0.4, 0.1, 0.3], vec![1, 1, 2, 2]);
        let recurrence = Node::new(NodeKind::KdaRecurrence {
            q: q.clone(),
            k: q.clone(),
            v: q.clone(),
            log_decay: leaf_shape(vec![-0.1, -0.2, -0.3, -0.4], vec![1, 1, 2, 2]),
            beta: leaf_shape(vec![0.5, 0.25], vec![1, 1, 2, 1]),
            scale: 0.5,
            layer: 0,
        })
        .unwrap();
        let executable = compile(&[recurrence], options(false), 1024)
            .unwrap()
            .executable;
        let transaction_index = executable
            .memory
            .segments
            .iter()
            .position(|segment| segment.ownership == SegmentOwnership::StateTransaction)
            .expect("stateful compilation declares a transaction segment");

        let invocation = acquire_segments(&executable).unwrap();
        assert!(invocation.retentions[transaction_index].is_none());
    }
}

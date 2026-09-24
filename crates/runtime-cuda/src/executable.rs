//! Fixed, typed CUDA execution. Every temporary is declared before invocation.
use crate::buffer::CudaBuffer;
use crate::capabilities::CudaCapabilities;
use crate::lowering::{
    Command, CommandKind, CudaLoweredProgram, CudaProgramBuilder, BF16_LINEAR_BIAS_KERNEL,
};
use crate::value::element_count;
use crate::workspace::{self, CudaMemorySpace, InvocationResources, CUDA_STORAGE_ALIGNMENT};
use crate::{CudaDevice, CudaValue};
use cudarc::driver::{sys, CudaGraph, DeviceRepr, LaunchConfig, PushKernelArg};
use effect_torch_compiler::{
    build_executable_diagnostics, CompileOptions, CompilerDriver, CompilerWorkReport,
    DiagnosticsInput, GraphIndex, LoweringUnit, MemoryPlannerConfig, ProgramRequest,
    StateCursorSlot, ARTIFACT_ASSEMBLY_PHASE, PHYSICAL_PLANNING_PHASE, PUBLICATION_PHASE,
};
use effect_torch_graph::{AttentionRounding, KvAttentionMode, Node, NodeKind, PositionOffset};
use effect_torch_runtime::{
    CancellationFlag, DType, ExecutableDiagnostics, GgmlKQuant, KvLayerDescriptor, MemoryPlan,
    StateAccessMode, ValueId,
};
use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

#[cfg(test)]
#[path = "kv_matmul_tests.rs"]
mod kv_matmul_tests;

/// By-value launch ABI shared by every typed CUDA kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaKernelArgs {
    pub(crate) inputs: [u64; 8],
    pub(crate) output: u64,
    pub(crate) scratch: [u64; 4],
    pub(crate) metadata: u64,
    pub(crate) elements: u64,
    pub(crate) integers: [u64; 16],
    pub(crate) scalars: [f64; 8],
    pub(crate) input_dtypes: [u32; 8],
    pub(crate) output_dtype: u32,
    pub(crate) compute_dtype: u32,
    pub(crate) operation: u32,
    pub(crate) error_context: u32,
}

// SAFETY: repr(C), scalar fields only, and all fields are initialized. The CUDA
// declaration has the same field order and widths.
unsafe impl DeviceRepr for CudaKernelArgs {}

pub(crate) fn dtype_code(dtype: DType) -> u32 {
    match dtype {
        DType::F64 => 0,
        DType::F32 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::I64 => 4,
        DType::U32 => 5,
        DType::U8 => 6,
    }
}

fn codec_code(codec: GgmlKQuant) -> u64 {
    match codec {
        GgmlKQuant::Q2K => 0,
        GgmlKQuant::Q3K => 1,
        GgmlKQuant::Q4K => 2,
        GgmlKQuant::Q5K => 3,
        GgmlKQuant::Q6K => 4,
    }
}

#[derive(Clone, Debug)]
pub(super) enum StateAccess {
    None,
    Rotary,
    LastToken {
        lane: usize,
    },
    Kv {
        layer: usize,
        heads: usize,
        dim: usize,
        batch: usize,
    },
    Kda {
        layer: Option<usize>,
        batch: usize,
        elements_per_sequence: usize,
    },
    Conv {
        layer: Option<usize>,
        batch: usize,
        elements_per_sequence: usize,
    },
}

#[derive(Clone)]
pub(super) struct KernelSpec {
    pub(super) name: &'static str,
    pub(super) args: CudaKernelArgs,
    pub(super) inputs: [Option<usize>; 8],
    pub(super) tail: Vec<u64>,
    pub(super) scratch: [Option<(usize, DType)>; 3],
    pub(super) state: StateAccess,
}

impl KernelSpec {
    pub(super) fn new(name: &'static str, inputs: &[Option<usize>]) -> Self {
        let mut roles = [None; 8];
        roles[..inputs.len()].copy_from_slice(inputs);
        Self {
            name,
            args: CudaKernelArgs::default(),
            inputs: roles,
            tail: Vec::new(),
            scratch: [None; 3],
            state: StateAccess::None,
        }
    }
}

fn product(values: &[usize]) -> Result<usize, String> {
    element_count(values)
}

impl Instruction {
    pub(super) fn kernel(&self) -> Result<KernelSpec, String> {
        let spec = match self {
            Self::GroupedExpertLinearRows { .. } => {
                return Err(
                    "compile: grouped experts require non-capturable command lowering".into(),
                );
            }
            Self::Binary { op, a, b, .. } => {
                let mut s = KernelSpec::new("et_binary", &[Some(*a), Some(*b)]);
                s.args.operation = *op;
                s
            }
            Self::Unary {
                op, a, parameter, ..
            } => {
                let mut s = KernelSpec::new(
                    if *op == 17 { "et_convert" } else { "et_unary" },
                    &[Some(*a)],
                );
                s.args.operation = *op;
                s.args.scalars[0] = *parameter;
                s
            }
            Self::Reindex {
                op, a, parameters, ..
            } => {
                let mut s = KernelSpec::new("et_reindex", &[Some(*a)]);
                s.args.operation = *op;
                s.tail = parameters.clone();
                s
            }
            Self::Where { cond, a, b, .. } => {
                KernelSpec::new("et_where", &[Some(*cond), Some(*a), Some(*b)])
            }
            Self::Concat { a, b, dim, .. } => {
                let mut s = KernelSpec::new("et_concat", &[Some(*a), Some(*b)]);
                s.args.integers[0] = u64::from(*dim);
                s
            }
            Self::Reduce {
                op,
                a,
                dims,
                count,
                trailing,
            } => {
                let mut s = KernelSpec::new("et_reduce", &[Some(*a)]);
                s.args.operation = *op;
                s.args.integers[0] = dims.len() as u64;
                s.args.integers[1] = *count as u64;
                s.args.integers[2] = u64::from(*trailing);
                s.tail = dims.iter().map(|d| *d as u64).collect();
                s
            }
            Self::Matmul { a, b, .. } => KernelSpec::new("et_matmul", &[Some(*a), Some(*b)]),
            Self::TopKIndices { a, k, width } => {
                let mut s = KernelSpec::new("et_top_k_indices", &[Some(*a)]);
                s.args.integers[0] = *k as u64;
                s.args.integers[1] = *width as u64;
                s
            }
            Self::Index {
                op,
                a,
                indexes,
                src,
                dim,
                width,
                trailing,
                ..
            } => {
                let mut s = KernelSpec::new("et_index", &[Some(*a), *indexes, *src]);
                s.args.operation = *op;
                s.args.integers[0] = u64::from(*dim);
                s.args.integers[1] = *width as u64;
                s.args.integers[2] = u64::from(*trailing);
                s
            }
            Self::RmsNorm { x, weight, eps, .. } => {
                let mut s = KernelSpec::new("et_rms_norm", &[Some(*x), *weight]);
                s.args.scalars[0] = *eps;
                s
            }
            Self::CrossEntropy {
                logits,
                target,
                ignore_index,
                backward,
                ..
            } => {
                let mut s = KernelSpec::new("et_cross_entropy", &[Some(*logits), Some(*target)]);
                s.args.operation = u32::from(*backward);
                s.args.integers[0] = *ignore_index as u64;
                s
            }
            Self::ChunkedHeadCe {
                output,
                x,
                weight,
                bias,
                target,
                gradient,
                ignore_index,
                ..
            } => {
                let mut s = KernelSpec::new(
                    "et_chunked_head_ce",
                    &[
                        Some(*x),
                        Some(*weight),
                        Some(*bias),
                        Some(*target),
                        *gradient,
                    ],
                );
                s.args.operation = output.map_or(0, |o| o + 1);
                s.args.integers[0] = *ignore_index as u64;
                s
            }
            Self::Conv {
                op,
                x,
                w,
                stride,
                padding,
                dilation,
                groups,
                ..
            } => {
                let mut s = KernelSpec::new("et_conv", &[Some(*x), Some(*w)]);
                s.args.operation = *op;
                s.args.integers[..4].copy_from_slice(&[
                    u64::from(*stride),
                    u64::from(*padding),
                    u64::from(*dilation),
                    u64::from(*groups),
                ]);
                s
            }
            Self::Linalg {
                op,
                a,
                b,
                a_shape,
                shape,
                dtype,
            } => {
                let mut s = KernelSpec::new("et_linalg", &[Some(*a), *b]);
                s.args.operation = *op;
                let n = *a_shape.last().ok_or("linalg requires a matrix")?;
                let batches = product(&a_shape[..a_shape.len() - 2])?;
                let rhs = if *op == 1 {
                    0
                } else if *op == 2 {
                    product(shape)?
                        .checked_div(product(&[batches, n])?)
                        .ok_or("solve has an empty matrix")?
                } else {
                    n
                };
                s.scratch[0] = Some((
                    product(&[
                        batches,
                        n,
                        n.checked_add(rhs).ok_or("linalg workspace overflow")?,
                    ])?,
                    if *dtype == DType::F64 {
                        DType::F64
                    } else {
                        DType::F32
                    },
                ));
                s
            }
            Self::Linear {
                x,
                weight,
                bias,
                k_width,
                n_width,
                ..
            } => {
                let mut s = KernelSpec::new("et_linear", &[Some(*x), Some(*weight), Some(*bias)]);
                s.args.integers[..2].copy_from_slice(&[u64::from(*k_width), u64::from(*n_width)]);
                s
            }
            Self::ExpertLinearRows {
                x,
                weight,
                indexes,
                rows,
                columns,
                inner,
                experts,
            } => {
                let mut s = KernelSpec::new(
                    "et_expert_linear_rows",
                    &[Some(*x), Some(*weight), Some(*indexes)],
                );
                s.args.integers[..4].copy_from_slice(&[
                    *rows as u64,
                    *columns as u64,
                    *inner as u64,
                    *experts as u64,
                ]);
                s
            }
            Self::QuantizedLinear {
                x,
                weight,
                bias,
                rows,
                columns,
                row_bytes,
                codec,
                ..
            } => {
                let mut s =
                    KernelSpec::new("et_quantized_linear", &[Some(*x), Some(*weight), *bias]);
                s.args.integers[..4].copy_from_slice(&[
                    codec_code(*codec),
                    u64::from(*rows),
                    u64::from(*columns),
                    u64::from(*row_bytes),
                ]);
                s
            }
            Self::QuantizedEmbedding {
                indexes,
                weight,
                rows,
                columns,
                row_bytes,
                codec,
                padding_index,
                ..
            } => {
                let mut s =
                    KernelSpec::new("et_quantized_embedding", &[Some(*indexes), Some(*weight)]);
                s.args.integers[..4].copy_from_slice(&[
                    codec_code(*codec),
                    u64::from(*rows),
                    u64::from(*columns),
                    u64::from(*row_bytes),
                ]);
                s.args.integers[4] = padding_index.unwrap_or(0) as u64;
                s.args.integers[5] = u64::from(padding_index.is_some());
                s
            }
            Self::LayerNorm {
                op,
                x,
                weight,
                other,
                width,
                rows,
                eps,
                ..
            } => {
                let mut s =
                    KernelSpec::new("et_layer_norm", &[Some(*x), Some(*weight), Some(*other)]);
                s.args.operation = *op;
                s.args.integers[..2].copy_from_slice(&[u64::from(*width), u64::from(*rows)]);
                s.args.scalars[0] = *eps;
                s
            }
            Self::Sdpa {
                op,
                q,
                k,
                v,
                g,
                scale,
                causal,
                window,
                ..
            } => {
                let mut s = KernelSpec::new("et_sdpa", &[Some(*q), Some(*k), Some(*v), *g]);
                s.args.operation = *op;
                s.args.integers[..2]
                    .copy_from_slice(&[u64::from(*causal), window.unwrap_or(0) as u64]);
                s.args.scalars[0] = *scale;
                s
            }
            Self::Rotary {
                x,
                shape,
                seq_len,
                theta,
                interleaved,
                backward,
                cursor,
                ..
            } => {
                let mut s = KernelSpec::new("et_rotary", &[Some(*x)]);
                s.args.integers[..3].copy_from_slice(&[
                    u64::from(*seq_len),
                    u64::from(*interleaved),
                    u64::from(*backward),
                ]);
                s.args.scalars[0] = *theta;
                if *cursor {
                    s.state = StateAccess::Rotary;
                    s.scratch[0] = Some((shape.first().copied().unwrap_or(1), DType::U32));
                }
                s
            }
            Self::Optimizer {
                kind,
                output,
                param,
                grad,
                state1,
                state2,
                lr,
                c1,
                c2,
                beta1,
                beta2,
                eps,
                weight_decay,
                dampening,
                nesterov,
                ..
            } => {
                let mut s = KernelSpec::new(
                    "et_optimizer",
                    &[
                        Some(*param),
                        Some(*grad),
                        Some(*state1),
                        Some(*state2),
                        Some(*lr),
                        Some(*c1),
                        Some(*c2),
                    ],
                );
                s.args.operation = *kind;
                s.args.integers[..2].copy_from_slice(&[u64::from(*output), u64::from(*nesterov)]);
                s.args.scalars[..5].copy_from_slice(&[
                    *beta1,
                    *beta2,
                    *eps,
                    *weight_decay,
                    *dampening,
                ]);
                s
            }
            Self::ShortConv {
                op,
                state_layer,
                x,
                weight,
                g,
                input_shape,
                weight_shape,
                ..
            } => {
                let mut s = KernelSpec::new("et_short_conv", &[Some(*x), Some(*weight), *g]);
                s.args.operation = *op;
                s.args.integers[0] = u64::from(state_layer.is_some());
                let rank = input_shape.len();
                s.args.integers[8] = input_shape[rank - 2] as u64;
                let batch = product(&input_shape[..rank - 2])?;
                let per = product(&[weight_shape[1].saturating_sub(1), input_shape[rank - 1]])?;
                let count = product(&[batch, per])?;
                s.scratch = [
                    Some((count, DType::F32)),
                    Some((count, DType::F32)),
                    Some((batch, DType::U32)),
                ];
                s.state = StateAccess::Conv {
                    layer: *state_layer,
                    batch,
                    elements_per_sequence: per,
                };
                s
            }
            Self::Kda {
                output,
                state_layer,
                q,
                k,
                v,
                decay,
                beta,
                g,
                q_shape,
                v_shape,
                scale,
                dtype,
                ..
            } => {
                let mut s = KernelSpec::new(
                    "et_kda",
                    &[Some(*q), Some(*k), Some(*v), Some(*decay), Some(*beta), *g],
                );
                s.args.operation = output.unwrap_or(0);
                s.args.integers[..2].copy_from_slice(&[
                    u64::from(output.is_some()),
                    u64::from(state_layer.is_some()),
                ]);
                s.args.scalars[0] = *scale;
                let rank = q_shape.len();
                s.args.integers[8] = q_shape[rank - 2] as u64;
                let scratch_dtype = if state_layer.is_some() {
                    DType::F32
                } else if *dtype == DType::F64 {
                    DType::F64
                } else {
                    DType::F32
                };
                let heads = if rank >= 3 { q_shape[rank - 3] } else { 1 };
                let batch = product(&q_shape[..rank.saturating_sub(3)])?;
                let per = product(&[heads, q_shape[rank - 1], v_shape[rank - 1]])?;
                let count = product(&[batch, per])?;
                let history = if output.is_some() {
                    q_shape[rank - 2]
                        .checked_add(1)
                        .ok_or("KDA history overflow")?
                } else {
                    1
                };
                s.scratch = [
                    Some((product(&[count, history])?, scratch_dtype)),
                    None,
                    Some((batch, DType::U32)),
                ];
                if output.is_some() {
                    s.scratch[1] = Some((product(&[count, 2])?, scratch_dtype));
                }
                s.state = StateAccess::Kda {
                    layer: *state_layer,
                    batch,
                    elements_per_sequence: per,
                };
                s
            }
            Self::LastTokenRow {
                a, lane, tokens, ..
            } => {
                let mut s = KernelSpec::new("et_last_token", &[Some(*a)]);
                s.args.integers[0] = *tokens as u64;
                s.state = StateAccess::LastToken { lane: *lane };
                s
            }
            Self::KvAttention {
                q,
                k,
                v,
                q_shape,
                k_shape,
                scale,
                layer,
                window,
                bidirectional,
                rounding,
                ..
            } => {
                let mut s = KernelSpec::new("et_kv_attention", &[Some(*q), Some(*k), Some(*v)]);
                s.args.scalars[0] = *scale;
                s.args.integers[3] = window.unwrap_or(0) as u64;
                s.args.integers[4] = u64::from(*bidirectional);
                s.args.integers[6] = u64::from(*rounding == AttentionRounding::Stepwise);
                s.args.integers[7] = q_shape[2] as u64;
                s.args.integers[10] = q_shape[3] as u64;
                s.args.integers[11] = q_shape[1] as u64;
                s.state = StateAccess::Kv {
                    layer: *layer,
                    heads: k_shape[1],
                    dim: k_shape[3],
                    batch: q_shape[0],
                };
                s.scratch[0] = Some((q_shape[0], DType::U32));
                s.scratch[1] = Some((q_shape[0], DType::U32));
                s
            }
            Self::Random {
                normal,
                lo,
                hi,
                provenance,
                ..
            } => {
                let mut s = KernelSpec::new("et_random", &[]);
                s.args.integers[..2].copy_from_slice(&[*provenance, u64::from(*normal)]);
                s.args.scalars[..2].copy_from_slice(&[*lo, *hi]);
                s
            }
            Self::Sequence {
                eye, start, step, ..
            } => {
                let mut s = KernelSpec::new("et_sequence", &[]);
                s.args.integers[0] = u64::from(*eye);
                s.args.scalars[..2].copy_from_slice(&[*start, *step]);
                s
            }
            Self::Value(_) | Self::Input { .. } | Self::StateCursor { .. } | Self::Alias { .. } => {
                return Err("compile: binding or alias is not a CUDA kernel".into())
            }
        };
        Ok(spec)
    }
}

#[derive(Clone)]
pub(super) enum Instruction {
    Value(CudaValue),
    Input {
        binding: usize,
        scalar: bool,
    },
    StateCursor {
        tensor: bool,
    },
    Binary {
        op: u32,
        a: usize,
        b: usize,
    },
    Unary {
        op: u32,
        a: usize,
        parameter: f64,
    },
    Alias {
        a: usize,
    },
    Reindex {
        op: u32,
        a: usize,
        parameters: Vec<u64>,
    },
    Where {
        cond: usize,
        a: usize,
        b: usize,
    },
    Concat {
        a: usize,
        b: usize,
        dim: u32,
    },
    Reduce {
        op: u32,
        a: usize,
        dims: Vec<usize>,
        count: usize,
        trailing: bool,
    },
    Matmul {
        a: usize,
        b: usize,
    },
    TopKIndices {
        a: usize,
        k: usize,
        width: usize,
    },
    Index {
        op: u32,
        a: usize,
        indexes: Option<usize>,
        src: Option<usize>,
        dim: u32,
        width: usize,
        trailing: bool,
    },
    RmsNorm {
        x: usize,
        weight: Option<usize>,
        eps: f64,
    },
    CrossEntropy {
        logits: usize,
        target: usize,
        ignore_index: i64,
        backward: bool,
    },
    ChunkedHeadCe {
        output: Option<u32>,
        x: usize,
        weight: usize,
        bias: usize,
        target: usize,
        gradient: Option<usize>,
        ignore_index: i64,
    },
    Conv {
        op: u32,
        x: usize,
        w: usize,
        stride: u32,
        padding: u32,
        dilation: u32,
        groups: u32,
    },
    Linalg {
        op: u32,
        a: usize,
        b: Option<usize>,
        shape: Vec<usize>,
        a_shape: Vec<usize>,
        dtype: DType,
    },
    Linear {
        x: usize,
        weight: usize,
        bias: usize,
        k_width: u32,
        n_width: u32,
    },
    ExpertLinearRows {
        x: usize,
        weight: usize,
        indexes: usize,
        rows: usize,
        columns: usize,
        inner: usize,
        experts: usize,
    },
    GroupedExpertLinearRows {
        x: usize,
        weight: usize,
        indexes: usize,
    },
    QuantizedLinear {
        x: usize,
        weight: usize,
        bias: Option<usize>,
        rows: u32,
        columns: u32,
        row_bytes: u32,
        codec: GgmlKQuant,
    },
    QuantizedEmbedding {
        indexes: usize,
        weight: usize,
        rows: u32,
        columns: u32,
        row_bytes: u32,
        codec: GgmlKQuant,
        padding_index: Option<usize>,
    },
    LayerNorm {
        op: u32,
        x: usize,
        weight: usize,
        other: usize,
        width: u32,
        rows: u32,
        eps: f64,
    },
    Sdpa {
        op: u32,
        q: usize,
        k: usize,
        v: usize,
        g: Option<usize>,
        scale: f64,
        causal: bool,
        window: Option<usize>,
    },
    Rotary {
        x: usize,
        shape: Vec<usize>,
        seq_len: u32,
        theta: f64,
        interleaved: bool,
        backward: bool,
        cursor: bool,
    },
    Optimizer {
        kind: u32,
        output: u32,
        param: usize,
        grad: usize,
        state1: usize,
        state2: usize,
        lr: usize,
        c1: usize,
        c2: usize,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        dampening: f64,
        nesterov: bool,
    },
    ShortConv {
        op: u32,
        state_layer: Option<usize>,
        x: usize,
        weight: usize,
        g: Option<usize>,
        input_shape: Vec<usize>,
        weight_shape: Vec<usize>,
    },
    Kda {
        output: Option<u32>,
        state_layer: Option<usize>,
        q: usize,
        k: usize,
        v: usize,
        decay: usize,
        beta: usize,
        g: Option<usize>,
        q_shape: Vec<usize>,
        v_shape: Vec<usize>,
        scale: f64,
        dtype: DType,
    },
    LastTokenRow {
        a: usize,
        lane: usize,
        tokens: usize,
    },
    KvAttention {
        q: usize,
        k: usize,
        v: usize,
        q_shape: Vec<usize>,
        k_shape: Vec<usize>,
        scale: f64,
        layer: usize,
        window: Option<usize>,
        bidirectional: bool,
        rounding: AttentionRounding,
    },
    Random {
        normal: bool,
        lo: f64,
        hi: f64,
        provenance: u64,
    },
    Sequence {
        eye: bool,
        start: f64,
        step: f64,
    },
}

pub(super) fn permutation_preserves_storage(shape: &[usize], dims: &[usize]) -> bool {
    if shape.len() != dims.len() {
        return false;
    }
    let mut seen = vec![false; dims.len()];
    for dim in dims {
        let Some(slot) = seen.get_mut(*dim) else {
            return false;
        };
        if *slot {
            return false;
        }
        *slot = true;
    }
    dims.iter()
        .copied()
        .filter(|dim| shape[*dim] != 1)
        .eq((0..shape.len()).filter(|dim| shape[*dim] != 1))
}

fn child_index(index: &GraphIndex, node: &Arc<Node>) -> Result<usize, String> {
    index
        .dense_id(node.id)
        .map(|id| id.index())
        .ok_or_else(|| "compile: CUDA dependency is missing".into())
}
fn checked_len(len: usize) -> Result<u32, String> {
    u32::try_from(len).map_err(|_| "compile: CUDA dimension exceeds u32".into())
}
fn semantic_instruction(
    node: &Node,
    index: &GraphIndex,
    device: &Arc<CudaDevice>,
    ordinal: u32,
    state_cursor: Option<(u32, bool)>,
) -> Result<Instruction, String> {
    Ok(match &node.kind {
        NodeKind::SdpaConfigured { .. } | NodeKind::RotaryEmbeddingExplicit { .. } => {
            return Err("compile: semantic operation escaped native preparation".into());
        }
        NodeKind::Leaf(slot) => {
            let value = slot
                .get::<CudaValue>()
                .map_err(|error| format!("compile: {error}"))?;
            if value.ordinal() != ordinal {
                return Err(format!(
                    "compile: concrete leaf is on CUDA device {}, expected {ordinal}",
                    value.ordinal()
                ));
            }
            Instruction::Value(value)
        }
        NodeKind::Input { slot, dtype, .. } => {
            if state_cursor.is_some_and(|(cursor_slot, _)| cursor_slot == *slot) {
                let tensor = state_cursor.is_some_and(|(_, tensor)| tensor);
                if !tensor || *dtype != DType::I64 {
                    return Err("compile: state cursor tensor has an invalid signature".to_string());
                }
                Instruction::StateCursor { tensor }
            } else {
                let binding = index.slots[..*slot as usize]
                    .iter()
                    .filter(|declaration| !declaration.scalar)
                    .count();
                Instruction::Input {
                    binding,
                    scalar: false,
                }
            }
        }
        NodeKind::ScalarInput { slot, dtype, .. } => {
            if state_cursor.is_some_and(|(cursor_slot, _)| cursor_slot == *slot) {
                let tensor = state_cursor.is_some_and(|(_, tensor)| tensor);
                if tensor || *dtype != DType::I64 {
                    return Err("compile: state cursor scalar has an invalid signature".to_string());
                }
                Instruction::StateCursor { tensor }
            } else {
                let binding = index.slots[..*slot as usize]
                    .iter()
                    .filter(|declaration| declaration.scalar)
                    .count();
                Instruction::Input {
                    binding,
                    scalar: true,
                }
            }
        }
        NodeKind::FromBytes {
            data, shape, dtype, ..
        } => Instruction::Value(CudaValue::from_dense_bytes(
            device.clone(),
            shape.clone(),
            *dtype,
            data,
        )?),
        NodeKind::Zeros { shape, dtype, .. } => {
            Instruction::Value(full_value(device.clone(), shape.clone(), *dtype, 0.0)?)
        }
        NodeKind::Ones { shape, dtype, .. } => {
            Instruction::Value(full_value(device.clone(), shape.clone(), *dtype, 1.0)?)
        }
        NodeKind::Full {
            shape,
            value,
            dtype,
            ..
        } => Instruction::Value(full_value(device.clone(), shape.clone(), *dtype, *value)?),
        NodeKind::Randn { .. } => Instruction::Random {
            normal: true,
            lo: 0.0,
            hi: 1.0,
            provenance: node.id,
        },
        NodeKind::Uniform { lo, hi, .. } => Instruction::Random {
            normal: false,
            lo: *lo,
            hi: *hi,
            provenance: node.id,
        },
        NodeKind::Arange { start, step, .. } => Instruction::Sequence {
            eye: false,
            start: *start,
            step: *step,
        },
        NodeKind::Eye { .. } => Instruction::Sequence {
            eye: true,
            start: 0.0,
            step: 0.0,
        },
        NodeKind::Add { a, b }
        | NodeKind::Sub { a, b }
        | NodeKind::Mul { a, b }
        | NodeKind::Div { a, b }
        | NodeKind::Maximum { a, b }
        | NodeKind::Minimum { a, b }
        | NodeKind::Eq { a, b }
        | NodeKind::Gt { a, b }
        | NodeKind::Lt { a, b }
        | NodeKind::Ge { a, b }
        | NodeKind::Le { a, b } => {
            let op = match &node.kind {
                NodeKind::Add { .. } => 0,
                NodeKind::Sub { .. } => 1,
                NodeKind::Mul { .. } => 2,
                NodeKind::Div { .. } => 3,
                NodeKind::Maximum { .. } => 4,
                NodeKind::Minimum { .. } => 5,
                NodeKind::Eq { .. } => 6,
                NodeKind::Gt { .. } => 7,
                NodeKind::Lt { .. } => 8,
                NodeKind::Ge { .. } => 9,
                NodeKind::Le { .. } => 10,
                _ => unreachable!(),
            };
            Instruction::Binary {
                op,
                a: child_index(&index, a)?,
                b: child_index(&index, b)?,
            }
        }
        NodeKind::Neg { a }
        | NodeKind::Abs { a }
        | NodeKind::Sqrt { a }
        | NodeKind::Exp { a }
        | NodeKind::Log { a }
        | NodeKind::Sin { a }
        | NodeKind::Cos { a }
        | NodeKind::Tanh { a }
        | NodeKind::Relu { a }
        | NodeKind::Erf { a }
        | NodeKind::Floor { a }
        | NodeKind::Ceil { a }
        | NodeKind::Round { a }
        | NodeKind::Sign { a }
        | NodeKind::Pow { a, .. }
        | NodeKind::Gelu { a, .. }
        | NodeKind::Cast { a, .. } => {
            let (op, parameter) = match &node.kind {
                NodeKind::Neg { .. } => (0, 0.0),
                NodeKind::Abs { .. } => (1, 0.0),
                NodeKind::Sqrt { .. } => (2, 0.0),
                NodeKind::Exp { .. } => (3, 0.0),
                NodeKind::Log { .. } => (4, 0.0),
                NodeKind::Sin { .. } => (5, 0.0),
                NodeKind::Cos { .. } => (6, 0.0),
                NodeKind::Tanh { .. } => (7, 0.0),
                NodeKind::Relu { .. } => (8, 0.0),
                NodeKind::Erf { .. } => (9, 0.0),
                NodeKind::Floor { .. } => (10, 0.0),
                NodeKind::Ceil { .. } => (11, 0.0),
                NodeKind::Round { .. } => (12, 0.0),
                NodeKind::Sign { .. } => (13, 0.0),
                NodeKind::Pow { exp, .. } => (14, *exp),
                NodeKind::Gelu { approximate, .. } => (if *approximate { 16 } else { 15 }, 0.0),
                NodeKind::Cast { .. } => (17, 0.0),
                _ => unreachable!(),
            };
            Instruction::Unary {
                op,
                a: child_index(&index, a)?,
                parameter,
            }
        }
        NodeKind::Reshape { a, .. }
        | NodeKind::StopGradient { a }
        | NodeKind::Checkpoint { a }
        | NodeKind::Expose { a, .. } => Instruction::Alias {
            a: child_index(&index, a)?,
        },
        NodeKind::BroadcastTo { a, .. } => Instruction::Reindex {
            op: 0,
            a: child_index(&index, a)?,
            parameters: Vec::new(),
        },
        NodeKind::Permute { a, dims } if permutation_preserves_storage(&a.shape, dims) => {
            Instruction::Alias {
                a: child_index(&index, a)?,
            }
        }
        NodeKind::Permute { a, dims } => Instruction::Reindex {
            op: 1,
            a: child_index(&index, a)?,
            parameters: dims.iter().map(|dim| *dim as u64).collect(),
        },
        NodeKind::Slice { a, ranges } => Instruction::Reindex {
            op: 2,
            a: child_index(&index, a)?,
            parameters: ranges
                .iter()
                .flat_map(|(start, _, stride)| [*start as u64, *stride as u64])
                .collect(),
        },
        NodeKind::Where { cond, a, b } => Instruction::Where {
            cond: child_index(&index, cond)?,
            a: child_index(&index, a)?,
            b: child_index(&index, b)?,
        },
        NodeKind::Concat { a, b, dim } => Instruction::Concat {
            a: child_index(&index, a)?,
            b: child_index(&index, b)?,
            dim: u32::try_from(*dim).map_err(|_| "CUDA concat dimension exceeds u32")?,
        },
        NodeKind::Sum { a, dims, .. }
        | NodeKind::Prod { a, dims, .. }
        | NodeKind::Mean { a, dims, .. }
        | NodeKind::Max { a, dims, .. }
        | NodeKind::Min { a, dims, .. } => {
            let op = match &node.kind {
                NodeKind::Sum { .. } => 0,
                NodeKind::Prod { .. } => 1,
                NodeKind::Max { .. } => 2,
                NodeKind::Min { .. } => 3,
                NodeKind::Mean { .. } => 4,
                _ => unreachable!(),
            };
            Instruction::Reduce {
                op,
                a: child_index(&index, a)?,
                dims: dims.clone(),
                count: dims.iter().try_fold(1_usize, |count, &dim| {
                    count
                        .checked_mul(a.shape[dim])
                        .ok_or("CUDA reduction size overflow")
                })?,
                trailing: dims.len() == 1 && dims[0] + 1 == a.shape.len(),
            }
        }
        NodeKind::Matmul { a, b } => Instruction::Matmul {
            a: child_index(&index, a)?,
            b: child_index(&index, b)?,
        },
        NodeKind::TopKIndices { a, k } => Instruction::TopKIndices {
            a: child_index(&index, a)?,
            k: *k,
            width: *a.shape.last().ok_or("topKIndices: missing last axis")?,
        },
        NodeKind::Argmax { a, dim } | NodeKind::Argmin { a, dim } => Instruction::Index {
            op: u32::from(matches!(node.kind, NodeKind::Argmin { .. })),
            a: child_index(&index, a)?,
            indexes: None,
            src: None,
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
            width: a.shape[*dim],
            trailing: *dim + 1 == a.shape.len(),
        },
        NodeKind::Cumsum { a, dim } => Instruction::Index {
            op: 2,
            a: child_index(&index, a)?,
            indexes: None,
            src: None,
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
            width: a.shape[*dim],
            trailing: *dim + 1 == a.shape.len(),
        },
        NodeKind::IndexSelect { a, dim, indexes } => Instruction::Index {
            op: 3,
            a: child_index(&index, a)?,
            indexes: Some(child_index(&index, indexes)?),
            src: None,
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
            width: a.shape[*dim],
            trailing: *dim + 1 == a.shape.len(),
        },
        NodeKind::Gather { a, dim, indexes } => Instruction::Index {
            op: 4,
            a: child_index(&index, a)?,
            indexes: Some(child_index(&index, indexes)?),
            src: None,
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
            width: a.shape[*dim],
            trailing: *dim + 1 == a.shape.len(),
        },
        NodeKind::ScatterAdd {
            a,
            dim,
            indexes,
            src,
        } => Instruction::Index {
            op: 5,
            a: child_index(&index, a)?,
            indexes: Some(child_index(&index, indexes)?),
            src: Some(child_index(&index, src)?),
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
            width: a.shape[*dim],
            trailing: *dim + 1 == a.shape.len(),
        },
        NodeKind::RmsNorm { x, weight, eps } => Instruction::RmsNorm {
            x: child_index(&index, x)?,
            weight: weight
                .as_ref()
                .map(|weight| child_index(&index, weight))
                .transpose()?,
            eps: *eps,
        },
        NodeKind::CrossEntropy {
            logits,
            target,
            ignore_index,
            ..
        } => Instruction::CrossEntropy {
            logits: child_index(&index, logits)?,
            target: child_index(&index, target)?,
            ignore_index: *ignore_index,
            backward: false,
        },
        NodeKind::CrossEntropyBackward {
            logits,
            target,
            ignore_index,
            ..
        } => Instruction::CrossEntropy {
            logits: child_index(&index, logits)?,
            target: child_index(&index, target)?,
            ignore_index: *ignore_index,
            backward: true,
        },
        NodeKind::ChunkedHeadCe {
            x,
            weight,
            bias,
            target,
            ignore_index,
        } => Instruction::ChunkedHeadCe {
            output: None,
            x: child_index(&index, x)?,
            weight: child_index(&index, weight)?,
            bias: child_index(&index, bias)?,
            target: child_index(&index, target)?,
            gradient: None,
            ignore_index: *ignore_index,
        },
        NodeKind::ChunkedHeadCeBackward {
            x,
            weight,
            bias,
            target,
            g,
            ignore_index,
        } => Instruction::ChunkedHeadCe {
            output: Some(0),
            x: child_index(&index, x)?,
            weight: child_index(&index, weight)?,
            bias: child_index(&index, bias)?,
            target: child_index(&index, target)?,
            gradient: Some(child_index(&index, g)?),
            ignore_index: *ignore_index,
        },
        NodeKind::ChunkedHeadCeBackwardOut { of, index: output } => {
            if *output == 0 {
                Instruction::Alias {
                    a: child_index(&index, of)?,
                }
            } else {
                let NodeKind::ChunkedHeadCeBackward {
                    x,
                    weight,
                    bias,
                    target,
                    g,
                    ignore_index,
                } = &of.kind
                else {
                    return Err("compile: invalid chunked-head backward output".to_string());
                };
                Instruction::ChunkedHeadCe {
                    output: Some(u32::from(*output)),
                    x: child_index(&index, x)?,
                    weight: child_index(&index, weight)?,
                    bias: child_index(&index, bias)?,
                    target: child_index(&index, target)?,
                    gradient: Some(child_index(&index, g)?),
                    ignore_index: *ignore_index,
                }
            }
        }
        NodeKind::PositionEmbedding { weight, .. } => Instruction::Reindex {
            op: 2,
            a: child_index(&index, weight)?,
            parameters: vec![0, 1, 0, 1],
        },
        NodeKind::ExpertLinearRows { x, weight, indexes } => Instruction::ExpertLinearRows {
            x: child_index(&index, x)?,
            weight: child_index(&index, weight)?,
            indexes: child_index(&index, indexes)?,
            rows: x.shape[0],
            columns: weight.shape[1],
            inner: x.shape[1],
            experts: weight.shape[0],
        },
        NodeKind::GroupedExpertLinearRows { x, weight, indexes } => {
            Instruction::GroupedExpertLinearRows {
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                indexes: child_index(&index, indexes)?,
            }
        }
        NodeKind::Linear { x, weight, bias } => Instruction::Linear {
            x: child_index(&index, x)?,
            weight: child_index(&index, weight)?,
            bias: child_index(&index, bias)?,
            k_width: u32::try_from(weight.shape[0])
                .map_err(|_| "CUDA linear input width exceeds u32")?,
            n_width: u32::try_from(weight.shape[1])
                .map_err(|_| "CUDA linear output width exceeds u32")?,
        },
        NodeKind::QuantizedLinear { x, weight, bias } => {
            let (codec, rows, columns, row_bytes) = packed_geometry(weight)?;
            Instruction::QuantizedLinear {
                x: child_index(index, x)?,
                weight: child_index(index, weight)?,
                bias: bias
                    .as_ref()
                    .map(|bias| child_index(index, bias))
                    .transpose()?,
                rows,
                columns,
                row_bytes,
                codec,
            }
        }
        NodeKind::QuantizedEmbedding {
            indexes,
            weight,
            padding_index,
        } => {
            let (codec, rows, columns, row_bytes) = packed_geometry(weight)?;
            Instruction::QuantizedEmbedding {
                indexes: child_index(index, indexes)?,
                weight: child_index(index, weight)?,
                rows,
                columns,
                row_bytes,
                codec,
                padding_index: *padding_index,
            }
        }
        NodeKind::LayerNorm {
            x,
            weight,
            bias,
            eps,
        } => {
            let width = element_count(&weight.shape)?;
            Instruction::LayerNorm {
                op: 0,
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                other: child_index(&index, bias)?,
                width: checked_len(width)?,
                rows: checked_len(element_count(&x.shape)? / width)?,
                eps: *eps,
            }
        }
        NodeKind::LayerNormBackward { x, weight, g, eps } => {
            let width = element_count(&weight.shape)?;
            Instruction::LayerNorm {
                op: 1,
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                other: child_index(&index, g)?,
                width: checked_len(width)?,
                rows: checked_len(element_count(&x.shape)? / width)?,
                eps: *eps,
            }
        }
        NodeKind::LayerNormBackwardOut { of, index: output } => {
            let NodeKind::LayerNormBackward { x, weight, g, eps } = &of.kind else {
                return Err("compile: invalid layer norm backward output".to_string());
            };
            let width = element_count(&weight.shape)?;
            Instruction::LayerNorm {
                op: match output {
                    1 => 2,
                    2 => 3,
                    _ => {
                        return Err("compile: invalid layer norm backward output index".to_string())
                    }
                },
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                other: child_index(&index, g)?,
                width: checked_len(width)?,
                rows: checked_len(element_count(&x.shape)? / width)?,
                eps: *eps,
            }
        }
        NodeKind::Sdpa {
            q,
            k,
            v,
            scale,
            causal,
            window,
        } => Instruction::Sdpa {
            op: 3,
            q: child_index(&index, q)?,
            k: child_index(&index, k)?,
            v: child_index(&index, v)?,
            g: None,
            scale: *scale,
            causal: *causal,
            window: window.local(),
        },
        NodeKind::SdpaBackward {
            q,
            k,
            v,
            g,
            scale,
            causal,
            window,
            ..
        } => Instruction::Sdpa {
            op: 0,
            q: child_index(&index, q)?,
            k: child_index(&index, k)?,
            v: child_index(&index, v)?,
            g: Some(child_index(&index, g)?),
            scale: *scale,
            causal: *causal,
            window: window.local(),
        },
        NodeKind::SdpaBackwardOut { of, index: output } => {
            let NodeKind::SdpaBackward {
                q,
                k,
                v,
                g,
                scale,
                causal,
                window,
                ..
            } = &of.kind
            else {
                return Err("compile: invalid attention backward output".to_string());
            };
            Instruction::Sdpa {
                op: u32::from(*output),
                q: child_index(&index, q)?,
                k: child_index(&index, k)?,
                v: child_index(&index, v)?,
                g: Some(child_index(&index, g)?),
                scale: *scale,
                causal: *causal,
                window: window.local(),
            }
        }
        NodeKind::RotaryEmbedding {
            x,
            seq_len,
            theta,
            offset,
            layout,
        } => Instruction::Rotary {
            x: child_index(&index, x)?,
            shape: node.shape.clone(),
            seq_len: u32::try_from(*seq_len).map_err(|_| "CUDA rotary sequence exceeds u32")?,
            theta: *theta,
            interleaved: matches!(layout, effect_torch_graph::RotaryLayout::InterleavedPairs),
            backward: false,
            cursor: *offset == PositionOffset::Cursor,
        },
        NodeKind::RotaryEmbeddingBackward {
            g,
            seq_len,
            theta,
            layout,
            ..
        } => Instruction::Rotary {
            x: child_index(&index, g)?,
            shape: node.shape.clone(),
            seq_len: u32::try_from(*seq_len).map_err(|_| "CUDA rotary sequence exceeds u32")?,
            theta: *theta,
            interleaved: matches!(layout, effect_torch_graph::RotaryLayout::InterleavedPairs),
            backward: true,
            cursor: false,
        },
        NodeKind::Conv1d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => Instruction::Conv {
            op: 0,
            x: child_index(&index, x)?,
            w: child_index(&index, w)?,
            stride: u32::try_from(*stride).map_err(|_| "CUDA convolution stride exceeds u32")?,
            padding: u32::try_from(*padding).map_err(|_| "CUDA convolution padding exceeds u32")?,
            dilation: u32::try_from(*dilation)
                .map_err(|_| "CUDA convolution dilation exceeds u32")?,
            groups: u32::try_from(*groups).map_err(|_| "CUDA convolution groups exceeds u32")?,
        },
        NodeKind::Conv2d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => Instruction::Conv {
            op: 1,
            x: child_index(&index, x)?,
            w: child_index(&index, w)?,
            stride: u32::try_from(*stride).map_err(|_| "CUDA convolution stride exceeds u32")?,
            padding: u32::try_from(*padding).map_err(|_| "CUDA convolution padding exceeds u32")?,
            dilation: u32::try_from(*dilation)
                .map_err(|_| "CUDA convolution dilation exceeds u32")?,
            groups: u32::try_from(*groups).map_err(|_| "CUDA convolution groups exceeds u32")?,
        },
        NodeKind::ConvTranspose1d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
            ..
        } => Instruction::Conv {
            op: 2,
            x: child_index(&index, x)?,
            w: child_index(&index, w)?,
            stride: u32::try_from(*stride).map_err(|_| "CUDA convolution stride exceeds u32")?,
            padding: u32::try_from(*padding).map_err(|_| "CUDA convolution padding exceeds u32")?,
            dilation: u32::try_from(*dilation)
                .map_err(|_| "CUDA convolution dilation exceeds u32")?,
            groups: u32::try_from(*groups).map_err(|_| "CUDA convolution groups exceeds u32")?,
        },
        NodeKind::ConvTranspose2d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
            ..
        } => Instruction::Conv {
            op: 3,
            x: child_index(&index, x)?,
            w: child_index(&index, w)?,
            stride: u32::try_from(*stride).map_err(|_| "CUDA convolution stride exceeds u32")?,
            padding: u32::try_from(*padding).map_err(|_| "CUDA convolution padding exceeds u32")?,
            dilation: u32::try_from(*dilation)
                .map_err(|_| "CUDA convolution dilation exceeds u32")?,
            groups: u32::try_from(*groups).map_err(|_| "CUDA convolution groups exceeds u32")?,
        },
        NodeKind::Conv1dBackwardW {
            x,
            g,
            stride,
            padding,
            dilation,
            groups,
            ..
        } => Instruction::Conv {
            op: 4,
            x: child_index(&index, x)?,
            w: child_index(&index, g)?,
            stride: u32::try_from(*stride).map_err(|_| "CUDA convolution stride exceeds u32")?,
            padding: u32::try_from(*padding).map_err(|_| "CUDA convolution padding exceeds u32")?,
            dilation: u32::try_from(*dilation)
                .map_err(|_| "CUDA convolution dilation exceeds u32")?,
            groups: u32::try_from(*groups).map_err(|_| "CUDA convolution groups exceeds u32")?,
        },
        NodeKind::Conv2dBackwardW {
            x,
            g,
            stride,
            padding,
            dilation,
            groups,
            ..
        } => Instruction::Conv {
            op: 5,
            x: child_index(&index, x)?,
            w: child_index(&index, g)?,
            stride: u32::try_from(*stride).map_err(|_| "CUDA convolution stride exceeds u32")?,
            padding: u32::try_from(*padding).map_err(|_| "CUDA convolution padding exceeds u32")?,
            dilation: u32::try_from(*dilation)
                .map_err(|_| "CUDA convolution dilation exceeds u32")?,
            groups: u32::try_from(*groups).map_err(|_| "CUDA convolution groups exceeds u32")?,
        },
        NodeKind::AdamWStep {
            param,
            grad,
            m,
            v,
            lr,
            c1,
            c2,
            beta1,
            beta2,
            eps,
            weight_decay,
        } => Instruction::Optimizer {
            kind: 0,
            output: 0,
            param: child_index(&index, param)?,
            grad: child_index(&index, grad)?,
            state1: child_index(&index, m)?,
            state2: child_index(&index, v)?,
            lr: child_index(&index, lr)?,
            c1: child_index(&index, c1)?,
            c2: child_index(&index, c2)?,
            beta1: *beta1,
            beta2: *beta2,
            eps: *eps,
            weight_decay: *weight_decay,
            dampening: 0.0,
            nesterov: false,
        },
        NodeKind::AdamWOut {
            step,
            index: output,
        } => {
            let NodeKind::AdamWStep {
                param,
                grad,
                m,
                v,
                lr,
                c1,
                c2,
                beta1,
                beta2,
                eps,
                weight_decay,
            } = &step.kind
            else {
                return Err("compile: invalid AdamW output selector".to_string());
            };
            Instruction::Optimizer {
                kind: 0,
                output: u32::from(*output),
                param: child_index(&index, param)?,
                grad: child_index(&index, grad)?,
                state1: child_index(&index, m)?,
                state2: child_index(&index, v)?,
                lr: child_index(&index, lr)?,
                c1: child_index(&index, c1)?,
                c2: child_index(&index, c2)?,
                beta1: *beta1,
                beta2: *beta2,
                eps: *eps,
                weight_decay: *weight_decay,
                dampening: 0.0,
                nesterov: false,
            }
        }
        NodeKind::SgdStep {
            param,
            grad,
            velocity,
            first,
            lr,
            momentum,
            dampening,
            nesterov,
            weight_decay,
        } => Instruction::Optimizer {
            kind: 1,
            output: 0,
            param: child_index(&index, param)?,
            grad: child_index(&index, grad)?,
            state1: child_index(&index, velocity)?,
            state2: child_index(&index, first)?,
            lr: child_index(&index, lr)?,
            c1: child_index(&index, lr)?,
            c2: child_index(&index, lr)?,
            beta1: *momentum,
            beta2: 0.0,
            eps: 0.0,
            weight_decay: *weight_decay,
            dampening: *dampening,
            nesterov: *nesterov,
        },
        NodeKind::SgdOut {
            step,
            index: output,
        } => {
            let NodeKind::SgdStep {
                param,
                grad,
                velocity,
                first,
                lr,
                momentum,
                dampening,
                nesterov,
                weight_decay,
            } = &step.kind
            else {
                return Err("compile: invalid SGD output selector".to_string());
            };
            Instruction::Optimizer {
                kind: 1,
                output: u32::from(*output),
                param: child_index(&index, param)?,
                grad: child_index(&index, grad)?,
                state1: child_index(&index, velocity)?,
                state2: child_index(&index, first)?,
                lr: child_index(&index, lr)?,
                c1: child_index(&index, lr)?,
                c2: child_index(&index, lr)?,
                beta1: *momentum,
                beta2: 0.0,
                eps: 0.0,
                weight_decay: *weight_decay,
                dampening: *dampening,
                nesterov: *nesterov,
            }
        }
        NodeKind::ShortConv1d { x, weight } => Instruction::ShortConv {
            op: 0,
            state_layer: None,
            x: child_index(&index, x)?,
            weight: child_index(&index, weight)?,
            g: None,
            input_shape: x.shape.clone(),
            weight_shape: weight.shape.clone(),
        },
        NodeKind::ShortConv1dBackwardX { x, weight, g } => Instruction::ShortConv {
            op: 1,
            state_layer: None,
            x: child_index(&index, x)?,
            weight: child_index(&index, weight)?,
            g: Some(child_index(&index, g)?),
            input_shape: x.shape.clone(),
            weight_shape: weight.shape.clone(),
        },
        NodeKind::ShortConv1dBackwardW { x, weight, g } => Instruction::ShortConv {
            op: 2,
            state_layer: None,
            x: child_index(&index, x)?,
            weight: child_index(&index, weight)?,
            g: Some(child_index(&index, g)?),
            input_shape: x.shape.clone(),
            weight_shape: weight.shape.clone(),
        },
        NodeKind::KdaChunk {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
        } => Instruction::Kda {
            output: None,
            state_layer: None,
            q: child_index(&index, q)?,
            k: child_index(&index, k)?,
            v: child_index(&index, v)?,
            decay: child_index(&index, log_decay)?,
            beta: child_index(&index, beta)?,
            g: None,
            q_shape: q.shape.clone(),
            v_shape: v.shape.clone(),
            scale: *scale,
            dtype: node.dtype,
        },
        NodeKind::KdaBackward {
            q,
            k,
            v,
            log_decay,
            beta,
            g,
            scale,
        } => Instruction::Kda {
            output: Some(0),
            state_layer: None,
            q: child_index(&index, q)?,
            k: child_index(&index, k)?,
            v: child_index(&index, v)?,
            decay: child_index(&index, log_decay)?,
            beta: child_index(&index, beta)?,
            g: Some(child_index(&index, g)?),
            q_shape: q.shape.clone(),
            v_shape: v.shape.clone(),
            scale: *scale,
            dtype: node.dtype,
        },
        NodeKind::KdaBackwardOut { of, index: output } => {
            let NodeKind::KdaBackward {
                q,
                k,
                v,
                log_decay,
                beta,
                g,
                scale,
            } = &of.kind
            else {
                return Err("compile: invalid KDA backward output selector".to_string());
            };
            Instruction::Kda {
                output: Some(u32::from(*output)),
                state_layer: None,
                q: child_index(&index, q)?,
                k: child_index(&index, k)?,
                v: child_index(&index, v)?,
                decay: child_index(&index, log_decay)?,
                beta: child_index(&index, beta)?,
                g: Some(child_index(&index, g)?),
                q_shape: q.shape.clone(),
                v_shape: v.shape.clone(),
                scale: *scale,
                dtype: node.dtype,
            }
        }
        NodeKind::ConvState { x, weight, layer } => Instruction::ShortConv {
            op: 0,
            state_layer: Some(*layer as usize),
            x: child_index(&index, x)?,
            weight: child_index(&index, weight)?,
            g: None,
            input_shape: x.shape.clone(),
            weight_shape: weight.shape.clone(),
        },
        NodeKind::KdaRecurrence {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
            layer,
        } => Instruction::Kda {
            output: None,
            state_layer: Some(*layer as usize),
            q: child_index(&index, q)?,
            k: child_index(&index, k)?,
            v: child_index(&index, v)?,
            decay: child_index(&index, log_decay)?,
            beta: child_index(&index, beta)?,
            g: None,
            q_shape: q.shape.clone(),
            v_shape: v.shape.clone(),
            scale: *scale,
            dtype: node.dtype,
        },
        NodeKind::LastTokenRow { a } => {
            let (lane, source, input_shape) = match &a.kind {
                NodeKind::Slice { ranges, .. } => (
                    ranges.first().map_or(0, |range| range.0),
                    child_index(&index, a)?,
                    a.shape.clone(),
                ),
                _ => (0, child_index(&index, a)?, a.shape.clone()),
            };
            let tokens = input_shape
                .get(input_shape.len().saturating_sub(2))
                .copied()
                .ok_or_else(|| {
                    "compile: last-token input must have rank at least two".to_string()
                })?;
            Instruction::LastTokenRow {
                a: source,
                lane,
                tokens,
            }
        }
        NodeKind::KvAttention {
            q,
            k,
            v,
            scale,
            layer,
            window,
            mode,
            rounding,
        } => Instruction::KvAttention {
            q: child_index(&index, q)?,
            k: child_index(&index, k)?,
            v: child_index(&index, v)?,
            q_shape: q.shape.clone(),
            k_shape: k.shape.clone(),
            scale: *scale,
            layer: *layer as usize,
            window: *window,
            bidirectional: *mode == KvAttentionMode::BidirectionalBlock,
            rounding: *rounding,
        },
        NodeKind::Inverse { a } => Instruction::Linalg {
            op: 0,
            a: child_index(&index, a)?,
            b: None,
            shape: node.shape.clone(),
            a_shape: a.shape.clone(),
            dtype: node.dtype,
        },
        NodeKind::Det { a } => Instruction::Linalg {
            op: 1,
            a: child_index(&index, a)?,
            b: None,
            shape: node.shape.clone(),
            a_shape: a.shape.clone(),
            dtype: node.dtype,
        },
        NodeKind::Solve { a, b } => Instruction::Linalg {
            op: 2,
            a: child_index(&index, a)?,
            b: Some(child_index(&index, b)?),
            shape: node.shape.clone(),
            a_shape: a.shape.clone(),
            dtype: node.dtype,
        },
    })
}

fn packed_geometry(weight: &Node) -> Result<(GgmlKQuant, u32, u32, u32), String> {
    let effect_torch_runtime::StorageRepresentation::Packed(
        effect_torch_runtime::PackedFormat::GgmlKQuant(codec),
    ) = weight.storage.representation
    else {
        return Err("compile: quantized weight is not a validated packed value".into());
    };
    if weight.shape.len() != 2 {
        return Err("compile: packed weight must be a matrix".into());
    }
    let rows = checked_len(weight.shape[0])?;
    let columns = checked_len(weight.shape[1])?;
    let row_bytes = checked_len(
        codec
            .encoded_row_bytes(weight.shape[1])
            .ok_or("compile: invalid packed row geometry")?,
    )?;
    Ok((codec, rows, columns, row_bytes))
}

pub struct CudaExecutable {
    state_layout: Option<CudaStateLayout>,
    device: Arc<CudaDevice>,
    program: CudaLoweredProgram,
    memory: MemoryPlan<CudaMemorySpace>,
    commands: Vec<Command>,
    metadata: Vec<Option<CudaBuffer<u64>>>,
    outputs: Box<[(Vec<usize>, DType)]>,
    diagnostics: ExecutableDiagnostics,
    compiler_work: CompilerWorkReport,
    runs: AtomicU64,
    graph_runtime: Mutex<GraphRuntime>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GraphKey {
    start: usize,
    end: usize,
    pointers: Arc<[u64]>,
}

struct CapturedGraph(CudaGraph);

// SAFETY: CUDA graph access is serialized by `graph_runtime`; cudarc binds the
// graph's context to the calling thread before each launch and destruction.
unsafe impl Send for CapturedGraph {}

#[derive(Default)]
struct GraphRuntime {
    cache: HashMap<GraphKey, CapturedGraph>,
    hits: u64,
    captures: u64,
    seen: HashSet<GraphKey>,
}

/// Physical KV layout frozen before lowering and memory planning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaStateLayout {
    pub capacity: u32,
    pub dtype: DType,
    pub slots: u32,
    pub packed_rows_per_sequence: Option<u32>,
    pub kv_layers: Vec<KvLayerDescriptor>,
    pub access: StateAccessMode,
}
impl CudaStateLayout {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.capacity == 0
            || self.slots == 0
            || self.packed_rows_per_sequence == Some(0)
            || !matches!(
                self.dtype,
                DType::F32 | DType::F16 | DType::BF16 | DType::U8
            )
        {
            return Err("compile: invalid CUDA state layout".into());
        }
        let mut ids = std::collections::HashSet::new();
        for layer in &self.kv_layers {
            if !ids.insert(layer.layer_id)
                || layer.kv_heads == 0
                || layer.head_dim == 0
                || !matches!(
                    layer.dtype,
                    DType::F32 | DType::F16 | DType::BF16 | DType::U8
                )
                || layer.row_bytes().is_none()
            {
                return Err("compile: invalid CUDA KV layer descriptor".into());
            }
        }
        Ok(())
    }
}

/// Immutable device pages. Forks and snapshots retain pages without copying KV.
#[derive(Clone, Default)]
pub struct CudaKvSnapshot {
    pub(crate) layers: Vec<CudaKvLayer>,
}
#[derive(Clone)]
pub(crate) struct CudaKvLayer {
    pub(crate) descriptor: effect_torch_runtime::KvLayerDescriptor,
    pub(crate) start_position: u32,
    pub(crate) pages: Vec<Arc<CudaKvPage>>,
}
pub(crate) struct CudaKvPage {
    pub(crate) start: u32,
    pub(crate) count: u32,
    pub(crate) keys: CudaBuffer<u8>,
    pub(crate) values: CudaBuffer<u8>,
    pub(crate) key_scales: Option<CudaBuffer<f32>>,
    pub(crate) value_scales: Option<CudaBuffer<f32>>,
}
#[derive(Clone)]
pub struct CudaSequenceState {
    pub cursor: u32,
    pub keys: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
    pub kda_states: Vec<Vec<f32>>,
    pub conv_states: Vec<Vec<f32>>,
    pub kv_storage: Option<CudaKvSnapshot>,
}
#[derive(Clone)]
pub(crate) struct CudaKvCache {
    pub(crate) sequences: Vec<CudaKvSnapshot>,
}
pub struct CudaStateInvocation {
    pub sequences: Vec<CudaSequenceState>,
    pub slots: Vec<u32>,
    pub valid_lengths: Vec<u32>,
    pub capacity: u32,
    pub cache_dtype: DType,
    pub packed_rows_per_sequence: Option<u32>,
    pub kv_layers: Vec<KvLayerDescriptor>,
    pub access: StateAccessMode,
    pub(crate) cache: Option<CudaKvCache>,
}

struct InvocationFence {
    stream: Arc<cudarc::driver::CudaStream>,
    worker_events: Vec<cudarc::driver::CudaEvent>,
    complete: bool,
}
impl Drop for InvocationFence {
    fn drop(&mut self) {
        if !self.complete {
            let _ = self.stream.synchronize();
        }
    }
}

struct StreamCaptureFence {
    stream: Arc<cudarc::driver::CudaStream>,
    active: bool,
}
impl Drop for StreamCaptureFence {
    fn drop(&mut self) {
        if self.active {
            let _ = self.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
            );
        }
    }
}

impl CudaExecutable {
    pub fn ordinal(&self) -> u32 {
        self.device.ordinal
    }
    pub fn diagnostics(&self) -> &ExecutableDiagnostics {
        &self.diagnostics
    }
    pub fn compiler_work(&self) -> &CompilerWorkReport {
        &self.compiler_work
    }
    pub fn outputs(&self) -> &[(Vec<usize>, DType)] {
        &self.outputs
    }
    pub fn instruction_count(&self) -> usize {
        self.commands.len()
    }
    pub fn tensor_input(&self, binding: usize) -> Option<(Vec<usize>, DType)> {
        self.commands.iter().find_map(|command| match command.kind {
            CommandKind::Input { binding: found } if found == binding => {
                let meta = &self.program.values[command.output?.index()];
                Some((meta.shape.clone(), meta.dtype))
            }
            _ => None,
        })
    }
    pub fn execute(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        cancelled: &CancellationFlag,
    ) -> Result<Vec<CudaValue>, String> {
        self.execute_inner(
            bindings,
            scalars,
            None,
            cancelled,
            #[cfg(test)]
            None,
            #[cfg(test)]
            None,
        )
    }
    /// Test-only injection immediately after a successful cuBLAS submission.
    #[cfg(test)]
    pub(crate) fn execute_with_gemm_hook(
        &self,
        bindings: &[CudaValue],
        cancelled: &CancellationFlag,
        after_gemm: &dyn Fn(),
    ) -> Result<Vec<CudaValue>, String> {
        self.execute_inner(bindings, &[], None, cancelled, Some(after_gemm), None)
    }
    pub fn execute_stateful(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        state: &mut CudaStateInvocation,
        cancelled: &CancellationFlag,
    ) -> Result<Vec<CudaValue>, String> {
        self.execute_state_transaction(
            bindings,
            scalars,
            state,
            cancelled,
            #[cfg(test)]
            None,
        )
    }
    #[cfg(test)]
    pub(crate) fn execute_stateful_with_kv_hook(
        &self,
        bindings: &[CudaValue],
        state: &mut CudaStateInvocation,
        cancelled: &CancellationFlag,
        after_kv: &dyn Fn(),
    ) -> Result<Vec<CudaValue>, String> {
        self.execute_state_transaction(bindings, &[], state, cancelled, Some(after_kv))
    }
    fn execute_state_transaction(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        state: &mut CudaStateInvocation,
        cancelled: &CancellationFlag,
        #[cfg(test)] after_kv: Option<&dyn Fn()>,
    ) -> Result<Vec<CudaValue>, String> {
        if !scalars.is_empty() {
            return Err("execute: stateful scalar bindings are unsupported".into());
        }
        self.prepare_state(state)?;
        let before = state.sequences.clone();
        let cache_before = state.cache.clone();
        let result = self.execute_inner(
            bindings,
            scalars,
            Some(state),
            cancelled,
            #[cfg(test)]
            None,
            #[cfg(test)]
            after_kv,
        );
        if result.is_err() {
            state.sequences = before;
            state.cache = cache_before;
        }
        result
    }
    pub fn execute_stateful_graphed(
        &self,
        _bindings: &[CudaValue],
        _tokens: &[u32],
        _state: &mut CudaStateInvocation,
        _cancelled: &CancellationFlag,
    ) -> Result<Option<CudaValue>, String> {
        // In particular, GroupedExpert commands are non-capturable: their
        // exact cuBLAS dimensions depend on a host group-control completion.
        // Stepwise BF16 KV GEMMs also remain non-capturable: P and Q depend on
        // the invocation cursor, retained start and valid query length.
        Ok(None)
    }
    pub fn execute_stateful_greedy(
        &self,
        _bindings: &[CudaValue],
        _tokens: &[u32],
        _state: &mut CudaStateInvocation,
        _cancelled: &CancellationFlag,
    ) -> Result<Option<u32>, String> {
        Ok(None)
    }
    fn buffer(
        &self,
        resources: &InvocationResources,
        id: ValueId,
    ) -> Result<CudaBuffer<u8>, String> {
        resources.buffer(
            &self.memory.locations[id.index()],
            0,
            self.program.values[id.index()].decl.bytes,
        )
    }
    fn planned_value(
        &self,
        resources: &InvocationResources,
        id: ValueId,
    ) -> Result<CudaValue, String> {
        CudaValue::from_planned_buffer(
            self.device.clone(),
            self.program.values[id.index()].spec(),
            self.buffer(resources, id)?,
        )
    }
    fn graph_segment_end(&self, start: usize) -> Option<usize> {
        fn captures_work(command: &Command) -> bool {
            command.output.is_some()
                && match &command.kind {
                    CommandKind::Gemm { .. }
                    | CommandKind::LinearBias { .. }
                    | CommandKind::FusedElementwise { .. } => true,
                    CommandKind::Kernel {
                        name,
                        checked,
                        state,
                        kv_matmul,
                        ..
                    } => {
                        !checked
                            && matches!(state, StateAccess::None)
                            && kv_matmul.is_none()
                            && !name.starts_with("et_random_")
                    }
                    _ => false,
                }
        }
        fn bridges_work(command: &Command) -> bool {
            matches!(
                command.kind,
                CommandKind::Prepare
                    | CommandKind::Value(_)
                    | CommandKind::Input { .. }
                    | CommandKind::Alias { .. }
            )
        }
        if !self.commands.get(start).is_some_and(captures_work) {
            return None;
        }
        let mut cursor = start + 1;
        let mut end = cursor;
        let mut work = 1usize;
        while let Some(command) = self.commands.get(cursor) {
            if captures_work(command) {
                work += 1;
                end = cursor + 1;
            } else if !bridges_work(command) {
                break;
            }
            cursor += 1;
        }
        (work >= 2).then_some(end)
    }
    fn materialize_graph_values(
        &self,
        resources: &InvocationResources,
        bindings: &[CudaValue],
        values: &mut [Option<CudaValue>],
        start: usize,
        end: usize,
    ) -> Result<(), String> {
        for command in &self.commands[start..end] {
            let Some(output) = command.output else {
                continue;
            };
            let meta = &self.program.values[output.index()];
            let value = match &command.kind {
                CommandKind::Prepare => continue,
                CommandKind::Value(value) => value.clone(),
                CommandKind::Input { binding } => {
                    let value = bindings
                        .get(*binding)
                        .ok_or_else(|| format!("execute: missing CUDA binding {binding}"))?;
                    if value.ordinal() != self.device.ordinal
                        || value.shape() != meta.shape
                        || value.dtype() != meta.dtype
                        || value.spec().storage.representation != meta.storage.representation
                    {
                        return Err(format!(
                            "execute: CUDA binding {binding} violates its value specification"
                        ));
                    }
                    value.clone()
                }
                CommandKind::Alias { source } => values[source.index()]
                    .as_ref()
                    .ok_or("execute: captured alias source is unavailable")?
                    .reshape_dense(meta.shape.clone())?,
                _ => self.planned_value(resources, output)?,
            };
            values[output.index()] = Some(value);
        }
        Ok(())
    }
    fn graph_key(
        &self,
        resources: &InvocationResources,
        values: &[Option<CudaValue>],
        start: usize,
        end: usize,
    ) -> Result<GraphKey, String> {
        let value_address = |id: ValueId| {
            values[id.index()]
                .as_ref()
                .map(CudaValue::storage_address)
                .ok_or_else(|| {
                    format!(
                        "execute: captured CUDA value {id} unavailable in segment {start}..{end}"
                    )
                })
        };
        let mut pointers = Vec::new();
        for (offset, command) in self.commands[start..end].iter().enumerate() {
            if matches!(command.kind, CommandKind::Prepare) {
                continue;
            }
            let Some(output) = command.output else {
                continue;
            };
            pointers.push(value_address(output)?);
            match &command.kind {
                CommandKind::Gemm {
                    x,
                    weight,
                    workspace,
                    ..
                } => {
                    pointers.extend([
                        value_address(*x)?,
                        value_address(*weight)?,
                        self.buffer(resources, *workspace)?.address(),
                    ]);
                }
                CommandKind::LinearBias {
                    accumulator, bias, ..
                } => {
                    pointers.extend([value_address(*accumulator)?, value_address(*bias)?]);
                }
                CommandKind::FusedElementwise { inputs, .. } => {
                    for input in inputs.iter().flatten() {
                        pointers.push(value_address(*input)?);
                    }
                }
                CommandKind::Kernel {
                    inputs, scratch, ..
                } => {
                    for input in inputs.iter().flatten() {
                        pointers.push(value_address(*input)?);
                    }
                    for scratch in scratch.iter().flatten() {
                        pointers.push(self.buffer(resources, *scratch)?.address());
                    }
                    pointers.push(
                        self.metadata[start + offset]
                            .as_ref()
                            .ok_or("execute: captured kernel metadata missing")?
                            .address(),
                    );
                }
                CommandKind::Prepare
                | CommandKind::Value(_)
                | CommandKind::Input { .. }
                | CommandKind::Alias { .. } => {}
                _ => return Err("execute: non-capturable command in CUDA graph segment".into()),
            }
        }
        Ok(GraphKey {
            start,
            end,
            pointers: pointers.into(),
        })
    }
    fn status_result(&self, failure: u64) -> Result<(), String> {
        if failure == 0 {
            return Ok(());
        }
        let code = failure as u32;
        let position = (failure >> 32) as usize;
        let name = match self.commands.get(position).map(|command| &command.kind) {
            Some(CommandKind::Kernel { name, .. }) => *name,
            _ => return Err("execute: CUDA error context is invalid".into()),
        };
        Err(match code {
            1 => format!("{name}: index is out of range"),
            2 => format!("{name}: no active targets"),
            3 => format!("{name}: matrix is singular"),
            5 => "topKIndices: NaN input".into(),
            6 => "expertLinearRows: expert index is out of range".into(),
            _ => format!("{name}: invalid arithmetic"),
        })
    }
    fn execute_inner(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        mut state: Option<&mut CudaStateInvocation>,
        cancelled: &CancellationFlag,
        #[cfg(test)] after_gemm: Option<&dyn Fn()>,
        #[cfg(test)] after_kv: Option<&dyn Fn()>,
    ) -> Result<Vec<CudaValue>, String> {
        let resources = workspace::acquire(self.device.ordinal, &self.memory.segments)?;
        // Drop the fence before returning leases on errors or interruption.
        let mut fence = InvocationFence {
            stream: self.device.stream.clone(),
            worker_events: Vec::new(),
            complete: false,
        };
        let run = self.runs.fetch_add(1, Ordering::Relaxed);
        // Opt-in diagnosis of a bounded run. Synchronization below attributes
        // asynchronous GEMMs to their own command rather than the next kernel.
        let trace = std::env::var("EFFECT_TORCH_CUDA_TRACE").is_ok_and(|value| value == "1");
        let graph_trace =
            std::env::var("EFFECT_TORCH_CUDA_GRAPH_TRACE").is_ok_and(|value| value == "1");
        let graphs_requested =
            std::env::var("EFFECT_TORCH_CUDA_GRAPHS").is_ok_and(|value| value == "1");
        // Node may make stderr nonblocking. Use an explicit append-only file
        // rather than allowing diagnostic output to panic on EAGAIN.
        let mut trace_file = if trace {
            let path = std::env::var_os("EFFECT_TORCH_CUDA_TRACE_PATH")
                .ok_or("execute: CUDA trace requires EFFECT_TORCH_CUDA_TRACE_PATH")?;
            Some(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|error| format!("execute: CUDA trace file: {error}"))?,
            )
        } else {
            None
        };
        let mut emit_trace = |record: serde_json::Value| -> Result<(), String> {
            if let Some(file) = trace_file.as_mut() {
                std::io::Write::write_all(file, format!("{record}\n").as_bytes())
                    .map_err(|error| format!("execute: CUDA trace write: {error}"))?;
            }
            Ok(())
        };
        let graph_candidate = self.commands.iter().any(|command| {
            matches!(
                command.kind,
                CommandKind::Kernel {
                    state: StateAccess::Kv { .. }
                        | StateAccess::Kda { layer: Some(_), .. }
                        | StateAccess::Conv { layer: Some(_), .. },
                    ..
                }
            )
        });
        #[cfg(test)]
        let graph_enabled = graph_candidate
            && graphs_requested
            && !trace
            && after_gemm.is_none()
            && after_kv.is_none();
        #[cfg(not(test))]
        let graph_enabled = graph_candidate && graphs_requested && !trace;
        let _graph_execution = graph_enabled
            .then(|| self.device.graph_execution.lock())
            .transpose()
            .map_err(|_| "execute: CUDA graph execution lock poisoned")?;
        let mut graph_runtime = graph_enabled
            .then(|| self.graph_runtime.lock())
            .transpose()
            .map_err(|_| "execute: CUDA graph cache lock poisoned")?;
        let mut capture_fence = StreamCaptureFence {
            stream: self.device.stream.clone(),
            active: false,
        };
        let mut capture: Option<(GraphKey, usize)> = None;
        let mut deferred_status = None;
        let mut grouped_routing = HashMap::<ValueId, Vec<u32>>::new();
        let mut values: Vec<Option<CudaValue>> = vec![None; self.program.values.len()];
        let mut position = 0usize;
        while position < self.commands.len() {
            if cancelled.is_cancelled() {
                return Err("operation aborted".into());
            }
            if capture.is_none() {
                if let (Some(runtime), Some(end)) =
                    (graph_runtime.as_mut(), self.graph_segment_end(position))
                {
                    self.materialize_graph_values(
                        &resources,
                        bindings,
                        &mut values,
                        position,
                        end,
                    )?;
                    let key = self.graph_key(&resources, &values, position, end)?;
                    if let Some(graph) = runtime.cache.get(&key) {
                        graph.0.launch().map_err(|error| error.to_string())?;
                        if graph_trace {
                            self.device.stream.synchronize().map_err(|error| {
                                format!(
                                    "CUDA graph replay {}..{} failed: {error}",
                                    key.start, key.end
                                )
                            })?;
                        }
                        runtime.hits += 1;
                        position = end;
                        continue;
                    }
                    // Loader programs often repeat with different weight addresses.
                    // Capture only after this exact command and pointer set repeats.
                    if !runtime.seen.insert(key.clone()) {
                        self.device
                            .stream
                            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                            .map_err(|error| format!("CUDA graph capture failed: {error}"))?;
                        capture_fence.active = true;
                        capture = Some((key, end));
                    }
                }
            }
            let command = &self.commands[position];
            let Some(output_id) = command.output else {
                position += 1;
                continue;
            };
            let meta = &self.program.values[output_id.index()];
            let traced = trace
                && !matches!(
                    command.kind,
                    CommandKind::Prepare
                        | CommandKind::Value(_)
                        | CommandKind::Input { .. }
                        | CommandKind::Alias { .. }
                );
            let started = if traced {
                self.device
                    .stream
                    .synchronize()
                    .map_err(|error| error.to_string())?;
                let value = |id: ValueId| {
                    let value = &self.program.values[id.index()];
                    serde_json::json!({ "shape": value.shape, "dtype": value.dtype.name() })
                };
                let operation = match &command.kind {
                    CommandKind::Kernel {
                        name,
                        inputs,
                        args,
                        checked,
                        state,
                        kv_matmul,
                        ..
                    } => serde_json::json!({
                        "kernel": if kv_matmul.is_some() { "kv_stepwise_bf16_gemm_active_rows" } else { name },
                        "capturable": !checked && kv_matmul.is_none(),
                        "dynamicGemmDimensions": kv_matmul.is_some(),
                        "workspaceBytes": kv_matmul.map(|plan| plan.bytes),
                        "inputs": inputs.iter().flatten().map(|id| value(*id)).collect::<Vec<_>>(),
                        "state": format!("{state:?}"),
                        "integers": args.integers,
                        "computeDtype": args.compute_dtype
                    }),
                    CommandKind::Gemm {
                        x, weight, plan, ..
                    } => serde_json::json!({
                        "kernel": "cublas_bf16_gemm", "inputs": [value(*x), value(*weight)],
                        "m": plan.m, "n": plan.n, "k": plan.k, "batch": plan.batch
                    }),
                    CommandKind::LinearBias { .. } => {
                        serde_json::json!({ "kernel": "et_linear_bias_f32" })
                    }
                    CommandKind::FusedElementwise { inputs, .. } => serde_json::json!({
                        "kernel": "fused_elementwise", "capturable": true,
                        "inputs": inputs.iter().flatten().map(|id| value(*id)).collect::<Vec<_>>()
                    }),
                    CommandKind::GroupedExpert {
                        rows,
                        columns,
                        inner,
                        experts,
                        indexes,
                        control,
                        reuse_routing,
                        ..
                    } => serde_json::json!({
                        "kernel": "grouped_expert_linear_rows", "rows": rows, "columns": columns,
                        "inner": inner, "experts": experts, "capturable": false,
                        "indexesValue": indexes.get(), "controlValue": control.get(),
                        "routingReused": reuse_routing,
                        "controlReadbackBytes": if *reuse_routing { 0 } else { (experts + 2) * 4 }
                    }),
                    CommandKind::Scalar { .. } => serde_json::json!({ "kernel": "et_fill" }),
                    CommandKind::Cursor { .. } => serde_json::json!({ "kernel": "cursor_upload" }),
                    _ => unreachable!(),
                };
                emit_trace(serde_json::json!({
                    "event": "begin", "program": format!("{self:p}"), "run": run, "instruction": position,
                    "operation": operation, "output": value(output_id),
                    "state": state.as_ref().map(|state| serde_json::json!({
                        "cursors": state.sequences.iter().map(|sequence| sequence.cursor).collect::<Vec<_>>(),
                        "validLengths": state.valid_lengths, "capacity": state.capacity,
                        "access": format!("{:?}", state.access)
                    }))
                }))?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            let output = match &command.kind {
                CommandKind::Prepare => {
                    position += 1;
                    continue;
                }
                CommandKind::Value(value) => value.clone(),
                CommandKind::Input { binding } => {
                    let value = bindings
                        .get(*binding)
                        .ok_or_else(|| format!("execute: missing CUDA binding {binding}"))?;
                    if value.ordinal() != self.device.ordinal
                        || value.shape() != meta.shape
                        || value.dtype() != meta.dtype
                        || value.spec().storage.representation != meta.storage.representation
                    {
                        return Err(format!(
                            "execute: CUDA binding {binding} violates its value specification"
                        ));
                    }
                    value.clone()
                }
                CommandKind::Alias { source } => values[source.index()]
                    .as_ref()
                    .ok_or("execute: alias source is unavailable")?
                    .reshape_dense(meta.shape.clone())?,
                CommandKind::Scalar { binding } => {
                    let value = *scalars
                        .get(*binding)
                        .ok_or("execute: scalar binding is missing")?;
                    let output = self.planned_value(&resources, output_id)?;
                    let args = CudaKernelArgs {
                        output: output.storage_address(),
                        elements: element_count(&meta.shape)? as u64,
                        output_dtype: dtype_code(meta.dtype),
                        scalars: [value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ..Default::default()
                    };
                    self.launch("et_fill", &args)?;
                    output
                }
                CommandKind::Cursor { tensor } => {
                    let state = state
                        .as_deref()
                        .ok_or("execute: state cursor requires state")?;
                    let mut cursors = vec![0i64; element_count(&meta.shape)?];
                    if *tensor {
                        let rows = state.packed_rows_per_sequence.unwrap_or(1) as usize;
                        for (request, &slot) in state.slots.iter().enumerate() {
                            for offset in 0..rows {
                                let target = cursors
                                    .get_mut(slot as usize * rows + offset)
                                    .ok_or("execute: cursor lane is invalid")?;
                                *target =
                                    i64::from(state.sequences[request].cursor) + offset as i64;
                            }
                        }
                    } else {
                        cursors[0] = i64::from(
                            state
                                .sequences
                                .first()
                                .ok_or("execute: cursor sequence missing")?
                                .cursor,
                        );
                    }
                    let output = self.planned_value(&resources, output_id)?;
                    let mut buffer = output.buffer.cast::<i64>(cursors.len())?;
                    self.device
                        .stream
                        .memcpy_htod(&cursors, &mut buffer)
                        .map_err(|e| e.to_string())?;
                    output
                }
                CommandKind::Gemm {
                    x,
                    weight,
                    weight_transposed,
                    plan,
                    out_f32,
                    workspace,
                } => {
                    let output = self.planned_value(&resources, output_id)?;
                    let x_value = values[x.index()]
                        .as_ref()
                        .ok_or("execute: GEMM activation is unavailable")?;
                    let weight_value = values[weight.index()]
                        .as_ref()
                        .ok_or("execute: GEMM weight is unavailable")?;
                    let workspace = self.buffer(&resources, *workspace)?;
                    // SAFETY: lowering checked geometry and binding types. Values
                    // and the invocation fence retain these allocations until the
                    // device stream completes, including cancellation and errors.
                    unsafe {
                        self.device.cublas.gemm_bf16(
                            *plan,
                            *weight_transposed,
                            x_value.storage_address(),
                            weight_value.storage_address(),
                            output.storage_address(),
                            *out_f32,
                            workspace.address(),
                        )?;
                    }
                    #[cfg(test)]
                    if let Some(after_gemm) = after_gemm {
                        after_gemm();
                    }
                    output
                }
                CommandKind::GroupedExpert {
                    x,
                    weight,
                    indexes,
                    rows,
                    columns,
                    inner,
                    experts,
                    control,
                    row_map,
                    gathered,
                    projected,
                    workspace,
                    reuse_routing,
                } => {
                    let output = self.planned_value(&resources, output_id)?;
                    if *rows != 0 {
                        let address = |id: &ValueId| -> Result<u64, String> {
                            Ok(values[id.index()]
                                .as_ref()
                                .ok_or("execute: grouped expert operand unavailable")?
                                .storage_address())
                        };
                        let control_id = *control;
                        let control = self
                            .buffer(&resources, control_id)?
                            .cast::<u32>(experts + 2)?;
                        let row_map = self.buffer(&resources, *row_map)?;
                        let gathered = self.buffer(&resources, *gathered)?;
                        let projected = self.buffer(&resources, *projected)?;
                        let mut args = CudaKernelArgs {
                            output: control.address(),
                            elements: (experts + 2) as u64,
                            output_dtype: dtype_code(DType::U32),
                            ..Default::default()
                        };
                        args.inputs[0] = address(indexes)?;
                        args.integers[0] = *experts as u64;
                        let offsets = if *reuse_routing {
                            grouped_routing
                                .get(&control_id)
                                .cloned()
                                .ok_or("execute: grouped expert routing cache is missing")?
                        } else {
                            let control_started = trace.then(std::time::Instant::now);
                            self.launch("et_fill", &args)?;
                            args.elements = *rows as u64;
                            self.launch("et_grouped_counts", &args)?;
                            args.elements = 1;
                            self.launch("et_grouped_offsets", &args)?;
                            // This is an explicit non-capturable host completion point.
                            // Only status and E+1 offsets cross the host. No padded
                            // groups, activations, weights, or row maps are read back.
                            let readback_started = trace.then(std::time::Instant::now);
                            let offsets = self
                                .device
                                .stream
                                .clone_dtoh(&control)
                                .map_err(|e| e.to_string())?;
                            if let Some(started) = control_started {
                                emit_trace(serde_json::json!({
                                    "event": "grouped_control", "program": format!("{self:p}"), "run": run,
                                    "instruction": position, "capturable": false, "bytes": (experts + 2) * 4,
                                    "milliseconds": started.elapsed().as_secs_f64() * 1000.0,
                                    "readbackAndWaitMilliseconds": readback_started.unwrap().elapsed().as_secs_f64() * 1000.0,
                                    "activeGroupRows": offsets[1..].windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>()
                                }))?;
                            }
                            if let Some(status) = deferred_status {
                                let status = self.buffer(&resources, status)?.cast::<u64>(1)?;
                                let failure = self
                                    .device
                                    .stream
                                    .clone_dtoh(&status)
                                    .map_err(|e| e.to_string())?[0];
                                self.status_result(failure)?;
                            }
                            grouped_routing.insert(control_id, offsets.clone());
                            offsets
                        };
                        if offsets[0] != 0 {
                            return Err(
                                "groupedExpertLinearRows: expert index is out of range".into()
                            );
                        }
                        if cancelled.is_cancelled() {
                            return Err("operation aborted".into());
                        }
                        if *columns != 0 && *inner == 0 {
                            args.output = output.storage_address();
                            args.output_dtype = dtype_code(meta.dtype);
                            args.elements = (*rows * *columns) as u64;
                            self.launch("et_fill", &args)?;
                        } else if *columns != 0 {
                            if !*reuse_routing {
                                args.elements = *rows as u64;
                                args.inputs[1] = control.address();
                                args.output = row_map.address();
                                self.launch("et_grouped_rows", &args)?;
                            }
                            args.inputs[0] = address(x)?;
                            args.inputs[1] = row_map.address();
                            args.output = gathered.address();
                            args.output_dtype = dtype_code(meta.dtype);
                            args.elements = (*rows * *inner) as u64;
                            args.integers[0] = *inner as u64;
                            self.launch("et_grouped_gather", &args)?;
                            let width = meta.dtype.size_in_bytes();
                            let weights = address(weight)?;
                            let workspace = workspace
                                .map(|id| self.buffer(&resources, id))
                                .transpose()?;
                            let mut bf16_groups = Vec::new();
                            for (expert, range) in offsets[1..].windows(2).enumerate() {
                                if cancelled.is_cancelled() {
                                    return Err("operation aborted".into());
                                }
                                let start = range[0] as usize;
                                let count = (range[1] - range[0]) as usize;
                                if count == 0 {
                                    continue;
                                }
                                let x = gathered.address() + (start * *inner * width) as u64;
                                let weight = weights + (expert * *columns * *inner * width) as u64;
                                let out = projected.address() + (start * *columns * width) as u64;
                                if meta.dtype == DType::BF16 {
                                    let plan = crate::cublas::Bf16GemmPlan {
                                        m: count,
                                        n: *columns,
                                        k: *inner,
                                        batch: 1,
                                        stride_x: count * *inner,
                                        stride_weight: 0,
                                        stride_out: count * *columns,
                                    };
                                    bf16_groups.push((plan, x, weight, out));
                                } else {
                                    let mut gemm = CudaKernelArgs {
                                        output: out,
                                        elements: (count * *columns) as u64,
                                        ..Default::default()
                                    };
                                    gemm.inputs[..2].copy_from_slice(&[x, weight]);
                                    gemm.integers[..2]
                                        .copy_from_slice(&[*columns as u64, *inner as u64]);
                                    self.launch("et_grouped_matmul_f32", &gemm)?;
                                }
                                #[cfg(test)]
                                if let Some(after_gemm) = after_gemm {
                                    after_gemm();
                                }
                            }
                            if !bf16_groups.is_empty() {
                                // SAFETY: every group uses checked geometry and
                                // invocation-planned allocations retained by the fence.
                                unsafe {
                                    let events = self.device.grouped_gemm_bf16(
                                        &bf16_groups,
                                        workspace
                                            .as_ref()
                                            .ok_or("grouped GEMM workspace missing")?
                                            .address(),
                                    )?;
                                    fence.worker_events.extend(events);
                                }
                            }
                            if cancelled.is_cancelled() {
                                return Err("operation aborted".into());
                            }
                            args.inputs[0] = projected.address();
                            args.inputs[1] = row_map.address();
                            args.output = output.storage_address();
                            args.elements = (*rows * *columns) as u64;
                            args.integers[0] = *columns as u64;
                            self.launch("et_grouped_scatter", &args)?;
                        }
                    }
                    output
                }
                CommandKind::LinearBias {
                    accumulator,
                    bias,
                    args,
                } => {
                    let output = self.planned_value(&resources, output_id)?;
                    let mut args = *args;
                    args.inputs[0] = values[accumulator.index()]
                        .as_ref()
                        .ok_or("execute: linear accumulator is unavailable")?
                        .storage_address();
                    args.inputs[1] = values[bias.index()]
                        .as_ref()
                        .ok_or("execute: linear bias is unavailable")?
                        .storage_address();
                    args.output = output.storage_address();
                    // This kernel cannot report a numerical error. Stream order
                    // makes the GEMM accumulator visible without status transfers
                    // or a host wait; the invocation fence handles launch errors.
                    self.launch(BF16_LINEAR_BIAS_KERNEL, &args)?;
                    output
                }
                CommandKind::FusedElementwise {
                    function,
                    args,
                    inputs,
                } => {
                    let output = self.planned_value(&resources, output_id)?;
                    let mut args = *args;
                    args.output = output.storage_address();
                    for (slot, input) in inputs.iter().enumerate() {
                        if let Some(input) = input {
                            args.inputs[slot] = values[input.index()]
                                .as_ref()
                                .ok_or("execute: fused CUDA input unavailable")?
                                .storage_address();
                        }
                    }
                    if args.elements != 0 {
                        let mut launch = self.device.stream.launch_builder(function);
                        launch.arg(&args);
                        unsafe {
                            launch.launch(LaunchConfig {
                                grid_dim: (args.elements.div_ceil(256).min(65535) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }
                        .map_err(|error| error.to_string())?;
                    }
                    output
                }
                CommandKind::Kernel {
                    name,
                    args,
                    inputs,
                    scratch,
                    status,
                    checked,
                    state: access,
                    state_buffers,
                    kv_matmul,
                    ..
                } => {
                    let output = self.planned_value(&resources, output_id)?;
                    let mut args = *args;
                    args.output = output.storage_address();
                    args.metadata = self.metadata[position]
                        .as_ref()
                        .ok_or("execute: kernel metadata missing")?
                        .address();
                    for (role, id) in inputs.iter().enumerate() {
                        if let Some(id) = id {
                            args.inputs[role] = values[id.index()]
                                .as_ref()
                                .ok_or("execute: input unavailable")?
                                .storage_address();
                        }
                    }
                    let scratch_buffers = scratch
                        .iter()
                        .map(|id| id.map(|id| self.buffer(&resources, id)).transpose())
                        .collect::<Result<Vec<_>, _>>()?;
                    for (slot, buffer) in scratch_buffers.iter().enumerate() {
                        if let Some(buffer) = buffer {
                            args.scratch[slot] = buffer.address();
                        }
                    }
                    if *checked {
                        let status = status.ok_or("execute: checked CUDA kernel has no status")?;
                        if deferred_status.is_some_and(|current| current != status) {
                            return Err("execute: checked CUDA kernels do not share status".into());
                        }
                        let mut buffer = self.buffer(&resources, status)?.cast::<u64>(1)?;
                        if deferred_status.is_none() {
                            self.device
                                .stream
                                .memcpy_htod(&[0u64], &mut buffer)
                                .map_err(|e| e.to_string())?;
                            deferred_status = Some(status);
                        }
                        args.scratch[3] = buffer.address();
                        args.error_context = u32::try_from(position)
                            .map_err(|_| "execute: CUDA error context exceeds u32")?;
                    }
                    if name.starts_with("et_random_") {
                        args.integers[0] =
                            args.integers[0].wrapping_add(run.wrapping_mul(0x9e3779b97f4a7c15));
                    }
                    let transaction_buffers = state_buffers
                        .iter()
                        .map(|id| id.map(|id| self.buffer(&resources, id)).transpose())
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut kv_ranges = Vec::new();
                    self.prepare_kernel_state(
                        access,
                        &mut args,
                        &scratch_buffers,
                        &transaction_buffers,
                        state.as_deref_mut(),
                        &mut kv_ranges,
                    )?;
                    if let Some(plan) = kv_matmul {
                        if traced {
                            emit_trace(serde_json::json!({
                                "event":"kv_gemm_geometry", "instruction":position, "capturable":false,
                                "lanes":kv_ranges.iter().map(|&(lane,p,q)|serde_json::json!({"lane":lane,"p":p,"q":q})).collect::<Vec<_>>(),
                                "workspaceBytes":plan.bytes, "kernelAndGemmSubmissions":2+5*kv_ranges.len(),
                            }))?;
                        }
                        crate::kv_matmul::execute(
                            &self.device,
                            &args,
                            *plan,
                            &kv_ranges,
                            |name, args| self.launch(name, args),
                        )?;
                    } else {
                        self.launch(name, &args)?;
                    }
                    #[cfg(test)]
                    if matches!(access, StateAccess::Kv { .. }) {
                        if let Some(hook) = after_kv {
                            hook();
                        }
                    }
                    self.commit_kernel_state(access, &scratch_buffers, state.as_deref_mut())?;
                    output
                }
            };
            if let Some(started) = started {
                self.device
                    .stream
                    .synchronize()
                    .map_err(|error| error.to_string())?;
                emit_trace(serde_json::json!({
                    "event": "end", "program": format!("{self:p}"), "run": run, "instruction": position,
                    "diagnostic": true, "streamSynchronized": true,
                    "milliseconds": started.elapsed().as_secs_f64() * 1000.0
                }))?;
            }
            values[output_id.index()] = Some(output);
            position += 1;
            if capture.as_ref().is_some_and(|(_, end)| *end == position) {
                let (key, _) = capture.take().unwrap();
                let graph = self
                    .device
                    .stream
                    .end_capture(
                        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
                    )
                    .map_err(|error| format!("CUDA graph instantiation failed: {error}"))?
                    .ok_or("CUDA graph capture produced no graph")?;
                capture_fence.active = false;
                graph.launch().map_err(|error| error.to_string())?;
                if graph_trace {
                    self.device.stream.synchronize().map_err(|error| {
                        format!(
                            "CUDA captured graph {}..{} failed: {error}",
                            key.start, key.end
                        )
                    })?;
                }
                graph_runtime
                    .as_mut()
                    .ok_or("execute: CUDA graph cache unavailable")?
                    .cache
                    .insert(key, CapturedGraph(graph));
                graph_runtime
                    .as_mut()
                    .ok_or("execute: CUDA graph cache unavailable")?
                    .captures += 1;
            }
        }
        let failure = if let Some(status) = deferred_status {
            let status = self.buffer(&resources, status)?.cast::<u64>(1)?;
            self.device
                .stream
                .clone_dtoh(&status)
                .map_err(|e| e.to_string())?[0]
        } else {
            self.device
                .stream
                .synchronize()
                .map_err(|e| e.to_string())?;
            0
        };
        fence.complete = true;
        self.status_result(failure)?;
        if cancelled.is_cancelled() {
            return Err("operation aborted".into());
        }
        if graph_trace {
            if let Some(runtime) = graph_runtime.as_ref() {
                eprintln!(
                    "CUDA graph run={run} cache={} hits={} captures={}",
                    runtime.cache.len(),
                    runtime.hits,
                    runtime.captures
                );
            }
        }
        self.program
            .outputs
            .iter()
            .map(|id| {
                values[id.index()]
                    .clone()
                    .ok_or_else(|| "execute: CUDA output unavailable".into())
            })
            .collect()
    }
    fn launch(&self, name: &str, args: &CudaKernelArgs) -> Result<(), String> {
        if name == "et_expert_linear_rows" {
            // One warp per output, or one per route when O=0. Bounded launch
            // dimensions; the kernel loops over remaining work with u64 offsets.
            let work = args.integers[0]
                .checked_mul(args.integers[1].max(1))
                .ok_or("expertLinearRows: work size overflow")?;
            if work == 0 {
                return Ok(());
            }
            let function = self.device.kernel(name)?;
            let mut launch = self.device.stream.launch_builder(function);
            launch.arg(args);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (work.div_ceil(8).min(65535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| e.to_string())?;
            return Ok(());
        }
        if args.elements == 0 {
            return Ok(());
        }
        if name == "et_scatter_add_inner" {
            let routes = args.integers[0];
            let inner = args.integers[1];
            let group_elements = routes
                .checked_mul(inner)
                .ok_or("scatterAdd: compact geometry overflow")?;
            if group_elements == 0 || args.elements % group_elements != 0 {
                return Err("scatterAdd: invalid compact geometry".into());
            }
            let rows = args.elements / group_elements;
            let blocks = rows
                .checked_mul(routes)
                .and_then(|groups| groups.checked_mul(inner.div_ceil(256)))
                .ok_or("scatterAdd: compact launch overflow")?;
            let function = self.device.kernel(name)?;
            let mut launch = self.device.stream.launch_builder(function);
            launch.arg(args);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        u32::try_from(blocks)
                            .map_err(|_| "scatterAdd: compact grid exceeds u32")?,
                        1,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| e.to_string())?;
            return Ok(());
        }
        if name == "et_rms_norm_f32" && args.integers[0] >= 1024 {
            let rows = args.elements / args.integers[0];
            let function = self.device.kernel("et_rms_norm_wide_f32")?;
            let mut launch = self.device.stream.launch_builder(function);
            launch.arg(args);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (rows.min(65535) as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| e.to_string())?;
            return Ok(());
        }
        // Current rows must be stored before query/head warps read them,
        // including future rows in a bidirectional canvas. Both launches use
        // the same stream and invocation-owned transaction/scratch storage.
        let wide_arg_index = name == "et_index"
            && args.operation <= 1
            && args.input_dtypes[0] == dtype_code(DType::F32)
            && args.integers[2] != 0
            && args.integers[1] >= 4096;
        if wide_arg_index {
            let function = self.device.kernel("et_arg_index_last_wide")?;
            let mut launch = self.device.stream.launch_builder(function);
            launch.arg(args);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (args.elements.min(65535) as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| e.to_string())?;
            return Ok(());
        }
        let warp_sum = name == "et_reduce_f32" && args.operation == 0;
        let wide_trailing =
            name == "et_reduce_f32" && args.integers[2] != 0 && args.integers[1] >= 4096;
        if wide_trailing && matches!(args.operation, 0 | 2 | 3) {
            let kernel = if args.operation == 0 {
                "et_sum_wide_f32"
            } else {
                "et_reduce_last_wide_f32"
            };
            let function = self.device.kernel(kernel)?;
            let mut launch = self.device.stream.launch_builder(function);
            launch.arg(args);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (args.elements.min(65535) as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| e.to_string())?;
            return Ok(());
        }
        let work_items = if warp_sum {
            args.elements
                .checked_mul(32)
                .ok_or("CUDA Sum grid overflow")?
        } else if name == "et_rms_norm_f32" {
            (args.elements / args.integers[0])
                .checked_mul(32)
                .ok_or("CUDA RMS grid overflow")?
        } else if name == "et_grouped_rows" {
            args.integers[0] * 32
        } else if name == "et_kv_attention" {
            self.launch("et_kv_store", args)?;
            (args.elements / args.integers[10])
                .checked_mul(32)
                .ok_or("CUDA KV grid overflow")?
        } else if name == "et_kv_store" {
            args.integers[5]
                .checked_mul(args.integers[2])
                .and_then(|n| n.checked_mul(args.integers[7]))
                .and_then(|n| n.checked_mul(256))
                .ok_or("CUDA KV store grid overflow")?
        } else if name == "et_top_k_indices" {
            // One block per row. Thread zero owns the stable insertion output.
            (args.elements / args.integers[0]) * 256
        } else {
            args.elements
        };
        let blocks = if name.starts_with("et_grouped_") || name == "et_rms_norm_f32" || warp_sum {
            work_items.div_ceil(256).clamp(1, 65535) as u32
        } else {
            u32::try_from(work_items.div_ceil(256)).map_err(|_| "CUDA grid overflow")?
        };
        let function = self.device.kernel(name)?;
        let mut launch = self.device.stream.launch_builder(function);
        launch.arg(args);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Counts backend submissions and required host completion points on the
/// successful path. Transfers count as commands; cuBLAS internals do not.
/// Compilation uploads, output readback by callers, and driver-dependent
/// implicit stalls are outside these invocation diagnostics.
pub(super) fn physical_counts(
    program: &CudaLoweredProgram,
    commands: &[Command],
) -> (usize, usize) {
    let mut submissions = 0;
    let mut completions = 1; // Final stream completion before output publication.
    let mut has_checked_kernel = false;
    for command in commands {
        match &command.kind {
            CommandKind::Gemm { .. } | CommandKind::Cursor { .. } => submissions += 1,
            CommandKind::GroupedExpert {
                rows,
                columns,
                inner,
                experts,
                reuse_routing,
                ..
            } => {
                if *rows != 0 {
                    if !*reuse_routing {
                        // Reset/count/prefix/readback. An initialized deferred
                        // status is read after the same completion point.
                        submissions += 4 + usize::from(has_checked_kernel);
                        completions += 1;
                    }
                    if *columns != 0 {
                        submissions += if *inner == 0 {
                            1
                        } else {
                            2 + usize::from(!*reuse_routing) + rows.min(experts)
                        };
                    }
                }
            }
            CommandKind::Scalar { .. } | CommandKind::LinearBias { .. } => {
                submissions += usize::from(
                    command
                        .output
                        .is_some_and(|id| program.values[id.index()].decl.bytes != 0),
                );
            }
            CommandKind::FusedElementwise { args, .. } => {
                submissions += usize::from(args.elements != 0);
            }
            CommandKind::Kernel {
                name,
                args,
                checked,
                state,
                kv_matmul,
                ..
            } => {
                // Checked kernels share one invocation status. Its reset and
                // final readback are counted once below.
                let launches = if *name == "et_expert_linear_rows" {
                    args.integers[0] != 0
                } else {
                    args.elements != 0
                };
                let kernel_submissions = if kv_matmul.is_some() {
                    // Zero output, store current rows, then gather/QK/softmax/PV/round
                    // per active lane. Static diagnostics bound all declared lanes.
                    let StateAccess::Kv { batch, .. } = state else {
                        unreachable!()
                    };
                    2 + 5 * batch
                } else if *name == "et_kv_attention" {
                    2
                } else {
                    1
                };
                submissions += usize::from(launches) * kernel_submissions;
                has_checked_kernel |= *checked;
                submissions += match state {
                    StateAccess::Rotary => 1,
                    StateAccess::Kv { .. } => 3,
                    StateAccess::Kda { .. } | StateAccess::Conv { .. } => 2,
                    StateAccess::None | StateAccess::LastToken { .. } => 0,
                };
                if matches!(
                    state,
                    StateAccess::Kda { layer: Some(_), .. }
                        | StateAccess::Conv { layer: Some(_), .. }
                ) {
                    submissions += 1; // Recurrent-state readback.
                    completions += 1;
                }
            }
            CommandKind::Prepare
            | CommandKind::Value(_)
            | CommandKind::Input { .. }
            | CommandKind::Alias { .. } => {}
        }
    }
    submissions += 2 * usize::from(has_checked_kernel);
    (submissions, completions)
}

pub fn compile(roots: Vec<Arc<Node>>, ordinal: u32) -> Result<CudaExecutable, String> {
    compile_with_options(roots, ordinal, CompileOptions::from_environment())
}
pub fn compile_with_options(
    roots: Vec<Arc<Node>>,
    ordinal: u32,
    options: CompileOptions,
) -> Result<CudaExecutable, String> {
    compile_inner(roots, ordinal, options, None, None)
}
pub fn compile_stateful(
    roots: Vec<Arc<Node>>,
    ordinal: u32,
    cursor_slot: u32,
    cursor_tensor: bool,
) -> Result<CudaExecutable, String> {
    compile_stateful_with_options(
        roots,
        ordinal,
        cursor_slot,
        cursor_tensor,
        CompileOptions::from_environment(),
    )
}
pub fn compile_stateful_with_options(
    roots: Vec<Arc<Node>>,
    ordinal: u32,
    cursor_slot: u32,
    cursor_tensor: bool,
    options: CompileOptions,
) -> Result<CudaExecutable, String> {
    compile_inner(
        roots,
        ordinal,
        options,
        Some((cursor_slot, cursor_tensor)),
        None,
    )
}
pub fn compile_stateful_with_layout(
    roots: Vec<Arc<Node>>,
    ordinal: u32,
    cursor_slot: u32,
    cursor_tensor: bool,
    options: CompileOptions,
    layout: CudaStateLayout,
) -> Result<CudaExecutable, String> {
    layout.validate()?;
    compile_inner(
        roots,
        ordinal,
        options,
        Some((cursor_slot, cursor_tensor)),
        Some(layout),
    )
}
fn compile_inner(
    roots: Vec<Arc<Node>>,
    ordinal: u32,
    options: CompileOptions,
    state_cursor: Option<(u32, bool)>,
    state_layout: Option<CudaStateLayout>,
) -> Result<CudaExecutable, String> {
    let device = CudaDevice::get(ordinal)?;
    let capabilities = CudaCapabilities::for_device(&device)?;
    let mut request = ProgramRequest::from_roots(roots, options);
    if let Some((slot, tensor)) = state_cursor {
        request = request.with_state_cursor(StateCursorSlot::new(slot, tensor));
    }
    let prepared = request.prepare()?;
    let mut driver = CompilerDriver::new(&prepared, &capabilities)?;
    let mut builder =
        CudaProgramBuilder::new(&prepared.index, state_layout.clone(), driver.legalization())?;
    driver.lower(|unit, index, optimization, plan| match unit {
        LoweringUnit::Node(dense) => {
            let node = index.node(dense).ok_or("compile: CUDA node missing")?;
            let instruction = semantic_instruction(node, index, &device, ordinal, state_cursor)?;
            builder.add(dense, node, index, optimization, instruction, plan)
        }
        LoweringUnit::Region(region) => {
            builder.add_region(index, optimization, region, plan, &device)
        }
    })?;
    driver.record_materialized_conversions(builder.conversion_count, builder.conversion_bytes);
    let (program, commands) = builder.finish(&prepared.index)?;
    let memory = driver
        .plan_memory(
            &program,
            &MemoryPlannerConfig::uniform(
                CudaMemorySpace::Device,
                usize::MAX / 2,
                CUDA_STORAGE_ALIGNMENT,
                CUDA_STORAGE_ALIGNMENT,
            ),
        )
        .map_err(|e| e.to_string())?;
    let metadata = driver.phase(PHYSICAL_PLANNING_PHASE, || {
        commands
            .iter()
            .map(|command| match &command.kind {
                CommandKind::Kernel { name, metadata, .. } => {
                    device.kernel(name)?;
                    device
                        .stream
                        .clone_htod(metadata)
                        .map(CudaBuffer::from_slice)
                        .map(Some)
                        .map_err(|e| e.to_string())
                }
                CommandKind::LinearBias { .. } => {
                    device.kernel(BF16_LINEAR_BIAS_KERNEL)?;
                    Ok(None)
                }
                CommandKind::GroupedExpert { .. } => {
                    for name in [
                        "et_fill",
                        "et_grouped_counts",
                        "et_grouped_offsets",
                        "et_grouped_rows",
                        "et_grouped_gather",
                        "et_grouped_scatter",
                        "et_grouped_matmul_f32",
                    ] {
                        device.kernel(name)?;
                    }
                    Ok(None)
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, String>>()
    })?;
    let outputs = prepared
        .roots
        .iter()
        .map(|root| (root.shape.clone(), root.dtype))
        .collect();
    let mut diagnostics = driver.phase(ARTIFACT_ASSEMBLY_PHASE, || {
        let (command_count, synchronization_count) = physical_counts(&program, &commands);
        Ok::<_, String>(build_executable_diagnostics(
            &program,
            &memory,
            &prepared.index,
            DiagnosticsInput {
                pipeline_count: 0,
                command_count,
                synchronization_count,
                compile_phases: Vec::new(),
            },
            |name| *name,
        ))
    })?;
    diagnostics.legalization = driver.legalization_diagnostics();
    let work = driver.finish_with_phase(&program, PUBLICATION_PHASE, std::time::Instant::now());
    let mut executable = CudaExecutable {
        state_layout,
        device,
        program,
        memory,
        commands,
        metadata,
        outputs,
        diagnostics,
        compiler_work: work,
        runs: AtomicU64::new(0),
        graph_runtime: Mutex::new(GraphRuntime::default()),
    };
    executable.diagnostics.compile_phases = executable.compiler_work.compile_phases.clone();
    Ok(executable)
}
fn full_value(
    device: Arc<CudaDevice>,
    shape: Vec<usize>,
    dtype: DType,
    value: f64,
) -> Result<CudaValue, String> {
    let count = element_count(&shape)?;
    CudaValue::from_host(device, shape, dtype, &vec![value; count])
}

impl CudaExecutable {
    /// Borrow immutable page tables; no prefix payload is uploaded or copied.
    pub fn prepare_state(&self, state: &mut CudaStateInvocation) -> Result<(), String> {
        let layout = self
            .state_layout
            .as_ref()
            .ok_or("execute: state layout is missing")?;
        if layout.capacity != state.capacity
            || layout.dtype != state.cache_dtype
            || layout.slots as usize != state.valid_lengths.len()
            || layout.packed_rows_per_sequence != state.packed_rows_per_sequence
            || layout.kv_layers != state.kv_layers
            || layout.access != state.access
        {
            return Err("execute: state layout differs from compiled layout".into());
        }
        if state.slots.len() != state.sequences.len() {
            return Err("execute: CUDA sequence slots are invalid".into());
        }
        let mut seen = vec![false; state.valid_lengths.len()];
        let bounded = state.kv_layers.iter().any(|layer| {
            layer
                .retention
                .is_none_or(|retention| retention > state.capacity as usize)
        });
        for (request, &slot) in state.slots.iter().enumerate() {
            let active = seen
                .get_mut(slot as usize)
                .ok_or("execute: invalid state slot")?;
            if *active {
                return Err("execute: duplicate state slot".into());
            }
            *active = true;
            let cursor = state.sequences[request].cursor;
            let end = cursor
                .checked_add(state.valid_lengths[slot as usize])
                .ok_or("execute: cursor overflow")?;
            if bounded
                && (cursor > state.capacity
                    || (state.access == StateAccessMode::Append && end > state.capacity))
            {
                return Err("execute: prefix exceeds KV capacity".into());
            }
        }
        if state.cache.is_none() {
            let mut sequences = Vec::with_capacity(state.sequences.len());
            for sequence in &state.sequences {
                let snapshot = sequence
                    .kv_storage
                    .clone()
                    .unwrap_or_else(|| CudaKvSnapshot {
                        layers: state
                            .kv_layers
                            .iter()
                            .map(|&descriptor| CudaKvLayer {
                                descriptor,
                                start_position: 0,
                                pages: Vec::new(),
                            })
                            .collect(),
                    });
                if snapshot
                    .layers
                    .iter()
                    .map(|layer| layer.descriptor)
                    .collect::<Vec<_>>()
                    != state.kv_layers
                {
                    return Err("execute: prefix layer schema differs".into());
                }
                for layer in &snapshot.layers {
                    let expected = sequence.cursor.saturating_sub(
                        layer
                            .descriptor
                            .retention
                            .unwrap_or(state.capacity as usize) as u32,
                    );
                    if layer.start_position != expected
                        || (sequence.cursor > layer.start_position && layer.pages.is_empty())
                    {
                        return Err("execute: prefix is missing retained KV rows".into());
                    }
                }
                sequences.push(snapshot);
            }
            state.cache = Some(CudaKvCache { sequences });
        }
        Ok(())
    }
    /// Publication retains device pages. Only inspect() exports host arrays.
    pub fn readback_state(&self, state: &mut CudaStateInvocation) -> Result<(), String> {
        if state.access == StateAccessMode::ReadOnly {
            return Ok(());
        }
        let Some(cache) = &mut state.cache else {
            return Ok(());
        };
        for (request, snapshot) in cache.sequences.iter_mut().enumerate() {
            let end = state.sequences[request]
                .cursor
                .checked_add(state.valid_lengths[state.slots[request] as usize])
                .ok_or("execute: cursor overflow")?;
            for layer in &mut snapshot.layers {
                let retained = layer
                    .descriptor
                    .retention
                    .unwrap_or(state.capacity as usize);
                layer.start_position = end.saturating_sub(
                    u32::try_from(retained).map_err(|_| "execute: retention exceeds u32")?,
                );
                layer
                    .pages
                    .retain(|page| page.start + page.count > layer.start_position);
            }
            state.sequences[request].kv_storage = Some(snapshot.clone());
        }
        Ok(())
    }
    fn upload_scratch<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        buffer: &Option<CudaBuffer<u8>>,
        values: &[T],
    ) -> Result<(), String> {
        let mut target = buffer
            .as_ref()
            .ok_or("execute: planned staging missing")?
            .cast::<T>(values.len())?;
        self.device
            .stream
            .memcpy_htod(values, &mut target)
            .map_err(|e| e.to_string())
    }
    fn prepare_kernel_state(
        &self,
        access: &StateAccess,
        args: &mut CudaKernelArgs,
        scratch: &[Option<CudaBuffer<u8>>],
        transactions: &[Option<CudaBuffer<u8>>],
        state: Option<&mut CudaStateInvocation>,
        kv_ranges: &mut Vec<(usize, usize, usize)>,
    ) -> Result<(), String> {
        match access {
            StateAccess::None => {}
            StateAccess::LastToken { lane } => {
                args.integers[1] = u64::from(
                    *state
                        .ok_or("execute: last-token selection requires state")?
                        .valid_lengths
                        .get(*lane)
                        .ok_or("execute: invalid last-token lane")?,
                );
            }
            StateAccess::Rotary => {
                let state = state.ok_or("execute: cursor rotary requires state")?;
                let count = scratch[0]
                    .as_ref()
                    .ok_or("execute: rotary staging missing")?
                    .len()
                    / 4;
                let rows = state.packed_rows_per_sequence.unwrap_or(1) as usize;
                let mut cursors = vec![0u32; count];
                for (request, &slot) in state.slots.iter().enumerate() {
                    for row in 0..rows {
                        *cursors
                            .get_mut(slot as usize * rows + row)
                            .ok_or("execute: invalid rotary lane")? = state.sequences[request]
                            .cursor
                            .checked_add(row as u32)
                            .ok_or("execute: cursor overflow")?;
                    }
                }
                self.upload_scratch(&scratch[0], &cursors)?;
            }
            StateAccess::Kv {
                layer,
                heads,
                dim,
                batch,
            } => {
                let state = state.ok_or("execute: KV attention requires state")?;
                let rows = state.packed_rows_per_sequence.unwrap_or(1) as usize;
                if product(&[state.valid_lengths.len(), rows])? != *batch {
                    return Err("execute: KV batch geometry differs".into());
                }
                let tokens = args.integers[7] as usize;
                let per_sequence = product(&[rows, tokens])?;
                let descriptor = *state
                    .kv_layers
                    .iter()
                    .find(|d| d.layer_id as usize == *layer)
                    .ok_or("execute: KV layer descriptor missing")?;
                if descriptor.kv_heads != *heads || descriptor.head_dim != *dim {
                    return Err("execute: KV layer geometry differs".into());
                }
                let cache = state
                    .cache
                    .as_mut()
                    .ok_or("execute: KV state was not prepared")?;
                let row_bytes = product(&[*heads, *dim, descriptor.dtype.size_in_bytes()])?;
                let mut table = vec![0u64; state.valid_lengths.len() * 4];
                let mut cursors = vec![0u32; *batch];
                let mut valid = vec![0u32; *batch];
                for (request, &slot) in state.slots.iter().enumerate() {
                    let slot = slot as usize;
                    let cursor = state.sequences[request].cursor;
                    let count = state.valid_lengths[slot] as usize;
                    if count > per_sequence || (rows > 1 && tokens != 1) {
                        return Err("execute: invalid KV token count".into());
                    }
                    let end = cursor
                        .checked_add(count as u32)
                        .ok_or("execute: cursor overflow")?;
                    let cached = cache.sequences[request]
                        .layers
                        .iter_mut()
                        .find(|l| l.descriptor.layer_id == descriptor.layer_id)
                        .ok_or("execute: KV layer missing")?;
                    cached.pages.retain(|p| p.start < cursor);
                    if count != 0 {
                        let byte_start = product(&[slot, per_sequence, row_bytes])?;
                        let byte_end = byte_start
                            .checked_add(count * row_bytes)
                            .ok_or("KV byte range overflow")?;
                        let slice = |role: usize| {
                            transactions[role]
                                .as_ref()
                                .ok_or("execute: KV transaction missing")?
                                .slice(byte_start..byte_end)
                        };
                        let scales = |role: usize| -> Result<Option<CudaBuffer<f32>>, String> {
                            if descriptor.dtype != DType::U8 {
                                return Ok(None);
                            }
                            let start = product(&[slot, per_sequence, *heads, 4])?;
                            let len = product(&[count, *heads])?;
                            Ok(Some(
                                transactions[role]
                                    .as_ref()
                                    .ok_or("execute: KV scale transaction missing")?
                                    .slice(start..start + len * 4)?
                                    .cast::<f32>(len)?,
                            ))
                        };
                        cached.pages.push(Arc::new(CudaKvPage {
                            start: cursor,
                            count: count as u32,
                            keys: slice(0)?,
                            values: slice(1)?,
                            key_scales: scales(2)?,
                            value_scales: scales(3)?,
                        }));
                    }
                    let start = cached.start_position;
                    let offset = table.len() as u64;
                    table[slot * 4..slot * 4 + 4].copy_from_slice(&[
                        u64::from(start),
                        u64::from(cursor),
                        u64::from(end),
                        offset,
                    ]);
                    let mut position = start;
                    for page in &cached.pages {
                        let from = page.start.max(start);
                        let to = (page.start + page.count).min(end);
                        if to <= from {
                            continue;
                        }
                        if from != position {
                            return Err("execute: prefix page table has a gap".into());
                        }
                        for pos in from..to {
                            let offset = (pos - page.start) as usize;
                            table.extend_from_slice(&[
                                page.keys.address() + (offset * row_bytes) as u64,
                                page.values.address() + (offset * row_bytes) as u64,
                                page.key_scales
                                    .as_ref()
                                    .map_or(0, |s| s.address() + (offset * heads * 4) as u64),
                                page.value_scales
                                    .as_ref()
                                    .map_or(0, |s| s.address() + (offset * heads * 4) as u64),
                            ]);
                        }
                        position = to;
                    }
                    if position != end {
                        return Err("execute: incomplete prefix page table".into());
                    }
                    for row in 0..rows {
                        cursors[slot * rows + row] =
                            cursor.checked_add(row as u32).ok_or("KV cursor overflow")?;
                        valid[slot * rows + row] = if rows == 1 {
                            count as u32
                        } else {
                            u32::from(row < count)
                        };
                        if valid[slot * rows + row] != 0 {
                            kv_ranges.push((
                                slot * rows + row,
                                (end - start) as usize,
                                valid[slot * rows + row] as usize,
                            ));
                        }
                    }
                }
                self.upload_scratch(&scratch[0], &valid)?;
                self.upload_scratch(&scratch[1], &cursors)?;
                let target = scratch[2]
                    .as_ref()
                    .ok_or("execute: KV pointer staging missing")?;
                let mut target = target.slice(0..table.len() * 8)?.cast::<u64>(table.len())?;
                self.device
                    .stream
                    .memcpy_htod(&table, &mut target)
                    .map_err(|e| e.to_string())?;
                args.inputs[3] = target.address();
                args.inputs[4] = args.scratch[2] + args.integers[12];
                args.inputs[7] = args.scratch[1];
                args.integers[1] = u64::from(dtype_code(descriptor.dtype));
                args.integers[2] = rows as u64;
                args.integers[5] = state.valid_lengths.len() as u64;
            }
            StateAccess::Kda {
                layer,
                batch,
                elements_per_sequence,
                ..
            }
            | StateAccess::Conv {
                layer,
                batch,
                elements_per_sequence,
            } => {
                let is_kda = matches!(access, StateAccess::Kda { .. });
                let bytes = scratch[0]
                    .as_ref()
                    .ok_or("execute: recurrent scratch missing")?
                    .len();
                let mut initial = vec![0u8; bytes];
                let mut valid = vec![
                    u32::try_from(args.integers[8])
                        .map_err(|_| "recurrent time exceeds u32")?;
                    *batch
                ];
                if let Some(layer) = layer {
                    let state = state.ok_or("execute: recurrent operation requires state")?;
                    if state.valid_lengths.len() != *batch
                        || state.packed_rows_per_sequence.is_some()
                    {
                        return Err("execute: recurrent packed rows unsupported".into());
                    }
                    valid.clone_from(&state.valid_lengths);
                    for (request, &slot) in state.slots.iter().enumerate() {
                        let states = if is_kda {
                            &state.sequences[request].kda_states
                        } else {
                            &state.sequences[request].conv_states
                        };
                        let values = states
                            .get(*layer)
                            .ok_or("execute: recurrent layer missing")?;
                        if values.len() != *elements_per_sequence {
                            return Err("execute: recurrent geometry differs".into());
                        }
                        let start = slot as usize * elements_per_sequence * 4;
                        for (element, value) in values.iter().enumerate() {
                            initial[start + element * 4..start + (element + 1) * 4]
                                .copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
                self.upload_scratch(&scratch[0], &initial)?;
                self.upload_scratch(&scratch[2], &valid)?;
            }
        }
        Ok(())
    }
    fn commit_kernel_state(
        &self,
        access: &StateAccess,
        scratch: &[Option<CudaBuffer<u8>>],
        state: Option<&mut CudaStateInvocation>,
    ) -> Result<(), String> {
        let (layer, per, slot, is_kda) = match access {
            StateAccess::Kda {
                layer: Some(layer),
                elements_per_sequence,
                ..
            } => (*layer, *elements_per_sequence, 0, true),
            StateAccess::Conv {
                layer: Some(layer),
                elements_per_sequence,
                ..
            } => (*layer, *elements_per_sequence, 1, false),
            _ => return Ok(()),
        };
        let state = state.ok_or("execute: recurrent state disappeared")?;
        let buffer = scratch[slot]
            .as_ref()
            .ok_or("execute: recurrent result staging missing")?;
        let values = self
            .device
            .stream
            .clone_dtoh(&buffer.cast::<f32>(buffer.len() / 4)?)
            .map_err(|e| e.to_string())?;
        for (request, &slot) in state.slots.iter().enumerate() {
            let states = if is_kda {
                &mut state.sequences[request].kda_states
            } else {
                &mut state.sequences[request].conv_states
            };
            states[layer] = values[slot as usize * per..(slot as usize + 1) * per].to_vec();
        }
        Ok(())
    }
}

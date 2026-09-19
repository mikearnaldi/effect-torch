//! Fixed, typed CUDA execution. Every temporary is declared before invocation.
use crate::buffer::CudaBuffer;
use crate::capabilities::CudaCapabilities;
use crate::lowering::{
    Command, CommandKind, CudaLoweredProgram, CudaProgramBuilder, StateComponent,
    BF16_LINEAR_BIAS_KERNEL,
};
use crate::value::element_count;
use crate::workspace::{self, CudaMemorySpace, InvocationResources, CUDA_STORAGE_ALIGNMENT};
use crate::{CudaDevice, CudaValue};
use cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
use effect_torch_compiler::{
    build_executable_diagnostics, CompileOptions, CompilerDriver, CompilerWorkReport,
    DiagnosticsInput, GraphIndex, LoweringUnit, MemoryPlannerConfig, ProgramRequest,
    StateCursorSlot, ARTIFACT_ASSEMBLY_PHASE, PHYSICAL_PLANNING_PHASE, PUBLICATION_PHASE,
};
use effect_torch_graph::{KvAttentionMode, Node, NodeKind, PositionOffset};
use effect_torch_runtime::{
    CancellationFlag, DType, ExecutableDiagnostics, GgmlKQuant, MemoryPlan, ValueId,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

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
    pub(crate) reserved: u32,
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
            Self::Reduce { op, a, dims, .. } => {
                let mut s = KernelSpec::new("et_reduce", &[Some(*a)]);
                s.args.operation = *op;
                s.args.integers[0] = dims.len() as u64;
                s.tail = dims.iter().map(|d| *d as u64).collect();
                s
            }
            Self::Matmul { a, b, .. } => KernelSpec::new("et_matmul", &[Some(*a), Some(*b)]),
            Self::Index {
                op,
                a,
                indexes,
                src,
                dim,
                ..
            } => {
                let mut s = KernelSpec::new("et_index", &[Some(*a), *indexes, *src]);
                s.args.operation = *op;
                s.args.integers[0] = u64::from(*dim);
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
                ..
            } => {
                let mut s = KernelSpec::new("et_kv_attention", &[Some(*q), Some(*k), Some(*v)]);
                s.args.scalars[0] = *scale;
                s.args.integers[3] = window.unwrap_or(0) as u64;
                s.args.integers[4] = u64::from(*bidirectional);
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
    },
    Matmul {
        a: usize,
        b: usize,
    },
    Index {
        op: u32,
        a: usize,
        indexes: Option<usize>,
        src: Option<usize>,
        dim: u32,
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
            }
        }
        NodeKind::Matmul { a, b } => Instruction::Matmul {
            a: child_index(&index, a)?,
            b: child_index(&index, b)?,
        },
        NodeKind::Argmax { a, dim } | NodeKind::Argmin { a, dim } => Instruction::Index {
            op: u32::from(matches!(node.kind, NodeKind::Argmin { .. })),
            a: child_index(&index, a)?,
            indexes: None,
            src: None,
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
        },
        NodeKind::Cumsum { a, dim } => Instruction::Index {
            op: 2,
            a: child_index(&index, a)?,
            indexes: None,
            src: None,
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
        },
        NodeKind::IndexSelect { a, dim, indexes } => Instruction::Index {
            op: 3,
            a: child_index(&index, a)?,
            indexes: Some(child_index(&index, indexes)?),
            src: None,
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
        },
        NodeKind::Gather { a, dim, indexes } => Instruction::Index {
            op: 4,
            a: child_index(&index, a)?,
            indexes: Some(child_index(&index, indexes)?),
            src: None,
            dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
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
}

/// Physical KV layout frozen before lowering and memory planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaStateLayout {
    pub capacity: u32,
    pub dtype: DType,
    pub slots: u32,
    pub packed_rows_per_sequence: Option<u32>,
}
impl CudaStateLayout {
    pub(crate) fn validate(self) -> Result<(), String> {
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
        Ok(())
    }
}

// Capture remains disabled until the new typed ABI has hardware lifetime coverage.
pub(crate) struct CudaDecodeGraph;

#[derive(Clone)]
pub struct CudaKvSnapshot {
    pub dtype: DType,
    pub keys: Vec<Vec<u8>>,
    pub values: Vec<Vec<u8>>,
    pub key_scales: Vec<Vec<f32>>,
    pub value_scales: Vec<Vec<f32>>,
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
    pub(crate) keys: Arc<CudaBuffer<u8>>,
    pub(crate) values: Arc<CudaBuffer<u8>>,
    pub(crate) key_scales: Option<Arc<CudaBuffer<f32>>>,
    pub(crate) value_scales: Option<Arc<CudaBuffer<f32>>>,
    pub(crate) layer_elements: usize,
    pub(crate) dtype: DType,
}
impl CudaKvCache {
    pub(crate) fn try_clone(&self, device: &Arc<CudaDevice>) -> Result<Self, String> {
        fn copy<T: DeviceRepr + Send + Sync + 'static>(
            device: &Arc<CudaDevice>,
            source: &CudaBuffer<T>,
        ) -> Result<Arc<CudaBuffer<T>>, String> {
            let mut buffer =
                unsafe { device.stream.alloc::<T>(source.len()) }.map_err(|e| e.to_string())?;
            device
                .stream
                .memcpy_dtod(source, &mut buffer)
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(CudaBuffer::from_slice(buffer)))
        }
        Ok(Self {
            keys: copy(device, &self.keys)?,
            values: copy(device, &self.values)?,
            key_scales: self
                .key_scales
                .as_ref()
                .map(|s| copy(device, s))
                .transpose()?,
            value_scales: self
                .value_scales
                .as_ref()
                .map(|s| copy(device, s))
                .transpose()?,
            layer_elements: self.layer_elements,
            dtype: self.dtype,
        })
    }
}
pub struct CudaStateInvocation {
    pub sequences: Vec<CudaSequenceState>,
    pub slots: Vec<u32>,
    pub valid_lengths: Vec<u32>,
    pub capacity: u32,
    pub cache_dtype: DType,
    pub packed_rows_per_sequence: Option<u32>,
    pub(crate) cache: Option<CudaKvCache>,
    pub(crate) cursor_device: Option<Arc<cudarc::driver::CudaSlice<u32>>>,
    pub(crate) valid_device: Option<Arc<cudarc::driver::CudaSlice<u32>>>,
    pub(crate) decode_graph: Option<CudaDecodeGraph>,
}

struct InvocationFence {
    stream: Arc<cudarc::driver::CudaStream>,
    complete: bool,
}
impl Drop for InvocationFence {
    fn drop(&mut self) {
        if !self.complete {
            let _ = self.stream.synchronize();
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
        self.execute_inner(bindings, &[], None, cancelled, Some(after_gemm))
    }
    pub fn execute_stateful(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        state: &mut CudaStateInvocation,
        cancelled: &CancellationFlag,
    ) -> Result<Vec<CudaValue>, String> {
        self.prepare_state(state)?;
        let before = state.sequences.clone();
        let result = self.execute_inner(
            bindings,
            scalars,
            Some(state),
            cancelled,
            #[cfg(test)]
            None,
        );
        if result.is_err() {
            state.sequences = before;
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
    fn execute_inner(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        mut state: Option<&mut CudaStateInvocation>,
        cancelled: &CancellationFlag,
        #[cfg(test)] after_gemm: Option<&dyn Fn()>,
    ) -> Result<Vec<CudaValue>, String> {
        let resources = workspace::acquire(self.device.ordinal, &self.memory.segments)?;
        // Drop the fence before returning leases on errors or interruption.
        let mut fence = InvocationFence {
            stream: self.device.stream.clone(),
            complete: false,
        };
        let run = self.runs.fetch_add(1, Ordering::Relaxed);
        let mut values: Vec<Option<CudaValue>> = vec![None; self.program.values.len()];
        for (position, command) in self.commands.iter().enumerate() {
            if cancelled.is_cancelled() {
                return Err("operation aborted".into());
            }
            if let CommandKind::StateCopy {
                persistent,
                transaction,
                component,
                commit,
            } = &command.kind
            {
                let state = state
                    .as_deref()
                    .ok_or("execute: state copy requires state")?;
                let mut persistent = self.state_buffer(
                    state,
                    *component,
                    self.program.values[persistent.index()].decl.bytes,
                )?;
                let mut transaction = self.buffer(&resources, *transaction)?;
                if *commit {
                    self.device
                        .stream
                        .memcpy_dtod(&transaction, &mut persistent)
                } else {
                    self.device
                        .stream
                        .memcpy_dtod(&persistent, &mut transaction)
                }
                .map_err(|e| e.to_string())?;
                continue;
            }
            let Some(output_id) = command.output else {
                continue;
            };
            let meta = &self.program.values[output_id.index()];
            let output = match &command.kind {
                CommandKind::Prepare | CommandKind::StateCopy { .. } => continue,
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
                CommandKind::Kernel {
                    name,
                    args,
                    inputs,
                    scratch,
                    status,
                    state: access,
                    state_buffers,
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
                    let mut status = self.buffer(&resources, *status)?.cast::<u32>(1)?;
                    self.device
                        .stream
                        .memcpy_htod(&[0u32], &mut status)
                        .map_err(|e| e.to_string())?;
                    args.scratch[3] = status.address();
                    if name.starts_with("et_random_") {
                        args.integers[0] =
                            args.integers[0].wrapping_add(run.wrapping_mul(0x9e3779b97f4a7c15));
                    }
                    self.prepare_kernel_state(
                        access,
                        &mut args,
                        &scratch_buffers,
                        state.as_deref_mut(),
                    )?;
                    for (slot, id) in state_buffers.iter().enumerate() {
                        if let Some(id) = id {
                            args.inputs[slot + 3] = self.buffer(&resources, *id)?.address();
                        }
                    }
                    self.launch(name, &args)?;
                    let code = self
                        .device
                        .stream
                        .clone_dtoh(&status)
                        .map_err(|e| e.to_string())?[0];
                    match code {
                        0 => {}
                        1 => return Err(format!("{name}: index is out of range")),
                        2 => return Err(format!("{name}: no active targets")),
                        3 => return Err(format!("{name}: matrix is singular")),
                        _ => return Err(format!("{name}: invalid arithmetic")),
                    }
                    self.commit_kernel_state(access, &scratch_buffers, state.as_deref_mut())?;
                    output
                }
            };
            values[output_id.index()] = Some(output);
        }
        self.device
            .stream
            .synchronize()
            .map_err(|e| e.to_string())?;
        fence.complete = true;
        if cancelled.is_cancelled() {
            return Err("operation aborted".into());
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
        if args.elements == 0 {
            return Ok(());
        }
        // KV keeps causal row order within each warp while its lanes compute
        // output dimensions in parallel. Small head dimensions still need a
        // complete warp for every logical sequence.
        let work_items = if name == "et_kv_attention" {
            args.integers[5]
                .checked_mul(32)
                .ok_or("CUDA KV grid overflow")?
        } else {
            args.elements
        };
        let blocks = u32::try_from(work_items.div_ceil(256)).map_err(|_| "CUDA grid overflow")?;
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
    for command in commands {
        match &command.kind {
            CommandKind::Gemm { .. }
            | CommandKind::StateCopy { .. }
            | CommandKind::Cursor { .. } => submissions += 1,
            CommandKind::Scalar { .. } | CommandKind::LinearBias { .. } => {
                submissions += usize::from(
                    command
                        .output
                        .is_some_and(|id| program.values[id.index()].decl.bytes != 0),
                );
            }
            CommandKind::Kernel { args, state, .. } => {
                // Status reset, optional launch, and status readback.
                submissions += 2 + usize::from(args.elements != 0);
                completions += 1;
                submissions += match state {
                    StateAccess::Rotary => 1,
                    StateAccess::Kv { .. } | StateAccess::Kda { .. } | StateAccess::Conv { .. } => {
                        2
                    }
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
        CudaProgramBuilder::new(&prepared.index, state_layout, driver.legalization())?;
    driver.lower(|unit, index, _, plan| {
        let LoweringUnit::Node(dense) = unit else {
            return Err("compile: CUDA regions unsupported".into());
        };
        let node = index.node(dense).ok_or("compile: CUDA node missing")?;
        let instruction = semantic_instruction(node, index, &device, ordinal, state_cursor)?;
        builder.add(dense, node, index, instruction, plan)
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
    /// Acquires explicit state-owned storage before encoding any operations.
    pub fn prepare_state(&self, state: &mut CudaStateInvocation) -> Result<(), String> {
        if let Some(layout) = self.state_layout {
            if layout.capacity != state.capacity
                || layout.dtype != state.cache_dtype
                || layout.slots as usize != state.valid_lengths.len()
                || layout.packed_rows_per_sequence != state.packed_rows_per_sequence
            {
                return Err("execute: state layout differs from compiled layout".into());
            }
        }
        if state.slots.len() != state.sequences.len()
            || state
                .slots
                .iter()
                .any(|slot| *slot as usize >= state.valid_lengths.len())
        {
            return Err("execute: CUDA sequence slots are invalid".into());
        }
        let mut geometry = None;
        let mut layers = 0usize;
        for command in &self.commands {
            if let CommandKind::Kernel {
                state: StateAccess::Kv {
                    layer, heads, dim, ..
                },
                ..
            } = &command.kind
            {
                if geometry.is_some_and(|old| old != (*heads, *dim)) {
                    return Err("execute: CUDA KV layer geometry differs".into());
                }
                geometry = Some((*heads, *dim));
                layers = layers.max(layer.checked_add(1).ok_or("KV layer count overflow")?);
            }
        }
        let Some((heads, dim)) = geometry else {
            return Ok(());
        };
        if state.capacity == 0
            || !matches!(
                state.cache_dtype,
                DType::F32 | DType::F16 | DType::BF16 | DType::U8
            )
        {
            return Err("execute: invalid CUDA KV storage contract".into());
        }
        let rows = product(&[state.valid_lengths.len(), state.capacity as usize, heads])?;
        let layer_elements = product(&[rows, dim])?;
        let width = state.cache_dtype.size_in_bytes();
        let bytes = product(&[layers, layer_elements, width])?;
        if let Some(cache) = &state.cache {
            if cache.layer_elements != layer_elements
                || cache.dtype != state.cache_dtype
                || cache.keys.len() != bytes
                || cache.values.len() != bytes
            {
                return Err("execute: CUDA state cache layout differs".into());
            }
            return Ok(());
        }
        let mut keys = vec![
            if state.cache_dtype == DType::U8 {
                128u8
            } else {
                0
            };
            bytes
        ];
        let mut values = keys.clone();
        let scale_elements = if state.cache_dtype == DType::U8 {
            product(&[layers, rows])?
        } else {
            0
        };
        let mut key_scales = vec![0.0f32; scale_elements];
        let mut value_scales = key_scales.clone();
        let row_elements = product(&[state.capacity as usize, heads, dim])?;
        let row_bytes = product(&[row_elements, width])?;
        let row_scales = product(&[state.capacity as usize, heads])?;
        for (request, &slot) in state.slots.iter().enumerate() {
            let sequence = &state.sequences[request];
            if let Some(snapshot) = &sequence.kv_storage {
                if snapshot.dtype != state.cache_dtype
                    || snapshot.keys.len() != layers
                    || snapshot.values.len() != layers
                {
                    return Err("execute: CUDA KV snapshot layout differs".into());
                }
                for layer in 0..layers {
                    if snapshot.keys[layer].len() != row_bytes
                        || snapshot.values[layer].len() != row_bytes
                    {
                        return Err("execute: CUDA KV snapshot byte length differs".into());
                    }
                    let start = layer * layer_elements * width + slot as usize * row_bytes;
                    keys[start..start + row_bytes].copy_from_slice(&snapshot.keys[layer]);
                    values[start..start + row_bytes].copy_from_slice(&snapshot.values[layer]);
                    if state.cache_dtype == DType::U8 {
                        let ks = snapshot
                            .key_scales
                            .get(layer)
                            .filter(|s| s.len() == row_scales)
                            .ok_or("execute: missing key scales")?;
                        let vs = snapshot
                            .value_scales
                            .get(layer)
                            .filter(|s| s.len() == row_scales)
                            .ok_or("execute: missing value scales")?;
                        let start = layer * rows + slot as usize * row_scales;
                        key_scales[start..start + row_scales].copy_from_slice(ks);
                        value_scales[start..start + row_scales].copy_from_slice(vs);
                    }
                }
            } else if sequence.cursor != 0 {
                if state.cache_dtype == DType::U8 {
                    return Err(
                        "execute: quantized KV restore requires exact codes and scales".into(),
                    );
                }
                if sequence.keys.len() != layers || sequence.values.len() != layers {
                    return Err("execute: nonempty sequence has no KV snapshot".into());
                }
                for layer in 0..layers {
                    if sequence.keys[layer].len() != row_elements
                        || sequence.values[layer].len() != row_elements
                    {
                        return Err("execute: dense KV snapshot geometry differs".into());
                    }
                    let start = layer * layer_elements * width + slot as usize * row_bytes;
                    keys[start..start + row_bytes].copy_from_slice(
                        &crate::value::dense_bytes_from_host(
                            &sequence.keys[layer],
                            state.cache_dtype,
                        ),
                    );
                    values[start..start + row_bytes].copy_from_slice(
                        &crate::value::dense_bytes_from_host(
                            &sequence.values[layer],
                            state.cache_dtype,
                        ),
                    );
                }
            }
        }
        let upload = |data: &[u8]| {
            self.device
                .stream
                .clone_htod(data)
                .map(CudaBuffer::from_slice)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        };
        let upload_scales = |data: &[f32]| {
            self.device
                .stream
                .clone_htod(data)
                .map(CudaBuffer::from_slice)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        };
        state.cache = Some(CudaKvCache {
            keys: upload(&keys)?,
            values: upload(&values)?,
            key_scales: if scale_elements > 0 {
                Some(upload_scales(&key_scales)?)
            } else {
                None
            },
            value_scales: if scale_elements > 0 {
                Some(upload_scales(&value_scales)?)
            } else {
                None
            },
            layer_elements,
            dtype: state.cache_dtype,
        });
        Ok(())
    }
    pub fn readback_state(&self, state: &mut CudaStateInvocation) -> Result<(), String> {
        let Some(cache) = &state.cache else {
            return Ok(());
        };
        let keys = self
            .device
            .stream
            .clone_dtoh(cache.keys.as_ref())
            .map_err(|e| e.to_string())?;
        let values = self
            .device
            .stream
            .clone_dtoh(cache.values.as_ref())
            .map_err(|e| e.to_string())?;
        let ks = cache
            .key_scales
            .as_ref()
            .map(|s| {
                self.device
                    .stream
                    .clone_dtoh(s.as_ref())
                    .map_err(|e| e.to_string())
            })
            .transpose()?
            .unwrap_or_default();
        let vs = cache
            .value_scales
            .as_ref()
            .map(|s| {
                self.device
                    .stream
                    .clone_dtoh(s.as_ref())
                    .map_err(|e| e.to_string())
            })
            .transpose()?
            .unwrap_or_default();
        let layer_bytes = product(&[cache.layer_elements, cache.dtype.size_in_bytes()])?;
        if layer_bytes == 0 || state.valid_lengths.is_empty() {
            return Err("execute: empty KV cache geometry".into());
        }
        let layers = keys.len() / layer_bytes;
        let sequence_bytes = layer_bytes / state.valid_lengths.len();
        let sequence_scales = if ks.is_empty() {
            0
        } else {
            ks.len() / layers / state.valid_lengths.len()
        };
        for (request, &slot) in state.slots.iter().enumerate() {
            let mut snapshot = CudaKvSnapshot {
                dtype: cache.dtype,
                keys: Vec::with_capacity(layers),
                values: Vec::with_capacity(layers),
                key_scales: Vec::with_capacity(layers),
                value_scales: Vec::with_capacity(layers),
            };
            for layer in 0..layers {
                let start = layer * layer_bytes + slot as usize * sequence_bytes;
                snapshot
                    .keys
                    .push(keys[start..start + sequence_bytes].to_vec());
                snapshot
                    .values
                    .push(values[start..start + sequence_bytes].to_vec());
                if sequence_scales != 0 {
                    let start =
                        (layer * state.valid_lengths.len() + slot as usize) * sequence_scales;
                    snapshot
                        .key_scales
                        .push(ks[start..start + sequence_scales].to_vec());
                    snapshot
                        .value_scales
                        .push(vs[start..start + sequence_scales].to_vec());
                }
            }
            state.sequences[request].kv_storage = Some(snapshot);
        }
        Ok(())
    }
    fn state_buffer(
        &self,
        state: &CudaStateInvocation,
        component: StateComponent,
        bytes: usize,
    ) -> Result<CudaBuffer<u8>, String> {
        let cache = state
            .cache
            .as_ref()
            .ok_or("execute: persistent KV storage missing")?;
        let (layer, buffer) = match component {
            StateComponent::Keys(layer) => (layer, cache.keys.as_ref().clone()),
            StateComponent::Values(layer) => (layer, cache.values.as_ref().clone()),
            StateComponent::KeyScales(layer) => {
                let source = cache
                    .key_scales
                    .as_ref()
                    .ok_or("execute: key scales missing")?;
                (
                    layer,
                    source
                        .cast::<u8>(source.len().checked_mul(4).ok_or("scale bytes overflow")?)?,
                )
            }
            StateComponent::ValueScales(layer) => {
                let source = cache
                    .value_scales
                    .as_ref()
                    .ok_or("execute: value scales missing")?;
                (
                    layer,
                    source
                        .cast::<u8>(source.len().checked_mul(4).ok_or("scale bytes overflow")?)?,
                )
            }
            _ => return Err("execute: recurrent state uses host transaction staging".into()),
        };
        let start = layer.checked_mul(bytes).ok_or("state offset overflow")?;
        buffer.slice(start..start.checked_add(bytes).ok_or("state range overflow")?)
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
        state: Option<&mut CudaStateInvocation>,
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
                let cache = state
                    .cache
                    .as_ref()
                    .ok_or("execute: KV state was not prepared")?;
                let layer_bytes = product(&[cache.layer_elements, cache.dtype.size_in_bytes()])?;
                let start = product(&[*layer, layer_bytes])?;
                args.inputs[3] = cache.keys.slice(start..start + layer_bytes)?.address();
                args.inputs[4] = cache.values.slice(start..start + layer_bytes)?.address();
                let layer_scales =
                    product(&[state.valid_lengths.len(), state.capacity as usize, *heads])?;
                if cache.dtype == DType::U8 {
                    let start = product(&[*layer, layer_scales])?;
                    args.inputs[5] = cache
                        .key_scales
                        .as_ref()
                        .ok_or("key scales missing")?
                        .slice(start..start + layer_scales)?
                        .address();
                    args.inputs[6] = cache
                        .value_scales
                        .as_ref()
                        .ok_or("value scales missing")?
                        .slice(start..start + layer_scales)?
                        .address();
                }
                let mut cursors = vec![0u32; *batch];
                let mut valid = vec![0u32; *batch];
                for (request, &slot) in state.slots.iter().enumerate() {
                    for row in 0..rows {
                        let lane = slot as usize * rows + row;
                        cursors[lane] = state.sequences[request]
                            .cursor
                            .checked_add(row as u32)
                            .ok_or("KV cursor overflow")?;
                        valid[lane] = if rows == 1 {
                            state.valid_lengths[slot as usize]
                        } else {
                            u32::from(row < state.valid_lengths[slot as usize] as usize)
                        };
                    }
                }
                self.upload_scratch(&scratch[0], &valid)?;
                self.upload_scratch(&scratch[1], &cursors)?;
                args.inputs[7] = args.scratch[1];
                args.integers[0] = u64::from(state.capacity);
                args.integers[1] = u64::from(dtype_code(cache.dtype));
                args.integers[2] = rows as u64;
                args.integers[5] = state.valid_lengths.len() as u64;
                if cache.layer_elements
                    != product(&[
                        state.valid_lengths.len(),
                        state.capacity as usize,
                        *heads,
                        *dim,
                    ])?
                {
                    return Err("execute: KV physical layout differs".into());
                }
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

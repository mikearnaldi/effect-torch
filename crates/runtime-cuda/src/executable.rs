use crate::buffer::CudaBuffer;
use crate::lowering::{lowered_program, CudaLoweredProgram};
use crate::value::element_count;
use crate::workspace::{self, CudaMemorySpace, InvocationResources, CUDA_STORAGE_ALIGNMENT};
use crate::{CudaDevice, CudaValue};
use cudarc::driver::{sys, CudaGraph, CudaSlice, LaunchConfig, PushKernelArg};
use effect_torch_compiler::{
    build_executable_diagnostics, CompileOptions, CompilerDriver, CompilerWorkReport,
    DiagnosticsInput, GraphIndex, LoweringUnit, MemoryPlannerConfig, ProgramRequest,
    StateCursorSlot, ARTIFACT_ASSEMBLY_PHASE, PHYSICAL_PLANNING_PHASE, PUBLICATION_PHASE,
};
use effect_torch_graph::{Device, KvAttentionMode, Node, NodeKind, PositionOffset};
use effect_torch_runtime::{
    CancellationFlag, DType, ExecutableDiagnostics, GgmlKQuant, MemoryPlan,
};
use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

static NEXT_EXECUTABLE_ID: AtomicU64 = AtomicU64::new(1);
const BF16_GEMM_MIN_VECTORS: usize = 16;
const MUSE_GQA_PARTITIONS: usize = 128;
const MUSE_GQA_PARTIAL_FLOATS: usize = MUSE_GQA_PARTITIONS * 32 * (128 + 2);

#[derive(Clone)]
pub(super) struct MuseAttentionPreparation {
    pub(super) q: usize,
    pub(super) k: usize,
    pub(super) q_weight: usize,
    pub(super) k_weight: usize,
    pub(super) q_eps: f64,
    pub(super) k_eps: f64,
    pub(super) rope_theta: Option<f64>,
}

#[derive(Clone)]
pub(super) enum Instruction {
    Noop,
    Value(CudaValue),
    Input {
        binding: usize,
        scalar: bool,
        shape: Vec<usize>,
        dtype: DType,
    },
    StateCursor {
        tensor: bool,
        shape: Vec<usize>,
        dtype: DType,
    },
    Binary {
        op: u32,
        a: usize,
        b: usize,
        shape: Vec<usize>,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        source_dtype: DType,
        dtype: DType,
    },
    Unary {
        op: u32,
        a: usize,
        shape: Vec<usize>,
        source_dtype: DType,
        dtype: DType,
        parameter: f64,
    },
    MuseGate {
        value: usize,
        gate: usize,
        multiplier: Option<usize>,
        shape: Vec<usize>,
    },
    MuseResidualRmsNorm {
        x: usize,
        weight: Option<usize>,
        residual: usize,
        shape: Vec<usize>,
        eps: f64,
    },
    Alias {
        a: usize,
        shape: Vec<usize>,
    },
    Reindex {
        op: u32,
        a: usize,
        shape: Vec<usize>,
        input_shape: Vec<usize>,
        parameters: Vec<u64>,
        dtype: DType,
    },
    Where {
        cond: usize,
        a: usize,
        b: usize,
        shape: Vec<usize>,
        cond_shape: Vec<usize>,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        dtype: DType,
    },
    Concat {
        a: usize,
        b: usize,
        shape: Vec<usize>,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        dim: u32,
        dtype: DType,
    },
    Reduce {
        op: u32,
        a: usize,
        shape: Vec<usize>,
        input_shape: Vec<usize>,
        dims: Vec<usize>,
        dtype: DType,
    },
    Matmul {
        a: usize,
        b: usize,
        shape: Vec<usize>,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        dtype: DType,
    },
    Index {
        op: u32,
        a: usize,
        indexes: Option<usize>,
        src: Option<usize>,
        shape: Vec<usize>,
        input_shape: Vec<usize>,
        index_shape: Vec<usize>,
        dim: u32,
        dtype: DType,
    },
    RmsNorm {
        x: usize,
        weight: Option<usize>,
        shape: Vec<usize>,
        eps: f64,
        dtype: DType,
    },
    CrossEntropy {
        logits: usize,
        target: usize,
        shape: Vec<usize>,
        logits_shape: Vec<usize>,
        ignore_index: i64,
        backward: bool,
        dtype: DType,
    },
    ChunkedHeadCe {
        output: Option<u32>,
        x: usize,
        weight: usize,
        bias: usize,
        target: usize,
        gradient: Option<usize>,
        shape: Vec<usize>,
        x_shape: Vec<usize>,
        weight_shape: Vec<usize>,
        ignore_index: i64,
        target_dtype: DType,
        dtype: DType,
    },
    Conv {
        op: u32,
        x: usize,
        w: usize,
        shape: Vec<usize>,
        x_shape: Vec<usize>,
        w_shape: Vec<usize>,
        stride: u32,
        padding: u32,
        dilation: u32,
        groups: u32,
        dtype: DType,
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
        shape: Vec<usize>,
        k_width: u32,
        n_width: u32,
        dtype: DType,
    },
    QuantizedLinear {
        x: usize,
        weight: usize,
        bias: Option<usize>,
        shape: Vec<usize>,
        rows: u32,
        columns: u32,
        row_bytes: u32,
        codec: GgmlKQuant,
        pair: Option<QuantizedLinearPair>,
    },
    QuantizedEmbedding {
        indexes: usize,
        weight: usize,
        shape: Vec<usize>,
        rows: u32,
        columns: u32,
        row_bytes: u32,
        codec: GgmlKQuant,
    },
    LayerNorm {
        op: u32,
        x: usize,
        weight: usize,
        other: usize,
        shape: Vec<usize>,
        width: u32,
        rows: u32,
        eps: f64,
        dtype: DType,
    },
    Sdpa {
        op: u32,
        q: usize,
        k: usize,
        v: usize,
        g: Option<usize>,
        shape: Vec<usize>,
        q_shape: Vec<usize>,
        k_shape: Vec<usize>,
        v_shape: Vec<usize>,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        dtype: DType,
    },
    Rotary {
        x: usize,
        shape: Vec<usize>,
        seq_len: u32,
        theta: f64,
        interleaved: bool,
        backward: bool,
        cursor: bool,
        dtype: DType,
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
        shape: Vec<usize>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        dampening: f64,
        nesterov: bool,
        dtype: DType,
    },
    ShortConv {
        op: u32,
        state_layer: Option<usize>,
        x: usize,
        weight: usize,
        g: Option<usize>,
        shape: Vec<usize>,
        input_shape: Vec<usize>,
        weight_shape: Vec<usize>,
        dtype: DType,
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
        shape: Vec<usize>,
        q_shape: Vec<usize>,
        v_shape: Vec<usize>,
        scale: f64,
        dtype: DType,
    },
    LastTokenRow {
        a: usize,
        lane: usize,
        shape: Vec<usize>,
        tokens: usize,
        dtype: DType,
    },
    KvAttention {
        q: usize,
        k: usize,
        v: usize,
        shape: Vec<usize>,
        q_shape: Vec<usize>,
        k_shape: Vec<usize>,
        scale: f64,
        layer: usize,
        window: Option<usize>,
        bidirectional: bool,
        dtype: DType,
        muse: Option<MuseAttentionPreparation>,
    },
    Random {
        shape: Vec<usize>,
        dtype: DType,
        normal: bool,
        lo: f64,
        hi: f64,
        provenance: u64,
    },
    Sequence {
        shape: Vec<usize>,
        dtype: DType,
        eye: bool,
        start: f64,
        step: f64,
    },
}

#[derive(Clone)]
pub(super) struct QuantizedLinearPair {
    pub(super) output: usize,
    pub(super) weight: usize,
    pub(super) codec: GgmlKQuant,
}

impl Instruction {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Noop => "noop",
            Self::Value(_) => "value",
            Self::Input { .. } => "input",
            Self::StateCursor { .. } => "state_cursor",
            Self::Binary { .. } => "binary",
            Self::Unary { .. } => "unary",
            Self::MuseGate { .. } => "muse_gate",
            Self::MuseResidualRmsNorm { .. } => "muse_residual_rms_norm",
            Self::Alias { .. } => "alias",
            Self::Reindex { .. } => "reindex",
            Self::Where { .. } => "where",
            Self::Concat { .. } => "concat",
            Self::Reduce { .. } => "reduce",
            Self::Matmul { .. } => "matmul",
            Self::Index { .. } => "index",
            Self::RmsNorm { .. } => "rms_norm",
            Self::CrossEntropy { .. } => "cross_entropy",
            Self::ChunkedHeadCe { .. } => "chunked_head_ce",
            Self::Conv { .. } => "conv",
            Self::Linalg { .. } => "linalg",
            Self::Linear { .. } => "linear",
            Self::QuantizedLinear { .. } => "quantized_linear",
            Self::QuantizedEmbedding { .. } => "quantized_embedding",
            Self::LayerNorm { .. } => "layer_norm",
            Self::Sdpa { .. } => "sdpa",
            Self::Rotary { .. } => "rotary",
            Self::Optimizer { .. } => "optimizer",
            Self::ShortConv { .. } => "short_conv",
            Self::Kda { .. } => "kda",
            Self::LastTokenRow { .. } => "last_token_row",
            Self::KvAttention { .. } => "kv_attention",
            Self::Random { .. } => "random",
            Self::Sequence { .. } => "sequence",
        }
    }

    pub(super) fn encodes_command(&self) -> bool {
        !matches!(
            self,
            Self::Noop | Self::Value(_) | Self::Input { scalar: false, .. } | Self::Alias { .. }
        )
    }

    pub(super) fn scratch_bytes(&self) -> Result<usize, String> {
        match self {
            Self::Linalg {
                op, shape, a_shape, ..
            } => linalg_workspace_len(*op, shape, a_shape)?
                .checked_mul(std::mem::size_of::<f64>())
                .ok_or_else(|| "CUDA matrix workspace byte size overflowed usize".to_string()),
            Self::QuantizedLinear {
                shape,
                rows,
                columns,
                ..
            } => quantized_linear_scratch_layout(shape, *rows, *columns).map(|layout| layout.bytes),
            Self::KvAttention {
                q_shape,
                k_shape,
                bidirectional,
                muse: Some(_),
                ..
            } if q_shape.as_slice() == [1, 32, 1, 128]
                && k_shape.as_slice() == [1, 2, 1, 128]
                && !bidirectional =>
            {
                MUSE_GQA_PARTIAL_FLOATS
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| "CUDA Muse GQA scratch byte size overflowed usize".to_string())
            }
            _ => Ok(0),
        }
    }

    fn static_metadata(&self) -> Option<Vec<u64>> {
        match self {
            Self::Binary {
                shape,
                a_shape,
                b_shape,
                ..
            } => {
                let mut metadata = shape.iter().map(|value| *value as u64).collect::<Vec<_>>();
                for input_shape in [a_shape, b_shape] {
                    metadata.extend(std::iter::repeat_n(1, shape.len() - input_shape.len()));
                    metadata.extend(input_shape.iter().map(|value| *value as u64));
                }
                Some(metadata)
            }
            Self::Reindex {
                shape,
                input_shape,
                parameters,
                ..
            } => Some(
                shape
                    .iter()
                    .chain(input_shape)
                    .map(|value| *value as u64)
                    .chain(parameters.iter().copied())
                    .collect(),
            ),
            Self::Where {
                shape,
                cond_shape,
                a_shape,
                b_shape,
                ..
            } => {
                let mut metadata = shape.iter().map(|value| *value as u64).collect::<Vec<_>>();
                for input_shape in [cond_shape, a_shape, b_shape] {
                    metadata.extend(std::iter::repeat_n(1, shape.len() - input_shape.len()));
                    metadata.extend(input_shape.iter().map(|value| *value as u64));
                }
                Some(metadata)
            }
            Self::Concat {
                shape,
                a_shape,
                b_shape,
                ..
            } => Some(
                shape
                    .iter()
                    .chain(a_shape)
                    .chain(b_shape)
                    .map(|value| *value as u64)
                    .collect(),
            ),
            Self::Reduce {
                input_shape, dims, ..
            } => Some(
                input_shape
                    .iter()
                    .map(|value| *value as u64)
                    .chain((0..input_shape.len()).map(|axis| u64::from(dims.contains(&axis))))
                    .collect(),
            ),
            Self::Matmul {
                shape,
                a_shape,
                b_shape,
                ..
            } => {
                let mut metadata = shape.iter().map(|value| *value as u64).collect::<Vec<_>>();
                for input_shape in [a_shape, b_shape] {
                    metadata.extend(std::iter::repeat_n(1, shape.len() - input_shape.len()));
                    metadata.extend(input_shape.iter().map(|value| *value as u64));
                }
                Some(metadata)
            }
            Self::Index {
                input_shape,
                index_shape,
                ..
            } => Some(
                input_shape
                    .iter()
                    .chain(index_shape)
                    .map(|value| *value as u64)
                    .collect(),
            ),
            Self::Conv {
                shape,
                x_shape,
                w_shape,
                ..
            } => {
                let mut metadata = Vec::with_capacity(12);
                for tensor_shape in [x_shape, w_shape, shape] {
                    metadata.extend(tensor_shape.iter().map(|value| *value as u64));
                    metadata.extend(std::iter::repeat_n(1, 4 - tensor_shape.len()));
                }
                Some(metadata)
            }
            Self::Sdpa {
                q_shape,
                k_shape,
                v_shape,
                ..
            } => Some(
                q_shape
                    .iter()
                    .chain(k_shape)
                    .chain(v_shape)
                    .map(|value| *value as u64)
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// Immutable CUDA execution plan for one graph generation.
pub struct CudaExecutable {
    id: u64,
    device: Arc<CudaDevice>,
    program: Arc<CudaLoweredProgram>,
    memory: MemoryPlan<CudaMemorySpace>,
    scratch_value: Option<usize>,
    instructions: Box<[Instruction]>,
    instruction_metadata: Box<[Option<CudaBuffer<u64>>]>,
    rms_quantized: Box<[bool]>,
    gate_quantized: Box<[bool]>,
    deferred_residuals: Box<[bool]>,
    roots: Box<[usize]>,
    outputs: Box<[(Vec<usize>, DType)]>,
    diagnostics: ExecutableDiagnostics,
    compiler_work: CompilerWorkReport,
    runs: AtomicU64,
}

#[derive(Clone)]
pub struct CudaSequenceState {
    pub cursor: u32,
    pub keys: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
    pub kda_states: Vec<Vec<f64>>,
    pub conv_states: Vec<Vec<f64>>,
}

pub struct CudaStateInvocation {
    pub sequences: Vec<CudaSequenceState>,
    pub slots: Vec<u32>,
    pub valid_lengths: Vec<u32>,
    pub capacity: u32,
    pub cache_dtype: DType,
    pub packed_rows_per_sequence: Option<u32>,
    pub(crate) key_cache: Option<Arc<CudaSlice<f32>>>,
    pub(crate) value_cache: Option<Arc<CudaSlice<f32>>>,
    pub(crate) cache_layer_size: Option<usize>,
    pub(crate) cursor_device: Option<Arc<CudaSlice<u32>>>,
    pub(crate) valid_device: Option<Arc<CudaSlice<u32>>>,
    pub(crate) decode_graph: Option<CudaDecodeGraph>,
    pub(crate) prepared_kv: Option<CudaPreparedKvBuffers>,
}

pub(crate) struct CudaPreparedKvBuffers {
    keys: CudaBuffer<f32>,
    values: CudaBuffer<f32>,
    cursor: CudaBuffer<u32>,
    valid: CudaBuffer<u32>,
}

struct MovableCudaGraph {
    graph: CudaGraph,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: CUDA graph handles have no host-thread affinity. This wrapper is not
// Clone or Sync, so moving it transfers the sole owner and safe code cannot
// access the graph concurrently.
unsafe impl Send for MovableCudaGraph {}

impl MovableCudaGraph {
    fn new(graph: CudaGraph) -> Self {
        Self {
            graph,
            _not_sync: PhantomData,
        }
    }

    fn launch(&self) -> Result<(), String> {
        self.graph.launch().map_err(|error| error.to_string())
    }
}

pub(crate) struct CudaDecodeGraph {
    graph: MovableCudaGraph,
    executable_id: u64,
    binding_address: u64,
    _resources: InvocationResources,
    output: CudaValue,
}

impl CudaStateInvocation {
    fn ensure_kv_metadata(
        &mut self,
        device: &Arc<CudaDevice>,
        batch: usize,
        rows_per_sequence: u32,
    ) -> Result<(), String> {
        self.prepared_kv = None;
        let mut cursors = vec![0u32; batch];
        let mut valid = vec![0u32; batch];
        for (request, &slot) in self.slots.iter().enumerate() {
            let sequence = self
                .sequences
                .get(request)
                .ok_or_else(|| "execute: CUDA KV sequence metadata is incomplete".to_string())?;
            if rows_per_sequence == 1 {
                cursors[slot as usize] = sequence.cursor;
                valid[slot as usize] = self.valid_lengths[slot as usize];
            } else {
                for row in 0..rows_per_sequence as usize {
                    let lane = slot as usize * rows_per_sequence as usize + row;
                    cursors[lane] = sequence.cursor.saturating_add(row as u32);
                    valid[lane] = u32::from(row < self.valid_lengths[slot as usize] as usize);
                }
            }
        }
        if self
            .cursor_device
            .as_ref()
            .is_some_and(|buffer| buffer.len() == batch)
        {
            device
                .stream
                .memcpy_htod(
                    &cursors,
                    Arc::get_mut(
                        self.cursor_device
                            .as_mut()
                            .expect("cursor metadata length was checked"),
                    )
                    .ok_or_else(|| "CUDA cursor metadata is still borrowed".to_string())?,
                )
                .map_err(|error| error.to_string())?;
        } else {
            self.cursor_device = Some(Arc::new(
                device
                    .stream
                    .clone_htod(&cursors)
                    .map_err(|error| error.to_string())?,
            ));
        }
        if self
            .valid_device
            .as_ref()
            .is_some_and(|buffer| buffer.len() == batch)
        {
            device
                .stream
                .memcpy_htod(
                    &valid,
                    Arc::get_mut(
                        self.valid_device
                            .as_mut()
                            .expect("validity metadata length was checked"),
                    )
                    .ok_or_else(|| "CUDA validity metadata is still borrowed".to_string())?,
                )
                .map_err(|error| error.to_string())?;
        } else {
            self.valid_device = Some(Arc::new(
                device
                    .stream
                    .clone_htod(&valid)
                    .map_err(|error| error.to_string())?,
            ));
        }
        Ok(())
    }

    fn ensure_kv_cache(
        &mut self,
        device: &Arc<CudaDevice>,
        layer_size: usize,
    ) -> Result<(), String> {
        if self.key_cache.is_some() {
            if self.cache_layer_size != Some(layer_size) {
                return Err("execute: CUDA KV cache geometry changed during invocation".to_string());
            }
            return Ok(());
        }
        let layers = self
            .sequences
            .first()
            .map_or(0, |sequence| sequence.keys.len());
        let row_size = layer_size / self.valid_lengths.len();
        if self.sequences.iter().any(|sequence| {
            sequence.keys.len() != layers
                || sequence.values.len() != layers
                || sequence.keys.iter().any(|layer| {
                    layer.len() != row_size && !(sequence.cursor == 0 && layer.is_empty())
                })
                || sequence.values.iter().any(|layer| {
                    layer.len() != row_size && !(sequence.cursor == 0 && layer.is_empty())
                })
        }) {
            return Err("execute: CUDA KV cache geometry is invalid".to_string());
        }
        let total = layer_size
            .checked_mul(layers)
            .ok_or_else(|| "execute: CUDA KV cache size overflowed".to_string())?;
        let total_u32 = checked_len(total)?;
        let mut keys =
            unsafe { device.stream.alloc::<f32>(total) }.map_err(|error| error.to_string())?;
        let mut values =
            unsafe { device.stream.alloc::<f32>(total) }.map_err(|error| error.to_string())?;
        if total != 0 {
            let dtype = dtype_code(DType::F32);
            for buffer in [&mut keys, &mut values] {
                let mut launch = device.stream.launch_builder(&device.f32.fill);
                launch.arg(buffer);
                launch.arg(&0.0f32);
                launch.arg(&total_u32);
                launch.arg(&dtype);
                unsafe { launch.launch(LaunchConfig::for_num_elems(total_u32)) }
                    .map_err(|error| error.to_string())?;
            }
        }
        for (request, sequence) in self.sequences.iter().enumerate() {
            if sequence.cursor == 0 {
                continue;
            }
            let slot = self.slots[request] as usize;
            for layer in 0..layers {
                let start = layer * layer_size + slot * row_size;
                let end = start + row_size;
                let host_keys = sequence.keys[layer]
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                let host_values = sequence.values[layer]
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>();
                device
                    .stream
                    .memcpy_htod(&host_keys, &mut keys.slice_mut(start..end))
                    .map_err(|error| error.to_string())?;
                device
                    .stream
                    .memcpy_htod(&host_values, &mut values.slice_mut(start..end))
                    .map_err(|error| error.to_string())?;
            }
        }
        self.key_cache = Some(Arc::new(keys));
        self.value_cache = Some(Arc::new(values));
        self.cache_layer_size = Some(layer_size);
        Ok(())
    }

    fn readback_kv_cache(&mut self, device: &Arc<CudaDevice>) -> Result<(), String> {
        let Some(layer_size) = self.cache_layer_size else {
            return Ok(());
        };
        let keys = device
            .stream
            .clone_dtoh(
                self.key_cache
                    .as_ref()
                    .expect("cache geometry requires keys")
                    .as_ref(),
            )
            .map_err(|error| error.to_string())?;
        let values = device
            .stream
            .clone_dtoh(
                self.value_cache
                    .as_ref()
                    .expect("cache geometry requires values")
                    .as_ref(),
            )
            .map_err(|error| error.to_string())?;
        let row_size = layer_size / self.valid_lengths.len();
        let layers = keys.len() / layer_size;
        for (request, sequence) in self.sequences.iter_mut().enumerate() {
            if sequence.keys.len() != layers
                || sequence.values.len() != layers
                || sequence.keys.iter().any(|layer| layer.len() != row_size)
                || sequence.values.iter().any(|layer| layer.len() != row_size)
            {
                sequence.keys = vec![vec![0.0; row_size]; layers];
                sequence.values = vec![vec![0.0; row_size]; layers];
            }
            let slot = self.slots[request] as usize;
            for layer in 0..sequence.keys.len() {
                let start = layer * layer_size + slot * row_size;
                let end = start + row_size;
                for (output, value) in sequence.keys[layer].iter_mut().zip(&keys[start..end]) {
                    *output = f64::from(*value);
                }
                for (output, value) in sequence.values[layer].iter_mut().zip(&values[start..end]) {
                    *output = f64::from(*value);
                }
            }
        }
        Ok(())
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
        self.program.instructions.len()
    }

    pub fn tensor_input(&self, binding: usize) -> Option<(Vec<usize>, DType)> {
        self.instructions
            .iter()
            .find_map(|instruction| match instruction {
                Instruction::Input {
                    binding: input,
                    scalar: false,
                    shape,
                    dtype,
                } if *input == binding => Some((shape.clone(), *dtype)),
                _ => None,
            })
    }

    pub fn execute(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        cancelled: &CancellationFlag,
    ) -> Result<Vec<CudaValue>, String> {
        self.execute_inner(bindings, scalars, None, cancelled)
    }

    pub fn execute_stateful(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        state: &mut CudaStateInvocation,
        cancelled: &CancellationFlag,
    ) -> Result<Vec<CudaValue>, String> {
        self.execute_inner(bindings, scalars, Some(state), cancelled)
    }

    pub fn execute_stateful_greedy(
        &self,
        bindings: &[CudaValue],
        input_tokens: &[u32],
        state: &mut CudaStateInvocation,
        cancelled: &CancellationFlag,
    ) -> Result<Option<u32>, String> {
        let Some(output) =
            self.execute_stateful_graphed(bindings, input_tokens, state, cancelled)?
        else {
            return Ok(None);
        };
        let sampled = output.greedy_argmax()?;
        if cancelled.is_cancelled() {
            return Err("operation aborted".to_string());
        }
        Ok(Some(sampled))
    }

    pub fn execute_stateful_graphed(
        &self,
        bindings: &[CudaValue],
        input_tokens: &[u32],
        state: &mut CudaStateInvocation,
        cancelled: &CancellationFlag,
    ) -> Result<Option<CudaValue>, String> {
        if !self.prepare_decode_graph_state(bindings, input_tokens, state)? {
            return Ok(None);
        }
        let binding_address = bindings[0].storage_address();
        if let Some(graph) = state.decode_graph.take() {
            if graph.executable_id == self.id && graph.binding_address == binding_address {
                if cancelled.is_cancelled() {
                    state.decode_graph = Some(graph);
                    return Err("operation aborted".to_string());
                }
                graph.graph.launch()?;
                let output = graph.output.clone();
                state.decode_graph = Some(graph);
                return Ok(Some(output));
            }
        }

        let resources = workspace::acquire(self.device.ordinal, &self.memory.segments)?;
        self.device
            .stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|error| error.to_string())?;
        let outputs = self.execute_with_resources(
            bindings,
            &[],
            Some(state),
            cancelled,
            &resources,
            false,
            true,
            false,
        );
        let captured = self.device.stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        let outputs = outputs?;
        let graph = captured
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "CUDA decode graph capture produced no graph".to_string())?;
        if outputs.len() != 1 {
            return Err("CUDA decode graph requires one output".to_string());
        }
        graph.upload().map_err(|error| error.to_string())?;
        graph.launch().map_err(|error| error.to_string())?;
        let captured = CudaDecodeGraph {
            graph: MovableCudaGraph::new(graph),
            executable_id: self.id,
            binding_address,
            _resources: resources,
            output: outputs
                .into_iter()
                .next()
                .expect("output length was checked"),
        };
        let output = captured.output.clone();
        state.decode_graph = Some(captured);
        Ok(Some(output))
    }

    fn prepare_decode_graph_state(
        &self,
        bindings: &[CudaValue],
        input_tokens: &[u32],
        state: &mut CudaStateInvocation,
    ) -> Result<bool, String> {
        if std::env::var_os("EFFECT_TORCH_CUDA_GRAPH_DEBUG").is_some() {
            eprintln!(
                "CUDA graph candidate: bindings={} elements={:?} outputs={:?} sequences={} slots={:?} valid={:?} packed={:?} dtype={} state_cursor={} cursor_rotary={} recurrent={}",
                bindings.len(),
                bindings.first().map(|binding| element_count(binding.shape())),
                self.outputs,
                state.sequences.len(),
                state.slots,
                state.valid_lengths,
                state.packed_rows_per_sequence,
                state.cache_dtype.name(),
                self.instructions
                    .iter()
                    .any(|instruction| matches!(instruction, Instruction::StateCursor { .. })),
                self.instructions.iter().any(
                    |instruction| matches!(instruction, Instruction::Rotary { cursor: true, .. })
                ),
                self.instructions.iter().any(|instruction| matches!(
                    instruction,
                    Instruction::Kda { .. } | Instruction::ShortConv { .. }
                )),
            );
        }
        if std::env::var("EFFECT_TORCH_CUDA_GRAPH").is_ok_and(|value| value == "false")
            || bindings.len() != 1
            || input_tokens.len() != 1
            || element_count(bindings[0].shape())? != 1
            || self.outputs.as_ref() != [(vec![202_048], DType::F32)]
            || state.sequences.len() != 1
            || state.slots != [0]
            || state.valid_lengths != [1]
            || state.packed_rows_per_sequence.is_some()
            || self.instructions.iter().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::StateCursor { .. }
                        | Instruction::Kda { .. }
                        | Instruction::ShortConv { .. }
                        | Instruction::Rotary { cursor: true, .. }
                )
            })
        {
            state.decode_graph = None;
            return Ok(false);
        }
        let embedding_rows = self
            .instructions
            .iter()
            .find_map(|instruction| match instruction {
                Instruction::QuantizedEmbedding { rows, .. } => Some(*rows),
                _ => None,
            });
        if embedding_rows.is_none_or(|rows| input_tokens[0] >= rows) {
            return Err(format!(
                "quantized_embedding: index {} is outside 0..{}",
                input_tokens[0],
                embedding_rows.unwrap_or(0)
            ));
        }
        let Some((q_shape, k_shape)) =
            self.instructions
                .iter()
                .find_map(|instruction| match instruction {
                    Instruction::KvAttention {
                        q_shape,
                        k_shape,
                        muse: Some(_),
                        ..
                    } => Some((q_shape, k_shape)),
                    _ => None,
                })
        else {
            state.decode_graph = None;
            return Ok(false);
        };
        if q_shape.as_slice() != [1, 32, 1, 128] || k_shape.as_slice() != [1, 2, 1, 128] {
            state.decode_graph = None;
            return Ok(false);
        }
        let cache_size = state.capacity as usize * k_shape[1] * k_shape[3];
        state.ensure_kv_metadata(&self.device, 1, 1)?;
        state.ensure_kv_cache(&self.device, cache_size)?;
        state.prepared_kv = Some(CudaPreparedKvBuffers {
            keys: CudaBuffer::from_arc_slice(Arc::clone(
                state.key_cache.as_ref().expect("KV cache was prepared"),
            )),
            values: CudaBuffer::from_arc_slice(Arc::clone(
                state
                    .value_cache
                    .as_ref()
                    .expect("KV value cache was prepared"),
            )),
            cursor: CudaBuffer::from_arc_slice(Arc::clone(
                state
                    .cursor_device
                    .as_ref()
                    .expect("KV cursor was prepared"),
            )),
            valid: CudaBuffer::from_arc_slice(Arc::clone(
                state
                    .valid_device
                    .as_ref()
                    .expect("KV validity was prepared"),
            )),
        });
        Ok(true)
    }

    pub fn readback_state(&self, state: &mut CudaStateInvocation) -> Result<(), String> {
        state.readback_kv_cache(&self.device)
    }

    fn planned_value(
        &self,
        resources: &InvocationResources,
        value_index: usize,
        shape: &[usize],
        dtype: DType,
        exact_i64: bool,
    ) -> Result<CudaValue, String> {
        let len = element_count(shape)?;
        let location = self
            .memory
            .locations
            .get(value_index)
            .ok_or_else(|| format!("CUDA value location {value_index} is out of range"))?;
        let element_bytes = if dtype == DType::F32 {
            std::mem::size_of::<f32>()
        } else {
            std::mem::size_of::<f64>()
        };
        let dense_bytes = len
            .checked_mul(element_bytes)
            .ok_or_else(|| "CUDA planned value byte size overflowed usize".to_string())?;
        let buffer: CudaBuffer<u8> = resources.buffer(location, 0, dense_bytes)?;
        let i64_buffer = if exact_i64 {
            let offset = len
                .checked_mul(std::mem::size_of::<f64>())
                .ok_or_else(|| "CUDA i64 sidecar offset overflowed usize".to_string())?;
            Some(resources.buffer(location, offset, len)?)
        } else {
            None
        };
        Ok(CudaValue::from_planned_buffers(
            Arc::clone(&self.device),
            shape.to_vec(),
            dtype,
            buffer,
            i64_buffer,
        ))
    }

    fn scratch_buffer<T: Send + Sync + 'static>(
        &self,
        resources: &InvocationResources,
        byte_offset: usize,
        len: usize,
    ) -> Result<CudaBuffer<T>, String> {
        let scratch = self
            .scratch_value
            .ok_or_else(|| "CUDA executable has no planned scratch storage".to_string())?;
        let location = self
            .memory
            .locations
            .get(scratch)
            .ok_or_else(|| "CUDA scratch location is out of range".to_string())?;
        resources.buffer(location, byte_offset, len)
    }

    fn fill_planned_value(
        &self,
        resources: &InvocationResources,
        value_index: usize,
        shape: &[usize],
        dtype: DType,
        value: f64,
    ) -> Result<CudaValue, String> {
        let len = element_count(shape)?;
        let len_u32 = checked_len(len)?;
        let output =
            self.planned_value(resources, value_index, shape, dtype, dtype == DType::I64)?;
        if len != 0 {
            let dtype_code = dtype_code(dtype);
            if dtype == DType::F32 {
                let value = value as f32;
                let mut fill = self.device.stream.launch_builder(&self.device.f32.fill);
                fill.arg(output.buffer.as_ref());
                fill.arg(&value);
                fill.arg(&len_u32);
                fill.arg(&dtype_code);
                unsafe { fill.launch(LaunchConfig::for_num_elems(len_u32)) }
                    .map_err(|error| error.to_string())?;
            } else {
                let mut fill = self.device.stream.launch_builder(&self.device.fill_f64);
                fill.arg(output.buffer.as_ref());
                fill.arg(&value);
                fill.arg(&len_u32);
                fill.arg(&dtype_code);
                unsafe { fill.launch(LaunchConfig::for_num_elems(len_u32)) }
                    .map_err(|error| error.to_string())?;
            }
            if dtype == DType::I64 {
                let mut cast = self.device.stream.launch_builder(&self.device.cast_to_i64);
                cast.arg(output.buffer.as_ref());
                cast.arg(output.i64_buffer()?);
                cast.arg(output.buffer.as_ref());
                cast.arg(&len_u32);
                unsafe { cast.launch(LaunchConfig::for_num_elems(len_u32)) }
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(output)
    }

    fn upload_planned_value(
        &self,
        resources: &InvocationResources,
        value_index: usize,
        shape: &[usize],
        dtype: DType,
        values: &[f64],
    ) -> Result<CudaValue, String> {
        let len = element_count(shape)?;
        if values.len() != len {
            return Err(format!(
                "CUDA host value count {} does not match shape element count {len}",
                values.len()
            ));
        }
        let location = self
            .memory
            .locations
            .get(value_index)
            .ok_or_else(|| format!("CUDA value location {value_index} is out of range"))?;
        let element_bytes = if dtype == DType::F32 {
            std::mem::size_of::<f32>()
        } else {
            std::mem::size_of::<f64>()
        };
        let dense_bytes = len
            .checked_mul(element_bytes)
            .ok_or_else(|| "CUDA planned value byte size overflowed usize".to_string())?;
        let buffer: CudaBuffer<u8> = resources.buffer(location, 0, dense_bytes)?;
        if dtype == DType::F32 {
            let mut typed = buffer.cast::<f32>(len)?;
            self.device
                .stream
                .memcpy_htod(
                    &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                    &mut typed,
                )
                .map_err(|error| error.to_string())?;
        } else {
            let mut typed = buffer.cast::<f64>(len)?;
            self.device
                .stream
                .memcpy_htod(values, &mut typed)
                .map_err(|error| error.to_string())?;
        }
        let i64_buffer = if dtype == DType::I64 {
            let offset = len
                .checked_mul(std::mem::size_of::<f64>())
                .ok_or_else(|| "CUDA i64 sidecar offset overflowed usize".to_string())?;
            let mut sidecar = resources.buffer(location, offset, len)?;
            self.device
                .stream
                .memcpy_htod(
                    &values.iter().map(|value| *value as i64).collect::<Vec<_>>(),
                    &mut sidecar,
                )
                .map_err(|error| error.to_string())?;
            Some(sidecar)
        } else {
            None
        };
        Ok(CudaValue::from_planned_buffers(
            Arc::clone(&self.device),
            shape.to_vec(),
            dtype,
            buffer,
            i64_buffer,
        ))
    }

    fn execute_inner(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        state: Option<&mut CudaStateInvocation>,
        cancelled: &CancellationFlag,
    ) -> Result<Vec<CudaValue>, String> {
        let resources = workspace::acquire(self.device.ordinal, &self.memory.segments)?;
        self.execute_with_resources(
            bindings, scalars, state, cancelled, &resources, true, false, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_with_resources(
        &self,
        bindings: &[CudaValue],
        scalars: &[f64],
        mut state: Option<&mut CudaStateInvocation>,
        cancelled: &CancellationFlag,
        resources: &InvocationResources,
        synchronize: bool,
        mut kv_prepared: bool,
        validate_embedding_indexes: bool,
    ) -> Result<Vec<CudaValue>, String> {
        let run = self.runs.fetch_add(1, Ordering::Relaxed);
        let mut values: Vec<Option<CudaValue>> = vec![None; self.instructions.len()];
        let mut quantized_input = None;
        for (index, instruction) in self.instructions.iter().enumerate() {
            if !synchronize
                && std::env::var_os("EFFECT_TORCH_CUDA_GRAPH_DEBUG").is_some()
                && instruction.encodes_command()
            {
                eprintln!(
                    "CUDA graph capture instruction {index}: {}",
                    instruction.name()
                );
            }
            if cancelled.is_cancelled() {
                return Err("operation aborted".to_string());
            }
            if matches!(instruction, Instruction::Noop) {
                continue;
            }
            if !matches!(instruction, Instruction::QuantizedLinear { .. })
                && instruction.scratch_bytes()? != 0
            {
                quantized_input = None;
            }
            let value = match instruction {
                Instruction::Noop => unreachable!("no-op instructions are skipped"),
                Instruction::Value(value) => value.clone(),
                Instruction::Input {
                    binding,
                    scalar,
                    shape,
                    dtype,
                } => {
                    if *scalar {
                        let scalar = scalars.get(*binding).ok_or_else(|| {
                            format!("execute: missing CUDA scalar binding {binding}")
                        })?;
                        self.fill_planned_value(&resources, index, shape, *dtype, *scalar)?
                    } else {
                        let value = bindings.get(*binding).ok_or_else(|| {
                            format!("execute: missing CUDA tensor binding {binding}")
                        })?;
                        if value.shape() != shape || value.dtype() != *dtype {
                            return Err(format!(
                                "execute: CUDA tensor binding {binding} has metadata {}:{:?}, expected {}:{shape:?}",
                                value.dtype().name(),
                                value.shape(),
                                dtype.name()
                            ));
                        }
                        value.clone()
                    }
                }
                Instruction::StateCursor {
                    tensor,
                    shape,
                    dtype,
                } => {
                    let state = state
                        .as_deref()
                        .ok_or_else(|| "execute: state cursor requires decode state".to_string())?;
                    if *tensor {
                        let mut cursors = vec![0.0; element_count(shape)?];
                        let rows = state.packed_rows_per_sequence.unwrap_or(1) as usize;
                        for (request, &slot) in state.slots.iter().enumerate() {
                            let cursor = state.sequences[request].cursor;
                            for offset in 0..rows {
                                let index = slot as usize * rows + offset;
                                if let Some(value) = cursors.get_mut(index) {
                                    *value = f64::from(cursor.saturating_add(offset as u32));
                                }
                            }
                        }
                        self.upload_planned_value(&resources, index, shape, *dtype, &cursors)?
                    } else {
                        let cursor = state
                            .sequences
                            .first()
                            .map_or(0.0, |sequence| f64::from(sequence.cursor));
                        self.fill_planned_value(&resources, index, shape, *dtype, cursor)?
                    }
                }
                Instruction::Binary {
                    op,
                    a,
                    b,
                    shape,
                    source_dtype,
                    dtype,
                    ..
                } => {
                    let a = values[*a].as_ref().ok_or_else(|| {
                        "CUDA executable referenced an unavailable input".to_string()
                    })?;
                    let b = values[*b].as_ref().ok_or_else(|| {
                        "CUDA executable referenced an unavailable input".to_string()
                    })?;
                    let len = element_count(shape)?;
                    let len_u32 = u32::try_from(len).map_err(|_| {
                        format!("CUDA element count {len} exceeds the first-slice limit")
                    })?;
                    let exact_i64 =
                        *source_dtype == DType::I64 && a.has_i64_buffer() && b.has_i64_buffer();
                    let output = self.planned_value(
                        &resources,
                        index,
                        shape,
                        *dtype,
                        *dtype == DType::I64 && exact_i64,
                    )?;
                    if len != 0 {
                        let rank = u32::try_from(shape.len())
                            .map_err(|_| "CUDA tensor rank exceeds u32".to_string())?;
                        let metadata = self.instruction_metadata[index]
                            .as_ref()
                            .expect("binary metadata is prepared at compilation");
                        if exact_i64 {
                            if output.has_i64_buffer() {
                                let mut launch =
                                    self.device.stream.launch_builder(&self.device.binary_i64);
                                launch.arg(op);
                                launch.arg(a.i64_buffer()?);
                                launch.arg(b.i64_buffer()?);
                                launch.arg(output.i64_buffer()?);
                                launch.arg(output.buffer.as_ref());
                                launch.arg(&len_u32);
                                launch.arg(&rank);
                                launch.arg(metadata);
                                unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                    .map_err(|error| error.to_string())?;
                            } else {
                                let mut launch =
                                    self.device.stream.launch_builder(&self.device.compare_i64);
                                launch.arg(op);
                                launch.arg(a.i64_buffer()?);
                                launch.arg(b.i64_buffer()?);
                                launch.arg(output.buffer.as_ref());
                                launch.arg(&len_u32);
                                launch.arg(&rank);
                                launch.arg(metadata);
                                unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                    .map_err(|error| error.to_string())?;
                            }
                        } else {
                            let dtype_code = dtype_code(*dtype);
                            let function = if *source_dtype == DType::F32 {
                                &self.device.f32.binary
                            } else {
                                &self.device.binary_f64
                            };
                            let mut launch = self.device.stream.launch_builder(function);
                            launch.arg(op);
                            launch.arg(a.buffer.as_ref());
                            launch.arg(b.buffer.as_ref());
                            launch.arg(output.buffer.as_ref());
                            launch.arg(&len_u32);
                            launch.arg(&rank);
                            launch.arg(metadata);
                            launch.arg(&dtype_code);
                            unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                .map_err(|error| error.to_string())?;
                        }
                    }
                    output
                }
                Instruction::Unary {
                    op,
                    a,
                    shape,
                    source_dtype,
                    dtype,
                    parameter,
                } => {
                    let a = values[*a].as_ref().ok_or_else(|| {
                        "CUDA executable referenced an unavailable input".to_string()
                    })?;
                    let len = element_count(shape)?;
                    let len_u32 = u32::try_from(len).map_err(|_| {
                        format!("CUDA element count {len} exceeds the first-slice limit")
                    })?;
                    let exact_i64 = *source_dtype == DType::I64 && a.has_i64_buffer();
                    let output_exact_i64 =
                        *dtype == DType::I64 && (exact_i64 || *source_dtype != DType::I64);
                    let output =
                        self.planned_value(&resources, index, shape, *dtype, output_exact_i64)?;
                    if len != 0 {
                        if exact_i64 && *dtype == DType::I64 {
                            let mut launch =
                                self.device.stream.launch_builder(&self.device.unary_i64);
                            launch.arg(op);
                            launch.arg(a.i64_buffer()?);
                            launch.arg(output.i64_buffer()?);
                            launch.arg(output.buffer.as_ref());
                            launch.arg(&len_u32);
                            unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                .map_err(|error| error.to_string())?;
                        } else if exact_i64 {
                            let dtype = dtype_code(*dtype);
                            let mut launch = self
                                .device
                                .stream
                                .launch_builder(&self.device.cast_from_i64);
                            launch.arg(a.i64_buffer()?);
                            launch.arg(output.buffer.as_ref());
                            launch.arg(&len_u32);
                            launch.arg(&dtype);
                            unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                .map_err(|error| error.to_string())?;
                        } else if *dtype == DType::I64 && *source_dtype != DType::I64 {
                            let mut launch =
                                self.device.stream.launch_builder(&self.device.cast_to_i64);
                            launch.arg(a.buffer.as_ref());
                            launch.arg(output.i64_buffer()?);
                            launch.arg(output.buffer.as_ref());
                            launch.arg(&len_u32);
                            unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                .map_err(|error| error.to_string())?;
                        } else {
                            let dtype_code = dtype_code(*dtype);
                            if *source_dtype == DType::F32 && *dtype == DType::F32 {
                                let parameter = *parameter as f32;
                                let mut launch =
                                    self.device.stream.launch_builder(&self.device.f32.unary);
                                launch.arg(op);
                                launch.arg(a.buffer.as_ref());
                                launch.arg(output.buffer.as_ref());
                                launch.arg(&len_u32);
                                launch.arg(&parameter);
                                launch.arg(&dtype_code);
                                unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                    .map_err(|error| error.to_string())?;
                            } else {
                                let mut launch =
                                    self.device.stream.launch_builder(&self.device.unary_f64);
                                launch.arg(op);
                                launch.arg(a.buffer.as_ref());
                                launch.arg(output.buffer.as_ref());
                                launch.arg(&len_u32);
                                launch.arg(parameter);
                                launch.arg(&dtype_code);
                                unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                    .map_err(|error| error.to_string())?;
                            }
                        }
                    }
                    output
                }
                Instruction::MuseGate {
                    value: input,
                    gate,
                    multiplier,
                    shape,
                } => {
                    let input = value(&values, *input)?;
                    if self.gate_quantized[index] {
                        input.clone()
                    } else {
                        let gate = value(&values, *gate)?;
                        let has_multiplier = u32::from(multiplier.is_some());
                        let multiplier = multiplier
                            .map(|index| value(&values, index))
                            .transpose()?
                            .unwrap_or(input);
                        let len = element_count(shape)?;
                        let len_u32 = checked_len(len)?;
                        let output =
                            self.planned_value(&resources, index, shape, DType::F32, false)?;
                        if len != 0 {
                            let mut launch = self
                                .device
                                .stream
                                .launch_builder(&self.device.f32.muse_gate);
                            launch.arg(input.buffer.as_ref());
                            launch.arg(gate.buffer.as_ref());
                            launch.arg(multiplier.buffer.as_ref());
                            launch.arg(output.buffer.as_ref());
                            launch.arg(&len_u32);
                            launch.arg(&has_multiplier);
                            unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                .map_err(|error| error.to_string())?;
                        }
                        output
                    }
                }
                Instruction::MuseResidualRmsNorm {
                    x,
                    weight,
                    residual,
                    shape,
                    eps,
                } => {
                    let x = value(&values, *x)?;
                    let residual = value(&values, *residual)?;
                    let weight_value = weight
                        .map(|index| value(&values, index))
                        .transpose()?
                        .unwrap_or(x);
                    let weighted = u32::from(weight.is_some());
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let width = checked_len(*shape.last().ok_or_else(|| {
                        "fused residual RMSNorm requires a non-scalar input".to_string()
                    })?)?;
                    let rows = len_u32 / width;
                    let eps = *eps as f32;
                    let output = self.planned_value(&resources, index, shape, DType::F32, false)?;
                    if len != 0 && !self.deferred_residuals[index] {
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.f32.muse_residual_rms_norm);
                        launch.arg(x.buffer.as_ref());
                        launch.arg(weight_value.buffer.as_ref());
                        launch.arg(residual.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&width);
                        launch.arg(&weighted);
                        launch.arg(&eps);
                        unsafe {
                            launch.launch(LaunchConfig {
                                grid_dim: (rows, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }
                        .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Alias { a, shape } => values[*a]
                    .as_ref()
                    .ok_or_else(|| "CUDA executable referenced an unavailable input".to_string())?
                    .with_shape(shape.clone()),
                Instruction::Reindex {
                    op,
                    a,
                    shape,
                    dtype,
                    ..
                } => {
                    let a = value(&values, *a)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let rank = u32::try_from(shape.len())
                        .map_err(|_| "CUDA tensor rank exceeds u32".to_string())?;
                    let metadata = self.instruction_metadata[index]
                        .as_ref()
                        .expect("reindex metadata is prepared at compilation");
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype_code = dtype_code(*dtype);
                        let function = if *dtype == DType::F32 {
                            &self.device.f32.reindex
                        } else {
                            &self.device.reindex_f64
                        };
                        let mut launch = self.device.stream.launch_builder(function);
                        launch.arg(op);
                        launch.arg(a.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&rank);
                        launch.arg(metadata);
                        launch.arg(&dtype_code);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Where {
                    cond,
                    a,
                    b,
                    shape,
                    dtype,
                    ..
                } => {
                    let cond = value(&values, *cond)?;
                    let a = value(&values, *a)?;
                    let b = value(&values, *b)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let rank = u32::try_from(shape.len())
                        .map_err(|_| "CUDA tensor rank exceeds u32".to_string())?;
                    let metadata = self.instruction_metadata[index]
                        .as_ref()
                        .expect("where metadata is prepared at compilation");
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.where_f64);
                        launch.arg(cond.buffer.as_ref());
                        launch.arg(a.buffer.as_ref());
                        launch.arg(b.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&rank);
                        launch.arg(metadata);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Concat {
                    a,
                    b,
                    shape,
                    dim,
                    dtype,
                    ..
                } => {
                    let a = value(&values, *a)?;
                    let b = value(&values, *b)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let rank = u32::try_from(shape.len())
                        .map_err(|_| "CUDA tensor rank exceeds u32".to_string())?;
                    let metadata = self.instruction_metadata[index]
                        .as_ref()
                        .expect("concat metadata is prepared at compilation");
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.concat_f64);
                        launch.arg(a.buffer.as_ref());
                        launch.arg(b.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&rank);
                        launch.arg(dim);
                        launch.arg(metadata);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Reduce {
                    op,
                    a,
                    shape,
                    input_shape,
                    dtype,
                    ..
                } => {
                    let a = value(&values, *a)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let rank = u32::try_from(input_shape.len())
                        .map_err(|_| "CUDA tensor rank exceeds u32".to_string())?;
                    let metadata = self.instruction_metadata[index]
                        .as_ref()
                        .expect("reduce metadata is prepared at compilation");
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.reduce_f64);
                        launch.arg(op);
                        launch.arg(a.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&rank);
                        launch.arg(metadata);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Matmul {
                    a, b, shape, dtype, ..
                } => {
                    let a = value(&values, *a)?;
                    let b = value(&values, *b)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let rank = u32::try_from(shape.len())
                        .map_err(|_| "CUDA tensor rank exceeds u32".to_string())?;
                    let metadata = self.instruction_metadata[index]
                        .as_ref()
                        .expect("matmul metadata is prepared at compilation");
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.matmul_f64);
                        launch.arg(a.buffer.as_ref());
                        launch.arg(b.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&rank);
                        launch.arg(metadata);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Index {
                    op,
                    a,
                    indexes,
                    src,
                    shape,
                    input_shape,
                    index_shape,
                    dim,
                    dtype,
                } => {
                    let a = value(&values, *a)?;
                    let indexes = indexes
                        .map(|index| value(&values, index))
                        .transpose()?
                        .unwrap_or(a);
                    let src = src
                        .map(|index| value(&values, index))
                        .transpose()?
                        .unwrap_or(a);
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let src_len = checked_len(element_count(index_shape)?)?;
                    let rank = u32::try_from(input_shape.len())
                        .map_err(|_| "CUDA tensor rank exceeds u32".to_string())?;
                    let metadata = self.instruction_metadata[index]
                        .as_ref()
                        .expect("index metadata is prepared at compilation");
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.index_f64);
                        launch.arg(op);
                        launch.arg(a.buffer.as_ref());
                        launch.arg(indexes.buffer.as_ref());
                        launch.arg(src.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&src_len);
                        launch.arg(&rank);
                        launch.arg(dim);
                        launch.arg(metadata);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::RmsNorm {
                    x,
                    weight,
                    shape,
                    eps,
                    dtype,
                } => {
                    let x = value(&values, *x)?;
                    if self.rms_quantized[index] {
                        x.with_shape(shape.clone())
                    } else {
                        let weight = weight.map(|index| value(&values, index)).transpose()?;
                        let len = element_count(shape)?;
                        let len_u32 = checked_len(len)?;
                        let width = u32::try_from(shape.last().copied().unwrap_or(1))
                            .map_err(|_| "CUDA RMS norm width exceeds u32".to_string())?;
                        let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                        if len != 0 {
                            let dtype_code = dtype_code(*dtype);
                            let weighted = u32::from(weight.is_some());
                            let weight = weight.unwrap_or(x);
                            let wide = width > 512;
                            let eps_f32 = *eps as f32;
                            let function = match (*dtype == DType::F32, wide) {
                                (true, true) => &self.device.f32.rms_norm_block,
                                (true, false) => &self.device.f32.rms_norm,
                                (false, true) => &self.device.rms_norm_block_f64,
                                (false, false) => &self.device.rms_norm_f64,
                            };
                            let mut launch = self.device.stream.launch_builder(function);
                            launch.arg(x.buffer.as_ref());
                            launch.arg(weight.buffer.as_ref());
                            launch.arg(output.buffer.as_ref());
                            launch.arg(&len_u32);
                            launch.arg(&width);
                            launch.arg(&weighted);
                            if *dtype == DType::F32 {
                                launch.arg(&eps_f32);
                            } else {
                                launch.arg(eps);
                            }
                            launch.arg(&dtype_code);
                            let rows = len_u32 / width;
                            let config = if wide {
                                LaunchConfig {
                                    grid_dim: (rows, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                }
                            } else {
                                const WARPS_PER_BLOCK: u32 = 4;
                                LaunchConfig {
                                    grid_dim: (rows.div_ceil(WARPS_PER_BLOCK), 1, 1),
                                    block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                                    shared_mem_bytes: 0,
                                }
                            };
                            unsafe { launch.launch(config) }.map_err(|error| error.to_string())?;
                        }
                        output
                    }
                }
                Instruction::CrossEntropy {
                    logits,
                    target,
                    shape,
                    logits_shape,
                    ignore_index,
                    backward,
                    dtype,
                } => {
                    let logits = value(&values, *logits)?;
                    let target = value(&values, *target)?;
                    let classes = u32::try_from(*logits_shape.last().unwrap_or(&0))
                        .map_err(|_| "CUDA cross-entropy class count exceeds u32".to_string())?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if *backward {
                        if len != 0 {
                            let dtype = dtype_code(*dtype);
                            let mut launch = self
                                .device
                                .stream
                                .launch_builder(&self.device.cross_entropy_backward_f64);
                            launch.arg(logits.buffer.as_ref());
                            launch.arg(target.buffer.as_ref());
                            launch.arg(output.buffer.as_ref());
                            launch.arg(&len_u32);
                            launch.arg(&classes);
                            launch.arg(ignore_index);
                            launch.arg(&dtype);
                            unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                                .map_err(|error| error.to_string())?;
                        }
                    } else {
                        let rows =
                            checked_len(element_count(&logits_shape[..logits_shape.len() - 1])?)?;
                        let dtype = dtype_code(*dtype);
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.cross_entropy_f64);
                        launch.arg(logits.buffer.as_ref());
                        launch.arg(target.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&rows);
                        launch.arg(&classes);
                        launch.arg(ignore_index);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                            .map_err(|error| error.to_string())?;
                        self.device
                            .stream
                            .synchronize()
                            .map_err(|error| error.to_string())?;
                        let host = self
                            .device
                            .stream
                            .clone_dtoh(&output.f64_buffer()?)
                            .map_err(|error| error.to_string())?;
                        if host.first().is_some_and(|value| value.is_nan()) {
                            return Err(
                                "cross_entropy: no active targets or target index is out of range"
                                    .to_string(),
                            );
                        }
                    }
                    output
                }
                Instruction::ChunkedHeadCe {
                    output: output_kind,
                    x,
                    weight,
                    bias,
                    target,
                    gradient,
                    shape,
                    x_shape,
                    weight_shape,
                    ignore_index,
                    target_dtype,
                    dtype,
                } => {
                    let x = value(&values, *x)?;
                    let weight = value(&values, *weight)?;
                    let bias = value(&values, *bias)?;
                    let target = value(&values, *target)?;
                    let rows = checked_len(element_count(&x_shape[..x_shape.len() - 1])?)?;
                    let inner = checked_len(*x_shape.last().unwrap_or(&0))?;
                    let vocab = checked_len(weight_shape[1])?;
                    let targets = target.readback_i64()?;
                    if targets.len() != rows as usize {
                        return Err("chunked head CE target geometry is invalid".to_string());
                    }
                    let ignore_is_representable = *target_dtype == DType::I64
                        || (0..=i64::from(u32::MAX)).contains(ignore_index);
                    let mut active = 0usize;
                    for &selected in &targets {
                        if ignore_is_representable && selected == *ignore_index {
                            continue;
                        }
                        active += 1;
                        if selected < 0 || selected >= i64::from(vocab) {
                            return Err(format!(
                                "cross_entropy: target out of range [0, {vocab}) at an active position"
                            ));
                        }
                    }
                    if active == 0 {
                        return Err(
                            "cross_entropy: no active targets (all positions are ignored)"
                                .to_string(),
                        );
                    }
                    let active = checked_len(active)?;
                    let target = self
                        .device
                        .stream
                        .clone_htod(&targets)
                        .map_err(|error| error.to_string())?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    let dtype = dtype_code(*dtype);
                    if let Some(output_kind) = output_kind {
                        let gradient = value(
                            &values,
                            gradient.ok_or_else(|| {
                                "chunked head CE backward gradient is missing".to_string()
                            })?,
                        )?;
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.chunked_head_ce_backward_f64);
                        launch.arg(output_kind);
                        launch.arg(x.buffer.as_ref());
                        launch.arg(weight.buffer.as_ref());
                        launch.arg(bias.buffer.as_ref());
                        launch.arg(&target);
                        launch.arg(gradient.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&rows);
                        launch.arg(&inner);
                        launch.arg(&vocab);
                        launch.arg(&active);
                        launch.arg(ignore_index);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    } else {
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.chunked_head_ce_f64);
                        launch.arg(x.buffer.as_ref());
                        launch.arg(weight.buffer.as_ref());
                        launch.arg(bias.buffer.as_ref());
                        launch.arg(&target);
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&rows);
                        launch.arg(&inner);
                        launch.arg(&vocab);
                        launch.arg(&active);
                        launch.arg(ignore_index);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Conv {
                    op,
                    x,
                    w,
                    shape,
                    stride,
                    padding,
                    dilation,
                    groups,
                    dtype,
                    ..
                } => {
                    let x = value(&values, *x)?;
                    let w = value(&values, *w)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let metadata = self.instruction_metadata[index]
                        .as_ref()
                        .expect("convolution metadata is prepared at compilation");
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.conv_f64);
                        launch.arg(op);
                        launch.arg(x.buffer.as_ref());
                        launch.arg(w.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(metadata);
                        launch.arg(stride);
                        launch.arg(padding);
                        launch.arg(dilation);
                        launch.arg(groups);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Linalg {
                    op,
                    a,
                    b,
                    shape,
                    a_shape,
                    dtype,
                } => {
                    let a = value(&values, *a)?;
                    let b = b
                        .map(|index| value(&values, index))
                        .transpose()?
                        .unwrap_or(a);
                    let n = u32::try_from(a_shape[a_shape.len() - 1])
                        .map_err(|_| "CUDA matrix size exceeds u32".to_string())?;
                    let batches = checked_len(element_count(&a_shape[..a_shape.len() - 2])?)?;
                    let rhs = if *op == 2 {
                        u32::try_from(*shape.last().unwrap_or(&1))
                            .map_err(|_| "CUDA right-hand side count exceeds u32".to_string())?
                    } else {
                        n
                    };
                    let workspace_len = linalg_workspace_len(*op, shape, a_shape)?;
                    let workspace: CudaBuffer<f64> =
                        self.scratch_buffer(&resources, 0, workspace_len)?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    let dtype_code = dtype_code(*dtype);
                    let mut launch = self.device.stream.launch_builder(&self.device.linalg_f64);
                    launch.arg(op);
                    launch.arg(a.buffer.as_ref());
                    launch.arg(b.buffer.as_ref());
                    launch.arg(output.buffer.as_ref());
                    launch.arg(&workspace);
                    launch.arg(&batches);
                    launch.arg(&n);
                    launch.arg(&rhs);
                    launch.arg(&dtype_code);
                    unsafe { launch.launch(LaunchConfig::for_num_elems(batches)) }
                        .map_err(|error| error.to_string())?;
                    if *op != 1 {
                        self.device
                            .stream
                            .synchronize()
                            .map_err(|error| error.to_string())?;
                        let host = self
                            .device
                            .stream
                            .clone_dtoh(&output.f64_buffer()?)
                            .map_err(|error| error.to_string())?;
                        if host.iter().any(|value| value.is_nan()) {
                            return Err("linalg: matrix is singular".to_string());
                        }
                    }
                    output
                }
                Instruction::Linear {
                    x,
                    weight,
                    bias,
                    shape,
                    k_width,
                    n_width,
                    dtype,
                } => {
                    let x = value(&values, *x)?;
                    let weight = value(&values, *weight)?;
                    let bias = value(&values, *bias)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.linear_f64);
                        launch.arg(x.buffer.as_ref());
                        launch.arg(weight.buffer.as_ref());
                        launch.arg(bias.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(k_width);
                        launch.arg(n_width);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::QuantizedLinear {
                    x,
                    weight,
                    bias,
                    shape,
                    rows,
                    columns,
                    row_bytes: _,
                    codec,
                    pair,
                } => {
                    let x_value = value(&values, *x)?;
                    let weight = value(&values, *weight)?;
                    let bias_value = bias
                        .map(|index| value(&values, index))
                        .transpose()?
                        .unwrap_or(x_value);
                    let packed_weight = weight.packed_buffer()?;
                    let has_bias = u32::from(bias.is_some());
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let vectors = len_u32
                        .checked_div(*rows)
                        .ok_or_else(|| "quantized_linear: weight has no rows".to_string())?;
                    let scratch = quantized_linear_scratch_layout(shape, *rows, *columns)?;
                    let q8: CudaBuffer<u8> = self.scratch_buffer(&resources, 0, scratch.bytes)?;
                    let output = self.planned_value(&resources, index, shape, DType::F32, false)?;
                    let paired = pair
                        .as_ref()
                        .map(|pair| {
                            let paired_weight = match self.instructions.get(pair.weight) {
                                Some(Instruction::Value(weight)) => weight,
                                _ => value(&values, pair.weight)?,
                            };
                            Ok::<_, String>((
                                paired_weight,
                                paired_weight.packed_buffer()?,
                                self.planned_value(
                                    &resources,
                                    pair.output,
                                    shape,
                                    DType::F32,
                                    false,
                                )?,
                            ))
                        })
                        .transpose()?;
                    if len != 0 {
                        let use_blas = vectors as usize >= BF16_GEMM_MIN_VECTORS && bias.is_none();
                        let bf16_elements = usize::try_from(vectors)
                            .ok()
                            .and_then(|vectors| vectors.checked_mul(*columns as usize))
                            .ok_or_else(|| {
                                "quantized_linear: BF16 input size overflowed usize".to_string()
                            })?;
                        let bf16_input = use_blas
                            .then(|| {
                                self.scratch_buffer::<half::bf16>(&resources, 0, bf16_elements)
                            })
                            .transpose()?;
                        let input_key =
                            (*x, scratch.bytes, scratch.padded_columns as usize, use_blas);
                        if quantized_input != Some(input_key) {
                            if let Some(bf16_input) = &bf16_input {
                                let bf16_elements = checked_len(bf16_elements)?;
                                let mut cast = self
                                    .device
                                    .stream
                                    .launch_builder(&self.device.cast_f32_bf16);
                                cast.arg(x_value.buffer.as_ref());
                                cast.arg(bf16_input);
                                cast.arg(&bf16_elements);
                                unsafe { cast.launch(LaunchConfig::for_num_elems(bf16_elements)) }
                                    .map_err(|error| error.to_string())?;
                            } else if self.gate_quantized[*x] {
                                let Instruction::MuseGate {
                                    value: gate_value,
                                    gate,
                                    multiplier,
                                    ..
                                } = &self.instructions[*x]
                                else {
                                    unreachable!("gate quantization marker requires MuseGate")
                                };
                                let gate_value = value(&values, *gate_value)?;
                                let gate = value(&values, *gate)?;
                                let has_multiplier = u32::from(multiplier.is_some());
                                let multiplier = multiplier
                                    .map(|index| value(&values, index))
                                    .transpose()?
                                    .unwrap_or(gate_value);
                                let mut quantize = self
                                    .device
                                    .stream
                                    .launch_builder(&self.device.muse_gate_quantize_q8_1_f32);
                                quantize.arg(gate_value.buffer.as_ref());
                                quantize.arg(gate.buffer.as_ref());
                                quantize.arg(multiplier.buffer.as_ref());
                                quantize.arg(&q8);
                                quantize.arg(columns);
                                quantize.arg(&scratch.padded_columns);
                                quantize.arg(&has_multiplier);
                                unsafe {
                                    quantize.launch(LaunchConfig {
                                        grid_dim: (
                                            scratch.padded_columns.div_ceil(256),
                                            vectors,
                                            1,
                                        ),
                                        block_dim: (256, 1, 1),
                                        shared_mem_bytes: 0,
                                    })
                                }
                                .map_err(|error| error.to_string())?;
                            } else if self.rms_quantized[*x] {
                                let Instruction::RmsNorm {
                                    x: source,
                                    weight,
                                    eps,
                                    ..
                                } = &self.instructions[*x]
                                else {
                                    unreachable!("RMS quantization marker requires RMSNorm")
                                };
                                let source_index = *source;
                                let source = value(&values, source_index)?;
                                let weight_value = weight
                                    .map(|index| value(&values, index))
                                    .transpose()?
                                    .unwrap_or(source);
                                let weighted = u32::from(weight.is_some());
                                let eps = *eps as f32;
                                let residual_index =
                                    resolve_alias_instruction(&self.instructions, source_index);
                                if self.deferred_residuals[residual_index] {
                                    let Instruction::MuseResidualRmsNorm {
                                        x: branch,
                                        weight: post_weight,
                                        residual,
                                        eps: post_eps,
                                        ..
                                    } = &self.instructions[residual_index]
                                    else {
                                        unreachable!(
                                            "deferred residual requires fused residual RMSNorm"
                                        )
                                    };
                                    let branch = value(&values, *branch)?;
                                    let residual = value(&values, *residual)?;
                                    let post_weight_value = post_weight
                                        .map(|index| value(&values, index))
                                        .transpose()?
                                        .unwrap_or(branch);
                                    let post_weighted = u32::from(post_weight.is_some());
                                    let post_eps = *post_eps as f32;
                                    let mut quantize = self.device.stream.launch_builder(
                                        &self.device.residual_rms_norm_quantize_q8_1_f32,
                                    );
                                    quantize.arg(branch.buffer.as_ref());
                                    quantize.arg(post_weight_value.buffer.as_ref());
                                    quantize.arg(residual.buffer.as_ref());
                                    quantize.arg(source.buffer.as_ref());
                                    quantize.arg(weight_value.buffer.as_ref());
                                    quantize.arg(&q8);
                                    quantize.arg(columns);
                                    quantize.arg(&scratch.padded_columns);
                                    quantize.arg(&post_weighted);
                                    quantize.arg(&weighted);
                                    quantize.arg(&post_eps);
                                    quantize.arg(&eps);
                                    unsafe {
                                        quantize.launch(LaunchConfig {
                                            grid_dim: (vectors, 1, 1),
                                            block_dim: (256, 1, 1),
                                            shared_mem_bytes: 0,
                                        })
                                    }
                                    .map_err(|error| error.to_string())?;
                                } else {
                                    let mut quantize = self
                                        .device
                                        .stream
                                        .launch_builder(&self.device.rms_norm_quantize_q8_1_f32);
                                    quantize.arg(source.buffer.as_ref());
                                    quantize.arg(weight_value.buffer.as_ref());
                                    quantize.arg(&q8);
                                    quantize.arg(columns);
                                    quantize.arg(&scratch.padded_columns);
                                    quantize.arg(&weighted);
                                    quantize.arg(&eps);
                                    unsafe {
                                        quantize.launch(LaunchConfig {
                                            grid_dim: (vectors, 1, 1),
                                            block_dim: (256, 1, 1),
                                            shared_mem_bytes: 0,
                                        })
                                    }
                                    .map_err(|error| error.to_string())?;
                                }
                            } else {
                                let mut quantize = self
                                    .device
                                    .stream
                                    .launch_builder(&self.device.quantize_q8_1_f32);
                                quantize.arg(x_value.buffer.as_ref());
                                quantize.arg(&q8);
                                quantize.arg(columns);
                                quantize.arg(&scratch.padded_columns);
                                unsafe {
                                    quantize.launch(LaunchConfig {
                                        grid_dim: (
                                            scratch.padded_columns.div_ceil(256),
                                            vectors,
                                            1,
                                        ),
                                        block_dim: (256, 1, 1),
                                        shared_mem_bytes: 0,
                                    })
                                }
                                .map_err(|error| error.to_string())?;
                            }
                            quantized_input = Some(input_key);
                        }
                        if let Some(bf16_input) = &bf16_input {
                            let dense_weight = weight.bf16_weight(*codec, *rows, *columns)?;
                            let mut dense_output = output.f32_buffer()?;
                            self.device.gemm_bf16_f32(
                                &dense_weight,
                                bf16_input,
                                &mut dense_output,
                                *rows,
                                vectors,
                                *columns,
                            )?;

                            if let Some((paired_weight, _, paired_output)) = &paired {
                                let paired_codec =
                                    pair.as_ref().expect("paired output is present").codec;
                                let dense_weight =
                                    paired_weight.bf16_weight(paired_codec, *rows, *columns)?;
                                let mut dense_output = paired_output.f32_buffer()?;
                                self.device.gemm_bf16_f32(
                                    &dense_weight,
                                    bf16_input,
                                    &mut dense_output,
                                    *rows,
                                    vectors,
                                    *columns,
                                )?;
                            }
                        } else {
                            let cooperative = *rows == 256
                                || (paired.is_none()
                                    && *codec == GgmlKQuant::Q3K
                                    && *columns == 19968);
                            let row_warps = if cooperative {
                                4
                            } else if *codec == GgmlKQuant::Q3K && *rows == 6656 {
                                4
                            } else if *rows >= 1024 {
                                8
                            } else {
                                1
                            };
                            let grid_x = if cooperative {
                                *rows
                            } else {
                                rows.div_ceil(row_warps)
                            };
                            if let Some((_, paired_weight, paired_output)) = &paired {
                                let function = match (
                                    *codec,
                                    pair.as_ref().expect("paired output is present").codec,
                                ) {
                                    (GgmlKQuant::Q2K, GgmlKQuant::Q2K) => {
                                        &self.device.mmvq_q2_k_f32_pair
                                    }
                                    (GgmlKQuant::Q3K, GgmlKQuant::Q3K) => {
                                        &self.device.mmvq_q3_k_f32_pair
                                    }
                                    (GgmlKQuant::Q2K, GgmlKQuant::Q3K) => {
                                        &self.device.mmvq_q2_q3_k_f32_pair
                                    }
                                    (GgmlKQuant::Q3K, GgmlKQuant::Q2K) => {
                                        &self.device.mmvq_q3_q2_k_f32_pair
                                    }
                                    _ => unreachable!("only Q2_K and Q3_K projections are paired"),
                                };
                                let mut launch = self.device.stream.launch_builder(function);
                                launch.arg(packed_weight);
                                launch.arg(*paired_weight);
                                launch.arg(&q8);
                                launch.arg(output.buffer.as_ref());
                                launch.arg(paired_output.buffer.as_ref());
                                launch.arg(columns);
                                launch.arg(rows);
                                launch.arg(&scratch.padded_columns);
                                unsafe {
                                    launch.launch(LaunchConfig {
                                        grid_dim: (grid_x, vectors, 2),
                                        block_dim: (32, row_warps, 1),
                                        shared_mem_bytes: 0,
                                    })
                                }
                                .map_err(|error| error.to_string())?;
                            } else {
                                let function = match (*codec, cooperative) {
                                    (GgmlKQuant::Q2K, false) => &self.device.mmvq_q2_k_f32,
                                    (GgmlKQuant::Q3K, false) => &self.device.mmvq_q3_k_f32,
                                    (GgmlKQuant::Q4K, false) => &self.device.mmvq_q4_k_f32,
                                    (GgmlKQuant::Q5K, false) => &self.device.mmvq_q5_k_f32,
                                    (GgmlKQuant::Q6K, false) => &self.device.mmvq_q6_k_f32,
                                    (GgmlKQuant::Q2K, true) => {
                                        &self.device.mmvq_q2_k_f32_cooperative
                                    }
                                    (GgmlKQuant::Q3K, true) => {
                                        &self.device.mmvq_q3_k_f32_cooperative
                                    }
                                    (GgmlKQuant::Q4K, true) => {
                                        &self.device.mmvq_q4_k_f32_cooperative
                                    }
                                    (GgmlKQuant::Q5K, true) => {
                                        &self.device.mmvq_q5_k_f32_cooperative
                                    }
                                    (GgmlKQuant::Q6K, true) => {
                                        &self.device.mmvq_q6_k_f32_cooperative
                                    }
                                };
                                let mut launch = self.device.stream.launch_builder(function);
                                launch.arg(packed_weight);
                                launch.arg(&q8);
                                launch.arg(bias_value.buffer.as_ref());
                                launch.arg(output.buffer.as_ref());
                                launch.arg(columns);
                                launch.arg(rows);
                                launch.arg(&scratch.padded_columns);
                                launch.arg(&has_bias);
                                unsafe {
                                    launch.launch(LaunchConfig {
                                        grid_dim: (grid_x, vectors, 1),
                                        block_dim: (32, row_warps, 1),
                                        shared_mem_bytes: 0,
                                    })
                                }
                                .map_err(|error| error.to_string())?;
                            }
                        }
                    }
                    if let Some((_, _, paired_output)) = paired {
                        values[pair.as_ref().expect("paired output is present").output] =
                            Some(paired_output);
                    }
                    output
                }
                Instruction::QuantizedEmbedding {
                    indexes,
                    weight,
                    shape,
                    rows,
                    columns,
                    row_bytes,
                    codec,
                } => {
                    let indexes = value(&values, *indexes)?;
                    let weight = value(&values, *weight)?;
                    let packed_weight = weight.packed_buffer()?;
                    if validate_embedding_indexes {
                        for index in indexes.readback()? {
                            if !index.is_finite()
                                || index.fract() != 0.0
                                || index < 0.0
                                || index >= f64::from(*rows)
                            {
                                return Err(format!(
                                    "quantized_embedding: index {index} is outside 0..{rows}"
                                ));
                            }
                        }
                    }
                    let codec = kquant_code(*codec);
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let output = self.planned_value(&resources, index, shape, DType::F32, false)?;
                    if len != 0 {
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.quantized_embedding_k_f32);
                        launch.arg(indexes.buffer.as_ref());
                        launch.arg(packed_weight);
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(columns);
                        launch.arg(row_bytes);
                        launch.arg(&codec);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::LayerNorm {
                    op,
                    x,
                    weight,
                    other,
                    shape,
                    width,
                    rows,
                    eps,
                    dtype,
                } => {
                    let x = value(&values, *x)?;
                    let weight = value(&values, *weight)?;
                    let other = value(&values, *other)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.layer_norm_f64);
                        launch.arg(op);
                        launch.arg(x.buffer.as_ref());
                        launch.arg(weight.buffer.as_ref());
                        launch.arg(other.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(width);
                        launch.arg(rows);
                        launch.arg(eps);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Sdpa {
                    op,
                    q,
                    k,
                    v,
                    g,
                    shape,
                    q_shape,
                    scale,
                    causal,
                    window,
                    dtype,
                    ..
                } => {
                    let q = value(&values, *q)?;
                    let k = value(&values, *k)?;
                    let v = value(&values, *v)?;
                    let g = g
                        .map(|index| value(&values, index))
                        .transpose()?
                        .unwrap_or(v);
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let rank = u32::try_from(q_shape.len())
                        .map_err(|_| "CUDA attention rank exceeds u32".to_string())?;
                    let metadata = self.instruction_metadata[index]
                        .as_ref()
                        .expect("attention metadata is prepared at compilation");
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let causal = u32::from(*causal);
                        let window = u32::try_from(window.unwrap_or(0))
                            .map_err(|_| "CUDA attention window exceeds u32".to_string())?;
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.sdpa_f64);
                        launch.arg(op);
                        launch.arg(q.buffer.as_ref());
                        launch.arg(k.buffer.as_ref());
                        launch.arg(v.buffer.as_ref());
                        launch.arg(g.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&rank);
                        launch.arg(metadata);
                        launch.arg(scale);
                        launch.arg(&causal);
                        launch.arg(&window);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Rotary {
                    x,
                    shape,
                    seq_len,
                    theta,
                    interleaved,
                    backward,
                    cursor,
                    dtype,
                } => {
                    let x = value(&values, *x)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let width = u32::try_from(*shape.last().unwrap_or(&0))
                        .map_err(|_| "CUDA rotary width exceeds u32".to_string())?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let leading = len
                            .checked_div(width as usize * *seq_len as usize)
                            .ok_or_else(|| "CUDA rotary shape is invalid".to_string())?;
                        let (offsets, groups) = if *cursor {
                            let state = state.as_deref().ok_or_else(|| {
                                "execute: cursor-relative rotary embedding requires decode state".to_string()
                            })?;
                            let packed_rows = state.packed_rows_per_sequence.unwrap_or(1) as usize;
                            let graph_rows = state
                                .valid_lengths
                                .len()
                                .checked_mul(packed_rows)
                                .ok_or_else(|| "CUDA rotary graph rows overflowed".to_string())?;
                            if graph_rows == 0 || !leading.is_multiple_of(graph_rows) {
                                return Err(
                                    "CUDA rotary graph rows do not match decode state".to_string()
                                );
                            }
                            let mut offsets = vec![0u32; graph_rows];
                            for (request, &slot) in state.slots.iter().enumerate() {
                                let base = state.sequences[request].cursor;
                                for row in 0..packed_rows {
                                    offsets[slot as usize * packed_rows + row] =
                                        base.saturating_add(row as u32);
                                }
                            }
                            (offsets, checked_len(leading / graph_rows)?)
                        } else {
                            (vec![0], checked_len(leading)?)
                        };
                        let offsets = self
                            .device
                            .stream
                            .clone_htod(&offsets)
                            .map_err(|error| error.to_string())?;
                        let interleaved = u32::from(*interleaved);
                        let backward = u32::from(*backward);
                        let dtype_code = dtype_code(*dtype);
                        let theta_f32 = *theta as f32;
                        let function = if *dtype == DType::F32 {
                            &self.device.f32.rotary
                        } else {
                            &self.device.rotary_f64
                        };
                        let mut launch = self.device.stream.launch_builder(function);
                        launch.arg(x.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&width);
                        launch.arg(seq_len);
                        launch.arg(&offsets);
                        launch.arg(&groups);
                        if *dtype == DType::F32 {
                            launch.arg(&theta_f32);
                        } else {
                            launch.arg(theta);
                        }
                        launch.arg(&interleaved);
                        launch.arg(&backward);
                        launch.arg(&dtype_code);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Optimizer {
                    kind,
                    output: output_kind,
                    param,
                    grad,
                    state1,
                    state2,
                    lr,
                    c1,
                    c2,
                    shape,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    dampening,
                    nesterov,
                    dtype,
                } => {
                    let param = value(&values, *param)?;
                    let grad = value(&values, *grad)?;
                    let state1 = value(&values, *state1)?;
                    let state2 = value(&values, *state2)?;
                    let lr = value(&values, *lr)?;
                    let c1 = value(&values, *c1)?;
                    let c2 = value(&values, *c2)?;
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let nesterov = u32::from(*nesterov);
                        let dtype = dtype_code(*dtype);
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.optimizer_f64);
                        launch.arg(kind);
                        launch.arg(output_kind);
                        launch.arg(param.buffer.as_ref());
                        launch.arg(grad.buffer.as_ref());
                        launch.arg(state1.buffer.as_ref());
                        launch.arg(state2.buffer.as_ref());
                        launch.arg(lr.buffer.as_ref());
                        launch.arg(c1.buffer.as_ref());
                        launch.arg(c2.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(beta1);
                        launch.arg(beta2);
                        launch.arg(eps);
                        launch.arg(weight_decay);
                        launch.arg(dampening);
                        launch.arg(&nesterov);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::ShortConv {
                    op,
                    state_layer,
                    x,
                    weight,
                    g,
                    shape,
                    input_shape,
                    weight_shape,
                    dtype,
                } => {
                    let x = value(&values, *x)?;
                    let weight = value(&values, *weight)?;
                    let g = g
                        .map(|index| value(&values, index))
                        .transpose()?
                        .unwrap_or(x);
                    let len = element_count(shape)?;
                    let len_u32 = checked_len(len)?;
                    let time = checked_len(input_shape[input_shape.len() - 2])?;
                    let channels = checked_len(input_shape[input_shape.len() - 1])?;
                    let outer = checked_len(element_count(&input_shape[..input_shape.len() - 2])?)?;
                    let kernel = checked_len(weight_shape[1])?;
                    let stateful = u32::from(state_layer.is_some());
                    let history_size = (outer as usize)
                        .checked_mul(kernel.saturating_sub(1) as usize)
                        .and_then(|value| value.checked_mul(channels as usize))
                        .ok_or_else(|| "CUDA convolution state size overflowed".to_string())?;
                    let mut initial_host = vec![0.0; history_size.max(1)];
                    let valid_host = if let Some(layer) = state_layer {
                        let invocation = state.as_deref().ok_or_else(|| {
                            "execute: convolution state requires decode state".to_string()
                        })?;
                        if invocation.valid_lengths.len() != outer as usize {
                            return Err("execute: convolution state does not support packed rows"
                                .to_string());
                        }
                        let layer_size = history_size / outer as usize;
                        for (request, &slot) in invocation.slots.iter().enumerate() {
                            let source = invocation.sequences[request]
                                .conv_states
                                .get(*layer)
                                .ok_or_else(|| {
                                    "execute: convolution state layer is missing".to_string()
                                })?;
                            if source.len() != layer_size {
                                return Err(
                                    "execute: convolution state geometry is invalid".to_string()
                                );
                            }
                            let start = slot as usize * layer_size;
                            initial_host[start..start + layer_size].copy_from_slice(source);
                        }
                        invocation.valid_lengths.clone()
                    } else {
                        vec![time; outer as usize]
                    };
                    let initial_state = self
                        .device
                        .stream
                        .clone_htod(&initial_host)
                        .map_err(|error| error.to_string())?;
                    let mut next_state =
                        unsafe { self.device.stream.alloc::<f64>(history_size.max(1)) }
                            .map_err(|error| error.to_string())?;
                    let valid = self
                        .device
                        .stream
                        .clone_htod(&valid_host)
                        .map_err(|error| error.to_string())?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let dtype = dtype_code(*dtype);
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.short_conv_f64);
                        launch.arg(op);
                        launch.arg(x.buffer.as_ref());
                        launch.arg(weight.buffer.as_ref());
                        launch.arg(g.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&outer);
                        launch.arg(&time);
                        launch.arg(&channels);
                        launch.arg(&kernel);
                        launch.arg(&initial_state);
                        launch.arg(&mut next_state);
                        launch.arg(&valid);
                        launch.arg(&stateful);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    if let Some(layer) = state_layer {
                        let next = self
                            .device
                            .stream
                            .clone_dtoh(&next_state)
                            .map_err(|error| error.to_string())?;
                        let invocation = state.as_deref_mut().ok_or_else(|| {
                            "execute: convolution state requires decode state".to_string()
                        })?;
                        let layer_size = history_size / outer as usize;
                        for (request, &slot) in invocation.slots.iter().enumerate() {
                            invocation.sequences[request].conv_states[*layer] = next
                                [slot as usize * layer_size..(slot as usize + 1) * layer_size]
                                .to_vec();
                        }
                    }
                    output
                }
                Instruction::Kda {
                    output: output_kind,
                    state_layer,
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    g,
                    shape,
                    q_shape,
                    v_shape,
                    scale,
                    dtype,
                } => {
                    let q = value(&values, *q)?;
                    let k = value(&values, *k)?;
                    let v = value(&values, *v)?;
                    let decay = value(&values, *decay)?;
                    let beta = value(&values, *beta)?;
                    let g = g
                        .map(|index| value(&values, index))
                        .transpose()?
                        .unwrap_or(v);
                    let rank = q_shape.len();
                    let time = checked_len(q_shape[rank - 2])?;
                    let dk = checked_len(q_shape[rank - 1])?;
                    let dv = checked_len(v_shape[rank - 1])?;
                    let outer = checked_len(element_count(&q_shape[..rank - 2])?)?;
                    let heads = checked_len(q_shape[rank - 3])?;
                    let graph_rows = outer
                        .checked_div(heads)
                        .ok_or_else(|| "CUDA KDA head geometry is invalid".to_string())?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    let state_size = (outer as usize)
                        .checked_mul(dk as usize)
                        .and_then(|value| value.checked_mul(dv as usize))
                        .ok_or_else(|| "CUDA KDA workspace size overflowed".to_string())?;
                    let dtype_code = dtype_code(*dtype);
                    if let Some(output_kind) = output_kind {
                        let history_size = state_size
                            .checked_mul(time as usize + 1)
                            .ok_or_else(|| "CUDA KDA history size overflowed".to_string())?;
                        let mut history = unsafe { self.device.stream.alloc::<f64>(history_size) }
                            .map_err(|error| error.to_string())?;
                        let mut grad_state =
                            unsafe { self.device.stream.alloc::<f64>(state_size * 2) }
                                .map_err(|error| error.to_string())?;
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.kda_backward_f64);
                        launch.arg(output_kind);
                        launch.arg(q.buffer.as_ref());
                        launch.arg(k.buffer.as_ref());
                        launch.arg(v.buffer.as_ref());
                        launch.arg(decay.buffer.as_ref());
                        launch.arg(beta.buffer.as_ref());
                        launch.arg(g.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&mut history);
                        launch.arg(&mut grad_state);
                        launch.arg(&outer);
                        launch.arg(&time);
                        launch.arg(&dk);
                        launch.arg(&dv);
                        launch.arg(scale);
                        launch.arg(&dtype_code);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(outer)) }
                            .map_err(|error| error.to_string())?;
                    } else {
                        let stateful = u32::from(state_layer.is_some());
                        let mut state_host = vec![0.0; state_size];
                        let valid_host = if let Some(layer) = state_layer {
                            let invocation = state.as_deref().ok_or_else(|| {
                                "execute: KDA recurrence requires decode state".to_string()
                            })?;
                            if invocation.valid_lengths.len() != graph_rows as usize {
                                return Err("execute: KDA recurrence does not support packed rows"
                                    .to_string());
                            }
                            let layer_size = heads as usize * dk as usize * dv as usize;
                            for (request, &slot) in invocation.slots.iter().enumerate() {
                                let source = invocation.sequences[request]
                                    .kda_states
                                    .get(*layer)
                                    .ok_or_else(|| {
                                    "execute: KDA state layer is missing".to_string()
                                })?;
                                if source.len() != layer_size {
                                    return Err(
                                        "execute: KDA state geometry is invalid".to_string()
                                    );
                                }
                                let start = slot as usize * layer_size;
                                state_host[start..start + layer_size].copy_from_slice(source);
                            }
                            invocation.valid_lengths.clone()
                        } else {
                            vec![time; graph_rows as usize]
                        };
                        let mut state_buffer = self
                            .device
                            .stream
                            .clone_htod(&state_host)
                            .map_err(|error| error.to_string())?;
                        let valid = self
                            .device
                            .stream
                            .clone_htod(&valid_host)
                            .map_err(|error| error.to_string())?;
                        let mut launch = self
                            .device
                            .stream
                            .launch_builder(&self.device.kda_forward_f64);
                        launch.arg(q.buffer.as_ref());
                        launch.arg(k.buffer.as_ref());
                        launch.arg(v.buffer.as_ref());
                        launch.arg(decay.buffer.as_ref());
                        launch.arg(beta.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&mut state_buffer);
                        launch.arg(&outer);
                        launch.arg(&time);
                        launch.arg(&dk);
                        launch.arg(&dv);
                        launch.arg(scale);
                        launch.arg(&valid);
                        launch.arg(&heads);
                        launch.arg(&stateful);
                        launch.arg(&dtype_code);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(outer)) }
                            .map_err(|error| error.to_string())?;
                        if let Some(layer) = state_layer {
                            let next = self
                                .device
                                .stream
                                .clone_dtoh(&state_buffer)
                                .map_err(|error| error.to_string())?;
                            let invocation = state.as_deref_mut().ok_or_else(|| {
                                "execute: KDA recurrence requires decode state".to_string()
                            })?;
                            let layer_size = heads as usize * dk as usize * dv as usize;
                            for (request, &slot) in invocation.slots.iter().enumerate() {
                                invocation.sequences[request].kda_states[*layer] = next
                                    [slot as usize * layer_size..(slot as usize + 1) * layer_size]
                                    .to_vec();
                            }
                        }
                    }
                    output
                }
                Instruction::LastTokenRow {
                    a,
                    lane,
                    shape,
                    tokens,
                    dtype,
                } => {
                    let input = value(&values, *a)?;
                    let width = checked_len(element_count(shape)?)?;
                    let tokens = checked_len(*tokens)?;
                    let valid = *state
                        .as_deref()
                        .and_then(|state| state.valid_lengths.get(*lane))
                        .ok_or_else(|| {
                            "execute: last-token selection requires decode state".to_string()
                        })?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if width != 0 {
                        let dtype_code = dtype_code(*dtype);
                        let function = if *dtype == DType::F32 {
                            &self.device.f32.last_token
                        } else {
                            &self.device.last_token_f64
                        };
                        let mut launch = self.device.stream.launch_builder(function);
                        launch.arg(input.buffer.as_ref());
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&width);
                        launch.arg(&tokens);
                        launch.arg(&valid);
                        launch.arg(&dtype_code);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(width)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::KvAttention {
                    q,
                    k,
                    v,
                    shape,
                    q_shape,
                    k_shape,
                    scale,
                    layer,
                    window,
                    bidirectional,
                    dtype,
                    muse,
                } => {
                    let q = value(&values, muse.as_ref().map_or(*q, |muse| muse.q))?;
                    let k = value(&values, muse.as_ref().map_or(*k, |muse| muse.k))?;
                    let v = value(&values, *v)?;
                    let state = state
                        .as_deref_mut()
                        .ok_or_else(|| "execute: KV attention requires decode state".to_string())?;
                    if q_shape.len() != 4
                        || k_shape.len() != 4
                        || q_shape[0]
                            != state
                                .valid_lengths
                                .len()
                                .checked_mul(state.packed_rows_per_sequence.unwrap_or(1) as usize)
                                .ok_or_else(|| {
                                    "execute: CUDA packed KV rows overflowed".to_string()
                                })?
                    {
                        return Err("execute: CUDA KV attention currently requires dense rank-four state rows".to_string());
                    }
                    let batch = checked_len(q_shape[0])?;
                    let logical_batch = checked_len(state.valid_lengths.len())?;
                    let rows_per_sequence = state.packed_rows_per_sequence.unwrap_or(1);
                    let q_heads = checked_len(q_shape[1])?;
                    let kv_heads = checked_len(k_shape[1])?;
                    let tokens = checked_len(q_shape[2])?;
                    let dim = checked_len(q_shape[3])?;
                    let capacity = state.capacity;
                    let row_size = (capacity as usize)
                        .checked_mul(kv_heads as usize)
                        .and_then(|value| value.checked_mul(dim as usize))
                        .ok_or_else(|| "execute: CUDA KV cache size overflowed".to_string())?;
                    let cache_size = (logical_batch as usize)
                        .checked_mul(row_size)
                        .ok_or_else(|| "execute: CUDA KV cache size overflowed".to_string())?;
                    if !kv_prepared {
                        state.ensure_kv_metadata(
                            &self.device,
                            batch as usize,
                            rows_per_sequence,
                        )?;
                        state.ensure_kv_cache(&self.device, cache_size)?;
                        kv_prepared = true;
                    }
                    let layer_start = layer
                        .checked_mul(cache_size)
                        .ok_or_else(|| "execute: CUDA KV layer offset overflowed".to_string())?;
                    let layer_end = layer_start
                        .checked_add(cache_size)
                        .ok_or_else(|| "execute: CUDA KV layer offset overflowed".to_string())?;
                    let (key_cache, value_cache, cursor_device, valid_device) =
                        if let Some(prepared) = &state.prepared_kv {
                            (
                                prepared.keys.slice(layer_start..layer_end)?,
                                prepared.values.slice(layer_start..layer_end)?,
                                prepared.cursor.clone(),
                                prepared.valid.clone(),
                            )
                        } else {
                            (
                                CudaBuffer::from_arc_slice(Arc::clone(
                                    state.key_cache.as_ref().expect("KV cache was initialized"),
                                ))
                                .slice(layer_start..layer_end)?,
                                CudaBuffer::from_arc_slice(Arc::clone(
                                    state
                                        .value_cache
                                        .as_ref()
                                        .expect("KV cache was initialized"),
                                ))
                                .slice(layer_start..layer_end)?,
                                CudaBuffer::from_arc_slice(Arc::clone(
                                    state
                                        .cursor_device
                                        .as_ref()
                                        .expect("KV cursor metadata was initialized"),
                                )),
                                CudaBuffer::from_arc_slice(Arc::clone(
                                    state
                                        .valid_device
                                        .as_ref()
                                        .expect("KV validity metadata was initialized"),
                                )),
                            )
                        };
                    let append_len = checked_len(element_count(k_shape)?)?;
                    let cache_dtype = dtype_code(state.cache_dtype);
                    let muse_gqa = state.cache_dtype != DType::U8
                        && rows_per_sequence == 1
                        && batch == 1
                        && q_heads == 32
                        && kv_heads == 2
                        && tokens == 1
                        && dim == 128
                        && !bidirectional
                        && muse.is_some();
                    if append_len != 0 && !muse_gqa {
                        let function = if rows_per_sequence == 1 {
                            &self.device.f32.kv_append
                        } else {
                            &self.device.f32.kv_append_packed
                        };
                        let mut launch = self.device.stream.launch_builder(function);
                        launch.arg(k.buffer.as_ref());
                        launch.arg(v.buffer.as_ref());
                        launch.arg(&key_cache);
                        launch.arg(&value_cache);
                        launch.arg(&cursor_device);
                        launch.arg(&valid_device);
                        launch.arg(&batch);
                        launch.arg(&kv_heads);
                        launch.arg(&tokens);
                        launch.arg(&dim);
                        launch.arg(&capacity);
                        if rows_per_sequence != 1 {
                            launch.arg(&rows_per_sequence);
                        }
                        launch.arg(&cache_dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(append_len)) }
                            .map_err(|error| error.to_string())?;
                    }
                    let output_len = checked_len(element_count(shape)?)?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if output_len != 0 {
                        let window = window.map_or(0, |window| window as u32);
                        let bidirectional = u32::from(*bidirectional);
                        let output_dtype = dtype_code(*dtype);
                        if muse_gqa {
                            let muse = muse.as_ref().expect("Muse GQA preparation is present");
                            let q_weight = value(&values, muse.q_weight)?;
                            let k_weight = value(&values, muse.k_weight)?;
                            let scale_f32 = *scale as f32;
                            let q_eps = muse.q_eps as f32;
                            let k_eps = muse.k_eps as f32;
                            let rope_theta = muse.rope_theta.unwrap_or(1.0) as f32;
                            let apply_rope = u32::from(muse.rope_theta.is_some());
                            let partials: CudaBuffer<f32> =
                                self.scratch_buffer(&resources, 0, MUSE_GQA_PARTIAL_FLOATS)?;
                            let mut launch = self
                                .device
                                .stream
                                .launch_builder(&self.device.f32.muse_gqa_append_attention_grouped);
                            launch.arg(q.buffer.as_ref());
                            launch.arg(k.buffer.as_ref());
                            launch.arg(v.buffer.as_ref());
                            launch.arg(q_weight.buffer.as_ref());
                            launch.arg(k_weight.buffer.as_ref());
                            launch.arg(&key_cache);
                            launch.arg(&value_cache);
                            launch.arg(&partials);
                            launch.arg(&cursor_device);
                            launch.arg(&capacity);
                            launch.arg(&scale_f32);
                            launch.arg(&q_eps);
                            launch.arg(&k_eps);
                            launch.arg(&rope_theta);
                            launch.arg(&apply_rope);
                            launch.arg(&window);
                            launch.arg(&cache_dtype);
                            unsafe {
                                launch.launch(LaunchConfig {
                                    grid_dim: ((2 * MUSE_GQA_PARTITIONS) as u32, 1, 1),
                                    block_dim: (512, 1, 1),
                                    shared_mem_bytes: 0,
                                })
                            }
                            .map_err(|error| error.to_string())?;
                            let mut merge = self
                                .device
                                .stream
                                .launch_builder(&self.device.f32.muse_gqa_merge_grouped);
                            merge.arg(&partials);
                            merge.arg(output.buffer.as_ref());
                            merge.arg(&output_dtype);
                            unsafe {
                                merge.launch(LaunchConfig {
                                    grid_dim: (8, 1, 1),
                                    block_dim: (128, 1, 1),
                                    shared_mem_bytes: 0,
                                })
                            }
                            .map_err(|error| error.to_string())?;
                            values[index] = Some(output);
                            continue;
                        }
                        let cooperative = dim <= 256;
                        let scale_f32 = *scale as f32;
                        let function = match (rows_per_sequence == 1, cooperative) {
                            (true, true) => &self.device.f32.kv_attention_warp,
                            (true, false) => &self.device.f32.kv_attention,
                            (false, true) => &self.device.f32.kv_attention_packed_warp,
                            (false, false) => &self.device.f32.kv_attention_packed,
                        };
                        let mut launch = self.device.stream.launch_builder(function);
                        launch.arg(q.buffer.as_ref());
                        launch.arg(&key_cache);
                        launch.arg(&value_cache);
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&cursor_device);
                        launch.arg(&valid_device);
                        launch.arg(&batch);
                        launch.arg(&q_heads);
                        launch.arg(&kv_heads);
                        launch.arg(&tokens);
                        launch.arg(&dim);
                        launch.arg(&capacity);
                        if rows_per_sequence != 1 {
                            launch.arg(&rows_per_sequence);
                        }
                        launch.arg(&scale_f32);
                        launch.arg(&window);
                        launch.arg(&bidirectional);
                        launch.arg(&output_dtype);
                        let config = if cooperative {
                            const WARPS_PER_BLOCK: u32 = 4;
                            let queries = output_len / dim;
                            LaunchConfig {
                                grid_dim: (queries.div_ceil(WARPS_PER_BLOCK), 1, 1),
                                block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                                shared_mem_bytes: 0,
                            }
                        } else {
                            LaunchConfig::for_num_elems(output_len)
                        };
                        unsafe { launch.launch(config) }.map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Random {
                    shape,
                    dtype,
                    normal,
                    lo,
                    hi,
                    provenance,
                } => {
                    let len = element_count(shape)?;
                    let len_u32 = u32::try_from(len).map_err(|_| {
                        format!("CUDA element count {len} exceeds the first-slice limit")
                    })?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let seed = provenance.wrapping_add(run.wrapping_mul(0x9e3779b97f4a7c15));
                        let normal = u32::from(*normal);
                        let dtype = dtype_code(*dtype);
                        let mut launch = self.device.stream.launch_builder(&self.device.random_f64);
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(&seed);
                        launch.arg(&normal);
                        launch.arg(lo);
                        launch.arg(hi);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
                Instruction::Sequence {
                    shape,
                    dtype,
                    eye,
                    start,
                    step,
                } => {
                    let len = element_count(shape)?;
                    let len_u32 = u32::try_from(len).map_err(|_| {
                        format!("CUDA element count {len} exceeds the first-slice limit")
                    })?;
                    let output = self.planned_value(&resources, index, shape, *dtype, false)?;
                    if len != 0 {
                        let op = u32::from(*eye);
                        let width = u32::try_from(shape.last().copied().unwrap_or(1))
                            .map_err(|_| "CUDA sequence width exceeds u32".to_string())?;
                        let dtype = dtype_code(*dtype);
                        let mut launch =
                            self.device.stream.launch_builder(&self.device.sequence_f64);
                        launch.arg(&op);
                        launch.arg(output.buffer.as_ref());
                        launch.arg(&len_u32);
                        launch.arg(start);
                        launch.arg(step);
                        launch.arg(&width);
                        launch.arg(&dtype);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
                            .map_err(|error| error.to_string())?;
                    }
                    output
                }
            };
            values[index] = Some(value);
        }
        if synchronize {
            self.device
                .stream
                .synchronize()
                .map_err(|error| error.to_string())?;
        }
        if cancelled.is_cancelled() {
            return Err("operation aborted".to_string());
        }
        self.roots
            .iter()
            .map(|root| {
                values[*root]
                    .as_ref()
                    .cloned()
                    .ok_or_else(|| "CUDA executable root was unavailable".to_string())
            })
            .collect()
    }
}

fn dtype_code(dtype: DType) -> u32 {
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

fn kquant_code(codec: GgmlKQuant) -> u32 {
    match codec {
        GgmlKQuant::Q2K => 0,
        GgmlKQuant::Q3K => 1,
        GgmlKQuant::Q4K => 2,
        GgmlKQuant::Q5K => 3,
        GgmlKQuant::Q6K => 4,
    }
}

fn checked_len(len: usize) -> Result<u32, String> {
    u32::try_from(len)
        .map_err(|_| format!("CUDA element count {len} exceeds the first-slice limit"))
}

struct QuantizedLinearScratchLayout {
    padded_columns: u32,
    bytes: usize,
}

fn quantized_linear_scratch_layout(
    shape: &[usize],
    rows: u32,
    columns: u32,
) -> Result<QuantizedLinearScratchLayout, String> {
    let len = element_count(shape)?;
    let rows = usize::try_from(rows).map_err(|_| "quantized_linear: rows exceed usize")?;
    if rows == 0 || !len.is_multiple_of(rows) {
        return Err("quantized_linear: output size is not divisible by rows".to_string());
    }
    let vectors = len / rows;
    let padded_columns = columns
        .checked_next_multiple_of(512)
        .ok_or_else(|| "quantized_linear: padded width overflow".to_string())?;
    let blocks = vectors
        .checked_mul(padded_columns as usize / 32)
        .ok_or_else(|| "quantized_linear: Q8_1 block count overflow".to_string())?;
    let bytes = blocks
        .checked_mul(36)
        .ok_or_else(|| "quantized_linear: scratch byte size overflow".to_string())?;
    let bf16_bytes = vectors
        .checked_mul(columns as usize)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<half::bf16>()))
        .ok_or_else(|| "quantized_linear: BF16 scratch byte size overflow".to_string())?;
    Ok(QuantizedLinearScratchLayout {
        padded_columns,
        bytes: bytes.max(bf16_bytes),
    })
}

fn linalg_workspace_len(op: u32, shape: &[usize], a_shape: &[usize]) -> Result<usize, String> {
    if a_shape.len() < 2 {
        return Err("CUDA linalg input rank must be at least two".to_string());
    }
    let n = a_shape[a_shape.len() - 1];
    let batches = element_count(&a_shape[..a_shape.len() - 2])?;
    let rhs = if op == 2 {
        shape.last().copied().unwrap_or(1)
    } else {
        n
    };
    let matrix_size = n
        .checked_mul(n)
        .ok_or_else(|| "CUDA matrix workspace size overflowed".to_string())?;
    let right_size = if op == 1 {
        0
    } else {
        n.checked_mul(rhs)
            .ok_or_else(|| "CUDA matrix workspace size overflowed".to_string())?
    };
    matrix_size
        .checked_add(right_size)
        .and_then(|size| size.checked_mul(batches))
        .ok_or_else(|| "CUDA matrix workspace size overflowed".to_string())
}

fn value(values: &[Option<CudaValue>], index: usize) -> Result<&CudaValue, String> {
    values[index]
        .as_ref()
        .ok_or_else(|| "CUDA executable referenced an unavailable input".to_string())
}

pub(super) fn decode(data: &[u8], dtype: DType) -> Result<Vec<f64>, String> {
    let width = match dtype {
        DType::F64 | DType::I64 => 8,
        DType::F32 | DType::U32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::U8 => 1,
    };
    if !data.len().is_multiple_of(width) {
        return Err(format!(
            "CUDA {} byte length {} is not divisible by {width}",
            dtype.name(),
            data.len(),
        ));
    }
    Ok(data
        .chunks_exact(width)
        .map(|bytes| match dtype {
            DType::F64 => f64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")),
            DType::F32 => f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")) as f64,
            DType::F16 => half::f16::from_bits(u16::from_le_bytes(
                bytes.try_into().expect("two-byte chunk"),
            ))
            .to_f64(),
            DType::BF16 => half::bf16::from_bits(u16::from_le_bytes(
                bytes.try_into().expect("two-byte chunk"),
            ))
            .to_f64(),
            DType::I64 => i64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")) as f64,
            DType::U32 => u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")) as f64,
            DType::U8 => bytes[0] as f64,
        })
        .collect())
}

fn full_value(
    device: Arc<CudaDevice>,
    shape: Vec<usize>,
    dtype: DType,
    value: f64,
) -> Result<CudaValue, String> {
    let len = element_count(&shape)?;
    if dtype == DType::I64 {
        return CudaValue::from_i64_host(device, shape, &vec![value as i64; len]);
    }
    if dtype == DType::F32 {
        return CudaValue::from_f32_host(device, shape, &vec![value as f32; len]);
    }
    let len_u32 = u32::try_from(len)
        .map_err(|_| format!("CUDA tensor element count {len} exceeds the first-slice limit"))?;
    let mut output =
        unsafe { device.stream.alloc::<f64>(len) }.map_err(|error| error.to_string())?;
    if len != 0 {
        let mut launch = device.stream.launch_builder(&device.fill_f64);
        let dtype = dtype_code(dtype);
        launch.arg(&mut output);
        launch.arg(&value);
        launch.arg(&len_u32);
        launch.arg(&dtype);
        unsafe { launch.launch(LaunchConfig::for_num_elems(len_u32)) }
            .map_err(|error| error.to_string())?;
    }
    CudaValue::from_buffer(device, shape, dtype, output)
}

fn child_index(index: &GraphIndex, node: &Arc<Node>) -> Result<usize, String> {
    index
        .dense_id(node.id)
        .map(|id| id.index())
        .ok_or_else(|| "CUDA compilation lost a graph dependency".to_string())
}

fn permutation_preserves_storage(shape: &[usize], dims: &[usize]) -> bool {
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

fn resolve_alias_instruction(instructions: &[Instruction], mut index: usize) -> usize {
    while let Instruction::Alias { a, .. } = &instructions[index] {
        index = *a;
    }
    index
}

struct MuseGatePattern {
    value: Arc<Node>,
    gate: Arc<Node>,
    multiplier: Option<Arc<Node>>,
    covered: Vec<Arc<Node>>,
}

fn is_full(node: &Arc<Node>, expected: f64) -> bool {
    matches!(&node.kind, NodeKind::Full { value, .. } if *value == expected)
}

fn sigmoid_pattern(node: &Arc<Node>) -> Option<(Arc<Node>, Vec<Arc<Node>>)> {
    let NodeKind::Add { a, b } = &node.kind else {
        return None;
    };
    let scaled = if is_full(a, 0.5) {
        b
    } else if is_full(b, 0.5) {
        a
    } else {
        return None;
    };
    let NodeKind::Div { a: tanh, b: two } = &scaled.kind else {
        return None;
    };
    if !is_full(two, 2.0) {
        return None;
    }
    let NodeKind::Tanh { a: half_gate } = &tanh.kind else {
        return None;
    };
    let NodeKind::Div { a: gate, b: two } = &half_gate.kind else {
        return None;
    };
    if !is_full(two, 2.0) {
        return None;
    }
    Some((
        gate.clone(),
        vec![
            half_gate.clone(),
            tanh.clone(),
            scaled.clone(),
            node.clone(),
        ],
    ))
}

fn silu_pattern(node: &Arc<Node>) -> Option<(Arc<Node>, Vec<Arc<Node>>)> {
    let NodeKind::Mul { a, b } = &node.kind else {
        return None;
    };
    for (gate, sigmoid) in [(a, b), (b, a)] {
        let Some((sigmoid_gate, mut covered)) = sigmoid_pattern(sigmoid) else {
            continue;
        };
        if sigmoid_gate.id == gate.id {
            covered.push(node.clone());
            return Some((gate.clone(), covered));
        }
    }
    None
}

fn muse_gate_pattern(node: &Arc<Node>) -> Option<MuseGatePattern> {
    if node.dtype != DType::F32 {
        return None;
    }
    let NodeKind::Mul { a, b } = &node.kind else {
        return None;
    };
    for (silu, multiplier) in [(a, b), (b, a)] {
        if let Some((gate, covered)) = silu_pattern(silu) {
            if gate.shape == node.shape && multiplier.shape == node.shape {
                return Some(MuseGatePattern {
                    value: gate.clone(),
                    gate,
                    multiplier: Some(multiplier.clone()),
                    covered,
                });
            }
        }
    }
    for (value, sigmoid) in [(a, b), (b, a)] {
        if let Some((gate, covered)) = sigmoid_pattern(sigmoid) {
            if value.shape == node.shape && gate.shape == node.shape {
                return Some(MuseGatePattern {
                    value: value.clone(),
                    gate,
                    multiplier: None,
                    covered,
                });
            }
        }
    }
    None
}

fn apply_muse_gate_fusions(index: &GraphIndex, instructions: &mut [Instruction]) {
    for (endpoint, node) in index.order.iter().enumerate() {
        let Some(pattern) = muse_gate_pattern(node) else {
            continue;
        };
        let mut allowed = pattern
            .covered
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        allowed.push(node.id);
        let exclusive = pattern.covered.iter().all(|covered| {
            index.dense_id(covered.id).is_some_and(|dense| {
                !index.roots.iter().any(|root| root.index() == dense.index())
                    && index.consumers[dense.index()]
                        .iter()
                        .all(|consumer| allowed.contains(&index.order[consumer.index()].id))
            })
        });
        if !exclusive {
            continue;
        }
        for covered in &pattern.covered {
            if let Some(dense) = index.dense_id(covered.id) {
                instructions[dense.index()] = Instruction::Noop;
            }
        }
        instructions[endpoint] = Instruction::MuseGate {
            value: index
                .dense_id(pattern.value.id)
                .expect("fused value belongs to graph")
                .index(),
            gate: index
                .dense_id(pattern.gate.id)
                .expect("fused gate belongs to graph")
                .index(),
            multiplier: pattern.multiplier.map(|multiplier| {
                index
                    .dense_id(multiplier.id)
                    .expect("fused multiplier belongs to graph")
                    .index()
            }),
            shape: node.shape.clone(),
        };
    }
}

fn apply_muse_projection_pair_fusions(instructions: &mut [Instruction]) {
    for first in 0..instructions.len() {
        let (x, shape, rows, columns) = match &instructions[first] {
            Instruction::QuantizedLinear {
                x,
                bias: None,
                shape,
                rows,
                columns,
                codec: GgmlKQuant::Q2K | GgmlKQuant::Q3K,
                pair: None,
                ..
            } => (*x, shape.clone(), *rows, *columns),
            _ => continue,
        };
        let Some(second) = (first + 1..instructions.len()).find(|&candidate| {
            matches!(
                &instructions[candidate],
                Instruction::QuantizedLinear {
                    x: candidate_x,
                    weight: candidate_weight,
                    bias: None,
                    shape: candidate_shape,
                    rows: candidate_rows,
                    columns: candidate_columns,
                    codec: candidate_codec,
                    pair: None,
                    ..
                } if *candidate_x == x
                    && *candidate_rows == rows
                    && *candidate_columns == columns
                    && matches!(candidate_codec, GgmlKQuant::Q2K | GgmlKQuant::Q3K)
                    && *candidate_shape == shape
                    && instructions
                        .get(*candidate_weight)
                        .is_some_and(|weight| matches!(weight, Instruction::Value(_)))
            )
        }) else {
            continue;
        };
        let (second_weight, second_codec) = match &instructions[second] {
            Instruction::QuantizedLinear { weight, codec, .. } => (*weight, *codec),
            _ => unreachable!("paired projection candidate changed during fusion"),
        };
        let Instruction::QuantizedLinear { pair, .. } = &mut instructions[first] else {
            unreachable!("paired projection candidate changed during fusion");
        };
        *pair = Some(QuantizedLinearPair {
            output: second,
            weight: second_weight,
            codec: second_codec,
        });
        instructions[second] = Instruction::Noop;
    }
}

fn apply_muse_residual_rms_fusions(index: &GraphIndex, instructions: &mut [Instruction]) {
    for (endpoint, node) in index.order.iter().enumerate() {
        if node.dtype != DType::F32 || node.shape.last().is_none_or(|width| *width <= 512) {
            continue;
        }
        let NodeKind::Add { a, b } = &node.kind else {
            continue;
        };
        let matched = [(a, b), (b, a)].into_iter().find_map(|(rms, residual)| {
            let NodeKind::RmsNorm { x, weight, eps } = &rms.kind else {
                return None;
            };
            (x.shape == node.shape && residual.shape == node.shape).then_some((
                rms,
                x,
                weight.as_ref(),
                residual,
                *eps,
            ))
        });
        let Some((rms, x, weight, residual, eps)) = matched else {
            continue;
        };
        let Some(rms_dense) = index.dense_id(rms.id) else {
            continue;
        };
        if !index.consumers[rms_dense.index()]
            .iter()
            .all(|consumer| consumer.index() == endpoint)
        {
            continue;
        }
        instructions[rms_dense.index()] = Instruction::Noop;
        instructions[endpoint] = Instruction::MuseResidualRmsNorm {
            x: index
                .dense_id(x.id)
                .expect("fused RMSNorm input belongs to graph")
                .index(),
            weight: weight.map(|weight| {
                index
                    .dense_id(weight.id)
                    .expect("fused RMSNorm weight belongs to graph")
                    .index()
            }),
            residual: index
                .dense_id(residual.id)
                .expect("fused residual belongs to graph")
                .index(),
            shape: node.shape.clone(),
            eps,
        };
    }
}

struct MusePreparedHead {
    x: Arc<Node>,
    weight: Arc<Node>,
    eps: f64,
    rope_theta: Option<f64>,
    covered: Vec<Arc<Node>>,
}

fn muse_prepared_head(node: &Arc<Node>) -> Option<MusePreparedHead> {
    let (rms, rope_theta, mut covered) = match &node.kind {
        NodeKind::RotaryEmbedding {
            x,
            seq_len: 1,
            theta,
            offset: PositionOffset::Cursor,
            layout: effect_torch_graph::RotaryLayout::InterleavedPairs,
        } => (x, Some(*theta), vec![node.clone()]),
        _ => (node, None, Vec::new()),
    };
    let NodeKind::RmsNorm { x, weight, eps } = &rms.kind else {
        return None;
    };
    let weight = weight.as_ref()?.clone();
    covered.push(rms.clone());
    Some(MusePreparedHead {
        x: x.clone(),
        weight,
        eps: *eps,
        rope_theta,
        covered,
    })
}

fn covered_is_exclusive(index: &GraphIndex, covered: &[Arc<Node>], endpoint: usize) -> bool {
    let mut allowed = covered.iter().map(|node| node.id).collect::<Vec<_>>();
    allowed.push(index.order[endpoint].id);
    covered.iter().all(|node| {
        index.dense_id(node.id).is_some_and(|dense| {
            !index.roots.iter().any(|root| root.index() == dense.index())
                && index.consumers[dense.index()]
                    .iter()
                    .all(|consumer| allowed.contains(&index.order[consumer.index()].id))
        })
    })
}

fn apply_muse_attention_fusions(index: &GraphIndex, instructions: &mut [Instruction]) {
    for (endpoint, node) in index.order.iter().enumerate() {
        let NodeKind::KvAttention { q, k, v, mode, .. } = &node.kind else {
            continue;
        };
        if node.dtype != DType::F32
            || *mode == KvAttentionMode::BidirectionalBlock
            || q.shape != [1, 32, 1, 128]
            || k.shape != [1, 2, 1, 128]
            || v.shape != [1, 2, 1, 128]
        {
            continue;
        }
        let Some(q) = muse_prepared_head(q) else {
            continue;
        };
        let Some(k) = muse_prepared_head(k) else {
            continue;
        };
        if q.rope_theta != k.rope_theta
            || !covered_is_exclusive(index, &q.covered, endpoint)
            || !covered_is_exclusive(index, &k.covered, endpoint)
        {
            continue;
        }
        for covered in q.covered.iter().chain(&k.covered) {
            instructions[index
                .dense_id(covered.id)
                .expect("fused Q/K preparation belongs to graph")
                .index()] = Instruction::Noop;
        }
        let Instruction::KvAttention { muse, .. } = &mut instructions[endpoint] else {
            unreachable!("KV semantic node lowers to KV instruction");
        };
        *muse = Some(MuseAttentionPreparation {
            q: index
                .dense_id(q.x.id)
                .expect("fused query input belongs to graph")
                .index(),
            k: index
                .dense_id(k.x.id)
                .expect("fused key input belongs to graph")
                .index(),
            q_weight: index
                .dense_id(q.weight.id)
                .expect("fused query weight belongs to graph")
                .index(),
            k_weight: index
                .dense_id(k.weight.id)
                .expect("fused key weight belongs to graph")
                .index(),
            q_eps: q.eps,
            k_eps: k.eps,
            rope_theta: q.rope_theta,
        });
    }
}

/// Compiles one nonempty, single-device CUDA graph.
pub fn compile(roots: Vec<Arc<Node>>, ordinal: u32) -> Result<CudaExecutable, String> {
    compile_with_options(roots, ordinal, CompileOptions::from_environment())
}

pub fn compile_with_options(
    roots: Vec<Arc<Node>>,
    ordinal: u32,
    options: CompileOptions,
) -> Result<CudaExecutable, String> {
    compile_inner(roots, ordinal, options, None)
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
    compile_inner(roots, ordinal, options, Some((cursor_slot, cursor_tensor)))
}

fn compile_inner(
    roots: Vec<Arc<Node>>,
    ordinal: u32,
    options: CompileOptions,
    state_cursor: Option<(u32, bool)>,
) -> Result<CudaExecutable, String> {
    if roots.is_empty() {
        return Err("compile: expected at least one root".to_string());
    }
    let expected_device = Device::Cuda(ordinal);
    if roots.iter().any(|root| root.device != expected_device) {
        return Err(format!(
            "compile: every root must use CUDA device {ordinal}"
        ));
    }
    let device = CudaDevice::get(ordinal)?;
    let mut request = ProgramRequest::from_roots(roots, options);
    if let Some((slot, tensor)) = state_cursor {
        request = request.with_state_cursor(StateCursorSlot::new(slot, tensor));
    }
    let prepared = request.prepare()?;
    let index = &prepared.index;
    let rms_quantized = index
        .order
        .iter()
        .enumerate()
        .map(|(dense, node)| {
            matches!(node.kind, NodeKind::RmsNorm { .. })
                && !index.consumers[dense].is_empty()
                && !index.roots.iter().any(|root| root.index() == dense)
                && index.consumers[dense].iter().all(|consumer| {
                    let consumer = &index.order[consumer.index()];
                    match &consumer.kind {
                        NodeKind::QuantizedLinear { weight_shape, .. } => {
                            element_count(&consumer.shape)
                                .is_ok_and(|len| len / weight_shape[0] < BF16_GEMM_MIN_VECTORS)
                        }
                        _ => false,
                    }
                })
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let mut driver = CompilerDriver::new(&prepared)?;
    let mut instructions = Vec::with_capacity(index.order.len());
    driver.lower(|unit, index, _| {
        let LoweringUnit::Node(dense) = unit else {
            return Err(
                "compile: the shared compiler selected a region CUDA cannot lower".to_string(),
            );
        };
        if dense.index() != instructions.len() {
            return Err("compile: CUDA lowering order is not dense postorder".to_string());
        }
        let node = index
            .node(dense)
            .ok_or_else(|| format!("compile: dense CUDA node {dense} is out of range"))?;
        if node.device != expected_device {
            return Err(format!(
                "compile: graph node {} is not on CUDA device {ordinal}",
                node.id
            ));
        }
        let instruction = match &node.kind {
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
            NodeKind::Input {
                slot, shape, dtype, ..
            } => {
                if state_cursor.is_some_and(|(cursor_slot, _)| cursor_slot == *slot) {
                    let tensor = state_cursor.is_some_and(|(_, tensor)| tensor);
                    if !tensor || *dtype != DType::I64 {
                        return Err(
                            "compile: state cursor tensor has an invalid signature".to_string()
                        );
                    }
                    Instruction::StateCursor {
                        tensor,
                        shape: shape.clone(),
                        dtype: *dtype,
                    }
                } else {
                    let binding = index.slots[..*slot as usize]
                        .iter()
                        .filter(|declaration| !declaration.scalar)
                        .count();
                    Instruction::Input {
                        binding,
                        scalar: false,
                        shape: shape.clone(),
                        dtype: *dtype,
                    }
                }
            }
            NodeKind::ScalarInput { slot, dtype, .. } => {
                if state_cursor.is_some_and(|(cursor_slot, _)| cursor_slot == *slot) {
                    let tensor = state_cursor.is_some_and(|(_, tensor)| tensor);
                    if tensor || *dtype != DType::I64 {
                        return Err(
                            "compile: state cursor scalar has an invalid signature".to_string()
                        );
                    }
                    Instruction::StateCursor {
                        tensor,
                        shape: Vec::new(),
                        dtype: *dtype,
                    }
                } else {
                    let binding = index.slots[..*slot as usize]
                        .iter()
                        .filter(|declaration| declaration.scalar)
                        .count();
                    Instruction::Input {
                        binding,
                        scalar: true,
                        shape: Vec::new(),
                        dtype: *dtype,
                    }
                }
            }
            NodeKind::FromBytes {
                data, shape, dtype, ..
            } => Instruction::Value(if *dtype == DType::I64 {
                CudaValue::from_i64_host(
                    device.clone(),
                    shape.clone(),
                    &data
                        .chunks_exact(8)
                        .map(|bytes| {
                            i64::from_le_bytes(bytes.try_into().expect("eight-byte chunk"))
                        })
                        .collect::<Vec<_>>(),
                )?
            } else {
                CudaValue::from_host(
                    device.clone(),
                    shape.clone(),
                    *dtype,
                    &decode(data, *dtype)?,
                )?
            }),
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
            NodeKind::Randn { shape, dtype, .. } => Instruction::Random {
                shape: shape.clone(),
                dtype: *dtype,
                normal: true,
                lo: 0.0,
                hi: 1.0,
                provenance: node.id,
            },
            NodeKind::Uniform {
                shape,
                dtype,
                lo,
                hi,
                ..
            } => Instruction::Random {
                shape: shape.clone(),
                dtype: *dtype,
                normal: false,
                lo: *lo,
                hi: *hi,
                provenance: node.id,
            },
            NodeKind::Arange {
                start, step, dtype, ..
            } => Instruction::Sequence {
                shape: node.shape.clone(),
                dtype: *dtype,
                eye: false,
                start: *start,
                step: *step,
            },
            NodeKind::Eye { n, dtype, .. } => Instruction::Sequence {
                shape: vec![*n, *n],
                dtype: *dtype,
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
                    shape: node.shape.clone(),
                    a_shape: a.shape.clone(),
                    b_shape: b.shape.clone(),
                    source_dtype: a.dtype,
                    dtype: node.dtype,
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
                    shape: node.shape.clone(),
                    source_dtype: a.dtype,
                    dtype: node.dtype,
                    parameter,
                }
            }
            NodeKind::Reshape { a, .. }
            | NodeKind::StopGradient { a }
            | NodeKind::Checkpoint { a }
            | NodeKind::Expose { a, .. } => Instruction::Alias {
                a: child_index(&index, a)?,
                shape: node.shape.clone(),
            },
            NodeKind::BroadcastTo { a, .. } => {
                let mut input_shape = vec![1; node.shape.len() - a.shape.len()];
                input_shape.extend_from_slice(&a.shape);
                Instruction::Reindex {
                    op: 0,
                    a: child_index(&index, a)?,
                    shape: node.shape.clone(),
                    input_shape,
                    parameters: Vec::new(),
                    dtype: node.dtype,
                }
            }
            NodeKind::Permute { a, dims } if permutation_preserves_storage(&a.shape, dims) => {
                Instruction::Alias {
                    a: child_index(&index, a)?,
                    shape: node.shape.clone(),
                }
            }
            NodeKind::Permute { a, dims } => Instruction::Reindex {
                op: 1,
                a: child_index(&index, a)?,
                shape: node.shape.clone(),
                input_shape: a.shape.clone(),
                parameters: dims.iter().map(|dim| *dim as u64).collect(),
                dtype: node.dtype,
            },
            NodeKind::Slice { a, ranges } => Instruction::Reindex {
                op: 2,
                a: child_index(&index, a)?,
                shape: node.shape.clone(),
                input_shape: a.shape.clone(),
                parameters: ranges
                    .iter()
                    .flat_map(|(start, _, stride)| [*start as u64, *stride as u64])
                    .collect(),
                dtype: node.dtype,
            },
            NodeKind::Where { cond, a, b } => Instruction::Where {
                cond: child_index(&index, cond)?,
                a: child_index(&index, a)?,
                b: child_index(&index, b)?,
                shape: node.shape.clone(),
                cond_shape: cond.shape.clone(),
                a_shape: a.shape.clone(),
                b_shape: b.shape.clone(),
                dtype: node.dtype,
            },
            NodeKind::Concat { a, b, dim } => Instruction::Concat {
                a: child_index(&index, a)?,
                b: child_index(&index, b)?,
                shape: node.shape.clone(),
                a_shape: a.shape.clone(),
                b_shape: b.shape.clone(),
                dim: u32::try_from(*dim).map_err(|_| "CUDA concat dimension exceeds u32")?,
                dtype: node.dtype,
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
                    shape: node.shape.clone(),
                    input_shape: a.shape.clone(),
                    dims: dims.clone(),
                    dtype: node.dtype,
                }
            }
            NodeKind::Matmul { a, b } => Instruction::Matmul {
                a: child_index(&index, a)?,
                b: child_index(&index, b)?,
                shape: node.shape.clone(),
                a_shape: a.shape.clone(),
                b_shape: b.shape.clone(),
                dtype: node.dtype,
            },
            NodeKind::Argmax { a, dim } | NodeKind::Argmin { a, dim } => {
                let mut index_shape = node.shape.clone();
                index_shape.insert(*dim, 1);
                Instruction::Index {
                    op: u32::from(matches!(node.kind, NodeKind::Argmin { .. })),
                    a: child_index(&index, a)?,
                    indexes: None,
                    src: None,
                    shape: node.shape.clone(),
                    input_shape: a.shape.clone(),
                    index_shape,
                    dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
                    dtype: node.dtype,
                }
            }
            NodeKind::Cumsum { a, dim } => Instruction::Index {
                op: 2,
                a: child_index(&index, a)?,
                indexes: None,
                src: None,
                shape: node.shape.clone(),
                input_shape: a.shape.clone(),
                index_shape: node.shape.clone(),
                dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
                dtype: node.dtype,
            },
            NodeKind::IndexSelect { a, dim, indexes } => Instruction::Index {
                op: 3,
                a: child_index(&index, a)?,
                indexes: Some(child_index(&index, indexes)?),
                src: None,
                shape: node.shape.clone(),
                input_shape: a.shape.clone(),
                index_shape: node.shape.clone(),
                dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
                dtype: node.dtype,
            },
            NodeKind::Gather { a, dim, indexes } => Instruction::Index {
                op: 4,
                a: child_index(&index, a)?,
                indexes: Some(child_index(&index, indexes)?),
                src: None,
                shape: node.shape.clone(),
                input_shape: a.shape.clone(),
                index_shape: indexes.shape.clone(),
                dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                input_shape: a.shape.clone(),
                index_shape: indexes.shape.clone(),
                dim: u32::try_from(*dim).map_err(|_| "CUDA index dimension exceeds u32")?,
                dtype: node.dtype,
            },
            NodeKind::RmsNorm { x, weight, eps } => Instruction::RmsNorm {
                x: child_index(&index, x)?,
                weight: weight
                    .as_ref()
                    .map(|weight| child_index(&index, weight))
                    .transpose()?,
                shape: node.shape.clone(),
                eps: *eps,
                dtype: node.dtype,
            },
            NodeKind::CrossEntropy {
                logits,
                target,
                ignore_index,
                ..
            } => Instruction::CrossEntropy {
                logits: child_index(&index, logits)?,
                target: child_index(&index, target)?,
                shape: node.shape.clone(),
                logits_shape: logits.shape.clone(),
                ignore_index: *ignore_index,
                backward: false,
                dtype: node.dtype,
            },
            NodeKind::CrossEntropyBackward {
                logits,
                target,
                ignore_index,
                ..
            } => Instruction::CrossEntropy {
                logits: child_index(&index, logits)?,
                target: child_index(&index, target)?,
                shape: node.shape.clone(),
                logits_shape: logits.shape.clone(),
                ignore_index: *ignore_index,
                backward: true,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                x_shape: x.shape.clone(),
                weight_shape: weight.shape.clone(),
                ignore_index: *ignore_index,
                target_dtype: target.dtype,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                x_shape: x.shape.clone(),
                weight_shape: weight.shape.clone(),
                ignore_index: *ignore_index,
                target_dtype: target.dtype,
                dtype: node.dtype,
            },
            NodeKind::ChunkedHeadCeBackwardOut { of, index: output } => {
                if *output == 0 {
                    Instruction::Alias {
                        a: child_index(&index, of)?,
                        shape: node.shape.clone(),
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
                        shape: node.shape.clone(),
                        x_shape: x.shape.clone(),
                        weight_shape: weight.shape.clone(),
                        ignore_index: *ignore_index,
                        target_dtype: target.dtype,
                        dtype: node.dtype,
                    }
                }
            }
            NodeKind::PositionEmbedding { weight, .. } => Instruction::Reindex {
                op: 2,
                a: child_index(&index, weight)?,
                shape: node.shape.clone(),
                input_shape: weight.shape.clone(),
                parameters: vec![0, 1, 0, 1],
                dtype: node.dtype,
            },
            NodeKind::Linear { x, weight, bias } => Instruction::Linear {
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                bias: child_index(&index, bias)?,
                shape: node.shape.clone(),
                k_width: u32::try_from(weight.shape[0])
                    .map_err(|_| "CUDA linear input width exceeds u32")?,
                n_width: u32::try_from(weight.shape[1])
                    .map_err(|_| "CUDA linear output width exceeds u32")?,
                dtype: node.dtype,
            },
            NodeKind::QuantizedLinear {
                x,
                weight,
                bias,
                codec,
                weight_shape,
            } => Instruction::QuantizedLinear {
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                bias: bias
                    .as_ref()
                    .map(|bias| child_index(&index, bias))
                    .transpose()?,
                shape: node.shape.clone(),
                rows: checked_len(weight_shape[0])?,
                columns: checked_len(weight_shape[1])?,
                row_bytes: checked_len(weight.shape[1])?,
                codec: *codec,
                pair: None,
            },
            NodeKind::QuantizedEmbedding {
                indexes,
                weight,
                codec,
                weight_shape,
                ..
            } => Instruction::QuantizedEmbedding {
                indexes: child_index(&index, indexes)?,
                weight: child_index(&index, weight)?,
                shape: node.shape.clone(),
                rows: checked_len(weight_shape[0])?,
                columns: checked_len(weight_shape[1])?,
                row_bytes: checked_len(weight.shape[1])?,
                codec: *codec,
            },
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
                    shape: node.shape.clone(),
                    width: checked_len(width)?,
                    rows: checked_len(element_count(&x.shape)? / width)?,
                    eps: *eps,
                    dtype: node.dtype,
                }
            }
            NodeKind::LayerNormBackward { x, weight, g, eps } => {
                let width = element_count(&weight.shape)?;
                Instruction::LayerNorm {
                    op: 1,
                    x: child_index(&index, x)?,
                    weight: child_index(&index, weight)?,
                    other: child_index(&index, g)?,
                    shape: node.shape.clone(),
                    width: checked_len(width)?,
                    rows: checked_len(element_count(&x.shape)? / width)?,
                    eps: *eps,
                    dtype: node.dtype,
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
                            return Err(
                                "compile: invalid layer norm backward output index".to_string()
                            )
                        }
                    },
                    x: child_index(&index, x)?,
                    weight: child_index(&index, weight)?,
                    other: child_index(&index, g)?,
                    shape: node.shape.clone(),
                    width: checked_len(width)?,
                    rows: checked_len(element_count(&x.shape)? / width)?,
                    eps: *eps,
                    dtype: node.dtype,
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
                shape: node.shape.clone(),
                q_shape: q.shape.clone(),
                k_shape: k.shape.clone(),
                v_shape: v.shape.clone(),
                scale: *scale,
                causal: *causal,
                window: window.local(),
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                q_shape: q.shape.clone(),
                k_shape: k.shape.clone(),
                v_shape: v.shape.clone(),
                scale: *scale,
                causal: *causal,
                window: window.local(),
                dtype: node.dtype,
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
                    shape: node.shape.clone(),
                    q_shape: q.shape.clone(),
                    k_shape: k.shape.clone(),
                    v_shape: v.shape.clone(),
                    scale: *scale,
                    causal: *causal,
                    window: window.local(),
                    dtype: node.dtype,
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
                dtype: node.dtype,
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
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                x_shape: x.shape.clone(),
                w_shape: w.shape.clone(),
                stride: u32::try_from(*stride)
                    .map_err(|_| "CUDA convolution stride exceeds u32")?,
                padding: u32::try_from(*padding)
                    .map_err(|_| "CUDA convolution padding exceeds u32")?,
                dilation: u32::try_from(*dilation)
                    .map_err(|_| "CUDA convolution dilation exceeds u32")?,
                groups: u32::try_from(*groups)
                    .map_err(|_| "CUDA convolution groups exceeds u32")?,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                x_shape: x.shape.clone(),
                w_shape: w.shape.clone(),
                stride: u32::try_from(*stride)
                    .map_err(|_| "CUDA convolution stride exceeds u32")?,
                padding: u32::try_from(*padding)
                    .map_err(|_| "CUDA convolution padding exceeds u32")?,
                dilation: u32::try_from(*dilation)
                    .map_err(|_| "CUDA convolution dilation exceeds u32")?,
                groups: u32::try_from(*groups)
                    .map_err(|_| "CUDA convolution groups exceeds u32")?,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                x_shape: x.shape.clone(),
                w_shape: w.shape.clone(),
                stride: u32::try_from(*stride)
                    .map_err(|_| "CUDA convolution stride exceeds u32")?,
                padding: u32::try_from(*padding)
                    .map_err(|_| "CUDA convolution padding exceeds u32")?,
                dilation: u32::try_from(*dilation)
                    .map_err(|_| "CUDA convolution dilation exceeds u32")?,
                groups: u32::try_from(*groups)
                    .map_err(|_| "CUDA convolution groups exceeds u32")?,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                x_shape: x.shape.clone(),
                w_shape: w.shape.clone(),
                stride: u32::try_from(*stride)
                    .map_err(|_| "CUDA convolution stride exceeds u32")?,
                padding: u32::try_from(*padding)
                    .map_err(|_| "CUDA convolution padding exceeds u32")?,
                dilation: u32::try_from(*dilation)
                    .map_err(|_| "CUDA convolution dilation exceeds u32")?,
                groups: u32::try_from(*groups)
                    .map_err(|_| "CUDA convolution groups exceeds u32")?,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                x_shape: x.shape.clone(),
                w_shape: g.shape.clone(),
                stride: u32::try_from(*stride)
                    .map_err(|_| "CUDA convolution stride exceeds u32")?,
                padding: u32::try_from(*padding)
                    .map_err(|_| "CUDA convolution padding exceeds u32")?,
                dilation: u32::try_from(*dilation)
                    .map_err(|_| "CUDA convolution dilation exceeds u32")?,
                groups: u32::try_from(*groups)
                    .map_err(|_| "CUDA convolution groups exceeds u32")?,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                x_shape: x.shape.clone(),
                w_shape: g.shape.clone(),
                stride: u32::try_from(*stride)
                    .map_err(|_| "CUDA convolution stride exceeds u32")?,
                padding: u32::try_from(*padding)
                    .map_err(|_| "CUDA convolution padding exceeds u32")?,
                dilation: u32::try_from(*dilation)
                    .map_err(|_| "CUDA convolution dilation exceeds u32")?,
                groups: u32::try_from(*groups)
                    .map_err(|_| "CUDA convolution groups exceeds u32")?,
                dtype: node.dtype,
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
                shape: node.shape.clone(),
                beta1: *beta1,
                beta2: *beta2,
                eps: *eps,
                weight_decay: *weight_decay,
                dampening: 0.0,
                nesterov: false,
                dtype: node.dtype,
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
                    shape: node.shape.clone(),
                    beta1: *beta1,
                    beta2: *beta2,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    dampening: 0.0,
                    nesterov: false,
                    dtype: node.dtype,
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
                shape: node.shape.clone(),
                beta1: *momentum,
                beta2: 0.0,
                eps: 0.0,
                weight_decay: *weight_decay,
                dampening: *dampening,
                nesterov: *nesterov,
                dtype: node.dtype,
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
                    shape: node.shape.clone(),
                    beta1: *momentum,
                    beta2: 0.0,
                    eps: 0.0,
                    weight_decay: *weight_decay,
                    dampening: *dampening,
                    nesterov: *nesterov,
                    dtype: node.dtype,
                }
            }
            NodeKind::ShortConv1d { x, weight } => Instruction::ShortConv {
                op: 0,
                state_layer: None,
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                g: None,
                shape: node.shape.clone(),
                input_shape: x.shape.clone(),
                weight_shape: weight.shape.clone(),
                dtype: node.dtype,
            },
            NodeKind::ShortConv1dBackwardX { x, weight, g } => Instruction::ShortConv {
                op: 1,
                state_layer: None,
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                g: Some(child_index(&index, g)?),
                shape: node.shape.clone(),
                input_shape: x.shape.clone(),
                weight_shape: weight.shape.clone(),
                dtype: node.dtype,
            },
            NodeKind::ShortConv1dBackwardW { x, weight, g } => Instruction::ShortConv {
                op: 2,
                state_layer: None,
                x: child_index(&index, x)?,
                weight: child_index(&index, weight)?,
                g: Some(child_index(&index, g)?),
                shape: node.shape.clone(),
                input_shape: x.shape.clone(),
                weight_shape: weight.shape.clone(),
                dtype: node.dtype,
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
                shape: node.shape.clone(),
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
                shape: node.shape.clone(),
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
                    shape: node.shape.clone(),
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
                shape: node.shape.clone(),
                input_shape: x.shape.clone(),
                weight_shape: weight.shape.clone(),
                dtype: node.dtype,
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
                shape: node.shape.clone(),
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
                    shape: node.shape.clone(),
                    tokens,
                    dtype: node.dtype,
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
                shape: node.shape.clone(),
                q_shape: q.shape.clone(),
                k_shape: k.shape.clone(),
                scale: *scale,
                layer: *layer as usize,
                window: *window,
                bidirectional: *mode == KvAttentionMode::BidirectionalBlock,
                dtype: node.dtype,
                muse: None,
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
        };
        instructions.push(instruction);
        Ok(())
    })?;
    apply_muse_gate_fusions(index, &mut instructions);
    apply_muse_residual_rms_fusions(index, &mut instructions);
    apply_muse_attention_fusions(index, &mut instructions);
    apply_muse_projection_pair_fusions(&mut instructions);
    let gate_quantized = instructions
        .iter()
        .enumerate()
        .map(|(instruction_index, instruction)| {
            matches!(instruction, Instruction::MuseGate { .. })
                && !index.consumers[instruction_index].is_empty()
                && !index
                    .roots
                    .iter()
                    .any(|root| root.index() == instruction_index)
                && index.consumers[instruction_index].iter().all(|consumer| {
                    let consumer = &index.order[consumer.index()];
                    match &consumer.kind {
                        NodeKind::QuantizedLinear { weight_shape, .. } => {
                            element_count(&consumer.shape)
                                .is_ok_and(|len| len / weight_shape[0] < BF16_GEMM_MIN_VECTORS)
                        }
                        _ => false,
                    }
                })
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let mut deferred_residuals = vec![false; instructions.len()];
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        if !rms_quantized[instruction_index] {
            continue;
        }
        let Instruction::RmsNorm { x, .. } = instruction else {
            continue;
        };
        let source = resolve_alias_instruction(&instructions, *x);
        if matches!(
            instructions[source],
            Instruction::MuseResidualRmsNorm { .. }
        ) {
            deferred_residuals[source] = true;
        }
    }
    let instruction_metadata = instructions
        .iter()
        .map(|instruction| {
            instruction
                .static_metadata()
                .map(|metadata| {
                    device
                        .stream
                        .clone_htod(&metadata)
                        .map(CudaBuffer::from_slice)
                        .map_err(|error| error.to_string())
                })
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let root_indices = index.roots.iter().map(|root| root.index()).collect();
    let outputs = prepared
        .roots
        .iter()
        .map(|root| (root.shape.clone(), root.dtype))
        .collect();
    let scratch_value = instructions
        .iter()
        .map(Instruction::scratch_bytes)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .any(|bytes| bytes != 0)
        .then_some(instructions.len());
    let program = Arc::new(lowered_program(index, &instructions)?);
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
        .map_err(|error| format!("compile: CUDA memory planning failed: {error}"))?;
    driver.phase(PHYSICAL_PLANNING_PHASE, || Ok::<_, String>(()))?;
    let command_count = instructions
        .iter()
        .filter(|instruction| instruction.encodes_command())
        .count();
    let artifact_assembly_started = Instant::now();
    let diagnostics = build_executable_diagnostics(
        &program,
        &memory,
        index,
        DiagnosticsInput {
            pipeline_count: 0,
            command_count,
            synchronization_count: 1,
            compile_phases: Vec::new(),
        },
        |name| *name,
    );
    let mut executable = CudaExecutable {
        id: NEXT_EXECUTABLE_ID.fetch_add(1, Ordering::Relaxed),
        device,
        program: Arc::clone(&program),
        memory,
        scratch_value,
        instructions: instructions.into_boxed_slice(),
        instruction_metadata: instruction_metadata.into_boxed_slice(),
        rms_quantized,
        gate_quantized,
        deferred_residuals: deferred_residuals.into_boxed_slice(),
        roots: root_indices,
        outputs,
        diagnostics,
        compiler_work: CompilerWorkReport::default(),
        runs: AtomicU64::new(0),
    };
    driver.record_phase(ARTIFACT_ASSEMBLY_PHASE, artifact_assembly_started.elapsed());
    let publication_started = Instant::now();
    let compiler_work = driver.finish_with_phase(&program, PUBLICATION_PHASE, publication_started);
    executable.diagnostics.compile_phases = compiler_work.compile_phases.clone();
    executable.compiler_work = compiler_work;
    Ok(executable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_runtime::{Location, StorageClass};

    #[test]
    fn shared_compiler_keeps_cuda_node_lowering_until_region_codegen_exists() {
        let device = Device::Cuda(2);
        let left = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![4],
            dtype: DType::F32,
            device: device.clone(),
        })
        .unwrap();
        let right = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![4],
            dtype: DType::F32,
            device,
        })
        .unwrap();
        let root = Node::new(NodeKind::Add { a: left, b: right }).unwrap();
        let prepared = ProgramRequest::from_roots(vec![root], CompileOptions::default())
            .prepare()
            .unwrap();
        let mut driver = CompilerDriver::new(&prepared).unwrap();
        let mut units = Vec::new();
        driver
            .lower(|unit, _, _| {
                units.push(unit);
                Ok(())
            })
            .unwrap();

        assert_eq!(units.len(), 3);
        assert!(units
            .iter()
            .all(|unit| matches!(unit, LoweringUnit::Node(_))));
        assert_eq!(driver.optimization().work.selected_regions, 0);

        let instructions = vec![
            Instruction::Input {
                binding: 0,
                scalar: false,
                shape: vec![4],
                dtype: DType::F32,
            },
            Instruction::Input {
                binding: 1,
                scalar: false,
                shape: vec![4],
                dtype: DType::F32,
            },
            Instruction::Binary {
                op: 0,
                a: 0,
                b: 1,
                shape: vec![4],
                a_shape: vec![4],
                b_shape: vec![4],
                source_dtype: DType::F32,
                dtype: DType::F32,
            },
        ];
        let program = lowered_program(&prepared.index, &instructions).unwrap();
        program.validate().unwrap();
        assert_eq!(program.values.len(), 3);
        assert_eq!(program.instructions.len(), 3);
        assert!(matches!(
            program.values[2].storage,
            effect_torch_compiler::ValueStorage::Planned {
                class: StorageClass::EscapingOutput,
                ..
            }
        ));
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
            .unwrap();
        assert!(matches!(memory.locations[2], Location::Segment { .. }));
        assert_eq!(memory.report.external_bytes, 32);
        assert_eq!(memory.report.output_bytes, CUDA_STORAGE_ALIGNMENT);
        let work = driver.finish(&program);
        assert_eq!(work.graph_index_builds, 1);
        assert_eq!(work.lowered_values, 3);
        assert_eq!(work.lowered_instructions, 3);
    }

    #[test]
    fn muse_gate_fusion_preserves_covered_roots() {
        let device = Device::Cuda(0);
        let gate = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![4],
            dtype: DType::F32,
            device: device.clone(),
        })
        .unwrap();
        let value = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![4],
            dtype: DType::F32,
            device: device.clone(),
        })
        .unwrap();
        let two = Node::new(NodeKind::Full {
            shape: Vec::new(),
            value: 2.0,
            dtype: DType::F32,
            device: device.clone(),
        })
        .unwrap();
        let half = Node::new(NodeKind::Full {
            shape: Vec::new(),
            value: 0.5,
            dtype: DType::F32,
            device,
        })
        .unwrap();
        let half_gate = Node::new(NodeKind::Div {
            a: gate,
            b: two.clone(),
        })
        .unwrap();
        let tanh = Node::new(NodeKind::Tanh { a: half_gate }).unwrap();
        let scaled = Node::new(NodeKind::Div { a: tanh, b: two }).unwrap();
        let sigmoid = Node::new(NodeKind::Add { a: half, b: scaled }).unwrap();
        let output = Node::new(NodeKind::Mul {
            a: value,
            b: sigmoid.clone(),
        })
        .unwrap();

        let index = GraphIndex::new(&[sigmoid, output.clone()]).unwrap();
        let mut instructions = index
            .order
            .iter()
            .map(|node| Instruction::Input {
                binding: 0,
                scalar: false,
                shape: node.shape.clone(),
                dtype: node.dtype,
            })
            .collect::<Vec<_>>();
        apply_muse_gate_fusions(&index, &mut instructions);
        let output = index.dense_id(output.id).unwrap().index();
        assert!(!matches!(
            instructions[output],
            Instruction::MuseGate { .. }
        ));
    }

    #[test]
    fn attention_preparation_preserves_covered_roots() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![4],
            dtype: DType::F32,
            device: Device::Cuda(0),
        })
        .unwrap();
        let output = Node::new(NodeKind::Tanh { a: input.clone() }).unwrap();
        let index = GraphIndex::new(&[input.clone(), output.clone()]).unwrap();
        let endpoint = index.dense_id(output.id).unwrap().index();

        assert!(!covered_is_exclusive(&index, &[input], endpoint));
    }

    #[test]
    fn projection_pairing_rejects_a_later_dynamic_weight() {
        let input = || Instruction::Input {
            binding: 0,
            scalar: false,
            shape: vec![1],
            dtype: DType::U8,
        };
        let linear = |weight| Instruction::QuantizedLinear {
            x: 0,
            weight,
            bias: None,
            shape: vec![4],
            rows: 4,
            columns: 256,
            row_bytes: 84,
            codec: GgmlKQuant::Q2K,
            pair: None,
        };
        let mut instructions = vec![input(), input(), linear(1), input(), linear(3)];

        apply_muse_projection_pair_fusions(&mut instructions);

        assert!(matches!(
            instructions[2],
            Instruction::QuantizedLinear { pair: None, .. }
        ));
        assert!(matches!(
            instructions[4],
            Instruction::QuantizedLinear { pair: None, .. }
        ));
    }

    #[test]
    fn quantized_linear_scratch_is_block_aligned_and_plannable() {
        let layout = quantized_linear_scratch_layout(&[4], 4, 32).unwrap();
        assert_eq!(layout.padded_columns, 512);
        assert_eq!(layout.bytes, 576);

        let instruction = Instruction::QuantizedLinear {
            x: 0,
            weight: 1,
            bias: None,
            shape: vec![4],
            rows: 4,
            columns: 32,
            row_bytes: 84,
            codec: GgmlKQuant::Q2K,
            pair: None,
        };
        assert_eq!(instruction.scratch_bytes(), Ok(576));
    }
}

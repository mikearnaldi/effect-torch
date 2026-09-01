//! Computation graph types for effect-torch.
//!
//! A graph is a lazy DAG of [`Node`]s. Each node wraps one [`NodeKind`]
//! operation. Nodes share values through `Arc`, so ten consumers can refer to
//! one value. [`Node::new`] calls [`NodeKind::metadata`] to compute the
//! node's shape, dtype, and device. It rejects shape mismatches, core dtype or
//! device violations such as f64 on Metal, and invalid attention or
//! convolution geometry. [`Node`] keeps its fields public for low-level
//! integration. Direct struct literals bypass these checks, and backends
//! validate their own constraints again during compilation.
//!
//! # Architecture
//!
//! - [`NodeKind::Leaf`] stores a user tensor in a [`LeafSlot`]. Clearing
//!   the slot releases its storage but preserves the node's cached metadata.
//!   Later use fails with [`ClearedLeaf`].
//! - [`NodeKind::Input`] and [`NodeKind::ScalarInput`] are placeholders
//!   for compiled programs. They evaluate only after a compiled program binds
//!   them to call arguments as specified by RFC 0008.
//! - Operations such as [`NodeKind::Sdpa`], [`NodeKind::Linear`],
//!   [`NodeKind::RotaryEmbedding`], and [`NodeKind::ChunkedHeadCe`] retain
//!   their high-level semantics in the graph. Native backends can replace
//!   their reference implementations with fused kernels. Graph rewrites such
//!   as autodiff, decode compilation, and checkpoint recomputation can identify
//!   the operations by structure. Fused kernels do not change graph semantics.
//! - The decode rewrite creates [`NodeKind::KvAttention`],
//!   [`NodeKind::KdaRecurrence`], [`NodeKind::ConvState`], and
//!   [`NodeKind::LastTokenRow`]. User code does not create them. They access
//!   per-sequence state through the run's decode context, while the graph stays
//!   a pure function of its inputs.
//! - Autodiff emits closed-form adjoints such as [`NodeKind::SdpaBackward`],
//!   [`NodeKind::KdaBackward`], and [`NodeKind::ChunkedHeadCeBackward`].
//!   Each computes several gradient tensors in one evaluation. Its `*Out`
//!   variants select one result by index. Backward nodes do not support
//!   second-order autodiff.
//! - [`NodeKind::AdamWStep`] and [`NodeKind::SgdStep`] store the learning
//!   rate, bias corrections, and first-step flag as 0-d tensor children rather
//!   than captured constants, so a frozen compiled graph cannot replay a stale
//!   step count.
//!
//! # Traversal and memory safety
//!
//! [`node_children`] returns a node's direct children. [`remap_children`]
//! rebuilds a kind with mapped children so callers can copy subgraphs with
//! fresh ids. Graphs can be deep, so [`Node`]'s `Drop` implementation uses
//! an explicit worklist instead of recursion. This prevents long chains from
//! overflowing worker-thread stacks. The crate contains no `unsafe` code.

use effect_torch_runtime::{DType, GgmlKQuant};
use std::any::Any;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// Node::new issues monotonic IDs that are unique within the process. External
// code that constructs Node directly must preserve this identity.
static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(0);

/// A tensor's compute device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu(u32),
    Metal(u32),
    Cuda(u32),
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu(_))
    }
    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal(_))
    }
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
    pub fn same_device(&self, other: &Device) -> bool {
        self == other
    }
    pub fn ordinal(&self) -> u32 {
        match self {
            Device::Cpu(ordinal) | Device::Metal(ordinal) | Device::Cuda(ordinal) => *ordinal,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Device::Cpu(_) => "cpu",
            Device::Metal(_) => "metal",
            Device::Cuda(_) => "cuda",
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.name(), self.ordinal())
    }
}

/// How cross-entropy reduces per-element losses to the scalar result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossEntropyReduction {
    Mean,
    Sum,
}

type CeReduction = CrossEntropyReduction;

/// A node's cached shape, dtype, and device.
///
/// [`NodeKind::metadata`] computes these values at construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeMetadata {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
}

/// A user tensor stored in a graph leaf.
///
/// The graph needs metadata and a downcast path for evaluators. The embedding
/// crate defines the concrete storage type. Implementations must be
/// `Send + Sync` because threads share graphs.
pub trait LeafValue: Any + Send + Sync {
    fn shape(&self) -> Vec<usize>;
    fn dtype(&self) -> DType;
    fn device(&self) -> Device;
    fn as_any(&self) -> &dyn Any;
}

/// A mutex-protected leaf slot with one-shot clearing.
///
/// [`LeafSlot::clear`] releases the tensor storage but leaves cached node
/// metadata valid. It returns whether it removed a value and has no effect
/// after the first successful call. Accessors hold the lock only for brief,
/// non-recursive sections. A panic in user `LeafValue` clone or drop code can
/// poison the lock. Accessors treat this as an invariant failure and call
/// `unwrap`.
pub struct LeafSlot(Mutex<Option<Arc<dyn LeafValue>>>);

impl LeafSlot {
    /// Creates a slot holding `value`.
    pub fn new(value: impl LeafValue) -> Self {
        Self(Mutex::new(Some(Arc::new(value))))
    }

    /// Removes the value, returning `true` if one was present. Afterward,
    /// [`get`](Self::get) and node construction through this slot fail with
    /// [`ClearedLeaf`].
    pub fn clear(&self) -> bool {
        self.0.lock().unwrap().take().is_some()
    }

    /// Clones and downcasts the value to its concrete type. Fails with
    /// [`ClearedLeaf`] if the slot was cleared or the type does not match.
    pub fn get<T: LeafValue + Clone>(&self) -> Result<T, ClearedLeaf> {
        self.0
            .lock()
            .unwrap()
            .as_ref()
            .ok_or(ClearedLeaf)?
            .as_any()
            .downcast_ref::<T>()
            .cloned()
            .ok_or(ClearedLeaf)
    }

    fn metadata(&self) -> Result<NodeMetadata, ClearedLeaf> {
        let guard = self.0.lock().unwrap();
        let value = guard.as_ref().ok_or(ClearedLeaf)?;
        Ok(NodeMetadata {
            shape: value.shape(),
            dtype: value.dtype(),
            device: value.device(),
        })
    }
}

/// Error returned when code reads a cleared [`LeafSlot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClearedLeaf;
impl fmt::Display for ClearedLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tensor was cleared")
    }
}
impl Error for ClearedLeaf {}

// Validates channels, dtypes, and devices for Conv1d and Conv2d.
fn conv_check(
    op: &str,
    x: &Node,
    w: &Node,
    stride: usize,
    _padding: usize,
    dilation: usize,
    groups: usize,
) -> std::result::Result<(), String> {
    if stride < 1 || dilation < 1 || groups < 1 {
        return Err(format!(
            "{op}: stride, dilation and groups must be >= 1, got {stride}, {dilation}, {groups}"
        ));
    }
    let c_in = x.shape[1];
    let c_out = w.shape[0];
    if c_in % groups != 0 || c_out % groups != 0 {
        return Err(format!(
            "{op}: channels [{c_in}, {c_out}] are not divisible into {groups} groups"
        ));
    }
    if w.shape[1] != c_in / groups {
        return Err(format!(
            "{op}: weight has {} input channels per group, expected {}",
            w.shape[1],
            c_in / groups
        ));
    }
    if !x.dtype.is_float() || x.dtype != w.dtype {
        return Err(format!(
            "{op}: dtypes must be floating point and match, got {:?} and {:?}",
            x.dtype, w.dtype
        ));
    }
    if !x.device.same_device(&w.device) {
        return Err(format!("{op}: input and weight must be on the same device"));
    }
    Ok(())
}

// Validates rank-2 attention and batched forms with q [.., Hq, T, D],
// k [.., Hkv, S, D], and v [.., Hkv, S, Dv]. Returns [.., Hq, T, Dv].
fn sdpa_check(op: &str, q: &Node, k: &Node, v: &Node) -> Result<Vec<usize>, String> {
    let rank = q.shape.len();
    if rank < 2 || k.shape.len() != rank || v.shape.len() != rank {
        return Err(format!(
            "{op}: q, k and v must share a rank >= 2, got {:?}, {:?} and {:?}",
            q.shape, k.shape, v.shape
        ));
    }
    let leading = if rank < 3 { rank - 2 } else { rank - 3 };
    if q.shape[..leading] != k.shape[..leading] || q.shape[..leading] != v.shape[..leading] {
        return Err(format!(
            "{op}: leading dims must match, got {:?}, {:?} and {:?}",
            q.shape, k.shape, v.shape
        ));
    }
    if rank >= 3 {
        let q_heads = q.shape[rank - 3];
        let kv_heads = k.shape[rank - 3];
        if v.shape[rank - 3] != kv_heads {
            return Err(format!(
                "{op}: k and v heads mismatch, got {:?} and {:?}",
                k.shape, v.shape
            ));
        }
        if kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "{op}: query heads {q_heads} must be divisible by K/V heads {kv_heads}"
            ));
        }
    }
    if q.shape[rank - 1] != k.shape[rank - 1] {
        return Err(format!(
            "{op}: q and k head dims mismatch, got {:?} and {:?}",
            q.shape, k.shape
        ));
    }
    if k.shape[rank - 2] != v.shape[rank - 2] {
        return Err(format!(
            "{op}: k and v sequence lengths mismatch, got {:?} and {:?}",
            k.shape, v.shape
        ));
    }
    if !matches!(q.dtype, DType::F32 | DType::F64 | DType::BF16) {
        return Err(format!(
            "{op}: dtype must be f32, f64 or bf16, got {:?}",
            q.dtype
        ));
    }
    if k.dtype != q.dtype || v.dtype != q.dtype {
        return Err(format!(
            "{op}: q, k and v must share a dtype, got {:?}, {:?} and {:?}",
            q.dtype, k.dtype, v.dtype
        ));
    }
    if !k.device.same_device(&q.device) || !v.device.same_device(&q.device) {
        return Err(format!("{op}: q, k and v must be on the same device"));
    }
    let mut out = q.shape[..rank - 1].to_vec();
    out.push(v.shape[rank - 1]);
    Ok(out)
}

// Validates kda_chunk operands. The shapes are q, k, and log_decay
// [.., H, T, Dk], v [.., H, T, Dv], and beta [.., H, T, 1]. Leading
// dimensions must match exactly. Returns [.., H, T, Dv].
fn kda_check(
    op: &str,
    q: &Node,
    k: &Node,
    v: &Node,
    log_decay: &Node,
    beta: &Node,
) -> Result<Vec<usize>, String> {
    let rank = q.shape.len();
    if rank < 2
        || k.shape.len() != rank
        || v.shape.len() != rank
        || log_decay.shape.len() != rank
        || beta.shape.len() != rank
    {
        return Err(format!(
            "{op}: q, k, v, log_decay and beta must share a rank >= 2, got {:?}, {:?}, {:?}, {:?} and {:?}",
            q.shape, k.shape, v.shape, log_decay.shape, beta.shape
        ));
    }
    if k.shape != q.shape || log_decay.shape != q.shape {
        return Err(format!(
            "{op}: q, k and log_decay must share a shape, got {:?}, {:?} and {:?}",
            q.shape, k.shape, log_decay.shape
        ));
    }
    if v.shape[..rank - 1] != q.shape[..rank - 1] {
        return Err(format!(
            "{op}: v must match q on all but the head dim, got {:?} and {:?}",
            v.shape, q.shape
        ));
    }
    let mut beta_shape = q.shape.clone();
    beta_shape[rank - 1] = 1;
    if beta.shape != beta_shape {
        return Err(format!(
            "{op}: beta must have shape {beta_shape:?}, got {:?}",
            beta.shape
        ));
    }
    if !matches!(q.dtype, DType::F32 | DType::F64 | DType::BF16) {
        return Err(format!(
            "{op}: dtype must be f32, f64 or bf16, got {:?}",
            q.dtype
        ));
    }
    for (name, t) in [("k", k), ("v", v), ("log_decay", log_decay), ("beta", beta)] {
        if t.dtype != q.dtype {
            return Err(format!(
                "{op}: all operands must share a dtype, got {:?} and {:?} for {name}",
                q.dtype, t.dtype
            ));
        }
        if !t.device.same_device(&q.device) {
            return Err(format!("{op}: all operands must be on the same device"));
        }
    }
    let mut out = q.shape.clone();
    out[rank - 1] = v.shape[rank - 1];
    Ok(out)
}

// Validates chunked head CE operands: x [.., K], weight [K, V], bias [1, V]
// or [V], and integer target [..]. Returns the vocabulary size.
fn head_ce_check(
    op: &str,
    x: &Node,
    weight: &Node,
    bias: &Node,
    target: &Node,
) -> Result<usize, String> {
    let rank = x.shape.len();
    if rank < 2 || weight.shape.len() != 2 {
        return Err(format!(
            "{op}: x must be [.., K] with rank >= 2 and weight [K, V], got {:?} and {:?}",
            x.shape, weight.shape
        ));
    }
    let (k, v) = (weight.shape[0], weight.shape[1]);
    if x.shape[rank - 1] != k {
        return Err(format!(
            "{op}: x's last dim {} does not match the weight's input dim {k}",
            x.shape[rank - 1]
        ));
    }
    if bias.shape.len() > 2
        || (bias.shape.len() >= 1 && bias.shape[bias.shape.len() - 1] != v)
        || bias.shape.iter().product::<usize>() != v
    {
        return Err(format!(
            "{op}: bias must hold {v} values, got {:?}",
            bias.shape
        ));
    }
    if target.shape != x.shape[..rank - 1] {
        return Err(format!(
            "{op}: target shape {:?} does not match x's leading shape {:?}",
            target.shape,
            &x.shape[..rank - 1]
        ));
    }
    if !matches!(target.dtype, DType::I64 | DType::U32) {
        return Err(format!(
            "{op}: targets must be i64 or u32, got {:?}",
            target.dtype
        ));
    }
    if !matches!(x.dtype, DType::F32 | DType::F64 | DType::BF16) {
        return Err(format!(
            "{op}: x must be f32, f64 or bf16, got {:?}",
            x.dtype
        ));
    }
    if weight.dtype != x.dtype || bias.dtype != x.dtype {
        return Err(format!(
            "{op}: weight and bias must share x's dtype, got {:?}, {:?} and {:?}",
            x.dtype, weight.dtype, bias.dtype
        ));
    }
    for node in [weight, bias, target] {
        if !node.device.same_device(&x.device) {
            return Err(format!("{op}: all operands must be on the same device"));
        }
    }
    Ok(v)
}

// Validates short_conv1d operands x [.., T, C] and weight [C, K].
fn short_conv_check(op: &str, x: &Node, weight: &Node) -> Result<(), String> {
    if x.shape.len() < 2 || weight.shape.len() != 2 {
        return Err(format!(
            "{op}: expected x [.., T, C] and weight [C, K], got {:?} and {:?}",
            x.shape, weight.shape
        ));
    }
    let c = x.shape[x.shape.len() - 1];
    if weight.shape[0] != c {
        return Err(format!(
            "{op}: weight has {} channels, expected {c}",
            weight.shape[0]
        ));
    }
    if weight.shape[1] == 0 {
        return Err(format!("{op}: kernel size must be >= 1"));
    }
    if !x.dtype.is_float() || x.dtype != weight.dtype {
        return Err(format!(
            "{op}: dtypes must be floating point and match, got {:?} and {:?}",
            x.dtype, weight.dtype
        ));
    }
    if !x.device.same_device(&weight.device) {
        return Err(format!("{op}: input and weight must be on the same device"));
    }
    Ok(())
}

// Computes one convolution output dimension as
// floor((in + 2*pad - dilation*(kernel-1) - 1) / stride) + 1. Rejects kernels
// that exceed the padded input.
fn conv_out_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> std::result::Result<usize, String> {
    let effective = dilation * (kernel - 1) + 1;
    if input + 2 * padding < effective {
        return Err(format!(
            "conv: kernel of effective size {effective} exceeds the padded input size {}",
            input + 2 * padding
        ));
    }
    Ok((input + 2 * padding - effective) / stride + 1)
}
/// The base position for a position-indexed operation.
///
/// User graphs start at zero. Decode-rewritten graphs use the sequence cursor.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PositionOffset {
    /// Positions are zero-based within the sequence.
    Absolute,
    /// Positions are offset by the decode run's sequence cursor.
    Cursor,
}

/// Sliding-window configuration of an attention node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionWindow {
    /// Take the window from the enclosing model configuration.
    Inherit,
    /// Attend to the whole context.
    Full,
    /// Attend to at most the last `window` positions. The window must be
    /// positive.
    Local(usize),
}

impl AttentionWindow {
    /// Resolves `Inherit` against the model-level `inherited` window.
    /// `Full` and `Local` override it. Returns `Some(n)` for a local
    /// window of `n` and `None` for full attention.
    pub const fn resolve(self, inherited: Option<usize>) -> Option<usize> {
        match self {
            Self::Inherit => inherited,
            Self::Full => None,
            Self::Local(window) => Some(window),
        }
    }

    /// Returns the local window, or `None` for other variants.
    pub const fn local(self) -> Option<usize> {
        self.resolve(None)
    }
}

/// Pairing layout for the last dimension of a rotary embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RotaryLayout {
    /// GPT-NeoX pairs `(x[j], x[j + D/2])`.
    HalfSplit,
    /// GPT-J pairs `(x[2j], x[2j + 1])`.
    InterleavedPairs,
}

/// Visibility of newly staged rows in stateful KV attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvAttentionMode {
    /// Query row `p` sees committed rows and current rows through `p`.
    Causal,
    /// Every query sees committed rows and the complete current block.
    BidirectionalBlock,
}

/// One operation in the computation graph.
///
/// Variants represent leaves, placeholders, constants, generators,
/// elementwise operations, reductions, indexing, and neural network
/// operations. They also represent closed-form backward operations and their
/// `*Out` projectors, decode and prefill state, optimizer steps,
/// [`Checkpoint`](NodeKind::Checkpoint), and
/// [`StopGradient`](NodeKind::StopGradient).
///
/// [`Node::new`] validates every variant. [`NodeKind::metadata`] enforces
/// the operand shape, dtype, and device contracts documented below.
pub enum NodeKind {
    /// A user tensor held in a clearable [`LeafSlot`].
    Leaf(std::sync::Arc<LeafSlot>),
    // RFC 0008 placeholder for a compiled program. An Input declares one call
    // argument's signature. CompiledProgram::run binds its slot to an argument
    // buffer before evaluation.
    Input {
        slot: u32,
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    ScalarInput {
        slot: u32,
        dtype: DType,
        device: Device,
    },
    /// A constant tensor decoded from the little-endian representation of
    /// `dtype`.
    FromBytes {
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Zeros {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Ones {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    /// Tensor of `shape` filled with `value`.
    Full {
        shape: Vec<usize>,
        value: f64,
        dtype: DType,
        device: Device,
    },
    /// A standard-normal random tensor drawn from the run's RNG state.
    Randn {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    /// A uniform random tensor over `[lo, hi)`. Requires a float dtype and
    /// `lo < hi`.
    Uniform {
        lo: f64,
        hi: f64,
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    /// A 1-D tensor of `ceil((end - start) / step)` values starting at
    /// `start`. Negative lengths become zero.
    Arange {
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
        device: Device,
    },
    /// The `n × n` identity matrix.
    Eye {
        n: usize,
        dtype: DType,
        device: Device,
    },
    /// Elementwise binary arithmetic with NumPy-style broadcasting. A 0-d
    /// float operand never promotes the other operand's dtype. This follows
    /// PyTorch scalar promotion rules.
    Add {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise subtraction. See [`Add`](NodeKind::Add) for the
    /// broadcasting and dtype rules.
    Sub {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise multiplication. See [`Add`](NodeKind::Add).
    Mul {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise division. See [`Add`](NodeKind::Add).
    Div {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise comparison with broadcasting. The output dtype is u8.
    Eq {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise `>` with u8 output. See [`Eq`](NodeKind::Eq).
    Gt {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise `<` with u8 output. See [`Eq`](NodeKind::Eq).
    Lt {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise `>=` with u8 output. See [`Eq`](NodeKind::Eq).
    Ge {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise `<=` with u8 output. See [`Eq`](NodeKind::Eq).
    Le {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise maximum with broadcasting. See [`Add`](NodeKind::Add).
    Maximum {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise minimum with broadcasting. See [`Add`](NodeKind::Add).
    Minimum {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise unary op preserving shape and dtype.
    Neg {
        a: Arc<Node>,
    },
    /// Elementwise absolute value.
    Abs {
        a: Arc<Node>,
    },
    /// Elementwise square root.
    Sqrt {
        a: Arc<Node>,
    },
    /// Elementwise exponential.
    Exp {
        a: Arc<Node>,
    },
    /// Elementwise natural logarithm.
    Log {
        a: Arc<Node>,
    },
    /// Elementwise sine.
    Sin {
        a: Arc<Node>,
    },
    /// Elementwise cosine.
    Cos {
        a: Arc<Node>,
    },
    /// Elementwise hyperbolic tangent.
    Tanh {
        a: Arc<Node>,
    },
    /// Elementwise rectified linear unit, `max(x, 0)`.
    Relu {
        a: Arc<Node>,
    },
    /// Elementwise Gauss error function.
    Erf {
        a: Arc<Node>,
    },
    // Tensor.gelu as one Gaussian error linear unit node. `approximate`
    // selects the tanh form instead of the exact erf form. This pointwise
    // unary operation folds into fusion regions like tanh and erf.
    Gelu {
        a: Arc<Node>,
        approximate: bool,
    },
    /// Elementwise floor.
    Floor {
        a: Arc<Node>,
    },
    /// Elementwise ceiling.
    Ceil {
        a: Arc<Node>,
    },
    /// Elementwise rounding to the nearest integer.
    Round {
        a: Arc<Node>,
    },
    /// Elementwise sign with values -1, 0, or 1.
    Sign {
        a: Arc<Node>,
    },
    /// Selects from `a` or `b` for each element of the u8 `cond`. All
    /// three broadcast together, and `a` and `b` must share a dtype.
    Where {
        cond: Arc<Node>,
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Elementwise `a^exp` with a compile-time constant exponent.
    Pow {
        a: Arc<Node>,
        exp: f64,
    },
    /// Elementwise cast to `dtype`. The graph records the target dtype, so
    /// device constraints never cause a silent downcast.
    Cast {
        a: Arc<Node>,
        dtype: DType,
    },
    /// Sum over `dims`. With `keepdims`, reduced dimensions have size 1.
    /// Otherwise, the output omits them.
    Sum {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    /// Mean over `dims`. See [`Sum`](NodeKind::Sum).
    Mean {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    /// Maximum over `dims`. See [`Sum`](NodeKind::Sum).
    Max {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    /// Minimum over `dims`. See [`Sum`](NodeKind::Sum).
    Min {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    /// Product over `dims`. See [`Sum`](NodeKind::Sum).
    Prod {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    /// Indices of the maxima along `dim` with i64 output. The output shape
    /// omits `dim`.
    Argmax {
        a: Arc<Node>,
        dim: usize,
    },
    /// Indices of the minima along `dim` with i64 output.
    Argmin {
        a: Arc<Node>,
        dim: usize,
    },
    /// Inclusive cumulative sum along `dim`, preserving shape.
    Cumsum {
        a: Arc<Node>,
        dim: usize,
    },
    /// Selects rows along `dim` using 1-D i64 or u32 `indexes`. The output
    /// shape replaces `dim` with `indexes.len()`.
    IndexSelect {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
    },
    /// Adds `src` into `a` along `dim` at i64 or u32 `indexes` with the
    /// same shape as `src`. The output keeps `a`'s shape and dtype.
    ScatterAdd {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
        src: Arc<Node>,
    },
    /// Gathers along `dim` with i64 or u32 `indexes` of the same rank as
    /// `a`. The output takes the indexes' shape.
    Gather {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
    },
    /// Cross-entropy of `logits [.., C]` against integer `target [..]`.
    /// Skips targets equal to `ignore_index` and reduces the scalar output
    /// according to `reduction`.
    CrossEntropy {
        logits: Arc<Node>,
        target: Arc<Node>,
        ignore_index: i64,
        reduction: CeReduction,
    },
    /// Gradient of [`CrossEntropy`](NodeKind::CrossEntropy) with respect to
    /// the logits. The shape matches `logits`.
    CrossEntropyBackward {
        logits: Arc<Node>,
        target: Arc<Node>,
        ignore_index: i64,
        reduction: CeReduction,
    },
    // Scaled dot-product attention as one operation. The graph records its
    // semantics, while native code chooses how to execute it. Evaluators
    // compose candle operations as the reference implementation. A fused
    // flash kernel can replace them without changing the graph or its
    // adjoints. The shapes are q [.., T, D], k [.., S, D], and
    // v [.., S, Dv] with equal leading dimensions. The output is [.., T, Dv].
    Sdpa {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        scale: f64,
        causal: bool,
        window: AttentionWindow,
    },
    // The closed-form backward pass recomputes P = softmax(scores) from q and
    // k, then produces dq, dk, and dv in one evaluation. Consumers read them
    // through SdpaBackwardOut. On Metal, the forward flash kernel stores the
    // per-row logsumexp in the evaluator. This node uses it for a chunked
    // backward pass with bounded memory. Other backends recompute P with
    // composed candle operations. `fwd` is the Sdpa node whose adjoint this
    // node computes. It reads that node's output and stored L. This node does
    // not support second-order differentiation.
    SdpaBackward {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        g: Arc<Node>,
        fwd: Arc<Node>,
        scale: f64,
        causal: bool,
        window: AttentionWindow,
    },
    /// Selects one output of [`SdpaBackward`](NodeKind::SdpaBackward).
    /// `index` 0 selects dq, 1 selects dk, and 2 selects dv.
    SdpaBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // Absolute position embedding as one operation. It selects rows
    // 0..seq_len from the [max_positions, E] weight table. This representation
    // lets the RFC 0010 decode rewrite offset positions by the runtime cursor
    // without inferring a position embedding from composed operations.
    PositionEmbedding {
        weight: Arc<Node>,
        seq_len: usize,
    },
    // RFC 0010 paged KV attention for decode and prefill. The decode rewrite
    // creates this node, and user code does not. `compile_decode` replaces
    // each causal Sdpa with one. It scatters the new tokens' k and v into the
    // sequence's pool blocks at the cursor, then attends q over cached
    // positions according to `mode`. A local `window` retains that many
    // committed rows. Block-bidirectional mode also retains every current
    // row. None retains the whole context. q and k have shape [1, H, T, D],
    // and v has shape [1, H, T, Dv]. T is the shared new-token length.
    // The run's KV context supplies the pool, block table, and cursor. The
    // graph remains a pure function of its inputs. This node is not
    // differentiable.
    KvAttention {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        scale: f64,
        layer: u32,
        window: Option<usize>,
        mode: KvAttentionMode,
    },
    // RFC 0018 Kimi Delta Attention implements gated delta-rule linear
    // attention as one operation. q, k, and log_decay are [.., H, T, Dk].
    // v is [.., H, T, Dv], and beta is [.., H, T, 1]. Their leading
    // dimensions match. log_decay contains raw per-channel log decay rates
    // before cumsum, with values <= 0. The gate activation occurs upstream.
    // beta is already sigmoid-transformed into [0, 1]. The recurrence is
    // S_t = (I - beta_t k_t k_t^T) Diag(exp(log_decay_t)) S_{t-1}
    //     + beta_t k_t v_t^T,   o_t = scale * S_t^T q_t
    // starting from a zero state. The output is [.., H, T, Dv]. Evaluators use
    // the chunked parallel form with chunk size 64, a WY representation, and a
    // UT transform as the reference implementation. Fused kernels can replace
    // it without changing the graph. The decode rewrite replaces this node
    // with a stateful KdaRecurrence. It is not yet differentiable. Phase 4
    // adds the closed-form backward pass.
    KdaChunk {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        log_decay: Arc<Node>,
        beta: Arc<Node>,
        scale: f64,
    },
    // RFC 0018 stateful KDA recurrence for decode and prefill. The decode
    // rewrite creates this node, and user code does not. `compile_decode`
    // replaces each KdaChunk with one. It has the same operand contract as
    // KdaChunk. The initial state comes from the sequence's slot in the run's
    // decode context, and the node writes the final state back there. The graph
    // remains a pure function of its inputs. This node is not differentiable.
    KdaRecurrence {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        log_decay: Arc<Node>,
        beta: Arc<Node>,
        scale: f64,
        layer: u32,
    },
    // The RFC 0018 phase 4 closed-form KDA backward pass produces dq, dk, dv,
    // dlog_decay, and dbeta in one evaluation. It uses the saved operands and
    // output cotangent g, which has the same shape as the forward output.
    // Consumers read the results through KdaBackwardOut. Evaluators run the
    // adjoint delta-rule recurrence in reverse and recompute each chunk. They
    // retain chunk start states and recompute per-token states within each
    // chunk. This bounds memory without retaining O(T) states. This node does
    // not support second-order differentiation.
    KdaBackward {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        log_decay: Arc<Node>,
        beta: Arc<Node>,
        g: Arc<Node>,
        scale: f64,
    },
    KdaBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // Causal depthwise short convolution over [.., T, C] with weight [C, K].
    // It computes y[t, c] = sum_j w[c, j] * x[t-K+1+j, c] with zero history,
    // which is left zero-padding of K-1. This representation lets the decode
    // rewrite carry the K-1-token window as sequence state without inferring
    // a short convolution from composed operations.
    ShortConv1d {
        x: Arc<Node>,
        weight: Arc<Node>,
    },
    // RFC 0018 stateful short convolution for decode and prefill. The decode
    // rewrite creates this node, and user code does not. It has the same
    // contract as ShortConv1d. The sequence's slot stores the previous K-1
    // inputs and receives the new window. This node is not differentiable.
    ConvState {
        x: Arc<Node>,
        weight: Arc<Node>,
        layer: u32,
    },
    // Inference-only decode output selector created by the decode rewrite.
    // User code does not create it. With the last-token-row flag,
    // `compile_decode` wraps each [1, T, V] root in one. It copies the row at
    // the run's token advance minus one from the sequence dimension and
    // produces [V]. At execution, it reads the row index from the decode
    // context, so the graph remains a pure function of its inputs. This node
    // is not differentiable and vmap rejects it.
    LastTokenRow {
        a: Arc<Node>,
    },
    // RFC 0016 phase 2, as built, combines the chunked head and cross-entropy
    // in one operation. Its forward result is the mean cross-entropy of
    // Linear(x, weight, bias) against target. Evaluation processes one row
    // chunk at a time and never materializes the full [rows, vocab] logits.
    // The adjoint can use the same chunk loop with one temporary grad-logits
    // workspace. The graph-rewrite version used one CE node per chunk and kept
    // every backward workspace alive until the head-parameter roots evaluated.
    // Consumer-count release cannot shorten those lifetimes.
    ChunkedHeadCe {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
        target: Arc<Node>,
        ignore_index: i64,
    },
    // The closed-form adjoint loops over the chunks in one evaluation. It
    // recomputes each chunk's logits and grad-logits in a temporary workspace,
    // then accumulates dx, dw, and db. Consumers read them through
    // ChunkedHeadCeBackwardOut. g is the loss's scalar cotangent. This node
    // does not support second-order differentiation.
    ChunkedHeadCeBackward {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
        target: Arc<Node>,
        g: Arc<Node>,
        ignore_index: i64,
    },
    /// Selects one output of
    /// [`ChunkedHeadCeBackward`](NodeKind::ChunkedHeadCeBackward). `index`
    /// 0 selects dx, 1 selects dweight, and 2 selects dbias.
    ChunkedHeadCeBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // RFC 0018 phase 4 ShortConv1d adjoints. dx is the full correlation of the
    // cotangent with the unflipped weight. dw is the per-tap correlation of
    // the cotangent and input over the zero-padded causal window. x supplies
    // shape and dtype metadata. These nodes do not support second-order
    // differentiation.
    ShortConv1dBackwardX {
        x: Arc<Node>,
        weight: Arc<Node>,
        g: Arc<Node>,
    },
    ShortConv1dBackwardW {
        x: Arc<Node>,
        weight: Arc<Node>,
        g: Arc<Node>,
    },
    // RoPE with GPT-NeoX half-split rotation. x has shape [.., T, D] with even
    // D. The last dimension rotates in half pairs by
    // (offset + position) * theta^(-2j/D). The tensor stores absolute
    // positions, while attention sees only offsets. This keeps cached K and V
    // valid as the context grows, unlike learned absolute embeddings.
    // User graphs set `offset` to Absolute. The decode rewrite changes it to
    // Cursor, which uses the run's KV cursor, without inferring positions from
    // composed operations.
    RotaryEmbedding {
        x: Arc<Node>,
        seq_len: usize,
        theta: f64,
        offset: PositionOffset,
        layout: RotaryLayout,
    },
    // Backward pass for RotaryEmbedding with absolute positions only. It
    // applies the transpose rotation through the same fused kernel with
    // negated angles. The input shape and seq_len provide metadata.
    RotaryEmbeddingBackward {
        g: Arc<Node>,
        shape: Vec<usize>,
        seq_len: usize,
        theta: f64,
        layout: RotaryLayout,
    },
    // Layer normalization over the last dimension:
    // y = (x - mean) / sqrt(variance + eps) * weight + bias.
    // The fused Metal kernel handles this operation in one launch, and decode
    // compilation can pass it through.
    LayerNorm {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
        eps: f64,
    },
    // RMS normalization over the last dimension. It computes
    // y = x / sqrt(mean(x^2) + eps), with an optional weight multiplier.
    RmsNorm {
        x: Arc<Node>,
        weight: Option<Arc<Node>>,
        eps: f64,
    },
    // The LayerNorm backward pass evaluates dx as its own value and stores dw
    // and db for LayerNormBackwardOut, as optimizer steps do.
    LayerNormBackward {
        x: Arc<Node>,
        weight: Arc<Node>,
        g: Arc<Node>,
        eps: f64,
    },
    // Reads one weight-side output of LayerNormBackward. Index 1 selects dw,
    // and index 2 selects db.
    LayerNormBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // Fused linear layer computing y = x * W + b in one GEMM launch. Metal
    // uses an addmm epilogue. Model.linear and attention projections build
    // this node directly.
    Linear {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
    },
    /// A linear layer with packed K-quant weights. It multiplies f32
    /// `x [.., columns]` by the dequantized
    /// `weight_shape = [rows, columns]` matrix and adds an optional f32 bias
    /// `[rows]`. The f32 output has shape `[.., rows]`. `weight` is the
    /// packed u8 tensor `[rows, encoded_row_bytes]`.
    QuantizedLinear {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Option<Arc<Node>>,
        codec: GgmlKQuant,
        weight_shape: [usize; 2],
    },
    /// An embedding lookup over a packed K-quant table. The i64 or u32
    /// `indexes` may have any shape and select rows of the dequantized
    /// `weight_shape = [rows, columns]` table. The f32 output has shape
    /// `[..indexes, columns]`. When set, `padding_index` zeroes that row's
    /// output and gradient.
    QuantizedEmbedding {
        indexes: Arc<Node>,
        weight: Arc<Node>,
        codec: GgmlKQuant,
        weight_shape: [usize; 2],
        padding_index: Option<usize>,
    },
    /// A 1-D convolution with x `[N, C_in, L]` and
    /// w `[C_out, C_in/groups, K]`. The output is `[N, C_out, L_out]`.
    Conv1d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    /// A 2-D convolution with x `[N, C_in, H, W]` and
    /// w `[C_out, C_in/groups, KH, KW]`.
    Conv2d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    /// A transposed 1-D convolution with output length
    /// `(L-1)*stride + dilation*(K-1) + output_padding + 1 - 2*padding`,
    /// and `w[1] * groups` channels.
    ConvTranspose1d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    /// A transposed 2-D convolution. See
    /// [`ConvTranspose1d`](NodeKind::ConvTranspose1d) for the size formula,
    /// applied per spatial dimension.
    ConvTranspose2d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    /// Weight gradient of [`Conv1d`](NodeKind::Conv1d):
    /// `[out_channels, C_in/groups, kernel]`.
    Conv1dBackwardW {
        x: Arc<Node>,
        g: Arc<Node>,
        kernel: usize,
        out_channels: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    /// Weight gradient of [`Conv2d`](NodeKind::Conv2d):
    /// `[out_channels, C_in/groups, kernel[0], kernel[1]]`.
    Conv2dBackwardW {
        x: Arc<Node>,
        g: Arc<Node>,
        kernel: [usize; 2],
        out_channels: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    /// Reinterprets `a` with `shape`. The element count must stay the same.
    Reshape {
        a: Arc<Node>,
        shape: Vec<usize>,
    },
    /// Reorders dimensions. `dims` must be a permutation of the rank.
    Permute {
        a: Arc<Node>,
        dims: Vec<usize>,
    },
    /// A strided view for each dimension. Each range is
    /// `(start, stop, stride)`, and the output dimension is
    /// `ceil((stop - start) / stride)`.
    Slice {
        a: Arc<Node>,
        ranges: Vec<(usize, usize, usize)>,
    },
    /// Concatenates `a` and `b` along `dim`. All other dimensions must
    /// match.
    Concat {
        a: Arc<Node>,
        b: Arc<Node>,
        dim: usize,
    },
    /// Broadcasts `a` to `shape`, which must have equal or higher rank.
    /// Source dimensions other than 1 must match the target.
    BroadcastTo {
        a: Arc<Node>,
        shape: Vec<usize>,
    },
    /// A batched matrix product over the last two dimensions with broadcast
    /// leading dimensions: `[.., M, K] @ [.., K, N] -> [.., M, N]`.
    Matmul {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    /// Matrix inverse over the last two square float dimensions.
    Inverse {
        a: Arc<Node>,
    },
    /// Determinant over the last two square float dimensions. The output drops
    /// both dimensions.
    Det {
        a: Arc<Node>,
    },
    /// Solves `a * x = b` for `x`. The last two dimensions of `a` form a
    /// square float matrix. `b [.., N, NRHS]` matches `a`'s leading
    /// dimensions and size.
    Solve {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    // lr, c1 = 1 - beta1^t, and c2 = 1 - beta2^t are 0-d tensor children.
    // Their changing values flow through the graph, so an RFC 0008 frozen
    // graph never replays a stale step count or learning rate.
    AdamWStep {
        param: Arc<Node>,
        grad: Arc<Node>,
        m: Arc<Node>,
        v: Arc<Node>,
        lr: Arc<Node>,
        c1: Arc<Node>,
        c2: Arc<Node>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    /// Selects one output of [`AdamWStep`](NodeKind::AdamWStep). `index` 0
    /// selects the updated parameter, 1 the first moment, and 2 the second
    /// moment.
    AdamWOut {
        step: Arc<Node>,
        index: u8,
    },
    // `first` is a 0-d flag that is 1.0 on the first step and 0.0 afterward.
    // It selects v = g instead of v = momentum * v + (1 - dampening) * g.
    // Velocity always has a real buffer initialized to zero, so it needs no
    // placeholder.
    SgdStep {
        param: Arc<Node>,
        grad: Arc<Node>,
        velocity: Arc<Node>,
        first: Arc<Node>,
        lr: Arc<Node>,
        momentum: f64,
        dampening: f64,
        nesterov: bool,
        weight_decay: f64,
    },
    /// Selects one output of [`SgdStep`](NodeKind::SgdStep). `index` 0
    /// selects the updated parameter, and 1 selects the updated velocity.
    SgdOut {
        step: Arc<Node>,
        index: u8,
    },
    /// Identity in the forward pass. Autodiff stops gradient propagation at
    /// this node.
    StopGradient {
        a: Arc<Node>,
    },
    /// Identity in the forward pass. During the backward pass, autodiff
    /// recomputes the wrapped subgraph instead of retaining its activations.
    Checkpoint {
        a: Arc<Node>,
    },
    /// Identity with a stable name. Graph walks can discover the wrapped value
    /// as an inference exposure. Autodiff treats it as identity, and lowering
    /// aliases it away at no cost.
    Expose {
        a: Arc<Node>,
        name: String,
    },
}

/// One computation graph node with an identity, cached metadata, and an
/// operation.
///
/// [`Node::new`] returns immutable nodes behind `Arc`. It issues monotonic
/// process-unique ids and caches metadata derived from `kind`. Public fields
/// allow low-level direct construction. Callers that construct a node directly
/// must provide a unique, stable id and metadata that exactly matches
/// [`NodeKind::metadata`]. The `id`, not the pointer, determines identity.
pub struct Node {
    pub id: u64,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
    pub kind: NodeKind,
}

impl Drop for Node {
    fn drop(&mut self) {
        // Default destruction follows Arc<Node> children recursively. Deep
        // operation chains and large fused backward graphs can overflow worker
        // thread stacks, so this destructor drains descendants through a
        // worklist. It clones children before dropping their parent's kind,
        // which prevents a Node destructor from running during that step. It
        // unwraps uniquely owned nodes, swaps each kind for a dummy leaf, and
        // adds their children to the worklist.
        let dummy = || NodeKind::Zeros {
            shape: Vec::new(),
            dtype: DType::F32,
            device: Device::Cpu(0),
        };
        let mut worklist: Vec<Arc<Node>> = Vec::new();
        let kind = std::mem::replace(&mut self.kind, dummy());
        worklist.extend(node_children(&kind));
        drop(kind);
        while let Some(arc) = worklist.pop() {
            let Ok(mut node) = Arc::try_unwrap(arc) else {
                continue;
            };
            let kind = std::mem::replace(&mut node.kind, dummy());
            // `node` drops with a dummy kind at the end of this iteration.
            // Its Drop finds no children and returns.
            worklist.extend(node_children(&kind));
            drop(kind);
        }
    }
}

impl Node {
    /// Validates `kind` and returns a shared node with cached metadata. See
    /// [`NodeKind::metadata`] for operand shape, dtype, and device checks.
    /// Unlike direct struct construction, this function guarantees a fresh id
    /// and consistent metadata.
    pub fn new(kind: NodeKind) -> Result<Arc<Self>, String> {
        let metadata = kind.metadata()?;
        Ok(Arc::new(Self {
            id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
            shape: metadata.shape,
            dtype: metadata.dtype,
            device: metadata.device,
            kind,
        }))
    }
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> std::result::Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        if da != db && da != 1 && db != 1 {
            return Err(format!("shapes {a:?} and {b:?} are not broadcastable"));
        }
        out.push(da.max(db));
    }
    Ok(out)
}

// A 0-d float operand never promotes a float tensor's dtype. A scalar-tensor
// operation keeps the tensor's dtype and casts the scalar to it during
// evaluation, matching PyTorch's scalar promotion rules. Integer dtype
// mismatches keep the existing first-operand rule.
fn scalar_aware_binary_dtype(a: &Node, b: &Node) -> DType {
    if a.dtype != b.dtype
        && a.dtype.is_float()
        && b.dtype.is_float()
        && a.shape.is_empty() != b.shape.is_empty()
    {
        if a.shape.is_empty() {
            b.dtype
        } else {
            a.dtype
        }
    } else {
        a.dtype
    }
}

fn reduced_shape(shape: &[usize], dims: &[usize], keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &d)| d)
            .collect()
    }
}

fn linear_out_shape(
    x: &[usize],
    weight: &[usize],
    bias: &[usize],
) -> std::result::Result<Vec<usize>, String> {
    let rank = x.len();
    if rank < 2 || weight.len() != 2 || x[rank - 1] != weight[0] || bias != [weight[1]] {
        return Err(format!(
            "linear: expected x [.., K], weight [K, N], bias [N], got {:?} x {:?} + {:?}",
            x, weight, bias
        ));
    }
    let mut out = x.to_vec();
    out[rank - 1] = weight[1];
    Ok(out)
}
impl NodeKind {
    /// Computes output metadata and validates all operand contracts.
    /// `Node::new` calls this for every node, including nodes rebuilt by
    /// compiler rewrites. The final check, `check_dtype_device`, validates
    /// device and dtype support.
    pub fn metadata(&self) -> Result<NodeMetadata, String> {
        let (shape, dtype, device) = match self {
            NodeKind::Leaf(slot) => {
                let metadata = slot.metadata().map_err(|e| e.to_string())?;
                (metadata.shape, metadata.dtype, metadata.device)
            }
            NodeKind::Input {
                shape,
                dtype,
                device,
                ..
            } => (shape.clone(), *dtype, device.clone()),
            NodeKind::ScalarInput { dtype, device, .. } => (vec![], *dtype, device.clone()),
            NodeKind::FromBytes {
                shape,
                dtype,
                device,
                ..
            }
            | NodeKind::Zeros {
                shape,
                dtype,
                device,
            }
            | NodeKind::Ones {
                shape,
                dtype,
                device,
            }
            | NodeKind::Randn {
                shape,
                dtype,
                device,
            } => (shape.clone(), *dtype, device.clone()),
            NodeKind::Uniform {
                lo,
                hi,
                shape,
                dtype,
                device,
            } => {
                if !dtype.is_float() {
                    return Err(format!(
                        "uniform: dtype must be floating point, got {dtype:?}"
                    ));
                }
                if hi <= lo {
                    return Err(format!("uniform: expected lo < hi, got lo={lo} hi={hi}"));
                }
                (shape.clone(), *dtype, device.clone())
            }
            NodeKind::Full {
                shape,
                dtype,
                device,
                ..
            } => (shape.clone(), *dtype, device.clone()),
            NodeKind::Arange {
                start,
                end,
                step,
                dtype,
                device,
            } => {
                let n = ((end - start) / step).ceil().max(0.0) as usize;
                (vec![n], *dtype, device.clone())
            }
            NodeKind::Eye { n, dtype, device } => (vec![*n, *n], *dtype, device.clone()),
            NodeKind::Add { a, b }
            | NodeKind::Sub { a, b }
            | NodeKind::Mul { a, b }
            | NodeKind::Div { a, b }
            | NodeKind::Maximum { a, b }
            | NodeKind::Minimum { a, b } => (
                broadcast_shapes(&a.shape, &b.shape)?,
                scalar_aware_binary_dtype(a, b),
                a.device.clone(),
            ),
            NodeKind::Eq { a, b }
            | NodeKind::Gt { a, b }
            | NodeKind::Lt { a, b }
            | NodeKind::Ge { a, b }
            | NodeKind::Le { a, b } => (
                broadcast_shapes(&a.shape, &b.shape)?,
                DType::U8,
                a.device.clone(),
            ),
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
            | NodeKind::Gelu { a, .. }
            | NodeKind::Floor { a }
            | NodeKind::Ceil { a }
            | NodeKind::Round { a }
            | NodeKind::Sign { a }
            | NodeKind::Checkpoint { a }
            | NodeKind::StopGradient { a }
            | NodeKind::Expose { a, .. } => (a.shape.clone(), a.dtype, a.device.clone()),
            NodeKind::Pow { a, .. } => (a.shape.clone(), a.dtype, a.device.clone()),
            NodeKind::Where { cond, a, b } => {
                if cond.dtype != DType::U8 {
                    return Err(format!("where: condition must be u8, got {:?}", cond.dtype));
                }
                if a.dtype != b.dtype {
                    return Err(format!(
                        "where: dtype mismatch, got {:?} and {:?}",
                        a.dtype, b.dtype
                    ));
                }
                let shape = broadcast_shapes(&cond.shape, &a.shape)?;
                let shape = broadcast_shapes(&shape, &b.shape)?;
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::Argmax { a, dim } | NodeKind::Argmin { a, dim } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "argmax/argmin: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                let mut shape = a.shape.clone();
                shape.remove(*dim);
                (shape, DType::I64, a.device.clone())
            }
            NodeKind::Cumsum { a, dim } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "cumsum: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                (a.shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::IndexSelect { a, dim, indexes } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "index_select: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                if !matches!(indexes.dtype, DType::I64 | DType::U32) {
                    return Err(format!(
                        "index_select: indexes must be i64 or u32, got {:?}",
                        indexes.dtype
                    ));
                }
                if indexes.shape.len() != 1 {
                    return Err(format!(
                        "index_select: indexes must be 1-D, got shape {:?}",
                        indexes.shape
                    ));
                }
                let mut shape = a.shape.clone();
                shape[*dim] = indexes.shape[0];
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::ScatterAdd {
                a,
                dim,
                indexes,
                src,
            } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "scatter_add: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                if !matches!(indexes.dtype, DType::I64 | DType::U32) {
                    return Err(format!(
                        "scatter_add: indexes must be i64 or u32, got {:?}",
                        indexes.dtype
                    ));
                }
                if indexes.shape != src.shape {
                    return Err(format!(
                        "scatter_add: indexes shape {:?} must match src shape {:?}",
                        indexes.shape, src.shape
                    ));
                }
                if src.dtype != a.dtype {
                    return Err(format!(
                        "scatter_add: src dtype {:?} does not match target dtype {:?}",
                        src.dtype, a.dtype
                    ));
                }
                (a.shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::Gather { a, dim, indexes } => {
                if a.shape.is_empty() || *dim >= a.shape.len() {
                    return Err(format!(
                        "gather: dim {dim} out of range for rank {}",
                        a.shape.len()
                    ));
                }
                if !matches!(indexes.dtype, DType::I64 | DType::U32) {
                    return Err(format!(
                        "gather: indexes must be i64 or u32, got {:?}",
                        indexes.dtype
                    ));
                }
                if indexes.shape.len() != a.shape.len() {
                    return Err(format!(
                        "gather: indexes rank {} must match input rank {}",
                        indexes.shape.len(),
                        a.shape.len()
                    ));
                }
                for i in 0..a.shape.len() {
                    if i != *dim && indexes.shape[i] > a.shape[i] {
                        return Err(format!(
                            "gather: indexes shape {:?} exceeds input shape {:?} at dim {i}",
                            indexes.shape, a.shape
                        ));
                    }
                }
                (indexes.shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::CrossEntropy {
                logits,
                target,
                ignore_index: _,
                reduction: _,
            } => {
                let rank = logits.shape.len();
                if rank < 1 {
                    return Err("cross_entropy: logits must have rank >= 1".to_string());
                }
                if logits.shape[rank - 1] == 0 {
                    return Err("cross_entropy: class dimension must be non-empty".to_string());
                }
                if !matches!(logits.dtype, DType::F32 | DType::F64 | DType::BF16) {
                    return Err(format!(
                        "cross_entropy: logits must be f32, f64 or bf16, got {:?}",
                        logits.dtype
                    ));
                }
                if !matches!(target.dtype, DType::I64 | DType::U32) {
                    return Err(format!(
                        "cross_entropy: targets must be i64 or u32, got {:?}",
                        target.dtype
                    ));
                }
                if target.shape != logits.shape[..rank - 1] {
                    return Err(format!(
                        "cross_entropy: targets shape {:?} does not match logits leading shape {:?}",
                        target.shape,
                        &logits.shape[..rank - 1]
                    ));
                }
                if !target.device.same_device(&logits.device) {
                    return Err(
                        "cross_entropy: logits and targets must be on the same device".to_string(),
                    );
                }
                (Vec::new(), logits.dtype, logits.device.clone())
            }
            NodeKind::CrossEntropyBackward { logits, .. } => {
                (logits.shape.clone(), logits.dtype, logits.device.clone())
            }
            NodeKind::Sdpa {
                q,
                k,
                v,
                causal,
                window,
                ..
            } => {
                if matches!(window, AttentionWindow::Local(0)) {
                    return Err("sdpa: window must be positive".to_string());
                }
                if !causal && !matches!(window, AttentionWindow::Inherit) {
                    return Err("sdpa: explicit window requires causal attention".to_string());
                }
                let out = sdpa_check("sdpa", q, k, v)?;
                (out, q.dtype, q.device.clone())
            }
            NodeKind::SdpaBackward {
                q,
                k,
                v,
                g,
                fwd,
                causal,
                window,
                ..
            } => {
                if matches!(window, AttentionWindow::Local(0)) {
                    return Err("sdpa backward: window must be positive".to_string());
                }
                if !causal && !matches!(window, AttentionWindow::Inherit) {
                    return Err(
                        "sdpa backward: explicit window requires causal attention".to_string()
                    );
                }
                let out = sdpa_check("sdpa", q, k, v)?;
                let rank = q.shape.len();
                if rank >= 3 && q.shape[rank - 3] != k.shape[rank - 3] {
                    return Err(
                        "sdpa backward: grouped-query attention is not differentiable".to_string(),
                    );
                }
                if !matches!(&fwd.kind, NodeKind::Sdpa { .. }) {
                    return Err("sdpa backward: fwd must be an sdpa node".to_string());
                }
                if g.shape != out {
                    return Err(format!(
                        "sdpa backward: grad shape {:?} does not match the attention output shape {out:?}",
                        g.shape
                    ));
                }
                if g.dtype != q.dtype || !g.device.same_device(&q.device) {
                    return Err(
                        "sdpa backward: grad must share dtype and device with q".to_string()
                    );
                }
                (q.shape.clone(), q.dtype, q.device.clone())
            }
            NodeKind::SdpaBackwardOut { of, index } => {
                let NodeKind::SdpaBackward { q, k, v, .. } = &of.kind else {
                    return Err(
                        "sdpa backward out: source must be an sdpa backward node".to_string()
                    );
                };
                let source = match index {
                    0 => q,
                    1 => k,
                    2 => v,
                    i => return Err(format!("sdpa backward out: index must be 0..=2, got {i}")),
                };
                (source.shape.clone(), source.dtype, source.device.clone())
            }
            NodeKind::PositionEmbedding { weight, seq_len } => {
                if weight.shape.len() != 2 {
                    return Err(format!(
                        "position_embedding: weight must be [maxPositions, E], got {:?}",
                        weight.shape
                    ));
                }
                if *seq_len == 0 || *seq_len > weight.shape[0] {
                    return Err(format!(
                        "position_embedding: seq_len {seq_len} out of range for {} positions",
                        weight.shape[0]
                    ));
                }
                if !weight.dtype.is_float() {
                    return Err(format!(
                        "position_embedding: weight must be a float dtype, got {:?}",
                        weight.dtype
                    ));
                }
                (
                    vec![*seq_len, weight.shape[1]],
                    weight.dtype,
                    weight.device.clone(),
                )
            }
            NodeKind::KvAttention { q, k, v, .. } => {
                let out = sdpa_check("kv_attention", q, k, v)?;
                let rank = q.shape.len();
                if q.shape[rank - 2] != k.shape[rank - 2] {
                    return Err(format!(
                        "kv_attention: q, k and v must share the new-token length, got {:?}, {:?} and {:?}",
                        q.shape, k.shape, v.shape
                    ));
                }
                if rank < 3 {
                    // Leading dimensions form the batch, with one per KV slot.
                    // RFC 0013 checks slot counts at run time.
                    return Err(format!(
                        "kv_attention: expected shape [B..., H, T, D], got {:?}",
                        q.shape
                    ));
                }
                (out, q.dtype, q.device.clone())
            }
            NodeKind::KdaChunk {
                q,
                k,
                v,
                log_decay,
                beta,
                ..
            } => {
                let out = kda_check("kda_chunk", q, k, v, log_decay, beta)?;
                (out, q.dtype, q.device.clone())
            }
            NodeKind::KdaRecurrence {
                q,
                k,
                v,
                log_decay,
                beta,
                ..
            } => {
                let out = kda_check("kda_recurrence", q, k, v, log_decay, beta)?;
                (out, q.dtype, q.device.clone())
            }
            NodeKind::ShortConv1d { x, weight } => {
                short_conv_check("short_conv1d", x, weight)?;
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::ConvState { x, weight, .. } => {
                short_conv_check("conv_state", x, weight)?;
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::LastTokenRow { a } => {
                if a.shape.len() != 3 || a.shape[0] != 1 {
                    return Err(format!(
                        "last_token_row: expected input [1, T, V], got {:?}",
                        a.shape
                    ));
                }
                (vec![a.shape[2]], a.dtype, a.device.clone())
            }
            NodeKind::ChunkedHeadCe {
                x,
                weight,
                bias,
                target,
                ..
            } => {
                head_ce_check("chunked_head_ce", x, weight, bias, target)?;
                (Vec::new(), x.dtype, x.device.clone())
            }
            NodeKind::ChunkedHeadCeBackward {
                x,
                weight,
                bias,
                target,
                g,
                ..
            } => {
                head_ce_check("chunked_head_ce_backward", x, weight, bias, target)?;
                if !g.shape.is_empty() && g.shape.iter().product::<usize>() != 1 {
                    return Err(format!(
                        "chunked head ce backward: grad must be a scalar, got {:?}",
                        g.shape
                    ));
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::ChunkedHeadCeBackwardOut { of, index } => {
                let NodeKind::ChunkedHeadCeBackward {
                    x, weight, bias, ..
                } = &of.kind
                else {
                    return Err(
                        "chunked head ce backward out: source must be a chunked head ce backward node"
                            .to_string(),
                    );
                };
                let source = match index {
                    0 => x,
                    1 => weight,
                    2 => bias,
                    i => {
                        return Err(format!(
                            "chunked head ce backward out: index must be 0..=2, got {i}"
                        ))
                    }
                };
                (source.shape.clone(), source.dtype, source.device.clone())
            }
            NodeKind::KdaBackward {
                q,
                k,
                v,
                log_decay,
                beta,
                g,
                ..
            } => {
                let out = kda_check("kda_backward", q, k, v, log_decay, beta)?;
                if g.shape != out {
                    return Err(format!(
                        "kda backward: grad shape {:?} does not match the kda output shape {out:?}",
                        g.shape
                    ));
                }
                if g.dtype != q.dtype || !g.device.same_device(&q.device) {
                    return Err("kda backward: grad must share dtype and device with q".to_string());
                }
                (q.shape.clone(), q.dtype, q.device.clone())
            }
            NodeKind::KdaBackwardOut { of, index } => {
                let NodeKind::KdaBackward {
                    q,
                    k,
                    v,
                    log_decay,
                    beta,
                    ..
                } = &of.kind
                else {
                    return Err("kda backward out: source must be a kda backward node".to_string());
                };
                let source = match index {
                    0 => q,
                    1 => k,
                    2 => v,
                    3 => log_decay,
                    4 => beta,
                    i => return Err(format!("kda backward out: index must be 0..=4, got {i}")),
                };
                (source.shape.clone(), source.dtype, source.device.clone())
            }
            NodeKind::ShortConv1dBackwardX { x, weight, g } => {
                short_conv_check("short_conv1d_backward", x, weight)?;
                if g.shape != x.shape {
                    return Err(format!(
                        "short_conv1d backward: grad shape {:?} does not match the input shape {:?}",
                        g.shape, x.shape
                    ));
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::ShortConv1dBackwardW { x, weight, g } => {
                short_conv_check("short_conv1d_backward", x, weight)?;
                if g.shape != x.shape {
                    return Err(format!(
                        "short_conv1d backward: grad shape {:?} does not match the input shape {:?}",
                        g.shape, x.shape
                    ));
                }
                (weight.shape.clone(), weight.dtype, weight.device.clone())
            }
            NodeKind::RotaryEmbedding { x, seq_len, .. } => {
                let rank = x.shape.len();
                if rank < 2 {
                    return Err(format!(
                        "rotary_embedding: expected [.., T, D], got {:?}",
                        x.shape
                    ));
                }
                let (t, d) = (x.shape[rank - 2], x.shape[rank - 1]);
                if *seq_len != t {
                    return Err(format!(
                        "rotary_embedding: seq_len {seq_len} does not match the input's T {t}"
                    ));
                }
                if d % 2 != 0 {
                    return Err(format!("rotary_embedding: head dim must be even, got {d}"));
                }
                if !matches!(x.dtype, DType::F32 | DType::BF16) {
                    return Err(format!(
                        "rotary_embedding: dtype must be f32 or bf16, got {:?}",
                        x.dtype
                    ));
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::RotaryEmbeddingBackward {
                g, shape, seq_len, ..
            } => {
                let rank = shape.len();
                if rank < 2 || shape[rank - 2] != *seq_len || g.shape != *shape {
                    return Err(format!(
                        "rotary_embedding_backward: expected grad of shape {shape:?}, got {:?}",
                        g.shape
                    ));
                }
                (shape.clone(), g.dtype, g.device.clone())
            }
            NodeKind::Linear { x, weight, bias } => {
                let out = linear_out_shape(&x.shape, &weight.shape, &bias.shape)?;
                (out, x.dtype, x.device.clone())
            }
            NodeKind::QuantizedLinear {
                x,
                weight,
                bias,
                codec,
                weight_shape: [rows, columns],
            } => {
                if x.shape.len() < 2 || x.shape.last() != Some(columns) {
                    return Err(format!(
                        "quantized_linear: expected input [.., {columns}], got {:?}",
                        x.shape
                    ));
                }
                let encoded_row_bytes = codec.encoded_row_bytes(*columns).ok_or_else(|| {
                    format!(
                        "quantized_linear: logical row width {columns} is invalid for {}",
                        codec.name()
                    )
                })?;
                if weight.shape != [*rows, encoded_row_bytes] || weight.dtype != DType::U8 {
                    return Err(format!(
                        "quantized_linear: expected packed {} weight [{rows}, {encoded_row_bytes}] u8, got {:?} {:?}",
                        codec.name(),
                        weight.shape,
                        weight.dtype
                    ));
                }
                if x.dtype != DType::F32 {
                    return Err(format!(
                        "quantized_linear: input must be f32, got {:?}",
                        x.dtype
                    ));
                }
                if !x.device.same_device(&weight.device) {
                    return Err(
                        "quantized_linear: input and weight must be on the same device".to_string(),
                    );
                }
                if let Some(bias) = bias {
                    if bias.shape != [*rows]
                        || bias.dtype != DType::F32
                        || !bias.device.same_device(&x.device)
                    {
                        return Err(format!(
                            "quantized_linear: bias must be [{rows}] f32 on the input device"
                        ));
                    }
                }
                let mut out = x.shape.clone();
                *out.last_mut().expect("validated input rank") = *rows;
                (out, DType::F32, x.device.clone())
            }
            NodeKind::QuantizedEmbedding {
                indexes,
                weight,
                codec,
                weight_shape: [rows, columns],
                padding_index,
            } => {
                let encoded_row_bytes = codec.encoded_row_bytes(*columns).ok_or_else(|| {
                    format!(
                        "quantized_embedding: logical row width {columns} is invalid for {}",
                        codec.name()
                    )
                })?;
                if weight.shape != [*rows, encoded_row_bytes] || weight.dtype != DType::U8 {
                    return Err(format!(
                        "quantized_embedding: expected packed {} weight [{rows}, {encoded_row_bytes}] u8, got {:?} {:?}",
                        codec.name(),
                        weight.shape,
                        weight.dtype
                    ));
                }
                if !matches!(indexes.dtype, DType::I64 | DType::U32) {
                    return Err(format!(
                        "quantized_embedding: indexes must be i64 or u32, got {:?}",
                        indexes.dtype
                    ));
                }
                if !indexes.device.same_device(&weight.device) {
                    return Err(
                        "quantized_embedding: indexes and weight must be on the same device"
                            .to_string(),
                    );
                }
                if padding_index.is_some_and(|index| index >= *rows) {
                    return Err(format!(
                        "quantized_embedding: padding index is outside 0..{rows}"
                    ));
                }
                let mut out = indexes.shape.clone();
                out.push(*columns);
                (out, DType::F32, indexes.device.clone())
            }
            NodeKind::LayerNorm {
                x, weight, bias, ..
            } => {
                let rank = x.shape.len();
                let k = weight.shape.len();
                if rank < k || x.shape[rank - k..] != weight.shape[..] || bias.shape != weight.shape
                {
                    return Err(format!(
                        "layer_norm: weight and bias must match the input's trailing dims {:?}, got {:?} and {:?}",
                        x.shape, weight.shape, bias.shape
                    ));
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::RmsNorm { x, weight, .. } => {
                if x.shape.is_empty() {
                    return Err("rms_norm: input must have rank at least 1".to_string());
                }
                if let Some(weight) = weight {
                    if weight.shape != [*x.shape.last().expect("validated input rank")]
                        || weight.dtype != x.dtype
                        || !weight.device.same_device(&x.device)
                    {
                        return Err(format!(
                            "rms_norm: weight must be [{}] {:?} on the input device",
                            x.shape.last().expect("validated input rank"),
                            x.dtype
                        ));
                    }
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::LayerNormBackward { x, weight, g, .. } => {
                let rank = x.shape.len();
                let k = weight.shape.len();
                if rank < k || x.shape[rank - k..] != weight.shape[..] || g.shape != x.shape {
                    return Err(format!(
                        "layer_norm_backward: expected grad of shape {:?}, got {:?}",
                        x.shape, g.shape
                    ));
                }
                (x.shape.clone(), x.dtype, x.device.clone())
            }
            NodeKind::LayerNormBackwardOut { of, index } => {
                let NodeKind::LayerNormBackward { weight, .. } = &of.kind else {
                    return Err(
                        "layer_norm_backward_out: parent is not a backward node".to_string()
                    );
                };
                if *index == 0 || *index > 2 {
                    return Err(format!(
                        "layer_norm_backward_out: index must be 1 (dw) or 2 (db), got {index}"
                    ));
                }
                (weight.shape.clone(), weight.dtype, weight.device.clone())
            }
            NodeKind::Conv1d {
                x,
                w,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if x.shape.len() != 3 || w.shape.len() != 3 {
                    return Err(format!(
                        "conv1d: expected rank-3 input and weight, got ranks {} and {}",
                        x.shape.len(),
                        w.shape.len()
                    ));
                }
                conv_check("conv1d", x, w, *stride, *padding, *dilation, *groups)?;
                let out = conv_out_dim(x.shape[2], w.shape[2], *stride, *padding, *dilation)?;
                (vec![x.shape[0], w.shape[0], out], x.dtype, x.device.clone())
            }
            NodeKind::Conv2d {
                x,
                w,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if x.shape.len() != 4 || w.shape.len() != 4 {
                    return Err(format!(
                        "conv2d: expected rank-4 input and weight, got ranks {} and {}",
                        x.shape.len(),
                        w.shape.len()
                    ));
                }
                conv_check("conv2d", x, w, *stride, *padding, *dilation, *groups)?;
                let oh = conv_out_dim(x.shape[2], w.shape[2], *stride, *padding, *dilation)?;
                let ow = conv_out_dim(x.shape[3], w.shape[3], *stride, *padding, *dilation)?;
                (
                    vec![x.shape[0], w.shape[0], oh, ow],
                    x.dtype,
                    x.device.clone(),
                )
            }
            NodeKind::ConvTranspose1d {
                x,
                w,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            } => {
                if x.shape.len() != 3 || w.shape.len() != 3 {
                    return Err("conv_transpose1d: expected rank-3 input and weight".to_string());
                }
                let out =
                    (x.shape[2] - 1) * stride + dilation * (w.shape[2] - 1) + output_padding + 1
                        - 2 * padding;
                (
                    vec![x.shape[0], w.shape[1] * groups, out],
                    x.dtype,
                    x.device.clone(),
                )
            }
            NodeKind::ConvTranspose2d {
                x,
                w,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            } => {
                if x.shape.len() != 4 || w.shape.len() != 4 {
                    return Err("conv_transpose2d: expected rank-4 input and weight".to_string());
                }
                let oh =
                    (x.shape[2] - 1) * stride + dilation * (w.shape[2] - 1) + output_padding + 1
                        - 2 * padding;
                let ow =
                    (x.shape[3] - 1) * stride + dilation * (w.shape[3] - 1) + output_padding + 1
                        - 2 * padding;
                (
                    vec![x.shape[0], w.shape[1] * groups, oh, ow],
                    x.dtype,
                    x.device.clone(),
                )
            }
            NodeKind::Conv1dBackwardW {
                x,
                kernel,
                out_channels,
                groups,
                ..
            } => (
                vec![*out_channels, x.shape[1] / groups, *kernel],
                x.dtype,
                x.device.clone(),
            ),
            NodeKind::Conv2dBackwardW {
                x,
                kernel,
                out_channels,
                groups,
                ..
            } => (
                vec![*out_channels, x.shape[1] / groups, kernel[0], kernel[1]],
                x.dtype,
                x.device.clone(),
            ),
            NodeKind::Cast { a, dtype } => (a.shape.clone(), *dtype, a.device.clone()),
            NodeKind::Sum { a, dims, keepdims }
            | NodeKind::Mean { a, dims, keepdims }
            | NodeKind::Max { a, dims, keepdims }
            | NodeKind::Min { a, dims, keepdims }
            | NodeKind::Prod { a, dims, keepdims } => (
                reduced_shape(&a.shape, dims, *keepdims),
                a.dtype,
                a.device.clone(),
            ),
            NodeKind::Reshape { a, shape } => {
                let before: usize = a.shape.iter().product();
                let after: usize = shape.iter().product();
                if before != after {
                    return Err(format!(
                        "reshape: cannot reshape {:?} ({before} elements) to {shape:?} ({after} elements)",
                        a.shape
                    ));
                }
                (shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::Permute { a, dims } => {
                if dims.len() != a.shape.len()
                    || dims.iter().any(|&d| d >= a.shape.len())
                    || (1..dims.len()).any(|i| dims[..i].contains(&dims[i]))
                {
                    return Err(format!(
                        "permute: dims {dims:?} are not a permutation of rank {}",
                        a.shape.len()
                    ));
                }
                (
                    dims.iter().map(|&d| a.shape[d]).collect(),
                    a.dtype,
                    a.device.clone(),
                )
            }
            NodeKind::Slice { a, ranges } => {
                if ranges.len() != a.shape.len() {
                    return Err(format!(
                        "slice: expected {} ranges, got {}",
                        a.shape.len(),
                        ranges.len()
                    ));
                }
                let shape = ranges
                    .iter()
                    .map(|&(start, stop, stride)| stop.saturating_sub(start).div_ceil(stride))
                    .collect();
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::Concat { a, b, dim } => {
                if a.shape.len() != b.shape.len() || *dim >= a.shape.len() {
                    return Err(format!(
                        "concat: rank/dim mismatch, {:?} vs {:?} along dim {dim}",
                        a.shape, b.shape
                    ));
                }
                let mut shape = a.shape.clone();
                for i in 0..shape.len() {
                    if i == *dim {
                        shape[i] += b.shape[i];
                    } else if a.shape[i] != b.shape[i] {
                        return Err(format!(
                            "concat: shape mismatch at dim {i}, {:?} vs {:?}",
                            a.shape, b.shape
                        ));
                    }
                }
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::BroadcastTo { a, shape } => {
                if shape.len() < a.shape.len() {
                    return Err(format!(
                        "broadcast_to: cannot broadcast {:?} to lower rank {shape:?}",
                        a.shape
                    ));
                }
                let offset = shape.len() - a.shape.len();
                for (i, &d) in a.shape.iter().enumerate() {
                    if d != shape[offset + i] && d != 1 {
                        return Err(format!(
                            "broadcast_to: cannot broadcast {:?} to {shape:?}",
                            a.shape
                        ));
                    }
                }
                (shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::Matmul { a, b } => {
                if a.shape.len() < 2 || b.shape.len() < 2 {
                    return Err(format!(
                        "matmul: expected tensors of rank >= 2, got {:?} and {:?}",
                        a.shape, b.shape
                    ));
                }
                let ar = a.shape.len();
                let br = b.shape.len();
                if a.shape[ar - 1] != b.shape[br - 2] {
                    return Err(format!(
                        "matmul: inner dimensions mismatch, got {:?} and {:?}",
                        a.shape, b.shape
                    ));
                }
                let mut shape = broadcast_shapes(&a.shape[..ar - 2], &b.shape[..br - 2])?;
                shape.push(a.shape[ar - 2]);
                shape.push(b.shape[br - 1]);
                (shape, a.dtype, a.device.clone())
            }
            NodeKind::Inverse { a } | NodeKind::Det { a } => {
                let rank = a.shape.len();
                if rank < 2 || a.shape[rank - 2] != a.shape[rank - 1] {
                    return Err(format!(
                        "linalg: expected a tensor square on its last two dimensions, got shape {:?}",
                        a.shape
                    ));
                }
                if !a.dtype.is_float() {
                    return Err(format!(
                        "linalg: dtype must be floating point, got {:?}",
                        a.dtype
                    ));
                }
                if matches!(self, NodeKind::Det { .. }) {
                    (a.shape[..rank - 2].to_vec(), a.dtype, a.device.clone())
                } else {
                    (a.shape.clone(), a.dtype, a.device.clone())
                }
            }
            NodeKind::Solve { a, b } => {
                let rank = a.shape.len();
                if rank < 2 || a.shape[rank - 2] != a.shape[rank - 1] {
                    return Err(format!(
                        "solve: expected a coefficient tensor square on its last two dimensions, got shape {:?}",
                        a.shape
                    ));
                }
                if b.shape.len() != rank
                    || b.shape[..rank - 2] != a.shape[..rank - 2]
                    || b.shape[rank - 2] != a.shape[rank - 1]
                {
                    return Err(format!(
                        "solve: expected a right-hand side of shape {:?} with {} rows, got shape {:?}",
                        &a.shape[..rank - 1],
                        a.shape[rank - 1],
                        b.shape
                    ));
                }
                if !a.dtype.is_float() || a.dtype != b.dtype {
                    return Err(format!(
                        "solve: dtypes must be floating point and match, got {:?} and {:?}",
                        a.dtype, b.dtype
                    ));
                }
                (b.shape.clone(), a.dtype, a.device.clone())
            }
            NodeKind::AdamWStep {
                param,
                grad,
                m,
                v,
                lr,
                c1,
                c2,
                ..
            } => {
                if !param.dtype.is_float() {
                    return Err(format!(
                        "adamw_step: dtype must be floating point, got {:?}",
                        param.dtype
                    ));
                }
                for (name, t) in [("grad", grad), ("m", m), ("v", v)] {
                    if t.shape != param.shape || t.dtype != param.dtype {
                        return Err(format!(
                            "adamw_step: {name} must match the parameter shape and dtype"
                        ));
                    }
                }
                for (name, t) in [("lr", lr), ("c1", c1), ("c2", c2)] {
                    if !t.shape.is_empty() {
                        return Err(format!("adamw_step: {name} must be a scalar (0-d) tensor"));
                    }
                }
                (param.shape.clone(), param.dtype, param.device.clone())
            }
            NodeKind::AdamWOut { step, index } => {
                if *index > 2 {
                    return Err(format!("adamw_out: index must be 0, 1 or 2, got {index}"));
                }
                (step.shape.clone(), step.dtype, step.device.clone())
            }
            NodeKind::SgdStep {
                param,
                grad,
                velocity,
                first,
                lr,
                ..
            } => {
                if !param.dtype.is_float() {
                    return Err(format!(
                        "sgd_step: dtype must be floating point, got {:?}",
                        param.dtype
                    ));
                }
                for (name, t) in [("grad", grad), ("velocity", velocity)] {
                    if t.shape != param.shape || t.dtype != param.dtype {
                        return Err(format!(
                            "sgd_step: {name} must match the parameter shape and dtype"
                        ));
                    }
                }
                for (name, t) in [("first", first), ("lr", lr)] {
                    if !t.shape.is_empty() {
                        return Err(format!("sgd_step: {name} must be a scalar (0-d) tensor"));
                    }
                }
                (param.shape.clone(), param.dtype, param.device.clone())
            }
            NodeKind::SgdOut { step, index } => {
                if *index > 1 {
                    return Err(format!("sgd_out: index must be 0 or 1, got {index}"));
                }
                (step.shape.clone(), step.dtype, step.device.clone())
            }
        };
        if node_children(self)
            .iter()
            .any(|child| !child.device.same_device(&device))
        {
            return Err(format!("node operands must use device {device}"));
        }
        check_dtype_device(dtype, &device)?;
        Ok(NodeMetadata {
            shape,
            dtype,
            device,
        })
    }
}
// RFC 0012 device dtype capabilities. Node::new enforces them during graph
// construction. Every lazy node passes through this check, including
// from-bytes leaves and nodes rebuilt by compiler or fusion rewrites. Metal's
// shading language has no f64. This check never downcasts or defers failure to
// compute time. CUDA supports the complete logical dtype set.
fn check_dtype_device(dtype: DType, device: &Device) -> std::result::Result<(), String> {
    if matches!(dtype, DType::F64) && device.is_metal() {
        return Err(
            "dtype f64 is not supported on device metal (supported: f32, f16, bf16, i64, u32, u8); cast explicitly or use device cpu"
                .to_string(),
        );
    }
    Ok(())
}
/// The direct children of a node, in operand order. Leaves, placeholders,
/// constants and generators have none.
///
/// Permuting contiguous storage only along extent-1 axes leaves its
/// linearization unchanged. Lowering can use a zero-cost alias instead of a
/// transpose kernel. `dims` maps each output axis to a source axis. This
/// check compares non-unit source axes in output order with input order.
pub fn permute_moves_only_unit_axes(shape: &[usize], dims: &[usize]) -> bool {
    let non_unit = shape
        .iter()
        .enumerate()
        .filter(|(_, extent)| **extent != 1)
        .map(|(axis, _)| axis);
    dims.iter()
        .filter(|source| shape[**source] != 1)
        .copied()
        .eq(non_unit)
}

/// The canonical child enumeration. Evaluators, autodiff, rewrites, and the
/// iterative [`Node`] destructor require it to be exhaustive.
pub fn node_children(kind: &NodeKind) -> Vec<Arc<Node>> {
    match kind {
        NodeKind::Leaf(_)
        | NodeKind::Input { .. }
        | NodeKind::ScalarInput { .. }
        | NodeKind::FromBytes { .. }
        | NodeKind::Zeros { .. }
        | NodeKind::Ones { .. }
        | NodeKind::Full { .. }
        | NodeKind::Randn { .. }
        | NodeKind::Uniform { .. }
        | NodeKind::Arange { .. }
        | NodeKind::Eye { .. } => vec![],
        NodeKind::Add { a, b }
        | NodeKind::Sub { a, b }
        | NodeKind::Mul { a, b }
        | NodeKind::Div { a, b }
        | NodeKind::Eq { a, b }
        | NodeKind::Gt { a, b }
        | NodeKind::Lt { a, b }
        | NodeKind::Ge { a, b }
        | NodeKind::Le { a, b }
        | NodeKind::Maximum { a, b }
        | NodeKind::Minimum { a, b }
        | NodeKind::Concat { a, b, .. }
        | NodeKind::Matmul { a, b } => vec![a.clone(), b.clone()],
        NodeKind::Solve { a, b } => vec![a.clone(), b.clone()],
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
        | NodeKind::Gelu { a, .. }
        | NodeKind::Floor { a }
        | NodeKind::Ceil { a }
        | NodeKind::Round { a }
        | NodeKind::Sign { a }
        | NodeKind::Argmax { a, .. }
        | NodeKind::Argmin { a, .. }
        | NodeKind::Inverse { a }
        | NodeKind::Det { a }
        | NodeKind::Cumsum { a, .. }
        | NodeKind::Pow { a, .. }
        | NodeKind::Cast { a, .. }
        | NodeKind::Sum { a, .. }
        | NodeKind::Mean { a, .. }
        | NodeKind::Max { a, .. }
        | NodeKind::Min { a, .. }
        | NodeKind::Prod { a, .. }
        | NodeKind::Reshape { a, .. }
        | NodeKind::Permute { a, .. }
        | NodeKind::Slice { a, .. }
        | NodeKind::BroadcastTo { a, .. }
        | NodeKind::Checkpoint { a }
        | NodeKind::StopGradient { a }
        | NodeKind::Expose { a, .. } => vec![a.clone()],
        NodeKind::Where { cond, a, b } => vec![cond.clone(), a.clone(), b.clone()],
        NodeKind::IndexSelect { a, indexes, .. } => vec![a.clone(), indexes.clone()],
        NodeKind::Gather { a, indexes, .. } => vec![a.clone(), indexes.clone()],
        NodeKind::CrossEntropy { logits, target, .. }
        | NodeKind::CrossEntropyBackward { logits, target, .. } => {
            vec![logits.clone(), target.clone()]
        }
        NodeKind::Sdpa { q, k, v, .. } => vec![q.clone(), k.clone(), v.clone()],
        NodeKind::KvAttention { q, k, v, .. } => vec![q.clone(), k.clone(), v.clone()],
        NodeKind::KdaChunk {
            q,
            k,
            v,
            log_decay,
            beta,
            ..
        } => vec![
            q.clone(),
            k.clone(),
            v.clone(),
            log_decay.clone(),
            beta.clone(),
        ],
        NodeKind::KdaRecurrence {
            q,
            k,
            v,
            log_decay,
            beta,
            ..
        } => vec![
            q.clone(),
            k.clone(),
            v.clone(),
            log_decay.clone(),
            beta.clone(),
        ],
        NodeKind::ShortConv1d { x, weight } | NodeKind::ConvState { x, weight, .. } => {
            vec![x.clone(), weight.clone()]
        }
        NodeKind::ChunkedHeadCe {
            x,
            weight,
            bias,
            target,
            ..
        } => vec![x.clone(), weight.clone(), bias.clone(), target.clone()],
        NodeKind::ChunkedHeadCeBackward {
            x,
            weight,
            bias,
            target,
            g,
            ..
        } => vec![
            x.clone(),
            weight.clone(),
            bias.clone(),
            target.clone(),
            g.clone(),
        ],
        NodeKind::ChunkedHeadCeBackwardOut { of, .. } => vec![of.clone()],
        NodeKind::KdaBackward {
            q,
            k,
            v,
            log_decay,
            beta,
            g,
            ..
        } => vec![
            q.clone(),
            k.clone(),
            v.clone(),
            log_decay.clone(),
            beta.clone(),
            g.clone(),
        ],
        NodeKind::KdaBackwardOut { of, .. } => vec![of.clone()],
        NodeKind::ShortConv1dBackwardX { x, weight, g }
        | NodeKind::ShortConv1dBackwardW { x, weight, g } => {
            vec![x.clone(), weight.clone(), g.clone()]
        }
        NodeKind::PositionEmbedding { weight, .. } => vec![weight.clone()],
        NodeKind::LastTokenRow { a } => vec![a.clone()],
        NodeKind::RotaryEmbedding { x, .. } => vec![x.clone()],
        NodeKind::RotaryEmbeddingBackward { g, .. } => vec![g.clone()],
        NodeKind::LayerNorm {
            x, weight, bias, ..
        } => vec![x.clone(), weight.clone(), bias.clone()],
        NodeKind::RmsNorm { x, weight, .. } => {
            let mut children = vec![x.clone()];
            children.extend(weight.iter().cloned());
            children
        }
        NodeKind::LayerNormBackward { x, weight, g, .. } => {
            vec![x.clone(), weight.clone(), g.clone()]
        }
        NodeKind::LayerNormBackwardOut { of, .. } => vec![of.clone()],
        NodeKind::Linear { x, weight, bias } => vec![x.clone(), weight.clone(), bias.clone()],
        NodeKind::QuantizedLinear {
            x, weight, bias, ..
        } => {
            let mut children = vec![x.clone(), weight.clone()];
            children.extend(bias.iter().cloned());
            children
        }
        NodeKind::QuantizedEmbedding {
            indexes, weight, ..
        } => vec![indexes.clone(), weight.clone()],
        NodeKind::SdpaBackward {
            q, k, v, g, fwd, ..
        } => {
            vec![q.clone(), k.clone(), v.clone(), g.clone(), fwd.clone()]
        }
        NodeKind::SdpaBackwardOut { of, .. } => vec![of.clone()],
        NodeKind::Conv1d { x, w, .. }
        | NodeKind::Conv2d { x, w, .. }
        | NodeKind::ConvTranspose1d { x, w, .. }
        | NodeKind::ConvTranspose2d { x, w, .. } => vec![x.clone(), w.clone()],
        NodeKind::Conv1dBackwardW { x, g, .. } | NodeKind::Conv2dBackwardW { x, g, .. } => {
            vec![x.clone(), g.clone()]
        }
        NodeKind::ScatterAdd {
            a, indexes, src, ..
        } => vec![a.clone(), indexes.clone(), src.clone()],
        NodeKind::AdamWStep {
            param,
            grad,
            m,
            v,
            lr,
            c1,
            c2,
            ..
        } => vec![
            param.clone(),
            grad.clone(),
            m.clone(),
            v.clone(),
            lr.clone(),
            c1.clone(),
            c2.clone(),
        ],
        NodeKind::AdamWOut { step, .. } => vec![step.clone()],
        NodeKind::SgdStep {
            param,
            grad,
            velocity,
            first,
            lr,
            ..
        } => vec![
            param.clone(),
            grad.clone(),
            velocity.clone(),
            first.clone(),
            lr.clone(),
        ],
        NodeKind::SgdOut { step, .. } => vec![step.clone()],
    }
}

/// Rebuilds a node kind with its children mapped through `f`. Checkpoint
/// recomputation uses it to copy subgraphs with fresh node ids.
///
/// Clones non-child fields without validation. Pass the result to
/// [`Node::new`] to validate it.
pub fn remap_children(kind: &NodeKind, f: &dyn Fn(&Arc<Node>) -> Arc<Node>) -> NodeKind {
    match kind {
        NodeKind::Leaf(t) => NodeKind::Leaf(t.clone()),
        NodeKind::Input {
            slot,
            shape,
            dtype,
            device,
        } => NodeKind::Input {
            slot: *slot,
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::ScalarInput {
            slot,
            dtype,
            device,
        } => NodeKind::ScalarInput {
            slot: *slot,
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::FromBytes {
            data,
            shape,
            dtype,
            device,
        } => NodeKind::FromBytes {
            data: data.clone(),
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Zeros {
            shape,
            dtype,
            device,
        } => NodeKind::Zeros {
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Ones {
            shape,
            dtype,
            device,
        } => NodeKind::Ones {
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Full {
            shape,
            value,
            dtype,
            device,
        } => NodeKind::Full {
            shape: shape.clone(),
            value: *value,
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Randn {
            shape,
            dtype,
            device,
        } => NodeKind::Randn {
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Uniform {
            lo,
            hi,
            shape,
            dtype,
            device,
        } => NodeKind::Uniform {
            lo: *lo,
            hi: *hi,
            shape: shape.clone(),
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Arange {
            start,
            end,
            step,
            dtype,
            device,
        } => NodeKind::Arange {
            start: *start,
            end: *end,
            step: *step,
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Eye { n, dtype, device } => NodeKind::Eye {
            n: *n,
            dtype: *dtype,
            device: device.clone(),
        },
        NodeKind::Add { a, b } => NodeKind::Add { a: f(a), b: f(b) },
        NodeKind::Sub { a, b } => NodeKind::Sub { a: f(a), b: f(b) },
        NodeKind::Mul { a, b } => NodeKind::Mul { a: f(a), b: f(b) },
        NodeKind::Div { a, b } => NodeKind::Div { a: f(a), b: f(b) },
        NodeKind::Eq { a, b } => NodeKind::Eq { a: f(a), b: f(b) },
        NodeKind::Gt { a, b } => NodeKind::Gt { a: f(a), b: f(b) },
        NodeKind::Lt { a, b } => NodeKind::Lt { a: f(a), b: f(b) },
        NodeKind::Ge { a, b } => NodeKind::Ge { a: f(a), b: f(b) },
        NodeKind::Le { a, b } => NodeKind::Le { a: f(a), b: f(b) },
        NodeKind::Maximum { a, b } => NodeKind::Maximum { a: f(a), b: f(b) },
        NodeKind::Minimum { a, b } => NodeKind::Minimum { a: f(a), b: f(b) },
        NodeKind::Concat { a, b, dim } => NodeKind::Concat {
            a: f(a),
            b: f(b),
            dim: *dim,
        },
        NodeKind::Matmul { a, b } => NodeKind::Matmul { a: f(a), b: f(b) },
        NodeKind::Solve { a, b } => NodeKind::Solve { a: f(a), b: f(b) },
        NodeKind::Where { cond, a, b } => NodeKind::Where {
            cond: f(cond),
            a: f(a),
            b: f(b),
        },
        NodeKind::IndexSelect { a, dim, indexes } => NodeKind::IndexSelect {
            a: f(a),
            dim: *dim,
            indexes: f(indexes),
        },
        NodeKind::Gather { a, dim, indexes } => NodeKind::Gather {
            a: f(a),
            dim: *dim,
            indexes: f(indexes),
        },
        NodeKind::CrossEntropy {
            logits,
            target,
            ignore_index,
            reduction,
        } => NodeKind::CrossEntropy {
            logits: f(logits),
            target: f(target),
            ignore_index: *ignore_index,
            reduction: *reduction,
        },
        NodeKind::CrossEntropyBackward {
            logits,
            target,
            ignore_index,
            reduction,
        } => NodeKind::CrossEntropyBackward {
            logits: f(logits),
            target: f(target),
            ignore_index: *ignore_index,
            reduction: *reduction,
        },
        NodeKind::Sdpa {
            q,
            k,
            v,
            scale,
            causal,
            window,
        } => NodeKind::Sdpa {
            q: f(q),
            k: f(k),
            v: f(v),
            scale: *scale,
            causal: *causal,
            window: *window,
        },
        NodeKind::SdpaBackward {
            q,
            k,
            v,
            g,
            fwd,
            scale,
            causal,
            window,
        } => NodeKind::SdpaBackward {
            q: f(q),
            k: f(k),
            v: f(v),
            g: f(g),
            fwd: f(fwd),
            scale: *scale,
            causal: *causal,
            window: *window,
        },
        NodeKind::SdpaBackwardOut { of, index } => NodeKind::SdpaBackwardOut {
            of: f(of),
            index: *index,
        },
        NodeKind::PositionEmbedding { weight, seq_len } => NodeKind::PositionEmbedding {
            weight: f(weight),
            seq_len: *seq_len,
        },
        NodeKind::KvAttention {
            q,
            k,
            v,
            scale,
            layer,
            window,
            mode,
        } => NodeKind::KvAttention {
            q: f(q),
            k: f(k),
            v: f(v),
            scale: *scale,
            layer: *layer,
            window: *window,
            mode: *mode,
        },
        NodeKind::KdaChunk {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
        } => NodeKind::KdaChunk {
            q: f(q),
            k: f(k),
            v: f(v),
            log_decay: f(log_decay),
            beta: f(beta),
            scale: *scale,
        },
        NodeKind::KdaRecurrence {
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
            layer,
        } => NodeKind::KdaRecurrence {
            q: f(q),
            k: f(k),
            v: f(v),
            log_decay: f(log_decay),
            beta: f(beta),
            scale: *scale,
            layer: *layer,
        },
        NodeKind::ShortConv1d { x, weight } => NodeKind::ShortConv1d {
            x: f(x),
            weight: f(weight),
        },
        NodeKind::ConvState { x, weight, layer } => NodeKind::ConvState {
            x: f(x),
            weight: f(weight),
            layer: *layer,
        },
        NodeKind::LastTokenRow { a } => NodeKind::LastTokenRow { a: f(a) },
        NodeKind::ChunkedHeadCe {
            x,
            weight,
            bias,
            target,
            ignore_index,
        } => NodeKind::ChunkedHeadCe {
            x: f(x),
            weight: f(weight),
            bias: f(bias),
            target: f(target),
            ignore_index: *ignore_index,
        },
        NodeKind::ChunkedHeadCeBackward {
            x,
            weight,
            bias,
            target,
            g,
            ignore_index,
        } => NodeKind::ChunkedHeadCeBackward {
            x: f(x),
            weight: f(weight),
            bias: f(bias),
            target: f(target),
            g: f(g),
            ignore_index: *ignore_index,
        },
        NodeKind::ChunkedHeadCeBackwardOut { of, index } => NodeKind::ChunkedHeadCeBackwardOut {
            of: f(of),
            index: *index,
        },
        NodeKind::KdaBackward {
            q,
            k,
            v,
            log_decay,
            beta,
            g,
            scale,
        } => NodeKind::KdaBackward {
            q: f(q),
            k: f(k),
            v: f(v),
            log_decay: f(log_decay),
            beta: f(beta),
            g: f(g),
            scale: *scale,
        },
        NodeKind::KdaBackwardOut { of, index } => NodeKind::KdaBackwardOut {
            of: f(of),
            index: *index,
        },
        NodeKind::ShortConv1dBackwardX { x, weight, g } => NodeKind::ShortConv1dBackwardX {
            x: f(x),
            weight: f(weight),
            g: f(g),
        },
        NodeKind::ShortConv1dBackwardW { x, weight, g } => NodeKind::ShortConv1dBackwardW {
            x: f(x),
            weight: f(weight),
            g: f(g),
        },
        NodeKind::RotaryEmbedding {
            x,
            seq_len,
            theta,
            offset,
            layout,
        } => NodeKind::RotaryEmbedding {
            x: f(x),
            seq_len: *seq_len,
            theta: *theta,
            offset: *offset,
            layout: *layout,
        },
        NodeKind::RotaryEmbeddingBackward {
            g,
            shape,
            seq_len,
            theta,
            layout,
        } => NodeKind::RotaryEmbeddingBackward {
            g: f(g),
            shape: shape.clone(),
            seq_len: *seq_len,
            theta: *theta,
            layout: *layout,
        },
        NodeKind::LayerNorm {
            x,
            weight,
            bias,
            eps,
        } => NodeKind::LayerNorm {
            x: f(x),
            weight: f(weight),
            bias: f(bias),
            eps: *eps,
        },
        NodeKind::RmsNorm { x, weight, eps } => NodeKind::RmsNorm {
            x: f(x),
            weight: weight.as_ref().map(f),
            eps: *eps,
        },
        NodeKind::LayerNormBackward { x, weight, g, eps } => NodeKind::LayerNormBackward {
            x: f(x),
            weight: f(weight),
            g: f(g),
            eps: *eps,
        },
        NodeKind::LayerNormBackwardOut { of, index } => NodeKind::LayerNormBackwardOut {
            of: f(of),
            index: *index,
        },
        NodeKind::Linear { x, weight, bias } => NodeKind::Linear {
            x: f(x),
            weight: f(weight),
            bias: f(bias),
        },
        NodeKind::QuantizedLinear {
            x,
            weight,
            bias,
            codec,
            weight_shape,
        } => NodeKind::QuantizedLinear {
            x: f(x),
            weight: f(weight),
            bias: bias.as_ref().map(f),
            codec: *codec,
            weight_shape: *weight_shape,
        },
        NodeKind::QuantizedEmbedding {
            indexes,
            weight,
            codec,
            weight_shape,
            padding_index,
        } => NodeKind::QuantizedEmbedding {
            indexes: f(indexes),
            weight: f(weight),
            codec: *codec,
            weight_shape: *weight_shape,
            padding_index: *padding_index,
        },
        NodeKind::Conv1d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => NodeKind::Conv1d {
            x: f(x),
            w: f(w),
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::Conv2d {
            x,
            w,
            stride,
            padding,
            dilation,
            groups,
        } => NodeKind::Conv2d {
            x: f(x),
            w: f(w),
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::ConvTranspose1d {
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => NodeKind::ConvTranspose1d {
            x: f(x),
            w: f(w),
            stride: *stride,
            padding: *padding,
            output_padding: *output_padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::ConvTranspose2d {
            x,
            w,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => NodeKind::ConvTranspose2d {
            x: f(x),
            w: f(w),
            stride: *stride,
            padding: *padding,
            output_padding: *output_padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::Conv1dBackwardW {
            x,
            g,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => NodeKind::Conv1dBackwardW {
            x: f(x),
            g: f(g),
            kernel: *kernel,
            out_channels: *out_channels,
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::Conv2dBackwardW {
            x,
            g,
            kernel,
            out_channels,
            stride,
            padding,
            dilation,
            groups,
        } => NodeKind::Conv2dBackwardW {
            x: f(x),
            g: f(g),
            kernel: *kernel,
            out_channels: *out_channels,
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        },
        NodeKind::ScatterAdd {
            a,
            dim,
            indexes,
            src,
        } => NodeKind::ScatterAdd {
            a: f(a),
            dim: *dim,
            indexes: f(indexes),
            src: f(src),
        },
        NodeKind::Neg { a } => NodeKind::Neg { a: f(a) },
        NodeKind::Abs { a } => NodeKind::Abs { a: f(a) },
        NodeKind::Sqrt { a } => NodeKind::Sqrt { a: f(a) },
        NodeKind::Exp { a } => NodeKind::Exp { a: f(a) },
        NodeKind::Log { a } => NodeKind::Log { a: f(a) },
        NodeKind::Sin { a } => NodeKind::Sin { a: f(a) },
        NodeKind::Cos { a } => NodeKind::Cos { a: f(a) },
        NodeKind::Tanh { a } => NodeKind::Tanh { a: f(a) },
        NodeKind::Relu { a } => NodeKind::Relu { a: f(a) },
        NodeKind::Erf { a } => NodeKind::Erf { a: f(a) },
        NodeKind::Gelu { a, approximate } => NodeKind::Gelu {
            a: f(a),
            approximate: *approximate,
        },
        NodeKind::Floor { a } => NodeKind::Floor { a: f(a) },
        NodeKind::Ceil { a } => NodeKind::Ceil { a: f(a) },
        NodeKind::Round { a } => NodeKind::Round { a: f(a) },
        NodeKind::Sign { a } => NodeKind::Sign { a: f(a) },
        NodeKind::Argmax { a, dim } => NodeKind::Argmax { a: f(a), dim: *dim },
        NodeKind::Argmin { a, dim } => NodeKind::Argmin { a: f(a), dim: *dim },
        NodeKind::Inverse { a } => NodeKind::Inverse { a: f(a) },
        NodeKind::Det { a } => NodeKind::Det { a: f(a) },
        NodeKind::Cumsum { a, dim } => NodeKind::Cumsum { a: f(a), dim: *dim },
        NodeKind::Pow { a, exp } => NodeKind::Pow { a: f(a), exp: *exp },
        NodeKind::Cast { a, dtype } => NodeKind::Cast {
            a: f(a),
            dtype: *dtype,
        },
        NodeKind::Sum { a, dims, keepdims } => NodeKind::Sum {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Mean { a, dims, keepdims } => NodeKind::Mean {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Max { a, dims, keepdims } => NodeKind::Max {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Min { a, dims, keepdims } => NodeKind::Min {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Prod { a, dims, keepdims } => NodeKind::Prod {
            a: f(a),
            dims: dims.clone(),
            keepdims: *keepdims,
        },
        NodeKind::Reshape { a, shape } => NodeKind::Reshape {
            a: f(a),
            shape: shape.clone(),
        },
        NodeKind::Permute { a, dims } => NodeKind::Permute {
            a: f(a),
            dims: dims.clone(),
        },
        NodeKind::Slice { a, ranges } => NodeKind::Slice {
            a: f(a),
            ranges: ranges.clone(),
        },
        NodeKind::BroadcastTo { a, shape } => NodeKind::BroadcastTo {
            a: f(a),
            shape: shape.clone(),
        },
        NodeKind::Expose { a, name } => NodeKind::Expose {
            a: f(a),
            name: name.clone(),
        },
        NodeKind::Checkpoint { a } => NodeKind::Checkpoint { a: f(a) },
        NodeKind::StopGradient { a } => NodeKind::StopGradient { a: f(a) },
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
        } => NodeKind::AdamWStep {
            param: f(param),
            grad: f(grad),
            m: f(m),
            v: f(v),
            lr: f(lr),
            c1: f(c1),
            c2: f(c2),
            beta1: *beta1,
            beta2: *beta2,
            eps: *eps,
            weight_decay: *weight_decay,
        },
        NodeKind::AdamWOut { step, index } => NodeKind::AdamWOut {
            step: f(step),
            index: *index,
        },
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
        } => NodeKind::SgdStep {
            param: f(param),
            grad: f(grad),
            velocity: f(velocity),
            first: f(first),
            lr: f(lr),
            momentum: *momentum,
            dampening: *dampening,
            nesterov: *nesterov,
            weight_decay: *weight_decay,
        },
        NodeKind::SgdOut { step, index } => NodeKind::SgdOut {
            step: f(step),
            index: *index,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_identity_includes_the_ordinal() {
        assert!(Device::Cpu(2).is_cpu());
        assert!(Device::Metal(3).is_metal());
        assert!(Device::Cuda(4).is_cuda());
        assert_eq!(Device::Metal(3).ordinal(), 3);
        assert!(!Device::Metal(3).same_device(&Device::Metal(4)));
        assert_eq!(Device::Cuda(4).to_string(), "cuda:4");
    }

    #[test]
    fn node_operands_must_use_the_same_ordinal() {
        let input = |slot, ordinal| {
            Node::new(NodeKind::Input {
                slot,
                shape: vec![1],
                dtype: DType::F32,
                device: Device::Metal(ordinal),
            })
            .unwrap()
        };
        let error = Node::new(NodeKind::Add {
            a: input(0, 1),
            b: input(1, 2),
        })
        .err()
        .unwrap();
        assert_eq!(error, "node operands must use device metal:1");
    }

    #[test]
    fn permute_moves_only_unit_axes_predicate() {
        // heads-first on [B, 1, H, W]: only the unit sequence axis moves.
        assert!(permute_moves_only_unit_axes(&[2, 1, 2, 2], &[0, 2, 1, 3]));
        // [1, 3] transposed to [3, 1] is a linearization no-op.
        assert!(permute_moves_only_unit_axes(&[1, 3], &[1, 0]));
        assert!(permute_moves_only_unit_axes(&[2, 3], &[0, 1]));
        // Real transposes and reorderings are not elided.
        assert!(!permute_moves_only_unit_axes(&[2, 3], &[1, 0]));
        assert!(!permute_moves_only_unit_axes(&[2, 2, 1], &[1, 0, 2]));
    }

    #[derive(Clone)]
    struct TestLeaf {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    }

    impl LeafValue for TestLeaf {
        fn shape(&self) -> Vec<usize> {
            self.shape.clone()
        }

        fn dtype(&self) -> DType {
            self.dtype
        }

        fn device(&self) -> Device {
            self.device.clone()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn semantic_nodes_own_authoritative_metadata_and_traversal() {
        let a = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2, 1],
            dtype: DType::F32,
            device: Device::Cpu(0),
        })
        .unwrap();
        let b = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![1, 3],
            dtype: DType::F32,
            device: Device::Cpu(0),
        })
        .unwrap();
        let add = Node::new(NodeKind::Add {
            a: a.clone(),
            b: b.clone(),
        })
        .unwrap();
        assert_eq!(add.shape, [2, 3]);
        assert_eq!(node_children(&add.kind).len(), 2);

        let remapped = remap_children(&add.kind, &|child| {
            if child.id == a.id {
                b.clone()
            } else {
                child.clone()
            }
        });
        let remapped = Node::new(remapped).unwrap();
        assert_eq!(remapped.shape, [1, 3]);
    }

    #[test]
    fn leaf_ownership_is_cleared_once_and_invalidates_graph_construction() {
        let slot = Arc::new(LeafSlot::new(TestLeaf {
            shape: vec![4],
            dtype: DType::F32,
            device: Device::Cpu(0),
        }));
        let leaf = Node::new(NodeKind::Leaf(slot.clone())).unwrap();
        assert_eq!(leaf.shape, [4]);
        assert!(slot.clear());
        assert!(!slot.clear());
        assert!(Node::new(NodeKind::Leaf(slot)).is_err());
    }

    #[test]
    fn metadata_validation_rejects_unsupported_device_dtype() {
        let error = Node::new(NodeKind::Zeros {
            shape: vec![1],
            dtype: DType::F64,
            device: Device::Metal(0),
        })
        .err()
        .unwrap();
        assert!(error.contains("f64 is not supported on device metal"));
    }

    #[test]
    fn last_token_row_validates_rank_and_row_contract() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![1, 7, 16],
            dtype: DType::F32,
            device: Device::Cpu(0),
        })
        .unwrap();
        let row = Node::new(NodeKind::LastTokenRow { a: input.clone() }).unwrap();
        assert_eq!(row.shape, [16]);
        assert_eq!(row.dtype, DType::F32);
        assert!(row.device.is_cpu());
        assert_eq!(node_children(&row.kind).len(), 1);

        let remapped = remap_children(&row.kind, &|_| input.clone());
        let NodeKind::LastTokenRow { a } = &remapped else {
            panic!("remap preserves the last-token-row kind")
        };
        assert!(Arc::ptr_eq(a, &input));

        for shape in [vec![7, 16], vec![2, 7, 16], vec![1, 7, 16, 1]] {
            let error = Node::new(NodeKind::LastTokenRow {
                a: Node::new(NodeKind::Input {
                    slot: 0,
                    shape: shape.clone(),
                    dtype: DType::F32,
                    device: Device::Cpu(0),
                })
                .unwrap(),
            })
            .err()
            .unwrap();
            assert_eq!(
                error,
                format!("last_token_row: expected input [1, T, V], got {shape:?}")
            );
        }
    }

    #[test]
    fn quantized_nodes_validate_packed_geometry_and_preserve_logical_outputs() {
        let input = Node::new(NodeKind::Input {
            slot: 0,
            shape: vec![2, 256],
            dtype: DType::F32,
            device: Device::Cpu(0),
        })
        .unwrap();
        let packed = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![3, 144],
            dtype: DType::U8,
            device: Device::Cpu(0),
        })
        .unwrap();
        let linear = Node::new(NodeKind::QuantizedLinear {
            x: input,
            weight: packed.clone(),
            bias: None,
            codec: GgmlKQuant::Q4K,
            weight_shape: [3, 256],
        })
        .unwrap();
        assert_eq!(linear.shape, [2, 3]);
        assert_eq!(linear.dtype, DType::F32);
        assert_eq!(node_children(&linear.kind).len(), 2);

        let indexes = Node::new(NodeKind::Input {
            slot: 2,
            shape: vec![2, 4],
            dtype: DType::U32,
            device: Device::Cpu(0),
        })
        .unwrap();
        let embedding = Node::new(NodeKind::QuantizedEmbedding {
            indexes,
            weight: packed.clone(),
            codec: GgmlKQuant::Q4K,
            weight_shape: [3, 256],
            padding_index: Some(2),
        })
        .unwrap();
        assert_eq!(embedding.shape, [2, 4, 256]);
        assert_eq!(embedding.dtype, DType::F32);

        let invalid = Node::new(NodeKind::QuantizedEmbedding {
            indexes: embedding,
            weight: packed,
            codec: GgmlKQuant::Q5K,
            weight_shape: [3, 256],
            padding_index: None,
        })
        .err()
        .unwrap();
        assert!(invalid.contains("expected packed Q5_K weight [3, 176] u8"));
    }
}

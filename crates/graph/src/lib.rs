use effect_torch_runtime::{DType, GgmlKQuant};
use std::any::Any;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Metal,
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal)
    }
    pub fn same_device(&self, other: &Device) -> bool {
        self == other
    }
    pub fn name(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossEntropyReduction {
    Mean,
    Sum,
}

type CeReduction = CrossEntropyReduction;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeMetadata {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
}

pub trait LeafValue: Any + Send + Sync {
    fn shape(&self) -> Vec<usize>;
    fn dtype(&self) -> DType;
    fn device(&self) -> Device;
    fn as_any(&self) -> &dyn Any;
}

pub struct LeafSlot(Mutex<Option<Arc<dyn LeafValue>>>);

impl LeafSlot {
    pub fn new(value: impl LeafValue) -> Self {
        Self(Mutex::new(Some(Arc::new(value))))
    }

    pub fn clear(&self) -> bool {
        self.0.lock().unwrap().take().is_some()
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClearedLeaf;
impl fmt::Display for ClearedLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tensor was cleared")
    }
}
impl Error for ClearedLeaf {}

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

// Validates rank-2 attention or q [.., Hq, T, D], k [.., Hkv, S, D],
// v [.., Hkv, S, Dv] and returns the output shape [.., Hq, T, Dv].
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

// Validates kda_chunk operands: q, k and log_decay [.., H, T, Dk], v
// [.., H, T, Dv], beta [.., H, T, 1]; returns the output shape
// [.., H, T, Dv]. Leading dims must match exactly.
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

// Validates a chunked head CE operand set: x [.., K], weight [K, V],
// bias [1, V] or [V], target [..] integer. Returns the vocab.
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

// Validates a short_conv1d pair: x [.., T, C], weight [C, K].
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
// Where a position-indexed semantic node reads its base position:
// zero in user graphs, the sequence cursor in decode-rewritten ones.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PositionOffset {
    Absolute,
    Cursor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionWindow {
    Inherit,
    Full,
    Local(usize),
}

impl AttentionWindow {
    pub const fn resolve(self, inherited: Option<usize>) -> Option<usize> {
        match self {
            Self::Inherit => inherited,
            Self::Full => None,
            Self::Local(window) => Some(window),
        }
    }

    pub const fn local(self) -> Option<usize> {
        self.resolve(None)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RotaryLayout {
    HalfSplit,
    InterleavedPairs,
}

pub enum NodeKind {
    Leaf(std::sync::Arc<LeafSlot>),
    // RFC 0008: placeholder leaves for compiled programs. An Input carries
    // the declared signature of one call argument; it evaluates only inside
    // CompiledProgram::run, which binds the slot to an argument buffer.
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
    Full {
        shape: Vec<usize>,
        value: f64,
        dtype: DType,
        device: Device,
    },
    Randn {
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Uniform {
        lo: f64,
        hi: f64,
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
    },
    Arange {
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
        device: Device,
    },
    Eye {
        n: usize,
        dtype: DType,
        device: Device,
    },
    Add {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Sub {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Mul {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Div {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Eq {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Gt {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Lt {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Ge {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Le {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Maximum {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Minimum {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Neg {
        a: Arc<Node>,
    },
    Abs {
        a: Arc<Node>,
    },
    Sqrt {
        a: Arc<Node>,
    },
    Exp {
        a: Arc<Node>,
    },
    Log {
        a: Arc<Node>,
    },
    Sin {
        a: Arc<Node>,
    },
    Cos {
        a: Arc<Node>,
    },
    Tanh {
        a: Arc<Node>,
    },
    Relu {
        a: Arc<Node>,
    },
    Erf {
        a: Arc<Node>,
    },
    // Gaussian error linear unit as a single node (Tensor.gelu).
    // `approximate` selects the tanh form over the exact erf form. A
    // pointwise unary op: folds into fusion regions like tanh/erf.
    Gelu {
        a: Arc<Node>,
        approximate: bool,
    },
    Floor {
        a: Arc<Node>,
    },
    Ceil {
        a: Arc<Node>,
    },
    Round {
        a: Arc<Node>,
    },
    Sign {
        a: Arc<Node>,
    },
    Where {
        cond: Arc<Node>,
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Pow {
        a: Arc<Node>,
        exp: f64,
    },
    Cast {
        a: Arc<Node>,
        dtype: DType,
    },
    Sum {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Mean {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Max {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Min {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Prod {
        a: Arc<Node>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Argmax {
        a: Arc<Node>,
        dim: usize,
    },
    Argmin {
        a: Arc<Node>,
        dim: usize,
    },
    Cumsum {
        a: Arc<Node>,
        dim: usize,
    },
    IndexSelect {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
    },
    ScatterAdd {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
        src: Arc<Node>,
    },
    Gather {
        a: Arc<Node>,
        dim: usize,
        indexes: Arc<Node>,
    },
    CrossEntropy {
        logits: Arc<Node>,
        target: Arc<Node>,
        ignore_index: i64,
        reduction: CeReduction,
    },
    CrossEntropyBackward {
        logits: Arc<Node>,
        target: Arc<Node>,
        ignore_index: i64,
        reduction: CeReduction,
    },
    // Scaled dot-product attention as one semantic node (the SgdStep
    // precedent: semantics in the graph, execution strategy native). The
    // eval arms compose candle ops as the reference implementation; a
    // fused flash kernel can replace them without touching the graph or
    // its adjoints. Shapes: q [.., T, D], k [.., S, D], v [.., S, Dv]
    // with equal leading dims; the output is [.., T, Dv].
    Sdpa {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        scale: f64,
        causal: bool,
        window: AttentionWindow,
    },
    // Closed-form backward: recomputes P = softmax(scores) from q/k and
    // produces (dq, dk, dv) in one eval; consumers read them through
    // SdpaBackwardOut. On Metal the forward's flash kernel stashes the
    // per-row logsumexp in the evaluator, and this node runs the
    // chunked-recompute backward against it (bounded memory); elsewhere
    // it recomputes P with composed candle ops. `fwd` is the Sdpa node
    // this is the adjoint of (reads its output and stashed L). Not
    // differentiable (no second-order).
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
    SdpaBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // Absolute position embedding as one semantic node: rows 0..seq_len
    // of the [max_positions, E] weight table (the Sdpa precedent —
    // semantics in the graph, execution strategy native). Semantic so
    // the RFC 0010 decode rewrite can offset the positions by the
    // runtime cursor instead of re-deriving "this gather is a position
    // embedding" from composed ops.
    PositionEmbedding {
        weight: Arc<Node>,
        seq_len: usize,
    },
    // RFC 0010: paged KV attention, the decode/prefill semantic node
    // produced by the decode rewrite (never written by user code —
    // `compile_decode` turns each causal Sdpa into one). Scatters the
    // new tokens' k/v into the sequence's pool blocks at the cursor,
    // then attends q causally over the last `window` cached positions
    // (None: the whole context). q, k and v are [1, H, T, D]/[1, H, T, Dv]
    // with a shared T (the new tokens); the pool, block table and
    // cursor arrive via the run's kv context, keeping the graph a pure
    // function of its inputs. Not differentiable.
    KvAttention {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        scale: f64,
        layer: u32,
        window: Option<usize>,
    },
    // Kimi Delta Attention (RFC 0018): gated delta-rule linear attention
    // as one semantic node (the Sdpa precedent — semantics in the graph,
    // execution strategy native). q, k and log_decay are [.., H, T, Dk],
    // v is [.., H, T, Dv] and beta is [.., H, T, 1], all with equal
    // leading dims. log_decay holds the raw per-channel log decay rates
    // (<= 0, pre-cumsum; the gate activation lives upstream) and beta is
    // already sigmoided into [0, 1]. The recurrence is
    // S_t = (I - beta_t k_t k_t^T) Diag(exp(log_decay_t)) S_{t-1}
    //     + beta_t k_t v_t^T,   o_t = scale * S_t^T q_t
    // starting from a zero state; the output is [.., H, T, Dv]. The
    // eval arms compose the chunked parallel form (chunk 64, WY
    // representation + UT transform) as the reference implementation;
    // fused kernels can replace them without touching the graph. The
    // decode rewrite turns this into a stateful KdaRecurrence. Not yet
    // differentiable (phase 4 adds the closed-form backward).
    KdaChunk {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        log_decay: Arc<Node>,
        beta: Arc<Node>,
        scale: f64,
    },
    // RFC 0018: stateful KDA recurrence, the decode/prefill semantic
    // node produced by the decode rewrite (never written by user code —
    // `compile_decode` turns each KdaChunk into one). Same operand
    // contract as KdaChunk, but the initial state comes from the
    // sequence's slot in the run's decode context and the final state is
    // written back to it, keeping the graph a pure function of its
    // inputs. Not differentiable.
    KdaRecurrence {
        q: Arc<Node>,
        k: Arc<Node>,
        v: Arc<Node>,
        log_decay: Arc<Node>,
        beta: Arc<Node>,
        scale: f64,
        layer: u32,
    },
    // Closed-form KDA backward (RFC 0018 phase 4): produces (dq, dk, dv,
    // dlog_decay, dbeta) in one eval from the saved operands and the
    // output cotangent g (same shape as the forward output); consumers
    // read them through KdaBackwardOut. The eval arms run the adjoint
    // delta-rule recurrence in reverse with per-chunk recompute (chunk
    // start states retained, per-token states recomputed within the
    // chunk) — bounded memory, no O(T) state retention. Not
    // differentiable (no second-order).
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
    // Causal depthwise short convolution over [.., T, C] with weight
    // [C, K] as one semantic node: y[t, c] = sum_j w[c, j] * x[t-K+1+j, c]
    // with zero history (left zero-padding of K-1). Semantic so the
    // decode rewrite can carry the K-1-token window as sequence state
    // instead of re-deriving "this pad+conv is a short conv" from
    // composed ops.
    ShortConv1d {
        x: Arc<Node>,
        weight: Arc<Node>,
    },
    // RFC 0018: stateful short convolution, the decode/prefill semantic
    // node produced by the decode rewrite (never written by user code).
    // Same contract as ShortConv1d, but the K-1 previous inputs ride the
    // sequence's slot and the new window is written back. Not
    // differentiable.
    ConvState {
        x: Arc<Node>,
        weight: Arc<Node>,
        layer: u32,
    },
    // Inference-only decode output selector produced by the decode
    // rewrite (never written by user code — `compile_decode` with the
    // last-token-row flag wraps each [1, T, V] root in one). Copies the
    // row selected by the run's token advance (advance - 1) out of the
    // sequence dimension, producing [V]; the row index is read from the
    // decode context at execution, so the graph stays a pure function of
    // its inputs. Not differentiable and rejected under vmap.
    LastTokenRow {
        a: Arc<Node>,
    },
    // RFC 0016 phase 2 (as-built revision): the chunked head +
    // cross-entropy as one semantic node. Forward semantics are exactly
    // Mean cross-entropy of Linear(x, weight, bias) against target, but
    // evaluation processes one row-chunk at a time so the [rows, vocab]
    // logits never materialize whole. Semantic so the adjoint can run
    // the same chunk loop with a single transient grad-logits workspace
    // — the graph-rewrite version (per-chunk CE nodes) kept every
    // chunk's backward workspace alive until the head-parameter roots
    // evaluated, which consumer-count release cannot fix.
    ChunkedHeadCe {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
        target: Arc<Node>,
        ignore_index: i64,
    },
    // Closed-form adjoint: one eval loops the chunks, recomputing each
    // chunk's logits and grad-logits in a transient workspace and
    // accumulating (dx, dw, db); consumers read them through
    // ChunkedHeadCeBackwardOut. g is the scalar cotangent of the loss.
    // Not differentiable (no second-order).
    ChunkedHeadCeBackward {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
        target: Arc<Node>,
        g: Arc<Node>,
        ignore_index: i64,
    },
    ChunkedHeadCeBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // ShortConv1d adjoints (RFC 0018 phase 4): dx is the full
    // correlation of the cotangent with the (unflipped) weight, dw the
    // per-tap correlation of cotangent and input over the zero-padded
    // causal window. x rides along for shape/dtype metadata. Not
    // differentiable (no second-order).
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
    // RoPE (GPT-NeoX half-split rotary) as one semantic node: x is
    // [.., T, D] with D even; the last dim rotates in half pairs by
    // (offset + position) * theta^(-2j/D). Absolute positions ride the
    // tensor, attention sees only offsets — so cached K/V stay valid as
    // the context grows, which learned absolute embeddings cannot do.
    // `offset` is Absolute in user graphs; the decode rewrite flips it
    // to Cursor (the run's kv cursor) instead of re-deriving "this
    // arange is a position" from composed ops.
    RotaryEmbedding {
        x: Arc<Node>,
        seq_len: usize,
        theta: f64,
        offset: PositionOffset,
        layout: RotaryLayout,
    },
    // Backward of RotaryEmbedding (absolute positions only): the
    // transpose rotation, evaluated by the same fused kernel with
    // negated angles. Carries the input's shape/seq_len for metadata.
    RotaryEmbeddingBackward {
        g: Arc<Node>,
        shape: Vec<usize>,
        seq_len: usize,
        theta: f64,
        layout: RotaryLayout,
    },
    // Layer normalization over the last dim: y = (x − μ)/√(σ² + eps) ·
    // weight + bias. Semantic node (like RotaryEmbedding) so the fused
    // Metal kernel handles it as one launch and decode compilation can
    // pass it through.
    LayerNorm {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
        eps: f64,
    },
    // RMS normalization over the last dimension:
    // y = x / sqrt(mean(x^2) + eps), optionally multiplied by weight.
    RmsNorm {
        x: Arc<Node>,
        weight: Option<Arc<Node>>,
        eps: f64,
    },
    // Backward of LayerNorm: evaluates dx (its own value) and stores
    // (dw, db) for LayerNormBackwardOut, like the optimizer steps.
    LayerNormBackward {
        x: Arc<Node>,
        weight: Arc<Node>,
        g: Arc<Node>,
        eps: f64,
    },
    // Reads one weight-side output of a LayerNormBackward (1 = dw,
    // 2 = db).
    LayerNormBackwardOut {
        of: Arc<Node>,
        index: u8,
    },
    // Fused linear layer: y = x·W + b in one gemm launch (addmm
    // epilogue on Metal). Semantic node — Model.linear and attention
    // projections build it directly.
    Linear {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Arc<Node>,
    },
    QuantizedLinear {
        x: Arc<Node>,
        weight: Arc<Node>,
        bias: Option<Arc<Node>>,
        codec: GgmlKQuant,
        weight_shape: [usize; 2],
    },
    QuantizedEmbedding {
        indexes: Arc<Node>,
        weight: Arc<Node>,
        codec: GgmlKQuant,
        weight_shape: [usize; 2],
        padding_index: Option<usize>,
    },
    Conv1d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv2d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose1d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose2d {
        x: Arc<Node>,
        w: Arc<Node>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
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
    Reshape {
        a: Arc<Node>,
        shape: Vec<usize>,
    },
    Permute {
        a: Arc<Node>,
        dims: Vec<usize>,
    },
    Slice {
        a: Arc<Node>,
        ranges: Vec<(usize, usize, usize)>,
    },
    Concat {
        a: Arc<Node>,
        b: Arc<Node>,
        dim: usize,
    },
    BroadcastTo {
        a: Arc<Node>,
        shape: Vec<usize>,
    },
    Matmul {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    Inverse {
        a: Arc<Node>,
    },
    Det {
        a: Arc<Node>,
    },
    Solve {
        a: Arc<Node>,
        b: Arc<Node>,
    },
    // lr, c1 (1 - beta1^t) and c2 (1 - beta2^t) are 0-d tensor children:
    // step-varying values flow through the graph so a frozen graph (RFC
    // 0008) never replays a stale step count or learning rate.
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
    AdamWOut {
        step: Arc<Node>,
        index: u8,
    },
    // `first` is a 0-d flag (1.0 on the first step, 0.0 after) selecting
    // v = g over v = momentum * v + (1 - dampening) * g; velocity is always
    // a real buffer (zeros at init), so no placeholder is needed.
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
    SgdOut {
        step: Arc<Node>,
        index: u8,
    },
    StopGradient {
        a: Arc<Node>,
    },
    Checkpoint {
        a: Arc<Node>,
    },
}

pub struct Node {
    pub id: u64,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
    pub kind: NodeKind,
}

impl Drop for Node {
    fn drop(&mut self) {
        // Deep graphs (long op chains, large fused backward graphs)
        // recurse in default destructor glue through Arc<Node> children
        // and overflow worker-thread stacks, so descendants drain into a
        // worklist: children are cloned out BEFORE their parent's kind
        // drops (so no Node destructor can fire mid-drain), uniquely
        // owned nodes are unwrapped and their kind swapped for a dummy
        // leaf, and their children rejoin the worklist.
        let dummy = || NodeKind::Zeros {
            shape: Vec::new(),
            dtype: DType::F32,
            device: Device::Cpu,
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
            // `node` drops at the end of this iteration with a dummy
            // kind; its Drop sees no children and returns immediately.
            worklist.extend(node_children(&kind));
            drop(kind);
        }
    }
}

impl Node {
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

// A 0-d float operand never promotes a float tensor's dtype: scalar ×
// tensor keeps the tensor's dtype (the scalar is cast to it at
// evaluation), matching PyTorch's scalar promotion rules. Mismatches
// involving integer dtypes keep the legacy first-operand rule.
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
            | NodeKind::StopGradient { a } => (a.shape.clone(), a.dtype, a.device.clone()),
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
                    // Leading dims are the batch (one per kv slot);
                    // slot counts are checked at run time (RFC 0013).
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
        check_dtype_device(dtype, &device)?;
        Ok(NodeMetadata {
            shape,
            dtype,
            device,
        })
    }
}
// RFC 0012: device dtype capabilities, enforced at graph construction
// (Node::new is the single choke point — every lazy node, including
// from-bytes leaves and nodes rebuilt by compile/fuse rewrites, passes
// through here). Metal's shading language has no f64. Never a silent
// downcast, never deferred to compute time.
fn check_dtype_device(dtype: DType, device: &Device) -> std::result::Result<(), String> {
    if matches!(dtype, DType::F64) && matches!(device, Device::Metal) {
        return Err(
            "dtype f64 is not supported on device metal (supported: f32, f16, bf16, i64, u32, u8); cast explicitly or use device cpu"
                .to_string(),
        );
    }
    Ok(())
}
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
        | NodeKind::StopGradient { a } => vec![a.clone()],
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

// Rebuilds a node kind with its children mapped through `f`. Used to
// deep-copy subgraphs with fresh node ids (checkpoint recompute).
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
        } => NodeKind::KvAttention {
            q: f(q),
            k: f(k),
            v: f(v),
            scale: *scale,
            layer: *layer,
            window: *window,
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
            device: Device::Cpu,
        })
        .unwrap();
        let b = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![1, 3],
            dtype: DType::F32,
            device: Device::Cpu,
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
            device: Device::Cpu,
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
            device: Device::Metal,
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
            device: Device::Cpu,
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
                    device: Device::Cpu,
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
            device: Device::Cpu,
        })
        .unwrap();
        let packed = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![3, 144],
            dtype: DType::U8,
            device: Device::Cpu,
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
            device: Device::Cpu,
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

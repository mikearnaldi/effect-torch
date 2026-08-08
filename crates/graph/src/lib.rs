use effect_torch_runtime::DType;
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

pub trait FusionExpression: Clone + Send + Sync + 'static {
    type ReduceOp: Copy + Send + Sync + 'static;
    fn lane_strides(lane: &[usize], out: &[usize]) -> Option<Vec<usize>>;
}

fn conv_check<E: FusionExpression>(
    op: &str,
    x: &Node<E>,
    w: &Node<E>,
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

// Validates q [.., T, D], k [.., S, D], v [.., S, Dv] and returns the
// attention output shape [.., T, Dv]. Leading dims must match exactly.
fn sdpa_check<E: FusionExpression>(
    op: &str,
    q: &Node<E>,
    k: &Node<E>,
    v: &Node<E>,
) -> Result<Vec<usize>, String> {
    let rank = q.shape.len();
    if rank < 2 || k.shape.len() != rank || v.shape.len() != rank {
        return Err(format!(
            "{op}: q, k and v must share a rank >= 2, got {:?}, {:?} and {:?}",
            q.shape, k.shape, v.shape
        ));
    }
    if q.shape[..rank - 2] != k.shape[..rank - 2] || q.shape[..rank - 2] != v.shape[..rank - 2] {
        return Err(format!(
            "{op}: leading dims must match, got {:?}, {:?} and {:?}",
            q.shape, k.shape, v.shape
        ));
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
fn kda_check<E: FusionExpression>(
    op: &str,
    q: &Node<E>,
    k: &Node<E>,
    v: &Node<E>,
    log_decay: &Node<E>,
    beta: &Node<E>,
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

// Validates a short_conv1d pair: x [.., T, C], weight [C, K].
fn short_conv_check<E: FusionExpression>(
    op: &str,
    x: &Node<E>,
    weight: &Node<E>,
) -> Result<(), String> {
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

pub enum NodeKind<E: FusionExpression> {
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
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Sub {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Mul {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Div {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Eq {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Gt {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Lt {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Ge {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Le {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Maximum {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Minimum {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Neg {
        a: Arc<Node<E>>,
    },
    Abs {
        a: Arc<Node<E>>,
    },
    Sqrt {
        a: Arc<Node<E>>,
    },
    Exp {
        a: Arc<Node<E>>,
    },
    Log {
        a: Arc<Node<E>>,
    },
    Sin {
        a: Arc<Node<E>>,
    },
    Cos {
        a: Arc<Node<E>>,
    },
    Tanh {
        a: Arc<Node<E>>,
    },
    Relu {
        a: Arc<Node<E>>,
    },
    Erf {
        a: Arc<Node<E>>,
    },
    // Gaussian error linear unit as a single node (Tensor.gelu).
    // `approximate` selects the tanh form over the exact erf form. A
    // pointwise unary op: folds into fusion regions like tanh/erf.
    Gelu {
        a: Arc<Node<E>>,
        approximate: bool,
    },
    Floor {
        a: Arc<Node<E>>,
    },
    Ceil {
        a: Arc<Node<E>>,
    },
    Round {
        a: Arc<Node<E>>,
    },
    Sign {
        a: Arc<Node<E>>,
    },
    Where {
        cond: Arc<Node<E>>,
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Pow {
        a: Arc<Node<E>>,
        exp: f64,
    },
    Cast {
        a: Arc<Node<E>>,
        dtype: DType,
    },
    Sum {
        a: Arc<Node<E>>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Mean {
        a: Arc<Node<E>>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Max {
        a: Arc<Node<E>>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Min {
        a: Arc<Node<E>>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Prod {
        a: Arc<Node<E>>,
        dims: Vec<usize>,
        keepdims: bool,
    },
    Argmax {
        a: Arc<Node<E>>,
        dim: usize,
    },
    Argmin {
        a: Arc<Node<E>>,
        dim: usize,
    },
    Cumsum {
        a: Arc<Node<E>>,
        dim: usize,
    },
    IndexSelect {
        a: Arc<Node<E>>,
        dim: usize,
        indexes: Arc<Node<E>>,
    },
    ScatterAdd {
        a: Arc<Node<E>>,
        dim: usize,
        indexes: Arc<Node<E>>,
        src: Arc<Node<E>>,
    },
    Gather {
        a: Arc<Node<E>>,
        dim: usize,
        indexes: Arc<Node<E>>,
    },
    CrossEntropy {
        logits: Arc<Node<E>>,
        target: Arc<Node<E>>,
        ignore_index: i64,
        reduction: CeReduction,
    },
    CrossEntropyBackward {
        logits: Arc<Node<E>>,
        target: Arc<Node<E>>,
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
        q: Arc<Node<E>>,
        k: Arc<Node<E>>,
        v: Arc<Node<E>>,
        scale: f64,
        causal: bool,
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
        q: Arc<Node<E>>,
        k: Arc<Node<E>>,
        v: Arc<Node<E>>,
        g: Arc<Node<E>>,
        fwd: Arc<Node<E>>,
        scale: f64,
        causal: bool,
    },
    SdpaBackwardOut {
        of: Arc<Node<E>>,
        index: u8,
    },
    // Absolute position embedding as one semantic node: rows 0..seq_len
    // of the [max_positions, E] weight table (the Sdpa precedent —
    // semantics in the graph, execution strategy native). Semantic so
    // the RFC 0010 decode rewrite can offset the positions by the
    // runtime cursor instead of re-deriving "this gather is a position
    // embedding" from composed ops.
    PositionEmbedding {
        weight: Arc<Node<E>>,
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
        q: Arc<Node<E>>,
        k: Arc<Node<E>>,
        v: Arc<Node<E>>,
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
        q: Arc<Node<E>>,
        k: Arc<Node<E>>,
        v: Arc<Node<E>>,
        log_decay: Arc<Node<E>>,
        beta: Arc<Node<E>>,
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
        q: Arc<Node<E>>,
        k: Arc<Node<E>>,
        v: Arc<Node<E>>,
        log_decay: Arc<Node<E>>,
        beta: Arc<Node<E>>,
        scale: f64,
        layer: u32,
    },
    // Causal depthwise short convolution over [.., T, C] with weight
    // [C, K] as one semantic node: y[t, c] = sum_j w[c, j] * x[t-K+1+j, c]
    // with zero history (left zero-padding of K-1). Semantic so the
    // decode rewrite can carry the K-1-token window as sequence state
    // instead of re-deriving "this pad+conv is a short conv" from
    // composed ops.
    ShortConv1d {
        x: Arc<Node<E>>,
        weight: Arc<Node<E>>,
    },
    // RFC 0018: stateful short convolution, the decode/prefill semantic
    // node produced by the decode rewrite (never written by user code).
    // Same contract as ShortConv1d, but the K-1 previous inputs ride the
    // sequence's slot and the new window is written back. Not
    // differentiable.
    ConvState {
        x: Arc<Node<E>>,
        weight: Arc<Node<E>>,
        layer: u32,
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
        x: Arc<Node<E>>,
        seq_len: usize,
        theta: f64,
        offset: PositionOffset,
    },
    // Backward of RotaryEmbedding (absolute positions only): the
    // transpose rotation, evaluated by the same fused kernel with
    // negated angles. Carries the input's shape/seq_len for metadata.
    RotaryEmbeddingBackward {
        g: Arc<Node<E>>,
        shape: Vec<usize>,
        seq_len: usize,
        theta: f64,
    },
    // Layer normalization over the last dim: y = (x − μ)/√(σ² + eps) ·
    // weight + bias. Semantic node (like RotaryEmbedding) so the fused
    // Metal kernel handles it as one launch and decode compilation can
    // pass it through.
    LayerNorm {
        x: Arc<Node<E>>,
        weight: Arc<Node<E>>,
        bias: Arc<Node<E>>,
        eps: f64,
    },
    // Backward of LayerNorm: evaluates dx (its own value) and stores
    // (dw, db) for LayerNormBackwardOut, like the optimizer steps.
    LayerNormBackward {
        x: Arc<Node<E>>,
        weight: Arc<Node<E>>,
        g: Arc<Node<E>>,
        eps: f64,
    },
    // Reads one weight-side output of a LayerNormBackward (1 = dw,
    // 2 = db).
    LayerNormBackwardOut {
        of: Arc<Node<E>>,
        index: u8,
    },
    // Fused linear layer: y = x·W + b in one gemm launch (addmm
    // epilogue on Metal). Semantic node — Model.linear and attention
    // projections build it directly.
    Linear {
        x: Arc<Node<E>>,
        weight: Arc<Node<E>>,
        bias: Arc<Node<E>>,
    },
    // RFC 0016 phase 3 — created only by the evaluation-time epilogue
    // pass: y = x·W + b + residual in one gemm launch (the residual add
    // rides the epilogue; the standalone proj output never materializes).
    // Never in user graphs, so autodiff and vmap reject it.
    LinearResidual {
        x: Arc<Node<E>>,
        weight: Arc<Node<E>>,
        bias: Arc<Node<E>>,
        residual: Arc<Node<E>>,
    },
    // RFC 0016 phase 3 — created only by the evaluation-time epilogue
    // pass: y = gelu(x·W + b) in one gemm launch. `dual` writes the
    // pre-activation as output 0 as well (backward needs it); consumers
    // read the outputs through FusedPick. Never in user graphs.
    LinearGelu {
        x: Arc<Node<E>>,
        weight: Arc<Node<E>>,
        bias: Arc<Node<E>>,
        approximate: bool,
        dual: bool,
    },
    Conv1d {
        x: Arc<Node<E>>,
        w: Arc<Node<E>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv2d {
        x: Arc<Node<E>>,
        w: Arc<Node<E>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose1d {
        x: Arc<Node<E>>,
        w: Arc<Node<E>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose2d {
        x: Arc<Node<E>>,
        w: Arc<Node<E>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv1dBackwardW {
        x: Arc<Node<E>>,
        g: Arc<Node<E>>,
        kernel: usize,
        out_channels: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Conv2dBackwardW {
        x: Arc<Node<E>>,
        g: Arc<Node<E>>,
        kernel: [usize; 2],
        out_channels: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Reshape {
        a: Arc<Node<E>>,
        shape: Vec<usize>,
    },
    Permute {
        a: Arc<Node<E>>,
        dims: Vec<usize>,
    },
    Slice {
        a: Arc<Node<E>>,
        ranges: Vec<(usize, usize, usize)>,
    },
    Concat {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
        dim: usize,
    },
    BroadcastTo {
        a: Arc<Node<E>>,
        shape: Vec<usize>,
    },
    Matmul {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    Inverse {
        a: Arc<Node<E>>,
    },
    Det {
        a: Arc<Node<E>>,
    },
    Solve {
        a: Arc<Node<E>>,
        b: Arc<Node<E>>,
    },
    // lr, c1 (1 - beta1^t) and c2 (1 - beta2^t) are 0-d tensor children:
    // step-varying values flow through the graph so a frozen graph (RFC
    // 0008) never replays a stale step count or learning rate.
    AdamWStep {
        param: Arc<Node<E>>,
        grad: Arc<Node<E>>,
        m: Arc<Node<E>>,
        v: Arc<Node<E>>,
        lr: Arc<Node<E>>,
        c1: Arc<Node<E>>,
        c2: Arc<Node<E>>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    AdamWOut {
        step: Arc<Node<E>>,
        index: u8,
    },
    // Freeze-time grouping of same-shape AdamW steps (the endgame for
    // the optimizer: one fused launch per group instead of one per
    // parameter — ≤4 params, Metal's 31-buffer limit: 4 lanes + 3
    // outputs each plus 3 scalars).
    AdamWStepGroup {
        params: Vec<Arc<Node<E>>>,
        grads: Vec<Arc<Node<E>>>,
        ms: Vec<Arc<Node<E>>>,
        vs: Vec<Arc<Node<E>>>,
        lr: Arc<Node<E>>,
        c1: Arc<Node<E>>,
        c2: Arc<Node<E>>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    // One output of a grouped step: `param`-th parameter's updated
    // param (0), m (1), or v (2).
    AdamWGroupOut {
        of: Arc<Node<E>>,
        param: u32,
        index: u8,
    },
    // `first` is a 0-d flag (1.0 on the first step, 0.0 after) selecting
    // v = g over v = momentum * v + (1 - dampening) * g; velocity is always
    // a real buffer (zeros at init), so no placeholder is needed.
    SgdStep {
        param: Arc<Node<E>>,
        grad: Arc<Node<E>>,
        velocity: Arc<Node<E>>,
        first: Arc<Node<E>>,
        lr: Arc<Node<E>>,
        momentum: f64,
        dampening: f64,
        nesterov: bool,
        weight_decay: f64,
    },
    SgdOut {
        step: Arc<Node<E>>,
        index: u8,
    },
    // Created only by the evaluation-time fusion rewrite (RFC 0007 phase
    // 2): a maximal chain of elementwise ops compiled to one kernel. Never
    // appears in user graphs, so autodiff and vmap reject it. Input lanes
    // may be broadcast-smaller than the output: `strides` gives each
    // lane's strides in output-dim space (0 = broadcast along that dim).
    FusedElementwise {
        inputs: Vec<Arc<Node<E>>>,
        strides: Vec<Vec<usize>>,
        shape: Vec<usize>,
        expr: E,
    },
    // Created by the multi-output post-pass (RFC 0007): a shared fused
    // prefix and its fused continuations compiled to one kernel with one
    // store per output. Consumers are FusedPick nodes.
    FusedElementwiseMulti {
        inputs: Vec<Arc<Node<E>>>,
        strides: Vec<Vec<usize>>,
        shape: Vec<usize>,
        exprs: Vec<E>,
    },
    // Reads one output of a FusedElementwiseMulti.
    FusedPick {
        of: Arc<Node<E>>,
        index: u8,
    },
    // Created only by the evaluation-time fusion rewrite (RFC 0007 phase
    // 3a): an elementwise chain terminated by a single reduce, compiled to
    // one kernel that evaluates the chain inside the reduce loop — the
    // chain's intermediate never materializes. `strides` are per-lane in
    // input-dim space; `dims` sorted ascending; `shape` is the reduced
    // shape with keepdims applied.
    FusedReduce {
        inputs: Vec<Arc<Node<E>>>,
        strides: Vec<Vec<usize>>,
        in_shape: Vec<usize>,
        expr: E,
        op: <E as FusionExpression>::ReduceOp,
        dims: Vec<usize>,
        keepdims: bool,
        shape: Vec<usize>,
    },
    StopGradient {
        a: Arc<Node<E>>,
    },
    Checkpoint {
        a: Arc<Node<E>>,
    },
}

pub struct Node<E: FusionExpression> {
    pub id: u64,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
    pub kind: NodeKind<E>,
}

impl<E: FusionExpression> Node<E> {
    pub fn new(kind: NodeKind<E>) -> Result<Arc<Self>, String> {
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
fn scalar_aware_binary_dtype<E: FusionExpression>(a: &Node<E>, b: &Node<E>) -> DType {
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
impl<E: FusionExpression> NodeKind<E> {
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
            NodeKind::Sdpa { q, k, v, .. } => {
                let out = sdpa_check("sdpa", q, k, v)?;
                (out, q.dtype, q.device.clone())
            }
            NodeKind::SdpaBackward {
                q, k, v, g, fwd, ..
            } => {
                let out = sdpa_check("sdpa", q, k, v)?;
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
            NodeKind::LinearGelu {
                x, weight, bias, ..
            } => {
                let out = linear_out_shape(&x.shape, &weight.shape, &bias.shape)?;
                (out, x.dtype, x.device.clone())
            }
            NodeKind::LinearResidual {
                x,
                weight,
                bias,
                residual,
            } => {
                let out = linear_out_shape(&x.shape, &weight.shape, &bias.shape)?;
                if residual.shape != out {
                    return Err(format!(
                        "linear residual: residual shape {:?} does not match output {:?}",
                        residual.shape, out
                    ));
                }
                if residual.dtype != x.dtype {
                    return Err(format!(
                        "linear residual: residual dtype {:?} does not match {:?}",
                        residual.dtype, x.dtype
                    ));
                }
                (out, x.dtype, x.device.clone())
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
            NodeKind::AdamWStepGroup {
                params,
                grads,
                ms,
                vs,
                lr,
                c1,
                c2,
                ..
            } => {
                let n = params.len();
                if n == 0 || n > 4 {
                    return Err(format!(
                        "adamw_step_group: groups hold 1..=4 params (the 31-buffer limit: 4 lanes + 3 outputs each plus 3 scalars), got {n}"
                    ));
                }
                if grads.len() != n || ms.len() != n || vs.len() != n {
                    return Err(
                        "adamw_step_group: params, grads, ms and vs must be equally long"
                            .to_string(),
                    );
                }
                let first = &params[0];
                if !first.dtype.is_float() {
                    return Err(format!(
                        "adamw_step_group: dtype must be floating point, got {:?}",
                        first.dtype
                    ));
                }
                for (name, tensors) in [("param", params), ("grad", grads), ("m", ms), ("v", vs)] {
                    for t in tensors {
                        if t.shape != first.shape || t.dtype != first.dtype {
                            return Err(format!(
                                "adamw_step_group: {name} must share the group's shape and dtype"
                            ));
                        }
                    }
                }
                for (name, t) in [("lr", lr), ("c1", c1), ("c2", c2)] {
                    if !t.shape.is_empty() {
                        return Err(format!(
                            "adamw_step_group: {name} must be a scalar (0-d) tensor"
                        ));
                    }
                }
                (first.shape.clone(), first.dtype, first.device.clone())
            }
            NodeKind::AdamWGroupOut { of, param, index } => {
                let NodeKind::AdamWStepGroup { params, .. } = &of.kind else {
                    return Err("adamw_group_out: parent is not a step group".to_string());
                };
                if *index > 2 || *param as usize >= params.len() {
                    return Err(format!(
                        "adamw_group_out: param {param} or index {index} out of range"
                    ));
                }
                (of.shape.clone(), of.dtype, of.device.clone())
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
            NodeKind::FusedElementwise {
                inputs,
                strides,
                shape,
                ..
            } => {
                if inputs.is_empty() {
                    return Err("fused: at least one input lane is required".to_string());
                }
                if strides.len() != inputs.len() {
                    return Err(format!(
                        "fused: got {} stride entries for {} inputs",
                        strides.len(),
                        inputs.len()
                    ));
                }
                let first = &inputs[0];
                for (input, stride) in inputs.iter().zip(strides.iter()) {
                    if input.dtype != first.dtype || !input.device.same_device(&first.device) {
                        return Err("fused: all inputs must share dtype and device".to_string());
                    }
                    if E::lane_strides(&input.shape, shape).as_ref() != Some(stride) {
                        return Err(format!(
                            "fused: input shape {:?} does not broadcast to {:?} with strides {stride:?}",
                            input.shape, shape
                        ));
                    }
                }
                if !first.dtype.is_float() {
                    return Err(format!(
                        "fused: dtype must be floating point, got {:?}",
                        first.dtype
                    ));
                }
                (shape.clone(), first.dtype, first.device.clone())
            }
            NodeKind::FusedElementwiseMulti {
                inputs,
                strides,
                shape,
                exprs,
            } => {
                if exprs.is_empty() {
                    return Err("fused multi: at least one output is required".to_string());
                }
                if inputs.is_empty() {
                    return Err("fused multi: at least one input lane is required".to_string());
                }
                if strides.len() != inputs.len() {
                    return Err(format!(
                        "fused multi: got {} stride entries for {} inputs",
                        strides.len(),
                        inputs.len()
                    ));
                }
                let first = &inputs[0];
                for (input, stride) in inputs.iter().zip(strides.iter()) {
                    if input.dtype != first.dtype || !input.device.same_device(&first.device) {
                        return Err(
                            "fused multi: all inputs must share dtype and device".to_string()
                        );
                    }
                    if E::lane_strides(&input.shape, shape).as_ref() != Some(stride) {
                        return Err(format!(
                            "fused multi: input shape {:?} does not broadcast to {:?} with strides {stride:?}",
                            input.shape, shape
                        ));
                    }
                }
                if !first.dtype.is_float() {
                    return Err(format!(
                        "fused multi: dtype must be floating point, got {:?}",
                        first.dtype
                    ));
                }
                (shape.clone(), first.dtype, first.device.clone())
            }
            NodeKind::FusedPick { of, .. } => (of.shape.clone(), of.dtype, of.device.clone()),
            NodeKind::FusedReduce {
                inputs,
                strides,
                in_shape,
                dims,
                keepdims,
                shape,
                ..
            } => {
                if inputs.is_empty() {
                    return Err("fused reduce: at least one input lane is required".to_string());
                }
                if strides.len() != inputs.len() {
                    return Err(format!(
                        "fused reduce: got {} stride entries for {} inputs",
                        strides.len(),
                        inputs.len()
                    ));
                }
                if dims.is_empty()
                    || dims.iter().any(|&d| d >= in_shape.len())
                    || !dims.windows(2).all(|w| w[0] < w[1])
                {
                    return Err(format!(
                        "fused reduce: dims {dims:?} are not sorted unique dims of {in_shape:?}"
                    ));
                }
                if &reduced_shape(in_shape, dims, *keepdims) != shape {
                    return Err(format!(
                        "fused reduce: shape {shape:?} is not {in_shape:?} reduced over {dims:?} (keepdims {keepdims})"
                    ));
                }
                let first = &inputs[0];
                for (input, stride) in inputs.iter().zip(strides.iter()) {
                    if input.dtype != first.dtype || !input.device.same_device(&first.device) {
                        return Err(
                            "fused reduce: all inputs must share dtype and device".to_string()
                        );
                    }
                    if E::lane_strides(&input.shape, in_shape).as_ref() != Some(stride) {
                        return Err(format!(
                            "fused reduce: input shape {:?} does not broadcast to {:?} with strides {stride:?}",
                            input.shape, in_shape
                        ));
                    }
                }
                if !first.dtype.is_float() {
                    return Err(format!(
                        "fused reduce: dtype must be floating point, got {:?}",
                        first.dtype
                    ));
                }
                (shape.clone(), first.dtype, first.device.clone())
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
pub fn node_children<E: FusionExpression>(kind: &NodeKind<E>) -> Vec<Arc<Node<E>>> {
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
        NodeKind::PositionEmbedding { weight, .. } => vec![weight.clone()],
        NodeKind::RotaryEmbedding { x, .. } => vec![x.clone()],
        NodeKind::RotaryEmbeddingBackward { g, .. } => vec![g.clone()],
        NodeKind::LayerNorm {
            x, weight, bias, ..
        } => vec![x.clone(), weight.clone(), bias.clone()],
        NodeKind::LayerNormBackward { x, weight, g, .. } => {
            vec![x.clone(), weight.clone(), g.clone()]
        }
        NodeKind::LayerNormBackwardOut { of, .. } => vec![of.clone()],
        NodeKind::Linear { x, weight, bias } => vec![x.clone(), weight.clone(), bias.clone()],
        NodeKind::LinearGelu {
            x, weight, bias, ..
        } => vec![x.clone(), weight.clone(), bias.clone()],
        NodeKind::LinearResidual {
            x,
            weight,
            bias,
            residual,
        } => vec![x.clone(), weight.clone(), bias.clone(), residual.clone()],
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
        NodeKind::AdamWStepGroup {
            params,
            grads,
            ms,
            vs,
            lr,
            c1,
            c2,
            ..
        } => params
            .iter()
            .chain(grads.iter())
            .chain(ms.iter())
            .chain(vs.iter())
            .cloned()
            .chain([lr.clone(), c1.clone(), c2.clone()])
            .collect(),
        NodeKind::AdamWGroupOut { of, .. } => vec![of.clone()],
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
        NodeKind::FusedElementwise { inputs, .. } => inputs.clone(),
        NodeKind::FusedElementwiseMulti { inputs, .. } => inputs.clone(),
        NodeKind::FusedReduce { inputs, .. } => inputs.clone(),
        NodeKind::FusedPick { of, .. } => vec![of.clone()],
    }
}

// Rebuilds a node kind with its children mapped through `f`. Used to
// deep-copy subgraphs with fresh node ids (checkpoint recompute).
pub fn remap_children<E: FusionExpression>(
    kind: &NodeKind<E>,
    f: &dyn Fn(&Arc<Node<E>>) -> Arc<Node<E>>,
) -> NodeKind<E> {
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
        } => NodeKind::Sdpa {
            q: f(q),
            k: f(k),
            v: f(v),
            scale: *scale,
            causal: *causal,
        },
        NodeKind::SdpaBackward {
            q,
            k,
            v,
            g,
            fwd,
            scale,
            causal,
        } => NodeKind::SdpaBackward {
            q: f(q),
            k: f(k),
            v: f(v),
            g: f(g),
            fwd: f(fwd),
            scale: *scale,
            causal: *causal,
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
        NodeKind::RotaryEmbedding {
            x,
            seq_len,
            theta,
            offset,
        } => NodeKind::RotaryEmbedding {
            x: f(x),
            seq_len: *seq_len,
            theta: *theta,
            offset: *offset,
        },
        NodeKind::RotaryEmbeddingBackward {
            g,
            shape,
            seq_len,
            theta,
        } => NodeKind::RotaryEmbeddingBackward {
            g: f(g),
            shape: shape.clone(),
            seq_len: *seq_len,
            theta: *theta,
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
        NodeKind::LinearGelu {
            x,
            weight,
            bias,
            approximate,
            dual,
        } => NodeKind::LinearGelu {
            x: f(x),
            weight: f(weight),
            bias: f(bias),
            approximate: *approximate,
            dual: *dual,
        },
        NodeKind::LinearResidual {
            x,
            weight,
            bias,
            residual,
        } => NodeKind::LinearResidual {
            x: f(x),
            weight: f(weight),
            bias: f(bias),
            residual: f(residual),
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
        NodeKind::AdamWStepGroup {
            params,
            grads,
            ms,
            vs,
            lr,
            c1,
            c2,
            beta1,
            beta2,
            eps,
            weight_decay,
        } => NodeKind::AdamWStepGroup {
            params: params.iter().map(|p| f(p)).collect(),
            grads: grads.iter().map(|g| f(g)).collect(),
            ms: ms.iter().map(|m| f(m)).collect(),
            vs: vs.iter().map(|v| f(v)).collect(),
            lr: f(lr),
            c1: f(c1),
            c2: f(c2),
            beta1: *beta1,
            beta2: *beta2,
            eps: *eps,
            weight_decay: *weight_decay,
        },
        NodeKind::AdamWGroupOut { of, param, index } => NodeKind::AdamWGroupOut {
            of: f(of),
            param: *param,
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
        NodeKind::FusedElementwise {
            inputs,
            strides,
            shape,
            expr,
        } => NodeKind::FusedElementwise {
            inputs: inputs.iter().map(|i| f(i)).collect(),
            strides: strides.clone(),
            shape: shape.clone(),
            expr: expr.clone(),
        },
        NodeKind::FusedElementwiseMulti {
            inputs,
            strides,
            shape,
            exprs,
        } => NodeKind::FusedElementwiseMulti {
            inputs: inputs.iter().map(|i| f(i)).collect(),
            strides: strides.clone(),
            shape: shape.clone(),
            exprs: exprs.clone(),
        },
        NodeKind::FusedPick { of, index } => NodeKind::FusedPick {
            of: f(of),
            index: *index,
        },
        NodeKind::FusedReduce {
            inputs,
            strides,
            in_shape,
            expr,
            op,
            dims,
            keepdims,
            shape,
        } => NodeKind::FusedReduce {
            inputs: inputs.iter().map(|i| f(i)).collect(),
            strides: strides.clone(),
            in_shape: in_shape.clone(),
            expr: expr.clone(),
            op: *op,
            dims: dims.clone(),
            keepdims: *keepdims,
            shape: shape.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestExpr;

    impl FusionExpression for TestExpr {
        type ReduceOp = ();

        fn lane_strides(lane: &[usize], out: &[usize]) -> Option<Vec<usize>> {
            if lane == out {
                Some(vec![1; out.len()])
            } else {
                None
            }
        }
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

    type TestNode = Node<TestExpr>;
    type TestNodeKind = NodeKind<TestExpr>;

    #[test]
    fn semantic_nodes_own_authoritative_metadata_and_traversal() {
        let a = TestNode::new(TestNodeKind::Input {
            slot: 0,
            shape: vec![2, 1],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let b = TestNode::new(TestNodeKind::Input {
            slot: 1,
            shape: vec![1, 3],
            dtype: DType::F32,
            device: Device::Cpu,
        })
        .unwrap();
        let add = TestNode::new(TestNodeKind::Add {
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
        let remapped = TestNode::new(remapped).unwrap();
        assert_eq!(remapped.shape, [1, 3]);
    }

    #[test]
    fn leaf_ownership_is_cleared_once_and_invalidates_graph_construction() {
        let slot = Arc::new(LeafSlot::new(TestLeaf {
            shape: vec![4],
            dtype: DType::F32,
            device: Device::Cpu,
        }));
        let leaf = TestNode::new(TestNodeKind::Leaf(slot.clone())).unwrap();
        assert_eq!(leaf.shape, [4]);
        assert!(slot.clear());
        assert!(!slot.clear());
        assert!(TestNode::new(TestNodeKind::Leaf(slot)).is_err());
    }

    #[test]
    fn metadata_validation_rejects_unsupported_device_dtype() {
        let error = TestNode::new(TestNodeKind::Zeros {
            shape: vec![1],
            dtype: DType::F64,
            device: Device::Metal,
        })
        .err()
        .unwrap();
        assert!(error.contains("f64 is not supported on device metal"));
    }
}

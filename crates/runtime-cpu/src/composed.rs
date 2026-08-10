use super::tensor::{source_index, CpuBuffer, CpuDestination, CpuTensorRequirement, Elem, Tensor};
use effect_torch_graph::CrossEntropyReduction;
use effect_torch_runtime::{DType, Layout};
use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossEntropyForwardTopology {
    RowsThenStatus {
        row_passes: usize,
        status_passes: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossEntropyForwardRequirements {
    pub loss: CpuTensorRequirement,
    pub status: CpuTensorRequirement,
    pub nll_scratch: CpuTensorRequirement,
    pub flags_scratch: CpuTensorRequirement,
    pub topology: CrossEntropyForwardTopology,
    pub rows: usize,
    pub classes: usize,
    pub logits_dtype: DType,
    pub target_dtype: DType,
    pub reduction: CrossEntropyReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossEntropyBackwardTopology {
    Rows {
        row_passes: usize,
    },
    CountThenRows {
        count_passes: usize,
        row_passes: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossEntropyBackwardRequirements {
    pub grad: CpuTensorRequirement,
    pub count_status: Option<CpuTensorRequirement>,
    pub topology: CrossEntropyBackwardTopology,
    pub rows: usize,
    pub classes: usize,
    pub logits_dtype: DType,
    pub target_dtype: DType,
    pub reduction: CrossEntropyReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkedHeadCeTopology {
    Forward {
        chunk_len: usize,
        chunks: usize,
        passes_per_chunk: usize,
        final_passes: usize,
    },
    Backward {
        chunk_len: usize,
        chunks: usize,
        passes_per_chunk: usize,
        final_passes: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedHeadCeForwardRequirements {
    pub loss: CpuTensorRequirement,
    pub status: CpuTensorRequirement,
    pub logits_scratch: CpuTensorRequirement,
    pub nll_scratch: CpuTensorRequirement,
    pub flags_scratch: CpuTensorRequirement,
    pub topology: ChunkedHeadCeTopology,
    pub rows: usize,
    pub inner: usize,
    pub vocab: usize,
    pub chunk_len: usize,
    pub dtype: DType,
    pub target_dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedHeadCeBackwardRequirements {
    pub dx: CpuTensorRequirement,
    pub dweight: CpuTensorRequirement,
    pub dbias: CpuTensorRequirement,
    pub status: CpuTensorRequirement,
    pub logits_scratch: CpuTensorRequirement,
    pub grad_logits_scratch: CpuTensorRequirement,
    pub dweight_scratch: CpuTensorRequirement,
    pub dbias_scratch: CpuTensorRequirement,
    pub topology: ChunkedHeadCeTopology,
    pub rows: usize,
    pub inner: usize,
    pub vocab: usize,
    pub chunk_len: usize,
    pub dtype: DType,
    pub target_dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpaForwardTopology {
    OnlineRows {
        score_passes: usize,
        value_passes: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdpaForwardRequirements {
    pub output: CpuTensorRequirement,
    pub logsumexp: CpuTensorRequirement,
    pub topology: SdpaForwardTopology,
    pub batch_heads: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub query_depth: usize,
    pub value_depth: usize,
    pub dtype: DType,
    pub work_dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpaBackwardTopology {
    RowDotThenRecompute {
        row_dot_passes: usize,
        gradient_passes: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdpaBackwardRequirements {
    pub dq: CpuTensorRequirement,
    pub dk: CpuTensorRequirement,
    pub dv: CpuTensorRequirement,
    pub d_vec_scratch: CpuTensorRequirement,
    pub topology: SdpaBackwardTopology,
    pub batch_heads: usize,
    pub query_len: usize,
    pub key_len: usize,
    pub query_depth: usize,
    pub value_depth: usize,
    pub dtype: DType,
    pub work_dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerNormTopology {
    Rows { row_passes: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerNormForwardRequirements {
    pub output: CpuTensorRequirement,
    pub topology: LayerNormTopology,
    pub rows: usize,
    pub normalized_elements: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerNormBackwardRequirements {
    pub dx: CpuTensorRequirement,
    pub dweight: CpuTensorRequirement,
    pub dbias: CpuTensorRequirement,
    pub normalized_scratch: CpuTensorRequirement,
    pub topology: LayerNormTopology,
    pub rows: usize,
    pub normalized_elements: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerTopology {
    Elementwise { passes: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdamWRequirements {
    pub param: CpuTensorRequirement,
    pub first_moment: CpuTensorRequirement,
    pub second_moment: CpuTensorRequirement,
    pub topology: OptimizerTopology,
    pub elements: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SgdRequirements {
    pub param: CpuTensorRequirement,
    pub velocity: CpuTensorRequirement,
    pub topology: OptimizerTopology,
    pub elements: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaTopology {
    ForwardScan { chunk: usize, passes: usize },
    BackwardChunkRecompute { chunk: usize, passes: usize },
    Decode { passes: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdaForwardRequirements {
    pub output: CpuTensorRequirement,
    pub state_next: Option<CpuTensorRequirement>,
    pub state_scratch: Option<CpuTensorRequirement>,
    pub topology: KdaTopology,
    pub batch_heads: usize,
    pub steps: usize,
    pub dk: usize,
    pub dv: usize,
    pub dtype: DType,
    pub work_dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdaBackwardRequirements {
    pub dq: CpuTensorRequirement,
    pub dk: CpuTensorRequirement,
    pub dv: CpuTensorRequirement,
    pub dlog_decay: CpuTensorRequirement,
    pub dbeta: CpuTensorRequirement,
    pub scratch: CpuTensorRequirement,
    pub topology: KdaTopology,
    pub batch_heads: usize,
    pub steps: usize,
    pub key_depth: usize,
    pub value_depth: usize,
    pub chunks: usize,
    pub dtype: DType,
    pub work_dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdaDecodeRequirements {
    pub output: CpuTensorRequirement,
    pub state_next: CpuTensorRequirement,
    pub topology: KdaTopology,
    pub heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub dtype: DType,
    pub work_dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortConvTopology {
    Direct { passes: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortConvForwardRequirements {
    pub output: CpuTensorRequirement,
    pub state_next: Option<CpuTensorRequirement>,
    pub topology: ShortConvTopology,
    pub batch: usize,
    pub steps: usize,
    pub channels: usize,
    pub kernel: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortConvBackwardRequirements {
    pub dx: CpuTensorRequirement,
    pub dweight: CpuTensorRequirement,
    pub topology: ShortConvTopology,
    pub batch: usize,
    pub steps: usize,
    pub channels: usize,
    pub kernel: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortConvBackwardXRequirements {
    pub output: CpuTensorRequirement,
    pub topology: ShortConvTopology,
    pub batch: usize,
    pub steps: usize,
    pub channels: usize,
    pub kernel: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortConvBackwardWRequirements {
    pub output: CpuTensorRequirement,
    pub topology: ShortConvTopology,
    pub batch: usize,
    pub steps: usize,
    pub channels: usize,
    pub kernel: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryTopology {
    Pairs { passes: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotaryRequirements {
    pub output: CpuTensorRequirement,
    pub topology: RotaryTopology,
    pub rows: usize,
    pub steps: usize,
    pub head_dim: usize,
    pub dtype: DType,
}

fn checked_numel(shape: &[usize], operation: &str) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| format!("{operation}: element count overflow"))
}

fn checked_requirement(
    shape: &[usize],
    dtype: DType,
    operation: &str,
) -> Result<CpuTensorRequirement, String> {
    checked_numel(shape, operation)?
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| format!("{operation}: byte size overflow"))?;
    Ok(CpuTensorRequirement::new(shape, dtype))
}

fn work_dtype(dtype: DType) -> DType {
    if dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    }
}

fn validate_float_dtype(dtype: DType, operation: &str) -> Result<(), String> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(format!("{operation}: expected a float dtype, got {dtype}"))
    }
}

fn cross_entropy_geometry(
    logits_shape: &[usize],
    logits_dtype: DType,
    target_shape: &[usize],
    target_dtype: DType,
) -> Result<(usize, usize), String> {
    validate_float_dtype(logits_dtype, "cross_entropy")?;
    if !matches!(target_dtype, DType::I64 | DType::U32) {
        return Err("cross_entropy: target must be i64 or u32".to_string());
    }
    let Some(&classes) = logits_shape.last() else {
        return Err("cross_entropy: logits must have rank >= 1".to_string());
    };
    if classes == 0 {
        return Err("cross_entropy: zero classes are unsupported".to_string());
    }
    let elements = checked_numel(logits_shape, "cross_entropy logits")?;
    let rows = elements / classes;
    if rows == 0 || checked_numel(target_shape, "cross_entropy target")? != rows {
        return Err("cross_entropy: target shape does not match logits rows".to_string());
    }
    Ok((rows, classes))
}

pub fn cross_entropy_forward_requirements(
    logits: &Tensor,
    target: &Tensor,
    reduction: CrossEntropyReduction,
) -> Result<CrossEntropyForwardRequirements, String> {
    let (rows, classes) = cross_entropy_geometry(
        logits.shape(),
        logits.dtype(),
        target.shape(),
        target.dtype(),
    )?;
    Ok(CrossEntropyForwardRequirements {
        loss: checked_requirement(&[], logits.dtype(), "cross_entropy loss")?,
        status: checked_requirement(&[3], DType::F64, "cross_entropy status")?,
        nll_scratch: checked_requirement(&[rows], DType::F64, "cross_entropy nll scratch")?,
        flags_scratch: checked_requirement(&[rows], DType::U32, "cross_entropy flags scratch")?,
        topology: CrossEntropyForwardTopology::RowsThenStatus {
            row_passes: 2,
            status_passes: 1,
        },
        rows,
        classes,
        logits_dtype: logits.dtype(),
        target_dtype: target.dtype(),
        reduction,
    })
}

pub fn cross_entropy_backward_requirements(
    logits: &Tensor,
    target: &Tensor,
    reduction: CrossEntropyReduction,
) -> Result<CrossEntropyBackwardRequirements, String> {
    let (rows, classes) = cross_entropy_geometry(
        logits.shape(),
        logits.dtype(),
        target.shape(),
        target.dtype(),
    )?;
    let (count_status, topology) = match reduction {
        CrossEntropyReduction::Mean => (
            Some(checked_requirement(
                &[1],
                DType::F64,
                "cross_entropy count status",
            )?),
            CrossEntropyBackwardTopology::CountThenRows {
                count_passes: 1,
                row_passes: 2,
            },
        ),
        CrossEntropyReduction::Sum => (None, CrossEntropyBackwardTopology::Rows { row_passes: 2 }),
    };
    Ok(CrossEntropyBackwardRequirements {
        grad: checked_requirement(logits.shape(), logits.dtype(), "cross_entropy gradient")?,
        count_status,
        topology,
        rows,
        classes,
        logits_dtype: logits.dtype(),
        target_dtype: target.dtype(),
        reduction,
    })
}

fn sdpa_geometry(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<(usize, usize, usize, usize, usize), String> {
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
        return Err("sdpa: q/k/v dtypes must match".to_string());
    }
    if !matches!(q.dtype(), DType::F32 | DType::F64) {
        return Err(format!("sdpa: unsupported CPU dtype {}", q.dtype()));
    }
    let r = q.shape().len();
    if r < 2 || k.shape().len() != r || v.shape().len() != r {
        return Err("sdpa: q/k/v must have matching rank >= 2".to_string());
    }
    if q.shape()[..r - 2] != k.shape()[..r - 2] || q.shape()[..r - 2] != v.shape()[..r - 2] {
        return Err("sdpa: q/k/v leading dimensions must match".to_string());
    }
    let (t, s, d, kd, vs, dv) = (
        q.shape()[r - 2],
        k.shape()[r - 2],
        q.shape()[r - 1],
        k.shape()[r - 1],
        v.shape()[r - 2],
        v.shape()[r - 1],
    );
    if d != kd || s != vs || t == 0 || s == 0 || d == 0 || dv == 0 {
        return Err("sdpa: incompatible or zero-sized q/k/v dimensions".to_string());
    }
    let bh = checked_numel(&q.shape()[..r - 2], "sdpa batch/head")?;
    Ok((bh, t, s, d, dv))
}

pub fn sdpa_forward_requirements(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<SdpaForwardRequirements, String> {
    let (batch_heads, query_len, key_len, query_depth, value_depth) = sdpa_geometry(q, k, v)?;
    let r = q.shape().len();
    let mut output_shape = q.shape().to_vec();
    output_shape[r - 1] = value_depth;
    let lse_shape = &q.shape()[..r - 1];
    let work_dtype = work_dtype(q.dtype());
    Ok(SdpaForwardRequirements {
        output: checked_requirement(&output_shape, q.dtype(), "sdpa output")?,
        logsumexp: checked_requirement(lse_shape, work_dtype, "sdpa logsumexp")?,
        topology: SdpaForwardTopology::OnlineRows {
            score_passes: 2,
            value_passes: 1,
        },
        batch_heads,
        query_len,
        key_len,
        query_depth,
        value_depth,
        dtype: q.dtype(),
        work_dtype,
    })
}

pub fn sdpa_backward_requirements(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<SdpaBackwardRequirements, String> {
    let (batch_heads, query_len, key_len, query_depth, value_depth) = sdpa_geometry(q, k, v)?;
    let work_dtype = work_dtype(q.dtype());
    Ok(SdpaBackwardRequirements {
        dq: checked_requirement(q.shape(), q.dtype(), "sdpa dq")?,
        dk: checked_requirement(k.shape(), k.dtype(), "sdpa dk")?,
        dv: checked_requirement(v.shape(), v.dtype(), "sdpa dv")?,
        d_vec_scratch: checked_requirement(
            &q.shape()[..q.shape().len() - 1],
            work_dtype,
            "sdpa row-dot scratch",
        )?,
        topology: SdpaBackwardTopology::RowDotThenRecompute {
            row_dot_passes: 1,
            gradient_passes: 3,
        },
        batch_heads,
        query_len,
        key_len,
        query_depth,
        value_depth,
        dtype: q.dtype(),
        work_dtype,
    })
}

fn layer_norm_geometry(x: &Tensor, weight: &Tensor) -> Result<(usize, usize), String> {
    validate_float_dtype(x.dtype(), "layer_norm")?;
    if weight.dtype() != x.dtype()
        || weight.shape().is_empty()
        || weight.shape().len() > x.shape().len()
        || x.shape()[x.shape().len() - weight.shape().len()..] != *weight.shape()
    {
        return Err("layer_norm: weight must match a non-empty suffix of x".to_string());
    }
    let normalized = checked_numel(weight.shape(), "layer_norm normalized shape")?;
    let elements = checked_numel(x.shape(), "layer_norm input")?;
    if normalized == 0 || elements == 0 {
        return Err("layer_norm: zero-sized dimensions are unsupported".to_string());
    }
    Ok((elements / normalized, normalized))
}

pub fn layer_norm_forward_requirements(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<LayerNormForwardRequirements, String> {
    let (rows, normalized_elements) = layer_norm_geometry(x, weight)?;
    if bias.dtype() != x.dtype() || bias.shape() != weight.shape() {
        return Err("layer_norm: bias must match weight shape and dtype".to_string());
    }
    Ok(LayerNormForwardRequirements {
        output: checked_requirement(x.shape(), x.dtype(), "layer_norm output")?,
        topology: LayerNormTopology::Rows { row_passes: 3 },
        rows,
        normalized_elements,
        dtype: x.dtype(),
    })
}

pub fn layer_norm_backward_requirements(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
) -> Result<LayerNormBackwardRequirements, String> {
    let (rows, normalized_elements) = layer_norm_geometry(x, weight)?;
    if gradient.dtype() != x.dtype() || gradient.shape() != x.shape() {
        return Err("layer_norm: gradient must match x shape and dtype".to_string());
    }
    Ok(LayerNormBackwardRequirements {
        dx: checked_requirement(x.shape(), x.dtype(), "layer_norm dx")?,
        dweight: checked_requirement(weight.shape(), weight.dtype(), "layer_norm dweight")?,
        dbias: checked_requirement(weight.shape(), weight.dtype(), "layer_norm dbias")?,
        normalized_scratch: checked_requirement(
            x.shape(),
            x.dtype(),
            "layer_norm normalized scratch",
        )?,
        topology: LayerNormTopology::Rows { row_passes: 5 },
        rows,
        normalized_elements,
        dtype: x.dtype(),
    })
}

fn rank(t: &Tensor) -> usize {
    t.shape().len()
}

fn unsqueeze_last(t: &Tensor) -> Tensor {
    let mut shape = t.shape().to_vec();
    shape.push(1);
    t.contiguous().view(Layout::contiguous(shape))
}

fn squeeze_last(t: &Tensor) -> Tensor {
    let mut shape = t.shape().to_vec();
    assert_eq!(shape.pop(), Some(1));
    t.contiguous().view(Layout::contiguous(shape))
}

fn transpose_last2(t: &Tensor) -> Tensor {
    let r = rank(t);
    let mut axes: Vec<usize> = (0..r).collect();
    axes.swap(r - 2, r - 1);
    t.view(t.layout.permute(&axes)).contiguous()
}

fn narrow(t: &Tensor, dim: usize, start: usize, len: usize) -> Tensor {
    t.view(t.layout.narrow(dim, start, len)).contiguous()
}

fn full_like(t: &Tensor, value: f64) -> Tensor {
    Tensor::full(t.shape(), value, t.dtype())
}

pub fn softmax_lastdim(x: &Tensor) -> Tensor {
    let r = rank(x);
    let m = x.max(&[r - 1]);
    let e = x.sub(&m).exp();
    let s = e.sum(&[r - 1]);
    e.div(&s)
}

pub fn logsumexp_lastdim(x: &Tensor) -> Tensor {
    let r = rank(x);
    let m = x.max(&[r - 1]);
    let e = x.sub(&m).exp();
    let s = e.sum(&[r - 1]);
    m.add(&s.log())
}

fn causal_allowed(t: usize, s: usize) -> Tensor {
    let off = s.saturating_sub(t) as i64;
    let mut data = Vec::with_capacity(t * s);
    for i in 0..t as i64 {
        for j in 0..s as i64 {
            data.push((j <= i + off) as u8);
        }
    }
    Tensor::from_vec(data, vec![t, s])
}

fn causal_additive_mask(t: usize, s: usize, dtype: DType) -> Tensor {
    let allowed = causal_allowed(t, s);
    let zeros = Tensor::zeros(&[t, s], dtype);
    let neg = Tensor::full(&[t, s], f64::NEG_INFINITY, dtype);
    zeros.where_(&allowed, &neg)
}

fn causal_gate(t: usize, s: usize, dtype: DType) -> Tensor {
    let allowed = causal_allowed(t, s);
    let ones = Tensor::ones(&[t, s], dtype);
    let zeros = Tensor::zeros(&[t, s], dtype);
    ones.where_(&allowed, &zeros)
}

fn sdpa_scores(q: &Tensor, k: &Tensor, scale: f64, causal: bool) -> Tensor {
    let r = rank(q);
    let kt = transpose_last2(k);
    let s = q.matmul(&kt);
    let s = s.mul(&full_like(&s, scale));
    if causal {
        let dims = s.shape();
        let (t, sq) = (dims[r - 2], dims[r - 1]);
        s.add(&causal_additive_mask(t, sq, s.dtype()))
    } else {
        s
    }
}

pub fn sdpa_forward(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: bool) -> Tensor {
    let s = sdpa_scores(q, k, scale, causal);
    let p = softmax_lastdim(&s);
    p.matmul(v)
}

pub fn sdpa_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    scale: f64,
    causal: bool,
) -> (Tensor, Tensor, Tensor) {
    let r = rank(q);
    let s = sdpa_scores(q, k, scale, causal);
    let p = softmax_lastdim(&s);
    let g = g.contiguous();
    let dv = transpose_last2(&p).matmul(&g);
    let dp = g.matmul(&transpose_last2(v));
    let dp_sum = p.mul(&dp).sum(&[r - 1]);
    let mut ds = p.mul(&dp.sub(&dp_sum));
    if causal {
        let dims = ds.shape();
        let (t, sq) = (dims[r - 2], dims[r - 1]);
        ds = ds.mul(&causal_gate(t, sq, ds.dtype()));
    }
    let dq_raw = ds.matmul(&k.contiguous());
    let dq = dq_raw.mul(&full_like(&dq_raw, scale));
    let dk_raw = transpose_last2(&ds).matmul(&q.contiguous());
    let dk = dk_raw.mul(&full_like(&dk_raw, scale));
    (dq, dk, dv)
}

fn sdpa_allowed(query: usize, key: usize, query_len: usize, key_len: usize, causal: bool) -> bool {
    !causal || key <= query + key_len.saturating_sub(query_len)
}

#[allow(clippy::too_many_arguments)]
fn sdpa_forward_into_impl<T: Elem>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
    output: &mut CpuDestination<'_>,
    logsumexp: &mut CpuDestination<'_>,
    batch_heads: usize,
    query_len: usize,
    key_len: usize,
    query_depth: usize,
    value_depth: usize,
) -> Result<(), String> {
    let q_values = tensor_values::<T>(q, "sdpa")?;
    let k_values = tensor_values::<T>(k, "sdpa")?;
    let v_values = tensor_values::<T>(v, "sdpa")?;
    let output_shape = output.shape();
    if output_shape.len() != q.shape().len()
        || output_shape[..output_shape.len() - 1] != q.shape()[..q.shape().len() - 1]
        || output_shape[output_shape.len() - 1] != value_depth
    {
        return Err("sdpa: output shape mismatch".to_string());
    }
    logsumexp.write::<T, _>("sdpa logsumexp", &q.shape()[..q.shape().len() - 1], |lse| {
        output.write_current::<T, _>("sdpa output", |out| {
            for bh in 0..batch_heads {
                for query in 0..query_len {
                    let q_base = (bh * query_len + query) * query_depth;
                    let row = bh * query_len + query;
                    let output_base = row * value_depth;
                    out[output_base..output_base + value_depth].fill(T::default());
                    let mut maximum = f64::NEG_INFINITY;
                    let mut denominator = 0.0f64;
                    for key in 0..key_len {
                        if !sdpa_allowed(query, key, query_len, key_len, causal) {
                            continue;
                        }
                        let k_base = (bh * key_len + key) * query_depth;
                        let mut score = 0.0f64;
                        for depth in 0..query_depth {
                            score += logical_value(q_values, q, q_base + depth).to_f64()
                                * logical_value(k_values, k, k_base + depth).to_f64();
                        }
                        let score = score * scale;
                        let next_maximum = maximum.max(score);
                        let previous_scale = if maximum == f64::NEG_INFINITY {
                            0.0
                        } else {
                            (maximum - next_maximum).exp()
                        };
                        let weight = (score - next_maximum).exp();
                        for value_index in 0..value_depth {
                            let output_index = output_base + value_index;
                            let value = out[output_index].to_f64() * previous_scale
                                + weight
                                    * logical_value(
                                        v_values,
                                        v,
                                        (bh * key_len + key) * value_depth + value_index,
                                    )
                                    .to_f64();
                            out[output_index] = T::from_f64(value);
                        }
                        denominator = denominator * previous_scale + weight;
                        maximum = next_maximum;
                    }
                    for value_index in 0..value_depth {
                        let output_index = output_base + value_index;
                        out[output_index] = T::from_f64(out[output_index].to_f64() / denominator);
                    }
                    lse[row] = T::from_f64(maximum + denominator.ln());
                }
            }
        })
    })?
}

pub fn sdpa_forward_into(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
    output: &mut CpuDestination<'_>,
    logsumexp: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch_heads, query_len, key_len, query_depth, value_depth) = sdpa_geometry(q, k, v)?;
    match q.dtype() {
        DType::F32 => sdpa_forward_into_impl::<f32>(
            q,
            k,
            v,
            scale,
            causal,
            output,
            logsumexp,
            batch_heads,
            query_len,
            key_len,
            query_depth,
            value_depth,
        ),
        DType::F64 => sdpa_forward_into_impl::<f64>(
            q,
            k,
            v,
            scale,
            causal,
            output,
            logsumexp,
            batch_heads,
            query_len,
            key_len,
            query_depth,
            value_depth,
        ),
        _ => unreachable!("sdpa geometry validated the CPU dtype"),
    }
}

#[allow(clippy::too_many_arguments)]
fn sdpa_logsumexp_into_impl<T: Elem>(
    q: &Tensor,
    k: &Tensor,
    scale: f64,
    causal: bool,
    logsumexp: &mut CpuDestination<'_>,
    batch_heads: usize,
    query_len: usize,
    key_len: usize,
    query_depth: usize,
) -> Result<(), String> {
    let q_values = tensor_values::<T>(q, "sdpa logsumexp")?;
    let k_values = tensor_values::<T>(k, "sdpa logsumexp")?;
    logsumexp.write::<T, _>("sdpa logsumexp", &q.shape()[..q.shape().len() - 1], |lse| {
        for bh in 0..batch_heads {
            for query in 0..query_len {
                let q_base = (bh * query_len + query) * query_depth;
                let mut maximum = f64::NEG_INFINITY;
                let mut denominator = 0.0f64;
                for key in 0..key_len {
                    if !sdpa_allowed(query, key, query_len, key_len, causal) {
                        continue;
                    }
                    let k_base = (bh * key_len + key) * query_depth;
                    let mut score = 0.0f64;
                    for depth in 0..query_depth {
                        score += logical_value(q_values, q, q_base + depth).to_f64()
                            * logical_value(k_values, k, k_base + depth).to_f64();
                    }
                    let score = score * scale;
                    let next_maximum = maximum.max(score);
                    let previous_scale = if maximum == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (maximum - next_maximum).exp()
                    };
                    denominator = denominator * previous_scale + (score - next_maximum).exp();
                    maximum = next_maximum;
                }
                lse[bh * query_len + query] = T::from_f64(maximum + denominator.ln());
            }
        }
    })?;
    Ok(())
}

pub fn sdpa_logsumexp_into(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
    logsumexp: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch_heads, query_len, key_len, query_depth, _) = sdpa_geometry(q, k, v)?;
    match q.dtype() {
        DType::F32 => sdpa_logsumexp_into_impl::<f32>(
            q,
            k,
            scale,
            causal,
            logsumexp,
            batch_heads,
            query_len,
            key_len,
            query_depth,
        ),
        DType::F64 => sdpa_logsumexp_into_impl::<f64>(
            q,
            k,
            scale,
            causal,
            logsumexp,
            batch_heads,
            query_len,
            key_len,
            query_depth,
        ),
        _ => unreachable!("sdpa geometry validated the CPU dtype"),
    }
}

#[allow(clippy::too_many_arguments)]
fn sdpa_backward_into_impl<T: Elem>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &Tensor,
    logsumexp: &Tensor,
    gradient: &Tensor,
    scale: f64,
    causal: bool,
    dq: &mut CpuDestination<'_>,
    dk: &mut CpuDestination<'_>,
    dv: &mut CpuDestination<'_>,
    d_vec_scratch: &mut CpuDestination<'_>,
    batch_heads: usize,
    query_len: usize,
    key_len: usize,
    query_depth: usize,
    value_depth: usize,
) -> Result<(), String> {
    let q_values = tensor_values::<T>(q, "sdpa backward")?;
    let k_values = tensor_values::<T>(k, "sdpa backward")?;
    let v_values = tensor_values::<T>(v, "sdpa backward")?;
    let o_values = tensor_values::<T>(output, "sdpa backward")?;
    let lse_values = tensor_values::<T>(logsumexp, "sdpa backward")?;
    let g_values = tensor_values::<T>(gradient, "sdpa backward")?;
    d_vec_scratch.write::<T, _>(
        "sdpa row-dot scratch",
        &q.shape()[..q.shape().len() - 1],
        |d_vec| {
            for row in 0..batch_heads * query_len {
                let mut value = 0.0f64;
                for depth in 0..value_depth {
                    value += logical_value(o_values, output, row * value_depth + depth).to_f64()
                        * logical_value(g_values, gradient, row * value_depth + depth).to_f64();
                }
                d_vec[row] = T::from_f64(value);
            }
            dq.write::<T, _>("sdpa dq", q.shape(), |dq_values| {
                dk.write::<T, _>("sdpa dk", k.shape(), |dk_values| {
                    dv.write::<T, _>("sdpa dv", v.shape(), |dv_values| {
                        dq_values.fill(T::default());
                        dk_values.fill(T::default());
                        dv_values.fill(T::default());
                        for bh in 0..batch_heads {
                            for query in 0..query_len {
                                let q_base = (bh * query_len + query) * query_depth;
                                let row = bh * query_len + query;
                                for key in 0..key_len {
                                    if !sdpa_allowed(query, key, query_len, key_len, causal) {
                                        continue;
                                    }
                                    let k_base = (bh * key_len + key) * query_depth;
                                    let v_base = (bh * key_len + key) * value_depth;
                                    let mut score = 0.0f64;
                                    for depth in 0..query_depth {
                                        score += logical_value(q_values, q, q_base + depth)
                                            .to_f64()
                                            * logical_value(k_values, k, k_base + depth).to_f64();
                                    }
                                    let probability = (score * scale
                                        - logical_value(lse_values, logsumexp, row).to_f64())
                                    .exp();
                                    let mut dp = 0.0f64;
                                    for value_index in 0..value_depth {
                                        dp += logical_value(
                                            g_values,
                                            gradient,
                                            row * value_depth + value_index,
                                        )
                                        .to_f64()
                                            * logical_value(v_values, v, v_base + value_index)
                                                .to_f64();
                                    }
                                    let ds = probability * (dp - d_vec[row].to_f64()) * scale;
                                    for depth in 0..query_depth {
                                        dq_values[q_base + depth] = T::from_f64(
                                            dq_values[q_base + depth].to_f64()
                                                + ds * logical_value(k_values, k, k_base + depth)
                                                    .to_f64(),
                                        );
                                        dk_values[k_base + depth] = T::from_f64(
                                            dk_values[k_base + depth].to_f64()
                                                + ds * logical_value(q_values, q, q_base + depth)
                                                    .to_f64(),
                                        );
                                    }
                                    for value_index in 0..value_depth {
                                        dv_values[v_base + value_index] = T::from_f64(
                                            dv_values[v_base + value_index].to_f64()
                                                + probability
                                                    * logical_value(
                                                        g_values,
                                                        gradient,
                                                        row * value_depth + value_index,
                                                    )
                                                    .to_f64(),
                                        );
                                    }
                                }
                            }
                        }
                    })
                })
            })
        },
    )????;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sdpa_backward_into(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &Tensor,
    logsumexp: &Tensor,
    gradient: &Tensor,
    scale: f64,
    causal: bool,
    dq: &mut CpuDestination<'_>,
    dk: &mut CpuDestination<'_>,
    dv: &mut CpuDestination<'_>,
    d_vec_scratch: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch_heads, query_len, key_len, query_depth, value_depth) = sdpa_geometry(q, k, v)?;
    let expected_output_elements = batch_heads
        .checked_mul(query_len)
        .and_then(|value| value.checked_mul(value_depth));
    let rank = q.shape().len();
    if output.dtype() != q.dtype()
        || gradient.dtype() != q.dtype()
        || output.numel() != expected_output_elements.unwrap_or(usize::MAX)
        || gradient.shape() != output.shape()
        || output.shape().len() != rank
        || output.shape()[..rank - 1] != q.shape()[..rank - 1]
        || output.shape()[rank - 1] != value_depth
        || logsumexp.dtype() != work_dtype(q.dtype())
        || logsumexp.shape() != &q.shape()[..rank - 1]
    {
        return Err("sdpa backward: output, logsumexp, or gradient mismatch".to_string());
    }
    match q.dtype() {
        DType::F32 => sdpa_backward_into_impl::<f32>(
            q,
            k,
            v,
            output,
            logsumexp,
            gradient,
            scale,
            causal,
            dq,
            dk,
            dv,
            d_vec_scratch,
            batch_heads,
            query_len,
            key_len,
            query_depth,
            value_depth,
        ),
        DType::F64 => sdpa_backward_into_impl::<f64>(
            q,
            k,
            v,
            output,
            logsumexp,
            gradient,
            scale,
            causal,
            dq,
            dk,
            dv,
            d_vec_scratch,
            batch_heads,
            query_len,
            key_len,
            query_depth,
            value_depth,
        ),
        _ => unreachable!("sdpa geometry validated the CPU dtype"),
    }
}

pub fn layer_norm_forward(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
    let r = rank(x);
    let k = weight.shape().len();
    let dims: Vec<usize> = (r - k..r).collect();
    let mean = x.mean(&dims);
    let centered = x.sub(&mean);
    let var = centered.mul(&centered).mean(&dims);
    let inv = var
        .add(&Tensor::full(var.shape(), eps, var.dtype()))
        .sqrt()
        .powf(-1.0);
    centered.mul(&inv).mul(weight).add(bias)
}

pub fn layer_norm_backward(
    x: &Tensor,
    weight: &Tensor,
    g: &Tensor,
    eps: f64,
) -> (Tensor, Tensor, Tensor) {
    let r = rank(x);
    let k = weight.shape().len();
    let dims: Vec<usize> = (r - k..r).collect();
    let reduce_dims: Vec<usize> = (0..r - k).collect();
    let mean = x.mean(&dims);
    let centered = x.sub(&mean);
    let var = centered.mul(&centered).mean(&dims);
    let rstd = var
        .add(&Tensor::full(var.shape(), eps, var.dtype()))
        .sqrt()
        .powf(-1.0);
    let xh = centered.mul(&rstd);
    // dx = (dyw − mean(dyw) − x̂·mean(dyw·x̂)) · rstd
    let dyw = g.mul(weight);
    let m1 = dyw.mean(&dims);
    let m2 = dyw.mul(&xh).mean(&dims);
    let dx = dyw.sub(&m1).sub(&xh.mul(&m2)).mul(&rstd);
    let dw = g.mul(&xh).sum(&reduce_dims).squeeze_dims(&reduce_dims);
    let db = g.sum(&reduce_dims).squeeze_dims(&reduce_dims);
    (dx, dw, db)
}

fn layer_norm_forward_into_impl<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
    output: &mut CpuDestination<'_>,
    rows: usize,
    normalized_elements: usize,
) -> Result<(), String> {
    let x_values = tensor_values::<T>(x, "layer_norm")?;
    let weight_values = tensor_values::<T>(weight, "layer_norm")?;
    let bias_values = tensor_values::<T>(bias, "layer_norm")?;
    output.write::<T, _>("layer_norm output", x.shape(), |out| {
        for row in 0..rows {
            let base = row * normalized_elements;
            let mut mean = 0.0f64;
            for index in 0..normalized_elements {
                mean += logical_value(x_values, x, base + index).to_f64();
            }
            mean /= normalized_elements as f64;
            let mut variance = 0.0f64;
            for index in 0..normalized_elements {
                let centered = logical_value(x_values, x, base + index).to_f64() - mean;
                variance += centered * centered;
            }
            variance /= normalized_elements as f64;
            let rstd = 1.0 / (variance + eps).sqrt();
            for index in 0..normalized_elements {
                out[base + index] = T::from_f64(
                    (logical_value(x_values, x, base + index).to_f64() - mean)
                        * rstd
                        * logical_value(weight_values, weight, index).to_f64()
                        + logical_value(bias_values, bias, index).to_f64(),
                );
            }
        }
    })
}

pub fn layer_norm_forward_into(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
    output: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (rows, normalized_elements) = layer_norm_geometry(x, weight)?;
    if bias.dtype() != x.dtype() || bias.shape() != weight.shape() {
        return Err("layer_norm: bias must match weight shape and dtype".to_string());
    }
    match x.dtype() {
        DType::F32 => layer_norm_forward_into_impl::<f32>(
            x,
            weight,
            bias,
            eps,
            output,
            rows,
            normalized_elements,
        ),
        DType::F64 => layer_norm_forward_into_impl::<f64>(
            x,
            weight,
            bias,
            eps,
            output,
            rows,
            normalized_elements,
        ),
        DType::F16 => layer_norm_forward_into_impl::<f16>(
            x,
            weight,
            bias,
            eps,
            output,
            rows,
            normalized_elements,
        ),
        DType::BF16 => layer_norm_forward_into_impl::<bf16>(
            x,
            weight,
            bias,
            eps,
            output,
            rows,
            normalized_elements,
        ),
        _ => unreachable!("layer_norm geometry validated float dtype"),
    }
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_backward_into_impl<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
    eps: f64,
    dx: &mut CpuDestination<'_>,
    dweight: &mut CpuDestination<'_>,
    dbias: &mut CpuDestination<'_>,
    normalized_scratch: &mut CpuDestination<'_>,
    rows: usize,
    normalized_elements: usize,
) -> Result<(), String> {
    let x_values = tensor_values::<T>(x, "layer_norm backward")?;
    let weight_values = tensor_values::<T>(weight, "layer_norm backward")?;
    let gradient_values = tensor_values::<T>(gradient, "layer_norm backward")?;
    normalized_scratch.write::<T, _>(
        "layer_norm normalized scratch",
        x.shape(),
        |normalized| {
            dx.write::<T, _>("layer_norm dx", x.shape(), |dx_values| {
                for row in 0..rows {
                    let base = row * normalized_elements;
                    let mut mean = 0.0f64;
                    for index in 0..normalized_elements {
                        mean += logical_value(x_values, x, base + index).to_f64();
                    }
                    mean /= normalized_elements as f64;
                    let mut variance = 0.0f64;
                    for index in 0..normalized_elements {
                        let centered = logical_value(x_values, x, base + index).to_f64() - mean;
                        variance += centered * centered;
                    }
                    variance /= normalized_elements as f64;
                    let rstd = 1.0 / (variance + eps).sqrt();
                    let mut mean_dyw = 0.0f64;
                    let mut mean_dyw_xhat = 0.0f64;
                    for index in 0..normalized_elements {
                        let xhat =
                            (logical_value(x_values, x, base + index).to_f64() - mean) * rstd;
                        normalized[base + index] = T::from_f64(xhat);
                        let dyw = logical_value(gradient_values, gradient, base + index).to_f64()
                            * logical_value(weight_values, weight, index).to_f64();
                        mean_dyw += dyw;
                        mean_dyw_xhat += dyw * xhat;
                    }
                    mean_dyw /= normalized_elements as f64;
                    mean_dyw_xhat /= normalized_elements as f64;
                    for index in 0..normalized_elements {
                        let xhat = normalized[base + index].to_f64();
                        let dyw = logical_value(gradient_values, gradient, base + index).to_f64()
                            * logical_value(weight_values, weight, index).to_f64();
                        dx_values[base + index] =
                            T::from_f64((dyw - mean_dyw - xhat * mean_dyw_xhat) * rstd);
                    }
                }
            })?;
            dweight.write::<T, _>("layer_norm dweight", weight.shape(), |dw| {
                dbias.write::<T, _>("layer_norm dbias", weight.shape(), |db| {
                    for index in 0..normalized_elements {
                        let mut weight_sum = 0.0f64;
                        let mut bias_sum = 0.0f64;
                        for row in 0..rows {
                            let offset = row * normalized_elements + index;
                            let value = logical_value(gradient_values, gradient, offset).to_f64();
                            weight_sum += value * normalized[offset].to_f64();
                            bias_sum += value;
                        }
                        dw[index] = T::from_f64(weight_sum);
                        db[index] = T::from_f64(bias_sum);
                    }
                })
            })
        },
    )???;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward_into(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
    eps: f64,
    dx: &mut CpuDestination<'_>,
    dweight: &mut CpuDestination<'_>,
    dbias: &mut CpuDestination<'_>,
    normalized_scratch: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (rows, normalized_elements) = layer_norm_geometry(x, weight)?;
    if gradient.dtype() != x.dtype() || gradient.shape() != x.shape() {
        return Err("layer_norm: gradient must match x shape and dtype".to_string());
    }
    match x.dtype() {
        DType::F32 => layer_norm_backward_into_impl::<f32>(
            x,
            weight,
            gradient,
            eps,
            dx,
            dweight,
            dbias,
            normalized_scratch,
            rows,
            normalized_elements,
        ),
        DType::F64 => layer_norm_backward_into_impl::<f64>(
            x,
            weight,
            gradient,
            eps,
            dx,
            dweight,
            dbias,
            normalized_scratch,
            rows,
            normalized_elements,
        ),
        DType::F16 => layer_norm_backward_into_impl::<f16>(
            x,
            weight,
            gradient,
            eps,
            dx,
            dweight,
            dbias,
            normalized_scratch,
            rows,
            normalized_elements,
        ),
        DType::BF16 => layer_norm_backward_into_impl::<bf16>(
            x,
            weight,
            gradient,
            eps,
            dx,
            dweight,
            dbias,
            normalized_scratch,
            rows,
            normalized_elements,
        ),
        _ => unreachable!("layer_norm geometry validated float dtype"),
    }
}

fn ce_ignored_mask(target: &Tensor, ignore_index: i64) -> Tensor {
    match target.dtype() {
        DType::I64 => {
            let ii = Tensor::full(target.shape(), ignore_index as f64, DType::I64);
            target.eq(&ii)
        }
        DType::U32 => {
            if ignore_index < 0 || ignore_index > u32::MAX as i64 {
                Tensor::zeros(target.shape(), DType::U8)
            } else {
                let ii = Tensor::full(target.shape(), ignore_index as f64, DType::U32);
                target.eq(&ii)
            }
        }
        _ => panic!("cross_entropy: target must be i64 or u32"),
    }
}

fn ce_active_count(ignored: &Tensor, total: usize) -> f64 {
    let s = ignored.cast(DType::F64).sum(&all_dims(ignored));
    total as f64 - scalar(&s)
}

fn all_dims(t: &Tensor) -> Vec<usize> {
    (0..rank(t)).collect()
}

fn scalar(t: &Tensor) -> f64 {
    let c = t.cast(DType::F64).contiguous();
    let CpuBuffer::F64(v) = &c.buffer else {
        unreachable!()
    };
    v[0]
}

fn ce_check_labels(target: &Tensor, ignored: &Tensor, classes: usize) -> Result<(), String> {
    let invalid = match target.dtype() {
        DType::I64 => {
            let lo = target.lt(&Tensor::full(target.shape(), 0.0, DType::I64));
            let hi = target.ge(&Tensor::full(target.shape(), classes as f64, DType::I64));
            lo.maximum(&hi)
        }
        DType::U32 => target.ge(&Tensor::full(target.shape(), classes as f64, DType::U32)),
        _ => unreachable!(),
    };
    let active = ignored.eq(&Tensor::zeros(ignored.shape(), DType::U8));
    let invalid_active = invalid.mul(&active).cast(DType::F64);
    if scalar(&invalid_active.sum(&all_dims(&invalid_active))) > 0.0 {
        return Err(format!(
            "cross_entropy: target out of range [0, {classes}) at an active position"
        ));
    }
    Ok(())
}

fn target_to_ids(target: &Tensor) -> Tensor {
    match target.dtype() {
        DType::I64 => target.clone(),
        _ => target.cast(DType::I64),
    }
}

enum TargetValues<'a> {
    I64(&'a [i64]),
    U32(&'a [u32]),
}

impl TargetValues<'_> {
    fn get(&self, target: &Tensor, index: usize) -> i64 {
        let index = source_index(&target.layout, index);
        match self {
            Self::I64(values) => values[index],
            Self::U32(values) => values[index] as i64,
        }
    }
}

fn target_values(target: &Tensor) -> Result<TargetValues<'_>, String> {
    match &target.buffer {
        CpuBuffer::I64(values) => Ok(TargetValues::I64(values)),
        CpuBuffer::U32(values) => Ok(TargetValues::U32(values)),
        _ => Err("cross_entropy: target must be i64 or u32".to_string()),
    }
}

fn target_is_ignored(target: &Tensor, value: i64, ignore_index: i64) -> bool {
    match target.dtype() {
        DType::I64 => value == ignore_index,
        DType::U32 => (0..=u32::MAX as i64).contains(&ignore_index) && value == ignore_index,
        _ => false,
    }
}

fn target_summary(
    target: &Tensor,
    ignore_index: i64,
    classes: usize,
) -> Result<(usize, usize), String> {
    let values = target_values(target)?;
    let mut active = 0usize;
    let mut invalid = 0usize;
    for index in 0..target.numel() {
        let value = values.get(target, index);
        if target_is_ignored(target, value, ignore_index) {
            continue;
        }
        active += 1;
        invalid += (value < 0 || value as usize >= classes) as usize;
    }
    Ok((active, invalid))
}

fn tensor_values<'a, T: Elem>(tensor: &'a Tensor, operation: &str) -> Result<&'a [T], String> {
    T::storage_of(&tensor.buffer)
        .map(|storage| storage.as_slice())
        .ok_or_else(|| format!("{operation}: unexpected tensor dtype {}", tensor.dtype()))
}

fn logical_value<T: Elem>(values: &[T], tensor: &Tensor, index: usize) -> T {
    values[source_index(&tensor.layout, index)]
}

#[allow(clippy::too_many_arguments)]
fn cross_entropy_forward_into_impl<T: Elem>(
    logits: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    reduction: CrossEntropyReduction,
    loss: &mut CpuDestination<'_>,
    status: &mut CpuDestination<'_>,
    nll_scratch: &mut CpuDestination<'_>,
    flags_scratch: &mut CpuDestination<'_>,
    rows: usize,
    classes: usize,
) -> Result<(), String> {
    let logits_values = tensor_values::<T>(logits, "cross_entropy")?;
    let targets = target_values(target)?;
    nll_scratch.write::<f64, _>("cross_entropy nll scratch", &[rows], |nll| {
        flags_scratch.write::<u32, _>("cross_entropy flags scratch", &[rows], |flags| {
            for row in 0..rows {
                let target_value = targets.get(target, row);
                let ignored = target_is_ignored(target, target_value, ignore_index);
                let invalid = !ignored && (target_value < 0 || target_value as usize >= classes);
                flags[row] = (ignored as u32) | ((invalid as u32) << 1);
                if ignored || invalid {
                    nll[row] = 0.0;
                    continue;
                }
                let base = row * classes;
                let mut maximum = f64::NEG_INFINITY;
                for class in 0..classes {
                    maximum =
                        maximum.max(logical_value(logits_values, logits, base + class).to_f64());
                }
                let mut sum = 0.0f64;
                for class in 0..classes {
                    sum += (logical_value(logits_values, logits, base + class).to_f64() - maximum)
                        .exp();
                }
                let picked =
                    logical_value(logits_values, logits, base + target_value as usize).to_f64();
                nll[row] = maximum + sum.ln() - picked;
            }
            let mut total = 0.0f64;
            let mut active = 0usize;
            let mut invalid = 0usize;
            for row in 0..rows {
                total += nll[row];
                active += (flags[row] & 1 == 0) as usize;
                invalid += (flags[row] & 2 != 0) as usize;
            }
            let result = if reduction == CrossEntropyReduction::Mean && active != 0 {
                total / active as f64
            } else {
                total
            };
            status.write::<f64, _>("cross_entropy status", &[3], |values| {
                values.copy_from_slice(&[result, active as f64, invalid as f64]);
            })?;
            loss.write::<T, _>("cross_entropy loss", &[], |output| {
                output[0] = T::from_f64(result);
            })?;
            if active == 0 && reduction == CrossEntropyReduction::Mean {
                return Err(
                    "cross_entropy: no active targets (all positions are ignored)".to_string(),
                );
            }
            if invalid != 0 {
                return Err(format!(
                    "cross_entropy: target out of range [0, {classes}) at an active position"
                ));
            }
            Ok(())
        })?
    })?
}

#[allow(clippy::too_many_arguments)]
pub fn cross_entropy_forward_into(
    logits: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    reduction: CrossEntropyReduction,
    loss: &mut CpuDestination<'_>,
    status: &mut CpuDestination<'_>,
    nll_scratch: &mut CpuDestination<'_>,
    flags_scratch: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (rows, classes) = cross_entropy_geometry(
        logits.shape(),
        logits.dtype(),
        target.shape(),
        target.dtype(),
    )?;
    match logits.dtype() {
        DType::F32 => cross_entropy_forward_into_impl::<f32>(
            logits,
            target,
            ignore_index,
            reduction,
            loss,
            status,
            nll_scratch,
            flags_scratch,
            rows,
            classes,
        ),
        DType::F64 => cross_entropy_forward_into_impl::<f64>(
            logits,
            target,
            ignore_index,
            reduction,
            loss,
            status,
            nll_scratch,
            flags_scratch,
            rows,
            classes,
        ),
        DType::F16 => cross_entropy_forward_into_impl::<f16>(
            logits,
            target,
            ignore_index,
            reduction,
            loss,
            status,
            nll_scratch,
            flags_scratch,
            rows,
            classes,
        ),
        DType::BF16 => cross_entropy_forward_into_impl::<bf16>(
            logits,
            target,
            ignore_index,
            reduction,
            loss,
            status,
            nll_scratch,
            flags_scratch,
            rows,
            classes,
        ),
        _ => unreachable!("cross_entropy geometry validated float logits"),
    }
}

#[allow(clippy::too_many_arguments)]
fn cross_entropy_backward_into_impl<T: Elem>(
    logits: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    reduction: CrossEntropyReduction,
    grad: &mut CpuDestination<'_>,
    count_status: Option<&mut CpuDestination<'_>>,
    rows: usize,
    classes: usize,
) -> Result<(), String> {
    let (active, invalid) = target_summary(target, ignore_index, classes)?;
    if active == 0 && reduction == CrossEntropyReduction::Mean {
        return Err("cross_entropy: no active targets (all positions are ignored)".to_string());
    }
    if invalid != 0 {
        return Err(format!(
            "cross_entropy: target out of range [0, {classes}) at an active position"
        ));
    }
    match (reduction, count_status) {
        (CrossEntropyReduction::Mean, Some(status)) => {
            status.write::<f64, _>("cross_entropy count status", &[1], |output| {
                output[0] = active as f64;
            })?;
        }
        (CrossEntropyReduction::Mean, None) => {
            return Err("cross_entropy: mean backward requires count status".to_string());
        }
        (CrossEntropyReduction::Sum, Some(_)) => {
            return Err("cross_entropy: sum backward has no count status".to_string());
        }
        (CrossEntropyReduction::Sum, None) => {}
    }
    let logits_values = tensor_values::<T>(logits, "cross_entropy backward")?;
    let targets = target_values(target)?;
    let scale = if reduction == CrossEntropyReduction::Mean {
        1.0 / active as f64
    } else {
        1.0
    };
    grad.write::<T, _>("cross_entropy gradient", logits.shape(), |output| {
        for row in 0..rows {
            let target_value = targets.get(target, row);
            if target_is_ignored(target, target_value, ignore_index) {
                output[row * classes..(row + 1) * classes].fill(T::default());
                continue;
            }
            let base = row * classes;
            let mut maximum = f64::NEG_INFINITY;
            for class in 0..classes {
                maximum = maximum.max(logical_value(logits_values, logits, base + class).to_f64());
            }
            let mut sum = 0.0f64;
            for class in 0..classes {
                sum +=
                    (logical_value(logits_values, logits, base + class).to_f64() - maximum).exp();
            }
            for class in 0..classes {
                let probability =
                    (logical_value(logits_values, logits, base + class).to_f64() - maximum).exp()
                        / sum;
                output[base + class] = T::from_f64(
                    (probability - (class == target_value as usize) as u8 as f64) * scale,
                );
            }
        }
    })
}

pub fn cross_entropy_backward_into(
    logits: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    reduction: CrossEntropyReduction,
    grad: &mut CpuDestination<'_>,
    count_status: Option<&mut CpuDestination<'_>>,
) -> Result<(), String> {
    let (rows, classes) = cross_entropy_geometry(
        logits.shape(),
        logits.dtype(),
        target.shape(),
        target.dtype(),
    )?;
    match logits.dtype() {
        DType::F32 => cross_entropy_backward_into_impl::<f32>(
            logits,
            target,
            ignore_index,
            reduction,
            grad,
            count_status,
            rows,
            classes,
        ),
        DType::F64 => cross_entropy_backward_into_impl::<f64>(
            logits,
            target,
            ignore_index,
            reduction,
            grad,
            count_status,
            rows,
            classes,
        ),
        DType::F16 => cross_entropy_backward_into_impl::<f16>(
            logits,
            target,
            ignore_index,
            reduction,
            grad,
            count_status,
            rows,
            classes,
        ),
        DType::BF16 => cross_entropy_backward_into_impl::<bf16>(
            logits,
            target,
            ignore_index,
            reduction,
            grad,
            count_status,
            rows,
            classes,
        ),
        _ => unreachable!("cross_entropy geometry validated float logits"),
    }
}

pub fn cross_entropy_forward(
    logits: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    reduction: CrossEntropyReduction,
) -> Result<Tensor, String> {
    let r = rank(logits);
    let classes = logits.shape()[r - 1];
    let ignored = ce_ignored_mask(target, ignore_index);
    let count = ce_active_count(&ignored, target.numel());
    if count == 0.0 && reduction == CrossEntropyReduction::Mean {
        return Err("cross_entropy: no active targets (all positions are ignored)".to_string());
    }
    ce_check_labels(target, &ignored, classes)?;
    let lse = logsumexp_lastdim(logits);
    // ignored positions may hold out-of-range values; gather at 0 and mask
    let zero_ids = Tensor::zeros(target.shape(), DType::I64);
    let safe_target = zero_ids.where_(&ignored, &target_to_ids(target));
    let picked = logits.gather(r - 1, &unsqueeze_last(&safe_target));
    let picked = squeeze_last(&picked);
    let per_position = squeeze_last(&lse).sub(&picked);
    let masked =
        Tensor::zeros(per_position.shape(), per_position.dtype()).where_(&ignored, &per_position);
    let total = masked.sum(&all_dims(&masked));
    match reduction {
        CrossEntropyReduction::Mean => {
            let scale = Tensor::full(total.shape(), 1.0 / count, total.dtype());
            Ok(total.mul(&scale))
        }
        CrossEntropyReduction::Sum => Ok(total),
    }
}

pub fn cross_entropy_backward(
    logits: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    reduction: CrossEntropyReduction,
) -> Result<Tensor, String> {
    let r = rank(logits);
    let ignored = ce_ignored_mask(target, ignore_index);
    let count = ce_active_count(&ignored, target.numel());
    if count == 0.0 && reduction == CrossEntropyReduction::Mean {
        return Err("cross_entropy: no active targets (all positions are ignored)".to_string());
    }
    let p = softmax_lastdim(logits);
    let zero_ids = Tensor::zeros(target.shape(), DType::I64);
    let safe_target = zero_ids.where_(&ignored, &target_to_ids(target));
    let ids = unsqueeze_last(&safe_target);
    let neg_ones = Tensor::full(ids.shape(), -1.0, logits.dtype());
    // p[target] -= 1 at every position
    let p = p.scatter_add(r - 1, &ids, &neg_ones);
    let keep = ignored.eq(&Tensor::zeros(ignored.shape(), DType::U8));
    let keep = unsqueeze_last(&keep);
    let masked = p.where_(&keep, &Tensor::zeros(p.shape(), p.dtype()));
    match reduction {
        CrossEntropyReduction::Mean => {
            let scale = Tensor::full(masked.shape(), 1.0 / count, masked.dtype());
            Ok(masked.mul(&scale))
        }
        CrossEntropyReduction::Sum => Ok(masked),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_step(
    p: &Tensor,
    g: &Tensor,
    m: &Tensor,
    v: &Tensor,
    lr: &Tensor,
    c1: &Tensor,
    c2: &Tensor,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> (Tensor, Tensor, Tensor) {
    let fl = |t: &Tensor, x: f64| full_like(t, x);
    let next_m = m.mul(&fl(m, beta1)).add(&g.mul(&fl(g, 1.0 - beta1)));
    let gg = g.mul(g);
    let next_v = v.mul(&fl(v, beta2)).add(&gg.mul(&fl(&gg, 1.0 - beta2)));
    let m_hat = next_m.div(c1);
    let v_hat = next_v.div(c2);
    let adjusted = m_hat.div(&v_hat.sqrt().add(&fl(&v_hat, eps))).mul(lr);
    let next_p = if weight_decay == 0.0 {
        p.sub(&adjusted)
    } else {
        let decay = p.mul(&lr.mul(&fl(lr, weight_decay)));
        p.sub(&decay).sub(&adjusted)
    };
    (next_p, next_m, next_v)
}

pub fn adamw_step_requirements(
    param: &Tensor,
    gradient: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
) -> Result<AdamWRequirements, String> {
    validate_float_dtype(param.dtype(), "adamw")?;
    if gradient.dtype() != param.dtype()
        || first_moment.dtype() != param.dtype()
        || second_moment.dtype() != param.dtype()
        || gradient.shape() != param.shape()
        || first_moment.shape() != param.shape()
        || second_moment.shape() != param.shape()
    {
        return Err("adamw: parameter, gradient, and moments must match".to_string());
    }
    let output = checked_requirement(param.shape(), param.dtype(), "adamw output")?;
    Ok(AdamWRequirements {
        param: output.clone(),
        first_moment: output.clone(),
        second_moment: output,
        topology: OptimizerTopology::Elementwise { passes: 1 },
        elements: param.numel(),
        dtype: param.dtype(),
    })
}

fn broadcast_value<T: Elem>(
    values: &[T],
    tensor: &Tensor,
    output_shape: &[usize],
    output_index: usize,
) -> T {
    let extra = output_shape.len() - tensor.shape().len();
    let mut source = tensor.layout.offset();
    let mut remainder = output_index;
    for dimension in (0..output_shape.len()).rev() {
        let width = output_shape[dimension].max(1);
        let coordinate = remainder % width;
        remainder /= width;
        if dimension >= extra {
            let input_dimension = dimension - extra;
            if tensor.shape()[input_dimension] != 1 {
                source += coordinate * tensor.layout.strides()[input_dimension];
            }
        }
    }
    values[source]
}

fn validate_broadcast_to(tensor: &Tensor, shape: &[usize], operation: &str) -> Result<(), String> {
    if tensor.shape().len() > shape.len() {
        return Err(format!("{operation}: input cannot broadcast to output"));
    }
    let extra = shape.len() - tensor.shape().len();
    for dimension in 0..tensor.shape().len() {
        let source = tensor.shape()[dimension];
        let destination = shape[extra + dimension];
        if source != 1 && source != destination {
            return Err(format!("{operation}: input cannot broadcast to output"));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn adamw_step_into_impl<T: Elem>(
    param: &Tensor,
    gradient: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
    lr: &Tensor,
    c1: &Tensor,
    c2: &Tensor,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    next_param: &mut CpuDestination<'_>,
    next_first_moment: &mut CpuDestination<'_>,
    next_second_moment: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let p = tensor_values::<T>(param, "adamw")?;
    let g = tensor_values::<T>(gradient, "adamw")?;
    let m = tensor_values::<T>(first_moment, "adamw")?;
    let v = tensor_values::<T>(second_moment, "adamw")?;
    let lr_values = tensor_values::<T>(lr, "adamw")?;
    let c1_values = tensor_values::<T>(c1, "adamw")?;
    let c2_values = tensor_values::<T>(c2, "adamw")?;
    next_param.write::<T, _>("adamw parameter", param.shape(), |p_out| {
        next_first_moment.write::<T, _>("adamw first moment", param.shape(), |m_out| {
            next_second_moment.write::<T, _>("adamw second moment", param.shape(), |v_out| {
                for index in 0..param.numel() {
                    let p_value = logical_value(p, param, index).to_f64();
                    let g_value = logical_value(g, gradient, index).to_f64();
                    let m_value = beta1 * logical_value(m, first_moment, index).to_f64()
                        + (1.0 - beta1) * g_value;
                    let v_value = beta2 * logical_value(v, second_moment, index).to_f64()
                        + (1.0 - beta2) * g_value * g_value;
                    let lr_value = broadcast_value(lr_values, lr, param.shape(), index).to_f64();
                    let adjusted = (m_value
                        / broadcast_value(c1_values, c1, param.shape(), index).to_f64())
                        / ((v_value
                            / broadcast_value(c2_values, c2, param.shape(), index).to_f64())
                        .sqrt()
                            + eps)
                        * lr_value;
                    p_out[index] =
                        T::from_f64(p_value - p_value * lr_value * weight_decay - adjusted);
                    m_out[index] = T::from_f64(m_value);
                    v_out[index] = T::from_f64(v_value);
                }
            })
        })
    })???;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_step_into(
    param: &Tensor,
    gradient: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
    lr: &Tensor,
    c1: &Tensor,
    c2: &Tensor,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    next_param: &mut CpuDestination<'_>,
    next_first_moment: &mut CpuDestination<'_>,
    next_second_moment: &mut CpuDestination<'_>,
) -> Result<(), String> {
    if !param.dtype().is_float()
        || gradient.dtype() != param.dtype()
        || first_moment.dtype() != param.dtype()
        || second_moment.dtype() != param.dtype()
        || gradient.shape() != param.shape()
        || first_moment.shape() != param.shape()
        || second_moment.shape() != param.shape()
    {
        return Err("adamw: parameter, gradient, and moments must match".to_string());
    }
    for tensor in [lr, c1, c2] {
        if tensor.dtype() != param.dtype() {
            return Err("adamw: scalar inputs must match parameter dtype".to_string());
        }
        validate_broadcast_to(tensor, param.shape(), "adamw")?;
    }
    match param.dtype() {
        DType::F32 => adamw_step_into_impl::<f32>(
            param,
            gradient,
            first_moment,
            second_moment,
            lr,
            c1,
            c2,
            beta1,
            beta2,
            eps,
            weight_decay,
            next_param,
            next_first_moment,
            next_second_moment,
        ),
        DType::F64 => adamw_step_into_impl::<f64>(
            param,
            gradient,
            first_moment,
            second_moment,
            lr,
            c1,
            c2,
            beta1,
            beta2,
            eps,
            weight_decay,
            next_param,
            next_first_moment,
            next_second_moment,
        ),
        DType::F16 => adamw_step_into_impl::<f16>(
            param,
            gradient,
            first_moment,
            second_moment,
            lr,
            c1,
            c2,
            beta1,
            beta2,
            eps,
            weight_decay,
            next_param,
            next_first_moment,
            next_second_moment,
        ),
        DType::BF16 => adamw_step_into_impl::<bf16>(
            param,
            gradient,
            first_moment,
            second_moment,
            lr,
            c1,
            c2,
            beta1,
            beta2,
            eps,
            weight_decay,
            next_param,
            next_first_moment,
            next_second_moment,
        ),
        _ => unreachable!("adamw requirements validated float dtype"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sgd_step(
    p: &Tensor,
    g: &Tensor,
    v: &Tensor,
    lr: &Tensor,
    first: &Tensor,
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    weight_decay: f64,
) -> (Tensor, Tensor) {
    let fl = |t: &Tensor, x: f64| full_like(t, x);
    let g = if weight_decay == 0.0 {
        g.clone()
    } else {
        g.add(&p.mul(&fl(p, weight_decay)))
    };
    // next_v = first ? g : momentum * v + (1 - dampening) * g, as
    // arithmetic selection (velocity is zeros on the first step).
    let continued = v
        .mul(&fl(v, momentum))
        .add(&g.mul(&fl(&g, 1.0 - dampening)));
    let not_first = first.mul(&fl(first, -1.0)).add(&fl(first, 1.0));
    let next_v = first.mul(&g).add(&not_first.mul(&continued));
    let used = if nesterov {
        g.add(&next_v.mul(&fl(&next_v, momentum)))
    } else {
        next_v.clone()
    };
    let next_p = p.sub(&used.mul(lr));
    (next_p, next_v)
}

pub fn sgd_step_requirements(
    param: &Tensor,
    gradient: &Tensor,
    velocity: &Tensor,
) -> Result<SgdRequirements, String> {
    validate_float_dtype(param.dtype(), "sgd")?;
    if gradient.dtype() != param.dtype()
        || velocity.dtype() != param.dtype()
        || gradient.shape() != param.shape()
        || velocity.shape() != param.shape()
    {
        return Err("sgd: parameter, gradient, and velocity must match".to_string());
    }
    let output = checked_requirement(param.shape(), param.dtype(), "sgd output")?;
    Ok(SgdRequirements {
        param: output.clone(),
        velocity: output,
        topology: OptimizerTopology::Elementwise { passes: 1 },
        elements: param.numel(),
        dtype: param.dtype(),
    })
}

#[allow(clippy::too_many_arguments)]
fn sgd_step_into_impl<T: Elem>(
    param: &Tensor,
    gradient: &Tensor,
    velocity: &Tensor,
    lr: &Tensor,
    first: &Tensor,
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    weight_decay: f64,
    next_param: &mut CpuDestination<'_>,
    next_velocity: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let p = tensor_values::<T>(param, "sgd")?;
    let g = tensor_values::<T>(gradient, "sgd")?;
    let velocity_values = tensor_values::<T>(velocity, "sgd")?;
    let lr_values = tensor_values::<T>(lr, "sgd")?;
    let first_values = tensor_values::<T>(first, "sgd")?;
    next_param.write::<T, _>("sgd parameter", param.shape(), |p_out| {
        next_velocity.write::<T, _>("sgd velocity", param.shape(), |velocity_out| {
            for index in 0..param.numel() {
                let p_value = logical_value(p, param, index).to_f64();
                let gradient_value =
                    logical_value(g, gradient, index).to_f64() + weight_decay * p_value;
                let first_value =
                    broadcast_value(first_values, first, param.shape(), index).to_f64();
                let continued = momentum * logical_value(velocity_values, velocity, index).to_f64()
                    + (1.0 - dampening) * gradient_value;
                let next_velocity_value =
                    first_value * gradient_value + (1.0 - first_value) * continued;
                let used = if nesterov {
                    gradient_value + momentum * next_velocity_value
                } else {
                    next_velocity_value
                };
                let lr_value = broadcast_value(lr_values, lr, param.shape(), index).to_f64();
                p_out[index] = T::from_f64(p_value - lr_value * used);
                velocity_out[index] = T::from_f64(next_velocity_value);
            }
        })
    })??;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sgd_step_into(
    param: &Tensor,
    gradient: &Tensor,
    velocity: &Tensor,
    lr: &Tensor,
    first: &Tensor,
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    weight_decay: f64,
    next_param: &mut CpuDestination<'_>,
    next_velocity: &mut CpuDestination<'_>,
) -> Result<(), String> {
    if !param.dtype().is_float()
        || gradient.dtype() != param.dtype()
        || velocity.dtype() != param.dtype()
        || gradient.shape() != param.shape()
        || velocity.shape() != param.shape()
    {
        return Err("sgd: parameter, gradient, and velocity must match".to_string());
    }
    for tensor in [lr, first] {
        if tensor.dtype() != param.dtype() {
            return Err("sgd: scalar inputs must match parameter dtype".to_string());
        }
        validate_broadcast_to(tensor, param.shape(), "sgd")?;
    }
    match param.dtype() {
        DType::F32 => sgd_step_into_impl::<f32>(
            param,
            gradient,
            velocity,
            lr,
            first,
            momentum,
            dampening,
            nesterov,
            weight_decay,
            next_param,
            next_velocity,
        ),
        DType::F64 => sgd_step_into_impl::<f64>(
            param,
            gradient,
            velocity,
            lr,
            first,
            momentum,
            dampening,
            nesterov,
            weight_decay,
            next_param,
            next_velocity,
        ),
        DType::F16 => sgd_step_into_impl::<f16>(
            param,
            gradient,
            velocity,
            lr,
            first,
            momentum,
            dampening,
            nesterov,
            weight_decay,
            next_param,
            next_velocity,
        ),
        DType::BF16 => sgd_step_into_impl::<bf16>(
            param,
            gradient,
            velocity,
            lr,
            first,
            momentum,
            dampening,
            nesterov,
            weight_decay,
            next_param,
            next_velocity,
        ),
        _ => unreachable!("sgd requirements validated float dtype"),
    }
}

pub fn rotary_forward(
    x: &Tensor,
    offsets: &[usize],
    theta: f64,
    sign: f64,
) -> Result<Tensor, String> {
    let dims = x.shape();
    let r = dims.len();
    let (t, d) = (dims[r - 2], dims[r - 1]);
    let batch = dims[0];
    if offsets.len() != 1 && offsets.len() != batch {
        return Err(format!(
            "rotary embedding: {} offsets for batch {batch}",
            offsets.len()
        ));
    }
    let half = d / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 / d as f64) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, vec![1, half]);
    let positions: Vec<f32> = if offsets.len() == 1 {
        (0..t).map(|p| (offsets[0] + p) as f32).collect()
    } else {
        offsets
            .iter()
            .flat_map(|base| (0..t).map(move |p| (*base + p) as f32))
            .collect()
    };
    let rows = if offsets.len() == 1 { 1 } else { batch };
    let positions = Tensor::from_vec(positions, vec![rows * t, 1]);
    let angles =
        positions
            .matmul(&inv_freq)
            .mul(&Tensor::full(&[rows * t, half], sign, DType::F32));
    let mut table_shape = vec![1usize; r - 2];
    if offsets.len() != 1 {
        table_shape[0] = batch;
    }
    table_shape.extend([t, half]);
    let cos = angles
        .cos()
        .contiguous()
        .view(Layout::contiguous(table_shape.clone()));
    let sin = angles
        .sin()
        .contiguous()
        .view(Layout::contiguous(table_shape));
    let first = narrow(x, r - 1, 0, half);
    let second = narrow(x, r - 1, half, half);
    let out_first = first.mul(&cos).sub(&second.mul(&sin));
    let out_second = second.mul(&cos).add(&first.mul(&sin));
    Ok(Tensor::cat(&[&out_first, &out_second], r - 1).contiguous())
}

pub fn rotary_requirements(x: &Tensor) -> Result<RotaryRequirements, String> {
    validate_float_dtype(x.dtype(), "rotary")?;
    let rank = x.shape().len();
    if rank < 2 || x.shape()[rank - 1] % 2 != 0 {
        return Err("rotary: expected rank >= 2 and an even head dimension".to_string());
    }
    let rows = checked_numel(&x.shape()[..rank - 2], "rotary rows")?;
    Ok(RotaryRequirements {
        output: checked_requirement(x.shape(), x.dtype(), "rotary output")?,
        topology: RotaryTopology::Pairs { passes: 1 },
        rows,
        steps: x.shape()[rank - 2],
        head_dim: x.shape()[rank - 1],
        dtype: x.dtype(),
    })
}

pub fn rotary_forward_requirements(x: &Tensor) -> Result<RotaryRequirements, String> {
    rotary_requirements(x)
}

fn rotary_forward_into_impl<T: Elem>(
    x: &Tensor,
    offsets: &[usize],
    theta: f64,
    sign: f64,
    output: &mut CpuDestination<'_>,
    rows: usize,
    steps: usize,
    head_dim: usize,
) -> Result<(), String> {
    let values = tensor_values::<T>(x, "rotary")?;
    let batch = x.shape()[0];
    let rows_per_batch = rows / batch;
    let half = head_dim / 2;
    output.write::<T, _>("rotary output", x.shape(), |out| {
        for row in 0..rows {
            let batch_index = row / rows_per_batch;
            let offset = if offsets.len() == 1 {
                offsets[0]
            } else {
                offsets[batch_index]
            };
            for step in 0..steps {
                let base = (row * steps + step) * head_dim;
                for index in 0..half {
                    let angle = sign
                        * (offset + step) as f64
                        * theta.powf(-2.0 * index as f64 / head_dim as f64);
                    let (sin, cos) = angle.sin_cos();
                    let first = logical_value(values, x, base + index).to_f64();
                    let second = logical_value(values, x, base + half + index).to_f64();
                    out[base + index] = T::from_f64(first * cos - second * sin);
                    out[base + half + index] = T::from_f64(second * cos + first * sin);
                }
            }
        }
    })
}

pub fn rotary_forward_into(
    x: &Tensor,
    offsets: &[usize],
    theta: f64,
    sign: f64,
    output: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let rank = x.shape().len();
    if rank < 2 || x.shape()[rank - 1] % 2 != 0 || !x.dtype().is_float() {
        return Err("rotary: expected a float rank >= 2 tensor with even head dimension".into());
    }
    let batch = x.shape()[0];
    if offsets.len() != 1 && offsets.len() != batch {
        return Err(format!(
            "rotary embedding: {} offsets for batch {batch}",
            offsets.len()
        ));
    }
    let rows = x.shape()[..rank - 2]
        .iter()
        .try_fold(1usize, |count, value| count.checked_mul(*value))
        .ok_or_else(|| "rotary: row count overflow".to_string())?;
    let steps = x.shape()[rank - 2];
    let head_dim = x.shape()[rank - 1];
    match x.dtype() {
        DType::F32 => {
            rotary_forward_into_impl::<f32>(x, offsets, theta, sign, output, rows, steps, head_dim)
        }
        DType::F64 => {
            rotary_forward_into_impl::<f64>(x, offsets, theta, sign, output, rows, steps, head_dim)
        }
        DType::F16 => {
            rotary_forward_into_impl::<f16>(x, offsets, theta, sign, output, rows, steps, head_dim)
        }
        DType::BF16 => {
            rotary_forward_into_impl::<bf16>(x, offsets, theta, sign, output, rows, steps, head_dim)
        }
        _ => unreachable!("rotary validated float dtype"),
    }
}

// --- Chunked head cross-entropy (RFC 0016 phase 2, semantic form) ---

fn head_ce_chunk_len(rows: usize, vocab: usize, chunk_size: usize) -> usize {
    let elements = rows.saturating_mul(vocab);
    let chunks = (elements / chunk_size).clamp(2, 64).min(rows);
    rows.div_ceil(chunks.max(1))
}

fn chunked_head_geometry(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    chunk_size: usize,
) -> Result<(usize, usize, usize, usize), String> {
    if chunk_size == 0 {
        return Err("chunked head CE: chunk size must be non-zero".to_string());
    }
    if !matches!(x.dtype(), DType::F32 | DType::F64)
        || weight.dtype() != x.dtype()
        || bias.dtype() != x.dtype()
        || weight.shape().len() != 2
        || x.shape().is_empty()
    {
        return Err("chunked head CE: unsupported dtype or input rank".to_string());
    }
    if !matches!(target.dtype(), DType::I64 | DType::U32) {
        return Err("cross_entropy: target must be i64 or u32".to_string());
    }
    let rank = x.shape().len();
    let rows = checked_numel(&x.shape()[..rank - 1], "chunked head CE rows")?;
    let (inner, vocab) = (weight.shape()[0], weight.shape()[1]);
    if rows == 0
        || vocab == 0
        || x.shape()[rank - 1] != inner
        || bias.numel() != vocab
        || target.numel() != rows
    {
        return Err("chunked head CE: inconsistent input geometry".to_string());
    }
    Ok((
        rows,
        inner,
        vocab,
        head_ce_chunk_len(rows, vocab, chunk_size),
    ))
}

pub fn chunked_head_ce_forward_requirements(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    chunk_size: usize,
) -> Result<ChunkedHeadCeForwardRequirements, String> {
    let (rows, inner, vocab, chunk_len) =
        chunked_head_geometry(x, weight, bias, target, chunk_size)?;
    let chunks = rows.div_ceil(chunk_len);
    Ok(ChunkedHeadCeForwardRequirements {
        loss: checked_requirement(&[], x.dtype(), "chunked head CE loss")?,
        status: checked_requirement(&[3], DType::F64, "chunked head CE status")?,
        logits_scratch: checked_requirement(
            &[chunk_len, vocab],
            x.dtype(),
            "chunked head CE logits scratch",
        )?,
        nll_scratch: checked_requirement(&[chunk_len], DType::F64, "chunked head CE nll scratch")?,
        flags_scratch: checked_requirement(
            &[chunk_len],
            DType::U32,
            "chunked head CE flags scratch",
        )?,
        topology: ChunkedHeadCeTopology::Forward {
            chunk_len,
            chunks,
            passes_per_chunk: 3,
            final_passes: 1,
        },
        rows,
        inner,
        vocab,
        chunk_len,
        dtype: x.dtype(),
        target_dtype: target.dtype(),
    })
}

pub fn chunked_head_ce_backward_requirements(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    chunk_size: usize,
) -> Result<ChunkedHeadCeBackwardRequirements, String> {
    let (rows, inner, vocab, chunk_len) =
        chunked_head_geometry(x, weight, bias, target, chunk_size)?;
    let chunks = rows.div_ceil(chunk_len);
    Ok(ChunkedHeadCeBackwardRequirements {
        dx: checked_requirement(x.shape(), x.dtype(), "chunked head CE dx")?,
        dweight: checked_requirement(weight.shape(), weight.dtype(), "chunked head CE dweight")?,
        dbias: checked_requirement(bias.shape(), bias.dtype(), "chunked head CE dbias")?,
        status: checked_requirement(&[3], DType::F64, "chunked head CE status")?,
        logits_scratch: checked_requirement(
            &[chunk_len, vocab],
            x.dtype(),
            "chunked head CE logits scratch",
        )?,
        grad_logits_scratch: checked_requirement(
            &[chunk_len, vocab],
            DType::F32,
            "chunked head CE grad-logits scratch",
        )?,
        dweight_scratch: checked_requirement(
            &[inner, vocab],
            DType::F32,
            "chunked head CE dweight scratch",
        )?,
        dbias_scratch: checked_requirement(&[vocab], DType::F32, "chunked head CE dbias scratch")?,
        topology: ChunkedHeadCeTopology::Backward {
            chunk_len,
            chunks,
            passes_per_chunk: 5,
            final_passes: 1,
        },
        rows,
        inner,
        vocab,
        chunk_len,
        dtype: x.dtype(),
        target_dtype: target.dtype(),
    })
}

#[allow(clippy::too_many_arguments)]
fn chunked_head_ce_forward_into_impl<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    loss: &mut CpuDestination<'_>,
    status: &mut CpuDestination<'_>,
    logits_scratch: &mut CpuDestination<'_>,
    nll_scratch: &mut CpuDestination<'_>,
    flags_scratch: &mut CpuDestination<'_>,
    rows: usize,
    inner: usize,
    vocab: usize,
    chunk_len: usize,
) -> Result<(), String> {
    let x_values = tensor_values::<T>(x, "chunked head CE")?;
    let weight_values = tensor_values::<T>(weight, "chunked head CE")?;
    let bias_values = tensor_values::<T>(bias, "chunked head CE")?;
    let targets = target_values(target)?;
    logits_scratch.write::<T, _>(
        "chunked head CE logits scratch",
        &[chunk_len, vocab],
        |logits| {
            nll_scratch.write::<f64, _>(
                "chunked head CE nll scratch",
                &[chunk_len],
                |nll| {
                    flags_scratch.write::<u32, _>(
                        "chunked head CE flags scratch",
                        &[chunk_len],
                        |flags| {
                            let mut total = 0.0f64;
                            let mut active = 0usize;
                            let mut invalid = 0usize;
                            let mut offset = 0usize;
                            while offset < rows {
                                let length = chunk_len.min(rows - offset);
                                for local_row in 0..length {
                                    let row = offset + local_row;
                                    for class in 0..vocab {
                                        let mut value = 0.0f64;
                                        for index in 0..inner {
                                            value += logical_value(
                                                x_values,
                                                x,
                                                row * inner + index,
                                            )
                                            .to_f64()
                                                * logical_value(
                                                    weight_values,
                                                    weight,
                                                    index * vocab + class,
                                                )
                                                .to_f64();
                                        }
                                        value += logical_value(bias_values, bias, class).to_f64();
                                        logits[local_row * vocab + class] = T::from_f64(value);
                                    }
                                    let target_value = targets.get(target, row);
                                    let ignored =
                                        target_is_ignored(target, target_value, ignore_index);
                                    let invalid_row = !ignored
                                        && (target_value < 0 || target_value as usize >= vocab);
                                    flags[local_row] =
                                        ignored as u32 | ((invalid_row as u32) << 1);
                                    if ignored || invalid_row {
                                        nll[local_row] = 0.0;
                                    } else {
                                        let base = local_row * vocab;
                                        let mut maximum = f64::NEG_INFINITY;
                                        for class in 0..vocab {
                                            maximum =
                                                maximum.max(logits[base + class].to_f64());
                                        }
                                        let mut sum = 0.0f64;
                                        for class in 0..vocab {
                                            sum += (logits[base + class].to_f64() - maximum).exp();
                                        }
                                        nll[local_row] = maximum + sum.ln()
                                            - logits[base + target_value as usize].to_f64();
                                    }
                                    total += nll[local_row];
                                    active += (!ignored) as usize;
                                    invalid += invalid_row as usize;
                                }
                                offset += length;
                            }
                            let result = if active == 0 {
                                0.0
                            } else {
                                total / active as f64
                            };
                            status.write::<f64, _>(
                                "chunked head CE status",
                                &[3],
                                |values| {
                                    values.copy_from_slice(&[
                                        result,
                                        active as f64,
                                        invalid as f64,
                                    ]);
                                },
                            )?;
                            loss.write::<T, _>("chunked head CE loss", &[], |output| {
                                output[0] = T::from_f64(result);
                            })?;
                            if active == 0 {
                                return Err(
                                    "cross_entropy: no active targets (all positions are ignored)"
                                        .to_string(),
                                );
                            }
                            if invalid != 0 {
                                return Err(format!(
                                    "cross_entropy: target out of range [0, {vocab}) at an active position"
                                ));
                            }
                            Ok(())
                        },
                    )?
                },
            )?
        },
    )?
}

#[allow(clippy::too_many_arguments)]
pub fn chunked_head_ce_forward_into(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    chunk_size: usize,
    loss: &mut CpuDestination<'_>,
    status: &mut CpuDestination<'_>,
    logits_scratch: &mut CpuDestination<'_>,
    nll_scratch: &mut CpuDestination<'_>,
    flags_scratch: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (rows, inner, vocab, chunk_len) =
        chunked_head_geometry(x, weight, bias, target, chunk_size)?;
    match x.dtype() {
        DType::F32 => chunked_head_ce_forward_into_impl::<f32>(
            x,
            weight,
            bias,
            target,
            ignore_index,
            loss,
            status,
            logits_scratch,
            nll_scratch,
            flags_scratch,
            rows,
            inner,
            vocab,
            chunk_len,
        ),
        DType::F64 => chunked_head_ce_forward_into_impl::<f64>(
            x,
            weight,
            bias,
            target,
            ignore_index,
            loss,
            status,
            logits_scratch,
            nll_scratch,
            flags_scratch,
            rows,
            inner,
            vocab,
            chunk_len,
        ),
        _ => unreachable!("chunked head geometry validated CPU matmul dtype"),
    }
}

#[allow(clippy::too_many_arguments)]
fn chunked_head_ce_backward_into_impl<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    gradient: &Tensor,
    ignore_index: i64,
    dx: &mut CpuDestination<'_>,
    dweight: &mut CpuDestination<'_>,
    dbias: &mut CpuDestination<'_>,
    status: &mut CpuDestination<'_>,
    logits_scratch: &mut CpuDestination<'_>,
    grad_logits_scratch: &mut CpuDestination<'_>,
    dweight_scratch: &mut CpuDestination<'_>,
    dbias_scratch: &mut CpuDestination<'_>,
    rows: usize,
    inner: usize,
    vocab: usize,
    chunk_len: usize,
) -> Result<(), String> {
    let (active, invalid) = target_summary(target, ignore_index, vocab)?;
    status.write::<f64, _>("chunked head CE status", &[3], |values| {
        values.copy_from_slice(&[0.0, active as f64, invalid as f64]);
    })?;
    if active == 0 {
        return Err("cross_entropy: no active targets (all positions are ignored)".to_string());
    }
    if invalid != 0 {
        return Err(format!(
            "cross_entropy: target out of range [0, {vocab}) at an active position"
        ));
    }
    let x_values = tensor_values::<T>(x, "chunked head CE backward")?;
    let weight_values = tensor_values::<T>(weight, "chunked head CE backward")?;
    let bias_values = tensor_values::<T>(bias, "chunked head CE backward")?;
    let gradient_values = tensor_values::<T>(gradient, "chunked head CE backward")?;
    let targets = target_values(target)?;
    let scalar_gradient = logical_value(gradient_values, gradient, 0).to_f64() as f32;
    let scale = scalar_gradient / active as f32;
    logits_scratch.write::<T, _>(
        "chunked head CE logits scratch",
        &[chunk_len, vocab],
        |logits| {
            grad_logits_scratch.write::<f32, _>(
                "chunked head CE grad-logits scratch",
                &[chunk_len, vocab],
                |grad_logits| {
                    dweight_scratch.write::<f32, _>(
                        "chunked head CE dweight scratch",
                        &[inner, vocab],
                        |dweight_accumulator| {
                            dbias_scratch.write::<f32, _>(
                                "chunked head CE dbias scratch",
                                &[vocab],
                                |dbias_accumulator| {
                                    dx.write::<T, _>("chunked head CE dx", x.shape(), |dx_values| {
                                        dweight.write::<T, _>(
                                            "chunked head CE dweight",
                                            weight.shape(),
                                            |dweight_values| {
                                                dbias.write::<T, _>(
                                                    "chunked head CE dbias",
                                                    bias.shape(),
                                                    |dbias_values| {
                                                        dweight_accumulator.fill(0.0);
                                                        dbias_accumulator.fill(0.0);
                                                        let mut offset = 0usize;
                                                        while offset < rows {
                                                            let length =
                                                                chunk_len.min(rows - offset);
                                                            for local_row in 0..length {
                                                                let row = offset + local_row;
                                                                for class in 0..vocab {
                                                                    let mut value = 0.0f64;
                                                                    for index in 0..inner {
                                                                        value += logical_value(
                                                                            x_values,
                                                                            x,
                                                                            row * inner + index,
                                                                        )
                                                                        .to_f64()
                                                                            * logical_value(
                                                                                weight_values,
                                                                                weight,
                                                                                index * vocab
                                                                                    + class,
                                                                            )
                                                                            .to_f64();
                                                                    }
                                                                    value += logical_value(
                                                                        bias_values,
                                                                        bias,
                                                                        class,
                                                                    )
                                                                    .to_f64();
                                                                    logits[local_row * vocab
                                                                        + class] =
                                                                        T::from_f64(value);
                                                                }
                                                                let target_value =
                                                                    targets.get(target, row);
                                                                let ignored = target_is_ignored(
                                                                    target,
                                                                    target_value,
                                                                    ignore_index,
                                                                );
                                                                let base = local_row * vocab;
                                                                if ignored {
                                                                    grad_logits[base..base + vocab]
                                                                        .fill(0.0);
                                                                } else {
                                                                    let mut maximum =
                                                                        f64::NEG_INFINITY;
                                                                    for class in 0..vocab {
                                                                        maximum = maximum.max(
                                                                            logits[base + class]
                                                                                .to_f64(),
                                                                        );
                                                                    }
                                                                    let mut sum = 0.0f64;
                                                                    for class in 0..vocab {
                                                                        sum += (logits
                                                                            [base + class]
                                                                            .to_f64()
                                                                            - maximum)
                                                                            .exp();
                                                                    }
                                                                    for class in 0..vocab {
                                                                        let probability = (logits
                                                                            [base + class]
                                                                            .to_f64()
                                                                            - maximum)
                                                                            .exp()
                                                                            / sum;
                                                                        grad_logits[base + class] =
                                                                            ((probability
                                                                                - (class
                                                                                    == target_value
                                                                                        as usize)
                                                                                    as u8
                                                                                    as f64)
                                                                                as f32)
                                                                                * scale;
                                                                    }
                                                                }
                                                                for index in 0..inner {
                                                                    let mut value = 0.0f32;
                                                                    for class in 0..vocab {
                                                                        value += grad_logits
                                                                            [base + class]
                                                                            * logical_value(
                                                                                weight_values,
                                                                                weight,
                                                                                index * vocab
                                                                                    + class,
                                                                            )
                                                                            .to_f64()
                                                                                as f32;
                                                                        dweight_accumulator[index
                                                                            * vocab
                                                                            + class] +=
                                                                            logical_value(
                                                                                x_values,
                                                                                x,
                                                                                row * inner + index,
                                                                            )
                                                                            .to_f64()
                                                                                as f32
                                                                                * grad_logits
                                                                                    [base + class];
                                                                    }
                                                                    dx_values
                                                                        [row * inner + index] =
                                                                        T::from_f64(value as f64);
                                                                }
                                                                for class in 0..vocab {
                                                                    dbias_accumulator[class] +=
                                                                        grad_logits[base + class];
                                                                }
                                                            }
                                                            offset += length;
                                                        }
                                                        for index in 0..inner * vocab {
                                                            dweight_values[index] = T::from_f64(
                                                                dweight_accumulator[index] as f64,
                                                            );
                                                        }
                                                        for class in 0..vocab {
                                                            dbias_values[class] = T::from_f64(
                                                                dbias_accumulator[class] as f64,
                                                            );
                                                        }
                                                    },
                                                )
                                            },
                                        )
                                    })
                                },
                            )
                        },
                    )
                },
            )
        },
    )???????;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn chunked_head_ce_backward_into(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    gradient: &Tensor,
    ignore_index: i64,
    chunk_size: usize,
    dx: &mut CpuDestination<'_>,
    dweight: &mut CpuDestination<'_>,
    dbias: &mut CpuDestination<'_>,
    status: &mut CpuDestination<'_>,
    logits_scratch: &mut CpuDestination<'_>,
    grad_logits_scratch: &mut CpuDestination<'_>,
    dweight_scratch: &mut CpuDestination<'_>,
    dbias_scratch: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (rows, inner, vocab, chunk_len) =
        chunked_head_geometry(x, weight, bias, target, chunk_size)?;
    if gradient.dtype() != x.dtype() || gradient.numel() != 1 {
        return Err("chunked head CE: gradient must be a scalar of the input dtype".to_string());
    }
    match x.dtype() {
        DType::F32 => chunked_head_ce_backward_into_impl::<f32>(
            x,
            weight,
            bias,
            target,
            gradient,
            ignore_index,
            dx,
            dweight,
            dbias,
            status,
            logits_scratch,
            grad_logits_scratch,
            dweight_scratch,
            dbias_scratch,
            rows,
            inner,
            vocab,
            chunk_len,
        ),
        DType::F64 => chunked_head_ce_backward_into_impl::<f64>(
            x,
            weight,
            bias,
            target,
            gradient,
            ignore_index,
            dx,
            dweight,
            dbias,
            status,
            logits_scratch,
            grad_logits_scratch,
            dweight_scratch,
            dbias_scratch,
            rows,
            inner,
            vocab,
            chunk_len,
        ),
        _ => unreachable!("chunked head geometry validated CPU matmul dtype"),
    }
}

// Mean cross-entropy of Linear(x, weight, bias) against target,
// evaluated one row-chunk at a time so the [rows, vocab] logits never
// materialize whole.
pub fn chunked_head_ce_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    ignore_index: i64,
    chunk_size: usize,
) -> Result<Tensor, String> {
    let dims = x.shape().to_vec();
    let r = rank(x);
    let (inner, vocab) = (weight.shape()[0], weight.shape()[1]);
    let rows: usize = dims[..r - 1].iter().product();
    let x2 = x.contiguous().view(Layout::contiguous(vec![rows, inner]));
    let t1 = target.contiguous().view(Layout::contiguous(vec![rows]));
    // Match Mean semantics exactly: zero-active error before label
    // checks, in the plain path's order.
    let ignored = ce_ignored_mask(&t1, ignore_index);
    let count = ce_active_count(&ignored, rows);
    if count == 0.0 {
        return Err("cross_entropy: no active targets (all positions are ignored)".to_string());
    }
    ce_check_labels(&t1, &ignored, vocab)?;
    let chunk_len = head_ce_chunk_len(rows, vocab, chunk_size);
    let mut total = Tensor::zeros(&[], DType::F32);
    let mut off = 0;
    while off < rows {
        let end = (off + chunk_len).min(rows);
        let x_c = narrow(&x2, 0, off, end - off);
        let t_c = narrow(&t1, 0, off, end - off);
        let logits = x_c.matmul(weight).add(bias);
        let nll = cross_entropy_forward(&logits, &t_c, ignore_index, CrossEntropyReduction::Sum)?;
        total = total.add(&nll.cast(DType::F32));
        off = end;
    }
    let mean = total.div(&Tensor::full(&[], count, DType::F32));
    Ok(if mean.dtype() == x.dtype() {
        mean
    } else {
        mean.cast(x.dtype())
    })
}

// Closed-form adjoint: recomputes each chunk's logits and grad-logits
// in a transient workspace and accumulates (dx, dw, db); grad-logits
// never outlive their chunk.
pub fn chunked_head_ce_backward(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    target: &Tensor,
    g: &Tensor,
    ignore_index: i64,
    chunk_size: usize,
) -> Result<(Tensor, Tensor, Tensor), String> {
    let dims = x.shape().to_vec();
    let r = rank(x);
    let (inner, vocab) = (weight.shape()[0], weight.shape()[1]);
    let rows: usize = dims[..r - 1].iter().product();
    let x2 = x.contiguous().view(Layout::contiguous(vec![rows, inner]));
    let t1 = target.contiguous().view(Layout::contiguous(vec![rows]));
    let ignored = ce_ignored_mask(&t1, ignore_index);
    let count = ce_active_count(&ignored, rows);
    let scale = Tensor::full(&[], scalar(&g.cast(DType::F64)) / count, DType::F32);
    let chunk_len = head_ce_chunk_len(rows, vocab, chunk_size);
    let w32t = transpose_last2(&weight.cast(DType::F32));
    let mut dx_chunks: Vec<Tensor> = Vec::new();
    let mut dw = Tensor::zeros(&[inner, vocab], DType::F32);
    let mut db = Tensor::zeros(&[vocab], DType::F32);
    let mut off = 0;
    while off < rows {
        let end = (off + chunk_len).min(rows);
        let x_c = narrow(&x2, 0, off, end - off);
        let t_c = narrow(&t1, 0, off, end - off);
        let logits = x_c.matmul(weight).add(bias);
        let gb = cross_entropy_backward(&logits, &t_c, ignore_index, CrossEntropyReduction::Sum)?;
        let gb32 = gb.cast(DType::F32).mul(&scale);
        dx_chunks.push(gb32.matmul(&w32t));
        dw = dw.add(&transpose_last2(&x_c.cast(DType::F32)).matmul(&gb32));
        db = db.add(&gb32.sum(&[0]));
        off = end;
    }
    let dx = Tensor::cat(&dx_chunks.iter().collect::<Vec<_>>(), 0)
        .contiguous()
        .view(Layout::contiguous(dims))
        .cast(x.dtype());
    let dw = dw.cast(weight.dtype());
    let db = db
        .cast(bias.dtype())
        .contiguous()
        .view(Layout::contiguous(bias.shape().to_vec()));
    Ok((dx, dw, db))
}

// --- Kimi Delta Attention (RFC 0018) ---

fn kda_geometry(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
) -> Result<(usize, usize, usize, usize), String> {
    let rank = q.shape().len();
    if rank < 2
        || k.shape().len() != rank
        || v.shape().len() != rank
        || log_decay.shape().len() != rank
        || beta.shape().len() != rank
        || q.dtype() != k.dtype()
        || q.dtype() != v.dtype()
        || q.dtype() != log_decay.dtype()
        || q.dtype() != beta.dtype()
        || !q.dtype().is_float()
    {
        return Err("kda: inputs must have matching float dtype and rank >= 2".to_string());
    }
    let (steps, dk, dv) = (
        q.shape()[rank - 2],
        q.shape()[rank - 1],
        v.shape()[rank - 1],
    );
    if q.shape() != k.shape()
        || q.shape() != log_decay.shape()
        || q.shape()[..rank - 1] != v.shape()[..rank - 1]
        || beta.shape()[..rank - 1] != q.shape()[..rank - 1]
        || beta.shape()[rank - 1] != 1
        || steps == 0
        || dk == 0
        || dv == 0
    {
        return Err("kda: inconsistent or zero-sized input geometry".to_string());
    }
    let batch_heads = checked_numel(&q.shape()[..rank - 2], "kda batch/head")?;
    Ok((batch_heads, steps, dk, dv))
}

pub fn kda_forward_requirements(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    write_state_next: bool,
) -> Result<KdaForwardRequirements, String> {
    let (batch_heads, steps, dk, dv) = kda_geometry(q, k, v, log_decay, beta)?;
    let mut output_shape = q.shape().to_vec();
    let rank = output_shape.len();
    output_shape[rank - 1] = dv;
    let work_dtype = work_dtype(q.dtype());
    let state = checked_requirement(&[batch_heads, dk, dv], work_dtype, "kda forward state")?;
    Ok(KdaForwardRequirements {
        output: checked_requirement(&output_shape, q.dtype(), "kda forward output")?,
        state_next: write_state_next.then(|| state.clone()),
        state_scratch: (!write_state_next).then_some(state),
        topology: KdaTopology::ForwardScan {
            chunk: 64,
            passes: 1,
        },
        batch_heads,
        steps,
        dk,
        dv,
        dtype: q.dtype(),
        work_dtype,
    })
}

pub fn kda_chunk_forward_requirements(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    write_state_next: bool,
) -> Result<KdaForwardRequirements, String> {
    kda_forward_requirements(q, k, v, log_decay, beta, write_state_next)
}

pub fn kda_backward_requirements(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
) -> Result<KdaBackwardRequirements, String> {
    let (batch_heads, steps, dk, dv) = kda_geometry(q, k, v, log_decay, beta)?;
    let chunks = steps.div_ceil(64);
    let state_elements = dk
        .checked_mul(dv)
        .ok_or_else(|| "kda backward: state size overflow".to_string())?;
    let scratch_per_head = chunks
        .checked_add(65)
        .and_then(|states| states.checked_mul(state_elements))
        .and_then(|states| states.checked_add(64usize.checked_mul(dv)?))
        .ok_or_else(|| "kda backward: scratch size overflow".to_string())?;
    let mut dv_shape = q.shape().to_vec();
    let rank = dv_shape.len();
    dv_shape[rank - 1] = dv;
    let mut dbeta_shape = q.shape().to_vec();
    dbeta_shape[rank - 1] = 1;
    let work_dtype = work_dtype(q.dtype());
    Ok(KdaBackwardRequirements {
        dq: checked_requirement(q.shape(), q.dtype(), "kda backward dq")?,
        dk: checked_requirement(k.shape(), k.dtype(), "kda backward dk")?,
        dv: checked_requirement(&dv_shape, v.dtype(), "kda backward dv")?,
        dlog_decay: checked_requirement(
            log_decay.shape(),
            log_decay.dtype(),
            "kda backward dlog_decay",
        )?,
        dbeta: checked_requirement(&dbeta_shape, beta.dtype(), "kda backward dbeta")?,
        scratch: checked_requirement(
            &[batch_heads, scratch_per_head],
            work_dtype,
            "kda backward scratch",
        )?,
        topology: KdaTopology::BackwardChunkRecompute {
            chunk: 64,
            passes: 3,
        },
        batch_heads,
        steps,
        key_depth: dk,
        value_depth: dv,
        chunks,
        dtype: q.dtype(),
        work_dtype,
    })
}

pub fn kda_chunk_backward_requirements(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
) -> Result<KdaBackwardRequirements, String> {
    kda_backward_requirements(q, k, v, log_decay, beta)
}

pub fn kda_decode_requirements(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
) -> Result<KdaDecodeRequirements, String> {
    if q.shape().len() != 2
        || k.shape() != q.shape()
        || log_decay.shape() != q.shape()
        || v.shape().len() != 2
        || beta.shape().len() != 2
        || v.shape()[0] != q.shape()[0]
        || beta.shape() != [q.shape()[0], 1]
        || q.dtype() != k.dtype()
        || q.dtype() != v.dtype()
        || q.dtype() != log_decay.dtype()
        || q.dtype() != beta.dtype()
        || !q.dtype().is_float()
    {
        return Err("kda decode: inconsistent inputs".to_string());
    }
    let (heads, dk, dv) = (q.shape()[0], q.shape()[1], v.shape()[1]);
    let work_dtype = work_dtype(q.dtype());
    Ok(KdaDecodeRequirements {
        output: checked_requirement(&[heads, dv], q.dtype(), "kda decode output")?,
        state_next: checked_requirement(&[heads, dk, dv], work_dtype, "kda decode state-next")?,
        topology: KdaTopology::Decode { passes: 1 },
        heads,
        dk,
        dv,
        dtype: q.dtype(),
        work_dtype,
    })
}

#[allow(clippy::too_many_arguments)]
fn kda_forward_into_impl<I: Elem, W: Elem>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    scale: f64,
    initial_state: Option<&Tensor>,
    output: &mut CpuDestination<'_>,
    state: &mut CpuDestination<'_>,
    batch_heads: usize,
    steps: usize,
    dk: usize,
    dv: usize,
) -> Result<(), String> {
    let q_values = tensor_values::<I>(q, "kda forward")?;
    let k_values = tensor_values::<I>(k, "kda forward")?;
    let v_values = tensor_values::<I>(v, "kda forward")?;
    let decay_values = tensor_values::<I>(log_decay, "kda forward")?;
    let beta_values = tensor_values::<I>(beta, "kda forward")?;
    let initial_values = initial_state
        .map(|tensor| tensor_values::<W>(tensor, "kda initial state"))
        .transpose()?;
    state.write::<W, _>(
        "kda forward state",
        &[batch_heads, dk, dv],
        |state_values| {
            if let (Some(initial), Some(initial_tensor)) = (initial_values, initial_state) {
                for index in 0..state_values.len() {
                    state_values[index] = logical_value(initial, initial_tensor, index);
                }
            } else {
                state_values.fill(W::default());
            }
            output.write_current::<I, _>("kda forward output", |out| {
                for bh in 0..batch_heads {
                    let state_base = bh * dk * dv;
                    for step in 0..steps {
                        let q_base = (bh * steps + step) * dk;
                        let v_base = (bh * steps + step) * dv;
                        let beta_value =
                            logical_value(beta_values, beta, bh * steps + step).to_f64();
                        for value_index in 0..dv {
                            let mut prediction = 0.0f64;
                            for depth in 0..dk {
                                let alpha = logical_value(decay_values, log_decay, q_base + depth)
                                    .to_f64()
                                    .exp();
                                prediction += alpha
                                    * state_values[state_base + depth * dv + value_index].to_f64()
                                    * logical_value(k_values, k, q_base + depth).to_f64();
                            }
                            let delta = logical_value(v_values, v, v_base + value_index).to_f64()
                                - prediction;
                            for depth in 0..dk {
                                let alpha = logical_value(decay_values, log_decay, q_base + depth)
                                    .to_f64()
                                    .exp();
                                let index = state_base + depth * dv + value_index;
                                state_values[index] = W::from_f64(
                                    alpha * state_values[index].to_f64()
                                        + beta_value
                                            * logical_value(k_values, k, q_base + depth).to_f64()
                                            * delta,
                                );
                            }
                        }
                        for value_index in 0..dv {
                            let mut value = 0.0f64;
                            for depth in 0..dk {
                                value += state_values[state_base + depth * dv + value_index]
                                    .to_f64()
                                    * logical_value(q_values, q, q_base + depth).to_f64();
                            }
                            out[v_base + value_index] = I::from_f64(value * scale);
                        }
                    }
                }
            })
        },
    )??;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kda_forward_into<'a>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    scale: f64,
    initial_state: Option<&Tensor>,
    output: &mut CpuDestination<'_>,
    state_next: Option<&mut CpuDestination<'a>>,
    state_scratch: Option<&mut CpuDestination<'a>>,
) -> Result<(), String> {
    let (batch_heads, steps, dk, dv) = kda_geometry(q, k, v, log_decay, beta)?;
    let rank = q.shape().len();
    if output.dtype() != q.dtype()
        || output.shape().len() != rank
        || output.shape()[..rank - 1] != q.shape()[..rank - 1]
        || output.shape()[rank - 1] != dv
    {
        return Err("kda forward: output shape or dtype mismatch".to_string());
    }
    let state = match (state_next, state_scratch) {
        (Some(state), None) | (None, Some(state)) => state,
        _ => {
            return Err(
                "kda forward: exactly one state-next or state scratch destination is required"
                    .to_string(),
            )
        }
    };
    let expected_work = work_dtype(q.dtype());
    if state.dtype() != expected_work {
        return Err(format!(
            "kda forward: state destination must have dtype {expected_work}"
        ));
    }
    if let Some(initial) = initial_state {
        if initial.dtype() != expected_work || initial.shape() != [batch_heads, dk, dv] {
            return Err("kda forward: initial state shape or dtype mismatch".to_string());
        }
    }
    match q.dtype() {
        DType::F32 => kda_forward_into_impl::<f32, f32>(
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
            initial_state,
            output,
            state,
            batch_heads,
            steps,
            dk,
            dv,
        ),
        DType::F64 => kda_forward_into_impl::<f64, f64>(
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
            initial_state,
            output,
            state,
            batch_heads,
            steps,
            dk,
            dv,
        ),
        DType::F16 => kda_forward_into_impl::<f16, f32>(
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
            initial_state,
            output,
            state,
            batch_heads,
            steps,
            dk,
            dv,
        ),
        DType::BF16 => kda_forward_into_impl::<bf16, f32>(
            q,
            k,
            v,
            log_decay,
            beta,
            scale,
            initial_state,
            output,
            state,
            batch_heads,
            steps,
            dk,
            dv,
        ),
        _ => unreachable!("kda geometry validated float dtype"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn kda_chunk_forward_into<'a>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    scale: f64,
    initial_state: Option<&Tensor>,
    output: &mut CpuDestination<'_>,
    state_next: Option<&mut CpuDestination<'a>>,
    state_scratch: Option<&mut CpuDestination<'a>>,
) -> Result<(), String> {
    kda_forward_into(
        q,
        k,
        v,
        log_decay,
        beta,
        scale,
        initial_state,
        output,
        state_next,
        state_scratch,
    )
}

#[allow(clippy::too_many_arguments)]
fn kda_decode_into_impl<I: Elem, W: Elem>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f64,
    output: &mut CpuDestination<'_>,
    state_next: &mut CpuDestination<'_>,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<(), String> {
    let q_values = tensor_values::<I>(q, "kda decode")?;
    let k_values = tensor_values::<I>(k, "kda decode")?;
    let v_values = tensor_values::<I>(v, "kda decode")?;
    let decay_values = tensor_values::<I>(log_decay, "kda decode")?;
    let beta_values = tensor_values::<I>(beta, "kda decode")?;
    let state_values = tensor_values::<W>(state, "kda decode")?;
    state_next.write::<W, _>("kda decode state-next", &[heads, dk, dv], |next| {
        output.write::<I, _>("kda decode output", &[heads, dv], |out| {
            for head in 0..heads {
                let state_base = head * dk * dv;
                let input_base = head * dk;
                let value_base = head * dv;
                let beta_value = logical_value(beta_values, beta, head).to_f64();
                for value_index in 0..dv {
                    let mut prediction = 0.0f64;
                    for depth in 0..dk {
                        let alpha = logical_value(decay_values, log_decay, input_base + depth)
                            .to_f64()
                            .exp();
                        prediction += alpha
                            * logical_value(
                                state_values,
                                state,
                                state_base + depth * dv + value_index,
                            )
                            .to_f64()
                            * logical_value(k_values, k, input_base + depth).to_f64();
                    }
                    let delta =
                        logical_value(v_values, v, value_base + value_index).to_f64() - prediction;
                    for depth in 0..dk {
                        let alpha = logical_value(decay_values, log_decay, input_base + depth)
                            .to_f64()
                            .exp();
                        next[state_base + depth * dv + value_index] = W::from_f64(
                            alpha
                                * logical_value(
                                    state_values,
                                    state,
                                    state_base + depth * dv + value_index,
                                )
                                .to_f64()
                                + beta_value
                                    * logical_value(k_values, k, input_base + depth).to_f64()
                                    * delta,
                        );
                    }
                }
                for value_index in 0..dv {
                    let mut value = 0.0f64;
                    for depth in 0..dk {
                        value += next[state_base + depth * dv + value_index].to_f64()
                            * logical_value(q_values, q, input_base + depth).to_f64();
                    }
                    out[value_base + value_index] = I::from_f64(value * scale);
                }
            }
        })
    })??;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kda_decode_into(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f64,
    output: &mut CpuDestination<'_>,
    state_next: &mut CpuDestination<'_>,
) -> Result<(), String> {
    if q.shape().len() != 2
        || k.shape() != q.shape()
        || log_decay.shape() != q.shape()
        || v.shape().len() != 2
        || beta.shape() != [q.shape()[0], 1]
        || v.shape()[0] != q.shape()[0]
        || q.dtype() != k.dtype()
        || q.dtype() != v.dtype()
        || q.dtype() != log_decay.dtype()
        || q.dtype() != beta.dtype()
        || !q.dtype().is_float()
    {
        return Err("kda decode: inconsistent inputs".to_string());
    }
    let (heads, dk, dv) = (q.shape()[0], q.shape()[1], v.shape()[1]);
    if state.dtype() != work_dtype(q.dtype()) || state.shape() != [heads, dk, dv] {
        return Err("kda decode: state shape or dtype mismatch".to_string());
    }
    match q.dtype() {
        DType::F32 => kda_decode_into_impl::<f32, f32>(
            q, k, v, log_decay, beta, state, scale, output, state_next, heads, dk, dv,
        ),
        DType::F64 => kda_decode_into_impl::<f64, f64>(
            q, k, v, log_decay, beta, state, scale, output, state_next, heads, dk, dv,
        ),
        DType::F16 => kda_decode_into_impl::<f16, f32>(
            q, k, v, log_decay, beta, state, scale, output, state_next, heads, dk, dv,
        ),
        DType::BF16 => kda_decode_into_impl::<bf16, f32>(
            q, k, v, log_decay, beta, state, scale, output, state_next, heads, dk, dv,
        ),
        _ => unreachable!("kda decode validated float dtype"),
    }
}

fn unsqueeze(t: &Tensor, dim: usize) -> Tensor {
    let mut shape = t.shape().to_vec();
    shape.insert(dim, 1);
    t.contiguous().view(Layout::contiguous(shape))
}

fn eye(n: usize, dtype: DType) -> Tensor {
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            data.push((i == j) as u8 as f32);
        }
    }
    Tensor::from_vec(data, vec![n, n]).cast(dtype)
}

// tril mask (diagonal 0 includes the diagonal, -1 excludes it) as u8.
fn tril_mask(n: usize, diagonal: i64) -> Tensor {
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n as i64 {
        for j in 0..n as i64 {
            data.push((j <= i + diagonal) as u8);
        }
    }
    Tensor::from_vec(data, vec![n, n])
}

// Unit lower-triangular inverse: given strictly lower-triangular a
// [.., n, n], returns (I + a)^-1 via batched row-wise forward
// substitution x_i = e_i - a_i[:, :i] @ x_{:i} (RFC 0018 numerics
// contract: sequential substitution, never a series expansion).
fn unit_lower_inverse(a: &Tensor) -> Tensor {
    let dims = a.shape();
    let r = rank(a);
    let n = dims[r - 1];
    let batch: usize = dims[..r - 2].iter().product();
    let a3 = a.contiguous().view(Layout::contiguous(vec![batch, n, n]));
    let id = eye(n, a.dtype());
    let mut x = batch_row(&narrow(&id, 0, 0, 1), batch);
    for i in 1..n {
        let a_row = narrow(&a3, 1, i, 1);
        let a_left = narrow(&a_row, 2, 0, i);
        let contrib = a_left.matmul(&x);
        let e_i = batch_row(&narrow(&id, 0, i, 1), batch);
        let row = e_i.sub(&contrib);
        x = Tensor::cat(&[&x, &row], 1);
    }
    x.view(Layout::contiguous(dims.to_vec()))
}

// row [1, n] -> [batch, 1, n] via broadcast add against zeros.
fn batch_row(row: &Tensor, batch: usize) -> Tensor {
    let n = row.shape()[row.shape().len() - 1];
    Tensor::zeros(&[batch, 1, n], row.dtype()).add(row)
}

// Chunked gated delta-rule linear attention, reference implementation
// (RFC 0018; FLA `naive_chunk_kda` equivalent). q/k/log_decay
// [.., H, T, Dk], v [.., H, T, Dv], beta [.., H, T, 1]; computes in f32
// (f64 stays f64) from a zero initial state. Chunk 64, sub-chunk 16:
// intra-chunk blocks use the pivot-factored decay
// exp(g_i - g_j) = exp(g_i - g_p) * exp(g_p - g_j) so no reciprocal
// cumulative decay is ever formed.
pub fn kda_chunk_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    scale: f64,
) -> Tensor {
    let dims = q.shape().to_vec();
    let r = dims.len();
    let dk = dims[r - 1];
    let dv = v.shape()[r - 1];
    let bh: usize = dims[..r - 2].iter().product();
    let work = if q.dtype() == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let initial = Tensor::zeros(&[bh, dk, dv], work);
    kda_chunk_with_state(q, k, v, log_decay, beta, scale, &initial).0
}

// Stateful variant: starts from `initial_state` ([BH, Dk, Dv], work
// dtype) and returns the output alongside the final state. The decode
// path drives this per sequence slot.
pub fn kda_chunk_with_state(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    scale: f64,
    initial_state: &Tensor,
) -> (Tensor, Tensor) {
    const CHUNK: usize = 64;
    const SUB: usize = 16;
    let in_dtype = q.dtype();
    let work = if in_dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let q = q.cast(work);
    let k = k.cast(work);
    let v = v.cast(work);
    let log_decay = log_decay.cast(work);
    let beta = beta.cast(work);

    let dims = q.shape().to_vec();
    let r = dims.len();
    let (t, dk) = (dims[r - 2], dims[r - 1]);
    let dv = v.shape()[r - 1];
    let bh: usize = dims[..r - 2].iter().product();

    let q3 = q.contiguous().view(Layout::contiguous(vec![bh, t, dk]));
    let k3 = k.contiguous().view(Layout::contiguous(vec![bh, t, dk]));
    let v3 = v.contiguous().view(Layout::contiguous(vec![bh, t, dv]));
    let ld3 = log_decay
        .contiguous()
        .view(Layout::contiguous(vec![bh, t, dk]));
    let b3 = beta.contiguous().view(Layout::contiguous(vec![bh, t, 1]));

    let mut state = initial_state.cast(work).contiguous();
    let mut outs: Vec<Tensor> = Vec::new();
    let mut t0 = 0;
    while t0 < t {
        let c = CHUNK.min(t - t0);
        let qc = narrow(&q3, 1, t0, c);
        let kc = narrow(&k3, 1, t0, c);
        let vc = narrow(&v3, 1, t0, c);
        let bc = narrow(&b3, 1, t0, c);
        // Inclusive chunk-local cumulative log decay, [BH, c, Dk].
        let gc = narrow(&ld3, 1, t0, c).cumsum(1);

        // Intra-chunk attention matrices, assembled from SUB-sized
        // blocks: Aqk (lower-triangular, scaled) and Akk (strictly
        // lower, beta-weighted).
        let blocks = c.div_ceil(SUB);
        let mut aqk_rows: Vec<Tensor> = Vec::new();
        let mut akk_rows: Vec<Tensor> = Vec::new();
        for rb in 0..blocks {
            let rs = rb * SUB;
            let br = SUB.min(c - rs);
            let q_r = narrow(&qc, 1, rs, br);
            let k_r = narrow(&kc, 1, rs, br);
            let g_r = narrow(&gc, 1, rs, br);
            let g_p = narrow(&gc, 1, rs, 1);
            let b_r = narrow(&bc, 1, rs, br);
            let mut aqk_cols: Vec<Tensor> = Vec::new();
            let mut akk_cols: Vec<Tensor> = Vec::new();
            for cb in 0..blocks {
                let cs = cb * SUB;
                let cc = SUB.min(c - cs);
                if cb > rb {
                    aqk_cols.push(Tensor::zeros(&[bh, br, cc], work));
                    akk_cols.push(Tensor::zeros(&[bh, br, cc], work));
                    continue;
                }
                let k_c = narrow(&kc, 1, cs, cc);
                let g_c = narrow(&gc, 1, cs, cc);
                if cb == rb {
                    // Diagonal block: full per-channel decay matrix,
                    // masked by select (exp overflow on the masked
                    // triangle is discarded, never multiplied).
                    let d = unsqueeze(&g_r, 2).sub(&unsqueeze(&g_c, 1));
                    let e = d.exp();
                    let zeros = Tensor::zeros(e.shape(), work);
                    let m_incl = unsqueeze(&unsqueeze(&tril_mask(br, 0), 0), 3);
                    let m_strict = unsqueeze(&unsqueeze(&tril_mask(br, -1), 0), 3);
                    let e_incl = e.where_(&m_incl, &zeros);
                    let e_strict = e.where_(&m_strict, &zeros);
                    let qq = unsqueeze(&q_r, 2).mul(&unsqueeze(&k_c, 1));
                    let aqk = squeeze_last(&qq.mul(&e_incl).sum(&[3])).mul(&Tensor::full(
                        &[bh, br, cc],
                        scale,
                        work,
                    ));
                    let kk = unsqueeze(&k_r, 2).mul(&unsqueeze(&k_c, 1));
                    let akk = squeeze_last(&kk.mul(&e_strict).sum(&[3])).mul(&b_r);
                    aqk_cols.push(aqk);
                    akk_cols.push(akk);
                } else {
                    // Off-diagonal block: decay factors at the row
                    // block's pivot, both directions bounded by 1.
                    let qd = q_r.mul(&g_r.sub(&g_p).exp());
                    let kd = k_c.mul(&g_p.sub(&g_c).exp());
                    let aqk = qd.matmul(&transpose_last2(&kd)).mul(&Tensor::full(
                        &[bh, br, cc],
                        scale,
                        work,
                    ));
                    let kkd = k_r.mul(&g_r.sub(&g_p).exp());
                    let akk = kkd.matmul(&transpose_last2(&kd)).mul(&b_r);
                    aqk_cols.push(aqk);
                    akk_cols.push(akk);
                }
            }
            aqk_rows.push(Tensor::cat(&aqk_cols.iter().collect::<Vec<_>>(), 2));
            akk_rows.push(Tensor::cat(&akk_cols.iter().collect::<Vec<_>>(), 2));
        }
        let aqk = Tensor::cat(&aqk_rows.iter().collect::<Vec<_>>(), 1);
        let akk = Tensor::cat(&akk_rows.iter().collect::<Vec<_>>(), 1);

        // UT transform: M = (I + Akk)^-1, then the WY representation.
        let m = unit_lower_inverse(&akk);
        let w_in = kc.mul(&bc).mul(&gc.exp());
        let w = m.matmul(&w_in);
        let u = m.matmul(&vc.mul(&bc));

        let v_new = u.sub(&w.matmul(&state));
        let o_inter =
            qc.mul(&gc.exp())
                .matmul(&state)
                .mul(&Tensor::full(&[bh, c, dv], scale, work));
        let o_intra = aqk.matmul(&v_new);
        outs.push(o_inter.add(&o_intra));

        // State update: decay to the chunk end, then rank-c update with
        // the decayed keys kg = k * exp(g_last - g).
        let g_last = narrow(&gc, 1, c - 1, 1);
        let kg = kc.mul(&g_last.sub(&gc).exp());
        let decay = transpose_last2(&g_last.exp());
        state = state.mul(&decay).add(&transpose_last2(&kg).matmul(&v_new));
        t0 += c;
    }

    let out = Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1);
    let mut out_shape = dims;
    out_shape[r - 1] = dv;
    let out = out
        .contiguous()
        .view(Layout::contiguous(out_shape))
        .cast(in_dtype);
    (out, state)
}

fn short_conv_geometry(
    x: &Tensor,
    weight: &Tensor,
) -> Result<(usize, usize, usize, usize), String> {
    let rank = x.shape().len();
    if rank < 2 || weight.shape().len() != 2 || x.dtype() != weight.dtype() || !x.dtype().is_float()
    {
        return Err("short_conv1d: expected matching float tensors of rank >= 2 and 2".into());
    }
    let (steps, channels, kernel) = (x.shape()[rank - 2], x.shape()[rank - 1], weight.shape()[1]);
    if weight.shape()[0] != channels || steps == 0 || channels == 0 || kernel == 0 {
        return Err("short_conv1d: inconsistent or zero-sized geometry".into());
    }
    let batch = checked_numel(&x.shape()[..rank - 2], "short_conv1d batch")?;
    Ok((batch, steps, channels, kernel))
}

pub fn short_conv1d_forward_requirements(
    x: &Tensor,
    weight: &Tensor,
    write_state_next: bool,
) -> Result<ShortConvForwardRequirements, String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    if write_state_next && batch != 1 {
        return Err("short_conv1d state requires one sequence".into());
    }
    Ok(ShortConvForwardRequirements {
        output: checked_requirement(x.shape(), x.dtype(), "short_conv1d output")?,
        state_next: write_state_next
            .then(|| {
                checked_requirement(
                    &[kernel - 1, channels],
                    x.dtype(),
                    "short_conv1d state-next",
                )
            })
            .transpose()?,
        topology: ShortConvTopology::Direct { passes: 1 },
        batch,
        steps,
        channels,
        kernel,
        dtype: x.dtype(),
    })
}

pub fn short_conv1d_state_requirements(
    x: &Tensor,
    weight: &Tensor,
) -> Result<ShortConvForwardRequirements, String> {
    short_conv1d_forward_requirements(x, weight, true)
}

pub fn short_conv1d_backward_requirements(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
) -> Result<ShortConvBackwardRequirements, String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    if gradient.dtype() != x.dtype() || gradient.shape() != x.shape() {
        return Err("short_conv1d backward: gradient must match input".into());
    }
    Ok(ShortConvBackwardRequirements {
        dx: checked_requirement(x.shape(), x.dtype(), "short_conv1d dx")?,
        dweight: checked_requirement(weight.shape(), weight.dtype(), "short_conv1d dweight")?,
        topology: ShortConvTopology::Direct { passes: 2 },
        batch,
        steps,
        channels,
        kernel,
        dtype: x.dtype(),
    })
}

pub fn short_conv1d_backward_x_requirements(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
) -> Result<ShortConvBackwardXRequirements, String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    if gradient.dtype() != x.dtype() || gradient.shape() != x.shape() {
        return Err("short_conv1d backward-x: gradient must match input".into());
    }
    Ok(ShortConvBackwardXRequirements {
        output: checked_requirement(x.shape(), x.dtype(), "short_conv1d dx")?,
        topology: ShortConvTopology::Direct { passes: 1 },
        batch,
        steps,
        channels,
        kernel,
        dtype: x.dtype(),
    })
}

pub fn short_conv1d_backward_w_requirements(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
) -> Result<ShortConvBackwardWRequirements, String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    if gradient.dtype() != x.dtype() || gradient.shape() != x.shape() {
        return Err("short_conv1d backward-weight: gradient must match input".into());
    }
    Ok(ShortConvBackwardWRequirements {
        output: checked_requirement(weight.shape(), weight.dtype(), "short_conv1d dweight")?,
        topology: ShortConvTopology::Direct { passes: 1 },
        batch,
        steps,
        channels,
        kernel,
        dtype: x.dtype(),
    })
}

fn short_conv1d_forward_into_impl<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    output: &mut CpuDestination<'_>,
    batch: usize,
    steps: usize,
    channels: usize,
    kernel: usize,
) -> Result<(), String> {
    let x_values = tensor_values::<T>(x, "short_conv1d")?;
    let weight_values = tensor_values::<T>(weight, "short_conv1d")?;
    output.write::<T, _>("short_conv1d output", x.shape(), |out| {
        for batch_index in 0..batch {
            for step in 0..steps {
                for channel in 0..channels {
                    let mut value = 0.0f64;
                    for tap in 0..kernel {
                        let source = step + tap + 1;
                        if source >= kernel {
                            let source_step = source - kernel;
                            value += logical_value(
                                x_values,
                                x,
                                (batch_index * steps + source_step) * channels + channel,
                            )
                            .to_f64()
                                * logical_value(weight_values, weight, channel * kernel + tap)
                                    .to_f64();
                        }
                    }
                    out[(batch_index * steps + step) * channels + channel] = T::from_f64(value);
                }
            }
        }
    })
}

pub fn short_conv1d_forward_into(
    x: &Tensor,
    weight: &Tensor,
    output: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    match x.dtype() {
        DType::F32 => {
            short_conv1d_forward_into_impl::<f32>(x, weight, output, batch, steps, channels, kernel)
        }
        DType::F64 => {
            short_conv1d_forward_into_impl::<f64>(x, weight, output, batch, steps, channels, kernel)
        }
        DType::F16 => {
            short_conv1d_forward_into_impl::<f16>(x, weight, output, batch, steps, channels, kernel)
        }
        DType::BF16 => short_conv1d_forward_into_impl::<bf16>(
            x, weight, output, batch, steps, channels, kernel,
        ),
        _ => unreachable!("short_conv geometry validated float dtype"),
    }
}

#[allow(clippy::too_many_arguments)]
fn short_conv1d_with_state_into_impl<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    state: &Tensor,
    advance: usize,
    output: &mut CpuDestination<'_>,
    state_next: &mut CpuDestination<'_>,
    steps: usize,
    channels: usize,
    kernel: usize,
) -> Result<(), String> {
    let x_values = tensor_values::<T>(x, "short_conv1d state")?;
    let weight_values = tensor_values::<T>(weight, "short_conv1d state")?;
    let state_values = tensor_values::<T>(state, "short_conv1d state")?;
    output.write::<T, _>("short_conv1d state output", x.shape(), |out| {
        state_next.write::<T, _>("short_conv1d state-next", &[kernel - 1, channels], |next| {
            for step in 0..steps {
                for channel in 0..channels {
                    let mut value = 0.0f64;
                    for tap in 0..kernel {
                        let window_index = step + tap;
                        let source = if window_index < kernel - 1 {
                            logical_value(state_values, state, window_index * channels + channel)
                        } else {
                            logical_value(
                                x_values,
                                x,
                                (window_index - (kernel - 1)) * channels + channel,
                            )
                        };
                        value += source.to_f64()
                            * logical_value(weight_values, weight, channel * kernel + tap).to_f64();
                    }
                    out[step * channels + channel] = T::from_f64(value);
                }
            }
            for index in 0..kernel - 1 {
                let window_index = advance + index;
                for channel in 0..channels {
                    next[index * channels + channel] = if window_index < kernel - 1 {
                        logical_value(state_values, state, window_index * channels + channel)
                    } else {
                        logical_value(
                            x_values,
                            x,
                            (window_index - (kernel - 1)) * channels + channel,
                        )
                    };
                }
            }
        })
    })??;
    Ok(())
}

pub fn short_conv1d_with_state_into(
    x: &Tensor,
    weight: &Tensor,
    state: &Tensor,
    advance: usize,
    output: &mut CpuDestination<'_>,
    state_next: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    if batch != 1
        || advance > steps
        || state.dtype() != x.dtype()
        || state.shape() != [kernel - 1, channels]
    {
        return Err("short_conv1d state: inconsistent state or advance".into());
    }
    match x.dtype() {
        DType::F32 => short_conv1d_with_state_into_impl::<f32>(
            x, weight, state, advance, output, state_next, steps, channels, kernel,
        ),
        DType::F64 => short_conv1d_with_state_into_impl::<f64>(
            x, weight, state, advance, output, state_next, steps, channels, kernel,
        ),
        DType::F16 => short_conv1d_with_state_into_impl::<f16>(
            x, weight, state, advance, output, state_next, steps, channels, kernel,
        ),
        DType::BF16 => short_conv1d_with_state_into_impl::<bf16>(
            x, weight, state, advance, output, state_next, steps, channels, kernel,
        ),
        _ => unreachable!("short_conv geometry validated float dtype"),
    }
}

#[allow(clippy::too_many_arguments)]
fn short_conv1d_backward_into_impl<T: Elem>(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
    dx: &mut CpuDestination<'_>,
    dweight: &mut CpuDestination<'_>,
    batch: usize,
    steps: usize,
    channels: usize,
    kernel: usize,
) -> Result<(), String> {
    let x_values = tensor_values::<T>(x, "short_conv1d backward")?;
    let weight_values = tensor_values::<T>(weight, "short_conv1d backward")?;
    let gradient_values = tensor_values::<T>(gradient, "short_conv1d backward")?;
    dx.write::<T, _>("short_conv1d dx", x.shape(), |dx_values| {
        dweight.write::<T, _>("short_conv1d dweight", weight.shape(), |dweight_values| {
            for batch_index in 0..batch {
                for step in 0..steps {
                    for channel in 0..channels {
                        let mut value = 0.0f64;
                        for offset in 0..kernel {
                            if step + offset < steps {
                                value += logical_value(
                                    gradient_values,
                                    gradient,
                                    (batch_index * steps + step + offset) * channels + channel,
                                )
                                .to_f64()
                                    * logical_value(
                                        weight_values,
                                        weight,
                                        channel * kernel + kernel - 1 - offset,
                                    )
                                    .to_f64();
                            }
                        }
                        dx_values[(batch_index * steps + step) * channels + channel] =
                            T::from_f64(value);
                    }
                }
            }
            for channel in 0..channels {
                for tap in 0..kernel {
                    let mut value = 0.0f64;
                    for batch_index in 0..batch {
                        for step in 0..steps {
                            let source = step + tap + 1;
                            if source >= kernel {
                                let source_step = source - kernel;
                                value += logical_value(
                                    gradient_values,
                                    gradient,
                                    (batch_index * steps + step) * channels + channel,
                                )
                                .to_f64()
                                    * logical_value(
                                        x_values,
                                        x,
                                        (batch_index * steps + source_step) * channels + channel,
                                    )
                                    .to_f64();
                            }
                        }
                    }
                    dweight_values[channel * kernel + tap] = T::from_f64(value);
                }
            }
        })
    })??;
    Ok(())
}

pub fn short_conv1d_backward_into(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
    dx: &mut CpuDestination<'_>,
    dweight: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    if gradient.dtype() != x.dtype() || gradient.shape() != x.shape() {
        return Err("short_conv1d backward: gradient must match input".into());
    }
    match x.dtype() {
        DType::F32 => short_conv1d_backward_into_impl::<f32>(
            x, weight, gradient, dx, dweight, batch, steps, channels, kernel,
        ),
        DType::F64 => short_conv1d_backward_into_impl::<f64>(
            x, weight, gradient, dx, dweight, batch, steps, channels, kernel,
        ),
        DType::F16 => short_conv1d_backward_into_impl::<f16>(
            x, weight, gradient, dx, dweight, batch, steps, channels, kernel,
        ),
        DType::BF16 => short_conv1d_backward_into_impl::<bf16>(
            x, weight, gradient, dx, dweight, batch, steps, channels, kernel,
        ),
        _ => unreachable!("short_conv geometry validated float dtype"),
    }
}

pub fn short_conv1d_backward_x_into(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
    dx: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    if gradient.dtype() != x.dtype() || gradient.shape() != x.shape() {
        return Err("short_conv1d backward-x: gradient must match input".into());
    }
    macro_rules! run {
        ($type:ty) => {{
            let weights = tensor_values::<$type>(weight, "short_conv1d backward-x")?;
            let gradients = tensor_values::<$type>(gradient, "short_conv1d backward-x")?;
            dx.write::<$type, _>("short_conv1d dx", x.shape(), |output| {
                for batch_index in 0..batch {
                    for step in 0..steps {
                        for channel in 0..channels {
                            let mut value = 0.0f64;
                            for offset in 0..kernel {
                                if step + offset < steps {
                                    value += logical_value(
                                        gradients,
                                        gradient,
                                        (batch_index * steps + step + offset) * channels + channel,
                                    )
                                    .to_f64()
                                        * logical_value(
                                            weights,
                                            weight,
                                            channel * kernel + kernel - 1 - offset,
                                        )
                                        .to_f64();
                                }
                            }
                            output[(batch_index * steps + step) * channels + channel] =
                                <$type as Elem>::from_f64(value);
                        }
                    }
                }
            })
        }};
    }
    match x.dtype() {
        DType::F32 => run!(f32),
        DType::F64 => run!(f64),
        DType::F16 => run!(f16),
        DType::BF16 => run!(bf16),
        _ => unreachable!("short_conv geometry validated float dtype"),
    }
}

pub fn short_conv1d_backward_w_into(
    x: &Tensor,
    weight: &Tensor,
    gradient: &Tensor,
    dweight: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch, steps, channels, kernel) = short_conv_geometry(x, weight)?;
    if gradient.dtype() != x.dtype() || gradient.shape() != x.shape() {
        return Err("short_conv1d backward-weight: gradient must match input".into());
    }
    macro_rules! run {
        ($type:ty) => {{
            let inputs = tensor_values::<$type>(x, "short_conv1d backward-weight")?;
            let gradients = tensor_values::<$type>(gradient, "short_conv1d backward-weight")?;
            dweight.write::<$type, _>("short_conv1d dweight", weight.shape(), |output| {
                for channel in 0..channels {
                    for tap in 0..kernel {
                        let mut value = 0.0f64;
                        for batch_index in 0..batch {
                            for step in 0..steps {
                                let source = step + tap + 1;
                                if source >= kernel {
                                    value += logical_value(
                                        gradients,
                                        gradient,
                                        (batch_index * steps + step) * channels + channel,
                                    )
                                    .to_f64()
                                        * logical_value(
                                            inputs,
                                            x,
                                            (batch_index * steps + source - kernel) * channels
                                                + channel,
                                        )
                                        .to_f64();
                                }
                            }
                        }
                        output[channel * kernel + tap] = <$type as Elem>::from_f64(value);
                    }
                }
            })
        }};
    }
    match x.dtype() {
        DType::F32 => run!(f32),
        DType::F64 => run!(f64),
        DType::F16 => run!(f16),
        DType::BF16 => run!(bf16),
        _ => unreachable!("short_conv geometry validated float dtype"),
    }
}

// Causal depthwise short convolution over [.., T, C] with weight
// [C, K] and zero history: y[t] = sum_j w[:, j] * x[t-K+1+j].
pub fn short_conv1d_forward(x: &Tensor, weight: &Tensor) -> Tensor {
    let dims = x.shape().to_vec();
    let r = dims.len();
    let (t, c) = (dims[r - 2], dims[r - 1]);
    let kk = weight.shape()[1];
    let batch: usize = dims[..r - 2].iter().product();
    let x3 = x.contiguous().view(Layout::contiguous(vec![batch, t, c]));
    let history = Tensor::zeros(&[batch, kk - 1, c], x.dtype());
    let window = Tensor::cat(&[&history, &x3], 1);
    let mut acc = Tensor::zeros(&[batch, t, c], x.dtype());
    for j in 0..kk {
        let wj = transpose_last2(&narrow(weight, 1, j, 1));
        acc = acc.add(&narrow(&window, 1, j, t).mul(&wj));
    }
    acc.contiguous().view(Layout::contiguous(dims))
}

// Stateful per-slot variant: x [T, C], state [K-1, C]; returns the
// output and the new window. `advance` is the count of real tokens
// (chunked prefill right-pads): outputs are computed over the full
// padded window — causal, so real rows never see padding — but the
// stored window shifts in only the first `advance` rows.
pub fn short_conv1d_with_state(
    x: &Tensor,
    weight: &Tensor,
    state: &Tensor,
    advance: usize,
) -> (Tensor, Tensor) {
    let dims = x.shape().to_vec();
    let (t, kk) = (dims[0], weight.shape()[1]);
    let window = Tensor::cat(&[state, x], 0);
    let mut acc = Tensor::zeros(&dims, x.dtype());
    for j in 0..kk {
        let wj = transpose_last2(&narrow(weight, 1, j, 1));
        acc = acc.add(&narrow(&window, 0, j, t).mul(&wj));
    }
    let real = narrow(&window, 0, 0, kk - 1 + advance);
    let new_state = narrow(&real, 0, advance, kk - 1).contiguous();
    (acc, new_state)
}

// ShortConv1d adjoints (RFC 0018 phase 4). dx[s] = sum_j w[:, K-1-j] *
// g[s+j] (full correlation over the right-zero-padded cotangent);
// dw[:, j] = sum_t g[t] * x[t-K+1+j] (per-tap correlation over the
// causal window). g and x are [.., T, C]; weight is [C, K].
pub fn short_conv1d_backward_x(x: &Tensor, weight: &Tensor, g: &Tensor) -> Tensor {
    let dims = x.shape().to_vec();
    let r = dims.len();
    let (t, c) = (dims[r - 2], dims[r - 1]);
    let kk = weight.shape()[1];
    let batch: usize = dims[..r - 2].iter().product();
    let g3 = g.contiguous().view(Layout::contiguous(vec![batch, t, c]));
    let padded = Tensor::cat(&[&g3, &Tensor::zeros(&[batch, kk - 1, c], g3.dtype())], 1);
    let mut acc = Tensor::zeros(&[batch, t, c], g3.dtype());
    for j in 0..kk {
        let wj = transpose_last2(&narrow(weight, 1, kk - 1 - j, 1));
        acc = acc.add(&narrow(&padded, 1, j, t).mul(&wj));
    }
    acc.contiguous().view(Layout::contiguous(dims))
}

pub fn short_conv1d_backward_w(x: &Tensor, weight: &Tensor, g: &Tensor) -> Tensor {
    let dims = x.shape().to_vec();
    let r = dims.len();
    let (t, c) = (dims[r - 2], dims[r - 1]);
    let kk = weight.shape()[1];
    let batch: usize = dims[..r - 2].iter().product();
    let x3 = x.contiguous().view(Layout::contiguous(vec![batch, t, c]));
    let g3 = g.contiguous().view(Layout::contiguous(vec![batch, t, c]));
    let window = Tensor::cat(&[&Tensor::zeros(&[batch, kk - 1, c], x3.dtype()), &x3], 1);
    let mut cols: Vec<Tensor> = Vec::with_capacity(kk);
    for j in 0..kk {
        // [batch, 1, C]: sum over T of g * window[j .. j+T]
        let tap = g3.mul(&narrow(&window, 1, j, t)).sum(&[1]);
        cols.push(tap);
    }
    // [batch, K, C] -> sum over batch -> [1, K, C] -> [1, C, K] -> [C, K]
    let stacked = Tensor::cat(&cols.iter().collect::<Vec<_>>(), 1);
    let summed = stacked.sum(&[0]);
    transpose_last2(&summed)
        .contiguous()
        .view(Layout::contiguous(vec![c, kk]))
}

// Closed-form KDA backward (RFC 0018 phase 4). With S̃_t = Diag(α_t)
// S_{t-1}, δ_t = v_t − S̃_tᵀ k_t, S_t = S̃_t + β_t k_t δ_tᵀ and o_t =
// scale · S_tᵀ q_t, the adjoint state Λ_t = ∂L/∂S_t runs in reverse:
//
//   Λ_t   += scale · q_t g_tᵀ           (g = output cotangent)
//   dq_t   = scale · S_t g_t
//   dv_t   = β_t Λ_tᵀ k_t
//   dk_t   = β_t (Λ_t δ_t − S̃_t (Λ_tᵀ k_t))
//   dβ_t   = k_tᵀ Λ_t δ_t
//   dα_t   = sum_dv(S_{t-1} ⊙ M_t), M_t = (I − β_t k_t k_tᵀ) Λ_t
//   dg_t   = dα_t ⊙ α_t
//   Λ_{t-1} = Diag(α_t) M_t
//
// Memory stays bounded: pass 1 retains only the 64-token chunk start
// states; pass 2 walks chunks in reverse and recomputes the per-token
// states within each chunk (transient, one chunk at a time).
#[allow(clippy::too_many_arguments)]
fn kda_backward_into_impl<I: Elem, W: Elem>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    gradient: &Tensor,
    scale: f64,
    dq: &mut CpuDestination<'_>,
    dk_out: &mut CpuDestination<'_>,
    dv_out: &mut CpuDestination<'_>,
    dlog_decay: &mut CpuDestination<'_>,
    dbeta: &mut CpuDestination<'_>,
    scratch: &mut CpuDestination<'_>,
    batch_heads: usize,
    steps: usize,
    dk: usize,
    dv: usize,
    chunks: usize,
    scratch_per_head: usize,
) -> Result<(), String> {
    let q_values = tensor_values::<I>(q, "kda backward")?;
    let k_values = tensor_values::<I>(k, "kda backward")?;
    let v_values = tensor_values::<I>(v, "kda backward")?;
    let decay_values = tensor_values::<I>(log_decay, "kda backward")?;
    let beta_values = tensor_values::<I>(beta, "kda backward")?;
    let gradient_values = tensor_values::<I>(gradient, "kda backward")?;
    let state_elements = dk * dv;
    scratch.write::<W, _>(
        "kda backward scratch",
        &[batch_heads, scratch_per_head],
        |work| {
            dq.write::<I, _>("kda backward dq", q.shape(), |dq_values| {
                dk_out.write::<I, _>("kda backward dk", k.shape(), |dk_values| {
                    dv_out.write::<I, _>("kda backward dv", v.shape(), |dv_values| {
                        dlog_decay.write::<I, _>(
                            "kda backward dlog_decay",
                            log_decay.shape(),
                            |dlog_values| {
                                dbeta.write::<I, _>(
                                    "kda backward dbeta",
                                    beta.shape(),
                                    |dbeta_values| {
                                        for bh in 0..batch_heads {
                                            let root = bh * scratch_per_head;
                                            let starts = root;
                                            let states = starts + chunks * state_elements;
                                            let lam = states + 64 * state_elements;
                                            let deltas = lam + state_elements;
                                            work[lam..lam + state_elements].fill(W::default());

                                            // Pass 1: retain only each chunk's start state.
                                            for chunk in 0..chunks {
                                                let start = starts + chunk * state_elements;
                                                for index in 0..state_elements {
                                                    work[start + index] = work[lam + index];
                                                }
                                                let t0 = chunk * 64;
                                                let length = 64.min(steps - t0);
                                                for local in 0..length {
                                                    let step = t0 + local;
                                                    let key_base = (bh * steps + step) * dk;
                                                    let value_base = (bh * steps + step) * dv;
                                                    let beta_value = logical_value(
                                                        beta_values,
                                                        beta,
                                                        bh * steps + step,
                                                    )
                                                    .to_f64();
                                                    for value_index in 0..dv {
                                                        let mut prediction = 0.0f64;
                                                        for depth in 0..dk {
                                                            let alpha = logical_value(
                                                                decay_values,
                                                                log_decay,
                                                                key_base + depth,
                                                            )
                                                            .to_f64()
                                                            .exp();
                                                            prediction += alpha
                                                                * work[lam
                                                                    + depth * dv
                                                                    + value_index]
                                                                    .to_f64()
                                                                * logical_value(
                                                                    k_values,
                                                                    k,
                                                                    key_base + depth,
                                                                )
                                                                .to_f64();
                                                        }
                                                        let delta = logical_value(
                                                            v_values,
                                                            v,
                                                            value_base + value_index,
                                                        )
                                                        .to_f64()
                                                            - prediction;
                                                        for depth in 0..dk {
                                                            let alpha = logical_value(
                                                                decay_values,
                                                                log_decay,
                                                                key_base + depth,
                                                            )
                                                            .to_f64()
                                                            .exp();
                                                            let index =
                                                                lam + depth * dv + value_index;
                                                            work[index] = W::from_f64(
                                                                alpha * work[index].to_f64()
                                                                    + beta_value
                                                                        * logical_value(
                                                                            k_values,
                                                                            k,
                                                                            key_base + depth,
                                                                        )
                                                                        .to_f64()
                                                                        * delta,
                                                            );
                                                        }
                                                    }
                                                }
                                            }

                                            // Pass 2: reverse chunks, recomputing 64 states.
                                            work[lam..lam + state_elements].fill(W::default());
                                            for chunk in (0..chunks).rev() {
                                                let t0 = chunk * 64;
                                                let length = 64.min(steps - t0);
                                                let chunk_start = starts + chunk * state_elements;
                                                for local in 0..length {
                                                    let step = t0 + local;
                                                    let previous = if local == 0 {
                                                        chunk_start
                                                    } else {
                                                        states + (local - 1) * state_elements
                                                    };
                                                    let current = states + local * state_elements;
                                                    let key_base = (bh * steps + step) * dk;
                                                    let value_base = (bh * steps + step) * dv;
                                                    let beta_value = logical_value(
                                                        beta_values,
                                                        beta,
                                                        bh * steps + step,
                                                    )
                                                    .to_f64();
                                                    for value_index in 0..dv {
                                                        let mut prediction = 0.0f64;
                                                        for depth in 0..dk {
                                                            let alpha = logical_value(
                                                                decay_values,
                                                                log_decay,
                                                                key_base + depth,
                                                            )
                                                            .to_f64()
                                                            .exp();
                                                            prediction += alpha
                                                                * work[previous
                                                                    + depth * dv
                                                                    + value_index]
                                                                    .to_f64()
                                                                * logical_value(
                                                                    k_values,
                                                                    k,
                                                                    key_base + depth,
                                                                )
                                                                .to_f64();
                                                        }
                                                        let delta = logical_value(
                                                            v_values,
                                                            v,
                                                            value_base + value_index,
                                                        )
                                                        .to_f64()
                                                            - prediction;
                                                        work[deltas + local * dv + value_index] =
                                                            W::from_f64(delta);
                                                        for depth in 0..dk {
                                                            let alpha = logical_value(
                                                                decay_values,
                                                                log_decay,
                                                                key_base + depth,
                                                            )
                                                            .to_f64()
                                                            .exp();
                                                            work[current
                                                                + depth * dv
                                                                + value_index] = W::from_f64(
                                                                alpha
                                                                    * work[previous
                                                                        + depth * dv
                                                                        + value_index]
                                                                        .to_f64()
                                                                    + beta_value
                                                                        * logical_value(
                                                                            k_values,
                                                                            k,
                                                                            key_base + depth,
                                                                        )
                                                                        .to_f64()
                                                                        * delta,
                                                            );
                                                        }
                                                    }
                                                }
                                                for local in (0..length).rev() {
                                                    let step = t0 + local;
                                                    let previous = if local == 0 {
                                                        chunk_start
                                                    } else {
                                                        states + (local - 1) * state_elements
                                                    };
                                                    let current = states + local * state_elements;
                                                    let key_base = (bh * steps + step) * dk;
                                                    let value_base = (bh * steps + step) * dv;
                                                    let beta_value = logical_value(
                                                        beta_values,
                                                        beta,
                                                        bh * steps + step,
                                                    )
                                                    .to_f64();

                                                    for depth in 0..dk {
                                                        for value_index in 0..dv {
                                                            let index =
                                                                lam + depth * dv + value_index;
                                                            work[index] = W::from_f64(
                                                                work[index].to_f64()
                                                                    + scale
                                                                        * logical_value(
                                                                            q_values,
                                                                            q,
                                                                            key_base + depth,
                                                                        )
                                                                        .to_f64()
                                                                        * logical_value(
                                                                            gradient_values,
                                                                            gradient,
                                                                            value_base
                                                                                + value_index,
                                                                        )
                                                                        .to_f64(),
                                                            );
                                                        }
                                                    }
                                                    for depth in 0..dk {
                                                        let mut value = 0.0f64;
                                                        for value_index in 0..dv {
                                                            value += work[current
                                                                + depth * dv
                                                                + value_index]
                                                                .to_f64()
                                                                * logical_value(
                                                                    gradient_values,
                                                                    gradient,
                                                                    value_base + value_index,
                                                                )
                                                                .to_f64();
                                                        }
                                                        dq_values[key_base + depth] =
                                                            I::from_f64(value * scale);
                                                    }
                                                    for value_index in 0..dv {
                                                        let mut lam_k = 0.0f64;
                                                        for depth in 0..dk {
                                                            lam_k += work
                                                                [lam + depth * dv + value_index]
                                                                .to_f64()
                                                                * logical_value(
                                                                    k_values,
                                                                    k,
                                                                    key_base + depth,
                                                                )
                                                                .to_f64();
                                                        }
                                                        dv_values[value_base + value_index] =
                                                            I::from_f64(beta_value * lam_k);
                                                    }
                                                    let mut beta_gradient = 0.0f64;
                                                    for depth in 0..dk {
                                                        let alpha = logical_value(
                                                            decay_values,
                                                            log_decay,
                                                            key_base + depth,
                                                        )
                                                        .to_f64()
                                                        .exp();
                                                        let mut lam_delta = 0.0f64;
                                                        let mut sdec_lam_k = 0.0f64;
                                                        for value_index in 0..dv {
                                                            let mut lam_k = 0.0f64;
                                                            for inner in 0..dk {
                                                                lam_k += work[lam
                                                                    + inner * dv
                                                                    + value_index]
                                                                    .to_f64()
                                                                    * logical_value(
                                                                        k_values,
                                                                        k,
                                                                        key_base + inner,
                                                                    )
                                                                    .to_f64();
                                                            }
                                                            lam_delta += work
                                                                [lam + depth * dv + value_index]
                                                                .to_f64()
                                                                * work[deltas
                                                                    + local * dv
                                                                    + value_index]
                                                                    .to_f64();
                                                            sdec_lam_k += alpha
                                                                * work[previous
                                                                    + depth * dv
                                                                    + value_index]
                                                                    .to_f64()
                                                                * lam_k;
                                                        }
                                                        dk_values[key_base + depth] = I::from_f64(
                                                            beta_value * (lam_delta - sdec_lam_k),
                                                        );
                                                        beta_gradient += logical_value(
                                                            k_values,
                                                            k,
                                                            key_base + depth,
                                                        )
                                                        .to_f64()
                                                            * lam_delta;

                                                        let mut decay_gradient = 0.0f64;
                                                        for value_index in 0..dv {
                                                            let mut lam_k = 0.0f64;
                                                            for inner in 0..dk {
                                                                lam_k += work[lam
                                                                    + inner * dv
                                                                    + value_index]
                                                                    .to_f64()
                                                                    * logical_value(
                                                                        k_values,
                                                                        k,
                                                                        key_base + inner,
                                                                    )
                                                                    .to_f64();
                                                            }
                                                            let m = work
                                                                [lam + depth * dv + value_index]
                                                                .to_f64()
                                                                - beta_value
                                                                    * logical_value(
                                                                        k_values,
                                                                        k,
                                                                        key_base + depth,
                                                                    )
                                                                    .to_f64()
                                                                    * lam_k;
                                                            decay_gradient += work[previous
                                                                + depth * dv
                                                                + value_index]
                                                                .to_f64()
                                                                * m;
                                                        }
                                                        dlog_values[key_base + depth] =
                                                            I::from_f64(decay_gradient * alpha);
                                                    }
                                                    dbeta_values[bh * steps + step] =
                                                        I::from_f64(beta_gradient);

                                                    for value_index in 0..dv {
                                                        let mut lam_k = 0.0f64;
                                                        for depth in 0..dk {
                                                            lam_k += work
                                                                [lam + depth * dv + value_index]
                                                                .to_f64()
                                                                * logical_value(
                                                                    k_values,
                                                                    k,
                                                                    key_base + depth,
                                                                )
                                                                .to_f64();
                                                        }
                                                        for depth in 0..dk {
                                                            let alpha = logical_value(
                                                                decay_values,
                                                                log_decay,
                                                                key_base + depth,
                                                            )
                                                            .to_f64()
                                                            .exp();
                                                            let index =
                                                                lam + depth * dv + value_index;
                                                            work[index] = W::from_f64(
                                                                alpha
                                                                    * (work[index].to_f64()
                                                                        - beta_value
                                                                            * logical_value(
                                                                                k_values,
                                                                                k,
                                                                                key_base + depth,
                                                                            )
                                                                            .to_f64()
                                                                            * lam_k),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                )
                            },
                        )
                    })
                })
            })
        },
    )??????;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kda_backward_into(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    gradient: &Tensor,
    scale: f64,
    dq: &mut CpuDestination<'_>,
    dk_out: &mut CpuDestination<'_>,
    dv_out: &mut CpuDestination<'_>,
    dlog_decay: &mut CpuDestination<'_>,
    dbeta: &mut CpuDestination<'_>,
    scratch: &mut CpuDestination<'_>,
) -> Result<(), String> {
    let (batch_heads, steps, dk, dv) = kda_geometry(q, k, v, log_decay, beta)?;
    if gradient.dtype() != q.dtype()
        || gradient.shape()[..gradient.shape().len() - 1] != q.shape()[..q.shape().len() - 1]
        || gradient.shape()[gradient.shape().len() - 1] != dv
    {
        return Err("kda backward: gradient shape or dtype mismatch".to_string());
    }
    let chunks = steps.div_ceil(64);
    let state_elements = dk
        .checked_mul(dv)
        .ok_or_else(|| "kda backward: state size overflow".to_string())?;
    let scratch_per_head = chunks
        .checked_add(65)
        .and_then(|states| states.checked_mul(state_elements))
        .and_then(|states| states.checked_add(64usize.checked_mul(dv)?))
        .ok_or_else(|| "kda backward: scratch size overflow".to_string())?;
    match q.dtype() {
        DType::F32 => kda_backward_into_impl::<f32, f32>(
            q,
            k,
            v,
            log_decay,
            beta,
            gradient,
            scale,
            dq,
            dk_out,
            dv_out,
            dlog_decay,
            dbeta,
            scratch,
            batch_heads,
            steps,
            dk,
            dv,
            chunks,
            scratch_per_head,
        ),
        DType::F64 => kda_backward_into_impl::<f64, f64>(
            q,
            k,
            v,
            log_decay,
            beta,
            gradient,
            scale,
            dq,
            dk_out,
            dv_out,
            dlog_decay,
            dbeta,
            scratch,
            batch_heads,
            steps,
            dk,
            dv,
            chunks,
            scratch_per_head,
        ),
        DType::F16 => kda_backward_into_impl::<f16, f32>(
            q,
            k,
            v,
            log_decay,
            beta,
            gradient,
            scale,
            dq,
            dk_out,
            dv_out,
            dlog_decay,
            dbeta,
            scratch,
            batch_heads,
            steps,
            dk,
            dv,
            chunks,
            scratch_per_head,
        ),
        DType::BF16 => kda_backward_into_impl::<bf16, f32>(
            q,
            k,
            v,
            log_decay,
            beta,
            gradient,
            scale,
            dq,
            dk_out,
            dv_out,
            dlog_decay,
            dbeta,
            scratch,
            batch_heads,
            steps,
            dk,
            dv,
            chunks,
            scratch_per_head,
        ),
        _ => unreachable!("kda geometry validated float dtype"),
    }
}

pub fn kda_chunk_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    scale: f64,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    const CHUNK: usize = 64;
    let in_dtype = q.dtype();
    let work = if in_dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let q = q.cast(work);
    let k = k.cast(work);
    let v = v.cast(work);
    let log_decay = log_decay.cast(work);
    let beta = beta.cast(work);
    let g = g.cast(work);

    let dims = q.shape().to_vec();
    let r = dims.len();
    let (t_total, dk) = (dims[r - 2], dims[r - 1]);
    let dv = v.shape()[r - 1];
    let bh: usize = dims[..r - 2].iter().product();

    let q3 = q
        .contiguous()
        .view(Layout::contiguous(vec![bh, t_total, dk]));
    let k3 = k
        .contiguous()
        .view(Layout::contiguous(vec![bh, t_total, dk]));
    let v3 = v
        .contiguous()
        .view(Layout::contiguous(vec![bh, t_total, dv]));
    let ld3 = log_decay
        .contiguous()
        .view(Layout::contiguous(vec![bh, t_total, dk]));
    let b3 = beta
        .contiguous()
        .view(Layout::contiguous(vec![bh, t_total, 1]));
    let g3 = g
        .contiguous()
        .view(Layout::contiguous(vec![bh, t_total, dv]));

    let tok = |x: &Tensor, t: usize| narrow(x, 1, t, 1); // [BH, 1, D]
    let col = |x: &Tensor, t: usize| transpose_last2(&narrow(x, 1, t, 1)); // [BH, D, 1]

    // Pass 1: chunk start states via the per-token recurrence.
    let mut starts: Vec<Tensor> = Vec::new();
    let mut state = Tensor::zeros(&[bh, dk, dv], work);
    let mut t0 = 0;
    while t0 < t_total {
        let c = CHUNK.min(t_total - t0);
        starts.push(state.clone());
        for i in 0..c {
            let alpha = transpose_last2(&tok(&ld3, t0 + i).exp()); // [BH, dk, 1]
            let sdec = state.mul(&alpha);
            let k_col = col(&k3, t0 + i);
            let delta = col(&v3, t0 + i).sub(&transpose_last2(&sdec).matmul(&k_col)); // [BH, dv, 1]
            state = sdec.add(&k_col.matmul(&transpose_last2(&delta).mul(&tok(&b3, t0 + i))));
        }
        t0 += c;
    }

    // Pass 2: reverse adjoint walk with per-chunk forward recompute.
    let mut lam = Tensor::zeros(&[bh, dk, dv], work);
    let mut dq_rows: Vec<Tensor> = Vec::new();
    let mut dk_rows: Vec<Tensor> = Vec::new();
    let mut dv_rows: Vec<Tensor> = Vec::new();
    let mut dg_rows: Vec<Tensor> = Vec::new();
    let mut db_rows: Vec<Tensor> = Vec::new();
    for ci in (0..starts.len()).rev() {
        let t0 = ci * CHUNK;
        let c = CHUNK.min(t_total - t0);
        // Recompute this chunk's per-token states.
        let mut sdecs: Vec<Tensor> = Vec::with_capacity(c);
        let mut states_t: Vec<Tensor> = Vec::with_capacity(c);
        let mut deltas: Vec<Tensor> = Vec::with_capacity(c);
        let mut s = starts[ci].clone();
        for i in 0..c {
            let alpha = transpose_last2(&tok(&ld3, t0 + i).exp());
            let sdec = s.mul(&alpha);
            let k_col = col(&k3, t0 + i);
            let delta = col(&v3, t0 + i).sub(&transpose_last2(&sdec).matmul(&k_col));
            s = sdec.add(&k_col.matmul(&transpose_last2(&delta).mul(&tok(&b3, t0 + i))));
            sdecs.push(sdec);
            deltas.push(delta);
            states_t.push(s.clone());
        }
        for i in (0..c).rev() {
            let t = t0 + i;
            let q_col = col(&q3, t); // [BH, dk, 1]
            let g_row = tok(&g3, t); // [BH, 1, dv]
            let k_col = col(&k3, t);
            let b_t = tok(&b3, t); // [BH, 1, 1]
            lam = lam.add(
                &q_col
                    .matmul(&g_row)
                    .mul(&Tensor::full(&[bh, dk, dv], scale, work)),
            );
            let g_col = transpose_last2(&g_row); // [BH, dv, 1]
            dq_rows.push(transpose_last2(
                &states_t[i]
                    .matmul(&g_col)
                    .mul(&Tensor::full(&[bh, dk, 1], scale, work)),
            ));
            let lam_k = transpose_last2(&lam).matmul(&k_col); // [BH, dv, 1]
            dv_rows.push(transpose_last2(&lam_k.mul(&b_t)));
            let lam_delta = lam.matmul(&deltas[i]); // [BH, dk, 1]
            dk_rows.push(transpose_last2(
                &lam_delta
                    .sub(&sdecs[i].matmul(&lam_k))
                    .mul(&transpose_last2(&b_t)),
            ));
            db_rows.push(k_col.mul(&lam_delta).sum(&[1])); // [BH, 1, 1]
                                                           // M = (I - beta k kᵀ) Λ
            let m_ = lam.sub(
                &k_col
                    .matmul(&transpose_last2(&lam_k))
                    .mul(&transpose_last2(&b_t)),
            );
            let s_prev = if i == 0 {
                &starts[ci]
            } else {
                &states_t[i - 1]
            };
            let dalpha = s_prev.mul(&m_).sum(&[2]); // [BH, dk, 1]
            let alpha = transpose_last2(&tok(&ld3, t).exp());
            dg_rows.push(transpose_last2(&dalpha.mul(&alpha)));
            lam = alpha.mul(&m_);
        }
    }
    for rows in [
        &mut dq_rows,
        &mut dk_rows,
        &mut dv_rows,
        &mut dg_rows,
        &mut db_rows,
    ] {
        rows.reverse();
    }
    let assemble = |rows: Vec<Tensor>, width: usize| {
        let mut shape = dims.clone();
        shape[r - 1] = width;
        Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)
            .contiguous()
            .view(Layout::contiguous(shape))
            .cast(in_dtype)
    };
    (
        assemble(dq_rows, dk),
        assemble(dk_rows, dk),
        assemble(dv_rows, dv),
        assemble(dg_rows, dk),
        assemble(db_rows, 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ExecutableAllocationGuard;

    fn f32_data(t: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(v) = &t.buffer else {
            panic!()
        };
        v.as_slice().to_vec()
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) {
        if actual.shape() != expected.shape() {
            assert_eq!((actual.numel(), expected.numel()), (1, 1));
        }
        for (actual, expected) in f32_data(actual).iter().zip(f32_data(expected)) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} vs {expected} (tolerance {tolerance})"
            );
        }
    }

    // Deterministic pseudo-random f32 in [-1, 1] (xorshift).
    fn prand(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    // Per-token gated delta-rule recurrence, the ground truth for the
    // chunked implementation. Inputs [BH, T, D] f32.
    fn naive_kda(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        log_decay: &Tensor,
        beta: &Tensor,
        scale: f64,
    ) -> Tensor {
        let dims = q.shape();
        let (bh, t, dk) = (dims[0], dims[1], dims[2]);
        let dv = v.shape()[2];
        let mut state = Tensor::zeros(&[bh, dk, dv], DType::F32);
        let mut outs = Vec::new();
        for i in 0..t {
            let q_t = narrow(q, 1, i, 1); // [BH,1,dk]
            let k_t = narrow(k, 1, i, 1);
            let v_t = narrow(v, 1, i, 1); // [BH,1,dv]
            let alpha = narrow(log_decay, 1, i, 1).exp(); // [BH,1,dk]
            let b_t = narrow(beta, 1, i, 1); // [BH,1,1]
            let sd = state.mul(&transpose_last2(&alpha)); // Diag(alpha) S
            let kv_mem = transpose_last2(&sd).matmul(&transpose_last2(&k_t)); // [BH,dv,1]
            let delta = transpose_last2(&v_t).sub(&kv_mem).mul(&b_t); // [BH,dv,1]
            state = sd.add(&transpose_last2(&k_t).matmul(&transpose_last2(&delta)));
            let o = transpose_last2(&state)
                .matmul(&transpose_last2(&q_t))
                .mul(&Tensor::full(&[bh, dv, 1], scale, DType::F32));
            outs.push(transpose_last2(&o)); // [BH,1,dv]
        }
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1)
    }

    fn kda_case(t: usize, dk: usize, dv: usize, seed: u64) {
        let bh = 2;
        let q = Tensor::from_vec(prand(bh * t * dk, seed), vec![bh, t, dk]);
        let k = Tensor::from_vec(prand(bh * t * dk, seed + 1), vec![bh, t, dk]);
        let v = Tensor::from_vec(prand(bh * t * dv, seed + 2), vec![bh, t, dv]);
        // Log decays in [-3, -0.05]: a realistic gate range.
        let ld: Vec<f32> = prand(bh * t * dk, seed + 3)
            .into_iter()
            .map(|x| (x.abs() + 0.05) * -1.5)
            .collect();
        let ld = Tensor::from_vec(ld, vec![bh, t, dk]);
        let beta: Vec<f32> = prand(bh * t, seed + 4)
            .into_iter()
            .map(|x| x.abs() * 0.9 + 0.05)
            .collect();
        let beta = Tensor::from_vec(beta, vec![bh, t, 1]);
        let scale = 1.0 / (dk as f64).sqrt();

        let chunked = kda_chunk_forward(&q, &k, &v, &ld, &beta, scale);
        let naive = naive_kda(&q, &k, &v, &ld, &beta, scale);
        let (a, b) = (f32_data(&chunked), f32_data(&naive));
        assert_eq!(a.len(), b.len());
        let mut worst = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let rel = (x - y).abs() / y.abs().max(1e-3);
            worst = worst.max(rel);
        }
        assert!(
            worst < 5e-4,
            "kda chunk vs naive mismatch at t={t} dk={dk} dv={dv}: worst rel {worst}"
        );
    }

    #[test]
    fn kda_chunk_matches_naive_within_sub_chunk() {
        kda_case(13, 8, 8, 7);
    }

    #[test]
    fn kda_chunk_matches_naive_across_sub_chunks() {
        kda_case(40, 8, 16, 11);
    }

    #[test]
    fn kda_chunk_matches_naive_across_chunks_ragged() {
        kda_case(70, 16, 8, 13);
    }

    #[test]
    fn kda_chunk_matches_naive_multi_chunk() {
        kda_case(129, 16, 16, 17);
    }

    // Central finite differences of the forward are the oracle for the
    // closed-form adjoint (f64 for tight tolerances).
    fn kda_backward_case(t: usize, dk: usize, dv: usize, seed: u64) {
        let bh = 1;
        let f64_data = |t_: &Tensor| {
            let CpuBuffer::F64(v) = &t_.buffer else {
                panic!()
            };
            v.as_slice().to_vec()
        };
        let q = Tensor::from_vec(prand(bh * t * dk, seed), vec![bh, t, dk]).cast(DType::F64);
        let k = Tensor::from_vec(prand(bh * t * dk, seed + 1), vec![bh, t, dk]).cast(DType::F64);
        let v = Tensor::from_vec(prand(bh * t * dv, seed + 2), vec![bh, t, dv]).cast(DType::F64);
        let ld = Tensor::from_vec(
            prand(bh * t * dk, seed + 3)
                .into_iter()
                .map(|x| (x.abs() + 0.05) * -1.5)
                .collect(),
            vec![bh, t, dk],
        )
        .cast(DType::F64);
        let beta = Tensor::from_vec(
            prand(bh * t, seed + 4)
                .into_iter()
                .map(|x| x.abs() * 0.9 + 0.05)
                .collect(),
            vec![bh, t, 1],
        )
        .cast(DType::F64);
        let w = Tensor::from_vec(prand(bh * t * dv, seed + 5), vec![bh, t, dv]).cast(DType::F64);
        let scale = 1.0 / (dk as f64).sqrt();
        let loss = |q_: &Tensor, k_: &Tensor, v_: &Tensor, ld_: &Tensor, b_: &Tensor| -> f64 {
            let out = kda_chunk_forward(q_, k_, v_, ld_, b_, scale);
            f64_data(&out.mul(&w).sum(&[0, 1, 2]))[0]
        };
        let (dq, dk_, dv_, dld, db) = kda_chunk_backward(&q, &k, &v, &ld, &beta, &w, scale);
        let eps = 1e-6;
        let fd = |input: &Tensor, analytic: &Tensor, which: usize, name: &str| {
            let base = f64_data(input);
            let got = f64_data(analytic);
            let shape = input.shape().to_vec();
            let mut local_worst = 0f64;
            for i in 0..base.len() {
                let mut plus = base.clone();
                plus[i] += eps;
                let mut minus = base.clone();
                minus[i] -= eps;
                let plus_t = Tensor::from_vec(plus, shape.clone());
                let minus_t = Tensor::from_vec(minus, shape.clone());
                let (lp, lm) = match which {
                    0 => (
                        loss(&plus_t, &k, &v, &ld, &beta),
                        loss(&minus_t, &k, &v, &ld, &beta),
                    ),
                    1 => (
                        loss(&q, &plus_t, &v, &ld, &beta),
                        loss(&q, &minus_t, &v, &ld, &beta),
                    ),
                    2 => (
                        loss(&q, &k, &plus_t, &ld, &beta),
                        loss(&q, &k, &minus_t, &ld, &beta),
                    ),
                    3 => (
                        loss(&q, &k, &v, &plus_t, &beta),
                        loss(&q, &k, &v, &minus_t, &beta),
                    ),
                    _ => (
                        loss(&q, &k, &v, &ld, &plus_t),
                        loss(&q, &k, &v, &ld, &minus_t),
                    ),
                };
                let numeric = (lp - lm) / (2.0 * eps);
                let rel = (numeric - got[i]).abs() / got[i].abs().max(1e-6);
                local_worst = local_worst.max(rel);
                assert!(
                    rel < 1e-4,
                    "{name}[{i}]: analytic {} vs numeric {numeric}",
                    got[i]
                );
            }
            local_worst
        };
        let w0 = fd(&q, &dq, 0, "dq");
        let w1 = fd(&k, &dk_, 1, "dk");
        let w2 = fd(&v, &dv_, 2, "dv");
        let w3 = fd(&ld, &dld, 3, "dlog_decay");
        let w4 = fd(&beta, &db, 4, "dbeta");
        for (name, w_) in [("dq", w0), ("dk", w1), ("dv", w2), ("dld", w3), ("db", w4)] {
            eprintln!("  t={t} {name} worst rel {w_:e}");
        }
    }

    #[test]
    fn kda_backward_matches_finite_differences() {
        kda_backward_case(8, 4, 6, 21);
    }

    #[test]
    fn kda_backward_matches_finite_differences_across_chunk() {
        kda_backward_case(70, 4, 4, 23);
    }

    #[test]
    fn unit_lower_inverse_inverts() {
        // (I + a) with strictly-lower a; x = (I + a)^-1 must satisfy
        // (I + a) x = I.
        let mut a = vec![0f32; 4 * 4];
        a[4] = 0.5;
        a[8] = -0.25;
        a[9] = 0.75;
        a[12] = 0.1;
        a[13] = -0.4;
        a[14] = 0.3;
        let a = Tensor::from_vec(a, vec![1, 4, 4]);
        let x = unit_lower_inverse(&a);
        let prod = eye(4, DType::F32).add(&a).matmul(&x);
        let d = f32_data(&prod);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((d[i * 4 + j] - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 1., 1., 1.], vec![2, 3]);
        let p = softmax_lastdim(&x);
        let d = f32_data(&p);
        assert!((d[0] + d[1] + d[2] - 1.0).abs() < 1e-6);
        assert!((d[3] + d[4] + d[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn causal_mask_blocks_future() {
        let m = causal_additive_mask(3, 3, DType::F32);
        let d = f32_data(&m);
        assert_eq!(d[0], 0.0);
        assert!(d[1].is_infinite() && d[1] < 0.0);
        assert_eq!(d[4], 0.0);
    }

    #[test]
    fn layer_norm_normalizes() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![1, 4]);
        let w = Tensor::ones(&[4], DType::F32);
        let b = Tensor::zeros(&[4], DType::F32);
        let y = layer_norm_forward(&x, &w, &b, 1e-5);
        let d = f32_data(&y);
        let mean: f32 = d.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        let (dx, dw, db) = layer_norm_backward(&x, &w, &Tensor::ones(&[1, 4], DType::F32), 1e-5);
        assert_eq!(dx.shape(), &[1, 4]);
        assert_eq!(dw.shape(), &[4]);
        assert_eq!(db.shape(), &[4]);
    }

    #[test]
    fn cross_entropy_matches_log_softmax() {
        let logits = Tensor::from_vec(vec![2f32, 0., -1., 0., 3., 1.], vec![2, 3]);
        let target = Tensor::from_vec(vec![0i64, 1], vec![2]);
        let loss =
            cross_entropy_forward(&logits, &target, -100, CrossEntropyReduction::Mean).unwrap();
        let lse = logsumexp_lastdim(&logits);
        let CpuBuffer::F32(v) = &lse.buffer else {
            panic!()
        };
        let expect = ((v[0] - 2.0) + (v[1] - 3.0)) / 2.0;
        let got = f32_data(&loss)[0];
        assert!((got - expect).abs() < 1e-5, "{got} vs {expect}");
    }

    #[test]
    fn sdpa_shapes_and_backward() {
        let q = Tensor::from_vec(vec![0.1f32; 2 * 4 * 8], vec![1, 2, 4, 8]);
        let k = Tensor::from_vec(vec![0.2f32; 2 * 4 * 8], vec![1, 2, 4, 8]);
        let v = Tensor::from_vec(vec![0.3f32; 2 * 4 * 8], vec![1, 2, 4, 8]);
        let o = sdpa_forward(&q, &k, &v, 0.35, true);
        assert_eq!(o.shape(), &[1, 2, 4, 8]);
        let g = Tensor::from_vec(vec![1f32; 2 * 4 * 8], vec![1, 2, 4, 8]);
        let (dq, dk, dv) = sdpa_backward(&q, &k, &v, &g, 0.35, true);
        assert_eq!(dq.shape(), q.shape());
        assert_eq!(dk.shape(), k.shape());
        assert_eq!(dv.shape(), v.shape());
    }

    #[test]
    fn destination_requirements_are_exact_and_topologies_are_fixed() {
        let logits = Tensor::zeros(&[2, 3, 5], DType::F32);
        let target = Tensor::zeros(&[2, 3], DType::U32);
        let ce = cross_entropy_forward_requirements(&logits, &target, CrossEntropyReduction::Mean)
            .unwrap();
        assert_eq!(ce.loss.bytes, 4);
        assert_eq!(ce.status.bytes, 3 * 8);
        assert_eq!(ce.nll_scratch.bytes, 6 * 8);
        assert_eq!(ce.flags_scratch.bytes, 6 * 4);
        assert_eq!(
            ce.topology,
            CrossEntropyForwardTopology::RowsThenStatus {
                row_passes: 2,
                status_passes: 1,
            }
        );

        let x = Tensor::zeros(&[2, 4], DType::F32);
        let weight = Tensor::zeros(&[4, 7], DType::F32);
        let bias = Tensor::zeros(&[7], DType::F32);
        let head_target = Tensor::zeros(&[2], DType::I64);
        let head =
            chunked_head_ce_backward_requirements(&x, &weight, &bias, &head_target, 4).unwrap();
        assert_eq!(head.dx.bytes, 2 * 4 * 4);
        assert_eq!(head.dweight.bytes, 4 * 7 * 4);
        assert_eq!(head.dbias.bytes, 7 * 4);
        assert_eq!(head.logits_scratch.bytes, head.chunk_len * 7 * 4);
        assert_eq!(head.grad_logits_scratch.bytes, head.chunk_len * 7 * 4);

        let q = Tensor::zeros(&[2, 3, 5, 4], DType::F32);
        let k = Tensor::zeros(&[2, 3, 7, 4], DType::F32);
        let v = Tensor::zeros(&[2, 3, 7, 6], DType::F32);
        let attention = sdpa_forward_requirements(&q, &k, &v).unwrap();
        assert_eq!(attention.output.shape, [2, 3, 5, 6]);
        assert_eq!(attention.logsumexp.shape, [2, 3, 5]);
        assert_eq!(attention.logsumexp.bytes, 2 * 3 * 5 * 4);

        let ln_weight = Tensor::ones(&[3, 4], DType::F32);
        let ln_bias = Tensor::zeros(&[3, 4], DType::F32);
        let ln_x = Tensor::zeros(&[2, 3, 4], DType::F32);
        let ln = layer_norm_forward_requirements(&ln_x, &ln_weight, &ln_bias).unwrap();
        assert_eq!(ln.rows, 2);
        assert_eq!(ln.normalized_elements, 12);
        assert_eq!(ln.output.bytes, 2 * 3 * 4 * 4);

        let decay = Tensor::zeros(&[2, 5, 4], DType::F32);
        let beta = Tensor::zeros(&[2, 5, 1], DType::F32);
        let kda_v = Tensor::zeros(&[2, 5, 3], DType::F32);
        let kda = kda_backward_requirements(&decay, &decay, &kda_v, &decay, &beta).unwrap();
        let state = 4 * 3;
        let per_head = (1 + 65) * state + 64 * 3;
        assert_eq!(kda.scratch.bytes, 2 * per_head * 4);
        assert_eq!(
            kda.topology,
            KdaTopology::BackwardChunkRecompute {
                chunk: 64,
                passes: 3,
            }
        );

        let short_weight = Tensor::zeros(&[4, 3], DType::F32);
        let short = short_conv1d_forward_requirements(&ln_x, &short_weight, false).unwrap();
        assert_eq!(short.output.bytes, ln_x.numel() * 4);
        assert!(short.state_next.is_none());
        let rotary = rotary_forward_requirements(&q).unwrap();
        assert_eq!(rotary.output.bytes, q.numel() * 4);
        assert_eq!(rotary.topology, RotaryTopology::Pairs { passes: 1 });
    }

    #[test]
    fn ce_sdpa_and_layer_norm_destinations_match_under_guard() {
        let logits = Tensor::from_vec(
            vec![0.2f32, -0.4, 1.1, 0.7, -1.0, 0.3, 0.8, -0.2],
            vec![2, 4],
        );
        let target = Tensor::from_vec(vec![2u32, 99], vec![2]);
        let ce_req =
            cross_entropy_forward_requirements(&logits, &target, CrossEntropyReduction::Mean)
                .unwrap();
        let ce_bwd_req =
            cross_entropy_backward_requirements(&logits, &target, CrossEntropyReduction::Mean)
                .unwrap();
        let mut loss = Tensor::empty(&ce_req.loss.shape, ce_req.loss.dtype);
        let mut status = Tensor::empty(&ce_req.status.shape, ce_req.status.dtype);
        let mut nll = Tensor::empty(&ce_req.nll_scratch.shape, ce_req.nll_scratch.dtype);
        let mut flags = Tensor::empty(&ce_req.flags_scratch.shape, ce_req.flags_scratch.dtype);
        let mut ce_grad = Tensor::empty(&ce_bwd_req.grad.shape, ce_bwd_req.grad.dtype);
        let count_req = ce_bwd_req.count_status.as_ref().unwrap();
        let mut count = Tensor::empty(&count_req.shape, count_req.dtype);

        let q = Tensor::from_vec(prand(2 * 3, 101), vec![1, 1, 2, 3]);
        let k = Tensor::from_vec(prand(4 * 3, 102), vec![1, 1, 4, 3]);
        let v = Tensor::from_vec(prand(4 * 2, 103), vec![1, 1, 4, 2]);
        let gradient = Tensor::from_vec(prand(2 * 2, 104), vec![1, 1, 2, 2]);
        let sdpa_req = sdpa_forward_requirements(&q, &k, &v).unwrap();
        let sdpa_bwd_req = sdpa_backward_requirements(&q, &k, &v).unwrap();
        let mut attention = Tensor::empty(&sdpa_req.output.shape, sdpa_req.output.dtype);
        let mut lse = Tensor::empty(&sdpa_req.logsumexp.shape, sdpa_req.logsumexp.dtype);
        let mut dq = Tensor::empty(&sdpa_bwd_req.dq.shape, sdpa_bwd_req.dq.dtype);
        let mut dk = Tensor::empty(&sdpa_bwd_req.dk.shape, sdpa_bwd_req.dk.dtype);
        let mut dv = Tensor::empty(&sdpa_bwd_req.dv.shape, sdpa_bwd_req.dv.dtype);
        let mut d_vec = Tensor::empty(
            &sdpa_bwd_req.d_vec_scratch.shape,
            sdpa_bwd_req.d_vec_scratch.dtype,
        );

        let ln_x = Tensor::from_vec(prand(24, 110), vec![2, 3, 4]);
        let ln_weight = Tensor::from_vec(prand(12, 111), vec![3, 4]);
        let ln_bias = Tensor::from_vec(prand(12, 112), vec![3, 4]);
        let ln_gradient = Tensor::from_vec(prand(24, 113), vec![2, 3, 4]);
        let ln_req = layer_norm_forward_requirements(&ln_x, &ln_weight, &ln_bias).unwrap();
        let ln_bwd_req = layer_norm_backward_requirements(&ln_x, &ln_weight, &ln_gradient).unwrap();
        let mut ln_output = Tensor::empty(&ln_req.output.shape, ln_req.output.dtype);
        let mut ln_dx = Tensor::empty(&ln_bwd_req.dx.shape, ln_bwd_req.dx.dtype);
        let mut ln_dw = Tensor::empty(&ln_bwd_req.dweight.shape, ln_bwd_req.dweight.dtype);
        let mut ln_db = Tensor::empty(&ln_bwd_req.dbias.shape, ln_bwd_req.dbias.dtype);
        let mut normalized = Tensor::empty(
            &ln_bwd_req.normalized_scratch.shape,
            ln_bwd_req.normalized_scratch.dtype,
        );

        {
            let _guard = ExecutableAllocationGuard::enter();
            cross_entropy_forward_into(
                &logits,
                &target,
                99,
                CrossEntropyReduction::Mean,
                &mut loss.destination().unwrap(),
                &mut status.destination().unwrap(),
                &mut nll.destination().unwrap(),
                &mut flags.destination().unwrap(),
            )
            .unwrap();
            cross_entropy_backward_into(
                &logits,
                &target,
                99,
                CrossEntropyReduction::Mean,
                &mut ce_grad.destination().unwrap(),
                Some(&mut count.destination().unwrap()),
            )
            .unwrap();
            sdpa_forward_into(
                &q,
                &k,
                &v,
                0.5,
                true,
                &mut attention.destination().unwrap(),
                &mut lse.destination().unwrap(),
            )
            .unwrap();
            sdpa_backward_into(
                &q,
                &k,
                &v,
                &attention,
                &lse,
                &gradient,
                0.5,
                true,
                &mut dq.destination().unwrap(),
                &mut dk.destination().unwrap(),
                &mut dv.destination().unwrap(),
                &mut d_vec.destination().unwrap(),
            )
            .unwrap();
            layer_norm_forward_into(
                &ln_x,
                &ln_weight,
                &ln_bias,
                1e-5,
                &mut ln_output.destination().unwrap(),
            )
            .unwrap();
            layer_norm_backward_into(
                &ln_x,
                &ln_weight,
                &ln_gradient,
                1e-5,
                &mut ln_dx.destination().unwrap(),
                &mut ln_dw.destination().unwrap(),
                &mut ln_db.destination().unwrap(),
                &mut normalized.destination().unwrap(),
            )
            .unwrap();
        }

        assert_close(
            &loss,
            &cross_entropy_forward(&logits, &target, 99, CrossEntropyReduction::Mean).unwrap(),
            1e-5,
        );
        assert_close(
            &ce_grad,
            &cross_entropy_backward(&logits, &target, 99, CrossEntropyReduction::Mean).unwrap(),
            1e-5,
        );
        assert_close(&attention, &sdpa_forward(&q, &k, &v, 0.5, true), 2e-5);
        let (expected_dq, expected_dk, expected_dv) =
            sdpa_backward(&q, &k, &v, &gradient, 0.5, true);
        assert_close(&dq, &expected_dq, 3e-5);
        assert_close(&dk, &expected_dk, 3e-5);
        assert_close(&dv, &expected_dv, 3e-5);
        assert_close(
            &ln_output,
            &layer_norm_forward(&ln_x, &ln_weight, &ln_bias, 1e-5),
            3e-5,
        );
        let (expected_dx, expected_dw, expected_db) =
            layer_norm_backward(&ln_x, &ln_weight, &ln_gradient, 1e-5);
        assert_close(&ln_dx, &expected_dx, 3e-5);
        assert_close(&ln_dw, &expected_dw, 3e-5);
        assert_close(&ln_db, &expected_db, 3e-5);
    }

    #[test]
    fn chunked_head_optimizer_rotary_and_short_conv_match_under_guard() {
        let x = Tensor::from_vec(prand(8, 201), vec![4, 2]);
        let weight = Tensor::from_vec(prand(6, 202), vec![2, 3]);
        let bias = Tensor::from_vec(prand(3, 203), vec![3]);
        let target = Tensor::from_vec(vec![0i64, 2, -100, 1], vec![4]);
        let scalar_gradient = Tensor::from_vec(vec![0.7f32], vec![]);
        let forward = chunked_head_ce_forward_requirements(&x, &weight, &bias, &target, 4).unwrap();
        let backward =
            chunked_head_ce_backward_requirements(&x, &weight, &bias, &target, 4).unwrap();
        let mut loss = Tensor::empty(&forward.loss.shape, forward.loss.dtype);
        let mut status = Tensor::empty(&forward.status.shape, forward.status.dtype);
        let mut logits = Tensor::empty(&forward.logits_scratch.shape, forward.logits_scratch.dtype);
        let mut nll = Tensor::empty(&forward.nll_scratch.shape, forward.nll_scratch.dtype);
        let mut flags = Tensor::empty(&forward.flags_scratch.shape, forward.flags_scratch.dtype);
        let mut dx = Tensor::empty(&backward.dx.shape, backward.dx.dtype);
        let mut dw = Tensor::empty(&backward.dweight.shape, backward.dweight.dtype);
        let mut db = Tensor::empty(&backward.dbias.shape, backward.dbias.dtype);
        let mut backward_status = Tensor::empty(&backward.status.shape, backward.status.dtype);
        let mut backward_logits = Tensor::empty(
            &backward.logits_scratch.shape,
            backward.logits_scratch.dtype,
        );
        let mut grad_logits = Tensor::empty(
            &backward.grad_logits_scratch.shape,
            backward.grad_logits_scratch.dtype,
        );
        let mut dw_scratch = Tensor::empty(
            &backward.dweight_scratch.shape,
            backward.dweight_scratch.dtype,
        );
        let mut db_scratch =
            Tensor::empty(&backward.dbias_scratch.shape, backward.dbias_scratch.dtype);

        let parameter = Tensor::from_vec(prand(6, 210), vec![2, 3]);
        let parameter_gradient = Tensor::from_vec(prand(6, 211), vec![2, 3]);
        let moment = Tensor::from_vec(prand(6, 212), vec![2, 3]);
        let variance = Tensor::from_vec(
            prand(6, 213).into_iter().map(|value| value.abs()).collect(),
            vec![2, 3],
        );
        let lr = Tensor::from_vec(vec![0.01f32], vec![]);
        let c1 = Tensor::from_vec(vec![0.2f32], vec![]);
        let c2 = Tensor::from_vec(vec![0.3f32], vec![]);
        let first = Tensor::from_vec(vec![0.0f32], vec![]);
        let adam =
            adamw_step_requirements(&parameter, &parameter_gradient, &moment, &variance).unwrap();
        let sgd = sgd_step_requirements(&parameter, &parameter_gradient, &moment).unwrap();
        let mut adam_p = Tensor::empty(&adam.param.shape, adam.param.dtype);
        let mut adam_m = Tensor::empty(&adam.first_moment.shape, adam.first_moment.dtype);
        let mut adam_v = Tensor::empty(&adam.second_moment.shape, adam.second_moment.dtype);
        let mut sgd_p = Tensor::empty(&sgd.param.shape, sgd.param.dtype);
        let mut sgd_v = Tensor::empty(&sgd.velocity.shape, sgd.velocity.dtype);

        let rotary_x = Tensor::from_vec(prand(2 * 3 * 6, 220), vec![2, 3, 6]);
        let rotary_req = rotary_forward_requirements(&rotary_x).unwrap();
        let mut rotary_output = Tensor::empty(&rotary_req.output.shape, rotary_req.output.dtype);

        let conv_x = Tensor::from_vec(prand(2 * 5 * 3, 230), vec![2, 5, 3]);
        let conv_weight = Tensor::from_vec(prand(3 * 4, 231), vec![3, 4]);
        let conv_gradient = Tensor::from_vec(prand(2 * 5 * 3, 232), vec![2, 5, 3]);
        let conv_req =
            short_conv1d_backward_requirements(&conv_x, &conv_weight, &conv_gradient).unwrap();
        let mut conv_output = Tensor::empty(conv_x.shape(), conv_x.dtype());
        let mut conv_dx = Tensor::empty(&conv_req.dx.shape, conv_req.dx.dtype);
        let mut conv_dw = Tensor::empty(&conv_req.dweight.shape, conv_req.dweight.dtype);
        let state_x = Tensor::from_vec(prand(5 * 3, 233), vec![5, 3]);
        let state = Tensor::from_vec(prand(3 * 3, 234), vec![3, 3]);
        let state_req = short_conv1d_state_requirements(&state_x, &conv_weight).unwrap();
        let mut state_output = Tensor::empty(&state_req.output.shape, state_req.output.dtype);
        let state_next_req = state_req.state_next.as_ref().unwrap();
        let mut state_next = Tensor::empty(&state_next_req.shape, state_next_req.dtype);

        {
            let _guard = ExecutableAllocationGuard::enter();
            chunked_head_ce_forward_into(
                &x,
                &weight,
                &bias,
                &target,
                -100,
                4,
                &mut loss.destination().unwrap(),
                &mut status.destination().unwrap(),
                &mut logits.destination().unwrap(),
                &mut nll.destination().unwrap(),
                &mut flags.destination().unwrap(),
            )
            .unwrap();
            chunked_head_ce_backward_into(
                &x,
                &weight,
                &bias,
                &target,
                &scalar_gradient,
                -100,
                4,
                &mut dx.destination().unwrap(),
                &mut dw.destination().unwrap(),
                &mut db.destination().unwrap(),
                &mut backward_status.destination().unwrap(),
                &mut backward_logits.destination().unwrap(),
                &mut grad_logits.destination().unwrap(),
                &mut dw_scratch.destination().unwrap(),
                &mut db_scratch.destination().unwrap(),
            )
            .unwrap();
            adamw_step_into(
                &parameter,
                &parameter_gradient,
                &moment,
                &variance,
                &lr,
                &c1,
                &c2,
                0.9,
                0.99,
                1e-8,
                0.1,
                &mut adam_p.destination().unwrap(),
                &mut adam_m.destination().unwrap(),
                &mut adam_v.destination().unwrap(),
            )
            .unwrap();
            sgd_step_into(
                &parameter,
                &parameter_gradient,
                &moment,
                &lr,
                &first,
                0.8,
                0.1,
                true,
                0.05,
                &mut sgd_p.destination().unwrap(),
                &mut sgd_v.destination().unwrap(),
            )
            .unwrap();
            rotary_forward_into(
                &rotary_x,
                &[2, 5],
                10_000.0,
                1.0,
                &mut rotary_output.destination().unwrap(),
            )
            .unwrap();
            short_conv1d_forward_into(
                &conv_x,
                &conv_weight,
                &mut conv_output.destination().unwrap(),
            )
            .unwrap();
            short_conv1d_backward_into(
                &conv_x,
                &conv_weight,
                &conv_gradient,
                &mut conv_dx.destination().unwrap(),
                &mut conv_dw.destination().unwrap(),
            )
            .unwrap();
            short_conv1d_with_state_into(
                &state_x,
                &conv_weight,
                &state,
                3,
                &mut state_output.destination().unwrap(),
                &mut state_next.destination().unwrap(),
            )
            .unwrap();
        }

        assert_close(
            &loss,
            &chunked_head_ce_forward(&x, &weight, &bias, &target, -100, 4).unwrap(),
            2e-5,
        );
        let (expected_dx, expected_dw, expected_db) =
            chunked_head_ce_backward(&x, &weight, &bias, &target, &scalar_gradient, -100, 4)
                .unwrap();
        assert_close(&dx, &expected_dx, 3e-5);
        assert_close(&dw, &expected_dw, 3e-5);
        assert_close(&db, &expected_db, 3e-5);
        let (expected_p, expected_m, expected_v) = adamw_step(
            &parameter,
            &parameter_gradient,
            &moment,
            &variance,
            &lr,
            &c1,
            &c2,
            0.9,
            0.99,
            1e-8,
            0.1,
        );
        assert_close(&adam_p, &expected_p, 2e-5);
        assert_close(&adam_m, &expected_m, 2e-5);
        assert_close(&adam_v, &expected_v, 2e-5);
        let (expected_p, expected_v) = sgd_step(
            &parameter,
            &parameter_gradient,
            &moment,
            &lr,
            &first,
            0.8,
            0.1,
            true,
            0.05,
        );
        assert_close(&sgd_p, &expected_p, 2e-5);
        assert_close(&sgd_v, &expected_v, 2e-5);
        assert_close(
            &rotary_output,
            &rotary_forward(&rotary_x, &[2, 5], 10_000.0, 1.0).unwrap(),
            3e-5,
        );
        assert_close(
            &conv_output,
            &short_conv1d_forward(&conv_x, &conv_weight),
            2e-5,
        );
        assert_close(
            &conv_dx,
            &short_conv1d_backward_x(&conv_x, &conv_weight, &conv_gradient),
            2e-5,
        );
        assert_close(
            &conv_dw,
            &short_conv1d_backward_w(&conv_x, &conv_weight, &conv_gradient),
            2e-5,
        );
        let (expected_output, expected_state) =
            short_conv1d_with_state(&state_x, &conv_weight, &state, 3);
        assert_close(&state_output, &expected_output, 2e-5);
        assert_close(&state_next, &expected_state, 1e-6);
    }

    #[test]
    fn kda_forward_backward_and_decode_destinations_match_under_guard() {
        let (bh, steps, dk, dv) = (1, 70, 3, 2);
        let q = Tensor::from_vec(prand(bh * steps * dk, 301), vec![bh, steps, dk]);
        let k = Tensor::from_vec(prand(bh * steps * dk, 302), vec![bh, steps, dk]);
        let v = Tensor::from_vec(prand(bh * steps * dv, 303), vec![bh, steps, dv]);
        let decay = Tensor::from_vec(
            prand(bh * steps * dk, 304)
                .into_iter()
                .map(|value| -(value.abs() + 0.1))
                .collect(),
            vec![bh, steps, dk],
        );
        let beta = Tensor::from_vec(
            prand(bh * steps, 305)
                .into_iter()
                .map(|value| value.abs() * 0.5)
                .collect(),
            vec![bh, steps, 1],
        );
        let gradient = Tensor::from_vec(prand(bh * steps * dv, 306), vec![bh, steps, dv]);
        let scale = 1.0 / (dk as f64).sqrt();
        let forward = kda_forward_requirements(&q, &k, &v, &decay, &beta, false).unwrap();
        let backward = kda_backward_requirements(&q, &k, &v, &decay, &beta).unwrap();
        let mut output = Tensor::empty(&forward.output.shape, forward.output.dtype);
        let state_req = forward.state_scratch.as_ref().unwrap();
        let mut state_scratch = Tensor::empty(&state_req.shape, state_req.dtype);
        let mut dq = Tensor::empty(&backward.dq.shape, backward.dq.dtype);
        let mut dk_output = Tensor::empty(&backward.dk.shape, backward.dk.dtype);
        let mut dv_output = Tensor::empty(&backward.dv.shape, backward.dv.dtype);
        let mut ddecay = Tensor::empty(&backward.dlog_decay.shape, backward.dlog_decay.dtype);
        let mut dbeta = Tensor::empty(&backward.dbeta.shape, backward.dbeta.dtype);
        let mut scratch = Tensor::empty(&backward.scratch.shape, backward.scratch.dtype);

        let decode_q = Tensor::from_vec(prand(2 * 3, 310), vec![2, 3]);
        let decode_k = Tensor::from_vec(prand(2 * 3, 311), vec![2, 3]);
        let decode_v = Tensor::from_vec(prand(2 * 2, 312), vec![2, 2]);
        let decode_decay = Tensor::from_vec(
            prand(2 * 3, 313)
                .into_iter()
                .map(|value| -(value.abs() + 0.1))
                .collect(),
            vec![2, 3],
        );
        let decode_beta = Tensor::from_vec(vec![0.2f32, 0.4], vec![2, 1]);
        let decode_state = Tensor::from_vec(prand(2 * 3 * 2, 314), vec![2, 3, 2]);
        let decode =
            kda_decode_requirements(&decode_q, &decode_k, &decode_v, &decode_decay, &decode_beta)
                .unwrap();
        let mut decode_output = Tensor::empty(&decode.output.shape, decode.output.dtype);
        let mut decode_state_next =
            Tensor::empty(&decode.state_next.shape, decode.state_next.dtype);

        {
            let _guard = ExecutableAllocationGuard::enter();
            kda_forward_into(
                &q,
                &k,
                &v,
                &decay,
                &beta,
                scale,
                None,
                &mut output.destination().unwrap(),
                None,
                Some(&mut state_scratch.destination().unwrap()),
            )
            .unwrap();
            kda_backward_into(
                &q,
                &k,
                &v,
                &decay,
                &beta,
                &gradient,
                scale,
                &mut dq.destination().unwrap(),
                &mut dk_output.destination().unwrap(),
                &mut dv_output.destination().unwrap(),
                &mut ddecay.destination().unwrap(),
                &mut dbeta.destination().unwrap(),
                &mut scratch.destination().unwrap(),
            )
            .unwrap();
            kda_decode_into(
                &decode_q,
                &decode_k,
                &decode_v,
                &decode_decay,
                &decode_beta,
                &decode_state,
                scale,
                &mut decode_output.destination().unwrap(),
                &mut decode_state_next.destination().unwrap(),
            )
            .unwrap();
        }

        assert_close(
            &output,
            &kda_chunk_forward(&q, &k, &v, &decay, &beta, scale),
            8e-4,
        );
        let (expected_dq, expected_dk, expected_dv, expected_decay, expected_beta) =
            kda_chunk_backward(&q, &k, &v, &decay, &beta, &gradient, scale);
        assert_close(&dq, &expected_dq, 8e-4);
        assert_close(&dk_output, &expected_dk, 8e-4);
        assert_close(&dv_output, &expected_dv, 8e-4);
        assert_close(&ddecay, &expected_decay, 8e-4);
        assert_close(&dbeta, &expected_beta, 8e-4);

        let decode_q3 = decode_q.view(Layout::contiguous(vec![2, 1, 3]));
        let decode_k3 = decode_k.view(Layout::contiguous(vec![2, 1, 3]));
        let decode_v3 = decode_v.view(Layout::contiguous(vec![2, 1, 2]));
        let decode_decay3 = decode_decay.view(Layout::contiguous(vec![2, 1, 3]));
        let decode_beta3 = decode_beta.view(Layout::contiguous(vec![2, 1, 1]));
        let (expected_decode, expected_state) = kda_chunk_with_state(
            &decode_q3,
            &decode_k3,
            &decode_v3,
            &decode_decay3,
            &decode_beta3,
            scale,
            &decode_state,
        );
        let expected_decode = expected_decode.view(Layout::contiguous(vec![2, 2]));
        assert_close(&decode_output, &expected_decode, 4e-5);
        assert_close(&decode_state_next, &expected_state, 4e-5);
    }
}

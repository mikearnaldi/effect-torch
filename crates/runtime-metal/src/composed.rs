//! Primitive-built numerical references for the fused kernels (tests
//! only — `#[cfg(test)]` in lib.rs).
//!
//! Every function here implements one fused kernel's semantics using
//! only the primitive `ops` runners (binary/unary/reduce/matmul/
//! gather/...), so the parity tests compare fused destination kernels
//! against an independent composition of already-verified primitives
//! rather than against themselves. These references are deliberately
//! naive: they materialize intermediates the fused kernels avoid (full
//! score matrices, per-token states) and are the readability-first
//! source of truth for the exact numerics contract — masking,
//! scaling, ignore-index semantics, decay factoring, and reduction
//! order — that the fused implementations must reproduce.

use super::ops::{
    binary, broadcast_to, cast, cat, compare, fill, gather, matmul, permute, reduce, unary, where_,
    BinOp, UnOp,
};
use crate::fusion::ReduceOp;
use crate::runtime::dtype::DType;
use crate::runtime::layout::Layout;
use crate::runtime::metal::run::MetalTensor;

fn rank(t: &MetalTensor) -> usize {
    t.layout.shape().len()
}

fn unsqueeze_last(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    let mut shape = t.layout.shape().to_vec();
    shape.push(1);
    Ok(MetalTensor {
        buffer: t.buffer.clone(),
        layout: Layout::contiguous(shape),
        dtype: t.dtype,
    })
}

fn squeeze_last(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    let mut shape = t.layout.shape().to_vec();
    shape.pop();
    Ok(MetalTensor {
        buffer: t.buffer.clone(),
        layout: Layout::contiguous(shape),
        dtype: t.dtype,
    })
}

fn transpose_last2(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    let r = rank(t);
    let mut axes: Vec<usize> = (0..r).collect();
    axes.swap(r - 2, r - 1);
    permute(t, &axes)
}

fn narrow(t: &MetalTensor, dim: usize, start: usize, len: usize) -> crate::err::Res<MetalTensor> {
    let v = MetalTensor {
        buffer: t.buffer.clone(),
        layout: t.layout.narrow(dim, start, len),
        dtype: t.dtype,
    };
    super::ops::contiguous(&v)
}

fn full_like(t: &MetalTensor, value: f64) -> crate::err::Res<MetalTensor> {
    fill(t.layout.shape(), value, t.dtype)
}

fn zeros_like(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
    fill(t.layout.shape(), 0.0, t.dtype)
}

fn mean_dims(t: &MetalTensor, dims: &[usize]) -> crate::err::Res<MetalTensor> {
    let count: usize = dims.iter().map(|&d| t.layout.shape()[d]).product();
    let s = reduce(t, dims, true, ReduceOp::Sum)?;
    let c = fill(s.layout.shape(), count as f64, s.dtype)?;
    binary(&s, &c, BinOp::Div)
}

/// Reference softmax over the last dim (max-subtracted).
pub fn softmax_lastdim(x: &MetalTensor) -> crate::err::Res<MetalTensor> {
    let r = rank(x);
    let m = reduce(x, &[r - 1], true, ReduceOp::Max)?;
    let e = unary(&binary(x, &m, BinOp::Sub)?, UnOp::Exp)?;
    let s = reduce(&e, &[r - 1], true, ReduceOp::Sum)?;
    binary(&e, &s, BinOp::Div)
}

/// Reference logsumexp over the last dim (max-subtracted), kept
/// reduced (same rank as the input).
pub fn logsumexp_lastdim(x: &MetalTensor) -> crate::err::Res<MetalTensor> {
    let r = rank(x);
    let m = reduce(x, &[r - 1], true, ReduceOp::Max)?;
    let e = unary(&binary(x, &m, BinOp::Sub)?, UnOp::Exp)?;
    let s = reduce(&e, &[r - 1], true, ReduceOp::Sum)?;
    binary(&m, &unary(&s, UnOp::Log)?, BinOp::Add)
}

fn causal_allowed(t: usize, s: usize) -> crate::err::Res<MetalTensor> {
    let off = s.saturating_sub(t) as i64;
    let mut data = Vec::with_capacity(t * s);
    for i in 0..t as i64 {
        for j in 0..s as i64 {
            data.push((j <= i + off) as u8);
        }
    }
    Ok(MetalTensor {
        buffer: crate::runtime::metal::device::MetalDevice::get().upload_bytes(&data),
        layout: Layout::contiguous(vec![t, s]),
        dtype: DType::U8,
    })
}

fn causal_additive_mask(t: usize, s: usize, dtype: DType) -> crate::err::Res<MetalTensor> {
    let allowed = causal_allowed(t, s)?;
    let zeros = fill(&[t, s], 0.0, dtype)?;
    let neg = fill(&[t, s], f64::NEG_INFINITY, dtype)?;
    where_(&allowed, &zeros, &neg)
}

fn causal_gate(t: usize, s: usize, dtype: DType) -> crate::err::Res<MetalTensor> {
    let allowed = causal_allowed(t, s)?;
    let ones = fill(&[t, s], 1.0, dtype)?;
    let zeros = fill(&[t, s], 0.0, dtype)?;
    where_(&allowed, &ones, &zeros)
}

fn sdpa_scores(
    q: &MetalTensor,
    k: &MetalTensor,
    scale: f64,
    causal: bool,
) -> crate::err::Res<MetalTensor> {
    let r = rank(q);
    let kt = transpose_last2(k)?;
    let s = matmul(q, &kt)?;
    let s = binary(&s, &full_like(&s, scale)?, BinOp::Mul)?;
    if causal {
        let dims = s.layout.shape();
        let (t, sq) = (dims[r - 2], dims[r - 1]);
        binary(&s, &causal_additive_mask(t, sq, s.dtype)?, BinOp::Add)
    } else {
        Ok(s)
    }
}

/// Reference scaled-dot-product attention forward:
/// `softmax(q·kᵀ·scale + causal_mask)·v`.
pub fn sdpa_forward(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    scale: f64,
    causal: bool,
) -> crate::err::Res<MetalTensor> {
    let s = sdpa_scores(q, k, scale, causal)?;
    let p = softmax_lastdim(&s)?;
    matmul(&p, v)
}

/// Reference SDPA backward; returns `(dq, dk, dv)`. Recomputes the
/// full score matrix and applies the softmax Jacobian
/// `p ⊙ (dp − Σ(p ⊙ dp))` with the causal gate re-applied after.
pub fn sdpa_backward(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    g: &MetalTensor,
    scale: f64,
    causal: bool,
) -> crate::err::Res<(MetalTensor, MetalTensor, MetalTensor)> {
    let r = rank(q);
    let s = sdpa_scores(q, k, scale, causal)?;
    let p = softmax_lastdim(&s)?;
    let g = super::ops::contiguous(g)?;
    let dv = matmul(&transpose_last2(&p)?, &g)?;
    let dp = matmul(&g, &transpose_last2(v)?)?;
    let dp_sum = reduce(&binary(&p, &dp, BinOp::Mul)?, &[r - 1], true, ReduceOp::Sum)?;
    let mut ds = binary(&p, &binary(&dp, &dp_sum, BinOp::Sub)?, BinOp::Mul)?;
    if causal {
        let dims = ds.layout.shape();
        let (t, sq) = (dims[r - 2], dims[r - 1]);
        ds = binary(&ds, &causal_gate(t, sq, ds.dtype)?, BinOp::Mul)?;
    }
    let dq_raw = matmul(&ds, &super::ops::contiguous(k)?)?;
    let dq = binary(&dq_raw, &full_like(&dq_raw, scale)?, BinOp::Mul)?;
    let dk_raw = matmul(&transpose_last2(&ds)?, &super::ops::contiguous(q)?)?;
    let dk = binary(&dk_raw, &full_like(&dk_raw, scale)?, BinOp::Mul)?;
    Ok((dq, dk, dv))
}

/// Reference layer-norm forward over the trailing `weight`-sized dims.
pub fn layer_norm_forward(
    x: &MetalTensor,
    weight: &MetalTensor,
    bias: &MetalTensor,
    eps: f64,
) -> crate::err::Res<MetalTensor> {
    let r = rank(x);
    let k = weight.layout.shape().len();
    let dims: Vec<usize> = (r - k..r).collect();
    let mean = mean_dims(x, &dims)?;
    let centered = binary(x, &mean, BinOp::Sub)?;
    let var = mean_dims(&binary(&centered, &centered, BinOp::Mul)?, &dims)?;
    let inv = unary(
        &binary(&var, &fill(var.layout.shape(), eps, var.dtype)?, BinOp::Add)?,
        UnOp::Sqrt,
    )?;
    let inv = unary(&inv, UnOp::Neg)?;
    let inv = unary(&inv, UnOp::Exp)?;
    binary(
        &binary(&binary(&centered, &inv, BinOp::Mul)?, weight, BinOp::Mul)?,
        bias,
        BinOp::Add,
    )
}

/// Reference layer-norm backward; returns `(dx, dw, db)`. Implements
/// `dx = (g·w − mean(g·w) − x̂·mean(g·w·x̂)) · rstd` with x̂ the
/// normalized activations, and `dw`/`db` as plain sums over the
/// non-normalized dims.
pub fn layer_norm_backward(
    x: &MetalTensor,
    weight: &MetalTensor,
    g: &MetalTensor,
    eps: f64,
) -> crate::err::Res<(MetalTensor, MetalTensor, MetalTensor)> {
    let r = rank(x);
    let k = weight.layout.shape().len();
    let dims: Vec<usize> = (r - k..r).collect();
    let reduce_dims: Vec<usize> = (0..r - k).collect();
    let mean = mean_dims(x, &dims)?;
    let centered = binary(x, &mean, BinOp::Sub)?;
    let var = mean_dims(&binary(&centered, &centered, BinOp::Mul)?, &dims)?;
    let rstd = unary(
        &binary(&var, &fill(var.layout.shape(), eps, var.dtype)?, BinOp::Add)?,
        UnOp::Sqrt,
    )?;
    let rstd = unary(&unary(&rstd, UnOp::Neg)?, UnOp::Exp)?;
    let xh = binary(&centered, &rstd, BinOp::Mul)?;
    // dx = (dyw − mean(dyw) − x̂·mean(dyw·x̂)) · rstd
    let dyw = binary(g, weight, BinOp::Mul)?;
    let m1 = mean_dims(&dyw, &dims)?;
    let m2 = mean_dims(&binary(&dyw, &xh, BinOp::Mul)?, &dims)?;
    let dx = binary(
        &binary(&dyw, &m1, BinOp::Sub)?,
        &binary(&xh, &m2, BinOp::Mul)?,
        BinOp::Sub,
    )?;
    let dx = binary(&dx, &rstd, BinOp::Mul)?;
    let dw = reduce(
        &binary(g, &xh, BinOp::Mul)?,
        &reduce_dims,
        false,
        ReduceOp::Sum,
    )?;
    let db = reduce(g, &reduce_dims, false, ReduceOp::Sum)?;
    Ok((dx, dw, db))
}

fn ce_ignored_mask(target: &MetalTensor, ignore_index: i64) -> crate::err::Res<MetalTensor> {
    match target.dtype {
        DType::I64 => {
            let ii = fill(target.layout.shape(), ignore_index as f64, DType::I64)?;
            compare(target, &ii, BinOp::Eq)
        }
        DType::U32 => {
            if ignore_index < 0 || ignore_index > u32::MAX as i64 {
                fill(target.layout.shape(), 0.0, DType::U8)
            } else {
                let ii = fill(target.layout.shape(), ignore_index as f64, DType::U32)?;
                compare(target, &ii, BinOp::Eq)
            }
        }
        _ => crate::err::err("cross_entropy: target must be i64 or u32"),
    }
}

fn to_f32_vec(t: &MetalTensor) -> crate::err::Res<Vec<f32>> {
    let dev = crate::runtime::metal::device::MetalDevice::get();
    let tc = crate::runtime::metal::kernels::strided_copy(dev, t)?;
    let t32 = if tc.dtype == DType::F32 {
        tc
    } else {
        cast(&tc, DType::F32)?
    };
    dev.synchronize()?;
    Ok(t32.read_f32()?)
}

fn scalar_f64(t: &MetalTensor) -> crate::err::Res<f64> {
    let v = to_f32_vec(t)?;
    Ok(v[0] as f64)
}

fn ce_active_count(ignored: &MetalTensor, total: usize) -> crate::err::Res<f64> {
    let ignored32 = cast(ignored, DType::F32)?;
    let all: Vec<usize> = (0..rank(ignored)).collect();
    let s = reduce(&ignored32, &all, true, ReduceOp::Sum)?;
    Ok(total as f64 - scalar_f64(&s)?)
}

fn ce_check_labels(
    target: &MetalTensor,
    ignored: &MetalTensor,
    classes: usize,
) -> crate::err::Res<()> {
    let invalid = match target.dtype {
        DType::I64 => {
            let lo = compare(
                target,
                &fill(target.layout.shape(), 0.0, DType::I64)?,
                BinOp::Lt,
            )?;
            let hi = compare(
                target,
                &fill(target.layout.shape(), classes as f64, DType::I64)?,
                BinOp::Ge,
            )?;
            let hi8 = cast(&hi, DType::U8)?;
            binary(&lo, &hi8, BinOp::Max)
        }
        DType::U32 => compare(
            target,
            &fill(target.layout.shape(), classes as f64, DType::U32)?,
            BinOp::Ge,
        ),
        _ => unreachable!(),
    }?;
    let active = compare(
        ignored,
        &fill(ignored.layout.shape(), 0.0, DType::U8)?,
        BinOp::Eq,
    )?;
    let invalid_active = cast(&binary(&invalid, &active, BinOp::Mul)?, DType::F32)?;
    let all: Vec<usize> = (0..rank(&invalid_active)).collect();
    let s = reduce(&invalid_active, &all, true, ReduceOp::Sum)?;
    if scalar_f64(&s)? > 0.0 {
        return crate::err::err(format!(
            "cross_entropy: target out of range [0, {classes}) at an active position"
        ));
    }
    Ok(())
}

fn target_to_ids(target: &MetalTensor) -> crate::err::Res<MetalTensor> {
    match target.dtype {
        DType::I64 => Ok(target.clone()),
        _ => cast(target, DType::I64),
    }
}

/// Reference cross-entropy forward with ignore-index masking, active
/// count and label validation (mean over an all-ignored batch and
/// out-of-range active labels are hard errors, in this order).
pub fn cross_entropy_forward(
    logits: &MetalTensor,
    target: &MetalTensor,
    ignore_index: i64,
    reduction: crate::CeReduction,
) -> crate::err::Res<MetalTensor> {
    let r = rank(logits);
    let classes = logits.layout.shape()[r - 1];
    let ignored = ce_ignored_mask(target, ignore_index)?;
    let count = ce_active_count(&ignored, target.numel())?;
    if count == 0.0 && reduction == crate::CeReduction::Mean {
        return crate::err::err("cross_entropy: no active targets (all positions are ignored)");
    }
    ce_check_labels(target, &ignored, classes)?;
    let lse = logsumexp_lastdim(logits)?;
    let zero_ids = fill(target.layout.shape(), 0.0, DType::I64)?;
    let safe_target = where_(&ignored, &zero_ids, &target_to_ids(target)?)?;
    let safe_ids = unsqueeze_last(&safe_target)?;
    let safe_ids32 = super::ops::cast(&safe_ids, DType::U32)?;
    let picked = gather(logits, r - 1, &safe_ids32, safe_ids.layout.shape())?;
    let picked = squeeze_last(&picked)?;
    let per_position = binary(&squeeze_last(&lse)?, &picked, BinOp::Sub)?;
    let masked = where_(&ignored, &zeros_like(&per_position)?, &per_position)?;
    let all: Vec<usize> = (0..rank(&masked)).collect();
    let total = reduce(&masked, &all, true, ReduceOp::Sum)?;
    match reduction {
        crate::CeReduction::Mean => binary(
            &total,
            &fill(total.layout.shape(), 1.0 / count, total.dtype)?,
            BinOp::Mul,
        ),
        crate::CeReduction::Sum => Ok(total),
    }
}

/// Reference cross-entropy backward: `softmax − one_hot` at active
/// positions, zeros at ignored ones, scaled by `1/active` for mean.
pub fn cross_entropy_backward(
    logits: &MetalTensor,
    target: &MetalTensor,
    ignore_index: i64,
    reduction: crate::CeReduction,
) -> crate::err::Res<MetalTensor> {
    let r = rank(logits);
    let ignored = ce_ignored_mask(target, ignore_index)?;
    let count = ce_active_count(&ignored, target.numel())?;
    if count == 0.0 && reduction == crate::CeReduction::Mean {
        return crate::err::err("cross_entropy: no active targets (all positions are ignored)");
    }
    let p = softmax_lastdim(logits)?;
    let zero_ids = fill(target.layout.shape(), 0.0, DType::I64)?;
    let safe_target = where_(&ignored, &zero_ids, &target_to_ids(target)?)?;
    let ids = unsqueeze_last(&safe_target)?;
    let neg_ones = fill(ids.layout.shape(), -1.0, logits.dtype)?;
    let ids32 = super::ops::cast(&ids, DType::U32)?;
    let p = super::ops::scatter_add(&p, r - 1, &ids32, &neg_ones)?;
    let keep = compare(
        &ignored,
        &fill(ignored.layout.shape(), 0.0, DType::U8)?,
        BinOp::Eq,
    )?;
    let keep = unsqueeze_last(&keep)?;
    let masked = where_(&keep, &p, &zeros_like(&p)?)?;
    match reduction {
        crate::CeReduction::Mean => binary(
            &masked,
            &fill(masked.layout.shape(), 1.0 / count, masked.dtype)?,
            BinOp::Mul,
        ),
        crate::CeReduction::Sum => Ok(masked),
    }
}

/// Reference rotary embedding (GPT-NeoX half-split): builds the
/// angle table host-side, rotates the two halves with cos/sin. `sign`
/// = −1 yields the transpose rotation (the backward).
pub fn rotary_forward(
    x: &MetalTensor,
    offsets: &[usize],
    theta: f64,
    sign: f64,
) -> crate::err::Res<MetalTensor> {
    let dims = x.layout.shape();
    let r = dims.len();
    let (t, d) = (dims[r - 2], dims[r - 1]);
    let batch = dims[0];
    if offsets.len() != 1 && offsets.len() != batch {
        return crate::err::err(format!(
            "rotary embedding: {} offsets for batch {batch}",
            offsets.len()
        ));
    }
    let half = d / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 / d as f64) as f32)
        .collect();
    let inv_freq = MetalTensor::from_f32(
        crate::runtime::metal::device::MetalDevice::get(),
        inv_freq,
        vec![1, half],
    );
    let positions: Vec<f32> = if offsets.len() == 1 {
        (0..t).map(|p| (offsets[0] + p) as f32).collect()
    } else {
        offsets
            .iter()
            .flat_map(|base| (0..t).map(move |p| (*base + p) as f32))
            .collect()
    };
    let rows = if offsets.len() == 1 { 1 } else { batch };
    let positions = MetalTensor::from_f32(
        crate::runtime::metal::device::MetalDevice::get(),
        positions,
        vec![rows * t, 1],
    );
    let angles = matmul(&positions, &inv_freq)?;
    let angles = binary(
        &angles,
        &fill(&[rows * t, half], sign, DType::F32)?,
        BinOp::Mul,
    )?;
    let mut table_shape = vec![1usize; r - 2];
    if offsets.len() != 1 {
        table_shape[0] = batch;
    }
    table_shape.extend([t, half]);
    let cos = unary(&angles, UnOp::Cos)?;
    let sin = unary(&angles, UnOp::Sin)?;
    let cos = broadcast_to(&cos, &table_shape)?;
    let sin = broadcast_to(&sin, &table_shape)?;
    let first = narrow(x, r - 1, 0, half)?;
    let second = narrow(x, r - 1, half, half)?;
    let out_first = binary(
        &binary(&first, &cos, BinOp::Mul)?,
        &binary(&second, &sin, BinOp::Mul)?,
        BinOp::Sub,
    )?;
    let out_second = binary(
        &binary(&second, &cos, BinOp::Mul)?,
        &binary(&first, &sin, BinOp::Mul)?,
        BinOp::Add,
    )?;
    cat(&out_first, &out_second, r - 1)
}

// --- Chunked head cross-entropy (RFC 0016 phase 2, semantic form) ---

fn head_ce_chunk_len(rows: usize, vocab: usize, chunk_size: usize) -> usize {
    let elements = rows.saturating_mul(vocab);
    let chunks = (elements / chunk_size).clamp(2, 64).min(rows);
    rows.div_ceil(chunks.max(1))
}

// Active count and label checks from one host readback: targets are
// integers far below 2^24, so the f32 readback is exact, and this runs
// once per eval (the emitter rejects u8 comparisons).
fn head_ce_check_target(t1: &MetalTensor, ignore_index: i64, vocab: usize) -> crate::err::Res<f64> {
    let host = super::ops::to_f32(t1)?.read_f32()?;
    let mut active = 0usize;
    for &value in &host {
        let t = value as i64;
        if t == ignore_index {
            continue;
        }
        if t < 0 || t as usize >= vocab {
            return Err(format!(
                "cross_entropy: target out of range [0, {vocab}) at an active position"
            ));
        }
        active += 1;
    }
    Ok(active as f64)
}

/// Reference chunked-head cross-entropy forward (RFC 0016 phase 2):
/// mean CE of `Linear(x, weight, bias)` against `target`, evaluated
/// one row-chunk at a time so the `[rows, vocab]` logits never
/// materialize whole. Mean semantics match the plain path exactly:
/// zero-active error before label checks, in the same order.
pub fn chunked_head_ce_forward(
    x: &MetalTensor,
    weight: &MetalTensor,
    bias: &MetalTensor,
    target: &MetalTensor,
    ignore_index: i64,
    chunk_size: usize,
) -> crate::err::Res<MetalTensor> {
    let dims = x.layout.shape().to_vec();
    let r = dims.len();
    let (inner, vocab) = (weight.layout.shape()[0], weight.layout.shape()[1]);
    let rows: usize = dims[..r - 1].iter().product();
    let x2 = super::ops::contiguous(&MetalTensor {
        buffer: x.buffer.clone(),
        layout: Layout::contiguous(vec![rows, inner]),
        dtype: x.dtype,
    })?;
    let t1 = super::ops::contiguous(&MetalTensor {
        buffer: target.buffer.clone(),
        layout: Layout::contiguous(vec![rows]),
        dtype: target.dtype,
    })?;
    // Match Mean semantics exactly: zero-active error before label
    // checks, in the plain path's order.
    let count = head_ce_check_target(&t1, ignore_index, vocab)?;
    if count == 0.0 {
        return Err("cross_entropy: no active targets (all positions are ignored)".to_string());
    }
    let chunk_len = head_ce_chunk_len(rows, vocab, chunk_size);
    let mut total = fill(&[1], 0.0, DType::F32)?;
    let mut off = 0;
    while off < rows {
        let end = (off + chunk_len).min(rows);
        let x_c = narrow(&x2, 0, off, end - off)?;
        let t_c = narrow(&t1, 0, off, end - off)?;
        let logits = binary(&matmul(&x_c, weight)?, bias, BinOp::Add)?;
        let (nll, _status) =
            crate::loss::ce_forward(&logits, &t_c, ignore_index, crate::CeReduction::Sum)?;
        total = binary(&total, &super::ops::to_f32(&nll)?, BinOp::Add)?;
        off = end;
    }
    let mean = binary(&total, &fill(&[1], count, DType::F32)?, BinOp::Div)?;
    if mean.dtype == x.dtype {
        Ok(mean)
    } else {
        super::ops::from_f32(&mean, x.dtype)
    }
}

/// Reference chunked-head CE backward (closed-form adjoint):
/// recomputes each chunk's logits and grad-logits in a transient
/// workspace and accumulates `(dx, dw, db)`; grad-logits never outlive
/// their chunk.
pub fn chunked_head_ce_backward(
    x: &MetalTensor,
    weight: &MetalTensor,
    bias: &MetalTensor,
    target: &MetalTensor,
    g: &MetalTensor,
    ignore_index: i64,
    chunk_size: usize,
) -> crate::err::Res<(MetalTensor, MetalTensor, MetalTensor)> {
    let dims = x.layout.shape().to_vec();
    let r = dims.len();
    let (inner, vocab) = (weight.layout.shape()[0], weight.layout.shape()[1]);
    let rows: usize = dims[..r - 1].iter().product();
    let x2 = super::ops::contiguous(&MetalTensor {
        buffer: x.buffer.clone(),
        layout: Layout::contiguous(vec![rows, inner]),
        dtype: x.dtype,
    })?;
    let t1 = super::ops::contiguous(&MetalTensor {
        buffer: target.buffer.clone(),
        layout: Layout::contiguous(vec![rows]),
        dtype: target.dtype,
    })?;
    let count = head_ce_check_target(&t1, ignore_index, vocab)?;
    let scale = fill(&[1], scalar_f64(g)? / count, DType::F32)?;
    let chunk_len = head_ce_chunk_len(rows, vocab, chunk_size);
    let w32t = transpose_last2(&super::ops::to_f32(weight)?)?;
    let mut dx_chunks: Vec<MetalTensor> = Vec::new();
    let mut dw = fill(&[inner, vocab], 0.0, DType::F32)?;
    let mut db = fill(&[1, vocab], 0.0, DType::F32)?;
    let mut off = 0;
    while off < rows {
        let end = (off + chunk_len).min(rows);
        let x_c = narrow(&x2, 0, off, end - off)?;
        let t_c = narrow(&t1, 0, off, end - off)?;
        let logits = binary(&matmul(&x_c, weight)?, bias, BinOp::Add)?;
        let (gb, _count) =
            crate::loss::ce_backward(&logits, &t_c, ignore_index, crate::CeReduction::Sum)?;
        let gb32 = binary(&super::ops::to_f32(&gb)?, &scale, BinOp::Mul)?;
        dx_chunks.push(matmul(&gb32, &w32t)?);
        dw = binary(
            &dw,
            &matmul(&transpose_last2(&super::ops::to_f32(&x_c)?)?, &gb32)?,
            BinOp::Add,
        )?;
        db = binary(&db, &reduce(&gb32, &[0], true, ReduceOp::Sum)?, BinOp::Add)?;
        off = end;
    }
    let mut cat_checkpoint = |_live: &[usize]| {};
    let dx = super::ops::from_f32(
        &super::ops::contiguous(&MetalTensor {
            buffer: cat_tree(dx_chunks, 0, &mut cat_checkpoint)?.buffer.clone(),
            layout: Layout::contiguous(dims),
            dtype: DType::F32,
        })?,
        x.dtype,
    )?;
    let dw = super::ops::from_f32(&dw, weight.dtype)?;
    let db_flat = super::ops::from_f32(&db, bias.dtype)?;
    let db = super::ops::contiguous(&MetalTensor {
        buffer: db_flat.buffer.clone(),
        layout: Layout::contiguous(bias.layout.shape().to_vec()),
        dtype: db_flat.dtype,
    })?;
    Ok((dx, dw, db))
}

// --- Kimi Delta Attention (RFC 0018) ---

fn unsqueeze(t: &MetalTensor, dim: usize) -> crate::err::Res<MetalTensor> {
    let mut shape = t.layout.shape().to_vec();
    shape.insert(dim, 1);
    Ok(MetalTensor {
        buffer: t.buffer.clone(),
        layout: Layout::contiguous(shape),
        dtype: t.dtype,
    })
}

// tril mask (diagonal 0 includes the diagonal, -1 excludes it) as u8.
fn tril_mask(n: usize, diagonal: i64) -> crate::err::Res<MetalTensor> {
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n as i64 {
        for j in 0..n as i64 {
            data.push((j <= i + diagonal) as u8);
        }
    }
    Ok(MetalTensor {
        buffer: crate::runtime::metal::device::MetalDevice::get().upload_bytes(&data),
        layout: Layout::contiguous(vec![n, n]),
        dtype: DType::U8,
    })
}

// row [1, n] -> [batch, 1, n] via broadcast add against zeros.
fn batch_row(row: &MetalTensor, batch: usize) -> crate::err::Res<MetalTensor> {
    let n = row.layout.shape()[row.layout.shape().len() - 1];
    binary(&fill(&[batch, 1, n], 0.0, row.dtype)?, row, BinOp::Add)
}

// Unit lower-triangular inverse: given strictly lower-triangular a
// [.., n, n], returns (I + a)^-1 via batched row-wise forward
// substitution x_i = e_i - a_i[:, :i] @ x_{:i} (RFC 0018 numerics
// contract: sequential substitution, never a series expansion).
fn unit_lower_inverse(a: &MetalTensor) -> crate::err::Res<MetalTensor> {
    let dims = a.layout.shape().to_vec();
    let r = dims.len();
    let n = dims[r - 1];
    let batch: usize = dims[..r - 2].iter().product();
    let a3 = super::ops::contiguous(&MetalTensor {
        buffer: a.buffer.clone(),
        layout: Layout::contiguous(vec![batch, n, n]),
        dtype: a.dtype,
    })?;
    let id = super::ops::eye(n, a.dtype)?;
    let mut x = batch_row(&narrow(&id, 0, 0, 1)?, batch)?;
    for i in 1..n {
        let a_row = narrow(&a3, 1, i, 1)?;
        let a_left = narrow(&a_row, 2, 0, i)?;
        let contrib = matmul(&a_left, &x)?;
        let e_i = batch_row(&narrow(&id, 0, i, 1)?, batch)?;
        let row = binary(&e_i, &contrib, BinOp::Sub)?;
        x = cat(&x, &row, 1)?;
    }
    Ok(MetalTensor {
        buffer: x.buffer.clone(),
        layout: Layout::contiguous(dims),
        dtype: x.dtype,
    })
}

fn cat_all(ts: &[MetalTensor], dim: usize) -> crate::err::Res<MetalTensor> {
    let mut it = ts.iter();
    let mut acc = it
        .next()
        .ok_or_else(|| "cat_all: empty".to_string())?
        .clone();
    for t in it {
        acc = cat(&acc, t, dim)?;
    }
    Ok(acc)
}

// Balanced pairwise concat: a left-deep fold holds every partial result
// until the fold completes (64 chunks ≈ 8 GiB live at once); the tree
// keeps one fold level live at a time. The callback lets callers expose
// each fixed fold level to their command planner.
fn cat_tree(
    ts: Vec<MetalTensor>,
    dim: usize,
    checkpoint: &mut dyn FnMut(&[usize]),
) -> crate::err::Res<MetalTensor> {
    let mut level = ts;
    while level.len() > 1 {
        let mut next: Vec<MetalTensor> = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            next.push(if pair.len() == 2 {
                cat(&pair[0], &pair[1], dim)?
            } else {
                pair[0].clone()
            });
        }
        let live: Vec<usize> = next
            .iter()
            .map(|t| std::sync::Arc::as_ptr(&t.buffer) as usize)
            .collect();
        checkpoint(&live);
        level = next;
    }
    level
        .into_iter()
        .next()
        .ok_or_else(|| "cat_tree: empty".to_string())
}

/// Reference chunked gated delta-rule linear attention (RFC 0018; FLA
/// `naive_chunk_kda` equivalent). `q/k/log_decay [.., H, T, Dk]`,
/// `v [.., H, T, Dv]`, `beta [.., H, T, 1]`; computes in f32 from a
/// zero initial state. Chunk 64, sub-chunk 16: intra-chunk blocks use
/// the pivot-factored decay `exp(g_i − g_j) = exp(g_i − g_p) *
/// exp(g_p − g_j)` so no reciprocal cumulative decay is ever formed.
pub fn kda_chunk_forward(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    log_decay: &MetalTensor,
    beta: &MetalTensor,
    scale: f64,
) -> crate::err::Res<MetalTensor> {
    let dims = q.layout.shape().to_vec();
    let r = dims.len();
    let dk = dims[r - 1];
    let dv = v.layout.shape()[r - 1];
    let bh: usize = dims[..r - 2].iter().product();
    let initial = fill(&[bh, dk, dv], 0.0, DType::F32)?;
    Ok(kda_chunk_with_state(q, k, v, log_decay, beta, scale, &initial)?.0)
}

/// Stateful variant of [`kda_chunk_forward`]: starts from
/// `initial_state` (`[BH, Dk, Dv]` f32) and returns the output
/// alongside the final state. The decode path drives this per
/// sequence slot.
pub fn kda_chunk_with_state(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    log_decay: &MetalTensor,
    beta: &MetalTensor,
    scale: f64,
    initial_state: &MetalTensor,
) -> crate::err::Res<(MetalTensor, MetalTensor)> {
    const CHUNK: usize = 64;
    const SUB: usize = 16;
    let in_dtype = q.dtype;
    let q = super::ops::to_f32(q)?;
    let k = super::ops::to_f32(k)?;
    let v = super::ops::to_f32(v)?;
    let log_decay = super::ops::to_f32(log_decay)?;
    let beta = super::ops::to_f32(beta)?;
    let work = DType::F32;

    let dims = q.layout.shape().to_vec();
    let r = dims.len();
    let (t, dk) = (dims[r - 2], dims[r - 1]);
    let dv = v.layout.shape()[r - 1];
    let bh: usize = dims[..r - 2].iter().product();

    let flatten = |x: &MetalTensor, d: usize| -> crate::err::Res<MetalTensor> {
        super::ops::contiguous(&MetalTensor {
            buffer: x.buffer.clone(),
            layout: Layout::contiguous(vec![bh, t, d]),
            dtype: x.dtype,
        })
    };
    let q3 = flatten(&q, dk)?;
    let k3 = flatten(&k, dk)?;
    let v3 = flatten(&v, dv)?;
    let ld3 = flatten(&log_decay, dk)?;
    let b3 = flatten(&beta, 1)?;

    let mut state = super::ops::contiguous(initial_state)?;
    let mut outs: Vec<MetalTensor> = Vec::new();
    let mut t0 = 0;
    while t0 < t {
        let c = CHUNK.min(t - t0);
        let qc = narrow(&q3, 1, t0, c)?;
        let kc = narrow(&k3, 1, t0, c)?;
        let vc = narrow(&v3, 1, t0, c)?;
        let bc = narrow(&b3, 1, t0, c)?;
        // Inclusive chunk-local cumulative log decay, [BH, c, Dk].
        let gc = super::ops::cumsum(&narrow(&ld3, 1, t0, c)?, 1)?;

        // Intra-chunk attention matrices, assembled from SUB-sized
        // blocks: Aqk (lower-triangular, scaled) and Akk (strictly
        // lower, beta-weighted).
        let blocks = c.div_ceil(SUB);
        let mut aqk_rows: Vec<MetalTensor> = Vec::new();
        let mut akk_rows: Vec<MetalTensor> = Vec::new();
        for rb in 0..blocks {
            let rs = rb * SUB;
            let br = SUB.min(c - rs);
            let q_r = narrow(&qc, 1, rs, br)?;
            let k_r = narrow(&kc, 1, rs, br)?;
            let g_r = narrow(&gc, 1, rs, br)?;
            let g_p = narrow(&gc, 1, rs, 1)?;
            let b_r = narrow(&bc, 1, rs, br)?;
            let mut aqk_cols: Vec<MetalTensor> = Vec::new();
            let mut akk_cols: Vec<MetalTensor> = Vec::new();
            for cb in 0..blocks {
                let cs = cb * SUB;
                let cc = SUB.min(c - cs);
                if cb > rb {
                    aqk_cols.push(fill(&[bh, br, cc], 0.0, work)?);
                    akk_cols.push(fill(&[bh, br, cc], 0.0, work)?);
                    continue;
                }
                let k_c = narrow(&kc, 1, cs, cc)?;
                let g_c = narrow(&gc, 1, cs, cc)?;
                if cb == rb {
                    // Diagonal block: full per-channel decay matrix,
                    // masked by select (exp overflow on the masked
                    // triangle is discarded, never multiplied).
                    let d = binary(&unsqueeze(&g_r, 2)?, &unsqueeze(&g_c, 1)?, BinOp::Sub)?;
                    let e = unary(&d, UnOp::Exp)?;
                    let zeros = fill(e.layout.shape(), 0.0, work)?;
                    let m_incl = unsqueeze(&unsqueeze(&tril_mask(br, 0)?, 0)?, 3)?;
                    let m_strict = unsqueeze(&unsqueeze(&tril_mask(br, -1)?, 0)?, 3)?;
                    let e_incl = where_(&m_incl, &e, &zeros)?;
                    let e_strict = where_(&m_strict, &e, &zeros)?;
                    let qq = binary(&unsqueeze(&q_r, 2)?, &unsqueeze(&k_c, 1)?, BinOp::Mul)?;
                    let aqk = binary(
                        &squeeze_last(&reduce(
                            &binary(&qq, &e_incl, BinOp::Mul)?,
                            &[3],
                            true,
                            ReduceOp::Sum,
                        )?)?,
                        &fill(&[bh, br, cc], scale, work)?,
                        BinOp::Mul,
                    )?;
                    let kk = binary(&unsqueeze(&k_r, 2)?, &unsqueeze(&k_c, 1)?, BinOp::Mul)?;
                    let akk = binary(
                        &squeeze_last(&reduce(
                            &binary(&kk, &e_strict, BinOp::Mul)?,
                            &[3],
                            true,
                            ReduceOp::Sum,
                        )?)?,
                        &b_r,
                        BinOp::Mul,
                    )?;
                    aqk_cols.push(aqk);
                    akk_cols.push(akk);
                } else {
                    // Off-diagonal block: decay factors at the row
                    // block's pivot, both directions bounded by 1.
                    let qd = binary(
                        &q_r,
                        &unary(&binary(&g_r, &g_p, BinOp::Sub)?, UnOp::Exp)?,
                        BinOp::Mul,
                    )?;
                    let kd = binary(
                        &k_c,
                        &unary(&binary(&g_p, &g_c, BinOp::Sub)?, UnOp::Exp)?,
                        BinOp::Mul,
                    )?;
                    let aqk = binary(
                        &matmul(&qd, &transpose_last2(&kd)?)?,
                        &fill(&[bh, br, cc], scale, work)?,
                        BinOp::Mul,
                    )?;
                    let kkd = binary(
                        &k_r,
                        &unary(&binary(&g_r, &g_p, BinOp::Sub)?, UnOp::Exp)?,
                        BinOp::Mul,
                    )?;
                    let akk = binary(&matmul(&kkd, &transpose_last2(&kd)?)?, &b_r, BinOp::Mul)?;
                    aqk_cols.push(aqk);
                    akk_cols.push(akk);
                }
            }
            aqk_rows.push(cat_all(&aqk_cols, 2)?);
            akk_rows.push(cat_all(&akk_cols, 2)?);
        }
        let aqk = cat_all(&aqk_rows, 1)?;
        let akk = cat_all(&akk_rows, 1)?;

        // UT transform: M = (I + Akk)^-1, then the WY representation.
        let m = unit_lower_inverse(&akk)?;
        let w_in = binary(
            &binary(&kc, &bc, BinOp::Mul)?,
            &unary(&gc, UnOp::Exp)?,
            BinOp::Mul,
        )?;
        let w = matmul(&m, &w_in)?;
        let u = matmul(&m, &binary(&vc, &bc, BinOp::Mul)?)?;

        let v_new = binary(&u, &matmul(&w, &state)?, BinOp::Sub)?;
        let o_inter = binary(
            &matmul(&binary(&qc, &unary(&gc, UnOp::Exp)?, BinOp::Mul)?, &state)?,
            &fill(&[bh, c, dv], scale, work)?,
            BinOp::Mul,
        )?;
        let o_intra = matmul(&aqk, &v_new)?;
        outs.push(binary(&o_inter, &o_intra, BinOp::Add)?);

        // State update: decay to the chunk end, then rank-c update with
        // the decayed keys kg = k * exp(g_last - g).
        let g_last = narrow(&gc, 1, c - 1, 1)?;
        let kg = binary(
            &kc,
            &unary(&binary(&g_last, &gc, BinOp::Sub)?, UnOp::Exp)?,
            BinOp::Mul,
        )?;
        let decay = transpose_last2(&unary(&g_last, UnOp::Exp)?)?;
        state = binary(
            &binary(&state, &decay, BinOp::Mul)?,
            &matmul(&transpose_last2(&kg)?, &v_new)?,
            BinOp::Add,
        )?;
        t0 += c;
    }

    let out = cat_all(&outs, 1)?;
    let mut out_shape = dims;
    out_shape[r - 1] = dv;
    let out = super::ops::contiguous(&MetalTensor {
        buffer: out.buffer.clone(),
        layout: Layout::contiguous(out_shape),
        dtype: out.dtype,
    })?;
    Ok((super::ops::from_f32(&out, in_dtype)?, state))
}

/// Reference causal depthwise short convolution over `[.., T, C]` with
/// weight `[C, K]` and zero history: `y[t] = Σ_j w[:, j] * x[t-K+1+j]`.
pub fn short_conv1d_forward(x: &MetalTensor, weight: &MetalTensor) -> crate::err::Res<MetalTensor> {
    let dims = x.layout.shape().to_vec();
    let r = dims.len();
    let (t, c) = (dims[r - 2], dims[r - 1]);
    let kk = weight.layout.shape()[1];
    let batch: usize = dims[..r - 2].iter().product();
    let x3 = super::ops::contiguous(&MetalTensor {
        buffer: x.buffer.clone(),
        layout: Layout::contiguous(vec![batch, t, c]),
        dtype: x.dtype,
    })?;
    let history = fill(&[batch, kk - 1, c], 0.0, x.dtype)?;
    let window = cat(&history, &x3, 1)?;
    let mut acc = fill(&[batch, t, c], 0.0, x.dtype)?;
    for j in 0..kk {
        let wj = transpose_last2(&narrow(weight, 1, j, 1)?)?;
        acc = binary(
            &acc,
            &binary(&narrow(&window, 1, j, t)?, &wj, BinOp::Mul)?,
            BinOp::Add,
        )?;
    }
    super::ops::contiguous(&MetalTensor {
        buffer: acc.buffer.clone(),
        layout: Layout::contiguous(dims),
        dtype: acc.dtype,
    })
}

/// Stateful per-slot variant: `x [T, C]`, `state [K-1, C]`; returns
/// the output and the new window. `advance` is the count of real
/// tokens (chunked prefill right-pads): outputs are computed over the
/// full padded window — causal, so real rows never see padding — but
/// the stored window shifts in only the first `advance` rows.
pub fn short_conv1d_with_state(
    x: &MetalTensor,
    weight: &MetalTensor,
    state: &MetalTensor,
    advance: usize,
) -> crate::err::Res<(MetalTensor, MetalTensor)> {
    let dims = x.layout.shape().to_vec();
    let (t, kk) = (dims[0], weight.layout.shape()[1]);
    let window = cat(state, x, 0)?;
    let mut acc = fill(&dims, 0.0, x.dtype)?;
    for j in 0..kk {
        let wj = transpose_last2(&narrow(weight, 1, j, 1)?)?;
        acc = binary(
            &acc,
            &binary(&narrow(&window, 0, j, t)?, &wj, BinOp::Mul)?,
            BinOp::Add,
        )?;
    }
    let real = narrow(&window, 0, 0, kk - 1 + advance)?;
    let new_state = narrow(&real, 0, advance, kk - 1)?;
    Ok((acc, new_state))
}

/// Reference ShortConv1d backward-x (RFC 0018 phase 4):
/// `dx[s] = Σ_j w[:, K-1-j] * g[s+j]` — full correlation over the
/// right-zero-padded cotangent. `g` and `x` are `[.., T, C]`; weight
/// is `[C, K]`.
pub fn short_conv1d_backward_x(
    x: &MetalTensor,
    weight: &MetalTensor,
    g: &MetalTensor,
) -> crate::err::Res<MetalTensor> {
    let dims = x.layout.shape().to_vec();
    let r = dims.len();
    let (t, c) = (dims[r - 2], dims[r - 1]);
    let kk = weight.layout.shape()[1];
    let batch: usize = dims[..r - 2].iter().product();
    let g3 = super::ops::contiguous(&MetalTensor {
        buffer: g.buffer.clone(),
        layout: Layout::contiguous(vec![batch, t, c]),
        dtype: g.dtype,
    })?;
    let padded = cat(&g3, &fill(&[batch, kk - 1, c], 0.0, g3.dtype)?, 1)?;
    let mut acc = fill(&[batch, t, c], 0.0, g3.dtype)?;
    for j in 0..kk {
        let wj = transpose_last2(&narrow(weight, 1, kk - 1 - j, 1)?)?;
        acc = binary(
            &acc,
            &binary(&narrow(&padded, 1, j, t)?, &wj, BinOp::Mul)?,
            BinOp::Add,
        )?;
    }
    super::ops::contiguous(&MetalTensor {
        buffer: acc.buffer.clone(),
        layout: Layout::contiguous(dims),
        dtype: acc.dtype,
    })
}

/// Reference ShortConv1d backward-w: `dw[:, j] = Σ_t g[t] *
/// x[t-K+1+j]` — per-tap correlation over the causal window, summed
/// over batch and time into `[C, K]`.
pub fn short_conv1d_backward_w(
    x: &MetalTensor,
    weight: &MetalTensor,
    g: &MetalTensor,
) -> crate::err::Res<MetalTensor> {
    let dims = x.layout.shape().to_vec();
    let r = dims.len();
    let (t, c) = (dims[r - 2], dims[r - 1]);
    let kk = weight.layout.shape()[1];
    let batch: usize = dims[..r - 2].iter().product();
    let view3 = |t_: &MetalTensor| {
        super::ops::contiguous(&MetalTensor {
            buffer: t_.buffer.clone(),
            layout: Layout::contiguous(vec![batch, t, c]),
            dtype: t_.dtype,
        })
    };
    let x3 = view3(x)?;
    let g3 = view3(g)?;
    let window = cat(&fill(&[batch, kk - 1, c], 0.0, x3.dtype)?, &x3, 1)?;
    let mut cols: Vec<MetalTensor> = Vec::with_capacity(kk);
    for j in 0..kk {
        cols.push(reduce(
            &binary(&g3, &narrow(&window, 1, j, t)?, BinOp::Mul)?,
            &[1],
            true,
            ReduceOp::Sum,
        )?);
    }
    // [batch, K, C] -> sum over batch -> [1, K, C] -> [1, C, K] -> [C, K]
    let stacked = cat_all(&cols, 1)?;
    let summed = reduce(&stacked, &[0], true, ReduceOp::Sum)?;
    let transposed = transpose_last2(&summed)?;
    super::ops::contiguous(&MetalTensor {
        buffer: transposed.buffer.clone(),
        layout: Layout::contiguous(vec![c, kk]),
        dtype: transposed.dtype,
    })
}

/// Reference closed-form KDA backward (RFC 0018 phase 4). With
/// S̃_t = Diag(α_t) S_{t-1}, δ_t = v_t − S̃_tᵀ k_t,
/// S_t = S̃_t + β_t k_t δ_tᵀ and o_t = scale · S_tᵀ q_t, the adjoint
/// state Λ_t = ∂L/∂S_t runs in reverse:
///
/// ```text
///   Λ_t   += scale · q_t g_tᵀ           (g = output cotangent)
///   dq_t   = scale · S_t g_t
///   dv_t   = β_t Λ_tᵀ k_t
///   dk_t   = β_t (Λ_t δ_t − S̃_t (Λ_tᵀ k_t))
///   dβ_t   = k_tᵀ Λ_t δ_t
///   dα_t   = sum_dv(S_{t-1} ⊙ M_t), M_t = (I − β_t k_t k_tᵀ) Λ_t
///   dg_t   = dα_t ⊙ α_t
///   Λ_{t-1} = Diag(α_t) M_t
/// ```
///
/// Memory stays bounded: pass 1 retains only the 64-token chunk start
/// states; pass 2 walks chunks in reverse and recomputes the per-token
/// states within each chunk (transient, one chunk at a time).
/// Returns `(dq, dk, dv, dlog_decay, dbeta)` in the input dtype.
#[allow(clippy::too_many_arguments)]
pub fn kda_chunk_backward(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    log_decay: &MetalTensor,
    beta: &MetalTensor,
    g: &MetalTensor,
    scale: f64,
) -> crate::err::Res<(
    MetalTensor,
    MetalTensor,
    MetalTensor,
    MetalTensor,
    MetalTensor,
)> {
    const CHUNK: usize = 64;
    let in_dtype = q.dtype;
    let q = super::ops::to_f32(q)?;
    let k = super::ops::to_f32(k)?;
    let v = super::ops::to_f32(v)?;
    let log_decay = super::ops::to_f32(log_decay)?;
    let beta = super::ops::to_f32(beta)?;
    let g = super::ops::to_f32(g)?;
    let work = DType::F32;

    let dims = q.layout.shape().to_vec();
    let r = dims.len();
    let (t_total, dk) = (dims[r - 2], dims[r - 1]);
    let dv = v.layout.shape()[r - 1];
    let bh: usize = dims[..r - 2].iter().product();

    let flatten = |x: &MetalTensor, d: usize| -> crate::err::Res<MetalTensor> {
        super::ops::contiguous(&MetalTensor {
            buffer: x.buffer.clone(),
            layout: Layout::contiguous(vec![bh, t_total, d]),
            dtype: x.dtype,
        })
    };
    let q3 = flatten(&q, dk)?;
    let k3 = flatten(&k, dk)?;
    let v3 = flatten(&v, dv)?;
    let ld3 = flatten(&log_decay, dk)?;
    let b3 = flatten(&beta, 1)?;
    let g3 = flatten(&g, dv)?;

    let tok = |x: &MetalTensor, t: usize| narrow(x, 1, t, 1); // [BH, 1, D]
    let col = |x: &MetalTensor, t: usize| transpose_last2(&narrow(x, 1, t, 1)?); // [BH, D, 1]

    // Pass 1: chunk start states via the per-token recurrence.
    let mut starts: Vec<MetalTensor> = Vec::new();
    let mut state = fill(&[bh, dk, dv], 0.0, work)?;
    let mut t0 = 0;
    while t0 < t_total {
        let c = CHUNK.min(t_total - t0);
        starts.push(state.clone());
        for i in 0..c {
            let alpha = transpose_last2(&unary(&tok(&ld3, t0 + i)?, UnOp::Exp)?)?; // [BH, dk, 1]
            let sdec = binary(&state, &alpha, BinOp::Mul)?;
            let k_col = col(&k3, t0 + i)?;
            let delta = binary(
                &col(&v3, t0 + i)?,
                &matmul(&transpose_last2(&sdec)?, &k_col)?,
                BinOp::Sub,
            )?; // [BH, dv, 1]
            state = binary(
                &sdec,
                &matmul(
                    &k_col,
                    &binary(&transpose_last2(&delta)?, &tok(&b3, t0 + i)?, BinOp::Mul)?,
                )?,
                BinOp::Add,
            )?;
        }
        t0 += c;
    }

    // Pass 2: reverse adjoint walk with per-chunk forward recompute.
    let mut lam = fill(&[bh, dk, dv], 0.0, work)?;
    let mut dq_rows: Vec<MetalTensor> = Vec::new();
    let mut dk_rows: Vec<MetalTensor> = Vec::new();
    let mut dv_rows: Vec<MetalTensor> = Vec::new();
    let mut dg_rows: Vec<MetalTensor> = Vec::new();
    let mut db_rows: Vec<MetalTensor> = Vec::new();
    for ci in (0..starts.len()).rev() {
        let t0 = ci * CHUNK;
        let c = CHUNK.min(t_total - t0);
        // Recompute this chunk's per-token states.
        let mut sdecs: Vec<MetalTensor> = Vec::with_capacity(c);
        let mut states_t: Vec<MetalTensor> = Vec::with_capacity(c);
        let mut deltas: Vec<MetalTensor> = Vec::with_capacity(c);
        let mut s = starts[ci].clone();
        for i in 0..c {
            let alpha = transpose_last2(&unary(&tok(&ld3, t0 + i)?, UnOp::Exp)?)?;
            let sdec = binary(&s, &alpha, BinOp::Mul)?;
            let k_col = col(&k3, t0 + i)?;
            let delta = binary(
                &col(&v3, t0 + i)?,
                &matmul(&transpose_last2(&sdec)?, &k_col)?,
                BinOp::Sub,
            )?;
            s = binary(
                &sdec,
                &matmul(
                    &k_col,
                    &binary(&transpose_last2(&delta)?, &tok(&b3, t0 + i)?, BinOp::Mul)?,
                )?,
                BinOp::Add,
            )?;
            sdecs.push(sdec);
            deltas.push(delta);
            states_t.push(s.clone());
        }
        for i in (0..c).rev() {
            let t = t0 + i;
            let q_col = col(&q3, t)?; // [BH, dk, 1]
            let g_row = tok(&g3, t)?; // [BH, 1, dv]
            let k_col = col(&k3, t)?;
            let b_t = tok(&b3, t)?; // [BH, 1, 1]
            lam = binary(
                &lam,
                &binary(
                    &matmul(&q_col, &g_row)?,
                    &fill(&[bh, dk, dv], scale, work)?,
                    BinOp::Mul,
                )?,
                BinOp::Add,
            )?;
            let g_col = transpose_last2(&g_row)?; // [BH, dv, 1]
            dq_rows.push(transpose_last2(&binary(
                &matmul(&states_t[i], &g_col)?,
                &fill(&[bh, dk, 1], scale, work)?,
                BinOp::Mul,
            )?)?);
            let lam_k = matmul(&transpose_last2(&lam)?, &k_col)?; // [BH, dv, 1]
            dv_rows.push(transpose_last2(&binary(&lam_k, &b_t, BinOp::Mul)?)?);
            let lam_delta = matmul(&lam, &deltas[i])?; // [BH, dk, 1]
            dk_rows.push(transpose_last2(&binary(
                &binary(&lam_delta, &matmul(&sdecs[i], &lam_k)?, BinOp::Sub)?,
                &transpose_last2(&b_t)?,
                BinOp::Mul,
            )?)?);
            db_rows.push(reduce(
                &binary(&k_col, &lam_delta, BinOp::Mul)?,
                &[1],
                true,
                ReduceOp::Sum,
            )?); // [BH, 1, 1]
                 // M = (I - beta k kᵀ) Λ
            let m_ = binary(
                &lam,
                &binary(
                    &matmul(&k_col, &transpose_last2(&lam_k)?)?,
                    &transpose_last2(&b_t)?,
                    BinOp::Mul,
                )?,
                BinOp::Sub,
            )?;
            let s_prev = if i == 0 {
                &starts[ci]
            } else {
                &states_t[i - 1]
            };
            let dalpha = reduce(&binary(s_prev, &m_, BinOp::Mul)?, &[2], true, ReduceOp::Sum)?; // [BH, dk, 1]
            let alpha = transpose_last2(&unary(&tok(&ld3, t)?, UnOp::Exp)?)?;
            dg_rows.push(transpose_last2(&binary(&dalpha, &alpha, BinOp::Mul)?)?);
            lam = binary(&alpha, &m_, BinOp::Mul)?;
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
    let assemble = |rows: Vec<MetalTensor>, width: usize| -> crate::err::Res<MetalTensor> {
        let mut shape = dims.clone();
        shape[r - 1] = width;
        let joined = cat_all(&rows, 1)?;
        let out = super::ops::contiguous(&MetalTensor {
            buffer: joined.buffer.clone(),
            layout: Layout::contiguous(shape),
            dtype: work,
        })?;
        super::ops::from_f32(&out, in_dtype)
    };
    Ok((
        assemble(dq_rows, dk)?,
        assemble(dk_rows, dk)?,
        assemble(dv_rows, dv)?,
        assemble(dg_rows, dk)?,
        assemble(db_rows, 1)?,
    ))
}

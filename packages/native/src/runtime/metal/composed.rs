use crate::fusion::ReduceOp;
use crate::runtime::dtype::DType;
use crate::runtime::layout::Layout;
use crate::runtime::metal::run::MetalTensor;
use super::ops::{binary, broadcast_to, cat, cast, compare, fill, gather, matmul, permute, reduce, unary, where_, BinOp, UnOp};

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

pub fn softmax_lastdim(x: &MetalTensor) -> crate::err::Res<MetalTensor> {
    let r = rank(x);
    let m = reduce(x, &[r - 1], true, ReduceOp::Max)?;
    let e = unary(&binary(x, &m, BinOp::Sub)?, UnOp::Exp)?;
    let s = reduce(&e, &[r - 1], true, ReduceOp::Sum)?;
    binary(&e, &s, BinOp::Div)
}

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
    let cpu = crate::runtime::cpu::Tensor::from_vec(data, vec![t, s]);
    upload_cpu(&cpu)
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

fn sdpa_scores(q: &MetalTensor, k: &MetalTensor, scale: f64, causal: bool) -> crate::err::Res<MetalTensor> {
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

pub fn sdpa_forward(q: &MetalTensor, k: &MetalTensor, v: &MetalTensor, scale: f64, causal: bool) -> crate::err::Res<MetalTensor> {
    let s = sdpa_scores(q, k, scale, causal)?;
    let p = softmax_lastdim(&s)?;
    matmul(&p, v)
}

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

pub fn layer_norm_forward(x: &MetalTensor, weight: &MetalTensor, bias: &MetalTensor, eps: f64) -> crate::err::Res<MetalTensor> {
    let r = rank(x);
    let k = weight.layout.shape().len();
    let dims: Vec<usize> = (r - k..r).collect();
    let mean = mean_dims(x, &dims)?;
    let centered = binary(x, &mean, BinOp::Sub)?;
    let var = mean_dims(&binary(&centered, &centered, BinOp::Mul)?, &dims)?;
    let inv = unary(&binary(&var, &fill(var.layout.shape(), eps, var.dtype)?, BinOp::Add)?, UnOp::Sqrt)?;
    let inv = unary(&inv, UnOp::Neg)?;
    let inv = unary(&inv, UnOp::Exp)?;
    binary(&binary(&binary(&centered, &inv, BinOp::Mul)?, weight, BinOp::Mul)?, bias, BinOp::Add)
}

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
    let rstd = unary(&binary(&var, &fill(var.layout.shape(), eps, var.dtype)?, BinOp::Add)?, UnOp::Sqrt)?;
    let rstd = unary(&unary(&rstd, UnOp::Neg)?, UnOp::Exp)?;
    let xh = binary(&centered, &rstd, BinOp::Mul)?;
    // dx = (dyw − mean(dyw) − x̂·mean(dyw·x̂)) · rstd
    let dyw = binary(g, weight, BinOp::Mul)?;
    let m1 = mean_dims(&dyw, &dims)?;
    let m2 = mean_dims(&binary(&dyw, &xh, BinOp::Mul)?, &dims)?;
    let dx = binary(&binary(&dyw, &m1, BinOp::Sub)?, &binary(&xh, &m2, BinOp::Mul)?, BinOp::Sub)?;
    let dx = binary(&dx, &rstd, BinOp::Mul)?;
    let dw = reduce(&binary(g, &xh, BinOp::Mul)?, &reduce_dims, false, ReduceOp::Sum)?;
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

fn ce_check_labels(target: &MetalTensor, ignored: &MetalTensor, classes: usize) -> crate::err::Res<()> {
    let invalid = match target.dtype {
        DType::I64 => {
            let lo = compare(target, &fill(target.layout.shape(), 0.0, DType::I64)?, BinOp::Lt)?;
            let hi = compare(target, &fill(target.layout.shape(), classes as f64, DType::I64)?, BinOp::Ge)?;
            let hi8 = cast(&hi, DType::U8)?;
            binary(&lo, &hi8, BinOp::Max)
        }
        DType::U32 => compare(target, &fill(target.layout.shape(), classes as f64, DType::U32)?, BinOp::Ge),
        _ => unreachable!(),
    }?;
    let active = compare(ignored, &fill(ignored.layout.shape(), 0.0, DType::U8)?, BinOp::Eq)?;
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

pub fn cross_entropy_forward(
    logits: &MetalTensor,
    target: &MetalTensor,
    ignore_index: i64,
) -> crate::err::Res<MetalTensor> {
    let r = rank(logits);
    let classes = logits.layout.shape()[r - 1];
    let ignored = ce_ignored_mask(target, ignore_index)?;
    let count = ce_active_count(&ignored, target.numel())?;
    if count == 0.0 {
        return crate::err::err("cross_entropy: no active targets (all positions are ignored)");
    }
    ce_check_labels(target, &ignored, classes)?;
    let lse = logsumexp_lastdim(logits)?;
    let zero_ids = fill(target.layout.shape(), 0.0, DType::I64)?;
    let safe_target = where_(&ignored, &zero_ids, &target_to_ids(target)?)?;
    let safe_ids = unsqueeze_last(&safe_target)?;
    let picked = gather(logits, r - 1, &safe_ids.to_u32_vec()?, safe_ids.layout.shape())?;
    let picked = squeeze_last(&picked)?;
    let per_position = binary(&squeeze_last(&lse)?, &picked, BinOp::Sub)?;
    let masked = where_(&ignored, &zeros_like(&per_position)?, &per_position)?;
    let all: Vec<usize> = (0..rank(&masked)).collect();
    let total = reduce(&masked, &all, true, ReduceOp::Sum)?;
    binary(&total, &fill(total.layout.shape(), 1.0 / count, total.dtype)?, BinOp::Mul)
}

pub fn cross_entropy_backward(
    logits: &MetalTensor,
    target: &MetalTensor,
    ignore_index: i64,
) -> crate::err::Res<MetalTensor> {
    let r = rank(logits);
    let ignored = ce_ignored_mask(target, ignore_index)?;
    let count = ce_active_count(&ignored, target.numel())?;
    if count == 0.0 {
        return crate::err::err("cross_entropy: no active targets (all positions are ignored)");
    }
    let p = softmax_lastdim(logits)?;
    let zero_ids = fill(target.layout.shape(), 0.0, DType::I64)?;
    let safe_target = where_(&ignored, &zero_ids, &target_to_ids(target)?)?;
    let ids = unsqueeze_last(&safe_target)?;
    let neg_ones = fill(ids.layout.shape(), -1.0, logits.dtype)?;
    let p = super::ops::scatter_add(&p, r - 1, &ids.to_u32_vec()?, &neg_ones)?;
    let keep = compare(&ignored, &fill(ignored.layout.shape(), 0.0, DType::U8)?, BinOp::Eq)?;
    let keep = unsqueeze_last(&keep)?;
    let masked = where_(&keep, &p, &zeros_like(&p)?)?;
    binary(&masked, &fill(masked.layout.shape(), 1.0 / count, masked.dtype)?, BinOp::Mul)
}

pub fn rotary_forward(x: &MetalTensor, offsets: &[usize], theta: f64, sign: f64) -> crate::err::Res<MetalTensor> {
    let dims = x.layout.shape();
    let r = dims.len();
    let (t, d) = (dims[r - 2], dims[r - 1]);
    let batch = dims[0];
    if offsets.len() != 1 && offsets.len() != batch {
        return crate::err::err(format!("rotary embedding: {} offsets for batch {batch}", offsets.len()));
    }
    let half = d / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 / d as f64) as f32)
        .collect();
    let inv_freq = upload_cpu(&crate::runtime::cpu::Tensor::from_vec(inv_freq, vec![1, half]))?;
    let positions: Vec<f32> = if offsets.len() == 1 {
        (0..t).map(|p| (offsets[0] + p) as f32).collect()
    } else {
        offsets
            .iter()
            .flat_map(|base| (0..t).map(move |p| (*base + p) as f32))
            .collect()
    };
    let rows = if offsets.len() == 1 { 1 } else { batch };
    let positions = upload_cpu(&crate::runtime::cpu::Tensor::from_vec(positions, vec![rows * t, 1]))?;
    let angles = matmul(&positions, &inv_freq)?;
    let angles = binary(&angles, &fill(&[rows * t, half], sign, DType::F32)?, BinOp::Mul)?;
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
    let out_first = binary(&binary(&first, &cos, BinOp::Mul)?, &binary(&second, &sin, BinOp::Mul)?, BinOp::Sub)?;
    let out_second = binary(&binary(&second, &cos, BinOp::Mul)?, &binary(&first, &sin, BinOp::Mul)?, BinOp::Add)?;
    cat(&out_first, &out_second, r - 1)
}

pub fn upload_cpu(t: &crate::runtime::cpu::Tensor) -> crate::err::Res<MetalTensor> {
    let t = t.contiguous();
    let bytes: Vec<u8> = match &t.buffer {
        crate::runtime::cpu::CpuBuffer::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        crate::runtime::cpu::CpuBuffer::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        crate::runtime::cpu::CpuBuffer::F16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        crate::runtime::cpu::CpuBuffer::BF16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        crate::runtime::cpu::CpuBuffer::U8(v) => v.as_slice().to_vec(),
        crate::runtime::cpu::CpuBuffer::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        crate::runtime::cpu::CpuBuffer::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    Ok(MetalTensor {
        buffer: crate::runtime::metal::device::MetalDevice::get().upload_bytes(&bytes),
        layout: Layout::contiguous(t.shape().to_vec()),
        dtype: t.dtype(),
    })
}

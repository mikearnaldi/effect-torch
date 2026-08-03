use super::tensor::{CpuBuffer, Tensor};
use crate::runtime::dtype::DType;
use crate::runtime::layout::Layout;

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

pub fn layer_norm_forward(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
    let r = rank(x);
    let k = weight.shape().len();
    let dims: Vec<usize> = (r - k..r).collect();
    let mean = x.mean(&dims);
    let centered = x.sub(&mean);
    let var = centered.mul(&centered).mean(&dims);
    let inv = var.add(&Tensor::full(var.shape(), eps, var.dtype())).sqrt().powf(-1.0);
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
    let rstd = var.add(&Tensor::full(var.shape(), eps, var.dtype())).sqrt().powf(-1.0);
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
    let CpuBuffer::F64(v) = &c.buffer else { unreachable!() };
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

pub fn cross_entropy_forward(
    logits: &Tensor,
    target: &Tensor,
    ignore_index: i64,
) -> Result<Tensor, String> {
    let r = rank(logits);
    let classes = logits.shape()[r - 1];
    let ignored = ce_ignored_mask(target, ignore_index);
    let count = ce_active_count(&ignored, target.numel());
    if count == 0.0 {
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
    let masked = Tensor::zeros(per_position.shape(), per_position.dtype()).where_(&ignored, &per_position);
    let total = masked.sum(&all_dims(&masked));
    let scale = Tensor::full(total.shape(), 1.0 / count, total.dtype());
    Ok(total.mul(&scale))
}

pub fn cross_entropy_backward(
    logits: &Tensor,
    target: &Tensor,
    ignore_index: i64,
) -> Result<Tensor, String> {
    let r = rank(logits);
    let ignored = ce_ignored_mask(target, ignore_index);
    let count = ce_active_count(&ignored, target.numel());
    if count == 0.0 {
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
    let scale = Tensor::full(masked.shape(), 1.0 / count, masked.dtype());
    Ok(masked.mul(&scale))
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
    let continued = v.mul(&fl(v, momentum)).add(&g.mul(&fl(&g, 1.0 - dampening)));
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

pub fn rotary_forward(x: &Tensor, offsets: &[usize], theta: f64, sign: f64) -> Result<Tensor, String> {
    let dims = x.shape();
    let r = dims.len();
    let (t, d) = (dims[r - 2], dims[r - 1]);
    let batch = dims[0];
    if offsets.len() != 1 && offsets.len() != batch {
        return Err(format!("rotary embedding: {} offsets for batch {batch}", offsets.len()));
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
    let angles = positions.matmul(&inv_freq).mul(&Tensor::full(&[rows * t, half], sign, DType::F32));
    let mut table_shape = vec![1usize; r - 2];
    if offsets.len() != 1 {
        table_shape[0] = batch;
    }
    table_shape.extend([t, half]);
    let cos = angles.cos().contiguous().view(Layout::contiguous(table_shape.clone()));
    let sin = angles.sin().contiguous().view(Layout::contiguous(table_shape));
    let first = narrow(x, r - 1, 0, half);
    let second = narrow(x, r - 1, half, half);
    let out_first = first.mul(&cos).sub(&second.mul(&sin));
    let out_second = second.mul(&cos).add(&first.mul(&sin));
    Ok(Tensor::cat(&[&out_first, &out_second], r - 1).contiguous())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_data(t: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(v) = &t.buffer else { panic!() };
        v.as_slice().to_vec()
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
        let loss = cross_entropy_forward(&logits, &target, -100).unwrap();
        let lse = logsumexp_lastdim(&logits);
        let CpuBuffer::F32(v) = &lse.buffer else { panic!() };
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
}

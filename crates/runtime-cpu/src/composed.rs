use super::tensor::{CpuBuffer, Tensor};
use effect_torch_graph::CrossEntropyReduction;
use effect_torch_runtime::{DType, Layout};

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

// --- Kimi Delta Attention (RFC 0018) ---

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

    fn f32_data(t: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(v) = &t.buffer else {
            panic!()
        };
        v.as_slice().to_vec()
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
        let mut fd = |input: &Tensor, analytic: &Tensor, which: usize, name: &str| {
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
}

use super::device::{set_buffer, MetalDevice};
use super::run::MetalTensor;
use objc2_metal::MTLComputeCommandEncoder;

fn key(parts: &[u64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for p in parts {
        p.hash(&mut h);
    }
    h.finish()
}

const HEADER: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

fn dims3(t: &MetalTensor) -> (usize, usize, usize) {
    let s = t.layout.shape();
    (s[0], s[1], s[2])
}

fn dims4(t: &MetalTensor) -> (usize, usize, usize, usize) {
    let s = t.layout.shape();
    (s[0], s[1], s[2], s[3])
}

fn grid_for(dev: &MetalDevice, pipeline: &super::device::Pipeline, n: usize, binds: &[(usize, &super::device::Buffer, usize)], scalars: &[(usize, u32)]) {
    let padded = n.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        for &(idx, buf, off) in binds {
            set_buffer(e, idx, buf, off);
        }
        for &(idx, v) in scalars {
            super::device::set_bytes(e, idx, &v);
        }
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
}

pub fn conv1d(dev: &MetalDevice, x: &MetalTensor, w: &MetalTensor, stride: usize, padding: usize, dilation: usize, groups: usize) -> Result<MetalTensor, String> {
    let (n, c_in, l) = dims3(x);
    let (c_out, c_per, k) = dims3(w);
    let cout_per = c_out / groups;
    let l_out = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    let out = MetalTensor::zeros(dev, vec![n, c_out, l_out], x.dtype);
    let total = n * c_out * l_out;
    if total == 0 {
        return Ok(out);
    }
    let src = format!(
        r#"{HEADER}
kernel void et_conv1d(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    const uint i = gid % {l_out}u;
    const uint oc = (gid / {l_out}u) % {c_out}u;
    const uint b = gid / ({l_out}u * {c_out}u);
    const uint g = oc / {cout_per}u;
    float acc = 0.0f;
    for (uint ci = 0u; ci < {c_per}u; ++ci) {{
        const uint ic = g * {c_per}u + ci;
        for (uint kk = 0u; kk < {k}u; ++kk) {{
            const long pos = (long)(i * {stride}u + kk * {dilation}u) - {padding}l;
            if (pos >= 0 && pos < {l}l) {{
                acc += x[(b * {c_in}u + ic) * {l}u + (uint)pos] * w[(oc * {c_per}u + ci) * {k}u + kk];
            }}
        }}
    }}
    out[gid] = acc;
}}
"#
    );
    let pipeline = dev.compile(key(&[0xC0A1, n as u64, c_in as u64, l as u64, c_out as u64, k as u64, stride as u64, padding as u64, dilation as u64, groups as u64]), &src, "et_conv1d")?;
    grid_for(dev, &pipeline, total, &[
        (0, &x.buffer, x.layout.offset() * 4),
        (1, &w.buffer, w.layout.offset() * 4),
        (2, &out.buffer, 0),
    ], &[]);
    Ok(out)
}

pub fn conv2d(dev: &MetalDevice, x: &MetalTensor, w: &MetalTensor, stride: usize, padding: usize, dilation: usize, groups: usize) -> Result<MetalTensor, String> {
    let (n, c_in, h, wd) = dims4(x);
    let (c_out, c_per, kh, kw) = dims4(w);
    let cout_per = c_out / groups;
    let oh = (h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
    let ow = (wd + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
    let out = MetalTensor::zeros(dev, vec![n, c_out, oh, ow], x.dtype);
    let total = n * c_out * oh * ow;
    if total == 0 {
        return Ok(out);
    }
    let src = format!(
        r#"{HEADER}
kernel void et_conv2d(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    const uint j = gid % {ow}u;
    const uint i = (gid / {ow}u) % {oh}u;
    const uint oc = (gid / ({ow}u * {oh}u)) % {c_out}u;
    const uint b = gid / ({ow}u * {oh}u * {c_out}u);
    const uint g = oc / {cout_per}u;
    float acc = 0.0f;
    for (uint ci = 0u; ci < {c_per}u; ++ci) {{
        const uint ic = g * {c_per}u + ci;
        for (uint ky = 0u; ky < {kh}u; ++ky) {{
            const long py = (long)(i * {stride}u + ky * {dilation}u) - {padding}l;
            if (py < 0 || py >= {h}l) continue;
            for (uint kx = 0u; kx < {kw}u; ++kx) {{
                const long px = (long)(j * {stride}u + kx * {dilation}u) - {padding}l;
                if (px < 0 || px >= {wd}l) continue;
                acc += x[((b * {c_in}u + ic) * {h}u + (uint)py) * {wd}u + (uint)px]
                       * w[((oc * {c_per}u + ci) * {kh}u + ky) * {kw}u + kx];
            }}
        }}
    }}
    out[gid] = acc;
}}
"#
    );
    let pipeline = dev.compile(key(&[0xC0A2, n as u64, c_in as u64, h as u64, wd as u64, c_out as u64, kh as u64, kw as u64, stride as u64, padding as u64, dilation as u64, groups as u64]), &src, "et_conv2d")?;
    grid_for(dev, &pipeline, total, &[
        (0, &x.buffer, x.layout.offset() * 4),
        (1, &w.buffer, w.layout.offset() * 4),
        (2, &out.buffer, 0),
    ], &[]);
    Ok(out)
}

pub fn conv_transpose1d(dev: &MetalDevice, x: &MetalTensor, w: &MetalTensor, stride: usize, padding: usize, output_padding: usize, dilation: usize, groups: usize) -> Result<MetalTensor, String> {
    let (n, c_in, l) = dims3(x);
    assert_eq!(c_in, w.layout.shape()[0], "conv_transpose1d: weight dim 0 must equal input channels");
    let (c_out_per_g, k) = (w.layout.shape()[1], w.layout.shape()[2]);
    let cin_per = c_in / groups;
    let c_out = c_out_per_g * groups;
    let l_out = (l - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;
    let out = MetalTensor::zeros(dev, vec![n, c_out, l_out], x.dtype);
    let total = n * c_out * l_out;
    if total == 0 {
        return Ok(out);
    }
    let src = format!(
        r#"{HEADER}
kernel void et_convt1d(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    const uint o = gid % {l_out}u;
    const uint oc = (gid / {l_out}u) % {c_out}u;
    const uint b = gid / ({l_out}u * {c_out}u);
    const uint g = oc / {c_out_per_g}u;
    float acc = 0.0f;
    for (uint ci = 0u; ci < {cin_per}u; ++ci) {{
        const uint ic = g * {cin_per}u + ci;
        for (uint kk = 0u; kk < {k}u; ++kk) {{
            const long num = (long)(o + {padding}u) - (long)(kk * {dilation}u);
            if (num < 0) continue;
            if ((uint)num % {stride}u != 0u) continue;
            const uint i = (uint)num / {stride}u;
            if (i < {l}u) {{
                acc += x[(b * {c_in}u + ic) * {l}u + i] * w[(ic * {c_out_per_g}u + (oc % {c_out_per_g}u)) * {k}u + kk];
            }}
        }}
    }}
    out[gid] = acc;
}}
"#
    );
    let pipeline = dev.compile(key(&[0xC0B1, n as u64, c_in as u64, l as u64, c_out as u64, k as u64, stride as u64, padding as u64, output_padding as u64, dilation as u64, groups as u64]), &src, "et_convt1d")?;
    grid_for(dev, &pipeline, total, &[
        (0, &x.buffer, x.layout.offset() * 4),
        (1, &w.buffer, w.layout.offset() * 4),
        (2, &out.buffer, 0),
    ], &[]);
    Ok(out)
}

pub fn conv_transpose2d(dev: &MetalDevice, x: &MetalTensor, w: &MetalTensor, stride: usize, padding: usize, output_padding: usize, dilation: usize, groups: usize) -> Result<MetalTensor, String> {
    let (n, c_in, h, wd) = dims4(x);
    assert_eq!(c_in, w.layout.shape()[0], "conv_transpose2d: weight dim 0 must equal input channels");
    let (c_out_per_g, kh, kw) = (w.layout.shape()[1], w.layout.shape()[2], w.layout.shape()[3]);
    let cin_per = c_in / groups;
    let c_out = c_out_per_g * groups;
    let oh = (h - 1) * stride - 2 * padding + dilation * (kh - 1) + output_padding + 1;
    let ow = (wd - 1) * stride - 2 * padding + dilation * (kw - 1) + output_padding + 1;
    let out = MetalTensor::zeros(dev, vec![n, c_out, oh, ow], x.dtype);
    let total = n * c_out * oh * ow;
    if total == 0 {
        return Ok(out);
    }
    let src = format!(
        r#"{HEADER}
kernel void et_convt2d(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    const uint oj = gid % {ow}u;
    const uint oi = (gid / {ow}u) % {oh}u;
    const uint oc = (gid / ({ow}u * {oh}u)) % {c_out}u;
    const uint b = gid / ({ow}u * {oh}u * {c_out}u);
    const uint g = oc / {c_out_per_g}u;
    float acc = 0.0f;
    for (uint ci = 0u; ci < {cin_per}u; ++ci) {{
        const uint ic = g * {cin_per}u + ci;
        for (uint ky = 0u; ky < {kh}u; ++ky) {{
            const long numy = (long)(oi + {padding}u) - (long)(ky * {dilation}u);
            if (numy < 0 || (uint)numy % {stride}u != 0u) continue;
            const uint i = (uint)numy / {stride}u;
            if (i >= {h}u) continue;
            for (uint kx = 0u; kx < {kw}u; ++kx) {{
                const long numx = (long)(oj + {padding}u) - (long)(kx * {dilation}u);
                if (numx < 0 || (uint)numx % {stride}u != 0u) continue;
                const uint j = (uint)numx / {stride}u;
                if (j >= {wd}u) continue;
                acc += x[((b * {c_in}u + ic) * {h}u + i) * {wd}u + j]
                       * w[((ic * {c_out_per_g}u + (oc % {c_out_per_g}u)) * {kh}u + ky) * {kw}u + kx];
            }}
        }}
    }}
    out[gid] = acc;
}}
"#
    );
    let pipeline = dev.compile(key(&[0xC0B2, n as u64, c_in as u64, h as u64, wd as u64, c_out as u64, kh as u64, kw as u64, stride as u64, padding as u64, output_padding as u64, dilation as u64, groups as u64]), &src, "et_convt2d")?;
    grid_for(dev, &pipeline, total, &[
        (0, &x.buffer, x.layout.offset() * 4),
        (1, &w.buffer, w.layout.offset() * 4),
        (2, &out.buffer, 0),
    ], &[]);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_w(dev: &MetalDevice, x: &MetalTensor, g: &MetalTensor, kernel: [usize; 2], out_channels: usize, stride: usize, padding: usize, dilation: usize, groups: usize) -> Result<MetalTensor, String> {
    let (n, c_in, _, _) = dims4(x);
    let (_, _, oh, ow) = dims4(g);
    let (kh, kw) = (kernel[0], kernel[1]);
    let c_per = c_in / groups;
    let cout_per = out_channels / groups;
    let out = MetalTensor::zeros(dev, vec![out_channels, c_per, kh, kw], x.dtype);
    let total = out_channels * c_per * kh * kw;
    if total == 0 {
        return Ok(out);
    }
    let src = format!(
        r#"{HEADER}
kernel void et_conv2d_bw(
    device const float* x [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    const uint kx = gid % {kw}u;
    const uint ky = (gid / {kw}u) % {kh}u;
    const uint ci = (gid / ({kw}u * {kh}u)) % {c_per}u;
    const uint oc = gid / ({kw}u * {kh}u * {c_per}u);
    const uint gi = oc / {cout_per}u;
    const uint ic = gi * {c_per}u + ci;
    float acc = 0.0f;
    for (uint b = 0u; b < {n}u; ++b) {{
        for (uint i = 0u; i < {oh}u; ++i) {{
            const long py = (long)(i * {stride}u + ky * {dilation}u) - {padding}l;
            if (py < 0 || py >= {h_dim}l) continue;
            for (uint j = 0u; j < {ow}u; ++j) {{
                const long px = (long)(j * {stride}u + kx * {dilation}u) - {padding}l;
                if (px < 0 || px >= {w_dim}l) continue;
                acc += g[((b * {out_channels}u + oc) * {oh}u + i) * {ow}u + j]
                       * x[((b * {c_in}u + ic) * {h_dim}u + (uint)py) * {w_dim}u + (uint)px];
            }}
        }}
    }}
    out[gid] = acc;
}}
"#,
        h_dim = x.layout.shape()[2],
        w_dim = x.layout.shape()[3]
    );
    let pipeline = dev.compile(key(&[0xC0C2, n as u64, c_in as u64, oh as u64, ow as u64, out_channels as u64, kh as u64, kw as u64, stride as u64, padding as u64, dilation as u64, groups as u64]), &src, "et_conv2d_bw")?;
    grid_for(dev, &pipeline, total, &[
        (0, &x.buffer, x.layout.offset() * 4),
        (1, &g.buffer, g.layout.offset() * 4),
        (2, &out.buffer, 0),
    ], &[]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_basic() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, vec![1f32, 2., 3., 4.], vec![1, 1, 4]);
        let w = MetalTensor::from_f32(dev, vec![1f32, 1.], vec![1, 1, 2]);
        let y = conv1d(dev, &x, &w, 1, 0, 1, 1).unwrap();
        dev.synchronize();
        assert_eq!(y.read_f32(), vec![3., 5., 7.]);
    }

    #[test]
    fn conv2d_basic() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, (1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3]);
        let w = MetalTensor::from_f32(dev, vec![1f32, 0., 0., 1.], vec![1, 1, 2, 2]);
        let y = conv2d(dev, &x, &w, 1, 0, 1, 1).unwrap();
        dev.synchronize();
        assert_eq!(y.read_f32(), vec![6., 8., 12., 14.]);
    }

    #[test]
    fn conv_transpose1d_basic() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, vec![1f32, 2.], vec![1, 1, 2]);
        let w = MetalTensor::from_f32(dev, vec![1f32, 1.], vec![1, 1, 2]);
        let y = conv_transpose1d(dev, &x, &w, 2, 0, 0, 1, 1).unwrap();
        dev.synchronize();
        assert_eq!(y.read_f32(), vec![1., 1., 2., 2.]);
    }

    #[test]
    fn conv2d_backward_w_basic() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, (1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3]);
        let g = MetalTensor::from_f32(dev, vec![1f32, 1., 1., 1.], vec![1, 1, 2, 2]);
        let dw = conv2d_backward_w(dev, &x, &g, [2, 2], 1, 1, 0, 1, 1).unwrap();
        dev.synchronize();
        assert_eq!(dw.read_f32(), vec![12., 16., 24., 28.]);
    }
}

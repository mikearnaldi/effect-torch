//! Fused causal depthwise short convolution on Metal (RFC 0018): the
//! KDA local-mixing conv y[t,c] = sum_j w[c,j] * x[t-K+1+j, c] with zero
//! or carried history. One launch per call replaces the composed
//! pad/conv1d/transpose chain; the stateful ConvState decode/prefill
//! path passes the slot's [K-1, C] window as history and receives the
//! shifted window back (only the `advance` real rows shift in — chunked
//! prefill right-pads). All channel counts and kernel sizes are
//! supported; the dtype gate fails loud.

use crate::runtime::dtype::DType;

pub fn supported_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16)
}

#[cfg(target_os = "macos")]
pub use metal::{backward_w, backward_x, forward};

#[cfg(target_os = "macos")]
mod metal {
    use crate::runtime::dtype::DType;
    use crate::runtime::metal::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
    use crate::runtime::metal::run::MetalTensor;

    use objc2_metal::MTLComputeCommandEncoder;

    fn wrap_contig(t: &MetalTensor) -> crate::err::Res<MetalTensor> {
        if t.layout.is_contiguous() && t.layout.offset() == 0 {
            Ok(t.clone())
        } else {
            crate::runtime::metal::kernels::strided_copy(MetalDevice::get(), t)
        }
    }

    fn ty_of(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "float",
            DType::BF16 => "bfloat",
            other => unreachable!("shortconv: unsupported dtype {other:?}"),
        }
    }

    fn fwd_source(dtype: DType) -> String {
        let ty = ty_of(dtype);
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}

// One thread per output element, addressed by a 3D grid (channel, row,
// batch) — no div/mod index math. x [.., steps, C], optional f32
// history [K-1, C] (single-sequence use only), y mirrors x; when flag 2
// is set the last K-1 real rows (of `advance`) plus surviving history
// write the shifted window to WIN.
kernel void et_shortconv_fwd(
    device const T* X [[buffer(0)]],
    device const T* W [[buffer(1)]],
    device const float* HIST [[buffer(2)]],
    device T* Y [[buffer(3)]],
    device float* WIN [[buffer(4)]],
    constant uint& steps [[buffer(5)]],
    constant uint& C [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& advance [[buffer(8)]],
    constant uint& flags [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= C) return;
    const bool has_hist = (flags & 1u) != 0u;
    const bool write_win = (flags & 2u) != 0u;
    const ulong rowBase = (ulong)gid.z * steps * C;
    if (t < steps) {{
        float acc = 0.0f;
        for (uint j = 0; j < K; j++) {{
            const long s = long(t) + long(j) - long(K - 1);
            if (s >= 0) {{
                acc += float(W[c * K + j]) * float(X[rowBase + ulong(s) * C + c]);
            }} else if (has_hist) {{
                acc += float(W[c * K + j]) * HIST[ulong(K - 1 + s) * C + c];
            }}
        }}
        Y[rowBase + ulong(t) * C + c] = T(acc);
    }}
    if (write_win && t < K - 1 && gid.z == 0) {{
        // Window row t holds real position advance-(K-1)+t (negative:
        // the surviving history row K-1+s).
        const long s = long(advance) - long(K - 1) + long(t);
        float val = 0.0f;
        if (s >= 0) {{
            val = float(X[rowBase + ulong(s) * C + c]);
        }} else if (has_hist) {{
            val = HIST[ulong(K - 1 + s) * C + c];
        }}
        WIN[ulong(t) * C + c] = val;
    }}
}}
"#,
            ty = ty,
        )
    }
    fn bwd_x_source(dtype: DType) -> String {
        let ty = ty_of(dtype);
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}

// dx[s] = sum_j w[:, K-1-j] * g[s+j]: the full correlation against the
// right-zero-padded cotangent. One thread per element.
kernel void et_shortconv_bwd_x(
    device const T* G [[buffer(0)]],
    device const T* W [[buffer(1)]],
    device T* dX [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& steps [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= C || t >= steps) return;
    const ulong rowBase = (ulong)gid.z * steps * C;
    float acc = 0.0f;
    for (uint j = 0; j < K; j++) {{
        if (t + j < steps) {{
            acc += float(W[c * K + (K - 1 - j)]) * float(G[rowBase + ulong(t + j) * C + c]);
        }}
    }}
    dX[rowBase + ulong(t) * C + c] = T(acc);
}}
"#,
            ty = ty,
        )
    }

    fn bwd_w_source(dtype: DType) -> String {
        let ty = ty_of(dtype);
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

#define T {ty}

// dw[c, j] = sum over batch and t of g[t, c] * x[t-K+1+j, c] over the
// causal window. One thread per (channel, tap); the loops are serial.
kernel void et_shortconv_bwd_w(
    device const T* X [[buffer(0)]],
    device const T* G [[buffer(1)]],
    device T* dW [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& steps [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 gid2 [[thread_position_in_grid]]
) {{
    const ulong i = ulong(gid2.y) * {wide}ul + ulong(gid2.x);
    if (i >= ulong(C) * K) return;
    const uint j = uint(i % K);
    const uint c = uint(i / K);
    float acc = 0.0f;
    for (uint b = 0; b < batch; b++) {{
        const ulong base = ulong(b) * steps * C;
        for (uint t = K - 1 - j; t < steps; t++) {{
            acc += float(G[base + ulong(t) * C + c])
                * float(X[base + ulong(t - (K - 1 - j)) * C + c]);
        }}
    }}
    dW[i] = T(acc);
}}
"#,
            ty = ty,
            wide = MetalDevice::WIDE,
        )
    }

    fn pipeline(
        key_tag: u32,
        dtype: DType,
        name: &'static str,
        make_src: impl Fn() -> String,
    ) -> crate::err::Res<Pipeline> {
        MetalDevice::get().compile_lazy(((key_tag as u64) << 8) | dtype as u64, name, make_src)
    }

    fn grid_over(n: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
        // grid_flat yields total-THREAD sizes: pair with dispatchThreads.
        MetalDevice::grid_flat(n.div_ceil(256) * 256)
    }

    // 3D (channels, rows, batch) thread grid, 32×8 threadgroups.
    fn grid3d(
        c: usize,
        steps: usize,
        batch: usize,
    ) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
        (
            MetalDevice::grid(c.div_ceil(32) * 32, steps.div_ceil(8) * 8, batch),
            MetalDevice::grid(32, 8, 1),
        )
    }

    // Forward: x [.., steps, C] (any leading batch without history; one
    // sequence with history), weight [C, K]; optional f32 history
    // [K-1, C]; when requested returns the shifted f32 window.
    pub fn forward(
        x: &MetalTensor,
        weight: &MetalTensor,
        history: Option<&MetalTensor>,
        advance: usize,
        write_window: bool,
    ) -> crate::err::Res<(MetalTensor, Option<MetalTensor>)> {
        let dtype = x.dtype;
        let shape = x.layout.shape().to_vec();
        let r = shape.len();
        let (steps, c) = (shape[r - 2], shape[r - 1]);
        let batch: usize = shape[..r - 2].iter().product();
        let kk = weight.layout.shape()[1];
        let x = wrap_contig(x)?;
        let weight = wrap_contig(weight)?;
        let dev = MetalDevice::get();
        let pipe = pipeline(0xC041, dtype, "et_shortconv_fwd", || fwd_source(dtype))?;
        let out = dev.alloc(batch * steps * c, dtype);
        let dummy = dev.alloc(1, DType::F32);
        let hist_buf = match history {
            Some(h) => wrap_contig(h)?.buffer.clone(),
            None => dummy.clone(),
        };
        let win_buf = if write_window {
            dev.alloc((kk - 1) * c, DType::F32)
        } else {
            dummy.clone()
        };
        let mut flags = 0u32;
        if history.is_some() {
            flags |= 1;
        }
        if write_window {
            flags |= 2;
        }
        let elem = |off: usize| off * dtype.size_in_bytes();
        let (grid, tg) = grid3d(c, steps.max(if write_window { kk - 1 } else { 0 }), batch);
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, elem(x.layout.offset()));
            set_buffer(e, 1, &weight.buffer, elem(weight.layout.offset()));
            set_buffer(e, 2, &hist_buf, 0);
            set_buffer(e, 3, &out, 0);
            set_buffer(e, 4, &win_buf, 0);
            set_bytes(e, 5, &(steps as u32));
            set_bytes(e, 6, &(c as u32));
            set_bytes(e, 7, &(kk as u32));
            set_bytes(e, 8, &(advance as u32));
            set_bytes(e, 9, &flags);
            e.dispatchThreads_threadsPerThreadgroup(grid, tg);
        });
        let y = MetalTensor {
            buffer: out,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype,
        };
        let window = if write_window {
            Some(MetalTensor {
                buffer: win_buf,
                layout: crate::runtime::layout::Layout::contiguous(vec![kk - 1, c]),
                dtype: DType::F32,
            })
        } else {
            None
        };
        Ok((y, window))
    }

    // dx of the short conv: g [.., steps, C], weight [C, K] -> g's shape.
    pub fn backward_x(g: &MetalTensor, weight: &MetalTensor) -> crate::err::Res<MetalTensor> {
        let dtype = g.dtype;
        let shape = g.layout.shape().to_vec();
        let r = shape.len();
        let (steps, c) = (shape[r - 2], shape[r - 1]);
        let batch: usize = shape[..r - 2].iter().product();
        let kk = weight.layout.shape()[1];
        let g = wrap_contig(g)?;
        let weight = wrap_contig(weight)?;
        let dev = MetalDevice::get();
        let pipe = pipeline(0xC042, dtype, "et_shortconv_bwd_x", || bwd_x_source(dtype))?;
        let out = dev.alloc(batch * steps * c, dtype);
        let elem = |off: usize| off * dtype.size_in_bytes();
        let (grid, tg) = grid3d(c, steps, batch);
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &g.buffer, elem(g.layout.offset()));
            set_buffer(e, 1, &weight.buffer, elem(weight.layout.offset()));
            set_buffer(e, 2, &out, 0);
            set_bytes(e, 3, &(batch as u32));
            set_bytes(e, 4, &(steps as u32));
            set_bytes(e, 5, &(c as u32));
            set_bytes(e, 6, &(kk as u32));
            e.dispatchThreads_threadsPerThreadgroup(grid, tg);
        });
        Ok(MetalTensor {
            buffer: out,
            layout: crate::runtime::layout::Layout::contiguous(shape),
            dtype,
        })
    }

    // dw of the short conv: x and g [.., steps, C] -> [C, K] (summed over
    // batch and time).
    pub fn backward_w(
        x: &MetalTensor,
        g: &MetalTensor,
        kernel: usize,
    ) -> crate::err::Res<MetalTensor> {
        let dtype = g.dtype;
        let shape = g.layout.shape().to_vec();
        let r = shape.len();
        let (steps, c) = (shape[r - 2], shape[r - 1]);
        let batch: usize = shape[..r - 2].iter().product();
        let x = wrap_contig(x)?;
        let g = wrap_contig(g)?;
        let dev = MetalDevice::get();
        let pipe = pipeline(0xC043, dtype, "et_shortconv_bwd_w", || bwd_w_source(dtype))?;
        let out = dev.alloc(c * kernel, dtype);
        let elem = |off: usize| off * dtype.size_in_bytes();
        let (grid, tg) = grid_over(c * kernel);
        dev.with_encoder(|e| {
            e.setComputePipelineState(pipe.as_raw());
            set_buffer(e, 0, &x.buffer, elem(x.layout.offset()));
            set_buffer(e, 1, &g.buffer, elem(g.layout.offset()));
            set_buffer(e, 2, &out, 0);
            set_bytes(e, 3, &(batch as u32));
            set_bytes(e, 4, &(steps as u32));
            set_bytes(e, 5, &(c as u32));
            set_bytes(e, 6, &(kernel as u32));
            e.dispatchThreads_threadsPerThreadgroup(grid, tg);
        });
        Ok(MetalTensor {
            buffer: out,
            layout: crate::runtime::layout::Layout::contiguous(vec![c, kernel]),
            dtype,
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::runtime::metal::device::MetalDevice;
    use crate::runtime::metal::run::MetalTensor as MT;

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

    fn max_diff(x: &[f32], y: &[f32]) -> f32 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn kernels_match_composed() {
        let dev = MetalDevice::get();
        for (b, t, c, k) in [
            (2usize, 9usize, 8usize, 4usize),
            (1, 3, 5, 2),
            (2, 70, 64, 4),
        ] {
            let x = MT::from_f32(dev, prand(b * t * c, 31), vec![b, t, c]);
            let w = MT::from_f32(dev, prand(c * k, 32), vec![c, k]);
            let g = MT::from_f32(dev, prand(b * t * c, 33), vec![b, t, c]);

            let fwd = super::metal::forward(&x, &w, None, t, false).unwrap().0;
            let composed_fwd =
                crate::runtime::metal::composed::short_conv1d_forward(&x, &w).unwrap();
            let dx = super::metal::backward_x(&g, &w).unwrap();
            let composed_dx =
                crate::runtime::metal::composed::short_conv1d_backward_x(&x, &w, &g).unwrap();
            let dw = super::metal::backward_w(&x, &g, k).unwrap();
            let composed_dw =
                crate::runtime::metal::composed::short_conv1d_backward_w(&x, &w, &g).unwrap();
            dev.synchronize().unwrap();
            let df = max_diff(&fwd.read_f32().unwrap(), &composed_fwd.read_f32().unwrap());
            let ddx = max_diff(&dx.read_f32().unwrap(), &composed_dx.read_f32().unwrap());
            let ddw = max_diff(&dw.read_f32().unwrap(), &composed_dw.read_f32().unwrap());
            assert!(
                df < 1e-4 && ddx < 1e-4 && ddw < 1e-4,
                "({b},{t},{c},{k}): fwd {df} dx {ddx} dw {ddw}"
            );
        }
    }

    #[test]
    fn history_window_matches_stateless() {
        let dev = MetalDevice::get();
        let (t, c, k) = (11usize, 8usize, 4usize);
        let x = MT::from_f32(dev, prand(t * c, 41), vec![t, c]);
        let w = MT::from_f32(dev, prand(c * k, 42), vec![c, k]);
        // Split at 5: the second half with the first half's trailing
        // window must equal the full-sequence output tail.
        let full = super::metal::forward(&x, &w, None, t, false).unwrap().0;
        let first = MT {
            buffer: x.buffer.clone(),
            layout: x.layout.narrow(0, 0, 5),
            dtype: x.dtype,
        };
        let (_, window) = super::metal::forward(&first, &w, None, 5, true).unwrap();
        let second = MT {
            buffer: x.buffer.clone(),
            layout: x.layout.narrow(0, 5, t - 5),
            dtype: x.dtype,
        };
        let (tail, _) = super::metal::forward(&second, &w, window.as_ref(), t - 5, false).unwrap();
        let full_tail = crate::runtime::metal::ops::contiguous(&MT {
            buffer: full.buffer.clone(),
            layout: full.layout.narrow(0, 5, t - 5),
            dtype: full.dtype,
        })
        .unwrap();
        dev.synchronize().unwrap();
        let diff = max_diff(&full_tail.read_f32().unwrap(), &tail.read_f32().unwrap());
        assert!(diff < 1e-5, "stateful tail diff {diff}");
    }
}

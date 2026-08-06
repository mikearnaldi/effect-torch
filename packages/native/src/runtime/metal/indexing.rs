use super::device::{set_buffer, set_bytes, MetalDevice};
use super::run::MetalTensor;
use crate::runtime::dtype::DType;
use objc2_metal::MTLComputeCommandEncoder;

fn msl_type(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        DType::F64 => "double",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::U8 => "uchar",
        DType::U32 => "uint",
        DType::I64 => "long",
    }
}

fn key(parts: &[u64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for p in parts {
        p.hash(&mut h);
    }
    h.finish()
}

/// Host-side indices uploaded once — for op callers whose indices were
/// never on-device. Hot paths (gather/scatter under a compiled step)
/// keep indices on the device and never read them back.
pub fn ids_from_host(dev: &MetalDevice, ids: &[u32]) -> MetalTensor {
    MetalTensor {
        buffer: dev.alloc_with_data_u32(ids),
        layout: crate::runtime::layout::Layout::contiguous(vec![ids.len()]),
        dtype: DType::U32,
    }
}

pub fn index_select(dev: &MetalDevice, x: &MetalTensor, dim: usize, ids: &MetalTensor) -> Result<MetalTensor, String> {
    assert_eq!(ids.dtype, DType::U32, "index_select: ids must be u32");
    let shape = x.layout.shape();
    let rank = shape.len();
    let l = ids.numel();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = l;
    let total: usize = out_shape.iter().product();
    let out = MetalTensor::empty(dev, out_shape.clone(), x.dtype);
    if total == 0 {
        return Ok(out);
    }
    let ty = msl_type(x.dtype);
    let offset = x.layout.offset();
    let strides = x.layout.strides();
    let mut decompose = String::new();
    for d in (0..rank).rev() {
        let c = out_shape[d];
        let s = strides[d];
        let coord = if d == rank - 1 {
            format!("gid % {c}u")
        } else {
            let div: usize = out_shape[d + 1..].iter().product();
            format!("(gid / {div}u) % {c}u")
        };
        if d == dim {
            decompose.push_str(&format!("        base += ids[{coord}] * {s}u;\n"));
        } else {
            decompose.push_str(&format!("        base += ({coord}) * {s}u;\n"));
        }
    }
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_isel(
    device const {ty}* x [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device {ty}* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    uint base = {offset}u;
{decompose}    out[gid] = x[base];
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0x15E1, x.dtype as u64, dim as u64, l as u64, key(&out_shape.iter().map(|&v| v as u64).collect::<Vec<_>>()), key(&strides.iter().map(|&v| v as u64).collect::<Vec<_>>())]), "et_isel", make_src)?;
    let padded = total.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &ids.buffer, ids.layout.offset() * 4);
        set_buffer(e, 2, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn gather(dev: &MetalDevice, x: &MetalTensor, dim: usize, ids: &MetalTensor, ids_shape: &[usize]) -> Result<MetalTensor, String> {
    assert_eq!(ids.dtype, DType::U32, "gather: ids must be u32");
    let shape = x.layout.shape();
    let rank = shape.len();
    assert_eq!(ids_shape.len(), rank, "gather: rank mismatch");
    let total: usize = ids_shape.iter().product();
    let out = MetalTensor::empty(dev, ids_shape.to_vec(), x.dtype);
    if total == 0 {
        return Ok(out);
    }
    let ty = msl_type(x.dtype);
    let strides = x.layout.strides();
    let mut decompose = String::new();
    for d in (0..rank).rev() {
        let c = ids_shape[d];
        let div: usize = ids_shape[d + 1..].iter().product::<usize>().max(1);
        let coord = format!("((gid / {div}u) % {c}u)");
        if d == dim {
            decompose.push_str(&format!("        base += ids[gid] * {}u;\n", strides[d]));
        } else {
            let s = strides[d];
            if s == 1 {
                decompose.push_str(&format!("        base += {coord};\n"));
            } else {
                decompose.push_str(&format!("        base += {coord} * {s}u;\n"));
            }
        }
    }
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_gather(
    device const {ty}* x [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device {ty}* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    uint base = 0u;
{decompose}    out[gid] = x[base];
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0x6A7E, x.dtype as u64, dim as u64, key(&ids_shape.iter().map(|&v| v as u64).collect::<Vec<_>>()), key(&strides.iter().map(|&v| v as u64).collect::<Vec<_>>())]), "et_gather", make_src)?;
    let padded = total.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &ids.buffer, ids.layout.offset() * 4);
        set_buffer(e, 2, &out.buffer, 0);
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn scatter_add(dev: &MetalDevice, x: &MetalTensor, dim: usize, ids: &MetalTensor, src_t: &MetalTensor) -> Result<MetalTensor, String> {
    assert_eq!(ids.dtype, DType::U32, "scatter_add: ids must be u32");
    if x.dtype != DType::F32 {
        // No bf16 atomics in MSL: accumulate in f32 (the more precise
        // order anyway), then cast back to the caller's dtype.
        let x32 = super::kernels::cast(dev, x, DType::F32)?;
        let src32 = super::kernels::cast(dev, src_t, DType::F32)?;
        let out32 = scatter_add(dev, &x32, dim, ids, &src32)?;
        return super::kernels::cast(dev, &out32, x.dtype);
    }
    let out = super::kernels::strided_copy(dev, x)?;
    let shape = x.layout.shape();
    let rank = shape.len();
    let ids_shape = src_t.layout.shape().to_vec();
    let total: usize = ids_shape.iter().product();
    if total == 0 {
        return Ok(out);
    }
    let out_strides = crate::runtime::layout::Layout::contiguous(shape.to_vec());
    let os = out_strides.strides().to_vec();
    let src_strides = src_t.layout.strides().to_vec();
    let mut decompose = String::new();
    let mut src_decompose = String::new();
    for d in (0..rank).rev() {
        let c = ids_shape[d];
        let div: usize = ids_shape[d + 1..].iter().product::<usize>().max(1);
        let coord = format!("((gid / {div}u) % {c}u)");
        if d == dim {
            decompose.push_str(&format!("        base += ids[gid] * {}u;\n", os[d]));
        } else {
            decompose.push_str(&format!("        base += {coord} * {}u;\n", os[d]));
        }
        let ss = src_strides[d];
        if ss == 1 {
            src_decompose.push_str(&format!("        src_off += {coord};\n"));
        } else {
            src_decompose.push_str(&format!("        src_off += {coord} * {ss}u;\n"));
        }
    }
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_sadd(
    device float* out [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device const float* src [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    uint base = 0u;
{decompose}    uint src_off = 0u;
{src_decompose}    atomic_fetch_add_explicit((device atomic_float*)&out[base], src[src_off], memory_order_relaxed);
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0x5ADD, x.dtype as u64, dim as u64, key(&ids_shape.iter().map(|&v| v as u64).collect::<Vec<_>>()), key(&os.iter().map(|&v| v as u64).collect::<Vec<_>>()), key(&src_strides.iter().map(|&v| v as u64).collect::<Vec<_>>())]), "et_sadd", make_src)?;
    let padded = total.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &out.buffer, 0);
        set_buffer(e, 1, &ids.buffer, ids.layout.offset() * 4);
        set_buffer(e, 2, &src_t.buffer, src_t.layout.offset() * src_t.dtype.size_in_bytes());
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(out)
}

pub fn cat(dev: &MetalDevice, tensors: &[&MetalTensor], dim: usize) -> Result<MetalTensor, String> {
    assert!(!tensors.is_empty());
    let dtype = tensors[0].dtype;
    let rank = tensors[0].layout.shape().len();
    let mut out_shape = tensors[0].layout.shape().to_vec();
    for t in tensors {
        assert_eq!(t.dtype, dtype, "mixed dtypes");
        assert_eq!(t.layout.shape().len(), rank, "cat rank mismatch");
        for d in 0..rank {
            if d != dim {
                assert_eq!(t.layout.shape()[d], out_shape[d], "cat shape mismatch");
            }
        }
    }
    out_shape[dim] = tensors.iter().map(|t| t.layout.shape()[dim]).sum();
    let out = MetalTensor::empty(dev, out_shape.clone(), dtype);
    let ty = msl_type(dtype);
    let esz = dtype.size_in_bytes();
    let inner: usize = out_shape[dim + 1..].iter().product();
    let outer: usize = out_shape[..dim].iter().product();
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_cat(
    device const {ty}* src [[buffer(0)]],
    device {ty}* out [[buffer(1)]],
    constant uint& tdim [[buffer(2)]],
    constant uint& dim_off [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint inner = {inner}u;
    const uint seg_n = tdim * inner * {outer}u;
    if (gid >= seg_n) return;
    const uint o = gid / (tdim * inner);
    const uint r = gid % (tdim * inner);
    out[o * {outdim}u * inner + dim_off * inner + r] = src[gid];
}}
"#,
        outdim = out_shape[dim]
    );
    let pipeline = dev.compile_lazy(key(&[0xCA7, dtype as u64, dim as u64, inner as u64, outer as u64, out_shape[dim] as u64]), "et_cat", make_src)?;
    let mut dim_off = 0usize;
    for t in tensors {
        let tc = super::kernels::strided_copy(dev, t)?;
        let tdim = t.layout.shape()[dim];
        let n = outer * tdim * inner;
        if n > 0 {
            let (tdu, dou) = (tdim as u32, dim_off as u32);
            let padded = n.div_ceil(256) * 256;
            dev.with_encoder(|e| {
                e.setComputePipelineState(pipeline.as_raw());
                set_buffer(e, 0, &tc.buffer, tc.layout.offset() * esz);
                set_buffer(e, 1, &out.buffer, 0);
                set_bytes(e, 2, &tdu);
                set_bytes(e, 3, &dou);
                e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
            });
        }
        dim_off += tdim;
    }
    Ok(out)
}

pub fn scatter_set(dev: &MetalDevice, x: &MetalTensor, dim: usize, ids: &[u32], src_t: &MetalTensor) -> Result<(), String> {
    let shape = x.layout.shape();
    let rank = shape.len();
    let ids_shape = src_t.layout.shape().to_vec();
    let total: usize = ids_shape.iter().product();
    if total == 0 {
        return Ok(());
    }
    let ty = msl_type(x.dtype);
    let out_strides = crate::runtime::layout::Layout::contiguous(shape.to_vec());
    let os = out_strides.strides().to_vec();
    let src_strides = src_t.layout.strides().to_vec();
    let mut decompose = String::new();
    let mut src_decompose = String::new();
    for d in (0..rank).rev() {
        let c = ids_shape[d];
        let div: usize = ids_shape[d + 1..].iter().product::<usize>().max(1);
        let coord = format!("((gid / {div}u) % {c}u)");
        if d == dim {
            decompose.push_str(&format!("        base += ids[gid] * {}u;\n", os[d]));
        } else {
            decompose.push_str(&format!("        base += {coord} * {}u;\n", os[d]));
        }
        let ss = src_strides[d];
        if ss == 1 {
            src_decompose.push_str(&format!("        src_off += {coord};\n"));
        } else {
            src_decompose.push_str(&format!("        src_off += {coord} * {ss}u;\n"));
        }
    }
    let make_src = || format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_sset(
    device {ty}* out [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device const {ty}* src [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= {total}u) return;
    uint base = 0u;
{decompose}    uint src_off = 0u;
{src_decompose}    out[base] = src[src_off];
}}
"#
    );
    let pipeline = dev.compile_lazy(key(&[0x55E7, x.dtype as u64, dim as u64, key(&ids_shape.iter().map(|&v| v as u64).collect::<Vec<_>>()), key(&os.iter().map(|&v| v as u64).collect::<Vec<_>>()), key(&src_strides.iter().map(|&v| v as u64).collect::<Vec<_>>())]), "et_sset", make_src)?;
    let ids_buf = dev.alloc_with_data_u32(ids);
    let padded = total.div_ceil(256) * 256;
    dev.with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        set_buffer(e, 0, &x.buffer, x.layout.offset() * x.dtype.size_in_bytes());
        set_buffer(e, 1, &ids_buf, 0);
        set_buffer(e, 2, &src_t.buffer, src_t.layout.offset() * src_t.dtype.size_in_bytes());
        e.dispatchThreads_threadsPerThreadgroup(MetalDevice::grid(padded, 1, 1), MetalDevice::grid(256, 1, 1));
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_select_rows() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, (0..6).map(|v| v as f32).collect(), vec![2, 3]);
        let out = index_select(dev, &x, 1, &ids_from_host(dev, &[2, 0])).unwrap();
        dev.synchronize();
        assert_eq!(out.read_f32().unwrap(), vec![2., 0., 5., 3.]);
    }

    #[test]
    fn gather_rows() {
        let dev = MetalDevice::get();
        let x = MetalTensor::from_f32(dev, (0..6).map(|v| v as f32).collect(), vec![2, 3]);
        let out = gather(dev, &x, 0, &ids_from_host(dev, &[1, 1, 1, 0, 0, 0, 1, 1, 1]), &[3, 3]).unwrap();
        dev.synchronize();
        assert_eq!(out.read_f32().unwrap(), vec![3., 4., 5., 0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn scatter_add_embedding() {
        let dev = MetalDevice::get();
        let table = MetalTensor::zeros(dev, vec![4, 2], DType::F32);
        let src = MetalTensor::from_f32(dev, vec![1f32, 2., 3., 4.], vec![2, 2]);
        let out = scatter_add(dev, &table, 0, &ids_from_host(dev, &[1, 1, 3, 3]), &src).unwrap();
        dev.synchronize();
        assert_eq!(out.read_f32().unwrap(), vec![0., 0., 1., 2., 0., 0., 3., 4.]);
    }

    #[test]
    fn cat_dims() {
        let dev = MetalDevice::get();
        let a = MetalTensor::from_f32(dev, vec![1f32, 2.], vec![1, 2]);
        let b = MetalTensor::from_f32(dev, vec![3f32, 4.], vec![1, 2]);
        let c = cat(dev, &[&a, &b], 0).unwrap();
        dev.synchronize();
        assert_eq!(c.read_f32().unwrap(), vec![1., 2., 3., 4.]);
        let d = cat(dev, &[&a, &a], 1).unwrap();
        dev.synchronize();
        assert_eq!(d.read_f32().unwrap(), vec![1., 2., 1., 2.]);
    }
}

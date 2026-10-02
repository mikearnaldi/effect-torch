//! Typed paged attention with invocation-owned current K/V and explicit rounding.
use crate::{
    device::{set_buffer, set_bytes, MetalDevice},
    err,
    run::MetalTensor,
};
use effect_torch_graph::{AttentionRounding, KvAttentionMode};
use effect_torch_runtime::DType;
use objc2_metal::MTLComputeCommandEncoder;
use std::hash::{Hash, Hasher};

fn key(dtype: DType, storage: DType, width: usize, scatter: bool) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    (0x2757u32, dtype, storage, width, scatter).hash(&mut hash);
    hash.finish()
}
fn ty(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::U8 => "uchar",
        _ => unreachable!(),
    }
}
fn source(dtype: DType, storage: DType, width: usize) -> String {
    r#"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#define T $TYPE
#define S $STORAGE
#define D $WIDTH
#define QUANTIZED $QUANTIZED
struct Params {
    uint batch, heads, kv_heads, time, block_size, max_blocks;
    uint window, bidirectional, stepwise, read_only;
    float scale;
    uint qs[4], ks[4], vs[4];
};
float rounded(float x, constant Params& p) { return p.stepwise ? float(T(x)) : x; }
uint offset(uint b, uint h, uint t, uint d, constant uint* s) {
    return b*s[0] + h*s[1] + t*s[2] + d*s[3];
}
float cached(device const S* slab, device const float* scales, device const T* current,
             device const uint* table, uint position, uint cursor, uint base, uint b,
             uint h, uint d, constant uint* strides, constant Params& p) {
    if (position >= cursor) return float(current[offset(b,h,position-cursor,d,strides)]);
    uint row = table[position / p.block_size - base] * p.block_size + position % p.block_size;
    float value = float(slab[(row*p.kv_heads+h)*D+d]);
#if QUANTIZED
    value = (value-128.0f)*scales[row*p.kv_heads+h];
#endif
    return value;
}
float score(device const T* q, device const T* k, device const S* slab, device const float* scales,
            device const uint* table, uint position, uint cursor, uint base, uint b,
            uint h, uint kh, uint query, constant Params& p) {
    float dot = 0.0f;
    for (uint d=0; d<D; d++) dot += float(q[offset(b,h,query,d,p.qs)]) * cached(slab,scales,k,table,position,cursor,base,b,kh,d,p.ks,p);
    return rounded(rounded(dot,p) * p.scale,p);
}
kernel void et_state_attention(
    device const T* q [[buffer(0)]], device const T* k [[buffer(1)]], device const T* v [[buffer(2)]],
    device const S* keys [[buffer(3)]], device const S* values [[buffer(4)]],
    device const float* key_scales [[buffer(5)]], device const float* value_scales [[buffer(6)]],
    device const uint* tables [[buffer(7)]], device const uint* lengths [[buffer(8)]],
    device const uint* bases [[buffer(9)]], device const uint* advances [[buffer(10)]],
    device const uint* starts [[buffer(11)]], device T* out [[buffer(12)]],
    constant Params& p [[buffer(13)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.batch*p.heads*p.time) return;
    uint query=gid%p.time, h=(gid/p.time)%p.heads, b=gid/(p.time*p.heads);
    uint advance=advances[b];
    if (query>=advance) { for (uint d=0;d<D;d++) out[gid*D+d]=T(0); return; }
    uint cursor=lengths[b]-advance, end=cursor+(p.bidirectional ? advance : query+1);
    uint begin=starts[b];
    if (p.window) {
        uint bound=p.bidirectional ? cursor : end;
        begin=max(begin,bound>p.window ? bound-p.window : 0u);
    }
    uint kh=h/(p.heads/p.kv_heads), base=bases[b];
    device const uint* table=tables+b*p.max_blocks;
    float maximum=-INFINITY;
    for(uint pos=begin;pos<end;pos++) maximum=max(maximum,score(q,k,keys,key_scales,table,pos,cursor,base,b,h,kh,query,p));
    float denominator=0.0f;
    for(uint pos=begin;pos<end;pos++) denominator+=exp(score(q,k,keys,key_scales,table,pos,cursor,base,b,h,kh,query,p)-maximum);
    float result[D];
    for(uint d=0;d<D;d++) result[d]=0.0f;
    for(uint pos=begin;pos<end;pos++) {
        float probability=rounded(exp(score(q,k,keys,key_scales,table,pos,cursor,base,b,h,kh,query,p)-maximum)/denominator,p);
        for(uint d=0;d<D;d++) result[d]+=probability*cached(values,value_scales,v,table,pos,cursor,base,b,kh,d,p.vs,p);
    }
    for(uint d=0;d<D;d++) out[gid*D+d]=T(result[d]);
}
kernel void et_state_scatter(
    device const T* k [[buffer(0)]], device const T* v [[buffer(1)]],
    device S* keys [[buffer(2)]], device S* values [[buffer(3)]],
    device float* key_scales [[buffer(4)]], device float* value_scales [[buffer(5)]],
    device const uint* tables [[buffer(6)]], device const uint* lengths [[buffer(7)]],
    device const uint* bases [[buffer(8)]], device const uint* advances [[buffer(9)]],
    constant Params& p [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
    if(gid>=p.batch*p.kv_heads*p.time) return;
    uint t=gid%p.time,h=(gid/p.time)%p.kv_heads,b=gid/(p.time*p.kv_heads),advance=advances[b];
    if(t>=advance) return;
    uint position=lengths[b]-advance+t;
    uint row=tables[b*p.max_blocks+position/p.block_size-bases[b]]*p.block_size+position%p.block_size;
    float ks=1.0f,vs=1.0f;
#if QUANTIZED
    ks=0.0f; vs=0.0f;
    for(uint d=0;d<D;d++) { ks=max(ks,abs(float(k[offset(b,h,t,d,p.ks)]))); vs=max(vs,abs(float(v[offset(b,h,t,d,p.vs)]))); }
    ks=ks/127.0f+1e-12f; vs=vs/127.0f+1e-12f;
    key_scales[row*p.kv_heads+h]=ks; value_scales[row*p.kv_heads+h]=vs;
#endif
    for(uint d=0;d<D;d++) {
        float x=float(k[offset(b,h,t,d,p.ks)]),y=float(v[offset(b,h,t,d,p.vs)]);
#if QUANTIZED
        x=clamp(round(x/ks),-127.0f,127.0f)+128.0f; y=clamp(round(y/vs),-127.0f,127.0f)+128.0f;
#endif
        keys[(row*p.kv_heads+h)*D+d]=S(x); values[(row*p.kv_heads+h)*D+d]=S(y);
    }
}
"#.replace("$TYPE",ty(dtype)).replace("$STORAGE",ty(storage)).replace("$WIDTH",&width.to_string()).replace("$QUANTIZED",if storage==DType::U8 {"1"} else {"0"})
}

pub(crate) fn warm(dtype: DType, storage: DType, width: usize) -> err::Res<()> {
    for (scatter, name) in [(false, "et_state_attention"), (true, "et_state_scatter")] {
        MetalDevice::get().compile_lazy(key(dtype, storage, width, scatter), name, || {
            source(dtype, storage, width)
        })?;
    }
    Ok(())
}

#[repr(C)]
struct Params {
    batch: u32,
    heads: u32,
    kv_heads: u32,
    time: u32,
    block_size: u32,
    max_blocks: u32,
    window: u32,
    bidirectional: u32,
    stepwise: u32,
    read_only: u32,
    scale: f32,
    qs: [u32; 4],
    ks: [u32; 4],
    vs: [u32; 4],
}
fn strides(tensor: &MetalTensor) -> err::Res<[u32; 4]> {
    let rank = tensor.layout.shape().len();
    if rank != 4 {
        return Err("typed paged attention requires rank-4 tensors".to_string());
    }
    tensor
        .layout
        .strides()
        .iter()
        .map(|&stride| {
            u32::try_from(stride).map_err(|_| "paged attention stride exceeds u32".to_string())
        })
        .collect::<err::Res<Vec<_>>>()?
        .try_into()
        .map_err(|_| "invalid attention strides".to_string())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_into(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    keys: &MetalTensor,
    values: &MetalTensor,
    scales: Option<(&MetalTensor, &MetalTensor)>,
    staging: &[MetalTensor],
    output: &MetalTensor,
    block_size: usize,
    scale: f64,
    window: Option<usize>,
    mode: KvAttentionMode,
    rounding: AttentionRounding,
    read_only: bool,
    state_only: bool,
) -> err::Res<()> {
    if q.dtype != k.dtype || q.dtype != v.dtype {
        return Err("paged attention input dtypes differ".to_string());
    }
    output.validate_destination("typed paged attention", q.layout.shape(), q.dtype)?;
    let shape = q.layout.shape();
    let p = Params {
        batch: shape[0] as u32,
        heads: shape[1] as u32,
        kv_heads: k.layout.shape()[1] as u32,
        time: shape[2] as u32,
        block_size: block_size as u32,
        max_blocks: staging[0].layout.shape()[1] as u32,
        window: window.unwrap_or(0) as u32,
        bidirectional: u32::from(mode == KvAttentionMode::BidirectionalBlock),
        stepwise: u32::from(rounding == AttentionRounding::Stepwise),
        read_only: u32::from(read_only),
        scale: scale as f32,
        qs: strides(q)?,
        ks: strides(k)?,
        vs: strides(v)?,
    };
    let bind = |encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
                index,
                tensor: &MetalTensor| {
        set_buffer(
            encoder,
            index,
            &tensor.buffer,
            tensor.layout.offset() * tensor.dtype.size_in_bytes(),
        );
    };
    let (ks, vs) = scales.unwrap_or((keys, values));
    if !read_only {
        let pipeline = MetalDevice::get()
            .pipeline_cached(key(q.dtype, keys.dtype, shape[3], true))
            .ok_or("typed paged scatter pipeline is not warm")?;
        MetalDevice::get().with_encoder(|e| {
            e.setComputePipelineState(pipeline.as_raw());
            for (index, tensor) in [
                k,
                v,
                keys,
                values,
                ks,
                vs,
                &staging[0],
                &staging[1],
                &staging[2],
                &staging[3],
            ]
            .into_iter()
            .enumerate()
            {
                bind(e, index, tensor);
            }
            set_bytes(e, 10, &p);
            let (grid, group) = MetalDevice::grid_flat((p.batch * p.kv_heads * p.time) as usize);
            e.dispatchThreads_threadsPerThreadgroup(grid, group);
        });
    }
    if state_only {
        return Ok(());
    }
    let pipeline = MetalDevice::get()
        .pipeline_cached(key(q.dtype, keys.dtype, shape[3], false))
        .ok_or("typed paged attention pipeline is not warm")?;
    MetalDevice::get().with_encoder(|e| {
        e.setComputePipelineState(pipeline.as_raw());
        for (index, tensor) in [
            q,
            k,
            v,
            keys,
            values,
            ks,
            vs,
            &staging[0],
            &staging[1],
            &staging[2],
            &staging[3],
            &staging[4],
            output,
        ]
        .into_iter()
        .enumerate()
        {
            bind(e, index, tensor);
        }
        set_bytes(e, 13, &p);
        let (grid, group) = MetalDevice::grid_flat((p.batch * p.heads * p.time) as usize);
        e.dispatchThreads_threadsPerThreadgroup(grid, group);
    });
    Ok(())
}

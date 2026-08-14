//! GPU-resident next-token sampling for one logits row.
//!
//! Greedy and unfiltered categorical sampling use one 32-thread threadgroup.
//! Filtered sampling first computes 32 sorted shard-local top-k lists, then a
//! second 32-thread dispatch merges them, applies top-p, and draws the token.
//! The shared u32 result pair is `[status, token]`: status zero is success and
//! status one reports a non-finite logit whose index is in `token`.

use crate::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
use crate::run::MetalTensor;
use crate::runtime::dtype::DType;
use effect_torch_runtime::{SamplingOptions, MAX_SAMPLING_VOCABULARY};
use objc2_metal::{MTLComputeCommandEncoder, MTLResource, MTLStorageMode};

const MAX_TOP_K: usize = 64;
const FILTER_GROUPS: usize = 32;
const FILTER_THREADS: usize = 32;
const PARTIAL_CANDIDATES: usize = FILTER_GROUPS * MAX_TOP_K;
const PARTIAL_VALUES_BYTES: usize = PARTIAL_CANDIDATES * std::mem::size_of::<f32>();
const PARTIAL_TOKENS_BYTES: usize = PARTIAL_CANDIDATES * std::mem::size_of::<u32>();
const PARTIAL_INVALIDS_BYTES: usize = FILTER_GROUPS * std::mem::size_of::<u32>();
const FILTER_SCRATCH_BYTES: usize =
    PARTIAL_VALUES_BYTES + PARTIAL_TOKENS_BYTES + PARTIAL_INVALIDS_BYTES;
pub const STATUS_OK: u32 = 0;
pub const STATUS_NONFINITE: u32 = 1;
const PIPELINE_REVISION: u64 = 6;
const KERNEL_NAME: &str = "et_sample_logits";
const PARTIALS_KERNEL_NAME: &str = "et_sample_filtered_partials";
const MERGE_KERNEL_NAME: &str = "et_sample_filtered_merge";

#[derive(Clone, Copy)]
enum Mode {
    Greedy = 0,
    Categorical = 1,
    Filtered = 2,
}

#[derive(Clone, Copy)]
struct Dispatch {
    mode: Mode,
    top_k: u32,
    temperature: f32,
    top_p: f32,
}

fn validate(logits: &MetalTensor, options: SamplingOptions) -> crate::err::Res<Dispatch> {
    if logits.layout.rank() != 1 {
        return Err(format!(
            "sample: logits must be rank 1, got rank {}",
            logits.layout.rank()
        ));
    }
    if !matches!(logits.dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(format!(
            "sample: logits dtype must be f16, bf16, or f32, got {}",
            logits.dtype.name()
        ));
    }
    let length = logits.numel();
    if length == 0 {
        return Err("sample: logits must be non-empty".to_string());
    }
    if length > MAX_SAMPLING_VOCABULARY {
        return Err(format!(
            "sample: vocabulary {length} exceeds limit {MAX_SAMPLING_VOCABULARY}"
        ));
    }
    let required_bytes = logits
        .layout
        .checked_byte_size(logits.dtype)
        .ok_or_else(|| "sample: logits layout overflows".to_string())?;
    if required_bytes > logits.buffer.size {
        return Err(format!(
            "sample: logits layout requires {required_bytes} bytes, buffer has {}",
            logits.buffer.size
        ));
    }
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(format!(
            "sample: temperature must be finite and non-negative, got {}",
            options.temperature
        ));
    }
    if !options.top_p.is_finite() || options.top_p <= 0.0 || options.top_p > 1.0 {
        return Err(format!(
            "sample: topP must be finite and in (0, 1], got {}",
            options.top_p
        ));
    }
    if let Some(top_k) = options.top_k {
        if top_k == 0 || top_k > length {
            return Err(format!(
                "sample: topK must be in [1, {length}], got {top_k}"
            ));
        }
    }

    if options.temperature == 0.0 {
        return Ok(Dispatch {
            mode: Mode::Greedy,
            top_k: 0,
            temperature: 0.0,
            top_p: options.top_p as f32,
        });
    }

    let mode =
        match options.top_k {
            None if options.top_p == 1.0 => Mode::Categorical,
            None => return Err(
                "sample: topP < 1 requires topK in [1, 64] for positive-temperature Metal sampling"
                    .to_string(),
            ),
            Some(top_k) if top_k > MAX_TOP_K => {
                return Err(format!(
                    "sample: topK {top_k} exceeds Metal positive-temperature limit {MAX_TOP_K}"
                ))
            }
            Some(top_k) if top_k == length && options.top_p == 1.0 => Mode::Categorical,
            Some(_) => Mode::Filtered,
        };
    let temperature = (options.temperature as f32).clamp(f32::MIN_POSITIVE, f32::MAX);
    Ok(Dispatch {
        mode,
        top_k: options.top_k.unwrap_or(0) as u32,
        temperature,
        top_p: options.top_p as f32,
    })
}

#[derive(Clone, Copy, Hash)]
enum Kernel {
    Base,
    Partials,
    Merge,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Self::Base => KERNEL_NAME,
            Self::Partials => PARTIALS_KERNEL_NAME,
            Self::Merge => MERGE_KERNEL_NAME,
        }
    }
}

fn pipeline_key(dtype: DType, kernel: Kernel) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    ("sampling", PIPELINE_REVISION, dtype, kernel).hash(&mut hasher);
    hasher.finish()
}

fn source(dtype: DType) -> String {
    let ty = match dtype {
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::F32 => "float",
        _ => unreachable!("sampling dtype was validated"),
    };
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

constant uint ET_THREADS = 32u;
constant uint ET_GROUPS = 32u;
constant uint ET_MAX_K = 64u;
constant uint ET_INVALID = 0xffffffffu;

inline bool et_better(float left_value, uint left_token,
                      float right_value, uint right_token) {{
    return left_value > right_value ||
        (left_value == right_value && left_token < right_token);
}}

inline bool et_worse(float left_value, uint left_token,
                     float right_value, uint right_token) {{
    return left_value < right_value ||
        (left_value == right_value && left_token > right_token);
}}

inline void et_sift_heap(threadgroup float* values, threadgroup uint* tokens,
                         uint base, uint count, uint position) {{
    while (true) {{
        const uint left = position * 2u + 1u;
        if (left >= count) break;
        const uint right = left + 1u;
        uint worse = left;
        if (right < count &&
            et_worse(values[base + right], tokens[base + right],
                     values[base + left], tokens[base + left])) {{
            worse = right;
        }}
        if (!et_worse(values[base + worse], tokens[base + worse],
                      values[base + position], tokens[base + position])) break;
        const float swap_value = values[base + position];
        const uint swap_token = tokens[base + position];
        values[base + position] = values[base + worse];
        tokens[base + position] = tokens[base + worse];
        values[base + worse] = swap_value;
        tokens[base + worse] = swap_token;
        position = worse;
    }}
}}

inline void et_offer_heap(threadgroup float* values, threadgroup uint* tokens,
                          uint base, uint count, float value, uint token) {{
    if (!et_better(value, token, values[base], tokens[base])) return;
    values[base] = value;
    tokens[base] = token;
    et_sift_heap(values, tokens, base, count, 0u);
}}

inline void et_sort_heap_desc(threadgroup float* values, threadgroup uint* tokens,
                              uint base, uint count) {{
    for (uint end = count; end > 1u; --end) {{
        const uint last = end - 1u;
        const float swap_value = values[base];
        const uint swap_token = tokens[base];
        values[base] = values[base + last];
        tokens[base] = tokens[base + last];
        values[base + last] = swap_value;
        tokens[base + last] = swap_token;
        et_sift_heap(values, tokens, base, last, 0u);
    }}
}}

inline float et_random_unit(ulong seed, ulong counter) {{
    ulong value = seed ^ (counter * 0x9e3779b97f4a7c15ul);
    value += 0x9e3779b97f4a7c15ul;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    value ^= value >> 31;
    // Metal has no f64, so round the baseline's 53-bit unit value to f32 and
    // clamp the one possible upward rounding at the open upper endpoint.
    return min(float(value >> 11) * 0x1.0p-53f, 0x1.fffffep-1f);
}}

inline float et_scaled_delta(float value, float maximum, float temperature) {{
    const float delta = value - maximum;
    // Opposite finite f32 extremes can overflow their subtraction even though
    // the temperature-scaled difference is representable.
    return isinf(delta) ? value / temperature - maximum / temperature
                        : delta / temperature;
}}

inline float et_draw(ulong seed, ulong counter, float total) {{
    // f32 multiplication can round a unit draw up to the closed endpoint.
    return min(et_random_unit(seed, counter) * total, nextafter(total, 0.0f));
}}

kernel void et_sample_logits(
    device const {ty}* logits [[buffer(0)]],
    device uint* result [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant ulong& stride [[buffer(3)]],
    constant uint& mode [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant float& top_p [[buffer(7)]],
    constant ulong& seed [[buffer(8)]],
    constant ulong& counter [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]]) {{
    threadgroup uint invalids[ET_THREADS];
    threadgroup float partial_values[ET_THREADS];
    threadgroup uint partial_tokens[ET_THREADS];
    uint invalid = ET_INVALID;
    if (mode == 0u) {{
        float best_value = -INFINITY;
        uint best_token = ET_INVALID;
        for (uint token = tid; token < length; token += ET_THREADS) {{
            const float value = float(logits[ulong(token) * stride]);
            if (!isfinite(value)) {{
                invalid = min(invalid, token);
            }} else if (et_better(value, token, best_value, best_token)) {{
                best_value = value;
                best_token = token;
            }}
        }}
        invalids[tid] = invalid;
        partial_values[tid] = best_value;
        partial_tokens[tid] = best_token;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {{
            uint first_invalid = ET_INVALID;
            float best_value = -INFINITY;
            uint best_token = ET_INVALID;
            for (uint lane = 0; lane < ET_THREADS; ++lane) {{
                first_invalid = min(first_invalid, invalids[lane]);
                if (et_better(partial_values[lane], partial_tokens[lane],
                              best_value, best_token)) {{
                    best_value = partial_values[lane];
                    best_token = partial_tokens[lane];
                }}
            }}
            result[0] = first_invalid == ET_INVALID ? 0u : 1u;
            result[1] = first_invalid == ET_INVALID ? best_token : first_invalid;
        }}
        return;
    }}

    if (mode == 1u) {{
        const uint chunk_start = uint((ulong(length) * ulong(tid)) / ulong(ET_THREADS));
        const uint chunk_end = uint((ulong(length) * ulong(tid + 1u)) / ulong(ET_THREADS));
        float local_max = -INFINITY;
        for (uint token = chunk_start; token < chunk_end; ++token) {{
            const float value = float(logits[ulong(token) * stride]);
            if (!isfinite(value)) invalid = min(invalid, token);
            else local_max = max(local_max, value);
        }}
        invalids[tid] = invalid;
        partial_values[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {{
            uint first_invalid = ET_INVALID;
            float row_max = -INFINITY;
            for (uint lane = 0; lane < ET_THREADS; ++lane) {{
                first_invalid = min(first_invalid, invalids[lane]);
                row_max = max(row_max, partial_values[lane]);
            }}
            invalids[0] = first_invalid;
            partial_values[0] = row_max;
            if (first_invalid != ET_INVALID) {{
                result[0] = 1u;
                result[1] = first_invalid;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (invalids[0] != ET_INVALID) return;

        const float row_max = partial_values[0];
        float local_total = 0.0f;
        for (uint token = chunk_start; token < chunk_end; ++token) {{
            const float value = float(logits[ulong(token) * stride]);
            local_total += exp(et_scaled_delta(value, row_max, temperature));
        }}
        partial_values[tid] = local_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {{
            float total = 0.0f;
            for (uint lane = 0; lane < ET_THREADS; ++lane) total += partial_values[lane];
            float draw = et_draw(seed, counter, total);
            uint selected_lane = ET_INVALID;
            for (uint lane = 0; lane < ET_THREADS; ++lane) {{
                draw -= partial_values[lane];
                if (draw < 0.0f) {{
                    draw += partial_values[lane];
                    selected_lane = lane;
                    break;
                }}
            }}
            partial_tokens[0] = selected_lane;
            partial_values[0] = draw;
            if (selected_lane == ET_INVALID) {{
                result[0] = 0u;
                result[1] = length - 1u;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == partial_tokens[0]) {{
            float draw = partial_values[0];
            uint selected = chunk_end - 1u;
            for (uint token = chunk_start; token < chunk_end; ++token) {{
                const float value = float(logits[ulong(token) * stride]);
                draw -= exp(et_scaled_delta(value, row_max, temperature));
                if (draw < 0.0f) {{
                    selected = token;
                    break;
                }}
            }}
            result[0] = 0u;
            result[1] = selected;
        }}
        return;
    }}
}}

kernel void et_sample_filtered_partials(
    device const {ty}* logits [[buffer(0)]],
    device float* partial_values [[buffer(1)]],
    device uint* partial_tokens [[buffer(2)]],
    device uint* partial_invalids [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant ulong& stride [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]]) {{
    threadgroup uint invalids[ET_THREADS];
    threadgroup float local_values[ET_THREADS * ET_MAX_K];
    threadgroup uint local_tokens[ET_THREADS * ET_MAX_K];

    const uint group = group_position.x;
    const uint base = tid * ET_MAX_K;
    for (uint i = 0; i < top_k; ++i) {{
        local_values[base + i] = -INFINITY;
        local_tokens[base + i] = ET_INVALID;
    }}
    uint invalid = ET_INVALID;
    for (uint token = group * ET_THREADS + tid; token < length;
         token += ET_GROUPS * ET_THREADS) {{
        const float value = float(logits[ulong(token) * stride]);
        if (!isfinite(value)) invalid = min(invalid, token);
        else et_offer_heap(local_values, local_tokens, base, top_k, value, token);
    }}
    et_sort_heap_desc(local_values, local_tokens, base, top_k);
    invalids[tid] = invalid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {{
        uint first_invalid = ET_INVALID;
        for (uint lane = 0; lane < ET_THREADS; ++lane)
            first_invalid = min(first_invalid, invalids[lane]);
        partial_invalids[group] = first_invalid;

        uint positions[ET_THREADS];
        for (uint lane = 0; lane < ET_THREADS; ++lane) positions[lane] = 0u;
        const uint output_base = group * ET_MAX_K;
        for (uint output = 0; output < top_k; ++output) {{
            float best_value = -INFINITY;
            uint best_token = ET_INVALID;
            uint best_lane = ET_INVALID;
            for (uint lane = 0; lane < ET_THREADS; ++lane) {{
                const uint position = positions[lane];
                const uint candidate = lane * ET_MAX_K + position;
                const float value = local_values[candidate];
                const uint token = local_tokens[candidate];
                if (best_lane == ET_INVALID ||
                    et_better(value, token, best_value, best_token)) {{
                    best_value = value;
                    best_token = token;
                    best_lane = lane;
                }}
            }}
            partial_values[output_base + output] = best_value;
            partial_tokens[output_base + output] = best_token;
            positions[best_lane] += 1u;
        }}
    }}
}}

kernel void et_sample_filtered_merge(
    device const float* partial_values [[buffer(0)]],
    device const uint* partial_tokens [[buffer(1)]],
    device const uint* partial_invalids [[buffer(2)]],
    device uint* result [[buffer(3)]],
    constant float& temperature [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant float& top_p [[buffer(6)]],
    constant ulong& seed [[buffer(7)]],
    constant ulong& counter [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]) {{
    threadgroup float merged_values[ET_MAX_K];
    threadgroup uint merged_tokens[ET_MAX_K];

    uint first_invalid = partial_invalids[tid];
    for (uint offset = ET_THREADS / 2u; offset > 0u; offset /= 2u)
        first_invalid = min(first_invalid, simd_shuffle_down(first_invalid, offset));
    first_invalid = simd_broadcast(first_invalid, 0u);
    if (first_invalid != ET_INVALID) {{
        if (tid == 0u) {{
            result[0] = 1u;
            result[1] = first_invalid;
        }}
        return;
    }}

    uint position = 0u;
    for (uint output = 0; output < top_k; ++output) {{
        const uint candidate = tid * ET_MAX_K + position;
        float best_value = partial_values[candidate];
        uint best_token = partial_tokens[candidate];
        uint best_group = tid;
        for (uint offset = ET_THREADS / 2u; offset > 0u; offset /= 2u) {{
            const float other_value = simd_shuffle_down(best_value, offset);
            const uint other_token = simd_shuffle_down(best_token, offset);
            const uint other_group = simd_shuffle_down(best_group, offset);
            if (tid + offset < ET_GROUPS &&
                et_better(other_value, other_token, best_value, best_token)) {{
                best_value = other_value;
                best_token = other_token;
                best_group = other_group;
            }}
        }}
        best_value = simd_broadcast(best_value, 0u);
        best_token = simd_broadcast(best_token, 0u);
        best_group = simd_broadcast(best_group, 0u);
        if (tid == 0u) {{
            merged_values[output] = best_value;
            merged_tokens[output] = best_token;
        }}
        if (tid == best_group) position += 1u;
    }}

    if (tid == 0u) {{
        const float row_max = merged_values[0];
        float total = 0.0f;
        for (uint i = 0; i < top_k; ++i) {{
            merged_values[i] = exp(et_scaled_delta(merged_values[i], row_max, temperature));
            total += merged_values[i];
        }}
        uint retained = top_k;
        if (top_p < 1.0f) {{
            const float threshold = top_p * total;
            float cumulative = 0.0f;
            for (uint i = 0; i < top_k; ++i) {{
                cumulative += merged_values[i];
                if (cumulative >= threshold) {{
                    retained = i + 1u;
                    total = cumulative;
                    break;
                }}
            }}
        }}

        float draw = et_draw(seed, counter, total);
        uint selected = merged_tokens[retained - 1u];
        for (uint i = 0; i < retained; ++i) {{
            draw -= merged_values[i];
            if (draw < 0.0f) {{
                selected = merged_tokens[i];
                break;
            }}
        }}
        result[0] = 0u;
        result[1] = selected;
    }}
}}
"#,
    )
}

fn compile_pipeline(dtype: DType, kernel: Kernel) -> crate::err::Res<Pipeline> {
    MetalDevice::get().compile_lazy(pipeline_key(dtype, kernel), kernel.name(), || source(dtype))
}

fn cached_pipeline(dtype: DType, kernel: Kernel) -> crate::err::Res<Pipeline> {
    MetalDevice::get()
        .pipeline_cached(pipeline_key(dtype, kernel))
        .ok_or_else(|| "sample: pipeline is not warm; call sampling::warm_exact first".to_string())
}

/// Precompiles the exact logits dtype needed by [`sample_into`]. Validation is
/// identical to dispatch validation, including the supported filter modes.
pub fn warm_exact(logits: &MetalTensor, options: SamplingOptions) -> crate::err::Res<()> {
    let dispatch = validate(logits, options)?;
    match dispatch.mode {
        Mode::Greedy | Mode::Categorical => {
            compile_pipeline(logits.dtype, Kernel::Base)?;
        }
        Mode::Filtered => {
            compile_pipeline(logits.dtype, Kernel::Partials)?;
            compile_pipeline(logits.dtype, Kernel::Merge)?;
        }
    }
    Ok(())
}

/// Backing allocation size for a logical shared u32 result tensor. The fixed
/// tail is reused by each filtered sample encoded against the tensor.
pub fn required_result_allocation_bytes(logical_u32_elements: usize) -> crate::err::Res<usize> {
    logical_u32_elements
        .checked_mul(DType::U32.size_in_bytes())
        .and_then(|bytes| bytes.checked_add(FILTER_SCRATCH_BYTES))
        .ok_or_else(|| "sample: result allocation byte size overflows".to_string())
}

fn validate_result(result: &MetalTensor, result_index: usize) -> crate::err::Res<()> {
    if result.dtype != DType::U32 || result.layout.rank() != 1 || !result.layout.is_contiguous() {
        return Err(format!(
            "sample: result must be a contiguous rank-1 u32 tensor, got {:?}:{}",
            result.layout,
            result.dtype.name()
        ));
    }
    let end = result_index
        .checked_add(2)
        .ok_or_else(|| "sample: result index overflows".to_string())?;
    if end > result.numel() {
        return Err(format!(
            "sample: result index {result_index} requires two u32 elements, result has {}",
            result.numel()
        ));
    }
    let allocation_elements = result
        .layout
        .offset()
        .checked_add(result.numel())
        .ok_or_else(|| "sample: result layout overflows".to_string())?;
    let required_bytes = required_result_allocation_bytes(allocation_elements)?;
    if required_bytes > result.buffer.size {
        return Err(format!(
            "sample: result including sampling scratch requires {required_bytes} bytes, buffer has {}",
            result.buffer.size
        ));
    }
    if result.buffer.as_raw().storageMode() != MTLStorageMode::Shared {
        return Err("sample: result must use shared Metal storage".to_string());
    }
    Ok(())
}

/// Encodes one allocation-free sample into `result[result_index..][..2]`.
///
/// `result` must use shared storage so the status pair can be read after
/// synchronization. The exact logits pipeline must already have been warmed by
/// [`warm_exact`]; a cache miss is an error rather than an implicit compile.
pub fn sample_into(
    logits: &MetalTensor,
    result: &MetalTensor,
    result_index: usize,
    options: SamplingOptions,
) -> crate::err::Res<()> {
    let dispatch = validate(logits, options)?;
    validate_result(result, result_index)?;
    let base_pipeline = match dispatch.mode {
        Mode::Greedy | Mode::Categorical => Some(cached_pipeline(logits.dtype, Kernel::Base)?),
        Mode::Filtered => None,
    };
    let filtered_pipelines = match dispatch.mode {
        Mode::Greedy | Mode::Categorical => None,
        Mode::Filtered => Some((
            cached_pipeline(logits.dtype, Kernel::Partials)?,
            cached_pipeline(logits.dtype, Kernel::Merge)?,
        )),
    };
    let device = MetalDevice::get();
    device.synchronize_buffer_producer(&logits.buffer)?;
    device.mark_buffer_write(&result.buffer)?;

    let length = logits.numel() as u32;
    let stride = logits.layout.strides()[0] as u64;
    let mode = dispatch.mode as u32;
    let seed = options.seed;
    let counter = options.counter;
    let logits_offset = logits.layout.offset() * logits.dtype.size_in_bytes();
    let result_offset = (result.layout.offset() + result_index) * DType::U32.size_in_bytes();
    let scratch_offset = (result.layout.offset() + result.numel()) * DType::U32.size_in_bytes();
    let partial_tokens_offset = scratch_offset + PARTIAL_VALUES_BYTES;
    let partial_invalids_offset = partial_tokens_offset + PARTIAL_TOKENS_BYTES;
    device.with_encoder(|encoder| {
        if let Some(pipeline) = base_pipeline.as_ref() {
            encoder.setComputePipelineState(pipeline.as_raw());
            set_buffer(encoder, 0, &logits.buffer, logits_offset);
            set_buffer(encoder, 1, &result.buffer, result_offset);
            set_bytes(encoder, 2, &length);
            set_bytes(encoder, 3, &stride);
            set_bytes(encoder, 4, &mode);
            set_bytes(encoder, 5, &dispatch.temperature);
            set_bytes(encoder, 6, &dispatch.top_k);
            set_bytes(encoder, 7, &dispatch.top_p);
            set_bytes(encoder, 8, &seed);
            set_bytes(encoder, 9, &counter);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MetalDevice::grid(1, 1, 1),
                MetalDevice::grid(FILTER_THREADS, 1, 1),
            );
            return;
        }

        let (partials, merge) = filtered_pipelines
            .as_ref()
            .expect("filtered sampling pipelines were resolved");
        encoder.setComputePipelineState(partials.as_raw());
        set_buffer(encoder, 0, &logits.buffer, logits_offset);
        set_buffer(encoder, 1, &result.buffer, scratch_offset);
        set_buffer(encoder, 2, &result.buffer, partial_tokens_offset);
        set_buffer(encoder, 3, &result.buffer, partial_invalids_offset);
        set_bytes(encoder, 4, &length);
        set_bytes(encoder, 5, &stride);
        set_bytes(encoder, 6, &dispatch.top_k);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MetalDevice::grid(FILTER_GROUPS, 1, 1),
            MetalDevice::grid(FILTER_THREADS, 1, 1),
        );
        encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

        encoder.setComputePipelineState(merge.as_raw());
        set_buffer(encoder, 0, &result.buffer, scratch_offset);
        set_buffer(encoder, 1, &result.buffer, partial_tokens_offset);
        set_buffer(encoder, 2, &result.buffer, partial_invalids_offset);
        set_buffer(encoder, 3, &result.buffer, result_offset);
        set_bytes(encoder, 4, &dispatch.temperature);
        set_bytes(encoder, 5, &dispatch.top_k);
        set_bytes(encoder, 6, &dispatch.top_p);
        set_bytes(encoder, 7, &seed);
        set_bytes(encoder, 8, &counter);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MetalDevice::grid(1, 1, 1),
            MetalDevice::grid(FILTER_THREADS, 1, 1),
        );
        encoder.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
    });
    Ok(())
}

/// Allocates a shared `[status, token]` u32 result, warms the required pipeline,
/// and encodes one sample. The returned tensor remains GPU-resident.
pub fn sample(logits: &MetalTensor, options: SamplingOptions) -> crate::err::Res<MetalTensor> {
    warm_exact(logits, options)?;
    let result = MetalTensor {
        buffer: MetalDevice::get().alloc_raw_checked(required_result_allocation_bytes(2)?)?,
        layout: crate::runtime::layout::Layout::contiguous(vec![2]),
        dtype: DType::U32,
    };
    sample_into(logits, &result, 0, options)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::layout::Layout;

    fn options() -> SamplingOptions {
        SamplingOptions {
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            seed: 7,
            counter: 3,
        }
    }

    fn result(logits: &MetalTensor, options: SamplingOptions) -> [u32; 2] {
        let result = sample(logits, options).unwrap();
        MetalDevice::get()
            .synchronize_buffer(&result.buffer)
            .unwrap();
        let values = result.to_u32_vec().unwrap();
        [values[0], values[1]]
    }

    fn cpu_order(values: &[f32]) -> Vec<usize> {
        let mut tokens = (0..values.len()).collect::<Vec<_>>();
        tokens.sort_unstable_by(|left, right| {
            if values[*left] > values[*right] {
                std::cmp::Ordering::Less
            } else if values[*left] < values[*right] {
                std::cmp::Ordering::Greater
            } else {
                left.cmp(right)
            }
        });
        tokens
    }

    #[test]
    fn result_allocation_includes_one_fixed_scratch_tail() {
        assert_eq!(
            required_result_allocation_bytes(2).unwrap(),
            2 * DType::U32.size_in_bytes() + FILTER_SCRATCH_BYTES
        );
        assert!(required_result_allocation_bytes(usize::MAX).is_err());
    }

    #[test]
    fn sample_into_rejects_a_logical_only_result_allocation() {
        let device = MetalDevice::get();
        let logits = MetalTensor::from_f32(device, vec![1.0, 3.0], vec![2]);
        let options = SamplingOptions {
            temperature: 0.0,
            ..options()
        };
        warm_exact(&logits, options).unwrap();
        let output = MetalTensor {
            buffer: device
                .alloc_raw_checked(2 * DType::U32.size_in_bytes())
                .unwrap(),
            layout: Layout::contiguous(vec![2]),
            dtype: DType::U32,
        };
        assert!(sample_into(&logits, &output, 0, options)
            .unwrap_err()
            .contains("including sampling scratch"));
    }

    #[test]
    fn greedy_prefers_lower_token_and_ignores_filters() {
        let logits = MetalTensor::from_f32(MetalDevice::get(), vec![1.0, 4.0, 4.0, 2.0], vec![4]);
        assert_eq!(
            result(
                &logits,
                SamplingOptions {
                    temperature: 0.0,
                    top_k: Some(4),
                    top_p: 0.1,
                    ..options()
                }
            ),
            [STATUS_OK, 1]
        );
    }

    #[test]
    fn greedy_reads_strided_f16_bf16_and_f32() {
        let values = [100.0_f32, 1.0, -100.0, 5.0, -100.0, 3.0];
        let cases = [
            (
                DType::F32,
                values
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                DType::F16,
                values
                    .iter()
                    .flat_map(|value| half::f16::from_f32(*value).to_bits().to_ne_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                DType::BF16,
                values
                    .iter()
                    .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_ne_bytes())
                    .collect::<Vec<_>>(),
            ),
        ];
        for (dtype, bytes) in cases {
            let logits = MetalTensor {
                buffer: MetalDevice::get().upload_bytes(&bytes),
                layout: Layout::new(vec![3], vec![2], 1),
                dtype,
            };
            assert_eq!(
                result(
                    &logits,
                    SamplingOptions {
                        temperature: 0.0,
                        ..options()
                    }
                ),
                [STATUS_OK, 1],
                "{dtype}"
            );
        }
    }

    #[test]
    fn identical_seed_and_counter_replay_categorical_draw() {
        let logits = MetalTensor::from_f32(MetalDevice::get(), vec![0.0; 257], vec![257]);
        let first = result(&logits, options());
        let second = result(&logits, options());
        assert_eq!(first, second);
        assert_eq!(first[0], STATUS_OK);
        assert!(first[1] < 257);
    }

    #[test]
    fn categorical_scales_finite_extremes_without_subtraction_overflow() {
        let logits = MetalTensor::from_f32(MetalDevice::get(), vec![f32::MAX, -f32::MAX], vec![2]);
        assert_eq!(
            result(
                &logits,
                SamplingOptions {
                    temperature: f32::MAX as f64,
                    seed: 0,
                    counter: 0,
                    ..options()
                }
            ),
            [STATUS_OK, 1]
        );
    }

    #[test]
    fn muse_width_top_k_40_then_top_p_stays_in_candidates() {
        const MUSE_VOCAB: usize = 151_936;
        let mut values = vec![-20.0; MUSE_VOCAB];
        for rank in 0..40 {
            values[10_000 + rank] = 10.0 - rank as f32 * 0.05;
        }
        let logits = MetalTensor::from_f32(MetalDevice::get(), values, vec![MUSE_VOCAB]);
        let sampled = result(
            &logits,
            SamplingOptions {
                temperature: 0.7,
                top_k: Some(40),
                top_p: 0.95,
                seed: 11,
                counter: 29,
            },
        );
        assert_eq!(sampled[0], STATUS_OK);
        assert!((10_000..10_040).contains(&(sampled[1] as usize)));
    }

    #[test]
    fn filtered_random_vectors_match_cpu_greedy_and_top_k_membership() {
        const DRAWS: usize = 8;
        let device = MetalDevice::get();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;

        for length in [67, 257, 4099] {
            let mut values = Vec::with_capacity(length);
            for index in 0..length {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = if index.is_multiple_of(97) {
                    3.25
                } else {
                    ((state >> 32) as u32 % 8192) as f32 / 37.0 - 100.0
                };
                values.push(value);
            }
            let ordered = cpu_order(&values);
            let logits = MetalTensor::from_f32(device, values, vec![length]);
            assert_eq!(
                result(
                    &logits,
                    SamplingOptions {
                        temperature: 0.0,
                        ..options()
                    }
                ),
                [STATUS_OK, ordered[0] as u32]
            );

            let filtered = SamplingOptions {
                temperature: 0.7,
                top_k: Some(40),
                top_p: 0.95,
                seed: 41,
                counter: 0,
            };
            warm_exact(&logits, filtered).unwrap();
            let output = MetalTensor {
                buffer: device
                    .alloc_raw_checked(required_result_allocation_bytes(DRAWS * 2).unwrap())
                    .unwrap(),
                layout: Layout::contiguous(vec![DRAWS * 2]),
                dtype: DType::U32,
            };
            for counter in 0..DRAWS {
                sample_into(
                    &logits,
                    &output,
                    counter * 2,
                    SamplingOptions {
                        counter: counter as u64,
                        ..filtered
                    },
                )
                .unwrap();
            }
            device.synchronize_buffer(&output.buffer).unwrap();
            let sampled = output.to_u32_vec().unwrap();
            for pair in sampled.chunks_exact(2) {
                assert_eq!(pair[0], STATUS_OK);
                assert!(ordered[..40].contains(&(pair[1] as usize)));
            }
        }
    }

    #[test]
    fn filtered_equal_logits_prefer_the_lower_token() {
        let logits = MetalTensor::from_f32(MetalDevice::get(), vec![2.0; 4099], vec![4099]);
        assert_eq!(
            result(
                &logits,
                SamplingOptions {
                    temperature: 1.0,
                    top_k: Some(40),
                    top_p: 0.01,
                    seed: 11,
                    counter: 29,
                }
            ),
            [STATUS_OK, 0]
        );
    }

    #[test]
    fn top_p_retains_and_can_sample_the_crossing_token() {
        let logits = MetalTensor::from_f32(MetalDevice::get(), vec![2.0, 1.0, -20.0], vec![3]);
        assert_eq!(
            result(
                &logits,
                SamplingOptions {
                    temperature: 1.0,
                    top_k: Some(3),
                    top_p: 0.8,
                    seed: 0,
                    counter: 0,
                }
            ),
            [STATUS_OK, 1]
        );
    }

    #[test]
    fn positive_temperature_rejects_top_p_only_and_top_k_65() {
        let logits = MetalTensor::from_f32(MetalDevice::get(), vec![0.0; 128], vec![128]);
        let top_p_only = SamplingOptions {
            top_p: 0.9,
            ..options()
        };
        let top_k_65 = SamplingOptions {
            top_k: Some(65),
            ..options()
        };
        assert!(sample(&logits, top_p_only)
            .unwrap_err()
            .contains("requires topK"));
        assert!(sample(&logits, top_k_65).unwrap_err().contains("limit 64"));
    }

    #[test]
    fn every_mode_reports_the_first_nonfinite_logit() {
        let logits = MetalTensor::from_f32(
            MetalDevice::get(),
            vec![0.0, f32::INFINITY, f32::NAN, 1.0],
            vec![4],
        );
        let cases = [
            SamplingOptions {
                temperature: 0.0,
                ..options()
            },
            options(),
            SamplingOptions {
                top_k: Some(3),
                top_p: 0.8,
                ..options()
            },
        ];
        for case in cases {
            assert_eq!(result(&logits, case), [STATUS_NONFINITE, 1]);
        }
    }

    #[test]
    fn sample_into_writes_at_the_requested_result_index() {
        let logits = MetalTensor::from_f32(MetalDevice::get(), vec![1.0, 3.0], vec![2]);
        let options = SamplingOptions {
            temperature: 0.0,
            ..options()
        };
        warm_exact(&logits, options).unwrap();
        let output = MetalTensor {
            buffer: MetalDevice::get()
                .alloc_raw_checked(required_result_allocation_bytes(4).unwrap())
                .unwrap(),
            layout: Layout::contiguous(vec![4]),
            dtype: DType::U32,
        };
        sample_into(&logits, &output, 2, options).unwrap();
        MetalDevice::get()
            .synchronize_buffer(&output.buffer)
            .unwrap();
        let values = output.to_u32_vec().unwrap();
        assert_eq!(&values[2..], &[STATUS_OK, 1]);
    }

    #[test]
    fn prewarmed_sample_into_is_legal_when_allocations_and_misses_are_forbidden() {
        let device = MetalDevice::get();
        let logits = MetalTensor::from_f32(device, vec![1.0, 3.0], vec![2]);
        let options = SamplingOptions {
            temperature: 0.0,
            ..options()
        };
        warm_exact(&logits, options).unwrap();
        let output = MetalTensor {
            buffer: device
                .alloc_raw_checked(required_result_allocation_bytes(2).unwrap())
                .unwrap(),
            layout: Layout::contiguous(vec![2]),
            dtype: DType::U32,
        };
        {
            let _guard = device.begin_executable_dispatch().unwrap();
            sample_into(&logits, &output, 0, options).unwrap();
            device.commit_executable_command();
        }
        device.synchronize_buffer(&output.buffer).unwrap();
        assert_eq!(output.to_u32_vec().unwrap(), [STATUS_OK, 1]);
    }
}

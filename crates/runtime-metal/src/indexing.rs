//! Indexing kernels: `index_select`, `gather`, `scatter_add`,
//! `scatter_set`, and `cat`.
//!
//! # Conventions
//!
//! - **Strides baked in.** Input/output stride decompositions are emitted
//!   as constant arithmetic over the flat thread id; pipeline keys hash
//!   the exact layouts, so a layout change is a different kernel.
//!   Destinations are contiguous; cat and the gather family consume
//!   strided sources directly with no scratch copy.
//! - **Index tensors.** Kernels read u32 indexes; other dtypes (or
//!   non-contiguous index layouts) are converted by a layout-keyed
//!   `et_index_convert` kernel into caller-provided `ids` scratch.
//! - **scatter_add.** Runs in f32 with relaxed-order device atomics:
//!   duplicate indexes accumulate in an unspecified (but lossless) order,
//!   so results are deterministic in value for exact sums but not in
//!   rounding. Non-f32 operands round-trip through f32 `accumulator` /
//!   `source_cast` scratch.
//! - **Requirements contract.** `*_requirements` report the exact output,
//!   scratch, staging, and pipeline count; the `*_into` entry points
//!   validate against them, require precompiled pipelines, and allocate
//!   nothing.
//! - **Dispatch.** One thread per output element (per source element for
//!   scatter/cat), 256-wide threadgroups, 32-bit flat indexing — indexing
//!   workloads are index-tensor sized, far from the u32 limit.

use super::device::{set_buffer, set_bytes, MetalDevice, Pipeline};
use super::run::MetalTensor;
use crate::runtime::dtype::DType;
use crate::runtime::layout::Layout;
use objc2_metal::MTLComputeCommandEncoder;

/// An exact tensor (contiguous shape + dtype) an indexing operation
/// requires as output, scratch, or staging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRequirement {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Alias kept for the executable planner's vocabulary.
pub type BufferRequirement = TensorRequirement;

impl TensorRequirement {
    /// Byte size of the described tensor, checked for overflow.
    pub fn bytes(&self) -> Result<usize, String> {
        checked_product(&self.shape)?
            .checked_mul(self.dtype.size_in_bytes())
            .ok_or_else(|| {
                format!(
                    "indexing tensor byte size overflow for {:?}:{:?}",
                    self.shape, self.dtype
                )
            })
    }
}

/// Exact caller-owned resources for one indexing operation.
///
/// `ids` stores converted/compacted U32 indexes. Non-f32 scatter-add also
/// needs an f32 `accumulator` and `source_cast`. Host scatter-set callers use
/// `staging` for the uploaded U32 indexes. Cat and layout-aware reads need no
/// copy scratch because their kernels consume source strides directly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexingRequirements {
    /// The contiguous output tensor (absent for in-place scatter_set).
    pub output: Option<TensorRequirement>,
    /// Converted contiguous u32 indexes (absent when the ids are already
    /// contiguous u32).
    pub ids: Option<TensorRequirement>,
    /// f32 accumulation buffer for non-f32 scatter_add.
    pub accumulator: Option<TensorRequirement>,
    /// f32 cast of the scatter_add source for non-f32 inputs.
    pub source_cast: Option<TensorRequirement>,
    /// Uploaded u32 index buffer for host-side scatter_set callers.
    pub staging: Option<TensorRequirement>,
    /// Byte size of `output`.
    pub output_bytes: usize,
    /// Combined byte size of `ids` + `accumulator` + `source_cast`.
    pub scratch_bytes: usize,
    /// Byte size of `staging`.
    pub staging_bytes: usize,
    /// Always zero for indexing: no per-invocation status buffer.
    pub status_bytes: usize,
    /// Distinct pipelines the operation needs precompiled.
    pub pipeline_count: usize,
}

fn finalize_requirements(
    mut requirements: IndexingRequirements,
    pipeline_count: usize,
) -> Result<IndexingRequirements, String> {
    requirements.output_bytes = requirements
        .output
        .as_ref()
        .map_or(Ok(0), TensorRequirement::bytes)?;
    requirements.scratch_bytes = requirements.scratch_bytes()?;
    requirements.staging_bytes = requirements.staging_bytes()?;
    requirements.pipeline_count = pipeline_count;
    Ok(requirements)
}

impl IndexingRequirements {
    /// Combined byte size of the `ids`, `accumulator`, and `source_cast`
    /// scratch tensors.
    pub fn scratch_bytes(&self) -> Result<usize, String> {
        [&self.ids, &self.accumulator, &self.source_cast]
            .into_iter()
            .flatten()
            .try_fold(0usize, |total, requirement| {
                total
                    .checked_add(requirement.bytes()?)
                    .ok_or_else(|| "indexing scratch byte size overflow".to_string())
            })
    }

    /// Byte size of the `staging` tensor.
    pub fn staging_bytes(&self) -> Result<usize, String> {
        self.staging
            .as_ref()
            .map_or(Ok(0), TensorRequirement::bytes)
    }
}

fn msl_type(dtype: DType) -> &'static str {
    match dtype {
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
    let mut hasher = DefaultHasher::new();
    for part in parts {
        part.hash(&mut hasher);
    }
    hasher.finish()
}

fn layout_key(layout: &Layout) -> u64 {
    let mut parts = Vec::with_capacity(layout.rank() * 2);
    parts.extend(layout.shape().iter().map(|&value| value as u64));
    parts.extend(layout.strides().iter().map(|&value| value as u64));
    key(&parts)
}

fn cached_pipeline(dev: &MetalDevice, pipeline_key: u64, name: &str) -> Result<Pipeline, String> {
    dev.pipeline_cached(pipeline_key).ok_or_else(|| {
        format!("{name}: exact pipeline is not warm; call the matching exact warm function")
    })
}

fn convert_key(layout: &Layout, source: DType, destination: DType) -> u64 {
    key(&[
        0xC012,
        source as u64,
        destination as u64,
        layout_key(layout),
    ])
}

fn index_select_key(layout: &Layout, dtype: DType, dim: usize, ids_len: usize) -> u64 {
    key(&[
        0x15E1,
        dtype as u64,
        dim as u64,
        ids_len as u64,
        layout_key(layout),
    ])
}

fn gather_key(layout: &Layout, dtype: DType, dim: usize, ids_shape: &[usize]) -> u64 {
    key(&[
        0x6A7E,
        dtype as u64,
        dim as u64,
        key(&ids_shape
            .iter()
            .map(|&value| value as u64)
            .collect::<Vec<_>>()),
        layout_key(layout),
    ])
}

fn scatter_add_key(output_shape: &[usize], source_layout: &Layout, dim: usize) -> u64 {
    key(&[
        0x5ADD,
        dim as u64,
        layout_key(&Layout::contiguous(output_shape.to_vec())),
        layout_key(source_layout),
    ])
}

fn cat_key(source_layout: &Layout, output_shape: &[usize], dtype: DType, dim: usize) -> u64 {
    key(&[
        0xCA7,
        dtype as u64,
        dim as u64,
        layout_key(source_layout),
        key(&output_shape
            .iter()
            .map(|&value| value as u64)
            .collect::<Vec<_>>()),
    ])
}

fn scatter_set_key(
    destination_layout: &Layout,
    source_layout: &Layout,
    dtype: DType,
    dim: usize,
) -> u64 {
    key(&[
        0x55E7,
        dtype as u64,
        dim as u64,
        layout_key(destination_layout),
        layout_key(source_layout),
    ])
}

fn checked_product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1usize, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| "indexing element count overflow".to_string())
    })
}

fn contiguous_requirement(shape: &[usize], dtype: DType) -> TensorRequirement {
    TensorRequirement {
        shape: shape.to_vec(),
        dtype,
    }
}

fn ids_requirement(layout: &Layout, dtype: DType) -> Option<TensorRequirement> {
    if dtype == DType::U32 && layout.is_contiguous() {
        None
    } else {
        Some(contiguous_requirement(layout.shape(), DType::U32))
    }
}

fn validate_dim(layout: &Layout, dim: usize, name: &str) -> Result<(), String> {
    if dim >= layout.rank() {
        return Err(format!(
            "{name}: dimension {dim} is out of range for rank {}",
            layout.rank()
        ));
    }
    Ok(())
}

fn validate_destination(
    destination: &MetalTensor,
    requirement: &TensorRequirement,
    name: &str,
) -> Result<(), String> {
    if destination.layout.shape() != requirement.shape || destination.dtype != requirement.dtype {
        return Err(format!(
            "{name}: destination mismatch, expected {:?}:{:?}, got {:?}:{:?}",
            requirement.shape,
            requirement.dtype,
            destination.layout.shape(),
            destination.dtype
        ));
    }
    if !destination.layout.is_contiguous() {
        return Err(format!("{name}: destination must be contiguous"));
    }
    let offset = destination
        .layout
        .offset()
        .checked_mul(destination.dtype.size_in_bytes())
        .ok_or_else(|| format!("{name}: destination offset overflow"))?;
    let required = requirement.bytes()?;
    if offset
        .checked_add(required)
        .is_none_or(|end| end > destination.buffer.size)
    {
        return Err(format!(
            "{name}: destination buffer has {} bytes, needs {required} bytes at offset {offset}",
            destination.buffer.size
        ));
    }
    MetalDevice::get().mark_buffer_write(&destination.buffer)?;
    Ok(())
}

fn validate_scratch(
    scratch: Option<&MetalTensor>,
    requirement: &Option<TensorRequirement>,
    name: &str,
) -> Result<(), String> {
    match (scratch, requirement) {
        (None, None) => Ok(()),
        (Some(_), None) => Err(format!("{name}: unexpected scratch tensor")),
        (None, Some(requirement)) => Err(format!(
            "{name}: missing scratch tensor {:?}:{:?}",
            requirement.shape, requirement.dtype
        )),
        (Some(scratch), Some(requirement)) => {
            if scratch.layout.offset() != 0 {
                return Err(format!("{name}: scratch must have zero layout offset"));
            }
            validate_destination(scratch, requirement, name)
        }
    }
}

fn offset_bytes(tensor: &MetalTensor) -> usize {
    tensor.layout.offset() * tensor.dtype.size_in_bytes()
}

fn decompose(layout: &Layout, index: &str, variable: &str) -> String {
    let contiguous = Layout::contiguous(layout.shape().to_vec());
    let mut source = String::new();
    for dimension in 0..layout.rank() {
        if layout.shape()[dimension] == 1 || layout.strides()[dimension] == 0 {
            continue;
        }
        let coordinate = if dimension + 1 == layout.rank() {
            format!("({index} % {}ul)", layout.shape()[dimension])
        } else {
            format!(
                "(({index} / {}ul) % {}ul)",
                contiguous.strides()[dimension],
                layout.shape()[dimension]
            )
        };
        source.push_str(&format!(
            "    {variable} += {coordinate} * {}ul;\n",
            layout.strides()[dimension]
        ));
    }
    source
}

fn convert_pipeline(
    dev: &MetalDevice,
    source_layout: &Layout,
    source_dtype: DType,
    destination_dtype: DType,
) -> Result<Pipeline, String> {
    let total = source_layout.numel();
    let source_ty = msl_type(source_dtype);
    let destination_ty = msl_type(destination_dtype);
    let decomposition = decompose(source_layout, "i", "source_offset");
    let make_source = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_index_convert(device const {source_ty}* source [[buffer(0)]], device {destination_ty}* destination [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    const ulong i = (ulong)gid;
    if (i >= {total}ul) return;
    ulong source_offset = 0ul;
{decomposition}    destination[i] = ({destination_ty})source[source_offset];
}}
"#
        )
    };
    dev.compile_lazy(
        convert_key(source_layout, source_dtype, destination_dtype),
        "et_index_convert",
        make_source,
    )
}

fn convert_into(
    dev: &MetalDevice,
    source: &MetalTensor,
    destination: &MetalTensor,
) -> Result<(), String> {
    let requirement = contiguous_requirement(source.layout.shape(), destination.dtype);
    validate_destination(destination, &requirement, "index conversion")?;
    let total = source.numel();
    if total == 0 {
        return Ok(());
    }
    let pipeline = cached_pipeline(
        dev,
        convert_key(&source.layout, source.dtype, destination.dtype),
        "index conversion",
    )?;
    dispatch(
        dev,
        &pipeline,
        total,
        &[
            (0, &source.buffer, offset_bytes(source)),
            (1, &destination.buffer, offset_bytes(destination)),
        ],
        &[],
    );
    Ok(())
}

fn dispatch(
    dev: &MetalDevice,
    pipeline: &Pipeline,
    total: usize,
    buffers: &[(usize, &super::device::Buffer, usize)],
    scalars: &[(usize, u32)],
) {
    let padded = total.div_ceil(256) * 256;
    dev.with_encoder(|encoder| {
        encoder.setComputePipelineState(pipeline.as_raw());
        for &(index, buffer, offset) in buffers {
            set_buffer(encoder, index, buffer, offset);
        }
        for &(index, value) in scalars {
            set_bytes(encoder, index, &value);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MetalDevice::grid(padded, 1, 1),
            MetalDevice::grid(256, 1, 1),
        );
    });
}

fn prepare_ids<'a>(
    dev: &MetalDevice,
    ids: &'a MetalTensor,
    scratch: Option<&'a MetalTensor>,
    requirement: &Option<TensorRequirement>,
    name: &str,
) -> Result<&'a MetalTensor, String> {
    validate_scratch(scratch, requirement, name)?;
    if let Some(scratch) = scratch {
        convert_into(dev, ids, scratch)?;
        Ok(scratch)
    } else {
        Ok(ids)
    }
}

/// Uploads host u32 indexes as a contiguous device tensor.
pub fn ids_from_host(dev: &MetalDevice, ids: &[u32]) -> MetalTensor {
    MetalTensor {
        buffer: dev.alloc_with_data_u32(ids),
        layout: Layout::contiguous(vec![ids.len()]),
        dtype: DType::U32,
    }
}

/// Exact resources for `index_select(input, dim, ids)`: the output shape
/// replaces `dim`'s extent with the index count.
pub fn index_select_requirements(
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
) -> Result<IndexingRequirements, String> {
    validate_dim(input_layout, dim, "index_select")?;
    let mut shape = input_layout.shape().to_vec();
    shape[dim] = ids_layout.numel();
    let requirements = IndexingRequirements {
        output: Some(contiguous_requirement(&shape, input_dtype)),
        ids: ids_requirement(ids_layout, ids_dtype),
        ..IndexingRequirements::default()
    };
    let pipelines = usize::from(checked_product(&shape)? != 0)
        + usize::from(requirements.ids.is_some() && ids_layout.numel() != 0);
    finalize_requirements(requirements, pipelines)
}

/// Exact resources for `gather(input, dim, ids)`; `ids_shape` must have
/// the input's rank and may not exceed the input extent outside `dim`.
pub fn gather_requirements(
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    ids_shape: &[usize],
) -> Result<IndexingRequirements, String> {
    validate_dim(input_layout, dim, "gather")?;
    if ids_shape.len() != input_layout.rank() {
        return Err(format!(
            "gather: index rank {} does not match input rank {}",
            ids_shape.len(),
            input_layout.rank()
        ));
    }
    if checked_product(ids_shape)? != ids_layout.numel() {
        return Err(format!(
            "gather: index shape {ids_shape:?} has a different element count from {:?}",
            ids_layout.shape()
        ));
    }
    for dimension in 0..ids_shape.len() {
        if dimension != dim && ids_shape[dimension] > input_layout.shape()[dimension] {
            return Err(format!(
                "gather: index extent {} exceeds input extent {} at dimension {dimension}",
                ids_shape[dimension],
                input_layout.shape()[dimension]
            ));
        }
    }
    let requirements = IndexingRequirements {
        output: Some(contiguous_requirement(ids_shape, input_dtype)),
        ids: ids_requirement(ids_layout, ids_dtype),
        ..IndexingRequirements::default()
    };
    let pipelines = usize::from(checked_product(ids_shape)? != 0)
        + usize::from(requirements.ids.is_some() && ids_layout.numel() != 0);
    finalize_requirements(requirements, pipelines)
}

/// Exact resources for `scatter_add(input, dim, ids, source)`; non-f32
/// dtypes additionally require the f32 accumulator and source cast.
pub fn scatter_add_requirements(
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    source_layout: &Layout,
    source_dtype: DType,
) -> Result<IndexingRequirements, String> {
    validate_dim(input_layout, dim, "scatter_add")?;
    if source_layout.rank() != input_layout.rank() {
        return Err("scatter_add: source and input ranks must match".to_string());
    }
    if source_dtype != input_dtype {
        return Err(format!(
            "scatter_add: source dtype {source_dtype:?} does not match input dtype {input_dtype:?}"
        ));
    }
    if ids_layout.numel() != source_layout.numel() {
        return Err(format!(
            "scatter_add: indexes have {} elements but source has {}",
            ids_layout.numel(),
            source_layout.numel()
        ));
    }
    for dimension in 0..source_layout.rank() {
        if dimension != dim && source_layout.shape()[dimension] > input_layout.shape()[dimension] {
            return Err(format!(
                "scatter_add: source extent exceeds input at dimension {dimension}"
            ));
        }
    }
    let needs_cast = input_dtype != DType::F32;
    let requirements = IndexingRequirements {
        output: Some(contiguous_requirement(input_layout.shape(), input_dtype)),
        ids: ids_requirement(ids_layout, ids_dtype),
        accumulator: needs_cast.then(|| contiguous_requirement(input_layout.shape(), DType::F32)),
        source_cast: needs_cast.then(|| contiguous_requirement(source_layout.shape(), DType::F32)),
        staging: None,
        ..IndexingRequirements::default()
    };
    let conversions = usize::from(input_layout.numel() != 0)
        + usize::from(needs_cast && input_layout.numel() != 0)
        + usize::from(needs_cast && source_layout.numel() != 0);
    let pipelines = conversions
        + usize::from(source_layout.numel() != 0)
        + usize::from(requirements.ids.is_some() && ids_layout.numel() != 0);
    finalize_requirements(requirements, pipelines)
}

/// Exact resources for concatenating same-rank, same-dtype tensors along
/// `dim`.
pub fn cat_requirements(
    layouts: &[&Layout],
    dtypes: &[DType],
    dim: usize,
) -> Result<IndexingRequirements, String> {
    if layouts.is_empty() || layouts.len() != dtypes.len() {
        return Err("cat: layouts and dtypes must be non-empty and have equal length".to_string());
    }
    validate_dim(layouts[0], dim, "cat")?;
    let dtype = dtypes[0];
    let mut shape = layouts[0].shape().to_vec();
    let mut concatenated = 0usize;
    for (layout, &candidate_dtype) in layouts.iter().zip(dtypes) {
        if candidate_dtype != dtype || layout.rank() != shape.len() {
            return Err("cat: all inputs must have the same dtype and rank".to_string());
        }
        for dimension in 0..shape.len() {
            if dimension != dim && layout.shape()[dimension] != shape[dimension] {
                return Err(format!("cat: shape mismatch at dimension {dimension}"));
            }
        }
        concatenated = concatenated
            .checked_add(layout.shape()[dim])
            .ok_or_else(|| "cat: concatenated dimension overflow".to_string())?;
    }
    shape[dim] = concatenated;
    let requirements = IndexingRequirements {
        output: Some(contiguous_requirement(&shape, dtype)),
        ..IndexingRequirements::default()
    };
    let pipelines = layouts
        .iter()
        .zip(dtypes)
        .filter(|(layout, _)| layout.numel() != 0)
        .map(|(layout, &dtype)| cat_key(layout, &shape, dtype, dim))
        .collect::<std::collections::HashSet<_>>()
        .len();
    finalize_requirements(requirements, pipelines)
}

/// Exact resources for in-place `scatter_set(destination, dim, ids,
/// source)` (no output allocation).
pub fn scatter_set_requirements(
    destination_layout: &Layout,
    destination_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    source_layout: &Layout,
    source_dtype: DType,
) -> Result<IndexingRequirements, String> {
    validate_dim(destination_layout, dim, "scatter_set")?;
    if source_layout.rank() != destination_layout.rank() || source_dtype != destination_dtype {
        return Err("scatter_set: source rank and dtype must match destination".to_string());
    }
    if ids_layout.numel() != source_layout.shape()[dim] {
        return Err(format!(
            "scatter_set: expected {} indexes, got {}",
            source_layout.shape()[dim],
            ids_layout.numel()
        ));
    }
    for dimension in 0..source_layout.rank() {
        if dimension != dim
            && source_layout.shape()[dimension] != destination_layout.shape()[dimension]
        {
            return Err(format!(
                "scatter_set: source shape mismatch at dimension {dimension}"
            ));
        }
    }
    let requirements = IndexingRequirements {
        ids: ids_requirement(ids_layout, ids_dtype),
        ..IndexingRequirements::default()
    };
    let pipelines = usize::from(source_layout.numel() != 0)
        + usize::from(requirements.ids.is_some() && ids_layout.numel() != 0);
    finalize_requirements(requirements, pipelines)
}

/// [`scatter_set_requirements`] for host-supplied u32 indexes, adding the
/// upload staging buffer.
pub fn scatter_set_staging_requirements(
    destination_layout: &Layout,
    destination_dtype: DType,
    dim: usize,
    ids_len: usize,
    source_layout: &Layout,
    source_dtype: DType,
) -> Result<IndexingRequirements, String> {
    let ids_layout = Layout::contiguous(vec![ids_len]);
    let mut requirements = scatter_set_requirements(
        destination_layout,
        destination_dtype,
        dim,
        &ids_layout,
        DType::U32,
        source_layout,
        source_dtype,
    )?;
    requirements.staging = Some(contiguous_requirement(&[ids_len], DType::U32));
    requirements.staging_bytes = requirements.staging_bytes()?;
    Ok(requirements)
}

fn index_select_pipeline(
    dev: &MetalDevice,
    input_layout: &Layout,
    dtype: DType,
    dim: usize,
    ids_len: usize,
) -> Result<Pipeline, String> {
    let mut output_shape = input_layout.shape().to_vec();
    output_shape[dim] = ids_len;
    let total = checked_product(&output_shape)?;
    let ty = msl_type(dtype);
    let mut decomposition = String::new();
    for dimension in (0..input_layout.rank()).rev() {
        let count = output_shape[dimension];
        let divisor = output_shape[dimension + 1..]
            .iter()
            .product::<usize>()
            .max(1);
        let coordinate = format!("((gid / {divisor}u) % {count}u)");
        if dimension == dim {
            decomposition.push_str(&format!(
                "    source_offset += ids[{coordinate}] * {}ul;\n",
                input_layout.strides()[dimension]
            ));
        } else {
            decomposition.push_str(&format!(
                "    source_offset += {coordinate} * {}ul;\n",
                input_layout.strides()[dimension]
            ));
        }
    }
    let make_source = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_isel(device const {ty}* input [[buffer(0)]], device const uint* ids [[buffer(1)]], device {ty}* output [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return;
    ulong source_offset = 0ul;
{decomposition}    output[gid] = input[source_offset];
}}
"#
        )
    };
    dev.compile_lazy(
        index_select_key(input_layout, dtype, dim, ids_len),
        "et_isel",
        make_source,
    )
}

fn gather_pipeline(
    dev: &MetalDevice,
    input_layout: &Layout,
    dtype: DType,
    dim: usize,
    ids_shape: &[usize],
) -> Result<Pipeline, String> {
    let total = checked_product(ids_shape)?;
    let ty = msl_type(dtype);
    let mut decomposition = String::new();
    for dimension in (0..input_layout.rank()).rev() {
        let count = ids_shape[dimension];
        let divisor = ids_shape[dimension + 1..].iter().product::<usize>().max(1);
        let coordinate = format!("((gid / {divisor}u) % {count}u)");
        if dimension == dim {
            decomposition.push_str(&format!(
                "    source_offset += ids[gid] * {}ul;\n",
                input_layout.strides()[dimension]
            ));
        } else {
            decomposition.push_str(&format!(
                "    source_offset += {coordinate} * {}ul;\n",
                input_layout.strides()[dimension]
            ));
        }
    }
    let make_source = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_gather(device const {ty}* input [[buffer(0)]], device const uint* ids [[buffer(1)]], device {ty}* output [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return;
    ulong source_offset = 0ul;
{decomposition}    output[gid] = input[source_offset];
}}
"#
        )
    };
    dev.compile_lazy(
        gather_key(input_layout, dtype, dim, ids_shape),
        "et_gather",
        make_source,
    )
}

fn scatter_add_pipeline(
    dev: &MetalDevice,
    output_shape: &[usize],
    source_layout: &Layout,
    dim: usize,
) -> Result<Pipeline, String> {
    let total = source_layout.numel();
    let output_layout = Layout::contiguous(output_shape.to_vec());
    let source_contiguous = Layout::contiguous(source_layout.shape().to_vec());
    let mut output_decomposition = String::new();
    let mut source_decomposition = String::new();
    for dimension in (0..source_layout.rank()).rev() {
        let count = source_layout.shape()[dimension];
        let divisor = source_contiguous.strides()[dimension];
        let coordinate = format!("((gid / {divisor}u) % {count}u)");
        if dimension == dim {
            output_decomposition.push_str(&format!(
                "    output_offset += ids[gid] * {}ul;\n",
                output_layout.strides()[dimension]
            ));
        } else {
            output_decomposition.push_str(&format!(
                "    output_offset += {coordinate} * {}ul;\n",
                output_layout.strides()[dimension]
            ));
        }
        source_decomposition.push_str(&format!(
            "    source_offset += {coordinate} * {}ul;\n",
            source_layout.strides()[dimension]
        ));
    }
    let make_source = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_sadd(device float* output [[buffer(0)]], device const uint* ids [[buffer(1)]], device const float* source [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return;
    ulong output_offset = 0ul;
{output_decomposition}    ulong source_offset = 0ul;
{source_decomposition}    atomic_fetch_add_explicit((device atomic_float*)&output[output_offset], source[source_offset], memory_order_relaxed);
}}
"#
        )
    };
    dev.compile_lazy(
        scatter_add_key(output_shape, source_layout, dim),
        "et_sadd",
        make_source,
    )
}

fn cat_pipeline(
    dev: &MetalDevice,
    source_layout: &Layout,
    output_shape: &[usize],
    dtype: DType,
    dim: usize,
) -> Result<Pipeline, String> {
    let total = source_layout.numel();
    let ty = msl_type(dtype);
    let inner = checked_product(&output_shape[dim + 1..])?;
    let source_decomposition = decompose(source_layout, "gid", "source_offset");
    let source_dim = source_layout.shape()[dim];
    let output_dim = output_shape[dim];
    let make_source = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_cat(device const {ty}* source [[buffer(0)]], device {ty}* output [[buffer(1)]], constant uint& dim_offset [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return;
    ulong source_offset = 0ul;
{source_decomposition}    const uint outer = gid / ({source_dim}u * {inner}u);
    const uint remainder = gid % ({source_dim}u * {inner}u);
    output[outer * {output_dim}u * {inner}u + dim_offset * {inner}u + remainder] = source[source_offset];
}}
"#
        )
    };
    dev.compile_lazy(
        cat_key(source_layout, output_shape, dtype, dim),
        "et_cat",
        make_source,
    )
}

fn scatter_set_pipeline(
    dev: &MetalDevice,
    destination_layout: &Layout,
    source_layout: &Layout,
    dtype: DType,
    dim: usize,
) -> Result<Pipeline, String> {
    let total = source_layout.numel();
    let ty = msl_type(dtype);
    let source_contiguous = Layout::contiguous(source_layout.shape().to_vec());
    let mut destination_decomposition = String::new();
    let mut source_decomposition = String::new();
    for dimension in (0..source_layout.rank()).rev() {
        let count = source_layout.shape()[dimension];
        let divisor = source_contiguous.strides()[dimension];
        let coordinate = format!("((gid / {divisor}u) % {count}u)");
        if dimension == dim {
            destination_decomposition.push_str(&format!(
                "    destination_offset += ids[{coordinate}] * {}ul;\n",
                destination_layout.strides()[dimension]
            ));
        } else {
            destination_decomposition.push_str(&format!(
                "    destination_offset += {coordinate} * {}ul;\n",
                destination_layout.strides()[dimension]
            ));
        }
        source_decomposition.push_str(&format!(
            "    source_offset += {coordinate} * {}ul;\n",
            source_layout.strides()[dimension]
        ));
    }
    let make_source = || {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;
kernel void et_sset(device {ty}* destination [[buffer(0)]], device const uint* ids [[buffer(1)]], device const {ty}* source [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= {total}u) return;
    ulong destination_offset = 0ul;
{destination_decomposition}    ulong source_offset = 0ul;
{source_decomposition}    destination[destination_offset] = source[source_offset];
}}
"#
        )
    };
    dev.compile_lazy(
        scatter_set_key(destination_layout, source_layout, dtype, dim),
        "et_sset",
        make_source,
    )
}

/// Destination form of [`index_select`]; `ids_scratch` must match the
/// `ids` requirement (converted u32 indexes).
pub fn index_select_into(
    dev: &MetalDevice,
    input: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    destination: &MetalTensor,
    ids_scratch: Option<&MetalTensor>,
) -> Result<(), String> {
    let requirements =
        index_select_requirements(&input.layout, input.dtype, dim, &ids.layout, ids.dtype)?;
    let output = requirements.output.as_ref().expect("index_select output");
    validate_destination(destination, output, "index_select")?;
    let ids = prepare_ids(dev, ids, ids_scratch, &requirements.ids, "index_select ids")?;
    let total = destination.numel();
    if total != 0 {
        let pipeline = cached_pipeline(
            dev,
            index_select_key(&input.layout, input.dtype, dim, ids.numel()),
            "index_select",
        )?;
        dispatch(
            dev,
            &pipeline,
            total,
            &[
                (0, &input.buffer, offset_bytes(input)),
                (1, &ids.buffer, offset_bytes(ids)),
                (2, &destination.buffer, offset_bytes(destination)),
            ],
            &[],
        );
    }
    Ok(())
}

/// Destination form of [`gather`].
pub fn gather_into(
    dev: &MetalDevice,
    input: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    ids_shape: &[usize],
    destination: &MetalTensor,
    ids_scratch: Option<&MetalTensor>,
) -> Result<(), String> {
    let requirements = gather_requirements(
        &input.layout,
        input.dtype,
        dim,
        &ids.layout,
        ids.dtype,
        ids_shape,
    )?;
    let output = requirements.output.as_ref().expect("gather output");
    validate_destination(destination, output, "gather")?;
    let ids = prepare_ids(dev, ids, ids_scratch, &requirements.ids, "gather ids")?;
    let total = destination.numel();
    if total != 0 {
        let pipeline = cached_pipeline(
            dev,
            gather_key(&input.layout, input.dtype, dim, ids_shape),
            "gather",
        )?;
        dispatch(
            dev,
            &pipeline,
            total,
            &[
                (0, &input.buffer, offset_bytes(input)),
                (1, &ids.buffer, offset_bytes(ids)),
                (2, &destination.buffer, offset_bytes(destination)),
            ],
            &[],
        );
    }
    Ok(())
}

/// Destination form of [`scatter_add`]: copies/converts the input into
/// the destination (or f32 accumulator), atomically accumulates the
/// source, and casts back for non-f32 dtypes.
#[allow(clippy::too_many_arguments)]
pub fn scatter_add_into(
    dev: &MetalDevice,
    input: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    source: &MetalTensor,
    destination: &MetalTensor,
    ids_scratch: Option<&MetalTensor>,
    accumulator: Option<&MetalTensor>,
    source_cast: Option<&MetalTensor>,
) -> Result<(), String> {
    let requirements = scatter_add_requirements(
        &input.layout,
        input.dtype,
        dim,
        &ids.layout,
        ids.dtype,
        &source.layout,
        source.dtype,
    )?;
    let output = requirements.output.as_ref().expect("scatter_add output");
    validate_destination(destination, output, "scatter_add")?;
    validate_scratch(
        accumulator,
        &requirements.accumulator,
        "scatter_add accumulator",
    )?;
    validate_scratch(
        source_cast,
        &requirements.source_cast,
        "scatter_add source cast",
    )?;
    let ids = prepare_ids(dev, ids, ids_scratch, &requirements.ids, "scatter_add ids")?;
    let (accumulator, source) = if input.dtype == DType::F32 {
        convert_into(dev, input, destination)?;
        (destination, source)
    } else {
        let accumulator = accumulator.expect("validated scatter_add accumulator");
        let source_cast = source_cast.expect("validated scatter_add source cast");
        convert_into(dev, input, accumulator)?;
        convert_into(dev, source, source_cast)?;
        (accumulator, source_cast)
    };
    if source.numel() != 0 {
        let pipeline = cached_pipeline(
            dev,
            scatter_add_key(input.layout.shape(), &source.layout, dim),
            "scatter_add",
        )?;
        dispatch(
            dev,
            &pipeline,
            source.numel(),
            &[
                (0, &accumulator.buffer, offset_bytes(accumulator)),
                (1, &ids.buffer, offset_bytes(ids)),
                (2, &source.buffer, offset_bytes(source)),
            ],
            &[],
        );
    }
    if input.dtype != DType::F32 {
        convert_into(dev, accumulator, destination)?;
    }
    Ok(())
}

/// Destination form of [`cat`]: one dispatch per non-empty input, each
/// writing its slice at the running `dim` offset.
pub fn cat_into(
    dev: &MetalDevice,
    tensors: &[&MetalTensor],
    dim: usize,
    destination: &MetalTensor,
) -> Result<(), String> {
    let layouts = tensors
        .iter()
        .map(|tensor| &tensor.layout)
        .collect::<Vec<_>>();
    let dtypes = tensors
        .iter()
        .map(|tensor| tensor.dtype)
        .collect::<Vec<_>>();
    let requirements = cat_requirements(&layouts, &dtypes, dim)?;
    let output = requirements.output.as_ref().expect("cat output");
    validate_destination(destination, output, "cat")?;
    let mut dim_offset = 0usize;
    for tensor in tensors {
        let total = tensor.numel();
        if total != 0 {
            let pipeline = cached_pipeline(
                dev,
                cat_key(&tensor.layout, &output.shape, tensor.dtype, dim),
                "cat",
            )?;
            dispatch(
                dev,
                &pipeline,
                total,
                &[
                    (0, &tensor.buffer, offset_bytes(tensor)),
                    (1, &destination.buffer, offset_bytes(destination)),
                ],
                &[(2, dim_offset as u32)],
            );
        }
        dim_offset += tensor.layout.shape()[dim];
    }
    Ok(())
}

/// In-place scatter-set: writes `source` slices into `destination` at the
/// indexed positions along `dim`.
pub fn scatter_set_into(
    dev: &MetalDevice,
    destination: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    source: &MetalTensor,
    ids_scratch: Option<&MetalTensor>,
) -> Result<(), String> {
    let requirements = scatter_set_requirements(
        &destination.layout,
        destination.dtype,
        dim,
        &ids.layout,
        ids.dtype,
        &source.layout,
        source.dtype,
    )?;
    let ids = prepare_ids(dev, ids, ids_scratch, &requirements.ids, "scatter_set ids")?;
    dev.mark_buffer_write(&destination.buffer)?;
    if source.numel() != 0 {
        let pipeline = cached_pipeline(
            dev,
            scatter_set_key(&destination.layout, &source.layout, destination.dtype, dim),
            "scatter_set",
        )?;
        dispatch(
            dev,
            &pipeline,
            source.numel(),
            &[
                (0, &destination.buffer, offset_bytes(destination)),
                (1, &ids.buffer, offset_bytes(ids)),
                (2, &source.buffer, offset_bytes(source)),
            ],
            &[],
        );
    }
    Ok(())
}

fn allocate_requirement(dev: &MetalDevice, requirement: &TensorRequirement) -> MetalTensor {
    MetalTensor::empty(dev, requirement.shape.clone(), requirement.dtype)
}

/// Allocating index_select: `output[.., ids[i], ..] = input[.., i, ..]`
/// along `dim`.
pub fn index_select(
    dev: &MetalDevice,
    input: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
) -> Result<MetalTensor, String> {
    let requirements =
        index_select_requirements(&input.layout, input.dtype, dim, &ids.layout, ids.dtype)?;
    compile_index_select_layout_exact(
        dev,
        &input.layout,
        input.dtype,
        dim,
        &ids.layout,
        ids.dtype,
    )?;
    let output = allocate_requirement(
        dev,
        requirements.output.as_ref().expect("index_select output"),
    );
    let ids_scratch = requirements
        .ids
        .as_ref()
        .map(|requirement| allocate_requirement(dev, requirement));
    index_select_into(dev, input, dim, ids, &output, ids_scratch.as_ref())?;
    Ok(output)
}

/// Allocating gather: per output element, one index lookup along `dim`.
pub fn gather(
    dev: &MetalDevice,
    input: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    ids_shape: &[usize],
) -> Result<MetalTensor, String> {
    let requirements = gather_requirements(
        &input.layout,
        input.dtype,
        dim,
        &ids.layout,
        ids.dtype,
        ids_shape,
    )?;
    compile_gather_layout(
        dev,
        &input.layout,
        input.dtype,
        dim,
        &ids.layout,
        ids.dtype,
        ids_shape,
    )?;
    let output = allocate_requirement(dev, requirements.output.as_ref().expect("gather output"));
    let ids_scratch = requirements
        .ids
        .as_ref()
        .map(|requirement| allocate_requirement(dev, requirement));
    gather_into(
        dev,
        input,
        dim,
        ids,
        ids_shape,
        &output,
        ids_scratch.as_ref(),
    )?;
    Ok(output)
}

/// Allocating scatter_add: `input` plus `source` accumulated at the
/// indexed positions along `dim`.
pub fn scatter_add(
    dev: &MetalDevice,
    input: &MetalTensor,
    dim: usize,
    ids: &MetalTensor,
    source: &MetalTensor,
) -> Result<MetalTensor, String> {
    let requirements = scatter_add_requirements(
        &input.layout,
        input.dtype,
        dim,
        &ids.layout,
        ids.dtype,
        &source.layout,
        source.dtype,
    )?;
    compile_scatter_add_layouts(
        dev,
        &input.layout,
        input.dtype,
        dim,
        &ids.layout,
        ids.dtype,
        &source.layout,
        source.dtype,
    )?;
    let output = allocate_requirement(
        dev,
        requirements.output.as_ref().expect("scatter_add output"),
    );
    let ids_scratch = requirements
        .ids
        .as_ref()
        .map(|requirement| allocate_requirement(dev, requirement));
    let accumulator = requirements
        .accumulator
        .as_ref()
        .map(|requirement| allocate_requirement(dev, requirement));
    let source_cast = requirements
        .source_cast
        .as_ref()
        .map(|requirement| allocate_requirement(dev, requirement));
    scatter_add_into(
        dev,
        input,
        dim,
        ids,
        source,
        &output,
        ids_scratch.as_ref(),
        accumulator.as_ref(),
        source_cast.as_ref(),
    )?;
    Ok(output)
}

/// Allocating concatenation along `dim`.
pub fn cat(dev: &MetalDevice, tensors: &[&MetalTensor], dim: usize) -> Result<MetalTensor, String> {
    let layouts = tensors
        .iter()
        .map(|tensor| &tensor.layout)
        .collect::<Vec<_>>();
    let dtypes = tensors
        .iter()
        .map(|tensor| tensor.dtype)
        .collect::<Vec<_>>();
    let requirements = cat_requirements(&layouts, &dtypes, dim)?;
    compile_cat_layouts(dev, &layouts, &dtypes, dim)?;
    let output = allocate_requirement(dev, requirements.output.as_ref().expect("cat output"));
    cat_into(dev, tensors, dim, &output)?;
    Ok(output)
}

/// In-place scatter_set with host u32 indexes (uploaded through the
/// staging buffer).
pub fn scatter_set(
    dev: &MetalDevice,
    destination: &MetalTensor,
    dim: usize,
    ids: &[u32],
    source: &MetalTensor,
) -> Result<(), String> {
    let requirements = scatter_set_staging_requirements(
        &destination.layout,
        destination.dtype,
        dim,
        ids.len(),
        &source.layout,
        source.dtype,
    )?;
    let staging = requirements.staging.as_ref().expect("scatter_set staging");
    debug_assert_eq!(staging.bytes()?, ids.len() * std::mem::size_of::<u32>());
    compile_scatter_set_layouts(
        dev,
        &destination.layout,
        destination.dtype,
        dim,
        &Layout::contiguous(vec![ids.len()]),
        DType::U32,
        &source.layout,
        source.dtype,
    )?;
    let ids = ids_from_host(dev, ids);
    scatter_set_into(dev, destination, dim, &ids, source, None)
}

/// Precompiles the index conversion (if needed) and index_select pipelines
/// for exact layouts.
pub fn compile_index_select_layout_exact(
    dev: &MetalDevice,
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
) -> Result<(), String> {
    let requirements =
        index_select_requirements(input_layout, input_dtype, dim, ids_layout, ids_dtype)?;
    if let Some(ids) = &requirements.ids {
        if ids_layout.numel() != 0 {
            convert_pipeline(dev, ids_layout, ids_dtype, ids.dtype)?;
        }
    }
    if requirements
        .output
        .as_ref()
        .expect("index_select output")
        .bytes()?
        != 0
    {
        index_select_pipeline(dev, input_layout, input_dtype, dim, ids_layout.numel())?;
    }
    Ok(())
}

/// [`compile_index_select_layout_exact`] for contiguous shapes on the
/// process-wide device.
pub fn warm_index_select(
    shape: &[usize],
    dtype: DType,
    dim: usize,
    ids_len: usize,
) -> Result<(), String> {
    warm_index_select_layout_exact(
        &Layout::contiguous(shape.to_vec()),
        dtype,
        dim,
        &Layout::contiguous(vec![ids_len]),
        DType::U32,
    )
}

/// [`warm_index_select_layout_exact`] with contiguous u32 indexes of
/// length `ids_len`.
pub fn warm_index_select_layout(
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_len: usize,
) -> Result<(), String> {
    warm_index_select_layout_exact(
        input_layout,
        input_dtype,
        dim,
        &Layout::contiguous(vec![ids_len]),
        DType::U32,
    )
}

/// Precompiles the index conversion (if needed) and gather pipelines for
/// exact layouts.
pub fn compile_gather_layout(
    dev: &MetalDevice,
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    ids_shape: &[usize],
) -> Result<(), String> {
    let requirements = gather_requirements(
        input_layout,
        input_dtype,
        dim,
        ids_layout,
        ids_dtype,
        ids_shape,
    )?;
    if let Some(ids) = &requirements.ids {
        if ids_layout.numel() != 0 {
            convert_pipeline(dev, ids_layout, ids_dtype, ids.dtype)?;
        }
    }
    if requirements
        .output
        .as_ref()
        .expect("gather output")
        .bytes()?
        != 0
    {
        gather_pipeline(dev, input_layout, input_dtype, dim, ids_shape)?;
    }
    Ok(())
}

/// [`compile_gather_layout`] for contiguous shapes on the process-wide
/// device.
pub fn warm_gather(
    shape: &[usize],
    dtype: DType,
    dim: usize,
    ids_shape: &[usize],
) -> Result<(), String> {
    warm_gather_layout(
        &Layout::contiguous(shape.to_vec()),
        dtype,
        dim,
        &Layout::contiguous(ids_shape.to_vec()),
        DType::U32,
        ids_shape,
    )
}

/// Precompiles every pipeline scatter_add needs for exact layouts: ids
/// conversion, f32 casts (non-f32 inputs), the atomic accumulation, and
/// the cast back.
#[allow(clippy::too_many_arguments)]
pub fn compile_scatter_add_layouts(
    dev: &MetalDevice,
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    source_layout: &Layout,
    source_dtype: DType,
) -> Result<(), String> {
    let requirements = scatter_add_requirements(
        input_layout,
        input_dtype,
        dim,
        ids_layout,
        ids_dtype,
        source_layout,
        source_dtype,
    )?;
    if let Some(ids) = &requirements.ids {
        if ids_layout.numel() != 0 {
            convert_pipeline(dev, ids_layout, ids_dtype, ids.dtype)?;
        }
    }
    if input_layout.numel() != 0 {
        convert_pipeline(dev, input_layout, input_dtype, DType::F32)?;
    }
    let effective_source = if source_dtype == DType::F32 {
        source_layout.clone()
    } else {
        if source_layout.numel() != 0 {
            convert_pipeline(dev, source_layout, source_dtype, DType::F32)?;
        }
        Layout::contiguous(source_layout.shape().to_vec())
    };
    if source_layout.numel() != 0 {
        scatter_add_pipeline(dev, input_layout.shape(), &effective_source, dim)?;
    }
    if input_dtype != DType::F32 && input_layout.numel() != 0 {
        convert_pipeline(
            dev,
            &Layout::contiguous(input_layout.shape().to_vec()),
            DType::F32,
            input_dtype,
        )?;
    }
    Ok(())
}

/// [`compile_scatter_add_layouts`] for contiguous shapes on the
/// process-wide device.
pub fn warm_scatter_add(
    shape: &[usize],
    dtype: DType,
    dim: usize,
    ids_shape: &[usize],
) -> Result<(), String> {
    warm_scatter_add_layouts(
        &Layout::contiguous(shape.to_vec()),
        dtype,
        dim,
        &Layout::contiguous(ids_shape.to_vec()),
        DType::U32,
        &Layout::contiguous(ids_shape.to_vec()),
        dtype,
    )
}

/// Precompiles one cat pipeline per distinct non-empty input layout.
pub fn compile_cat_layouts(
    dev: &MetalDevice,
    layouts: &[&Layout],
    dtypes: &[DType],
    dim: usize,
) -> Result<(), String> {
    let requirements = cat_requirements(layouts, dtypes, dim)?;
    let output = requirements.output.as_ref().expect("cat output");
    for (layout, &dtype) in layouts.iter().zip(dtypes) {
        if layout.numel() != 0 {
            cat_pipeline(dev, layout, &output.shape, dtype, dim)?;
        }
    }
    Ok(())
}

/// [`compile_cat_layouts`] for contiguous shapes on the process-wide
/// device.
pub fn warm_cat(shapes: &[&[usize]], dtype: DType, dim: usize) -> Result<(), String> {
    if shapes.is_empty() {
        return Ok(());
    }
    let layouts = shapes
        .iter()
        .map(|shape| Layout::contiguous(shape.to_vec()))
        .collect::<Vec<_>>();
    let references = layouts.iter().collect::<Vec<_>>();
    warm_cat_layouts(&references, &vec![dtype; shapes.len()], dim)
}

/// Precompiles the ids conversion (if needed) and scatter_set pipelines
/// for exact layouts.
pub fn compile_scatter_set_layouts(
    dev: &MetalDevice,
    destination_layout: &Layout,
    destination_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    source_layout: &Layout,
    source_dtype: DType,
) -> Result<(), String> {
    let requirements = scatter_set_requirements(
        destination_layout,
        destination_dtype,
        dim,
        ids_layout,
        ids_dtype,
        source_layout,
        source_dtype,
    )?;
    if let Some(ids) = &requirements.ids {
        if ids_layout.numel() != 0 {
            convert_pipeline(dev, ids_layout, ids_dtype, ids.dtype)?;
        }
    }
    if source_layout.numel() != 0 {
        scatter_set_pipeline(
            dev,
            destination_layout,
            source_layout,
            destination_dtype,
            dim,
        )?;
    }
    Ok(())
}

/// [`compile_index_select_layout_exact`] against the process-wide device.
pub fn warm_index_select_layout_exact(
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
) -> Result<(), String> {
    compile_index_select_layout_exact(
        MetalDevice::get(),
        input_layout,
        input_dtype,
        dim,
        ids_layout,
        ids_dtype,
    )
}

/// [`compile_gather_layout`] against the process-wide device.
pub fn warm_gather_layout(
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    ids_shape: &[usize],
) -> Result<(), String> {
    compile_gather_layout(
        MetalDevice::get(),
        input_layout,
        input_dtype,
        dim,
        ids_layout,
        ids_dtype,
        ids_shape,
    )
}

/// [`compile_scatter_add_layouts`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_scatter_add_layouts(
    input_layout: &Layout,
    input_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    source_layout: &Layout,
    source_dtype: DType,
) -> Result<(), String> {
    compile_scatter_add_layouts(
        MetalDevice::get(),
        input_layout,
        input_dtype,
        dim,
        ids_layout,
        ids_dtype,
        source_layout,
        source_dtype,
    )
}

/// [`compile_cat_layouts`] against the process-wide device.
pub fn warm_cat_layouts(layouts: &[&Layout], dtypes: &[DType], dim: usize) -> Result<(), String> {
    compile_cat_layouts(MetalDevice::get(), layouts, dtypes, dim)
}

/// [`compile_scatter_set_layouts`] against the process-wide device.
#[allow(clippy::too_many_arguments)]
pub fn warm_scatter_set_layouts(
    destination_layout: &Layout,
    destination_dtype: DType,
    dim: usize,
    ids_layout: &Layout,
    ids_dtype: DType,
    source_layout: &Layout,
    source_dtype: DType,
) -> Result<(), String> {
    compile_scatter_set_layouts(
        MetalDevice::get(),
        destination_layout,
        destination_dtype,
        dim,
        ids_layout,
        ids_dtype,
        source_layout,
        source_dtype,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocate_optional(
        dev: &MetalDevice,
        requirement: &Option<TensorRequirement>,
    ) -> Option<MetalTensor> {
        requirement
            .as_ref()
            .map(|requirement| allocate_requirement(dev, requirement))
    }

    #[test]
    fn destination_apis_match_allocating_wrappers() {
        let dev = MetalDevice::get();
        let input =
            MetalTensor::from_f32(dev, (0..6).map(|value| value as f32).collect(), vec![2, 3]);

        let select_ids = ids_from_host(dev, &[2, 0]);
        let selected = index_select(dev, &input, 1, &select_ids).unwrap();
        let select_requirements = index_select_requirements(
            &input.layout,
            input.dtype,
            1,
            &select_ids.layout,
            select_ids.dtype,
        )
        .unwrap();
        let selected_into = allocate_requirement(dev, select_requirements.output.as_ref().unwrap());
        index_select_into(dev, &input, 1, &select_ids, &selected_into, None).unwrap();

        let gather_ids = ids_from_host(dev, &[1, 1, 1, 0, 0, 0, 1, 1, 1]);
        let gathered = gather(dev, &input, 0, &gather_ids, &[3, 3]).unwrap();
        let gather_requirements = gather_requirements(
            &input.layout,
            input.dtype,
            0,
            &gather_ids.layout,
            gather_ids.dtype,
            &[3, 3],
        )
        .unwrap();
        let gathered_into = allocate_requirement(dev, gather_requirements.output.as_ref().unwrap());
        gather_into(dev, &input, 0, &gather_ids, &[3, 3], &gathered_into, None).unwrap();

        let table = MetalTensor::zeros(dev, vec![4, 2], DType::F32);
        let source = MetalTensor::from_f32(dev, vec![1., 2., 3., 4.], vec![2, 2]);
        let scatter_ids = ids_from_host(dev, &[1, 1, 3, 3]);
        let scattered = scatter_add(dev, &table, 0, &scatter_ids, &source).unwrap();
        let scatter_requirements = scatter_add_requirements(
            &table.layout,
            table.dtype,
            0,
            &scatter_ids.layout,
            scatter_ids.dtype,
            &source.layout,
            source.dtype,
        )
        .unwrap();
        let scattered_into =
            allocate_requirement(dev, scatter_requirements.output.as_ref().unwrap());
        scatter_add_into(
            dev,
            &table,
            0,
            &scatter_ids,
            &source,
            &scattered_into,
            None,
            None,
            None,
        )
        .unwrap();

        let a = MetalTensor::from_f32(dev, vec![1., 2.], vec![1, 2]);
        let b = MetalTensor::from_f32(dev, vec![3., 4.], vec![1, 2]);
        let concatenated = cat(dev, &[&a, &b], 0).unwrap();
        let cat_requirements =
            cat_requirements(&[&a.layout, &b.layout], &[a.dtype, b.dtype], 0).unwrap();
        let concatenated_into =
            allocate_requirement(dev, cat_requirements.output.as_ref().unwrap());
        cat_into(dev, &[&a, &b], 0, &concatenated_into).unwrap();

        let set_destination = MetalTensor::zeros(dev, vec![4, 2], DType::F32);
        let set_destination_into = MetalTensor::zeros(dev, vec![4, 2], DType::F32);
        scatter_set(dev, &set_destination, 0, &[1, 3], &source).unwrap();
        let set_ids = ids_from_host(dev, &[1, 3]);
        scatter_set_into(dev, &set_destination_into, 0, &set_ids, &source, None).unwrap();

        dev.synchronize().unwrap();
        assert_eq!(
            selected.read_f32().unwrap(),
            selected_into.read_f32().unwrap()
        );
        assert_eq!(
            gathered.read_f32().unwrap(),
            gathered_into.read_f32().unwrap()
        );
        assert_eq!(
            scattered.read_f32().unwrap(),
            scattered_into.read_f32().unwrap()
        );
        assert_eq!(
            concatenated.read_f32().unwrap(),
            concatenated_into.read_f32().unwrap()
        );
        assert_eq!(
            set_destination.read_f32().unwrap(),
            set_destination_into.read_f32().unwrap()
        );
    }

    #[test]
    fn requirements_expose_cast_and_staging_storage_exactly() {
        let input = Layout::contiguous(vec![4, 2]);
        let ids = Layout::new(vec![2, 2], vec![1, 2], 0);
        let source = Layout::contiguous(vec![2, 2]);
        let requirements = scatter_add_requirements(
            &input,
            DType::BF16,
            0,
            &ids,
            DType::I64,
            &source,
            DType::BF16,
        )
        .unwrap();
        assert_eq!(requirements.ids.as_ref().unwrap().bytes().unwrap(), 16);
        assert_eq!(
            requirements.accumulator.as_ref().unwrap().bytes().unwrap(),
            32
        );
        assert_eq!(
            requirements.source_cast.as_ref().unwrap().bytes().unwrap(),
            16
        );
        assert_eq!(requirements.scratch_bytes().unwrap(), 64);
        assert_eq!(requirements.scratch_bytes, 64);
        assert_eq!(requirements.output_bytes, 16);
        assert_eq!(requirements.staging_bytes, 0);
        assert_eq!(requirements.pipeline_count, 5);

        let staging =
            scatter_set_staging_requirements(&input, DType::F32, 0, 2, &source, DType::F32)
                .unwrap();
        assert_eq!(staging.staging_bytes().unwrap(), 8);
        assert_eq!(staging.staging_bytes, 8);
    }

    #[test]
    fn into_uses_no_device_allocation_after_warm() {
        let dev = MetalDevice::new(0).unwrap();
        let input =
            MetalTensor::from_f32(&dev, (0..6).map(|value| value as f32).collect(), vec![2, 3]);
        let ids = ids_from_host(&dev, &[2, 0]);
        let requirements =
            index_select_requirements(&input.layout, input.dtype, 1, &ids.layout, ids.dtype)
                .unwrap();
        let destination = allocate_requirement(&dev, requirements.output.as_ref().unwrap());
        let ids_scratch = allocate_optional(&dev, &requirements.ids);
        let error = index_select_into(&dev, &input, 1, &ids, &destination, ids_scratch.as_ref())
            .unwrap_err();
        assert!(error.contains("not warm"), "{error}");
        compile_index_select_layout_exact(
            &dev,
            &input.layout,
            input.dtype,
            1,
            &ids.layout,
            ids.dtype,
        )
        .unwrap();
        let _dispatch_guard = dev.begin_executable_dispatch().unwrap();
        let result = index_select_into(&dev, &input, 1, &ids, &destination, ids_scratch.as_ref());
        result.unwrap();
        dev.synchronize().unwrap();
    }
}

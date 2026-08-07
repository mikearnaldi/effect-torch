//! Metal bridges for the backend-neutral fusion IR.

use super::err::Res;
use super::value::Value;
use crate::device::MetalDevice;
use crate::run::MetalTensor;
pub use effect_torch_compiler::{adamw_exprs, contiguous_strides, sgd_exprs, Expr, ReduceOp};
use effect_torch_graph::Device;
use effect_torch_runtime::{DType, Layout};

fn fusion_phase_nanos(phase: usize, nanos: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static PHASES: [AtomicU64; 4] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    static CALLS: AtomicU64 = AtomicU64::new(0);
    const NAMES: [&str; 4] = ["hash", "pipeline", "buffers", "dispatch"];
    PHASES[phase].fetch_add(nanos, Ordering::Relaxed);
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if phase == 3 && calls.is_multiple_of(2048) {
        let totals: Vec<u64> = PHASES
            .iter()
            .map(|value| value.load(Ordering::Relaxed))
            .collect();
        eprintln!(
            "[fusion-timing] {calls} calls: {}",
            NAMES
                .iter()
                .zip(totals.iter())
                .map(|(name, total)| format!("{name} {:.1}ms", *total as f64 / 1e6))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}

fn run_metal(
    exprs: &[Expr],
    inputs: &[Value],
    lane_strides: &[Vec<usize>],
    scalars: &[Value],
    n: usize,
    shape: &[usize],
) -> Res<Vec<Value>> {
    let scalar_slots = usize::from(!scalars.is_empty());
    if inputs.len() + exprs.len() + scalar_slots > 31 {
        return Err(format!(
            "fusion multi: {} buffer arguments exceed Metal's limit of 31",
            inputs.len() + exprs.len() + scalar_slots
        ));
    }
    let phase_timing = std::env::var_os("EFFECT_TORCH_FUSION_TIMING").is_some();
    let started = std::time::Instant::now();
    let native: Vec<MetalTensor> = inputs
        .iter()
        .map(|value| value.as_metal().cloned())
        .collect::<Res<_>>()?;
    let references: Vec<&MetalTensor> = native.iter().collect();
    let device = MetalDevice::get();
    let metal_scalars: Option<Vec<MetalTensor>> = scalars
        .iter()
        .map(|value| {
            let tensor = value.as_metal().ok()?;
            (tensor.dtype == DType::F32).then(|| tensor.clone())
        })
        .collect();
    let (scalar_buffer, scalar_count) = if scalars.is_empty() {
        (None, 0)
    } else if let Some(tensors) = metal_scalars {
        let views: Vec<MetalTensor> = tensors
            .iter()
            .map(|tensor| MetalTensor {
                buffer: tensor.buffer.clone(),
                layout: Layout::contiguous(vec![1]),
                dtype: DType::F32,
            })
            .collect();
        let references: Vec<&MetalTensor> = views.iter().collect();
        let packed = crate::indexing::cat(device, &references, 0)?;
        (Some(packed.buffer.clone()), scalars.len())
    } else {
        device.synchronize()?;
        let values: Vec<f32> = scalars
            .iter()
            .map(|value| value.to_f32_vec().map(|values| values[0]))
            .collect::<Res<_>>()?;
        (Some(device.alloc_with_data(&values)), values.len())
    };
    if phase_timing {
        fusion_phase_nanos(0, started.elapsed().as_nanos() as u64);
    }
    crate::run::run_elementwise_scalar_buf(
        device,
        exprs,
        &references,
        lane_strides,
        scalar_buffer.as_deref(),
        scalar_count,
        n,
        shape,
    )
    .map(|outputs| outputs.into_iter().map(Value).collect())
}

#[allow(clippy::too_many_arguments)]
fn run_reduce_metal(
    op: ReduceOp,
    expr: &Expr,
    inputs: &[Value],
    lane_strides: &[Vec<usize>],
    input_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    output_shape: &[usize],
) -> Res<Value> {
    if inputs.len() + 1 > 31 {
        return Err(format!(
            "fusion reduce: {} buffer arguments exceed Metal's limit of 31",
            inputs.len() + 1
        ));
    }
    let native: Vec<MetalTensor> = inputs
        .iter()
        .map(|value| value.as_metal().cloned())
        .collect::<Res<_>>()?;
    let references: Vec<&MetalTensor> = native.iter().collect();
    crate::run::run_reduce(
        MetalDevice::get(),
        op,
        expr,
        &references,
        lane_strides,
        input_shape,
        dims,
        keepdims,
        output_shape,
    )
    .map(Value)
}

pub fn is_supported(device: &Device, dtype: DType) -> bool {
    device.is_metal() && matches!(dtype, DType::F32 | DType::BF16)
}

pub struct GroupPlan {
    pub exprs: Vec<Expr>,
    pub strides: Vec<Vec<usize>>,
    pub key: u64,
    pub num_scalars: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_group_plan(
    params_len: usize,
    shape: &[usize],
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> std::sync::Arc<GroupPlan> {
    type Key = (usize, Vec<usize>, u64, u64, u64, u64);
    static PLANS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<Key, std::sync::Arc<GroupPlan>>>,
    > = std::sync::OnceLock::new();
    let key = (
        params_len,
        shape.to_vec(),
        beta1.to_bits(),
        beta2.to_bits(),
        eps.to_bits(),
        weight_decay.to_bits(),
    );
    let cache = PLANS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(plan) = cache.lock().unwrap().get(&key) {
        return plan.clone();
    }
    let base = adamw_exprs(beta1, beta2, eps, weight_decay);
    let mut exprs = Vec::with_capacity(params_len * 3);
    for index in 0..params_len {
        let remap: std::collections::HashMap<u32, u32> = (0u32..4)
            .map(|lane| (lane, (index * 4) as u32 + lane))
            .collect();
        for expr in &base {
            exprs.push(expr.remap_lanes(&remap));
        }
    }
    let strides = vec![contiguous_strides(shape); params_len * 4];
    let element_count = shape.iter().product();
    let plan = std::sync::Arc::new(GroupPlan {
        key: crate::run::elementwise_key(&exprs, &strides, shape, element_count, 3, DType::F32),
        exprs,
        strides,
        num_scalars: 3,
    });
    cache.lock().unwrap().insert(key, plan.clone());
    plan
}

pub fn pack_scalars_metal(scalars: &[Value]) -> Res<Value> {
    let device = MetalDevice::get();
    let mut views = Vec::with_capacity(scalars.len());
    for value in scalars {
        let tensor = value.as_metal()?;
        let tensor = if tensor.layout.is_contiguous() && tensor.layout.offset() == 0 {
            tensor.clone()
        } else {
            crate::kernels::strided_copy(device, tensor)?
        };
        let tensor = if tensor.dtype == DType::F32 {
            tensor
        } else {
            crate::kernels::cast(device, &tensor, DType::F32)?
        };
        views.push(MetalTensor {
            buffer: tensor.buffer.clone(),
            layout: Layout::contiguous(vec![1]),
            dtype: DType::F32,
        });
    }
    let references: Vec<&MetalTensor> = views.iter().collect();
    crate::indexing::cat(device, &references, 0).map(Value)
}

pub fn run_group_metal(
    plan: &GroupPlan,
    inputs: &[Value],
    packed_scalars: &Value,
    shape: &[usize],
) -> Res<Vec<Value>> {
    if inputs.len() + plan.exprs.len() + 1 > 31 {
        return Err(format!(
            "fusion group: {} buffer arguments exceed Metal's limit of 31",
            inputs.len() + plan.exprs.len() + 1
        ));
    }
    let device = MetalDevice::get();
    let mut owned = Vec::with_capacity(inputs.len());
    for value in inputs {
        let tensor = value.as_metal()?;
        owned.push(if tensor.layout.is_contiguous() {
            tensor.clone()
        } else {
            crate::kernels::strided_copy(device, tensor)?
        });
    }
    let references: Vec<&MetalTensor> = owned.iter().collect();
    let packed = packed_scalars.as_metal()?;
    crate::run::run_elementwise_prekeyed(
        device,
        plan.key,
        &plan.exprs,
        &references,
        &plan.strides,
        Some(&packed.buffer),
        plan.num_scalars,
        shape.iter().product(),
        shape,
    )
    .map(|outputs| outputs.into_iter().map(Value).collect())
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    exprs: &[Expr],
    inputs: &[Value],
    strides: Option<&[Vec<usize>]>,
    scalars: &[Value],
    n: usize,
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Res<Vec<Value>> {
    if !is_supported(device, dtype) {
        return Err(format!(
            "fusion: unsupported device/dtype {device:?} {dtype:?}"
        ));
    }
    if let Some(strides) = strides {
        if strides.len() != inputs.len() {
            return Err(format!(
                "fusion: got {} stride entries for {} inputs",
                strides.len(),
                inputs.len()
            ));
        }
    }
    let mut owned = Vec::with_capacity(inputs.len() + scalars.len());
    for value in inputs.iter().chain(scalars) {
        let tensor = value.as_metal()?;
        owned.push(Value(if tensor.layout.is_contiguous() {
            tensor.clone()
        } else {
            crate::kernels::strided_copy(MetalDevice::get(), tensor)?
        }));
    }
    for scalar in scalars {
        if scalar.numel() != 1 {
            return Err(format!(
                "fusion: scalar lanes must have exactly one element, got {}",
                scalar.numel()
            ));
        }
    }
    if n == 0 {
        return Ok(exprs
            .iter()
            .map(|_| {
                Value(MetalTensor::zeros(
                    MetalDevice::get(),
                    shape.to_vec(),
                    dtype,
                ))
            })
            .collect());
    }
    let inputs = &owned[..inputs.len()];
    let scalars = &owned[inputs.len()..];
    let contiguous;
    let lane_strides = match strides {
        Some(strides) => strides,
        None => {
            contiguous = vec![contiguous_strides(shape); inputs.len()];
            &contiguous
        }
    };
    run_metal(exprs, inputs, lane_strides, scalars, n, shape)
}

#[allow(clippy::too_many_arguments)]
pub fn run_reduce(
    op: ReduceOp,
    expr: &Expr,
    inputs: &[Value],
    strides: &[Vec<usize>],
    input_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    output_shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Res<Value> {
    if !is_supported(device, dtype) {
        return Err(format!(
            "fusion: unsupported device/dtype {device:?} {dtype:?}"
        ));
    }
    if strides.len() != inputs.len() {
        return Err(format!(
            "fusion: got {} stride entries for {} inputs",
            strides.len(),
            inputs.len()
        ));
    }
    let mut owned = Vec::with_capacity(inputs.len());
    for value in inputs {
        let tensor = value.as_metal()?;
        owned.push(Value(if tensor.layout.is_contiguous() {
            tensor.clone()
        } else {
            crate::kernels::strided_copy(MetalDevice::get(), tensor)?
        }));
    }
    if output_shape.iter().product::<usize>() == 0 {
        return Ok(Value(MetalTensor::zeros(
            MetalDevice::get(),
            output_shape.to_vec(),
            dtype,
        )));
    }
    run_reduce_metal(
        op,
        expr,
        &owned,
        strides,
        input_shape,
        dims,
        keepdims,
        output_shape,
    )
}

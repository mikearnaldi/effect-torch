use crate::buffer::CudaBuffer;
use crate::device::{CUDA_TOP_K_BLOCKS, CUDA_TOP_K_LIMIT};
use crate::CudaDevice;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use effect_torch_graph::{Device, LeafValue};
use effect_torch_runtime::{DType, GgmlKQuant, MAX_SAMPLING_VOCABULARY};
use half::bf16;
use std::any::Any;
use std::sync::{Arc, Mutex};

type Bf16Weight = (GgmlKQuant, u32, u32, Arc<CudaSlice<bf16>>);

pub(crate) fn element_count(shape: &[usize]) -> Result<usize, String> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| "CUDA tensor element count overflowed usize".to_string())
    })
}

/// A contiguous device allocation and its logical metadata.
#[derive(Clone)]
pub struct CudaValue {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) buffer: Arc<CudaBuffer<u8>>,
    packed_buffer: Option<Arc<CudaBuffer<u8>>>,
    bf16_weight: Option<Arc<Mutex<Option<Bf16Weight>>>>,
    i64_buffer: Option<Arc<CudaBuffer<i64>>>,
    shape: Arc<[usize]>,
    dtype: DType,
}

impl CudaValue {
    pub(crate) fn from_host(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        dtype: DType,
        values: &[f64],
    ) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if values.len() != expected {
            return Err(format!(
                "CUDA host value count {} does not match shape element count {expected}",
                values.len()
            ));
        }
        let buffer = if dtype == DType::F32 {
            let values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
            let buffer = CudaBuffer::from_slice(
                device
                    .stream
                    .clone_htod(&values)
                    .map_err(|error| error.to_string())?,
            );
            buffer.cast::<u8>(values.len() * std::mem::size_of::<f32>())?
        } else {
            let buffer = CudaBuffer::from_slice(
                device
                    .stream
                    .clone_htod(values)
                    .map_err(|error| error.to_string())?,
            );
            buffer.cast::<u8>(values.len() * std::mem::size_of::<f64>())?
        };
        let i64_buffer = if dtype == DType::I64 {
            Some(Arc::new(
                device
                    .stream
                    .clone_htod(&values.iter().map(|value| *value as i64).collect::<Vec<_>>())
                    .map_err(|error| error.to_string())?,
            ))
        } else {
            None
        };
        Ok(Self {
            device,
            buffer: Arc::new(buffer),
            packed_buffer: None,
            bf16_weight: None,
            i64_buffer: i64_buffer.map(|buffer| Arc::new(CudaBuffer::from_arc_slice(buffer))),
            shape: shape.into(),
            dtype,
        })
    }

    pub(crate) fn from_f32_host(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        values: &[f32],
    ) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if values.len() != expected {
            return Err(format!(
                "CUDA f32 value count {} does not match shape element count {expected}",
                values.len()
            ));
        }
        let buffer = CudaBuffer::from_slice(
            device
                .stream
                .clone_htod(values)
                .map_err(|error| error.to_string())?,
        );
        Ok(Self {
            device,
            buffer: Arc::new(buffer.cast::<u8>(values.len() * std::mem::size_of::<f32>())?),
            packed_buffer: None,
            bf16_weight: None,
            i64_buffer: None,
            shape: shape.into(),
            dtype: DType::F32,
        })
    }

    pub(crate) fn from_packed_host(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        values: &[u8],
    ) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if values.len() != expected {
            return Err(format!(
                "CUDA packed byte count {} does not match shape element count {expected}",
                values.len()
            ));
        }
        let buffer = device
            .stream
            .clone_htod(values)
            .map_err(|error| error.to_string())?;
        let dense = CudaBuffer::from_slice(buffer).cast::<u8>(values.len())?;
        Ok(Self {
            device,
            buffer: Arc::new(dense.clone()),
            packed_buffer: Some(Arc::new(dense)),
            bf16_weight: Some(Arc::new(Mutex::new(None))),
            i64_buffer: None,
            shape: shape.into(),
            dtype: DType::U8,
        })
    }

    pub(crate) fn from_i64_host(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        values: &[i64],
    ) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if values.len() != expected {
            return Err(format!(
                "CUDA i64 value count {} does not match shape element count {expected}",
                values.len()
            ));
        }
        let buffer = CudaBuffer::from_slice(
            device
                .stream
                .clone_htod(&values.iter().map(|value| *value as f64).collect::<Vec<_>>())
                .map_err(|error| error.to_string())?,
        );
        let i64_buffer = device
            .stream
            .clone_htod(values)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            device,
            buffer: Arc::new(buffer.cast::<u8>(values.len() * std::mem::size_of::<f64>())?),
            packed_buffer: None,
            bf16_weight: None,
            i64_buffer: Some(Arc::new(CudaBuffer::from_slice(i64_buffer))),
            shape: shape.into(),
            dtype: DType::I64,
        })
    }

    pub(crate) fn write_host(&self, values: &[f64]) -> Result<(), String> {
        let expected = element_count(&self.shape)?;
        if values.len() != expected {
            return Err(format!(
                "CUDA host value count {} does not match shape element count {expected}",
                values.len()
            ));
        }
        if self.dtype == DType::F32 {
            let values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
            let mut buffer = self.buffer.cast::<f32>(expected)?;
            self.device
                .stream
                .memcpy_htod(&values, &mut buffer)
                .map_err(|error| error.to_string())
        } else {
            let mut buffer = self.buffer.cast::<f64>(expected)?;
            self.device
                .stream
                .memcpy_htod(values, &mut buffer)
                .map_err(|error| error.to_string())
        }
    }

    pub(crate) fn write_i64_host(&self, values: &[i64]) -> Result<(), String> {
        let expected = element_count(&self.shape)?;
        if values.len() != expected {
            return Err(format!(
                "CUDA i64 value count {} does not match shape element count {expected}",
                values.len()
            ));
        }
        let mut buffer = self.buffer.cast::<f64>(expected)?;
        self.device
            .stream
            .memcpy_htod(
                &values.iter().map(|value| *value as f64).collect::<Vec<_>>(),
                &mut buffer,
            )
            .map_err(|error| error.to_string())?;
        let mut sidecar = self
            .i64_buffer
            .as_ref()
            .ok_or_else(|| "CUDA value has no exact i64 sidecar".to_string())?
            .as_ref()
            .clone();
        self.device
            .stream
            .memcpy_htod(values, &mut sidecar)
            .map_err(|error| error.to_string())
    }

    pub(crate) fn from_buffer(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        dtype: DType,
        buffer: CudaSlice<f64>,
    ) -> Result<Self, String> {
        let len = buffer.len();
        let buffer = CudaBuffer::from_slice(buffer);
        Ok(Self {
            device,
            buffer: Arc::new(buffer.cast::<u8>(len * std::mem::size_of::<f64>())?),
            packed_buffer: None,
            bf16_weight: None,
            i64_buffer: None,
            shape: shape.into(),
            dtype,
        })
    }

    pub(crate) fn from_f32_buffer(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        buffer: CudaSlice<f32>,
    ) -> Result<Self, String> {
        let len = buffer.len();
        let buffer = CudaBuffer::from_slice(buffer);
        Ok(Self {
            device,
            buffer: Arc::new(buffer.cast::<u8>(len * std::mem::size_of::<f32>())?),
            packed_buffer: None,
            bf16_weight: None,
            i64_buffer: None,
            shape: shape.into(),
            dtype: DType::F32,
        })
    }

    pub(crate) fn from_planned_buffers(
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        dtype: DType,
        buffer: CudaBuffer<u8>,
        i64_buffer: Option<CudaBuffer<i64>>,
    ) -> Self {
        Self {
            device,
            buffer: Arc::new(buffer),
            packed_buffer: None,
            bf16_weight: None,
            i64_buffer: i64_buffer.map(Arc::new),
            shape: shape.into(),
            dtype,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ordinal(&self) -> u32 {
        self.device.ordinal
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn storage_bytes(&self) -> Result<usize, String> {
        if let Some(packed) = &self.packed_buffer {
            return Ok(packed.len());
        }
        let dense = self.buffer.len();
        let exact_i64 = self
            .i64_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.len())
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or_else(|| "CUDA value storage byte size overflowed usize".to_string())?;
        dense
            .checked_add(exact_i64)
            .ok_or_else(|| "CUDA value storage byte size overflowed usize".to_string())
    }

    pub(crate) fn with_shape(&self, shape: Vec<usize>) -> Self {
        Self {
            device: self.device.clone(),
            buffer: self.buffer.clone(),
            packed_buffer: self.packed_buffer.clone(),
            bf16_weight: self.bf16_weight.clone(),
            i64_buffer: self.i64_buffer.clone(),
            shape: shape.into(),
            dtype: self.dtype,
        }
    }

    pub(crate) fn storage_address(&self) -> u64 {
        self.buffer.address()
    }

    pub(crate) fn packed_buffer(&self) -> Result<&CudaBuffer<u8>, String> {
        self.packed_buffer
            .as_deref()
            .ok_or_else(|| "CUDA value does not have packed byte storage".to_string())
    }

    pub(crate) fn bf16_weight(
        &self,
        codec: GgmlKQuant,
        rows: u32,
        columns: u32,
    ) -> Result<Arc<CudaSlice<bf16>>, String> {
        let cache = self
            .bf16_weight
            .as_ref()
            .ok_or_else(|| "CUDA value does not have a BF16 weight cache".to_string())?;
        let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
        if let Some((cached_codec, cached_rows, cached_columns, weight)) = &*cache {
            if (*cached_codec, *cached_rows, *cached_columns) != (codec, rows, columns) {
                return Err("CUDA packed weight was reused with incompatible metadata".to_string());
            }
            return Ok(Arc::clone(weight));
        }
        let weight = Arc::new(self.device.dequantize_weight_bf16(
            self.packed_buffer()?,
            codec,
            rows,
            columns,
        )?);
        *cache = Some((codec, rows, columns, Arc::clone(&weight)));
        Ok(weight)
    }

    pub(crate) fn f32_buffer(&self) -> Result<CudaBuffer<f32>, String> {
        if self.dtype != DType::F32 {
            return Err(format!("CUDA value is {}, not f32", self.dtype.name()));
        }
        self.buffer.cast(element_count(&self.shape)?)
    }

    pub(crate) fn f64_buffer(&self) -> Result<CudaBuffer<f64>, String> {
        if self.dtype == DType::F32 || self.dtype == DType::U8 {
            return Err(format!(
                "CUDA value is {}, not f64-backed",
                self.dtype.name()
            ));
        }
        self.buffer.cast(element_count(&self.shape)?)
    }

    pub(crate) fn i64_buffer(&self) -> Result<&CudaBuffer<i64>, String> {
        self.i64_buffer
            .as_deref()
            .ok_or_else(|| "CUDA value does not have i64 storage".to_string())
    }

    pub(crate) fn has_i64_buffer(&self) -> bool {
        self.i64_buffer.is_some()
    }

    pub(crate) fn greedy_argmax(&self) -> Result<u32, String> {
        let logits = self.f32_buffer()?;
        let len = element_count(&self.shape)?;
        let len = u32::try_from(len)
            .map_err(|_| "CUDA greedy argmax input exceeds u32 indexing".to_string())?;
        if len == 0 {
            return Err("sample: logits must not be empty".to_string());
        }
        let mut output = self
            .device
            .greedy_argmax_output
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut launch = self
            .device
            .stream
            .launch_builder(&self.device.f32.greedy_argmax);
        launch.arg(&logits);
        launch.arg(&len);
        launch.arg(&mut *output);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| error.to_string())?;
        let result = self
            .device
            .stream
            .clone_dtoh(&*output)
            .map_err(|error| error.to_string())?;
        if result[1] != u32::MAX {
            return Err(format!("sample: logit {} is not finite", result[1]));
        }
        Ok(result[0])
    }

    pub(crate) fn topk(&self, k: usize) -> Result<(Vec<f64>, Vec<u32>), String> {
        let logits = self.f32_buffer()?;
        let len = element_count(&self.shape)?;
        if len == 0 {
            return Err("sample: logits must be non-empty".to_string());
        }
        if len > MAX_SAMPLING_VOCABULARY {
            return Err(format!(
                "sample: vocabulary {len} exceeds limit {MAX_SAMPLING_VOCABULARY}"
            ));
        }
        if k == 0 || k > len {
            return Err(format!("sample: topK must be in [1, {len}], got {k}"));
        }
        if k > CUDA_TOP_K_LIMIT {
            return Err(format!("CUDA top-k exceeds limit {CUDA_TOP_K_LIMIT}"));
        }
        let len =
            u32::try_from(len).map_err(|_| "CUDA top-k input exceeds u32 indexing".to_string())?;
        let k = u32::try_from(k).expect("CUDA top-k limit fits u32");
        let mut output = self
            .device
            .topk_output
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut launch = self.device.stream.launch_builder(&self.device.f32.topk);
        launch.arg(&logits);
        launch.arg(&len);
        launch.arg(&k);
        launch.arg(&mut *output);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (CUDA_TOP_K_BLOCKS as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| error.to_string())?;
        let result = self
            .device
            .stream
            .clone_dtoh(&*output)
            .map_err(|error| error.to_string())?;
        let output_count = CUDA_TOP_K_BLOCKS * CUDA_TOP_K_LIMIT;
        let invalid = result[2 * output_count..]
            .iter()
            .copied()
            .filter(|index| *index >= 0.0)
            .min_by(f32::total_cmp);
        if let Some(invalid) = invalid {
            return Err(format!("sample: logit {} is not finite", invalid as u32));
        }
        let k = k as usize;
        let mut values = Vec::with_capacity(CUDA_TOP_K_BLOCKS * k);
        let mut tokens = Vec::with_capacity(CUDA_TOP_K_BLOCKS * k);
        for block in 0..CUDA_TOP_K_BLOCKS {
            let offset = block * CUDA_TOP_K_LIMIT;
            values.extend(result[offset..offset + k].iter().copied().map(f64::from));
            tokens.extend(
                result[output_count + offset..output_count + offset + k]
                    .iter()
                    .map(|token| *token as u32),
            );
        }
        Ok((values, tokens))
    }

    pub fn readback_i64(&self) -> Result<Vec<i64>, String> {
        if let Some(buffer) = &self.i64_buffer {
            return self
                .device
                .stream
                .clone_dtoh(buffer.as_ref())
                .map_err(|error| error.to_string());
        }
        self.readback()
            .map(|values| values.into_iter().map(|value| value as i64).collect())
    }

    pub fn readback(&self) -> Result<Vec<f64>, String> {
        if let Some(buffer) = &self.packed_buffer {
            return self
                .device
                .stream
                .clone_dtoh(buffer.as_ref())
                .map(|values| values.into_iter().map(f64::from).collect())
                .map_err(|error| error.to_string());
        }
        let len = element_count(&self.shape)?;
        if self.dtype == DType::F32 {
            return self
                .device
                .stream
                .clone_dtoh(&self.buffer.cast::<f32>(len)?)
                .map(|values| values.into_iter().map(f64::from).collect())
                .map_err(|error| error.to_string());
        }
        self.device
            .stream
            .clone_dtoh(&self.buffer.cast::<f64>(len)?)
            .map_err(|error| error.to_string())
    }
}

impl LeafValue for CudaValue {
    fn shape(&self) -> Vec<usize> {
        self.shape.to_vec()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        Device::Cuda(self.device.ordinal)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::element_count;

    #[test]
    fn zero_extent_short_circuits_overflow() {
        assert_eq!(element_count(&[usize::MAX, usize::MAX, 0]), Ok(0));
    }

    #[test]
    fn rejects_nonzero_shape_overflow() {
        assert!(element_count(&[usize::MAX, 2]).is_err());
    }
}

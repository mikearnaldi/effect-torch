//! Diagnostic upload + existing chain + packed readback, preserving every fence.
use super::{chain96, literal89};
use crate::{CudaDevice, CudaExecutable, CudaValue};
use effect_torch_graph::Node;
use effect_torch_runtime::CancellationFlag;
use std::sync::Arc;

pub(super) fn execute(
    device: Arc<CudaDevice>,
    upload: Arc<Node>,
    bindings: Vec<CudaValue>,
    body: impl FnOnce(&[CudaValue]) -> Result<Vec<CudaValue>, String>,
    head: Option<&CudaExecutable>,
    sampler: &CudaExecutable,
    temperature: f64,
    cancelled: &CancellationFlag,
) -> Result<(CudaValue, Vec<u8>), String> {
    // literal89 owns its upload source, gate and completion fence through return.
    let mut uploaded = literal89::materialize(device, &[upload], cancelled)?;
    uploaded.extend(bindings);
    let outputs = chain96::execute(|| body(&uploaded), head, sampler, temperature, cancelled)?;
    if cancelled.is_cancelled() {
        return Err("operation aborted".into());
    }
    let statistics = outputs
        .get(1)
        .ok_or("chain97: missing packed statistics")?
        .read_storage_bytes()?;
    if cancelled.is_cancelled() {
        return Err("operation aborted".into());
    }
    let feedback = outputs
        .into_iter()
        .next()
        .ok_or("chain97: missing feedback")?;
    Ok((feedback, statistics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_with_options;
    use effect_torch_compiler::CompileOptions;
    use effect_torch_graph::{Device, NodeKind};
    use effect_torch_runtime::{DType, StorageMetadata};

    #[test]
    #[ignore = "requires CUDA; chain97 upload/readback RNG, retention and cancellation"]
    fn chain97_hardware_upload_readback_cancellation() {
        let node = |kind| Node::new(kind).unwrap();
        let device = CudaDevice::get(0).unwrap();
        let tokens = node(NodeKind::Input {
            slot: 0,
            shape: vec![1, 16],
            dtype: DType::U32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        });
        let cast = node(NodeKind::Cast {
            a: tokens,
            dtype: DType::F32,
        });
        let body = compile_with_options(vec![cast], 0, CompileOptions::default()).unwrap();
        let x = node(NodeKind::Input {
            slot: 0,
            shape: vec![1, 16],
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        });
        let temp = node(NodeKind::ScalarInput {
            slot: 1,
            dtype: DType::F32,
            device: Device::Cuda(0),
        });
        let noise = node(NodeKind::Uniform {
            lo: 0.,
            hi: 1.,
            shape: vec![1, 16],
            dtype: DType::F32,
            device: Device::Cuda(0),
        });
        let sum = node(NodeKind::Add { a: x, b: noise });
        let out = node(NodeKind::Mul { a: sum, b: temp });
        let options = || CompileOptions {
            random_seed: Some(73),
            ..CompileOptions::default()
        };
        let sampler = compile_with_options(vec![out.clone(), out.clone()], 0, options()).unwrap();
        let control = compile_with_options(vec![out.clone(), out], 0, options()).unwrap();
        let bytes = (0u32..16).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        let upload = node(NodeKind::FromBytes {
            data: bytes.clone(),
            shape: vec![1, 16],
            dtype: DType::U32,
            device: Device::Cuda(0),
        });
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        assert!(execute(
            device.clone(),
            upload.clone(),
            vec![],
            |_| panic!("pre-cancelled body"),
            None,
            &sampler,
            0.5,
            &cancelled
        )
        .is_err());
        let during = CancellationFlag::new();
        assert!(execute(
            device.clone(),
            upload.clone(),
            vec![],
            |values| {
                during.cancel();
                Ok(values.to_vec())
            },
            None,
            &sampler,
            0.5,
            &during
        )
        .is_err());
        let mut retained = Vec::new();
        for _ in 0..3 {
            let flag = CancellationFlag::new();
            let (feedback, statistics) = execute(
                device.clone(),
                upload.clone(),
                vec![],
                |values| body.execute(values, &[], &flag),
                None,
                &sampler,
                0.5,
                &flag,
            )
            .unwrap();
            let concrete =
                literal89::materialize(device.clone(), &[upload.clone()], &flag).unwrap();
            let hidden = body.execute(&concrete, &[], &flag).unwrap();
            let reference = control.execute(&hidden, &[0.5], &flag).unwrap();
            assert_eq!(statistics, reference[1].read_storage_bytes().unwrap());
            assert_eq!(feedback.read_storage_bytes().unwrap(), statistics);
            retained.push((feedback, statistics));
        }
        drop(upload);
        for (feedback, expected) in retained {
            assert_eq!(feedback.read_storage_bytes().unwrap(), expected);
        }
    }
}

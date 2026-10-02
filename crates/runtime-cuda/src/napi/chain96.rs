//! Diagnostic sequencing only: existing executors retain their fences and RNGs.
use crate::{CudaExecutable, CudaValue};
use effect_torch_runtime::CancellationFlag;

pub(super) fn execute(
    body: impl FnOnce() -> Result<Vec<CudaValue>, String>,
    head: Option<&CudaExecutable>,
    sampler: &CudaExecutable,
    temperature: f64,
    cancelled: &CancellationFlag,
) -> Result<Vec<CudaValue>, String> {
    if cancelled.is_cancelled() {
        return Err("operation aborted".into());
    }
    let hidden = body()?;
    if cancelled.is_cancelled() {
        return Err("operation aborted".into());
    }
    let logits = match head {
        Some(head) => {
            let logits = head.execute(&hidden, &[], cancelled)?;
            drop(hidden);
            logits
        }
        None => hidden,
    };
    if cancelled.is_cancelled() {
        return Err("operation aborted".into());
    }
    sampler.execute(&logits, &[temperature], cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile_with_options, CudaDevice};
    use effect_torch_compiler::CompileOptions;
    use effect_torch_graph::{Device, Node, NodeKind};
    use effect_torch_runtime::{DType, StorageMetadata};

    #[test]
    #[ignore = "requires CUDA; chain96 seeded progression, retention and cancellation"]
    fn chain96_hardware_two_three_stages_rng_and_cancellation() {
        let node = |kind| Node::new(kind).unwrap();
        let input = node(NodeKind::Input {
            slot: 0,
            shape: vec![16],
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        });
        let scalar = node(NodeKind::ScalarInput {
            slot: 1,
            dtype: DType::F32,
            device: Device::Cuda(0),
        });
        let noise = node(NodeKind::Uniform {
            lo: 0.,
            hi: 1.,
            shape: vec![16],
            dtype: DType::F32,
            device: Device::Cuda(0),
        });
        let sum = node(NodeKind::Add {
            a: input.clone(),
            b: noise,
        });
        let result = node(NodeKind::Mul { a: sum, b: scalar });
        let options = || CompileOptions {
            random_seed: Some(12345),
            ..CompileOptions::default()
        };
        let chain_sampler =
            compile_with_options(vec![result.clone(), result.clone()], 0, options()).unwrap();
        let separate_sampler =
            compile_with_options(vec![result.clone(), result], 0, options()).unwrap();
        let head = compile_with_options(vec![input], 0, CompileOptions::default()).unwrap();
        assert!(head.chain96_successor(&[]));
        assert!(!head.chain96_successor(&[DType::F32]));
        assert!(chain_sampler.chain96_successor(&[DType::F32]));
        assert!(!chain_sampler.chain96_successor(&[]));
        assert!(!chain_sampler.chain96_successor(&[DType::F64]));
        let device = CudaDevice::get(0).unwrap();
        let bytes = (0..16)
            .flat_map(|v| (v as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let borrowed = CudaValue::from_dense_bytes(device, vec![16], DType::F32, &bytes).unwrap();
        let mut retained = Vec::new();
        for with_head in [false, true, false, true] {
            let cancelled = CancellationFlag::new();
            let actual = execute(
                || Ok(vec![borrowed.clone()]),
                with_head.then_some(&head),
                &chain_sampler,
                0.75,
                &cancelled,
            )
            .unwrap();
            let expected = separate_sampler
                .execute(&[borrowed.clone()], &[0.75], &cancelled)
                .unwrap();
            assert_eq!(
                actual[0].read_storage_bytes().unwrap(),
                expected[0].read_storage_bytes().unwrap()
            );
            retained.push((actual, expected[0].read_storage_bytes().unwrap()));
        }
        // Cancellation before or immediately after the body must not consume a sampler draw.
        let pre = CancellationFlag::new();
        pre.cancel();
        assert!(execute(
            || panic!("pre-cancel must not run body"),
            None,
            &chain_sampler,
            1.,
            &pre
        )
        .is_err());
        let between = CancellationFlag::new();
        assert!(execute(
            || {
                between.cancel();
                Ok(vec![borrowed.clone()])
            },
            Some(&head),
            &chain_sampler,
            1.,
            &between
        )
        .is_err());
        assert!(execute(
            || Err("body failure".into()),
            None,
            &chain_sampler,
            1.,
            &CancellationFlag::new()
        )
        .is_err());
        let actual = execute(
            || Ok(vec![borrowed.clone()]),
            None,
            &chain_sampler,
            1.,
            &CancellationFlag::new(),
        )
        .unwrap();
        let expected = separate_sampler
            .execute(&[borrowed.clone()], &[1.], &CancellationFlag::new())
            .unwrap();
        assert_eq!(
            actual[0].read_storage_bytes().unwrap(),
            expected[0].read_storage_bytes().unwrap()
        );
        drop(actual);
        for (mut values, expected) in retained {
            values.remove(0);
            assert_eq!(values[0].read_storage_bytes().unwrap(), expected);
        }
        assert_eq!(borrowed.read_storage_bytes().unwrap(), bytes);
    }
}

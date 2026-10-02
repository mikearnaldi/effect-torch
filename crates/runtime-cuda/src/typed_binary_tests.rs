use super::*;
use crate::{CudaDevice, CudaValue};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{CancellationFlag, DType, StorageMetadata};
use std::sync::Arc;

fn raw_value(device: &Arc<CudaDevice>, shape: Vec<usize>, dtype: DType, seed: usize) -> CudaValue {
    let count: usize = shape.iter().product();
    let mut bytes = Vec::new();
    for i in 0..count {
        if dtype == DType::BF16 {
            bytes.extend_from_slice(&((i + seed) as u16).to_le_bytes());
        } else {
            let specials = [
                0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc12345, 0xff812345,
                0x3f808000,
            ];
            let bits = if i % 7 == 0 {
                specials[(i / 7 + seed) % specials.len()]
            } else {
                ((i + seed) as u32).wrapping_mul(0x9e3779b9)
            };
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
    }
    CudaValue::from_dense_bytes(device.clone(), shape, dtype, &bytes).unwrap()
}

fn launch_reference_or_candidate(
    name: &str,
    device: &Arc<CudaDevice>,
    bindings: &[CudaValue],
    n: usize,
    op: u32,
    types: [u32; 3],
    mode: u64,
) -> Vec<u8> {
    let dtype = if types[2] == 3 {
        DType::BF16
    } else {
        DType::F32
    };
    let output = CudaValue::from_host(device.clone(), vec![n], dtype, &vec![0.; n]).unwrap();
    let mut args = CudaKernelArgs {
        elements: n as u64,
        operation: op,
        compute_dtype: 1,
        output_dtype: types[2],
        output: output.storage_address(),
        ..Default::default()
    };
    args.inputs[0] = bindings[0].storage_address();
    args.inputs[1] = bindings[1].storage_address();
    args.input_dtypes[..2].copy_from_slice(&types[..2]);
    args.integers[2] = 1;
    args.integers[3] = mode;
    let mut launch = device.stream.launch_builder(device.kernel(name).unwrap());
    launch.arg(&args);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();
    output.read_storage_bytes().unwrap()
}

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_TYPED_BINARY=1"]
fn typed_binary_compiled_exact_raw_bits_scalar_cancel_concurrent_and_retained() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_TYPED_BINARY").as_deref(),
        Ok("1")
    );
    let device = CudaDevice::get(0).unwrap();
    let mut retained = Vec::new();
    for types in [[3, 3, 3], [1, 1, 1], [3, 1, 1]] {
        for mode in [1, 2] {
            for op in 0..6 {
                for n in [1, 31, 257, 65536] {
                    let dtype = |code| if code == 3 { DType::BF16 } else { DType::F32 };
                    let input = |slot, shape, dt| {
                        Node::new(NodeKind::Input {
                            slot,
                            shape,
                            dtype: dt,
                            device: Device::Cuda(0),
                            storage: StorageMetadata::dense(),
                        })
                        .unwrap()
                    };
                    let a = input(0, vec![n], dtype(types[0]));
                    let b = input(1, vec![if mode == 2 { 1 } else { n }], dtype(types[1]));
                    // Mixed semantic arithmetic requires an explicit widening cast.
                    // With optimization disabled this cast materializes F32; the
                    // mixed physical wrapper is checked separately below.
                    let a = if types[0] != types[1] {
                        Node::new(NodeKind::Cast {
                            a,
                            dtype: DType::F32,
                        })
                        .unwrap()
                    } else {
                        a
                    };
                    let node = Node::new(match op {
                        0 => NodeKind::Add { a, b },
                        1 => NodeKind::Sub { a, b },
                        2 => NodeKind::Mul { a, b },
                        3 => NodeKind::Div { a, b },
                        4 => NodeKind::Maximum { a, b },
                        _ => NodeKind::Minimum { a, b },
                    })
                    .unwrap();
                    let executable = Arc::new(
                        crate::compile_with_options(
                            vec![node],
                            0,
                            effect_torch_compiler::CompileOptions {
                                optimize: false,
                                ..Default::default()
                            },
                        )
                        .unwrap(),
                    );
                    // For one element, a [1] RHS is contiguous, not broadcast.
                    let physical_mode = if n == 1 { 1 } else { mode };
                    let expected_name = format!(
                        "et_binary_fixed_{op}_{}_{}_{}_{physical_mode}",
                        types[2], types[1], types[2]
                    );
                    assert!(
                        executable
                            .diagnostics()
                            .instructions
                            .iter()
                            .any(|i| i.kind == expected_name),
                        "missing {expected_name}: {:?}",
                        executable.diagnostics().instructions
                    );
                    let bindings = vec![
                        raw_value(&device, vec![n], dtype(types[0]), 17),
                        raw_value(
                            &device,
                            vec![if mode == 2 { 1 } else { n }],
                            dtype(types[1]),
                            313,
                        ),
                    ];
                    let expected = launch_reference_or_candidate(
                        "et_binary",
                        &device,
                        &bindings,
                        n,
                        op,
                        types,
                        mode,
                    );
                    if types[0] != types[1] {
                        let mixed_name = format!("et_binary_fixed_{op}_3_1_1_{mode}");
                        let mixed = launch_reference_or_candidate(
                            &mixed_name,
                            &device,
                            &bindings,
                            n,
                            op,
                            types,
                            mode,
                        );
                        assert_eq!(mixed, expected, "mixed physical kernel {mixed_name} n={n}");
                    }
                    let cancelled = CancellationFlag::new();
                    cancelled.cancel();
                    assert!(executable.execute(&bindings, &[], &cancelled).is_err());
                    let actual = executable
                        .execute(&bindings, &[], &CancellationFlag::new())
                        .unwrap()
                        .remove(0);
                    assert_eq!(
                        actual.read_storage_bytes().unwrap(),
                        expected,
                        "op={op} types={types:?} mode={mode} n={n}"
                    );
                    if n == 257 && op == 0 {
                        let handles = (0..2)
                            .map(|_| {
                                let exe = executable.clone();
                                let values = bindings.clone();
                                let bytes = expected.clone();
                                std::thread::spawn(move || {
                                    let output = exe
                                        .execute(&values, &[], &CancellationFlag::new())
                                        .unwrap();
                                    assert_eq!(output[0].read_storage_bytes().unwrap(), bytes);
                                })
                            })
                            .collect::<Vec<_>>();
                        for handle in handles {
                            handle.join().unwrap();
                        }
                    }
                    drop(executable);
                    retained.push((actual, expected));
                }
            }
        }
    }
    for (output, bytes) in retained {
        assert_eq!(output.read_storage_bytes().unwrap(), bytes);
    }
}

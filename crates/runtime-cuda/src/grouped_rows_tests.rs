use crate::executable::{dtype_code, CudaKernelArgs};
use crate::{CudaDevice, CudaValue};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use effect_torch_runtime::DType;

#[test]
#[ignore = "requires CUDA and EFFECT_TORCH_CUDA_GROUPED_ROWS_BLOCK=1"]
fn grouped_rows_block_matches_cpu_existing_kernel_and_grid_stride() {
    assert_eq!(
        std::env::var("EFFECT_TORCH_CUDA_GROUPED_ROWS_BLOCK").unwrap(),
        "1"
    );
    let device = CudaDevice::get(0).unwrap();
    let mut retained = Vec::new();
    for rows in [0usize, 1, 33, 129, 2048, 4097] {
        for experts in [3usize, 128, 257] {
            for pattern in 0..4 {
                let ids = (0..rows)
                    .map(|row| match pattern {
                        0 => (row * 113 + 17) % experts,
                        1 => experts - 1,
                        2 => (row / 17) % experts,
                        _ => {
                            if row % 7 == 0 {
                                experts + 1
                            } else {
                                row % experts
                            }
                        }
                    })
                    .collect::<Vec<_>>();
                let mut control = vec![0u32; experts + 2];
                control[0] = 0x8000_0000;
                for &id in &ids {
                    if id < experts {
                        control[id + 2] += 1;
                    } else {
                        control[0] |= 6;
                    }
                }
                let mut total = 0;
                for e in 0..experts {
                    let count = control[e + 2];
                    control[e + 1] = total;
                    total += count;
                }
                control[experts + 1] = total;
                let mut expected = (0..experts)
                    .flat_map(|e| {
                        ids.iter()
                            .enumerate()
                            .filter_map(move |(row, &id)| (id == e).then_some(row as u32))
                    })
                    .collect::<Vec<_>>();
                expected.resize(rows + 16, 0xdead_beef);
                let host = |values: &[u32]| {
                    CudaValue::from_host(
                        device.clone(),
                        vec![values.len()],
                        DType::U32,
                        &values.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    )
                    .unwrap()
                };
                let input = host(&ids.iter().map(|&id| id as u32).collect::<Vec<_>>());
                let offsets = host(&control);
                let old = host(&vec![0xdead_beef; rows + 16]);
                let block = host(&vec![0xdead_beef; rows + 16]);
                let mut args = CudaKernelArgs {
                    elements: rows as u64,
                    output_dtype: dtype_code(DType::U32),
                    ..Default::default()
                };
                args.inputs[0] = input.storage_address();
                args.inputs[1] = offsets.storage_address();
                args.integers[0] = experts as u64;
                for (name, output, grid) in [
                    ("et_grouped_rows", &old, (experts * 32).div_ceil(256)),
                    ("et_grouped_rows_block", &block, experts.min(7)),
                ] {
                    args.output = output.storage_address();
                    let mut launch = device.stream.launch_builder(device.kernel(name).unwrap());
                    launch.arg(&args);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (grid as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .unwrap();
                }
                device.stream.synchronize().unwrap();
                let bytes = expected
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                assert_eq!(
                    old.read_storage_bytes().unwrap(),
                    bytes,
                    "ordinary rows={rows} experts={experts} pattern={pattern}"
                );
                assert_eq!(
                    block.read_storage_bytes().unwrap(),
                    bytes,
                    "block rows={rows} experts={experts} pattern={pattern}"
                );
                assert_eq!(
                    offsets.read_storage_bytes().unwrap(),
                    control
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                );
                retained.push((block, bytes));
            }
        }
    }
    for (value, bytes) in retained {
        assert_eq!(value.read_storage_bytes().unwrap(), bytes);
    }
}

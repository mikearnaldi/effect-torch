use super::{dtype_code, grouped_copy_launch, CudaKernelArgs};
use crate::{CudaDevice, CudaValue};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use effect_torch_runtime::DType;

#[test]
fn grouped_vector_copy_requires_supported_aligned_bf16_rows_and_opt_in() {
    let mut args = CudaKernelArgs {
        output: 0x2000,
        output_dtype: dtype_code(DType::BF16),
        ..Default::default()
    };
    args.inputs[0] = 0x1000;
    args.integers[..3].copy_from_slice(&[704, 17, 1]);
    assert_eq!(
        grouped_copy_launch("et_grouped_gather", &args),
        ("et_grouped_gather_vector", 128)
    );
    for width in [1408, 2816] {
        args.integers[0] = width;
        assert_eq!(
            grouped_copy_launch("et_grouped_scatter", &args),
            ("et_grouped_scatter_vector", 256)
        );
    }
    let valid = args;
    for change in 0..6 {
        let mut args = valid;
        match change {
            0 => args.integers[2] = 0,
            1 => args.output_dtype = dtype_code(DType::F32),
            2 => args.integers[0] = 705,
            3 => args.inputs[0] += 2,
            4 => args.output += 2,
            _ => args.integers[1] = 0,
        }
        assert_eq!(
            grouped_copy_launch("et_grouped_gather", &args),
            ("et_grouped_gather", 256)
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn grouped_vector_copy_preserves_raw_bits_permutations_repeats_and_unaligned_fallback() {
    let device = CudaDevice::get(0).unwrap();
    for rows in [1usize, 33, 256] {
        let map = (0..rows)
            .map(|r| ((r * 13 + 7) % rows) as u32)
            .collect::<Vec<_>>();
        let map_bytes = map.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>();
        let map_value =
            CudaValue::from_dense_bytes(device.clone(), vec![rows], DType::U32, &map_bytes)
                .unwrap();
        for width in [704usize, 1408, 2816, 705] {
            for offset in [0usize, 1] {
                let count = rows * width;
                let data = (0..count + offset)
                    .map(|i| (i * 173 + 117) as u16)
                    .collect::<Vec<_>>();
                let bytes = data
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>();
                let input = CudaValue::from_dense_bytes(
                    device.clone(),
                    vec![data.len()],
                    DType::BF16,
                    &bytes,
                )
                .unwrap();
                for scatter in [false, true] {
                    let initial = vec![0xa5u8; bytes.len()];
                    let output = CudaValue::from_dense_bytes(
                        device.clone(),
                        vec![data.len()],
                        DType::BF16,
                        &initial,
                    )
                    .unwrap();
                    let source_rows = rows / 2 + 1;
                    let mut args = CudaKernelArgs {
                        output: output.storage_address() + (offset * 2) as u64,
                        output_dtype: dtype_code(DType::BF16),
                        elements: count as u64,
                        ..Default::default()
                    };
                    args.inputs[0] = input.storage_address() + (offset * 2) as u64;
                    args.inputs[1] = map_value.storage_address();
                    args.integers[..3].copy_from_slice(&[width as u64, source_rows as u64, 1]);
                    let name = if scatter {
                        "et_grouped_scatter"
                    } else {
                        "et_grouped_gather"
                    };
                    let (selected, threads) = grouped_copy_launch(name, &args);
                    assert_eq!(selected != name, offset == 0 && width != 705);
                    let mut launch = device
                        .stream
                        .launch_builder(device.kernel(selected).unwrap());
                    launch.arg(&args);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .unwrap();
                    let actual = output.read_storage_bytes().unwrap();
                    let mut expected = initial;
                    for (r, &mapped) in map.iter().enumerate() {
                        let source = if scatter {
                            r
                        } else {
                            mapped as usize % source_rows
                        };
                        let destination = if scatter { mapped as usize } else { r };
                        let from = 2 * (offset + source * width);
                        let to = 2 * (offset + destination * width);
                        expected[to..to + 2 * width]
                            .copy_from_slice(&bytes[from..from + 2 * width]);
                    }
                    assert_eq!(
                        actual, expected,
                        "rows{rows}/width{width}/offset{offset}/scatter{scatter}"
                    );
                    assert_eq!(input.read_storage_bytes().unwrap(), bytes);
                }
            }
        }
    }
}

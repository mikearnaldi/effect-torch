use super::*;
use crate::executable::{dtype_code, CudaKernelArgs};
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[test]
#[ignore = "requires CUDA"]
fn static_storage_generated_kernels_match_generic_mixed_types_views_and_reductions() {
    let device = crate::CudaDevice::get(0).unwrap();
    let scalar = crate::CudaValue::from_host(device.clone(), vec![1], DType::F32, &[0.5]).unwrap();
    for dtype in [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::U32,
        DType::U8,
    ] {
        let data = (0..58)
            .map(|i| match i % 11 {
                0 => -0.,
                1 => 0.,
                2 => f64::NAN,
                3 => f64::INFINITY,
                4 => f64::NEG_INFINITY,
                5 => f32::from_bits(1) as f64,
                _ => i as f64 - 32.25,
            })
            .collect::<Vec<_>>();
        let input = crate::CudaValue::from_host(device.clone(), vec![58], dtype, &data).unwrap();
        let generic = elementwise(
            &KernelExpr::Add(
                Box::new(KernelExpr::Input(0)),
                Box::new(KernelExpr::Input(1)),
            ),
            &[
                vec![1, 17].into_boxed_slice(),
                vec![0, 0].into_boxed_slice(),
            ],
            &[vec![0, 0].into_boxed_slice(), vec![0, 0].into_boxed_slice()],
            &[7, 0],
            &[17, 3],
        )
        .unwrap();
        let typed = specialize_storage_dtypes(
            generic.clone(),
            &[dtype_code(dtype), dtype_code(DType::F32)],
            dtype_code(dtype),
        )
        .unwrap();
        let make_output = || {
            crate::CudaValue::from_host(device.clone(), vec![17, 3], dtype, &vec![0.; 51]).unwrap()
        };
        let a = make_output();
        let b = make_output();
        let mut args = CudaKernelArgs {
            elements: 51,
            output_dtype: dtype_code(dtype),
            compute_dtype: dtype_code(DType::F32),
            ..Default::default()
        };
        args.inputs[0] = input.storage_address();
        args.inputs[1] = scalar.storage_address();
        args.input_dtypes[0] = dtype_code(dtype);
        args.input_dtypes[1] = dtype_code(DType::F32);
        for (source, out) in [(&generic, &a), (&typed, &b)] {
            args.output = out.storage_address();
            let function = device.fused_elementwise(source).unwrap();
            let mut launch = device.stream.launch_builder(&function);
            launch.arg(&args);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        }
        device.stream.synchronize().unwrap();
        assert_eq!(
            a.read_storage_bytes().unwrap(),
            b.read_storage_bytes().unwrap(),
            "dtype{dtype:?}"
        );
    }
    let width = 4097;
    let data = (0..2 * width)
        .map(|i| match i % 113 {
            0 => -0.,
            1 => f64::NAN,
            2 => f64::INFINITY,
            3 => f64::NEG_INFINITY,
            _ => ((i * 113) % 997) as f64 / 128. - 3.,
        })
        .collect::<Vec<_>>();
    let input =
        crate::CudaValue::from_host(device.clone(), vec![2, width], DType::F32, &data).unwrap();
    for arg in [false, true] {
        let generic = elementwise_with_sum(
            !arg,
            arg,
            &KernelExpr::Input(0),
            &[vec![width, 1].into_boxed_slice()],
            &[vec![0, 0].into_boxed_slice()],
            &[0],
            &[2, width],
        )
        .unwrap();
        let dtype = if arg { DType::I64 } else { DType::F32 };
        let typed = specialize_storage_dtypes(generic.clone(), &[1], dtype_code(dtype)).unwrap();
        let make_output =
            || crate::CudaValue::from_host(device.clone(), vec![2], dtype, &[0.; 2]).unwrap();
        let a = make_output();
        let b = make_output();
        let mut args = CudaKernelArgs {
            elements: 2,
            output_dtype: dtype_code(dtype),
            compute_dtype: 1,
            ..Default::default()
        };
        args.inputs[0] = input.storage_address();
        args.input_dtypes[0] = 1;
        args.integers[1] = width as u64;
        for (source, out) in [(&generic, &a), (&typed, &b)] {
            args.output = out.storage_address();
            let function = device.fused_elementwise(source).unwrap();
            let mut launch = device.stream.launch_builder(&function);
            launch.arg(&args);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (2, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        }
        device.stream.synchronize().unwrap();
        assert_eq!(
            a.read_storage_bytes().unwrap(),
            b.read_storage_bytes().unwrap(),
            "arg{arg}"
        );
    }
}

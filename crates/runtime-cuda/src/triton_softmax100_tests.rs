use super::*;

fn invocation() -> Invocation100 {
    Invocation100 {
        input: 0x1000_0000,
        output: 0x2000_0000,
        scratch: 0x3000_0000,
    }
}
fn manifest() -> serde_json::Value {
    serde_json::json!({"abi":1,"shape":[ROWS,WIDTH],"variants":IMAGES.iter().enumerate().map(|(i,&(symbol,hash,grid,block,shared,pointers,integers))| {
        serde_json::json!({"file":format!("stage{i}.ptx"),"symbol":symbol,"sha256":hash,"grid":grid,"block":block,"shared":shared,
            "parameters":([vec!["u64";pointers],vec!["u32";integers],vec!["u64";2]].concat()),
            "metadata":{"num_warps":block/32,"shared":shared,"global_scratch_size":0,"profile_scratch_size":0}})
    }).collect::<Vec<_>>()})
}
#[test]
fn softmax100_validates_all_ranges_before_first_enqueue() {
    let original = invocation();
    assert!(original.validate().is_ok());
    for field in 0..3 {
        for pointer in [
            0,
            2,
            u64::MAX - 15,
            original.input + 16,
            original.output + 16,
        ] {
            let mut bad = original;
            match field {
                0 => bad.input = pointer,
                1 => bad.output = pointer,
                _ => bad.scratch = pointer,
            }
            // A pointer in its own original range is still a valid fresh binding.
            if (field == 0 && pointer == original.input + 16)
                || (field == 1 && pointer == original.output + 16)
            {
                continue;
            }
            assert!(bad.validate().is_err(), "field {field}, pointer {pointer}");
        }
    }
    let mut adjacent = original;
    adjacent.output = adjacent.input + MATRIX_BYTES;
    adjacent.scratch = adjacent.output + MATRIX_BYTES;
    assert!(adjacent.validate().is_ok());
}
#[test]
fn softmax100_manifest_requires_all_exact_fixed_images_geometry_and_abi() {
    let original = manifest();
    assert!(validate_manifest(&original).is_ok());
    for stage in 0..5 {
        for field in [
            "file",
            "symbol",
            "sha256",
            "grid",
            "block",
            "shared",
            "parameters",
        ] {
            let mut bad = original.clone();
            bad["variants"][stage][field] = serde_json::json!("wrong");
            assert!(validate_manifest(&bad).is_err(), "{stage} {field}");
        }
        for field in [
            "num_warps",
            "shared",
            "global_scratch_size",
            "profile_scratch_size",
        ] {
            let mut bad = original.clone();
            bad["variants"][stage]["metadata"][field] = serde_json::json!(999);
            assert!(validate_manifest(&bad).is_err(), "{stage} {field}");
        }
    }
    let mut bad = original.clone();
    bad["abi"] = serde_json::json!(2);
    assert!(validate_manifest(&bad).is_err());
    let mut bad = original.clone();
    bad["shape"] = serde_json::json!([ROWS, WIDTH / 2]);
    assert!(validate_manifest(&bad).is_err());
    let mut bad = original;
    bad["variants"].as_array_mut().unwrap().pop();
    assert!(validate_manifest(&bad).is_err());
}
#[test]
fn softmax100_admission_requires_fixed_extent_dense_bf16() {
    use effect_torch_runtime::DType;
    assert!(admitted(ROWS, WIDTH, DType::BF16, true));
    for (rows, width, dtype, dense) in [
        (ROWS / 2, WIDTH, DType::BF16, true),
        (ROWS, WIDTH / 2, DType::BF16, true),
        (ROWS, WIDTH, DType::F32, true),
        (ROWS, WIDTH, DType::BF16, false),
        (0, WIDTH, DType::BF16, true),
    ] {
        assert!(!admitted(rows, width, dtype, dense));
    }
}
fn input_bits(contrast: bool) -> Vec<u16> {
    (0..ROWS * WIDTH)
        .map(|index| {
            let offset = ((index / WIDTH) % 17) as f32 * 0.25;
            let value = offset + if contrast && index % 2 == 1 { 1.0 } else { 0.0 };
            (value.to_bits() >> 16) as u16
        })
        .collect()
}
fn check_probabilities(values: &[u16], contrast: bool) {
    assert_eq!(values.len(), ROWS * WIDTH);
    let denominator = if contrast {
        (WIDTH / 2) as f32 * (1.0 + 1.0_f32.exp())
    } else {
        WIDTH as f32
    };
    for (index, &bits) in values.iter().enumerate() {
        let expected = if contrast && index % 2 == 1 {
            1.0_f32.exp() / denominator
        } else {
            1.0 / denominator
        };
        let got = f32::from_bits(u32::from(bits) << 16);
        assert!(
            got.is_finite() && (got - expected).abs() <= 0.0078125 * expected,
            "index {index}: {got} vs {expected}"
        );
    }
}
#[test]
#[ignore = "requires SM120 and TRITON_SOFTMAX100_DIRECTORY"]
fn softmax100_hardware_fixed_images_preserve_inputs_and_retained_outputs() {
    use cudarc::driver::{CudaSlice, DevicePtr};
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let plan = Softmax100::from_env(&context)
        .unwrap()
        .expect("softmax100 directory required");
    let pointer = |value: &CudaSlice<u16>| {
        let (address, event) = value.device_ptr(&stream);
        drop(event);
        address
    };
    let mut retained = Vec::new();
    for contrast in [false, true] {
        let host = input_bits(contrast);
        let input = stream.clone_htod(&host).unwrap();
        let output = stream.alloc_zeros::<u16>(ROWS * WIDTH).unwrap();
        let scratch = stream.alloc_zeros::<f32>(SCRATCH_ELEMENTS).unwrap();
        stream.synchronize().unwrap();
        let (address, event) = scratch.device_ptr(&stream);
        drop(event);
        let args = Invocation100 {
            input: pointer(&input),
            output: pointer(&output),
            scratch: address,
        };
        unsafe {
            plan.launch(&stream, args).unwrap();
        }
        stream.synchronize().unwrap();
        let saved = stream.clone_dtoh(&output).unwrap();
        check_probabilities(&saved, contrast);
        assert_eq!(stream.clone_dtoh(&input).unwrap(), host);
        let mut invalid = args;
        invalid.scratch = args.input;
        assert!(unsafe { plan.launch(&stream, invalid) }.is_err());
        assert_eq!(stream.clone_dtoh(&output).unwrap(), saved);
        unsafe {
            plan.launch(&stream, args).unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(stream.clone_dtoh(&output).unwrap(), saved);
        retained.push((output, saved));
    }
    drop(plan);
    for (output, saved) in retained {
        assert_eq!(stream.clone_dtoh(&output).unwrap(), saved);
    }
}
#[test]
#[ignore = "requires SM120, BF16_SOFTMAX=1 and TRITON_SOFTMAX100_DIRECTORY"]
fn softmax100_hardware_compiled_lifecycle_cancellation_and_retained_outputs() {
    use crate::{CudaDevice, CudaValue};
    use effect_torch_compiler::CompileOptions;
    use effect_torch_graph::{Device, Node, NodeKind};
    use effect_torch_runtime::{CancellationFlag, DType, StorageMetadata};
    let node = |kind| Node::new(kind).unwrap();
    let input = node(NodeKind::Input {
        slot: 0,
        shape: vec![ROWS, WIDTH],
        dtype: DType::BF16,
        device: Device::Cuda(0),
        storage: StorageMetadata::dense(),
    });
    let floating = node(NodeKind::Cast {
        a: input,
        dtype: DType::F32,
    });
    let maximum = node(NodeKind::Max {
        a: floating.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let shifted = node(NodeKind::Sub {
        a: floating,
        b: maximum,
    });
    let exponential = node(NodeKind::Exp { a: shifted });
    let denominator = node(NodeKind::Sum {
        a: exponential.clone(),
        dims: vec![1],
        keepdims: true,
    });
    let quotient = node(NodeKind::Div {
        a: exponential,
        b: denominator,
    });
    let output = node(NodeKind::Cast {
        a: quotient,
        dtype: DType::BF16,
    });
    let program =
        Arc::new(crate::compile_with_options(vec![output], 0, CompileOptions::default()).unwrap());
    let count = |name| {
        program
            .diagnostics()
            .instructions
            .iter()
            .filter(|i| i.kind == name)
            .map(|i| i.count)
            .sum::<usize>()
    };
    assert_eq!(count("triton_softmax100"), 1);
    assert_eq!(count("et_bf16_softmax_prepare"), 0);
    assert_eq!(count("et_bf16_softmax_store"), 0);
    let device = CudaDevice::get(0).unwrap();
    let mut retained = Vec::new();
    for contrast in [false, true] {
        let host = input_bits(contrast)
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let value =
            CudaValue::from_dense_bytes(device.clone(), vec![ROWS, WIDTH], DType::BF16, &host)
                .unwrap();
        let bindings = vec![value];
        let cancelled = CancellationFlag::new();
        cancelled.cancel();
        assert!(program.execute(&bindings, &[], &cancelled).is_err());
        assert!(program.execute(&[], &[], &CancellationFlag::new()).is_err());
        let cancelled = CancellationFlag::new();
        let reached = std::cell::Cell::new(false);
        assert!(program
            .execute_with_softmax100_hook(&bindings, &cancelled, &|| {
                reached.set(true);
                cancelled.cancel();
            })
            .is_err());
        assert!(
            reached.get(),
            "must cancel after the actual five-kernel submission"
        );
        let actual = program
            .execute(&bindings, &[], &CancellationFlag::new())
            .unwrap();
        let saved = actual[0].read_storage_bytes().unwrap();
        let probabilities = saved
            .chunks_exact(2)
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .collect::<Vec<_>>();
        check_probabilities(&probabilities, contrast);
        drop(probabilities);
        assert_eq!(bindings[0].read_storage_bytes().unwrap(), host);
        let mut workers = Vec::new();
        for _ in 0..2 {
            let program = program.clone();
            let bindings = bindings.clone();
            let expected = saved.clone();
            workers.push(std::thread::spawn(move || {
                let got = program
                    .execute(&bindings, &[], &CancellationFlag::new())
                    .unwrap();
                assert_eq!(got[0].read_storage_bytes().unwrap(), expected);
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(bindings[0].read_storage_bytes().unwrap(), host);
        retained.push((actual, saved));
    }
    drop(program);
    for (values, saved) in retained {
        assert_eq!(values[0].read_storage_bytes().unwrap(), saved);
    }
}

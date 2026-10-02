use super::*;
fn invocation() -> Invocation101 {
    Invocation101 {
        packed: 0x100_0000,
        query_weight: 0x200_0000,
        key_weight: 0x300_0000,
        cosine: 0x400_0000,
        sine: 0x500_0000,
        cosine_stride: 128,
        sine_stride: 128,
        table: 0x600_0000,
        positions: 0x700_0000,
        sum_q: 0x800_0000,
        sum_k: 0x900_0000,
        raw_query: 0xa00_0000,
        raw_key: 0xb00_0000,
        raw_value: 0xc00_0000,
        query: 0xd00_0000,
        key: 0xe00_0000,
        value: 0xf00_0000,
    }
}
#[test]
fn normrope101_validates_complete_pipeline_before_mutation() {
    let original = invocation();
    assert!(original.validate().is_ok());
    let mut shared = original;
    shared.key_weight = shared.query_weight;
    assert!(shared.validate().is_ok());
    for field in 0..10 {
        let mut bad = original;
        match field {
            0 => bad.table = original.cosine,
            1 => bad.positions = original.packed,
            2 => bad.sum_q = original.query_weight,
            3 => bad.sum_k = original.key_weight,
            4 => bad.raw_query = original.packed,
            5 => bad.raw_key = original.table,
            6 => bad.raw_value = original.raw_query,
            7 => bad.query = original.raw_query,
            8 => bad.key = original.query + 16,
            _ => bad.value = original.key,
        }
        assert!(bad.validate().is_err(), "mutable field {field}");
    }
}
#[test]
fn normrope101_rejects_unaligned_null_and_overflowing_ranges() {
    for pointer in [0, 2, 0x100_0002, u64::MAX - 15] {
        let mut bad = invocation();
        bad.packed = pointer;
        assert!(bad.validate().is_err());
        let mut bad = invocation();
        bad.query = pointer;
        assert!(bad.validate().is_err());
    }
}
fn manifest() -> serde_json::Value {
    let parameters = |p, n| {
        [
            vec![serde_json::json!({"type":"u64"}); p],
            vec![serde_json::json!({"type":"u32"}); n],
            vec![serde_json::json!({"type":"u64"}); 2],
        ]
        .concat()
    };
    serde_json::json!({"abi":1,"computeCapability":[12,0],"tokens":256,"qHeads":16,"kvHeads":8,"headWidth":256,"helperAbi":2,"helperSourceTableStrides":[128,256],"variants":[
        {"role":"reduction","file":"reduction.ptx","symbol":REDUCTION,"sha256":REDUCTION_SHA,"grid":[128,1,1],"block":[512,1,1],"numWarps":16,"dynamicSharedBytes":256,"xBlock":64,"rBlock":64,"globalScratchSize":0,"profileScratchSize":0,"parameters":parameters(4,3)},
        {"role":"pointwise","file":"pointwise.ptx","symbol":POINTWISE,"sha256":POINTWISE_SHA,"grid":[1536,1,1],"block":[256,1,1],"numWarps":8,"dynamicSharedBytes":0,"xBlock":512,"rBlock":null,"globalScratchSize":0,"profileScratchSize":0,"parameters":parameters(11,2)}]})
}
#[test]
fn normrope101_manifest_rejects_generic_name_wrong_binary_geometry_or_abi() {
    let original = manifest();
    assert!(manifest_rows(&original).is_ok());
    for field in [
        "symbol",
        "sha256",
        "numWarps",
        "dynamicSharedBytes",
        "xBlock",
        "rBlock",
        "globalScratchSize",
        "profileScratchSize",
        "block",
        "grid",
    ] {
        let mut bad = original.clone();
        bad["variants"][0][field] = serde_json::json!("another same-name variant");
        assert!(manifest_rows(&bad).is_err(), "{field}");
    }
    for field in ["tokens", "qHeads", "kvHeads", "headWidth", "helperAbi"] {
        let mut bad = original.clone();
        bad[field] = serde_json::json!(1);
        assert!(manifest_rows(&bad).is_err());
    }
    for strides in [
        serde_json::Value::Null,
        serde_json::json!([128]),
        serde_json::json!([256, 128]),
    ] {
        let mut bad = original.clone();
        bad["helperSourceTableStrides"] = strides;
        assert!(manifest_rows(&bad).is_err());
    }
    let mut bad = original.clone();
    bad["variants"][0]["file"] = serde_json::json!("../reduction.ptx");
    assert!(manifest_rows(&bad).is_err());
    let mut bad = original.clone();
    bad["variants"][1]["role"] = serde_json::json!("reduction");
    assert!(manifest_rows(&bad).is_err());
    let mut bad = original;
    bad["variants"][1]["parameters"][11]["type"] = serde_json::json!("u64");
    assert!(manifest_rows(&bad).is_err());
}
#[test]
fn normrope101_validates_independent_strides_and_full_borrowed_extents() {
    let original = invocation();
    for stride in [0, 64, 512, u32::MAX] {
        let mut bad = original;
        bad.cosine_stride = stride;
        assert!(bad.validate().is_err());
        let mut bad = original;
        bad.sine_stride = stride;
        assert!(bad.validate().is_err());
    }
    for cosine in [false, true] {
        let mut args = original;
        args.table = if cosine { args.cosine } else { args.sine } + HALF_TABLE_BYTES as u64 + 16;
        assert!(
            args.validate().is_ok(),
            "compact range ends before mutable table"
        );
        if cosine {
            args.cosine_stride = 256
        } else {
            args.sine_stride = 256
        };
        assert!(
            args.validate().is_err(),
            "full borrowed range must include second half"
        );
    }
    for cosine_stride in [128, 256] {
        for sine_stride in [128, 256] {
            let mut args = original;
            args.cosine_stride = cosine_stride;
            args.sine_stride = sine_stride;
            assert!(args.validate().is_ok());
        }
    }
}
#[test]
#[ignore = "requires SM120 and ABI2 stride-aware TRITON_NORMROPE101_DIRECTORY"]
fn normrope101_hardware_fixed_images_preserve_inputs_and_retained_outputs() {
    use cudarc::driver::{CudaSlice, DevicePtr};
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let plan = NormRope101::from_env(&context)
        .unwrap()
        .expect("normrope101 directory required");
    let pointer = |buffer: &CudaSlice<u16>| {
        let (address, event) = buffer.device_ptr(&stream);
        drop(event);
        address
    };
    let float_pointer = |buffer: &CudaSlice<f32>| {
        let (address, event) = buffer.device_ptr(&stream);
        drop(event);
        address
    };
    let long_pointer = |buffer: &CudaSlice<i64>| {
        let (address, event) = buffer.device_ptr(&stream);
        drop(event);
        address
    };
    let narrow = |value: f32| {
        let bits = value.to_bits();
        (bits.wrapping_add(0x7fff + (bits >> 16 & 1)) >> 16) as u16
    };
    let decode = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut retained = Vec::new();
    for cosine_stride in [128, 256] {
        for sine_stride in [128, 256] {
            for magnitude in [0.25_f32, 1.0] {
                let input = vec![narrow(magnitude); PACKED_BYTES / 2];
                let make_table = |stride: usize, sine: bool| {
                    (0..TOKENS * stride)
                        .map(|index| {
                            let row = index / stride;
                            let column = index % 128;
                            let angle =
                                row as f32 * 0.017 + column as f32 * 0.023 + magnitude * 0.1;
                            narrow(if sine { angle.sin() } else { angle.cos() })
                        })
                        .collect::<Vec<_>>()
                };
                let hcos = make_table(cosine_stride as usize, false);
                let hsin = make_table(sine_stride as usize, true);
                let raw = stream.clone_htod(&input).unwrap();
                let qw = stream.clone_htod(&vec![narrow(1.125); WIDTH]).unwrap();
                let kw = stream.clone_htod(&vec![narrow(0.875); WIDTH]).unwrap();
                let cos = stream.clone_htod(&hcos).unwrap();
                let sin = stream.clone_htod(&hsin).unwrap();
                let table = stream.alloc_zeros::<u16>(TABLE_BYTES / 2).unwrap();
                let positions = stream.alloc_zeros::<i64>(TOKENS).unwrap();
                let sum_q = stream.alloc_zeros::<f32>(Q_SUM_BYTES / 4).unwrap();
                let sum_k = stream.alloc_zeros::<f32>(K_SUM_BYTES / 4).unwrap();
                let scratch = [Q_BYTES, KV_BYTES, KV_BYTES]
                    .map(|bytes| stream.alloc_zeros::<u16>(bytes / 2).unwrap());
                let output = [Q_BYTES, KV_BYTES, KV_BYTES]
                    .map(|bytes| stream.alloc_zeros::<u16>(bytes / 2).unwrap());
                stream.synchronize().unwrap();
                let args = Invocation101 {
                    packed: pointer(&raw),
                    query_weight: pointer(&qw),
                    key_weight: pointer(&kw),
                    cosine: pointer(&cos),
                    sine: pointer(&sin),
                    cosine_stride,
                    sine_stride,
                    table: pointer(&table),
                    positions: long_pointer(&positions),
                    sum_q: float_pointer(&sum_q),
                    sum_k: float_pointer(&sum_k),
                    raw_query: pointer(&scratch[0]),
                    raw_key: pointer(&scratch[1]),
                    raw_value: pointer(&scratch[2]),
                    query: pointer(&output[0]),
                    key: pointer(&output[1]),
                    value: pointer(&output[2]),
                };
                unsafe {
                    plan.launch(&stream, &args).unwrap();
                }
                stream.synchronize().unwrap();
                let saved = output
                    .iter()
                    .map(|value| stream.clone_dtoh(value).unwrap())
                    .collect::<Vec<_>>();
                let inverse = 1.0_f32 / (magnitude * magnitude + 1e-6).sqrt();
                for (role, values) in saved.iter().enumerate() {
                    for (index, &bits) in values.iter().enumerate() {
                        let token = (index / WIDTH) % TOKENS;
                        let column = index % WIDTH;
                        let half_column = column % 128;
                        let c = decode(hcos[token * cosine_stride as usize + half_column]);
                        let s = decode(hsin[token * sine_stride as usize + half_column]);
                        let normalized = magnitude * inverse * [1.125, 0.875, 1.0][role];
                        let expected = if role == 2 {
                            normalized
                        } else if column < 128 {
                            normalized.mul_add(c, -normalized * s)
                        } else {
                            normalized.mul_add(c, normalized * s)
                        };
                        let got = decode(bits);
                        assert!(
                            got.is_finite()
                                && (got - expected).abs() <= 0.0078125 * expected.abs().max(0.05),
                            "stride{cosine_stride}/{sine_stride}, role{role}, index{index}: {got} vs {expected}"
                        );
                    }
                }
                assert_eq!(stream.clone_dtoh(&raw).unwrap(), input);
                assert_eq!(stream.clone_dtoh(&qw).unwrap(), vec![narrow(1.125); WIDTH]);
                assert_eq!(stream.clone_dtoh(&kw).unwrap(), vec![narrow(0.875); WIDTH]);
                assert_eq!(stream.clone_dtoh(&cos).unwrap(), hcos);
                assert_eq!(stream.clone_dtoh(&sin).unwrap(), hsin);
                let packed_table = stream.clone_dtoh(&table).unwrap();
                for row in 0..TOKENS {
                    for column in 0..128 {
                        assert_eq!(
                            packed_table[row * 256 + column],
                            hcos[row * cosine_stride as usize + column]
                        );
                        assert_eq!(
                            packed_table[row * 256 + 128 + column],
                            hsin[row * sine_stride as usize + column]
                        );
                    }
                }
                assert_eq!(
                    stream.clone_dtoh(&positions).unwrap(),
                    (0..TOKENS as i64).collect::<Vec<_>>()
                );
                for sums in [&sum_q, &sum_k] {
                    for sum in stream.clone_dtoh(sums).unwrap() {
                        assert_eq!(sum, magnitude * magnitude * WIDTH as f32);
                    }
                }
                unsafe {
                    plan.launch(&stream, &args).unwrap();
                }
                stream.synchronize().unwrap();
                for (value, saved) in output.iter().zip(&saved) {
                    assert_eq!(stream.clone_dtoh(value).unwrap(), *saved);
                }
                retained.push((output, saved));
            }
        }
    }
    drop(plan);
    for (output, saved) in retained {
        for (value, saved) in output.iter().zip(saved) {
            assert_eq!(stream.clone_dtoh(value).unwrap(), saved);
        }
    }
}

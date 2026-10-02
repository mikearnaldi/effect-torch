use super::*;

fn entrance() -> Entrance98 {
    let mut args = Entrance98 {
        rows: 256,
        attention: 0,
        attention_weight: 0,
        hidden: 0,
        rho_bf16: 0,
        router_weight: 0,
        expert_weight: 0,
        dense_weight: 0,
        sum_a: 0,
        router: 0,
        expert: 0,
        dense: 0,
    };
    let addresses = (1..=11).map(|n| n * 0x100_0000).collect::<Vec<_>>();
    args.attention = addresses[0];
    args.attention_weight = addresses[1];
    args.hidden = addresses[2];
    args.rho_bf16 = addresses[3];
    args.router_weight = addresses[4];
    args.expert_weight = addresses[5];
    args.dense_weight = addresses[6];
    args.sum_a = addresses[7];
    args.router = addresses[8];
    args.expert = addresses[9];
    args.dense = addresses[10];
    args
}

fn tail() -> Tail98 {
    Tail98 {
        rows: 256,
        private_attention: 0x100_0000,
        dense: 0x200_0000,
        expert: 0x300_0000,
        dense_weight: 0x400_0000,
        expert_weight: 0x500_0000,
        combined_weight: 0x600_0000,
        sum_a: 0x700_0000,
        attention_weight: 0x800_0000,
        hidden: 0x900_0000,
        scale: 0xa00_0000,
        next_weight: 0xb00_0000,
        combined_f32: 0xc00_0000,
        next_norm: 0xd00_0000,
    }
}

#[test]
fn rejects_escaped_or_overlapping_mutable_storage_before_enqueue() {
    let original = entrance();
    assert!(original.validate().is_ok());
    let mut bad = original;
    bad.dense = original.hidden;
    assert!(bad.validate().is_err());
    bad = original;
    bad.sum_a = original.dense + 16;
    assert!(bad.validate().is_err());
    // Read-only model weights may legitimately share one constant allocation.
    let mut shared = original;
    shared.router_weight = shared.expert_weight;
    assert!(shared.validate().is_ok());

    let original = tail();
    assert!(original.validate().is_ok());
    let mut bad = original;
    bad.private_attention = original.hidden;
    assert!(bad.validate().is_err());
    bad = original;
    bad.next_norm = original.private_attention;
    assert!(bad.validate().is_err());
    bad = original;
    bad.combined_f32 = original.dense + 16;
    assert!(bad.validate().is_err());
}

#[test]
fn rejects_unsupported_geometry_alignment_and_pointer_overflow() {
    let original = entrance();
    for rows in [0, 1, 32, 65, 512, u32::MAX] {
        let mut bad = original;
        bad.rows = rows;
        assert!(bad.validate().is_err());
    }
    for pointer in [0, original.hidden + 2, u64::MAX - 15] {
        let mut bad = original;
        bad.hidden = pointer;
        assert!(bad.validate().is_err());
    }
    let mut supported = original;
    supported.rows = 64;
    assert!(supported.validate().is_ok());
    let mut bad = tail();
    bad.scale += 2;
    assert!(bad.validate().is_err());
}

fn manifest() -> serde_json::Value {
    let parameters = |count| {
        [
            vec![serde_json::json!({"type":"u64"}); count],
            vec![serde_json::json!({"type":"u32"}); 2],
            vec![serde_json::json!({"type":"u64"}); 2],
        ]
        .concat()
    };
    let entrance_parameters = parameters(11);
    let tail_parameters = parameters(13);
    serde_json::json!({
        "abi": 1, "computeCapability": [12, 0], "variants": [
            { "role": "entrance", "entry": ENTRANCE, "file": "entrance.ptx",
              "sha256": ENTRANCE_SHA, "numWarps": 8, "dynamicSharedBytes": SHARED,
              "globalScratchSize": 0, "profileScratchSize": 0,
              "parameters": entrance_parameters },
            { "role": "tail", "entry": TAIL, "file": "tail.ptx",
              "sha256": TAIL_SHA, "numWarps": 8, "dynamicSharedBytes": SHARED,
              "globalScratchSize": 0, "profileScratchSize": 0,
              "parameters": tail_parameters }
        ]
    })
}

#[test]
fn manifest_rejects_same_name_different_tail_image_and_launch_abi() {
    let original = manifest();
    assert!(manifest_rows(&original).is_ok());
    for field in [
        "entry",
        "sha256",
        "numWarps",
        "dynamicSharedBytes",
        "globalScratchSize",
    ] {
        let mut bad = original.clone();
        bad["variants"][1][field] = serde_json::json!("another same-name variant");
        assert!(manifest_rows(&bad).is_err(), "{field}");
    }
    let mut bad = original.clone();
    bad["variants"][0]["file"] = serde_json::json!("../entrance.ptx");
    assert!(manifest_rows(&bad).is_err());
    let mut bad = original.clone();
    bad["variants"][1]["role"] = serde_json::json!("entrance");
    assert!(manifest_rows(&bad).is_err());
    let mut bad = original.clone();
    bad["variants"][1]["parameters"][13]["type"] = serde_json::json!("u64");
    assert!(manifest_rows(&bad).is_err());
    let mut bad = original;
    bad["computeCapability"] = serde_json::json!([12, 1]);
    assert!(manifest_rows(&bad).is_err());
}

#[test]
fn scalar_conversion_preserves_f32_rounding_and_bf16_ties() {
    let rho = 2816_f32.sqrt().recip();
    let converted = f32::from_bits(u32::from(narrow(rho)) << 16);
    assert_eq!(narrow(rho), 0x3c9a);
    assert_ne!(converted, rho);
    assert_eq!(narrow(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(narrow(f32::from_bits(0x3f81_8000)), 0x3f82);
    assert_eq!(narrow(-0.0), 0x8000);
}

#[test]
#[ignore = "requires CUDA SM120 and the screened TRITON_NORM98_DIRECTORY"]
fn hardware_fixed_image_constant_pipeline_retains_readonly_inputs_and_outputs() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let image = Norm98::from_env(&context)
        .unwrap()
        .expect("norm98 directory required");
    let rho = Rho98::new(&stream, 0.03125).unwrap();
    let pointer = |x: &CudaSlice<u16>| {
        let (address, ready) = x.device_ptr(&stream);
        drop(ready);
        address
    };
    // Constant rows provide an independent closed-form pipeline reference.
    // Both row counts, changed inputs, retained prior outputs and unchanged
    // borrowed inputs cross the real Rust driver/ABI boundary.
    let mut retained = Vec::new();
    for rows in [64_u32, 256] {
        for magnitude in [0.25_f32, 1.0] {
            let n = rows as usize * WIDTH as usize;
            let a = stream.clone_htod(&vec![narrow(magnitude); n]).unwrap();
            let h = stream.clone_htod(&vec![narrow(0.5); n]).unwrap();
            let weights = stream
                .clone_htod(&vec![narrow(1.0); WIDTH as usize])
                .unwrap();
            let scale = stream.clone_htod(&[narrow(1.0)]).unwrap();
            let private_a = stream.clone_htod(&vec![narrow(magnitude); n]).unwrap();
            let packed = stream.clone_htod(&vec![0_u16; 3 * n]).unwrap();
            let next_norm = stream.clone_htod(&vec![0_u16; n]).unwrap();
            let sum_a = stream.clone_htod(&vec![0_f32; rows as usize]).unwrap();
            let combined = stream.clone_htod(&vec![0_f32; n]).unwrap();
            let (sum_ptr, sum_ready) = sum_a.device_ptr(&stream);
            drop(sum_ready);
            let (combined_ptr, combined_ready) = combined.device_ptr(&stream);
            drop(combined_ready);
            stream.synchronize().unwrap();
            let packed_ptr = pointer(&packed);
            let args = Entrance98 {
                rows,
                attention: pointer(&private_a),
                attention_weight: pointer(&weights),
                hidden: pointer(&h),
                rho_bf16: rho.address(&context).unwrap(),
                router_weight: pointer(&weights),
                expert_weight: pointer(&weights),
                dense_weight: pointer(&weights),
                sum_a: sum_ptr,
                router: packed_ptr + (4 * n) as u64,
                expert: packed_ptr + (2 * n) as u64,
                dense: packed_ptr,
            };
            let tail = Tail98 {
                rows,
                private_attention: pointer(&private_a),
                dense: pointer(&a),
                expert: pointer(&a),
                dense_weight: pointer(&weights),
                expert_weight: pointer(&weights),
                combined_weight: pointer(&weights),
                sum_a: sum_ptr,
                attention_weight: pointer(&weights),
                hidden: pointer(&h),
                scale: pointer(&scale),
                next_weight: pointer(&weights),
                combined_f32: combined_ptr,
                next_norm: pointer(&next_norm),
            };
            let result = unsafe {
                image
                    .launch_entrance(&stream, &args)
                    .and_then(|()| image.launch_tail(&stream, &tail))
            };
            stream.synchronize().unwrap();
            result.unwrap();
            let packed_host = stream.clone_dtoh(&packed).unwrap();
            let state_host = stream.clone_dtoh(&private_a).unwrap();
            let next_host = stream.clone_dtoh(&next_norm).unwrap();
            let attention_norm = magnitude / (magnitude * magnitude + 1e-6).sqrt();
            let residual = attention_norm + 0.5;
            let normalized = residual / (residual * residual + 1e-6).sqrt();
            let combined_value = 2.0 * attention_norm;
            let next = combined_value / (combined_value * combined_value + 1e-6).sqrt() + residual;
            let inv_next = (next * next + 1e-6).sqrt().recip();
            let wide = |bits: u16| f32::from_bits(u32::from(bits) << 16);
            let bounded = |value: f32, reference: f32| {
                assert!((value - reference).abs() <= 0.0078125 * reference.abs().max(1.0))
            };
            for i in 0..n {
                bounded(wide(packed_host[i]), normalized);
                bounded(wide(packed_host[n + i]), normalized);
                bounded(wide(packed_host[2 * n + i]), normalized * 0.03125);
                bounded(wide(state_host[i]), next);
                bounded(wide(next_host[i]), wide(state_host[i]) * inv_next);
            }
            assert_eq!(stream.clone_dtoh(&a).unwrap(), vec![narrow(magnitude); n]);
            assert_eq!(stream.clone_dtoh(&h).unwrap(), vec![narrow(0.5); n]);
            assert_eq!(
                stream.clone_dtoh(&weights).unwrap(),
                vec![narrow(1.0); WIDTH as usize]
            );
            retained.push((
                private_a,
                state_host,
                next_norm,
                next_host,
                packed,
                packed_host,
            ));
        }
    }
    drop(image);
    drop(rho);
    for (state, state_host, next, next_host, packed, packed_host) in retained {
        assert_eq!(stream.clone_dtoh(&state).unwrap(), state_host);
        assert_eq!(stream.clone_dtoh(&next).unwrap(), next_host);
        assert_eq!(stream.clone_dtoh(&packed).unwrap(), packed_host);
    }
}

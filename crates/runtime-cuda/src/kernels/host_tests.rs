use super::*;
use effect_torch_runtime::{decode_ggml_k_block, GgmlKQuant};
use std::collections::HashSet;
use std::io::Write;
use std::process::Command;

#[test]
fn production_registry_matches_descriptor_entrypoints() {
    let mut names = HashSet::new();
    for name in TYPED_KERNELS {
        assert!(names.insert(name.to_string()));
        let signature = match *name {
            "et_grouped_pointer_banks" => "EtGroupedPointerBanks a",
            "et_expert_partial_pointer_banks" => "EtGroupedPointerBanks a, unsigned int first",
            _ => "CudaKernelArgs a",
        };
        assert!(
            [
                TYPED_SOURCE,
                TYPED_BINARY_SOURCE,
                GROUPED_COPY_SOURCE,
                GROUPED_ROWS_SOURCE,
                EXPERT_ROUTE_RANK_SOURCE,
                ORDERED_SORTED_SOURCE,
                GROUPED_INVERSE_SOURCE,
                FFN_TAIL_SOURCE,
                FFN_NEXT_NORM63_SOURCE,
                ROUTER_TAIL_SOURCE,
                DUAL_ARGMAX_SOURCE,
                RMS_RESIDUAL_SOURCE,
                ATTN_FFN_ENTRANCE_SOURCE
            ]
            .iter()
            .any(|source| source.contains(&format!("void {name}({signature})"))),
            "missing typed entrypoint {name}"
        );
    }
    for module in COMPUTE_MODULES {
        assert!(COMPUTE_WRAPPERS.contains(module.define.trim_start_matches("#define ")));
        // Match the source set passed to compile_module in device.rs. Some
        // entrypoints live beside their implementation rather than in the
        // common wrappers; each registered name must still have one definition.
        let sources = [
            TYPED_HEADER,
            F32_PRELUDE,
            COMMON_SOURCE,
            module.define,
            module.source,
            RNG_ARG80_SOURCE,
            SAMPLER83_SOURCE,
            COMPUTE_WRAPPERS,
            ENTROPY_SOURCE,
            ENTROPY81_SOURCE,
            SMALL_SOFTMAX_SOURCE,
            NORM_ROPE_SOURCE,
        ];
        for name in module.kernels {
            let signature = format!("void {name}(CudaKernelArgs a)");
            assert_eq!(
                sources
                    .iter()
                    .map(|source| source.matches(&signature).count())
                    .sum::<usize>(),
                1,
                "expected exactly one {} module entrypoint {name}",
                module.name
            );
            for suffix in ["f32", "f64"] {
                assert!(names.insert(format!("{name}_{suffix}")));
            }
        }
    }
    for name in ["et_quantized_linear", "et_quantized_embedding"] {
        assert!(QUANTIZED_SOURCE.contains(&format!("void {name}(CudaKernelArgs a)")));
    }
    assert!(CACHE_SOURCE.contains("void et_kv_attention(CudaKernelArgs a)"));
    assert!(COMPUTE_WRAPPERS.contains("void et_mean256_f32(CudaKernelArgs a)"));
}

fn run_host_kernel_test(argument: &str) {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("cuda-binary-host-{}-{unique}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
    let source =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels/host-tests.cpp");
    for f32_compute in [false, true] {
        let executable = directory.join(if f32_compute { "host-f32" } else { "host-f64" });
        let mut command = Command::new(&compiler);
        command.args(["-std=c++17", "-O2", "-ffp-contract=off"]);
        if f32_compute {
            command.arg("-DET_TEST_F32");
        }
        let output = command
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new(&executable).arg(argument).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
#[ignore = "requires a C++17 compiler with _Float16; does not require CUDA"]
fn host_binary_broadcast_and_scalar_coercion() {
    run_host_kernel_test("--binary-broadcast");
}

#[test]
#[ignore = "requires a C++17 compiler with _Float16; does not require CUDA"]
fn host_top_k_bitonic_matches_stable_order() {
    run_host_kernel_test("--top-k");
}

#[test]
#[ignore = "requires a C++17 compiler with _Float16; does not require CUDA"]
fn host_scalar_abi_and_canonical_packed_fixtures() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("cuda-kernel-host-{}-{unique}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let fixture_path = directory.join("canonical.bin");
    let mut fixture = std::fs::File::create(&fixture_path).unwrap();
    let mut seed = 0x517cc1b727220a95_u64;
    for (code, codec) in [
        GgmlKQuant::Q2K,
        GgmlKQuant::Q3K,
        GgmlKQuant::Q4K,
        GgmlKQuant::Q5K,
        GgmlKQuant::Q6K,
    ]
    .into_iter()
    .enumerate()
    {
        for _ in 0..32 {
            let mut block = vec![0_u8; codec.block_bytes()];
            for byte in &mut block {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *byte = seed as u8;
            }
            let scale_offsets: &[usize] = match codec {
                GgmlKQuant::Q2K => &[80, 82],
                GgmlKQuant::Q3K => &[108],
                GgmlKQuant::Q4K | GgmlKQuant::Q5K => &[0, 2],
                GgmlKQuant::Q6K => &[208],
            };
            for offset in scale_offsets {
                // Finite nonzero scale with varied sign, exponent and mantissa.
                block[offset + 1] = (block[offset + 1] & 0x83) | 0x30;
            }
            let mut decoded = [0_f32; 256];
            decode_ggml_k_block(codec, &block, &mut decoded).unwrap();
            fixture.write_all(&(code as u32).to_le_bytes()).unwrap();
            fixture.write_all(&block).unwrap();
            for value in decoded {
                fixture.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }
    drop(fixture);
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
    let source = std::path::Path::new(file!());
    let source = if source.is_absolute() {
        source.with_file_name("host-tests.cpp")
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels/host-tests.cpp")
    };
    for f32_compute in [false, true] {
        let executable = directory.join(if f32_compute { "host-f32" } else { "host-f64" });
        let mut command = Command::new(&compiler);
        command.args(["-std=c++17", "-O2", "-ffp-contract=off"]);
        if f32_compute {
            command.arg("-DET_TEST_F32");
        }
        let output = command
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new(&executable)
            .arg(&fixture_path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
#[ignore = "requires a C++17 compiler with _Float16; does not require CUDA"]
fn attention82_fallback_store_preserves_rounding_layout_and_redzones() {
    run_host_kernel_test("--attention82-round");
}

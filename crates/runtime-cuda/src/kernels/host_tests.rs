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
        assert!(TYPED_SOURCE.contains(&format!("void {name}(CudaKernelArgs a)")));
    }
    for module in COMPUTE_MODULES {
        assert!(COMPUTE_WRAPPERS.contains(module.define.trim_start_matches("#define ")));
        for name in module.kernels {
            assert!(COMPUTE_WRAPPERS.contains(&format!("void {name}(CudaKernelArgs a)")));
            for suffix in ["f32", "f64"] {
                assert!(names.insert(format!("{name}_{suffix}")));
            }
        }
    }
    for name in ["et_quantized_linear", "et_quantized_embedding"] {
        assert!(QUANTIZED_SOURCE.contains(&format!("void {name}(CudaKernelArgs a)")));
    }
    assert!(CACHE_SOURCE.contains("void et_kv_attention(CudaKernelArgs a)"));
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

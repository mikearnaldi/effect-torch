use super::*;
use crate::CudaValue;
use cudarc::driver::DevicePtr;
use effect_torch_runtime::DType;

#[test]
#[ignore = "requires pinned CUDA, merged PTX and EFFECT_TORCH_CUDA_EXPERT_GEMV_OVERLAP=1"]
fn gemv_overlap_independent_bank_exact_chunks_fallback_guards_failure_and_retention() {
    assert!(expert_gemv_overlap::enabled());
    let device = CudaDevice::get(0).unwrap();
    assert!(device.exact_splitk_fingerprint && device.merged_expert.is_some());
    assert!(device.kernel("et_expert_gemv_second").is_ok());
    let (n, k) = (2816, 704);
    let bytes = CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS
        + EXPERT_GROUPED_POINTER_BYTES
        + 2 * EXPERT_SPLITK_DESCRIPTOR_BYTES;
    let reference = CudaBlas::new(device.stream.clone()).unwrap();
    let mut mixed = vec![1; 33];
    mixed.extend([2, 17, 257, 513]);
    let mut retained = Vec::new();
    for (phase, rows) in [mixed, vec![1; 128], vec![2, 33], vec![]]
        .into_iter()
        .enumerate()
    {
        let total: usize = rows.iter().sum();
        for pattern in 0..4 {
            let values = |len: usize, weight: bool| {
                (0..len)
                    .map(|i| {
                        let h = (i as u32).wrapping_mul(1664525).wrapping_add(if weight {
                            313
                        } else {
                            17
                        });
                        let bits = match pattern {
                            0 => {
                                (((h >> 16) & 0x8000) | ((120 + (h % 12)) << 7) | ((h >> 8) & 127))
                                    as u16
                            }
                            1 => [0u16, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x3f80, 0xbf80][i % 8],
                            2 => [0x7f80u16, 0xff80, 0x7fc1, 0xffc1, 0x7f7f, 0xff7f, 0x8000, 0]
                                [i % 8],
                            _ => {
                                if weight {
                                    0x7180
                                } else if i % 2 == 0 {
                                    1
                                } else {
                                    0x8001
                                }
                            }
                        };
                        half::bf16::from_bits(bits).to_f64()
                    })
                    .collect::<Vec<_>>()
            };
            let x = CudaValue::from_host(
                device.clone(),
                vec![total, k],
                DType::BF16,
                &values(total * k, false),
            )
            .unwrap();
            let w = CudaValue::from_host(
                device.clone(),
                vec![n, k],
                DType::BF16,
                &values(n * k, true),
            )
            .unwrap();
            let expected = CudaValue::from_host(
                device.clone(),
                vec![total, n],
                DType::BF16,
                &vec![0.; total * n],
            )
            .unwrap();
            let groups = |output: &CudaValue| {
                let mut offset = 0;
                rows.iter()
                    .map(|&m| {
                        let group = (
                            Bf16GemmPlan {
                                m,
                                n,
                                k,
                                batch: 1,
                                stride_x: m * k,
                                stride_weight: 0,
                                stride_out: m * n,
                            },
                            x.storage_address() + (offset * k * 2) as u64,
                            w.storage_address(),
                            output.storage_address() + (offset * n * 2) as u64,
                        );
                        offset += m;
                        group
                    })
                    .collect::<Vec<_>>()
            };
            let mut workspace = unsafe { device.stream.alloc::<u8>(bytes + 512) }.unwrap();
            for range in [0..256, bytes + 256..bytes + 512] {
                device
                    .stream
                    .memcpy_htod(&vec![0xa5u8; 256], &mut workspace.slice_mut(range))
                    .unwrap();
            }
            let (base, guard) = workspace.device_ptr(&device.stream);
            let address = base + 256;
            for (plan, x, w, out) in groups(&expected) {
                unsafe { reference.gemm_bf16(plan, true, x, w, out, false, address) }.unwrap();
            }
            let expected_bits = expected.read_storage_bytes().unwrap();
            let poison = expected_bits.iter().map(|b| b ^ 0xff).collect::<Vec<_>>();
            let mut actual =
                CudaValue::from_dense_bytes(device.clone(), vec![total, n], DType::BF16, &poison)
                    .unwrap();
            if phase == 0 && pattern == 0 {
                assert!(unsafe {
                    device.grouped_gemm_bf16(
                        &groups(&actual),
                        address,
                        true,
                        bytes - EXPERT_SPLITK_DESCRIPTOR_BYTES,
                    )
                }
                .is_err());
                FAIL_EXPERT_SUBMISSION_AFTER_MERGED.with(|flag| flag.set(true));
                assert_eq!(
                    unsafe { device.grouped_gemm_bf16(&groups(&actual), address, true, bytes) }
                        .unwrap_err(),
                    "injected expert submission failure after merged launch"
                );
                FAIL_EXPERT_SUBMISSION_BEFORE_JOIN.with(|flag| flag.set(true));
                assert_eq!(
                    unsafe { device.grouped_gemm_bf16(&groups(&actual), address, true, bytes) }
                        .unwrap_err(),
                    "injected expert submission failure before join"
                );
            }
            actual =
                CudaValue::from_dense_bytes(device.clone(), vec![total, n], DType::BF16, &poison)
                    .unwrap();
            let events =
                unsafe { device.grouped_gemm_bf16(&groups(&actual), address, true, bytes) }
                    .unwrap();
            let actual_bits = actual.read_storage_bytes().unwrap();
            assert_eq!(actual_bits.len(), expected_bits.len());
            assert!(
                actual_bits == expected_bits,
                "raw BF16 mismatch phase={phase} pattern={pattern}"
            );
            for range in [0..256, bytes + 256..bytes + 512] {
                assert_eq!(
                    device.stream.clone_dtoh(&workspace.slice(range)).unwrap(),
                    vec![0xa5u8; 256]
                );
            }
            drop((events, guard));
            drop((workspace, x, w));
            retained.push((actual, expected_bits));
            for (output, expected) in &retained {
                assert!(
                    output.read_storage_bytes().unwrap() == *expected,
                    "retained output changed"
                );
            }
        }
    }
}

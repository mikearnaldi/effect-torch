use crate::executable::CudaKernelArgs;

// Uses existing executable::CudaKernelArgs in integration. No environment reads
// at launch time: capability flag participates in the executable fingerprint.
pub(crate) fn typed_binary_kernel(
    name: &str,
    args: &CudaKernelArgs,
    enabled: bool,
) -> Option<&'static str> {
    if !enabled
        || name != "et_binary"
        || args.elements == 0
        || args.compute_dtype != 1
        || args.operation > 5
        || args.integers[0] != 0
        || args.integers[1] != 0
        || args.integers[2] != 1
        || !matches!(args.integers[3], 1 | 2)
    {
        return None;
    }
    match (
        args.operation,
        args.input_dtypes[0],
        args.input_dtypes[1],
        args.output_dtype,
        args.integers[3],
    ) {
        (0, 3, 3, 3, 1) => Some("et_binary_fixed_0_3_3_3_1"),
        (1, 3, 3, 3, 1) => Some("et_binary_fixed_1_3_3_3_1"),
        (2, 3, 3, 3, 1) => Some("et_binary_fixed_2_3_3_3_1"),
        (3, 3, 3, 3, 1) => Some("et_binary_fixed_3_3_3_3_1"),
        (4, 3, 3, 3, 1) => Some("et_binary_fixed_4_3_3_3_1"),
        (5, 3, 3, 3, 1) => Some("et_binary_fixed_5_3_3_3_1"),
        (0, 3, 3, 3, 2) => Some("et_binary_fixed_0_3_3_3_2"),
        (1, 3, 3, 3, 2) => Some("et_binary_fixed_1_3_3_3_2"),
        (2, 3, 3, 3, 2) => Some("et_binary_fixed_2_3_3_3_2"),
        (3, 3, 3, 3, 2) => Some("et_binary_fixed_3_3_3_3_2"),
        (4, 3, 3, 3, 2) => Some("et_binary_fixed_4_3_3_3_2"),
        (5, 3, 3, 3, 2) => Some("et_binary_fixed_5_3_3_3_2"),
        (0, 1, 1, 1, 1) => Some("et_binary_fixed_0_1_1_1_1"),
        (1, 1, 1, 1, 1) => Some("et_binary_fixed_1_1_1_1_1"),
        (2, 1, 1, 1, 1) => Some("et_binary_fixed_2_1_1_1_1"),
        (3, 1, 1, 1, 1) => Some("et_binary_fixed_3_1_1_1_1"),
        (4, 1, 1, 1, 1) => Some("et_binary_fixed_4_1_1_1_1"),
        (5, 1, 1, 1, 1) => Some("et_binary_fixed_5_1_1_1_1"),
        (0, 1, 1, 1, 2) => Some("et_binary_fixed_0_1_1_1_2"),
        (1, 1, 1, 1, 2) => Some("et_binary_fixed_1_1_1_1_2"),
        (2, 1, 1, 1, 2) => Some("et_binary_fixed_2_1_1_1_2"),
        (3, 1, 1, 1, 2) => Some("et_binary_fixed_3_1_1_1_2"),
        (4, 1, 1, 1, 2) => Some("et_binary_fixed_4_1_1_1_2"),
        (5, 1, 1, 1, 2) => Some("et_binary_fixed_5_1_1_1_2"),
        (0, 3, 1, 1, 1) => Some("et_binary_fixed_0_3_1_1_1"),
        (1, 3, 1, 1, 1) => Some("et_binary_fixed_1_3_1_1_1"),
        (2, 3, 1, 1, 1) => Some("et_binary_fixed_2_3_1_1_1"),
        (3, 3, 1, 1, 1) => Some("et_binary_fixed_3_3_1_1_1"),
        (4, 3, 1, 1, 1) => Some("et_binary_fixed_4_3_1_1_1"),
        (5, 3, 1, 1, 1) => Some("et_binary_fixed_5_3_1_1_1"),
        (0, 3, 1, 1, 2) => Some("et_binary_fixed_0_3_1_1_2"),
        (1, 3, 1, 1, 2) => Some("et_binary_fixed_1_3_1_1_2"),
        (2, 3, 1, 1, 2) => Some("et_binary_fixed_2_3_1_1_2"),
        (3, 3, 1, 1, 2) => Some("et_binary_fixed_3_3_1_1_2"),
        (4, 3, 1, 1, 2) => Some("et_binary_fixed_4_3_1_1_2"),
        (5, 3, 1, 1, 2) => Some("et_binary_fixed_5_3_1_1_2"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_exactly_matches_approved_dtype_operation_geometry_matrix() {
        // Exhaustive dispatch table independent of selector match arm names.
        for op in 0..12 {
            for left in 0..8 {
                for right in 0..8 {
                    for out in 0..8 {
                        for left_mode in 0..6 {
                            for right_mode in 0..6 {
                                let mut a = CudaKernelArgs {
                                    elements: 129,
                                    compute_dtype: 1,
                                    operation: op,
                                    output_dtype: out,
                                    ..Default::default()
                                };
                                a.input_dtypes[..2].copy_from_slice(&[left, right]);
                                a.integers[2] = left_mode;
                                a.integers[3] = right_mode;
                                let expected = op <= 5
                                    && left_mode == 1
                                    && matches!(right_mode, 1 | 2)
                                    && matches!(
                                        (left, right, out),
                                        (3, 3, 3) | (1, 1, 1) | (3, 1, 1)
                                    );
                                let actual = typed_binary_kernel("et_binary", &a, true);
                                assert_eq!(actual.is_some(), expected);
                                if expected {
                                    assert_eq!(
                                        actual.unwrap(),
                                        format!(
                                            "et_binary_fixed_{op}_{left}_{right}_{out}_{right_mode}"
                                        )
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_coercion_compute_and_disabled_paths_stay_generic() {
        let mut base = CudaKernelArgs {
            elements: 129,
            compute_dtype: 1,
            output_dtype: 3,
            ..Default::default()
        };
        base.input_dtypes[..2].copy_from_slice(&[3, 3]);
        base.integers[2] = 1;
        base.integers[3] = 1;
        assert!(typed_binary_kernel("et_binary", &base, true).is_some());
        assert!(typed_binary_kernel("et_binary", &base, false).is_none());
        assert!(typed_binary_kernel("et_convert", &base, true).is_none());
        for dtype in [0, 2, 3, 4, 5, 6, 7] {
            let mut a = base;
            a.compute_dtype = dtype;
            assert!(typed_binary_kernel("et_binary", &a, true).is_none());
        }
        for role in 0..2 {
            for target in 1..10 {
                let mut a = base;
                a.integers[role] = target;
                assert!(typed_binary_kernel("et_binary", &a, true).is_none());
            }
        }
        let mut empty = base;
        empty.elements = 0;
        assert!(typed_binary_kernel("et_binary", &empty, true).is_none());
        // Shapes/tails are intentionally not keys; byte-offset input pointers are untouched.
        for count in [1, 127, 128, 129, 16384, 720896] {
            let mut a = base;
            a.elements = count;
            assert!(typed_binary_kernel("et_binary", &a, true).is_some());
        }
    }
}

#[cfg(test)]
#[path = "typed_binary_tests.rs"]
mod hardware_tests;

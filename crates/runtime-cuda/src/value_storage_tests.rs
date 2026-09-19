use super::{dense_bytes_from_host, element_count, validate_storage_bytes};
use effect_torch_runtime::{DType, GgmlKQuant, PackedFormat, StorageMetadata, ValueSpec};

#[test]
fn dense_storage_uses_exact_scalar_widths() {
    for (dtype, width) in [
        (DType::F64, 8),
        (DType::F32, 4),
        (DType::F16, 2),
        (DType::BF16, 2),
        (DType::I64, 8),
        (DType::U32, 4),
        (DType::U8, 1),
    ] {
        let spec = ValueSpec::dense(dtype, &[2, 3]);
        assert_eq!(validate_storage_bytes(spec, 6 * width), Ok(()));
        assert!(validate_storage_bytes(spec, 6 * width + 1).is_err());
        assert!(validate_storage_bytes(spec, 6 * width - 1).is_err());
        if width < 8 {
            assert!(validate_storage_bytes(spec, 6 * 8).is_err());
        }
        assert_eq!(dense_bytes_from_host(&[1.0; 6], dtype).len(), 6 * width);
    }
}

#[test]
fn zero_extent_short_circuits_overflow() {
    assert_eq!(element_count(&[usize::MAX, usize::MAX, 0]), Ok(0));
    assert_eq!(
        validate_storage_bytes(
            ValueSpec::dense(DType::F64, &[usize::MAX, usize::MAX, 0]),
            0
        ),
        Ok(())
    );
    assert!(element_count(&[usize::MAX, 2]).is_err());
    assert!(validate_storage_bytes(ValueSpec::dense(DType::F64, &[usize::MAX]), 0).is_err());
}

#[test]
fn packed_storage_keeps_logical_f32_geometry() {
    for (codec, block_bytes) in [
        (GgmlKQuant::Q2K, 84),
        (GgmlKQuant::Q3K, 110),
        (GgmlKQuant::Q4K, 144),
        (GgmlKQuant::Q5K, 176),
        (GgmlKQuant::Q6K, 210),
    ] {
        let metadata = StorageMetadata::packed(codec);
        let spec = ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &[3, 512],
            storage: metadata.as_spec(),
        };
        assert_eq!(validate_storage_bytes(spec, 6 * block_bytes), Ok(()));
        let physical = spec.canonical_geometry().unwrap();
        assert_eq!(physical.physical_shape, [3, 2 * block_bytes]);
        assert_eq!(physical.physical_dtype, DType::U8);
        assert_eq!(spec.packed_matrix().unwrap(), (codec, [3, 512]));
        assert!(validate_storage_bytes(spec, 3 * 512 * 4).is_err());
        assert!(validate_storage_bytes(
            ValueSpec {
                semantic_dtype: DType::U8,
                ..spec
            },
            6 * block_bytes
        )
        .is_err());
        assert!(validate_storage_bytes(
            ValueSpec {
                logical_shape: &[3, 255],
                ..spec
            },
            0
        )
        .is_err());
        assert!(validate_storage_bytes(
            ValueSpec {
                logical_shape: &[usize::MAX, 512],
                ..spec
            },
            0
        )
        .is_err());
        assert!(validate_storage_bytes(
            ValueSpec {
                logical_shape: &[],
                ..spec
            },
            0
        )
        .is_err());
    }
    for format in ["Q4", "GGML", "NVFP4", "q4_k", ""] {
        assert!(PackedFormat::from_name(format).is_err());
    }
}

#[test]
fn host_conversion_rounds_half_storage_and_keeps_signed_zero() {
    assert_eq!(
        dense_bytes_from_host(&[1.0, -0.0, 1.0006], DType::F16),
        [0x00, 0x3c, 0x00, 0x80, 0x01, 0x3c]
    );
    assert_eq!(
        dense_bytes_from_host(&[1.0, -0.0, 1.005], DType::BF16),
        [0x80, 0x3f, 0x00, 0x80, 0x81, 0x3f]
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn device_storage_roundtrip_and_typed_views() {
    use super::CudaValue;
    let device = crate::CudaDevice::get(0).unwrap();
    for dtype in [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::U32,
        DType::U8,
    ] {
        let bytes = (0..3 * dtype.size_in_bytes())
            .map(|index| (index * 17) as u8)
            .collect::<Vec<_>>();
        let value = CudaValue::from_dense_bytes(device.clone(), vec![3], dtype, &bytes).unwrap();
        assert_eq!(value.storage_bytes(), bytes.len());
        assert_eq!(value.read_storage_bytes().unwrap(), bytes);
        assert!(value.write_storage_bytes(&[]).is_err());
        assert_eq!(value.read_storage_bytes().unwrap(), bytes);
        assert_eq!(value.typed_buffer::<f64>().is_ok(), dtype == DType::F64);
        assert_eq!(value.typed_buffer::<f32>().is_ok(), dtype == DType::F32);
        assert_eq!(
            value.typed_buffer::<half::f16>().is_ok(),
            dtype == DType::F16
        );
        assert_eq!(
            value.typed_buffer::<half::bf16>().is_ok(),
            dtype == DType::BF16
        );
        assert_eq!(value.typed_buffer::<i64>().is_ok(), dtype == DType::I64);
        assert_eq!(value.typed_buffer::<u32>().is_ok(), dtype == DType::U32);
        assert_eq!(value.typed_buffer::<u8>().is_ok(), dtype == DType::U8);
        assert!(value.reshape_dense(vec![4]).is_err());
        let reshaped = value.reshape_dense(vec![1, 3]).unwrap();
        assert_eq!(reshaped.storage_address(), value.storage_address());
        drop(value);
        assert_eq!(reshaped.read_storage_bytes().unwrap(), bytes);
    }
    for (dtype, exponent, positive_bits) in
        [(DType::F16, -11, 0x3c01u16), (DType::BF16, -8, 0x3f81)]
    {
        let value = 1.0 + 2.0f64.powi(exponent) + 2.0f64.powi(-40);
        let tensor =
            CudaValue::from_host(device.clone(), vec![2], dtype, &[value, -value]).unwrap();
        let expected = [positive_bits, positive_bits | 0x8000]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(tensor.storage_bytes(), 4);
        assert_eq!(tensor.read_storage_bytes().unwrap(), expected);
        tensor
            .write_storage_bytes(&dense_bytes_from_host(&[-value, value], dtype))
            .unwrap();
        let expected = [positive_bits | 0x8000, positive_bits]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(tensor.read_storage_bytes().unwrap(), expected);
    }
    let bytes = vec![0u8; 144];
    let packed = CudaValue::from_packed_bytes(
        device,
        vec![1, 256],
        PackedFormat::GgmlKQuant(GgmlKQuant::Q4K),
        &bytes,
    )
    .unwrap();
    assert_eq!(packed.shape(), [1, 256]);
    assert_eq!(packed.dtype(), DType::F32);
    assert_eq!(packed.read_storage_bytes().unwrap(), bytes);
    assert!(packed.typed_buffer::<f32>().is_err());
    assert!(packed.typed_buffer::<u8>().is_err());
    assert!(packed.readback().is_err());
    assert!(packed.reshape_dense(vec![256]).is_err());
}

#[test]
#[ignore = "requires a CUDA device"]
fn planned_views_keep_owner_and_retention_until_last_handle_drops() {
    use super::CudaValue;
    use crate::buffer::CudaBuffer;
    use effect_torch_graph::LeafSlot;
    use std::sync::Arc;

    let device = crate::CudaDevice::get(0).unwrap();
    let owner = Arc::new(device.stream.clone_htod(&[0x17u8; 32]).unwrap());
    let retention = Arc::new(());
    let buffer =
        CudaBuffer::<u8>::from_segment(owner.clone(), 8, 8, Some(retention.clone())).unwrap();
    assert!(buffer.cast::<u32>(1).is_err());
    assert!(buffer.cast::<u32>(3).is_err());
    assert_eq!(buffer.cast::<u32>(2).unwrap().len(), 2);
    assert!(CudaBuffer::<u8>::from_segment(owner.clone(), 1, 8, None)
        .unwrap()
        .cast::<f64>(1)
        .is_err());
    let misaligned = CudaBuffer::<u8>::from_segment(owner.clone(), 1, 8, None).unwrap();
    assert!(CudaValue::from_planned_buffer(
        device.clone(),
        ValueSpec::dense(DType::F64, &[1]),
        misaligned
    )
    .is_err());
    let value =
        CudaValue::from_planned_buffer(device, ValueSpec::dense(DType::F64, &[1]), buffer).unwrap();
    let slot = LeafSlot::new(value);
    let retained = slot
        .get::<CudaValue>()
        .unwrap()
        .reshape_dense(vec![1, 1])
        .unwrap();
    drop(owner);
    assert!(slot.clear());
    assert!(!slot.clear());
    assert_eq!(retained.read_storage_bytes().unwrap(), [0x17; 8]);
    assert_eq!(Arc::strong_count(&retention), 2);
    drop(retained);
    assert_eq!(Arc::strong_count(&retention), 1);
}

#[test]
fn direct_f64_half_conversion_keeps_sticky_bits() {
    let bf16_value = 1.0 + 2.0f64.powi(-8) + 2.0f64.powi(-40);
    assert_eq!(
        dense_bytes_from_host(&[bf16_value, -bf16_value], DType::BF16),
        [0x81, 0x3f, 0x81, 0xbf]
    );
    let f16_value = 1.0 + 2.0f64.powi(-11) + 2.0f64.powi(-40);
    assert_eq!(
        dense_bytes_from_host(&[f16_value, -f16_value], DType::F16),
        [0x01, 0x3c, 0x01, 0xbc]
    );
}

#[test]
fn host_half_conversion_rounds_every_finite_midpoint_once() {
    fn convert(value: f64, dtype: DType) -> u16 {
        u16::from_le_bytes(dense_bytes_from_host(&[value], dtype).try_into().unwrap())
    }
    for (dtype, last_finite, infinity, overflow_midpoint) in [
        (DType::F16, 0x7bffu16, 0x7c00, 65520.0),
        (
            DType::BF16,
            0x7f7fu16,
            0x7f80,
            half::bf16::MAX.to_f64() + 2.0f64.powi(119),
        ),
    ] {
        let decode = |bits| match dtype {
            DType::F16 => half::f16::from_bits(bits).to_f64(),
            _ => half::bf16::from_bits(bits).to_f64(),
        };
        for lower in 0..last_finite {
            let midpoint = (decode(lower) + decode(lower + 1)) * 0.5;
            let below = f64::from_bits(midpoint.to_bits() - 1);
            let above = f64::from_bits(midpoint.to_bits() + 1);
            let even = if lower & 1 == 0 { lower } else { lower + 1 };
            for (value, expected) in [(below, lower), (midpoint, even), (above, lower + 1)] {
                assert_eq!(
                    convert(value, dtype),
                    expected,
                    "{dtype} midpoint above {lower:#06x}, value {value:?}"
                );
                assert_eq!(convert(-value, dtype), expected | 0x8000);
            }
        }
        for (value, expected) in [
            (0.0, 0),
            (f64::from_bits(1), 0),
            (f64::from_bits(overflow_midpoint.to_bits() - 1), last_finite),
            (overflow_midpoint, infinity),
            (f64::INFINITY, infinity),
        ] {
            assert_eq!(convert(value, dtype), expected);
            assert_eq!(convert(-value, dtype), expected | 0x8000);
        }
        for bits in [0x7ff0000000000001, 0xfff8000000001234] {
            let encoded = convert(f64::from_bits(bits), dtype);
            assert_eq!(encoded & infinity, infinity);
            assert_ne!(encoded & !(infinity | 0x8000), 0);
            assert_eq!(encoded & 0x8000, ((bits >> 48) & 0x8000) as u16);
        }
    }
}

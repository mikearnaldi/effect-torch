use super::*;

fn fixture(dtype: DType) -> (NativeKvPool, NativeKvSequence, CudaKvSnapshot, BlockKey) {
    let tokens = vec![1u32; 16];
    let key = BlockKey::new(format!(
        "full:16:{}",
        tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    ));
    let snapshot = CudaKvSnapshot {
        dtype,
        keys: vec![(0..64 * dtype.size_in_bytes())
            .map(|index| (index * 17) as u8)
            .collect()],
        values: vec![(0..64 * dtype.size_in_bytes())
            .map(|index| (index * 29) as u8)
            .collect()],
        key_scales: if dtype == DType::U8 {
            vec![vec![0.125; 32]]
        } else {
            Vec::new()
        },
        value_scales: if dtype == DType::U8 {
            vec![vec![0.25; 32]]
        } else {
            Vec::new()
        },
    };
    let state = SequenceState {
        cursor: 16,
        tokens,
        kv_storage: Some(snapshot.clone()),
        kda_states: vec![vec![1.0, f32::from_bits(0x80000000)]],
        conv_states: vec![vec![f32::from_bits(0x7f800123), 2.0]],
        block_keys: vec![key.clone()],
    };
    let inner = Arc::new(PoolInner {
        ordinal: 0,
        layers: 1,
        kv_heads: 1,
        head_dim: 2,
        max_tokens: 32,
        block_size: 16,
        dtype,
        recurrent: NativeRecurrentStateSchema {
            kda_layers: 1,
            kda_heads: 1,
            kda_head_dim: 1,
            kda_value_dim: 2,
            conv_layers: 1,
            conv_channels: 2,
            conv_kernel: 2,
        },
        usage: Mutex::new(PoolUsage {
            blocks: HashMap::from([(key.clone(), 1)]),
            snapshots: HashMap::from([(key.clone(), Arc::new(state.clone()))]),
        }),
    });
    let sequence = NativeKvSequence {
        inner: Arc::new(SequenceInner {
            pool: inner.clone(),
            state: Mutex::new(state),
            device_cache: Mutex::new(None),
            released: AtomicBool::new(false),
            running: AtomicBool::new(false),
        }),
    };
    (NativeKvPool { inner }, sequence, snapshot, key)
}

fn assert_snapshot(sequence: &NativeKvSequence, expected: &CudaKvSnapshot) {
    let state = sequence.inner.state.lock().unwrap();
    let actual = state.kv_storage.as_ref().unwrap();
    assert_eq!(actual.dtype, expected.dtype);
    assert_eq!(actual.keys, expected.keys);
    assert_eq!(actual.values, expected.values);
    assert_eq!(actual.key_scales, expected.key_scales);
    assert_eq!(actual.value_scales, expected.value_scales);
    assert_eq!(state.kda_states[0][1].to_bits(), 0x80000000);
    assert_eq!(state.conv_states[0][0].to_bits(), 0x7f800123);
}

#[test]
fn fork_and_prefix_restore_preserve_exact_snapshot_bits() {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
        let (pool, source, snapshot, key) = fixture(dtype);
        let fork = source.fork().unwrap();
        assert_snapshot(&fork, &snapshot);
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 2);
        source.release();
        source.release();
        assert!(source.inner.state.lock().unwrap().kv_storage.is_none());
        assert_snapshot(&fork, &snapshot);
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 1);
        let restored = pool.make_sequence();
        assert_eq!(restored.prefill_match(vec![1; 17]).unwrap(), 16);
        assert_snapshot(&restored, &snapshot);
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 2);
        fork.release();
        restored.release();
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 0);
    }
}

#[test]
fn leased_and_released_sequences_reject_snapshot_operations() {
    let (_, sequence, snapshot, _) = fixture(DType::U8);
    let lease = sequence.lease().unwrap();
    assert!(sequence.fork().is_err());
    assert!(sequence.prefill_match(vec![1; 17]).is_err());
    assert_snapshot(&sequence, &snapshot);
    drop(lease);
    let fork = sequence.fork().unwrap();
    fork.release();
    sequence.release();
    assert!(sequence.fork().is_err());
    assert!(sequence.prefill_match(vec![1; 17]).is_err());
}

#[test]
#[ignore = "requires a CUDA device"]
fn device_fork_copies_exact_cache_bytes_and_scales() {
    use crate::buffer::CudaBuffer;
    let device = CudaDevice::get(0).unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
        let (_, source, snapshot, _) = fixture(dtype);
        let upload_bytes = |bytes: &[u8]| {
            Arc::new(CudaBuffer::from_slice(
                device.stream.clone_htod(bytes).unwrap(),
            ))
        };
        let upload_scales = |scales: &[f32]| {
            Arc::new(CudaBuffer::from_slice(
                device.stream.clone_htod(scales).unwrap(),
            ))
        };
        let keys = upload_bytes(&snapshot.keys[0]);
        let original_address = keys.address();
        *source.inner.device_cache.lock().unwrap() = Some(SequenceDeviceCache {
            kv: CudaKvCache {
                keys,
                values: upload_bytes(&snapshot.values[0]),
                key_scales: snapshot
                    .key_scales
                    .first()
                    .map(|scales| upload_scales(scales)),
                value_scales: snapshot
                    .value_scales
                    .first()
                    .map(|scales| upload_scales(scales)),
                layer_elements: 64,
                dtype,
            },
            cursor: None,
            valid: None,
            decode_graph: None,
        });
        let fork = source.fork().unwrap();
        source.release();
        {
            let guard = fork.inner.device_cache.lock().unwrap();
            let cache = &guard.as_ref().unwrap().kv;
            assert_ne!(cache.keys.address(), original_address);
            assert_eq!(cache.keys.len(), 64 * dtype.size_in_bytes());
            assert_eq!(cache.values.len(), 64 * dtype.size_in_bytes());
            assert_eq!(
                device.stream.clone_dtoh(cache.keys.as_ref()).unwrap(),
                snapshot.keys[0]
            );
            assert_eq!(
                device.stream.clone_dtoh(cache.values.as_ref()).unwrap(),
                snapshot.values[0]
            );
            if dtype == DType::U8 {
                assert_eq!(
                    device
                        .stream
                        .clone_dtoh(cache.key_scales.as_ref().unwrap().as_ref())
                        .unwrap(),
                    snapshot.key_scales[0]
                );
                assert_eq!(
                    device
                        .stream
                        .clone_dtoh(cache.value_scales.as_ref().unwrap().as_ref())
                        .unwrap(),
                    snapshot.value_scales[0]
                );
            } else {
                assert!(cache.key_scales.is_none());
                assert!(cache.value_scales.is_none());
            }
        }
        fork.release();
    }
}

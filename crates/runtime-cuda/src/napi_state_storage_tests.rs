use super::*;
use crate::executable::CudaKvLayer;
use effect_torch_graph::AttentionRounding;

#[test]
fn exported_accounting_deduplicates_slots_and_releases_clone_groups_once() {
    let owner = Arc::new(());
    let identity = Arc::as_ptr(&owner) as usize;
    let baseline = external_memory_bytes();
    let original = Arc::new(ExportedTensorAccounting::new((identity, 4096)));
    let derived_clone = original.clone();
    let retained_slot = ExportedTensorAccounting::new((identity, 4096));
    assert_eq!(external_memory_bytes(), baseline + 4096.0);
    drop(original);
    assert_eq!(external_memory_bytes(), baseline + 4096.0);
    derived_clone.release();
    derived_clone.release();
    drop(derived_clone);
    assert_eq!(external_memory_bytes(), baseline + 4096.0);
    drop(retained_slot);
    assert_eq!(external_memory_bytes(), baseline);
}

#[test]
#[ignore = "requires a CUDA device"]
fn device_tensor_retain_and_alias_clear_account_for_backing_allocations_once() {
    use crate::buffer::CudaBuffer;
    let device = CudaDevice::get(0).unwrap();
    let baseline = external_memory_bytes();
    let owner = Arc::new(device.stream.clone_htod(&[0u8; 128]).unwrap());
    let view = |offset| {
        CudaValue::from_planned_buffer(
            device.clone(),
            ValueSpec::dense(DType::F32, &[2]),
            CudaBuffer::from_segment(owner.clone(), offset, 8, None).unwrap(),
        )
        .unwrap()
    };
    let original = NativeTensor::wrap(view(0));
    let original_clone = original.clone();
    let retained = original.retain().unwrap();
    let other_view = NativeTensor::wrap(view(16));
    assert!(!Arc::ptr_eq(&original.slot, &retained.slot));
    assert!(Arc::ptr_eq(&original.slot, &original_clone.slot));
    assert_eq!(external_memory_bytes(), baseline + 128.0);
    original.clear();
    original.clear();
    assert!(original_clone.value().is_err());
    assert!(original.retain().is_err());
    assert_eq!(retained.value().unwrap().readback().unwrap(), vec![0.0; 2]);
    assert_eq!(external_memory_bytes(), baseline + 128.0);
    drop(original);
    drop(original_clone);
    drop(retained);
    assert_eq!(external_memory_bytes(), baseline + 128.0);
    other_view.clear();
    assert_eq!(external_memory_bytes(), baseline);
    drop(other_view);
    // An unexported CUDA owner is excluded even while its allocation remains alive.
    assert_eq!(external_memory_bytes(), baseline);
    let drop_only = NativeTensor::wrap(view(32));
    let clone = drop_only.clone();
    assert_eq!(external_memory_bytes(), baseline + 128.0);
    drop(drop_only);
    assert_eq!(external_memory_bytes(), baseline + 128.0);
    drop(clone);
    assert_eq!(external_memory_bytes(), baseline);
}

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
    // Host-only fixtures retain no device rows. Recurrent bytes and page descriptors
    // still exercise the ownership and prefix-cache paths without loading CUDA.
    let descriptor = KvLayerDescriptor {
        layer_id: 0,
        kv_heads: 1,
        head_dim: 2,
        dtype,
        retention: Some(0),
    };
    let snapshot = CudaKvSnapshot {
        layers: vec![CudaKvLayer {
            descriptor,
            start_position: 16,
            pages: Vec::new(),
        }],
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
        max_tokens: 32,
        block_size: 16,
        kv_layers: vec![descriptor],
        explicit_layers: true,
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
            released: AtomicBool::new(false),
            running: AtomicBool::new(false),
        }),
    };
    (NativeKvPool { inner }, sequence, snapshot, key)
}

fn assert_snapshot(sequence: &NativeKvSequence, expected: &CudaKvSnapshot) {
    let state = sequence.inner.state.lock().unwrap();
    let actual = state.kv_storage.as_ref().unwrap();
    assert_eq!(actual.layers.len(), expected.layers.len());
    assert_eq!(actual.layers[0].descriptor, expected.layers[0].descriptor);
    assert_eq!(
        actual.layers[0].start_position,
        expected.layers[0].start_position
    );
    assert!(actual.layers[0].pages.is_empty());
    assert_eq!(state.kda_states[0][1].to_bits(), 0x80000000);
    assert_eq!(state.conv_states[0][0].to_bits(), 0x7f800123);
}

#[test]
fn fork_snapshot_and_prefix_restore_preserve_metadata_and_recurrent_bits() {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
        let (pool, source, snapshot, key) = fixture(dtype);
        let prefix = source.snapshot().unwrap();
        let fork = prefix.fork().unwrap();
        assert_snapshot(&fork, &snapshot);
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 3);
        source.release();
        source.release();
        assert!(source.inner.state.lock().unwrap().kv_storage.is_none());
        assert_snapshot(&fork, &snapshot);
        let borrowed = prefix.borrowed().unwrap();
        prefix.release();
        assert!(prefix.fork().is_err());
        assert_eq!(borrowed.state.cursor, 16);
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 2);
        drop(borrowed);
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 1);
        let restored = pool.make_sequence();
        assert_eq!(restored.prefill_match(vec![1; 17]).unwrap(), 16);
        assert_snapshot(&restored, &snapshot);
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 2);
        drop(fork);
        restored.release();
        assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 0);
    }
}

#[test]
fn leased_release_defers_cleanup_until_the_borrow_finishes() {
    let (pool, sequence, snapshot, key) = fixture(DType::U8);
    let lease = sequence.lease().unwrap();
    assert!(sequence.fork().is_err());
    assert!(sequence.snapshot().is_err());
    assert!(sequence.prefill_match(vec![1; 17]).is_err());
    sequence.release();
    assert_snapshot(&sequence, &snapshot);
    assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 1);
    drop(lease);
    assert!(sequence.inner.state.lock().unwrap().kv_storage.is_none());
    assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 0);
    assert!(sequence.fork().is_err());
}

#[test]
fn released_prefix_getters_reject_and_empty_inspection_never_loads_cuda() {
    let (_, sequence, _, _) = fixture(DType::BF16);
    let prefix = sequence.snapshot().unwrap();
    let inspection = prefix.inspect().unwrap();
    assert_eq!(inspection.cursor, 16);
    assert_eq!(inspection.layers[0].start_position, 16);
    assert_eq!(inspection.retained_bytes, 0.0);
    assert!(inspection.layers[0].keys.is_empty());
    prefix.release();
    assert!(prefix.cursor().is_err());
    assert!(prefix.retained_bytes().is_err());
    assert!(prefix.shared_bytes().is_err());
    assert!(prefix.copied_bytes().is_err());
    assert!(prefix.inspect().is_err());
}

fn schema_for(pool: &PoolInner) -> CudaStateSchema {
    CudaStateSchema {
        max_tokens: pool.max_tokens,
        block_size: pool.block_size,
        kv_dtype: pool.kv_layers[0].dtype,
        batch: 1,
        packed_rows_per_sequence: None,
        access: StateAccessMode::Append,
        geometry: DecodeGeometry {
            kv_layers: pool.kv_layers.clone(),
            layers: 1,
            kv_heads: 1,
            head_dim: 2,
            allows_window_eviction: true,
            cursor_slot: 0,
            cursor_tensor: false,
            kda: effect_torch_compiler::KdaGeometry {
                layers: 1,
                heads: 1,
                head_dim: 1,
                value_dim: 2,
                dtype: DType::F32,
            },
            conv: effect_torch_compiler::ConvGeometry {
                layers: 1,
                channels: 2,
                kernel: 2,
            },
        },
    }
}

#[test]
fn descriptor_validation_checks_retention_dtype_capacity_and_device() {
    let (pool, _, _, _) = fixture(DType::F32);
    let schema = schema_for(&pool.inner);
    schema.validate_pool(&pool.inner, 0).unwrap();
    assert!(schema.validate_pool(&pool.inner, 1).is_err());
    let mut changed = schema.clone();
    changed.max_tokens += 16;
    assert!(changed.validate_pool(&pool.inner, 0).is_err());
    let mut changed = schema.clone();
    changed.block_size = 8;
    assert!(changed.validate_pool(&pool.inner, 0).is_err());
    for layer in [
        KvLayerDescriptor {
            retention: None,
            ..schema.geometry.kv_layers[0]
        },
        KvLayerDescriptor {
            dtype: DType::F16,
            ..schema.geometry.kv_layers[0]
        },
        KvLayerDescriptor {
            head_dim: 3,
            ..schema.geometry.kv_layers[0]
        },
    ] {
        let mut changed = schema.clone();
        changed.geometry.kv_layers[0] = layer;
        assert!(changed.validate_pool(&pool.inner, 0).is_err());
    }
    let mut legacy = schema.clone();
    legacy.geometry.kv_layers[0].retention = None;
    let (legacy_pool, legacy_sequence, _, _) = fixture(DType::F32);
    drop(legacy_sequence);
    let mut legacy_pool = Arc::try_unwrap(legacy_pool.inner).ok().unwrap();
    legacy_pool.explicit_layers = false;
    legacy.validate_pool(&legacy_pool, 0).unwrap();
    legacy.geometry.kv_layers[0].head_dim += 1;
    assert!(legacy.validate_pool(&legacy_pool, 0).is_err());
}

#[test]
fn failed_multi_step_publication_restores_sequence_and_block_references() {
    let (pool, sequence, _, key) = fixture(DType::F32);
    let schema = schema_for(&pool.inner);
    let leases = vec![sequence.lease().unwrap()];
    let mut invocation = CudaStateInvocation {
        sequences: Vec::new(),
        slots: vec![0],
        valid_lengths: vec![1],
        capacity: 32,
        cache_dtype: DType::F32,
        packed_rows_per_sequence: None,
        kv_layers: schema.geometry.kv_layers.clone(),
        access: StateAccessMode::Append,
        cache: None,
    };
    let result: Result<()> =
        Executable::with_state_transaction(&leases, &[0], &mut invocation, |_| {
            let mut state = sequence.inner.state.lock().unwrap();
            state.cursor = 23;
            state.kda_states[0][0] = 99.0;
            state.block_keys.clear();
            drop(state);
            pool.inner
                .usage
                .lock()
                .unwrap()
                .blocks
                .insert(key.clone(), 0);
            Err(Error::new(
                Status::Cancelled,
                "operation aborted after a completed step",
            ))
        });
    assert!(result.is_err());
    assert_eq!(sequence.cursor(), 16);
    assert_eq!(pool.inner.usage.lock().unwrap().blocks[&key], 1);
    assert_eq!(sequence.inner.state.lock().unwrap().kda_states[0][0], 1.0);
}

fn gpu_program(
    dtype: DType,
    kv_dtype: DType,
    access: &str,
    time: usize,
    capacity: u32,
    retention: Option<u32>,
    bad_gather: bool,
) -> Executable {
    let input = |slot| {
        Node::new(NodeKind::Input {
            slot,
            shape: vec![1, 1, time, 1],
            dtype,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    };
    let mut root = Node::new(NodeKind::SdpaConfigured {
        q: input(0),
        k: input(1),
        v: input(2),
        scale: 0.3,
        causal: true,
        window: AttentionWindow::Full,
        rounding: AttentionRounding::Stepwise,
        layer_id: Some(0),
        retention: retention.map_or(AttentionWindow::Full, |value| {
            AttentionWindow::Local(value as usize)
        }),
    })
    .unwrap();
    if bad_gather {
        let indexes = Node::new(NodeKind::Full {
            shape: vec![1, 1, time, 1],
            value: 99.0,
            dtype: DType::I64,
            device: Device::Cuda(0),
        })
        .unwrap();
        root = Node::new(NodeKind::Gather {
            a: root,
            dim: 3,
            indexes,
        })
        .unwrap();
    }
    let runtime = CudaRuntime::new(Some(0)).unwrap();
    runtime
        .compile(
            vec![&LazyTensor { node: root }],
            Some(NativeCompileOptions {
                optimize: Some(false),
                random_seed: None,
                constant_weights: None,
            }),
            Some(NativeKvStateSchema {
                access: Some(access.to_string()),
                max_tokens: capacity,
                block_size: 2,
                kv_dtype: kv_dtype.name().to_string(),
                window: None,
                current_block_attention: Some(if access == "Append" {
                    NativeCurrentBlockAttention::Causal
                } else {
                    NativeCurrentBlockAttention::Bidirectional
                }),
                batch: 1,
                packed_causal_chains: None,
                last_token_row: None,
                output_selections: None,
            }),
        )
        .unwrap()
}

fn gpu_value(values: &[f64], dtype: DType) -> NativeTensor {
    NativeTensor::wrap(
        CudaValue::from_host(
            CudaDevice::get(0).unwrap(),
            vec![1, 1, values.len(), 1],
            dtype,
            values,
        )
        .unwrap(),
    )
}

fn gpu_pool(dtype: DType, capacity: u32, retention: Option<u32>) -> NativeKvPool {
    NativeKvPool::new(
        0,
        1,
        1,
        1,
        capacity,
        Some(2),
        Some(dtype.name().to_string()),
        None,
        Some(vec![NativeKvLayerDescriptor {
            layer_id: 0,
            kv_heads: 1,
            head_dim: 1,
            dtype: dtype.name().to_string(),
            retention_window: retention,
        }]),
    )
    .unwrap()
}

async fn gpu_append(
    program: &Executable,
    sequence: &NativeKvSequence,
    values: &[f64],
    dtype: DType,
) -> Result<Vec<NativeTensor>> {
    let q = gpu_value(&vec![1.0; values.len()], dtype);
    let k = gpu_value(values, dtype);
    let v = gpu_value(values, dtype);
    program
        .execute_stateful(
            vec![&q, &k, &v],
            vec![sequence],
            vec![0],
            vec![true],
            vec![values.len() as u32],
            vec![values.len() as u32],
            vec![vec![1; values.len()]],
            None,
        )
        .await
}

#[tokio::test]
#[ignore = "requires a CUDA device"]
async fn device_snapshots_share_pages_and_failed_append_preserves_prefix() {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::U8] {
        let pool = gpu_pool(dtype, 16, None);
        let sequence = pool.make_sequence();
        let initial = gpu_program(DType::F32, dtype, "Append", 3, 16, None, false);
        gpu_append(&initial, &sequence, &[1.0, -2.0, 3.0], DType::F32)
            .await
            .unwrap();
        let prefix = sequence.snapshot().unwrap();
        let before = prefix.inspect().unwrap();
        let child = prefix.fork().unwrap();
        {
            let source = sequence.inner.state.lock().unwrap();
            let target = child.inner.state.lock().unwrap();
            let source_page = &source.kv_storage.as_ref().unwrap().layers[0].pages[0];
            let target_page = &target.kv_storage.as_ref().unwrap().layers[0].pages[0];
            assert!(Arc::ptr_eq(source_page, target_page));
        }
        assert!(prefix.retained_bytes().unwrap() > 0.0);
        assert_eq!(
            prefix.retained_bytes().unwrap(),
            prefix.shared_bytes().unwrap()
        );
        assert_eq!(prefix.copied_bytes().unwrap(), 0.0);
        let broken = gpu_program(DType::F32, dtype, "Append", 1, 16, None, true);
        assert!(gpu_append(&broken, &child, &[9.0], DType::F32)
            .await
            .is_err());
        assert_eq!(child.cursor(), 3);
        assert_eq!(
            prefix.inspect().unwrap().layers[0].values,
            before.layers[0].values
        );
        let append = gpu_program(DType::F32, dtype, "Append", 1, 16, None, false);
        gpu_append(&append, &child, &[7.0], DType::F32)
            .await
            .unwrap();
        assert_eq!(child.cursor(), 4);
        assert_eq!(
            prefix.inspect().unwrap().layers[0].values,
            before.layers[0].values
        );
        sequence.release();
        let borrowed = prefix.borrowed().unwrap();
        prefix.release();
        let retained = NativeKvPrefix {
            inner: Mutex::new(Some(borrowed)),
        };
        assert_eq!(
            retained.inspect().unwrap().layers[0].values,
            before.layers[0].values
        );
    }
}

#[tokio::test]
#[ignore = "requires a CUDA device"]
async fn device_read_only_full_pool_is_concurrent_and_half_rounding_is_exact() {
    let pool = gpu_pool(DType::BF16, 2, None);
    let sequence = pool.make_sequence();
    let empty = sequence.snapshot().unwrap();
    let decoder = gpu_program(DType::BF16, DType::BF16, "ReadOnly", 2, 2, None, false);
    let q = gpu_value(&[1.0, 1.0], DType::BF16);
    let k = gpu_value(&[3.0, -2.0], DType::BF16);
    let v = gpu_value(&[0.0, 1.0], DType::BF16);
    let outputs = decoder
        .execute_read_only(
            vec![&q, &k, &v],
            vec![&empty],
            vec![0],
            vec![true],
            vec![2],
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        outputs[0].value().unwrap().readback().unwrap(),
        vec![0.1826171875; 2]
    );
    let encoder = gpu_program(DType::BF16, DType::BF16, "Append", 2, 2, None, false);
    gpu_append(&encoder, &sequence, &[1.0, 2.0], DType::BF16)
        .await
        .unwrap();
    let prefix = sequence.snapshot().unwrap();
    let before = prefix.inspect().unwrap();
    let other = gpu_value(&[9.0, 10.0], DType::BF16);
    let (left, right) = tokio::join!(
        decoder.execute_read_only(
            vec![&q, &k, &v],
            vec![&prefix],
            vec![0],
            vec![true],
            vec![2],
            None
        ),
        decoder.execute_read_only(
            vec![&q, &k, &other],
            vec![&prefix],
            vec![0],
            vec![true],
            vec![2],
            None
        )
    );
    let left = left.unwrap();
    let right = right.unwrap();
    assert_ne!(
        left[0].value().unwrap().readback().unwrap(),
        right[0].value().unwrap().readback().unwrap()
    );
    assert_eq!(
        prefix.inspect().unwrap().layers[0].values,
        before.layers[0].values
    );
    let cancellation = CancellationToken {
        state: Arc::new(CancellationState::new()),
        notify: Arc::new(tokio::sync::Notify::new()),
    };
    cancellation.cancel();
    assert!(decoder
        .execute_read_only(
            vec![&q, &k, &v],
            vec![&prefix],
            vec![0],
            vec![true],
            vec![2],
            Some(&cancellation)
        )
        .await
        .is_err());
    assert_eq!(
        prefix.inspect().unwrap().layers[0].values,
        before.layers[0].values
    );
    prefix.release();
    sequence.release();
    assert!(left[0].value().unwrap().readback().unwrap()[0].is_finite());
}

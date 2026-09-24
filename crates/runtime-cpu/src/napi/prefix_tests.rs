use super::*;
use effect_torch_graph::AttentionRounding;

fn state_pool(dtype: NativeDType, retention: Option<u32>, capacity: u32) -> NativeKvPool {
    NativeKvPool::new(
        1,
        1,
        1,
        capacity,
        Some(2),
        Some(dtype),
        None,
        Some(vec![NativeKvLayerDescriptor {
            layer_id: 0,
            kv_heads: 1,
            head_dim: 1,
            dtype,
            retention_window: retention,
        }]),
    )
    .unwrap()
}

fn attention_program(
    dtype: NativeDType,
    access: &str,
    time: u32,
    retention: Option<u32>,
    capacity: u32,
    bad_gather: bool,
) -> Executable {
    let input = |slot| LazyTensor::input(slot, vec![1, 1, time, 1], Some(dtype), None).unwrap();
    let q = input(0);
    let k = input(1);
    let v = input(2);
    let mut root = LazyTensor {
        node: Node::new(NodeKind::SdpaConfigured {
            q: q.node,
            k: k.node,
            v: v.node,
            scale: 0.3,
            causal: true,
            window: AttentionWindow::Full,
            rounding: AttentionRounding::Stepwise,
            layer_id: Some(0),
            retention: retention.map_or(AttentionWindow::Full, |value| {
                AttentionWindow::Local(value as usize)
            }),
        })
        .unwrap(),
    };
    if bad_gather {
        let indices = LazyTensor {
            node: Node::new(NodeKind::Leaf(Arc::new(LeafSlot::new(Value::dense(
                Tensor::from_vec(vec![99i64; time as usize], vec![1, 1, time as usize, 1]),
            )))))
            .unwrap(),
        };
        root = root.gather(3, &indices).unwrap();
    }
    compile(
        vec![&root],
        Some(NativeCompileOptions {
            optimize: Some(false),
            constant_weights: None,
        }),
        Some(NativeKvStateSchema {
            access: Some(access.to_string()),
            max_tokens: capacity,
            block_size: 2,
            kv_dtype: dtype,
            window: None,
            batch: 1,
            packed_causal_chains: None,
            last_token_row: None,
            output_selections: None,
            current_block_attention: Some(if access == "Append" {
                NativeCurrentBlockAttention::Causal
            } else {
                NativeCurrentBlockAttention::Bidirectional
            }),
        }),
        None,
    )
    .unwrap()
}

fn value(values: &[f32], dtype: DType) -> NativeTensor {
    NativeTensor::wrap(Value::dense(
        Tensor::from_vec(values.to_vec(), vec![1, 1, values.len(), 1]).cast(dtype),
    ))
}

async fn append(
    executable: &Executable,
    sequence: &NativeKvSequence,
    data: &[f32],
    dtype: DType,
) -> Result<Vec<NativeTensor>> {
    let q = value(&vec![1.0; data.len()], dtype);
    let k = value(data, dtype);
    let v = value(data, dtype);
    executable
        .execute(
            vec![&q, &k, &v],
            vec![],
            Some(vec![sequence]),
            Some(vec![0]),
            Some(vec![true]),
            Some(vec![data.len() as u32]),
            Some(vec![data.len() as u32]),
            Some(vec![vec![7; data.len()]]),
            None,
        )
        .await
}

fn context(sequence: &NativeKvSequence, access: StateAccessMode) -> Arc<KvContext> {
    sequence.state.lock().unwrap().advance = 1;
    Arc::new(KvContext {
        access,
        valid_lengths: None,
        pool: sequence.pool.clone(),
        slots: vec![Some(sequence.state.clone())],
        advances: vec![1],
        packed: None,
        window: None,
        kda: KdaGeometry::default(),
        conv: ConvGeometry::default(),
        transaction: Mutex::new(None),
    })
}

#[tokio::test]
async fn failed_append_and_late_cancellation_restore_shared_tail() {
    let pool = state_pool(NativeDType::F32, None, 8);
    let encoder = attention_program(NativeDType::F32, "Append", 1, None, 8, false);
    let sequence = pool.make_sequence();
    append(&encoder, &sequence, &[2.0], DType::F32)
        .await
        .unwrap();
    let prefix = sequence.snapshot().unwrap();
    let before = prefix.inspect().unwrap();
    let blocks = sequence.state.lock().unwrap().blocks.clone();
    let available = pool.free_blocks();
    let broken = attention_program(NativeDType::F32, "Append", 1, None, 8, true);
    assert!(append(&broken, &sequence, &[9.0], DType::F32)
        .await
        .is_err());
    assert_eq!(sequence.cursor(), 1);
    assert_eq!(sequence.state.lock().unwrap().blocks, blocks);
    assert_eq!(sequence.state.lock().unwrap().copied_bytes, 0);
    assert_eq!(pool.free_blocks(), available);
    assert_eq!(
        prefix.inspect().unwrap().layers[0].values,
        before.layers[0].values
    );

    let context = context(&sequence, StateAccessMode::Append);
    let input = Value::dense(Tensor::from_vec(vec![7.0f32], vec![1, 1, 1, 1]));
    let cancelled = CancellationFlag::new();
    let mut observed_private_tail = false;
    let error = executable::execute_stateful_before_commit(
        &encoder.inner.executable,
        &[input.clone(), input.clone(), input],
        &encoder.inner.generated_bindings,
        &cancelled,
        &context,
        &|| !cancelled.is_cancelled(),
        &mut |_| {
            observed_private_tail = sequence.state.lock().unwrap().blocks != blocks;
            cancelled.cancel();
            Ok(())
        },
    )
    .unwrap_err();
    assert!(error.contains("abort"));
    assert!(observed_private_tail);
    assert_eq!(sequence.state.lock().unwrap().blocks, blocks);
    assert_eq!(sequence.state.lock().unwrap().copied_bytes, 0);
    assert_eq!(pool.free_blocks(), available);
    assert_eq!(
        prefix.inspect().unwrap().layers[0].values,
        before.layers[0].values
    );
}

#[tokio::test]
async fn read_only_uses_no_pool_pages_and_borrow_survives_release() {
    let pool = state_pool(NativeDType::F32, None, 2);
    let encoder = attention_program(NativeDType::F32, "Append", 2, None, 2, false);
    let sequence = pool.make_sequence();
    append(&encoder, &sequence, &[1.0, 2.0], DType::F32)
        .await
        .unwrap();
    let prefix = sequence.snapshot().unwrap();
    let borrower = prefix.borrowed().unwrap();
    let before = prefix.inspect().unwrap();
    prefix.release();
    sequence.release();
    assert_eq!(pool.free_blocks(), 0);
    let live = NativeKvPrefix {
        inner: Mutex::new(Some(borrower)),
    };
    let decoder = attention_program(NativeDType::F32, "ReadOnly", 2, None, 2, false);
    let q = value(&[1.0, 1.0], DType::F32);
    let k = value(&[3.0, 4.0], DType::F32);
    let v = value(&[5.0, 6.0], DType::F32);
    let outputs = decoder
        .execute_read_only(
            vec![&q, &k, &v],
            vec![&live],
            vec![0],
            vec![true],
            vec![2],
            None,
        )
        .await
        .unwrap();
    assert_eq!(outputs[0].shape().unwrap(), vec![1, 1, 2, 1]);
    assert_eq!(pool.free_blocks(), 0);
    assert_eq!(
        live.inspect().unwrap().layers[0].values,
        before.layers[0].values
    );
    let private = live.fork().unwrap();
    let context = context(&private, StateAccessMode::ReadOnly);
    let cancelled = CancellationFlag::new();
    assert!(executable::execute_stateful_before_commit(
        &decoder.inner.executable,
        &[
            q.value_cloned().unwrap(),
            k.value_cloned().unwrap(),
            v.value_cloned().unwrap()
        ],
        &decoder.inner.generated_bindings,
        &cancelled,
        &context,
        &|| !cancelled.is_cancelled(),
        &mut |_| {
            cancelled.cancel();
            Ok(())
        }
    )
    .is_err());
    assert_eq!(
        live.inspect().unwrap().layers[0].values,
        before.layers[0].values
    );
    drop(private);
    live.release();
    assert_eq!(pool.free_blocks(), 1);
    assert!(outputs[0].value_cloned().unwrap().to_f32_vec().unwrap()[0].is_finite());
}

#[tokio::test]
async fn local_retention_tracks_absolute_positions_beyond_pool_capacity() {
    let pool = state_pool(NativeDType::F32, Some(1), 4);
    let encoder = attention_program(NativeDType::F32, "Append", 1, Some(1), 4, false);
    let sequence = pool.make_sequence();
    for token in 0..12 {
        append(&encoder, &sequence, &[token as f32], DType::F32)
            .await
            .unwrap();
    }
    let prefix = sequence.snapshot().unwrap();
    let inspection = prefix.inspect().unwrap();
    assert_eq!(inspection.cursor, 12);
    assert_eq!(inspection.layers[0].start_position, 11);
    assert_eq!(inspection.layers[0].values, vec![11.0]);
    assert_eq!(prefix.retained_bytes().unwrap(), 16.0);
    let decoder = attention_program(NativeDType::F32, "ReadOnly", 2, Some(1), 4, false);
    let q = value(&[0.0, 0.0], DType::F32);
    let k = value(&[0.0, 0.0], DType::F32);
    let v = value(&[20.0, 30.0], DType::F32);
    let outputs = decoder
        .execute_read_only(
            vec![&q, &k, &v],
            vec![&prefix],
            vec![0],
            vec![true],
            vec![2],
            None,
        )
        .await
        .unwrap();
    let result = outputs[0].value_cloned().unwrap().to_f32_vec().unwrap();
    for actual in result {
        assert!((actual - 61.0 / 3.0).abs() < 1e-5);
    }
}

#[tokio::test]
async fn half_stepwise_paged_scale_uses_f32_scalar_then_rounds_the_score() {
    for (native, dtype, expected) in [
        (NativeDType::BF16, DType::BF16, 0.1826171875),
        (NativeDType::F16, DType::F16, 0.182373046875),
    ] {
        let pool = state_pool(native, None, 4);
        let sequence = pool.make_sequence();
        let prefix = sequence.snapshot().unwrap();
        let decoder = attention_program(native, "ReadOnly", 2, None, 4, false);
        let q = value(&[1.0, 1.0], dtype);
        let k = value(&[3.0, -2.0], dtype);
        let v = value(&[0.0, 1.0], dtype);
        let outputs = decoder
            .execute_read_only(
                vec![&q, &k, &v],
                vec![&prefix],
                vec![0],
                vec![true],
                vec![2],
                None,
            )
            .await
            .unwrap();
        assert_eq!(outputs[0].value_cloned().unwrap().dtype(), dtype);
        assert_eq!(
            outputs[0].value_cloned().unwrap().to_f32_vec().unwrap(),
            vec![expected; 2]
        );
        assert_eq!(pool.free_blocks(), 2);
    }
}

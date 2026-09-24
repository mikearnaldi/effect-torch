// Immutable handles retain page references independently of sequences and borrowers.
#[napi(object, object_from_js = false)]
pub struct NativeKvLayerSnapshot {
    pub layer_id: u32,
    pub start_position: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub dtype: NativeDType,
    pub keys: Vec<f64>,
    pub values: Vec<f64>,
}

#[napi(object, object_from_js = false)]
pub struct NativeKvSnapshotInspection {
    pub cursor: u32,
    pub retained_bytes: f64,
    pub shared_bytes: f64,
    pub copied_bytes: f64,
    pub layers: Vec<NativeKvLayerSnapshot>,
}

struct PrefixData {
    pool: Arc<PoolInner>,
    blocks: Vec<u32>,
    head: usize,
    cursor: usize,
    last_hash: u64,
    pending: Vec<u32>,
    recurrent: RecurrentSnapshot,
    copied_bytes: usize,
}

impl Drop for PrefixData {
    fn drop(&mut self) {
        for &block in self.live_blocks() {
            self.pool.unref_block(block);
        }
    }
}

impl PrefixData {
    fn live_blocks(&self) -> &[u32] {
        &self.blocks
    }
    fn physical_block(&self, position: usize) -> u32 {
        self.blocks[position / self.pool.block_size - self.head]
    }

    fn fork_sequence(&self) -> Result<NativeKvSequence> {
        with_device(self.pool.device_ordinal, || {
            let sequence = make_managed_sequence(&self.pool)?;
            let mut state = sequence
                .state
                .lock()
                .map_err(|error| to_napi_err(error.to_string()))?;
            self.recurrent
                .restore_into(&mut state)
                .map_err(to_napi_err)?;
            self.pool
                .ref_blocks(self.live_blocks())
                .map_err(to_napi_err)?;
            state.blocks = self.blocks.clone();
            state.head = self.head;
            state.cursor = self.cursor;
            state.last_hash = self.last_hash;
            state.pending = self.pending.clone();
            state.copied_bytes = self.copied_bytes;
            drop(state);
            Ok(sequence)
        })
    }

    fn export_layer(&self, index: usize, rows: &[u32]) -> Result<(Vec<f64>, Vec<f64>)> {
        let layer = self.pool.kv_layers[index];
        let export = |slab: &PoolSlab, scale_index: usize| -> Result<Vec<f64>> {
            let tensor = slab.metal().map_err(to_napi_err)?;
            let mut values = Vec::with_capacity(rows.len() * layer.kv_heads * layer.head_dim);
            for &row in rows {
                for head in 0..layer.kv_heads {
                    for column in 0..layer.head_dim {
                        let offset = tensor.layout.offset()
                            + (row as usize * layer.kv_heads + head) * layer.head_dim
                            + column;
                        // Snapshot page references keep these committed shared-memory rows immutable.
                        let value = unsafe {
                            match layer.dtype {
                                DType::F32 => {
                                    *tensor.buffer.contents_ptr().cast::<f32>().add(offset) as f64
                                }
                                DType::F16 => half::f16::from_bits(
                                    *tensor.buffer.contents_ptr().cast::<u16>().add(offset),
                                )
                                .to_f64(),
                                DType::BF16 => half::bf16::from_bits(
                                    *tensor.buffer.contents_ptr().cast::<u16>().add(offset),
                                )
                                .to_f64(),
                                DType::U8 => {
                                    let scales = self.pool.scales[scale_index]
                                        .metal()
                                        .map_err(to_napi_err)?;
                                    let scale = *scales.buffer.contents_ptr().cast::<f32>().add(
                                        scales.layout.offset()
                                            + row as usize * layer.kv_heads
                                            + head,
                                    );
                                    (*tensor.buffer.contents_ptr().cast::<u8>().add(offset) as f64
                                        - 128.0)
                                        * scale as f64
                                }
                                _ => {
                                    return Err(to_napi_err(
                                        "kv snapshot: unsupported storage dtype".to_string(),
                                    ))
                                }
                            }
                        };
                        values.push(value);
                    }
                }
            }
            Ok(values)
        };
        Ok((
            export(&self.pool.k[index], 2 * index)?,
            export(&self.pool.v[index], 2 * index + 1)?,
        ))
    }

    fn retained_bytes(&self, shared_only: bool) -> usize {
        let store = self
            .pool
            .blocks
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let pages = self
            .live_blocks()
            .iter()
            .filter(|&&block| !shared_only || store.refcounts[block as usize] > 1)
            .count();
        self.pool
            .kv_layers
            .iter()
            .map(|layer| {
                let scales = if layer.dtype == DType::U8 {
                    layer.kv_heads * 8
                } else {
                    0
                };
                pages * self.pool.block_size * (layer.row_bytes().unwrap_or(0) + scales)
            })
            .sum()
    }
}

#[napi]
pub struct NativeKvPrefix {
    inner: Mutex<Option<Arc<PrefixData>>>,
}

impl NativeKvPrefix {
    fn borrowed(&self) -> Result<Arc<PrefixData>> {
        self.inner
            .lock()
            .map_err(|error| to_napi_err(error.to_string()))?
            .clone()
            .ok_or_else(|| Error::new(Status::InvalidArg, "kv prefix is released"))
    }
}

#[napi]
impl NativeKvPrefix {
    #[napi]
    pub fn release(&self) {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
    }

    #[napi(getter)]
    pub fn cursor(&self) -> Result<u32> {
        Ok(self.borrowed()?.cursor as u32)
    }

    #[napi(getter)]
    pub fn retained_bytes(&self) -> Result<f64> {
        Ok(self.borrowed()?.retained_bytes(false) as f64)
    }

    #[napi(getter)]
    pub fn shared_bytes(&self) -> Result<f64> {
        Ok(self.borrowed()?.retained_bytes(true) as f64)
    }

    #[napi(getter)]
    pub fn copied_bytes(&self) -> Result<f64> {
        Ok(self.borrowed()?.copied_bytes as f64)
    }

    #[napi]
    pub fn fork(&self) -> Result<NativeKvSequence> {
        self.borrowed()?.fork_sequence()
    }

    #[napi]
    pub fn inspect(&self) -> Result<NativeKvSnapshotInspection> {
        let prefix = self.borrowed()?;
        let mut layers = Vec::with_capacity(prefix.pool.kv_layers.len());
        for (index, &layer) in prefix.pool.kv_layers.iter().enumerate() {
            let start = layer
                .retention
                .map_or(0, |retention| prefix.cursor.saturating_sub(retention))
                .max(prefix.head * prefix.pool.block_size);
            let rows = (start..prefix.cursor)
                .map(|position| {
                    prefix.physical_block(position) * prefix.pool.block_size as u32
                        + (position % prefix.pool.block_size) as u32
                })
                .collect::<Vec<_>>();
            let descriptor = NativeKvLayerDescriptor::from(layer);
            let (keys, values) = prefix.export_layer(index, &rows)?;
            layers.push(NativeKvLayerSnapshot {
                layer_id: layer.layer_id,
                start_position: start as u32,
                kv_heads: descriptor.kv_heads,
                head_dim: descriptor.head_dim,
                dtype: descriptor.dtype,
                keys,
                values,
            });
        }
        Ok(NativeKvSnapshotInspection {
            cursor: prefix.cursor as u32,
            retained_bytes: prefix.retained_bytes(false) as f64,
            shared_bytes: prefix.retained_bytes(true) as f64,
            copied_bytes: prefix.copied_bytes as f64,
            layers,
        })
    }
}

#[napi]
impl NativeKvSequence {
    #[napi]
    pub fn snapshot(&self) -> Result<NativeKvPrefix> {
        let _run = self
            .run_lock
            .lock()
            .map_err(|error| to_napi_err(error.to_string()))?;
        if self.released.load(Ordering::Acquire) {
            return Err(Error::new(Status::InvalidArg, "kv sequence is released"));
        }
        let state = self
            .state
            .lock()
            .map_err(|error| to_napi_err(error.to_string()))?;
        let recurrent = RecurrentSnapshot::capture(&state)
            .ok_or_else(|| to_napi_err("kv snapshot: recurrent capture failed".to_string()))?;
        self.pool.ref_blocks(&state.blocks).map_err(to_napi_err)?;
        Ok(NativeKvPrefix {
            inner: Mutex::new(Some(Arc::new(PrefixData {
                pool: self.pool.clone(),
                blocks: state.blocks.clone(),
                head: state.head,
                cursor: state.cursor,
                last_hash: state.last_hash,
                pending: state.pending.clone(),
                recurrent,
                copied_bytes: state.copied_bytes,
            }))),
        })
    }
}

#[napi]
impl Executable {
    #[napi]
    pub async fn execute_read_only(
        &self,
        bindings: Vec<&NativeTensor>,
        prefixes: Vec<&NativeKvPrefix>,
        slots: Vec<u32>,
        active_mask: Vec<bool>,
        valid_lengths: Vec<u32>,
        cancellation_token: Option<&CancellationToken>,
    ) -> Result<Vec<NativeTensor>> {
        let state = self.state.as_ref().ok_or_else(|| {
            Error::new(Status::InvalidArg, "executeReadOnly: state schema required")
        })?;
        let schema = &state.schema;
        if schema.access != StateAccessMode::ReadOnly
            || schema.kda.layers != 0
            || schema.conv.layers != 0
            || state.packed_rows_per_sequence.is_some()
        {
            return Err(Error::new(
                Status::InvalidArg,
                "executeReadOnly: requires a dense read-only KV executable",
            ));
        }
        let tokens = slots
            .iter()
            .map(|&slot| {
                valid_lengths
                    .get(slot as usize)
                    .map(|&length| vec![0; length as usize])
                    .ok_or_else(|| {
                        Error::new(Status::InvalidArg, "executeReadOnly: slot out of range")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        validate_fixed_lanes(
            schema.batch,
            prefixes.len(),
            &slots,
            &active_mask,
            &valid_lengths,
            &valid_lengths,
            &tokens,
        )?;
        // Independent block tables and run locks retain shared pages until native completion.
        let mut sequences = Vec::with_capacity(prefixes.len());
        for prefix in prefixes {
            match prefix.fork() {
                Ok(sequence) => sequences.push(sequence),
                Err(error) => {
                    for sequence in &sequences {
                        sequence.return_blocks();
                    }
                    return Err(error);
                }
            }
        }
        let result = self
            .execute_stateful(
                bindings,
                sequences.iter().collect(),
                slots,
                active_mask,
                valid_lengths.clone(),
                valid_lengths,
                tokens,
                StatefulInvocation::Tensors,
                cancellation_token,
            )
            .await;
        for sequence in &sequences {
            sequence.return_blocks();
        }
        match result? {
            StatefulExecutionOutput::Tensors(outputs) => Ok(outputs),
            StatefulExecutionOutput::Samples(_) => unreachable!(),
        }
    }
}

impl PoolInner {
    fn copy_shared_tail(&self, state: &mut SeqState) -> err::Res<()> {
        if state.cursor % self.block_size == 0 || state.blocks.is_empty() {
            return Ok(());
        }
        let index = state.cursor / self.block_size - state.head;
        let old = state.blocks[index];
        if state.transaction_tail == Some(old) {
            return Ok(());
        }
        if self
            .blocks
            .lock()
            .map_err(|error| error.to_string())?
            .refcounts[old as usize]
            <= 1
        {
            return Ok(());
        }
        let new = self
            .alloc_block_with_cache_eviction(true)
            .ok_or("kv pool: exhausted while copying shared tail")?;
        let result = (|| {
            let mut bytes = 0;
            for slab in self.k.iter().chain(&self.v).chain(&self.scales) {
                let tensor = slab.metal()?;
                let size =
                    self.block_size * tensor.layout.shape()[1] * tensor.dtype.size_in_bytes();
                if size == 0 {
                    continue;
                }
                crate::kernels::copy_bytes_into(
                    device::MetalDevice::get(),
                    &tensor.buffer,
                    old as usize * size,
                    &tensor.buffer,
                    new as usize * size,
                    size,
                )?;
                bytes += size;
            }
            Ok(bytes)
        })();
        match result {
            Ok(bytes) => {
                state.blocks[index] = new;
                state.copied_bytes += bytes;
                self.unref_block(old);
                Ok(())
            }
            Err(error) => {
                self.unref_block(new);
                Err(error)
            }
        }
    }

    fn matches_layers(&self, layers: &[KvLayerDescriptor]) -> bool {
        self.kv_layers.len() == layers.len()
            && self.kv_layers.iter().zip(layers).all(|(pool, compiled)| {
                let mut expected = *compiled;
                if !self.explicit_layers {
                    expected.retention = pool.retention;
                }
                *pool == expected
            })
    }

    fn retention_start(&self, cursor: usize) -> usize {
        self.kv_layers
            .iter()
            .map(|layer| {
                layer
                    .retention
                    .map_or(0, |retention| cursor.saturating_sub(retention))
            })
            .min()
            .unwrap_or(0)
    }
}

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
        &self.blocks[self.head..]
    }
    fn physical_block(&self, position: usize) -> u32 {
        self.blocks[position / self.pool.block_size]
    }

    fn fork_sequence(&self) -> Result<NativeKvSequence> {
        let sequence = NativeKvSequence::new(self.pool.clone());
        let mut state = sequence
            .state
            .lock()
            .map_err(|error| to_napi_err(error.to_string()))?;
        if !restore_recurrent_snapshot(&mut state, &self.recurrent) {
            return Err(Error::new(
                Status::GenericFailure,
                "kv prefix: recurrent state restore failed",
            ));
        }
        for &block in self.live_blocks() {
            self.pool.ref_block(block);
        }
        state.blocks = self.blocks.clone();
        state.head = self.head;
        state.cursor = self.cursor;
        state.last_hash = self.last_hash;
        state.pending = self.pending.clone();
        state.copied_bytes = self.copied_bytes;
        drop(state);
        Ok(sequence)
    }

    fn export_layer(&self, index: usize, rows: &[u32]) -> Result<(Vec<f64>, Vec<f64>)> {
        let layer = self.pool.kv_layers[index];
        let export = |slab: &pool::Slab, scale_index: usize| {
            let values = slab.read_rows_f32(rows);
            let values = if layer.dtype == DType::U8 {
                pool::dequantize_int8(
                    &values,
                    &self.pool.scales[scale_index].read_rows_f32(rows),
                    rows.len(),
                    layer.kv_heads,
                    layer.head_dim,
                )
            } else {
                values
            };
            values.into_iter().map(f64::from).collect()
        };
        Ok((
            export(&self.pool.k[index], 2 * index),
            export(&self.pool.v[index], 2 * index + 1),
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
        let schema = self.state.as_ref().ok_or_else(|| {
            Error::new(Status::InvalidArg, "executeReadOnly: state schema required")
        })?;
        if schema.access != StateAccessMode::ReadOnly
            || schema.kda.layers != 0
            || schema.conv.layers != 0
            || schema.packed_rows_per_sequence.is_some()
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
            schema,
            prefixes.len(),
            &slots,
            &active_mask,
            &valid_lengths,
            &valid_lengths,
            &tokens,
        )?;
        // Each call owns a distinct block table and run lock. Pages remain shared.
        let sequences = prefixes
            .iter()
            .map(|prefix| prefix.fork())
            .collect::<Result<Vec<_>>>()?;
        let result = self
            .execute_stateful(
                bindings,
                sequences.iter().collect(),
                slots,
                valid_lengths,
                tokens,
                StatefulInvocation::Tensors,
                cancellation_token,
            )
            .await?;
        match result {
            StatefulExecutionOutput::Tensors(outputs) => Ok(outputs),
            StatefulExecutionOutput::Samples(_) => unreachable!(),
        }
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
        let recurrent = capture_recurrent_snapshot(&state).ok_or_else(|| {
            Error::new(
                Status::GenericFailure,
                "kv snapshot: recurrent capture failed",
            )
        })?;
        for &block in &state.blocks[state.head..] {
            self.pool.ref_block(block);
        }
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

impl PoolInner {
    fn copy_shared_tail(&self, state: &mut SeqState) -> err::Res<()> {
        if state.cursor % self.block_size == 0 {
            return Ok(());
        }
        let index = state.cursor / self.block_size;
        let Some(&old) = state.blocks.get(index) else {
            return Ok(());
        };
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
            .alloc_block()
            .ok_or_else(|| "kv pool: exhausted while copying shared tail".to_string())?;
        for slab in self.k.iter().chain(&self.v).chain(&self.scales) {
            if slab.rows == 0 {
                continue;
            }
            slab.copy_rows(
                old as usize * self.block_size,
                new as usize * self.block_size,
                self.block_size,
            );
            state.copied_bytes += self.block_size * slab.row_width * slab.dtype.size_in_bytes();
        }
        state.blocks[index] = new;
        self.unref_block(old);
        Ok(())
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

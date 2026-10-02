#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn kv_attention_into(
    context: &KvContext,
    layer: u32,
    q: &Value,
    k: &Value,
    v: &Value,
    scale: f64,
    window: Option<usize>,
    mode: KvAttentionMode,
    output: &mut CpuDestination<'_>,
    eviction_starts: &mut [usize],
) -> err::Res<()> {
    kv_attention_configured_into(
        context,
        layer,
        q,
        k,
        v,
        scale,
        window,
        mode,
        effect_torch_graph::AttentionRounding::Fused,
        output,
        eviction_starts,
    )
}

#[allow(clippy::too_many_arguments)]
fn kv_attention_configured_into(
    context: &KvContext,
    layer: u32,
    q: &Value,
    k: &Value,
    v: &Value,
    scale: f64,
    window: Option<usize>,
    mode: KvAttentionMode,
    rounding: effect_torch_graph::AttentionRounding,
    output: &mut CpuDestination<'_>,
    eviction_starts: &mut [usize],
) -> err::Res<()> {
    match q.dtype() {
        DType::F32 => kv_attention_typed::<f32>(
            context,
            layer,
            q,
            k,
            v,
            scale,
            window,
            mode,
            rounding,
            output,
            eviction_starts,
        ),
        DType::F16 => kv_attention_typed::<half::f16>(
            context,
            layer,
            q,
            k,
            v,
            scale,
            window,
            mode,
            rounding,
            output,
            eviction_starts,
        ),
        DType::BF16 => kv_attention_typed::<half::bf16>(
            context,
            layer,
            q,
            k,
            v,
            scale,
            window,
            mode,
            rounding,
            output,
            eviction_starts,
        ),
        _ => Err("kv attention: unsupported query dtype".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn kv_attention_typed<T: Elem>(
    context: &KvContext,
    layer: u32,
    q: &Value,
    k: &Value,
    v: &Value,
    scale: f64,
    window: Option<usize>,
    mode: KvAttentionMode,
    rounding: effect_torch_graph::AttentionRounding,
    output: &mut CpuDestination<'_>,
    eviction_starts: &mut [usize],
) -> err::Res<()> {
    let dimensions = q.shape();
    let rank = dimensions.len();
    if rank < 4
        || k.shape().len() != rank
        || v.shape().len() != rank
        || q.dtype() != k.dtype()
        || q.dtype() != v.dtype()
        || output.shape() != dimensions
        || output.dtype() != q.dtype()
    {
        return Err(
            "kv attention: expected matching rank-4-or-higher floating tensors".to_string(),
        );
    }
    let batch = dimensions[..rank - 3].iter().product::<usize>();
    let (heads, tokens, width) = (
        dimensions[rank - 3],
        dimensions[rank - 2],
        dimensions[rank - 1],
    );
    let kv_heads = k.shape()[rank - 3];
    if k.shape()[..rank - 3] != dimensions[..rank - 3]
        || k.shape() != v.shape()
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || k.shape()[rank - 2] != tokens
        || k.shape()[rank - 1] != width
        || batch != context.graph_rows()
    {
        return Err("kv attention: incompatible grouped-query shapes".to_string());
    }
    let index = layer as usize;
    let descriptor = context
        .pool
        .kv_layers
        .get(index)
        .ok_or("kv attention: missing layer")?;
    if descriptor.kv_heads != kv_heads || descriptor.head_dim != width {
        return Err("kv attention: layer descriptor mismatch".to_string());
    }
    let read_only = context.access == StateAccessMode::ReadOnly;
    let stepwise = rounding == effect_torch_graph::AttentionRounding::Stepwise;
    let round = |value: f32| {
        if stepwise {
            T::from_f64(value as f64).to_f64() as f32
        } else {
            value
        }
    };
    let scale = scale as f32;
    let query_values =
        T::storage_of(&q.tensor().buffer).ok_or("kv attention: query dtype mismatch")?;
    let query_stride = q.tensor().layout.strides()[rank - 1];
    output.write::<T, _>("kv attention", &dimensions, |out| -> err::Res<()> {
        out.fill(T::from_f64(0.0));
        for lane in 0..batch {
            let Some((physical_lane, explicit_position)) = context.graph_row(lane) else {
                continue;
            };
            let slot = context.slots[physical_lane]
                .as_ref()
                .ok_or("kv attention: inactive lane")?;
            let mut state = slot.lock().map_err(|error| error.to_string())?;
            let cursor = state.cursor;
            let valid = if context.packed.is_some() {
                1
            } else {
                context
                    .valid_lengths
                    .as_ref()
                    .map_or(state.advance, |lengths| lengths[physical_lane])
            };
            let row_position = if context.packed.is_some() {
                explicit_position
            } else {
                cursor
            };
            if valid == 0 || valid > tokens {
                return Err("kv attention: invalid query length".to_string());
            }
            let prefix_start = descriptor
                .retention
                .map_or(0, |retention| cursor.saturating_sub(retention))
                .max(state.head * context.pool.block_size);
            if !read_only {
                let planned = if context.packed.is_some() {
                    state.advance
                } else {
                    tokens
                };
                kv_prepare(
                    &context.pool,
                    &mut state,
                    index,
                    context.window,
                    mode,
                    kv_heads,
                    width,
                    planned,
                )?;
            }
            let physical = |position: usize| -> usize {
                state.blocks[position / context.pool.block_size] as usize * context.pool.block_size
                    + position % context.pool.block_size
            };
            let input =
                |value: &Value, token: usize, head: usize, column: usize| -> err::Res<f32> {
                    Ok(tensor_element::<T>(
                        value.tensor(),
                        ((lane * kv_heads + head) * tokens + token) * width + column,
                        "kv attention",
                    )?
                    .to_f64() as f32)
                };
            if !read_only {
                let write_tokens = if context.packed.is_some() {
                    1
                } else {
                    state.advance
                };
                for (value, slab, scale_index) in [
                    (k, &context.pool.k[index], 2 * index),
                    (v, &context.pool.v[index], 2 * index + 1),
                ] {
                    if descriptor.dtype == DType::U8 {
                        context.pool.scales[scale_index].write(|mut scales| {
                            slab.write(|mut slab| -> err::Res<()> {
                                for token in 0..write_tokens {
                                    for head in 0..kv_heads {
                                        let row = physical(row_position + token);
                                        let mut maximum = 0.0f32;
                                        for column in 0..width {
                                            maximum = maximum
                                                .max(input(value, token, head, column)?.abs());
                                        }
                                        let scale = maximum / 127.0 + 1e-12;
                                        scales.set_f32(row * kv_heads + head, scale);
                                        for column in 0..width {
                                            slab.set_u8(
                                                (row * kv_heads + head) * width + column,
                                                ((input(value, token, head, column)? / scale)
                                                    .round()
                                                    .clamp(-127.0, 127.0)
                                                    + 128.0)
                                                    as u8,
                                            );
                                        }
                                    }
                                }
                                Ok(())
                            })
                        })?;
                    } else {
                        slab.write(|mut slab| -> err::Res<()> {
                            for token in 0..write_tokens {
                                for head in 0..kv_heads {
                                    for column in 0..width {
                                        slab.set_f32(
                                            (physical(row_position + token) * kv_heads + head)
                                                * width
                                                + column,
                                            input(value, token, head, column)?,
                                        );
                                    }
                                }
                            }
                            Ok(())
                        })?;
                    }
                }
            }
            let mut attend = |keys: &pool::SlabReader<'_>,
                              values: &pool::SlabReader<'_>,
                              key_scales: Option<&pool::SlabReader<'_>>,
                              value_scales: Option<&pool::SlabReader<'_>>|
             -> err::Res<()> {
                let cached = |reader: &pool::SlabReader<'_>,
                              scales: Option<&pool::SlabReader<'_>>,
                              current: &Value,
                              position: usize,
                              head: usize,
                              column: usize|
                 -> err::Res<f32> {
                    // The canvas remains an invocation input. It is never written by read-only calls.
                    if position >= row_position && read_only {
                        return input(current, position - row_position, head, column);
                    }
                    let row = physical(position);
                    let raw = reader.get_f32((row * kv_heads + head) * width + column);
                    Ok(scales.map_or(raw, |scales| {
                        (raw - 128.0) * scales.get_f32(row * kv_heads + head)
                    }))
                };
                for head in 0..heads {
                    let kv_head = head / (heads / kv_heads);
                    for query in 0..valid {
                        let end = if mode == KvAttentionMode::BidirectionalBlock {
                            row_position + valid
                        } else {
                            row_position + query + 1
                        };
                        let begin = window.map_or(prefix_start, |window| {
                            if mode == KvAttentionMode::BidirectionalBlock {
                                cursor.saturating_sub(window).max(prefix_start)
                            } else {
                                end.saturating_sub(window).max(prefix_start)
                            }
                        });
                        let query_base = crate::tensor::source_index(
                            &q.tensor().layout,
                            ((lane * heads + head) * tokens + query) * width,
                        );
                        let score = |position: usize| -> err::Res<f32> {
                            let mut dot = 0.0f32;
                            for column in 0..width {
                                let q = query_values[query_base + column * query_stride].to_f64()
                                    as f32;
                                dot += q * cached(keys, key_scales, k, position, kv_head, column)?;
                            }
                            Ok(round(round(dot) * scale))
                        };
                        let mut maximum = f32::NEG_INFINITY;
                        for position in begin..end {
                            maximum = maximum.max(score(position)?);
                        }
                        let mut denominator = 0.0f32;
                        for position in begin..end {
                            denominator += (score(position)? - maximum).exp();
                        }
                        for column in 0..width {
                            let mut result = 0.0f32;
                            for position in begin..end {
                                let exponential = (score(position)? - maximum).exp();
                                let probability = if stepwise {
                                    round(exponential / denominator)
                                } else {
                                    exponential
                                };
                                result += probability
                                    * cached(values, value_scales, v, position, kv_head, column)?;
                            }
                            out[((lane * heads + head) * tokens + query) * width + column] =
                                T::from_f64(if stepwise {
                                    result
                                } else {
                                    result / denominator
                                } as f64);
                        }
                    }
                }
                Ok(())
            };
            context.pool.k[index].read(|keys| {
                context.pool.v[index].read(|values| {
                    if descriptor.dtype == DType::U8 {
                        context.pool.scales[2 * index].read(|ks| {
                            context.pool.scales[2 * index + 1]
                                .read(|vs| attend(&keys, &values, Some(&ks), Some(&vs)))
                        })
                    } else {
                        attend(&keys, &values, None, None)
                    }
                })
            })?;
            if !read_only {
                let start = if context.pool.explicit_layers {
                    context.pool.retention_start(cursor + state.advance)
                } else {
                    context
                        .window
                        .map_or(0, |window| (cursor + state.advance).saturating_sub(window))
                };
                eviction_starts[physical_lane] = start;
            }
        }
        Ok(())
    })??;
    Ok(())
}

//! Backend-neutral next-token sampling over one dense logits row.

/// Maximum row width accepted by native sampling. This bounds filtered
/// candidate storage and the non-interruptible portions of native selection.
pub const MAX_SAMPLING_VOCABULARY: usize = 1_048_576;

/// Validated controls for one stateless token draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingOptions {
    /// Non-negative softmax temperature; zero selects greedy argmax.
    pub temperature: f64,
    /// Candidate count retained before top-p, or `None` to disable top-k.
    pub top_k: Option<usize>,
    /// Smallest descending probability prefix to retain, in `(0, 1]`.
    pub top_p: f64,
    /// Stateless random stream key.
    pub seed: u64,
    /// Draw position within the keyed stream.
    pub counter: u64,
}

#[derive(Clone, Copy)]
struct Candidate {
    token: usize,
    value: f64,
}

fn descending(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
    right
        .value
        .total_cmp(&left.value)
        .then_with(|| left.token.cmp(&right.token))
}

fn random_unit(seed: u64, counter: u64) -> f64 {
    let mut value = seed ^ counter.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn validate(length: usize, options: SamplingOptions) -> Result<(), String> {
    if length == 0 {
        return Err("sample: logits must be non-empty".to_string());
    }
    if length > u32::MAX as usize {
        return Err("sample: vocabulary exceeds u32 token range".to_string());
    }
    if length > MAX_SAMPLING_VOCABULARY {
        return Err(format!(
            "sample: vocabulary {length} exceeds limit {MAX_SAMPLING_VOCABULARY}"
        ));
    }
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(format!(
            "sample: temperature must be finite and non-negative, got {}",
            options.temperature
        ));
    }
    if !options.top_p.is_finite() || options.top_p <= 0.0 || options.top_p > 1.0 {
        return Err(format!(
            "sample: topP must be finite and in (0, 1], got {}",
            options.top_p
        ));
    }
    if let Some(top_k) = options.top_k {
        if top_k == 0 || top_k > length {
            return Err(format!(
                "sample: topK must be in [1, {length}], got {top_k}"
            ));
        }
    }
    Ok(())
}

/// Samples one token from `length` logical logits without taking ownership of
/// their storage. `cancelled` is polled during scans and before sorting.
pub fn sample_logits(
    length: usize,
    mut value_at: impl FnMut(usize) -> f64,
    options: SamplingOptions,
    mut cancelled: impl FnMut() -> bool,
) -> Result<u32, String> {
    validate(length, options)?;

    if options.temperature == 0.0 {
        let mut selected = 0usize;
        let mut best = f64::NEG_INFINITY;
        for token in 0..length {
            if token & 4095 == 0 && cancelled() {
                return Err("operation aborted".to_string());
            }
            let value = value_at(token);
            if !value.is_finite() {
                return Err(format!("sample: logit {token} is not finite"));
            }
            if value > best {
                best = value;
                selected = token;
            }
        }
        return Ok(selected as u32);
    }

    let top_k = options.top_k.filter(|&top_k| top_k < length);
    let filtered = top_k.is_some() || options.top_p < 1.0;
    if !filtered {
        let mut max = f64::NEG_INFINITY;
        for token in 0..length {
            if token & 4095 == 0 && cancelled() {
                return Err("operation aborted".to_string());
            }
            let value = value_at(token);
            if !value.is_finite() {
                return Err(format!("sample: logit {token} is not finite"));
            }
            max = max.max(value);
        }
        let mut total = 0.0;
        for token in 0..length {
            if token & 4095 == 0 && cancelled() {
                return Err("operation aborted".to_string());
            }
            total += ((value_at(token) - max) / options.temperature).exp();
        }
        let mut draw = random_unit(options.seed, options.counter) * total;
        for token in 0..length {
            if token & 4095 == 0 && cancelled() {
                return Err("operation aborted".to_string());
            }
            draw -= ((value_at(token) - max) / options.temperature).exp();
            if draw < 0.0 {
                return Ok(token as u32);
            }
        }
        return Ok((length - 1) as u32);
    }

    let mut candidates = Vec::with_capacity(length);
    for token in 0..length {
        if token & 4095 == 0 && cancelled() {
            return Err("operation aborted".to_string());
        }
        let value = value_at(token);
        if !value.is_finite() {
            return Err(format!("sample: logit {token} is not finite"));
        }
        candidates.push(Candidate { token, value });
    }
    if cancelled() {
        return Err("operation aborted".to_string());
    }
    if let Some(top_k) = top_k {
        if top_k < candidates.len() {
            candidates.select_nth_unstable_by(top_k, descending);
            candidates.truncate(top_k);
        }
    }
    if cancelled() {
        return Err("operation aborted".to_string());
    }
    candidates.sort_unstable_by(descending);
    if cancelled() {
        return Err("operation aborted".to_string());
    }

    let max = candidates[0].value;
    let mut total = 0.0;
    for (index, candidate) in candidates.iter_mut().enumerate() {
        if index & 4095 == 0 && cancelled() {
            return Err("operation aborted".to_string());
        }
        candidate.value = ((candidate.value - max) / options.temperature).exp();
        total += candidate.value;
    }
    let mut retained = candidates.len();
    if options.top_p < 1.0 {
        let threshold = options.top_p * total;
        let mut cumulative = 0.0;
        for (index, candidate) in candidates.iter().enumerate() {
            if index & 4095 == 0 && cancelled() {
                return Err("operation aborted".to_string());
            }
            cumulative += candidate.value;
            if cumulative >= threshold {
                retained = index + 1;
                total = cumulative;
                break;
            }
        }
    }

    let mut draw = random_unit(options.seed, options.counter) * total;
    for (index, candidate) in candidates[..retained].iter().enumerate() {
        if index & 4095 == 0 && cancelled() {
            return Err("operation aborted".to_string());
        }
        draw -= candidate.value;
        if draw < 0.0 {
            return Ok(candidate.token as u32);
        }
    }
    Ok(candidates[retained - 1].token as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> SamplingOptions {
        SamplingOptions {
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            seed: 7,
            counter: 3,
        }
    }

    #[test]
    fn greedy_prefers_the_lower_token_on_ties() {
        let logits = [1.0, 4.0, 4.0, 2.0];
        assert_eq!(
            sample_logits(
                logits.len(),
                |index| logits[index],
                SamplingOptions {
                    temperature: 0.0,
                    ..options()
                },
                || false,
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn top_k_one_is_greedy() {
        let logits = [1.0, 4.0, 3.0];
        assert_eq!(
            sample_logits(
                logits.len(),
                |index| logits[index],
                SamplingOptions {
                    top_k: Some(1),
                    ..options()
                },
                || false,
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn top_p_retains_the_crossing_token() {
        let logits = [2.0, 1.0, 0.0];
        let sampled = sample_logits(
            logits.len(),
            |index| logits[index],
            SamplingOptions {
                top_p: 0.8,
                seed: 1,
                counter: 0,
                ..options()
            },
            || false,
        )
        .unwrap();
        assert!(sampled < 2);
    }

    #[test]
    fn identical_keys_replay() {
        let logits = [0.0, 0.0, 0.0, 0.0];
        let first =
            sample_logits(logits.len(), |index| logits[index], options(), || false).unwrap();
        let second =
            sample_logits(logits.len(), |index| logits[index], options(), || false).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn full_width_top_k_is_a_no_op() {
        let logits = [0.0, 1.0, 2.0];
        let unfiltered =
            sample_logits(logits.len(), |index| logits[index], options(), || false).unwrap();
        let filtered = sample_logits(
            logits.len(),
            |index| logits[index],
            SamplingOptions {
                top_k: Some(logits.len()),
                ..options()
            },
            || false,
        )
        .unwrap();
        assert_eq!(unfiltered, filtered);
    }
}

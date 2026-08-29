//! Backend-neutral next-token sampling over one dense logits row.

/// Maximum logits row width for native sampling. It limits candidate storage
/// and the work native selection performs between cancellation checks.
pub const MAX_SAMPLING_VOCABULARY: usize = 1_048_576;

/// Controls for one stateless token draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingOptions {
    /// Non-negative softmax temperature. Zero selects greedy argmax.
    pub temperature: f64,
    /// Number of candidates kept before top-p, or `None` to disable top-k.
    pub top_k: Option<usize>,
    /// Top-p threshold in `(0, 1]`. Sampling keeps the smallest descending
    /// prefix whose cumulative probability reaches this threshold.
    pub top_p: f64,
    /// Stateless random stream key.
    pub seed: u64,
    /// Draw position within the keyed stream.
    pub counter: u64,
}

/// Independent deterministic draw domains for speculative generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum SamplingPurpose {
    Proposal = 0x243f_6a88_85a3_08d3,
    Accept = 0x1319_8a2e_0370_7344,
    Residual = 0xa409_3822_299f_31d0,
    Target = 0x082e_fa98_ec4e_6c89,
}

/// Logical coordinate for one stateless sampling draw. It stores every field
/// at full width and excludes physical lanes, packed rows, and request order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingCoordinate {
    pub seed: u64,
    pub sequence_id: u64,
    pub absolute_position: u64,
    pub purpose: SamplingPurpose,
    pub subcounter: u64,
}

/// Returns a lossless logical RNG coordinate.
pub const fn sampling_coordinate(
    seed: u64,
    sequence_id: u64,
    absolute_position: u64,
    purpose: SamplingPurpose,
    subcounter: u64,
) -> SamplingCoordinate {
    SamplingCoordinate {
        seed,
        sequence_id,
        absolute_position,
        purpose,
        subcounter,
    }
}

/// Derives a draw counter from an absolute output position and purpose. The
/// counter does not depend on a physical lane, request index, or packed row.
pub fn purpose_counter(position: u64, purpose: SamplingPurpose, subcounter: u64) -> u64 {
    let mut value = position ^ purpose as u64;
    value = value.wrapping_add(subcounter.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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

/// Returns one deterministic uniform draw in `[0, 1)` for a stateless key.
pub fn random_unit(seed: u64, counter: u64) -> f64 {
    let mut value = seed ^ counter.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Returns one deterministic uniform draw for a full logical coordinate.
pub fn random_unit_at(coordinate: SamplingCoordinate) -> f64 {
    let mut key = coordinate.seed;
    for component in [
        coordinate.sequence_id,
        coordinate.absolute_position,
        coordinate.purpose as u64,
        coordinate.subcounter,
    ] {
        key = purpose_counter(key, SamplingPurpose::Target, component);
    }
    random_unit(key, 0)
}

/// Returns the normalized distribution after temperature, top-k, and top-p.
/// Exact speculative acceptance uses it to keep target and proposal
/// probabilities available after selection.
pub fn effective_probabilities(
    length: usize,
    mut value_at: impl FnMut(usize) -> f64,
    options: SamplingOptions,
    mut cancelled: impl FnMut() -> bool,
) -> Result<Vec<f64>, String> {
    validate(length, options)?;
    let mut probabilities = vec![0.0; length];
    if options.temperature == 0.0 {
        let token = sample_logits(length, value_at, options, cancelled)? as usize;
        probabilities[token] = 1.0;
        return Ok(probabilities);
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
    if let Some(top_k) = options.top_k.filter(|&top_k| top_k < length) {
        candidates.select_nth_unstable_by(top_k, descending);
        candidates.truncate(top_k);
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
    for candidate in &candidates[..retained] {
        probabilities[candidate.token] = candidate.value / total;
    }
    Ok(probabilities)
}

/// Samples a token from non-negative unnormalized weights with a stateless key.
/// Exact speculative residual sampling calls this after `max(0, p - q)`.
pub fn sample_probabilities(
    probabilities: &[f64],
    seed: u64,
    counter: u64,
    mut cancelled: impl FnMut() -> bool,
) -> Result<u32, String> {
    sample_probabilities_with_draw(probabilities, random_unit(seed, counter), &mut cancelled)
}

/// Samples non-negative weights at a full-width logical RNG coordinate.
pub fn sample_probabilities_at(
    probabilities: &[f64],
    coordinate: SamplingCoordinate,
    mut cancelled: impl FnMut() -> bool,
) -> Result<u32, String> {
    sample_probabilities_with_draw(probabilities, random_unit_at(coordinate), &mut cancelled)
}

fn sample_probabilities_with_draw(
    probabilities: &[f64],
    unit_draw: f64,
    mut cancelled: impl FnMut() -> bool,
) -> Result<u32, String> {
    if probabilities.is_empty() || probabilities.len() > u32::MAX as usize {
        return Err("sample: probability row has invalid vocabulary size".to_string());
    }
    let mut total = 0.0;
    for (index, &probability) in probabilities.iter().enumerate() {
        if index & 4095 == 0 && cancelled() {
            return Err("operation aborted".to_string());
        }
        if !probability.is_finite() || probability < 0.0 {
            return Err(format!("sample: probability {index} is invalid"));
        }
        total += probability;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err("sample: probability row has no positive mass".to_string());
    }
    let mut draw = unit_draw * total;
    for (token, &probability) in probabilities.iter().enumerate() {
        draw -= probability;
        if draw < 0.0 {
            return Ok(token as u32);
        }
    }
    Ok((probabilities.len() - 1) as u32)
}

/// Comparison of a target-distribution draw with one deterministic or
/// non-factorized proposal token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetSampleMatchResult {
    Accepted,
    Rejected(u32),
}

/// Draws from the target distribution at one candidate position. If the draw
/// differs from `candidate`, returns the draw as the correction.
pub fn target_sample_match(
    target: &[f64],
    candidate: u32,
    coordinate: SamplingCoordinate,
    cancelled: impl FnMut() -> bool,
) -> Result<TargetSampleMatchResult, String> {
    let candidate = usize::try_from(candidate)
        .map_err(|_| "target sample match: candidate is outside the vocabulary".to_string())?;
    if candidate >= target.len() {
        return Err("target sample match: candidate is outside the vocabulary".to_string());
    }
    let sampled = sample_probabilities_at(target, coordinate, cancelled)?;
    if sampled as usize == candidate {
        Ok(TargetSampleMatchResult::Accepted)
    } else {
        Ok(TargetSampleMatchResult::Rejected(sampled))
    }
}

/// Result of one exact speculative rejection test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectionResult {
    Accepted,
    Rejected(u32),
}

/// Accepts with probability `min(1, p(x) / q(x))`. On rejection, samples
/// from `normalize(max(0, p - q))`. Acceptance and residual draws use
/// independent purpose domains.
pub fn rejection_sample(
    target: &[f64],
    proposal: &[f64],
    candidate: u32,
    seed: u64,
    position: u64,
    mut cancelled: impl FnMut() -> bool,
) -> Result<RejectionResult, String> {
    if target.len() != proposal.len() || target.is_empty() {
        return Err("speculative sampling: target and proposal vocabularies differ".to_string());
    }
    let candidate = candidate as usize;
    let (&p, &q) = target
        .get(candidate)
        .zip(proposal.get(candidate))
        .ok_or_else(|| "speculative sampling: candidate is outside the vocabulary".to_string())?;
    if !q.is_finite() || q <= 0.0 || !p.is_finite() || p < 0.0 {
        return Err("speculative sampling: candidate has invalid probability".to_string());
    }
    let accepted =
        p >= q || random_unit(seed, purpose_counter(position, SamplingPurpose::Accept, 0)) < p / q;
    if accepted {
        return Ok(RejectionResult::Accepted);
    }
    let residual = target
        .iter()
        .zip(proposal)
        .map(|(&target, &proposal)| (target - proposal).max(0.0))
        .collect::<Vec<_>>();
    let token = sample_probabilities(
        &residual,
        seed,
        purpose_counter(position, SamplingPurpose::Residual, 0),
        &mut cancelled,
    )?;
    Ok(RejectionResult::Rejected(token))
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

/// Samples one token from `length` logical logits read through `value_at`.
/// Polls `cancelled` during scans and before sorting.
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
    fn effective_probabilities_apply_filters_and_normalize() {
        let logits = [3.0, 2.0, 1.0, 0.0];
        let probabilities = effective_probabilities(
            logits.len(),
            |index| logits[index],
            SamplingOptions {
                top_k: Some(3),
                top_p: 0.7,
                ..options()
            },
            || false,
        )
        .unwrap();
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probabilities[0] > 0.0);
        assert!(probabilities[1] > 0.0);
        assert_eq!(probabilities[2], 0.0);
        assert_eq!(probabilities[3], 0.0);
    }

    #[test]
    fn residual_probability_sampling_replays_its_key() {
        let probabilities = [0.0, 0.25, 0.75];
        let first = sample_probabilities(&probabilities, 17, 9, || false).unwrap();
        let second = sample_probabilities(&probabilities, 17, 9, || false).unwrap();
        assert_eq!(first, second);
        assert!(matches!(first, 1 | 2));
    }

    #[test]
    fn speculative_purposes_are_replayable_and_disjoint() {
        let position = 41;
        let proposal = purpose_counter(position, SamplingPurpose::Proposal, 0);
        assert_eq!(
            proposal,
            purpose_counter(position, SamplingPurpose::Proposal, 0)
        );
        assert_ne!(
            proposal,
            purpose_counter(position, SamplingPurpose::Accept, 0)
        );
        assert_ne!(
            proposal,
            purpose_counter(position, SamplingPurpose::Residual, 0)
        );
        assert_ne!(
            proposal,
            purpose_counter(position, SamplingPurpose::Target, 0)
        );
        assert_ne!(
            proposal,
            purpose_counter(position + 1, SamplingPurpose::Proposal, 0)
        );
    }

    #[test]
    fn full_width_coordinates_retain_every_logical_component() {
        let coordinate = sampling_coordinate(
            u64::MAX,
            u64::MAX - 1,
            1 << 63,
            SamplingPurpose::Residual,
            u64::MAX - 2,
        );
        assert_eq!(coordinate.seed, u64::MAX);
        assert_eq!(coordinate.sequence_id, u64::MAX - 1);
        assert_eq!(coordinate.absolute_position, 1 << 63);
        assert_eq!(coordinate.purpose, SamplingPurpose::Residual);
        assert_eq!(coordinate.subcounter, u64::MAX - 2);
        assert_eq!(random_unit_at(coordinate), random_unit_at(coordinate));

        for distinct in [
            sampling_coordinate(
                u64::MAX - 1,
                coordinate.sequence_id,
                coordinate.absolute_position,
                coordinate.purpose,
                coordinate.subcounter,
            ),
            sampling_coordinate(
                coordinate.seed,
                u64::MAX,
                coordinate.absolute_position,
                coordinate.purpose,
                coordinate.subcounter,
            ),
            sampling_coordinate(
                coordinate.seed,
                coordinate.sequence_id,
                coordinate.absolute_position + 1,
                coordinate.purpose,
                coordinate.subcounter,
            ),
            sampling_coordinate(
                coordinate.seed,
                coordinate.sequence_id,
                coordinate.absolute_position,
                SamplingPurpose::Accept,
                coordinate.subcounter,
            ),
            sampling_coordinate(
                coordinate.seed,
                coordinate.sequence_id,
                coordinate.absolute_position,
                coordinate.purpose,
                u64::MAX - 1,
            ),
        ] {
            assert_ne!(random_unit_at(coordinate), random_unit_at(distinct));
        }
    }

    #[test]
    fn target_sample_matching_replays_acceptance_and_correction() {
        let coordinate = sampling_coordinate(17, 29, 41, SamplingPurpose::Target, 0);
        let target = [0.0, 0.0, 1.0];
        assert_eq!(
            target_sample_match(&target, 2, coordinate, || false).unwrap(),
            TargetSampleMatchResult::Accepted
        );
        assert_eq!(
            target_sample_match(&target, 1, coordinate, || false).unwrap(),
            TargetSampleMatchResult::Rejected(2)
        );
        assert_eq!(
            sample_probabilities_at(&target, coordinate, || false).unwrap(),
            2
        );
        assert!(target_sample_match(&target, 3, coordinate, || false)
            .unwrap_err()
            .contains("outside the vocabulary"));
        assert_eq!(
            target_sample_match(&target, 1, coordinate, || false).unwrap(),
            TargetSampleMatchResult::Rejected(2)
        );
    }

    #[test]
    fn rejection_sampling_accepts_equal_rows_and_corrects_disjoint_rows() {
        let equal = [0.25, 0.75];
        assert_eq!(
            rejection_sample(&equal, &equal, 1, 7, 0, || false).unwrap(),
            RejectionResult::Accepted
        );
        assert_eq!(
            rejection_sample(&[1.0, 0.0], &[0.0, 1.0], 1, 7, 0, || false).unwrap(),
            RejectionResult::Rejected(0)
        );
    }

    #[test]
    fn rejection_sampling_pins_poor_and_degenerate_drafts() {
        let target = [0.1, 0.2, 0.3, 0.4];
        let poor = [0.55, 0.3, 0.1, 0.05];
        let acceptance_probability = target[0] / poor[0];
        assert!(
            random_unit(7, purpose_counter(6, SamplingPurpose::Accept, 0)) < acceptance_probability
        );
        assert!(
            random_unit(7, purpose_counter(2, SamplingPurpose::Accept, 0)) > acceptance_probability
        );
        assert_eq!(
            rejection_sample(&target, &poor, 0, 7, 6, || false).unwrap(),
            RejectionResult::Accepted
        );
        assert_eq!(
            rejection_sample(&target, &poor, 0, 7, 2, || false).unwrap(),
            RejectionResult::Rejected(2)
        );

        assert_eq!(
            rejection_sample(&[0.0, 1.0], &[0.0, 1.0], 1, 7, 2, || false).unwrap(),
            RejectionResult::Accepted
        );
        assert_eq!(
            rejection_sample(&[0.0, 0.25, 0.75], &[1.0, 0.0, 0.0], 0, 7, 2, || false).unwrap(),
            RejectionResult::Rejected(1)
        );
    }

    #[test]
    fn rejection_sampling_uses_filtered_positive_temperature_distributions() {
        let target_logits = [3.0, 2.0, 1.0, 0.0];
        let proposal_logits = [0.0, 3.0, 2.0, 1.0];
        let filtered = SamplingOptions {
            temperature: 0.7,
            top_k: Some(3),
            top_p: 0.8,
            ..options()
        };
        let target = effective_probabilities(
            target_logits.len(),
            |token| target_logits[token],
            filtered,
            || false,
        )
        .unwrap();
        let proposal = effective_probabilities(
            proposal_logits.len(),
            |token| proposal_logits[token],
            filtered,
            || false,
        )
        .unwrap();

        assert_eq!(target[2..], [0.0, 0.0]);
        assert_eq!(proposal[0], 0.0);
        assert_eq!(proposal[3], 0.0);
        assert_eq!(
            rejection_sample(&target, &proposal, 1, 7, 2, || false).unwrap(),
            RejectionResult::Rejected(0)
        );
    }

    #[test]
    fn rejection_sampling_matches_target_frequencies_for_poor_and_perfect_drafts() {
        const DRAWS: u64 = 50_000;
        const TOLERANCE: f64 = 0.01;
        let target = [0.1, 0.2, 0.3, 0.4];

        for proposal in [[0.55, 0.3, 0.1, 0.05], target] {
            let mut counts = [0u64; 4];
            for position in 0..DRAWS {
                let seed = position.wrapping_mul(0xd134_2543_de82_ef95);
                let candidate = sample_probabilities(
                    &proposal,
                    seed,
                    purpose_counter(position, SamplingPurpose::Proposal, 0),
                    || false,
                )
                .unwrap();
                let output =
                    match rejection_sample(&target, &proposal, candidate, seed, position, || false)
                        .unwrap()
                    {
                        RejectionResult::Accepted => candidate,
                        RejectionResult::Rejected(token) => token,
                    };
                counts[output as usize] += 1;
            }
            for (count, expected) in counts.into_iter().zip(target) {
                let observed = count as f64 / DRAWS as f64;
                assert!(
                    (observed - expected).abs() < TOLERANCE,
                    "observed {observed}, expected {expected}, proposal {proposal:?}"
                );
            }
        }
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

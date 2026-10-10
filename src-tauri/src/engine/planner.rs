//! A duration-constrained beam search over independently scored beat transitions.
//! Shortening uses forward skips and lengthening uses backward repeats. Candidate
//! cuts share a four-beat phase and compare surrounding chroma, timbre, loudness,
//! and waveform correlation. The search is bounded, so an exhausted search reports
//! the closest explored duration rather than truncating, padding, or stretching.

use std::collections::HashMap;

use super::{
    Analysis, BeatFeatures, EngineError, RemixPlan, RemixSettings, Segment, MAX_SAMPLES,
    MAX_SECONDS,
};

const BEAM_WIDTH: usize = 48;
const MATCHES_PER_BOUNDARY: usize = 6;
const MAX_SEARCH_EDITS: usize = 2048;

#[derive(Clone, Copy)]
struct Transition {
    end: usize,
    start: usize,
    cost: f64,
}

#[derive(Clone, Copy)]
struct SearchNode {
    parent: Option<usize>,
    transition: Option<Transition>,
    source_start: usize,
    output_frames: i64,
    cost: f64,
    transition_cost: f64,
}

#[derive(Clone, Copy)]
struct Candidate {
    parent: usize,
    transition: Transition,
    output_frames: i64,
    cost: f64,
    transition_cost: f64,
    rank: f64,
}

/// Find passages within the requested tolerance while retaining both source ends.
pub fn plan(analysis: &Analysis, settings: &RemixSettings) -> Result<RemixPlan, EngineError> {
    validate_settings(analysis, settings)?;
    if within_tolerance(
        analysis.source_frames as i64,
        analysis.sample_rate,
        settings,
    ) {
        return assemble_plan(analysis, settings, &[]);
    }
    let graph = transition_graph(analysis, settings);
    let mut arena = vec![SearchNode {
        parent: None,
        transition: None,
        source_start: 0,
        output_frames: analysis.source_frames as i64,
        cost: 0.0,
        transition_cost: 0.0,
    }];
    let mut beam = vec![0_usize];
    let target_frames = settings.target_seconds * f64::from(analysis.sample_rate);
    let tolerance_frames = settings.tolerance_seconds * f64::from(analysis.sample_rate);
    let beat_frames = f64::from(analysis.sample_rate) * 60.0 / analysis.tempo_bpm;
    let quantum =
        (tolerance_frames / 8.0).clamp(1.0, f64::from(analysis.sample_rate) * 0.01) as i64;
    let maximum_edits =
        ((target_frames / (beat_frames * settings.minimum_beats as f64)).ceil() as usize + 1)
            .min(MAX_SEARCH_EDITS);
    let mut closest_frames = analysis.source_frames as i64;
    let mut best: Option<(usize, f64, usize)> = None;
    for depth in 1..=maximum_edits {
        let mut unique: HashMap<(usize, i64), Candidate> = HashMap::new();
        for parent in &beam {
            let node = arena[*parent];
            let remaining = target_frames - node.output_frames as f64;
            for end in candidate_ends(analysis, settings, node.source_start) {
                let mut transitions = graph[end].clone();
                for divisor in [1.0, 2.0, 4.0, 8.0] {
                    let wanted = analysis.boundaries[end] as f64 - remaining / divisor;
                    let insertion = analysis
                        .boundaries
                        .partition_point(|frame| (*frame as f64) < wanted);
                    let phase_index = insertion.saturating_sub(insertion % 4) + end % 4;
                    for start in [
                        phase_index.saturating_sub(4),
                        phase_index,
                        phase_index.saturating_add(4),
                    ] {
                        if let Some(transition) = make_transition(analysis, settings, end, start) {
                            if !transitions.iter().any(|existing| existing.start == start) {
                                transitions.push(transition);
                            }
                        }
                    }
                }
                for transition in transitions {
                    let delta = analysis.boundaries[transition.end] as i64
                        - analysis.boundaries[transition.start] as i64;
                    let output_frames = node.output_frames + delta;
                    let maximum_frames = (MAX_SECONDS * f64::from(analysis.sample_rate)) as i64;
                    if output_frames <= 0
                        || output_frames > maximum_frames
                        || output_frames as usize > MAX_SAMPLES / usize::from(analysis.channels)
                    {
                        continue;
                    }
                    let extends = settings.target_seconds > analysis.duration_seconds;
                    if (extends && output_frames as f64 > target_frames + tolerance_frames)
                        || (!extends && (output_frames as f64) < target_frames - tolerance_frames)
                    {
                        continue;
                    }
                    if (output_frames as f64 - target_frames).abs()
                        < (closest_frames as f64 - target_frames).abs()
                    {
                        closest_frames = output_frames;
                    }
                    let segment_beats = transition.end - node.source_start;
                    let cost = node.cost
                        + transition.cost
                        + edit_penalty(settings)
                        + segment_penalty(segment_beats, settings);
                    let transition_cost = node.transition_cost + transition.cost;
                    let rank =
                        cost + (output_frames as f64 - target_frames).abs() / beat_frames * 0.20;
                    let candidate = Candidate {
                        parent: *parent,
                        transition,
                        output_frames,
                        cost,
                        transition_cost,
                        rank,
                    };
                    let key = (transition.start, output_frames / quantum);
                    match unique.get(&key) {
                        Some(existing) if existing.rank <= rank => {}
                        _ => {
                            unique.insert(key, candidate);
                        }
                    }
                }
            }
        }
        let mut candidates: Vec<Candidate> = unique.into_values().collect();
        candidates.sort_by(|left, right| {
            left.rank
                .total_cmp(&right.rank)
                .then_with(|| left.transition.end.cmp(&right.transition.end))
                .then_with(|| left.transition.start.cmp(&right.transition.start))
                .then_with(|| left.parent.cmp(&right.parent))
        });
        // Evaluate every feasible completion before pruning the continuing beam.
        for candidate in &candidates {
            if within_tolerance(candidate.output_frames, analysis.sample_rate, settings) {
                let final_cost = candidate.cost
                    + segment_penalty(analysis.beat_count - candidate.transition.start, settings);
                if best.is_none_or(|(_, cost, _)| final_cost < cost) {
                    arena.push(candidate_node(*candidate));
                    best = Some((arena.len() - 1, final_cost, depth));
                }
            }
        }
        if let Some((_, _, found_depth)) = best {
            let additional = ((1.0 - settings.edit_length) * 6.0).ceil() as usize;
            if depth >= found_depth + additional {
                break;
            }
        }
        beam = candidates
            .into_iter()
            .filter(|candidate| {
                !within_tolerance(candidate.output_frames, analysis.sample_rate, settings)
            })
            .filter(|candidate| best.is_none_or(|(_, cost, _)| candidate.cost < cost))
            .take(BEAM_WIDTH)
            .map(|candidate| {
                arena.push(candidate_node(candidate));
                arena.len() - 1
            })
            .collect();
        if beam.is_empty() {
            break;
        }
    }
    match best {
        Some((index, _, _)) => {
            let mut transitions = Vec::new();
            let mut cursor = index;
            while let Some(parent) = arena[cursor].parent {
                transitions.push(
                    arena[cursor]
                        .transition
                        .expect("non-root search node has an edit"),
                );
                cursor = parent;
            }
            transitions.reverse();
            assemble_plan(analysis, settings, &transitions)
        }
        None => Err(EngineError::UnreachableTarget {
            target_seconds: settings.target_seconds,
            tolerance_seconds: settings.tolerance_seconds,
            closest_seconds: closest_frames as f64 / f64::from(analysis.sample_rate),
        }),
    }
}

fn candidate_node(candidate: Candidate) -> SearchNode {
    SearchNode {
        parent: Some(candidate.parent),
        transition: Some(candidate.transition),
        source_start: candidate.transition.start,
        output_frames: candidate.output_frames,
        cost: candidate.cost,
        transition_cost: candidate.transition_cost,
    }
}

fn validate_settings(analysis: &Analysis, settings: &RemixSettings) -> Result<(), EngineError> {
    for (field, value) in [
        ("targetSeconds", settings.target_seconds),
        ("toleranceSeconds", settings.tolerance_seconds),
        ("editLength", settings.edit_length),
        ("featureBalance", settings.feature_balance),
        ("crossfadeMs", settings.crossfade_ms),
    ] {
        if !value.is_finite() {
            return Err(EngineError::InvalidSettings {
                field,
                reason: "value must be finite".to_owned(),
            });
        }
    }
    if settings.target_seconds <= 0.0 || settings.target_seconds > MAX_SECONDS {
        return Err(EngineError::InvalidSettings {
            field: "targetSeconds",
            reason: format!(
                "{} must be greater than zero and at most {MAX_SECONDS}",
                settings.target_seconds
            ),
        });
    }
    if settings.tolerance_seconds < 0.0 || settings.tolerance_seconds > MAX_SECONDS {
        return Err(EngineError::InvalidSettings {
            field: "toleranceSeconds",
            reason: format!(
                "{} must be between 0 and {MAX_SECONDS}",
                settings.tolerance_seconds
            ),
        });
    }
    for (field, value) in [
        ("editLength", settings.edit_length),
        ("featureBalance", settings.feature_balance),
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Err(EngineError::InvalidSettings {
                field,
                reason: format!("{value} must be between 0 and 1"),
            });
        }
    }
    if settings.minimum_beats == 0 || settings.minimum_beats > analysis.beat_count {
        return Err(EngineError::InvalidSettings {
            field: "minimumBeats",
            reason: format!(
                "{} must be between 1 and the {} analyzed beat intervals",
                settings.minimum_beats, analysis.beat_count
            ),
        });
    }
    if !(0.0..=500.0).contains(&settings.crossfade_ms) {
        return Err(EngineError::InvalidSettings {
            field: "crossfadeMs",
            reason: format!(
                "{} must be between 0 and 500 milliseconds",
                settings.crossfade_ms
            ),
        });
    }
    let shortest_segment = analysis
        .boundaries
        .windows(settings.minimum_beats + 1)
        .map(|bounds| bounds[bounds.len() - 1] - bounds[0])
        .min()
        .expect("validated minimum fits the beat grid");
    if crossfade_frames(analysis, settings) > shortest_segment / 2 {
        return Err(EngineError::InvalidSettings {
            field: "crossfadeMs",
            reason: format!(
                "{}ms exceeds half the shortest permitted musical segment",
                settings.crossfade_ms
            ),
        });
    }
    let samples = (settings.target_seconds * f64::from(analysis.sample_rate)).ceil() as usize
        * usize::from(analysis.channels);
    if samples > MAX_SAMPLES {
        return Err(EngineError::AudioTooLarge {
            samples,
            maximum_samples: MAX_SAMPLES,
        });
    }
    Ok(())
}

fn crossfade_frames(analysis: &Analysis, settings: &RemixSettings) -> usize {
    (settings.crossfade_ms / 1000.0 * f64::from(analysis.sample_rate)).round() as usize
}

fn within_tolerance(frames: i64, sample_rate: u32, settings: &RemixSettings) -> bool {
    (frames as f64 / f64::from(sample_rate) - settings.target_seconds).abs()
        <= settings.tolerance_seconds + f64::EPSILON * settings.target_seconds.max(1.0) * 4.0
}

fn transition_graph(analysis: &Analysis, settings: &RemixSettings) -> Vec<Vec<Transition>> {
    (0..=analysis.beat_count)
        .map(|end| {
            let mut transitions: Vec<Transition> = (end % 4..=analysis.beat_count)
                .step_by(4)
                .filter_map(|start| make_transition(analysis, settings, end, start))
                .collect();
            transitions.sort_by(|left, right| {
                left.cost
                    .total_cmp(&right.cost)
                    .then_with(|| left.start.cmp(&right.start))
            });
            transitions.truncate(MATCHES_PER_BOUNDARY);
            transitions
        })
        .collect()
}

fn make_transition(
    analysis: &Analysis,
    settings: &RemixSettings,
    end: usize,
    start: usize,
) -> Option<Transition> {
    let last = analysis.beat_count.checked_sub(settings.minimum_beats)?;
    if end < settings.minimum_beats
        || end > last
        || start < settings.minimum_beats
        || start > last
        || end % 4 != start % 4
        || end.abs_diff(start) < 4
    {
        return None;
    }
    if (settings.target_seconds < analysis.duration_seconds && start < end)
        || (settings.target_seconds > analysis.duration_seconds && start > end)
    {
        return None;
    }
    let left = &analysis.contexts[end];
    let right = &analysis.contexts[start];
    let correlation = dot(&left.phase, &right.phase);
    if correlation < -0.80 {
        return None;
    }
    let before = feature_cost(&left.before, &right.before, settings.feature_balance);
    let after = feature_cost(&left.after, &right.after, settings.feature_balance);
    let phase_cost = (1.0 - correlation.clamp(-1.0, 1.0)) * 0.5;
    let variation = variation_cost(end, start, settings.variation);
    Some(Transition {
        end,
        start,
        cost: before * 0.35 + after * 0.65 + phase_cost * 0.20 + variation,
    })
}

fn feature_cost(left: &BeatFeatures, right: &BeatFeatures, balance: f64) -> f64 {
    let harmonic = 1.0 - dot(&left.chroma, &right.chroma).clamp(0.0, 1.0);
    let timbre = 1.0 - dot(&left.timbre, &right.timbre).clamp(0.0, 1.0);
    let energy = f64::from((left.log_energy - right.log_energy).abs()).min(6.0) / 6.0;
    harmonic * balance + timbre * (1.0 - balance) + energy * 0.15
}

fn dot<const N: usize>(left: &[f32; N], right: &[f32; N]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn variation_cost(end: usize, start: usize, variation: u32) -> f64 {
    let value = (end as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (start as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ u64::from(variation).wrapping_mul(0x94d0_49bb_1331_11eb);
    let mixed = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    ((mixed ^ (mixed >> 27)) & 0xffff) as f64 / 65535.0 * 0.04
}

fn edit_penalty(settings: &RemixSettings) -> f64 {
    0.10 + settings.edit_length * 1.5
}

fn segment_penalty(beats: usize, settings: &RemixSettings) -> f64 {
    let preferred = settings.minimum_beats as f64 + 4.0 + settings.edit_length * 60.0;
    let ratio = beats as f64 / preferred;
    (1.0 - settings.edit_length) * (ratio - 1.0).max(0.0).powi(2) * 0.12
        + settings.edit_length * (1.0 / ratio - 1.0).max(0.0).powi(2) * 0.06
}

fn candidate_ends(analysis: &Analysis, settings: &RemixSettings, start: usize) -> Vec<usize> {
    let minimum = start + settings.minimum_beats;
    let maximum = analysis.beat_count.saturating_sub(settings.minimum_beats);
    if minimum > maximum {
        return Vec::new();
    }
    let stride = ((maximum - minimum) / 96).max(1);
    let mut ends: Vec<usize> = (minimum..=maximum).step_by(stride).collect();
    ends.push(maximum);
    for length in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128] {
        let end = start + settings.minimum_beats + length;
        if end <= maximum {
            ends.push(end);
        }
    }
    ends.sort_unstable();
    ends.dedup();
    ends
}

fn assemble_plan(
    analysis: &Analysis,
    settings: &RemixSettings,
    transitions: &[Transition],
) -> Result<RemixPlan, EngineError> {
    let crossfade = if transitions.is_empty() {
        0
    } else {
        crossfade_frames(analysis, settings)
    };
    let left_handle = crossfade / 2;
    let right_handle = crossfade - left_handle;
    let mut intervals = Vec::with_capacity(transitions.len() + 1);
    let mut source_start = 0;
    for transition in transitions {
        intervals.push((
            source_start,
            analysis.boundaries[transition.end] + right_handle,
        ));
        source_start = analysis.boundaries[transition.start] - left_handle;
    }
    intervals.push((source_start, analysis.source_frames));
    let mut cursor = 0_usize;
    let mut segments = Vec::with_capacity(intervals.len());
    for (index, (start, end)) in intervals.iter().copied().enumerate() {
        let output_start = if index == 0 { 0 } else { cursor - crossfade };
        cursor = output_start + end - start;
        let rate = f64::from(analysis.sample_rate);
        segments.push(Segment {
            source_start_seconds: start as f64 / rate,
            source_end_seconds: end as f64 / rate,
            output_start_seconds: output_start as f64 / rate,
            output_end_seconds: cursor as f64 / rate,
        });
    }
    let samples = cursor * usize::from(analysis.channels);
    if samples > MAX_SAMPLES {
        return Err(EngineError::AudioTooLarge {
            samples,
            maximum_samples: MAX_SAMPLES,
        });
    }
    if !within_tolerance(cursor as i64, analysis.sample_rate, settings) {
        return Err(EngineError::UnreachableTarget {
            target_seconds: settings.target_seconds,
            tolerance_seconds: settings.tolerance_seconds,
            closest_seconds: cursor as f64 / f64::from(analysis.sample_rate),
        });
    }
    Ok(RemixPlan {
        segments,
        duration_seconds: cursor as f64 / f64::from(analysis.sample_rate),
        crossfade_frames: crossfade,
        mean_transition_cost: if transitions.is_empty() {
            0.0
        } else {
            transitions
                .iter()
                .map(|transition| transition.cost)
                .sum::<f64>()
                / transitions.len() as f64
        },
        sample_rate: analysis.sample_rate,
        channels: analysis.channels,
        source_frames: analysis.source_frames,
        source_id: analysis.source_id,
        target_seconds: settings.target_seconds,
        tolerance_seconds: settings.tolerance_seconds,
    })
}

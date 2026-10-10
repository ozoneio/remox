//! Original-rate multichannel rendering with complementary cosine crossfades.
//! Beat-centered source handles are overlapped exactly once, so transition fades
//! consume their declared frame count without moving the beat or changing duration.

use std::f64::consts::PI;

use super::{source_id, validate_audio, Audio, EngineError, RemixPlan, MAX_SAMPLES};

#[derive(Clone, Copy)]
struct FrameSegment {
    source_start: usize,
    source_end: usize,
    output_start: usize,
    output_end: usize,
}

/// Render a validated plan without resampling, time stretching, padding, or clipping.
pub fn render(audio: &Audio, plan: &RemixPlan) -> Result<Audio, EngineError> {
    let source_frames = validate_audio(audio)?;
    if source_frames != plan.source_frames
        || audio.sample_rate != plan.sample_rate
        || audio.channels != plan.channels
        || source_id(audio) != plan.source_id
    {
        return Err(EngineError::SourceMismatch);
    }
    let segments = frame_segments(plan)?;
    let frames = segments
        .last()
        .expect("validated plan contains a segment")
        .output_end;
    let channels = usize::from(audio.channels);
    let samples = frames
        .checked_mul(channels)
        .ok_or_else(|| EngineError::InvalidPlan {
            reason: "output sample count overflowed".to_owned(),
        })?;
    if samples > MAX_SAMPLES {
        return Err(EngineError::AudioTooLarge {
            samples,
            maximum_samples: MAX_SAMPLES,
        });
    }
    let mut output = Vec::with_capacity(samples);
    for (index, segment) in segments.iter().enumerate() {
        let source = &audio.samples[segment.source_start * channels..segment.source_end * channels];
        if index == 0 {
            output.extend_from_slice(source);
            continue;
        }
        for frame in 0..plan.crossfade_frames {
            let progress = (frame as f64 + 0.5) / plan.crossfade_frames as f64;
            let incoming = (0.5 - 0.5 * (PI * progress).cos()) as f32;
            for channel in 0..channels {
                let output_index = (segment.output_start + frame) * channels + channel;
                output[output_index] = output[output_index] * (1.0 - incoming)
                    + source[frame * channels + channel] * incoming;
            }
        }
        output.extend_from_slice(&source[plan.crossfade_frames * channels..]);
    }
    if output.len() != samples {
        return Err(EngineError::InvalidPlan {
            reason: format!(
                "rendered {} samples but the plan specifies {samples}",
                output.len()
            ),
        });
    }
    Ok(Audio {
        samples: output,
        sample_rate: audio.sample_rate,
        channels: audio.channels,
    })
}

/// Return channel-inclusive peak bins normalized for display; PCM is unchanged.
pub fn waveform(audio: &Audio, bins: usize) -> Result<Vec<f32>, EngineError> {
    if bins == 0 {
        return Err(EngineError::InvalidSettings {
            field: "waveformBins",
            reason: "bin count must be greater than zero".to_owned(),
        });
    }
    let frames = validate_audio(audio)?;
    let channels = usize::from(audio.channels);
    let count = bins.min(frames);
    let peaks: Vec<f32> = (0..count)
        .map(|index| {
            let start = index * frames / count;
            let end = (index + 1) * frames / count;
            audio.samples[start * channels..end * channels]
                .iter()
                .fold(0.0_f32, |peak, value| peak.max(value.abs()))
        })
        .collect();
    let maximum = peaks.iter().copied().fold(0.0_f32, f32::max);
    Ok(peaks
        .into_iter()
        .map(|peak| if maximum > 0.0 { peak / maximum } else { 0.0 })
        .collect())
}

fn frame_segments(plan: &RemixPlan) -> Result<Vec<FrameSegment>, EngineError> {
    if plan.segments.is_empty() {
        return Err(EngineError::InvalidPlan {
            reason: "no source segments were provided".to_owned(),
        });
    }
    let rate = f64::from(plan.sample_rate);
    let mut result = Vec::with_capacity(plan.segments.len());
    let mut previous_end = 0;
    for (index, segment) in plan.segments.iter().enumerate() {
        let values = [
            segment.source_start_seconds,
            segment.source_end_seconds,
            segment.output_start_seconds,
            segment.output_end_seconds,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(EngineError::InvalidPlan {
                reason: format!("segment {index} has a negative or nonfinite time"),
            });
        }
        let source_start = (segment.source_start_seconds * rate).round() as usize;
        let source_end = (segment.source_end_seconds * rate).round() as usize;
        let output_start = (segment.output_start_seconds * rate).round() as usize;
        let output_end = (segment.output_end_seconds * rate).round() as usize;
        if source_end <= source_start || source_end > plan.source_frames {
            return Err(EngineError::InvalidPlan { reason: format!("segment {index} source frames {source_start}..{source_end} are outside the recording") });
        }
        let length = source_end - source_start;
        if length <= plan.crossfade_frames || (index > 0 && previous_end < plan.crossfade_frames) {
            return Err(EngineError::InvalidPlan {
                reason: format!("segment {index} is shorter than its crossfade"),
            });
        }
        let expected_start = if index == 0 {
            0
        } else {
            previous_end - plan.crossfade_frames
        };
        if output_start != expected_start || output_end.checked_sub(output_start) != Some(length) {
            return Err(EngineError::InvalidPlan {
                reason: format!(
                    "segment {index} does not match the overlap-accounted output timeline"
                ),
            });
        }
        if (index == 0 && source_start != 0)
            || (index == plan.segments.len() - 1 && source_end != plan.source_frames)
        {
            return Err(EngineError::InvalidPlan {
                reason: "the original opening and ending must be retained".to_owned(),
            });
        }
        result.push(FrameSegment {
            source_start,
            source_end,
            output_start,
            output_end,
        });
        previous_end = output_end;
    }
    let duration = previous_end as f64 / rate;
    let roundoff = f64::EPSILON * plan.target_seconds.max(1.0) * 4.0;
    if !plan.duration_seconds.is_finite()
        || (duration - plan.duration_seconds).abs() > roundoff
        || (duration - plan.target_seconds).abs() > plan.tolerance_seconds + roundoff
    {
        return Err(EngineError::InvalidPlan { reason: format!("rendered duration {duration:.6}s violates the planned duration or target tolerance") });
    }
    Ok(result)
}

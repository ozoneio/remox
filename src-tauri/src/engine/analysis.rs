//! Streaming-size FFT features and independently designed onset-period beat tracking.
//! Only a low-rate mono analysis buffer and compact frame/beat features are retained;
//! rendered PCM is always read from the original interleaved source.

use std::f32::consts::PI;

use rustfft::{num_complex::Complex, FftPlanner};

use super::{
    source_id, validate_audio, Analysis, Audio, BeatFeatures, BoundaryContext, EngineError,
};

const ANALYSIS_RATE: u32 = 11_025;
const FFT_SIZE: usize = 1024;
const HOP: usize = 256;

#[derive(Clone)]
struct SpectralFrame {
    features: BeatFeatures,
    novelty: f32,
}

/// Analyze harmonic, timbral, rhythmic, and waveform context without changing PCM.
pub fn analyze(audio: &Audio) -> Result<Analysis, EngineError> {
    let source_frames = validate_audio(audio)?;
    let peak = audio
        .samples
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    if peak < 0.000_01 {
        return Err(EngineError::SilentAudio);
    }
    let rate = audio.sample_rate.min(ANALYSIS_RATE);
    let mono = analysis_mono(audio, rate);
    let frames = spectral_frames(&mono, rate);
    let novelty: Vec<f32> = frames.iter().map(|frame| frame.novelty).collect();
    let (period, offset, confidence, onset_count) = estimate_period(&novelty, rate)?;
    let duration = source_frames as f64 / f64::from(audio.sample_rate);
    let beat_seconds = period * HOP as f64 / f64::from(rate);
    let offset_seconds = (offset * HOP as f64 + FFT_SIZE as f64 / 2.0) / f64::from(rate);
    let boundaries = beat_boundaries(
        &novelty,
        period,
        offset_seconds,
        beat_seconds,
        audio.sample_rate,
        rate,
        source_frames,
    );
    if boundaries.len() < 9 {
        return Err(EngineError::UnreliableBeat {
            confidence,
            onsets: onset_count,
        });
    }
    let features: Vec<BeatFeatures> = boundaries
        .windows(2)
        .map(|bounds| interval_features(&frames, bounds[0], bounds[1], audio.sample_rate, rate))
        .collect();
    let contexts = boundaries
        .iter()
        .enumerate()
        .map(|(index, frame)| BoundaryContext {
            before: features[index.saturating_sub(1)].clone(),
            after: features[index.min(features.len() - 1)].clone(),
            phase: phase_signature(audio, *frame),
        })
        .collect();
    let beats = boundaries
        .iter()
        .map(|frame| *frame as f64 / f64::from(audio.sample_rate))
        .collect();
    Ok(Analysis {
        tempo_bpm: 60.0 / beat_seconds,
        beat_count: boundaries.len() - 1,
        duration_seconds: duration,
        beats,
        boundaries,
        contexts,
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        source_frames,
        source_id: source_id(audio),
    })
}

/// Analyze one fixed highest-energy channel, avoiding stereo phase cancellation.
/// Box averaging limits high-frequency aliasing in this analysis-only decimation.
fn analysis_mono(audio: &Audio, rate: u32) -> Vec<f32> {
    let channels = usize::from(audio.channels);
    let source_frames = audio.samples.len() / channels;
    let energies: Vec<f64> = (0..channels)
        .map(|channel| {
            audio
                .samples
                .iter()
                .skip(channel)
                .step_by(channels)
                .map(|sample| f64::from(*sample).powi(2))
                .sum()
        })
        .collect();
    let selected_channel = (0..channels)
        .max_by(|left, right| energies[*left].total_cmp(&energies[*right]))
        .expect("validated audio has at least one channel");
    let count = (source_frames as u64 * u64::from(rate) / u64::from(audio.sample_rate)) as usize;
    (0..count)
        .map(|index| {
            let start = (index as u64 * u64::from(audio.sample_rate) / u64::from(rate)) as usize;
            let end = (((index + 1) as u64 * u64::from(audio.sample_rate) / u64::from(rate))
                as usize)
                .max(start + 1)
                .min(source_frames);
            let sum: f32 = (start..end)
                .map(|frame| audio.samples[frame * channels + selected_channel])
                .sum();
            sum / (end - start) as f32
        })
        .collect()
}

fn spectral_frames(mono: &[f32], rate: u32) -> Vec<SpectralFrame> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut buffer = vec![Complex::new(0.0, 0.0); FFT_SIZE];
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / (FFT_SIZE - 1) as f32).cos())
        .collect();
    let mut previous = vec![0.0_f32; FFT_SIZE / 2 + 1];
    let mut result = Vec::with_capacity(mono.len() / HOP + 1);
    for start in (0..mono.len()).step_by(HOP) {
        let mut energy = 0.0_f32;
        for (index, value) in buffer.iter_mut().enumerate() {
            let sample = mono.get(start + index).copied().unwrap_or(0.0);
            energy += sample * sample;
            *value = Complex::new(sample * window[index], 0.0);
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        let mut chroma = [0.0_f32; 12];
        let mut timbre = [0.0_f32; 8];
        let mut flux = 0.0_f32;
        for (bin, value) in buffer.iter().take(FFT_SIZE / 2 + 1).enumerate().skip(1) {
            let magnitude = value.norm() / FFT_SIZE as f32;
            flux += (magnitude - previous[bin]).max(0.0);
            previous[bin] = magnitude;
            let frequency = bin as f32 * rate as f32 / FFT_SIZE as f32;
            let band = ((frequency / 62.5).log2().floor() as i32).clamp(0, 7) as usize;
            timbre[band] += magnitude;
            if (55.0..=5000.0).contains(&frequency) {
                let note = 69.0 + 12.0 * (frequency / 440.0).log2();
                let pitch = (note.round() as i32).rem_euclid(12) as usize;
                chroma[pitch] += magnitude;
            }
        }
        result.push(SpectralFrame {
            features: BeatFeatures {
                chroma: normalize(chroma),
                timbre: normalize(timbre),
                log_energy: (energy / FFT_SIZE as f32 + 1.0e-12).ln(),
            },
            novelty: flux,
        });
    }
    result
}

fn normalize<const N: usize>(mut values: [f32; N]) -> [f32; N] {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        values.iter_mut().for_each(|value| *value /= norm);
    }
    values
}

/// Autocorrelation is normalized on matching overlaps and refined between hops.
fn estimate_period(novelty: &[f32], rate: u32) -> Result<(f64, f64, f64, usize), EngineError> {
    let average = novelty.iter().map(|value| f64::from(*value)).sum::<f64>() / novelty.len() as f64;
    let deviation = (novelty
        .iter()
        .map(|value| (f64::from(*value) - average).powi(2))
        .sum::<f64>()
        / novelty.len() as f64)
        .sqrt();
    let threshold = average + deviation * 0.5;
    let onsets = novelty
        .windows(3)
        .filter(|values| {
            f64::from(values[1]) > threshold && values[1] > values[0] && values[1] >= values[2]
        })
        .count();
    if onsets < 6 {
        return Err(EngineError::UnreliableBeat {
            confidence: 0.0,
            onsets,
        });
    }
    // A three-hop triangular kernel removes the accidental advantage of tempo
    // multiples whose attack spacing happens to fall on integer FFT hops.
    let centered: Vec<f64> = novelty
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let previous = novelty[index.saturating_sub(1)];
            let following = novelty[(index + 1).min(novelty.len() - 1)];
            (f64::from(previous) * 0.25 + f64::from(*value) * 0.5 + f64::from(following) * 0.25
                - average)
                .max(0.0)
        })
        .collect();
    let minimum = (f64::from(rate) * 60.0 / (200.0 * HOP as f64)).floor() as usize;
    let maximum = (f64::from(rate) * 60.0 / (60.0 * HOP as f64)).ceil() as usize;
    let correlation: Vec<f64> = (0..=maximum + 1)
        .map(|lag| {
            let mut product = 0.0;
            let mut left_energy = 0.0;
            let mut right_energy = 0.0;
            for index in lag..centered.len() {
                product += centered[index] * centered[index - lag];
                left_energy += centered[index] * centered[index];
                right_energy += centered[index - lag] * centered[index - lag];
            }
            if left_energy * right_energy > 0.0 {
                product / (left_energy * right_energy).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    let lag = (minimum..=maximum)
        .max_by(|left, right| {
            tempo_score(correlation[*left], *left, rate).total_cmp(&tempo_score(
                correlation[*right],
                *right,
                rate,
            ))
        })
        .expect("60–200 BPM includes analysis hop lags");
    let confidence = correlation[lag];
    if confidence < 0.20 {
        return Err(EngineError::UnreliableBeat { confidence, onsets });
    }
    let curvature = correlation[lag - 1] - 2.0 * correlation[lag] + correlation[lag + 1];
    let correction = if curvature.abs() > 1.0e-9 {
        (0.5 * (correlation[lag - 1] - correlation[lag + 1]) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let period = lag as f64 + correction;
    let offset = (0..lag)
        .max_by(|left, right| {
            phase_strength(&centered, *left as f64, period).total_cmp(&phase_strength(
                &centered,
                *right as f64,
                period,
            ))
        })
        .expect("beat period is positive") as f64;
    Ok((period, offset, confidence, onsets))
}

fn tempo_score(correlation: f64, lag: usize, rate: u32) -> f64 {
    let bpm = f64::from(rate) * 60.0 / (lag * HOP) as f64;
    correlation * (1.0 - 0.20 * (bpm / 120.0).ln().abs())
}

fn phase_strength(novelty: &[f64], offset: f64, period: f64) -> f64 {
    let mut position = offset;
    let mut sum = 0.0;
    let mut count = 0;
    while position < novelty.len() as f64 {
        let center = position.round() as usize;
        if center < novelty.len() {
            sum += novelty[center];
            count += 1;
        }
        position += period;
    }
    sum / count.max(1) as f64
}

fn beat_boundaries(
    novelty: &[f32],
    period: f64,
    offset_seconds: f64,
    beat_seconds: f64,
    sample_rate: u32,
    analysis_rate: u32,
    source_frames: usize,
) -> Vec<usize> {
    let duration = source_frames as f64 / f64::from(sample_rate);
    let mut position = offset_seconds;
    while position > beat_seconds * 0.35 {
        position -= beat_seconds;
    }
    let mut result = vec![0];
    while position < duration - beat_seconds * 0.35 {
        if position > beat_seconds * 0.35 {
            let frame_index = ((position * f64::from(analysis_rate) - FFT_SIZE as f64 / 2.0)
                / HOP as f64)
                .round()
                .max(0.0) as usize;
            let radius = (period * 0.15).ceil() as usize;
            let start = frame_index.saturating_sub(radius);
            let end = (frame_index + radius + 1).min(novelty.len());
            let peak =
                (start..end).max_by(|left, right| novelty[*left].total_cmp(&novelty[*right]));
            let aligned = peak
                .map(|index| (index * HOP + FFT_SIZE / 2) as f64 / f64::from(analysis_rate))
                .expect("beat search interval intersects analyzed audio");
            let source_frame = (aligned * f64::from(sample_rate)).round() as usize;
            if source_frame > *result.last().expect("opening boundary exists")
                && source_frame < source_frames
            {
                result.push(source_frame);
            }
        }
        position += beat_seconds;
    }
    result.push(source_frames);
    result
}

fn interval_features(
    frames: &[SpectralFrame],
    start: usize,
    end: usize,
    sample_rate: u32,
    analysis_rate: u32,
) -> BeatFeatures {
    let start_index =
        (start as f64 * f64::from(analysis_rate) / (f64::from(sample_rate) * HOP as f64)) as usize;
    let end_index =
        ((end as f64 * f64::from(analysis_rate) / (f64::from(sample_rate) * HOP as f64)) as usize)
            .max(start_index + 1)
            .min(frames.len());
    let mut result = BeatFeatures {
        chroma: [0.0; 12],
        timbre: [0.0; 8],
        log_energy: 0.0,
    };
    for frame in &frames[start_index..end_index] {
        for (sum, value) in result.chroma.iter_mut().zip(frame.features.chroma) {
            *sum += value;
        }
        for (sum, value) in result.timbre.iter_mut().zip(frame.features.timbre) {
            *sum += value;
        }
        result.log_energy += frame.features.log_energy;
    }
    result.chroma = normalize(result.chroma);
    result.timbre = normalize(result.timbre);
    result.log_energy /= (end_index - start_index) as f32;
    result
}

fn phase_signature(audio: &Audio, center: usize) -> [f32; 32] {
    let radius = (audio.sample_rate as usize / 100).max(1);
    let source_frames = audio.samples.len() / usize::from(audio.channels);
    let channels = usize::from(audio.channels);
    let signature = std::array::from_fn(|index| {
        let frame = center.saturating_sub(radius) + index * 2 * radius / 31;
        let bounded = frame.min(source_frames - 1);
        audio.samples[bounded * channels]
    });
    normalize(signature)
}

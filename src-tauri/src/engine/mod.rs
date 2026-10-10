//! Independently designed musical duration editing from a public behavioral brief.
//!
//! Behavioral reference: <https://helpx.adobe.com/uk/audition/desktop/mixing-multitrack-sessions/creating-remix.html>.
//! No proprietary implementation, assets, source separation model, or prior remix
//! engine is used. This implementation measures spectral flux, chroma, timbre,
//! energy, and waveform phase, then searches a beat transition graph. It preserves
//! the original opening, ending, sample rate, and channels; it does not time stretch.
//! Analysis assumes reasonably steady 60–200 BPM music and a four-beat transition
//! phase. Rubato, irregular meter, and recordings without reliable rhythmic attacks
//! can fail explicitly. Source and target lengths are limited to fifteen minutes.

mod analysis;
mod planner;
mod rendering;

pub use analysis::analyze;
pub use planner::plan;
pub use rendering::{render, waveform};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(crate) const MAX_SECONDS: f64 = 900.0;
pub(crate) const MAX_SAMPLES: usize = 64_000_000;

/// Decoded, finite PCM samples interleaved by channel, at their original rate.
#[derive(Clone, Debug)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// All preferences are explicit; no missing request field receives a default.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemixSettings {
    pub target_seconds: f64,
    pub tolerance_seconds: f64,
    pub edit_length: f64,
    pub feature_balance: f64,
    pub minimum_beats: usize,
    pub crossfade_ms: f64,
    pub variation: u32,
}

/// Beat times and tempo are measured from PCM rather than metadata tags.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Analysis {
    pub tempo_bpm: f64,
    pub beat_count: usize,
    pub duration_seconds: f64,
    pub beats: Vec<f64>,
    #[serde(skip)]
    pub(crate) boundaries: Vec<usize>,
    #[serde(skip)]
    pub(crate) contexts: Vec<BoundaryContext>,
    #[serde(skip)]
    pub(crate) sample_rate: u32,
    #[serde(skip)]
    pub(crate) channels: u16,
    #[serde(skip)]
    pub(crate) source_frames: usize,
    #[serde(skip)]
    pub(crate) source_id: u64,
}

/// Source intervals include fade handles; output intervals overlap at each join.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub source_start_seconds: f64,
    pub source_end_seconds: f64,
    pub output_start_seconds: f64,
    pub output_end_seconds: f64,
}

/// A bounded graph-search result with a frame-exact, overlap-accounted duration.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemixPlan {
    pub segments: Vec<Segment>,
    pub duration_seconds: f64,
    pub crossfade_frames: usize,
    pub mean_transition_cost: f64,
    #[serde(skip)]
    pub(crate) sample_rate: u32,
    #[serde(skip)]
    pub(crate) channels: u16,
    #[serde(skip)]
    pub(crate) source_frames: usize,
    #[serde(skip)]
    pub(crate) source_id: u64,
    #[serde(skip)]
    pub(crate) target_seconds: f64,
    #[serde(skip)]
    pub(crate) tolerance_seconds: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct BeatFeatures {
    pub chroma: [f32; 12],
    pub timbre: [f32; 8],
    pub log_energy: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundaryContext {
    pub before: BeatFeatures,
    pub after: BeatFeatures,
    pub phase: [f32; 32],
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Invalid PCM audio: {reason}")]
    InvalidAudio { reason: String },
    #[error("Source duration {seconds:.3}s exceeds the {maximum_seconds:.0}s analysis limit")]
    SourceTooLong { seconds: f64, maximum_seconds: f64 },
    #[error(
        "PCM audio exceeds the {maximum_samples} sample memory limit ({samples} samples requested)"
    )]
    AudioTooLarge {
        samples: usize,
        maximum_samples: usize,
    },
    #[error(
        "The recording is silent or below the usable analysis level; choose audible rhythmic music"
    )]
    SilentAudio,
    #[error("No reliable 60–200 BPM beat grid was found (periodicity {confidence:.3}, {onsets} rhythmic attacks); choose music with a steady audible beat")]
    UnreliableBeat { confidence: f64, onsets: usize },
    #[error("Invalid remix setting '{field}': {reason}")]
    InvalidSettings { field: &'static str, reason: String },
    #[error("Cannot reach {target_seconds:.3}s ± {tolerance_seconds:.3}s with compatible beat edits and the requested minimum segment; closest explored duration was {closest_seconds:.3}s. Increase tolerance, reduce minimum beats, or choose another target")]
    UnreachableTarget {
        target_seconds: f64,
        tolerance_seconds: f64,
        closest_seconds: f64,
    },
    #[error("Invalid remix plan: {reason}")]
    InvalidPlan { reason: String },
    #[error(
        "The remix plan belongs to a different source; analyze and plan the loaded audio again"
    )]
    SourceMismatch,
}

pub(crate) fn validate_audio(audio: &Audio) -> Result<usize, EngineError> {
    if !(8_000..=192_000).contains(&audio.sample_rate) {
        return Err(EngineError::InvalidAudio {
            reason: format!(
                "sample rate {} Hz is outside 8000–192000 Hz",
                audio.sample_rate
            ),
        });
    }
    if !(1..=8).contains(&audio.channels) {
        return Err(EngineError::InvalidAudio {
            reason: format!("channel count {} is outside 1–8", audio.channels),
        });
    }
    if audio.samples.is_empty()
        || !audio
            .samples
            .len()
            .is_multiple_of(usize::from(audio.channels))
    {
        return Err(EngineError::InvalidAudio {
            reason: format!(
                "{} samples do not form nonempty complete {}-channel frames",
                audio.samples.len(),
                audio.channels
            ),
        });
    }
    if audio.samples.len() > MAX_SAMPLES {
        return Err(EngineError::AudioTooLarge {
            samples: audio.samples.len(),
            maximum_samples: MAX_SAMPLES,
        });
    }
    if let Some(index) = audio.samples.iter().position(|sample| !sample.is_finite()) {
        return Err(EngineError::InvalidAudio {
            reason: format!("sample {index} is not finite"),
        });
    }
    let frames = audio.samples.len() / usize::from(audio.channels);
    let duration = frames as f64 / f64::from(audio.sample_rate);
    if duration > MAX_SECONDS {
        return Err(EngineError::SourceTooLong {
            seconds: duration,
            maximum_seconds: MAX_SECONDS,
        });
    }
    Ok(frames)
}

/// A lightweight identity check prevents accidentally rendering another loaded file.
pub(crate) fn source_id(audio: &Audio) -> u64 {
    let stride = (audio.samples.len() / 4096).max(1);
    audio
        .samples
        .iter()
        .step_by(stride)
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
            (hash ^ u64::from(sample.to_bits())).wrapping_mul(0x100_0000_01b3)
        })
        ^ audio.samples.len() as u64
        ^ (u64::from(audio.sample_rate) << 16)
        ^ u64::from(audio.channels)
}

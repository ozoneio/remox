//! Codec and filesystem boundaries for the independent remix engine.
//! Symphonia decodes locally; previews and exports contain the same float WAV bytes.

use std::{fs::File, io::ErrorKind, path::Path};

use symphonia::core::{
    audio::SampleBuffer,
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    errors::Error as DecodeError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};
use thiserror::Error;

use crate::engine::Audio;

const MAX_SAMPLES: usize = 64_000_000;
const MAX_SECONDS: f64 = 900.0;

#[derive(Debug, Error)]
pub enum MediaError {
    #[error("Cannot access audio file {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("Cannot decode audio file {path}: {reason}")]
    Decode { path: String, reason: String },
    #[error("Invalid audio file {path}: {reason}")]
    Invalid { path: String, reason: String },
    #[error("Cannot write WAV file {path}: {source}")]
    Write { path: String, source: hound::Error },
}

/// Decode a finite mono/stereo file without skipping corrupt packets or changing its rate.
pub fn decode_audio(path: &Path) -> Result<Audio, MediaError> {
    let label = path.display().to_string();
    let file = File::open(path).map_err(|source| MediaError::Read {
        path: label.clone(),
        source,
    })?;
    let bytes = file
        .metadata()
        .map_err(|source| MediaError::Read {
            path: label.clone(),
            source,
        })?
        .len();
    if bytes > 256 * 1024 * 1024 {
        return Err(MediaError::Invalid {
            path: label,
            reason: "File exceeds the 256 MB import limit".into(),
        });
    }
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }
    let source = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| MediaError::Decode {
            path: label.clone(),
            reason: error.to_string(),
        })?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .filter(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| MediaError::Invalid {
            path: label.clone(),
            reason: "No supported audio track found".into(),
        })?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    let expected_frames = params.n_frames;
    let sample_rate = params.sample_rate.ok_or_else(|| MediaError::Invalid {
        path: label.clone(),
        reason: "Audio track does not declare a sample rate".into(),
    })?;
    let channels = params
        .channels
        .ok_or_else(|| MediaError::Invalid {
            path: label.clone(),
            reason: "Audio track does not declare a channel layout".into(),
        })?
        .count();
    if !(8_000..=192_000).contains(&sample_rate) || !(1..=2).contains(&channels) {
        return Err(MediaError::Invalid {
            path: label,
            reason: format!(
                "Expected mono/stereo at 8–192 kHz; found {channels} channels at {sample_rate} Hz"
            ),
        });
    }
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|error| MediaError::Decode {
            path: label.clone(),
            reason: error.to_string(),
        })?;
    let mut samples: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(DecodeError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => {
                return Err(MediaError::Decode {
                    path: label,
                    reason: error.to_string(),
                })
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .map_err(|error| MediaError::Decode {
                path: label.clone(),
                reason: format!("Packet at timestamp {} failed: {error}", packet.ts()),
            })?;
        if decoded.spec().rate != sample_rate || decoded.spec().channels.count() != channels {
            return Err(MediaError::Invalid {
                path: label,
                reason: "Sample rate or channel layout changes within this file".into(),
            });
        }
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buffer.copy_interleaved_ref(decoded);
        if samples.len() + buffer.samples().len() > MAX_SAMPLES {
            return Err(MediaError::Invalid {
                path: label,
                reason: "Decoded audio exceeds the 256 MB processing limit".into(),
            });
        }
        if buffer.samples().iter().any(|sample| !sample.is_finite()) {
            return Err(MediaError::Invalid {
                path: label,
                reason: "Decoded audio contains non-finite samples".into(),
            });
        }
        samples.extend_from_slice(buffer.samples());
        if samples.len() as f64 / channels as f64 / sample_rate as f64 > MAX_SECONDS {
            return Err(MediaError::Invalid {
                path: label,
                reason: "Track exceeds the 15 minute import limit".into(),
            });
        }
    }
    if samples.is_empty() || !samples.len().is_multiple_of(channels) {
        return Err(MediaError::Invalid {
            path: label,
            reason: "Audio has no complete sample frames".into(),
        });
    }
    if let Some(frames) = expected_frames {
        // Lossless PCM declares an exact frame count; reject a truncated RIFF payload.
        if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
            && samples.len() / channels != frames as usize
        {
            return Err(MediaError::Invalid {
                path: label,
                reason: format!(
                    "Truncated WAV: header declares {frames} frames; decoded {}",
                    samples.len() / channels
                ),
            });
        }
    }
    Ok(Audio {
        samples,
        sample_rate,
        channels: channels as u16,
    })
}

/// Write unaltered finite samples as 32-bit float WAV, without normalization or clipping.
pub fn write_wav(path: &Path, audio: &Audio) -> Result<(), MediaError> {
    let label = path.display().to_string();
    if audio.channels == 0
        || audio.sample_rate == 0
        || audio.samples.is_empty()
        || !audio
            .samples
            .len()
            .is_multiple_of(usize::from(audio.channels))
        || audio.samples.iter().any(|sample| !sample.is_finite())
    {
        return Err(MediaError::Invalid {
            path: label,
            reason: "Output must contain finite, complete audio frames at a positive rate".into(),
        });
    }
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|source| MediaError::Write {
        path: label.clone(),
        source,
    })?;
    for sample in &audio.samples {
        writer
            .write_sample(*sample)
            .map_err(|source| MediaError::Write {
                path: label.clone(),
                source,
            })?;
    }
    writer.finalize().map_err(|source| MediaError::Write {
        path: label,
        source,
    })
}

use std::{f64::consts::TAU, path::PathBuf};

use remox_desktop_lib::engine::Audio;

/// Real PCM with a 120 BPM drum envelope, repeating harmonic phrases, and stereo phase.
pub fn rhythmic_audio(seconds: usize, sample_rate: u32) -> Audio {
    let frame_count = seconds * sample_rate as usize;
    let samples = (0..frame_count)
        .flat_map(|frame| {
            let time = frame as f64 / f64::from(sample_rate);
            let beat_time = time % 0.5;
            let phrase = ((time / 4.0) as usize) % 4;
            let root = [220.0, 261.6256, 196.0, 220.0][phrase];
            let kick = (TAU * 60.0 * beat_time).sin() * (-beat_time * 45.0).exp() * 0.6;
            let snap = ((TAU * 3337.0 * beat_time).sin() + (TAU * 5111.0 * beat_time).sin())
                * (-beat_time * 85.0).exp()
                * 0.11;
            let left = (TAU * root * time).sin() * 0.11
                + (TAU * root * 1.5 * time).sin() * 0.055
                + kick
                + snap;
            let right = (TAU * root * time + 0.37).sin() * 0.09
                + (TAU * root * 1.5 * time + 0.71).sin() * 0.045
                + kick * 0.85
                + snap;
            [left as f32, right as f32]
        })
        .collect();
    Audio {
        samples,
        sample_rate,
        channels: 2,
    }
}

pub fn temporary_wav(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("remox-{label}-{}.wav", uuid::Uuid::new_v4()))
}

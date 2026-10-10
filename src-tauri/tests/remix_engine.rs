mod common;

use remox_desktop_lib::{
    engine::{analyze, plan, render, waveform, Audio, EngineError, RemixSettings},
    media::{decode_audio, write_wav},
};

fn settings(target_seconds: f64) -> RemixSettings {
    RemixSettings {
        target_seconds,
        tolerance_seconds: 0.10,
        edit_length: 0.5,
        feature_balance: 0.6,
        minimum_beats: 4,
        crossfade_ms: 40.0,
        variation: 0,
    }
}

#[test]
fn stereo_wav_shortens_and_extends_with_preserved_ends_and_frame_exact_fades() {
    let source = common::rhythmic_audio(16, 48_000);
    let source_path = common::temporary_wav("rhythmic-source");
    write_wav(&source_path, &source).expect("write the real rhythmic PCM fixture");
    let decoded = decode_audio(&source_path).expect("decode rhythmic stereo WAV");
    let analysis = analyze(&decoded).expect("find a reliable beat grid in rhythmic audio");
    assert!(
        (analysis.tempo_bpm - 120.0).abs() < 5.0,
        "tempo {}",
        analysis.tempo_bpm
    );
    assert!(analysis.beat_count >= 28);
    for target in [10.0, 22.0] {
        let preferences = settings(target);
        let remix = plan(&analysis, &preferences).expect("find compatible edits for the target");
        let output = render(&decoded, &remix).expect("render at the original rate and channels");
        assert_eq!(output.sample_rate, decoded.sample_rate);
        assert_eq!(output.channels, decoded.channels);
        let actual_seconds = output.samples.len() as f64 / 2.0 / f64::from(output.sample_rate);
        assert!((actual_seconds - target).abs() <= preferences.tolerance_seconds);
        assert!((actual_seconds - remix.duration_seconds).abs() < 1.0e-9);
        assert!(remix.segments.len() > 1);
        assert!(output.samples.iter().all(|sample| sample.is_finite()));
        assert!(
            output
                .samples
                .iter()
                .map(|sample| sample * sample)
                .sum::<f32>()
                > 1.0
        );
        assert_eq!(&output.samples[..256], &decoded.samples[..256]);
        assert_eq!(
            &output.samples[output.samples.len() - 256..],
            &decoded.samples[decoded.samples.len() - 256..]
        );
        let copied_duration: f64 = remix
            .segments
            .iter()
            .map(|segment| segment.source_end_seconds - segment.source_start_seconds)
            .sum();
        let overlap = (remix.segments.len() - 1) as f64 * remix.crossfade_frames as f64
            / f64::from(output.sample_rate);
        assert!((copied_duration - overlap - actual_seconds).abs() < 1.0e-9);
        for join in remix.segments.windows(2) {
            assert!(
                (join[0].output_end_seconds
                    - join[1].output_start_seconds
                    - remix.crossfade_frames as f64 / f64::from(output.sample_rate))
                .abs()
                    < 1.0e-9
            );
        }
        let export_path = common::temporary_wav("rendered-remix");
        write_wav(&export_path, &output).expect("export remixed float PCM");
        let exported = decode_audio(&export_path).expect("decode remixed export");
        assert_eq!(exported.samples, output.samples);
        std::fs::remove_file(export_path).expect("remove temporary exported WAV");
    }
    std::fs::remove_file(source_path).expect("remove temporary source WAV");
}

#[test]
fn invalid_settings_silence_and_unreachable_targets_fail_explicitly() {
    let source = common::rhythmic_audio(16, 16_000);
    let analysis = analyze(&source).expect("analyze rhythmic fixture");
    let mut invalid = settings(16.0);
    invalid.minimum_beats = 0;
    assert!(matches!(
        plan(&analysis, &invalid),
        Err(EngineError::InvalidSettings {
            field: "minimumBeats",
            ..
        })
    ));
    invalid.minimum_beats = 4;
    invalid.feature_balance = f64::NAN;
    assert!(matches!(
        plan(&analysis, &invalid),
        Err(EngineError::InvalidSettings {
            field: "featureBalance",
            ..
        })
    ));
    let mut impossible = settings(1.0);
    impossible.tolerance_seconds = 0.0;
    assert!(matches!(
        plan(&analysis, &impossible),
        Err(EngineError::UnreachableTarget { .. })
    ));
    let silence = Audio {
        samples: vec![0.0; 16_000 * 8],
        sample_rate: 16_000,
        channels: 1,
    };
    assert!(matches!(analyze(&silence), Err(EngineError::SilentAudio)));
    let tone = Audio {
        samples: (0..16_000 * 8)
            .map(|index| (std::f32::consts::TAU * 440.0 * index as f32 / 16_000.0).sin() * 0.2)
            .collect(),
        sample_rate: 16_000,
        channels: 1,
    };
    assert!(matches!(
        analyze(&tone),
        Err(EngineError::UnreliableBeat { .. })
    ));
}

#[test]
fn anti_phase_stereo_is_analyzed_and_waveform_normalizes_display_only() {
    let source = common::rhythmic_audio(16, 16_000);
    let anti_phase = Audio {
        samples: source
            .samples
            .chunks_exact(2)
            .flat_map(|frame| [frame[0] * 4.0, -frame[0] * 4.0])
            .collect(),
        sample_rate: source.sample_rate,
        channels: 2,
    };
    let analysis =
        analyze(&anti_phase).expect("analyze the strongest channel without cancellation");
    let remix = plan(&analysis, &settings(10.0)).expect("plan anti-phase stereo music");
    let output = render(&anti_phase, &remix).expect("render anti-phase stereo music");
    assert!(output
        .samples
        .chunks_exact(2)
        .all(|frame| (frame[0] + frame[1]).abs() < 1.0e-6));
    let peaks = waveform(&anti_phase, 80).expect("build normalized waveform");
    assert_eq!(peaks.len(), 80);
    assert!(peaks.iter().all(|peak| (0.0..=1.0).contains(peak)));
    assert_eq!(peaks.iter().copied().fold(0.0_f32, f32::max), 1.0);
    assert!(anti_phase.samples.iter().any(|sample| sample.abs() > 1.0));
    assert!(matches!(
        waveform(&anti_phase, 0),
        Err(EngineError::InvalidSettings {
            field: "waveformBins",
            ..
        })
    ));
}

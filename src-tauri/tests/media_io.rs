mod common;

use remox_desktop_lib::media::{decode_audio, write_wav};

#[test]
fn float_wav_roundtrip_preserves_pcm_rate_and_channels() {
    let source = common::rhythmic_audio(1, 44_100);
    let path = common::temporary_wav("float-roundtrip");
    write_wav(&path, &source).expect("write float WAV");
    let decoded = decode_audio(&path).expect("decode float WAV");
    assert_eq!(decoded.sample_rate, source.sample_rate);
    assert_eq!(decoded.channels, source.channels);
    assert_eq!(decoded.samples, source.samples);
    let reader = hound::WavReader::open(&path).expect("inspect exported WAV header");
    assert_eq!(reader.spec().bits_per_sample, 32);
    assert_eq!(reader.spec().sample_format, hound::SampleFormat::Float);
    std::fs::remove_file(path).expect("remove roundtrip WAV");
}

#[test]
fn malformed_and_truncated_wav_are_rejected() {
    let malformed = common::temporary_wav("malformed");
    std::fs::write(&malformed, b"not a RIFF audio stream").expect("write malformed file");
    assert!(decode_audio(&malformed).is_err());
    std::fs::remove_file(malformed).expect("remove malformed WAV");
    let truncated = common::temporary_wav("truncated");
    write_wav(&truncated, &common::rhythmic_audio(1, 44_100))
        .expect("write valid WAV before truncation");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&truncated)
        .expect("open fixture for truncation");
    let original_bytes = file.metadata().expect("read fixture length").len();
    file.set_len(original_bytes - 8000)
        .expect("truncate PCM payload without changing declared length");
    drop(file);
    assert!(
        decode_audio(&truncated).is_err(),
        "a truncated WAV payload must not be accepted as a shorter song"
    );
    std::fs::remove_file(truncated).expect("remove truncated WAV");
}

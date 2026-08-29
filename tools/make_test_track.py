"""Generate a synthetic test track with clear sections, chords and drums.

Not a unit test - a fixture generator, so the pipeline can be exercised
end-to-end on audio with real metrical and harmonic structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM

# Section name -> (bars, chord roots in semitones from A2, brightness, has_drums, sings)
SECTIONS = [
    ("intro", 4, [0, 0, 5, 5], 0.3, False, False),
    ("verse", 8, [0, 0, 5, 7], 0.5, True, True),
    ("chorus", 8, [5, 7, 0, 0], 1.0, True, True),
    ("verse2", 8, [0, 0, 5, 7], 0.5, True, True),
    ("chorus2", 8, [5, 7, 0, 0], 1.0, True, True),
    ("bridge", 4, [3, 3, 10, 10], 0.4, False, False),
    ("chorus3", 8, [5, 7, 0, 0], 1.0, True, True),
    ("outro", 4, [0, 0, 0, 0], 0.25, False, False),
]


def _note(freq: float, n: int, brightness: float) -> np.ndarray:
    t = np.arange(n) / SR
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.5 * brightness * np.sin(2 * np.pi * freq * 2 * t)
    wave += 0.25 * brightness * np.sin(2 * np.pi * freq * 3 * t)
    env = np.exp(-2.5 * t)
    return wave * env


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    sweep = 110 * np.exp(-25 * t) + 45
    return np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-14 * t)


# Vowel formants (Hz) and bandwidths, roughly /a/ and /i/.
VOWELS = {
    "a": [(730, 80), (1090, 90), (2440, 120)],
    "i": [(270, 60), (2290, 100), (3010, 130)],
}

# Scale degrees of the sung phrase, in semitones above the chord root.
PHRASE = [0, 2, 3, 5, 3, 2, 0, -2]


def _resonator(x: np.ndarray, freq: float, bandwidth: float) -> np.ndarray:
    """Two-pole formant filter."""
    r = np.exp(-np.pi * bandwidth / SR)
    theta = 2 * np.pi * freq / SR
    return lfilter([1.0 - r], [1.0, -2 * r * np.cos(theta), r * r], x)


def _voice(freqs: np.ndarray, vowel_at: np.ndarray) -> np.ndarray:
    """Formant-filtered glottal pulse train over a continuous pitch contour.

    Rendered as one unbroken signal per phrase: real singing sustains across
    beat boundaries, and that continuity is exactly what the splice search has
    to detect. Rendering note-by-note would leave a gap at every beat and make
    every boundary look safe to cut.
    """
    n = len(freqs)
    t = np.arange(n) / SR
    vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(freqs * vibrato) / SR

    # Bandlimited-ish pulse train: first 24 harmonics with 1/k rolloff, plus a
    # little breath noise so the stem reads as voice rather than synth lead.
    source = sum(np.sin(phase * k) / k for k in range(1, 25))
    source += 0.02 * np.random.default_rng(1).standard_normal(n)

    out = np.zeros(n)
    for vowel in VOWELS:
        mask = (vowel_at == list(VOWELS).index(vowel)).astype(float)
        voiced = sum(_resonator(source, f, bw) for f, bw in VOWELS[vowel])
        out += voiced * mask

    attack, release = int(0.04 * SR), int(0.12 * SR)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[n - release :] = np.linspace(1, 0, release)
    return out * env


def _phrase(root: int, n_beats: int) -> np.ndarray:
    """Build a continuous sung phrase of ``n_beats`` beats over a chord root."""
    per_beat = int(BEAT * SR)
    freqs = np.empty(per_beat * n_beats)
    vowels = np.empty(per_beat * n_beats)

    for b in range(n_beats):
        degree = PHRASE[b % len(PHRASE)]
        freq = 220.0 * 2 ** ((root + degree) / 12.0)
        span = slice(b * per_beat, (b + 1) * per_beat)
        freqs[span] = freq
        vowels[span] = b % len(VOWELS)

    # Short glide between notes keeps the contour continuous rather than
    # stepped, which is both more voice-like and kinder to the separator.
    glide = max(1, int(0.02 * SR))
    kernel = np.ones(glide) / glide
    freqs = np.convolve(freqs, kernel, mode="same")
    freqs[:glide] = freqs[glide]
    freqs[-glide:] = freqs[-glide - 1]

    return _voice(freqs, vowels)


def _snare(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    return rng.standard_normal(n) * np.exp(-24 * t)


def _hat(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    return rng.standard_normal(n) * np.exp(-90 * t)


def build(with_vocals: bool = False) -> np.ndarray:
    """Render the fixture mix.

    Args:
        with_vocals: add a formant-synthesised lead that sings two bars then
            rests two bars, giving clear phrase gaps for the splice search to
            find (and clear phrases for it to avoid chopping).
    """
    return render(with_vocals=with_vocals)[0]


def build_with_stem() -> tuple[np.ndarray, np.ndarray]:
    """Render the mix and the isolated vocal bus at matching gain.

    Tests use the stem as ground truth instead of running a separator: the
    synthetic voice is not convincing enough for a real separation model
    (htdemucs routes it to 'other'), so relying on Demucs here would test the
    model rather than remox.
    """
    return render(with_vocals=True)


def render(with_vocals: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Render the fixture, returning ``(mix, vocal_bus)``."""
    rng = np.random.default_rng(0)
    total_bars = sum(bars for _, bars, _, _, _, _ in SECTIONS)
    total = int(total_bars * 4 * BEAT * SR) + SR
    out = np.zeros(total, dtype=np.float32)
    voice = np.zeros(total, dtype=np.float32)

    def into(buffer: np.ndarray, signal: np.ndarray, at: int, gain: float) -> None:
        end = min(at + len(signal), total)
        if end > at:
            buffer[at:end] += signal[: end - at].astype(np.float32) * gain

    def mix(signal: np.ndarray, at: int, gain: float) -> None:
        into(out, signal, at, gain)

    beat_index = 0
    for _, bars, roots, brightness, drums, sings in SECTIONS:
        for bar in range(bars):
            root = roots[bar % len(roots)]
            bar_start = beat_index

            # Chord: root, third, fifth held for the bar.
            chord_n = int(4 * BEAT * SR)
            for interval in (0, 4, 7):
                freq = 110.0 * 2 ** ((root + interval) / 12.0)
                mix(_note(freq, chord_n, brightness), int(bar_start * BEAT * SR), 0.18)

            # Bass on every beat.
            for b in range(4):
                freq = 55.0 * 2 ** (root / 12.0)
                at = int((bar_start + b) * BEAT * SR)
                mix(_note(freq, int(BEAT * SR), 0.2), at, 0.3)

            # Two bars sung, two bars silent - the sung span is continuous, so
            # the boundaries inside it are genuinely unsafe to cut, while the
            # rests are the places a good remix should choose.
            if with_vocals and sings and (bar % 4) == 0:
                phrase = _phrase(root, 8)
                at = int(bar_start * BEAT * SR)
                mix(phrase, at, 0.55)
                into(voice, phrase, at, 0.55)

            if drums:
                for b in range(4):
                    at = int((bar_start + b) * BEAT * SR)
                    if b in (0, 2):
                        mix(_kick(int(0.25 * SR)), at, 0.9)
                    if b in (1, 3):
                        mix(_snare(int(0.15 * SR), rng), at, 0.35)
                    for eighth in (0, 1):
                        mix(
                            _hat(int(0.05 * SR), rng),
                            at + int(eighth * BEAT * SR / 2),
                            0.12,
                        )

            beat_index += 4

    # Scale both buses by the same factor so the stem stays aligned with the
    # mix in absolute level - the vocal gate compares the two.
    peak = float(np.max(np.abs(out))) or 1.0
    scale = 0.9 / peak
    return (out * scale).astype(np.float32), (voice * scale).astype(np.float32)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_vocals = "--vocals" in sys.argv[1:]
    dest = Path(args[0]) if args else Path("tests/fixtures/track.wav")
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio = build(with_vocals=with_vocals)
    sf.write(str(dest), audio, SR)
    kind = "with vocals" if with_vocals else "instrumental"
    print(f"{dest}  {len(audio) / SR:.2f}s  {BPM:.0f} BPM  ({kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

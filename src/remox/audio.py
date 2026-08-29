"""Audio I/O and channel handling.

Analysis runs on a mono, fixed-rate signal; rendering happens on the original
full-rate (possibly stereo) signal so the output keeps the source fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

#: Sample rate used for all analysis (beat tracking, chroma, MFCC).
#: 22050 is the librosa default and is plenty for beat/harmony features,
#: while being ~2x cheaper than 44.1k.
ANALYSIS_SR = 22050


@dataclass
class Audio:
    """A loaded track.

    Attributes:
        samples: float32 array of shape ``(channels, n_samples)``. Always 2-D,
            even for mono input, so downstream code has one case to handle.
        sr: sample rate of ``samples``.
        path: source file, for error messages.
    """

    samples: np.ndarray
    sr: int
    path: Path

    @property
    def n_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.n_samples / self.sr

    def to_mono(self) -> np.ndarray:
        """Channel-averaged 1-D view for analysis."""
        if self.n_channels == 1:
            return self.samples[0]
        return self.samples.mean(axis=0)

    def analysis_signal(self) -> np.ndarray:
        """Mono signal resampled to :data:`ANALYSIS_SR`."""
        mono = self.to_mono()
        if self.sr == ANALYSIS_SR:
            return mono
        return librosa.resample(mono, orig_sr=self.sr, target_sr=ANALYSIS_SR)


def load(path: str | Path) -> Audio:
    """Read an audio file at its native sample rate, preserving channels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    # sr=None keeps the native rate; mono=False keeps channel layout.
    samples, sr = librosa.load(str(path), sr=None, mono=False)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]

    if samples.shape[1] == 0:
        raise ValueError(f"audio file is empty: {path}")

    return Audio(samples=samples, sr=int(sr), path=path)


def save(path: str | Path, samples: np.ndarray, sr: int, subtype: str | None = None) -> None:
    """Write ``(channels, n_samples)`` audio to disk.

    soundfile expects ``(n_samples, channels)``, so we transpose here rather
    than making every caller remember the convention.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if samples.ndim == 1:
        samples = samples[np.newaxis, :]

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        # Crossfade sums can nudge past full scale. Pull back rather than clip.
        samples = samples / peak

    sf.write(str(path), samples.T, sr, subtype=subtype)

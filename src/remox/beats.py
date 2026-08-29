"""Beat and downbeat estimation.

Splices only ever happen on beat boundaries, and only between beats that sit at
the same position within the bar. Everything downstream therefore depends on
the grid produced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from remox.audio import ANALYSIS_SR

#: STFT hop for onset/beat analysis. 512 @ 22050 Hz => ~23 ms resolution,
#: comfortably finer than the ~100 ms precision a beat tracker can deliver.
HOP_LENGTH = 512

#: Below this many beats there is nothing meaningful to re-edit.
MIN_BEATS = 16


class BeatTrackingError(RuntimeError):
    """Raised when the track has no usable beat grid."""


@dataclass
class BeatGrid:
    """Beat boundaries and metrical structure for a track.

    ``times`` holds ``n_beats + 1`` boundaries: beat ``i`` spans
    ``times[i] .. times[i + 1]``. Storing boundaries rather than onsets means
    the duration of every beat is well defined, including the last one.

    ``downbeat_phase`` is the offset such that beat ``i`` begins a bar when
    ``(i - downbeat_phase) % meter == 0``.
    """

    times: np.ndarray
    tempo: float
    meter: int
    downbeat_phase: int

    @property
    def n_beats(self) -> int:
        return len(self.times) - 1

    @property
    def start(self) -> float:
        """Time of the first tracked beat; audio before this is the intro."""
        return float(self.times[0])

    @property
    def end(self) -> float:
        """Time of the last tracked beat; audio after this is the outro."""
        return float(self.times[-1])

    def durations(self) -> np.ndarray:
        """Length in seconds of each beat."""
        return np.diff(self.times)

    def phase(self, i: int | np.ndarray):
        """Position of beat ``i`` within its bar (0 == downbeat)."""
        return (i - self.downbeat_phase) % self.meter

    def bar_index(self, i: int | np.ndarray):
        """Zero-based bar number containing beat ``i``."""
        return (i - self.downbeat_phase) // self.meter


def _estimate_downbeat_phase(
    beat_frames: np.ndarray,
    onset_env: np.ndarray,
    chroma: np.ndarray,
    meter: int,
) -> int:
    """Pick the bar phase that best explains where accents fall.

    Two cues are combined:

    * onset strength - downbeats usually carry the loudest transient (kick).
    * chroma flux - chord changes tend to land on downbeats.

    Both are z-scored so neither dominates purely because of its scale.
    """
    n_beats = len(beat_frames)
    if n_beats < meter * 2:
        return 0

    # Onset strength sampled at each beat.
    idx = np.clip(beat_frames, 0, len(onset_env) - 1)
    beat_accent = onset_env[idx].astype(np.float64)

    # Harmonic change entering each beat. First beat has no predecessor, so it
    # gets the mean rather than an artificial zero.
    beat_chroma = _sync_to_beats(chroma, beat_frames)
    flux = np.abs(np.diff(beat_chroma, axis=1)).sum(axis=0)
    flux = np.concatenate([[flux.mean() if flux.size else 0.0], flux])

    def z(x: np.ndarray) -> np.ndarray:
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

    score_signal = z(beat_accent) + 0.5 * z(flux)

    scores = [score_signal[phase::meter].mean() for phase in range(meter)]
    return int(np.argmax(scores))


def _sync_to_beats(frames: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    """Average frame-rate features within each beat.

    Returns an array of shape ``(n_features, len(beat_frames))``. The final beat
    is aggregated up to the end of the feature matrix.
    """
    n_frames = frames.shape[1]
    bounds = np.append(beat_frames, n_frames)
    out = np.empty((frames.shape[0], len(beat_frames)), dtype=frames.dtype)
    for i in range(len(beat_frames)):
        lo = int(np.clip(bounds[i], 0, n_frames - 1))
        hi = int(np.clip(bounds[i + 1], lo + 1, n_frames))
        out[:, i] = frames[:, lo:hi].mean(axis=1)
    return out


def track(y: np.ndarray, sr: int = ANALYSIS_SR, meter: int = 4) -> BeatGrid:
    """Estimate the beat grid of a mono analysis signal.

    Args:
        y: mono float signal.
        sr: sample rate of ``y``.
        meter: beats per bar. 4 covers the overwhelming majority of the music
            this tool is aimed at; 3 is offered for waltz-time material.

    Raises:
        BeatTrackingError: if too few beats were found to re-edit anything.
    """
    if meter < 2:
        raise ValueError("meter must be at least 2")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
        trim=False,
    )
    tempo = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)

    if len(beat_frames) < MIN_BEATS:
        raise BeatTrackingError(
            f"only {len(beat_frames)} beats detected (need >= {MIN_BEATS}); "
            "the track may be too short, too quiet, or have no steady pulse"
        )

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    phase = _estimate_downbeat_phase(beat_frames, onset_env, chroma, meter)

    times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    # `times` are beat onsets; we need one extra boundary to close the final
    # beat. Extrapolate using the median beat period.
    period = float(np.median(np.diff(times))) if len(times) > 1 else 60.0 / max(tempo, 1e-6)
    closing = min(times[-1] + period, len(y) / sr)
    if closing <= times[-1]:
        # Last beat sits at the very end of the file: drop it instead of
        # creating a zero-length interval.
        times = times[:-1]
        closing = float(len(y) / sr)
    times = np.append(times, closing)

    return BeatGrid(times=times, tempo=tempo, meter=meter, downbeat_phase=phase)

"""Per-beat feature extraction.

Each beat is reduced to a single vector describing what is happening musically
during that beat. Splice quality is judged entirely by distances between these
vectors, so the choice and weighting of features is what makes cuts sound
intentional rather than random.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from remox.audio import ANALYSIS_SR
from remox.beats import HOP_LENGTH, BeatGrid


@dataclass
class FeatureWeights:
    """Relative influence of each feature block on splice cost.

    Each block is normalised to unit average norm before weighting, so these
    numbers are directly comparable.

    * ``chroma`` - pitch class content. Governs whether a cut is harmonically
      plausible; the single most important cue, hence the largest weight.
    * ``timbre`` - MFCCs. Catches arrangement changes (drums drop in, synth
      pad enters) that chroma is blind to.
    * ``energy`` - loudness. Prevents cutting from a quiet breakdown straight
      into a full chorus.
    """

    chroma: float = 1.0
    timbre: float = 0.6
    energy: float = 0.5


def _block_normalise(block: np.ndarray) -> np.ndarray:
    """Z-score each dimension, then scale the block to unit average norm.

    Per-dimension z-scoring stops one loud MFCC coefficient from swamping the
    other twelve. The second step puts every block on the same footing so
    :class:`FeatureWeights` means what it says.
    """
    mean = block.mean(axis=0, keepdims=True)
    std = block.std(axis=0, keepdims=True)
    block = (block - mean) / np.maximum(std, 1e-9)

    avg_norm = np.linalg.norm(block, axis=1).mean()
    if avg_norm > 1e-9:
        block = block / avg_norm
    return block


@dataclass
class FeatureBlocks:
    """Per-beat feature blocks, normalised but not yet weighted.

    Kept separate from the combined matrix because the expensive work (frame-
    rate chroma, MFCC and RMS over the whole track) does not depend on the
    weights. Re-weighting for another variant is then just three multiplies.

    Each block has shape ``(n_beats, n_dims)``.
    """

    chroma: np.ndarray
    timbre: np.ndarray
    energy: np.ndarray

    @property
    def n_beats(self) -> int:
        return self.chroma.shape[0]

    def combine(self, weights: FeatureWeights | None = None) -> np.ndarray:
        """Apply weights and concatenate into a single feature matrix."""
        weights = weights or FeatureWeights()
        return np.hstack(
            [
                self.chroma * weights.chroma,
                self.timbre * weights.timbre,
                self.energy * weights.energy,
            ]
        )


def blocks(y: np.ndarray, grid: BeatGrid, sr: int = ANALYSIS_SR) -> FeatureBlocks:
    """Compute normalised per-beat feature blocks.

    Args:
        y: mono analysis signal.
        grid: beat grid for ``y``.
        sr: sample rate of ``y``.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
    # Coefficient 0 is overall loudness, which `rms` already covers and which
    # varies with mastering rather than content. Drop it from the timbre block.
    mfcc = mfcc[1:]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    rms = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)

    n_frames = min(chroma.shape[1], mfcc.shape[1], rms.shape[1])
    chroma, mfcc, rms = chroma[:, :n_frames], mfcc[:, :n_frames], rms[:, :n_frames]

    # Beat boundaries in frame units. `grid.times` has n_beats + 1 entries.
    bounds = librosa.time_to_frames(grid.times, sr=sr, hop_length=HOP_LENGTH)
    bounds = np.clip(bounds, 0, n_frames)

    aggregated = []
    for frames in (chroma, mfcc, rms):
        agg = np.empty((grid.n_beats, frames.shape[0]), dtype=np.float64)
        for i in range(grid.n_beats):
            lo = int(bounds[i])
            hi = max(int(bounds[i + 1]), lo + 1)
            hi = min(hi, n_frames)
            if lo >= hi:
                lo = max(hi - 1, 0)
            agg[i] = frames[:, lo:hi].mean(axis=1)
        aggregated.append(_block_normalise(agg))

    return FeatureBlocks(*aggregated)


def extract(
    y: np.ndarray,
    grid: BeatGrid,
    sr: int = ANALYSIS_SR,
    weights: FeatureWeights | None = None,
) -> np.ndarray:
    """Build the per-beat feature matrix.

    Args:
        y: mono analysis signal.
        grid: beat grid for ``y``.
        sr: sample rate of ``y``.
        weights: block weighting; defaults to :class:`FeatureWeights`.

    Returns:
        Array of shape ``(grid.n_beats, n_features)``.
    """
    return blocks(y, grid, sr).combine(weights)

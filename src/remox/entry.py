"""Choosing where a remix should begin.

By default an edit opens at beat one, which is rarely what anyone wants from a
30-second cut: the first bars of a track are usually the least characteristic
part of it. This module scores every beat as a candidate opening so the solver
can start at a chorus, a drop, or wherever the music actually asserts itself.

Four cues, combined into a single "badness" score in ``[0, 1]``:

* **Novelty** - how much the music changes at that point. Section boundaries
  make intentional-sounding openings; starting mid-phrase sounds like a botched
  fade-in.
* **Energy** - loud sections are what people mean by "start at the chorus".
* **Vocal safety** - reuses the phrase-crossing score, so a cut never opens
  halfway through a sung word.
* **Phrase alignment** - four-bar boundaries beat plain downbeats.

Novelty and energy are rank-normalised rather than min-maxed, so a single
outlier bar cannot flatten the rest of the scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from remox.beats import BeatGrid
from remox.features import FeatureBlocks

#: Beats either side of a candidate compared when measuring novelty. Eight is
#: two bars at 4/4 - long enough to span a section change, short enough not to
#: blur across two of them.
NOVELTY_WINDOW = 8


@dataclass
class EntryWeights:
    """Relative influence of each opening cue."""

    #: Prefer points where the music changes character.
    novelty: float = 1.0
    #: Prefer loud, fully-arranged sections over sparse ones.
    energy: float = 1.0
    #: Avoid opening in the middle of a sung phrase.
    vocal: float = 1.5
    #: Prefer 4-bar phrase boundaries over bare downbeats.
    phrase: float = 0.5


def _rank01(values: np.ndarray) -> np.ndarray:
    """Map values to ``[0, 1]`` by rank, so outliers cannot dominate."""
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(np.argsort(values))
    return order / (len(values) - 1)


def novelty(features: np.ndarray, window: int = NOVELTY_WINDOW) -> np.ndarray:
    """Foote-style novelty: how different the music is either side of each beat.

    Compares the mean feature vector of the ``window`` beats before a beat
    against the ``window`` beats from it onwards. Peaks land on section
    boundaries.
    """
    n = len(features)
    scores = np.zeros(n, dtype=np.float64)
    if n < 4:
        return scores

    # Cumulative sums let each window mean be computed in constant time.
    padded = np.vstack([np.zeros((1, features.shape[1])), np.cumsum(features, axis=0)])

    def block_mean(lo: int, hi: int) -> np.ndarray:
        return (padded[hi] - padded[lo]) / max(hi - lo, 1)

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window)
        # Need real material on both sides, otherwise the score reflects
        # padding rather than a musical change.
        if i - lo < 2 or hi - i < 2:
            continue
        scores[i] = float(np.linalg.norm(block_mean(lo, i) - block_mean(i, hi)))
    return scores


def costs(
    blocks: FeatureBlocks,
    grid: BeatGrid,
    crossing: np.ndarray | None = None,
    weights: EntryWeights | None = None,
    align: str = "phrase",
    allow_first_beat: bool = True,
) -> np.ndarray:
    """Score every beat as an opening point.

    Args:
        blocks: per-beat feature blocks.
        grid: beat grid.
        crossing: vocal phrase-crossing scores, length ``n_beats + 1``.
        weights: cue weighting.
        align: ``"phrase"`` restricts openings to 4-bar boundaries, ``"bar"``
            to any downbeat, ``"beat"`` to anywhere.
        allow_first_beat: always keep beat 0 as a candidate, so "auto" can
            still choose the natural beginning when it genuinely scores best.

    Returns:
        Array of length ``n_beats``. Finite values are costs in ``[0, 1]``
        where lower is better; ``inf`` marks a beat that may not be used.
    """
    weights = weights or EntryWeights()
    n = grid.n_beats
    beats = np.arange(n)

    if align == "phrase":
        allowed = (grid.phase(beats) == 0) & (grid.bar_index(beats) % 4 == 0)
    elif align == "bar":
        allowed = grid.phase(beats) == 0
    elif align == "beat":
        allowed = np.ones(n, dtype=bool)
    else:
        raise ValueError(f"unknown align mode: {align!r}")

    if allow_first_beat:
        allowed = allowed.copy()
        allowed[0] = True

    if not allowed.any():
        allowed[0] = True

    combined = blocks.combine()
    good_novelty = _rank01(novelty(combined))
    good_energy = _rank01(blocks.energy[:, 0])

    badness = weights.novelty * (1.0 - good_novelty)
    badness = badness + weights.energy * (1.0 - good_energy)
    total = weights.novelty + weights.energy

    if crossing is not None and np.any(crossing > 0):
        if len(crossing) != n + 1:
            raise ValueError(f"crossing has length {len(crossing)}, expected {n + 1}")
        badness = badness + weights.vocal * np.asarray(crossing[:n], dtype=np.float64)
        total += weights.vocal

    if align != "phrase":
        # Nudge towards phrase boundaries without forbidding the rest.
        off_phrase = ((grid.phase(beats) != 0) | (grid.bar_index(beats) % 4 != 0)).astype(float)
        badness = badness + weights.phrase * off_phrase
        total += weights.phrase

    badness = badness / max(total, 1e-9)
    return np.where(allowed, badness, np.inf)


def resolve_fixed(grid: BeatGrid, when: float, align: str = "bar") -> np.ndarray:
    """Entry costs that force the opening to a specific point in time.

    The requested time is snapped forward to the next permitted boundary, so
    an explicit ``--start`` still lands on the grid rather than mid-bar.
    """
    n = grid.n_beats
    beats = np.arange(n)
    if align == "phrase":
        allowed = (grid.phase(beats) == 0) & (grid.bar_index(beats) % 4 == 0)
    elif align == "bar":
        allowed = grid.phase(beats) == 0
    else:
        allowed = np.ones(n, dtype=bool)

    candidates = beats[allowed & (grid.times[:n] >= when)]
    if len(candidates) == 0:
        # Requested a point past the last usable boundary; fall back to the
        # latest one that exists rather than failing outright.
        candidates = beats[allowed]
    if len(candidates) == 0:
        candidates = np.array([0])

    chosen = int(candidates[0])
    out = np.full(n, np.inf)
    out[chosen] = 0.0
    return out

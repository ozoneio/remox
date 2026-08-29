"""Splice candidate discovery.

A *jump* means: after playing beat ``src``, continue at beat ``dst`` instead of
``src + 1``. Backward jumps repeat material (used to lengthen), forward jumps
skip material (used to shorten).

Cost is measured over a context window rather than a single beat. Comparing
lone beats is not enough - two beats can have near-identical spectra while the
bars leading into them are completely different, which is exactly the splice
that sounds wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from remox.beats import BeatGrid

#: How many beats of lead-in are compared when scoring a splice. Four beats
#: (one bar at 4/4) is the smallest window that reliably captures the groove.
CONTEXT_BEATS = 4

#: Weight decay across the context window. The beat immediately before the cut
#: matters most; earlier beats matter progressively less.
CONTEXT_DECAY = 0.6


@dataclass
class Jumps:
    """Accepted splice candidates.

    Attributes:
        src: source beat indices.
        dst: destination beat indices.
        cost: splice cost, normalised so 1.0 is the median candidate.
    """

    src: np.ndarray
    dst: np.ndarray
    cost: np.ndarray

    def __len__(self) -> int:
        return len(self.src)

    def by_source(self, n_beats: int) -> list[list[tuple[int, float]]]:
        """Adjacency list: ``out[i]`` holds ``(dst, cost)`` for jumps from beat ``i``."""
        out: list[list[tuple[int, float]]] = [[] for _ in range(n_beats)]
        for s, d, c in zip(self.src, self.dst, self.cost):
            out[int(s)].append((int(d), float(c)))
        return out

    def pairs(self) -> set[tuple[int, int]]:
        """Every ``(src, dst)`` splice in this candidate set."""
        return {(int(s), int(d)) for s, d in zip(self.src, self.dst)}

    def exclude(self, forbidden: set[tuple[int, int]]) -> "Jumps":
        """Drop the given splices.

        Used to force plan diversity: after accepting a plan, its splices are
        forbidden so the next solve has to find genuinely different cuts rather
        than rediscovering the same optimum.
        """
        if not forbidden:
            return self
        keep = np.array(
            [(int(s), int(d)) not in forbidden for s, d in zip(self.src, self.dst)],
            dtype=bool,
        )
        return Jumps(src=self.src[keep], dst=self.dst[keep], cost=self.cost[keep])


def _context_distance(features: np.ndarray, context: int) -> np.ndarray:
    """Windowed distance matrix ``C[i, j]``.

    ``C[i, j]`` compares the ``context`` beats ending at ``i`` (what the
    listener just heard) against the ``context`` beats ending at ``j - 1``
    (what normally precedes ``j``). Low values mean beat ``j`` is a natural
    continuation of the material before the cut.
    """
    n = len(features)
    base = cdist(features, features, metric="euclidean")

    weights = CONTEXT_DECAY ** np.arange(context)
    weights /= weights.sum()

    out = np.zeros((n, n), dtype=np.float64)
    for w, weight in enumerate(weights):
        if n - w - 1 <= 0:
            break
        # C[i, j] += weight * base[i - w, j - 1 - w]
        out[w:, w + 1 :] += weight * base[: n - w, : n - w - 1]
    return out


def find(
    features: np.ndarray,
    grid: BeatGrid,
    context: int = CONTEXT_BEATS,
    min_jump_bars: int = 2,
    align: str = "bar",
    keep_percentile: float = 12.0,
    max_candidates: int = 4000,
    vocal_crossing: np.ndarray | None = None,
    vocal_weight: float = 3.0,
    vocal_veto: float | None = None,
) -> Jumps:
    """Score every legal splice and keep the best ones.

    Args:
        features: per-beat feature matrix from :func:`remox.features.extract`.
        grid: beat grid.
        context: lead-in length in beats used for scoring.
        min_jump_bars: minimum distance a jump must cover, in bars. Short jumps
            produce stutter artefacts even when they score well.
        align: metrical constraint. ``"beat"`` allows any same-phase splice,
            ``"bar"`` restricts cuts to bar lines, ``"phrase"`` further
            restricts them to 4-bar phrase boundaries (most musical, fewest
            options).
        keep_percentile: keep only splices in this best percentile of cost.
        max_candidates: hard cap on retained candidates, for solver speed.
        vocal_crossing: per-boundary phrase-continuity scores from
            :func:`remox.vocals.crossing_scores`, length ``n_beats + 1``.
        vocal_weight: cost added per unit of vocal crossing, in units of
            median splice cost. Applied to both ends of the splice.
        vocal_veto: if set, reject any splice whose crossing exceeds this on
            either end, regardless of how good it sounds spectrally.

    Returns:
        The retained candidates, cost-normalised against their own median.
    """
    n = grid.n_beats
    if n <= context + 2:
        return Jumps(np.array([], int), np.array([], int), np.array([], float))

    acoustic = _context_distance(features, context)

    idx = np.arange(n)
    src = idx[:, None]
    dst = idx[None, :]

    # Both endpoints need a full context window behind them, otherwise the
    # score is computed from padding rather than music.
    valid = (src >= context - 1) & (dst >= context)

    # The replaced beat (src + 1) and the destination must occupy the same
    # position in the bar, or the groove flips phase at the cut.
    valid &= grid.phase(src + 1) == grid.phase(dst)

    if align in ("bar", "phrase"):
        valid &= grid.phase(dst) == 0
    if align == "phrase":
        valid &= (grid.bar_index(dst) % 4) == (grid.bar_index(src + 1) % 4)
    elif align not in ("beat", "bar"):
        raise ValueError(f"unknown align mode: {align!r}")

    # Exclude the do-nothing edge and anything too short to be heard as an edit.
    min_beats = max(min_jump_bars * grid.meter, grid.meter)
    valid &= np.abs(dst - (src + 1)) >= min_beats

    # A jump must leave somewhere to go.
    valid &= dst < n

    if not valid.any():
        return Jumps(np.array([], int), np.array([], int), np.array([], float))

    # Normalise the acoustic cost against the median legal splice *before*
    # adding the vocal term, so `vocal_weight` is expressed in "median splice"
    # units and is comparable across tracks with different feature scales.
    median = float(np.median(acoustic[valid])) or 1.0
    score = acoustic / median

    if vocal_crossing is not None:
        crossing = np.asarray(vocal_crossing, dtype=np.float64)
        # Validate shape before the all-zero short circuit: a malformed array
        # must be an error even when its contents happen to be harmless.
        if len(crossing) != n + 1:
            raise ValueError(
                f"vocal_crossing has length {len(crossing)}, expected {n + 1}"
            )
        if np.any(crossing > 0):
            # A cut chops a phrase if vocals span the boundary it leaves from
            # (src + 1) or the one it lands on (dst).
            leaving = crossing[np.arange(n) + 1][:, None]
            landing = crossing[np.arange(n)][None, :]
            score = score + vocal_weight * (leaving + landing)
            if vocal_veto is not None:
                valid = valid & (leaving <= vocal_veto) & (landing <= vocal_veto)
                if not valid.any():
                    return Jumps(
                        np.array([], int), np.array([], int), np.array([], float)
                    )

    scores = np.where(valid, score, np.inf)
    finite = scores[np.isfinite(scores)]

    cutoff = np.percentile(finite, keep_percentile)
    keep = np.argwhere(scores <= cutoff)

    kept_costs = scores[keep[:, 0], keep[:, 1]]
    if len(keep) > max_candidates:
        best = np.argsort(kept_costs)[:max_candidates]
        keep, kept_costs = keep[best], kept_costs[best]

    return Jumps(src=keep[:, 0], dst=keep[:, 1], cost=kept_costs)

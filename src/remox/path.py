"""Constrained shortest-path search for the edit plan.

Retargeting is posed as: walk the beat grid from the first beat to the last,
optionally taking splices, such that the number of beats emitted lands on the
target and the total splice cost is minimal.

The search state is ``(current_beat, beats_emitted)``. Counting beats rather
than seconds keeps the state space integral and small; the exact second-level
duration is computed afterwards from the real beat lengths.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from remox.beats import BeatGrid
from remox.jumps import Jumps


class NoPlanError(RuntimeError):
    """Raised when no edit path can hit the requested duration."""


@dataclass
class Segment:
    """A run of consecutive beats played verbatim, ``[start, end]`` inclusive."""

    start: int
    end: int

    @property
    def n_beats(self) -> int:
        return self.end - self.start + 1


@dataclass
class Plan:
    """The chosen edit.

    Attributes:
        segments: source ranges to play back to back.
        splice_costs: cost of each junction between consecutive segments.
        beats_emitted: total beats in the output.
        duration: length in seconds of the spliced region (intro/outro excluded).
    """

    segments: list[Segment] = field(default_factory=list)
    splice_costs: list[float] = field(default_factory=list)
    beats_emitted: int = 0
    duration: float = 0.0

    @property
    def n_edits(self) -> int:
        return len(self.splice_costs)

    @property
    def repeats(self) -> int:
        """Count of segments that duplicate the one immediately before them.

        A cheap backward splice forms a cycle the solver can ride repeatedly,
        since ``beats_emitted`` still increases each time. The result is an
        audible stutter - the same bar played several times back to back -
        which no splice-cost term catches, because each individual junction is
        genuinely low-cost. Reachable with a very low ``edit_penalty`` combined
        with ``align="beat"``.
        """
        return sum(
            1
            for i in range(1, len(self.segments))
            if (self.segments[i].start, self.segments[i].end)
            == (self.segments[i - 1].start, self.segments[i - 1].end)
        )


def _reconstruct(parent: dict, terminal: tuple[int, int]) -> tuple[list[int], list[float]]:
    """Walk parent pointers back to the start, returning played beats and splice costs.

    Only jump edges contribute a cost; the free "advance to the next beat"
    edges are not edits and must not be counted as such.
    """
    beats: list[int] = []
    costs: list[float] = []

    state = terminal
    while state is not None:
        i, _ = state
        beats.append(i)
        prev, splice_cost, is_jump = parent.get(state, (None, None, False))
        if prev is not None and is_jump:
            costs.append(splice_cost)
        state = prev

    beats.reverse()
    costs.reverse()
    return beats, costs


def _segments(beats: list[int]) -> list[Segment]:
    """Collapse a beat sequence into contiguous runs."""
    segments: list[Segment] = []
    start = prev = beats[0]
    for b in beats[1:]:
        if b == prev + 1:
            prev = b
            continue
        segments.append(Segment(start, prev))
        start = prev = b
    segments.append(Segment(start, prev))
    return segments


def solve(
    grid: BeatGrid,
    jumps: Jumps,
    target_beats: int,
    tolerance_beats: int,
    edit_penalty: float = 1.0,
    max_states: int = 2_000_000,
    entry_costs: np.ndarray | None = None,
) -> Plan:
    """Find the cheapest edit path emitting ``target_beats`` beats.

    Args:
        grid: beat grid of the source.
        jumps: candidate splices.
        target_beats: desired number of beats in the output.
        tolerance_beats: acceptable deviation from ``target_beats``.
        edit_penalty: cost added per splice, in units of "median splice cost".
            Raise it for fewer, larger edits; lower it for more, smaller ones.
        max_states: safety cap on search size.
        entry_costs: per-beat cost of opening the edit at that beat, from
            :func:`remox.entry.costs`. ``inf`` forbids a beat. When omitted the
            edit starts at beat 0, as it always did.

    Raises:
        NoPlanError: if the target is unreachable with the available splices.
    """
    n = grid.n_beats
    if n == 0:
        raise NoPlanError("beat grid is empty")

    lo = max(1, target_beats - tolerance_beats)
    hi = target_beats + tolerance_beats

    adjacency = jumps.by_source(n)

    # Seeding the queue with several starts turns this into a multi-source
    # shortest path: whichever opening leads to the cheapest complete edit wins,
    # rather than beat 0 being assumed.
    if entry_costs is None:
        seeds = [(0, 0.0)]
    else:
        if len(entry_costs) != n:
            raise ValueError(
                f"entry_costs has length {len(entry_costs)}, expected {n}"
            )
        seeds = [
            (i, float(c)) for i, c in enumerate(entry_costs) if np.isfinite(c)
        ]
        if not seeds:
            raise NoPlanError("no permitted entry point")

    best: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], tuple] = {}
    heap: list[tuple[float, int, int]] = []
    for beat, cost in seeds:
        state = (beat, 0)
        if cost < best.get(state, float("inf")):
            best[state] = cost
            parent[state] = (None, None, False)
            heapq.heappush(heap, (cost, beat, 0))

    popped = 0

    while heap:
        cost, i, emitted = heapq.heappop(heap)
        if cost > best.get((i, emitted), float("inf")):
            continue

        popped += 1
        if popped > max_states:
            raise NoPlanError(
                "search space exhausted; try a looser --tolerance or a coarser --align"
            )

        # Playing beat `i` emits one beat.
        next_emitted = emitted + 1
        if next_emitted > hi:
            # Overshot; no edge can shorten what has already been emitted.
            continue

        if i == n - 1 and lo <= next_emitted <= hi:
            beats, costs = _reconstruct(parent, (i, emitted))
            segments = _segments(beats)
            durations = grid.durations()
            total = float(sum(durations[b] for b in beats))
            return Plan(
                segments=segments,
                splice_costs=costs,
                beats_emitted=len(beats),
                duration=total,
            )

        def relax(dest: int, added: float, splice_cost: float | None) -> None:
            """Relax an edge. ``splice_cost`` is None for free advance edges.

            The recorded cost excludes ``edit_penalty`` so reports show the
            actual acoustic quality of the cut, not the tuning knob.
            """
            state = (dest, next_emitted)
            new_cost = cost + added
            if new_cost < best.get(state, float("inf")):
                best[state] = new_cost
                parent[state] = ((i, emitted), splice_cost, splice_cost is not None)
                heapq.heappush(heap, (new_cost, dest, next_emitted))

        if i + 1 < n:
            relax(i + 1, 0.0, None)

        for dest, splice_cost in adjacency[i]:
            relax(dest, splice_cost + edit_penalty, splice_cost)

    raise NoPlanError(
        "no edit path reaches the target duration; try a looser --tolerance, "
        "--align beat, or a target closer to the source length"
    )


def target_beats_for(grid: BeatGrid, target_seconds: float) -> int:
    """Convert a target duration into a beat count using the mean beat length."""
    mean_beat = float(np.mean(grid.durations()))
    if mean_beat <= 0:
        raise NoPlanError("degenerate beat grid")
    return max(1, int(round(target_seconds / mean_beat)))

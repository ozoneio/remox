"""Rendering: turn an edit plan into audio.

Splices are hidden with a short equal-power crossfade placed *before* the
junction. The incoming segment therefore still begins exactly on its own beat
boundary, so its transient stays sharp; only the tail of the outgoing segment
is faded against the incoming segment's pre-roll.
"""

from __future__ import annotations

import numpy as np

from remox.audio import Audio
from remox.beats import BeatGrid
from remox.path import Plan

#: Default crossfade length. Long enough to mask a discontinuity, short enough
#: that it stays inside a single beat at fast tempos.
DEFAULT_FADE_MS = 40.0


def _equal_power(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Constant-power fade-out / fade-in pair of ``length`` samples."""
    t = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)


def render(
    audio: Audio,
    grid: BeatGrid,
    plan: Plan,
    fade_ms: float = DEFAULT_FADE_MS,
    keep_intro: bool = True,
    keep_outro: bool = True,
) -> np.ndarray:
    """Assemble the remixed audio.

    Args:
        audio: the source, at its original rate and channel count.
        grid: beat grid (times are in seconds, so rate conversion is implicit).
        plan: edit plan from :func:`remox.path.solve`.
        fade_ms: crossfade length in milliseconds.
        keep_intro: include audio before the first tracked beat.
        keep_outro: include audio after the last tracked beat.

    Returns:
        Array of shape ``(channels, n_samples)``.
    """
    sr = audio.sr
    total = audio.n_samples

    def at(time: float) -> int:
        return int(np.clip(round(time * sr), 0, total))

    fade_len = max(1, int(round(fade_ms / 1000.0 * sr)))
    pieces: list[np.ndarray] = []

    # The intro is the audio *before* beat 0. It only belongs in the output
    # when the edit actually opens there; a plan that starts at the chorus must
    # not have the track's opening bars glued to the front of it.
    opens_at_start = bool(plan.segments) and plan.segments[0].start == 0
    if keep_intro and opens_at_start and grid.start > 0:
        head = audio.samples[:, : at(grid.start)]
        if head.shape[1]:
            pieces.append(head)

    for index, segment in enumerate(plan.segments):
        begin = at(grid.times[segment.start])
        finish = at(grid.times[segment.end + 1])
        block = audio.samples[:, begin:finish]
        if block.shape[1] == 0:
            continue

        if index == 0 or not pieces:
            pieces.append(block.copy())
            continue

        previous = pieces[-1]
        # The fade cannot be longer than either the outgoing tail, the incoming
        # segment, or the pre-roll available before the incoming segment.
        n = min(fade_len, previous.shape[1], block.shape[1], begin)
        if n <= 1:
            pieces.append(block.copy())
            continue

        fade_out, fade_in = _equal_power(n)
        pre_roll = audio.samples[:, begin - n : begin]

        previous[:, -n:] = previous[:, -n:] * fade_out + pre_roll * fade_in
        pieces.append(block.copy())

    if keep_outro and grid.end < audio.duration:
        tail = audio.samples[:, at(grid.end) :]
        if tail.shape[1]:
            pieces.append(tail)

    if not pieces:
        raise ValueError("edit plan produced no audio")

    return np.concatenate(pieces, axis=1)

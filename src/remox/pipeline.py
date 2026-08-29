"""End-to-end retargeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from remox import audio as audio_io
from remox import beats, entry, features, jumps, path, render, vocals
from remox.audio import ANALYSIS_SR

#: Progressively looser alignment modes used when the strict one has no solution.
_RELAX_ORDER = ["phrase", "bar", "beat"]


@dataclass
class RemixOptions:
    """Tunable parameters for :func:`remix`."""

    #: Beats per bar.
    meter: int = 4
    #: Metrical constraint on splices: "phrase", "bar" or "beat".
    align: str = "bar"
    #: Minimum span of a splice, in bars.
    min_jump_bars: int = 2
    #: Cost added per edit. Higher => fewer, larger edits.
    edit_penalty: float = 1.0
    #: Acceptable deviation from the target duration, in seconds.
    tolerance: float = 2.0
    #: Crossfade length at each splice.
    fade_ms: float = render.DEFAULT_FADE_MS
    #: Lead-in length in beats used to score splices.
    context_beats: int = jumps.CONTEXT_BEATS
    #: Feature block weighting.
    weights: features.FeatureWeights = field(default_factory=features.FeatureWeights)
    #: Keep audio before the first / after the last tracked beat.
    keep_intro: bool = True
    keep_outro: bool = True
    #: Retry with looser alignment and tolerance if the target is unreachable.
    auto_relax: bool = True

    #: Where the edit opens. ``"beginning"`` starts at beat one as before,
    #: ``"auto"`` lets the solver choose a strong entry point (a chorus or
    #: drop), or pass a time in seconds to force a specific opening.
    start: str | float = "beginning"
    #: How strongly entry quality counts against splice cost, in median-splice
    #: units. Raise it to insist on a great opening even at the cost of a
    #: rougher cut elsewhere.
    entry_weight: float = 4.0
    #: Metrical restriction on where the edit may open.
    entry_align: str = "phrase"
    #: Weighting of the opening-point cues.
    entry_weights: entry.EntryWeights = field(default_factory=entry.EntryWeights)

    #: Stem isolation for the rendered output: ``"none"`` keeps the full mix,
    #: ``"instrumental"`` removes the vocals, ``"vocals"`` keeps only them.
    #: Analysis always runs on the full mix, which tracks beats more reliably
    #: than a separated stem.
    isolate: str = "none"

    #: Vocal-aware splicing: "auto" uses Demucs when installed and silently
    #: skips it otherwise, "on" requires it, "off" disables it.
    vocals: str = "auto"
    #: Cost per unit of vocal phrase crossing, in median-splice units.
    vocal_weight: float = 3.0
    #: Hard rejection threshold for cuts through a vocal phrase. None = penalty only.
    vocal_veto: float | None = None
    #: Demucs model used for separation.
    vocal_model: str = "htdemucs"
    #: Reuse cached separation envelopes.
    vocal_cache: bool = True


@dataclass
class Remix:
    """Result of a retargeting run."""

    samples: np.ndarray
    sr: int
    plan: path.Plan
    grid: beats.BeatGrid
    source_duration: float
    target_duration: float
    align_used: str
    requested_align: str
    vocal_activity: vocals.VocalActivity | None = None
    #: The analysis this remix was built from, so callers can reach the
    #: isolated stem or beat grid without recomputing anything.
    analysis: "Analysis | None" = None

    @property
    def duration(self) -> float:
        return self.samples.shape[1] / self.sr

    def save(self, out_path: str | Path) -> None:
        audio_io.save(out_path, self.samples, self.sr)


@dataclass
class Analysis:
    """Everything about a track that does not depend on the target duration.

    Beat tracking, feature extraction and vocal separation are all per-track,
    and separation in particular costs tens of seconds. Computing them once and
    reusing them is what makes generating many variants nearly free.
    """

    track: audio_io.Audio
    signal: np.ndarray
    grid: beats.BeatGrid
    blocks: features.FeatureBlocks
    activity: vocals.VocalActivity | None = None
    #: Audio the plan is rendered from. Same as ``track`` unless a stem was
    #: isolated, in which case timing still comes from the full mix while the
    #: samples come from the isolated version.
    render_source: audio_io.Audio | None = None

    @property
    def output_audio(self) -> audio_io.Audio:
        """Whichever audio the output should be cut from."""
        return self.render_source or self.track

    @property
    def crossing(self) -> np.ndarray | None:
        """Vocal phrase-crossing scores, or None when there are no vocals."""
        if self.activity is None or not self.activity.present:
            return None
        return self.activity.crossing

    def opens_at_beginning(self, options: RemixOptions) -> bool:
        """Whether the edit is guaranteed to start at beat one.

        Only the literal ``"beginning"`` setting guarantees it. ``"auto"`` may
        well choose beat 0, but the duration arithmetic has to be settled
        before the solver runs, so ``"auto"`` always excludes the intro. That
        keeps the output length deterministic rather than depending on which
        opening the solver happens to pick.
        """
        return options.start == "beginning"

    def fixed_duration(self, options: RemixOptions) -> float:
        """Seconds of intro and outro that pass through unedited."""
        fixed = 0.0
        if options.keep_intro and self.opens_at_beginning(options):
            fixed += self.grid.start
        if options.keep_outro:
            fixed += self.track.duration - self.grid.end
        return fixed

    def entry_costs(self, options: RemixOptions) -> np.ndarray | None:
        """Per-beat opening costs for this configuration, or None for beat 0."""
        if options.start == "beginning":
            return None

        if options.start == "auto":
            raw = entry.costs(
                self.blocks,
                self.grid,
                crossing=self.crossing,
                weights=options.entry_weights,
                align=options.entry_align,
            )
        else:
            try:
                when = float(options.start)
            except (TypeError, ValueError):
                raise ValueError(
                    f"unknown start mode {options.start!r}; "
                    "use 'beginning', 'auto', or a time in seconds"
                ) from None
            raw = entry.resolve_fixed(self.grid, when, align=options.entry_align)

        return raw * options.entry_weight

    def targeting(
        self, target_duration: float, options: RemixOptions
    ) -> tuple[int, int]:
        """Convert a target duration into ``(target_beats, tolerance_beats)``.

        Raises:
            ValueError: if the target is shorter than the untouched intro+outro.
        """
        fixed = self.fixed_duration(options)
        beat_region = target_duration - fixed
        if beat_region <= 0:
            raise ValueError(
                f"target of {target_duration:.1f}s is shorter than the untouched "
                f"intro+outro ({fixed:.1f}s); use --no-intro/--no-outro or a longer target"
            )
        mean_beat = float(np.mean(self.grid.durations()))
        return (
            path.target_beats_for(self.grid, beat_region),
            max(1, int(round(options.tolerance / mean_beat))),
        )


def analyse(source: str | Path, options: RemixOptions | None = None) -> Analysis:
    """Run all per-track analysis for ``source``.

    Args:
        source: path to the input audio file.
        options: only the analysis-affecting fields are read (``meter``, the
            ``vocal_*`` settings and ``isolate``).

    Raises:
        remox.beats.BeatTrackingError: if no usable beat grid is found.
        remox.vocals.SeparationUnavailable: if isolation is requested without
            Demucs installed.
    """
    options = options or RemixOptions()
    track = audio_io.load(source)
    signal = track.analysis_signal()
    grid = beats.track(signal, sr=ANALYSIS_SR, meter=options.meter)
    block_set = features.blocks(signal, grid, sr=ANALYSIS_SR)

    render_source = None
    if options.isolate != "none":
        # Separation is the expensive step, so take the vocal envelope from the
        # same pass rather than running the model a second time.
        render_source, vocal_signal = vocals.isolate_cached(
            track,
            keep=options.isolate,
            model=options.vocal_model,
            device=None,
            use_cache=options.vocal_cache,
        )
        activity = (
            None
            if options.vocals == "off"
            else vocals.from_stem(vocal_signal, signal, grid)
        )
    else:
        activity = _vocal_activity(track, grid, options)

    return Analysis(
        track=track,
        signal=signal,
        grid=grid,
        blocks=block_set,
        activity=activity,
        render_source=render_source,
    )


def plan_for(
    analysis: Analysis,
    target_duration: float,
    options: RemixOptions | None = None,
    forbidden: set[tuple[int, int]] | None = None,
) -> tuple[path.Plan, str]:
    """Solve for an edit plan against a prepared analysis.

    Args:
        analysis: output of :func:`analyse`.
        target_duration: desired output length in seconds.
        options: tuning parameters for this plan.
        forbidden: splices to exclude, used to force diversity between variants.

    Returns:
        ``(plan, align_used)``.
    """
    options = options or RemixOptions()
    target_beats, tolerance_beats = analysis.targeting(target_duration, options)
    return _solve_with_relaxation(
        analysis, target_beats, tolerance_beats, options, forbidden
    )


def render_plan(
    analysis: Analysis, plan: path.Plan, options: RemixOptions | None = None
) -> np.ndarray:
    """Render an edit plan to audio."""
    options = options or RemixOptions()
    return render.render(
        analysis.output_audio,
        analysis.grid,
        plan,
        fade_ms=options.fade_ms,
        keep_intro=options.keep_intro,
        keep_outro=options.keep_outro,
    )


def remix(
    source: str | Path,
    target_duration: float,
    options: RemixOptions | None = None,
) -> Remix:
    """Re-edit ``source`` to approximately ``target_duration`` seconds.

    Args:
        source: path to the input audio file.
        target_duration: desired output length in seconds.
        options: tuning parameters.

    Raises:
        ValueError: if the target duration is not positive.
        remox.beats.BeatTrackingError: if no usable beat grid is found.
        remox.path.NoPlanError: if no edit path reaches the target.
    """
    options = options or RemixOptions()
    if target_duration <= 0:
        raise ValueError("target duration must be positive")

    analysis = analyse(source, options)
    plan, align_used = plan_for(analysis, target_duration, options)
    samples = render_plan(analysis, plan, options)

    return Remix(
        samples=samples,
        sr=analysis.track.sr,
        plan=plan,
        grid=analysis.grid,
        source_duration=analysis.track.duration,
        target_duration=target_duration,
        align_used=align_used,
        requested_align=options.align,
        vocal_activity=analysis.activity,
        analysis=analysis,
    )


def _vocal_activity(
    track: audio_io.Audio, grid: beats.BeatGrid, options: RemixOptions
) -> vocals.VocalActivity | None:
    """Run vocal separation according to the ``vocals`` policy.

    ``"auto"`` is the default because separation is an optional extra: a user
    without Demucs installed should still get a working remix, just without
    phrase protection.
    """
    if options.vocals == "off":
        return None
    if options.vocals not in ("auto", "on"):
        raise ValueError(f"unknown vocals mode: {options.vocals!r}")

    try:
        return vocals.analyse(
            track,
            grid,
            model=options.vocal_model,
            use_cache=options.vocal_cache,
        )
    except vocals.SeparationUnavailable:
        if options.vocals == "on":
            raise
        return None


def _solve_with_relaxation(
    analysis: Analysis,
    target_beats: int,
    tolerance_beats: int,
    options: RemixOptions,
    forbidden: set[tuple[int, int]] | None = None,
) -> tuple[path.Plan, str]:
    """Try the requested alignment, then progressively looser ones.

    Strict alignment gives the most musical cuts but can leave no path to the
    target - especially on short tracks or extreme duration changes. Rather
    than fail outright, fall back to looser constraints and report which one
    was actually used.
    """
    if options.auto_relax:
        start = _RELAX_ORDER.index(options.align)
        attempts = _RELAX_ORDER[start:]
    else:
        attempts = [options.align]

    feature_matrix = analysis.blocks.combine(options.weights)
    entry_costs = analysis.entry_costs(options)

    last_error: Exception | None = None
    for align in attempts:
        candidates = jumps.find(
            feature_matrix,
            analysis.grid,
            context=options.context_beats,
            min_jump_bars=options.min_jump_bars,
            align=align,
            vocal_crossing=analysis.crossing,
            vocal_weight=options.vocal_weight,
            vocal_veto=options.vocal_veto,
        )
        if forbidden:
            candidates = candidates.exclude(forbidden)
        try:
            plan = path.solve(
                analysis.grid,
                candidates,
                target_beats=target_beats,
                tolerance_beats=tolerance_beats,
                edit_penalty=options.edit_penalty,
                entry_costs=entry_costs,
            )
        except path.NoPlanError as exc:
            last_error = exc
            continue
        return plan, align

    raise last_error or path.NoPlanError("no edit path found")

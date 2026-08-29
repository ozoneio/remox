"""Generate several distinct edits of the same track at the same duration.

Two mechanisms, used in that order:

1. **Curated presets** - configurations that emphasise different things, so
   each version has a stated reason for sounding different.
2. **Forced diversity** - if presets converge on the same plan (they often do,
   because one splice is frequently the obvious optimum), the splices already
   used are forbidden and the solver is re-run. That guarantees the requested
   number of genuinely distinct edits.

Every variant is solved against the same target duration and the same
tolerance. Presets deliberately never override ``tolerance``: a version that
drifted outside the length you asked for would not be a usable alternative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from remox import path as path_mod
from remox.features import FeatureWeights
from remox.pipeline import Analysis, RemixOptions, analyse, plan_for, render_plan
from remox import audio as audio_io


@dataclass(frozen=True)
class Preset:
    """A named configuration and the reason it exists."""

    name: str
    description: str
    overrides: dict = field(default_factory=dict)

    def apply(self, base: RemixOptions) -> RemixOptions:
        return replace(base, **self.overrides)


#: Ordered by how likely each is to produce a genuinely different edit, so that
#: asking for a small number of variants still gives contrast.
PRESETS: tuple[Preset, ...] = (
    Preset("balanced", "Default settings: bar-aligned, moderate edit count."),
    Preset(
        "minimal",
        "Fewest, largest edits - one big cut rather than several small ones.",
        {"edit_penalty": 8.0, "min_jump_bars": 4},
    ),
    Preset(
        "granular",
        "Many small edits, spreading the change across the track.",
        {"edit_penalty": 0.2, "min_jump_bars": 1},
    ),
    Preset(
        "harmonic",
        "Prioritises chord continuity over timbre and level.",
        {"weights": FeatureWeights(chroma=1.6, timbre=0.3, energy=0.3)},
    ),
    Preset(
        "textural",
        "Prioritises arrangement and instrumentation matching.",
        {"weights": FeatureWeights(chroma=0.5, timbre=1.5, energy=0.4)},
    ),
    Preset(
        "phrase",
        "Cuts only on 4-bar phrase boundaries where the target allows.",
        {"align": "phrase"},
    ),
    Preset(
        "dynamic",
        "Matches energy level across the cut, avoiding loud/quiet jumps.",
        {"weights": FeatureWeights(chroma=0.7, timbre=0.5, energy=1.5)},
    ),
    Preset(
        "vocalsafe",
        "Hard-rejects any cut through a sung phrase (no effect if instrumental).",
        {"vocal_veto": 0.25},
    ),
)


@dataclass
class Variant:
    """One planned edit, not yet rendered.

    Audio is rendered on demand rather than held for every variant at once: a
    five-minute stereo track is ~50 MB per rendered version.
    """

    index: int
    name: str
    description: str
    options: RemixOptions
    plan: path_mod.Plan
    align_used: str
    source: str

    @property
    def signature(self) -> tuple:
        """Identity of the resulting edit, used to detect duplicates."""
        return tuple((s.start, s.end) for s in self.plan.segments)

    @property
    def worst_cut(self) -> float:
        return max(self.plan.splice_costs) if self.plan.splice_costs else 0.0

    @property
    def mean_cut(self) -> float:
        costs = self.plan.splice_costs
        return sum(costs) / len(costs) if costs else 0.0

    def summary(self, analysis: Analysis) -> dict:
        """JSON-friendly record of this variant."""
        grid = analysis.grid
        return {
            "index": self.index,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "align_used": self.align_used,
            "edits": self.plan.n_edits,
            "duration": round(self.plan.duration + analysis.fixed_duration(self.options), 3),
            "mean_cut_cost": round(self.mean_cut, 4),
            "worst_cut_cost": round(self.worst_cut, 4),
            "segments": [
                {
                    "start_beat": s.start,
                    "end_beat": s.end,
                    "start_time": round(float(grid.times[s.start]), 3),
                    "end_time": round(float(grid.times[s.end + 1]), 3),
                    "beats": s.n_beats,
                }
                for s in self.plan.segments
            ],
        }


@dataclass
class VariantSet:
    """A set of distinct edits sharing one analysis."""

    analysis: Analysis
    variants: list[Variant]
    target_duration: float
    requested: int

    def __len__(self) -> int:
        return len(self.variants)

    @property
    def shortfall(self) -> int:
        """How many fewer variants were found than requested."""
        return max(0, self.requested - len(self.variants))

    def render(self, variant: Variant):
        """Render one variant to audio."""
        return render_plan(self.analysis, variant.plan, variant.options)

    def save_all(self, directory: str | Path, stem: str = "v") -> list[Path]:
        """Render and write every variant, plus a manifest.

        Variants are rendered one at a time so peak memory stays at a single
        version rather than all of them.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        records = []
        for variant in self.variants:
            out = directory / f"{stem}{variant.index}_{variant.name}.wav"
            samples = self.render(variant)
            audio_io.save(out, samples, self.analysis.track.sr)
            written.append(out)

            record = variant.summary(self.analysis)
            record["file"] = out.name
            record["rendered_duration"] = round(samples.shape[1] / self.analysis.track.sr, 3)
            records.append(record)

        manifest = {
            "source": str(self.analysis.track.path),
            "source_duration": round(self.analysis.track.duration, 3),
            "target_duration": self.target_duration,
            "tempo": round(self.analysis.grid.tempo, 2),
            "meter": self.analysis.grid.meter,
            "beats": self.analysis.grid.n_beats,
            "requested_variants": self.requested,
            "produced_variants": len(self.variants),
            "vocals_detected": bool(
                self.analysis.activity is not None and self.analysis.activity.present
            ),
            "variants": records,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return written


def generate(
    source: str | Path,
    target_duration: float,
    count: int = 3,
    options: RemixOptions | None = None,
    analysis: Analysis | None = None,
) -> VariantSet:
    """Produce up to ``count`` distinct edits at ``target_duration``.

    Args:
        source: input audio file.
        target_duration: desired length, in seconds, for every variant.
        count: how many distinct versions to aim for.
        options: base configuration; presets override individual fields.
        analysis: reuse a prepared analysis instead of recomputing it.

    Raises:
        ValueError: if ``count`` is not positive or the target is invalid.
        remox.path.NoPlanError: if not even one plan reaches the target.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if target_duration <= 0:
        raise ValueError("target duration must be positive")

    options = options or RemixOptions()
    analysis = analysis or analyse(source, options)

    variants: list[Variant] = []
    seen: set[tuple] = set()
    first_error: Exception | None = None

    for preset in PRESETS:
        if len(variants) >= count:
            break
        candidate = preset.apply(options)
        try:
            plan, align_used = plan_for(analysis, target_duration, candidate)
        except path_mod.NoPlanError as exc:
            first_error = first_error or exc
            continue

        signature = tuple((s.start, s.end) for s in plan.segments)
        if signature in seen:
            continue
        if plan.repeats:
            # A stutter loop is never a usable alternative version, however
            # cheap its individual splices score.
            continue
        seen.add(signature)
        variants.append(
            Variant(
                index=len(variants) + 1,
                name=preset.name,
                description=preset.description,
                options=candidate,
                plan=plan,
                align_used=align_used,
                source="preset",
            )
        )

    if not variants:
        raise first_error or path_mod.NoPlanError(
            "no edit path reaches the target duration"
        )

    _fill_by_exclusion(analysis, target_duration, options, count, variants, seen)
    return VariantSet(
        analysis=analysis,
        variants=variants,
        target_duration=target_duration,
        requested=count,
    )


def _fill_by_exclusion(
    analysis: Analysis,
    target_duration: float,
    options: RemixOptions,
    count: int,
    variants: list[Variant],
    seen: set[tuple],
) -> None:
    """Top up the list by forbidding splices that have already been used.

    Presets often converge - when one cut is clearly the best, several
    configurations find it. Excluding the splices already chosen forces the
    solver to return the next-best genuinely different edit. These are ranked
    below the preset variants by construction, and are labelled as such.
    """
    forbidden: set[tuple[int, int]] = set()
    for variant in variants:
        forbidden |= _splices_of(variant.plan)

    while len(variants) < count:
        try:
            plan, align_used = plan_for(
                analysis, target_duration, options, forbidden=forbidden
            )
        except path_mod.NoPlanError:
            break  # Nothing distinct left to find; report a shortfall instead.

        signature = tuple((s.start, s.end) for s in plan.segments)
        if signature in seen:
            break  # Excluding made no difference; further attempts will loop.
        if plan.repeats:
            break  # Only stutter loops remain reachable; stop rather than emit one.

        used = _splices_of(plan)
        if not used:
            break  # An uncut plan cannot be excluded any further.

        seen.add(signature)
        forbidden |= used
        variants.append(
            Variant(
                index=len(variants) + 1,
                name=f"alt{len(variants) + 1}",
                description="Next-best edit avoiding all splices used above.",
                options=options,
                plan=plan,
                align_used=align_used,
                source="diversity",
            )
        )


def _splices_of(plan: path_mod.Plan) -> set[tuple[int, int]]:
    """The ``(src, dst)`` splices a plan uses at its segment junctions."""
    return {
        (plan.segments[i].end, plan.segments[i + 1].start)
        for i in range(len(plan.segments) - 1)
    }

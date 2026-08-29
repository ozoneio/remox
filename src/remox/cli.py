"""Command-line interface."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from remox import __version__
from remox.audio import save as audio_save
from remox.beats import BeatTrackingError
from remox.features import FeatureWeights
from remox.fetch import AUDIO_FORMATS, FetchError, default_output_name, fetch, is_url
from remox.path import NoPlanError
from remox.pipeline import RemixOptions, analyse, remix
from remox.variants import PRESETS, generate
from remox.vocals import SeparationUnavailable

_CLOCK = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)$")
_UNITS = re.compile(r"^(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$", re.IGNORECASE)
_OUTPUT_DIR = Path("output")


def parse_duration(text: str) -> float:
    """Parse ``90``, ``90s``, ``1m30s``, ``1:30`` or ``1:02:30`` into seconds."""
    text = text.strip()
    if not text:
        raise argparse.ArgumentTypeError("duration is empty")

    clock = _CLOCK.match(text)
    if clock:
        hours, minutes, seconds = clock.groups()
        return int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)

    try:
        return float(text)
    except ValueError:
        pass

    units = _UNITS.match(text)
    if units and any(units.groups()):
        minutes, seconds = units.groups()
        return float(minutes or 0) * 60 + float(seconds or 0)

    raise argparse.ArgumentTypeError(
        f"cannot parse duration {text!r}; try '90', '90s', '1m30s' or '1:30'"
    )


def _parse_start(text: str) -> str | float:
    """Parse ``--start``: a keyword, or a time offset into the source."""
    lowered = text.strip().lower()
    if lowered in ("beginning", "start", "0"):
        return "beginning"
    if lowered == "auto":
        return "auto"
    try:
        return parse_duration(text)
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(
            f"cannot parse --start {text!r}; use 'beginning', 'auto', or a time like 1:12"
        ) from None


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:05.2f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remox",
        description="Re-edit a music track to a target duration, on the beat.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="source audio file, or a media URL to download",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=parse_duration,
        help="target duration: 90, 90s, 1m30s or 1:30 (omit only with --fetch-only)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (default: output/<input>.remox.wav)",
    )
    parser.add_argument(
        "-n",
        "--variants",
        type=int,
        default=1,
        metavar="N",
        help="produce N distinct edits at the same duration, written to a "
        "subfolder with a manifest (default: 1)",
    )
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=None,
        help="folder for multiple variants (default: output/<input>_variants)",
    )
    parser.add_argument(
        "--list-presets", action="store_true", help="print the variant presets and exit"
    )
    parser.add_argument("--version", action="version", version=f"remox {__version__}")

    musical = parser.add_argument_group("musical structure")
    musical.add_argument("--meter", type=int, default=4, help="beats per bar (default: 4)")
    musical.add_argument(
        "--align",
        choices=["phrase", "bar", "beat"],
        default="bar",
        help="where cuts may land; looser modes give more options but rougher edits "
        "(default: bar)",
    )
    musical.add_argument(
        "--min-jump-bars",
        type=int,
        default=2,
        help="minimum span of a single edit, in bars (default: 2)",
    )

    musical.add_argument(
        "--start",
        default="beginning",
        metavar="WHERE",
        help="where the edit opens: 'beginning' (default), 'auto' to pick a "
        "strong entry point such as a chorus, or a time like 1:12",
    )
    musical.add_argument(
        "--entry-align",
        choices=["phrase", "bar", "beat"],
        default="phrase",
        help="metrical restriction on the opening point (default: phrase)",
    )
    musical.add_argument(
        "--entry-weight",
        type=float,
        default=4.0,
        help="how strongly opening quality counts against cut quality (default: 4.0)",
    )

    shaping = parser.add_argument_group("edit shaping")
    shaping.add_argument(
        "--edits",
        type=float,
        default=1.0,
        dest="edit_penalty",
        help="cost per edit; higher means fewer, larger edits (default: 1.0)",
    )
    shaping.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="acceptable deviation from the target, in seconds (default: 2.0)",
    )
    shaping.add_argument(
        "--context",
        type=int,
        default=4,
        help="beats of lead-in compared when scoring a cut (default: 4)",
    )
    shaping.add_argument(
        "--no-relax",
        action="store_true",
        help="fail instead of loosening --align when the target is unreachable",
    )

    weights = parser.add_argument_group("feature weights")
    weights.add_argument("--w-chroma", type=float, default=1.0, help="harmony weight")
    weights.add_argument("--w-timbre", type=float, default=0.6, help="timbre weight")
    weights.add_argument("--w-energy", type=float, default=0.5, help="loudness weight")

    vocal = parser.add_argument_group("vocal-aware splicing")
    vocal.add_argument(
        "--vocals",
        choices=["auto", "on", "off"],
        default="auto",
        help="separate vocals and avoid cutting mid-phrase; 'auto' skips it if "
        "Demucs is not installed (default: auto)",
    )
    vocal.add_argument(
        "--vocal-weight",
        type=float,
        default=3.0,
        help="cost per unit of vocal phrase crossing (default: 3.0)",
    )
    vocal.add_argument(
        "--vocal-veto",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="reject any cut whose vocal crossing exceeds THRESHOLD (0-1); "
        "stricter than the default penalty and can make targets unreachable",
    )
    vocal.add_argument(
        "--isolate",
        choices=["none", "instrumental", "vocals"],
        default="none",
        help="separate stems for the output: 'instrumental' removes the vocals, "
        "'vocals' keeps only them. Omit -d to just export the isolated track "
        "without remixing (default: none)",
    )
    vocal.add_argument(
        "--keep-isolated",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="also save the full-length isolated track when remixing. Give a "
        "path, or omit it for <output>.instrumental.mp3",
    )
    vocal.add_argument(
        "--vocal-model", default="htdemucs", help="Demucs model (default: htdemucs)"
    )
    vocal.add_argument(
        "--no-vocal-cache", action="store_true", help="re-run separation instead of caching"
    )

    download = parser.add_argument_group("downloading")
    download.add_argument(
        "--audio-format",
        choices=list(AUDIO_FORMATS),
        default="mp3",
        help="format to convert downloaded audio to (default: mp3)",
    )
    download.add_argument(
        "--audio-quality", default="192", help="bitrate in kbps for lossy formats (default: 192)"
    )
    download.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="where to keep downloads (default: ~/Downloads/Music, or "
        "$REMOX_DOWNLOAD_DIR)",
    )
    download.add_argument(
        "--no-download-cache", action="store_true", help="re-download even if already cached"
    )
    download.add_argument(
        "--fetch-only",
        action="store_true",
        help="download the audio and exit without remixing",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--fade-ms", type=float, default=40.0, help="crossfade length (default: 40)"
    )
    output.add_argument("--no-intro", action="store_true", help="drop audio before beat 1")
    output.add_argument("--no-outro", action="store_true", help="drop audio after the last beat")
    output.add_argument("--quiet", action="store_true", help="suppress the report")

    return parser


def _report(result, stream) -> None:
    plan = result.plan
    delta = result.duration - result.target_duration
    print(f"source     {_format_duration(result.source_duration)}", file=stream)
    print(
        f"output     {_format_duration(result.duration)}  "
        f"(target {_format_duration(result.target_duration)}, {delta:+.2f}s)",
        file=stream,
    )
    print(
        f"tempo      {result.grid.tempo:.1f} BPM, {result.grid.meter}/4, "
        f"{result.grid.n_beats} beats",
        file=stream,
    )
    print(f"edits      {plan.n_edits} (align: {result.align_used})", file=stream)
    opening = result.grid.times[plan.segments[0].start] if plan.segments else 0.0
    if plan.segments and plan.segments[0].start != 0:
        print(
            f"opens at   {_format_duration(float(opening))} "
            f"(beat {plan.segments[0].start}) - intro dropped",
            file=stream,
        )
    if result.align_used != result.requested_align:
        print(
            f"note       --align {result.requested_align} had no path to the target; "
            f"relaxed to {result.align_used}. A larger --tolerance may keep it.",
            file=stream,
        )
    if plan.splice_costs:
        worst = max(plan.splice_costs)
        mean = sum(plan.splice_costs) / len(plan.splice_costs)
        print(f"cut cost   mean {mean:.2f}, worst {worst:.2f}", file=stream)
    if plan.repeats:
        print(
            f"warning    {plan.repeats} segment(s) repeat back to back - this will "
            "stutter. Raise --edits or --min-jump-bars.",
            file=stream,
        )

    activity = result.vocal_activity
    if activity is None:
        print("vocals     not analysed", file=stream)
    elif not activity.present:
        print("vocals     none detected (instrumental)", file=stream)
    else:
        crossings = [activity.crossing[s.start] for s in plan.segments[1:]]
        worst_cross = max(crossings) if crossings else 0.0
        print(
            f"vocals     {activity.voiced_fraction:.0%} of track voiced; "
            f"worst cut crossing {worst_cross:.2f}",
            file=stream,
        )
    for i, segment in enumerate(plan.segments):
        start = result.grid.times[segment.start]
        end = result.grid.times[segment.end + 1]
        print(
            f"  seg {i + 1:>2}  {_format_duration(start)} -> {_format_duration(end)}"
            f"  ({segment.n_beats} beats)",
            file=stream,
        )


def _report_variants(result, stream) -> None:
    """Print the comparison table for a multi-variant run."""
    analysis = result.analysis
    print(f"source     {_format_duration(analysis.track.duration)}", file=stream)
    print(
        f"tempo      {analysis.grid.tempo:.1f} BPM, {analysis.grid.meter}/4, "
        f"{analysis.grid.n_beats} beats",
        file=stream,
    )
    voiced = analysis.activity is not None and analysis.activity.present
    print(f"vocals     {'detected' if voiced else 'none detected'}", file=stream)
    print(
        f"variants   {len(result)} of {result.requested} requested "
        f"at {_format_duration(result.target_duration)}",
        file=stream,
    )
    if result.shortfall:
        print(
            f"note       only {len(result)} distinct edits exist at this duration; "
            "a looser --tolerance or --align beat would allow more.",
            file=stream,
        )
    print("", file=stream)

    header = f"  {'#':<3}{'name':<11}{'edits':>6}{'worst':>8}  {'align':<7}why"
    print(header, file=stream)
    print(f"  {'-' * (len(header) - 2)}", file=stream)
    for variant in result.variants:
        print(
            f"  {variant.index:<3}{variant.name:<11}{variant.plan.n_edits:>6}"
            f"{variant.worst_cut:>8.2f}  {variant.align_used:<7}{variant.description}",
            file=stream,
        )


def _run_variants(args, source, options, stream) -> int:
    """Plan, render and write multiple variants."""
    directory = args.variants_dir or _OUTPUT_DIR / f"{source.stem}_variants"
    try:
        result = generate(source, args.duration, count=args.variants, options=options)
    except (
        FileNotFoundError,
        ValueError,
        BeatTrackingError,
        NoPlanError,
        SeparationUnavailable,
    ) as exc:
        print(f"remox: {exc}", file=sys.stderr)
        return 1

    written = result.save_all(directory)
    _save_isolated(args, result.analysis, directory / "isolated.wav", stream)
    if not args.quiet:
        _report_variants(result, stream)
        print("", file=stream)
        for out in written:
            print(f"written    {out}", file=stream)
        print(f"manifest   {directory / 'manifest.json'}", file=stream)
    return 0


def _build_options(args) -> RemixOptions:
    """Translate parsed arguments into a :class:`RemixOptions`."""
    return RemixOptions(
        meter=args.meter,
        align=args.align,
        min_jump_bars=args.min_jump_bars,
        edit_penalty=args.edit_penalty,
        tolerance=args.tolerance,
        fade_ms=args.fade_ms,
        context_beats=args.context,
        weights=FeatureWeights(
            chroma=args.w_chroma, timbre=args.w_timbre, energy=args.w_energy
        ),
        keep_intro=not args.no_intro,
        keep_outro=not args.no_outro,
        auto_relax=not args.no_relax,
        start=_parse_start(args.start),
        entry_align=args.entry_align,
        entry_weight=args.entry_weight,
        vocals=args.vocals,
        vocal_weight=args.vocal_weight,
        vocal_veto=args.vocal_veto,
        vocal_model=args.vocal_model,
        vocal_cache=not args.no_vocal_cache,
        isolate=args.isolate,
    )


def _run_isolate(args, source, options, stream) -> int:
    """Export an isolated stem without remixing (no -d given)."""
    try:
        analysis = analyse(source, options)
    except (
        FileNotFoundError,
        ValueError,
        BeatTrackingError,
        SeparationUnavailable,
    ) as exc:
        print(f"remox: {exc}", file=sys.stderr)
        return 1

    output = args.output or _OUTPUT_DIR / source.with_suffix(f".{args.isolate}.wav").name
    isolated = analysis.output_audio
    audio_save(output, isolated.samples, isolated.sr)

    if not args.quiet:
        print(f"source     {_format_duration(analysis.track.duration)}", file=stream)
        print(f"isolated   {args.isolate} (model: {options.vocal_model})", file=stream)
        print(f"written    {output}", file=stream)
    return 0


def _save_isolated(args, analysis, output: Path, stream) -> None:
    """Write the full-length isolated track next to the remix, if asked."""
    if args.keep_isolated is None or analysis.render_source is None:
        return

    if args.keep_isolated == "auto":
        target = output.with_suffix("").with_suffix(f".{args.isolate}.mp3")
    else:
        target = Path(args.keep_isolated)

    isolated = analysis.render_source
    audio_save(target, isolated.samples, isolated.sr)
    if not args.quiet:
        print(f"isolated   {target}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        for index, preset in enumerate(PRESETS, start=1):
            print(f"{index:<3}{preset.name:<12}{preset.description}")
        return 0

    if args.input is None:
        parser.error("input is required (an audio file or a URL)")

    source = args.input
    download = None

    if is_url(source):
        try:
            download = fetch(
                source,
                dest_dir=args.download_dir,
                audio_format=args.audio_format,
                quality=args.audio_quality,
                use_cache=not args.no_download_cache,
                quiet=args.quiet,
            )
        except FetchError as exc:
            print(f"remox: {exc}", file=sys.stderr)
            return 1
        source = download.path
        if not args.quiet:
            print(f"fetched    {download.attribution()}", file=sys.stdout)
            print(f"source     {download.url}", file=sys.stdout)
            print(f"saved      {download.path}", file=sys.stdout)
    else:
        source = Path(source)

    if args.fetch_only:
        if download is None:
            print("remox: --fetch-only requires a URL as input", file=sys.stderr)
            return 1
        return 0

    if args.duration is None:
        if args.isolate != "none":
            options = _build_options(args)
            return _run_isolate(args, source, options, sys.stdout)
        parser.error(
            "-d/--duration is required unless --fetch-only or --isolate is given"
        )
    if args.variants < 1:
        parser.error("-n/--variants must be at least 1")

    options = _build_options(args)

    if args.variants > 1:
        return _run_variants(args, source, options, sys.stdout)

    if args.output is not None:
        output = args.output
    elif download is not None:
        output = _OUTPUT_DIR / default_output_name(download)
    else:
        output = _OUTPUT_DIR / source.with_suffix(".remox.wav").name

    try:
        result = remix(source, args.duration, options)
    except (
        FileNotFoundError,
        ValueError,
        BeatTrackingError,
        NoPlanError,
        SeparationUnavailable,
    ) as exc:
        print(f"remox: {exc}", file=sys.stderr)
        return 1

    result.save(output)
    _save_isolated(args, result.analysis, output, sys.stdout)
    if not args.quiet:
        _report(result, sys.stdout)
        print(f"written    {output}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

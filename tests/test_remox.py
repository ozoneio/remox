"""End-to-end and unit tests for remox.

The audio fixture is generated once per session rather than committed, so the
repo stays free of binaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import make_test_track  # noqa: E402

from remox import audio as audio_io  # noqa: E402
from remox import beats, cache, cli, entry, pipeline, variants  # noqa: E402
from remox import fetch as fetch_mod  # noqa: E402
from remox import features, jumps, path, render, vocals  # noqa: E402
from remox.cli import main, parse_duration  # noqa: E402
from remox.pipeline import RemixOptions, remix  # noqa: E402

SOURCE_DURATION = 105.0


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path_factory, monkeypatch):
    """Point the cache at a throwaway directory for every test.

    Without this, tests write stub separations into the real user cache and
    later runs read them back as if they were genuine - which silently broke
    an assertion here once already. Tests must never touch a user's cache.
    """
    monkeypatch.setenv("REMOX_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))


@pytest.fixture(scope="session")
def track(tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("fixtures") / "track.wav"
    sf.write(str(dest), make_test_track.build(), make_test_track.SR)
    return dest


@pytest.fixture(scope="session")
def analysis(track):
    loaded = audio_io.load(track)
    signal = loaded.analysis_signal()
    grid = beats.track(signal)
    matrix = features.extract(signal, grid)
    return loaded, signal, grid, matrix


# --------------------------------------------------------------------------- audio


def test_load_normalises_to_2d(track):
    loaded = audio_io.load(track)
    assert loaded.samples.ndim == 2
    assert loaded.n_channels == 1
    assert loaded.duration == pytest.approx(SOURCE_DURATION, abs=0.5)


def test_analysis_signal_is_resampled(track):
    loaded = audio_io.load(track)
    signal = loaded.analysis_signal()
    assert signal.ndim == 1
    expected = loaded.duration * audio_io.ANALYSIS_SR
    assert len(signal) == pytest.approx(expected, rel=0.01)


def test_save_limits_peak(tmp_path):
    loud = np.full((1, 1000), 3.0, dtype=np.float32)
    out = tmp_path / "loud.wav"
    audio_io.save(out, loud, 44100)
    written, _ = sf.read(str(out))
    assert np.max(np.abs(written)) <= 1.0


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        audio_io.load(tmp_path / "nope.wav")


# --------------------------------------------------------------------------- beats


def test_tempo_and_beat_count(analysis):
    _, _, grid, _ = analysis
    assert grid.tempo == pytest.approx(make_test_track.BPM, rel=0.05)
    expected_beats = SOURCE_DURATION / (60.0 / make_test_track.BPM)
    assert grid.n_beats == pytest.approx(expected_beats, rel=0.05)


def test_beat_boundaries_are_monotonic(analysis):
    _, _, grid, _ = analysis
    assert np.all(np.diff(grid.times) > 0)
    assert len(grid.times) == grid.n_beats + 1


def test_phase_and_bar_indexing_agree(analysis):
    _, _, grid, _ = analysis
    downbeats = [i for i in range(grid.n_beats) if grid.phase(i) == 0]
    # Downbeats are exactly `meter` beats apart.
    assert np.all(np.diff(downbeats) == grid.meter)


def test_too_short_input_rejected():
    silence = np.zeros(audio_io.ANALYSIS_SR, dtype=np.float32)
    with pytest.raises(beats.BeatTrackingError):
        beats.track(silence)


# --------------------------------------------------------------------------- features


def test_feature_matrix_shape(analysis):
    _, _, grid, matrix = analysis
    assert matrix.shape[0] == grid.n_beats
    assert matrix.shape[1] == 12 + 12 + 1  # chroma + mfcc[1:] + rms
    assert np.all(np.isfinite(matrix))


def test_weights_scale_blocks(analysis):
    _, signal, grid, _ = analysis
    quiet = features.extract(
        signal, grid, weights=features.FeatureWeights(chroma=1.0, timbre=0.0, energy=0.0)
    )
    # Zero-weighted blocks collapse to zero, leaving only chroma active.
    assert np.allclose(quiet[:, 12:], 0.0)
    assert not np.allclose(quiet[:, :12], 0.0)


def test_repeated_sections_are_similar(analysis):
    """The fixture repeats the chorus verbatim; features should reflect that."""
    _, _, grid, matrix = analysis
    beats_per_bar = grid.meter
    # chorus starts at bar 12, chorus2 at bar 28 (see make_test_track.SECTIONS).
    a = matrix[12 * beats_per_bar : 16 * beats_per_bar]
    b = matrix[28 * beats_per_bar : 32 * beats_per_bar]
    unrelated = matrix[4 * beats_per_bar : 8 * beats_per_bar]
    assert np.linalg.norm(a - b) < np.linalg.norm(a - unrelated)


# --------------------------------------------------------------------------- jumps


def test_jump_candidates_respect_constraints(analysis):
    _, _, grid, matrix = analysis
    found = jumps.find(matrix, grid, align="bar", min_jump_bars=2)
    assert len(found) > 0
    # Destination must be a downbeat under "bar" alignment...
    assert np.all(grid.phase(found.dst) == 0)
    # ...and share the phase of the beat it replaces.
    assert np.all(grid.phase(found.src + 1) == grid.phase(found.dst))
    # No stutter-length jumps.
    assert np.all(np.abs(found.dst - (found.src + 1)) >= 2 * grid.meter)


def test_phrase_alignment_is_stricter(analysis):
    _, _, grid, matrix = analysis
    bar = jumps.find(matrix, grid, align="bar")
    phrase = jumps.find(matrix, grid, align="phrase")
    assert len(phrase) < len(bar)


def test_unknown_align_rejected(analysis):
    _, _, grid, matrix = analysis
    with pytest.raises(ValueError):
        jumps.find(matrix, grid, align="sideways")


# --------------------------------------------------------------------------- path


def test_plan_hits_target_beat_count(analysis):
    _, _, grid, matrix = analysis
    found = jumps.find(matrix, grid, align="bar")
    plan = path.solve(grid, found, target_beats=120, tolerance_beats=4)
    assert abs(plan.beats_emitted - 120) <= 4
    assert plan.segments[0].start == 0
    assert plan.segments[-1].end == grid.n_beats - 1


def test_segment_beat_counts_sum_to_emitted(analysis):
    _, _, grid, matrix = analysis
    found = jumps.find(matrix, grid, align="bar")
    plan = path.solve(grid, found, target_beats=120, tolerance_beats=4)
    assert sum(s.n_beats for s in plan.segments) == plan.beats_emitted
    # One splice per junction between segments.
    assert plan.n_edits == len(plan.segments) - 1


def test_identity_target_needs_no_edits(analysis):
    _, _, grid, matrix = analysis
    found = jumps.find(matrix, grid, align="bar")
    plan = path.solve(grid, found, target_beats=grid.n_beats, tolerance_beats=0)
    assert plan.n_edits == 0
    assert plan.beats_emitted == grid.n_beats


def test_edit_penalty_discourages_edits(analysis):
    _, _, grid, matrix = analysis
    found = jumps.find(matrix, grid, align="beat", min_jump_bars=1)
    cheap = path.solve(grid, found, 120, 8, edit_penalty=0.0)
    dear = path.solve(grid, found, 120, 8, edit_penalty=50.0)
    assert dear.n_edits <= cheap.n_edits


def test_impossible_target_raises(analysis):
    _, _, grid, matrix = analysis
    empty = jumps.Jumps(np.array([], int), np.array([], int), np.array([], float))
    with pytest.raises(path.NoPlanError):
        path.solve(grid, empty, target_beats=10, tolerance_beats=0)


# --------------------------------------------------------------------------- render


def test_render_length_matches_plan(analysis):
    loaded, _, grid, matrix = analysis
    found = jumps.find(matrix, grid, align="bar")
    plan = path.solve(grid, found, target_beats=120, tolerance_beats=4)
    out = render.render(loaded, grid, plan)

    intro = grid.start
    outro = loaded.duration - grid.end
    expected = plan.duration + intro + outro
    assert out.shape[1] / loaded.sr == pytest.approx(expected, abs=0.05)
    assert out.shape[0] == loaded.n_channels


def test_render_has_no_clicks_or_silence(analysis):
    """A bad splice shows up as a discontinuity far larger than the source's."""
    loaded, _, grid, matrix = analysis
    found = jumps.find(matrix, grid, align="bar")
    plan = path.solve(grid, found, target_beats=120, tolerance_beats=4)
    out = render.render(loaded, grid, plan)[0]

    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0.1
    source_max_step = np.max(np.abs(np.diff(loaded.samples[0])))
    assert np.max(np.abs(np.diff(out))) <= source_max_step * 1.5


def test_dropping_intro_outro_shortens_output(analysis):
    loaded, _, grid, matrix = analysis
    found = jumps.find(matrix, grid, align="bar")
    plan = path.solve(grid, found, target_beats=120, tolerance_beats=4)
    full = render.render(loaded, grid, plan)
    bare = render.render(loaded, grid, plan, keep_intro=False, keep_outro=False)
    assert bare.shape[1] <= full.shape[1]


# --------------------------------------------------------------------------- pipeline


@pytest.mark.parametrize("target", [30.0, 60.0, 90.0, 150.0])
def test_remix_hits_target_duration(track, target):
    result = remix(track, target, RemixOptions(tolerance=2.0, vocals="off"))
    assert result.duration == pytest.approx(target, abs=2.5)


def test_lengthening_repeats_material(track):
    result = remix(track, 150.0, RemixOptions(vocals="off"))
    assert result.duration > result.source_duration
    assert result.plan.n_edits >= 1


def test_target_shorter_than_intro_outro_rejected(track):
    with pytest.raises(ValueError, match="intro"):
        remix(track, 0.05, RemixOptions(vocals="off"))


def test_non_positive_target_rejected(track):
    with pytest.raises(ValueError):
        remix(track, 0.0)


def test_auto_relax_falls_back(track):
    """Phrase alignment is too coarse for a tight tolerance here; it should relax."""
    result = remix(track, 60.0, RemixOptions(align="phrase", tolerance=2.0, vocals="off"))
    assert result.requested_align == "phrase"
    assert result.align_used in ("bar", "beat")


def test_no_relax_propagates_failure(track):
    with pytest.raises(path.NoPlanError):
        remix(track, 60.0, RemixOptions(align="phrase", tolerance=2.0, auto_relax=False, vocals="off"))


@pytest.fixture(scope="session")
def vocal_track(tmp_path_factory) -> tuple[Path, np.ndarray]:
    """Mix with vocals, plus the isolated vocal bus as ground truth.

    Demucs is deliberately bypassed here: the synthetic voice is not
    convincing enough for a real separator (htdemucs routes it to 'other'), so
    driving these tests through the model would test the model, not remox.
    """
    mix, stem = make_test_track.build_with_stem()
    dest = tmp_path_factory.mktemp("fixtures") / "vocal.wav"
    sf.write(str(dest), mix, make_test_track.SR)
    return dest, stem


@pytest.fixture(scope="session")
def vocal_analysis(vocal_track):
    import librosa

    path, stem = vocal_track
    loaded = audio_io.load(path)
    grid = beats.track(loaded.analysis_signal())
    resampled = librosa.resample(
        stem, orig_sr=make_test_track.SR, target_sr=audio_io.ANALYSIS_SR
    )
    return loaded, grid, resampled


@pytest.fixture
def fake_separation(monkeypatch, vocal_analysis):
    """Make vocals.analyse return the ground-truth stem instead of separating."""
    _, _, stem = vocal_analysis
    monkeypatch.setattr(vocals, "_separate", lambda *a, **k: stem)


# --------------------------------------------------------------------------- vocals


def test_instrumental_track_reports_no_vocals(analysis):
    """Separation residue on an instrumental must not be rescaled into 'vocals'."""
    loaded, signal, _, _ = analysis
    residue = np.random.default_rng(0).standard_normal(len(signal)) * 1e-4
    envelope, present = vocals._envelope(residue, signal)
    assert not present
    assert np.all(envelope == 0)


def test_ground_truth_stem_is_detected(vocal_analysis):
    loaded, _, stem = vocal_analysis
    envelope, present = vocals._envelope(stem, loaded.analysis_signal())
    assert present
    # Fixture sings 2 bars in 4, across 5 of 8 sections: ~38% of the track.
    assert 0.30 < (envelope > 0.5).mean() < 0.45


def test_crossing_high_inside_phrase_low_in_rests(vocal_analysis):
    loaded, grid, stem = vocal_analysis
    envelope, _ = vocals._envelope(stem, loaded.analysis_signal())
    crossing = vocals.crossing_scores(envelope, grid)

    assert len(crossing) == grid.n_beats + 1
    assert crossing.min() >= 0.0 and crossing.max() <= 1.0
    # The fixture alternates sung and silent 2-bar spans, so downbeats must
    # split into clearly unsafe and clearly safe groups.
    downbeats = [i for i in range(grid.n_beats) if grid.phase(i) == 0]
    unsafe = sum(crossing[i] > 0.5 for i in downbeats)
    safe = sum(crossing[i] < 0.1 for i in downbeats)
    assert unsafe >= 10
    assert safe >= 20


def test_vocal_penalty_raises_cost_of_unsafe_cuts(vocal_analysis):
    loaded, grid, stem = vocal_analysis
    matrix = features.extract(loaded.analysis_signal(), grid)
    envelope, _ = vocals._envelope(stem, loaded.analysis_signal())
    crossing = vocals.crossing_scores(envelope, grid)

    plain = jumps.find(matrix, grid, align="bar")
    guarded = jumps.find(matrix, grid, align="bar", vocal_crossing=crossing, vocal_weight=5.0)

    def worst(found):
        return max(
            max(crossing[s + 1], crossing[d]) for s, d in zip(found.src, found.dst)
        )

    assert len(guarded) > 0
    assert worst(guarded) <= worst(plain)


def test_vocal_veto_excludes_phrase_cuts(vocal_analysis):
    loaded, grid, stem = vocal_analysis
    matrix = features.extract(loaded.analysis_signal(), grid)
    envelope, _ = vocals._envelope(stem, loaded.analysis_signal())
    crossing = vocals.crossing_scores(envelope, grid)

    found = jumps.find(matrix, grid, align="bar", vocal_crossing=crossing, vocal_veto=0.3)
    assert len(found) > 0
    assert np.all(crossing[found.src + 1] <= 0.3)
    assert np.all(crossing[found.dst] <= 0.3)


def test_model_load_prefers_the_local_cache(monkeypatch):
    """Cached weights must load without contacting the Hub."""
    import os
    import sys
    import types

    attempts = []

    class FakeModel:
        sources = ["drums", "bass", "other", "vocals"]

    def fake_get_model(name):
        attempts.append(os.environ.get("HF_HUB_OFFLINE"))
        return FakeModel()

    module = types.ModuleType("demucs.pretrained")
    module.get_model = fake_get_model
    monkeypatch.setitem(sys.modules, "demucs.pretrained", module)

    model = vocals._load_pretrained("htdemucs")
    assert model.sources == FakeModel.sources
    assert attempts == ["1"]  # offline only; no online round-trip


def test_model_load_falls_back_online_when_uncached(monkeypatch):
    """A first run with no cached weights must still download them."""
    import os
    import sys
    import types

    attempts = []

    class FakeModel:
        sources = ["drums", "bass", "other", "vocals"]

    def fake_get_model(name):
        offline = os.environ.get("HF_HUB_OFFLINE")
        attempts.append(offline)
        if offline == "1":
            raise OSError("not cached locally")
        return FakeModel()

    module = types.ModuleType("demucs.pretrained")
    module.get_model = fake_get_model
    monkeypatch.setitem(sys.modules, "demucs.pretrained", module)

    model = vocals._load_pretrained("htdemucs")
    assert model.sources == FakeModel.sources
    assert attempts == ["1", "0"]  # tried cache, then fetched


def test_model_load_restores_offline_env(monkeypatch):
    """The env var must be left as it was found, not clobbered."""
    import os
    import sys
    import types

    class FakeModel:
        sources = ["vocals"]

    module = types.ModuleType("demucs.pretrained")
    module.get_model = lambda name: FakeModel()
    monkeypatch.setitem(sys.modules, "demucs.pretrained", module)

    monkeypatch.setenv("HF_HUB_OFFLINE", "sentinel")
    vocals._load_pretrained("htdemucs")
    assert os.environ["HF_HUB_OFFLINE"] == "sentinel"



def test_wrong_length_crossing_rejected(analysis):
    _, _, grid, matrix = analysis
    with pytest.raises(ValueError, match="length"):
        jumps.find(matrix, grid, vocal_crossing=np.zeros(3))


def test_zero_crossing_is_a_noop(analysis):
    _, _, grid, matrix = analysis
    plain = jumps.find(matrix, grid, align="bar")
    zeroed = jumps.find(
        matrix, grid, align="bar", vocal_crossing=np.zeros(grid.n_beats + 1)
    )
    assert np.allclose(plain.cost, zeroed.cost)


def test_pipeline_respects_veto(vocal_track, fake_separation):
    path, _ = vocal_track
    result = remix(
        path,
        60.0,
        RemixOptions(vocals="on", vocal_veto=0.3, vocal_cache=False, tolerance=4.0),
    )
    activity = result.vocal_activity
    assert activity is not None and activity.present
    for segment in result.plan.segments[1:]:
        assert activity.crossing[segment.start] <= 0.3


def test_vocals_off_skips_separation(track, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("separation should not run when vocals='off'")

    monkeypatch.setattr(vocals, "_separate", explode)
    result = remix(track, 60.0, RemixOptions(vocals="off"))
    assert result.vocal_activity is None


def test_vocals_auto_tolerates_missing_demucs(track, monkeypatch):
    def unavailable(*args, **kwargs):
        raise vocals.SeparationUnavailable("no demucs")

    monkeypatch.setattr(vocals, "_separate", unavailable)
    result = remix(track, 60.0, RemixOptions(vocals="auto", vocal_cache=False))
    assert result.vocal_activity is None


def test_vocals_on_requires_demucs(track, monkeypatch):
    def unavailable(*args, **kwargs):
        raise vocals.SeparationUnavailable("no demucs")

    monkeypatch.setattr(vocals, "_separate", unavailable)
    with pytest.raises(vocals.SeparationUnavailable):
        remix(track, 60.0, RemixOptions(vocals="on", vocal_cache=False))


def test_unknown_vocals_mode_rejected(track):
    with pytest.raises(ValueError, match="vocals mode"):
        remix(track, 60.0, RemixOptions(vocals="maybe"))


def test_envelope_cache_round_trip(vocal_track, vocal_analysis, monkeypatch, tmp_path):
    """Second call must reuse the cached envelope rather than separate again."""
    path, _ = vocal_track
    _, grid, stem = vocal_analysis
    monkeypatch.setenv("REMOX_CACHE_DIR", str(tmp_path))

    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return stem

    monkeypatch.setattr(vocals, "_separate", counted)
    loaded = audio_io.load(path)

    first = vocals.analyse(loaded, grid)
    second = vocals.analyse(loaded, grid)

    assert len(calls) == 1
    assert first.present == second.present
    assert np.allclose(first.crossing, second.crossing)


# --------------------------------------------------------------------------- fetch


class _FakeYoutubeDL:
    """Stand-in for yt_dlp.YoutubeDL that fabricates a file instead of downloading.

    Tests must never touch the network, both for speed and because the tool has
    no business pulling third-party media during a test run.
    """

    calls: list[bool] = []
    info: dict = {}
    payload: bytes = b""

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        type(self).calls.append(download)
        if download:
            template = Path(self.options["outtmpl"])
            directory = template.parent
            directory.mkdir(parents=True, exist_ok=True)
            codec = self.options["postprocessors"][0]["preferredcodec"]
            produced = directory / f"{type(self).info['id']}.{codec}"
            produced.write_bytes(type(self).payload or b"fake audio")
            return dict(type(self).info, requested_downloads=[{"filepath": str(produced)}])
        return dict(type(self).info)


@pytest.fixture
def fake_ytdlp(monkeypatch):
    """Install the fake extractor and reset its recorded calls."""
    _FakeYoutubeDL.calls = []
    _FakeYoutubeDL.info = {
        "id": "abc123XYZ_-",
        "title": "Sunlit / Fields (Demo)",
        "uploader": "Example Artist",
        "webpage_url": "https://www.youtube.com/watch?v=abc123XYZ_-",
        "duration": 105.0,
        "license": "Creative Commons Attribution license (reuse allowed)",
    }
    module = type("module", (), {"YoutubeDL": _FakeYoutubeDL})
    monkeypatch.setattr(fetch_mod, "_load_yt_dlp", lambda: module)
    return _FakeYoutubeDL


@pytest.mark.parametrize(
    "text,expected",
    [
        ("https://youtube.com/watch?v=x", True),
        ("HTTPS://YOUTU.BE/x", True),
        ("http://example.com/a.mp3", True),
        ("song.mp3", False),
        (r"C:\music\song.wav", False),
        ("/home/me/song.flac", False),
        ("", False),
    ],
)
def test_is_url(text, expected):
    assert fetch_mod.is_url(text) is expected


def test_ffmpeg_is_resolvable():
    """Either a system ffmpeg or the imageio-ffmpeg static build must be found."""
    resolved = fetch_mod.resolve_ffmpeg()
    assert resolved is not None
    assert Path(resolved).exists()


def test_ffmpeg_actually_converts_to_readable_mp3(track, tmp_path):
    """Smoke-test the real conversion path without downloading anything.

    Verifies the two things the download feature depends on: the bundled ffmpeg
    runs, and libsndfile can read the mp3 it produces.
    """
    import subprocess

    ffmpeg = fetch_mod.resolve_ffmpeg()
    out = tmp_path / "converted.mp3"
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(track), "-b:a", "192k", str(out)],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert out.exists() and out.stat().st_size > 1000

    loaded = audio_io.load(out)
    assert loaded.duration == pytest.approx(SOURCE_DURATION, abs=0.5)


def test_fetch_returns_metadata_and_writes_sidecar(fake_ytdlp, tmp_path):
    result = fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path)

    assert result.path.exists()
    assert result.path.suffix == ".mp3"
    assert result.video_id == "abc123XYZ_-"
    assert result.title == "Sunlit / Fields (Demo)"
    assert result.uploader == "Example Artist"
    assert result.duration == pytest.approx(105.0)
    assert "Creative Commons" in result.license

    sidecar = result.path.with_suffix(".json")
    assert sidecar.exists()
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["url"] == fake_ytdlp.info["webpage_url"]
    assert record["license"] == fake_ytdlp.info["license"]


def test_fetch_reuses_cached_download(fake_ytdlp, tmp_path):
    first = fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path)
    downloads = fake_ytdlp.calls.count(True)

    second = fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path)

    assert fake_ytdlp.calls.count(True) == downloads  # no second transfer
    assert second.path == first.path
    assert second.title == first.title


def test_fetch_can_bypass_cache(fake_ytdlp, tmp_path):
    fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path)
    fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path, use_cache=False)
    assert fake_ytdlp.calls.count(True) == 2


def test_fetch_rejects_playlists(fake_ytdlp, tmp_path):
    fake_ytdlp.info = dict(fake_ytdlp.info, _type="playlist", entries=[])
    with pytest.raises(fetch_mod.FetchError, match="playlist"):
        fetch_mod.fetch("https://youtube.com/playlist?list=x", dest_dir=tmp_path)


def test_fetch_rejects_unknown_format(tmp_path):
    with pytest.raises(fetch_mod.FetchError, match="unsupported audio format"):
        fetch_mod.fetch("https://youtu.be/x", dest_dir=tmp_path, audio_format="aiff")


def test_fetch_requires_ffmpeg(fake_ytdlp, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "resolve_ffmpeg", lambda: None)
    with pytest.raises(fetch_mod.FetchError, match="ffmpeg"):
        fetch_mod.fetch("https://youtu.be/x", dest_dir=tmp_path)


def test_fetch_requires_yt_dlp(monkeypatch, tmp_path):
    def missing():
        raise fetch_mod.FetchError("needs yt-dlp")

    monkeypatch.setattr(fetch_mod, "_load_yt_dlp", missing)
    with pytest.raises(fetch_mod.FetchError, match="yt-dlp"):
        fetch_mod.fetch("https://youtu.be/x", dest_dir=tmp_path)


def test_fetch_surfaces_extractor_errors(monkeypatch, tmp_path):
    class Exploding(_FakeYoutubeDL):
        def extract_info(self, url, download=False):
            raise RuntimeError("Video unavailable")

    module = type("module", (), {"YoutubeDL": Exploding})
    monkeypatch.setattr(fetch_mod, "_load_yt_dlp", lambda: module)
    with pytest.raises(fetch_mod.FetchError, match="Video unavailable"):
        fetch_mod.fetch("https://youtu.be/x", dest_dir=tmp_path)


def test_attribution_and_output_name(fake_ytdlp, tmp_path):
    result = fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path)
    assert "Sunlit" in result.attribution()
    assert "Example Artist" in result.attribution()

    name = fetch_mod.default_output_name(result)
    assert name.name.endswith(".remox.wav")
    # Slashes and parentheses must not leak into a filename.
    assert not set(name.name) & set('/\\:*?"<>|')


def test_cli_accepts_a_url(track, tmp_path, monkeypatch, capsys):
    """A URL input should download, then remix the downloaded file."""
    landed = tmp_path / "abc123XYZ_-.mp3"
    landed.write_bytes(track.read_bytes())
    stub = fetch_mod.Download(
        path=landed,
        video_id="abc123XYZ_-",
        title="Sunlit / Fields (Demo)",
        uploader="Example Artist",
        url="https://www.youtube.com/watch?v=abc123XYZ_-",
        duration=105.0,
        license="Creative Commons Attribution license (reuse allowed)",
    )
    monkeypatch.setattr(cli, "fetch", lambda url, **kwargs: stub)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["https://youtu.be/abc123XYZ_-", "-d", "45", "--vocals", "off"])
    assert code == 0

    out = capsys.readouterr().out
    assert "fetched" in out and "Example Artist" in out
    assert "youtube.com/watch?v=abc123XYZ_-" in out

    produced = tmp_path / "output" / "Sunlit_Fields_Demo.remox.wav"
    assert produced.exists()


def test_cli_fetch_only_skips_remixing(track, tmp_path, monkeypatch, capsys):
    landed = tmp_path / "abc.mp3"
    landed.write_bytes(track.read_bytes())
    stub = fetch_mod.Download(
        path=landed, video_id="abc", title="T", uploader="U", url="https://x/y"
    )
    monkeypatch.setattr(cli, "fetch", lambda url, **kwargs: stub)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["https://youtu.be/abc", "--fetch-only"]) == 0
    assert not list(tmp_path.glob("*.remox.wav"))


def test_cli_fetch_only_requires_a_url(track, capsys):
    assert cli.main([str(track), "--fetch-only"]) == 1
    assert "requires a URL" in capsys.readouterr().err


def test_cli_requires_duration_without_fetch_only(track):
    with pytest.raises(SystemExit):
        cli.main([str(track)])


def test_cli_reports_download_failure(monkeypatch, capsys):
    def boom(url, **kwargs):
        raise fetch_mod.FetchError("Video unavailable")

    monkeypatch.setattr(cli, "fetch", boom)
    assert cli.main(["https://youtu.be/x", "-d", "30"]) == 1
    assert "Video unavailable" in capsys.readouterr().err


# --------------------------------------------------------------------------- variants


@pytest.fixture(scope="session")
def prepared(track):
    """One analysis shared by the variant tests."""
    return pipeline.analyse(track, RemixOptions(vocals="off"))


def test_feature_blocks_combine_matches_extract(analysis):
    _, signal, grid, matrix = analysis
    blocks = features.blocks(signal, grid)
    assert np.allclose(blocks.combine(features.FeatureWeights()), matrix)
    assert blocks.n_beats == grid.n_beats


def test_reweighting_changes_the_matrix(prepared):
    a = prepared.blocks.combine(features.FeatureWeights(1.6, 0.3, 0.3))
    b = prepared.blocks.combine(features.FeatureWeights(0.5, 1.5, 0.4))
    assert not np.allclose(a, b)


def test_analysis_is_reused_not_recomputed(track, monkeypatch):
    """generate() must not re-run beat tracking per variant."""
    calls = []
    original = beats.track

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(beats, "track", counted)
    monkeypatch.setattr(pipeline.beats, "track", counted)
    variants.generate(track, 30.0, count=5, options=RemixOptions(vocals="off"))
    assert len(calls) == 1


def test_generate_returns_distinct_plans(prepared, track):
    result = variants.generate(
        track, 30.0, count=6, options=RemixOptions(vocals="off"), analysis=prepared
    )
    assert len(result) == 6
    signatures = [v.signature for v in result.variants]
    assert len(set(signatures)) == len(signatures)


def test_all_variants_hit_the_target(prepared, track):
    options = RemixOptions(vocals="off", tolerance=2.0)
    result = variants.generate(track, 30.0, count=6, options=options, analysis=prepared)
    for variant in result.variants:
        total = variant.plan.duration + prepared.fixed_duration(variant.options)
        assert total == pytest.approx(30.0, abs=2.5)


def test_presets_never_override_tolerance():
    """A variant outside the requested length would not be a usable option."""
    for preset in variants.PRESETS:
        assert "tolerance" not in preset.overrides


def test_preset_overrides_apply():
    base = RemixOptions(vocals="off")
    minimal = next(p for p in variants.PRESETS if p.name == "minimal")
    applied = minimal.apply(base)
    assert applied.edit_penalty == 8.0
    assert applied.min_jump_bars == 4
    assert base.edit_penalty == 1.0  # base untouched


def test_variants_are_indexed_and_labelled(prepared, track):
    result = variants.generate(
        track, 30.0, count=4, options=RemixOptions(vocals="off"), analysis=prepared
    )
    assert [v.index for v in result.variants] == [1, 2, 3, 4]
    assert result.variants[0].name == "balanced"
    assert all(v.source in ("preset", "diversity") for v in result.variants)
    assert all(v.description for v in result.variants)


def test_single_variant_matches_plain_remix(prepared, track):
    options = RemixOptions(vocals="off")
    result = variants.generate(track, 30.0, count=1, options=options, analysis=prepared)
    plan, _ = pipeline.plan_for(prepared, 30.0, options)
    assert result.variants[0].signature == tuple((s.start, s.end) for s in plan.segments)


def test_count_must_be_positive(track, prepared):
    with pytest.raises(ValueError):
        variants.generate(track, 30.0, count=0, analysis=prepared)


def test_shortfall_is_reported_not_faked(prepared, track):
    """Asking for more versions than exist must return fewer, not duplicates."""
    result = variants.generate(
        track, 30.0, count=200, options=RemixOptions(vocals="off"), analysis=prepared
    )
    assert len(result) < 200
    assert result.shortfall == 200 - len(result)
    signatures = [v.signature for v in result.variants]
    assert len(set(signatures)) == len(signatures)


def test_save_all_writes_audio_and_manifest(prepared, track, tmp_path):
    result = variants.generate(
        track, 30.0, count=3, options=RemixOptions(vocals="off"), analysis=prepared
    )
    written = result.save_all(tmp_path)

    assert len(written) == 3
    for out in written:
        assert out.exists()
        data, sr = sf.read(str(out))
        assert len(data) / sr == pytest.approx(30.0, abs=2.5)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["produced_variants"] == 3
    assert manifest["requested_variants"] == 3
    assert manifest["target_duration"] == 30.0
    assert len(manifest["variants"]) == 3
    first = manifest["variants"][0]
    assert first["file"] == written[0].name
    assert first["segments"] and "start_time" in first["segments"][0]
    assert first["rendered_duration"] == pytest.approx(30.0, abs=2.5)


def test_rendered_variants_actually_differ(prepared, track, tmp_path):
    """Distinct plans must produce distinct audio, not identical files."""
    result = variants.generate(
        track, 30.0, count=3, options=RemixOptions(vocals="off"), analysis=prepared
    )
    written = result.save_all(tmp_path)
    contents = [out.read_bytes() for out in written]
    assert len(set(contents)) == len(contents)


# --------------------------------------------------------------------------- stutter


def test_plan_detects_repeated_segments():
    plan = path.Plan(
        segments=[path.Segment(0, 5), path.Segment(9, 12), path.Segment(9, 12)],
        splice_costs=[0.1, 0.1],
    )
    assert plan.repeats == 1


def test_clean_plan_reports_no_repeats(prepared):
    plan, _ = pipeline.plan_for(prepared, 30.0, RemixOptions(vocals="off"))
    assert plan.repeats == 0


def test_extreme_settings_can_produce_stutter(prepared):
    """Documents the known defect the variant generator must screen out."""
    loose = RemixOptions(
        vocals="off", edit_penalty=0.001, min_jump_bars=1, align="beat"
    )
    plan, _ = pipeline.plan_for(prepared, 30.0, loose)
    assert plan.repeats > 0


def test_variants_never_include_stutter(prepared, track):
    loose = RemixOptions(vocals="off", edit_penalty=0.001, min_jump_bars=1, align="beat")
    result = variants.generate(track, 30.0, count=8, options=loose, analysis=prepared)
    assert len(result) >= 1
    for variant in result.variants:
        assert variant.plan.repeats == 0


def test_cli_warns_about_stutter(track, capsys):
    code = main(
        [
            str(track), "-d", "30", "--vocals", "off", "--quiet",
            "--edits", "0.001", "--min-jump-bars", "1", "--align", "beat",
        ]
    )
    assert code == 0
    # --quiet suppresses the report, so re-run loudly to see the warning.
    main(
        [
            str(track), "-d", "30", "--vocals", "off",
            "--edits", "0.001", "--min-jump-bars", "1", "--align", "beat",
        ]
    )
    assert "stutter" in capsys.readouterr().out


def test_default_download_dir_is_downloads_music(monkeypatch, tmp_path):
    monkeypatch.delenv("REMOX_DOWNLOAD_DIR", raising=False)
    monkeypatch.setattr(fetch_mod, "_windows_downloads", lambda: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert fetch_mod.download_dir() == tmp_path / "Downloads" / "Music"


def test_download_dir_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("REMOX_DOWNLOAD_DIR", str(tmp_path / "elsewhere"))
    assert fetch_mod.download_dir() == tmp_path / "elsewhere"


def test_download_dir_follows_windows_redirect(monkeypatch, tmp_path):
    """A redirected Downloads folder must win over the literal ~/Downloads."""
    monkeypatch.delenv("REMOX_DOWNLOAD_DIR", raising=False)
    redirected = tmp_path / "OneDrive" / "Downloads"
    monkeypatch.setattr(fetch_mod, "_windows_downloads", lambda: redirected)
    assert fetch_mod.download_dir() == redirected / "Music"


def test_download_dir_is_not_the_cache(monkeypatch, tmp_path):
    """Clearing the cache must never delete the user's downloaded media."""
    monkeypatch.delenv("REMOX_DOWNLOAD_DIR", raising=False)
    monkeypatch.setenv("REMOX_CACHE_DIR", str(tmp_path / "cache"))
    assert cache.cache_dir() not in fetch_mod.download_dir().parents
    assert fetch_mod.download_dir() != cache.cache_dir("downloads")


def test_fetch_uses_the_default_dir(fake_ytdlp, monkeypatch, tmp_path):
    monkeypatch.setenv("REMOX_DOWNLOAD_DIR", str(tmp_path / "Music"))
    result = fetch_mod.fetch("https://youtu.be/abc123XYZ_-")
    assert result.path.parent == tmp_path / "Music"
    assert result.path.exists()


def test_fetch_creates_the_default_dir(fake_ytdlp, monkeypatch, tmp_path):
    target = tmp_path / "does" / "not" / "exist" / "Music"
    monkeypatch.setenv("REMOX_DOWNLOAD_DIR", str(target))
    assert not target.exists()
    fetch_mod.fetch("https://youtu.be/abc123XYZ_-")
    assert target.is_dir()


# --------------------------------------------------------------------------- entry


def _section_of(beat, grid):
    """Name the fixture section containing a beat, for readable assertions."""
    at = 0
    bar = int(grid.bar_index(beat))
    for name, bars, *_ in make_test_track.SECTIONS:
        if at <= bar < at + bars:
            return name
        at += bars
    return "?"


def test_novelty_peaks_at_section_boundaries(prepared):
    """Foote novelty should spike where the fixture changes section."""
    combined = prepared.blocks.combine()
    curve = entry.novelty(combined)
    grid = prepared.grid

    boundaries, at = [], 0
    for _, bars, *_ in make_test_track.SECTIONS[:-1]:
        at += bars
        boundaries.append(at)

    # Beats at a section change, versus beats in the middle of a section.
    edge_beats = [b for b in range(grid.n_beats) if grid.bar_index(b) in boundaries]
    inner_beats = [
        b
        for b in range(grid.n_beats)
        if grid.bar_index(b) not in boundaries and grid.phase(b) == 0
    ]
    assert np.mean(curve[edge_beats]) > np.mean(curve[inner_beats])


def test_entry_costs_respect_alignment(prepared):
    grid = prepared.grid
    phrase = entry.costs(prepared.blocks, grid, align="phrase")
    bar = entry.costs(prepared.blocks, grid, align="bar")

    assert len(phrase) == grid.n_beats
    allowed = np.isfinite(phrase)
    beats_idx = np.arange(grid.n_beats)
    # Phrase mode may only open on 4-bar boundaries (beat 0 always allowed).
    assert np.all(grid.phase(beats_idx[allowed]) == 0)
    # Bar mode is strictly more permissive.
    assert np.isfinite(bar).sum() > allowed.sum()


def test_entry_costs_are_normalised(prepared):
    scores = entry.costs(prepared.blocks, prepared.grid, align="bar")
    finite = scores[np.isfinite(scores)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0


def test_entry_avoids_vocal_phrases(vocal_analysis):
    loaded, grid, stem = vocal_analysis
    blocks = features.blocks(loaded.analysis_signal(), grid)
    envelope, _ = vocals._envelope(stem, loaded.analysis_signal())
    crossing = vocals.crossing_scores(envelope, grid)

    plain = entry.costs(blocks, grid, align="bar")
    guarded = entry.costs(blocks, grid, crossing=crossing, align="bar")

    best_plain = int(np.argmin(plain))
    best_guarded = int(np.argmin(guarded))
    assert crossing[best_guarded] <= crossing[best_plain]


def test_entry_rejects_wrong_length_crossing(prepared):
    with pytest.raises(ValueError, match="length"):
        entry.costs(prepared.blocks, prepared.grid, crossing=np.ones(5))


def test_resolve_fixed_snaps_forward(prepared):
    grid = prepared.grid
    scores = entry.resolve_fixed(grid, 60.0, align="phrase")
    chosen = int(np.argmin(scores))
    assert np.isfinite(scores[chosen])
    assert np.isfinite(scores).sum() == 1  # exactly one permitted opening
    assert grid.times[chosen] >= 60.0
    assert grid.phase(chosen) == 0


def test_resolve_fixed_past_end_falls_back(prepared):
    scores = entry.resolve_fixed(prepared.grid, 10_000.0, align="bar")
    assert np.isfinite(scores).sum() == 1


def test_auto_start_opens_at_a_chorus(prepared):
    """The whole point: don't open on the intro."""
    plan, _ = pipeline.plan_for(
        prepared, 30.0, RemixOptions(vocals="off", start="auto")
    )
    opening = plan.segments[0].start
    assert opening != 0
    assert _section_of(opening, prepared.grid).startswith("chorus")


def test_beginning_start_is_unchanged(prepared):
    plan, _ = pipeline.plan_for(prepared, 30.0, RemixOptions(vocals="off"))
    assert plan.segments[0].start == 0


def test_explicit_start_is_honoured(prepared):
    plan, _ = pipeline.plan_for(prepared, 30.0, RemixOptions(vocals="off", start=60.0))
    assert prepared.grid.times[plan.segments[0].start] >= 60.0


def test_unknown_start_mode_rejected(prepared):
    with pytest.raises(ValueError, match="start mode"):
        prepared.entry_costs(RemixOptions(start="sideways"))


def test_solver_rejects_wrong_length_entry_costs(prepared):
    matrix = prepared.blocks.combine()
    found = jumps.find(matrix, prepared.grid, align="bar")
    with pytest.raises(ValueError, match="entry_costs"):
        path.solve(prepared.grid, found, 60, 4, entry_costs=np.zeros(3))


def test_solver_errors_when_no_entry_permitted(prepared):
    matrix = prepared.blocks.combine()
    found = jumps.find(matrix, prepared.grid, align="bar")
    blocked = np.full(prepared.grid.n_beats, np.inf)
    with pytest.raises(path.NoPlanError, match="entry point"):
        path.solve(prepared.grid, found, 60, 4, entry_costs=blocked)


@pytest.mark.parametrize("target", [20.0, 30.0, 45.0])
def test_auto_start_still_hits_the_target(track, target):
    result = remix(track, target, RemixOptions(vocals="off", start="auto"))
    assert result.duration == pytest.approx(target, abs=2.5)


def test_intro_is_not_glued_on_when_starting_late(track):
    """A late opening must not have the track's first bars prepended."""
    result = remix(track, 30.0, RemixOptions(vocals="off", start="auto"))
    grid, plan = result.grid, result.plan
    assert plan.segments[0].start != 0

    expected = plan.duration + (result.source_duration - grid.end)
    assert result.duration == pytest.approx(expected, abs=0.05)


def test_render_skips_intro_for_late_start(prepared):
    late = path.Plan(
        segments=[path.Segment(80, prepared.grid.n_beats - 1)],
        splice_costs=[],
        beats_emitted=prepared.grid.n_beats - 80,
    )
    early = path.Plan(
        segments=[path.Segment(0, prepared.grid.n_beats - 1)],
        splice_costs=[],
        beats_emitted=prepared.grid.n_beats,
    )
    late_audio = render.render(prepared.track, prepared.grid, late)
    early_audio = render.render(prepared.track, prepared.grid, early)

    # The early plan carries the pre-beat-0 intro; the late one must not.
    late_expected = prepared.track.n_samples - int(
        prepared.grid.times[80] * prepared.track.sr
    )
    assert late_audio.shape[1] == pytest.approx(late_expected, abs=200)
    assert early_audio.shape[1] > late_audio.shape[1]


def test_cli_auto_start(track, tmp_path, capsys):
    out = tmp_path / "auto.wav"
    code = main(
        [str(track), "-d", "30", "--start", "auto", "-o", str(out), "--vocals", "off"]
    )
    assert code == 0
    assert out.exists()
    report = capsys.readouterr().out
    assert "opens at" in report and "intro dropped" in report


def test_cli_start_accepts_clock_time(track, tmp_path):
    out = tmp_path / "at.wav"
    assert (
        main([str(track), "-d", "30", "--start", "1:00", "-o", str(out),
              "--vocals", "off", "--quiet"])
        == 0
    )
    assert out.exists()


@pytest.mark.parametrize("text,expected", [
    ("beginning", "beginning"),
    ("0", "beginning"),
    ("auto", "auto"),
    ("1:00", 60.0),
    ("90s", 90.0),
])
def test_parse_start(text, expected):
    assert cli._parse_start(text) == expected


def test_parse_start_rejects_garbage():
    with pytest.raises(Exception):
        cli._parse_start("chorus-ish")


# --------------------------------------------------------------------------- isolation


def _fake_apply(n_samples=1000, sr=44100, channels=2):
    """Build a stub _apply returning four known, distinguishable stems."""
    import torch

    sources = ["drums", "bass", "other", "vocals"]
    stems = torch.zeros(4, channels, n_samples)
    for i in range(4):
        stems[i] = i + 1  # drums=1, bass=2, other=3, vocals=4
    return lambda track, model, device: (stems, sources, sr)


def test_isolate_sums_non_vocal_stems(track, monkeypatch):
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals, "_apply", _fake_apply(loaded.n_samples, loaded.sr, loaded.n_channels)
    )
    instrumental, _ = vocals.isolate(loaded, keep="instrumental")
    # drums + bass + other = 1 + 2 + 3, with vocals (4) excluded.
    assert np.allclose(instrumental.samples, 6.0)


def test_isolate_vocals_keeps_only_the_vocal_stem(track, monkeypatch):
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals, "_apply", _fake_apply(loaded.n_samples, loaded.sr, loaded.n_channels)
    )
    acapella, _ = vocals.isolate(loaded, keep="vocals")
    assert np.allclose(acapella.samples, 4.0)


def test_isolate_returns_vocal_signal_for_analysis(track, monkeypatch):
    """The vocal stem comes back too, so no second separation pass is needed."""
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals, "_apply", _fake_apply(loaded.n_samples, loaded.sr, loaded.n_channels)
    )
    _, vocal_signal = vocals.isolate(loaded, keep="instrumental")
    assert vocal_signal.ndim == 1
    # Resampling to the analysis rate rings slightly at the edges, so compare
    # the bulk of the signal rather than demanding an exact constant.
    assert float(np.median(vocal_signal)) == pytest.approx(4.0, abs=0.05)


def test_isolate_preserves_length_and_channels(track, monkeypatch):
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals, "_apply", _fake_apply(loaded.n_samples, loaded.sr, loaded.n_channels)
    )
    instrumental, _ = vocals.isolate(loaded, keep="instrumental")
    assert instrumental.n_samples == loaded.n_samples
    assert instrumental.n_channels == loaded.n_channels
    assert instrumental.sr == loaded.sr


def test_isolate_rejects_unknown_mode(track, monkeypatch):
    loaded = audio_io.load(track)
    with pytest.raises(ValueError, match="isolate mode"):
        vocals.isolate(loaded, keep="drums")


@pytest.mark.parametrize("produced", [900, 1100])
def test_to_track_rate_pads_or_trims(track, produced):
    """A stem a few samples off would shift every beat time; it must be exact."""
    loaded = audio_io.load(track)
    stub = audio_io.Audio(
        samples=np.zeros((1, 1000), dtype=np.float32), sr=loaded.sr, path=loaded.path
    )
    signal = np.ones((1, produced), dtype=np.float32)
    fixed = vocals._to_track_rate(signal, stub.sr, stub)
    assert fixed.shape == (1, 1000)


def test_to_track_rate_downmixes_to_source_layout(track):
    loaded = audio_io.load(track)  # fixture is mono
    stereo = np.ones((2, loaded.n_samples), dtype=np.float32)
    fixed = vocals._to_track_rate(stereo, loaded.sr, loaded)
    assert fixed.shape[0] == loaded.n_channels


def test_from_stem_builds_activity(vocal_analysis):
    loaded, grid, stem = vocal_analysis
    activity = vocals.from_stem(stem, loaded.analysis_signal(), grid)
    assert activity.present
    assert len(activity.crossing) == grid.n_beats + 1


def test_analyse_sets_render_source_when_isolating(track, monkeypatch):
    loaded = audio_io.load(track)
    silent = audio_io.Audio(
        samples=np.zeros_like(loaded.samples), sr=loaded.sr, path=loaded.path
    )
    calls = []

    def fake_isolate(t, keep="instrumental", model="htdemucs", device=None):
        calls.append(keep)
        return silent, np.zeros(1000, dtype=np.float32)

    monkeypatch.setattr(vocals, "isolate", fake_isolate)
    result = pipeline.analyse(track, RemixOptions(isolate="instrumental"))

    assert calls == ["instrumental"]
    assert result.render_source is silent
    assert result.output_audio is silent
    # Timing still comes from the full mix.
    assert result.track is not silent
    assert result.grid.n_beats > 0


def test_analyse_separates_only_once_when_isolating(track, monkeypatch):
    """Isolation and the vocal envelope must share one separation pass."""
    loaded = audio_io.load(track)
    separate_calls = []
    monkeypatch.setattr(
        vocals, "_separate", lambda *a, **k: separate_calls.append(1) or np.zeros(1000)
    )
    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            audio_io.Audio(np.zeros_like(loaded.samples), loaded.sr, loaded.path),
            np.zeros(1000, dtype=np.float32),
        ),
    )
    pipeline.analyse(track, RemixOptions(isolate="instrumental"))
    assert separate_calls == []  # _separate never used; isolate supplied the stem


def test_no_isolation_leaves_render_source_unset(prepared):
    assert prepared.render_source is None
    assert prepared.output_audio is prepared.track


def test_remix_renders_from_the_isolated_audio(track, monkeypatch):
    """The output must be cut from the instrumental, not the original mix."""
    loaded = audio_io.load(track)
    marker = audio_io.Audio(
        samples=np.full_like(loaded.samples, 0.25), sr=loaded.sr, path=loaded.path
    )
    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            marker,
            np.zeros(1000, dtype=np.float32),
        ),
    )
    result = remix(track, 30.0, RemixOptions(isolate="instrumental", vocals="off"))
    # Every sample came from the constant-valued stand-in, not the real mix.
    # Splice crossfades briefly sum two copies (0.25 * sqrt(2) at the midpoint),
    # so allow that band rather than demanding an exact constant everywhere.
    samples = np.abs(result.samples)
    assert samples.min() >= 0.24
    assert samples.max() <= 0.36
    assert float(np.isclose(samples, 0.25).mean()) > 0.99


def test_cli_isolate_without_duration_exports_stem(track, tmp_path, monkeypatch, capsys):
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            audio_io.Audio(np.full_like(loaded.samples, 0.5), loaded.sr, loaded.path),
            np.zeros(1000, dtype=np.float32),
        ),
    )
    out = tmp_path / "inst.wav"
    code = main([str(track), "--isolate", "instrumental", "-o", str(out)])
    assert code == 0
    assert out.exists()
    data, sr = sf.read(str(out))
    assert len(data) / sr == pytest.approx(loaded.duration, abs=0.05)
    assert "isolated   instrumental" in capsys.readouterr().out


def test_cli_isolate_default_output_name(track, tmp_path, monkeypatch):
    loaded = audio_io.load(track)
    copy = tmp_path / "song.wav"
    copy.write_bytes(track.read_bytes())
    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="vocals", model="htdemucs", device=None: (
            audio_io.Audio(np.zeros_like(loaded.samples), loaded.sr, loaded.path),
            np.zeros(1000, dtype=np.float32),
        ),
    )
    monkeypatch.chdir(tmp_path)
    assert main([str(copy), "--isolate", "vocals", "--quiet"]) == 0
    assert (tmp_path / "output" / "song.vocals.wav").exists()


def test_cli_isolate_with_duration_remixes(track, tmp_path, monkeypatch):
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            audio_io.Audio(np.full_like(loaded.samples, 0.5), loaded.sr, loaded.path),
            np.zeros(1000, dtype=np.float32),
        ),
    )
    out = tmp_path / "cut.wav"
    code = main(
        [str(track), "-d", "30", "--isolate", "instrumental", "-o", str(out),
         "--vocals", "off", "--quiet"]
    )
    assert code == 0
    data, sr = sf.read(str(out))
    assert len(data) / sr == pytest.approx(30.0, abs=2.5)


def test_cli_still_requires_duration_without_isolate(track):
    with pytest.raises(SystemExit):
        main([str(track)])


# --------------------------------------------------------------------------- reuse


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=BmfTc54KE-g", "BmfTc54KE-g"),
        ("https://youtu.be/BmfTc54KE-g", "BmfTc54KE-g"),
        ("https://youtube.com/shorts/BmfTc54KE-g", "BmfTc54KE-g"),
        ("https://www.youtube.com/embed/BmfTc54KE-g", "BmfTc54KE-g"),
        ("https://www.youtube.com/watch?v=BmfTc54KE-g&list=X&t=30", "BmfTc54KE-g"),
        ("https://example.com/song.mp3", None),
        ("not a url", None),
    ],
)
def test_id_from_url(url, expected):
    assert fetch_mod.id_from_url(url) == expected


def test_cached_download_makes_no_network_calls(fake_ytdlp, tmp_path):
    """A track already on disk must be reused without contacting the extractor."""
    fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path)
    before = len(fake_ytdlp.calls)

    again = fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path)

    assert len(fake_ytdlp.calls) == before  # no probe, no download
    assert again.video_id == "abc123XYZ_-"
    assert again.title == "Sunlit / Fields (Demo)"


def test_unparseable_url_still_probes(fake_ytdlp, tmp_path):
    """URLs whose id we cannot read locally must fall back to asking."""
    fake_ytdlp.info = dict(fake_ytdlp.info, id="opaque1")
    fetch_mod.fetch("https://example.com/track/opaque1", dest_dir=tmp_path)
    before = len(fake_ytdlp.calls)
    fetch_mod.fetch("https://example.com/track/opaque1", dest_dir=tmp_path)
    assert len(fake_ytdlp.calls) > before


def test_cache_bypass_still_redownloads(fake_ytdlp, tmp_path):
    fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path)
    before = fake_ytdlp.calls.count(True)
    fetch_mod.fetch("https://youtu.be/abc123XYZ_-", dest_dir=tmp_path, use_cache=False)
    assert fake_ytdlp.calls.count(True) == before + 1


def test_isolate_cache_round_trip(track, tmp_path, monkeypatch):
    """The second call must reuse the FLAC instead of separating again."""
    monkeypatch.setenv("REMOX_CACHE_DIR", str(tmp_path))
    loaded = audio_io.load(track)
    calls = []

    def counted(t, keep="instrumental", model="htdemucs", device=None):
        calls.append(keep)
        return (
            audio_io.Audio(np.full_like(loaded.samples, 0.3), loaded.sr, loaded.path),
            np.full(2048, 0.1, dtype=np.float32),
        )

    monkeypatch.setattr(vocals, "isolate", counted)

    first, first_vocal = vocals.isolate_cached(loaded, keep="instrumental")
    second, second_vocal = vocals.isolate_cached(loaded, keep="instrumental")

    assert calls == ["instrumental"]  # separated exactly once
    assert second.sr == first.sr
    assert second.n_samples == first.n_samples
    assert second.n_channels == first.n_channels
    assert np.allclose(second.samples, first.samples, atol=1e-4)
    assert np.allclose(second_vocal, first_vocal, atol=1e-4)


def test_isolate_cache_separates_per_mode(track, tmp_path, monkeypatch):
    """instrumental and vocals are different outputs and must not collide."""
    monkeypatch.setenv("REMOX_CACHE_DIR", str(tmp_path))
    loaded = audio_io.load(track)
    values = {"instrumental": 0.3, "vocals": 0.7}

    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            audio_io.Audio(
                np.full_like(loaded.samples, values[keep]), loaded.sr, loaded.path
            ),
            np.zeros(2048, dtype=np.float32),
        ),
    )

    inst, _ = vocals.isolate_cached(loaded, keep="instrumental")
    voc, _ = vocals.isolate_cached(loaded, keep="vocals")
    assert not np.allclose(inst.samples, voc.samples)


def test_isolate_cache_can_be_disabled(track, tmp_path, monkeypatch):
    monkeypatch.setenv("REMOX_CACHE_DIR", str(tmp_path))
    loaded = audio_io.load(track)
    calls = []

    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            calls.append(1)
            or audio_io.Audio(np.zeros_like(loaded.samples), loaded.sr, loaded.path),
            np.zeros(2048, dtype=np.float32),
        ),
    )
    vocals.isolate_cached(loaded, use_cache=False)
    vocals.isolate_cached(loaded, use_cache=False)
    assert len(calls) == 2


def test_corrupt_isolate_cache_is_recovered(track, tmp_path, monkeypatch):
    """A truncated cache entry must trigger re-separation, not a crash."""
    monkeypatch.setenv("REMOX_CACHE_DIR", str(tmp_path))
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            audio_io.Audio(np.zeros_like(loaded.samples), loaded.sr, loaded.path),
            np.zeros(2048, dtype=np.float32),
        ),
    )
    vocals.isolate_cached(loaded, keep="instrumental")
    for corrupt in (tmp_path / "isolated").glob("*.flac"):
        corrupt.write_bytes(b"not audio")

    result, _ = vocals.isolate_cached(loaded, keep="instrumental")
    assert result.n_samples == loaded.n_samples


def test_cli_keep_isolated_writes_full_length(track, tmp_path, monkeypatch, capsys):
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            audio_io.Audio(np.full_like(loaded.samples, 0.4), loaded.sr, loaded.path),
            np.zeros(2048, dtype=np.float32),
        ),
    )
    out = tmp_path / "cut.wav"
    code = main(
        [str(track), "-d", "30", "--isolate", "instrumental", "-o", str(out),
         "--keep-isolated", "--vocals", "off", "--no-vocal-cache"]
    )
    assert code == 0

    kept = tmp_path / "cut.instrumental.mp3"
    assert kept.exists()
    data, sr = sf.read(str(kept))
    # The kept file is the whole track, not the 30s cut.
    assert len(data) / sr == pytest.approx(loaded.duration, abs=0.2)
    assert sf.read(str(out))[0].shape[0] / sr == pytest.approx(30.0, abs=2.5)
    assert "isolated" in capsys.readouterr().out


def test_cli_keep_isolated_explicit_path(track, tmp_path, monkeypatch):
    loaded = audio_io.load(track)
    monkeypatch.setattr(
        vocals,
        "isolate",
        lambda t, keep="instrumental", model="htdemucs", device=None: (
            audio_io.Audio(np.zeros_like(loaded.samples), loaded.sr, loaded.path),
            np.zeros(2048, dtype=np.float32),
        ),
    )
    kept = tmp_path / "backing.flac"
    assert (
        main([str(track), "-d", "30", "--isolate", "instrumental",
              "-o", str(tmp_path / "c.wav"), "--keep-isolated", str(kept),
              "--vocals", "off", "--no-vocal-cache", "--quiet"])
        == 0
    )
    assert kept.exists()


def test_keep_isolated_ignored_without_isolation(track, tmp_path):
    """Asking to keep a stem that was never produced must not crash."""
    out = tmp_path / "plain.wav"
    assert (
        main([str(track), "-d", "30", "-o", str(out), "--keep-isolated",
              "--vocals", "off", "--quiet"])
        == 0
    )
    assert out.exists()
    assert not list(tmp_path.glob("*.instrumental.*"))


# --------------------------------------------------------------------------- cli


@pytest.mark.parametrize(
    "text,expected",
    [
        ("90", 90.0),
        ("90s", 90.0),
        ("1m30s", 90.0),
        ("1:30", 90.0),
        ("1:02:30", 3750.0),
        ("2m", 120.0),
        ("30.5", 30.5),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "--"])
def test_parse_duration_rejects_garbage(text):
    with pytest.raises(Exception):
        parse_duration(text)


def test_cli_writes_output(track, tmp_path, capsys):
    out = tmp_path / "out.wav"
    code = main([str(track), "-d", "1:00", "-o", str(out), "--vocals", "off"])
    assert code == 0
    assert out.exists()
    written, sr = sf.read(str(out))
    assert len(written) / sr == pytest.approx(60.0, abs=2.5)
    assert "edits" in capsys.readouterr().out


def test_cli_reports_error_for_missing_file(tmp_path, capsys):
    code = main([str(tmp_path / "nope.wav"), "-d", "30", "--vocals", "off"])
    assert code == 1
    assert "remox:" in capsys.readouterr().err


def test_cli_default_output_path(track, tmp_path, monkeypatch):
    copy = tmp_path / "song.wav"
    copy.write_bytes(track.read_bytes())
    monkeypatch.chdir(tmp_path)
    assert main([str(copy), "-d", "45", "--quiet", "--vocals", "off"]) == 0
    assert (tmp_path / "output" / "song.remox.wav").exists()


def test_cli_writes_variants_to_a_folder(track, tmp_path, capsys):
    out = tmp_path / "versions"
    code = main(
        [str(track), "-d", "30", "-n", "3", "--variants-dir", str(out), "--vocals", "off"]
    )
    assert code == 0
    assert out.is_dir()
    wavs = sorted(out.glob("*.wav"))
    assert len(wavs) == 3
    assert (out / "manifest.json").exists()

    report = capsys.readouterr().out
    assert "variants   3 of 3 requested" in report
    assert "balanced" in report


def test_cli_variants_default_folder_name(track, tmp_path, monkeypatch):
    copy = tmp_path / "song.wav"
    copy.write_bytes(track.read_bytes())
    monkeypatch.chdir(tmp_path)
    assert main([str(copy), "-d", "30", "-n", "2", "--vocals", "off", "--quiet"]) == 0
    assert (tmp_path / "output" / "song_variants").is_dir()


def test_cli_variants_reports_shortfall(track, tmp_path, capsys):
    out = tmp_path / "many"
    assert (
        main([str(track), "-d", "30", "-n", "150", "--variants-dir", str(out), "--vocals", "off"])
        == 0
    )
    report = capsys.readouterr().out
    assert "of 150 requested" in report
    assert "only" in report and "distinct edits" in report


def test_cli_rejects_zero_variants(track):
    with pytest.raises(SystemExit):
        main([str(track), "-d", "30", "-n", "0"])


def test_cli_lists_presets(capsys):
    assert main(["--list-presets"]) == 0
    out = capsys.readouterr().out
    assert "balanced" in out and "minimal" in out
    assert len(out.strip().splitlines()) == len(variants.PRESETS)

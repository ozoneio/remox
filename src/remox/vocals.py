"""Vocal activity detection via source separation.

The most audible artefact in an automatic remix is a cut that lands in the
middle of a sung phrase. Chroma, timbre and loudness features are all blind to
this - a splice can be perfectly beat-aligned and harmonically clean while
still chopping a word in half.

This module isolates the vocal stem with Demucs, reduces it to a per-boundary
"is a phrase crossing here" score, and hands that to the splice search as an
extra cost. Separation is expensive, so envelopes are cached on disk keyed by
file content.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter

from remox.audio import ANALYSIS_SR, Audio
from remox.beats import HOP_LENGTH, BeatGrid
from remox.cache import cache_dir

#: Half-width of the window examined either side of a cut, in seconds.
#: Roughly one syllable - long enough to see a phrase continuing, short enough
#: not to blur across a genuine gap between lines.
BOUNDARY_WINDOW = 0.15

#: Frames quieter than (stem peak - this) are treated as silence rather than
#: quiet singing. Guards against separation bleed being read as vocal activity.
STEM_GATE_DB = 45.0

#: A vocal stem this far below the mix level means the track is instrumental
#: and the stem contains nothing but separation residue.
INSTRUMENTAL_DB = 35.0

#: Fraction of the track that must be voiced before we believe there are vocals.
PRESENCE_FRACTION = 0.02


class SeparationUnavailable(RuntimeError):
    """Raised when vocal separation is requested but Demucs is not installed."""


@dataclass
class VocalActivity:
    """Per-boundary vocal continuity for a track.

    Attributes:
        crossing: length ``n_beats + 1``. ``crossing[b]`` is high when a vocal
            phrase spans the boundary between beat ``b - 1`` and beat ``b``,
            i.e. cutting there would chop a word.
        envelope: frame-rate activity in ``[0, 1]``, kept for inspection.
        present: False when the track appears to be instrumental, in which case
            ``crossing`` is all zeros and the vocal cost is a no-op.
    """

    crossing: np.ndarray
    envelope: np.ndarray
    present: bool

    @property
    def voiced_fraction(self) -> float:
        """Share of the track where vocals are active."""
        return float((self.envelope > 0.5).mean()) if self.envelope.size else 0.0


def _fingerprint(path: Path, model: str) -> str:
    """Content hash of the source file, so cache entries survive renames."""
    digest = hashlib.sha256()
    digest.update(model.encode())
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def _set_hub_offline(enabled: bool) -> None:
    """Toggle HuggingFace Hub offline mode.

    The env var is read once at ``huggingface_hub`` import time, so the
    already-imported constant is patched too when the module is loaded.
    """
    os.environ["HF_HUB_OFFLINE"] = "1" if enabled else "0"
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None:
        hub.HF_HUB_OFFLINE = enabled


def _load_pretrained(model_name: str):
    """Load Demucs weights, preferring the local cache.

    ``huggingface_hub`` contacts the Hub on every load even when the weights
    are already cached. That costs about half a second per run and prints an
    unauthenticated-request warning that has nothing to do with remox and
    confuses anyone who did not realise a model was involved.

    Trying offline first avoids both. If the weights genuinely are not cached
    the offline attempt fails and we fall back to a normal online fetch, so a
    first run still works.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    previous = os.environ.get("HF_HUB_OFFLINE")

    _set_hub_offline(True)
    try:
        from demucs.pretrained import get_model

        return get_model(model_name)
    except Exception:
        pass  # Not cached (or offline lookup unsupported); fetch normally.
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous

    _set_hub_offline(False)
    from demucs.pretrained import get_model

    return get_model(model_name)


def _apply(track: Audio, model_name: str, device: str | None):
    """Run Demucs and return ``(stems, sources, model_sample_rate)``.

    ``stems`` is a torch tensor of shape ``(n_sources, channels, n_samples)`` at
    the model's own sample rate. Kept as a tensor so callers can sum sources
    before converting, which avoids materialising four full-rate stereo arrays
    when only their sum is wanted.
    """
    try:
        import torch
        from demucs.apply import apply_model
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise SeparationUnavailable(
            "vocal separation needs Demucs; install with: pip install 'remox[vocals]'"
        ) from exc

    model = _load_pretrained(model_name)
    if "vocals" not in model.sources:
        raise SeparationUnavailable(f"model {model_name!r} has no vocals stem")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Demucs expects stereo at its own sample rate.
    samples = track.samples
    if track.sr != model.samplerate:
        samples = librosa.resample(samples, orig_sr=track.sr, target_sr=model.samplerate)
    if samples.shape[0] == 1:
        samples = np.repeat(samples, model.audio_channels, axis=0)
    elif samples.shape[0] > model.audio_channels:
        samples = samples[: model.audio_channels]

    wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))

    # Demucs is trained on standardised input; undo the scaling afterwards.
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std().clamp_min(1e-8)
    wav = (wav - mean) / std

    with torch.no_grad():
        stems = apply_model(model, wav[None], device=device, split=True, progress=False)
    stems = stems[0] * std + mean

    return stems, list(model.sources), int(model.samplerate)


def _to_track_rate(signal: np.ndarray, from_sr: int, track: Audio) -> np.ndarray:
    """Resample a separated stem back to the source rate, layout and length.

    Resampling rarely lands on exactly the original sample count, and a stem
    that is a few samples short would shift every beat time computed against
    the original grid. Pad or trim to make the mapping exact.
    """
    if from_sr != track.sr:
        signal = librosa.resample(signal, orig_sr=from_sr, target_sr=track.sr)

    if signal.shape[0] > track.n_channels:
        signal = signal.mean(axis=0, keepdims=True)
    elif signal.shape[0] < track.n_channels:
        signal = np.repeat(signal, track.n_channels, axis=0)

    if signal.shape[1] > track.n_samples:
        signal = signal[:, : track.n_samples]
    elif signal.shape[1] < track.n_samples:
        pad = track.n_samples - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad)))

    return np.ascontiguousarray(signal, dtype=np.float32)


def isolate(
    track: Audio,
    keep: str = "instrumental",
    model: str = "htdemucs",
    device: str | None = None,
) -> tuple[Audio, np.ndarray]:
    """Split a track into the requested part and the vocal analysis signal.

    Args:
        track: the source audio.
        keep: ``"instrumental"`` sums every stem except vocals; ``"vocals"``
            keeps only the vocal stem (an acapella).
        model: Demucs pretrained model name.
        device: torch device; auto-selects CUDA when available.

    Returns:
        ``(isolated, vocal_signal)`` where ``isolated`` is a new :class:`Audio`
        at the source rate and channel count, and ``vocal_signal`` is the mono
        vocal stem at :data:`ANALYSIS_SR` - returned alongside so callers get
        the phrase-crossing envelope without a second separation pass.

    Raises:
        SeparationUnavailable: if Demucs is not installed.
        ValueError: for an unknown ``keep`` mode.
    """
    if keep not in ("instrumental", "vocals"):
        raise ValueError(f"unknown isolate mode: {keep!r}")

    stems, sources, model_sr = _apply(track, model, device)
    vocal_index = sources.index("vocals")

    if keep == "vocals":
        wanted = stems[vocal_index]
    else:
        # Sum in tensor form: cheaper than converting four stems separately.
        wanted = sum(
            stems[i] for i in range(len(sources)) if i != vocal_index
        )

    isolated = Audio(
        samples=_to_track_rate(wanted.cpu().numpy(), model_sr, track),
        sr=track.sr,
        path=track.path,
    )

    vocal_mono = stems[vocal_index].cpu().numpy().mean(axis=0)
    if model_sr != ANALYSIS_SR:
        vocal_mono = librosa.resample(vocal_mono, orig_sr=model_sr, target_sr=ANALYSIS_SR)

    return isolated, vocal_mono


def isolate_cached(
    track: Audio,
    keep: str = "instrumental",
    model: str = "htdemucs",
    device: str | None = None,
    use_cache: bool = True,
) -> tuple[Audio, np.ndarray]:
    """:func:`isolate` with the result persisted between runs.

    Separation costs tens of seconds, so the isolated audio and the vocal
    analysis signal are both stored as FLAC. FLAC rather than MP3 because a
    cache entry gets re-read and re-cut repeatedly; a lossy round trip would
    compound artefacts every time.
    """
    if not use_cache:
        return isolate(track, keep=keep, model=model, device=device)

    stem = _fingerprint(track.path, f"{model}:{keep}")
    directory = cache_dir("isolated")
    isolated_path = directory / f"{stem}.{keep}.flac"
    vocal_path = directory / f"{stem}.vocal.flac"

    if isolated_path.exists() and vocal_path.exists():
        try:
            samples, sr = sf.read(str(isolated_path), dtype="float32", always_2d=True)
            vocal, _ = sf.read(str(vocal_path), dtype="float32")
            return (
                Audio(samples=samples.T, sr=int(sr), path=track.path),
                np.asarray(vocal, dtype=np.float32),
            )
        except (OSError, RuntimeError):
            pass  # Corrupt entry; separate again and overwrite.

    isolated, vocal_signal = isolate(track, keep=keep, model=model, device=device)

    try:
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(str(isolated_path), isolated.samples.T, isolated.sr)
        sf.write(str(vocal_path), vocal_signal, ANALYSIS_SR)
    except (OSError, RuntimeError):
        pass  # A cache write failure must not fail the run.

    return isolated, vocal_signal


def _separate(track: Audio, model_name: str, device: str | None) -> np.ndarray:
    """Return the mono vocal stem, resampled to :data:`ANALYSIS_SR`."""
    stems, sources, model_sr = _apply(track, model_name, device)
    vocals = stems[sources.index("vocals")].cpu().numpy().mean(axis=0)
    if model_sr != ANALYSIS_SR:
        vocals = librosa.resample(vocals, orig_sr=model_sr, target_sr=ANALYSIS_SR)
    return vocals


def _envelope(vocals: np.ndarray, mix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Reduce a vocal stem to a gated activity envelope in ``[0, 1]``.

    The gate is anchored to both the stem's own peak and the mix level. Using
    the stem alone would rescale pure separation residue on an instrumental
    track into apparent full-scale singing.
    """
    frames = librosa.feature.rms(y=vocals, hop_length=HOP_LENGTH)[0]
    stem_db = librosa.amplitude_to_db(np.maximum(frames, 1e-10), ref=1.0)

    mix_rms = float(np.sqrt(np.mean(mix**2))) if mix.size else 0.0
    mix_db = 20 * np.log10(max(mix_rms, 1e-10))

    top = float(np.percentile(stem_db, 95))
    if top < mix_db - INSTRUMENTAL_DB:
        return np.zeros_like(stem_db), False

    floor = max(top - STEM_GATE_DB, mix_db - INSTRUMENTAL_DB)
    if top - floor < 1e-6:
        return np.zeros_like(stem_db), False

    activity = np.clip((stem_db - floor) / (top - floor), 0.0, 1.0)
    # Smoothstep pushes ambiguous mid values towards 0 or 1 so the boundary
    # score reflects a decision rather than a gradient.
    activity = activity * activity * (3.0 - 2.0 * activity)
    # ~120 ms median filter removes single-frame flicker between syllables.
    activity = median_filter(activity, size=5, mode="nearest")

    present = bool((activity > 0.5).mean() >= PRESENCE_FRACTION)
    return activity, present


def crossing_scores(envelope: np.ndarray, grid: BeatGrid, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Score each beat boundary by how strongly a vocal phrase spans it.

    A boundary is only dangerous when vocals are active on *both* sides, hence
    the min: singing that stops right at the boundary is a safe place to cut.
    """
    n_frames = len(envelope)
    half = max(1, int(round(BOUNDARY_WINDOW * sr / HOP_LENGTH)))
    scores = np.zeros(len(grid.times), dtype=np.float64)

    for b, time in enumerate(grid.times):
        centre = int(round(time * sr / HOP_LENGTH))
        before = envelope[max(0, centre - half) : centre]
        after = envelope[centre : min(n_frames, centre + half)]
        if before.size == 0 or after.size == 0:
            continue  # Track edges are never splice points anyway.
        scores[b] = min(float(before.mean()), float(after.mean()))

    return scores


def from_stem(
    vocal_signal: np.ndarray, mix: np.ndarray, grid: BeatGrid
) -> VocalActivity:
    """Build phrase-crossing scores from an already-separated vocal stem.

    Lets callers that have run separation for another reason - isolating an
    instrumental, say - get the splice-guard data without paying for a second
    pass through the model.

    Args:
        vocal_signal: mono vocal stem at :data:`ANALYSIS_SR`.
        mix: the mono mix at :data:`ANALYSIS_SR`, used to gauge whether the
            stem is real singing or separation residue.
        grid: beat grid of the track.
    """
    envelope, present = _envelope(vocal_signal, mix)
    crossing = (
        crossing_scores(envelope, grid) if present else np.zeros(len(grid.times))
    )
    return VocalActivity(crossing=crossing, envelope=envelope, present=present)


def analyse(
    track: Audio,
    grid: BeatGrid,
    model: str = "htdemucs",
    device: str | None = None,
    use_cache: bool = True,
) -> VocalActivity:
    """Separate vocals and score every beat boundary for phrase continuity.

    Args:
        track: the source audio.
        grid: its beat grid.
        model: Demucs pretrained model name.
        device: torch device; auto-selects CUDA when available.
        use_cache: reuse a previously separated envelope for this file.

    Raises:
        SeparationUnavailable: if Demucs is not installed.
    """
    envelope: np.ndarray | None = None
    present = False
    path = cache_dir("stems") / f"{_fingerprint(track.path, model)}.npz" if use_cache else None

    if path is not None and path.exists():
        try:
            cached = np.load(path)
            envelope, present = cached["envelope"], bool(cached["present"])
        except (OSError, KeyError, ValueError):
            envelope = None  # Corrupt cache entry; separate again.

    if envelope is None:
        vocals = _separate(track, model, device)
        envelope, present = _envelope(vocals, track.analysis_signal())
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, envelope=envelope, present=present)

    crossing = crossing_scores(envelope, grid) if present else np.zeros(len(grid.times))
    return VocalActivity(crossing=crossing, envelope=envelope, present=present)

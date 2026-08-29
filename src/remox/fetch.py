"""Fetch audio from a media URL via yt-dlp.

remox retargets music to a duration, and the material people retarget usually
starts life as a URL rather than a local file. This module turns one into the
other and records where it came from.

Provenance is written to a JSON sidecar next to every download. The tool cannot
tell a Creative Commons track from a commercial release, so it keeps the title,
uploader, licence field and source URL instead of producing an anonymous file.
Whether a given download is permitted is the operator's call, not this module's.

Downloads land in ``~/Downloads/Music`` by default. They are the user's own
media rather than a regenerable artefact, so they deliberately do not live in
the cache directory - clearing the cache must not delete them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

#: Formats worth requesting. mp3 is the default because libsndfile reads it
#: directly; wav avoids a second lossy encode and is better for analysis.
AUDIO_FORMATS = ("mp3", "wav", "flac", "m4a", "opus")

#: Windows known-folder GUID for Downloads.
_DOWNLOADS_GUID = "{374DE290-123F-4565-9164-39C4925E467B}"

_URL = re.compile(r"^https?://", re.IGNORECASE)
_UNSAFE = re.compile(r"[^\w\-. ]+")

#: URL shapes whose video id can be read without asking the network.
_ID_PATTERNS = (
    re.compile(r"[?&]v=([\w-]{6,})"),
    re.compile(r"youtu\.be/([\w-]{6,})"),
    re.compile(r"/shorts/([\w-]{6,})"),
    re.compile(r"/embed/([\w-]{6,})"),
    re.compile(r"/live/([\w-]{6,})"),
)


def id_from_url(url: str) -> str | None:
    """Extract a video id from a URL without contacting the network.

    Lets a cached download be reused with no requests at all. Returns None for
    URL shapes we cannot read locally, in which case the caller falls back to
    asking the extractor.
    """
    for pattern in _ID_PATTERNS:
        found = pattern.search(url)
        if found:
            return found.group(1)
    return None


class FetchError(RuntimeError):
    """Raised when a URL cannot be downloaded or converted."""


@dataclass
class Download:
    """A downloaded track and where it came from.

    Attributes:
        path: the converted audio file on disk.
        video_id: extractor-specific identifier, used as the cache key.
        title: track title as reported by the source.
        uploader: channel or account that published it.
        url: canonical page URL.
        duration: length in seconds as reported by the source, if known.
        license: licence string when the source exposes one. YouTube only
            populates this for Creative Commons uploads, so ``None`` means
            "unspecified", not "unrestricted".
    """

    path: Path
    video_id: str
    title: str
    uploader: str
    url: str
    duration: float | None = None
    license: str | None = None

    def attribution(self) -> str:
        """One-line human-readable credit for logs and reports."""
        who = f" - {self.uploader}" if self.uploader else ""
        lic = f" [{self.license}]" if self.license else ""
        return f"{self.title}{who}{lic}"


def _windows_downloads() -> Path | None:
    """Resolve the real Downloads folder from the Windows shell registry.

    Downloads is frequently redirected (OneDrive, or a second drive). Reading
    the known-folder entry avoids creating a stray ``C:\\Users\\x\\Downloads``
    on machines where the real one lives elsewhere. Returns None on any
    failure so the caller can fall back to the literal path.
    """
    if os.name != "nt":
        return None
    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            raw, _ = winreg.QueryValueEx(key, _DOWNLOADS_GUID)
        expanded = os.path.expandvars(raw)
        return Path(expanded) if expanded else None
    except (ImportError, OSError, ValueError):
        return None


def download_dir() -> Path:
    """Default location for downloaded audio: ``~/Downloads/Music``.

    Downloads are kept here rather than in the cache because they are the
    user's own media, not a regenerable artefact - clearing the cache should
    not delete them.

    ``REMOX_DOWNLOAD_DIR`` overrides the whole path.
    """
    override = os.environ.get("REMOX_DOWNLOAD_DIR")
    if override:
        return Path(override)
    base = _windows_downloads() or (Path.home() / "Downloads")
    return base / "Music"


def is_url(text: str) -> bool:
    """True when ``text`` looks like an http(s) URL rather than a file path."""
    return bool(_URL.match(text.strip()))


def _slug(text: str, fallback: str = "track") -> str:
    """Filesystem-safe version of a title, for default output names.

    Unsafe characters become spaces rather than being deleted, so "A / B"
    yields "A_B" instead of "A__B".
    """
    cleaned = _UNSAFE.sub(" ", text)
    cleaned = "_".join(cleaned.split())
    return cleaned[:80].strip("_.") or fallback


def resolve_ffmpeg() -> str | None:
    """Locate an ffmpeg binary.

    Prefers one already on PATH, then falls back to the static build shipped by
    ``imageio-ffmpeg`` so users do not need a system-wide install. Returns None
    if neither is available.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - binary missing for this platform
        return None


def _load_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:
        raise FetchError(
            "downloading from a URL needs yt-dlp; install with: pip install 'remox[youtube]'"
        ) from exc
    return yt_dlp


def _sidecar(download: Download) -> None:
    """Write provenance next to the audio file."""
    record = asdict(download)
    record["path"] = download.path.name
    target = download.path.with_suffix(".json")
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def _cached(directory: Path, video_id: str, audio_format: str) -> Download | None:
    """Return a previous download of the same video in the same format."""
    audio = directory / f"{video_id}.{audio_format}"
    meta = audio.with_suffix(".json")
    if not (audio.exists() and meta.exists()):
        return None
    try:
        record = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    record.pop("path", None)
    try:
        return Download(path=audio, **record)
    except TypeError:  # Sidecar from an older version; re-download.
        return None


def fetch(
    url: str,
    dest_dir: str | Path | None = None,
    audio_format: str = "mp3",
    quality: str = "192",
    use_cache: bool = True,
    quiet: bool = True,
) -> Download:
    """Download the audio track behind ``url``.

    Args:
        url: page URL. Playlists are rejected rather than silently expanded.
        dest_dir: where to write; defaults to :func:`download_dir`
            (``~/Downloads/Music``).
        audio_format: container/codec to convert to. See :data:`AUDIO_FORMATS`.
        quality: bitrate in kbps for lossy formats.
        use_cache: reuse a previous download of the same video and format.
        quiet: suppress yt-dlp's own progress output.

    Raises:
        FetchError: if yt-dlp is missing, ffmpeg is needed but absent, the URL
            cannot be resolved, or conversion fails.
    """
    if audio_format not in AUDIO_FORMATS:
        raise FetchError(
            f"unsupported audio format {audio_format!r}; choose from {', '.join(AUDIO_FORMATS)}"
        )

    yt_dlp = _load_yt_dlp()
    directory = Path(dest_dir) if dest_dir else download_dir()
    directory.mkdir(parents=True, exist_ok=True)

    # Check the cache before touching the network. The video id is usually
    # readable straight from the URL, so a track already on disk costs no
    # requests at all and works offline.
    if use_cache:
        local_id = id_from_url(url)
        if local_id:
            hit = _cached(directory, local_id, audio_format)
            if hit is not None:
                return hit

    ffmpeg = resolve_ffmpeg()
    if ffmpeg is None:
        raise FetchError(
            "ffmpeg is required to convert downloaded audio; install it, or run: "
            "pip install imageio-ffmpeg"
        )

    # Resolve metadata first so the cache can be checked before transferring
    # any media.
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(directory / "%(id)s.%(ext)s"),
        "ffmpeg_location": ffmpeg,
        "noplaylist": True,
        "quiet": quiet,
        "no_warnings": quiet,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": quality,
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(dict(options, skip_download=True)) as probe:
            info = probe.extract_info(url, download=False)
    except Exception as exc:
        raise FetchError(f"could not read {url}: {exc}") from exc

    if info is None:
        raise FetchError(f"no media found at {url}")
    if info.get("_type") == "playlist" or "entries" in info:
        raise FetchError("URL points at a playlist; pass a single track URL")

    video_id = str(info.get("id") or "")
    if use_cache and video_id:
        hit = _cached(directory, video_id, audio_format)
        if hit is not None:
            return hit

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except Exception as exc:
        raise FetchError(f"download failed for {url}: {exc}") from exc

    path = _downloaded_path(info, directory, video_id, audio_format)
    download = Download(
        path=path,
        video_id=video_id,
        title=str(info.get("title") or video_id or "unknown"),
        uploader=str(info.get("uploader") or info.get("channel") or ""),
        url=str(info.get("webpage_url") or url),
        duration=float(info["duration"]) if info.get("duration") else None,
        license=info.get("license") or None,
    )
    _sidecar(download)
    return download


def _downloaded_path(info: dict, directory: Path, video_id: str, audio_format: str) -> Path:
    """Work out where the post-processor left the converted file.

    yt-dlp reports the final path in ``requested_downloads`` after
    post-processing, but that key is not guaranteed across versions, so fall
    back to the predictable output template.
    """
    for entry in info.get("requested_downloads") or []:
        candidate = entry.get("filepath")
        if candidate and Path(candidate).exists():
            return Path(candidate)

    expected = directory / f"{video_id}.{audio_format}"
    if expected.exists():
        return expected

    matches = sorted(directory.glob(f"{video_id}.*"))
    audio = [m for m in matches if m.suffix.lstrip(".") in AUDIO_FORMATS]
    if audio:
        return audio[0]

    raise FetchError(
        f"download completed but no {audio_format} file was produced for {video_id}"
    )


def default_output_name(download: Download, suffix: str = ".remox.wav") -> Path:
    """Output filename derived from the track title."""
    return Path(f"{_slug(download.title, download.video_id)}{suffix}")

# Remøx

Re-edit a music track to an arbitrary target duration, on the beat.

Give remox a song and a length. It analyses the beat grid, finds places where
the music can be spliced without sounding broken, and solves for the set of
cuts that lands closest to your target at the lowest cost. Intro and outro are
preserved; the edits happen in between.

Conceptually the same job as Adobe Audition's Remix / Premiere's Remix Tool.

```
.\remox.exe song.mp3 -d 30s
```

```
source     1:45.00
output     0:29.00  (target 0:30.00, -1.00s)
tempo      117.5 BPM, 4/4, 209 beats
edits      1 (align: bar)
cut cost   mean 0.16, worst 0.16
vocals     none detected (instrumental)
  seg  1  0:00.02 -> 0:14.02  (28 beats)
  seg  2  1:30.02 -> 1:44.54  (29 beats)
written    output/song.remox.wav
```

## Install

### Windows

Download the Windows executable from the [latest GitHub release](../../releases/latest).
It is self-contained, so Python and pip are not required. You can rename it to
`remox.exe` and run it from PowerShell:

```powershell
.\remox.exe song.mp3 -d 30s
```

The executable includes local audio processing and URL download support. Vocal
isolation is not bundled because its Torch and Demucs dependencies are very
large; use the Python development installation below when you need that feature.

## Usage

Generated remixes, isolated stems, and variant sets are written beneath
`output/` by default. Use `--output` or `--variants-dir` to choose another
location explicitly.

```powershell
.\remox.exe song.wav -d 90              # 90 seconds
.\remox.exe song.wav -d 1:30            # same, clock format
.\remox.exe song.wav -d 2m15s           # same idea
.\remox.exe song.wav -d 150             # lengthen a 105s track by repeating material
```

Shortening and lengthening use the same machinery: a forward jump skips
material, a backward jump repeats it.

### Removing vocals

```powershell
.\remox.exe song.mp3 --isolate instrumental                 # full-length instrumental
.\remox.exe song.mp3 --isolate vocals                       # acapella
.\remox.exe song.mp3 -d 30s --isolate instrumental          # 30s instrumental cut
.\remox.exe song.mp3 -d 30s --isolate instrumental `
      --keep-isolated                                 # cut + the full instrumental
```

Isolation happens **before** the cut: the whole track is separated, then the
edit is planned and rendered from the isolated audio. `--keep-isolated` saves
that full-length version alongside the remix (`<output>.instrumental.mp3` by
default, or give it a path — the extension picks the format).

Separated results are cached as FLAC keyed on file content, so re-running the
same track skips the model entirely: ~26 s the first time, ~0.1 s after.
Lossless rather than MP3 because a cache entry is re-read and re-cut
repeatedly, and a lossy round trip would compound artefacts each time. Disable
with `--no-vocal-cache`.

Demucs already separates four stems to compute the vocal envelope; the
instrumental is the sum of everything except the vocal stem, so it costs no
extra model pass. Omit `-d` to export the isolated track without retargeting.

Analysis always runs on the **full mix** — beats track more reliably with the
vocal present — while the output is cut from the isolated audio. The isolated
stem is padded or trimmed to the source length so the beat grid still indexes
it exactly.

Needs the `vocals` extra. Quality is whatever `htdemucs` gives you: good on
well-produced material, with audible bleed and artefacts on dense mixes,
heavy reverb, or backing vocals.

### Choosing where it opens

By default an edit starts at beat one. For a short cut that is rarely what you
want — the opening bars of a track are usually its least characteristic part.

```powershell
.\remox.exe song.wav -d 30s --start auto      # open at a chorus or drop
.\remox.exe song.wav -d 30s --start 1:12      # open at a specific point
.\remox.exe song.wav -d 30s                   # open at the beginning (default)
```

`--start auto` scores every beat as a candidate opening on four cues: novelty
(how much the music changes there, so openings land on section boundaries),
energy (loud, fully-arranged sections), vocal safety (never open mid-word), and
phrase alignment. The solver then treats the opening as part of the same
shortest-path problem as the cuts, so it trades entry quality against splice
quality rather than choosing them independently.

Anything other than `--start beginning` drops the intro, and the output length
stays exact because that is decided before solving rather than depending on
which opening wins. Tune with `--entry-weight` (how much a good opening is
worth relative to a good cut) and `--entry-align`.

### Downloading

```powershell
.\remox.exe "https://youtu.be/VIDEO_ID" -d 30s
.\remox.exe "https://youtu.be/VIDEO_ID" --fetch-only --audio-format wav
```

Downloads go to `~/Downloads/Music` by default. They are your own media rather
than a regenerable artefact, so they deliberately do not live in the cache —
clearing the cache will not delete them. Override with `--download-dir` or the
`REMOX_DOWNLOAD_DIR` environment variable. On Windows the real Downloads
location is read from the shell registry, so a OneDrive-redirected folder is
followed rather than creating a stray `C:\Users\you\Downloads`.

A track already in the download folder is reused without any network request —
the video id is read straight from the URL, so re-running an identical command
is instant and works offline. Force a fresh download with `--no-download-cache`.

Every download writes a JSON sidecar recording title, uploader, canonical URL,
duration and the licence field. See [Licensing of source material](#licensing-of-source-material).

### Tuning

| Flag | Default | Effect |
|---|---|---|
| `--start WHERE` | `beginning` | Where the edit opens: `beginning`, `auto`, or a time like `1:12`. |
| `--entry-weight N` | `4.0` | How much a strong opening is worth relative to a clean cut. |
| `--entry-align {phrase,bar,beat}` | `phrase` | Metrical restriction on the opening point. |
| `--align {phrase,bar,beat}` | `bar` | Where cuts may land. `phrase` is the most musical but has the fewest options. |
| `--edits N` | `1.0` | Cost per edit. Raise for fewer, larger cuts. |
| `--tolerance SECONDS` | `2.0` | How far from the target is acceptable. |
| `--min-jump-bars N` | `2` | Minimum span of a single edit. Stops stutter. |
| `--context N` | `4` | Beats of lead-in compared when scoring a cut. |
| `--vocals {auto,on,off}` | `auto` | Vocal-aware splicing. `auto` skips it if Demucs is absent. |
| `--isolate {none,instrumental,vocals}` | `none` | Remove the vocals, or keep only them. |
| `--vocal-veto 0..1` | off | Hard-reject cuts through a sung phrase. |
| `--fade-ms N` | `40` | Crossfade length at each splice. |
| `--no-intro` / `--no-outro` | off | Drop the untouched head/tail. |

`.\remox.exe --help` lists everything.

### As a library

```python
from remox import remix, RemixOptions

result = remix("song.mp3", 30.0, RemixOptions(align="bar", vocal_veto=0.3))
print(result.plan.n_edits, result.duration)
result.save("spot.wav")
```

## How it works

**1. Beat tracking** (`beats.py`) — spectral-flux onset envelope, tempo by
autocorrelation, beats by dynamic programming. Downbeats are estimated by
scoring each candidate bar phase on onset accent plus chroma flux, since chord
changes tend to land on downbeats.

**2. Per-beat features** (`features.py`) — chroma (harmony), MFCCs (timbre) and
RMS (energy), each block z-scored and normalised to unit average norm so the
weights are directly comparable.

**3. Splice scoring** (`jumps.py`) — cost of cutting from beat `i` to beat `j`
is measured over a **4-beat context window**, comparing the run ending at `i`
against the run ending at `j-1`. Single-beat comparison is not enough: two
beats can be near-identical while the bars leading into them are unrelated,
which is exactly the cut that sounds wrong. Candidates must share bar phase and
clear a minimum span.

**4. Path solving** (`path.py`) — Dijkstra over states of
`(current_beat, beats_emitted)`. Advancing one beat is free; taking a splice
costs its score plus `--edits`. The search returns the cheapest path that
reaches the last beat with a beat count matching the target.

Counting **beats** rather than seconds keeps the state space integral and
small — quantising time into 100 ms buckets would give millions of states on a
5-minute track. Exact second-level duration is computed afterwards from real
beat lengths.

**5. Rendering** (`render.py`) — equal-power crossfade placed *before* each
junction, so the incoming segment still starts exactly on its own beat
boundary and its transient stays sharp. Centring the fade on the junction
smears the downbeat attack audibly.

### Vocal-aware splicing

The most audible artefact in an automatic remix is a cut landing mid-word.
Chroma, timbre and loudness are all blind to it — a splice can be perfectly
beat-aligned and harmonically clean while chopping a lyric in half.

With the `vocals` extra installed, remox separates the vocal stem with Demucs,
reduces it to a per-boundary "is a phrase crossing here" score, and feeds that
to the splice search as extra cost. Separated envelopes are cached by file
content hash.

Measured on the synthetic test fixture across 41 target durations from
20–100 s, using ground-truth phrase locations:

| | Off | On |
|---|---|---|
| Candidate splices cutting through a phrase | 29% | 0% |
| Plans that differed | — | 18 of 41 |
| …of which cut at a strictly safer point | — | 17 of 18 |

Worth being straight about the other 23: the plans were identical, because the
acoustic features had already picked a safe cut. The feature removes a tail of
bad outcomes rather than improving every track.

## Known limitations

- **Downbeat detection is heuristic.** The onset-accent + chroma-flux vote
  works on steady 4/4 and will misfire on syncopated, rubato or heavily live
  material. This is the biggest quality lever left; a neural tracker
  (`beat_this`, `allin1`) would be a drop-in replacement behind `BeatGrid`.
- **`--align phrase` often can't hit tight targets.** Four-bar granularity is
  8 s at 120 BPM, so a 2 s tolerance is frequently unsatisfiable. remox
  auto-relaxes to `bar` and tells you it did. A larger `--tolerance` keeps it.
- **Duration is approximate, not exact.** Output lands within `--tolerance`.
  There is no residual time-stretch, so remox cannot currently guarantee a
  frame-exact 30.00 s for broadcast.
- **Splice cost is relative, not absolute.** Candidates are filtered by
  percentile and normalised against their own median, so on a track with no
  good cuts anywhere remox returns the best of a bad lot without warning.
- **Free-tempo material analyses poorly.** Ambient, rubato and unmetered music
  has no usable beat grid; remox will refuse rather than guess.
- **Live YouTube downloads have no test coverage.** yt-dlp extractors break
  when sites change. If a download fails oddly, try `pip install -U yt-dlp`.

## Development

The source installation requires Python 3.10+. Choose the editable install
that matches the features you need:

```bash
pip install -e .                        # core: analysis + retargeting
pip install -e ".[youtube]"             # + download from a URL
pip install -e ".[vocals]"              # + vocal isolation and awareness
pip install -e ".[vocals,youtube]"      # everything
```

The core install needs no system dependencies. The `youtube` extra includes a
static FFmpeg binary through `imageio-ffmpeg`. The `vocals` extra adds roughly
500 MB of Torch plus a roughly 300 MB Demucs model downloaded on first use.

For a complete development and test environment on Windows:

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[vocals,youtube]" pytest
.venv/Scripts/python -m pytest tests -q
```

Build the standalone Windows release executable with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_release.ps1
```

The executable and its SHA-256 checksum are generated in `build/`. Pushing a
version tag such as `v0.1.0` runs the release workflow and uploads both files to
the corresponding GitHub release.

81 tests, ~22 s. No network access — yt-dlp is mocked and vocal separation is
driven by a ground-truth stem rather than the real model.

Generate the test fixture standalone:

```bash
python tools/make_test_track.py out.wav --vocals
```

A note on that fixture: its synthetic voice is **not** convincing enough for a
real separator (htdemucs routes it to `other`, not `vocals`), so the vocal
tests inject the known vocal bus instead of running Demucs. That tests remox;
running it through the model would test the model. The consequence is that the
separation step itself is only verified on instrumental input, where it
correctly reports "none detected".

### Layout

| File | Role |
|---|---|
| `audio.py` | Load/save, mono downmix, analysis-rate resampling |
| `beats.py` | Onset envelope, tempo, beat tracking, downbeat phase |
| `features.py` | Per-beat chroma + MFCC + RMS matrix |
| `jumps.py` | Context-windowed splice costs, metrical constraints |
| `path.py` | Constrained shortest path to the target |
| `entry.py` | Scoring where the edit should open |
| `render.py` | Equal-power crossfade splicing |
| `vocals.py` | Demucs separation, phrase-crossing scores, stem isolation |
| `fetch.py` | yt-dlp download with provenance sidecar |
| `cache.py` | Shared on-disk cache location |
| `pipeline.py` | Wiring, alignment auto-relaxation |
| `cli.py` | Argument parsing and reporting |

## Licensing of source material

remox is MIT licensed. **That says nothing about the music you feed it.**

Downloading from YouTube is against its Terms of Service, and most commercial
music there is not licensed for reuse or derivative editing. remox cannot tell
a Creative Commons track from a commercial release, so it records provenance
rather than producing anonymous files — but the decision is yours, not the
tool's.

Straightforwardly fine: your own recordings, YouTube Audio Library tracks,
Creative Commons and public-domain material, anything you have a licence for.

Note that a YouTube `license` field of `None` means *unspecified*, not
*unrestricted* — the field is only populated for Creative Commons uploads.

## Third-party licenses

| Dependency | License | Used for |
|---|---|---|
| librosa | ISC | Beat tracking, features |
| numpy, scipy | BSD-3-Clause | Numerics |
| soundfile / libsndfile | BSD-3-Clause / LGPL-2.1 | Audio I/O |
| demucs | MIT | Vocal separation (optional) |
| torch | BSD-3-Clause | Demucs runtime (optional) |
| yt-dlp | Unlicense | Downloading (optional) |
| imageio-ffmpeg | BSD-2-Clause | Bundled ffmpeg binary (optional) |

The ffmpeg binary shipped by `imageio-ffmpeg` is distributed under its own
terms (LGPL or GPL depending on the build) and is **not** covered by this
project's MIT license. If you redistribute remox with that binary bundled,
check ffmpeg's licensing for your situation.

## License

MIT — see [LICENSE](LICENSE).

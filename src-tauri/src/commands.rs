//! Native desktop boundary: dialogs grant input paths, cached WAVs grant playback paths.
//! The musical engine has no filesystem, Tauri, network, or mutable session dependency.

use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use serde::Serialize;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;
use uuid::Uuid;

use crate::{
    engine::{self, Analysis, Audio, RemixSettings},
    media, youtube,
};

#[derive(Debug, Serialize)]
pub struct CommandError {
    kind: &'static str,
    message: String,
}

fn command_error(kind: &'static str, message: String) -> CommandError {
    CommandError { kind, message }
}

struct ImportedTrack {
    id: String,
    source_path: PathBuf,
    audio: Arc<Audio>,
    analysis: Arc<Analysis>,
}

struct GeneratedRemix {
    id: String,
    path: PathBuf,
}

struct Session {
    track: Option<ImportedTrack>,
    remix: Option<GeneratedRemix>,
}

/// Connector state belongs only to the native window; analysis and planning stay pure.
pub struct DesktopSession {
    current: Mutex<Session>,
    busy: Arc<AtomicBool>,
    cache_id: String,
}

impl DesktopSession {
    pub(crate) fn new() -> Self {
        Self {
            current: Mutex::new(Session {
                track: None,
                remix: None,
            }),
            busy: Arc::new(AtomicBool::new(false)),
            cache_id: Uuid::new_v4().to_string(),
        }
    }

    fn begin(&self) -> Result<Operation, CommandError> {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                command_error(
                    "busy",
                    "An audio operation is already running; wait for it to finish".into(),
                )
            })?;
        Ok(Operation {
            busy: Arc::clone(&self.busy),
        })
    }
}

struct Operation {
    busy: Arc<AtomicBool>,
}

impl Drop for Operation {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::Release);
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackSummary {
    id: String,
    name: String,
    duration_seconds: f64,
    sample_rate: u32,
    channels: u16,
    tempo_bpm: f64,
    beat_count: usize,
    waveform: Vec<f32>,
    preview_path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentSummary {
    source_start_seconds: f64,
    source_end_seconds: f64,
    output_start_seconds: f64,
    output_end_seconds: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemixSummary {
    id: String,
    track_id: String,
    duration_seconds: f64,
    target_seconds: f64,
    tempo_bpm: f64,
    segments: Vec<SegmentSummary>,
    mean_transition_cost: f64,
    waveform: Vec<f32>,
    preview_path: String,
}

fn cache_path(app: &AppHandle, cache_id: &str, name: &str) -> Result<PathBuf, CommandError> {
    let root = app
        .path()
        .app_cache_dir()
        .map_err(|error| command_error("cache", format!("Cannot locate the app cache: {error}")))?
        .join("previews")
        .join(cache_id);
    std::fs::create_dir_all(&root).map_err(|error| {
        command_error(
            "cache",
            format!("Cannot create preview folder {}: {error}", root.display()),
        )
    })?;
    Ok(root.join(name))
}

fn preview_path(app: &AppHandle, path: &Path) -> Result<String, CommandError> {
    let label = path
        .to_str()
        .ok_or_else(|| {
            command_error(
                "path",
                format!(
                    "Preview path cannot be represented as UTF-8: {}",
                    path.display()
                ),
            )
        })?
        .to_owned();
    app.asset_protocol_scope()
        .allow_file(path)
        .map_err(|error| {
            command_error(
                "playback",
                format!("Cannot grant access to preview {}: {error}", path.display()),
            )
        })?;
    Ok(label)
}

fn prepare_imported_track(
    app: &AppHandle,
    source_path: PathBuf,
    cache_id: &str,
    name: String,
) -> Result<(ImportedTrack, TrackSummary), CommandError> {
    let audio = media::decode_audio(&source_path)
        .map_err(|error| command_error("decode", error.to_string()))?;
    let analysis =
        engine::analyze(&audio).map_err(|error| command_error("analysis", error.to_string()))?;
    let id = Uuid::new_v4().to_string();
    let preview = cache_path(app, cache_id, &format!("{id}-source.wav"))?;
    media::write_wav(&preview, &audio)
        .map_err(|error| command_error("preview", error.to_string()))?;
    let summary = TrackSummary {
        id: id.clone(),
        name,
        duration_seconds: analysis.duration_seconds,
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        tempo_bpm: analysis.tempo_bpm,
        beat_count: analysis.beat_count,
        waveform: engine::waveform(&audio, 1000)
            .map_err(|error| command_error("waveform", error.to_string()))?,
        preview_path: preview_path(app, &preview)?,
    };
    Ok((
        ImportedTrack {
            id,
            source_path,
            audio: Arc::new(audio),
            analysis: Arc::new(analysis),
        },
        summary,
    ))
}

fn store_imported_track(state: &DesktopSession, track: ImportedTrack) -> Result<(), CommandError> {
    let mut current = state.current.lock().map_err(|error| {
        command_error("session", format!("Cannot update imported track: {error}"))
    })?;
    *current = Session {
        track: Some(track),
        remix: None,
    };
    Ok(())
}

/// Pick and decode on a worker thread; cancellation leaves the existing track intact.
#[tauri::command]
pub async fn import_audio(
    app: AppHandle,
    state: State<'_, DesktopSession>,
) -> Result<Option<TrackSummary>, CommandError> {
    let _operation = state.begin()?;
    let worker_app = app.clone();
    let cache_id = state.cache_id.clone();
    let imported = tauri::async_runtime::spawn_blocking(move || {
        let selected = worker_app
            .dialog()
            .file()
            .add_filter("Music", &["wav", "mp3", "flac", "ogg", "m4a", "aac"])
            .blocking_pick_file();
        let Some(selected) = selected else {
            return Ok(None);
        };
        let source_path = selected.into_path().map_err(|error| {
            command_error(
                "path",
                format!("Selected file is not a local path: {error}"),
            )
        })?;
        let name = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                command_error(
                    "path",
                    format!(
                        "Selected filename cannot be represented as UTF-8: {}",
                        source_path.display()
                    ),
                )
            })?
            .to_owned();
        prepare_imported_track(&worker_app, source_path, &cache_id, name).map(Some)
    })
    .await
    .map_err(|error| command_error("worker", format!("Import worker failed: {error}")))??;
    let Some((track, summary)) = imported else {
        return Ok(None);
    };
    store_imported_track(&state, track)?;
    Ok(Some(summary))
}

/// Import one public YouTube video's audio; failed downloads preserve the active session.
#[tauri::command]
pub async fn import_youtube(
    app: AppHandle,
    state: State<'_, DesktopSession>,
    url: String,
) -> Result<TrackSummary, CommandError> {
    let _operation = state.begin()?;
    let cache_id = state.cache_id.clone();
    let imported = tauri::async_runtime::spawn_blocking(move || {
        let tools = app
            .path()
            .resource_dir()
            .map_err(|error| {
                command_error(
                    "youtube",
                    format!("Cannot locate bundled media tools: {error}"),
                )
            })?
            .join("media-tools");
        let folder = cache_path(&app, &cache_id, &format!("{}-youtube", Uuid::new_v4()))?;
        let downloaded = youtube::download_audio(&tools, &folder, &url)
            .map_err(|error| command_error("youtube", error.to_string()))?;
        prepare_imported_track(&app, downloaded.path, &cache_id, downloaded.title)
    })
    .await
    .map_err(|error| command_error("worker", format!("YouTube import worker failed: {error}")))??;
    store_imported_track(&state, imported.0)?;
    Ok(imported.1)
}

#[tauri::command]
pub async fn generate_remix(
    app: AppHandle,
    state: State<'_, DesktopSession>,
    track_id: String,
    settings: RemixSettings,
) -> Result<RemixSummary, CommandError> {
    let _operation = state.begin()?;
    let (audio, analysis) = {
        let current = state.current.lock().map_err(|error| {
            command_error("session", format!("Cannot access imported track: {error}"))
        })?;
        let track = current
            .track
            .as_ref()
            .filter(|track| track.id == track_id)
            .ok_or_else(|| {
                command_error(
                    "track",
                    format!("Track {track_id} is not the active import; import your music again"),
                )
            })?;
        (Arc::clone(&track.audio), Arc::clone(&track.analysis))
    };
    let cache_id = state.cache_id.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let plan = engine::plan(&analysis, &settings)
            .map_err(|error| command_error("planning", error.to_string()))?;
        let rendered = engine::render(&audio, &plan)
            .map_err(|error| command_error("render", error.to_string()))?;
        let id = Uuid::new_v4().to_string();
        let path = cache_path(&app, &cache_id, &format!("{id}-remix.wav"))?;
        media::write_wav(&path, &rendered)
            .map_err(|error| command_error("preview", error.to_string()))?;
        let summary = RemixSummary {
            id: id.clone(),
            track_id,
            duration_seconds: rendered.samples.len() as f64
                / f64::from(rendered.channels)
                / f64::from(rendered.sample_rate),
            target_seconds: settings.target_seconds,
            tempo_bpm: analysis.tempo_bpm,
            segments: plan
                .segments
                .iter()
                .map(|segment| SegmentSummary {
                    source_start_seconds: segment.source_start_seconds,
                    source_end_seconds: segment.source_end_seconds,
                    output_start_seconds: segment.output_start_seconds,
                    output_end_seconds: segment.output_end_seconds,
                })
                .collect(),
            mean_transition_cost: plan.mean_transition_cost,
            waveform: engine::waveform(&rendered, 1000)
                .map_err(|error| command_error("waveform", error.to_string()))?,
            preview_path: preview_path(&app, &path)?,
        };
        Ok::<_, CommandError>((GeneratedRemix { id, path }, summary))
    })
    .await
    .map_err(|error| command_error("worker", format!("Remix worker failed: {error}")))??;
    let mut current = state
        .current
        .lock()
        .map_err(|error| command_error("session", format!("Cannot store remix result: {error}")))?;
    current.remix = Some(result.0);
    Ok(result.1)
}

/// Save exactly the previewed WAV bytes; the imported source cannot be overwritten.
#[tauri::command]
pub async fn export_remix(
    app: AppHandle,
    state: State<'_, DesktopSession>,
    remix_id: String,
) -> Result<Option<String>, CommandError> {
    let _operation = state.begin()?;
    let (preview, source, name) = {
        let current = state
            .current
            .lock()
            .map_err(|error| command_error("session", format!("Cannot access export: {error}")))?;
        let remix = current
            .remix
            .as_ref()
            .filter(|remix| remix.id == remix_id)
            .ok_or_else(|| {
                command_error(
                    "remix",
                    format!("Remix {remix_id} is no longer active; generate it again"),
                )
            })?;
        let track = current
            .track
            .as_ref()
            .ok_or_else(|| command_error("track", "No imported track for this remix".into()))?;
        let name = track
            .source_path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                command_error(
                    "path",
                    "Source filename cannot be represented as UTF-8".into(),
                )
            })?;
        (
            remix.path.clone(),
            track.source_path.clone(),
            format!("{name}.remix.wav"),
        )
    };
    tauri::async_runtime::spawn_blocking(move || {
        let selected = app
            .dialog()
            .file()
            .set_file_name(name)
            .add_filter("Lossless WAV", &["wav"])
            .blocking_save_file();
        let Some(selected) = selected else {
            return Ok(None);
        };
        let destination = selected.into_path().map_err(|error| {
            command_error(
                "path",
                format!("Export destination is not a local path: {error}"),
            )
        })?;
        if !destination
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        {
            return Err(command_error(
                "export",
                format!("Choose a .wav filename; received {}", destination.display()),
            ));
        }
        if destination.exists() {
            let target = std::fs::canonicalize(&destination).map_err(|error| {
                command_error(
                    "export",
                    format!(
                        "Cannot resolve destination {}: {error}",
                        destination.display()
                    ),
                )
            })?;
            let original = std::fs::canonicalize(&source).map_err(|error| {
                command_error(
                    "export",
                    format!("Cannot resolve source {}: {error}", source.display()),
                )
            })?;
            if target == original {
                return Err(command_error(
                    "export",
                    "Choose a different filename: exporting cannot overwrite the imported source"
                        .into(),
                ));
            }
        }
        std::fs::copy(&preview, &destination).map_err(|error| {
            command_error(
                "export",
                format!(
                    "Cannot copy preview {} to {}: {error}",
                    preview.display(),
                    destination.display()
                ),
            )
        })?;
        let exported_path = destination
            .to_str()
            .ok_or_else(|| {
                command_error("path", "Export path cannot be represented as UTF-8".into())
            })?
            .to_owned();
        Ok(Some(exported_path))
    })
    .await
    .map_err(|error| command_error("worker", format!("Export worker failed: {error}")))?
}

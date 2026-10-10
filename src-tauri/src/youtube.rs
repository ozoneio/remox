//! YouTube connector: bundled executables download one validated video into session storage.
//! The independent audio engine never depends on the network or these tools.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use thiserror::Error;
use url::Url;

#[derive(Debug, Error)]
pub enum YoutubeError {
    #[error("Invalid YouTube URL {input:?}: {reason}. Paste an HTTPS watch, Shorts, live-video, or youtu.be link for one video")]
    InvalidUrl { input: String, reason: String },
    #[error("Cannot access bundled media tool {path}: {source}. Reinstall Remox with its media-tools resources")]
    Tool {
        path: String,
        source: std::io::Error,
    },
    #[error("Cannot {operation} {path}: {source}")]
    File {
        operation: &'static str,
        path: String,
        source: std::io::Error,
    },
    #[error("Cannot start {executable} for {url} with parameters {parameters}: {source}")]
    Start {
        executable: String,
        url: String,
        parameters: String,
        source: std::io::Error,
    },
    #[error("Cannot {operation} for the YouTube downloader: {source}")]
    ProcessControl {
        operation: &'static str,
        source: std::io::Error,
    },
    #[error("YouTube import failed for {url} with parameters {parameters}; status: {status}. stdout: {stdout}; stderr: {stderr}. Check your connection and that the video is public and available without signing in; restricted videos can be imported from a local audio file")]
    Process {
        url: String,
        parameters: String,
        status: String,
        stdout: String,
        stderr: String,
    },
    #[error("Invalid metadata for {url}: {source}; response: {body}")]
    Metadata {
        url: String,
        source: serde_json::Error,
        body: String,
    },
    #[error("Cannot import {url}: {reason}")]
    UnsupportedVideo { url: String, reason: String },
    #[error("YouTube import currently requires the bundled Windows x64 media tools")]
    UnsupportedPlatform,
}

#[derive(Debug, PartialEq, Eq)]
pub struct VideoUrl {
    pub id: String,
    pub url: String,
}

#[derive(Debug)]
pub struct DownloadedAudio {
    pub path: PathBuf,
    pub title: String,
}

#[derive(Deserialize)]
struct VideoMetadata {
    id: String,
    title: String,
    duration: Option<f64>,
    is_live: bool,
    live_status: Option<String>,
}

fn invalid_url(input: &str, reason: &str) -> YoutubeError {
    YoutubeError::InvalidUrl {
        input: input.to_owned(),
        reason: reason.to_owned(),
    }
}

/// Accept only explicit HTTPS YouTube video URLs and remove playlist/tracking parameters.
pub fn validate_url(input: &str) -> Result<VideoUrl, YoutubeError> {
    let text = input.trim();
    if text.bytes().any(|value| value.is_ascii_control()) || text.contains('\\') {
        return Err(invalid_url(
            input,
            "Control characters and backslashes are not allowed",
        ));
    }
    let (scheme, remainder) = text
        .split_once("://")
        .ok_or_else(|| invalid_url(input, "The link must include https://"))?;
    if !scheme.eq_ignore_ascii_case("https") {
        return Err(invalid_url(input, "Only HTTPS links are accepted"));
    }
    let authority = remainder.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains('@') || authority.contains(':') {
        return Err(invalid_url(
            input,
            "User information and explicit ports are not allowed",
        ));
    }
    let parsed = Url::parse(text).map_err(|error| invalid_url(input, &error.to_string()))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| invalid_url(input, "The link has no host"))?;
    let id = match host {
        "youtube.com" | "www.youtube.com" | "m.youtube.com" | "music.youtube.com" => {
            match parsed.path() {
                "/watch" => {
                    let ids: Vec<String> = parsed
                        .query_pairs()
                        .filter(|(key, _)| key == "v")
                        .map(|(_, value)| value.into_owned())
                        .collect();
                    if ids.len() != 1 {
                        return Err(invalid_url(input, "A watch link must contain exactly one v video ID; playlist-only links are not accepted"));
                    }
                    ids[0].clone()
                }
                path => {
                    let segments: Vec<&str> = path.split('/').collect();
                    if segments.len() != 3 || !matches!(segments[1], "shorts" | "live") {
                        return Err(invalid_url(
                            input,
                            "This YouTube path is not a supported single-video link",
                        ));
                    }
                    segments[2].to_owned()
                }
            }
        }
        "youtu.be" => {
            let segments: Vec<&str> = parsed.path().split('/').collect();
            if segments.len() != 2 {
                return Err(invalid_url(
                    input,
                    "A youtu.be link must have exactly one video ID path segment",
                ));
            }
            segments[1].to_owned()
        }
        _ => {
            return Err(invalid_url(
                input,
                "The host must be youtube.com or youtu.be",
            ))
        }
    };
    if id.len() != 11
        || !id
            .bytes()
            .all(|value| value.is_ascii_alphanumeric() || matches!(value, b'_' | b'-'))
    {
        return Err(invalid_url(
            input,
            "The video ID must contain 11 ASCII letters, digits, underscores, or hyphens",
        ));
    }
    Ok(VideoUrl {
        url: format!("https://www.youtube.com/watch?v={id}"),
        id,
    })
}

fn read_metadata(video: &VideoUrl, body: &[u8]) -> Result<VideoMetadata, YoutubeError> {
    let metadata: VideoMetadata =
        serde_json::from_slice(body).map_err(|source| YoutubeError::Metadata {
            url: video.url.clone(),
            source,
            body: String::from_utf8_lossy(body).into_owned(),
        })?;
    let reason = if metadata.id != video.id {
        Some(format!(
            "The returned video ID {:?} does not match {}",
            metadata.id, video.id
        ))
    } else if metadata.title.trim().is_empty() {
        Some("YouTube did not return a nonempty video title".to_owned())
    } else if metadata.is_live || metadata.live_status.as_deref() == Some("is_upcoming") {
        Some(
            "Livestreams and upcoming streams are not supported; choose a completed video"
                .to_owned(),
        )
    } else {
        match metadata.duration {
            None => {
                Some("YouTube did not return a known duration; choose a completed video".to_owned())
            }
            Some(duration) if !duration.is_finite() || duration <= 0.0 => {
                Some(format!("YouTube returned an invalid duration: {duration}"))
            }
            Some(duration) if duration > 900.0 => Some(format!(
                "The video lasts {duration:.1} seconds and exceeds the 15 minute import limit"
            )),
            Some(_) => None,
        }
    };
    if let Some(reason) = reason {
        return Err(YoutubeError::UnsupportedVideo {
            url: video.url.clone(),
            reason,
        });
    }
    Ok(metadata)
}

fn bundled_tool(directory: &Path, name: &str) -> Result<PathBuf, YoutubeError> {
    let path = directory.join(name);
    let metadata = std::fs::metadata(&path).map_err(|source| YoutubeError::Tool {
        path: path.display().to_string(),
        source,
    })?;
    if !path.is_absolute() || !metadata.is_file() {
        return Err(YoutubeError::Tool {
            path: path.display().to_string(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Expected an absolute executable file path",
            ),
        });
    }
    Ok(path)
}

fn downloader_arguments(tools: &Path, deno: &Path) -> Vec<OsString> {
    let mut runtime = OsString::from("deno:");
    runtime.push(deno);
    let mut arguments: Vec<OsString> = [
        "--ignore-config",
        "--no-plugin-dirs",
        "--no-cache-dir",
        "--no-playlist",
        "--no-js-runtimes",
        "--no-remote-components",
        "--no-progress",
        "--no-color",
        "--abort-on-error",
        "--abort-on-unavailable-fragments",
        "--socket-timeout",
        "20",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--extractor-retries",
        "3",
        "--file-access-retries",
        "3",
        "--retry-sleep",
        "http:exp=1:4",
        "--retry-sleep",
        "fragment:exp=1:4",
        "--retry-sleep",
        "extractor:exp=1:4",
        "--retry-sleep",
        "file_access:exp=1:4",
        "--format",
        "bestaudio",
        "--ffmpeg-location",
    ]
    .into_iter()
    .map(OsString::from)
    .collect();
    arguments.push(tools.as_os_str().to_owned());
    arguments.push(OsString::from("--js-runtimes"));
    arguments.push(runtime);
    arguments
}

/// Download with only bundled tools into a new directory; preserve the video's display title.
pub fn download_audio(
    tools: &Path,
    folder: &Path,
    input: &str,
) -> Result<DownloadedAudio, YoutubeError> {
    let video = validate_url(input)?;
    let executable = bundled_tool(tools, "yt-dlp.exe")?;
    let deno = bundled_tool(tools, "deno.exe")?;
    bundled_tool(tools, "ffmpeg.exe")?;
    bundled_tool(tools, "ffprobe.exe")?;
    std::fs::create_dir(folder).map_err(|source| YoutubeError::File {
        operation: "create YouTube import folder",
        path: folder.display().to_string(),
        source,
    })?;
    let arguments = downloader_arguments(tools, &deno);
    let metadata_arguments: Vec<OsString> = arguments
        .iter()
        .cloned()
        .chain(
            ["--dump-single-json", "--skip-download", "--", &video.url]
                .into_iter()
                .map(OsString::from),
        )
        .collect();
    let body = run_downloader(
        &executable,
        folder,
        &video.url,
        "metadata",
        &metadata_arguments,
    )?;
    let metadata = read_metadata(&video, &body)?;
    let download_arguments: Vec<OsString> = arguments
        .into_iter()
        .chain(
            [
                "--extract-audio",
                "--audio-format",
                "wav",
                "--max-filesize",
                "256M",
                "--match-filter",
                "!is_live & duration > 0 & duration <= 900",
                "--output",
                "source.%(ext)s",
                "--",
                &video.url,
            ]
            .into_iter()
            .map(OsString::from),
        )
        .collect();
    run_downloader(
        &executable,
        folder,
        &video.url,
        "download",
        &download_arguments,
    )?;
    let path = folder.join("source.wav");
    let file = std::fs::metadata(&path).map_err(|source| YoutubeError::File {
        operation:
            "read downloaded WAV (the video may have been rejected by its size or duration limit)",
        path: path.display().to_string(),
        source,
    })?;
    if !file.is_file() || file.len() == 0 || file.len() > 256 * 1024 * 1024 {
        return Err(YoutubeError::UnsupportedVideo {
            url: video.url,
            reason: format!(
                "Downloaded WAV must be a nonempty file under 256 MB; found {} bytes",
                file.len()
            ),
        });
    }
    Ok(DownloadedAudio {
        path,
        title: metadata.title,
    })
}

#[cfg(not(windows))]
fn run_downloader(
    _executable: &Path,
    _folder: &Path,
    _url: &str,
    _operation: &str,
    _arguments: &[OsString],
) -> Result<Vec<u8>, YoutubeError> {
    Err(YoutubeError::UnsupportedPlatform)
}

#[cfg(windows)]
fn run_downloader(
    executable: &Path,
    folder: &Path,
    url: &str,
    operation: &str,
    arguments: &[OsString],
) -> Result<Vec<u8>, YoutubeError> {
    use std::{
        fs::File,
        os::windows::process::CommandExt,
        process::Command,
        time::{Duration, Instant},
    };
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

    let stdout_path = folder.join(format!("{operation}-stdout.log"));
    let stderr_path = folder.join(format!("{operation}-stderr.log"));
    let output_file = |path: &Path| {
        File::create(path).map_err(|source| YoutubeError::File {
            operation: "create downloader log",
            path: path.display().to_string(),
            source,
        })
    };
    let job = windows_process::ProcessJob::create()?;
    let mut child = Command::new(executable)
        .args(arguments)
        .current_dir(folder)
        .stdin(std::process::Stdio::null())
        .stdout(output_file(&stdout_path)?)
        .stderr(output_file(&stderr_path)?)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|source| YoutubeError::Start {
            executable: executable.display().to_string(),
            url: url.to_owned(),
            parameters: format!("{arguments:?}"),
            source,
        })?;
    if let Err(error) = job.assign(&child) {
        child
            .kill()
            .map_err(|source| YoutubeError::ProcessControl {
                operation: "stop uncontained process",
                source,
            })?;
        child
            .wait()
            .map_err(|source| YoutubeError::ProcessControl {
                operation: "wait for uncontained process",
                source,
            })?;
        return Err(error);
    }
    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|source| YoutubeError::ProcessControl {
                operation: "wait for process",
                source,
            })?
        {
            break (status, false);
        }
        if started.elapsed() >= Duration::from_secs(240) {
            job.terminate()?;
            let status = child
                .wait()
                .map_err(|source| YoutubeError::ProcessControl {
                    operation: "wait for terminated process",
                    source,
                })?;
            break (status, true);
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    // Closing the job also stops any descendant that outlived the downloader.
    drop(job);
    let read_log = |path: &Path| {
        std::fs::read(path).map_err(|source| YoutubeError::File {
            operation: "read downloader log",
            path: path.display().to_string(),
            source,
        })
    };
    let stdout = read_log(&stdout_path)?;
    let stderr = String::from_utf8_lossy(&read_log(&stderr_path)?).into_owned();
    if !stderr.trim().is_empty() {
        log::warn!(url = url, operation = operation, stderr = stderr.as_str(); "YouTube downloader diagnostics");
    }
    if timed_out || !status.success() {
        return Err(YoutubeError::Process {
            url: url.to_owned(),
            parameters: format!("{arguments:?}"),
            status: if timed_out {
                format!("timed out after 240 seconds ({status})")
            } else {
                status.to_string()
            },
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr,
        });
    }
    Ok(stdout)
}

#[cfg(windows)]
mod windows_process {
    use super::YoutubeError;
    use std::{os::windows::io::AsRawHandle, process::Child};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, HANDLE},
        System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        },
    };

    /// Own the native job handle so downloader children are contained on timeout or failure.
    pub struct ProcessJob {
        handle: HANDLE,
    }

    impl ProcessJob {
        pub fn create() -> Result<Self, YoutubeError> {
            // No name or security attributes are supplied; the handle belongs only to this call.
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return Err(YoutubeError::ProcessControl {
                    operation: "create process job",
                    source: std::io::Error::last_os_error(),
                });
            }
            let job = Self { handle };
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // The pointer and size describe the initialized limits structure for this job.
            let result = unsafe {
                SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    std::mem::size_of_val(&limits) as u32,
                )
            };
            if result == 0 {
                return Err(YoutubeError::ProcessControl {
                    operation: "configure process job",
                    source: std::io::Error::last_os_error(),
                });
            }
            Ok(job)
        }

        pub fn assign(&self, child: &Child) -> Result<(), YoutubeError> {
            // Child keeps its process handle alive for this API call.
            if unsafe { AssignProcessToJobObject(self.handle, child.as_raw_handle()) } == 0 {
                return Err(YoutubeError::ProcessControl {
                    operation: "contain downloader process",
                    source: std::io::Error::last_os_error(),
                });
            }
            Ok(())
        }

        pub fn terminate(&self) -> Result<(), YoutubeError> {
            // The owned job includes the downloader and its inherited child processes.
            if unsafe { TerminateJobObject(self.handle, 1) } == 0 {
                return Err(YoutubeError::ProcessControl {
                    operation: "terminate timed-out process job",
                    source: std::io::Error::last_os_error(),
                });
            }
            Ok(())
        }
    }

    impl Drop for ProcessJob {
        fn drop(&mut self) {
            // The handle is uniquely owned and never escapes this connector.
            if unsafe { CloseHandle(self.handle) } == 0 {
                log::error!(error = std::io::Error::last_os_error().to_string().as_str(); "Cannot close YouTube process job");
            }
        }
    }
}

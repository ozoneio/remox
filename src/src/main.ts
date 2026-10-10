import "./styles.css";
import { exportRemix, generateRemix, importAudio, importYoutube, nativeRuntime, NativeCommandError, previewUrl } from "./bridge";
import type { RemixSettings, RemixSummary, Segment, TrackSummary } from "./bridge";
import { drawWaveform, formatExactTime, formatTime, joinPosition } from "./waveform";

type Job = "import" | "youtube" | "generate" | "export" | null;
type Player = "source" | "output";

interface WorkspaceState {
  track: TrackSummary | null;
  remix: RemixSummary | null;
  job: Job;
  activePlayer: Player;
  auditionEnd: number | null;
  activeJoin: number | null;
  variation: number;
  invalidated: boolean;
}

function element(id: string): HTMLElement {
  const result = document.getElementById(id);
  if (result === null) throw new Error(`Required workspace element #${id} is missing.`);
  return result;
}

function input(id: string): HTMLInputElement {
  const result = element(id);
  if (!(result instanceof HTMLInputElement)) throw new Error(`Workspace element #${id} must be an input.`);
  return result;
}

function button(id: string): HTMLButtonElement {
  const result = element(id);
  if (!(result instanceof HTMLButtonElement)) throw new Error(`Workspace element #${id} must be a button.`);
  return result;
}

function canvas(id: string): HTMLCanvasElement {
  const result = element(id);
  if (!(result instanceof HTMLCanvasElement)) throw new Error(`Workspace element #${id} must be a canvas.`);
  return result;
}

function audio(id: string): HTMLAudioElement {
  const result = element(id);
  if (!(result instanceof HTMLAudioElement)) throw new Error(`Workspace element #${id} must be an audio player.`);
  return result;
}

function fieldset(id: string): HTMLFieldSetElement {
  const result = element(id);
  if (!(result instanceof HTMLFieldSetElement)) throw new Error(`Workspace element #${id} must be a fieldset.`);
  return result;
}

function dialog(id: string): HTMLDialogElement {
  const result = element(id);
  if (!(result instanceof HTMLDialogElement)) throw new Error(`Workspace element #${id} must be a dialog.`);
  return result;
}

function youtubeUrl(value: string): string {
  let url: URL;
  try {
    url = new URL(value.trim());
  } catch (error: unknown) {
    if (!(error instanceof TypeError)) throw error;
    throw new RangeError("Enter a complete YouTube video link beginning with https://.");
  }
  if (url.protocol !== "https:" || url.username !== "" || url.password !== "" || url.port !== "") {
    throw new RangeError("Use an HTTPS YouTube video link without credentials or a custom port.");
  }
  let videoId: string | null;
  if (url.hostname === "youtu.be") {
    videoId = url.pathname.match(/^\/([A-Za-z0-9_-]{11})\/?$/)?.[1] ?? null;
  } else if (["youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"].includes(url.hostname)) {
    videoId = url.pathname === "/watch" ? url.searchParams.get("v") : url.pathname.match(/^\/(?:shorts|live)\/([A-Za-z0-9_-]{11})\/?$/)?.[1] ?? null;
  } else {
    throw new RangeError("Use a video link from youtube.com or youtu.be.");
  }
  if (videoId === null || !/^[A-Za-z0-9_-]{11}$/.test(videoId)) {
    throw new RangeError("Paste a single YouTube video link, such as a watch, Shorts, live, or youtu.be link.");
  }
  return `https://www.youtube.com/watch?v=${videoId}`;
}

function sourceState(current: WorkspaceState, track: TrackSummary): WorkspaceState {
  return { ...current, track, remix: null, variation: 0, invalidated: false, activePlayer: "source", auditionEnd: null, activeJoin: null };
}

function targetSeconds(value: string): number {
  const text = value.trim();
  let seconds: number;
  if (/^\d+(?:\.\d+)?$/.test(text)) {
    seconds = Number(text);
  } else if (/^\d+:\d{1,2}(?:\.\d+)?$/.test(text)) {
    const parts = text.split(":");
    const minutes = Number(parts[0]);
    const remainder = Number(parts[1]);
    if (remainder >= 60) throw new RangeError("Use a seconds value below 60 after the colon.");
    seconds = minutes * 60 + remainder;
  } else {
    throw new RangeError("Enter a duration as minutes:seconds (1:30), or seconds (90).");
  }
  if (!Number.isFinite(seconds) || seconds <= 0) throw new RangeError("Target duration must be greater than zero.");
  return seconds;
}

function boundedNumber(control: HTMLInputElement, label: string): number {
  const value = control.valueAsNumber;
  if (!Number.isFinite(value) || value < Number(control.min) || value > Number(control.max)) {
    throw new RangeError(`${label} must be between ${control.min} and ${control.max}.`);
  }
  return value;
}

function errorText(error: unknown): string {
  if (error instanceof NativeCommandError) return `${error.message} (${error.kind})`;
  if (error instanceof Error) return error.message;
  return `The workspace received an unrecognized error: ${String(error)}`;
}

function playbackIcon(paused: boolean): string {
  return paused
    ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m5 3 8 5-8 5z"/></svg>'
    : '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3h2v10H5zm4 0h2v10H9z"/></svg>';
}

function createWorkspace(): void {
  const sourceAudio = audio("source-audio");
  const outputAudio = audio("output-audio");
  const sourceCanvas = canvas("source-waveform");
  const outputCanvas = canvas("output-waveform");
  const importButton = button("import-audio");
  const youtubeButton = button("import-youtube");
  const youtubeDialog = dialog("youtube-dialog");
  const youtubeInput = input("youtube-url");
  const youtubeSubmit = button("youtube-submit");
  const youtubeCancel = button("youtube-cancel");
  const generateButton = button("generate-remix");
  const exportButton = button("export-remix");
  const sourcePlay = button("play-source");
  const outputPlay = button("play-output");
  const targetInput = input("target-duration");
  const editInput = input("edit-length");
  const featureInput = input("feature-balance");
  const beatInput = input("minimum-beats");
  const slackInput = input("maximum-slack");
  const crossfadeInput = input("crossfade");
  const fields = fieldset("settings-fields");
  let state: WorkspaceState = { track: null, remix: null, job: null, activePlayer: "source", auditionEnd: null, activeJoin: null, variation: 0, invalidated: false };
  let jobSequence = 0;

  function notify(message: string): void {
    const notice = element("notification");
    notice.classList.remove("error");
    notice.setAttribute("role", "status");
    element("notification-message").textContent = message;
    notice.hidden = false;
  }

  function reportError(error: unknown): void {
    const notice = element("notification");
    notice.classList.add("error");
    notice.setAttribute("role", "alert");
    element("notification-message").textContent = errorText(error);
    notice.hidden = false;
  }

  function readSettings(): RemixSettings {
    const minimumBeats = boundedNumber(beatInput, "Minimum segment");
    if (!Number.isInteger(minimumBeats)) throw new RangeError("Minimum segment must be a whole number of beats.");
    return {
      targetSeconds: targetSeconds(targetInput.value),
      toleranceSeconds: boundedNumber(slackInput, "Maximum slack"),
      editLength: boundedNumber(editInput, "Edit length") / 100,
      featureBalance: boundedNumber(featureInput, "Transition balance") / 100,
      minimumBeats,
      crossfadeMs: boundedNumber(crossfadeInput, "Crossfade"),
      variation: state.variation,
    };
  }

  function clearAudition(): void {
    state = { ...state, auditionEnd: null, activeJoin: null };
    element("join-list").querySelectorAll(".active").forEach((join: Element): void => join.classList.remove("active"));
  }

  function updatePlayers(): void {
    sourcePlay.innerHTML = playbackIcon(sourceAudio.paused);
    outputPlay.innerHTML = playbackIcon(outputAudio.paused);
    sourcePlay.setAttribute("aria-label", sourceAudio.paused ? "Play source track" : "Pause source track");
    outputPlay.setAttribute("aria-label", outputAudio.paused ? "Play generated remix" : "Pause generated remix");
    sourcePlay.setAttribute("aria-pressed", String(!sourceAudio.paused));
    outputPlay.setAttribute("aria-pressed", String(!outputAudio.paused));
    element("source-current").textContent = formatTime(sourceAudio.currentTime);
    element("output-current").textContent = formatTime(outputAudio.currentTime);
  }

  function drawWaveforms(): void {
    if (state.track !== null) drawWaveform(sourceCanvas, { samples: state.track.waveform, durationSeconds: state.track.durationSeconds, currentSeconds: sourceAudio.currentTime, segments: [] });
    if (state.remix !== null) drawWaveform(outputCanvas, { samples: state.remix.waveform, durationSeconds: state.remix.durationSeconds, currentSeconds: outputAudio.currentTime, segments: state.remix.segments });
  }

  function refreshControls(): void {
    const busy = state.job !== null;
    const processing = state.job === "import" || state.job === "youtube" || state.job === "generate";
    importButton.disabled = busy || !nativeRuntime;
    youtubeButton.disabled = busy || !nativeRuntime;
    youtubeInput.disabled = busy || !nativeRuntime;
    youtubeSubmit.disabled = busy || !nativeRuntime;
    youtubeCancel.disabled = busy;
    generateButton.disabled = busy || state.track === null || !nativeRuntime;
    exportButton.disabled = busy || state.remix === null;
    sourcePlay.disabled = busy || state.track === null;
    outputPlay.disabled = busy || state.remix === null;
    button("restart-output").disabled = busy || state.remix === null;
    fields.disabled = busy || state.track === null;
    sourceCanvas.tabIndex = busy || state.track === null ? -1 : 0;
    outputCanvas.tabIndex = busy || state.remix === null ? -1 : 0;
    element("join-list").querySelectorAll("button").forEach((join: HTMLButtonElement): void => { join.disabled = busy; });
    element("import-label").textContent = state.job === "import" ? "Analyzing…" : state.track === null ? "Import audio" : "Replace audio";
    element("youtube-submit-label").textContent = state.job === "youtube" ? "Downloading audio…" : "Import from YouTube";
    element("youtube-progress").hidden = state.job !== "youtube";
    element("generate-label").textContent = state.job === "generate" ? "Generating…" : state.remix === null ? "Generate remix" : "Generate another";
    element("output-busy").hidden = !processing;
    element("output-empty").hidden = state.remix !== null || processing;
    element("output-content").hidden = state.remix === null || processing;
    element("busy-title").textContent = state.job === "youtube" ? "Downloading audio…" : state.job === "import" ? "Analyzing source audio" : "Rendering arrangement";
    element("busy-description").textContent = state.job === "youtube" ? "Downloading your source audio and analyzing its musical structure." : state.job === "import" ? "Decoding your audio and analyzing its musical structure." : "Matching musical passages and rendering your audio.";
    const badge = element("arrangement-badge");
    badge.textContent = state.job === "youtube" ? "Importing" : busy ? "Processing" : state.remix === null ? "No arrangement" : "Render complete";
    badge.classList.toggle("ready", state.remix !== null && !busy);
    element("generation-note").textContent = !nativeRuntime ? "Open the desktop app to process local audio." : state.job === "youtube" ? "Downloading and analyzing your source audio…" : state.job === "import" ? "Analyzing the source track on your device…" : state.job === "generate" ? "Rendering your remix on your device…" : state.track === null ? "Import a track to get started." : "Beat-aware edits, rendered locally.";
    element("empty-title").textContent = state.invalidated ? "Arrangement needs rendering" : "No arrangement loaded";
    element("empty-description").textContent = state.invalidated ? "Your settings changed. Generate a new remix to preview the result." : "Import a source file, then set a target duration in Remix properties.";
    updatePlayers();
  }

  function invalidateRemix(): void {
    if (state.remix === null) return;
    outputAudio.pause();
    outputAudio.removeAttribute("src");
    outputAudio.load();
    clearAudition();
    state = { ...state, remix: null, invalidated: true, activePlayer: "source" };
    element("join-list").replaceChildren();
    refreshControls();
  }

  function renderTrack(track: TrackSummary): void {
    element("source-empty").hidden = true;
    element("source-content").hidden = false;
    element("track-name").textContent = track.name;
    element("track-name").title = track.name;
    const channels = track.channels === 1 ? "Mono" : track.channels === 2 ? "Stereo" : `${track.channels} channels`;
    element("track-format").textContent = `${(track.sampleRate / 1000).toLocaleString(undefined, { maximumFractionDigits: 1 })} kHz · ${channels} · ${track.beatCount.toLocaleString()} beats`;
    element("source-tempo").textContent = `${track.tempoBpm.toFixed(1)} BPM`;
    element("source-duration").textContent = formatTime(track.durationSeconds);
    element("source-duration-caption").textContent = formatExactTime(track.durationSeconds);
    sourceAudio.src = previewUrl(track.previewPath);
    sourceAudio.load();
    sourceAudio.currentTime = 0;
    drawWaveforms();
  }

  function applySource(track: TrackSummary): void {
    invalidateRemix();
    state = sourceState(state, track);
    renderTrack(track);
  }

  function renderSegments(remix: RemixSummary): void {
    const map = element("segment-map");
    const joins = element("join-list");
    map.replaceChildren();
    joins.replaceChildren();
    remix.segments.forEach((segment: Segment, index: number): void => {
      const segmentButton = document.createElement("button");
      segmentButton.type = "button";
      segmentButton.className = "segment-block";
      segmentButton.id = `source-segment-${index + 1}`;
      segmentButton.dataset.testid = segmentButton.id;
      segmentButton.style.left = `${(segment.outputStartSeconds / remix.durationSeconds) * 100}%`;
      segmentButton.style.width = `${((segment.outputEndSeconds - segment.outputStartSeconds) / remix.durationSeconds) * 100}%`;
      const description = `Segment ${index + 1}: source ${formatExactTime(segment.sourceStartSeconds)}–${formatExactTime(segment.sourceEndSeconds)}; output ${formatExactTime(segment.outputStartSeconds)}–${formatExactTime(segment.outputEndSeconds)}`;
      segmentButton.title = description;
      segmentButton.setAttribute("aria-label", `Seek to ${description}`);
      segmentButton.textContent = `${index + 1} · ${formatTime(segment.sourceStartSeconds)}`;
      segmentButton.addEventListener("click", (): void => {
        if (state.job !== null || state.remix?.id !== remix.id) return;
        clearAudition();
        state = { ...state, activePlayer: "output" };
        outputAudio.currentTime = segment.outputStartSeconds;
        updatePlayers();
        drawWaveforms();
      });
      map.append(segmentButton);
      const previous = remix.segments[index - 1];
      if (previous === undefined) return;
      const position = joinPosition(previous, segment);
      const joinButton = document.createElement("button");
      joinButton.type = "button";
      joinButton.className = "join-button";
      joinButton.id = `audition-join-${index}`;
      joinButton.dataset.testid = joinButton.id;
      joinButton.setAttribute("aria-label", `Audition join ${index} at ${formatExactTime(position)}`);
      joinButton.title = `Source ${formatExactTime(previous.sourceEndSeconds)} → ${formatExactTime(segment.sourceStartSeconds)}`;
      joinButton.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m5 3 8 5-8 5z"/></svg>';
      joinButton.append(document.createTextNode(`${String(index).padStart(2, "0")} · ${formatTime(position)}`));
      joinButton.addEventListener("click", (): void => { void auditionJoin(index, position, remix.id); });
      joins.append(joinButton);
    });
    if (remix.segments.length === 1) {
      const message = document.createElement("p");
      message.className = "join-empty";
      message.textContent = "One continuous passage. No joins to audition.";
      joins.append(message);
    }
  }

  function renderRemix(remix: RemixSummary): void {
    element("result-duration").textContent = formatExactTime(remix.durationSeconds);
    element("result-target").textContent = formatExactTime(remix.targetSeconds);
    const difference = remix.durationSeconds - remix.targetSeconds;
    element("duration-difference").textContent = Math.abs(difference) < .05 ? "On target" : `${difference > 0 ? "+" : "−"}${Math.abs(difference).toFixed(1)} sec from target`;
    element("output-duration").textContent = formatTime(remix.durationSeconds);
    element("result-edits").textContent = String(remix.segments.length - 1);
    element("result-tempo").textContent = `${remix.tempoBpm.toFixed(1)} BPM`;
    element("result-segments").textContent = String(remix.segments.length);
    element("segment-count").textContent = `${remix.segments.length} musical passages`;
    const scale = element("timeline-scale");
    scale.replaceChildren(...[0, .25, .5, .75, 1].map((fraction: number): HTMLSpanElement => {
      const label = document.createElement("span");
      label.textContent = formatTime(remix.durationSeconds * fraction);
      return label;
    }));
    renderSegments(remix);
    outputAudio.src = previewUrl(remix.previewPath);
    outputAudio.load();
    outputAudio.currentTime = 0;
  }

  async function importTrack(): Promise<void> {
    if (state.job !== null || !nativeRuntime) return;
    const job = ++jobSequence;
    sourceAudio.pause();
    outputAudio.pause();
    clearAudition();
    state = { ...state, job: "import" };
    element("notification").hidden = true;
    refreshControls();
    try {
      const track = await importAudio();
      if (job !== jobSequence || track === null) return;
      applySource(track);
    } catch (error: unknown) {
      reportError(error);
    } finally {
      if (job === jobSequence) {
        state = { ...state, job: null };
        refreshControls();
        drawWaveforms();
      }
    }
  }

  async function importYoutubeTrack(): Promise<void> {
    if (state.job !== null || !nativeRuntime) return;
    const message = element("youtube-error");
    message.hidden = true;
    youtubeInput.removeAttribute("aria-invalid");
    let url: string;
    try {
      url = youtubeUrl(youtubeInput.value);
    } catch (error: unknown) {
      message.textContent = errorText(error);
      message.hidden = false;
      youtubeInput.setAttribute("aria-invalid", "true");
      youtubeInput.focus();
      return;
    }
    const job = ++jobSequence;
    sourceAudio.pause();
    outputAudio.pause();
    clearAudition();
    state = { ...state, job: "youtube" };
    element("notification").hidden = true;
    refreshControls();
    try {
      const track = await importYoutube(url);
      if (job !== jobSequence) return;
      applySource(track);
      youtubeDialog.close();
    } catch (error: unknown) {
      message.textContent = errorText(error);
      message.hidden = false;
    } finally {
      if (job === jobSequence) {
        state = { ...state, job: null };
        refreshControls();
        drawWaveforms();
      }
    }
  }

  async function generateArrangement(): Promise<void> {
    if (state.job !== null || state.track === null) return;
    let settings: RemixSettings;
    try {
      settings = readSettings();
    } catch (error: unknown) {
      reportError(error);
      return;
    }
    const track = state.track;
    const job = ++jobSequence;
    sourceAudio.pause();
    outputAudio.pause();
    invalidateRemix();
    state = { ...state, job: "generate" };
    element("notification").hidden = true;
    refreshControls();
    try {
      const remix = await generateRemix(track.id, settings);
      if (job !== jobSequence) return;
      state = { ...state, remix, variation: state.variation + 1, invalidated: false, activePlayer: "output" };
      renderRemix(remix);
    } catch (error: unknown) {
      reportError(error);
    } finally {
      if (job === jobSequence) {
        state = { ...state, job: null };
        refreshControls();
        drawWaveforms();
      }
    }
  }

  async function saveArrangement(): Promise<void> {
    if (state.job !== null || state.remix === null) return;
    const remix = state.remix;
    state = { ...state, job: "export" };
    refreshControls();
    try {
      const path = await exportRemix(remix.id);
      if (path !== null) notify(`WAV saved to ${path}`);
    } catch (error: unknown) {
      reportError(error);
    } finally {
      state = { ...state, job: null };
      refreshControls();
    }
  }

  async function playSource(): Promise<void> {
    if (state.job !== null || state.track === null) return;
    const track = state.track;
    clearAudition();
    state = { ...state, activePlayer: "source" };
    outputAudio.pause();
    if (!sourceAudio.paused) {
      sourceAudio.pause();
      return;
    }
    try {
      await sourceAudio.play();
    } catch (error: unknown) {
      reportError(new Error(`Source audio playback failed for ${track.previewPath}: ${errorText(error)}`));
    }
  }

  async function playOutput(): Promise<void> {
    if (state.job !== null || state.remix === null) return;
    const remix = state.remix;
    clearAudition();
    state = { ...state, activePlayer: "output" };
    sourceAudio.pause();
    if (!outputAudio.paused) {
      outputAudio.pause();
      return;
    }
    try {
      await outputAudio.play();
    } catch (error: unknown) {
      reportError(new Error(`Remix audio playback failed for ${remix.previewPath}: ${errorText(error)}`));
    }
  }

  async function auditionJoin(index: number, position: number, remixId: string): Promise<void> {
    if (state.job !== null || state.remix === null || state.remix.id !== remixId) return;
    clearAudition();
    sourceAudio.pause();
    outputAudio.pause();
    outputAudio.currentTime = Math.max(0, position - 2);
    state = { ...state, activePlayer: "output", activeJoin: index, auditionEnd: Math.min(state.remix.durationSeconds, position + 2) };
    button(`audition-join-${index}`).classList.add("active");
    try {
      await outputAudio.play();
    } catch (error: unknown) {
      clearAudition();
      reportError(new Error(`Join ${index} playback failed: ${errorText(error)}`));
    }
  }

  function settingsChanged(): void {
    if (state.job !== null) return;
    invalidateRemix();
    element("edit-length-value").textContent = editInput.valueAsNumber < 35 ? "Short edits" : editInput.valueAsNumber > 65 ? "Long passages" : "Balanced";
    element("feature-balance-value").textContent = featureInput.valueAsNumber < 35 ? "Timbre-led" : featureInput.valueAsNumber > 65 ? "Harmony-led" : "Balanced";
    editInput.style.background = `linear-gradient(to right, var(--accent) ${editInput.value}%, var(--control-track) ${editInput.value}%)`;
    featureInput.style.background = `linear-gradient(to right, var(--accent) ${featureInput.value}%, var(--control-track) ${featureInput.value}%)`;
    const help = element("target-help");
    try {
      targetSeconds(targetInput.value);
      targetInput.removeAttribute("aria-invalid");
      help.classList.remove("invalid");
      help.textContent = "Use mm:ss or seconds";
    } catch (error: unknown) {
      targetInput.setAttribute("aria-invalid", "true");
      help.classList.add("invalid");
      help.textContent = errorText(error);
    }
  }

  function attachSeeking(waveform: HTMLCanvasElement, player: HTMLAudioElement, getDuration: () => number): void {
    waveform.addEventListener("pointerdown", (event: PointerEvent): void => {
      if (state.job !== null || getDuration() <= 0) return;
      const bounds = waveform.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
      clearAudition();
      state = { ...state, activePlayer: player === sourceAudio ? "source" : "output" };
      player.currentTime = position * getDuration();
      waveform.focus();
      updatePlayers();
      drawWaveforms();
    });
    waveform.addEventListener("keydown", (event: KeyboardEvent): void => {
      if (state.job !== null || getDuration() <= 0) return;
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      clearAudition();
      state = { ...state, activePlayer: player === sourceAudio ? "source" : "output" };
      player.currentTime = event.key === "Home" ? 0 : event.key === "End" ? getDuration() : Math.max(0, Math.min(getDuration(), player.currentTime + (event.key === "ArrowRight" ? 1 : -1)));
      updatePlayers();
      drawWaveforms();
    });
  }

  importButton.addEventListener("click", (): void => { void importTrack(); });
  youtubeButton.addEventListener("click", (): void => {
    if (state.job !== null || !nativeRuntime) return;
    element("youtube-error").hidden = true;
    youtubeInput.removeAttribute("aria-invalid");
    youtubeDialog.showModal();
    youtubeInput.focus();
  });
  element("youtube-form").addEventListener("submit", (event: SubmitEvent): void => {
    event.preventDefault();
    void importYoutubeTrack();
  });
  youtubeCancel.addEventListener("click", (): void => {
    if (state.job !== null) return;
    youtubeDialog.close();
  });
  youtubeDialog.addEventListener("cancel", (event: Event): void => {
    if (state.job !== null) event.preventDefault();
  });
  generateButton.addEventListener("click", (): void => { void generateArrangement(); });
  exportButton.addEventListener("click", (): void => { void saveArrangement(); });
  sourcePlay.addEventListener("click", (): void => { void playSource(); });
  outputPlay.addEventListener("click", (): void => { void playOutput(); });
  button("restart-output").addEventListener("click", (): void => {
    if (state.job !== null || state.remix === null) return;
    clearAudition();
    outputAudio.currentTime = 0;
    state = { ...state, activePlayer: "output" };
    updatePlayers();
    drawWaveforms();
  });
  button("dismiss-notification").addEventListener("click", (): void => { element("notification").hidden = true; });
  for (const control of [targetInput, editInput, featureInput, beatInput, slackInput, crossfadeInput]) control.addEventListener("input", settingsChanged);
  for (const player of [sourceAudio, outputAudio]) {
    player.addEventListener("play", (): void => {
      if (player === sourceAudio) outputAudio.pause();
      else sourceAudio.pause();
      updatePlayers();
    });
    player.addEventListener("pause", updatePlayers);
    player.addEventListener("ended", (): void => { clearAudition(); updatePlayers(); });
    player.addEventListener("timeupdate", (): void => {
      if (player === outputAudio && state.auditionEnd !== null && outputAudio.currentTime >= state.auditionEnd) {
        outputAudio.pause();
        clearAudition();
      }
      updatePlayers();
      drawWaveforms();
    });
    player.addEventListener("error", (): void => {
      if (player.error === null || player.getAttribute("src") === null) return;
      reportError(new Error(`Could not load ${player === sourceAudio ? "source" : "remix"} preview ${player.currentSrc}: ${player.error.message} (media error ${player.error.code}).`));
    });
  }
  attachSeeking(sourceCanvas, sourceAudio, (): number => state.track?.durationSeconds ?? 0);
  attachSeeking(outputCanvas, outputAudio, (): number => state.remix?.durationSeconds ?? 0);
  document.addEventListener("keydown", (event: KeyboardEvent): void => {
    if (event.code !== "Space" || event.repeat || event.altKey || event.ctrlKey || event.metaKey || state.job !== null) return;
    const focus = document.activeElement;
    if (focus instanceof HTMLInputElement || focus instanceof HTMLTextAreaElement || focus instanceof HTMLButtonElement || focus instanceof HTMLAnchorElement || (focus instanceof HTMLElement && focus.isContentEditable)) return;
    event.preventDefault();
    if (focus === sourceCanvas || (focus !== outputCanvas && state.activePlayer === "source")) void playSource();
    else void playOutput();
  });
  new ResizeObserver(drawWaveforms).observe(document.body);
  new ResizeObserver(drawWaveforms).observe(sourceCanvas);
  new ResizeObserver(drawWaveforms).observe(outputCanvas);
  if (!nativeRuntime) {
    notify("This workspace needs the remox desktop app. Launch the native app to import, remix, and export audio on your device.");
  }
  refreshControls();
}

createWorkspace();

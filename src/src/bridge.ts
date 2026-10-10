import { convertFileSrc, invoke, isTauri } from "@tauri-apps/api/core";

export interface TrackSummary {
  id: string;
  name: string;
  durationSeconds: number;
  sampleRate: number;
  channels: number;
  tempoBpm: number;
  beatCount: number;
  waveform: number[];
  previewPath: string;
}

export interface Segment {
  sourceStartSeconds: number;
  sourceEndSeconds: number;
  outputStartSeconds: number;
  outputEndSeconds: number;
}

export interface RemixSettings {
  targetSeconds: number;
  toleranceSeconds: number;
  editLength: number;
  featureBalance: number;
  minimumBeats: number;
  crossfadeMs: number;
  variation: number;
}

export interface RemixSummary {
  id: string;
  trackId: string;
  durationSeconds: number;
  targetSeconds: number;
  tempoBpm: number;
  segments: Segment[];
  meanTransitionCost: number;
  waveform: number[];
  previewPath: string;
}

interface CommandFailure {
  kind: string;
  message: string;
}

interface TrackResponse {
  id: unknown;
  name: unknown;
  durationSeconds: unknown;
  sampleRate: unknown;
  channels: unknown;
  tempoBpm: unknown;
  beatCount: unknown;
  waveform: unknown;
  previewPath: unknown;
}

interface SegmentResponse {
  sourceStartSeconds: unknown;
  sourceEndSeconds: unknown;
  outputStartSeconds: unknown;
  outputEndSeconds: unknown;
}

interface RemixResponse {
  id: unknown;
  trackId: unknown;
  durationSeconds: unknown;
  targetSeconds: unknown;
  tempoBpm: unknown;
  segments: unknown;
  meanTransitionCost: unknown;
  waveform: unknown;
  previewPath: unknown;
}

export class NativeCommandError extends Error {
  readonly kind: string;

  constructor(failure: CommandFailure) {
    super(failure.message);
    this.name = "NativeCommandError";
    this.kind = failure.kind;
  }
}

export const nativeRuntime = isTauri();

function trackResponse(value: unknown): TrackResponse {
  if (typeof value !== "object" || value === null || !("id" in value) || !("name" in value) || !("durationSeconds" in value) || !("sampleRate" in value) || !("channels" in value) || !("tempoBpm" in value) || !("beatCount" in value) || !("waveform" in value) || !("previewPath" in value)) {
    throw new NativeCommandError({ kind: "invalid_response", message: "Audio import returned a track with missing required fields." });
  }
  return value;
}

function segmentResponse(value: unknown, index: number): SegmentResponse {
  if (typeof value !== "object" || value === null || !("sourceStartSeconds" in value) || !("sourceEndSeconds" in value) || !("outputStartSeconds" in value) || !("outputEndSeconds" in value)) {
    throw new NativeCommandError({ kind: "invalid_response", message: `Segment ${index + 1} is missing required source or output boundaries.` });
  }
  return value;
}

function remixResponse(value: unknown): RemixResponse {
  if (typeof value !== "object" || value === null || !("id" in value) || !("trackId" in value) || !("durationSeconds" in value) || !("targetSeconds" in value) || !("tempoBpm" in value) || !("segments" in value) || !("meanTransitionCost" in value) || !("waveform" in value) || !("previewPath" in value)) {
    throw new NativeCommandError({ kind: "invalid_response", message: "Remix generation returned an arrangement with missing required fields." });
  }
  return value;
}

function responseString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new NativeCommandError({ kind: "invalid_response", message: `The audio engine returned an empty or invalid ${field}.` });
  }
  return value;
}

function responseNumber(value: unknown, field: string, minimum: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new NativeCommandError({ kind: "invalid_response", message: `The audio engine returned an invalid ${field}: ${String(value)}.` });
  }
  return value;
}

function responseInteger(value: unknown, field: string, minimum: number): number {
  const result = responseNumber(value, field, minimum);
  if (!Number.isInteger(result)) {
    throw new NativeCommandError({ kind: "invalid_response", message: `The audio engine returned a non-integer ${field}: ${result}.` });
  }
  return result;
}

function responseWaveform(value: unknown): number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new NativeCommandError({ kind: "invalid_response", message: "The audio engine returned no waveform samples." });
  }
  return value.map((sample: unknown, index: number): number => {
    const amplitude = responseNumber(sample, `waveform sample ${index}`, -1);
    if (amplitude > 1) {
      throw new NativeCommandError({ kind: "invalid_response", message: `Waveform sample ${index} is outside the normalized range: ${amplitude}.` });
    }
    return amplitude;
  });
}

function parseTrack(value: unknown): TrackSummary {
  const data = trackResponse(value);
  return {
    id: responseString(data.id, "track ID"),
    name: responseString(data.name, "track name"),
    durationSeconds: responseNumber(data.durationSeconds, "source duration", Number.MIN_VALUE),
    sampleRate: responseInteger(data.sampleRate, "sample rate", 1),
    channels: responseInteger(data.channels, "channel count", 1),
    tempoBpm: responseNumber(data.tempoBpm, "source tempo", 0),
    beatCount: responseInteger(data.beatCount, "beat count", 0),
    waveform: responseWaveform(data.waveform),
    previewPath: responseString(data.previewPath, "source preview path"),
  };
}

function parseSegment(value: unknown, index: number): Segment {
  const data = segmentResponse(value, index);
  const segment: Segment = {
    sourceStartSeconds: responseNumber(data.sourceStartSeconds, "source segment start", 0),
    sourceEndSeconds: responseNumber(data.sourceEndSeconds, "source segment end", Number.MIN_VALUE),
    outputStartSeconds: responseNumber(data.outputStartSeconds, "output segment start", 0),
    outputEndSeconds: responseNumber(data.outputEndSeconds, "output segment end", Number.MIN_VALUE),
  };
  if (segment.sourceEndSeconds <= segment.sourceStartSeconds || segment.outputEndSeconds <= segment.outputStartSeconds) {
    throw new NativeCommandError({ kind: "invalid_response", message: `Segment ${index + 1} has an end before its start.` });
  }
  return segment;
}

function parseRemix(value: unknown, trackId: string): RemixSummary {
  const data = remixResponse(value);
  if (!Array.isArray(data.segments) || data.segments.length === 0) {
    throw new NativeCommandError({ kind: "invalid_response", message: "The audio engine returned an arrangement without segments." });
  }
  const result: RemixSummary = {
    id: responseString(data.id, "remix ID"),
    trackId: responseString(data.trackId, "remix source ID"),
    durationSeconds: responseNumber(data.durationSeconds, "remix duration", Number.MIN_VALUE),
    targetSeconds: responseNumber(data.targetSeconds, "target duration", Number.MIN_VALUE),
    tempoBpm: responseNumber(data.tempoBpm, "remix tempo", 0),
    segments: data.segments.map((segment: unknown, index: number): Segment => parseSegment(segment, index)),
    meanTransitionCost: responseNumber(data.meanTransitionCost, "transition cost", 0),
    waveform: responseWaveform(data.waveform),
    previewPath: responseString(data.previewPath, "remix preview path"),
  };
  if (result.trackId !== trackId) {
    throw new NativeCommandError({ kind: "invalid_response", message: "The audio engine returned an arrangement for a different source track." });
  }
  return result;
}

function commandFailure(error: unknown): NativeCommandError {
  if (typeof error === "object" && error !== null && "kind" in error && "message" in error && typeof error.kind === "string" && typeof error.message === "string") {
    return new NativeCommandError({ kind: error.kind, message: error.message });
  }
  const message = error instanceof Error ? error.message : String(error);
  return new NativeCommandError({ kind: "native_command", message });
}

async function nativeCall(command: string, args: { trackId: string; settings: RemixSettings } | { remixId: string } | { url: string } | undefined): Promise<unknown> {
  if (!nativeRuntime) {
    throw new NativeCommandError({ kind: "native_runtime", message: "Open remox in its desktop app to import, process, and export local audio." });
  }
  try {
    return await invoke<unknown>(command, args);
  } catch (error: unknown) {
    throw commandFailure(error);
  }
}

export async function importAudio(): Promise<TrackSummary | null> {
  const result = await nativeCall("import_audio", undefined);
  return result === null ? null : parseTrack(result);
}

export async function importYoutube(url: string): Promise<TrackSummary> {
  return parseTrack(await nativeCall("import_youtube", { url }));
}

export async function generateRemix(trackId: string, settings: RemixSettings): Promise<RemixSummary> {
  return parseRemix(await nativeCall("generate_remix", { trackId, settings }), trackId);
}

export async function exportRemix(remixId: string): Promise<string | null> {
  const result = await nativeCall("export_remix", { remixId });
  return result === null ? null : responseString(result, "export path");
}

export function previewUrl(path: string): string {
  return convertFileSrc(path);
}

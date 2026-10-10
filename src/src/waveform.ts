import type { Segment } from "./bridge";

export interface WaveformView {
  samples: readonly number[];
  durationSeconds: number;
  currentSeconds: number;
  segments: readonly Segment[];
}

export function formatTime(seconds: number): string {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(wholeSeconds / 60);
  return `${minutes}:${String(wholeSeconds % 60).padStart(2, "0")}`;
}

export function formatExactTime(seconds: number): string {
  const tenths = Math.round(seconds * 10);
  const minutes = Math.floor(tenths / 600);
  const remainder = (tenths % 600) / 10;
  return `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

export function joinPosition(previous: Segment, next: Segment): number {
  return (previous.outputEndSeconds + next.outputStartSeconds) / 2;
}

export function drawWaveform(canvas: HTMLCanvasElement, view: WaveformView): void {
  const rectangle = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio;
  const width = Math.max(1, Math.round(rectangle.width));
  const height = Math.max(1, Math.round(rectangle.height));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("The desktop WebView could not create a canvas context for the waveform.");
  }
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const colors = getComputedStyle(canvas);
  const waveformColor = colors.getPropertyValue("--waveform");
  const playedColor = colors.getPropertyValue("--waveform-played");
  context.strokeStyle = colors.getPropertyValue("--waveform-grid");
  context.lineWidth = 1;
  context.beginPath();
  for (let division = 1; division < 8; division += 1) {
    const x = Math.round(width * division / 8) + .5;
    context.moveTo(x, 0);
    context.lineTo(x, height);
  }
  for (const fraction of [.25, .5, .75]) {
    const y = Math.round(height * fraction) + .5;
    context.moveTo(0, y);
    context.lineTo(width, y);
  }
  context.stroke();
  const bars = Math.max(1, Math.floor(width / 2));
  const progress = Math.min(1, Math.max(0, view.currentSeconds / view.durationSeconds));
  context.fillStyle = colors.getPropertyValue("--waveform-grid");
  context.fillRect(0, height / 2, width, 1);
  for (let bar = 0; bar < bars; bar += 1) {
    const start = Math.floor((bar / bars) * view.samples.length);
    const end = Math.max(start + 1, Math.floor(((bar + 1) / bars) * view.samples.length));
    let amplitude = 0;
    for (let sample = start; sample < end; sample += 1) {
      const value = view.samples[sample];
      if (value !== undefined) amplitude = Math.max(amplitude, Math.abs(value));
    }
    const x = (bar / bars) * width;
    const barHeight = Math.max(1, amplitude * (height - 14));
    context.fillStyle = x < progress * width ? playedColor : waveformColor;
    context.fillRect(x, (height - barHeight) / 2, 1.5, barHeight);
  }
  view.segments.forEach((segment: Segment, index: number): void => {
    if (index === 0) return;
    const previous = view.segments[index - 1];
    if (previous === undefined) return;
    const x = (joinPosition(previous, segment) / view.durationSeconds) * width;
    context.strokeStyle = colors.getPropertyValue("--waveform-join");
    context.setLineDash([2, 3]);
    context.beginPath();
    context.moveTo(x, 3);
    context.lineTo(x, height - 3);
    context.stroke();
  });
  context.setLineDash([]);
  if (view.currentSeconds > 0) {
    context.fillStyle = colors.getPropertyValue("--playhead");
    context.fillRect(Math.min(width - 1, progress * width), 0, 1, height);
  }
  canvas.setAttribute("aria-valuemax", String(view.durationSeconds));
  canvas.setAttribute("aria-valuenow", String(view.currentSeconds));
  canvas.setAttribute("aria-valuetext", `${formatTime(view.currentSeconds)} of ${formatTime(view.durationSeconds)}`);
}

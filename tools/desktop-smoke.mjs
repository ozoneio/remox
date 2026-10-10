import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn, execFile } from 'node:child_process';
import { once } from 'node:events';
import { closeSync, openSync } from 'node:fs';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from '@playwright/test';

/** @typedef {import('@playwright/test').Page} Page */
/** @typedef {import('node:child_process').ChildProcess} ChildProcess */
/** @typedef {{duration: number, currentTime: number, paused: boolean, source: string, errorCode: number | null}} AudioState */
/** @typedef {{duration: number, sampleRate: number, channels: number, rms: number, peak: number}} WaveSummary */

export const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const executeFile = promisify(execFile);

/** Create a real stereo PCM16 fixture with drums and changing harmonic sections. */
export async function writeFixture(filePath, sampleRate, duration) {
  const frames = Math.round(sampleRate * duration);
  const wave = Buffer.alloc(44 + frames * 4);
  wave.write('RIFF', 0);
  wave.writeUInt32LE(wave.length - 8, 4);
  wave.write('WAVEfmt ', 8);
  wave.writeUInt32LE(16, 16);
  wave.writeUInt16LE(1, 20);
  wave.writeUInt16LE(2, 22);
  wave.writeUInt32LE(sampleRate, 24);
  wave.writeUInt32LE(sampleRate * 4, 28);
  wave.writeUInt16LE(4, 32);
  wave.writeUInt16LE(16, 34);
  wave.write('data', 36);
  wave.writeUInt32LE(frames * 4, 40);
  const notes = [261.6256, 329.6276, 391.9954, 349.2282, 220, 293.6648];
  for (let frame = 0; frame < frames; frame += 1) {
    const time = frame / sampleRate;
    const beat = Math.floor(time * 2);
    const beatTime = time - beat * 0.5;
    const note = notes[Math.floor(beat / 8) % notes.length];
    const kick = 0.5 * Math.exp(-beatTime * 24) * Math.sin(2 * Math.PI * (48 * beatTime + 30 * (1 - Math.exp(-beatTime * 40)) / 40));
    const snare = beat % 2 === 1 ? 0.13 * Math.exp(-beatTime * 28) * (Math.sin(frame * 1.7) + Math.sin(frame * 2.13)) : 0;
    const hatTime = time % 0.25;
    const hat = 0.055 * Math.exp(-hatTime * 90) * Math.sin(2 * Math.PI * 7000 * time);
    const envelope = Math.min(1, time / 0.02, (duration - time) / 1.5);
    for (let channel = 0; channel < 2; channel += 1) {
      const melody = 0.11 * Math.sin(2 * Math.PI * note * time + channel * 0.12)
        + 0.065 * Math.sin(2 * Math.PI * note * 1.5 * time + channel * 0.3)
        + 0.05 * Math.sin(2 * Math.PI * note * 0.5 * time);
      const sample = (kick + snare + hat + melody) * envelope;
      wave.writeInt16LE(Math.round(sample * 32767), 44 + frame * 4 + channel * 2);
    }
  }
  await writeFile(filePath, wave);
}

/** Decode and inspect the WAV produced by the native export command. @returns {Promise<WaveSummary>} */
async function inspectWave(filePath, expectedDuration, tolerance, expectedSampleRate, expectedChannels) {
  const wave = await readFile(filePath);
  assert.equal(wave.toString('ascii', 0, 4), 'RIFF', `Export is not RIFF: ${filePath}`);
  assert.equal(wave.toString('ascii', 8, 12), 'WAVE', `Export is not WAV: ${filePath}`);
  let format = null;
  let samples = null;
  for (let offset = 12; offset + 8 <= wave.length;) {
    const chunk = wave.toString('ascii', offset, offset + 4);
    const size = wave.readUInt32LE(offset + 4);
    const start = offset + 8;
    assert.ok(start + size <= wave.length, `Truncated WAV chunk ${chunk}: ${filePath}`);
    if (chunk === 'fmt ') {
      assert.ok(size >= 16, `Invalid WAV format chunk: ${filePath}`);
      format = {
        encoding: wave.readUInt16LE(start),
        channels: wave.readUInt16LE(start + 2),
        sampleRate: wave.readUInt32LE(start + 4),
        bits: wave.readUInt16LE(start + 14),
        extensionSize: size >= 40 ? wave.readUInt16LE(start + 16) : null,
        validBits: size >= 40 ? wave.readUInt16LE(start + 18) : null,
        subtype: size >= 40 ? wave.subarray(start + 24, start + 40).toString('hex') : null,
      };
    }
    if (chunk === 'data') samples = wave.subarray(start, start + size);
    offset = start + size + size % 2;
  }
  assert.ok(format, `Export has no WAV format chunk: ${filePath}`);
  assert.ok(samples, `Export has no WAV audio chunk: ${filePath}`);
  assert.equal(format.encoding, 0xfffe, `Expected Hound extensible WAV export: ${filePath}`);
  assert.equal(format.bits, 32, `Expected native float32 WAV export: ${filePath}`);
  assert.equal(format.extensionSize, 22, `Invalid native extensible WAV header: ${filePath}`);
  assert.equal(format.validBits, 32, `Expected 32 valid bits per exported sample: ${filePath}`);
  assert.equal(format.subtype, '0300000000001000800000aa00389b71', `Expected IEEE float WAV subtype: ${filePath}`);
  assert.equal(format.channels, expectedChannels, `Native export changed the channel count: ${filePath}`);
  assert.equal(format.sampleRate, expectedSampleRate, `Native export changed the sample rate: ${filePath}`);
  assert.equal(samples.length % (expectedChannels * 4), 0, `Incomplete float32 frame: ${filePath}`);
  const duration = samples.length / (format.channels * format.sampleRate * 4);
  assert.ok(Math.abs(duration - expectedDuration) <= tolerance, `Export duration ${duration}s exceeds ${expectedDuration}s ± ${tolerance}s: ${filePath}`);
  let peak = 0;
  let squareSum = 0;
  for (let offset = 0; offset < samples.length; offset += 4) {
    const sample = samples.readFloatLE(offset);
    assert.ok(Number.isFinite(sample), `Non-finite exported sample at byte ${offset}: ${filePath}`);
    peak = Math.max(peak, Math.abs(sample));
    squareSum += sample * sample;
  }
  const rms = Math.sqrt(squareSum / (samples.length / 4));
  assert.ok(peak > 0.05 && rms > 0.005, `Native export is silent or near silent: peak=${peak}, rms=${rms}, file=${filePath}`);
  return { duration, channels: format.channels, sampleRate: format.sampleRate, peak, rms };
}

/** Allocate an unused debugging port rather than colliding with another desktop app. */
async function selectDebuggingPort() {
  const server = createServer();
  const listening = once(server, 'listening');
  server.listen(0, '127.0.0.1');
  await listening;
  const address = server.address();
  assert.ok(address && typeof address === 'object', 'Could not allocate a local WebView2 debugging port');
  await new Promise((resolveClose, rejectClose) => {
    server.close((error) => error ? rejectClose(error) : resolveClose());
  });
  return address.port;
}

/** Wait for the real WebView2 debugging endpoint, reporting transient startup errors. */
async function waitForWebView(app, address) {
  let lastError = new Error(`WebView2 did not start at ${address}`);
  for (let attempt = 1; attempt <= 30; attempt += 1) {
    assert.equal(app.exitCode, null, `Native app exited before WebView2 startup: exitCode=${app.exitCode}`);
    try {
      const response = await fetch(`${address}/json/version`, { signal: AbortSignal.timeout(1000) });
      if (!response.ok) throw new Error(`WebView2 startup request failed: url=${address}/json/version, status=${response.status}, body=${await response.text()}`);
      const version = await response.json();
      assert.ok(version && typeof version === 'object' && !Array.isArray(version), 'WebView2 debugging response is not an object');
      assert.equal(typeof version.webSocketDebuggerUrl, 'string', 'WebView2 debugging response has no webSocketDebuggerUrl');
      return;
    } catch (error) {
      if (!(error instanceof Error)) throw error;
      lastError = error;
      console.warn(JSON.stringify({ event: 'webview_startup_retry', attempt, error: error.message }));
      await delay(1000);
    }
  }
  throw lastError;
}

export async function confirmNativeDialog(processId, filePath, fileNameAutomationId) {
  assert.equal(typeof processId, 'number', 'Native app has no process ID');
  const result = await executeFile('powershell.exe', [
    '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
    '-File', resolve(root, 'tools/desktop-dialog.ps1'),
    '-ProcessId', String(processId), '-FilePath', filePath,
    '-FileNameAutomationId', fileNameAutomationId,
  ], { windowsHide: true, timeout: 25000 });
  const confirmation = JSON.parse(result.stdout.trim());
  assert.ok(confirmation && typeof confirmation === 'object' && !Array.isArray(confirmation), 'Native dialog helper response is not an object');
  assert.equal(confirmation.confirmed, true, `Native dialog did not confirm ${filePath}`);
  assert.equal(confirmation.processId, processId, 'Native dialog targeted a different process');
}

/** @param {Page} page @returns {Promise<AudioState>} */
export async function readAudioState(page, audioId) {
  return page.locator(`#${audioId}`).evaluate((audio) => {
    if (!(audio instanceof HTMLAudioElement)) throw new Error('Expected a native HTML audio element');
    return {
      duration: audio.duration, currentTime: audio.currentTime,
      paused: audio.paused, source: audio.currentSrc,
      errorCode: audio.error ? audio.error.code : null,
    };
  });
}

/** Verify decoded metadata and advancing real playback. @param {Page} page */
export async function verifyPlayback(page, audioId, buttonId, expectedDuration, tolerance) {
  await page.waitForFunction((id) => {
    const audio = document.getElementById(id);
    return audio instanceof HTMLAudioElement && Number.isFinite(audio.duration) && audio.readyState >= 2;
  }, audioId, { timeout: 15000 });
  const metadata = await readAudioState(page, audioId);
  assert.equal(metadata.errorCode, null, `Audio decoder failed: ${JSON.stringify(metadata)}`);
  assert.ok(metadata.source.startsWith('http://asset.localhost/'), `Playback is not using the real native asset protocol: ${metadata.source}`);
  assert.ok(Math.abs(metadata.duration - expectedDuration) <= tolerance, `Playback duration ${metadata.duration}s exceeds ${expectedDuration}s ± ${tolerance}s`);
  await page.locator(`#${buttonId}`).click();
  await page.waitForFunction((id) => {
    const audio = document.getElementById(id);
    return audio instanceof HTMLAudioElement && !audio.paused && audio.currentTime > 0.2;
  }, audioId, { timeout: 10000 });
  const playing = await readAudioState(page, audioId);
  assert.equal(playing.errorCode, null, `Real audio playback failed: ${JSON.stringify(playing)}`);
  await page.locator(`#${buttonId}`).click();
  assert.equal((await readAudioState(page, audioId)).paused, true, `Pause control failed: ${buttonId}`);
}

/** Audition a real generated join and verify that the bounded preview stops itself. @param {Page} page */
async function verifyJoinAudition(page) {
  const join = page.locator('#audition-join-1');
  await join.waitFor({ state: 'visible' });
  await join.click();
  assert.equal(await join.evaluate((button) => button.classList.contains('active')), true, 'Join audition did not activate');
  const started = await readAudioState(page, 'output-audio');
  await page.waitForFunction((start) => {
    const audio = document.getElementById('output-audio');
    return audio instanceof HTMLAudioElement && !audio.paused && audio.currentTime > start + 0.2;
  }, started.currentTime, { timeout: 3000 });
  await page.waitForFunction(() => {
    const audio = document.getElementById('output-audio');
    return audio instanceof HTMLAudioElement && audio.paused;
  }, null, { timeout: 6000 });
  const stopped = await readAudioState(page, 'output-audio');
  assert.equal(stopped.errorCode, null, `Join preview audio failed: ${JSON.stringify(stopped)}`);
  assert.equal(await join.evaluate((button) => button.classList.contains('active')), false, 'Join audition remained active after playback stopped');
  const playedSeconds = stopped.currentTime - started.currentTime;
  assert.ok(playedSeconds >= 3.5 && playedSeconds <= 4.5, `Join audition played ${playedSeconds}s instead of a bounded four-second preview`);
  return { startSeconds: started.currentTime, endSeconds: stopped.currentTime, playedSeconds };
}

/** Generate and export through the real native app UI. @param {Page} page @returns {Promise<WaveSummary>} */
export async function generateAndExport(page, processId, target, filePath, sampleRate, channels) {
  console.log(JSON.stringify({ event: 'native_generation_started', targetSeconds: target }));
  await page.locator('#target-duration').fill(String(target));
  await page.locator('#maximum-slack').fill('0.5');
  await page.locator('#generate-remix').click();
  await page.locator('#output-busy').waitFor({ state: 'hidden', timeout: 180000 });
  const notification = await page.locator('#notification-message').textContent();
  assert.equal(await page.locator('#output-content').isVisible(), true, `Native generation failed for target=${target}s: ${notification}`);
  await verifyPlayback(page, 'output-audio', 'play-output', target, 0.51);
  await page.locator('#export-remix').click();
  await confirmNativeDialog(processId, filePath, '1001');
  await page.waitForFunction(() => {
    const button = document.getElementById('export-remix');
    return button instanceof HTMLButtonElement && !button.disabled;
  }, null, { timeout: 15000 });
  const summary = await inspectWave(filePath, target, 0.51, sampleRate, channels);
  console.log(JSON.stringify({ event: 'native_export_verified', filePath, ...summary }));
  return summary;
}

/** Start the packaged app with its actual WebView2 and IPC bridge. */
export async function openNativeApp(runDirectory) {
  assert.equal(process.platform, 'win32', 'This smoke test requires the real Windows Tauri/WebView2 app');
  const configuredExecutable = process.env.REMOX_DESKTOP_EXECUTABLE;
  if (configuredExecutable !== undefined) assert.ok(configuredExecutable.trim().length > 0, 'REMOX_DESKTOP_EXECUTABLE must contain a native executable path');
  const executable = resolve(root, configuredExecutable === undefined ? 'src-tauri/target/release/remox-desktop.exe' : configuredExecutable);
  await access(executable);
  const port = await selectDebuggingPort();
  const address = `http://127.0.0.1:${port}`;
  const logPath = resolve(runDirectory, 'native-app.log');
  const logHandle = openSync(logPath, 'w');
  const app = spawn(executable, [], {
    cwd: root, windowsHide: true,
    env: {
      ...process.env,
      WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}`,
      WEBVIEW2_USER_DATA_FOLDER: resolve(runDirectory, 'webview2'),
    },
    stdio: ['ignore', logHandle, logHandle],
  });
  closeSync(logHandle);
  let browser = null;
  let page = null;
  try {
    await once(app, 'spawn');
    await writeFile(resolve(runDirectory, 'launch.json'), `${JSON.stringify({ processId: app.pid, address, executable }, null, 2)}\n`);
    await waitForWebView(app, address);
    browser = await chromium.connectOverCDP(address);
    const contexts = browser.contexts();
    assert.equal(contexts.length, 1, 'Expected one native WebView2 context');
    const pages = contexts[0].pages();
    assert.equal(pages.length, 1, 'Expected one native Remox app page');
    page = pages[0];
    page.setDefaultTimeout(15000);
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    await page.locator('#import-audio').waitFor({ state: 'visible' });
    assert.ok(page.url().startsWith('http://tauri.localhost/'), `Expected the packaged Tauri frontend: ${page.url()}`);
    assert.equal(await page.evaluate(() => typeof window.__TAURI_INTERNALS__.invoke), 'function', 'The actual Tauri IPC bridge is missing');
    return { app, browser, page, pageErrors, logPath };
  } catch (error) {
    await closeNativeApp(app, browser);
    throw error;
  }
}

/** Close only the native process tree created by this test. */
export async function closeNativeApp(app, browser) {
  if (app.exitCode === null) {
    await executeFile('taskkill.exe', ['/PID', String(app.pid), '/T', '/F'], { windowsHide: true, timeout: 10000 });
  }
  if (browser) await browser.close();
}

/** Capture reviewable artifacts without replacing the original test failure. @param {Page} page */
export async function captureNativeDiagnostics(page, runDirectory) {
  const diagnostics = await Promise.allSettled([
    page.screenshot({ path: resolve(runDirectory, 'failure.png'), timeout: 5000 }).then(() => resolve(runDirectory, 'failure.png')),
    page.locator('#notification-message').textContent({ timeout: 1000 }),
  ]);
  for (const result of diagnostics) {
    console.error(JSON.stringify({ event: 'failure_diagnostic', status: result.status, detail: result.status === 'fulfilled' ? result.value : String(result.reason) }));
  }
}

async function main() {
  const runDirectory = resolve(root, 'build/desktop-smoke', new Date().toISOString().replaceAll(/[:.]/g, '-'));
  await mkdir(runDirectory, { recursive: true });
  await mkdir(resolve(root, 'build/previews'), { recursive: true });
  const fixture = resolve(runDirectory, 'rhythmic-stereo.wav');
  await writeFixture(fixture, 44100, 48);
  const originalHash = createHash('sha256').update(await readFile(fixture)).digest('hex');
  const { app, browser, page, pageErrors, logPath } = await openNativeApp(runDirectory);
  try {
    await page.locator('#import-audio').click();
    await confirmNativeDialog(app.pid, fixture, '1148');
    console.log(JSON.stringify({ event: 'native_import_confirmed', fixture }));
    await page.locator('#generate-remix').waitFor({ state: 'visible' });
    await page.waitForFunction(() => {
      const button = document.getElementById('generate-remix');
      const notification = document.getElementById('notification');
      return (button instanceof HTMLButtonElement && !button.disabled)
        || (notification instanceof HTMLElement && !notification.hidden);
    }, null, { timeout: 120000 });
    assert.equal(await page.locator('#generate-remix').isEnabled(), true, `Native import failed: ${await page.locator('#notification-message').textContent()}`);
    assert.equal(await page.locator('#track-name').textContent(), 'rhythmic-stereo.wav', 'The native import did not load the selected fixture');
    await verifyPlayback(page, 'source-audio', 'play-source', 48, 0.01);
    const shortened = await generateAndExport(page, app.pid, 24, resolve(runDirectory, 'shortened.wav'), 44100, 2);
    const extended = await generateAndExport(page, app.pid, 72, resolve(runDirectory, 'extended.wav'), 44100, 2);
    const joinAudition = await verifyJoinAudition(page);
    assert.equal(createHash('sha256').update(await readFile(fixture)).digest('hex'), originalHash, 'Native import/generation/export modified the source audio');
    assert.deepEqual(pageErrors, [], 'The real app reported JavaScript errors');
    const screenshot = resolve(root, 'build/previews/desktop.png');
    await page.screenshot({ path: screenshot });
    const report = { fixture, originalHash, shortened, extended, joinAudition, screenshot, appLog: logPath };
    await writeFile(resolve(runDirectory, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({ event: 'desktop_smoke_passed', ...report }, null, 2));
  } catch (error) {
    console.error(JSON.stringify({ event: 'desktop_smoke_failed', error: error instanceof Error ? error.message : String(error), runDirectory }));
    await captureNativeDiagnostics(page, runDirectory);
    throw error;
  } finally {
    await closeNativeApp(app, browser);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();

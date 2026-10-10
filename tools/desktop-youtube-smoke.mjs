/** Real native YouTube integration using Kevin MacLeod's CC BY music.
 * Source: https://www.youtube.com/watch?v=LrM_Y39Gmhk
 * Composer/license: https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1400011
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';
import {
  captureNativeDiagnostics, closeNativeApp, confirmNativeDialog, generateAndExport,
  openNativeApp, readAudioState, root, verifyPlayback, writeFixture,
} from './desktop-smoke.mjs';

/** @typedef {import('@playwright/test').Page} Page */
/** @typedef {{name: string, previewUrl: string}} TrackIdentity */

const videoUrl = 'https://www.youtube.com/watch?v=LrM_Y39Gmhk';

/** @param {Page} page @returns {Promise<TrackIdentity>} */
async function readTrackIdentity(page) {
  const name = await page.locator('#track-name').textContent();
  assert.ok(typeof name === 'string' && name.length > 0, 'Native import returned no track name');
  const audio = await readAudioState(page, 'source-audio');
  assert.ok(audio.source.length > 0, 'Native source has no audio preview URL');
  return { name, previewUrl: audio.source };
}

/** Invalid URLs and cancelling the form must retain the currently imported source. @param {Page} page */
async function verifyYoutubeForm(page, previousTrack, screenshot) {
  await page.locator('#import-youtube').click();
  await page.locator('#youtube-dialog').waitFor({ state: 'visible' });
  await page.locator('#youtube-url').fill('https://example.com/watch?v=LrM_Y39Gmhk');
  await page.locator('#youtube-submit').click();
  await page.locator('#youtube-error').waitFor({ state: 'visible' });
  assert.equal(await page.locator('#youtube-url').getAttribute('aria-invalid'), 'true', 'Invalid URL was not marked invalid');
  const validationError = await page.locator('#youtube-error').textContent();
  assert.ok(validationError && validationError.trim().length > 0, 'Invalid URL has no actionable error');
  assert.deepEqual(await readTrackIdentity(page), previousTrack, 'Invalid YouTube URL changed the existing source');
  await page.screenshot({ path: screenshot });
  await page.locator('#youtube-cancel').click();
  await page.locator('#youtube-dialog').waitFor({ state: 'hidden' });
  await page.locator('#import-youtube').click();
  await page.locator('#youtube-url').fill(videoUrl);
  await page.locator('#youtube-cancel').click();
  await page.locator('#youtube-dialog').waitFor({ state: 'hidden' });
  assert.deepEqual(await readTrackIdentity(page), previousTrack, 'Cancelling the YouTube form changed the existing source');
  console.log(JSON.stringify({ event: 'youtube_validation_and_cancel_verified', validationError, previousTrack }));
  return validationError;
}

/** Download through the actual native command reached by the desktop UI. @param {Page} page */
async function importYoutube(page) {
  await page.locator('#import-youtube').click();
  await page.locator('#youtube-url').fill(videoUrl);
  await page.locator('#youtube-submit').click();
  await page.locator('#youtube-progress').waitFor({ state: 'visible' });
  assert.equal(await page.locator('#youtube-cancel').isDisabled(), true, 'YouTube controls were not disabled during native download');
  console.log(JSON.stringify({ event: 'native_youtube_download_started', videoUrl }));
  await page.waitForFunction(() => {
    const dialog = document.getElementById('youtube-dialog');
    const error = document.getElementById('youtube-error');
    return (dialog instanceof HTMLDialogElement && !dialog.open)
      || (error instanceof HTMLElement && !error.hidden);
  }, null, { timeout: 240000 });
  const error = await page.locator('#youtube-error').textContent();
  assert.equal(await page.locator('#youtube-dialog').isVisible(), false, `Actual YouTube import failed for ${videoUrl}: ${error}`);
  await page.locator('#output-busy').waitFor({ state: 'hidden' });
  assert.equal(await page.locator('#generate-remix').isEnabled(), true, 'Native YouTube import did not enable generation');
}

async function main() {
  const runDirectory = resolve(root, 'build/desktop-youtube-smoke', new Date().toISOString().replaceAll(/[:.]/g, '-'));
  await mkdir(runDirectory, { recursive: true });
  await mkdir(resolve(root, 'build/previews'), { recursive: true });
  const fixture = resolve(runDirectory, 'previous-source.wav');
  await writeFixture(fixture, 44100, 48);
  const fixtureHash = createHash('sha256').update(await readFile(fixture)).digest('hex');
  const { app, browser, page, pageErrors, logPath } = await openNativeApp(runDirectory);
  try {
    await page.locator('#import-audio').click();
    await confirmNativeDialog(app.pid, fixture, '1148');
    await page.waitForFunction(() => {
      const button = document.getElementById('generate-remix');
      const notification = document.getElementById('notification');
      return (button instanceof HTMLButtonElement && !button.disabled)
        || (notification instanceof HTMLElement && !notification.hidden);
    }, null, { timeout: 120000 });
    assert.equal(await page.locator('#generate-remix').isEnabled(), true, `Initial native import failed: ${await page.locator('#notification-message').textContent()}`);
    await verifyPlayback(page, 'source-audio', 'play-source', 48, 0.01);
    const previousTrack = await readTrackIdentity(page);
    const validationError = await verifyYoutubeForm(page, previousTrack, resolve(root, 'build/previews/desktop-youtube-invalid.png'));
    await importYoutube(page);
    await verifyPlayback(page, 'source-audio', 'play-source', 130, 0.51);
    const importedTrack = await readTrackIdentity(page);
    assert.notDeepEqual(importedTrack, previousTrack, 'Successful YouTube import did not replace the source');
    const sourceAudio = await readAudioState(page, 'source-audio');
    assert.ok(sourceAudio.duration > 120 && sourceAudio.duration < 140, `Unexpected duration for selected YouTube music: ${sourceAudio.duration}`);
    const previewUrl = new URL(sourceAudio.source);
    assert.equal(previewUrl.origin, 'http://asset.localhost', 'YouTube preview did not use the actual native asset protocol');
    const sourcePath = decodeURIComponent(previewUrl.pathname.slice(1));
    assert.ok(isAbsolute(sourcePath), `Native YouTube preview path is not absolute: ${sourcePath}`);
    const sourceBytes = await readFile(sourcePath);
    assert.equal(sourceBytes.toString('ascii', 0, 4), 'RIFF', 'Native YouTube preview is not WAV');
    const channels = sourceBytes.readUInt16LE(22);
    const sampleRate = sourceBytes.readUInt32LE(24);
    assert.ok(channels === 1 || channels === 2, `Unexpected imported YouTube channels: ${channels}`);
    assert.ok(sampleRate >= 8000 && sampleRate <= 192000, `Unexpected imported YouTube sample rate: ${sampleRate}`);
    const sourceHash = createHash('sha256').update(sourceBytes).digest('hex');
    console.log(JSON.stringify({ event: 'native_youtube_import_and_playback_verified', videoUrl, importedTrack, duration: sourceAudio.duration, sampleRate, channels }));
    const exported = await generateAndExport(page, app.pid, 60, resolve(runDirectory, 'youtube-remix.wav'), sampleRate, channels);
    assert.equal(createHash('sha256').update(await readFile(sourcePath)).digest('hex'), sourceHash, 'Remixing modified the native YouTube source audio');
    assert.equal(createHash('sha256').update(await readFile(fixture)).digest('hex'), fixtureHash, 'YouTube import modified the prior source file');
    assert.deepEqual(pageErrors, [], 'The native YouTube workflow reported JavaScript errors');
    const screenshot = resolve(root, 'build/previews/desktop-youtube.png');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: screenshot, fullPage: true });
    const report = {
      videoUrl, creator: 'Kevin MacLeod', title: importedTrack.name,
      licenseSource: 'https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1400011',
      sourceDuration: sourceAudio.duration, sourceHash, sampleRate, channels,
      validationError, cancellationPreservedSource: true, exported, screenshot, appLog: logPath,
    };
    await writeFile(resolve(runDirectory, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({ event: 'desktop_youtube_smoke_passed', ...report }, null, 2));
  } catch (error) {
    console.error(JSON.stringify({ event: 'desktop_youtube_smoke_failed', error: error instanceof Error ? error.message : String(error), runDirectory }));
    await captureNativeDiagnostics(page, runDirectory);
    console.error(JSON.stringify({ event: 'youtube_error_diagnostic', message: await page.locator('#youtube-error').textContent({ timeout: 1000 }) }));
    throw error;
  } finally {
    await closeNativeApp(app, browser);
  }
}

await main();

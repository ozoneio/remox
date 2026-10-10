# Remøx

Remox is a Tauri 2 desktop app with a Rust remix engine. Import a local music
file or a YouTube link, choose a target duration, preview the arrangement and
its joins, then export lossless WAV. Audio processing stays on your computer.
Use **YouTube** beside **Import audio** to paste a single video link. The app
downloads its audio, then uses the same waveform, preview and remix controls.
Failed imports preserve your current source and arrangement.

```powershell
pnpm install --frozen-lockfile
pnpm desktop          # develop the native app
pnpm desktop:build    # build the Windows executable and installer
```

Windows development needs Node.js, pnpm, Rust, Microsoft C++ Build Tools and
WebView2; see [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/).
The executable is `src-tauri/target/release/remox-desktop.exe`; installers are
in `src-tauri/target/release/bundle/nsis/`. `./build.ps1` installs dependencies
and builds the desktop app. Pushing a version tag such as `v0.2.0` runs the
Windows release workflow and publishes the installer to GitHub Releases.

Run `pnpm check`, `pnpm test` and `pnpm build` to validate the desktop types,
Rust engine and frontend build. `pnpm check:desktop` and `pnpm test:core` run
the same type checks and engine tests directly. After building,
`pnpm test:desktop` exercises native file import, playback and export;
`pnpm test:youtube` exercises an actual YouTube download and remix.
Set `REMOX_DESKTOP_EXECUTABLE` to test a particular executable.

Desktop commands prepare pinned, checksum-verified download tools and bundle
them with the app. YouTube import requires an internet connection; audio
analysis and remixing run locally.

The engine independently implements the publicly described duration-remix
workflow: beat analysis, content matching, skipped/repeated passages, minimum
segment length and target tolerance. Its source and design provenance are in
`src-tauri/src/engine/`; it does not use Adobe code, SDKs or models, and does not
claim identical output. The engine preserves opening/ending samples and tempo;
it reports an error when it cannot meet the requested tolerance. It targets
steady-tempo instrumental music; vocals and changing meters need listening
review. Pitch-preserving exact-duration stretching is not implemented.

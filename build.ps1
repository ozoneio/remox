[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = $PSScriptRoot
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    throw "pnpm is required to build Remox. Install it from https://pnpm.io/installation"
}

Push-Location $repositoryRoot
try {
    Write-Host "Building Remox desktop executable and installer..."
    & pnpm install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed with exit code $LASTEXITCODE" }
    & pnpm desktop:build
    if ($LASTEXITCODE -ne 0) { throw "Desktop build failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "Build complete. Artifacts are in $(Join-Path $repositoryRoot 'src-tauri\target\release')."

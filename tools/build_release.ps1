param()

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$projectFile = Join-Path $repositoryRoot "pyproject.toml"
$versionMatch = Select-String -LiteralPath $projectFile -Pattern '^version = "([^"]+)"$'

if (-not $versionMatch) {
    throw "Could not read the project version from pyproject.toml"
}

$version = $versionMatch.Matches[0].Groups[1].Value
$artifactName = "remox-v$version-windows-x86_64"
$buildDirectory = Join-Path $repositoryRoot "build"

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null

$env:UV_CACHE_DIR = Join-Path $buildDirectory ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $buildDirectory ".uv-python"

Push-Location $repositoryRoot
try {
    uv run --isolated --python 3.12 `
        --with pyinstaller `
        --with-editable ".[youtube]" `
        pyinstaller `
        --noconfirm `
        --clean `
        --onefile `
        --console `
        --name $artifactName `
        --distpath $buildDirectory `
        --workpath (Join-Path $buildDirectory "pyinstaller") `
        --specpath $buildDirectory `
        --runtime-hook (Join-Path $repositoryRoot "tools\pyinstaller_runtime_hook.py") `
        --collect-all imageio_ffmpeg `
        (Join-Path $repositoryRoot "src\remox\__main__.py")

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$artifact = Join-Path $buildDirectory "$artifactName.exe"
$checksum = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$artifact.sha256" -Encoding ascii -Value "$checksum  $artifactName.exe"

Write-Host "Built $artifact"
Write-Host "SHA256 $checksum"

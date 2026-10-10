# Build-only acquisition of pinned Windows tools. Runtime never downloads executables.
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $ExpectedSha256) { throw "Cached tool checksum mismatch at $Destination. Expected $ExpectedSha256; found $actual. Remove the corrupt file and rebuild." }
        return
    }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Destination -TimeoutSec 180 -Headers @{ 'User-Agent' = 'remox-build' }
            break
        }
        catch [Microsoft.PowerShell.Commands.HttpResponseException], [System.Net.Http.HttpRequestException], [System.Threading.Tasks.TaskCanceledException] {
            Write-Warning ([PSCustomObject]@{ event = 'media_tool_download_failed'; attempt = $attempt; attempts = 3; url = $Url; error = $_.Exception.Message } | ConvertTo-Json -Compress)
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds $attempt
        }
    }
    $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256) { throw "Downloaded tool checksum mismatch: URL=$Url; expected=$ExpectedSha256; actual=$actual; path=$Destination" }
}

function Copy-ArchiveEntry {
    param(
        [Parameter(Mandatory = $true)][System.IO.Compression.ZipArchive]$Archive,
        [Parameter(Mandatory = $true)][string]$EntryPath,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    $entry = $Archive.GetEntry($EntryPath)
    if ($null -eq $entry) { throw "Required media tool archive entry is missing: $EntryPath" }
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Destination, $true)
}

$repository = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$cache = Join-Path $repository 'build\media-tools'
$destination = Join-Path $repository 'src-tauri\resources\media-tools'
New-Item -ItemType Directory -Path $cache, $destination -Force | Out-Null
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'media-tools.json') -Raw | ConvertFrom-Json

foreach ($tool in $manifest.downloads) {
    if ([System.IO.Path]::GetFileName($tool.name) -ne $tool.name) { throw "Tool destination must be a filename: $($tool.name)" }
    $archivePath = Join-Path $cache $tool.name
    Get-VerifiedDownload -Url $tool.url -Destination $archivePath -ExpectedSha256 $tool.sha256
    Copy-Item -LiteralPath $archivePath -Destination (Join-Path $destination $tool.name) -Force
}
foreach ($tool in $manifest.archives) {
    if ([System.IO.Path]::GetFileName($tool.name) -ne $tool.name) { throw "Archive destination must be a filename: $($tool.name)" }
    $archivePath = Join-Path $cache $tool.name
    Get-VerifiedDownload -Url $tool.url -Destination $archivePath -ExpectedSha256 $tool.sha256
    $archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        foreach ($entry in $tool.entries) {
            if ([System.IO.Path]::GetFileName($entry.name) -ne $entry.name) { throw "Archive output must be a filename: $($entry.name)" }
            Copy-ArchiveEntry -Archive $archive -EntryPath $entry.path -Destination (Join-Path $destination $entry.name)
        }
    }
    finally { $archive.Dispose() }
}
Write-Output "Prepared pinned YouTube media tools in $destination"

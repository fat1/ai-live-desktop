# Base toolchain setup for the dev server: uv, Python 3.10, Git, ffmpeg.
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads BOM-less
# scripts as ANSI, so non-ASCII text would be corrupted.
# Usage: powershell -ExecutionPolicy Bypass -File setup_server.ps1
# Re-running is safe: finished steps are skipped.
# GitHub release downloads stall from the server network. If uv.zip / ffmpeg.zip
# are missing, download them elsewhere and copy them into C:\ai_live\downloads first.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = 'C:\ai_live'
$Tools = Join-Path $Root 'tools'
$Downloads = Join-Path $Root 'downloads'
foreach ($dir in $Root, $Tools, $Downloads, "$Root\repos", "$Root\models", "$Root\envs") {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function Get-RemoteFile {
    param([string]$Url, [string]$OutFile)
    if (Test-Path $OutFile) { return }
    Write-Host "Downloading $Url"
    curl.exe -L --fail --retry 3 --silent --show-error -o "$OutFile.part" $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    Move-Item -Force "$OutFile.part" $OutFile
}

function Add-MachinePath {
    param([string]$Dir)
    $current = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    if (($current -split ';') -notcontains $Dir) {
        [Environment]::SetEnvironmentVariable('Path', "$current;$Dir", 'Machine')
    }
    if (($env:Path -split ';') -notcontains $Dir) { $env:Path = "$env:Path;$Dir" }
}

# --- uv -------------------------------------------------------------------
$UvDir = Join-Path $Tools 'uv'
if (-not (Test-Path "$UvDir\uv.exe")) {
    $zip = Join-Path $Downloads 'uv.zip'
    Get-RemoteFile 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' $zip
    Expand-Archive -Force $zip $UvDir
}
Add-MachinePath $UvDir
# PyPI mirror inside China, and keep uv caches/pythons on a known path.
[Environment]::SetEnvironmentVariable('UV_DEFAULT_INDEX', 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple', 'Machine')
[Environment]::SetEnvironmentVariable('UV_PYTHON_INSTALL_DIR', "$Tools\python", 'Machine')
[Environment]::SetEnvironmentVariable('UV_CACHE_DIR', "$Root\uv_cache", 'Machine')
$env:UV_DEFAULT_INDEX = 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'
$env:UV_PYTHON_INSTALL_DIR = "$Tools\python"
$env:UV_CACHE_DIR = "$Root\uv_cache"
# python-build-standalone mirror (the default source is a GitHub release).
[Environment]::SetEnvironmentVariable('UV_PYTHON_INSTALL_MIRROR', 'https://registry.npmmirror.com/-/binary/python-build-standalone', 'Machine')
$env:UV_PYTHON_INSTALL_MIRROR = 'https://registry.npmmirror.com/-/binary/python-build-standalone'

# --- Python 3.10 ----------------------------------------------------------
& "$UvDir\uv.exe" python install 3.10
if ($LASTEXITCODE -ne 0) { throw 'uv python install failed' }

# --- Git ------------------------------------------------------------------
if (-not (Test-Path 'C:\Program Files\Git\cmd\git.exe')) {
    # Pinned version served from npmmirror; GitHub release downloads stall from this network.
    $installer = Join-Path $Downloads 'Git-2.47.1-64-bit.exe'
    Get-RemoteFile 'https://registry.npmmirror.com/-/binary/git-for-windows/v2.47.1.windows.1/Git-2.47.1-64-bit.exe' $installer
    Start-Process -Wait -FilePath $installer -ArgumentList '/VERYSILENT', '/NORESTART', '/NOCANCEL', '/SP-'
}
Add-MachinePath 'C:\Program Files\Git\cmd'

# --- ffmpeg ---------------------------------------------------------------
$FfmpegDir = Join-Path $Tools 'ffmpeg'
if (-not (Test-Path "$FfmpegDir\bin\ffmpeg.exe")) {
    $zip = Join-Path $Downloads 'ffmpeg.zip'
    Get-RemoteFile 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip' $zip
    $tmp = Join-Path $Downloads 'ffmpeg_extract'
    Expand-Archive -Force $zip $tmp
    $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
    if (Test-Path $FfmpegDir) { Remove-Item -Recurse -Force $FfmpegDir }
    Move-Item $inner.FullName $FfmpegDir
    Remove-Item -Recurse -Force $tmp
}
Add-MachinePath "$FfmpegDir\bin"

# --- Microsoft VC++ runtime -------------------------------------------------
# llama.cpp crashes on start with the old runtime shipped in the server image.
$redist = Join-Path $Downloads 'vc_redist.x64.exe'
Get-RemoteFile 'https://aka.ms/vs/17/release/vc_redist.x64.exe' $redist
Start-Process -Wait -FilePath $redist -ArgumentList '/install', '/quiet', '/norestart'

# --- MSVC build tools -------------------------------------------------------
# Needed to compile Python packages without Windows wheels (jieba_fast for GPT-SoVITS).
$VsRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
if (-not (Test-Path "$VsRoot\VC\Tools\MSVC")) {
    $bootstrap = Join-Path $Downloads 'vs_BuildTools.exe'
    Get-RemoteFile 'https://aka.ms/vs/17/release/vs_BuildTools.exe' $bootstrap
    $proc = Start-Process -Wait -PassThru -FilePath $bootstrap -ArgumentList '--quiet', '--wait', '--norestart', '--nocache',
        '--add', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '--add', 'Microsoft.VisualStudio.Component.Windows11SDK.22621'
    Write-Host "VS Build Tools installer exit code: $($proc.ExitCode)"
}

# --- Report ---------------------------------------------------------------
Write-Host '--- versions ---'
& "$UvDir\uv.exe" --version
& "$UvDir\uv.exe" python list --only-installed
& 'C:\Program Files\Git\cmd\git.exe' --version
& "$FfmpegDir\bin\ffmpeg.exe" -version | Select-Object -First 1

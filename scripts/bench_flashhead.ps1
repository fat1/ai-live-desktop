# Run SoulX-FlashHead Lite once on the dev server; report per-chunk time and peak GPU memory.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
# Install first with: setup_envs.ps1 -Only flashhead
# Usage: powershell -ExecutionPolicy Bypass -File bench_flashhead.ps1 -Image C:\ai_live\out\flashhead\meg.png `
#            -Audio C:\ai_live\out\flashhead\speech16k.wav -Out C:\ai_live\out\flashhead\meg_lite.mp4
# The input is a single image (the model does not take video); the output is a 512x512 head video.
# The first chunk includes torch.compile time (several minutes), so it is reported separately.

param(
    [Parameter(Mandatory)][string]$Image,
    [Parameter(Mandatory)][string]$Audio,
    [Parameter(Mandatory)][string]$Out,
    [switch]$NoFaceCrop
)

$Root = 'C:\ai_live'
$Repo = "$Root\repos\SoulX-FlashHead"
$Py = "$Root\envs\flashhead\Scripts\python.exe"
$Log = [IO.Path]::ChangeExtension($Out, '.txt')
$Peak = [IO.Path]::ChangeExtension($Out, '.vram.txt')

# Single-GPU stand-ins for Linux-only / audio-device imports; see flashhead_stubs.
$env:PYTHONPATH = "$PSScriptRoot\flashhead_stubs"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:Path = "$Root\tools\ffmpeg\bin;$env:Path"

# The upstream code splits paths on "/", so pass forward slashes.
$Image = $Image -replace '\\', '/'
$Audio = $Audio -replace '\\', '/'
$OutArg = $Out -replace '\\', '/'
$crop = if ($NoFaceCrop) { '' } else { '--use_face_crop True' }

$monitor = Start-Job -ArgumentList $Peak {
    param($file)
    $max = 0
    while ($true) {
        $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if ($used -gt $max) { $max = $used; Set-Content $file $max }
        Start-Sleep -Milliseconds 500
    }
}

$started = Get-Date
Push-Location $Repo   # the code reads its config through a relative path
cmd /c "$Py generate_video.py --ckpt_dir models/SoulX-FlashHead-1_3B --wav2vec_dir models/wav2vec2-base-960h --model_type lite --cond_image $Image --audio_path $Audio --audio_encode_mode stream $crop --save_file $OutArg > $Log 2>&1"
$code = $LASTEXITCODE
Pop-Location
Stop-Job $monitor
Remove-Job $monitor

"exit code $code, wall time {0:N0} s" -f ((Get-Date) - $started).TotalSeconds
if ($code -ne 0) { Get-Content $Log -Tail 15; exit $code }

$times = Get-Content $Log | Where-Object { $_ -match 'chunk-\d+ done, cost time: ([\d.]+)s' } | ForEach-Object { [double]$Matches[1] }
$rest = $times | Select-Object -Skip 1 | Sort-Object
"first chunk (includes compilation): {0:N1} s" -f $times[0]
if ($rest) {
    $median = $rest[[int](($rest.Count - 1) / 2)]
    # Lite generates 24 new frames per chunk at 25 fps.
    "later chunks: {0}, median {1:N3} s, worst {2:N3} s per 24 frames (about {3:N0} fps)" -f $rest.Count, $median, $rest[-1], (24 / $median)
}
"gpu memory peak: $(Get-Content $Peak) MiB"
"output: $Out"

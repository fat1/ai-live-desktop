# Run LLM, TTS and lip sync at full load at the same time and report speed and GPU memory.
# TTS and lip sync run continuously; the LLM is queried every -LlmInterval seconds.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
# Usage: powershell -ExecutionPolicy Bypass -File bench_concurrent.ps1 [-Backend vulkan|cuda]

param(
    [ValidateSet('vulkan', 'cuda')][string]$Backend = 'vulkan',
    [string]$BatchSizes = '8',
    # Seconds between LLM requests. 0 = saturate the LLM (worst case); 3 is close to a real stream.
    [double]$LlmInterval = 3
)

$Root = 'C:\ai_live'
$Out = "$Root\bench_concurrent\$Backend"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Remove-Item "$Out\*" -Force -ErrorAction SilentlyContinue
$LlamaDir = if ($Backend -eq 'cuda') { "$Root\tools\llama_cuda" } else { "$Root\tools\llama" }
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:Path = "$Root\tools\ffmpeg_shared\bin;$env:Path"

function Get-VramUsed { [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) }

$started = @()
try {
    Write-Host "vram idle: $(Get-VramUsed) MiB"

    # --- LLM server ---
    $started += Start-Process -PassThru -WindowStyle Hidden -FilePath "$LlamaDir\llama-server.exe" `
        -ArgumentList '-m', "$Root\models\llm\Qwen3-4B-Q4_K_M.gguf", '-ngl', '99', '-c', '4096', '--port', '8089', '--jinja' `
        -RedirectStandardOutput "$Out\llama_out.txt" -RedirectStandardError "$Out\llama_err.txt"
    foreach ($i in 1..120) {
        $health = curl.exe -s -o NUL -w '%{http_code}' http://127.0.0.1:8089/health
        if ($health -eq '200') { break }
        Start-Sleep -Seconds 1
    }
    if ($health -ne '200') { throw 'llama-server did not become ready' }
    Write-Host "vram with LLM loaded: $(Get-VramUsed) MiB"

    # --- GPU memory sampler ---
    $sampler = Start-Job -ArgumentList "$Out\vram_peak.txt" {
        param($file)
        $peak = 0
        while ($true) {
            $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
            if ($used -gt $peak) { $peak = $used; Set-Content $file $peak }
            Start-Sleep -Milliseconds 500
        }
    }

    # --- TTS loop ---
    $started += Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory "$Root\repos\GPT-SoVITS" `
        -FilePath "$Root\envs\sovits\Scripts\python.exe" `
        -ArgumentList "$Root\scripts\bench_tts.py", '--ref_audio', "$Root\bench_tts\ref.wav", '--ref_lang', 'en', `
            '--ref_text', '"Before I chased the way the wind would go, now I hunt a storm of darkness."', `
            '--out_dir', $Out, '--repeat', '40' `
        -RedirectStandardOutput "$Out\tts_out.txt" -RedirectStandardError "$Out\tts_err.txt"
    foreach ($i in 1..300) {
        if ((Test-Path "$Out\tts_out.txt") -and (Select-String -Path "$Out\tts_out.txt" -Pattern '^idx' -Quiet)) { break }
        Start-Sleep -Seconds 1
    }
    Write-Host "vram with LLM + TTS loaded: $(Get-VramUsed) MiB"

    # --- LLM request loop ---
    $started += Start-Process -PassThru -WindowStyle Hidden -FilePath "$Root\envs\sovits\Scripts\python.exe" `
        -ArgumentList "$Root\scripts\bench_llm_loop.py", '--seconds', '600', '--interval', $LlmInterval `
        -RedirectStandardOutput "$Out\llm_out.txt" -RedirectStandardError "$Out\llm_err.txt"

    # --- Lip sync benchmark in the foreground ---
    Push-Location "$Root\repos\MuseTalk"
    cmd /c "$Root\envs\musetalk\Scripts\python.exe $Root\scripts\bench_musetalk.py --avatar results\v15\avatars\avator_1 --audio data\audio\eng.wav --batch_sizes $BatchSizes --max_frames 1000 > $Out\musetalk_out.txt 2>&1"
    Pop-Location

    Write-Host "vram peak during concurrent run: $(Get-Content "$Out\vram_peak.txt") MiB"
    Write-Host '--- lip sync (while TTS and LLM are busy) ---'
    Get-Content "$Out\musetalk_out.txt" | Where-Object { $_ -match '^(batch| +\d+ \|)|Error|Traceback' }
    Write-Host '--- TTS (rows finished so far) ---'
    $rows = Get-Content "$Out\tts_out.txt" | Where-Object { $_ -match '^\s*\d+ \|' } | Select-Object -Skip 5
    $audio = 0.0; $synth = 0.0
    foreach ($row in $rows) { $cols = $row -split '\|'; $audio += [double]$cols[2]; $synth += [double]$cols[3] }
    if ($audio -gt 0) { Write-Host ('{0} sentences, {1:N1}s audio in {2:N1}s, RTF {3:N2}' -f $rows.Count, $audio, $synth, ($synth / $audio)) }
    Write-Host '--- LLM ---'
    Get-Content "$Out\llm_out.txt" -Encoding UTF8 -ErrorAction SilentlyContinue | Select-Object -Last 1
} finally {
    if ($sampler) { Stop-Job $sampler; Remove-Job $sampler }
    foreach ($proc in $started) { if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force } }
}

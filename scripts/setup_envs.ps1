# Create the two Python environments and download model weights on the dev server.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
# Usage: powershell -ExecutionPolicy Bypass -File setup_envs.ps1 [-Only musetalk|sovits|llm|cosyvoice|flashhead]
# Two separate environments are required: MuseTalk pins transformers==4.39 / torch 2.0.1,
# while GPT-SoVITS needs transformers>=4.51 and a recent torch.
# cosyvoice and flashhead are candidates that were evaluated and not adopted; they are
# installed only when named with -Only, never by the default 'all'.
# Each step is skipped when its marker file exists, so re-running is safe.

param([ValidateSet('all', 'musetalk', 'sovits', 'cosyvoice', 'flashhead', 'llm')][string]$Only = 'all')

$ProgressPreference = 'SilentlyContinue'
$Root = 'C:\ai_live'
$Uv = "$Root\tools\uv\uv.exe"
$Repos = "$Root\repos"
$Envs = "$Root\envs"
$Markers = "$Root\markers"
New-Item -ItemType Directory -Force -Path $Markers, "$Root\models\llm" | Out-Null

$env:UV_DEFAULT_INDEX = 'https://mirrors.cloud.tencent.com/pypi/simple'
$env:PIP_INDEX_URL = 'https://mirrors.cloud.tencent.com/pypi/simple'
$env:UV_PYTHON_INSTALL_DIR = "$Root\tools\python"
$env:UV_CACHE_DIR = "$Root\uv_cache"
$env:UV_HTTP_TIMEOUT = '300'
$env:HF_ENDPOINT = 'https://hf-mirror.com'
$env:Path = "$Root\tools\ffmpeg\bin;C:\Program Files\Git\cmd;$env:Path"

function Invoke-Step {
    # Run a named step once; a marker file records success.
    param([string]$Name, [scriptblock]$Body)
    $marker = Join-Path $Markers "$Name.done"
    if (Test-Path $marker) { Write-Host "[skip] $Name"; return }
    Write-Host "[run ] $Name"
    $global:LASTEXITCODE = 0
    try {
        & $Body
        if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
        New-Item -ItemType File -Force -Path $marker | Out-Null
        Write-Host "[ ok ] $Name"
    } catch {
        Write-Host "[FAIL] $Name : $_"
    }
}

function Get-RemoteFile {
    # Large downloads are often cut mid-transfer; resume until complete.
    param([string]$Url, [string]$OutFile)
    foreach ($attempt in 1..30) {
        curl.exe -L --fail -C - --silent --show-error -o $OutFile $Url
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "download interrupted (curl exit $LASTEXITCODE), resuming, attempt $attempt"
        Start-Sleep -Seconds 3
    }
}

# ---------------------------------------------------------------- MuseTalk
if ($Only -in 'all', 'musetalk') {
    $Py = "$Envs\musetalk\Scripts\python.exe"
    $Repo = "$Repos\MuseTalk"

    Invoke-Step 'musetalk_venv' {
        & $Uv venv --python 3.10 --seed "$Envs\musetalk"
    }
    Invoke-Step 'musetalk_torch' {
        & $Uv pip install --python $Py 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' 'torchaudio==2.0.2+cu118' `
            --index https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
    }
    Invoke-Step 'musetalk_requirements' {
        # tensorflow / tensorboard / gradio / gdown are only for training or the demo UI.
        $req = "$Root\downloads\musetalk_requirements.txt"
        Get-Content "$Repo\requirements.txt" |
            Where-Object { $_ -notmatch '^(tensorflow|tensorboard|gradio|gdown)' } |
            Set-Content -Encoding ascii $req
        & $Uv pip install --python $Py -r $req 'setuptools<81' 'torch==2.0.1+cu118' `
            --index https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
    }
    Invoke-Step 'musetalk_mmlab' {
        & $Py -m pip install -U openmim
        & $Py -m mim install mmengine
        & $Py -m mim install 'mmcv==2.0.1'
        & $Py -m mim install 'mmdet==3.1.0'
        # mmpose depends on chumpy, which cannot be built with current pip and is only
        # used for SMPL body models. Install mmpose without deps and add the rest by hand.
        & $Py -m pip install --no-deps 'mmpose==1.1.0'
        & $Py -m pip install json_tricks munkres xtcocotools scipy matplotlib 'numpy==1.23.5'
    }
    Invoke-Step 'musetalk_weights' {
        $m = "$Repo\models"
        $hf = "$Envs\musetalk\Scripts\huggingface-cli.exe"
        & $hf download TMElyralab/MuseTalk --local-dir $m
        if ($LASTEXITCODE -ne 0) { throw 'MuseTalk weights' }
        & $hf download stabilityai/sd-vae-ft-mse --local-dir "$m\sd-vae" --include config.json diffusion_pytorch_model.bin
        if ($LASTEXITCODE -ne 0) { throw 'sd-vae' }
        & $hf download openai/whisper-tiny --local-dir "$m\whisper" --include config.json pytorch_model.bin preprocessor_config.json
        if ($LASTEXITCODE -ne 0) { throw 'whisper' }
        & $hf download yzd-v/DWPose --local-dir "$m\dwpose" --include dw-ll_ucoco_384.pth
        if ($LASTEXITCODE -ne 0) { throw 'dwpose' }
        & $hf download ManyOtherFunctions/face-parse-bisent --local-dir "$m\face-parse-bisent" --include 79999_iter.pth resnet18-5c106cde.pth
    }
    Invoke-Step 'musetalk_check' {
        & $Py -c "import torch, mmpose, mmcv, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('mmcv', mmcv.__version__, 'mmpose', mmpose.__version__)"
    }
}

# -------------------------------------------------------------- GPT-SoVITS
if ($Only -in 'all', 'sovits') {
    $Py = "$Envs\sovits\Scripts\python.exe"
    $Repo = "$Repos\GPT-SoVITS"
    $Src = 'https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master'

    Invoke-Step 'sovits_venv' {
        & $Uv venv --python 3.10 --seed "$Envs\sovits"
    }
    Invoke-Step 'sovits_torch' {
        & $Uv pip install --python $Py torch torchaudio torchcodec `
            --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
    }
    Invoke-Step 'sovits_requirements' {
        # Drop "--no-binary=opencc": building it needs a C++ toolchain; the wheel works on Windows.
        $req = "$Root\downloads\sovits_requirements.txt"
        Get-Content "$Repo\requirements.txt" | Where-Object { $_ -notmatch '^--no-binary' } |
            Set-Content -Encoding ascii $req
        & $Uv pip install --python $Py --no-deps -r "$Repo\extra-req.txt"
        if ($LASTEXITCODE -ne 0) { throw 'extra-req' }
        # resampy is imported by TTS_infer_pack but missing from upstream requirements.
        & $Uv pip install --python $Py -r $req resampy `
            --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
    }
    Invoke-Step 'sovits_lipsync_runtime' {
        # MuseTalk's runtime (not its preprocessing) also runs in this environment, so that
        # TTS and lip sync share one process and one CUDA context. huggingface-hub must stay
        # below 1.0 for transformers 4.x; triton-windows is required by torch.compile.
        & $Uv pip install --python $Py torchvision diffusers einops omegaconf opencv-python accelerate `
            triton-windows 'huggingface-hub<1.0' `
            --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
    }
    Invoke-Step 'preset_tts_kokoro' {
        # Preset (non-cloned) voices: Kokoro, a small and fast TTS model with built-in Chinese voices.
        & $Uv pip install --python $Py kokoro 'misaki[zh]' 'huggingface-hub<1.0'
        if ($LASTEXITCODE -ne 0) { throw 'kokoro install' }
        $env:PYTHONUTF8 = '1'  # the hf CLI prints non-GBK characters
        & "$Envs\sovits\Scripts\huggingface-cli.exe" download hexgrad/Kokoro-82M-v1.1-zh --local-dir "$Root\models\kokoro-zh"
    }
    Invoke-Step 'sovits_pretrained' {
        $zip = "$Root\downloads\sovits_pretrained_models.zip"
        Get-RemoteFile "$Src/pretrained_models.zip" $zip
        if ($LASTEXITCODE -ne 0) { throw 'download' }
        Expand-Archive -Force $zip "$Repo\GPT_SoVITS"
        Remove-Item $zip
    }
    Invoke-Step 'sovits_g2pw' {
        $zip = "$Root\downloads\G2PWModel.zip"
        Get-RemoteFile "$Src/G2PWModel.zip" $zip
        if ($LASTEXITCODE -ne 0) { throw 'download' }
        Expand-Archive -Force $zip "$Repo\GPT_SoVITS\text"
        Remove-Item $zip
    }
    Invoke-Step 'sovits_nltk' {
        $zip = "$Root\downloads\nltk_data.zip"
        Get-RemoteFile "$Src/nltk_data.zip" $zip
        if ($LASTEXITCODE -ne 0) { throw 'download' }
        Expand-Archive -Force $zip "$Envs\sovits"
        Remove-Item $zip
    }
    Invoke-Step 'sovits_openjtalk' {
        $tar = "$Root\downloads\open_jtalk_dic.tar.gz"
        Get-RemoteFile "$Src/open_jtalk_dic_utf_8-1.11.tar.gz" $tar
        if ($LASTEXITCODE -ne 0) { throw 'download' }
        $target = (& $Py -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))").Trim()
        tar -xzf $tar -C $target
        Remove-Item $tar
    }
    Invoke-Step 'sovits_check' {
        & $Py -c "import torch, torchaudio, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"
    }
}

# ------------------------------------------------------- mouth enhancement
if ($Only -in 'all', 'sovits') {
    Invoke-Step 'enhance_weights' {
        # Small super-resolution net for the mouth region (Real-ESRGAN compact, BSD-3, 5 MB).
        # GitHub release downloads stall from the server network; if this fails, download the
        # file elsewhere and copy it to C:\ai_live\models\sr\.
        New-Item -ItemType Directory -Force -Path "$Root\models\sr" | Out-Null
        Get-RemoteFile 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth' "$Root\models\sr\realesr-general-x4v3.pth"
    }
}

# --------------------------------------------------------------- CosyVoice
# Candidate TTS evaluated for expressiveness (emotion instructions, streaming output). Not adopted.
if ($Only -eq 'cosyvoice') {
    $Py = "$Envs\cosyvoice\Scripts\python.exe"
    $Repo = "$Repos\CosyVoice"
    $Git = 'C:\Program Files\Git\cmd\git.exe'

    Invoke-Step 'cosyvoice_repo' {
        if (-not (Test-Path $Repo)) { & $Git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git $Repo }
        & $Git -C $Repo submodule update --init --recursive --depth 1
    }
    Invoke-Step 'cosyvoice_venv' {
        & $Uv venv --python 3.10 --seed "$Envs\cosyvoice"
    }
    Invoke-Step 'cosyvoice_requirements' {
        # Drop the extra index lines (uv takes indexes from the command line) and Linux-only pins.
        $req = "$Root\downloads\cosyvoice_requirements.txt"
        Get-Content "$Repo\requirements.txt" |
            Where-Object { $_ -notmatch '^--extra-index-url' -and $_ -notmatch "sys_platform == 'linux'" } |
            Set-Content -Encoding ascii $req
        # openai-whisper's setup.py imports pkg_resources, which setuptools 81+ no longer ships.
        $constraints = "$Root\downloads\build_constraints.txt"
        Set-Content -Encoding ascii $constraints 'setuptools<81'
        # 'setuptools<81' is also needed at run time: CosyVoice imports pkg_resources.
        & $Uv pip install --python $Py -r $req 'setuptools<81' --build-constraints $constraints `
            --index https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
    }
    Invoke-Step 'cosyvoice_model' {
        $env:PYTHONUTF8 = '1'
        & $Py -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir=r'$Root\models\Fun-CosyVoice3-0.5B')"
    }
    Invoke-Step 'cosyvoice_check' {
        & $Py -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    }
}

# ----------------------------------------------------- SoulX-FlashHead Lite
# Candidate talking-head model (512x512, generates the whole head from one image). Not adopted;
# see the README for the test results. Run it with bench_flashhead.ps1.
if ($Only -eq 'flashhead') {
    $Py = "$Envs\flashhead\Scripts\python.exe"
    $Repo = "$Repos\SoulX-FlashHead"
    $Git = 'C:\Program Files\Git\cmd\git.exe'
    $Models = "$Repo\models"

    Invoke-Step 'flashhead_repo' {
        if (-not (Test-Path $Repo)) { & $Git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git $Repo }
    }
    Invoke-Step 'flashhead_venv' {
        & $Uv venv --python 3.10 --seed "$Envs\flashhead"
    }
    Invoke-Step 'flashhead_requirements' {
        # Upstream pins torch 2.7.1; the versions below match the sovits environment instead, so uv
        # hard-links them from its cache and the environment costs well under 1 GB of disk.
        # Left out on purpose: xfuser (Linux-only; replaced by scripts\flashhead_stubs), flash_attn
        # and xformers (optional; the code falls back to torch attention), nvidia-nccl-cu12
        # (multi-GPU only), gradio and decord (not used by generate_video.py).
        & $Uv pip install --python $Py 'torch==2.14.1+cu126' 'torchvision==0.29.2' 'numpy==1.26.4' `
            'transformers==4.57.3' 'diffusers==0.39.0' 'accelerate==1.15.0' 'opencv-python==4.11.0.86' `
            'librosa==0.10.2' 'loguru==0.8.0' 'einops==0.8.2' 'safetensors==0.8.0' 'pyyaml==6.0.4' `
            'pillow==12.3.0' triton-windows 'mediapipe==0.10.9' easydict ftfy imageio imageio-ffmpeg `
            scikit-image pyloudnorm tqdm `
            --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
    }
    Invoke-Step 'flashhead_models' {
        # Lite needs Model_Lite (5.7 GB) and VAE_LTX (1.6 GB) only; Model_Pro and VAE_Wan are skipped.
        # huggingface-cli stalled on these files from the server network, so use resumable curl.
        $ms = 'https://www.modelscope.cn/models/Soul-AILab/SoulX-FlashHead-1_3B/resolve/master'
        $hf = 'https://hf-mirror.com/facebook/wav2vec2-base-960h/resolve/main'
        New-Item -ItemType Directory -Force -Path "$Models\SoulX-FlashHead-1_3B\Model_Lite", `
            "$Models\SoulX-FlashHead-1_3B\VAE_LTX", "$Models\wav2vec2-base-960h" | Out-Null
        foreach ($part in 'Model_Lite', 'VAE_LTX') {
            foreach ($name in 'config.json', 'diffusion_pytorch_model.safetensors') {
                Get-RemoteFile "$ms/$part/$name" "$Models\SoulX-FlashHead-1_3B\$part\$name"
                if ($LASTEXITCODE -ne 0) { throw "download $part/$name" }
            }
        }
        foreach ($name in 'config.json', 'preprocessor_config.json', 'feature_extractor_config.json', 'model.safetensors') {
            Get-RemoteFile "$hf/$name" "$Models\wav2vec2-base-960h\$name"
            if ($LASTEXITCODE -ne 0) { throw "download wav2vec2 $name" }
        }
    }
    Invoke-Step 'flashhead_check' {
        $env:PYTHONPATH = "$PSScriptRoot\flashhead_stubs"
        Push-Location $Repo   # the code reads its config through a relative path
        & $Py -c "import torch, flash_head.inference; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
        Pop-Location
    }
}

# --------------------------------------------------------------------- LLM
if ($Only -in 'all', 'llm') {
    Invoke-Step 'llm_qwen3_4b_gguf' {
        Get-RemoteFile 'https://www.modelscope.cn/models/Qwen/Qwen3-4B-GGUF/resolve/master/Qwen3-4B-Q4_K_M.gguf' "$Root\models\llm\Qwen3-4B-Q4_K_M.gguf"
    }
}

Write-Host '--- summary ---'
Get-ChildItem $Markers -Name
Get-PSDrive C | ForEach-Object { 'free GB: {0:N1}' -f ($_.Free / 1GB) }

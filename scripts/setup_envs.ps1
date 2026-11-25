# Create the two Python environments and download model weights on the dev server.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
# Usage: powershell -ExecutionPolicy Bypass -File setup_envs.ps1 [-Only musetalk|sovits|llm]
# Two separate environments are required: MuseTalk pins transformers==4.39 / torch 2.0.1,
# while GPT-SoVITS needs transformers>=4.51 and a recent torch.
# Each step is skipped when its marker file exists, so re-running is safe.

param([ValidateSet('all', 'musetalk', 'sovits', 'llm')][string]$Only = 'all')

$ProgressPreference = 'SilentlyContinue'
$Root = 'C:\ai_live'
$Uv = "$Root\tools\uv\uv.exe"
$Repos = "$Root\repos"
$Envs = "$Root\envs"
$Markers = "$Root\markers"
New-Item -ItemType Directory -Force -Path $Markers, "$Root\models\llm" | Out-Null

$env:UV_DEFAULT_INDEX = 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'
$env:PIP_INDEX_URL = 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'
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
    param([string]$Url, [string]$OutFile)
    curl.exe -L --fail --retry 3 -C - --silent --show-error -o $OutFile $Url
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
        & $Py -m mim install 'mmpose==1.1.0'
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
        & $Uv pip install --python $Py -r $req `
            --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
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

# --------------------------------------------------------------------- LLM
if ($Only -in 'all', 'llm') {
    Invoke-Step 'llm_qwen3_4b_gguf' {
        Get-RemoteFile 'https://www.modelscope.cn/models/Qwen/Qwen3-4B-GGUF/resolve/master/Qwen3-4B-Q4_K_M.gguf' "$Root\models\llm\Qwen3-4B-Q4_K_M.gguf"
    }
}

Write-Host '--- summary ---'
Get-ChildItem $Markers -Name
Get-PSDrive C | ForEach-Object { 'free GB: {0:N1}' -f ($_.Free / 1GB) }

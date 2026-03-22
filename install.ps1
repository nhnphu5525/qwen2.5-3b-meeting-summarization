# ===========================================================================
#  install.ps1 — Cài đặt môi trường cho Qwen2.5-3B Meeting Summarization
# ===========================================================================
#
#  Cách dùng:
#    .\install.ps1              (tạo venv mới + cài tất cả)
#    .\install.ps1 -SkipVenv    (chỉ cài thư viện, không tạo lại venv)
#
# ===========================================================================

param(
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"

# --- 1. Tạo venv ---
if (-not $SkipVenv) {
    Write-Host "`n[1/4] Tao moi truong ao (venv)..." -ForegroundColor Cyan
    if (Test-Path ".\venv") {
        Write-Host "  -> Xoa venv cu..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".\venv"
    }
    python -m venv venv
    Write-Host "  -> venv da tao." -ForegroundColor Green
} else {
    Write-Host "`n[1/4] Bo qua tao venv (SkipVenv)." -ForegroundColor Yellow
}

# --- 2. Upgrade pip + setuptools < 81 ---
Write-Host "`n[2/4] Upgrade pip, setuptools < 81, wheel..." -ForegroundColor Cyan
& .\venv\Scripts\python.exe -m pip install --upgrade pip "setuptools<81" wheel

# --- 3. Cài torch CUDA riêng (tránh bị kéo bản CPU) ---
Write-Host "`n[3/4] Cai PyTorch CUDA tu pytorch.org/whl/cu128..." -ForegroundColor Cyan
& .\venv\Scripts\pip.exe install -r requirements-torch.txt

# Kiểm tra torch CUDA
$torchCheck = & .\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
Write-Host "  -> Torch: $torchCheck" -ForegroundColor Green

# --- 4. Cài phần còn lại (không cho pip tạo build env mới) ---
Write-Host "`n[4/4] Cai cac thu vien con lai tu requirements.txt..." -ForegroundColor Cyan
& .\venv\Scripts\pip.exe install -r requirements.txt --no-build-isolation

# --- Xong ---
Write-Host "`n=============================" -ForegroundColor Green
Write-Host "  CAI DAT HOAN TAT!" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green
Write-Host "Kich hoat venv: .\venv\Scripts\activate"
Write-Host ""

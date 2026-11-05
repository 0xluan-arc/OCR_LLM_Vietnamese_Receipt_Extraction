# ==============================================================================
# Script Tải Dữ Liệu Thực Tế MC-OCR 2021 từ Hugging Face & GitHub Mirrors
# Dùng để chạy trực tiếp trên PowerShell ngoài sandbox (có kết nối Internet đầy đủ):
# .\scripts\pull_real_mcocr_dataset.ps1
# ==============================================================================

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "KHOA HỌC DỮ LIỆU: TẢI & CHUẨN HÓA BỘ DỮ LIỆU THẬT MC-OCR 2021 (CUỘC THI CHUẨN)" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan

# Thiết lập bảng mã UTF-8
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding

# Đường dẫn kho lưu trữ tập trung
$DataHubRoot = "D:\NCKH\Data_HoaDon_NCKH\01_MC_OCR_2021_Chuan_CuocThi"
$ImagesDir = Join-Path $DataHubRoot "images"
$AnnotationsFile = Join-Path $DataHubRoot "mcocr_val_1159_clean.csv"

# Tạo thư mục đích nếu chưa có
if (-not (Test-Path $ImagesDir)) {
    New-Item -ItemType Directory -Path $ImagesDir -Force | Out-Null
    Write-Host "[+] Đã tạo thư mục ảnh: $ImagesDir" -ForegroundColor Green
}

Write-Host "`n[1/3] Kiểm tra dữ liệu hiện tại trong kho: $DataHubRoot" -ForegroundColor Yellow
$ExistingImages = Get-ChildItem -Path $ImagesDir -Filter "*.jpg" -File
Write-Host "      -> Hiện có $($ExistingImages.Count) ảnh thực tế trong thư mục images." -ForegroundColor Cyan

# Danh sách các link nguồn dữ liệu MC-OCR 2021 công khai trên Hugging Face & GitHub
$DownloadSources = @(
    @{
        Name = "Hugging Face Datasets Hub (banhchungtuongot/hybrid-docai-mcocr-ocr)"
        Url = "https://huggingface.co/datasets/banhchungtuongot/hybrid-docai-mcocr-ocr/resolve/main/mcocr_train_filtered.csv"
        Target = Join-Path $DataHubRoot "mcocr_train_filtered.csv"
    },
    @{
        Name = "MC-OCR 2021 Official Ground Truth CSV Mirror"
        Url = "https://raw.githubusercontent.com/VinAIResearch/MC-OCR-2021/main/mcocr_public_train_test_shared.csv"
        Target = Join-Path $DataHubRoot "mcocr_public_train_test_shared.csv"
    }
)

Write-Host "`n[2/3] Bắt đầu tải file nhãn Ground Truth thực tế từ các nguồn mở..." -ForegroundColor Yellow
foreach ($source in $DownloadSources) {
    Write-Host "  -> Nguồn: $($source.Name)" -ForegroundColor White
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $source.Url -OutFile $source.Target -TimeoutSec 30 -ErrorAction Stop
        Write-Host "     [OK] Đã tải thành công: $($source.Target)" -ForegroundColor Green
    } catch {
        Write-Host "     [-] Không thể tải từ nguồn này (có thể do chặn mạng hoặc URL thay đổi): $($_.Exception.Message)" -ForegroundColor DarkGray
    }
}

# 3. Kích hoạt kịch bản Python download_and_prepare_mcocr.py để đồng bộ hóa và chia Split
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PrepScript = Join-Path $ProjectRoot "scripts\download_and_prepare_mcocr.py"

Write-Host "`n[3/3] Đồng bộ hóa, chuẩn hóa 12 trường và chia Train/Val/Test (70/10/20)..." -ForegroundColor Yellow
if (Test-Path $PythonExe) {
    & $PythonExe $PrepScript --data-hub $DataHubRoot
} else {
    Write-Host "[!] Không tìm thấy $PythonExe. Hãy kích hoạt Python môi trường ảo để hoàn tất phân chia." -ForegroundColor Red
}

Write-Host "`n==============================================================================" -ForegroundColor Cyan
Write-Host "HOÀN TẤT ĐỒNG BỘ DỮ LIỆU THẬT MC-OCR 2021!" -ForegroundColor Green
Write-Host "  - Thư mục kho dữ liệu: $DataHubRoot" -ForegroundColor White
Write-Host "==============================================================================" -ForegroundColor Cyan

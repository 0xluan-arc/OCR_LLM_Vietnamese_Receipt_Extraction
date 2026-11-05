# ==============================================================================
# Script Tải & Chuẩn Hóa 1.000+ Mẫu Hóa Đơn Việt Nam MC-OCR 2021
# Nguồn: Hugging Face Hub (banhchungtuongot/hybrid-docai-mcocr-ocr)
# Chạy trực tiếp trong PowerShell: .\scripts\run_download_1000_receipts.ps1
# ==============================================================================

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "NCKH: TẢI & CHUẨN HÓA 1.000+ HÓA ĐƠN VIỆT NAM MC-OCR 2021 (MỤC TIÊU MT1)" -ForegroundColor Green
Write-Host "Repository: banhchungtuongot/hybrid-docai-mcocr-ocr" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$env:PYTHONIOENCODING = "utf-8"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[!] Không tìm thấy Python Virtual Environment tại: $PythonExe" -ForegroundColor Red
    Exit 1
}

$FetchScript = Join-Path $ProjectRoot "scripts\fetch_1000_real_receipts.py"

Write-Host "`nĐang tải bộ dữ liệu Hóa đơn Việt Nam MC-OCR 2021 từ Hugging Face Hub..." -ForegroundColor Yellow
& $PythonExe $FetchScript --repo-id "banhchungtuongot/hybrid-docai-mcocr-ocr" --count 1155

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[✓] HOÀN TẤT THÀNH CÔNG! Kho dữ liệu Hóa đơn Việt Nam MC-OCR 2021 đã sẵn sàng." -ForegroundColor Green
    Write-Host "Thư mục ảnh : D:\NCKH\Data_HoaDon_NCKH\01_MC_OCR_2021_Chuan_CuocThi\images" -ForegroundColor Cyan
    Write-Host "Báo cáo tại : experiments\real_1000_receipts_audit_and_split.md" -ForegroundColor Cyan
} else {
    Write-Host "`n[!] Quá trình kết thúc với mã lỗi: $LASTEXITCODE" -ForegroundColor Red
}

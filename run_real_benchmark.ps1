# ==============================================================================
# Script Thực Thi Benchmark Mô Hình Thật Qwen2.5-1.5B (Zero Mock) Ngoài Sandbox
# Chạy trực tiếp trong PowerShell: .\run_real_benchmark.ps1
# ==============================================================================

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "KHOA HỌC DỮ LIỆU & DOCUMENT AI: BENCHMARK MÔ HÌNH THẬT QWEN2.5-1.5B-INSTRUCT" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan

# 1. Thiết lập bảng mã UTF-8 cho Console PowerShell
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$env:PYTHONIOENCODING = "utf-8"

# 2. Định vị thư mục dự án và venv
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[!] Không tìm thấy Python Virtual Environment tại: $PythonExe" -ForegroundColor Red
    Write-Host "[!] Vui lòng tạo môi trường ảo hoặc cài đặt theo hướng dẫn README." -ForegroundColor Yellow
    Exit 1
}

# 3. Kiểm tra kết nối tới Ollama trên cổng 11434
Write-Host "[1/3] Đang kiểm tra trạng thái dịch vụ Ollama tại http://localhost:11434..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 4 -ErrorAction Stop
    $models = $response.models | ForEach-Object { $_.name }
    Write-Host "      -> Kết nối Ollama thành công!" -ForegroundColor Green
    Write-Host "      -> Các mô hình phát hiện: $($models -join ', ')" -ForegroundColor Cyan
    
    $hasQwen = $models | Where-Object { $_ -like "*qwen2.5:1.5b-instruct*" }
    if (-not $hasQwen) {
        Write-Host "[!] CẢNH BÁO: Chưa tìm thấy 'qwen2.5:1.5b-instruct' trong Ollama." -ForegroundColor Red
        Write-Host "[!] Hãy chạy lệnh: ollama pull qwen2.5:1.5b-instruct" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[!] CẢNH BÁO: Không thể kết nối tới dịch vụ Ollama tại http://localhost:11434" -ForegroundColor Red
    Write-Host "    Lỗi chi tiết: $($_.Exception.Message)" -ForegroundColor DarkGray
    Write-Host "    Vui lòng đảm bảo Ollama đã được khởi động trên máy tính của bạn." -ForegroundColor Yellow
}

# 4. Thực thi kịch bản Benchmark bằng Python
Write-Host "`n[2/3] Bắt đầu chạy Benchmark 5 cấu hình với Zero Mock..." -ForegroundColor Yellow
$BenchmarkScript = Join-Path $ScriptDir "scripts\run_real_qwen25_benchmark.py"

& $PythonExe $BenchmarkScript

# 5. Thông báo kết quả
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[3/3] BENCHMARK HOÀN TẤT XUẤT SẮC!" -ForegroundColor Green
    Write-Host "Báo cáo khoa học đã được tạo tại:" -ForegroundColor Cyan
    Write-Host "  - experiments/real_qwen25_benchmark_results.json" -ForegroundColor White
    Write-Host "  - experiments/real_qwen25_benchmark_results.md" -ForegroundColor White
} else {
    Write-Host "`n[!] Benchmark kết thúc với mã lỗi: $LASTEXITCODE" -ForegroundColor Red
}

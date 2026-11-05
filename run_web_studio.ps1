# ==============================================================================
# Script Khởi Chạy Nhanh FastAPI Modern Web Studio
# Chạy trực tiếp trong PowerShell: .\run_web_studio.ps1
# ==============================================================================

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "KHỞI CHẠY VIETNAMESE RECEIPT DOCUMENT AI STUDIO (FASTAPI + MODERN SPA)" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[!] Không tìm thấy $PythonExe" -ForegroundColor Red
    Exit 1
}

Write-Host "`nĐang mở Web Studio tại: http://localhost:8000" -ForegroundColor Yellow
Write-Host "Swagger API Docs tại   : http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Nhấn Ctrl+C để dừng server.`n" -ForegroundColor DarkGray

& $PythonExe -m uvicorn receipt_extraction.api.server:app --host 127.0.0.1 --port 8000 --reload

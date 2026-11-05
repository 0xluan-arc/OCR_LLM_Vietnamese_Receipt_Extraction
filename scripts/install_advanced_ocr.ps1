# ==============================================================================
# KỊCH BẢN CÀI ĐẶT & KHỞI TẠO TRỌNG SỐ PADDLEOCR + VIETOCR (MỤC TIÊU MT2)
# Dự án: Nghiên cứu kết hợp OCR và LLM trích xuất thông tin Hóa đơn bán lẻ Việt Nam
# Chạy ngoài PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_advanced_ocr.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$WeightsDir  = "D:\NCKH\VietOCR_Weights"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " KÍCH HOẠT VŨ KHÍ OCR HẠNG NẶNG: PADDLEOCR (DBNet) + VIETOCR (Seq2Seq)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

if (-not (Test-Path $VenvPython)) {
    Write-Error "Không tìm thấy môi trường ảo tại: $VenvPython"
    exit 1
}

# 1. Tạo thư mục lưu trữ trọng số tập trung trên ổ D:\
if (-not (Test-Path $WeightsDir)) {
    New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null
}
$env:TORCH_HOME = $WeightsDir
$env:PADDLE_PDX_CACHE_DIR = $WeightsDir

Write-Host "`n[1/4] Đang cập nhật setuptools (<81 cho pkg_resources) và các gói nền tảng..." -ForegroundColor Yellow
& $VenvPython -m pip install "setuptools<81" wheel pyyaml requests tqdm pillow opencv-python

Write-Host "`n[2/4] Đang kiểm tra & cài đặt PyTorch + VietOCR..." -ForegroundColor Yellow
& $VenvPython -m pip install torch torchvision vietocr

Write-Host "`n[3/4] Đang cài đặt PaddlePaddle & PaddleOCR (Text Detection DBNet)..." -ForegroundColor Yellow
& $VenvPython -m pip install paddlepaddle "paddleocr>=2.7.0,<3.0.0"

Write-Host "`n[4/4] Đang tải trước trọng số VietOCR (vgg_seq2seq & vgg_transformer) và PaddleOCR về $WeightsDir..." -ForegroundColor Yellow

$InitScript = @"
import os
import urllib.request
from pathlib import Path

weights_dir = Path(r'$WeightsDir')
weights_dir.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(weights_dir)

# Tải trọng số VietOCR vgg_seq2seq và vgg_transformer từ kho chính thức
urls = {
    'vgg_seq2seq.pth': 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth',
    'vgg_transformer.pth': 'https://vocr.vn/data/vietocr/vgg_transformer.pth',
}
for fname, url in urls.items():
    dest = weights_dir / fname
    if not dest.exists() or dest.stat().st_size < 1024 * 1024:
        print(f'  -> Đang tải {fname} từ {url} ...')
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f'  [OK] Đã lưu: {dest} ({dest.stat().st_size / 1e6:.1f} MB)')
        except Exception as e:
            print(f'  [CẢNH BÁO] Không thể tải {fname} từ {url}: {e}')
    else:
        print(f'  [OK] Trọng số {fname} đã sẵn sàng ({dest.stat().st_size / 1e6:.1f} MB)')

# Khởi tạo thử PaddleOCR Detector
try:
    from paddleocr import PaddleOCR
    det = PaddleOCR(use_angle_cls=True, lang='vi', det=True, rec=False, show_log=False)
    print('  [OK] PaddleOCR Text Detector (DBNet) đã sẵn sàng!')
except Exception as e:
    print(f'  [CẢNH BÁO] Khởi tạo PaddleOCR Detector: {e}')

# Khởi tạo thử PaddleVietOCREngine
from receipt_extraction.ocr.factory import get_ocr_engine
eng = get_ocr_engine('paddleocr')
ok, msg = eng.is_available()
print(f'  -> Trạng thái PaddleVietOCREngine: available={ok} | {msg or "Sẵn sàng 100%"}')
"@

& $VenvPython -c $InitScript

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " HOÀN TẤT CÀI ĐẶT! BẠN CÓ THỂ CHẠY ĐÁNH GIÁ MT2 HOẶC WEB STUDIO NGAY:" -ForegroundColor Green
Write-Host "   .\.venv\Scripts\python.exe scripts/evaluate_ocr_engines.py" -ForegroundColor White
Write-Host "   .\.venv\Scripts\python.exe scripts/run_web_studio.py --port 8000" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Green

"""Integration tests cho OCR Engine thực tế và luồng xử lý trích xuất."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pytest

from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.ocr.tesseract_ocr import TesseractOCREngine
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.validation.receipt_validator import validate_receipt_data
from receipt_extraction.schemas.receipt import ExtractionResult, ExtractionMetadata


@pytest.fixture
def tesseract_or_skip():
    """Fixture kiểm tra Tesseract OCR trước khi chạy integration test."""
    engine = TesseractOCREngine()
    available, reason = engine.is_available()
    if not available:
        pytest.skip(f"Bỏ qua Integration Test do Tesseract chưa được cài đặt: {reason}")
    return engine


def test_tesseract_ocr_on_blank_image(tesseract_or_skip):
    """Kiểm tra Tesseract chạy trên ảnh trắng không có chữ không bị crash."""
    engine = tesseract_or_skip
    # Tạo ảnh trắng 400x400
    blank_img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    
    result = engine.recognize(blank_img)
    assert result.error_message is None
    assert result.engine_name == "tesseract"
    # Văn bản nhận dạng được từ ảnh trắng phải rỗng hoặc chỉ có khoảng trắng
    assert len(result.raw_text.strip()) == 0


def test_tesseract_ocr_on_synthetic_receipt_image(tesseract_or_skip):
    """Kiểm tra Tesseract nhận dạng chữ trên ảnh hóa đơn tổng hợp."""
    engine = tesseract_or_skip
    
    # Tạo ảnh có chữ rõ ràng
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "SIEU THI MINI MART", fill=(0, 0, 0))
    draw.text((20, 70), "Tong thanh toan: 150.000 VND", fill=(0, 0, 0))
    draw.text((20, 110), "MST: 0319998888", fill=(0, 0, 0))

    result = engine.recognize(img)
    assert result.error_message is None
    assert len(result.lines) > 0
    assert result.processing_time_sec > 0

    # Chuyển tiếp kết quả sang Regex Baseline
    extractor = RegexBaselineExtractor()
    data, evidence = extractor.extract(result)

    assert data.total_cost == 150000
    assert data.tax_id == "0319998888"

    # Kiểm tra qua validator
    issues, is_valid = validate_receipt_data(data, raw_ocr_text=result.raw_text)
    assert is_valid is True

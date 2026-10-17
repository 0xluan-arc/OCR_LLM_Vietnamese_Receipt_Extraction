"""Unit tests cho OCR Factory và các engine trừu tượng."""

import pytest
from receipt_extraction.ocr.factory import get_ocr_engine, UnimplementedOCREngine
from receipt_extraction.ocr.dummy_ocr import DummyOCREngine
from receipt_extraction.ocr.tesseract_ocr import TesseractOCREngine


def test_factory_returns_dummy_engine():
    """Kiểm tra factory trả về DummyOCREngine khi yêu cầu 'dummy'."""
    engine = get_ocr_engine("dummy")
    assert isinstance(engine, DummyOCREngine)
    assert engine.name == "dummy"
    is_avail, err = engine.is_available()
    assert is_avail is True
    assert err is None


def test_factory_returns_tesseract_engine():
    """Kiểm tra factory trả về TesseractOCREngine khi yêu cầu 'tesseract'."""
    engine = get_ocr_engine("tesseract")
    assert isinstance(engine, TesseractOCREngine)
    assert engine.name == "tesseract"


def test_factory_returns_unimplemented_for_future_engines():
    """Kiểm tra factory xử lý đúng các engine thuộc lộ trình sau (VietOCR, PaddleOCR)."""
    vietocr_engine = get_ocr_engine("vietocr")
    assert isinstance(vietocr_engine, UnimplementedOCREngine)
    is_avail, msg = vietocr_engine.is_available()
    assert is_avail is False
    assert "chưa được triển khai" in msg

    paddle_engine = get_ocr_engine("paddleocr")
    assert isinstance(paddle_engine, UnimplementedOCREngine)
    is_avail, msg = paddle_engine.is_available()
    assert is_avail is False
    assert "chưa được triển khai" in msg


def test_factory_raises_error_on_unknown_engine():
    """Kiểm tra factory báo lỗi rõ ràng khi tên engine không hợp lệ."""
    with pytest.raises(ValueError, match="Không nhận dạng được OCR engine"):
        get_ocr_engine("unknown_engine_xyz")

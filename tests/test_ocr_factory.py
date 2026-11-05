"""Unit tests cho OCR Factory và các engine (Dummy, Tesseract, VietOCR, PaddleVietOCR)."""

import pytest
from PIL import Image, ImageDraw
from receipt_extraction.ocr.base import OCRBoundingBox
from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.ocr.dummy_ocr import DummyOCREngine
from receipt_extraction.ocr.tesseract_ocr import TesseractOCREngine
from receipt_extraction.ocr.vietocr_engine import (
    PaddleVietOCREngine,
    VietOCREngine,
    detect_line_boxes_opencv,
)


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


def test_factory_returns_vietocr_and_paddleocr_engines():
    """Kiểm tra factory trả về VietOCREngine và PaddleVietOCREngine kèm hướng dẫn cài đặt an toàn."""
    vietocr_engine = get_ocr_engine("vietocr")
    assert isinstance(vietocr_engine, VietOCREngine)
    assert vietocr_engine.name == "vietocr"
    is_avail, msg = vietocr_engine.is_available()
    if not is_avail:
        assert "vietocr" in msg.lower() or "torch" in msg.lower()

    paddle_engine = get_ocr_engine("paddleocr")
    assert isinstance(paddle_engine, PaddleVietOCREngine)
    assert paddle_engine.name == "paddleocr"
    is_avail_p, msg_p = paddle_engine.is_available()
    if not is_avail_p:
        assert "paddleocr" in msg_p.lower() or "vietocr" in msg_p.lower()


def test_vietocr_engine_two_stage_pipeline_with_custom_callbacks():
    """Kiểm tra luồng nhận dạng 2 giai đoạn (Phát hiện dòng OpenCV + Nhận dạng VietOCR) của VietOCREngine."""
    img = Image.new("RGB", (400, 140), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 360, 45], fill=(0, 0, 0))
    draw.rectangle([20, 75, 320, 100], fill=(0, 0, 0))

    boxes = detect_line_boxes_opencv(img)
    assert len(boxes) >= 2

    mock_lines = ["SIÊU THỊ WINMART+", "Tổng cộng: 125.000"]
    call_idx = {"i": 0}

    def _custom_rec(_crop: Image.Image):
        idx = call_idx["i"]
        call_idx["i"] += 1
        text = mock_lines[idx] if idx < len(mock_lines) else "Dòng phụ"
        return text, 0.96

    engine = VietOCREngine(
        custom_detector=lambda im: detect_line_boxes_opencv(im),
        custom_recognizer=_custom_rec,
    )
    avail, err = engine.is_available()
    assert avail is True
    assert err is None

    res = engine.recognize(img)
    assert res.is_success is True
    assert len(res.lines) == 2
    assert "SIÊU THỊ WINMART+" in res.raw_text
    assert "125.000" in res.raw_text


def test_factory_raises_error_on_unknown_engine():
    """Kiểm tra factory báo lỗi rõ ràng khi tên engine không hợp lệ."""
    with pytest.raises(ValueError, match="Không nhận dạng được OCR engine"):
        get_ocr_engine("unknown_engine_xyz")

"""Factory khởi tạo OCR Engine theo cấu hình."""

from typing import Optional
from .base import BaseOCREngine
from .dummy_ocr import DummyOCREngine
from .tesseract_ocr import TesseractOCREngine
from receipt_extraction.config import config


def get_ocr_engine(engine_name: Optional[str] = None) -> BaseOCREngine:
    """Trả về instance của OCR engine tương ứng."""
    name = (engine_name or config.ocr.engine).lower().strip()
    
    if name == "dummy":
        return DummyOCREngine()
    elif name == "tesseract":
        return TesseractOCREngine()
    elif name in ("vietocr", "paddleocr"):
        # Với vietocr và paddleocr, sẽ cài đặt chi tiết ở Giai đoạn 3 (Tuần 3)
        # Tạm thời trả về dummy hoặc thông báo
        return DummyOCREngine()
    else:
        # Fallback về Dummy
        return DummyOCREngine()

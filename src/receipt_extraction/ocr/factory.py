"""Factory khởi tạo OCR Engine theo cấu hình và kiểm soát tính khả dụng."""

from typing import Optional, Tuple, Union, Any
from pathlib import Path
from .base import BaseOCREngine, OCRResult
from .dummy_ocr import DummyOCREngine
from .tesseract_ocr import TesseractOCREngine
from receipt_extraction.config import config


class UnimplementedOCREngine(BaseOCREngine):
    """Engine giữ chỗ cho các mô hình đang trong lộ trình phát triển (VietOCR, PaddleOCR)."""

    def __init__(self, name: str, planned_phase: str):
        super().__init__(name=name)
        self.planned_phase = planned_phase

    def is_available(self) -> Tuple[bool, Optional[str]]:
        return False, f"Engine '{self.name}' chưa được triển khai (dự kiến trong {self.planned_phase})."

    def recognize(self, image_input: Union[str, Path, bytes, Any]) -> OCRResult:
        return OCRResult(
            engine_name=self.name,
            error_message=(
                f"Engine '{self.name}' chưa sẵn sàng. Hãy chọn 'tesseract' (OCR thật) "
                f"hoặc 'dummy' (chế độ kiểm thử/mô phỏng)."
            )
        )


def get_ocr_engine(engine_name: Optional[str] = None) -> BaseOCREngine:
    """Trả về instance của OCR engine tương ứng.
    
    Quy tắc khoa học: Tuyệt đối KHÔNG giả lập OCR thật bằng Dummy nếu không được yêu cầu rõ ràng.
    """
    name = (engine_name or config.ocr.engine).lower().strip()
    
    if name == "dummy":
        return DummyOCREngine()
    elif name == "tesseract":
        return TesseractOCREngine()
    elif name == "vietocr":
        return UnimplementedOCREngine(name="vietocr", planned_phase="Giai đoạn 3 - Tuần 3 (VietOCR recognition)")
    elif name == "paddleocr":
        return UnimplementedOCREngine(name="paddleocr", planned_phase="Giai đoạn 3 - Tuần 3 (PaddleOCR text detection)")
    else:
        raise ValueError(
            f"Không nhận dạng được OCR engine '{name}'. "
            f"Các lựa chọn hợp lệ: 'dummy', 'tesseract', 'vietocr', 'paddleocr'."
        )

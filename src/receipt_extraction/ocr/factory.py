"""Factory khởi tạo OCR Engine theo cấu hình và kiểm soát tính khả dụng."""

from typing import Optional, Tuple, Union, Any
from pathlib import Path
from .base import BaseOCREngine, OCRResult
from .dummy_ocr import DummyOCREngine
from .tesseract_ocr import TesseractOCREngine
from .vietocr_engine import VietOCREngine, PaddleVietOCREngine
from receipt_extraction.config import config


class UnimplementedOCREngine(BaseOCREngine):
    """Engine giữ chỗ dự phòng cho các mô hình mở rộng tương lai."""

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


def get_ocr_engine(
    engine_name: Optional[str] = None,
    fallback_to_tesseract: bool = False,
) -> BaseOCREngine:
    """Trả về instance của OCR engine tương ứng.

    Nếu không truyền `engine_name` (dùng mặc định từ config là `paddleocr`) hoặc bật `fallback_to_tesseract=True`,
    hệ thống sẽ ưu tiên `PaddleVietOCREngine` và tự động fallback về `TesseractOCREngine` nếu máy chưa hỗ trợ.
    """
    use_default = engine_name is None
    name = (engine_name or config.ocr.engine or "paddleocr").lower().strip()

    if name == "dummy":
        return DummyOCREngine()
    elif name == "tesseract":
        return TesseractOCREngine()
    elif name == "vietocr":
        eng: BaseOCREngine = VietOCREngine()
    elif name in ("paddleocr", "paddle_vietocr"):
        eng = PaddleVietOCREngine()
    else:
        raise ValueError(
            f"Không nhận dạng được OCR engine '{name}'. "
            f"Các lựa chọn hợp lệ: 'dummy', 'tesseract', 'vietocr', 'paddleocr'."
        )

    if (use_default or fallback_to_tesseract) and not eng.is_available()[0]:
        tess = TesseractOCREngine()
        if tess.is_available()[0]:
            return tess
    return eng

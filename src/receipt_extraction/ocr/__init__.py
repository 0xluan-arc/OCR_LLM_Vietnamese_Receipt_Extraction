"""OCR modules and abstractions."""

from .base import OCRBoundingBox, OCRLine, OCRResult, BaseOCREngine
from .factory import get_ocr_engine

__all__ = [
    "OCRBoundingBox",
    "OCRLine",
    "OCRResult",
    "BaseOCREngine",
    "get_ocr_engine"
]

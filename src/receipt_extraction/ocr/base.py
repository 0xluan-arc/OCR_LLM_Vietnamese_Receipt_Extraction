"""Định nghĩa cấu trúc dữ liệu và interface trừu tượng cho các OCR Engine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from pathlib import Path


@dataclass
class OCRBoundingBox:
    """Tọa độ hộp bao của dòng/từ (x_min, y_min, x_max, y_max)."""
    x_min: int = 0
    y_min: int = 0
    x_max: int = 0
    y_max: int = 0

    @property
    def box_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass
class OCRLine:
    """Một dòng văn bản do OCR phát hiện kèm tọa độ và độ tin cậy."""
    text: str
    confidence: float = 1.0
    bounding_box: Optional[OCRBoundingBox] = None


@dataclass
class OCRResult:
    """Toàn bộ kết quả OCR cho một ảnh hóa đơn."""
    engine_name: str
    lines: List[OCRLine] = field(default_factory=list)
    raw_text: str = ""
    processing_time_sec: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.raw_text and self.lines:
            self.raw_text = "\n".join(line.text for line in self.lines if line.text.strip())


class BaseOCREngine(ABC):
    """Interface cơ sở cho mọi engine nhận dạng ký tự quang học (OCR)."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def recognize(self, image_input: Union[str, Path, bytes]) -> OCRResult:
        """Thực hiện nhận dạng văn bản từ ảnh đầu vào."""
        pass

    @abstractmethod
    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Kiểm tra engine có khả dụng trên hệ thống hiện tại hay không."""
        pass

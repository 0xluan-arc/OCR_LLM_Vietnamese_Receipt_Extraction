"""Interface cơ sở cho các bộ trích xuất thông tin KIE."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict

from receipt_extraction.schemas.receipt import ReceiptData
from receipt_extraction.ocr.base import OCRResult


class BaseExtractor(ABC):
    """Interface trừu tượng cho mọi thuật toán trích xuất KIE."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract(self, ocr_result: OCRResult) -> Tuple[ReceiptData, Dict[str, bool]]:
        """Trích xuất dữ liệu hóa đơn từ kết quả OCR.
        
        Returns:
            Tuple[ReceiptData, Dict[str, bool]]:
                - ReceiptData: dữ liệu đã được gán vào schema.
                - Dict[str, bool]: map đánh dấu trường có tìm thấy bằng chứng trong text OCR hay không.
        """
        pass

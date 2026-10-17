"""Mock/Dummy OCR Engine phục vụ kiểm thử đơn vị và demo offline."""

import time
from typing import Union, Tuple, Optional
from pathlib import Path

from .base import BaseOCREngine, OCRResult, OCRLine, OCRBoundingBox


SAMPLE_MOCK_RECEIPT = """SIÊU THỊ TIỆN LỢI CIRCLE K
Địa chỉ: 123 Nguyễn Thị Minh Khai, Phường Bến Nghé, Quận 1, TP.HCM
Mã số thuế: 0312456789
Số hóa đơn: CK-20261017-0892
Ngày giờ: 17/10/2026 08:30:15
----------------------------------------
TÊN HÀNG             SL   ĐƠN GIÁ  THÀNH TIỀN
Cà phê sữa đá 330ml   2    25.000      50.000
Bánh mì que pate      1    15.000      15.000
Nước khoáng Lavie     2    10.000      20.000
----------------------------------------
Tạm tính:                              85.000
Giảm giá thành viên:                   -5.000
Thuế VAT (8%):                          6.400
Tổng thanh toán:                       86.400
Hình thức thanh toán: Tiền mặt
Cảm ơn quý khách và hẹn gặp lại!
"""


class DummyOCREngine(BaseOCREngine):
    """Engine giả lập kết quả OCR cho mục đích kiểm thử."""

    def __init__(self, mock_text: Optional[str] = None):
        super().__init__(name="dummy")
        self.mock_text = mock_text or SAMPLE_MOCK_RECEIPT

    def is_available(self) -> Tuple[bool, Optional[str]]:
        return True, None

    def recognize(self, image_input: Union[str, Path, bytes]) -> OCRResult:
        start_time = time.time()
        lines = []
        raw_lines = self.mock_text.strip().split("\n")
        y = 10
        for i, text in enumerate(raw_lines):
            lines.append(
                OCRLine(
                    text=text,
                    confidence=0.98,
                    bounding_box=OCRBoundingBox(x_min=10, y_min=y, x_max=500, y_max=y + 25)
                )
            )
            y += 30

        elapsed = time.time() - start_time
        return OCRResult(
            engine_name=self.name,
            lines=lines,
            raw_text=self.mock_text.strip(),
            processing_time_sec=elapsed
        )

"""Module tích hợp Large Language Model (LLM) để trích xuất thông tin KIE."""

import json
import time
from typing import Tuple, Dict, Optional

from .base import BaseExtractor
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem
from receipt_extraction.ocr.base import OCRResult
from receipt_extraction.config import config


class LLMNotConfiguredError(Exception):
    """Ngoại lệ ném ra khi LLM chưa được cấu hình nhà cung cấp hoặc thiếu API key."""
    pass


RECEIPT_EXTRACTION_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia về Document AI và kế toán Việt Nam.
Nhiệm vụ của bạn là nhận vào văn bản OCR của hóa đơn bán lẻ Việt Nam và trích xuất thông tin chính xác theo cấu trúc JSON.

Quy tắc bắt buộc:
1. Chỉ trích xuất thông tin có thực trong văn bản OCR. Tuyệt đối KHÔNG tự bịa hoặc suy diễn thông tin không có căn cứ.
2. Với các trường số tiền (subtotal, discount, vat_amount, total_cost, unit_price, total_price): chuyển đổi về số nguyên VND (ví dụ: "86.400" -> 86400, "15k" -> 15000). Nếu không có, gán null.
3. Với trường ngày giờ (timestamp): chuẩn hóa về định dạng dd/mm/yyyy hh:mm hoặc dd/mm/yyyy nếu có thể.
4. Trường discount (giảm giá) luôn là số dương nguyên.
5. Trả về DUY NHẤT một khối JSON hợp lệ theo schema sau, không thêm lời giải thích hay ký tự markdown thừa ngoài khối JSON:

{
  "seller_name": "Tên cửa hàng hoặc null",
  "seller_address": "Địa chỉ hoặc null",
  "tax_id": "Mã số thuế hoặc null",
  "invoice_number": "Số hóa đơn hoặc null",
  "timestamp": "Ngày giờ hoặc null",
  "items": [
    {
      "item_name": "Tên hàng",
      "quantity": 1.0,
      "unit_price": 25000,
      "total_price": 25000
    }
  ],
  "subtotal": null,
  "discount": null,
  "vat_amount": null,
  "total_cost": 86400,
  "payment_method": "Tiền mặt hoặc null"
}
"""


class LLMExtractor(BaseExtractor):
    """Bộ trích xuất thông tin sử dụng Mô hình Ngôn ngữ Lớn (LLM)."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        super().__init__(name="llm_extractor")
        self.provider = (provider or config.llm.provider).lower()
        self.model = model or config.llm.model
        self.api_key = api_key or config.llm.api_key
        self.base_url = base_url or config.llm.base_url

    def is_configured(self) -> Tuple[bool, Optional[str]]:
        """Kiểm tra xem mô-đun LLM đã được cấu hình đầy đủ chưa."""
        if self.provider == "none":
            return False, "LLM chưa được kích hoạt (LLM_PROVIDER='none'). Hiện tại hệ thống đang chạy ở chế độ Baseline Regex B1."
        if self.provider in ("openai", "gemini", "groq") and not self.api_key:
            return False, f"Nhà cung cấp LLM '{self.provider}' yêu cầu LLM_API_KEY nhưng chưa được thiết lập trong biến môi trường."
        return True, None

    def extract(self, ocr_result: OCRResult) -> Tuple[ReceiptData, Dict[str, bool]]:
        configured, reason = self.is_configured()
        if not configured:
            raise LLMNotConfiguredError(reason)

        raw_text = ocr_result.raw_text or ""
        evidence: Dict[str, bool] = {}

        # Gọi API hoặc model cục bộ dựa theo provider
        # Trong giai đoạn này (Giai đoạn 1 MVP), nếu người dùng chưa kết nối API key,
        # hệ thống sẽ báo rõ ràng trạng thái.
        # Ở các giai đoạn tiếp theo (Tuần 4), logic gọi HTTP OpenAI-compatible endpoint sẽ được mở rộng chi tiết.
        raise NotImplementedError(
            f"Provider '{self.provider}' đã được khai báo nhưng chưa được tích hợp hoàn chỉnh trong bước MVP này."
        )

"""Định nghĩa cấu trúc dữ liệu Pydantic cho hóa đơn bán lẻ Việt Nam."""

from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class ValidationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationIssue(BaseModel):
    """Chi tiết một cảnh báo hoặc lỗi kiểm định."""
    code: str = Field(description="Mã lỗi kiểm định (ví dụ: MATH_MISMATCH, MISSING_REQUIRED_FIELD)")
    field_name: Optional[str] = Field(default=None, description="Tên trường liên quan")
    message: str = Field(description="Mô tả chi tiết nguyên nhân")
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING)


class ReceiptItem(BaseModel):
    """Một dòng hàng trong danh sách mặt hàng hóa đơn."""
    item_name: str = Field(description="Tên sản phẩm hoặc dịch vụ")
    quantity: Optional[float] = Field(default=1.0, description="Số lượng mua")
    unit_price: Optional[int] = Field(default=None, description="Đơn giá (VND)")
    total_price: Optional[int] = Field(default=None, description="Thành tiền (VND)")


class ReceiptData(BaseModel):
    """Dữ liệu hóa đơn có cấu trúc hoàn chỉnh (12 trường thông tin chuẩn)."""
    seller_name: Optional[str] = Field(default=None, description="Tên cửa hàng / Đơn vị bán hàng")
    seller_address: Optional[str] = Field(default=None, description="Địa chỉ cửa hàng")
    seller_phone: Optional[str] = Field(default=None, description="Số điện thoại cửa hàng")
    tax_id: Optional[str] = Field(default=None, description="Mã số thuế cửa hàng")
    invoice_number: Optional[str] = Field(default=None, description="Số hóa đơn / Số phiếu thanh toán")
    
    timestamp: Optional[str] = Field(default=None, description="Ngày và giờ mua hàng (dd/mm/yyyy hh:mm hoặc dd/mm/yyyy)")
    
    items: List[ReceiptItem] = Field(default_factory=list, description="Danh sách mặt hàng chi tiết")
    
    subtotal: Optional[int] = Field(default=None, description="Tạm tính / Tổng tiền trước thuế hoặc giảm giá (VND)")
    discount: Optional[int] = Field(default=None, description="Số tiền giảm giá hoặc chiết khấu (VND)")
    vat_amount: Optional[int] = Field(default=None, description="Tiền thuế VAT (VND)")
    total_cost: Optional[int] = Field(default=None, description="Tổng tiền thanh toán cuối cùng (VND)")
    payment_method: Optional[str] = Field(default=None, description="Hình thức thanh toán (Tiền mặt, Chuyển khoản, Thẻ...)")

    def to_mc_ocr_dict(self) -> Dict[str, Optional[str]]:
        """Trả về 4 trường chuẩn của MC-OCR 2021 để dùng chung bộ chấm điểm."""
        return {
            "SELLER": self.seller_name,
            "SELLER_ADDRESS": self.seller_address,
            "TIMESTAMP": self.timestamp,
            "TOTAL_COST": str(self.total_cost) if self.total_cost is not None else None,
        }


class ExtractionMetadata(BaseModel):
    """Thông tin kỹ thuật về quá trình trích xuất, thích nghi Nested Learning và kiểm định."""
    method: str = Field(description="Phương pháp trích xuất: 'regex_baseline', 'llm_zeroshot', 'llm_fewshot', v.v.")
    ocr_engine: str = Field(description="Engine OCR đã sử dụng")
    raw_ocr_text: str = Field(default="", description="Toàn bộ văn bản thô do OCR đọc được")
    processing_time_sec: float = Field(default=0.0, description="Thời gian thực thi trích xuất (giây)")
    ocr_character_count: int = Field(default=0, description="Số lượng ký tự OCR phát hiện được")
    field_evidence: Dict[str, bool] = Field(
        default_factory=dict,
        description="Đánh dấu trường có tìm thấy bằng chứng trong văn bản OCR thô hay không"
    )
    adaptation_mode: str = Field(
        default="regex_b1",
        description="Cấp độ thích nghi: regex_b1, zero_shot, few_shot_cms, few_shot_layout, lora, lora_few_shot"
    )
    retry_count: int = Field(
        default=0,
        description="Số lượt tự sửa lỗi qua Tín hiệu Ngạc nhiên Cục bộ (LSS Self-Correction)"
    )
    needs_human_review: bool = Field(
        default=False,
        description="Gắn cờ cần người kiểm tra khi vẫn còn lỗi sau khi tự sửa"
    )
    input_tokens: int = Field(
        default=0,
        description="Số token đầu vào để đo chi phí nén ngữ cảnh"
    )
    output_tokens: int = Field(
        default=0,
        description="Số token sinh ra"
    )


class ExtractionResult(BaseModel):
    """Kết quả tổng thể trả về cho người dùng hoặc API."""
    success: bool = Field(description="Trạng thái thực thi pipeline có thành công hay không")
    data: ReceiptData = Field(default_factory=ReceiptData, description="Dữ liệu hóa đơn trích xuất được")
    metadata: ExtractionMetadata = Field(description="Metadata kỹ thuật")
    validation_issues: List[ValidationIssue] = Field(
        default_factory=list,
        description="Danh sách cảnh báo / vi phạm ràng buộc nghiệp vụ"
    )
    is_valid: bool = Field(
        default=True,
        description="Hóa đơn có vượt qua tất cả các kiểm tra nghiệp vụ nghiêm ngặt hay không"
    )

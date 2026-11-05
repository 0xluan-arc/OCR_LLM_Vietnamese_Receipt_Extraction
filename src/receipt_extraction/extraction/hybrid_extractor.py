"""Bộ trích xuất định tuyến lai (Hybrid Extractor) kết hợp Fast Path Regex và Slow Path Nested LLM.

Triển khai các nguyên lý từ kỹ năng ECC cost-aware-llm-pipeline và regex-vs-llm-structured-text:
- Fast Path: Chi phí 0 token, độ trễ mili-giây qua RegexBaselineExtractor.
- Escalation: Tự động leo thang sang LLMExtractor khi phát hiện dị thường, vi phạm số học
  hoặc phát hiện bỏ sót trường trên OCR (`MISSING_DETECTED_FIELD`, LSS > 0).
- Nếu LLM chưa được cấu hình (hoặc server chưa bật), không được che giấu lỗi mà phải gắn cờ
  `last_needs_human_review = True` và ghi nhận `hybrid_fallback_unverified`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from receipt_extraction.extraction.base import BaseExtractor
from receipt_extraction.extraction.llm_extractor import LLMExtractor, LLMNotConfiguredError
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.ocr.base import OCRResult
from receipt_extraction.schemas.receipt import (
    ReceiptData,
    ValidationIssue,
    ValidationSeverity,
)
from receipt_extraction.validation.receipt_validator import (
    check_unextracted_evidence,
    sanitize_and_ground_receipt_data,
    validate_receipt_data,
)


class HybridNestedExtractor(BaseExtractor):
    """Bộ trích xuất lai tối ưu hóa chi phí và độ chính xác theo kiến trúc Nested Learning."""

    def __init__(
        self,
        regex_extractor: Optional[RegexBaselineExtractor] = None,
        llm_extractor: Optional[LLMExtractor] = None,
    ):
        super().__init__(name="hybrid_nested_extractor")
        self.regex_extractor = regex_extractor or RegexBaselineExtractor()
        self.llm_extractor = llm_extractor or LLMExtractor()

        self.last_adaptation_mode: str = "hybrid_fast_regex"
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_retry_count: int = 0
        self.last_needs_human_review: bool = False

        self.fast_path_count: int = 0
        self.escalated_path_count: int = 0

    def extract(self, ocr_result: OCRResult) -> Tuple[ReceiptData, Dict[str, bool]]:
        """Định tuyến linh hoạt giữa Regex Baseline và LLM dựa trên kiểm định chất lượng và Unextracted Evidence Scan."""
        raw_ocr = ocr_result.raw_text or ""

        # Bước 1: Fast Path (Tần số cực nhanh, chi phí 0 token)
        regex_data, regex_evidence = self.regex_extractor.extract(ocr_result)
        issues, is_valid = validate_receipt_data(regex_data, raw_ocr_text=raw_ocr)
        unextracted_issues = check_unextracted_evidence(regex_data, raw_ocr)

        has_warning_or_error = any(
            iss.severity in (ValidationSeverity.WARNING, ValidationSeverity.ERROR)
            for iss in issues
        ) or len(unextracted_issues) > 0

        has_core_fields = (
            regex_data.seller_name is not None
            and regex_data.total_cost is not None
            and regex_data.total_cost > 0
            and len(regex_data.items) > 0
        )

        # Bước 2: Điều kiện giữ lại Fast Path (Nghiêm ngặt: Không có lỗi toán học VÀ không bỏ sót trường nào trên OCR)
        if is_valid and not has_warning_or_error and has_core_fields:
            regex_data = sanitize_and_ground_receipt_data(regex_data, raw_ocr_text=raw_ocr, recover_math=True)
            self.last_adaptation_mode = "hybrid_fast_regex"
            self.last_input_tokens = 0
            self.last_output_tokens = 0
            self.last_retry_count = 0
            self.last_needs_human_review = False
            self.fast_path_count += 1
            return regex_data, regex_evidence

        # Bước 3: Slow Path (Leo thang sang Nested LLM khi LSS > 0 hoặc phát hiện bỏ sót trường)
        self.escalated_path_count += 1
        try:
            llm_data, llm_evidence = self.llm_extractor.extract(ocr_result)
            self.last_adaptation_mode = "hybrid_escalated_llm"
            self.last_input_tokens = self.llm_extractor.last_input_tokens
            self.last_output_tokens = self.llm_extractor.last_output_tokens
            self.last_retry_count = self.llm_extractor.last_retry_count
            self.last_needs_human_review = self.llm_extractor.last_needs_human_review
            return llm_data, llm_evidence
        except Exception:
            self.last_adaptation_mode = "hybrid_fallback_regex"
            self.last_input_tokens = 0
            self.last_output_tokens = 0
            self.last_retry_count = 0
            actionable = any(
                iss.severity in (ValidationSeverity.WARNING, ValidationSeverity.ERROR)
                for iss in issues
            ) or (len(unextracted_issues) > 0) or (not is_valid)
            self.last_needs_human_review = actionable
            return regex_data, regex_evidence

"""Unit tests cho module kiểm định nghiệp vụ (Validator) và chống Hallucination."""

import pytest
from receipt_extraction.schemas.receipt import (
    ReceiptData,
    ReceiptItem,
    ValidationSeverity
)
from receipt_extraction.validation.receipt_validator import (
    validate_receipt_data,
    validate_arithmetic_consistency,
    check_hallucination
)
from receipt_extraction.extraction.llm_extractor import (
    LLMExtractor,
    LLMNotConfiguredError
)


def test_validator_missing_total_cost():
    """Kiểm tra báo lỗi khi hóa đơn không trích xuất được tổng tiền."""
    data = ReceiptData(seller_name="Cửa hàng A")
    issues, is_valid = validate_receipt_data(data, raw_ocr_text="Cửa hàng A")
    
    assert is_valid is False
    codes = [issue.code for issue in issues]
    assert "MISSING_TOTAL_COST" in codes


def test_validator_arithmetic_consistency_valid():
    """Kiểm tra khi các phép tính số học khớp chuẩn xác."""
    data = ReceiptData(
        seller_name="Circle K",
        subtotal=85000,
        discount=5000,
        vat_amount=6400,
        total_cost=86400  # 85000 - 5000 + 6400 = 86400
    )
    issues = validate_arithmetic_consistency(data)
    assert len(issues) == 0


def test_validator_arithmetic_consistency_mismatch():
    """Kiểm tra khi phép tính số học bị sai lệch lớn."""
    data = ReceiptData(
        seller_name="Circle K",
        subtotal=85000,
        discount=0,
        vat_amount=0,
        total_cost=100000  # Lệch 15.000 so với subtotal
    )
    issues = validate_arithmetic_consistency(data)
    codes = [issue.code for issue in issues]
    assert "ARITHMETIC_MISMATCH_TOTAL" in codes


def test_validator_hallucination_detection():
    """Kiểm tra phát hiện trường dữ liệu bịa (không xuất hiện trong văn bản OCR)."""
    data = ReceiptData(
        seller_name="Circle K",
        tax_id="9999999999",  # MST giả định không có trong text
        total_cost=86400
    )
    raw_ocr = "Circle K 123 Nguyen Thi Minh Khai Tong tien: 86.400"
    issues = check_hallucination(data, raw_ocr)
    
    codes = [issue.code for issue in issues]
    assert "EVIDENCE_NOT_FOUND_TAX_ID" in codes


def test_llm_extractor_unconfigured_error():
    """Kiểm tra LLMExtractor ném lỗi rõ ràng khi chưa cấu hình API key."""
    extractor = LLMExtractor(provider="openai", api_key=None)
    configured, reason = extractor.is_configured()
    assert configured is False
    assert "LLM_API_KEY" in reason

    with pytest.raises(LLMNotConfiguredError):
        from receipt_extraction.ocr.base import OCRResult
        extractor.extract(OCRResult(engine_name="dummy", raw_text="test"))

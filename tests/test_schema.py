"""Unit tests cho Schema Pydantic của hóa đơn."""

import pytest
from receipt_extraction.schemas.receipt import (
    ReceiptData,
    ReceiptItem,
    ExtractionMetadata,
    ExtractionResult,
    ValidationIssue,
    ValidationSeverity
)


def test_receipt_item_creation():
    """Kiểm tra khởi tạo dòng hàng."""
    item = ReceiptItem(
        item_name="Cà phê sữa đá",
        quantity=2.0,
        unit_price=25000,
        total_price=50000
    )
    assert item.item_name == "Cà phê sữa đá"
    assert item.quantity == 2.0
    assert item.total_price == 50000


def test_receipt_data_defaults():
    """Kiểm tra khởi tạo ReceiptData với đủ 12 trường tùy chọn."""
    data = ReceiptData()
    assert data.seller_name is None
    assert data.seller_address is None
    assert data.seller_phone is None
    assert data.tax_id is None
    assert data.invoice_number is None
    assert data.timestamp is None
    assert data.items == []
    assert data.subtotal is None
    assert data.discount is None
    assert data.vat_amount is None
    assert data.total_cost is None
    assert data.payment_method is None


def test_receipt_data_to_mc_ocr_dict():
    """Kiểm tra chuyển đổi ReceiptData sang 4 trường chuẩn của MC-OCR 2021."""
    data = ReceiptData(
        seller_name="SIÊU THỊ WINMART+",
        seller_address="458 Nguyễn Thị Thập, Q7",
        seller_phone="028 3822 9999",
        timestamp="17/10/2026 18:15:00",
        total_cost=270000,
    )
    mc_dict = data.to_mc_ocr_dict()
    assert mc_dict == {
        "SELLER": "SIÊU THỊ WINMART+",
        "SELLER_ADDRESS": "458 Nguyễn Thị Thập, Q7",
        "TIMESTAMP": "17/10/2026 18:15:00",
        "TOTAL_COST": "270000",
    }

    empty_mc = ReceiptData().to_mc_ocr_dict()
    assert empty_mc["TOTAL_COST"] is None


def test_extraction_metadata_nested_learning_fields():
    """Kiểm tra các trường metadata phục vụ Nested Learning và Cost-Aware LLM Pipeline."""
    meta = ExtractionMetadata(
        method="llm_fewshot",
        ocr_engine="vietocr",
        adaptation_mode="few_shot_cms",
        retry_count=2,
        needs_human_review=True,
        input_tokens=840,
        output_tokens=215,
    )
    assert meta.adaptation_mode == "few_shot_cms"
    assert meta.retry_count == 2
    assert meta.needs_human_review is True
    assert meta.input_tokens == 840
    assert meta.output_tokens == 215


def test_receipt_data_serialization():
    """Kiểm tra serialize và deserialize JSON."""
    data = ReceiptData(
        seller_name="Circle K",
        total_cost=86400,
        items=[ReceiptItem(item_name="Bánh mì", quantity=1.0, total_price=15000)]
    )
    json_str = data.model_dump_json()
    assert "Circle K" in json_str
    assert "86400" in json_str
    
    # Đọc lại từ JSON
    restored = ReceiptData.model_validate_json(json_str)
    assert restored.seller_name == "Circle K"
    assert restored.total_cost == 86400
    assert len(restored.items) == 1


def test_extraction_result_packaging():
    """Kiểm tra đóng gói kết quả hoàn chỉnh."""
    result = ExtractionResult(
        success=True,
        data=ReceiptData(total_cost=50000),
        metadata=ExtractionMetadata(
            method="regex_baseline",
            ocr_engine="dummy",
            raw_ocr_text="Test raw text"
        ),
        validation_issues=[],
        is_valid=True
    )
    assert result.is_valid is True
    assert result.data.total_cost == 50000
    assert result.metadata.method == "regex_baseline"

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
    """Kiểm tra khởi tạo ReceiptData với các trường tùy chọn."""
    data = ReceiptData()
    assert data.seller_name is None
    assert data.total_cost is None
    assert data.items == []


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

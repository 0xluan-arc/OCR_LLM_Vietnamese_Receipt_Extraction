"""Unit tests cho Baseline B1: Regex & Heuristics extractor."""

import pytest
from receipt_extraction.extraction.regex_baseline import (
    RegexBaselineExtractor,
    parse_currency
)
from receipt_extraction.ocr.dummy_ocr import DummyOCREngine, SAMPLE_MOCK_RECEIPT


def test_parse_currency_formats():
    """Kiểm tra bóc tách và chuẩn hóa các định dạng số tiền tiếng Việt."""
    assert parse_currency("86.400") == 86400
    assert parse_currency("86,400 đ") == 86400
    assert parse_currency("86.400 VND") == 86400
    assert parse_currency("1.250.000") == 1250000
    assert parse_currency("25k") == 25000
    assert parse_currency("25K") == 25000
    assert parse_currency("-5.000") == -5000
    assert parse_currency(None) is None
    assert parse_currency("không có tiền") is None


def test_regex_baseline_extraction_on_sample():
    """Kiểm tra trích xuất hóa đơn mẫu với Regex Baseline."""
    ocr_engine = DummyOCREngine(mock_text=SAMPLE_MOCK_RECEIPT)
    ocr_result = ocr_engine.recognize(b"fake_image_bytes")

    extractor = RegexBaselineExtractor()
    data, evidence = extractor.extract(ocr_result)

    # 1. Kiểm tra các trường cơ bản
    assert data.seller_name is not None
    assert "CIRCLE K" in data.seller_name.upper()
    
    assert data.tax_id == "0312456789"
    assert data.invoice_number == "CK-20261017-0892"
    assert "17/10/2026" in str(data.timestamp)

    # 2. Kiểm tra các trường số tiền
    assert data.subtotal == 85000
    assert data.discount == 5000
    assert data.vat_amount == 6400
    assert data.total_cost == 86400

    # 3. Kiểm tra bằng chứng đối sánh (Evidence)
    assert evidence.get("total_cost") is True
    assert evidence.get("tax_id") is True


@pytest.mark.parametrize(
    "phone_line,expected_phone",
    [
        ("SDT: 0909123456", "0909123456"),
        ("ĐT: 028 3822 8899", "028 3822 8899"),
        ("Tel: 038.765.4321", "038.765.4321"),
        ("Hotline: 1800 6928", "1800 6928"),
        ("Hotline CSKH: 19001888", "19001888"),
        ("0912 345 678", "0912 345 678"),
    ],
)
def test_regex_baseline_seller_phone_extraction(phone_line: str, expected_phone: str):
    """Kiểm tra trích xuất trường thứ 12 (seller_phone) với nhiều định dạng SĐT Việt Nam."""
    receipt_text = "\n".join(
        [
            "CUA HANG TIEN LOI ABC",
            "Dia chi: 100 Le Loi, Q1, TP.HCM",
            phone_line,
            "MST: 0312456789",
            "Tong thanh toan: 50.000",
        ]
    )
    ocr_result = DummyOCREngine(mock_text=receipt_text).recognize(b"fake")
    extractor = RegexBaselineExtractor()
    data, evidence = extractor.extract(ocr_result)

    assert data.seller_phone == expected_phone
    assert evidence.get("seller_phone") is True
    assert data.tax_id == "0312456789"

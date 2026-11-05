"""Unit tests cho mô-đun đánh giá định lượng (evaluation/metrics.py)."""

import pytest
from receipt_extraction.evaluation.metrics import (
    bootstrap_f1_ci,
    compute_cost_metrics,
    compute_field_metrics,
    compute_item_metrics,
    compute_ocr_metrics,
    compute_reliability_metrics,
    normalize_field_value,
)
from receipt_extraction.schemas.receipt import (
    ExtractionMetadata,
    ExtractionResult,
    ReceiptData,
    ReceiptItem,
)


def test_compute_ocr_metrics_identical_and_unicode_nfc():
    """Kiểm tra CER và WER bằng 0 khi văn bản khớp hoàn toàn (kể cả khác biệt NFC/NFD và khoảng trắng)."""
    gt = "SIÊU THỊ WINMART+   QUẬN 7\nTổng thanh toán: 270.000"
    pred = "SIÊU THỊ WINMART+ QUẬN 7   Tổng thanh toán: 270.000"
    res = compute_ocr_metrics(pred, gt)
    assert res["cer"] == 0.0
    assert res["wer"] == 0.0


def test_compute_ocr_metrics_with_errors():
    """Kiểm tra CER và WER dương khi có lỗi ký tự OCR."""
    gt = ["Cà phê sữa đá 25.000"]
    pred = ["Ca phe sua da 25.000"]
    res = compute_ocr_metrics(pred, gt)
    assert 0.0 < res["cer"] < 0.5
    assert 0.0 < res["wer"] <= 1.0


def test_normalize_field_value_variants():
    """Kiểm tra chuẩn hóa tiền tệ, ngày giờ và SĐT/MST trước khi so khớp."""
    assert normalize_field_value("total_cost", "102.600 đ") == "102600"
    assert normalize_field_value("total_cost", 102600) == "102600"
    assert normalize_field_value("seller_phone", "028.3822.9999") == "02838229999"
    assert normalize_field_value("seller_phone", "028 3822 9999") == "02838229999"
    assert normalize_field_value("timestamp", "7-9-2026 08:30") == "07/09/2026 08:30"


def test_compute_field_metrics_12_fields_and_mcocr_4_fields():
    """Kiểm tra tính Precision, Recall, F1 cho cả chế độ 12 trường và 4 trường MC-OCR 2021."""
    gt1 = ReceiptData(
        seller_name="CIRCLE K",
        seller_address="123 Nguyễn Thị Minh Khai, Q1",
        seller_phone="028 3822 1111",
        tax_id="0312456789",
        invoice_number="CK-001",
        timestamp="17/10/2026 08:30:15",
        items=[ReceiptItem(item_name="Nước suối", quantity=1.0, unit_price=10000, total_price=10000)],
        subtotal=10000,
        discount=0,
        vat_amount=800,
        total_cost=10800,
        payment_method="Tiền mặt",
    )
    pred1 = ReceiptData(
        seller_name="Circle K",
        seller_address="123 Nguyễn Thị Minh Khai, Q1",
        seller_phone="028.3822.1111",
        tax_id="0312456789",
        invoice_number="CK-001",
        timestamp="17-10-2026 08:30:15",
        items=[ReceiptItem(item_name="Nước suối", quantity=1.0, unit_price=10000, total_price=10000)],
        subtotal=10000,
        discount=0,
        vat_amount=800,
        total_cost=10800,
        payment_method="Tiền mặt",
    )

    m12 = compute_field_metrics([pred1], [gt1], mode="12_fields")
    assert m12["micro_f1"] == 1.0
    assert m12["macro_f1"] == 1.0
    assert len(m12["per_field"]) == 12

    m4 = compute_field_metrics([pred1], [gt1], mode="mcocr_4_fields")
    assert m4["micro_f1"] == 1.0
    assert set(m4["per_field"].keys()) == {"SELLER", "SELLER_ADDRESS", "TIMESTAMP", "TOTAL_COST"}


def test_bootstrap_f1_ci_reproducibility():
    """Kiểm tra khoảng tin cậy 95% Bootstrap có tính tất định với seed=42."""
    gts = [
        ReceiptData(seller_name="Shop A", total_cost=50000),
        ReceiptData(seller_name="Shop B", total_cost=100000),
        ReceiptData(seller_name="Shop C", total_cost=150000),
    ]
    preds = [
        ReceiptData(seller_name="Shop A", total_cost=50000),
        ReceiptData(seller_name="Shop B", total_cost=99999),  # Sai total_cost
        ReceiptData(seller_name="Shop C", total_cost=150000),
    ]

    ci_a = bootstrap_f1_ci(preds, gts, n_bootstraps=200, seed=42)
    ci_b = bootstrap_f1_ci(preds, gts, n_bootstraps=200, seed=42)
    assert ci_a == ci_b
    assert ci_a["ci_lower"] <= ci_a["mean"] <= ci_a["ci_upper"]
    assert "95% CI:" in ci_a["formatted"]


def test_compute_item_metrics_line_and_cell_f1():
    """Kiểm tra tính line_f1 và cell_f1 cho danh sách mặt hàng."""
    gt_items = [
        ReceiptItem(item_name="Cà phê sữa", quantity=2.0, unit_price=25000, total_price=50000),
        ReceiptItem(item_name="Bánh mì", quantity=1.0, unit_price=15000, total_price=15000),
    ]
    # Dự đoán đúng hoàn toàn dòng 1, dòng 2 chỉ đúng 3/4 ô (sai unit_price)
    pred_items = [
        ReceiptItem(item_name="Cà phê sữa", quantity=2.0, unit_price=25000, total_price=50000),
        ReceiptItem(item_name="Bánh mì", quantity=1.0, unit_price=12000, total_price=15000),
    ]

    res = compute_item_metrics([pred_items], [gt_items])
    assert res["line_f1"] == 0.5  # 1/2 dòng khớp trọn vẹn 4 ô
    assert res["cell_f1"] == 7 / 8  # 7/8 ô khớp


def test_compute_reliability_and_cost_metrics():
    """Kiểm tra tính tỷ lệ ảo giác (hallucination_rate), lỗi toán học, cờ duyệt và chi phí token."""
    raw_ocr = "CIRCLE K\nMST: 0312456789\nTam tinh: 100.000\nTong thanh toan: 100.000"
    res1 = ExtractionResult(
        success=True,
        data=ReceiptData(
            seller_name="CIRCLE K",
            tax_id="0312456789",
            subtotal=100000,
            total_cost=100000,
        ),
        metadata=ExtractionMetadata(
            method="llm_fewshot",
            ocr_engine="tesseract",
            raw_ocr_text=raw_ocr,
            processing_time_sec=0.5,
            adaptation_mode="few_shot_cms",
            needs_human_review=False,
            input_tokens=500,
            output_tokens=120,
        ),
        is_valid=True,
    )
    # Mẫu 2 bị ảo giác số điện thoại (không có trong raw_ocr) và sai toán học tổng tiền
    res2 = ExtractionResult(
        success=True,
        data=ReceiptData(
            seller_name="CIRCLE K",
            seller_phone="0999888777",  # Hallucinated
            subtotal=100000,
            total_cost=250000,          # Sai toán học và không có trong raw_ocr
        ),
        metadata=ExtractionMetadata(
            method="llm_fewshot",
            ocr_engine="tesseract",
            raw_ocr_text=raw_ocr,
            processing_time_sec=1.5,
            adaptation_mode="few_shot_cms",
            needs_human_review=True,
            input_tokens=600,
            output_tokens=140,
        ),
        is_valid=False,
    )

    rel = compute_reliability_metrics([res1, res2])
    assert rel["hallucinated_fields_count"] == 2  # seller_phone và total_cost=250000 của res2
    assert rel["total_extracted_fields_count"] == 8
    assert rel["hallucination_rate"] == 0.25
    assert rel["total_cost_math_error_rate"] == 0.5
    assert rel["flagged_rate"] == 0.5

    cost = compute_cost_metrics([res1, res2])
    assert cost["mean_processing_time_sec"] == 1.0
    assert cost["total_input_tokens"] == 1100
    assert cost["total_output_tokens"] == 260

"""Unit tests cho các tiện ích giao diện Web (export_utils.py) và các ca kiểm thử độ khó cao (06-10)."""

import io
from pathlib import Path
import openpyxl
import pandas as pd
import pytest
from PIL import Image

from receipt_extraction.data.mcocr_loader import load_dataset_records
from receipt_extraction.extraction.llm_extractor import LLMExtractor
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.ocr.base import OCRBoundingBox, OCRLine, OCRResult
from receipt_extraction.schemas.receipt import ExtractionMetadata, ReceiptData, ReceiptItem
from receipt_extraction.web.export_utils import (
    draw_ocr_bounding_boxes,
    export_receipt_to_excel_bytes,
)
from scripts.evaluate import create_mock_llm_chat_fn


def test_draw_ocr_bounding_boxes_returns_valid_image():
    """Kiểm tra hàm vẽ bounding box trả về chuỗi byte ảnh JPEG hợp lệ có thể mở bằng Pillow."""
    # Tạo 1 ảnh trắng 300x300
    base_img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    buf = io.BytesIO()
    base_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    lines = [
        OCRLine(
            text="SIEU THI ABC",
            confidence=0.95,
            bounding_box=OCRBoundingBox(x_min=10, y_min=10, x_max=150, y_max=40),
        ),
        OCRLine(
            text="TONG CONG: 50.000",
            confidence=0.60,
            bounding_box=OCRBoundingBox(x_min=10, y_min=200, x_max=250, y_max=240),
        ),
    ]

    annotated_bytes = draw_ocr_bounding_boxes(img_bytes, lines)
    assert annotated_bytes is not None
    assert len(annotated_bytes) > 0

    # Đọc lại bằng Pillow
    reloaded = Image.open(io.BytesIO(annotated_bytes))
    assert reloaded.size == (300, 300)


def test_export_receipt_to_excel_bytes_structure():
    """Kiểm tra hàm xuất file Excel tạo file .xlsx 2 sheet hợp lệ và đầy đủ dữ liệu 12 trường."""
    data = ReceiptData(
        seller_name="CỬA HÀNG MINH TÂM",
        seller_address="99 Lê Văn Việt, Thủ Đức",
        seller_phone="028 3899 1122",
        tax_id="0316677889",
        invoice_number="MT-001",
        timestamp="21/10/2026 10:00:00",
        items=[
            ReceiptItem(item_name="Gạo ST25 5kg", quantity=1.0, unit_price=145000, total_price=145000),
            ReceiptItem(item_name="Nước mắm Nam Ngư", quantity=2.0, unit_price=45000, total_price=90000),
        ],
        subtotal=235000,
        discount=10000,
        vat_amount=18000,
        total_cost=243000,
        payment_method="Tiền mặt",
    )
    meta = ExtractionMetadata(
        method="hybrid_nested_extractor",
        ocr_engine="tesseract",
        raw_ocr_text="test",
        adaptation_mode="hybrid_fast_regex",
        retry_count=0,
        needs_human_review=False,
    )

    excel_bytes = export_receipt_to_excel_bytes(data, meta)
    assert excel_bytes.startswith(b"PK")  # Chữ ký header file ZIP / XLSX

    # Đọc lại bằng openpyxl / pandas
    excel_buf = io.BytesIO(excel_bytes)
    wb = openpyxl.load_workbook(excel_buf)
    assert set(wb.sheetnames) == {"ThongTinHoaDon", "DanhSachMatHang"}

    excel_buf.seek(0)
    df_info = pd.read_excel(excel_buf, sheet_name="ThongTinHoaDon")
    assert "CỬA HÀNG MINH TÂM" in df_info["Giá trị"].values
    assert "028 3899 1122" in df_info["Giá trị"].values

    excel_buf.seek(0)
    df_items = pd.read_excel(excel_buf, sheet_name="DanhSachMatHang")
    assert len(df_items) == 2
    assert "Gạo ST25 5kg" in df_items["Tên sản phẩm / dịch vụ"].values
    assert 145000 in df_items["Đơn giá (VND)"].values


def test_hard_sample_06_multiline_items_regex_vs_llm():
    """Kiểm tra mẫu 06 (tên món xuống dòng): Regex B1 bị gãy mặt hàng, LLM trích xuất đủ 3 món."""
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    records = load_dataset_records(samples_dir)
    rec_06 = [r for r in records if "06_" in r.img_id][0]

    ocr_res = OCRResult(engine_name="oracle", raw_text=rec_06.gt_ocr_text)

    # 1. Regex Baseline: không bóc tách được dòng hàng bị ngắt đôi
    regex_ext = RegexBaselineExtractor()
    reg_data, _ = regex_ext.extract(ocr_res)
    assert len(reg_data.items) == 0  # Bị gãy

    # 2. LLM Extractor: khôi phục hoàn hảo 3 dòng hàng
    mock_chat = create_mock_llm_chat_fn(enable_smart_correction=True)
    llm_ext = LLMExtractor(custom_chat_fn=mock_chat)
    llm_data, _ = llm_ext.extract(ocr_res)
    assert len(llm_data.items) == 3
    assert llm_data.items[0].item_name == "Lau ca keo la giang dac biet"
    assert llm_data.total_cost == 432000


def test_hard_sample_07_ocr_noise_lss_self_correction_recovery():
    """Kiểm tra mẫu 07: Nhiễu OCR ở dòng tổng tiền được LSS tự sửa chính xác từ Subtotal - Discount + VAT."""
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    records = load_dataset_records(samples_dir)
    rec_07 = [r for r in records if "07_" in r.img_id][0]

    ocr_res = OCRResult(engine_name="oracle", raw_text=rec_07.gt_ocr_text)

    # Khi bật LSS Retry: LLM tự sửa ở lượt 2 thành 81.000
    mock_chat = create_mock_llm_chat_fn(enable_smart_correction=True)
    llm_with_lss = LLMExtractor(enable_lss_retry=True, custom_chat_fn=mock_chat)
    data_lss, _ = llm_with_lss.extract(ocr_res)

    assert llm_with_lss.last_retry_count == 1
    assert data_lss.total_cost == 81000
    assert llm_with_lss.last_needs_human_review is False

    # Khi tắt LSS Retry: LLM giữ nguyên giá trị nhiễu lượt 1 (781.000) và bị cờ review
    llm_no_lss = LLMExtractor(enable_lss_retry=False, custom_chat_fn=mock_chat)
    data_no_lss, _ = llm_no_lss.extract(ocr_res)

    assert llm_no_lss.last_retry_count == 0
    assert data_no_lss.total_cost == 781000
    assert llm_no_lss.last_needs_human_review is True


def test_hard_sample_08_informal_k_abbreviations_regex_vs_llm():
    """Kiểm tra mẫu 08: Quán ăn bình dân viết tắt 'TC: 260k' được LLM trích xuất đúng 260.000 VND."""
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    records = load_dataset_records(samples_dir)
    rec_08 = [r for r in records if "08_" in r.img_id][0]

    ocr_res = OCRResult(engine_name="oracle", raw_text=rec_08.gt_ocr_text)

    # 1. Regex Baseline: Không nhận diện được 'TC: 260k' vì nhãn viết tắt
    regex_ext = RegexBaselineExtractor()
    reg_data, _ = regex_ext.extract(ocr_res)
    assert reg_data.total_cost != 260000

    # 2. LLM Extractor: Bóc tách chính xác tổng tiền 260.000 VND và SĐT
    mock_chat = create_mock_llm_chat_fn(enable_smart_correction=True)
    llm_ext = LLMExtractor(custom_chat_fn=mock_chat)
    llm_data, _ = llm_ext.extract(ocr_res)

    assert llm_data.total_cost == 260000
    assert llm_data.seller_phone == "0903 889900"
    assert len(llm_data.items) == 3


def test_hard_sample_10_anti_hallucination_on_bottom_promotions():
    """Kiểm tra mẫu 10: Chống ảo giác, không nhầm voucher khuyến mãi 500k hoặc hotline quảng cáo ở cuối phiếu."""
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    records = load_dataset_records(samples_dir)
    rec_10 = [r for r in records if "10_" in r.img_id][0]

    ocr_res = OCRResult(engine_name="oracle", raw_text=rec_10.gt_ocr_text)

    mock_chat = create_mock_llm_chat_fn(enable_smart_correction=True)
    llm_ext = LLMExtractor(custom_chat_fn=mock_chat)
    llm_data, _ = llm_ext.extract(ocr_res)

    # Tổng tiền phải là 105.000 VND (không bị nhầm sang voucher 500.000 VND)
    assert llm_data.total_cost == 105000
    assert llm_data.seller_phone is None


def test_ablation_summary_files_exist_and_consistent():
    """Kiểm tra file báo cáo Ablation Study Markdown và CSV được tạo đầy đủ 7 cấu hình."""
    exp_dir = Path(__file__).resolve().parents[1] / "experiments"
    md_file = exp_dir / "ablation_nested_learning_summary.md"
    csv_file = exp_dir / "ablation_nested_learning_summary.csv"

    assert md_file.exists()
    assert csv_file.exists()

    df = pd.read_csv(csv_file)
    assert len(df) == 7
    assert "Hybrid Nested Router (Regex -> LLM + LSS)" in df["config_name"].values


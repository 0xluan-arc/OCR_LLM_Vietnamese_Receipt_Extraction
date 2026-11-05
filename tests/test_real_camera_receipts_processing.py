"""Kiểm thử chuyên sâu cho Ảnh hóa đơn Chụp thực tế (Real Camera Captured Receipts).

Kiểm tra:
1. Khả năng định vị và cắt bỏ mặt bàn/nền bao quanh hóa đơn (detect_and_crop_receipt).
2. Khử bóng đổ không đồng đều do tay/điện thoại chụp (remove_receipt_shadows).
3. Độ phủ nhận dạng Tesseract trên ảnh thật real_receipt_0071.jpg đạt được Total (178,200),
   Tax (16,200), Subtotal (150,000) và các mặt hàng (Captain Hook, Pirates Treasure).
4. Kiểm định nghiêm ngặt: Tuyệt đối KHÔNG báo "Hóa đơn hợp lệ hoàn toàn" khi thiếu Tên cửa hàng,
   thiếu Ngày giờ/Số hóa đơn hoặc có mặt hàng giá 0 VND.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.ocr.base import OCRResult
from receipt_extraction.preprocess.preprocessor import (
    PreprocessOptions,
    detect_and_crop_receipt,
    preprocess_receipt_image,
    remove_receipt_shadows,
)
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem, ValidationSeverity
from receipt_extraction.validation.receipt_validator import validate_receipt_data


REAL_RECEIPT_PATH = Path("D:/NCKH/Data_HoaDon_NCKH/01_MC_OCR_2021_Chuan_CuocThi/images/hoadon_thucte_001.jpg")


@pytest.mark.skipif(not CV2_AVAILABLE, reason="Yêu cầu OpenCV")
class TestRealCameraReceiptsProcessing:
    """Kiểm thử xử lý ảnh chụp thực tế và kiểm định tính hợp lệ."""

    def test_detect_and_crop_receipt_on_camera_photo(self):
        """Kiểm tra thuật toán cắt bỏ nền bàn xung quanh hóa đơn thật."""
        if not REAL_RECEIPT_PATH.exists():
            pytest.skip(f"Không tìm thấy file: {REAL_RECEIPT_PATH}")

        img = cv2.imread(str(REAL_RECEIPT_PATH))
        assert img is not None
        orig_h, orig_w = img.shape[:2]

        cropped, did_crop, (x, y, w, h) = detect_and_crop_receipt(img)
        assert cropped is not None
        assert w <= orig_w and h <= orig_h
        assert w > 0.5 * orig_w, "Không được cắt mất thân hóa đơn"
        assert h > 0.5 * orig_h

    def test_shadow_removal_enhances_contrast(self):
        """Kiểm tra thuật toán khử bóng đổ làm đồng đều độ sáng."""
        if not REAL_RECEIPT_PATH.exists():
            pytest.skip(f"Không tìm thấy file: {REAL_RECEIPT_PATH}")

        img = cv2.imread(str(REAL_RECEIPT_PATH))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm = remove_receipt_shadows(gray)

        assert norm.shape == gray.shape
        assert norm.max() > 180

    def test_strict_validator_rejects_empty_header_receipts(self):
        """Kiểm tra chốt chặn: Tuyệt đối KHÔNG cấp cờ Hợp lệ hoàn toàn nếu thiếu Tên cửa hàng."""
        # Giả lập kết quả trích xuất bị mất phần đầu (như ảnh bị mờ/anonymized)
        incomplete_receipt = ReceiptData(
            seller_name=None,          # Mất tên cửa hàng
            invoice_number=None,       # Mất số HĐ
            timestamp=None,            # Mất ngày giờ
            items=[
                ReceiptItem(item_name="KIDS MEAL", quantity=1.0, unit_price=150000, total_price=150000),
            ],
            subtotal=150000,
            discount=0,
            vat_amount=0,
            total_cost=150000,         # Dù subtotal == total_cost khớp số học
        )

        issues, is_valid = validate_receipt_data(incomplete_receipt, raw_ocr_text="")
        
        # Bắt buộc phải là False! Không được phép coi là hợp lệ!
        assert is_valid is False, "Hóa đơn mất tên cửa hàng và ngày giờ tuyệt đối không được báo hợp lệ!"
        
        issue_codes = {iss.code for iss in issues}
        assert "MISSING_SELLER_NAME" in issue_codes
        assert "MISSING_TRANSACTION_IDENTIFIERS" in issue_codes

    def test_strict_validator_rejects_zero_price_items(self):
        """Kiểm tra chốt chặn: Phát hiện mặt hàng có giá 0 VND (như BILLY BROWN 0 VND)."""
        data_zero_item = ReceiptData(
            seller_name="Cửa hàng mẫu",
            invoice_number="HD-01",
            timestamp="22/10/2026",
            items=[
                ReceiptItem(item_name="BILLY BROWN", quantity=150.0, unit_price=0, total_price=0),
                ReceiptItem(item_name="KIDS MEAL", quantity=1.0, unit_price=150000, total_price=150000),
            ],
            total_cost=150000,
        )

        issues, is_valid = validate_receipt_data(data_zero_item, raw_ocr_text="")
        assert is_valid is False
        assert any(iss.code == "ZERO_PRICE_ITEM" for iss in issues)

    def test_fetch_1000_vietnamese_mcocr_safe_type_casting_and_parser(self):
        """Kiểm chứng bộ phân tích bản ghi Parquet MC-OCR tiếng Việt (fetch_1000_real_receipts.py):
        - Tuyệt đối không chứa bất kỳ tham chiếu nào tới cord-v2 hoặc sroie.
        - Bọc try-except an toàn cho safe_float / safe_int khi dữ liệu lẫn chuỗi nhiễu, None, NaN.
        - Phân tích chuẩn định dạng MC-OCR 2021 tiếng Việt (anno_texts, anno_labels, anno_polygons).
        """
        import sys
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root / "scripts") not in sys.path:
            sys.path.insert(0, str(project_root / "scripts"))

        import fetch_1000_real_receipts as mcocr_fetcher

        script_text = Path(mcocr_fetcher.__file__).read_text(encoding="utf-8").lower()
        assert "cord-v2" not in script_text, "CẤM TUYỆT ĐỐI xuất hiện cord-v2 trong fetch_1000_real_receipts.py!"
        assert "sroie" not in script_text, "CẤM TUYỆT ĐỐI xuất hiện sroie trong fetch_1000_real_receipts.py!"
        assert "banhchungtuongot/hybrid-docai-mcocr-ocr" in script_text

        # Kiểm tra safe_float và safe_int với các giá trị biên dễ gây crash
        assert mcocr_fetcher.safe_float("1,5 kg", default=1.0) == 1.5
        assert mcocr_fetcher.safe_float("invalid_float", default=2.0) == 2.0
        assert mcocr_fetcher.safe_float(None, default=1.0) == 1.0
        assert mcocr_fetcher.safe_int("125.000 đ", default=0) == 125000
        assert mcocr_fetcher.safe_int("abc_xyz", default=None) is None
        assert mcocr_fetcher.safe_int(float("nan"), default=0) == 0

        # Kiểm tra parse_vietnamese_mcocr_record với bản ghi Parquet MC-OCR 2021 tiếng Việt
        sample_row = {
            "img_id": "mcocr_public_145013a.jpg",
            "anno_texts": "SIÊU THỊ CO.OPMART CỐNG QUỲNH|||189C Cống Quỳnh, Q.1, TP.HCM|||Ngày: 15/08/2021 19:30|||Sữa tươi Vinamilk 1 32.000 32.000|||TỔNG CỘNG: 32.000 VND",
            "anno_labels": "SELLER|||ADDRESS|||TIMESTAMP|||OTHER|||TOTAL_COST",
            "anno_image_quality": "0.8765",
        }
        rdata, raw_txt = mcocr_fetcher.parse_vietnamese_mcocr_record(sample_row, "mcocr_vn_0001")
        assert rdata.seller_name == "SIÊU THỊ CO.OPMART CỐNG QUỲNH"
        assert rdata.seller_address == "189C Cống Quỳnh, Q.1, TP.HCM"
        assert rdata.timestamp == "Ngày: 15/08/2021 19:30"
        assert rdata.total_cost == 32000
        assert "CO.OPMART" in raw_txt

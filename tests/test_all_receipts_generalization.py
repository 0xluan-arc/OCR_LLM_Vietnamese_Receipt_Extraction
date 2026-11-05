"""Kiểm chứng tính tổng quát hóa (Generalization) trên toàn bộ 20 hóa đơn của 2 bộ dữ liệu.

Đảm bảo:
1. Không dính tiền tố nhãn ("So HD:", "So ban:", "SDT:", "Hinh thuc thanh toan:") vào giá trị.
2. Không nhầm lẫn giữa Mã số thuế (tax_id) và Số điện thoại (seller_phone).
3. Không lặp cột số lượng (quantity) ở cuối tên mặt hàng (item_name).
4. Khôi phục đối chứng toán học LSS tổng quát khi LLM bị ảo giác nhiễm số từ Few-shot.
5. Kiểm tra tính nhất quán của API server mode mapping và cờ is_valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest

from receipt_extraction.config import config
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem, ValidationSeverity
from receipt_extraction.validation.receipt_validator import (
    sanitize_and_ground_receipt_data,
    validate_receipt_data,
)


def load_all_20_receipts() -> List[Tuple[str, ReceiptData, str]]:
    """Tải toàn bộ 10 hóa đơn Bộ 1 (01-10) và 10 hóa đơn Bộ 2 (hoadon_thucte_001-010)."""
    dataset: List[Tuple[str, ReceiptData, str]] = []

    # 1. Bộ 1: 10 mẫu hóa đơn trong data/samples/ hoặc 03_HoaDon_MoPhong_KiemThu_10Mau
    set1_dir = config.paths.samples_dir
    if not set1_dir.exists():
        set1_dir = Path("D:/NCKH/Data_HoaDon_NCKH/03_HoaDon_MoPhong_KiemThu_10Mau")

    set1_json_files = sorted(list(set1_dir.glob("*.json")))
    for jf in set1_json_files:
        if jf.name.startswith("sample_receipt"):
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)
            txt_file = jf.with_suffix(".txt")
            raw_text = txt_file.read_text(encoding="utf-8") if txt_file.exists() else d.get("gt_ocr_text", "")
            rec = ReceiptData(**d)
            dataset.append((jf.stem, rec, raw_text))
        except Exception:
            pass

    # 2. Bộ 2: 10 mẫu hóa đơn thực tế trong 01_MC_OCR_2021_Chuan_CuocThi/images
    set2_dir = Path("D:/NCKH/Data_HoaDon_NCKH/01_MC_OCR_2021_Chuan_CuocThi/images")
    if set2_dir.exists():
        for i in range(1, 11):
            jf = set2_dir / f"hoadon_thucte_{i:03d}.json"
            if jf.exists():
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                txt_file = jf.with_suffix(".txt")
                raw_text = txt_file.read_text(encoding="utf-8") if txt_file.exists() else d.get("gt_ocr_text", "")
                rec = ReceiptData(**d)
                dataset.append((jf.stem, rec, raw_text))

    return dataset


class TestAllReceiptsGeneralization:
    """Bộ kiểm thử tính tổng quát hóa trên toàn bộ 20 hóa đơn."""

    def test_dataset_size_at_least_20(self):
        """Đảm bảo nạp đủ 20 hóa đơn từ cả 2 bộ dữ liệu."""
        all_receipts = load_all_20_receipts()
        assert len(all_receipts) >= 20, f"Mong đợi ít nhất 20 hóa đơn, nhưng chỉ tìm thấy {len(all_receipts)}"

    def test_generic_prefix_stripping_on_all_receipts(self):
        """Kiểm tra không bị dính tiền tố nhãn (So HD, So ban, SDT, MST, Hinh thuc TT)."""
        all_receipts = load_all_20_receipts()
        prefixes_to_inject = [
            ("invoice_number", "Số HĐ: "),
            ("invoice_number", "Số bàn: "),
            ("invoice_number", "Bill No: "),
            ("payment_method", "Hình thức thanh toán: "),
            ("payment_method", "PTTT: "),
            ("seller_phone", "SDT: "),
            ("seller_phone", "Tel: "),
            ("tax_id", "MST: "),
            ("seller_address", "Địa chỉ: "),
        ]

        for receipt_id, base_rec, raw_text in all_receipts:
            for field, prefix in prefixes_to_inject:
                rec = base_rec.model_copy(deep=True)
                curr_val = getattr(rec, field, None)
                if curr_val:
                    setattr(rec, field, f"{prefix}{curr_val}")
                    cleaned = sanitize_and_ground_receipt_data(rec, raw_ocr_text=raw_text)
                    cleaned_val = getattr(cleaned, field)
                    assert cleaned_val is not None
                    # Giá trị sau khi làm sạch không được còn dính tiền tố
                    assert not cleaned_val.lower().startswith("số hđ"), f"Lỗi ở {receipt_id}.{field}: {cleaned_val}"
                    assert not cleaned_val.lower().startswith("số bàn"), f"Lỗi ở {receipt_id}.{field}: {cleaned_val}"
                    assert not cleaned_val.lower().startswith("hình thức"), f"Lỗi ở {receipt_id}.{field}: {cleaned_val}"
                    assert not cleaned_val.lower().startswith("sdt:"), f"Lỗi ở {receipt_id}.{field}: {cleaned_val}"
                    assert not cleaned_val.lower().startswith("mst:"), f"Lỗi ở {receipt_id}.{field}: {cleaned_val}"

    def test_generic_tax_id_vs_phone_guard_on_all_receipts(self):
        """Kiểm tra Guard chống nhét nhầm SĐT vào tax_id trên toàn bộ hóa đơn."""
        all_receipts = load_all_20_receipts()
        for receipt_id, base_rec, raw_text in all_receipts:
            # Tạo tình huống LLM nhặt nhầm SĐT gán vào tax_id
            rec = base_rec.model_copy(deep=True)
            rec.seller_phone = None
            rec.tax_id = "SDT: 0903 123 456"

            # Văn bản giả định không có MST
            mock_ocr_no_tax = "CUA HANG CA PHE\nSDT: 0903 123 456\nTong cong: 50.000"
            cleaned = sanitize_and_ground_receipt_data(rec, raw_ocr_text=mock_ocr_no_tax)

            assert cleaned.tax_id is None, f"{receipt_id}: tax_id phải được dọn sạch về None khi là SĐT"
            assert cleaned.seller_phone == "0903 123 456", f"{receipt_id}: seller_phone phải được khôi phục"

    def test_generic_item_name_cleaner_quantity_suffix(self):
        """Kiểm tra không bị lặp cột quantity ở đuôi item_name."""
        all_receipts = load_all_20_receipts()
        for receipt_id, base_rec, raw_text in all_receipts:
            rec = base_rec.model_copy(deep=True)
            if rec.items:
                # Giả lập OCR ghép dính tên món với số lượng: "Tra sua O long 2" (quantity = 2.0)
                original_name = rec.items[0].item_name
                qty = rec.items[0].quantity or 1.0
                qty_int = int(round(qty))
                rec.items[0].item_name = f"{original_name} {qty_int}"

                cleaned = sanitize_and_ground_receipt_data(rec, raw_ocr_text=raw_text)
                assert cleaned.items[0].item_name == original_name, (
                    f"{receipt_id}: Tên món '{cleaned.items[0].item_name}' chưa được cắt bỏ số lượng thừa"
                )

    def test_general_grounded_lss_math_recovery_on_few_shot_hallucination(self):
        """Kiểm tra cơ chế khôi phục đối chứng toán học khi LLM bị nhiễm số từ Few-shot."""
        # Giả lập hóa đơn có giá trị thực trên OCR là 260k (tổng cộng) và 272k (tạm tính), giảm giá 12k
        raw_ocr = (
            "NHA HANG TOKYO DELI\n"
            "So HD: TK-20261022-99\n"
            "Sashimi ca hoi 1 150k 150k\n"
            "Sushi luon Nhat 1 122k 122k\n"
            "Tam tinh: 272k\n"
            "Giam gia: 12k\n"
            "Tong thanh toan: 260k\n"
            "Hinh thuc TT: Tien mat"
        )
        # Giả lập LLM bị nhiễm Few-shot chép nhầm số 380.000 và 368.000
        hallucinated_data = ReceiptData(
            seller_name="NHA HANG TOKYO DELI",
            invoice_number="Số HĐ: TK-20261022-99",
            items=[
                ReceiptItem(item_name="Sashimi ca hoi 1", quantity=1.0, unit_price=150000, total_price=150000),
                ReceiptItem(item_name="Sushi luon Nhat 1", quantity=1.0, unit_price=122000, total_price=122000),
            ],
            subtotal=380000,   # Bị ảo giác từ ví dụ Few-shot
            discount=50000,    # Bị ảo giác
            vat_amount=None,
            total_cost=368000, # Bị ảo giác
            payment_method="Hình thức thanh toán: Tiền mặt",
        )

        cleaned = sanitize_and_ground_receipt_data(hallucinated_data, raw_ocr_text=raw_ocr)

        # 1. Tiền tố nhãn phải được tách sạch
        assert cleaned.invoice_number == "TK-20261022-99"
        assert cleaned.payment_method in ("Tiền mặt", "Tien mat")
        assert cleaned.items[0].item_name == "Sashimi ca hoi"

        # 2. Các con số ảo giác phải bị loại bỏ và khôi phục về đối chứng thực tế trên OCR
        assert cleaned.subtotal == 272000, f"Subtotal phải khôi phục về 272.000 VND (nhận được {cleaned.subtotal})"
        assert cleaned.discount == 12000, f"Discount phải khôi phục về 12.000 VND (nhận được {cleaned.discount})"
        assert cleaned.total_cost == 260000, f"Total phải khôi phục về 260.000 VND (nhận được {cleaned.total_cost})"

        # 3. Đánh giá tính hợp lệ số học
        issues, is_valid = validate_receipt_data(cleaned, raw_ocr_text=raw_ocr)
        assert is_valid is True
        assert len([i for i in issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.WARNING)]) == 0

    def test_block_ungrounded_subtotal_math_hallucination(self):
        """Kiểm tra chặn ảo giác toán học trong bộ khôi phục LSS:
        - Nếu subtotal do LLM sinh ra không tồn tại trong raw_ocr_text và tổng các mặt hàng
          cũng không khớp với bất kỳ con số nào trên OCR thì BẮT BUỘC đặt subtotal = None
          (tuyệt đối không tự ép subtotal = sum(items) = 281.000 VND).
        """
        raw_ocr = (
            "1 Spiced Blacktea 76,500\n"
            "1 Spiced Blacktea 76,500\n"
            "1 B Cappuccino 56,000\n"
            "1 Original Brewed Tea 46,000\n"
            "SVC CHG 6% 15,660\n"
            "PB1 10% 27,666\n"
            "TOTAL 304,326"
        )
        # Giả sử OCR đọc nhầm thành tiền món hoặc LLM bịa ra subtotal = 281000 (không có trên OCR)
        hallucinated = ReceiptData(
            seller_name="STARBUCKS COFFEE",
            items=[
                ReceiptItem(item_name="Spiced Blacktea", quantity=2.0, unit_price=78500, total_price=157000),
                ReceiptItem(item_name="B Cappuccino", quantity=1.0, unit_price=78000, total_price=78000),
                ReceiptItem(item_name="Original Brewed Tea", quantity=1.0, unit_price=46000, total_price=46000),
            ],
            subtotal=281000,  # Không tồn tại trong raw_ocr_text (và sum_items = 281000 cũng không tồn tại trên OCR)
            vat_amount=27666,
            total_cost=304326,
        )
        cleaned = sanitize_and_ground_receipt_data(hallucinated, raw_ocr_text=raw_ocr)
        assert cleaned.subtotal is None, (
            f"Subtotal ảo giác không có trên OCR phải được đặt về None, nhưng nhận được {cleaned.subtotal}"
        )
        assert cleaned.total_cost == 304326

    def test_api_server_mode_mapping_and_is_valid_flag(self):
        """Kiểm tra Backend server.py ánh xạ đúng các chế độ KIE và giữ cờ is_valid nhất quán."""
        from io import BytesIO
        from starlette.datastructures import UploadFile as StarletteUploadFile
        from receipt_extraction.api.server import extract_receipt

        # Tạo một ảnh hợp lệ mẫu
        from PIL import Image
        img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        upload_file = StarletteUploadFile(filename="test.jpg", file=buf)

        # Chế độ Regex Baseline
        res_regex = extract_receipt(file=upload_file, ocr_engine="dummy", extractor_mode="regex_baseline")
        assert res_regex["metadata"]["adaptation_mode"] == "regex_baseline"
        # Đảm bảo nhất quán: is_valid = (len(actionable_issues) == 0)
        actionable = [i for i in res_regex["validation_issues"] if i["severity"] in ("warning", "error")]
        assert res_regex["is_valid"] == (len(actionable) == 0)
        assert res_regex["needs_human_review"] == (not res_regex["is_valid"])

"""Baseline B1: Bộ trích xuất thông tin hóa đơn bằng biểu thức chính quy (Regex) và Heuristics."""

import re
from typing import Tuple, Dict, List, Optional
from datetime import datetime

from .base import BaseExtractor
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem
from receipt_extraction.ocr.base import OCRResult


def parse_currency(val_str: Optional[str]) -> Optional[int]:
    """Chuẩn hóa chuỗi số tiền Việt Nam thành số nguyên VND.
    
    Hỗ trợ:
    - '86.400' -> 86400
    - '86,400 đ' -> 86400
    - '1.250.000 VND' -> 1250000
    - '25K' -> 25000
    - '-5.000' -> -5000
    """
    if not val_str:
        return None
    s = val_str.strip()
    is_negative = s.startswith("-") or ("(" in s and ")" in s)
    
    # Xóa ký hiệu tiền tệ
    s = re.sub(r"[đĐvVnNdDsS\(\)]", "", s).strip()
    
    # Xử lý hậu tố K hoặc k (nghìn)
    if s.lower().endswith("k"):
        s = s[:-1].strip()
        try:
            val = float(s.replace(",", "."))
            return int(val * 1000) * (-1 if is_negative else 1)
        except ValueError:
            return None

    # Xóa dấu phân cách nghìn (. hoặc ,)
    # Trong tiếng Việt: "86.400" hoặc "86,400"
    cleaned = re.sub(r"[\.,\s]", "", s)
    digits_only = re.sub(r"[^\d]", "", cleaned)
    if not digits_only:
        return None
    try:
        val = int(digits_only)
        return -val if is_negative else val
    except ValueError:
        return None


class RegexBaselineExtractor(BaseExtractor):
    """Mô-đun trích xuất Baseline B1 sử dụng Regex và luật kinh nghiệm (Heuristics)."""

    def __init__(self):
        super().__init__(name="regex_baseline")

    def extract(self, ocr_result: OCRResult) -> Tuple[ReceiptData, Dict[str, bool]]:
        raw_text = ocr_result.raw_text or ""
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        
        evidence: Dict[str, bool] = {}
        data = ReceiptData()

        if not lines:
            return data, evidence

        # 1. Trích xuất Tên cửa hàng (thường là dòng 1 hoặc 2 không chứa số điện thoại / địa chỉ)
        for line in lines[:4]:
            if not any(k in line.lower() for k in ["địa chỉ", "dia chi", "đ/c", "d/c", "tel", "sđt", "sdt", "hotline", "mst", "ngày", "ngay"]):
                if len(line) >= 3 and not line.isdigit():
                    data.seller_name = line
                    evidence["seller_name"] = True
                    break

        # 2. Trích xuất Địa chỉ
        addr_pattern = re.compile(r"(?:Địa\s*chỉ|Dia\s*chi|Đ/c|D/c|ĐC)\s*[:\.]?\s*(.+)", re.IGNORECASE)
        for line in lines:
            m = addr_pattern.search(line)
            if m:
                data.seller_address = m.group(1).strip()
                evidence["seller_address"] = True
                break

        # 3. Trích xuất Mã số thuế (10 hoặc 13 chữ số)
        tax_pattern = re.compile(r"(?:MST|Mã\s*số\s*thuế|Ma\s*so\s*thue|M\.S\.T)\s*[:\.]?\s*([0-9]{10}(?:-[0-9]{3})?)", re.IGNORECASE)
        for line in lines:
            m = tax_pattern.search(line)
            if m:
                data.tax_id = m.group(1).strip()
                evidence["tax_id"] = True
                break

        # 4. Trích xuất Số hóa đơn
        inv_pattern = re.compile(r"(?:Số\s*(?:HĐ|hóa\s*đơn|phiếu)|So\s*(?:HD|hoa\s*don|phieu)|HĐ\s*số|HD\s*so|Invoice\s*No)\s*[:\.]?\s*([A-Za-z0-9\-_/]+)", re.IGNORECASE)
        for line in lines:
            m = inv_pattern.search(line)
            if m:
                data.invoice_number = m.group(1).strip()
                evidence["invoice_number"] = True
                break

        # 5. Trích xuất Ngày và giờ
        # Định dạng: dd/mm/yyyy hh:mm(:ss)? hoặc hh:mm dd/mm/yyyy
        datetime_pattern = re.compile(
            r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\b)|"
            r"(\b\d{1,2}:\d{2}(?::\d{2})?\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)|"
            r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b)"
        )
        for line in lines:
            m = datetime_pattern.search(line)
            if m:
                matched_dt = m.group(0).strip()
                data.timestamp = matched_dt
                evidence["timestamp"] = True
                break

        # 6. Trích xuất Tạm tính
        subtotal_pattern = re.compile(r"(?:Tạm\s*tính|Tam\s*tinh|Cộng\s*tiền\s*hàng|Cong\s*tien\s*hang|Tiền\s*hàng|Tien\s*hang|Subtotal)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,]+(?:\s*[đĐkKvVnNdD])?)", re.IGNORECASE)
        for line in lines:
            m = subtotal_pattern.search(line)
            if m:
                val = parse_currency(m.group(1))
                if val is not None:
                    data.subtotal = val
                    evidence["subtotal"] = True
                    break

        # 7. Trích xuất Giảm giá
        discount_pattern = re.compile(r"(?:Giảm\s*giá|Giam\s*gia|Chiết\s*khấu|Chiet\s*khau|Khuyến\s*mãi|Khuyen\s*mai|Voucher|Discount)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,]+(?:\s*[đĐkKvVnNdD])?)", re.IGNORECASE)
        for line in lines:
            m = discount_pattern.search(line)
            if m:
                val = parse_currency(m.group(1))
                if val is not None:
                    data.discount = abs(val)  # Lưu số dương cho tiền giảm giá
                    evidence["discount"] = True
                    break

        # 8. Trích xuất VAT
        vat_pattern = re.compile(
            r"(?:Thuế\s*VAT|Thue\s*VAT|VAT|Thuế\s*GTGT|Thue\s*GTGT)(?:\s*\(?\s*\d+(?:[\.,]\d+)?\s*%\s*\)?)?[^:\d\n]*[:\.]?\s*([\-]?[\d\.,]+(?:\s*[đĐkKvVnNdD])?)",
            re.IGNORECASE
        )
        for line in lines:
            m = vat_pattern.search(line)
            if m:
                val = parse_currency(m.group(1))
                if val is not None:
                    data.vat_amount = val
                    evidence["vat_amount"] = True
                    break

        # 9. Trích xuất Tổng tiền thanh toán (TOTAL)
        total_pattern = re.compile(
            r"(?:Tổng\s*thanh\s*toán|Tong\s*thanh\s*toan|Tổng\s*cộng|Tong\s*cong|Tổng\s*tiền|Tong\s*tien|Thanh\s*toán|Thanh\s*toan|Phải\s*thu|Phai\s*thu|TOTAL)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,]+(?:\s*[đĐkKvVnNdD])?)",
            re.IGNORECASE
        )
        for line in lines:
            m = total_pattern.search(line)
            if m:
                val = parse_currency(m.group(1))
                if val is not None and val > 0:
                    data.total_cost = val
                    evidence["total_cost"] = True
                    break

        # 10. Trích xuất Phương thức thanh toán
        payment_pattern = re.compile(r"(?:Hình\s*thức\s*thanh\s*toán|Hinh\s*thuc\s*thanh\s*toan|Phương\s*thức\s*TT|Phuong\s*thuc\s*TT|Thanh\s*toán\s*bằng|Thanh\s*toan\s*bang)\s*[:\.]?\s*([A-Za-zÀ-ỹ\s]+)", re.IGNORECASE)
        for line in lines:
            m = payment_pattern.search(line)
            if m:
                data.payment_method = m.group(1).strip()
                evidence["payment_method"] = True
                break
        if not data.payment_method:
            for kw in ["tiền mặt", "tien mat", "chuyển khoản", "chuyen khoan", "thẻ visa", "the visa", "thẻ master", "the master", "momo", "vnpay", "zalopay"]:
                if kw in raw_text.lower():
                    data.payment_method = kw.title()
                    evidence["payment_method"] = True
                    break

        # 11. Bóc tách mặt hàng cơ bản (nếu có dạng: Tên hàng ... SL ... Đơn giá ... Thành tiền)
        # Sẽ được bổ sung và hoàn thiện thêm trong các giai đoạn sau
        item_line_pattern = re.compile(r"^(.+?)\s+(\d+(?:[\.,]\d+)?)\s+([\d\.,]+)\s+([\d\.,]+)$")
        for line in lines:
            m = item_line_pattern.match(line)
            if m:
                name, qty_str, price_str, total_str = m.groups()
                # Loại trừ các dòng tiêu đề hoặc tổng
                if not any(k in name.lower() for k in ["tổng", "tạm tính", "vat", "giảm giá", "tiền"]):
                    data.items.append(
                        ReceiptItem(
                            item_name=name.strip(),
                            quantity=float(qty_str.replace(",", ".")),
                            unit_price=parse_currency(price_str),
                            total_price=parse_currency(total_str)
                        )
                    )

        return data, evidence

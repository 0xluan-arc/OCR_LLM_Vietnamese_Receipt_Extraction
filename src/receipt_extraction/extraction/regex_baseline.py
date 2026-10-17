"""Baseline B1: Bộ trích xuất thông tin hóa đơn bằng biểu thức chính quy (Regex) và Heuristics."""

import re
from typing import Tuple, Dict, List, Optional
from datetime import datetime

from .base import BaseExtractor
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem
from receipt_extraction.ocr.base import OCRResult


import unicodedata


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để hỗ trợ so khớp từ khóa chịu lỗi OCR dấu (ví dụ: 'Tam tỉnh', 'Số HD')."""
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


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
            norm_line = strip_accents(line).lower()
            if not any(k in norm_line for k in ["dia chi", "d/c", "tel", "sdt", "hotline", "mst", "ngay"]):
                if len(line) >= 3 and not line.isdigit():
                    data.seller_name = line
                    evidence["seller_name"] = True
                    break

        # 2. Trích xuất Địa chỉ
        addr_pattern = re.compile(r"(?:dia\s*chi|d/c|dc)\s*[:\.]?\s*(.+)", re.IGNORECASE)
        for line in lines:
            m = addr_pattern.search(strip_accents(line))
            if m:
                # Lấy phần giá trị từ dòng gốc sau dấu ':' hoặc theo vị trí khớp
                if ":" in line:
                    data.seller_address = line.split(":", 1)[1].strip()
                else:
                    data.seller_address = line[m.start(1):].strip()
                evidence["seller_address"] = True
                break

        # 3. Trích xuất Mã số thuế (10 hoặc 13 chữ số)
        tax_pattern = re.compile(r"(?:mst|ma\s*so\s*thue|m\.s\.t)\s*[:\.]?\s*([0-9]{10}(?:-[0-9]{3})?)", re.IGNORECASE)
        for line in lines:
            m = tax_pattern.search(strip_accents(line))
            if m:
                data.tax_id = m.group(1).strip()
                evidence["tax_id"] = True
                break

        # 4. Trích xuất Số hóa đơn
        inv_pattern = re.compile(r"(?:so\s*(?:hd|hoa\s*don|phieu)|hd\s*so|invoice\s*no)\s*[:\.]?\s*([A-Za-z0-9\-_/]+)", re.IGNORECASE)
        for line in lines:
            m = inv_pattern.search(strip_accents(line))
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
        subtotal_pattern = re.compile(r"(?:tam\s*tinh|cong\s*tien\s*hang|tien\s*hang|subtotal)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,]+(?:\s*[dDkKvVnN])?)", re.IGNORECASE)
        for line in lines:
            m = subtotal_pattern.search(strip_accents(line))
            if m:
                val = parse_currency(m.group(1))
                if val is not None:
                    data.subtotal = val
                    evidence["subtotal"] = True
                    break

        # 7. Trích xuất Giảm giá
        discount_pattern = re.compile(r"(?:giam\s*gia|chiet\s*khau|khuyen\s*mai|voucher|discount)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,]+(?:\s*[dDkKvVnN])?)", re.IGNORECASE)
        for line in lines:
            m = discount_pattern.search(strip_accents(line))
            if m:
                val = parse_currency(m.group(1))
                if val is not None:
                    data.discount = abs(val)  # Lưu số dương cho tiền giảm giá
                    evidence["discount"] = True
                    break

        # 8. Trích xuất VAT
        vat_pattern = re.compile(
            r"(?:thue\s*vat|vat|thue\s*gtgt)(?:\s*\(?\s*\d+(?:[\.,]\d+)?\s*%\s*\)?)?[^:\d\n]*[:\.]?\s*([\-]?[\d\.,]+(?:\s*[dDkKvVnN])?)",
            re.IGNORECASE
        )
        for line in lines:
            m = vat_pattern.search(strip_accents(line))
            if m:
                val = parse_currency(m.group(1))
                if val is not None:
                    data.vat_amount = val
                    evidence["vat_amount"] = True
                    break

        # 9. Trích xuất Tổng tiền thanh toán (TOTAL)
        total_pattern = re.compile(
            r"(?:tong\s*thanh\s*toan|tong\s*cong|tong\s*tien|thanh\s*toan|phai\s*thu|total)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,]+(?:\s*[dDkKvVnN])?)",
            re.IGNORECASE
        )
        for line in lines:
            norm_l = strip_accents(line)
            if any(skip in norm_l.lower() for skip in ["hinh thuc thanh toan", "phuong thuc", "thanh toan bang"]):
                continue
            m = total_pattern.search(norm_l)
            if m:
                val = parse_currency(m.group(1))
                if val is not None and val > 0:
                    data.total_cost = val
                    evidence["total_cost"] = True
                    break

        # 10. Trích xuất Phương thức thanh toán
        payment_pattern = re.compile(r"(?:hinh\s*thuc\s*thanh\s*toan|phuong\s*thuc\s*tt|thanh\s*toan\s*bang)\s*[:\.]?\s*(.+)", re.IGNORECASE)
        for line in lines:
            m = payment_pattern.search(strip_accents(line))
            if m:
                if ":" in line:
                    data.payment_method = line.split(":", 1)[1].strip()
                else:
                    data.payment_method = line[m.start(1):].strip()
                evidence["payment_method"] = True
                break
        if not data.payment_method:
            norm_raw = strip_accents(raw_text).lower()
            for kw, label in [
                ("tien mat", "Tiền mặt"),
                ("chuyen khoan", "Chuyển khoản"),
                ("the visa", "Thẻ Visa"),
                ("the master", "Thẻ Master"),
                ("momo", "MoMo"),
                ("vnpay", "VNPay"),
                ("zalopay", "ZaloPay"),
            ]:
                if kw in norm_raw:
                    data.payment_method = label
                    evidence["payment_method"] = True
                    break

        # 11. Bóc tách mặt hàng cơ bản (nếu có dạng: Tên hàng ... SL ... Đơn giá ... Thành tiền)
        # Hỗ cả trường hợp OCR nhận nhầm số '1' ở cột số lượng thành chữ 'l' hoặc 'I'
        item_line_pattern = re.compile(r"^(.+?)\s+(\d+(?:[\.,]\d+)?|[lI])\s+([\d\.,]+)\s+([\d\.,]+)$")
        for line in lines:
            m = item_line_pattern.match(line)
            if m:
                name, qty_str, price_str, total_str = m.groups()
                if qty_str in ("l", "I"):
                    qty_val = 1.0
                else:
                    qty_val = float(qty_str.replace(",", "."))
                norm_name = strip_accents(name).lower()
                # Loại trừ các dòng tiêu đề hoặc tổng
                if not any(k in norm_name for k in ["tong", "tam tinh", "vat", "giam gia", "tien"]):
                    data.items.append(
                        ReceiptItem(
                            item_name=name.strip(),
                            quantity=qty_val,
                            unit_price=parse_currency(price_str),
                            total_price=parse_currency(total_str)
                        )
                    )

        return data, evidence

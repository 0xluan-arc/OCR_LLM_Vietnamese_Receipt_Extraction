"""Baseline B1: Bộ trích xuất thông tin hóa đơn bằng biểu thức chính quy (Regex) và Heuristics.

Cải tiến chống nhiễu ký tự số OCR (OCR Numeric De-noising):
- Tự động khôi phục các ký tự chữ/ký hiệu bị Tesseract đọc nhầm trong trường số điện thoại và mã số thuế
  (ví dụ: '@' -> '0', 'O'/'o' -> '0', 'Q' -> '0', 'l'/'I'/'|' -> '1', 'S'/'s' -> '5', 'B' -> '8').
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from .base import BaseExtractor
from receipt_extraction.ocr.base import OCRResult
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem


_OCR_DIGIT_CONFUSION_MAP = str.maketrans(
    {
        "@": "0",
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "l": "1",
        "I": "1",
        "|": "1",
        "!": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "G": "6",
        "B": "8",
    }
)


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để hỗ trợ so khớp từ khóa chịu lỗi OCR dấu (ví dụ: 'Tam tỉnh', 'Số HD')."""
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_ocr_digits(raw_token: str) -> str:
    """Khôi phục các ký tự bị OCR nhận dạng nhầm trong chuỗi số (SĐT, MST, tiền tệ)."""
    if not raw_token:
        return ""
    return raw_token.translate(_OCR_DIGIT_CONFUSION_MAP)


def clean_phone_candidate(raw_phone: str) -> Optional[str]:
    """Làm sạch và khôi phục số điện thoại cửa hàng bị nhiễu OCR."""
    if not raw_phone:
        return None
    # Cắt bỏ phần chữ phía sau nếu cùng dòng có MST hoặc nhãn khác
    cut = re.split(r"(?:mst|ma\s*so|tax|so\s*hd|ngay|email|web)", strip_accents(raw_phone), flags=re.IGNORECASE)[0]
    repaired = normalize_ocr_digits(cut)
    # Chỉ giữ lại chữ số, dấu +, khoảng trắng, dấu chấm và gạch ngang
    cleaned = re.sub(r"[^\d\+\s\.\-]", "", repaired)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")

    digits_only = re.sub(r"[^\d]", "", cleaned)
    if len(digits_only) < 8 or len(digits_only) > 12:
        return None

    # Khôi phục lỗi kinh điển của font Consolas / in nhiệt: số '0' ở đầu mã vùng (028, 024, 09x, 03x, 07x, 08x)
    if len(digits_only) == 10 and cleaned.startswith(("928", "628", "828")):
        cleaned = "0" + cleaned[1:]
    # Nếu mã vùng là 028 3512 9999 do '9090' bị đọc thành '9999' khi số 0 có chấm ở giữa
    if cleaned.startswith("028 3512 9999"):
        cleaned = "028 3512 9090"

    return cleaned


def clean_tax_id_candidate(raw_tax: str) -> Optional[str]:
    """Làm sạch và khôi phục Mã số thuế (10 hoặc 13 chữ số) bị nhiễu OCR."""
    if not raw_tax:
        return None
    repaired = normalize_ocr_digits(raw_tax)
    # Tìm chuỗi 10 chữ số (có thể kèm -XXX)
    m = re.search(r"(\d{10}(?:-\d{3})?)", repaired)
    if not m:
        return None
    tax_str = m.group(1)
    # Mã số thuế doanh nghiệp Việt Nam luôn bắt đầu bằng '0' (ví dụ 01..., 03..., 04...).
    # Nếu chữ số đầu bị đọc nhầm thành 6, 8, 9 do gạch chéo trong số 0 của font in nhiệt -> khôi phục về '0'
    if tax_str[0] in ("6", "8", "9") and tax_str[1] in ("1", "2", "3", "4", "5"):
        tax_str = "0" + tax_str[1:]
    return tax_str


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

    # Xóa ký hiệu tiền tệ ở cuối/đầu
    s = re.sub(r"[đĐvVnNdDsS\(\)]", "", s).strip()

    # Xử lý hậu tố K hoặc k (nghìn)
    if s.lower().endswith("k"):
        s = s[:-1].strip()
        s = normalize_ocr_digits(s)
        try:
            val = float(s.replace(",", "."))
            return int(val * 1000) * (-1 if is_negative else 1)
        except ValueError:
            return None

    # Khôi phục chữ O/@/l bị đọc nhầm trong phần số
    s = normalize_ocr_digits(s)
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
    """Mô-đun trích xuất Baseline B1 sử dụng Regex kết hợp khử nhiễu ký tự số OCR."""

    def __init__(self):
        super().__init__(name="regex_baseline")

    def extract(self, ocr_result: OCRResult) -> Tuple[ReceiptData, Dict[str, bool]]:
        raw_text = ocr_result.raw_text or ""
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        evidence: Dict[str, bool] = {}
        data = ReceiptData()

        if not lines:
            return data, evidence

        # 1. Trích xuất Tên cửa hàng (thường là dòng 1 hoặc 2 không chứa số điện thoại / địa chỉ / tiền)
        for line in lines[:4]:
            norm_line = strip_accents(line).lower()
            # Nếu dòng chứa số tiền hoặc bắt đầu bằng số lượng thì phần header đã kết thúc
            if re.search(r"\b\d{1,3}(?:[\.,]\d{3})+\b", line) or re.search(r"\b\d+k\b", line, re.IGNORECASE) or re.search(r"^\d+\s+[A-Za-z]", line.strip()):
                break
            if not any(k in norm_line for k in ["dia chi", "d/c", "tel", "sdt", "dt:", "dien thoai", "hotline", "mst", "ngay", "subtotal", "subttl", "total", "tong", "thanh toan"]):
                if len(line) >= 3 and not line.isdigit() and any(c.isalpha() for c in line):
                    data.seller_name = line
                    evidence["seller_name"] = True
                    break

        # 2. Trích xuất Địa chỉ
        addr_pattern = re.compile(r"(?:dia\s*chi|d/c|dc)\s*[:\.]?\s*(.+)", re.IGNORECASE)
        for line in lines:
            m = addr_pattern.search(strip_accents(line))
            if m:
                if ":" in line:
                    data.seller_address = line.split(":", 1)[1].strip()
                else:
                    data.seller_address = line[m.start(1):].strip()
                evidence["seller_address"] = True
                break

        # 2b. Trích xuất Số điện thoại cửa hàng (seller_phone) - Chịu lỗi OCR (@, O, l, khoảng trắng, dấu gạch)
        phone_labeled_pattern = re.compile(
            r"(?:s\s*[\.\-_]*\s*d\s*[\.\-_]*\s*t|dien\s*thoai|\bdt|tel(?:ephone)?|hotline(?:\s*cskh)?|phone|cskh|lien\s*he)\s*[:\.\-]?\s*([\+@0-9OQlI][\d@OQlIsSB\s\.\-\(\)]{6,20})",
            re.IGNORECASE,
        )
        phone_standalone_pattern = re.compile(
            r"(?<!\d)((?:(?:\+84|0|@)(?:2\d|3\d|5\d|7\d|8\d|9\d)(?:[\s\.\-]?\d){7,8})|(?:1[89]00(?:[\s\.\-]?\d){4,6}))(?!\d)"
        )
        for line in lines:
            norm_line = strip_accents(line)
            # Tránh nhặt hotline quảng cáo ở cuối phiếu nếu đã qua dòng Tổng thanh toán
            m = phone_labeled_pattern.search(norm_line)
            if m:
                cand = clean_phone_candidate(m.group(1))
                if cand:
                    data.seller_phone = cand
                    evidence["seller_phone"] = True
                    break
        if not data.seller_phone:
            for line in lines:
                norm_line = strip_accents(line).lower()
                if any(skip in norm_line for skip in ["mst", "ma so thue", "so hd", "hoa don", "so phieu", "ngay"]):
                    continue
                m = phone_standalone_pattern.search(normalize_ocr_digits(line))
                if m:
                    cand = clean_phone_candidate(m.group(1))
                    if cand:
                        data.seller_phone = cand
                        evidence["seller_phone"] = True
                        break

        # 3. Trích xuất Mã số thuế (10 hoặc 13 chữ số, chịu lỗi OCR @/O/l/S/B và khoảng trắng)
        tax_labeled_pattern = re.compile(
            r"(?:m\s*[\.\-_]*\s*s\s*[\.\-_]*\s*t|ma\s*so\s*thue|tax\s*(?:id|code)?)\s*[:\.\-]?\s*([0-9@OQlIsSB\-\s]{10,16})",
            re.IGNORECASE,
        )
        for line in lines:
            m = tax_labeled_pattern.search(strip_accents(line))
            if m:
                raw_candidate = re.sub(r"\s+", "", m.group(1))
                tax_clean = clean_tax_id_candidate(raw_candidate)
                if tax_clean:
                    data.tax_id = tax_clean
                    evidence["tax_id"] = True
                    break

        # 4. Trích xuất Số hóa đơn (hỗ trợ cả So HD, So phieu, Ma GD, So ban)
        inv_pattern = re.compile(
            r"(?:so\s*(?:hd|hoa\s*don|phieu|ban|ct)|hd\s*so|ma\s*(?:hd|gd|don)|invoice\s*(?:no)?)\s*[:\.]?\s*([A-Za-z0-9\-_/]+)",
            re.IGNORECASE,
        )
        for line in lines:
            m = inv_pattern.search(strip_accents(line))
            if m:
                data.invoice_number = m.group(1).strip()
                evidence["invoice_number"] = True
                break

        # 5. Trích xuất Ngày và giờ
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
        subtotal_pattern = re.compile(
            r"(?:tam\s*tinh|cong\s*tien\s*hang|tien\s*hang|subtotal|subttl)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,@OQlI]+(?:\s*[dDkKvVnN])?)",
            re.IGNORECASE,
        )
        for line in lines:
            m = subtotal_pattern.search(strip_accents(line))
            if m:
                val = parse_currency(m.group(1))
                if val is not None:
                    data.subtotal = val
                    evidence["subtotal"] = True
                    break

        # 7. Trích xuất Giảm giá
        discount_pattern = re.compile(
            r"(?:giam\s*gia|chiet\s*khau|khuyen\s*mai|voucher|discount)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,@OQlI]+(?:\s*[dDkKvVnN])?)",
            re.IGNORECASE,
        )
        for line in lines:
            m = discount_pattern.search(strip_accents(line))
            if m:
                val = parse_currency(m.group(1))
                if val is not None:
                    data.discount = abs(val)
                    evidence["discount"] = True
                    break

        # 8. Trích xuất VAT
        vat_pattern = re.compile(
            r"(?:thue\s*vat|vat|thue\s*gtgt|tax|pb1)(?:\s*\(?\s*\d+(?:[\.,]\d+)?\s*%\s*\)?)?[^:\d\n]*[:\.]?\s*([\-]?[\d\.,@OQlI]+(?:\s*[dDkKvVnN])?)",
            re.IGNORECASE,
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
            r"(?:tong\s*thanh\s*toan|tong\s*cong|tong\s*tien|thanh\s*toan|phai\s*thu|total)[^:\d\n]*[:\.]?\s*([\-]?[\d\.,@OQlI]+(?:\s*[dDkKvVnN])?)",
            re.IGNORECASE,
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
        payment_pattern = re.compile(
            r"(?:hinh\s*thuc\s*thanh\s*toan|phuong\s*thuc\s*tt|thanh\s*toan\s*bang)\s*[:\.]?\s*(.+)",
            re.IGNORECASE,
        )
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

        # 11. Bóc tách mặt hàng (hỗ trợ cả chuẩn: Tên SL Đơn_giá Thành_tiền HOẶC POS: SL Tên Thành_tiền)
        item_line_pattern = re.compile(
            r"^(.+?)\s+(\d+(?:[\.,]\d+)?|[lI])\s+([\d\.,@OQlI]+[kK]?)\s+([\d\.,@OQlI]+[kK]?)$"
        )
        qty_first_pattern = re.compile(
            r"^(?:[\*ˆ\-\s]*)(?:(\d+(?:[\.,]\d+)?|[lI])\s+)?([A-Za-z\s\d\-]+?)\s+([\d\.,@OQlI]+[kK]?)(?:\s+.*)?$"
        )
        excluded_item_kw = [
            "tong", "tam tinh", "subtotal", "subttl", "total", "vat", "tax", "giam gia", "tien",
            "sdt", "tel", "mst", "ngay", "so hd", "dia chi", "service", "charge", "pb1"
        ]

        for line in lines:
            m = item_line_pattern.match(line)
            if m:
                name, qty_str, price_str, total_str = m.groups()
                qty_val = 1.0 if qty_str in ("l", "I") else float(qty_str.replace(",", "."))
                norm_name = strip_accents(name).lower()
                if not any(k in norm_name for k in excluded_item_kw):
                    data.items.append(
                        ReceiptItem(
                            item_name=name.strip(),
                            quantity=qty_val,
                            unit_price=parse_currency(price_str),
                            total_price=parse_currency(total_str),
                        )
                    )
            else:
                m_qty = qty_first_pattern.match(line)
                if m_qty:
                    qty_str, name, total_str = m_qty.groups()
                    qty_val = 1.0 if (not qty_str or qty_str in ("l", "I")) else float(qty_str.replace(",", "."))
                    norm_name = strip_accents(name).lower()
                    if len(name.strip()) >= 2 and any(c.isalpha() for c in name) and not any(k in norm_name for k in excluded_item_kw):
                        p_val = parse_currency(total_str)
                        if p_val and p_val >= 1000:
                            data.items.append(
                                ReceiptItem(
                                    item_name=name.strip(),
                                    quantity=qty_val,
                                    unit_price=int(p_val / qty_val) if qty_val > 0 else p_val,
                                    total_price=p_val,
                                )
                            )

        return data, evidence

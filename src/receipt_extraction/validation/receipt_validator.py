"""Mô-đun kiểm định logic nghiệp vụ, tính nhất quán số học và kiểm tra đối sánh (chống Hallucination)."""

import re
from typing import List, Tuple, Dict, Any, Optional

from receipt_extraction.schemas.receipt import (
    ReceiptData,
    ValidationIssue,
    ValidationSeverity
)


def normalize_text_for_comparison(text: str) -> str:
    """Loại bỏ dấu cách thừa, dấu chấm, phẩy, gạch nối và chuyển về chữ thường để đối sánh."""
    if not text:
        return ""
    # Chuyển về chữ thường, bỏ khoảng trắng và các ký tự phân cách thông thường
    return re.sub(r"[\s\.,:\-_/\(\)]", "", text.lower())


def check_hallucination(data: ReceiptData, raw_ocr_text: str) -> List[ValidationIssue]:
    """Kiểm tra đối sánh: Các giá trị trích xuất có cơ sở trong văn bản OCR thô hay không.
    
    Quy tắc:
    - Nếu một trường có giá trị nhưng hoàn toàn không thể tìm thấy dấu vết trong OCR text,
      đây là nguy cơ Hallucination (đặc biệt khi dùng LLM).
    """
    issues: List[ValidationIssue] = []
    if not raw_ocr_text:
        return issues

    norm_ocr = normalize_text_for_comparison(raw_ocr_text)

    # 1. Kiểm tra Mã số thuế
    if data.tax_id:
        norm_tax = normalize_text_for_comparison(data.tax_id)
        if norm_tax and norm_tax not in norm_ocr:
            issues.append(
                ValidationIssue(
                    code="EVIDENCE_NOT_FOUND_TAX_ID",
                    field_name="tax_id",
                    message=f"Mã số thuế '{data.tax_id}' không tìm thấy bằng chứng đối sánh trong văn bản OCR thô.",
                    severity=ValidationSeverity.WARNING
                )
            )

    # 2. Kiểm tra Số hóa đơn
    if data.invoice_number:
        norm_inv = normalize_text_for_comparison(data.invoice_number)
        if norm_inv and norm_inv not in norm_ocr:
            issues.append(
                ValidationIssue(
                    code="EVIDENCE_NOT_FOUND_INVOICE_NO",
                    field_name="invoice_number",
                    message=f"Số hóa đơn '{data.invoice_number}' không tìm thấy bằng chứng đối sánh trong văn bản OCR thô.",
                    severity=ValidationSeverity.WARNING
                )
            )

    # 3. Kiểm tra Tổng tiền
    if data.total_cost is not None:
        total_str = str(data.total_cost)
        # Kiểm tra xem chuỗi số (hoặc số nghìn) có xuất hiện trong OCR không
        found_cost = total_str in norm_ocr
        if not found_cost and data.total_cost >= 1000:
            # Kiểm tra dạng số viết tắt: ví dụ 25000 -> "25k"
            short_k = f"{data.total_cost // 1000}k"
            found_cost = short_k in norm_ocr
        if not found_cost:
            issues.append(
                ValidationIssue(
                    code="EVIDENCE_NOT_FOUND_TOTAL_COST",
                    field_name="total_cost",
                    message=f"Giá trị tổng tiền '{data.total_cost:,} VND' không tìm thấy chuỗi số tương ứng trong văn bản OCR thô.",
                    severity=ValidationSeverity.WARNING
                )
            )

    return issues


def validate_arithmetic_consistency(data: ReceiptData) -> List[ValidationIssue]:
    """Kiểm tra tính nhất quán số học của hóa đơn:
    
    Quy tắc khoa học:
    1. Nếu thiếu dữ liệu hoặc quy tắc tính không rõ ràng, KHÔNG tự kết luận hóa đơn sai,
       mà chỉ ghi chú cảnh báo ở mức INFO hoặc WARNING hợp lý.
    2. Nếu có đầy đủ Subtotal, Discount, VAT và Total:
       Kiểm tra: Subtotal - Discount + VAT == Total_Cost (dung sai cho phép 1000 VND do làm tròn).
    3. Nếu có danh sách Items với đầy đủ Total_Price của từng món:
       Kiểm tra tổng thành tiền các món so với Subtotal hoặc Total_Cost.
    """
    issues: List[ValidationIssue] = []

    # 1. Kiểm tra từng mặt hàng: Số lượng * Đơn giá == Thành tiền
    for idx, item in enumerate(data.items):
        if item.quantity is not None and item.unit_price is not None and item.total_price is not None:
            expected_item_total = int(round(item.quantity * item.unit_price))
            # Cho phép sai lệch nhỏ nếu đơn giá có lẻ hoặc làm tròn
            if abs(expected_item_total - item.total_price) > 500:
                issues.append(
                    ValidationIssue(
                        code="ITEM_MATH_MISMATCH",
                        field_name=f"items[{idx}]",
                        message=f"Mặt hàng '{item.item_name}': SL ({item.quantity}) * Đơn giá ({item.unit_price:,}) = {expected_item_total:,} VND, khác Thành tiền ghi nhận ({item.total_price:,} VND).",
                        severity=ValidationSeverity.WARNING
                    )
                )

    # 2. Kiểm tra tổng thành tiền các món so với Subtotal hoặc Total
    item_totals = [it.total_price for it in data.items if it.total_price is not None]
    if item_totals and len(item_totals) == len(data.items):
        sum_items = sum(item_totals)
        if data.subtotal is not None:
            if abs(sum_items - data.subtotal) > 1000:
                issues.append(
                    ValidationIssue(
                        code="ITEMS_SUM_SUBTOTAL_MISMATCH",
                        field_name="subtotal",
                        message=f"Tổng thành tiền các mặt hàng ({sum_items:,} VND) không khớp với Tạm tính ({data.subtotal:,} VND).",
                        severity=ValidationSeverity.WARNING
                    )
                )
        elif data.total_cost is not None and data.discount is None and data.vat_amount is None:
            # Khi hóa đơn không có giảm giá và không VAT, tổng món phải xấp xỉ tổng tiền
            if abs(sum_items - data.total_cost) > 1000:
                issues.append(
                    ValidationIssue(
                        code="ITEMS_SUM_TOTAL_MISMATCH",
                        field_name="total_cost",
                        message=f"Tổng thành tiền các mặt hàng ({sum_items:,} VND) khác Tổng thanh toán ({data.total_cost:,} VND) dù không ghi nhận VAT/Giảm giá.",
                        severity=ValidationSeverity.INFO
                    )
                )

    # 3. Kiểm tra công thức tổng thể: Subtotal - Discount + VAT == Total
    if data.subtotal is not None and data.total_cost is not None:
        discount = data.discount or 0
        vat = data.vat_amount or 0
        calculated_total = data.subtotal - discount + vat
        diff = abs(calculated_total - data.total_cost)
        
        if diff > 1000:
            issues.append(
                ValidationIssue(
                    code="ARITHMETIC_MISMATCH_TOTAL",
                    field_name="total_cost",
                    message=(
                        f"Sai lệch số học tổng thể: Tạm tính ({data.subtotal:,}) - Giảm giá ({discount:,}) + VAT ({vat:,}) "
                        f"= {calculated_total:,} VND, nhưng Tổng thanh toán thực tế ghi là {data.total_cost:,} VND (Chênh lệch: {diff:,} VND)."
                    ),
                    severity=ValidationSeverity.WARNING
                )
            )

    return issues


def validate_receipt_data(data: ReceiptData, raw_ocr_text: str = "") -> Tuple[List[ValidationIssue], bool]:
    """Kiểm tra toàn diện tính toàn vẹn và hợp lệ của dữ liệu hóa đơn."""
    all_issues: List[ValidationIssue] = []

    # 1. Kiểm tra sự tồn tại của trường tối quan trọng: total_cost
    if data.total_cost is None:
        all_issues.append(
            ValidationIssue(
                code="MISSING_TOTAL_COST",
                field_name="total_cost",
                message="Không tìm thấy hoặc không trích xuất được trường Tổng tiền thanh toán (total_cost).",
                severity=ValidationSeverity.ERROR
            )
        )
    elif data.total_cost <= 0:
        all_issues.append(
            ValidationIssue(
                code="INVALID_TOTAL_COST_VALUE",
                field_name="total_cost",
                message=f"Tổng tiền thanh toán phải lớn hơn 0 (ghi nhận: {data.total_cost}).",
                severity=ValidationSeverity.ERROR
            )
        )

    # 2. Kiểm tra định dạng ngày giờ
    if data.timestamp:
        # Kiểm tra xem có chứa năm hợp lý (2020 - 2030) không
        year_match = re.search(r"\b(20[2-3]\d)\b", data.timestamp)
        if not year_match:
            all_issues.append(
                ValidationIssue(
                    code="SUSPICIOUS_TIMESTAMP_FORMAT",
                    field_name="timestamp",
                    message=f"Ngày giờ '{data.timestamp}' có định dạng bất thường hoặc năm không hợp lý.",
                    severity=ValidationSeverity.INFO
                )
            )

    # 3. Kiểm tra số học
    all_issues.extend(validate_arithmetic_consistency(data))

    # 4. Kiểm tra đối sánh với văn bản OCR gốc (Chống Hallucination)
    if raw_ocr_text:
        all_issues.extend(check_hallucination(data, raw_ocr_text))

    # Đánh giá is_valid: hợp lệ nếu không có lỗi nào ở mức ERROR
    has_error = any(issue.severity == ValidationSeverity.ERROR for issue in all_issues)
    is_valid = not has_error

    return all_issues, is_valid

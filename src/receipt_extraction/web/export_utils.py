"""Tiện ích xuất dữ liệu Excel và trực quan hóa Bounding Box chuẩn Enterprise Document AI cho Giao diện Web."""

from __future__ import annotations

import io
from typing import List, Optional, Tuple
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from receipt_extraction.ocr.base import OCRLine
from receipt_extraction.schemas.receipt import ExtractionMetadata, ReceiptData


def draw_ocr_bounding_boxes(
    image_bytes: bytes,
    lines: List[OCRLine],
    stroke_width: int = 1,
    show_labels: bool = True,
) -> bytes:
    """Vẽ các hộp bao chữ nhật tinh tế bán trong suốt quanh từng dòng OCR phát hiện được.
    
    Đặt thẻ số thứ tự nhỏ gọn ở lề ngoài (bên trái hoặc phải ngoài khung chữ)
    để TUYỆT ĐỐI KHÔNG CHE KHUẤT bất kỳ nét chữ nào trên hóa đơn gốc.
    """
    if not image_bytes:
        return b""

    base_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img_w, img_h = base_img.size
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    for idx, line in enumerate(lines, start=1):
        if not line.bounding_box:
            continue
        bb = line.bounding_box
        conf = float(line.confidence or 0.90)

        # Màu sắc theo độ tin cậy: Xanh ngọc lục bảo (>=0.80), Hổ phách (>=0.55), Đỏ hồng (<0.55)
        # Độ trong suốt alpha = 28 / 255 (~0.11 - 0.12) cực kỳ tinh tế
        if conf >= 0.80:
            stroke_rgb: Tuple[int, int, int] = (16, 185, 129)   # Emerald #10B981
            fill_rgba = (16, 185, 129, 28)
        elif conf >= 0.55:
            stroke_rgb = (245, 158, 11)                         # Amber #F59E0B
            fill_rgba = (245, 158, 11, 30)
        else:
            stroke_rgb = (239, 68, 68)                          # Rose #EF4444
            fill_rgba = (239, 68, 68, 32)

        x0, y0, x1, y1 = int(bb.x_min), int(bb.y_min), int(bb.x_max), int(bb.y_max)
        if x1 <= x0 or y1 <= y0:
            continue

        # 1. Vẽ đường viền mảnh tinh tế và nền bán trong suốt nhẹ
        draw.rectangle([(x0, y0), (x1, y1)], fill=fill_rgba, outline=(*stroke_rgb, 220), width=stroke_width)

        # 2. Đặt thẻ số thứ tự nhỏ gọn hoàn toàn ở lề ngoài (ngoài khung chữ để không đè lên ký tự)
        if show_labels:
            label_text = f"#{idx}"
            tag_w = max(20, len(label_text) * 7 + 4)
            tag_h = min(14, max(12, y1 - y0))

            # Ưu tiên 1: Lề trái ngoài khung chữ
            if x0 >= tag_w + 2:
                tag_x0 = x0 - tag_w - 2
                tag_y0 = y0
            # Ưu tiên 2: Lề phải ngoài khung chữ
            elif x1 + tag_w + 2 <= img_w:
                tag_x0 = x1 + 2
                tag_y0 = y0
            # Ưu tiên 3: Đặt phía trên khung chữ
            elif y0 >= tag_h + 1:
                tag_x0 = max(0, x0)
                tag_y0 = y0 - tag_h - 1
            else:
                # Nếu sát cả 4 góc, đặt phía dưới ngoài khung chữ
                tag_x0 = max(0, x0)
                tag_y0 = min(img_h - tag_h, y1 + 1)

            tag_x1 = tag_x0 + tag_w
            tag_y1 = tag_y0 + tag_h

            # Vẽ thẻ nền bán trong suốt và số thứ tự
            draw.rectangle([(tag_x0, tag_y0), (tag_x1, tag_y1)], fill=(*stroke_rgb, 200), outline=(*stroke_rgb, 255), width=1)
            draw.text((tag_x0 + 2, tag_y0 + 1), label_text, fill=(255, 255, 255, 255), font=font)

    combined = Image.alpha_composite(base_img, overlay).convert("RGB")
    out_buffer = io.BytesIO()
    combined.save(out_buffer, format="JPEG", quality=95)
    return out_buffer.getvalue()


def export_receipt_to_excel_bytes(
    data: ReceiptData,
    metadata: Optional[ExtractionMetadata] = None,
) -> bytes:
    """Xuất thông tin hóa đơn có cấu trúc và danh sách mặt hàng ra file Excel 2 sheet (.xlsx)."""
    meta_method = metadata.method if metadata else "N/A"
    meta_engine = metadata.ocr_engine if metadata else "N/A"
    meta_adapt = metadata.adaptation_mode if metadata else "N/A"
    meta_retry = metadata.retry_count if metadata else 0
    meta_review = "CÓ" if (metadata and metadata.needs_human_review) else "KHÔNG"

    info_rows = [
        {"Trường thông tin": "1. Tên cửa hàng (seller_name)", "Giá trị": data.seller_name or ""},
        {"Trường thông tin": "2. Địa chỉ cửa hàng (seller_address)", "Giá trị": data.seller_address or ""},
        {"Trường thông tin": "3. Số điện thoại cửa hàng (seller_phone)", "Giá trị": data.seller_phone or ""},
        {"Trường thông tin": "4. Mã số thuế (tax_id)", "Giá trị": data.tax_id or ""},
        {"Trường thông tin": "5. Số hóa đơn / phiếu (invoice_number)", "Giá trị": data.invoice_number or ""},
        {"Trường thông tin": "6. Ngày giờ giao dịch (timestamp)", "Giá trị": data.timestamp or ""},
        {"Trường thông tin": "7. Tạm tính (subtotal - VND)", "Giá trị": data.subtotal if data.subtotal is not None else ""},
        {"Trường thông tin": "8. Giảm giá / Chiết khấu (discount - VND)", "Giá trị": data.discount if data.discount is not None else ""},
        {"Trường thông tin": "9. Thuế VAT (vat_amount - VND)", "Giá trị": data.vat_amount if data.vat_amount is not None else ""},
        {"Trường thông tin": "10. Tổng tiền thanh toán (total_cost - VND)", "Giá trị": data.total_cost if data.total_cost is not None else ""},
        {"Trường thông tin": "11. Hình thức thanh toán (payment_method)", "Giá trị": data.payment_method or ""},
        {"Trường thông tin": "12. Số lượng mặt hàng (item_count)", "Giá trị": len(data.items)},
        {"Trường thông tin": "--- THÔNG TIN KỸ THUẬT (METADATA) ---", "Giá trị": "---"},
        {"Trường thông tin": "Phương pháp trích xuất", "Giá trị": meta_method},
        {"Trường thông tin": "Engine OCR", "Giá trị": meta_engine},
        {"Trường thông tin": "Cấp độ thích nghi Nested Learning", "Giá trị": meta_adapt},
        {"Trường thông tin": "Số lượt tự sửa LSS (retry_count)", "Giá trị": meta_retry},
        {"Trường thông tin": "Gắn cờ cần người kiểm tra", "Giá trị": meta_review},
    ]
    df_info = pd.DataFrame(info_rows)

    item_rows = []
    for idx, it in enumerate(data.items, start=1):
        item_rows.append(
            {
                "STT": idx,
                "Tên sản phẩm / dịch vụ": it.item_name,
                "Số lượng": it.quantity,
                "Đơn giá (VND)": it.unit_price,
                "Thành tiền (VND)": it.total_price,
            }
        )
    df_items = pd.DataFrame(item_rows) if item_rows else pd.DataFrame(
        columns=["STT", "Tên sản phẩm / dịch vụ", "Số lượng", "Đơn giá (VND)", "Thành tiền (VND)"]
    )

    out_buffer = io.BytesIO()
    with pd.ExcelWriter(out_buffer, engine="openpyxl") as writer:
        df_info.to_excel(writer, sheet_name="ThongTinHoaDon", index=False)
        df_items.to_excel(writer, sheet_name="DanhSachMatHang", index=False)

    return out_buffer.getvalue()

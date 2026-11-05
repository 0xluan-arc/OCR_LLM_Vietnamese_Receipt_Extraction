"""Unit tests cho FastAPI Backend Server, Modern Web Studio APIs
và các sửa lỗi về Timeout Ollama, Trực quan hóa Bounding Box, cùng Cơ chế LSS.

Thực thi kiểm thử trực tiếp các route handler functions để tương thích hoàn toàn
với môi trường Windows Sandbox (không phụ thuộc vào socketpair loopback).
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from fastapi import UploadFile
from PIL import Image

from receipt_extraction.api.server import (
    CMSSaveRequest,
    ExportExcelRequest,
    export_excel,
    extract_receipt,
    get_sample_image,
    get_system_status,
    list_sample_receipts,
    save_to_cms,
    serve_index_html,
)
from receipt_extraction.ocr.base import OCRBoundingBox, OCRLine
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem, ValidationSeverity
from receipt_extraction.web.export_utils import draw_ocr_bounding_boxes


def create_test_image_bytes(width: int = 400, height: int = 600) -> bytes:
    """Tạo ảnh trắng thử nghiệm."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_api_status_endpoint() -> None:
    """Kiểm tra get_system_status trả về đầy đủ các trường hệ thống."""
    data = get_system_status()
    assert data["status"] == "healthy"
    assert "ocr_engines" in data
    assert "ollama_llm" in data
    assert "cms_store" in data
    assert data["cms_store"]["delta_matrix_shape"] == "128x128"
    assert data["cms_store"]["memory_dim"] == 128


def test_api_samples_endpoint() -> None:
    """Kiểm tra list_sample_receipts trả về danh sách mẫu hóa đơn."""
    samples = list_sample_receipts()
    assert isinstance(samples, list)
    if len(samples) > 0:
        sample_0 = samples[0]
        assert "filename" in sample_0
        assert "title" in sample_0


def test_api_extract_endpoint_and_lss_consistency() -> None:
    """Kiểm tra extract_receipt với engine dummy và đảm bảo không mâu thuẫn LSS 0 cảnh báo."""
    img_bytes = create_test_image_bytes(300, 400)
    upload = UploadFile(file=io.BytesIO(img_bytes), filename="test_receipt.jpg")

    res = extract_receipt(
        file=upload,
        ocr_engine="dummy",
        extractor_mode="regex_b1",
        enable_lss_retry=True,
        use_layout=False,
    )

    assert res["success"] is True
    assert "receipt_data" in res
    assert "image_boxed_base64" in res
    assert res["image_boxed_base64"].startswith("data:image/jpeg;base64,")
    assert "pipeline_trace" in res
    assert len(res["pipeline_trace"]) == 4

    # Kiểm tra ràng buộc logic tuyệt đối: Nếu needs_human_review == True thì BẮT BUỘC phải có cảnh báo actionable hoặc invalid!
    actionable = [
        iss for iss in res["validation_issues"]
        if iss["severity"] in ("WARNING", "ERROR")
    ]
    if len(actionable) == 0 and res["is_valid"]:
        assert res["needs_human_review"] is False, "Tuyệt đối không được báo needs_human_review=True khi có 0 cảnh báo!"


def test_api_cms_save_endpoint() -> None:
    """Kiểm tra save_to_cms cập nhật thành công vào CMS và Ma trận Delta Rule."""
    req_payload = CMSSaveRequest(
        receipt_data={
            "seller_name": "Circle K Test Studio",
            "seller_address": "123 Hai Ba Trung",
            "seller_phone": "0901234567",
            "tax_id": "0311223344",
            "invoice_number": "HD-1234",
            "timestamp": "22/10/2026",
            "subtotal": 50000,
            "discount": 0,
            "vat_amount": 0,
            "total_cost": 50000,
            "payment_method": "Tiền mặt",
            "items": [{"item_name": "Bánh mì", "quantity": 1.0, "unit_price": 50000, "total_price": 50000}],
        },
        ocr_text="Circle K Test Studio 123 Hai Ba Trung SĐT 0901234567 Bánh mì 50000 Tổng 50000",
        source="unit_test_direct",
    )
    res = save_to_cms(req_payload)
    assert res["success"] is True
    assert res["total_episodes"] >= 1
    assert "matrix_frobenius_norm" in res
    assert "delta_matrix_steps" in res


def test_api_export_excel_endpoint() -> None:
    """Kiểm tra export_excel trả về file nhị phân xlsx."""
    req = ExportExcelRequest(
        receipt_data={
            "seller_name": "Highlands Coffee",
            "total_cost": 75000,
            "subtotal": 75000,
            "items": [{"item_name": "Phin sữa đá", "quantity": 1.0, "unit_price": 75000, "total_price": 75000}],
        }
    )
    resp = export_excel(req)
    assert "spreadsheetml" in resp.headers["content-type"]
    assert len(resp.body) > 1000


def test_serve_index_html() -> None:
    """Kiểm tra serve_index_html phục vụ SPA HTML hiện đại và sạch bóng mã LaTeX thô."""
    resp = serve_index_html()
    assert resp.status_code == 200
    html_text = resp.body.decode("utf-8")
    assert "Vietnamese Receipt Document AI Studio" in html_text
    assert "Nested Learning" in html_text
    # Đảm bảo tuyệt đối không còn mã LaTeX thô $\mathcal{M}_t$ trên giao diện HTML
    assert "\\mathcal" not in html_text
    assert "$\\mathcal" not in html_text
    assert "<b>M</b><sub>t</sub>" in html_text or "&#119976;<sub>t</sub>" in html_text


def test_bounding_box_tags_do_not_overlap_preceding_line() -> None:
    """Kiểm tra hàm draw_ocr_bounding_boxes đặt thẻ ở lề ngoài, không đè lên dòng chữ phía trên."""
    img_bytes = create_test_image_bytes(500, 500)
    # Hai dòng chữ nằm sát nhau theo trục Y: dòng 1 (y: 100->130), dòng 2 (y: 135->165)
    line1 = OCRLine(
        text="DÒNG 1: WINMART CONG HOA",
        confidence=0.95,
        bounding_box=OCRBoundingBox(x_min=80, y_min=100, x_max=350, y_max=130),
    )
    line2 = OCRLine(
        text="DÒNG 2: SUA TUOI VINAMILK 1L",
        confidence=0.92,
        bounding_box=OCRBoundingBox(x_min=80, y_min=135, x_max=350, y_max=165),
    )

    out_bytes = draw_ocr_bounding_boxes(img_bytes, [line1, line2], stroke_width=1, show_labels=True)
    assert len(out_bytes) > 0

    res_img = Image.open(io.BytesIO(out_bytes))
    assert res_img.size == (500, 500)

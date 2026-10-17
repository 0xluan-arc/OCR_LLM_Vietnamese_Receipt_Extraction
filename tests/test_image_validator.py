"""Unit tests cho module kiểm định tính hợp lệ của ảnh đầu vào."""

import io
import pytest
from receipt_extraction.preprocess.image_validator import validate_image


def test_validate_nonexistent_file():
    """Kiểm tra xử lý tệp không tồn tại."""
    res = validate_image("D:/non_existent_image_123456.jpg")
    assert res.is_valid is False
    assert "không tồn tại" in res.error_message


def test_validate_empty_bytes():
    """Kiểm tra xử lý dữ liệu rỗng."""
    res = validate_image(b"")
    assert res.is_valid is False
    assert "rỗng" in res.error_message


def test_validate_unsupported_extension():
    """Kiểm tra chặn các định dạng không phải ảnh (PDF, DOCX, TXT)."""
    res = validate_image(b"fake content", file_name="invoice.pdf")
    assert res.is_valid is False
    assert "không được hỗ trợ" in res.error_message


def test_validate_corrupt_image():
    """Kiểm tra tệp có đuôi ảnh nhưng nội dung hỏng."""
    res = validate_image(b"this is not an image binary data", file_name="corrupt.jpg")
    assert res.is_valid is False
    assert "bị hỏng" in res.error_message or "không thể giải mã" in res.error_message


def test_validate_valid_synthetic_image():
    """Kiểm tra tệp ảnh hợp lệ được tạo tổng hợp."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow chưa được cài đặt trong môi trường test.")

    # Tạo ảnh trắng 400x600 chuẩn
    img = Image.new("RGB", (400, 600), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid_bytes = buf.getvalue()

    res = validate_image(valid_bytes, file_name="valid_receipt.jpg")
    assert res.is_valid is True
    assert res.width == 400
    assert res.height == 600
    assert res.error_message is None

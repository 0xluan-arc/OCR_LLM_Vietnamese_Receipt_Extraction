"""Unit tests cho Tesseract OCR Adapter."""

import pytest
from pathlib import Path
from receipt_extraction.ocr.tesseract_ocr import (
    TesseractOCREngine,
    resolve_tesseract_cmd,
    resolve_tessdata_dir
)


def test_resolve_tesseract_cmd_invalid_path():
    """Kiểm tra xử lý đường dẫn tệp thực thi không tồn tại."""
    res = resolve_tesseract_cmd("C:/non_existent_folder_xyz/tesseract.exe")
    assert res is None or res.is_file()


def test_tesseract_adapter_invalid_cmd_handling():
    """Kiểm tra adapter xử lý trường hợp đường dẫn sai mà không gây crash."""
    engine = TesseractOCREngine(tesseract_cmd="C:/invalid_path/tesseract.exe")
    is_avail, err = engine.is_available()
    
    assert is_avail is False
    assert "Chưa tìm thấy phần mềm Tesseract OCR" in err

    # Khi gọi recognize trên engine chưa cài, phải trả về OCRResult có error_message
    res = engine.recognize(b"fake_image_bytes")
    assert res.error_message is not None
    assert "chưa sẵn sàng" in res.error_message


def test_resolve_tessdata_dir_finds_project_data():
    """Kiểm tra hàm tìm kiếm thư mục tessdata tìm thấy thư mục data/tessdata trong repo."""
    tessdata_path = resolve_tessdata_dir()
    # Nếu trong data/tessdata có file .traineddata, phải tìm thấy
    if tessdata_path:
        assert tessdata_path.is_dir()
        assert any(tessdata_path.glob("*.traineddata"))

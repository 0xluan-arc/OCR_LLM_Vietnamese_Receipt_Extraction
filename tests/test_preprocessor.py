"""Unit tests cho module tiền xử lý ảnh OpenCV."""

import numpy as np
import pytest
from receipt_extraction.preprocess.preprocessor import (
    preprocess_receipt_image,
    PreprocessOptions,
    estimate_skew_angle
)


def test_preprocess_options_defaults():
    """Kiểm tra giá trị mặc định của PreprocessOptions."""
    opts = PreprocessOptions()
    assert opts.enhance_contrast is True
    assert opts.to_grayscale is False
    assert opts.adaptive_threshold is False
    assert opts.deskew is False


def test_preprocess_to_grayscale():
    """Kiểm tra chuyển đổi ảnh màu sang ảnh xám."""
    # Tạo ảnh RGB 3 kênh 400x400
    dummy_color = np.full((400, 400, 3), 200, dtype=np.uint8)
    opts = PreprocessOptions(to_grayscale=True, enhance_contrast=False)
    
    proc, meta, err = preprocess_receipt_image(dummy_color, opts)
    assert err is None
    assert proc is not None
    assert len(proc.shape) == 2  # Ảnh 1 kênh (grayscale)
    assert "grayscale" in meta["applied_transforms"]


def test_preprocess_upscale_low_resolution():
    """Kiểm tra phóng to ảnh độ phân giải thấp."""
    # Tạo ảnh 200x300 (nhỏ hơn min_dimension=800)
    small_img = np.full((300, 200, 3), 255, dtype=np.uint8)
    opts = PreprocessOptions(min_dimension=800, enhance_contrast=False)
    
    proc, meta, err = preprocess_receipt_image(small_img, opts)
    assert err is None
    assert proc is not None
    assert min(proc.shape[:2]) >= 800
    assert any("upscale" in t for t in meta["applied_transforms"])


def test_preprocess_adaptive_threshold():
    """Kiểm tra phân ngưỡng nhị phân thích ứng."""
    dummy_color = np.full((500, 500, 3), 180, dtype=np.uint8)
    opts = PreprocessOptions(adaptive_threshold=True, enhance_contrast=False)
    
    proc, meta, err = preprocess_receipt_image(dummy_color, opts)
    assert err is None
    assert proc is not None
    assert len(proc.shape) == 2
    # Chỉ chứa các giá trị nhị phân 0 hoặc 255
    unique_vals = set(np.unique(proc))
    assert unique_vals.issubset({0, 255})
    assert "adaptive_threshold" in meta["applied_transforms"]


def test_estimate_skew_angle_blank_image():
    """Kiểm tra ước lượng góc nghiêng trên ảnh trắng không có chữ."""
    blank_gray = np.full((400, 400), 255, dtype=np.uint8)
    angle = estimate_skew_angle(blank_gray)
    assert angle == 0.0

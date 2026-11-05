"""Unit tests cho module tiền xử lý ảnh OpenCV."""

import numpy as np
import pytest
from receipt_extraction.preprocess.preprocessor import (
    preprocess_receipt_image,
    PreprocessOptions,
    estimate_skew_angle,
    illumination_compensation,
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


def test_illumination_compensation_removes_shadow_gradient():
    """Kiểm tra hàm bù sáng nền Gaussian (cv2.divide) làm phẳng vùng bóng râm trên hóa đơn nhiệt."""
    import cv2

    h, w = 400, 300
    # Tạo trường ánh sáng không đều (bóng râm bên phải tối hơn bên trái)
    gradient = np.tile(np.linspace(230, 110, w, dtype=np.uint8), (h, 1))
    # Vẽ các dòng chữ đen trên nền bóng râm
    cv2.putText(gradient, "HIGHLANDS COFFEE", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 25, 2)
    cv2.putText(gradient, "TOTAL: 125,000 VND", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 25, 2)

    compensated = illumination_compensation(gradient, kernel_size=51)
    assert compensated.shape == gradient.shape
    assert compensated.dtype == np.uint8
    # Nền giấy sau khi bù sáng phải sáng đồng đều hơn ảnh gốc
    assert float(np.percentile(compensated, 85)) >= 240.0


def test_preprocess_preserves_faint_thermal_header_without_bleaching():
    """Kiểm tra preprocess_receipt_image dùng CLAHE nhẹ trên kênh L hệ màu LAB, không tẩy trắng nét chữ in nhiệt mờ ở nửa trên."""
    import cv2

    # Tạo ảnh màu BGR giả lập hóa đơn in nhiệt bị phai mờ phần tiêu đề (mực xám nhạt 195 trên nền 235)
    img = np.full((600, 400, 3), 235, dtype=np.uint8)
    cv2.putText(img, "WINMART CONG HOA", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (195, 195, 195), 2)
    cv2.putText(img, "SDT: 028 3811 2233", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (195, 195, 195), 2)
    cv2.putText(img, "TONG CONG: 304,326", (40, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    opts = PreprocessOptions(
        crop_receipt=False,
        remove_shadows=False,
        illumination_compensation=False,
        deskew=True,
        enhance_contrast=True,
        clahe_clip_limit=2.0,
    )
    proc, meta, err = preprocess_receipt_image(img, opts)
    assert err is None
    assert proc is not None
    assert len(proc.shape) == 3  # Bảo toàn ảnh màu BGR 3 kênh
    assert "adaptive_threshold" not in meta["applied_transforms"]
    assert "clahe_contrast" in meta["applied_transforms"]

    # Kiểm tra vùng tiêu đề nửa trên (y: 30..120) vẫn còn nguyên các pixel nét chữ tối hơn nền (không bị tẩy trắng 255)
    header_roi = cv2.cvtColor(proc[30:120, 30:360], cv2.COLOR_BGR2GRAY)
    assert float(np.min(header_roi)) < float(np.max(header_roi)) - 15.0


def test_estimate_skew_angle_hough_lines_tilted_receipt():
    """Kiểm tra estimate_skew_angle bằng HoughLinesP phát hiện đúng góc nghiêng của các dòng chữ."""
    import cv2

    img = np.full((500, 500), 255, dtype=np.uint8)
    for y in range(100, 400, 45):
        # Vẽ các dòng chữ nghiêng khoảng +6 độ (dy = 31px trên dx = 300px)
        cv2.line(img, (80, y), (380, y + 31), 0, 5)

    angle = estimate_skew_angle(img)
    assert 4.0 <= angle <= 8.0


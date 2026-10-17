"""Bộ kiểm định tính hợp lệ của tệp ảnh hóa đơn đầu vào."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Union, Tuple
import io

from receipt_extraction.config import config


@dataclass
class ImageValidationResult:
    """Kết quả kiểm định ảnh."""
    is_valid: bool
    error_message: Optional[str] = None
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    is_blurry: bool = False
    blur_score: float = 0.0


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def validate_image(
    image_source: Union[str, Path, bytes, io.BytesIO],
    file_name: Optional[str] = None
) -> ImageValidationResult:
    """Kiểm tra toàn diện tính hợp lệ của ảnh đầu vào.
    
    Kiểm tra:
    1. Định dạng file có thuộc danh sách hỗ trợ không.
    2. Kích thước file không vượt ngưỡng cấu hình.
    3. Ảnh có giải mã (decode) được không hay bị corrupt.
    4. Kích thước chiều rộng và chiều cao tối thiểu.
    5. Đánh giá sơ bộ độ mờ (blur score) nếu có OpenCV.
    """
    file_bytes: bytes
    name_to_check: str = file_name or ""

    # 1. Trích xuất bytes và tên file
    if isinstance(image_source, (str, Path)):
        p = Path(image_source)
        if not p.exists():
            return ImageValidationResult(is_valid=False, error_message=f"Tệp không tồn tại: {p}")
        name_to_check = name_to_check or p.name
        file_bytes = p.read_bytes()
    elif isinstance(image_source, bytes):
        file_bytes = image_source
    elif isinstance(image_source, io.BytesIO):
        file_bytes = image_source.getvalue()
    else:
        return ImageValidationResult(is_valid=False, error_message="Định dạng nguồn ảnh không được hỗ trợ.")

    file_size = len(file_bytes)
    if file_size == 0:
        return ImageValidationResult(is_valid=False, error_message="Tệp ảnh rỗng (0 bytes).")

    # 2. Kiểm tra phần mở rộng
    if name_to_check:
        ext = Path(name_to_check).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return ImageValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error_message=f"Định dạng tệp '{ext}' không được hỗ trợ. Các định dạng cho phép: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    # 3. Kiểm tra dung lượng tối đa
    max_bytes = int(config.image.max_size_mb * 1024 * 1024)
    if file_size > max_bytes:
        return ImageValidationResult(
            is_valid=False,
            file_size_bytes=file_size,
            error_message=f"Dung lượng ảnh ({file_size / (1024*1024):.2f} MB) vượt quá giới hạn cho phép ({config.image.max_size_mb} MB)."
        )

    # 4. Kiểm tra khả năng decode ảnh (dùng Pillow hoặc OpenCV)
    width = 0
    height = 0
    try:
        from PIL import Image
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()  # Kiểm tra tính toàn vẹn
        with Image.open(io.BytesIO(file_bytes)) as img:
            width, height = img.size
    except Exception as e:
        return ImageValidationResult(
            is_valid=False,
            file_size_bytes=file_size,
            error_message=f"Tệp ảnh bị hỏng hoặc không thể giải mã: {str(e)}"
        )

    # 5. Kiểm tra kích thước tối thiểu
    if width < config.image.min_width or height < config.image.min_height:
        return ImageValidationResult(
            is_valid=False,
            width=width,
            height=height,
            file_size_bytes=file_size,
            error_message=f"Kích thước ảnh quá nhỏ ({width}x{height}px). Yêu cầu tối thiểu: {config.image.min_width}x{config.image.min_height}px."
        )

    # 6. Đánh giá độ mờ (Blur detection via Laplacian variance)
    blur_score = 0.0
    is_blurry = False
    try:
        import numpy as np
        import cv2
        nparr = np.frombuffer(file_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if cv_img is not None:
            blur_score = float(cv2.Laplacian(cv_img, cv2.CV_64F).var())
            if blur_score < config.image.blur_threshold:
                is_blurry = True
    except Exception:
        # Nếu chưa cài OpenCV hoặc gặp lỗi tính toán, không chặn mà coi như hợp lệ
        pass

    return ImageValidationResult(
        is_valid=True,
        width=width,
        height=height,
        file_size_bytes=file_size,
        is_blurry=is_blurry,
        blur_score=blur_score
    )

"""Module tiền xử lý ảnh hóa đơn trước khi đưa vào OCR."""

from typing import Union, Optional, Tuple
from pathlib import Path
import io

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def preprocess_receipt_image(
    image_input: Union[str, Path, bytes, "np.ndarray"],
    enhance_contrast: bool = True,
    max_dimension: Optional[int] = 2000
) -> Tuple[Optional["np.ndarray"], Optional[str]]:
    """Tiền xử lý ảnh phục vụ OCR: khử nhiễu, tăng độ tương phản và chuẩn hóa kích thước.
    
    Returns:
        (processed_image_ndarray, error_message)
    """
    if not CV2_AVAILABLE:
        return None, "OpenCV chưa được cài đặt trong môi trường hiện tại."

    img: Optional[np.ndarray] = None

    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            return None, f"Không thể đọc ảnh từ đường dẫn: {image_input}"
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Không thể giải mã dữ liệu bytes thành ảnh OpenCV."
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        return None, "Kiểu dữ liệu đầu vào không được hỗ trợ."

    h, w = img.shape[:2]

    # 1. Thu phóng nếu ảnh quá lớn (> max_dimension) nhằm tối ưu tốc độ OCR
    if max_dimension and (max(h, w) > max_dimension):
        scale = max_dimension / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Tăng cường độ tương phản cục bộ bằng CLAHE trên kênh độ sáng (L channel)
    if enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return img, None

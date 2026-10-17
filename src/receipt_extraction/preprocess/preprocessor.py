"""Module tiền xử lý ảnh hóa đơn trước khi đưa vào OCR."""

from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict, Any
from pathlib import Path
import io

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class PreprocessOptions:
    """Các tùy chọn tiền xử lý có thể bật/tắt độc lập."""
    to_grayscale: bool = False
    enhance_contrast: bool = True
    adaptive_threshold: bool = False
    deskew: bool = False
    min_dimension: Optional[int] = 800
    max_dimension: Optional[int] = 2400


def estimate_skew_angle(gray_img: "np.ndarray") -> float:
    """Ước lượng góc nghiêng của tài liệu dựa trên contour vùng chữ.
    
    Chỉ trả về góc nghiêng nếu phát hiện góc hợp lý (0.5 đến 45 độ).
    """
    if not CV2_AVAILABLE:
        return 0.0

    # Phân ngưỡng nhị phân nghịch đảo để lấy chữ
    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Tìm các tọa độ điểm trắng (nét chữ)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return 0.0

    # Tính hộp bao diện tích nhỏ nhất
    angle = cv2.minAreaRect(coords)[-1]
    
    # Chuẩn hóa góc quay OpenCV
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Chỉ áp dụng nếu góc nghiêng đáng kể nhưng không quá cực đoan (0.5 < |angle| < 30)
    if 0.5 <= abs(angle) <= 30.0:
        return float(angle)
    return 0.0


def rotate_image(img: "np.ndarray", angle: float) -> "np.ndarray":
    """Xoay ảnh theo góc ước lượng mà không làm mất góc cạnh."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Tính lại kích thước bounding box mới
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]
    
    return cv2.warpAffine(img, m, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_receipt_image(
    image_input: Union[str, Path, bytes, "np.ndarray"],
    options: Optional[PreprocessOptions] = None
) -> Tuple[Optional["np.ndarray"], Dict[str, Any], Optional[str]]:
    """Tiền xử lý ảnh hóa đơn với các bước được kiểm soát rõ ràng.
    
    Returns:
        (processed_image_ndarray, metadata_dict, error_message)
    """
    if not CV2_AVAILABLE:
        return None, {}, "OpenCV chưa được cài đặt trong môi trường hiện tại."

    opts = options or PreprocessOptions()
    meta: Dict[str, Any] = {"applied_transforms": []}

    img: Optional[np.ndarray] = None

    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            return None, {}, f"Không thể đọc ảnh từ đường dẫn: {image_input}"
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, {}, "Không thể giải mã dữ liệu bytes thành ảnh OpenCV."
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        return None, {}, "Kiểu dữ liệu đầu vào không được hỗ trợ."

    h, w = img.shape[:2]
    meta["original_size"] = (w, h)

    # 1. Phóng to ảnh nếu độ phân giải quá thấp (< min_dimension)
    if opts.min_dimension and (min(w, h) < opts.min_dimension):
        scale = opts.min_dimension / float(min(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        meta["applied_transforms"].append(f"upscale_{scale:.2f}x")
        h, w = img.shape[:2]

    # 2. Thu nhỏ nếu ảnh quá lớn (> max_dimension) để tối ưu thời gian OCR
    elif opts.max_dimension and (max(w, h) > opts.max_dimension):
        scale = opts.max_dimension / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        meta["applied_transforms"].append(f"downscale_{scale:.2f}x")
        h, w = img.shape[:2]

    # 3. Chỉnh nghiêng (Deskew) nếu được bật
    if opts.deskew:
        gray_for_skew = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        angle = estimate_skew_angle(gray_for_skew)
        if abs(angle) > 0.5:
            img = rotate_image(img, angle)
            meta["applied_transforms"].append(f"deskew_{angle:.2f}deg")
            meta["skew_angle"] = angle

    # 4. Tăng cường độ tương phản (CLAHE)
    if opts.enhance_contrast:
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        meta["applied_transforms"].append("clahe_contrast")

    # 5. Chuyển Grayscale
    if opts.to_grayscale or opts.adaptive_threshold:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            meta["applied_transforms"].append("grayscale")

    # 6. Phân ngưỡng nhị phân thích ứng (Adaptive Thresholding)
    if opts.adaptive_threshold:
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        meta["applied_transforms"].append("adaptive_threshold")

    meta["final_size"] = (img.shape[1], img.shape[0])
    return img, meta, None

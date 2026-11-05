"""Module tiền xử lý ảnh hóa đơn trước khi đưa vào OCR (Tối ưu hóa cho giấy in nhiệt - MT2)."""

from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict, Any, List
from pathlib import Path

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
    crop_receipt: bool = False
    remove_shadows: bool = False
    illumination_compensation: bool = False
    clahe_clip_limit: float = 2.0
    min_dimension: Optional[int] = 800
    max_dimension: Optional[int] = 1800


def detect_and_crop_receipt(img: "np.ndarray") -> Tuple["np.ndarray", bool, Tuple[int, int, int, int]]:
    """Tự động phát hiện và cắt bỏ vùng mặt bàn/nền xung quanh hóa đơn.

    Returns:
        (cropped_image, did_crop_bool, (x, y, w, h))
    """
    if not CV2_AVAILABLE:
        return img, False, (0, 0, img.shape[1], img.shape[0])

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    thresh_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.threshold(blurred, max(thresh_val * 0.72, 50), 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        best_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_cnt)
        if area > 0.12 * w * h:
            rx, ry, rw, rh = cv2.boundingRect(best_cnt)
            # Nếu khung bao đã chiếm gần trọn ảnh (>92% cả 2 chiều), không cần cắt
            if rw > 0.92 * w and rh > 0.92 * h:
                return img, False, (0, 0, w, h)
            pad = 24
            x0, y0 = max(0, rx - pad), max(0, ry - pad)
            x1, y1 = min(w, rx + rw + pad), min(h, ry + rh + pad)
            cropped = img[y0:y1, x0:x1]
            return cropped, True, (x0, y0, x1 - x0, y1 - y0)

    return img, False, (0, 0, w, h)


def illumination_compensation(
    img_input: "np.ndarray",
    kernel_size: int = 51,
    clahe_clip_limit: float = 2.0,
) -> "np.ndarray":
    """Bù sáng nền mềm mại (Non-Destructive Illumination Compensation) kết hợp CLAHE
    nhẹ trên kênh L của không gian màu LAB (hoặc Grayscale).

    TUYỆT ĐỐI KHÔNG sử dụng cv2.adaptiveThreshold, cv2.threshold hoặc mặt nạ tẩy trắng (bleaching)
    nhằm bảo toàn 100% các nét chữ in nhiệt mờ ở phần tiêu đề (Tên cửa hàng, Địa chỉ, SĐT)
    khi đưa vào mạng học sâu PaddleOCR (DBNet) + VietOCR.
    """
    if not CV2_AVAILABLE:
        return img_input

    is_color = len(img_input.shape) == 3
    if is_color:
        lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        gray = l_chan
    else:
        gray = img_input

    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return img_input

    # Nếu ảnh hoàn toàn đồng nhất (ảnh kiểm thử trắng/xám), trả về trực tiếp
    if float(np.std(gray)) < 1.5:
        return img_input

    ksize = max(kernel_size, 51)
    if ksize % 2 == 0:
        ksize += 1

    # Bước 1: Ước lượng trường sáng nền vĩ mô B(x,y) qua phép giãn nở + GaussianBlur kernel lớn (51x51)
    if min(h, w) > 500:
        small_w = max(64, w // 4)
        small_h = max(64, h // 4)
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_dil = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        small_bg = cv2.GaussianBlur(small_dil, (ksize, ksize), 0)
        bg = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        bg_seed = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        bg = cv2.GaussianBlur(bg_seed, (ksize, ksize), 0)

    bg = np.clip(bg, 1, 255)

    # Bước 2: Phép chia bù sáng nền mềm mại cv2.divide(gray, bg, scale=255)
    div = cv2.divide(gray, bg, scale=255)

    # Bước 3: Tăng cường tương phản nhẹ bằng CLAHE (clipLimit=2.0, tileGridSize=(8,8))
    # Không tẩy trắng hoặc cắt ngưỡng gắt, bảo toàn 100% nét chữ mờ
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit or 2.0), tileGridSize=(8, 8))
    l_enhanced = clahe.apply(div)

    if is_color:
        merged_lab = cv2.merge((l_enhanced, a_chan, b_chan))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return l_enhanced


def remove_receipt_shadows(gray_img: "np.ndarray") -> "np.ndarray":
    """Khử bóng đổ không đồng đều trên giấy in nhiệt kết hợp Illumination Compensation."""
    if not CV2_AVAILABLE:
        return gray_img
    return illumination_compensation(gray_img, kernel_size=51)


def estimate_skew_angle(gray_img: "np.ndarray") -> float:
    """Ước lượng góc nghiêng của tài liệu bằng Hough Line Transform (cv2.HoughLinesP)
    trên cấu trúc dòng văn bản ngang, kết hợp fallback về cv2.minAreaRect.

    Chỉ trả về góc nghiêng nếu phát hiện góc hợp lý (0.5 đến 30.0 độ).
    """
    if not CV2_AVAILABLE:
        return 0.0

    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    h, w = gray_img.shape[:2]
    if h < 40 or w < 40:
        return 0.0

    # Phân ngưỡng nhị phân nghịch đảo để lấy nét chữ
    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    nonzero_count = int(np.count_nonzero(thresh))
    if nonzero_count < 100 or nonzero_count > int(0.85 * h * w):
        return 0.0

    # 1. Phương pháp chính: Giãn nở ngang nối chữ thành dòng + Probabilistic Hough Line Transform
    kernel_w = max(15, w // 28)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    connected_lines = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horiz_kernel)
    edges = cv2.Canny(connected_lines, 50, 150, apertureSize=3)

    min_line_length = max(30, w // 8)
    max_line_gap = max(10, w // 25)
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=25,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    candidate_angles: List[float] = []
    weights: List[float] = []
    if lines is not None and len(lines) > 0:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 15.0:
                continue
            deg = float(np.degrees(np.arctan2(dy, dx)))
            # Chỉ xét các đoạn thẳng gần phương ngang (-30 độ đến +30 độ)
            if -30.0 <= deg <= 30.0:
                length = float(np.hypot(dx, dy))
                candidate_angles.append(deg)
                weights.append(length)

    if len(candidate_angles) >= 3:
        # Lấy trung vị có trọng số theo độ dài dòng chữ để kháng nhiễu biên/logo
        order = np.argsort(candidate_angles)
        sorted_angles = np.array(candidate_angles, dtype=np.float64)[order]
        sorted_weights = np.array(weights, dtype=np.float64)[order]
        cum_weights = np.cumsum(sorted_weights)
        median_idx = int(np.searchsorted(cum_weights, 0.5 * cum_weights[-1]))
        median_idx = min(max(0, median_idx), len(sorted_angles) - 1)
        est_angle = float(sorted_angles[median_idx])
        if 0.5 <= abs(est_angle) <= 30.0:
            return round(est_angle, 2)
        return 0.0

    # 2. Fallback: Sử dụng hộp bao diện tích nhỏ nhất (minAreaRect) trên các contour dòng chữ
    contours, _ = cv2.findContours(connected_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rect_angles: List[float] = []
    for cnt in contours:
        cw, ch = cv2.boundingRect(cnt)[2:]
        if cw < min_line_length or ch < 6 or cw < 2 * ch:
            continue
        rect_ang = cv2.minAreaRect(cnt)[-1]
        if rect_ang < -45:
            rect_ang = -(90 + rect_ang)
        elif rect_ang > 45:
            rect_ang = 90 - rect_ang
        else:
            rect_ang = -rect_ang
        if -30.0 <= rect_ang <= 30.0:
            rect_angles.append(-float(rect_ang))

    if rect_angles:
        med = float(np.median(rect_angles))
        if 0.5 <= abs(med) <= 30.0:
            return round(med, 2)

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

    # 1. Tự động phát hiện và cắt bỏ vùng mặt bàn/nền xung quanh hóa đơn
    if opts.crop_receipt:
        img, did_crop, crop_box = detect_and_crop_receipt(img)
        if did_crop:
            meta["applied_transforms"].append(f"crop_receipt_{crop_box[2]}x{crop_box[3]}")
            meta["crop_box"] = crop_box
            h, w = img.shape[:2]

    # 2. Phóng to ảnh nếu độ phân giải quá thấp (< min_dimension)
    if opts.min_dimension and (min(w, h) < opts.min_dimension):
        scale = opts.min_dimension / float(min(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        meta["applied_transforms"].append(f"upscale_{scale:.2f}x")
        h, w = img.shape[:2]

    # 3. Thu nhỏ nếu ảnh quá lớn (> max_dimension) để tối ưu thời gian OCR
    elif opts.max_dimension and (max(w, h) > opts.max_dimension):
        scale = opts.max_dimension / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        meta["applied_transforms"].append(f"downscale_{scale:.2f}x")
        h, w = img.shape[:2]

    # 4. Chỉnh nghiêng (Hough Line Deskew) nếu được bật
    if opts.deskew:
        gray_for_skew = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        angle = estimate_skew_angle(gray_for_skew)
        if abs(angle) >= 0.5:
            img = rotate_image(img, angle)
            meta["applied_transforms"].append(f"deskew_{angle:.2f}deg")
            meta["skew_angle"] = angle

    # 5. Bù sáng nền mềm mại (nếu bật) mà không tẩy trắng hay phân ngưỡng nhị phân
    did_illum = False
    if opts.remove_shadows or opts.illumination_compensation:
        img = illumination_compensation(img, kernel_size=51, clahe_clip_limit=float(opts.clahe_clip_limit or 2.0))
        meta["applied_transforms"].append("illumination_compensation")
        if opts.enhance_contrast:
            meta["applied_transforms"].append("clahe_contrast")
        did_illum = True

    # 6. Tăng tương phản nhẹ bằng CLAHE trên kênh L của không gian màu LAB (bảo toàn 100% nét chữ mờ)
    if opts.enhance_contrast and not did_illum:
        clip_lim = float(opts.clahe_clip_limit or 2.0)
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(8, 8))
            img = clahe.apply(img)
        meta["applied_transforms"].append("clahe_contrast")

    # 7. Chuyển Grayscale (chỉ khi người dùng bật tường minh hoặc dùng adaptive_threshold)
    if opts.to_grayscale or opts.adaptive_threshold:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if "grayscale" not in meta["applied_transforms"]:
            meta["applied_transforms"].append("grayscale")

    # 8. Phân ngưỡng nhị phân (mặc định TẮT hoàn toàn đối với PaddleVietOCREngine)
    if opts.adaptive_threshold:
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        meta["applied_transforms"].append("adaptive_threshold")

    meta["final_size"] = (img.shape[1], img.shape[0])
    return img, meta, None

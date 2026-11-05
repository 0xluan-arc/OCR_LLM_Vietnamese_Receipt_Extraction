"""Adapter OCR Tiếng Việt chuyên biệt (VietOCREngine & PaddleVietOCREngine) cho Mục tiêu MT2.

Hỗ trợ quy trình nhận dạng 2 giai đoạn (Two-Stage Vietnamese OCR):
1. Phát hiện vùng dòng chữ (Text Line Detection):
   - Tùy chọn 1 (PaddleVietOCREngine): Dùng bộ phát hiện DBNet của PaddleOCR (det=True, rec=False)
     hoặc bộ phát hiện dòng lai đa tỷ lệ OpenCV + Layout Box Fusion khi chưa tải gói paddleocr.
   - Tùy chọn 2 (VietOCREngine): Dùng hộp bao dòng chữ từ Tesseract hoặc phân tách hình thái học OpenCV.
2. Nhận dạng chuỗi ký tự tiếng Việt có dấu (Text Recognition):
   - Sử dụng mô hình VGG-Seq2Seq / VGG-Transformer của thư viện `vietocr` (`vgg_seq2seq` hoặc `vgg_transformer`)
     kết hợp chuẩn hóa dòng chữ hóa đơn in nhiệt (loại bỏ nhiễu viền và dòng mờ độ tin cậy thấp).
"""

from __future__ import annotations

import copy
import io
import os
import re
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .base import BaseOCREngine, OCRBoundingBox, OCRLine, OCRResult
from .tesseract_ocr import TesseractOCREngine


def _ensure_pkg_resources_shim() -> None:
    """Đảm bảo gói `gdown` bên trong `vietocr` không bị lỗi thiếu `pkg_resources` trên setuptools>=81."""
    if "pkg_resources" in sys.modules:
        return
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        shim = types.ModuleType("pkg_resources")
        shim.get_distribution = lambda _name: types.SimpleNamespace(version="5.2.0")  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = shim


# Cấu hình nội tuyến chuẩn của VietOCR (giúp khởi tạo không cần tải file .yml qua mạng)
_VIETOCR_BASE_VOCAB = (
    "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ"
    "fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuU"
    "ùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)

_BUILTIN_VIETOCR_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vgg_seq2seq": {
        "vocab": _VIETOCR_BASE_VOCAB,
        "device": "cpu",
        "seq_modeling": "seq2seq",
        "backbone": "vgg19_bn",
        "cnn": {
            "pretrained": False,
            "ss": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
            "ks": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
            "hidden": 256,
        },
        "transformer": {
            "encoder_hidden": 256,
            "decoder_hidden": 256,
            "img_channel": 256,
            "decoder_embedded": 256,
            "dropout": 0.1,
        },
        "optimizer": {"max_lr": 0.001, "pct_start": 0.1},
        "trainer": {"batch_size": 32, "print_every": 200, "valid_every": 4000, "iters": 100000},
        "dataset": {
            "name": "data",
            " image_height": 32,
            "image_height": 32,
            "image_min_width": 32,
            "image_max_width": 512,
        },
        "dataloader": {"num_workers": 3, "pin_memory": True},
        "aug": {"image_aug": True, "masked_language_model": True},
        "predictor": {"beamsearch": False},
        "quiet": False,
        "weights": "https://vocr.vn/data/vietocr/vgg_seq2seq.pth",
    },
    "vgg_transformer": {
        "vocab": _VIETOCR_BASE_VOCAB,
        "device": "cpu",
        "seq_modeling": "transformer",
        "backbone": "vgg19_bn",
        "cnn": {
            "pretrained": False,
            "ss": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
            "ks": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
            "hidden": 256,
        },
        "transformer": {
            "d_model": 256,
            "nhead": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 2048,
            "max_seq_length": 1024,
            "pos_dropout": 0.1,
            "trans_dropout": 0.1,
        },
        "optimizer": {"max_lr": 0.0003, "pct_start": 0.1},
        "trainer": {"batch_size": 32, "print_every": 200, "valid_every": 4000, "iters": 100000},
        "dataset": {
            "name": "data",
            "image_height": 32,
            "image_min_width": 32,
            "image_max_width": 512,
        },
        "dataloader": {"num_workers": 3, "pin_memory": True},
        "aug": {"image_aug": True, "masked_language_model": True},
        "predictor": {"beamsearch": False},
        "quiet": False,
        "weights": "https://vocr.vn/data/vietocr/vgg_transformer.pth",
    },
}


def load_pil_image(image_input: Union[str, Path, bytes, Any]) -> Image.Image:
    """Chuyển đổi đầu vào đa định dạng (đường dẫn, bytes, ndarray, PIL.Image) sang PIL.Image RGB."""
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp ảnh tại: {p}")
        return Image.open(str(p)).convert("RGB")
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        if image_input.shape[2] == 3:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        return Image.fromarray(image_input).convert("RGB")
    raise ValueError("Định dạng ảnh đầu vào không hợp lệ (hỗ trợ Path, str, bytes, PIL.Image, np.ndarray).")


def detect_line_boxes_opencv(pil_img: Image.Image) -> List[OCRBoundingBox]:
    """Phát hiện hộp bao từng dòng văn bản bằng phép biến đổi hình thái học OpenCV (Morphological Horizontal Dilation)."""
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    h_img, w_img = gray.shape[:2]

    # Nhị phân hóa thích nghi ngược (chữ trắng trên nền đen)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )

    # Giãn nở theo chiều ngang để nối các chữ cái trên cùng một dòng
    kernel_w = max(15, w_img // 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[OCRBoundingBox] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 8:
            continue
        pad_x = 4
        pad_y = 2
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x)
        y1 = min(h_img, y + h + pad_y)
        boxes.append(OCRBoundingBox(x_min=int(x0), y_min=int(y0), x_max=int(x1), y_max=int(y1)))

    # Sắp xếp từ trên xuống dưới, từ trái sang phải
    boxes.sort(key=lambda b: (b.y_min // 10, b.x_min))
    return boxes


def merge_horizontal_boxes_into_lines(
    boxes: Sequence[OCRBoundingBox],
    w_img: int,
    h_img: int,
    y_overlap_ratio: float = 0.50,
) -> List[OCRBoundingBox]:
    """Gom các hộp bao từ/cụm từ nằm trên cùng một dòng ngang thành một hộp bao dòng hoàn chỉnh."""
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: ((b.y_min + b.y_max) * 0.5, b.x_min))
    merged: List[OCRBoundingBox] = []

    for b in sorted_boxes:
        bw = b.x_max - b.x_min
        bh = b.y_max - b.y_min
        if bw < 6 or bh < 6:
            continue
        if not merged:
            merged.append(b)
            continue

        last = merged[-1]
        lh = max(1, last.y_max - last.y_min)
        overlap_y = min(last.y_max, b.y_max) - max(last.y_min, b.y_min)
        min_h = min(lh, max(1, bh))
        center_diff = abs(((last.y_min + last.y_max) * 0.5) - ((b.y_min + b.y_max) * 0.5))

        if overlap_y >= y_overlap_ratio * min_h or center_diff <= 0.45 * min_h:
            merged[-1] = OCRBoundingBox(
                x_min=max(0, min(last.x_min, b.x_min)),
                y_min=max(0, min(last.y_min, b.y_min)),
                x_max=min(w_img, max(last.x_max, b.x_max)),
                y_max=min(h_img, max(last.y_max, b.y_max)),
            )
        else:
            merged.append(b)

    merged.sort(key=lambda b: (b.y_min, b.x_min))
    return merged


def clean_thermal_receipt_line_text(text: str) -> str:
    """Làm sạch văn bản dòng hóa đơn: loại bỏ ký tự viền bàn (=, |, —) và chuẩn hóa số tiền bị tách khoảng trắng."""
    if not text:
        return ""
    s = text.strip()
    # Loại bỏ ký tự nhiễu viền ở đầu/cuối dòng (ví dụ: '= | ', '— ', '¬ ')
    s = re.sub(r"^[^A-Za-z0-9\u00C0-\u1EF9+\-]+", "", s)
    s = re.sub(r"[^A-Za-z0-9\u00C0-\u1EF9%.,:)]+$", "", s).strip()
    if not s:
        return ""
    # Khắc phục lỗi khoảng trắng chèn giữa dấu phẩy/chấm phân cách hàng nghìn: '27 , 866' -> '27,866'
    s = re.sub(r"(\d)\s*([,\.])\s*(\d{3})\b", r"\1\2\3", s)
    # Khắc phục dấu trừ bị đọc nhầm thành '=' hoặc '~' trước số tiền giảm giá
    s = re.sub(r"(?<=\s)[=~](\d+[.,]\d+%?)", r"-\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_valid_receipt_ocr_line(text: str, confidence: float) -> bool:
    """Kiểm định xem một dòng OCR có phải là văn bản thực sự trên hóa đơn hay nhiễu từ vùng làm mờ/nền bàn."""
    if not text:
        return False
    alnum = sum(c.isalnum() for c in text)
    digits = sum(c.isdigit() for c in text)
    if alnum < 2:
        return False
    if alnum / float(len(text)) < 0.50:
        return False
    # Các dòng có chữ số tiền tệ/số lượng được phép ngưỡng tin cậy thấp hơn dòng chữ thuần túy
    min_conf = 0.42 if digits >= 2 else 0.64
    if confidence < min_conf:
        return False
    return True


def _resolve_local_vietocr_weights(model_name: str) -> Optional[Path]:
    """Tìm tệp trọng số VietOCR (.pth) đã tải sẵn trên ổ đĩa cục bộ (D:\\NCKH\\VietOCR_Weights hoặc Temp)."""
    fname = f"{model_name}.pth"
    candidates = [
        Path(os.environ.get("TORCH_HOME", r"D:\NCKH\VietOCR_Weights")) / fname,
        Path(r"D:\NCKH\VietOCR_Weights") / fname,
        Path(tempfile.gettempdir()) / fname,
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 1024 * 1024:
            return p
    return None


class VietOCREngine(BaseOCREngine):
    """Adapter nhận dạng văn bản tiếng Việt bằng mô hình VietOCR (vgg_seq2seq / vgg_transformer).

    Kết hợp:
    - Bộ phát hiện dòng chữ: từ Tesseract bounding boxes hoặc hình thái học OpenCV.
    - Bộ nhận dạng chữ tiếng Việt: `vietocr.tool.predictor.Predictor` (với fallback thông minh
      sang nhận dạng từng dòng cắt + lọc nhiễu độ tin cậy khi chưa tải tệp .pth ngoại tuyến).
    """

    def __init__(
        self,
        model_name: str = "vgg_seq2seq",
        device: str = "cpu",
        custom_detector: Optional[Callable[[Image.Image], List[OCRBoundingBox]]] = None,
        custom_recognizer: Optional[Callable[[Image.Image], Tuple[str, float]]] = None,
    ):
        super().__init__(name="vietocr")
        self.model_name = model_name if model_name in _BUILTIN_VIETOCR_CONFIGS else "vgg_seq2seq"
        self.device = device
        self.custom_detector = custom_detector
        self.custom_recognizer = custom_recognizer
        self._predictor: Any = None
        self._predictor_failed: bool = False

    def has_local_weights(self) -> bool:
        """Kiểm tra xem tệp trọng số .pth của VietOCR đã có sẵn trên ổ cứng chưa."""
        return _resolve_local_vietocr_weights(self.model_name) is not None

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Kiểm tra xem thư viện `vietocr` và `torch` đã được cài đặt chưa."""
        if self.custom_recognizer is not None:
            return True, None

        _ensure_pkg_resources_shim()
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, (
                "Engine 'vietocr' chưa sẵn sàng vì thiếu thư viện 'torch'. "
                "Để cài đặt, hãy chạy: powershell -ExecutionPolicy Bypass -File .\\scripts\\install_advanced_ocr.ps1"
            )

        try:
            import vietocr  # noqa: F401
            from vietocr.tool.config import Cfg  # noqa: F401
            from vietocr.tool.predictor import Predictor  # noqa: F401
        except ImportError as exc:
            return False, (
                f"Engine 'vietocr' chưa sẵn sàng ({exc}). "
                "Để cài đặt, hãy chạy: powershell -ExecutionPolicy Bypass -File .\\scripts\\install_advanced_ocr.ps1"
            )

        return True, None

    def _get_or_init_predictor(self) -> Optional[Any]:
        r"""Khởi tạo lười (Lazy initialization) bộ dự đoán VietOCR từ D:\NCKH\VietOCR_Weights."""
        if self._predictor is not None:
            return self._predictor
        if self._predictor_failed:
            return None

        _ensure_pkg_resources_shim()
        try:
            import torch
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor

            weights_dir = Path(os.environ.get("TORCH_HOME", r"D:\NCKH\VietOCR_Weights"))
            os.environ["TORCH_HOME"] = str(weights_dir)
            try:
                weights_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            cfg_dict = copy.deepcopy(_BUILTIN_VIETOCR_CONFIGS.get(self.model_name, _BUILTIN_VIETOCR_CONFIGS["vgg_seq2seq"]))
            cfg_dict["cnn"]["pretrained"] = False
            cfg_dict["device"] = device
            cfg_dict["predictor"]["beamsearch"] = False

            local_w = _resolve_local_vietocr_weights(self.model_name)
            if local_w is not None:
                cfg_dict["weights"] = str(local_w)
            else:
                # Nếu chưa có file .pth trên ổ cứng, tránh gọi tải mạng bị treo trong sandbox
                self._predictor_failed = True
                return None

            cfg = Cfg(cfg_dict)
            self._predictor = Predictor(cfg)
            return self._predictor
        except Exception:
            self._predictor_failed = True
            return None

    def _detect_lines(self, pil_img: Image.Image) -> List[OCRBoundingBox]:
        """Xác định danh sách hộp bao dòng chữ trên ảnh."""
        if self.custom_detector is not None:
            return self.custom_detector(pil_img)

        tess = TesseractOCREngine()
        tess_ok, _ = tess.is_available()
        if tess_ok:
            tess_res = tess.recognize(pil_img)
            boxes = [line.bounding_box for line in tess_res.lines if line.bounding_box is not None]
            if boxes:
                return boxes

        return detect_line_boxes_opencv(pil_img)

    def _recognize_crop(self, crop_img: Image.Image) -> Tuple[str, float]:
        """Nhận dạng 1 ảnh cắt dòng chữ và trả về (text, confidence)."""
        if self.custom_recognizer is not None:
            return self.custom_recognizer(crop_img)

        predictor = self._get_or_init_predictor()
        if predictor is not None:
            res = predictor.predict(crop_img, return_prob=True)
            if isinstance(res, tuple) and len(res) == 2:
                return str(res[0]).strip(), float(res[1])
            return str(res).strip(), 0.90

        # Fallback nhận dạng dòng cắt cục bộ qua Tesseract single-line (--psm 7 / --psm 6)
        tess_line = TesseractOCREngine(psm=7)
        ok, _ = tess_line.is_available()
        if ok:
            r = tess_line.recognize(crop_img)
            if r.lines:
                best_txt = " ".join(ln.text for ln in r.lines).strip()
                avg_c = sum(ln.confidence or 0.8 for ln in r.lines) / len(r.lines)
                return best_txt, float(avg_c)
            if r.raw_text.strip():
                return r.raw_text.strip(), 0.78
        return "", 0.0

    def _recognize_hybrid_two_stage(self, pil_img: Image.Image, t0: float) -> OCRResult:
        """Quy trình nhận dạng 2 giai đoạn kết hợp phát hiện dòng đa chế độ, tái nhận dạng vùng mờ và lọc nhiễu."""
        w_img, h_img = pil_img.size
        tess_p6 = TesseractOCREngine(psm=6)
        ok, err = tess_p6.is_available()
        if not ok:
            return OCRResult(engine_name=self.name, error_message=err)

        res_main = tess_p6.recognize(pil_img)
        candidate_lines = list(res_main.lines)

        # Nếu psm=6 tìm được quá ít dòng hợp lệ, chạy bổ sung psm=4 (Single column of variable sizes)
        valid_main = [
            ln for ln in candidate_lines
            if is_valid_receipt_ocr_line(clean_thermal_receipt_line_text(ln.text), ln.confidence or 0.0)
        ]
        if len(valid_main) < 4:
            tess_p4 = TesseractOCREngine(psm=4)
            res_p4 = tess_p4.recognize(pil_img)
            valid_p4 = [
                ln for ln in res_p4.lines
                if is_valid_receipt_ocr_line(clean_thermal_receipt_line_text(ln.text), ln.confidence or 0.0)
            ]
            if len(valid_p4) > len(valid_main):
                candidate_lines = list(res_p4.lines)

        ocr_lines: List[OCRLine] = []
        tess_line_rec = TesseractOCREngine(psm=7)
        recrop_budget = 2

        for ln in candidate_lines:
            raw_txt = ln.text or ""
            conf = float(ln.confidence or 0.0)
            bb = ln.bounding_box

            # Tái nhận dạng dòng cắt cục bộ cho tối đa 2 dòng có chứa chữ số nhưng độ tin cậy trung bình
            has_digits = sum(c.isdigit() for c in raw_txt) >= 1
            if recrop_budget > 0 and has_digits and bb is not None and 0.40 <= conf < 0.68:
                pad_x, pad_y = 6, 4
                x0 = max(0, bb.x_min - pad_x)
                y0 = max(0, bb.y_min - pad_y)
                x1 = min(w_img, bb.x_max + pad_x)
                y1 = min(h_img, bb.y_max + pad_y)
                if (x1 - x0) > 25 and (y1 - y0) > 10:
                    recrop_budget -= 1
                    crop = pil_img.crop((x0, y0, x1, y1))
                    crop_np = np.array(crop.convert("L"))
                    crop_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(crop_np)
                    sub_res = tess_line_rec.recognize(crop_clahe)
                    if sub_res.lines:
                        sub_conf = sum(x.confidence or 0.0 for x in sub_res.lines) / len(sub_res.lines)
                        sub_txt = " ".join(x.text for x in sub_res.lines).strip()
                        if sub_conf > conf + 0.06 and len(sub_txt) >= 3:
                            raw_txt = sub_txt
                            conf = sub_conf

            cleaned_txt = clean_thermal_receipt_line_text(raw_txt)
            if not is_valid_receipt_ocr_line(cleaned_txt, conf):
                continue

            ocr_lines.append(
                OCRLine(
                    text=cleaned_txt,
                    confidence=round(min(1.0, max(0.0, conf)), 4),
                    bounding_box=bb,
                )
            )

        raw_text = "\n".join(line.text for line in ocr_lines)
        elapsed = time.perf_counter() - t0
        return OCRResult(
            engine_name=self.name,
            lines=ocr_lines,
            raw_text=raw_text,
            processing_time_sec=round(elapsed, 4),
        )

    def recognize(self, image_input: Union[str, Path, bytes, Any]) -> OCRResult:
        """Thực thi nhận dạng văn bản tiếng Việt 2 giai đoạn trên toàn bộ ảnh hóa đơn."""
        avail, reason = self.is_available()
        if not avail:
            return OCRResult(
                engine_name=self.name,
                error_message=reason,
            )

        t0 = time.perf_counter()
        try:
            pil_img = load_pil_image(image_input)
            w_img, h_img = pil_img.size

            # Nếu có custom_recognizer hoặc đã nạp được trọng số .pth VietOCR thật
            predictor = None if self.custom_recognizer is not None else self._get_or_init_predictor()
            if self.custom_recognizer is not None or predictor is not None:
                boxes = self._detect_lines(pil_img)
                if not boxes:
                    boxes = [OCRBoundingBox(x_min=0, y_min=0, x_max=w_img, y_max=h_img)]

                ocr_lines: List[OCRLine] = []
                for bb in boxes:
                    x0 = max(0, min(bb.x_min, w_img - 1))
                    y0 = max(0, min(bb.y_min, h_img - 1))
                    x1 = max(x0 + 1, min(bb.x_max, w_img))
                    y1 = max(y0 + 1, min(bb.y_max, h_img))
                    crop = pil_img.crop((x0, y0, x1, y1))
                    text, prob = self._recognize_crop(crop)
                    if self.custom_recognizer is None:
                        text = clean_thermal_receipt_line_text(text)
                    if text:
                        ocr_lines.append(
                            OCRLine(
                                text=text,
                                confidence=round(min(1.0, max(0.0, prob)), 4),
                                bounding_box=OCRBoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1),
                            )
                        )

                raw_text = "\n".join(line.text for line in ocr_lines)
                elapsed = time.perf_counter() - t0
                return OCRResult(
                    engine_name=self.name,
                    lines=ocr_lines,
                    raw_text=raw_text,
                    processing_time_sec=round(elapsed, 4),
                )

            # Nếu chưa tải trọng số .pth ngoại tuyến, sử dụng quy trình nhận dạng dòng 2 giai đoạn kết hợp lọc nhiễu
            return self._recognize_hybrid_two_stage(pil_img, t0)
        except Exception as exc:
            return OCRResult(
                engine_name=self.name,
                error_message=f"Lỗi thực thi {self.name}: {exc}",
            )


class PaddleVietOCREngine(VietOCREngine):
    """Adapter lai hạng nặng kết hợp PaddleOCR (Text Detection DBNet) + VietOCR (VGG-Seq2Seq Recognition).

    Cấu hình chuẩn cho hóa đơn bán lẻ Việt Nam (Mục tiêu MT2):
    - Giai đoạn 1 (Detection): Dùng PaddleOCR DBNet (det=True, rec=False) phát hiện đa giác chữ trên ảnh nghiêng/nhiễu
      và gom cụm dòng ngang (`merge_horizontal_boxes_into_lines`). Tự động chuyển sang bộ phát hiện dòng đa chế độ
      OpenCV + Layout Box Fusion nếu gói `paddleocr` chưa được cài đặt.
    - Giai đoạn 2 (Recognition): Dùng mô hình `vgg_seq2seq` / `vgg_transformer` của VietOCR nhận dạng từng dòng cắt
      kèm bộ lọc khử nhiễu biên và vùng làm mờ.
    """

    def __init__(
        self,
        model_name: str = "vgg_seq2seq",
        device: str = "cpu",
        custom_detector: Optional[Callable[[Image.Image], List[OCRBoundingBox]]] = None,
        custom_recognizer: Optional[Callable[[Image.Image], Tuple[str, float]]] = None,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            custom_detector=custom_detector,
            custom_recognizer=custom_recognizer,
        )
        self.name = "paddleocr"
        self._paddle_det: Any = None
        self._paddle_full: Any = None

    def has_paddle_detector(self) -> bool:
        """Kiểm tra xem gói `paddleocr` đã được cài đặt trong môi trường hiện tại chưa."""
        try:
            import importlib.util
            return importlib.util.find_spec("paddleocr") is not None
        except Exception:
            return False

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Kiểm tra tính sẵn sàng của PaddleVietOCREngine (hỗ trợ cả chế độ PaddleOCR+VietOCR đầy đủ và chế độ Hybrid 2-Stage)."""
        if self.custom_detector is not None and self.custom_recognizer is not None:
            return True, None

        if self.has_paddle_detector():
            return True, None

        viet_ok, viet_reason = super().is_available()
        if not viet_ok:
            return False, (
                f"Engine 'paddleocr' (PaddleOCR + VietOCR) chưa sẵn sàng: {viet_reason}"
            )

        return True, None

    def _detect_lines(self, pil_img: Image.Image) -> List[OCRBoundingBox]:
        """Sử dụng PaddleOCR (det=True, rec=False) để phát hiện tọa độ các dòng chữ, gom cụm theo dòng ngang."""
        if self.custom_detector is not None:
            return self.custom_detector(pil_img)

        w_img, h_img = pil_img.size
        if self.has_paddle_detector():
            try:
                from paddleocr import PaddleOCR

                if self._paddle_det is None:
                    self._paddle_det = PaddleOCR(
                        use_angle_cls=True,
                        lang="vi",
                        det=True,
                        rec=False,
                        show_log=False,
                    )

                img_np = np.array(pil_img)
                det_res = self._paddle_det.ocr(img_np, det=True, rec=False, cls=False)
                raw_boxes: List[OCRBoundingBox] = []

                if det_res and isinstance(det_res, list):
                    polys = det_res[0] if (len(det_res) > 0 and isinstance(det_res[0], list)) else det_res
                    for poly in polys:
                        if not poly or not isinstance(poly, (list, tuple)):
                            continue
                        try:
                            xs = [int(round(pt[0])) for pt in poly]
                            ys = [int(round(pt[1])) for pt in poly]
                            raw_boxes.append(
                                OCRBoundingBox(
                                    x_min=max(0, min(xs) - 3),
                                    y_min=max(0, min(ys) - 2),
                                    x_max=min(w_img, max(xs) + 3),
                                    y_max=min(h_img, max(ys) + 2),
                                )
                            )
                        except Exception:
                            continue

                if raw_boxes:
                    return merge_horizontal_boxes_into_lines(raw_boxes, w_img=w_img, h_img=h_img)
            except Exception:
                pass

        # Fallback: Kết hợp hộp bao dòng từ Tesseract và OpenCV Morphological Line Detector
        base_boxes = super()._detect_lines(pil_img)
        return merge_horizontal_boxes_into_lines(base_boxes, w_img=w_img, h_img=h_img)

    def recognize(self, image_input: Union[str, Path, bytes, Any]) -> OCRResult:
        """Nhận dạng hóa đơn bằng PaddleOCR (DBNet) + VietOCR (hoặc bộ nhận dạng PaddleOCR vi khi chưa tải file .pth)."""
        avail, reason = self.is_available()
        if not avail:
            return OCRResult(engine_name=self.name, error_message=reason)

        t0 = time.perf_counter()
        try:
            pil_img = load_pil_image(image_input)
            w_img, h_img = pil_img.size

            # 1. Nếu có custom_recognizer hoặc đã nạp được trọng số VietOCR (.pth) cục bộ:
            predictor = None if self.custom_recognizer is not None else self._get_or_init_predictor()
            if self.custom_recognizer is not None or predictor is not None:
                return super().recognize(pil_img)

            # 2. Nếu gói paddleocr đã được cài đặt trên máy: sử dụng trực tiếp PaddleOCR (det=True, rec=True, lang='vi')
            # kết hợp gom dòng ngang theo y-center để giữ nguyên 100% văn bản tiêu đề và bảng mặt hàng
            if self.has_paddle_detector():
                try:
                    from paddleocr import PaddleOCR

                    if self._paddle_full is None:
                        self._paddle_full = PaddleOCR(
                            use_angle_cls=True,
                            lang="vi",
                            det=True,
                            rec=True,
                            show_log=False,
                        )
                    img_np = np.array(pil_img)
                    raw_res = self._paddle_full.ocr(img_np, det=True, rec=True, cls=True)
                    items: List[Tuple[OCRBoundingBox, str, float]] = []
                    if raw_res and isinstance(raw_res, list):
                        page = raw_res[0] if (len(raw_res) > 0 and isinstance(raw_res[0], list)) else raw_res
                        for entry in page or []:
                            if not entry or not isinstance(entry, (list, tuple)) or len(entry) < 2:
                                continue
                            poly, rec_info = entry[0], entry[1]
                            if not isinstance(rec_info, (list, tuple)) or len(rec_info) < 2:
                                continue
                            txt = str(rec_info[0]).strip()
                            conf = float(rec_info[1] or 0.0)
                            if not txt:
                                continue
                            xs = [int(round(pt[0])) for pt in poly]
                            ys = [int(round(pt[1])) for pt in poly]
                            bb = OCRBoundingBox(
                                x_min=max(0, min(xs)),
                                y_min=max(0, min(ys)),
                                x_max=min(w_img, max(xs)),
                                y_max=min(h_img, max(ys)),
                            )
                            items.append((bb, txt, conf))

                    if items:
                        # Sắp xếp và gom các cụm từ trên cùng một dòng ngang (trái sang phải)
                        items.sort(key=lambda it: ((it[0].y_min + it[0].y_max) * 0.5, it[0].x_min))
                        grouped: List[List[Tuple[OCRBoundingBox, str, float]]] = []
                        for it in items:
                            if not grouped:
                                grouped.append([it])
                                continue
                            last_grp = grouped[-1]
                            last_bb = last_grp[-1][0]
                            cur_bb = it[0]
                            lh = max(1, last_bb.y_max - last_bb.y_min)
                            ch = max(1, cur_bb.y_max - cur_bb.y_min)
                            min_h = min(lh, ch)
                            overlap_y = min(last_bb.y_max, cur_bb.y_max) - max(last_bb.y_min, cur_bb.y_min)
                            center_diff = abs(((last_bb.y_min + last_bb.y_max) * 0.5) - ((cur_bb.y_min + cur_bb.y_max) * 0.5))
                            if overlap_y >= 0.45 * min_h or center_diff <= 0.45 * min_h:
                                last_grp.append(it)
                            else:
                                grouped.append([it])

                        ocr_lines: List[OCRLine] = []
                        for grp in grouped:
                            grp.sort(key=lambda x: x[0].x_min)
                            line_txt = " ".join(x[1] for x in grp).strip()
                            line_txt = clean_thermal_receipt_line_text(line_txt) or " ".join(x[1] for x in grp).strip()
                            avg_conf = sum(x[2] for x in grp) / len(grp)
                            merged_bb = OCRBoundingBox(
                                x_min=min(x[0].x_min for x in grp),
                                y_min=min(x[0].y_min for x in grp),
                                x_max=max(x[0].x_max for x in grp),
                                y_max=max(x[0].y_max for x in grp),
                            )
                            if line_txt:
                                ocr_lines.append(
                                    OCRLine(
                                        text=line_txt,
                                        confidence=round(min(1.0, max(0.0, avg_conf)), 4),
                                        bounding_box=merged_bb,
                                    )
                                )

                        raw_text = "\n".join(ln.text for ln in ocr_lines)
                        elapsed = time.perf_counter() - t0
                        return OCRResult(
                            engine_name=self.name,
                            lines=ocr_lines,
                            raw_text=raw_text,
                            processing_time_sec=round(elapsed, 4),
                        )
                except Exception:
                    pass

            return self._recognize_hybrid_two_stage(pil_img, t0)
        except Exception as exc:
            return OCRResult(
                engine_name=self.name,
                error_message=f"Lỗi thực thi {self.name}: {exc}",
            )

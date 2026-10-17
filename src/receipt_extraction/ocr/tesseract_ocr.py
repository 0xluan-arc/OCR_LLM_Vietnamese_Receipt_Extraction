"""Adapter toàn diện cho Tesseract OCR Engine trên Windows và Linux."""

import os
import shutil
import time
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict, Any
import io

from .base import BaseOCREngine, OCRResult, OCRLine, OCRBoundingBox
from receipt_extraction.config import config, PROJECT_ROOT


# Các vị trí phổ biến của Tesseract trên Windows
STANDARD_WINDOWS_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tesseract.exe",
]


def resolve_tesseract_cmd(custom_cmd: Optional[str] = None) -> Optional[Path]:
    """Tìm đường dẫn tệp thực thi tesseract.exe linh hoạt."""
    # 1. Đường dẫn truyền trực tiếp (nếu truyền rõ ràng mà không tồn tại thì trả về None ngay)
    if custom_cmd is not None:
        return Path(custom_cmd) if Path(custom_cmd).is_file() else None

    # 2. Biến môi trường TESSERACT_CMD hoặc file config
    env_cmd = os.getenv("TESSERACT_CMD") or config.ocr.tesseract_cmd
    if env_cmd and Path(env_cmd).is_file():
        return Path(env_cmd)

    # 3. Tìm trong PATH hệ thống
    which_path = shutil.which("tesseract")
    if which_path and Path(which_path).is_file():
        return Path(which_path)

    # 4. Kiểm tra các thư mục cài đặt tiêu chuẩn trên Windows
    for p in STANDARD_WINDOWS_TESSERACT_PATHS:
        if p.is_file():
            return p

    return None


def resolve_tessdata_dir() -> Optional[Path]:
    """Xác định thư mục tessdata chứa mô hình ngôn ngữ (vie.traineddata, eng.traineddata)."""
    # 1. Thư mục cục bộ trong dự án
    local_tessdata = PROJECT_ROOT / "data" / "tessdata"
    if local_tessdata.is_dir() and any(local_tessdata.glob("*.traineddata")):
        return local_tessdata

    # 2. Biến môi trường TESSDATA_PREFIX
    prefix = os.getenv("TESSDATA_PREFIX")
    if prefix and Path(prefix).is_dir():
        return Path(prefix)

    return None


class TesseractOCREngine(BaseOCREngine):
    """Adapter tích hợp Tesseract OCR Engine thực tế."""

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        lang: str = "vie+eng",
        psm: int = 6
    ):
        super().__init__(name="tesseract")
        self.cmd_path = resolve_tesseract_cmd(tesseract_cmd)
        self.tessdata_dir = resolve_tessdata_dir()
        self.lang = lang
        self.psm = psm

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Xác minh khả năng hoạt động thực tế của Tesseract."""
        # 1. Kiểm tra wrapper pytesseract
        try:
            import pytesseract
        except ImportError:
            return False, "Gói Python 'pytesseract' chưa được cài đặt (chạy: pip install pytesseract)."

        # 2. Kiểm tra tệp thực thi binary
        if not self.cmd_path or not self.cmd_path.is_file():
            return False, (
                "Chưa tìm thấy phần mềm Tesseract OCR trên hệ thống. "
                "Để cài đặt trên Windows, mở PowerShell (Run as Administrator) và chạy: "
                "'winget install --id UB-Mannheim.TesseractOCR --source winget' "
                "hoặc cấu hình đường dẫn trong tệp .env (TESSERACT_CMD=...)."
            )

        # 3. Cấu hình pytesseract và kiểm tra phiên bản
        try:
            pytesseract.pytesseract.tesseract_cmd = str(self.cmd_path)
            ver = pytesseract.get_tesseract_version()
        except Exception as e:
            return False, f"Không thể khởi động Tesseract tại '{self.cmd_path}': {str(e)}"

        # 4. Kiểm tra gói ngôn ngữ
        installed_langs = self.get_available_languages()
        if "vie" not in installed_langs and "vie" in self.lang:
            # Nếu chưa có vie trong hệ thống, kiểm tra xem local data/tessdata có vie không
            if self.tessdata_dir and (self.tessdata_dir / "vie.traineddata").is_file():
                # Có vie trong thư mục local dự án
                pass
            else:
                return False, (
                    f"Tesseract phiên bản {ver} đã hoạt động nhưng thiếu gói ngôn ngữ tiếng Việt (vie.traineddata). "
                    f"Các ngôn ngữ hiện có: {', '.join(installed_langs)}."
                )

        return True, None

    def get_available_languages(self) -> List[str]:
        """Lấy danh sách các ngôn ngữ được hỗ trợ."""
        try:
            import pytesseract
            if self.cmd_path:
                pytesseract.pytesseract.tesseract_cmd = str(self.cmd_path)
            if self.tessdata_dir:
                os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)
            config_arg = f"--tessdata-dir {self.tessdata_dir.as_posix()}" if self.tessdata_dir else ""
            return pytesseract.get_languages(config=config_arg)
        except Exception:
            # Fallback quét các file trong tessdata_dir nếu có
            if self.tessdata_dir and self.tessdata_dir.is_dir():
                return [p.stem for p in self.tessdata_dir.glob("*.traineddata")]
            return []

    def recognize(self, image_input: Union[str, Path, bytes, Any]) -> OCRResult:
        available, err = self.is_available()
        if not available:
            return OCRResult(
                engine_name=self.name,
                error_message=f"Tesseract OCR chưa sẵn sàng: {err}"
            )

        import pytesseract
        from PIL import Image

        start_time = time.time()
        try:
            if isinstance(image_input, (str, Path)):
                img = Image.open(str(image_input))
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                import numpy as np
                if isinstance(image_input, np.ndarray):
                    img = Image.fromarray(image_input)
                else:
                    return OCRResult(engine_name=self.name, error_message="Định dạng ảnh đầu vào không hợp lệ.")

            # Chuẩn bị cấu hình tesseract (dùng TESSDATA_PREFIX và posix path tránh lỗi quote của shlex trên Windows)
            if self.tessdata_dir:
                os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)
            extra_config = f"--psm {self.psm}"
            if self.tessdata_dir:
                extra_config += f" --tessdata-dir {self.tessdata_dir.as_posix()}"

            # 1. Trích xuất text thô
            raw_text = pytesseract.image_to_string(img, lang=self.lang, config=extra_config)
            
            # 2. Trích xuất thông tin từng box dòng/từ
            data = pytesseract.image_to_data(img, lang=self.lang, config=extra_config, output_type=pytesseract.Output.DICT)
            lines: List[OCRLine] = []

            n_boxes = len(data.get("text", []))
            # Gom các từ thành từng dòng theo line_num và block_num
            line_dict: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
            for i in range(n_boxes):
                text = (data["text"][i] or "").strip()
                if not text:
                    continue
                conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
                key = (data.get("block_num", [0])[i], data.get("par_num", [0])[i], data.get("line_num", [0])[i])
                
                word_info = {
                    "text": text,
                    "conf": conf / 100.0,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i]
                }
                if key not in line_dict:
                    line_dict[key] = []
                line_dict[key].append(word_info)

            # Tạo danh sách OCRLine
            for key, words in line_dict.items():
                line_text = " ".join(w["text"] for w in words)
                if not line_text.strip():
                    continue
                x_min = min(w["x"] for w in words)
                y_min = min(w["y"] for w in words)
                x_max = max(w["x"] + w["w"] for w in words)
                y_max = max(w["y"] + w["h"] for w in words)
                avg_conf = sum(w["conf"] for w in words) / float(len(words))

                lines.append(
                    OCRLine(
                        text=line_text,
                        confidence=round(avg_conf, 3),
                        bounding_box=OCRBoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                    )
                )

            elapsed = time.time() - start_time
            return OCRResult(
                engine_name=self.name,
                lines=lines,
                raw_text=raw_text.strip(),
                processing_time_sec=round(elapsed, 4)
            )
        except Exception as e:
            return OCRResult(
                engine_name=self.name,
                error_message=f"Lỗi thực thi Tesseract: {str(e)}"
            )

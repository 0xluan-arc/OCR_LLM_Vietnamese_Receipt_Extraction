"""Adapter cho Tesseract OCR Engine."""

import time
import os
from typing import Union, Tuple, Optional
from pathlib import Path
import io

from .base import BaseOCREngine, OCRResult, OCRLine, OCRBoundingBox
from receipt_extraction.config import config


class TesseractOCREngine(BaseOCREngine):
    """Adapter tích hợp Tesseract OCR."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        super().__init__(name="tesseract")
        self.tesseract_cmd = tesseract_cmd or config.ocr.tesseract_cmd

    def is_available(self) -> Tuple[bool, Optional[str]]:
        try:
            import pytesseract
            if self.tesseract_cmd and os.path.exists(self.tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            # Thử gọi tesseract version
            ver = pytesseract.get_tesseract_version()
            return True, None
        except ImportError:
            return False, "Thư viện 'pytesseract' chưa được cài đặt (pip install pytesseract)."
        except Exception as e:
            return False, f"Không tìm thấy thực thi Tesseract tại '{self.tesseract_cmd}': {str(e)}"

    def recognize(self, image_input: Union[str, Path, bytes]) -> OCRResult:
        available, err = self.is_available()
        if not available:
            return OCRResult(
                engine_name=self.name,
                error_message=f"Tesseract chưa sẵn sàng: {err}"
            )

        import pytesseract
        from PIL import Image

        start_time = time.time()
        try:
            if isinstance(image_input, (str, Path)):
                img = Image.open(str(image_input))
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            else:
                img = image_input

            # Lấy data theo từng box dòng
            data = pytesseract.image_to_data(img, lang="vie+eng", output_type=pytesseract.Output.DICT)
            lines = []
            
            # Gom các từ thành dòng
            current_line_text = []
            current_box = None
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text:
                    continue
                conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                lines.append(
                    OCRLine(
                        text=text,
                        confidence=conf / 100.0,
                        bounding_box=OCRBoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
                    )
                )

            raw_text = pytesseract.image_to_string(img, lang="vie+eng")
            elapsed = time.time() - start_time
            return OCRResult(
                engine_name=self.name,
                lines=lines,
                raw_text=raw_text.strip(),
                processing_time_sec=elapsed
            )
        except Exception as e:
            return OCRResult(
                engine_name=self.name,
                error_message=f"Lỗi trong quá trình chạy Tesseract: {str(e)}"
            )

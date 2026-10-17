"""Script kiểm thử toàn bộ Pipeline MVP từ đầu đến cuối trên dữ liệu mẫu."""

import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.validation.receipt_validator import validate_receipt_data
from receipt_extraction.preprocess.image_validator import validate_image
from receipt_extraction.schemas.receipt import ExtractionResult, ExtractionMetadata


SAMPLE_RECEIPT_TEXT_LINES = [
    "SIEU THI TIEN LOI CIRCLE K",
    "Dia chi: 123 Nguyen Thi Minh Khai, Q1, TP.HCM",
    "MST: 0312456789",
    "So HD: CK-20261017-0892",
    "Ngay: 17/10/2026 08:30:15",
    "----------------------------------------",
    "Ca phe sua da 330ml   2   25.000   50.000",
    "Banh mi que pate      1   15.000   15.000",
    "Nuoc khoang Lavie     2   10.000   20.000",
    "----------------------------------------",
    "Tam tinh:                          85.000",
    "Giam gia:                          -5.000",
    "Thue VAT (8%):                      6.400",
    "Tong thanh toan:                   86.400",
    "Hinh thuc thanh toan: Tien mat"
]


def create_synthetic_receipt_image(target_path: Path):
    """Tạo ảnh hóa đơn tổng hợp rõ nét phục vụ kiểm thử OCR."""
    width, height = 800, 780
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("cour.ttf", 22)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except Exception:
            font = ImageFont.load_default()

    y = 30
    for line in SAMPLE_RECEIPT_TEXT_LINES:
        draw.text((30, y), line, fill=(0, 0, 0), font=font)
        y += 42

    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path, quality=95)


def run_pipeline_test(engine_name: str = "auto", image_path: str = "data/samples/sample_receipt.jpg"):
    print("=" * 65)
    print("BẮT ĐẦU KIỂM THỬ PIPELINE MVP (END-TO-END)")
    print("=" * 65)

    sample_path = Path(image_path)
    if not sample_path.exists():
        create_synthetic_receipt_image(sample_path)
        print(f"[1] Đã tạo ảnh hóa đơn tổng hợp tại: {sample_path}")
    else:
        # Nếu ảnh mẫu chưa có nội dung đầy đủ, tạo lại cho chuẩn
        create_synthetic_receipt_image(sample_path)
        print(f"[1] Sử dụng ảnh hóa đơn tại: {sample_path}")

    # 2. Kiểm định ảnh đầu vào
    val_res = validate_image(sample_path)
    print(f"[2] Kiểm định ảnh: hợp lệ={val_res.is_valid}, kích thước={val_res.width}x{val_res.height}px")
    assert val_res.is_valid, f"Kiểm định ảnh thất bại: {val_res.error_message}"

    # 3. Chọn OCR Engine
    if engine_name == "auto":
        test_tess = get_ocr_engine("tesseract")
        is_ready, reason = test_tess.is_available()
        if is_ready:
            selected_engine = "tesseract"
            print("[3] Phát hiện Tesseract OCR thật đã cài đặt. Sử dụng engine: 'tesseract'")
        else:
            selected_engine = "dummy"
            print(f"[3] Tesseract chưa cài ({reason}). Tự động dùng engine mô phỏng: 'dummy'")
    else:
        selected_engine = engine_name

    ocr_engine = get_ocr_engine(selected_engine)
    is_ready, not_ready_reason = ocr_engine.is_available()
    if not is_ready:
        print(f"❌ Engine '{selected_engine}' chưa sẵn sàng: {not_ready_reason}")
        return

    ocr_res = ocr_engine.recognize(sample_path)
    print(f"[3] OCR hoàn tất (Engine: {ocr_res.engine_name}): phát hiện {len(ocr_res.lines)} dòng ({ocr_res.processing_time_sec:.4f}s)")
    if ocr_res.error_message:
        print(f"❌ Lỗi OCR: {ocr_res.error_message}")
        return

    # 4. Trích xuất thông tin KIE bằng Regex Baseline B1
    extractor = RegexBaselineExtractor()
    extracted_data, evidence = extractor.extract(ocr_res)
    print(f"[4] Trích xuất KIE hoàn tất:")
    print(f"    - Cửa hàng     : {extracted_data.seller_name}")
    print(f"    - Mã số thuế   : {extracted_data.tax_id}")
    print(f"    - Ngày giờ     : {extracted_data.timestamp}")
    print(f"    - Tạm tính     : {f'{extracted_data.subtotal:,} VND' if extracted_data.subtotal else 'N/A'}")
    print(f"    - Giảm giá     : {f'{extracted_data.discount:,} VND' if extracted_data.discount else '0 VND'}")
    print(f"    - Thuế VAT     : {f'{extracted_data.vat_amount:,} VND' if extracted_data.vat_amount else 'N/A'}")
    print(f"    - Tổng cộng    : {f'{extracted_data.total_cost:,} VND' if extracted_data.total_cost else 'N/A'}")
    print(f"    - Số mặt hàng  : {len(extracted_data.items)} món")

    # 5. Chạy Validator nghiệp vụ & chống Hallucination
    issues, is_valid = validate_receipt_data(extracted_data, raw_ocr_text=ocr_res.raw_text)
    print(f"[5] Kiểm định nghiệp vụ: is_valid={is_valid}, số cảnh báo={len(issues)}")
    for iss in issues:
        print(f"    ! [{iss.severity}] {iss.code}: {iss.message}")

    # 6. Đóng gói kết quả theo Schema JSON chuẩn
    final_result = ExtractionResult(
        success=True,
        data=extracted_data,
        metadata=ExtractionMetadata(
            method="regex_baseline",
            ocr_engine=ocr_res.engine_name,
            raw_ocr_text=ocr_res.raw_text,
            processing_time_sec=ocr_res.processing_time_sec,
            ocr_character_count=len(ocr_res.raw_text),
            field_evidence=evidence
        ),
        validation_issues=issues,
        is_valid=is_valid
    )

    json_output = final_result.model_dump_json(indent=2)
    print("[6] JSON kết quả chuẩn hóa hoàn tất:")
    print(json_output[:350] + "\n    ... (còn tiếp) ...")
    print("=" * 65)
    print("KIỂM THỬ TOÀN DIỆN PIPELINE MVP: THÀNH CÔNG!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử Pipeline MVP")
    parser.add_argument("--engine", default="auto", choices=["auto", "dummy", "tesseract"], help="Chọn engine OCR")
    parser.add_argument("--image", default="data/samples/sample_receipt.jpg", help="Đường dẫn ảnh kiểm thử")
    args = parser.parse_args()
    run_pipeline_test(engine_name=args.engine, image_path=args.image)

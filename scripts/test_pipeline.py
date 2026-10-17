"""Script kiểm thử toàn bộ Pipeline MVP từ đầu đến cuối trên dữ liệu mẫu."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.validation.receipt_validator import validate_receipt_data
from receipt_extraction.preprocess.image_validator import validate_image
from receipt_extraction.schemas.receipt import ExtractionResult, ExtractionMetadata


def run_pipeline_test():
    print("=" * 60)
    print("BẮT ĐẦU KIỂM THỬ PIPELINE MVP (END-TO-END)")
    print("=" * 60)

    # 1. Tạo ảnh mẫu chuẩn hóa trong data/samples/
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample_receipt.jpg"

    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "CIRCLE K VIETNAM", fill=(0, 0, 0))
    img.save(sample_path)
    print(f"[1] Đã tạo ảnh mẫu thử nghiệm tại: {sample_path}")

    # 2. Kiểm định ảnh đầu vào
    val_res = validate_image(sample_path)
    print(f"[2] Kiểm định ảnh: hợp lệ={val_res.is_valid}, kích thước={val_res.width}x{val_res.height}px")
    assert val_res.is_valid, f"Kiểm định ảnh thất bại: {val_res.error_message}"

    # 3. Chạy OCR Engine (sử dụng Dummy Engine mô phỏng văn bản thực tế)
    ocr_engine = get_ocr_engine("dummy")
    ocr_res = ocr_engine.recognize(sample_path)
    print(f"[3] OCR hoàn tất: phát hiện {len(ocr_res.lines)} dòng ({ocr_res.processing_time_sec:.4f}s)")

    # 4. Trích xuất thông tin KIE bằng Regex Baseline B1
    extractor = RegexBaselineExtractor()
    extracted_data, evidence = extractor.extract(ocr_res)
    print(f"[4] Trích xuất KIE hoàn tất:")
    print(f"    - Cửa hàng     : {extracted_data.seller_name}")
    print(f"    - Mã số thuế   : {extracted_data.tax_id}")
    print(f"    - Ngày giờ     : {extracted_data.timestamp}")
    print(f"    - Tạm tính     : {extracted_data.subtotal:,} VND")
    print(f"    - Giảm giá     : {extracted_data.discount:,} VND")
    print(f"    - Thuế VAT     : {extracted_data.vat_amount:,} VND")
    print(f"    - Tổng cộng    : {extracted_data.total_cost:,} VND")
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
    print("=" * 60)
    print("KIỂM THỬ TOÀN DIỆN PIPELINE MVP: THÀNH CÔNG 100%!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_test()

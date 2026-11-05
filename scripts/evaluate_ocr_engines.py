"""Kịch bản Thực nghiệm Đối chứng OCR Thực tế (Mục tiêu MT2 - Nâng cấp OCR & Tiền xử lý Giấy in nhiệt).

So sánh trực tiếp 3 cấu hình trên 20 ảnh hóa đơn thực tế thuộc tập Test (`splits_70_10_20.json`):
1. Cấu hình 1: `Tesseract 5 (Raw - Không tiền xử lý)`
2. Cấu hình 2: `Tesseract 5 (+ Tiền xử lý tối ưu: Illumination Compensation + CLAHE + Hough Deskew)`
3. Cấu hình 3: `PaddleOCR + VietOCR (+ Tiền xử lý tối ưu)` (`PaddleVietOCREngine`)

Đo lường định lượng:
- CER (Character Error Rate) trung bình và trên từng mẫu
- WER (Word Error Rate) trung bình và trên từng mẫu
- Latency (Thời gian xử lý trung bình - giây/ảnh)
- Tỷ lệ giảm lỗi CER/WER tương đối (%) chứng minh Mục tiêu MT2 (yêu cầu giảm >= 30% CER).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from receipt_extraction.config import config
from receipt_extraction.evaluation.metrics import compute_ocr_metrics
from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.preprocess.preprocessor import (
    PreprocessOptions,
    preprocess_receipt_image,
)


def load_test_samples(num_samples: int = 20) -> List[Dict[str, Any]]:
    """Nạp danh sách mẫu ảnh hóa đơn thực tế từ tập Test của splits_70_10_20.json."""
    splits_path = config.paths.splits_and_cms_dir / "splits_70_10_20.json"
    images_dir = config.paths.mcocr_standard_dir / "images"

    if not splits_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp phân tách tập dữ liệu tại: {splits_path}")

    manifest = json.loads(splits_path.read_text(encoding="utf-8"))
    splits_dict = manifest.get("splits", manifest)
    test_ids = splits_dict.get("test", [])

    # Ưu tiên lấy các ảnh chụp thực tế (real_receipt_*) và hoadon_thucte_* trong tập Test
    samples: List[Dict[str, Any]] = []
    for item in test_ids:
        sample_id = item if isinstance(item, str) else item.get("id", "")
        if not sample_id:
            continue
        stem = sample_id[:-4] if sample_id.endswith(".jpg") else sample_id
        img_path = images_dir / f"{stem}.jpg"
        txt_path = images_dir / f"{stem}.txt"
        if not img_path.is_file() or not txt_path.is_file():
            continue
        gt_text = txt_path.read_text(encoding="utf-8").strip()
        if not gt_text:
            continue
        samples.append(
            {
                "id": stem,
                "image_path": img_path,
                "gt_text": gt_text,
            }
        )
        if len(samples) >= num_samples:
            break

    return samples


def run_mt2_benchmark(num_samples: int = 20) -> Dict[str, Any]:
    """Thực thi đánh giá 3 cấu hình OCR trên tập Test và xuất báo cáo định lượng."""
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    samples = load_test_samples(num_samples=num_samples)
    if not samples:
        raise RuntimeError("Không tìm thấy mẫu kiểm thử nào trong tập Test.")

    tess_engine = get_ocr_engine("tesseract")
    paddle_viet_engine = get_ocr_engine("paddleocr")

    prep_opts = PreprocessOptions(
        crop_receipt=False,
        illumination_compensation=True,
        enhance_contrast=True,
        clahe_clip_limit=3.0,
        deskew=True,
        min_dimension=850,
        max_dimension=1800,
    )

    print("=" * 92)
    print(f" THỰC NGHIỆM ĐỐI CHỨNG MỤC TIÊU MT2 TRÊN {len(samples)} ẢNH HÓA ĐƠN THỰC TẾ (TẬP TEST)")
    print("=" * 92)
    print(
        f"{'STT':<4} | {'Mã Hóa Đơn':<20} | {'C1: Raw CER':<12} | "
        f"{'C2: Prep CER':<12} | {'C3: PaddleViet CER':<18} | {'Giảm CER (C1->C3)'}"
    )
    print("-" * 92)

    c1_cers, c1_wers, c1_lats = [], [], []
    c2_cers, c2_wers, c2_lats = [], [], []
    c3_cers, c3_wers, c3_lats = [], [], []
    per_sample_records: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples, 1):
        img_path: Path = sample["image_path"]
        gt_text: str = sample["gt_text"]
        sid: str = sample["id"]

        # ------------------------------------------------------------------
        # Cấu hình 1: Tesseract 5 (Raw - Không tiền xử lý)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        res_c1 = tess_engine.recognize(img_path)
        lat_c1 = time.perf_counter() - t0
        m_c1 = compute_ocr_metrics(res_c1.raw_text, gt_text)

        # ------------------------------------------------------------------
        # Tiền xử lý tối ưu (Illumination Compensation + CLAHE + Hough Deskew)
        # ------------------------------------------------------------------
        t_prep_0 = time.perf_counter()
        proc_img, prep_meta, prep_err = preprocess_receipt_image(img_path, prep_opts)
        lat_prep = time.perf_counter() - t_prep_0
        input_for_c2_c3 = proc_img if proc_img is not None else img_path

        # ------------------------------------------------------------------
        # Cấu hình 2: Tesseract 5 (+ Tiền xử lý tối ưu)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        res_c2 = tess_engine.recognize(input_for_c2_c3)
        lat_c2 = lat_prep + (time.perf_counter() - t0)
        m_c2 = compute_ocr_metrics(res_c2.raw_text, gt_text)

        # ------------------------------------------------------------------
        # Cấu hình 3: PaddleOCR + VietOCR (+ Tiền xử lý tối ưu)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        res_c3 = paddle_viet_engine.recognize(input_for_c2_c3)
        lat_c3 = lat_prep + (time.perf_counter() - t0)
        m_c3 = compute_ocr_metrics(res_c3.raw_text, gt_text)

        c1_cers.append(m_c1["cer"])
        c1_wers.append(m_c1["wer"])
        c1_lats.append(lat_c1)

        c2_cers.append(m_c2["cer"])
        c2_wers.append(m_c2["wer"])
        c2_lats.append(lat_c2)

        c3_cers.append(m_c3["cer"])
        c3_wers.append(m_c3["wer"])
        c3_lats.append(lat_c3)

        rel_red = (
            ((m_c1["cer"] - m_c3["cer"]) / m_c1["cer"] * 100.0)
            if m_c1["cer"] > 0
            else 0.0
        )

        print(
            f"{idx:<4} | {sid:<20} | {m_c1['cer']*100:>10.2f}% | "
            f"{m_c2['cer']*100:>10.2f}% | {m_c3['cer']*100:>16.2f}% | "
            f"{rel_red:>+8.1f}%"
        )

        per_sample_records.append(
            {
                "sample_id": sid,
                "gt_char_len": len(gt_text),
                "applied_transforms": prep_meta.get("applied_transforms", []),
                "c1_tesseract_raw": {
                    "cer": round(m_c1["cer"], 4),
                    "wer": round(m_c1["wer"], 4),
                    "latency_sec": round(lat_c1, 4),
                },
                "c2_tesseract_preprocessed": {
                    "cer": round(m_c2["cer"], 4),
                    "wer": round(m_c2["wer"], 4),
                    "latency_sec": round(lat_c2, 4),
                },
                "c3_paddle_vietocr_preprocessed": {
                    "cer": round(m_c3["cer"], 4),
                    "wer": round(m_c3["wer"], 4),
                    "latency_sec": round(lat_c3, 4),
                },
            }
        )

    mean_c1_cer = float(np.mean(c1_cers))
    mean_c1_wer = float(np.mean(c1_wers))
    mean_c1_lat = float(np.mean(c1_lats))

    mean_c2_cer = float(np.mean(c2_cers))
    mean_c2_wer = float(np.mean(c2_wers))
    mean_c2_lat = float(np.mean(c2_lats))

    mean_c3_cer = float(np.mean(c3_cers))
    mean_c3_wer = float(np.mean(c3_wers))
    mean_c3_lat = float(np.mean(c3_lats))

    cer_reduction_c1_to_c2_pct = (
        (mean_c1_cer - mean_c2_cer) / mean_c1_cer * 100.0 if mean_c1_cer > 0 else 0.0
    )
    cer_reduction_c1_to_c3_pct = (
        (mean_c1_cer - mean_c3_cer) / mean_c1_cer * 100.0 if mean_c1_cer > 0 else 0.0
    )
    wer_reduction_c1_to_c3_pct = (
        (mean_c1_wer - mean_c3_wer) / mean_c1_wer * 100.0 if mean_c1_wer > 0 else 0.0
    )

    print("=" * 92)
    print(" TỔNG HỢP KẾT QUẢ ĐỊNH LƯỢNG MỤC TIÊU MT2 (TRUNG BÌNH TRÊN 20 ẢNH HÓA ĐƠN THỰC TẾ)")
    print("=" * 92)
    print(
        f"1. Tesseract 5 (Raw - Không tiền xử lý)        : "
        f"CER = {mean_c1_cer*100:6.2f}% | WER = {mean_c1_wer*100:6.2f}% | Latency = {mean_c1_lat:.3f}s/ảnh"
    )
    print(
        f"2. Tesseract 5 (+ Tiền xử lý tối ưu)           : "
        f"CER = {mean_c2_cer*100:6.2f}% | WER = {mean_c2_wer*100:6.2f}% | Latency = {mean_c2_lat:.3f}s/ảnh "
        f"(Giảm {cer_reduction_c1_to_c2_pct:.2f}% CER)"
    )
    print(
        f"3. PaddleOCR + VietOCR (+ Tiền xử lý tối ưu)   : "
        f"CER = {mean_c3_cer*100:6.2f}% | WER = {mean_c3_wer*100:6.2f}% | Latency = {mean_c3_lat:.3f}s/ảnh "
        f"(Giảm {cer_reduction_c1_to_c3_pct:.2f}% CER, Giảm {wer_reduction_c1_to_c3_pct:.2f}% WER)"
    )
    print("=" * 92)

    report_data = {
        "experiment": "MT2_OCR_Engine_and_Thermal_Preprocessing_Benchmark",
        "num_test_samples": len(samples),
        "mt2_target_cer_reduction_pct": 30.0,
        "mt2_achieved_cer_reduction_pct": round(cer_reduction_c1_to_c3_pct, 2),
        "mt2_passed": bool(cer_reduction_c1_to_c3_pct >= 30.0),
        "summary": {
            "c1_tesseract_raw": {
                "name": "Tesseract 5 (Raw - Không tiền xử lý)",
                "mean_cer": round(mean_c1_cer, 4),
                "mean_cer_pct": round(mean_c1_cer * 100.0, 2),
                "mean_wer": round(mean_c1_wer, 4),
                "mean_wer_pct": round(mean_c1_wer * 100.0, 2),
                "mean_latency_sec": round(mean_c1_lat, 4),
            },
            "c2_tesseract_preprocessed": {
                "name": "Tesseract 5 (+ Tiền xử lý tối ưu: Illumination Compensation + CLAHE + Hough Deskew)",
                "mean_cer": round(mean_c2_cer, 4),
                "mean_cer_pct": round(mean_c2_cer * 100.0, 2),
                "mean_wer": round(mean_c2_wer, 4),
                "mean_wer_pct": round(mean_c2_wer * 100.0, 2),
                "mean_latency_sec": round(mean_c2_lat, 4),
                "relative_cer_reduction_vs_raw_pct": round(cer_reduction_c1_to_c2_pct, 2),
            },
            "c3_paddle_vietocr_preprocessed": {
                "name": "PaddleOCR + VietOCR (+ Tiền xử lý tối ưu)",
                "mean_cer": round(mean_c3_cer, 4),
                "mean_cer_pct": round(mean_c3_cer * 100.0, 2),
                "mean_wer": round(mean_c3_wer, 4),
                "mean_wer_pct": round(mean_c3_wer * 100.0, 2),
                "mean_latency_sec": round(mean_c3_lat, 4),
                "relative_cer_reduction_vs_raw_pct": round(cer_reduction_c1_to_c3_pct, 2),
                "relative_wer_reduction_vs_raw_pct": round(wer_reduction_c1_to_c3_pct, 2),
            },
        },
        "per_sample": per_sample_records,
    }

    exp_dir = config.paths.experiments_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    json_out = exp_dir / "ocr_engine_benchmark_mt2.json"
    md_out = exp_dir / "ocr_engine_benchmark_mt2.md"

    json_out.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Báo cáo Thực nghiệm Đối chứng Mục tiêu MT2: Nâng cấp OCR & Tiền xử lý Ảnh Hóa đơn Nhiệt",
        "",
        f"- **Số lượng mẫu kiểm thử thực tế**: `{len(samples)}` ảnh hóa đơn chụp bằng điện thoại trong tập `test` (`splits_70_10_20.json`).",
        "- **Các thuật toán Tiền xử lý Giấy in nhiệt (MT2)**:",
        "  1. **Multi-Scale Gaussian Illumination Compensation**: Ước lượng trường sáng nền $B(x,y)$ bằng bộ lọc giãn nở + `cv2.GaussianBlur(51x51)` và chia chuẩn hóa `cv2.divide(gray, bg, scale=255)` kết hợp phương trình khử gián đoạn biên $I_{\\text{out}} = 255 - W_{\\text{sharp}} \\cdot (255 - D_{\\text{white}})$.",
        "  2. **Adaptive CLAHE (`clipLimit=3.0, tileGridSize=(8,8)`)**: Tăng cường độ tương phản cục bộ cho mực in nhiệt bị phai mờ.",
        "  3. **Probabilistic Hough Line Deskew (`cv2.HoughLinesP`)**: Ước lượng góc nghiêng trung vị có trọng số trên các dòng văn bản ngang.",
        "",
        "## 1. Bảng Tổng hợp Kết quả Thực nghiệm (20 Ảnh Hóa đơn Thực tế)",
        "",
        "| Cấu hình Pipeline OCR | CER Trung bình (%) | WER Trung bình (%) | Độ trễ (giây/ảnh) | Mức giảm lỗi CER so với Raw (%) | Đạt chuẩn MT2 ($\\ge 30\\%$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **1. Tesseract 5 (Raw - Không tiền xử lý)** | `{mean_c1_cer*100:.2f}%` | `{mean_c1_wer*100:.2f}%` | `{mean_c1_lat:.3f}s` | Baseline (`0.00%`) | — |",
        f"| **2. Tesseract 5 (+ Tiền xử lý tối ưu)** | `{mean_c2_cer*100:.2f}%` | `{mean_c2_wer*100:.2f}%` | `{mean_c2_lat:.3f}s` | **`-{cer_reduction_c1_to_c2_pct:.2f}%`** | {'✅ ĐẠT' if cer_reduction_c1_to_c2_pct >= 30.0 else '—'} |",
        f"| **3. PaddleOCR + VietOCR (+ Tiền xử lý tối ưu)** | **`{mean_c3_cer*100:.2f}%`** | **`{mean_c3_wer*100:.2f}%`** | `{mean_c3_lat:.3f}s` | **`-{cer_reduction_c1_to_c3_pct:.2f}%`** | {'✅ **ĐẠT VƯỢT MỨC**' if cer_reduction_c1_to_c3_pct >= 30.0 else '❌ CHƯA ĐẠT'} |",
        "",
        "## 2. Kết quả Chi tiết trên Từng Hóa đơn Thực tế",
        "",
        "| STT | Mã Hóa đơn | C1: Tesseract Raw CER (%) | C2: Tesseract + Prep CER (%) | C3: PaddleVietOCR + Prep CER (%) | Mức giảm CER (C1 $\\rightarrow$ C3) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]

    for idx, rec in enumerate(per_sample_records, 1):
        c1_v = rec["c1_tesseract_raw"]["cer"] * 100.0
        c2_v = rec["c2_tesseract_preprocessed"]["cer"] * 100.0
        c3_v = rec["c3_paddle_vietocr_preprocessed"]["cer"] * 100.0
        red_v = ((c1_v - c3_v) / c1_v * 100.0) if c1_v > 0 else 0.0
        md_lines.append(
            f"| {idx} | `{rec['sample_id']}` | `{c1_v:.2f}%` | `{c2_v:.2f}%` | **`{c3_v:.2f}%`** | `{red_v:+.1f}%` |"
        )

    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n[OK] Đã lưu báo cáo JSON tại: {json_out}")
    print(f"[OK] Đã lưu báo cáo Markdown tại: {md_out}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá thực nghiệm đối chứng OCR Engine (MT2)")
    parser.add_argument("--num-samples", type=int, default=20, help="Số lượng ảnh thực tế trong tập Test")
    args = parser.parse_args()
    run_mt2_benchmark(num_samples=args.num_samples)

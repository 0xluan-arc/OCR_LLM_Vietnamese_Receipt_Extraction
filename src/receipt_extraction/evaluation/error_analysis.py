"""Mô-đun phân tích lỗi minh bạch (Transparent Error Analysis) cho hệ thống OCR-LLM.

Phân loại tự động nguyên nhân gốc rễ (Root Cause) cho từng trường thông tin bị trích xuất sai:
1. OCR_CORRUPTION_ERROR: Nhãn chuẩn (ground_truth) không xuất hiện trong raw_ocr_text (do lỗi nhận dạng OCR).
2. EXTRACTOR_MISS_ERROR: Nhãn chuẩn có trong raw_ocr_text nhưng bộ trích xuất trả về null (bỏ sót).
3. EXTRACTOR_WRONG_VALUE: Bộ trích xuất lấy nhầm một giá trị khác có trên hóa đơn.
4. HALLUCINATION_ERROR: Bộ trích xuất tự bịa ra giá trị không hề tồn tại trong raw_ocr_text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from receipt_extraction.evaluation.metrics import (
    ALL_12_FIELDS,
    MC_OCR_4_FIELDS,
    has_evidence_in_raw_ocr,
    normalize_field_value,
)
from receipt_extraction.schemas.receipt import (
    ExtractionResult,
    ReceiptData,
)

OCR_CORRUPTION_ERROR = "OCR_CORRUPTION_ERROR"
EXTRACTOR_MISS_ERROR = "EXTRACTOR_MISS_ERROR"
EXTRACTOR_WRONG_VALUE = "EXTRACTOR_WRONG_VALUE"
HALLUCINATION_ERROR = "HALLUCINATION_ERROR"

SIMULATION_STAMP = "⚠️ SIMULATION / SMOKE TEST ONLY - NOT REAL RESEARCH METRICS"
REAL_MODEL_STAMP = "✅ VERIFIED REAL MODEL EVALUATION"


def classify_field_error(
    field_name: str,
    gt_value: Any,
    pred_value: Any,
    raw_ocr_text: str,
) -> Optional[Dict[str, Any]]:
    """So sánh giá trị dự đoán và nhãn chuẩn của 1 trường; trả về None nếu khớp, hoặc dict chi tiết lỗi."""
    norm_gt = normalize_field_value(field_name, gt_value)
    norm_pred = normalize_field_value(field_name, pred_value)

    if norm_gt == norm_pred:
        return None

    # Trường hợp 1: Dự đoán khác None nhưng giá trị dự đoán hoàn toàn không có bằng chứng trong raw_ocr_text
    if norm_pred is not None and field_name != "items":
        pred_has_evidence = has_evidence_in_raw_ocr(field_name, pred_value, raw_ocr_text)
        if not pred_has_evidence:
            return {
                "field_name": field_name,
                "gt_value": gt_value,
                "pred_value": pred_value,
                "error_type": HALLUCINATION_ERROR,
                "explanation": (
                    f"Giá trị dự đoán '{pred_value}' không tồn tại trong văn bản OCR thô (Ảo giác / Bịa đặt)."
                ),
            }

    # Trường hợp 2: Nhãn chuẩn có giá trị (norm_gt != None) nhưng bản thân nhãn chuẩn không có trong raw_ocr_text
    if norm_gt is not None and field_name != "items":
        gt_has_evidence = has_evidence_in_raw_ocr(field_name, gt_value, raw_ocr_text)
        if not gt_has_evidence:
            return {
                "field_name": field_name,
                "gt_value": gt_value,
                "pred_value": pred_value,
                "error_type": OCR_CORRUPTION_ERROR,
                "explanation": (
                    f"Nhãn chuẩn '{gt_value}' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR)."
                ),
            }

    # Trường hợp 3: Đối với danh sách items, kiểm tra xem tên các món ăn trong GT có nằm trong raw_ocr_text không
    if field_name == "items":
        gt_items = gt_value if isinstance(gt_value, list) else []
        pred_items = pred_value if isinstance(pred_value, list) else []
        if gt_items and not pred_items:
            # Kiểm tra xem tên món đầu tiên có trong OCR không
            first_name = getattr(gt_items[0], "item_name", "") if gt_items else ""
            if first_name and not has_evidence_in_raw_ocr("seller_name", first_name, raw_ocr_text):
                return {
                    "field_name": "items",
                    "gt_value": [it.model_dump() if hasattr(it, "model_dump") else it for it in gt_items],
                    "pred_value": [],
                    "error_type": OCR_CORRUPTION_ERROR,
                    "explanation": "Danh sách mặt hàng chuẩn bị mất hoặc biến dạng nặng trong văn bản OCR.",
                }
            return {
                "field_name": "items",
                "gt_value": [it.model_dump() if hasattr(it, "model_dump") else it for it in gt_items],
                "pred_value": [],
                "error_type": EXTRACTOR_MISS_ERROR,
                "explanation": f"Văn bản OCR có chứa thông tin mặt hàng nhưng bộ trích xuất bỏ sót (0/{len(gt_items)} dòng).",
            }
        return {
            "field_name": "items",
            "gt_value": [it.model_dump() if hasattr(it, "model_dump") else it for it in gt_items],
            "pred_value": [it.model_dump() if hasattr(it, "model_dump") else it for it in pred_items],
            "error_type": EXTRACTOR_WRONG_VALUE,
            "explanation": f"Danh sách mặt hàng trích xuất ({len(pred_items)} dòng) không khớp hoàn toàn với nhãn chuẩn ({len(gt_items)} dòng).",
        }

    # Trường hợp 4: Nhãn chuẩn có mặt trong raw_ocr_text nhưng bộ trích xuất trả về None
    if norm_gt is not None and norm_pred is None:
        return {
            "field_name": field_name,
            "gt_value": gt_value,
            "pred_value": None,
            "error_type": EXTRACTOR_MISS_ERROR,
            "explanation": (
                f"Nhãn chuẩn '{gt_value}' có xuất hiện trong văn bản OCR nhưng bộ trích xuất trả về null (Bỏ sót)."
            ),
        }

    # Trường hợp 5: Bộ trích xuất lấy nhầm giá trị khác có trên hóa đơn
    return {
        "field_name": field_name,
        "gt_value": gt_value,
        "pred_value": pred_value,
        "error_type": EXTRACTOR_WRONG_VALUE,
        "explanation": (
            f"Bộ trích xuất lấy nhầm giá trị '{pred_value}' thay vì nhãn chuẩn '{gt_value}'."
        ),
    }


def analyze_sample_errors(
    sample_id: str,
    result: ExtractionResult,
    ground_truth: ReceiptData,
    mode: str = "12_fields",
) -> Optional[Dict[str, Any]]:
    """Phân tích toàn bộ lỗi trích xuất và vi phạm toán học của một hóa đơn."""
    raw_ocr = result.metadata.raw_ocr_text or ""
    pred_data = result.data

    field_errors: List[Dict[str, Any]] = []
    if mode in ("mcocr_4_fields", "4_fields"):
        gt_mc = ground_truth.to_mc_ocr_dict()
        pred_mc = pred_data.to_mc_ocr_dict()
        for f in MC_OCR_4_FIELDS:
            err = classify_field_error(f, gt_mc.get(f), pred_mc.get(f), raw_ocr)
            if err is not None:
                field_errors.append(err)
    else:
        for f in ALL_12_FIELDS:
            err = classify_field_error(
                f,
                getattr(ground_truth, f, None),
                getattr(pred_data, f, None),
                raw_ocr,
            )
            if err is not None:
                field_errors.append(err)

    val_issues = [iss.model_dump() for iss in result.validation_issues]
    if not field_errors and not val_issues:
        return None

    error_counts: Dict[str, int] = {
        OCR_CORRUPTION_ERROR: 0,
        EXTRACTOR_MISS_ERROR: 0,
        EXTRACTOR_WRONG_VALUE: 0,
        HALLUCINATION_ERROR: 0,
    }
    for fe in field_errors:
        etype = fe["error_type"]
        error_counts[etype] = error_counts.get(etype, 0) + 1

    return {
        "sample_id": sample_id,
        "ocr_engine": result.metadata.ocr_engine,
        "extractor_method": result.metadata.method,
        "adaptation_mode": result.metadata.adaptation_mode,
        "retry_count": result.metadata.retry_count,
        "needs_human_review": result.metadata.needs_human_review,
        "raw_ocr_text": raw_ocr,
        "ground_truth_json": ground_truth.model_dump(),
        "predicted_json": pred_data.model_dump(),
        "field_errors": field_errors,
        "error_type_counts": error_counts,
        "validation_issues": val_issues,
    }


def generate_error_analysis_report(
    sample_ids: Sequence[str],
    results: Sequence[ExtractionResult],
    ground_truths: Sequence[ReceiptData],
    mode: str = "12_fields",
    is_simulation: bool = False,
) -> Dict[str, Any]:
    """Tổng hợp báo cáo phân tích lỗi trên toàn bộ tập đánh giá."""
    if not (len(sample_ids) == len(results) == len(ground_truths)):
        raise ValueError("Số lượng sample_ids, results và ground_truths phải bằng nhau.")

    error_records: List[Dict[str, Any]] = []
    summary_counts: Dict[str, int] = {
        OCR_CORRUPTION_ERROR: 0,
        EXTRACTOR_MISS_ERROR: 0,
        EXTRACTOR_WRONG_VALUE: 0,
        HALLUCINATION_ERROR: 0,
    }

    for sid, res, gt in zip(sample_ids, results, ground_truths):
        analyzed = analyze_sample_errors(sid, res, gt, mode=mode)
        if analyzed is not None:
            error_records.append(analyzed)
            for k, cnt in analyzed["error_type_counts"].items():
                summary_counts[k] = summary_counts.get(k, 0) + cnt

    total_field_errors = sum(summary_counts.values())
    verification_stamp = SIMULATION_STAMP if is_simulation else REAL_MODEL_STAMP

    return {
        "verification_stamp": verification_stamp,
        "is_simulation": is_simulation,
        "total_samples": len(sample_ids),
        "samples_with_errors_or_warnings": len(error_records),
        "perfect_samples": len(sample_ids) - len(error_records),
        "total_field_errors": total_field_errors,
        "error_type_breakdown": summary_counts,
        "error_records": error_records,
    }


def save_error_analysis_files(
    error_report: Dict[str, Any],
    output_dir: Union[str, Path],
    run_tag: str,
) -> Dict[str, Path]:
    """Lưu báo cáo phân tích lỗi chi tiết ra file Markdown (.md) và JSON Lines (.jsonl)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"error_analysis_{run_tag}.md"
    jsonl_path = out / f"error_analysis_{run_tag}.jsonl"

    # 1. Ghi file JSONL (mỗi dòng 1 ca lỗi đầy đủ)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in error_report["error_records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 2. Ghi file Markdown
    stamp = error_report["verification_stamp"]
    breakdown = error_report["error_type_breakdown"]
    total_err = max(1, error_report["total_field_errors"])

    lines: List[str] = [
        f"# BÁO CÁO PHÂN TÍCH LỖI MINH BẠCH (ERROR ANALYSIS) — `{run_tag}`",
        "",
        f"> **{stamp}**",
        "",
        "## 1. Thống kê Tổng quan Nguyên nhân Lỗi (Root-Cause Breakdown)",
        "",
        f"- **Tổng số hóa đơn đánh giá**: `{error_report['total_samples']}`",
        f"- **Số hóa đơn khớp tuyệt đối & không cảnh báo**: `{error_report['perfect_samples']}`",
        f"- **Số hóa đơn có lỗi trích xuất hoặc cảnh báo toán học**: `{error_report['samples_with_errors_or_warnings']}`",
        f"- **Tổng số trường bị lỗi (`total_field_errors`)**: `{error_report['total_field_errors']}`",
        "",
        "| Mã phân loại lỗi (Error Type) | Số lượng trường lỗi | Tỷ trọng (%) | Ý nghĩa Khoa học |",
        "| :--- | :---: | :---: | :--- |",
        f"| `OCR_CORRUPTION_ERROR` | {breakdown[OCR_CORRUPTION_ERROR]} | {breakdown[OCR_CORRUPTION_ERROR]/total_err*100:.1f}% | Nhãn chuẩn bị mất/biến dạng từ tầng nhận dạng OCR |",
        f"| `EXTRACTOR_MISS_ERROR` | {breakdown[EXTRACTOR_MISS_ERROR]} | {breakdown[EXTRACTOR_MISS_ERROR]/total_err*100:.1f}% | Có văn bản trong OCR nhưng Extractor bỏ sót (`null`) |",
        f"| `EXTRACTOR_WRONG_VALUE` | {breakdown[EXTRACTOR_WRONG_VALUE]} | {breakdown[EXTRACTOR_WRONG_VALUE]/total_err*100:.1f}% | Extractor lấy nhầm dòng/giá trị khác trên hóa đơn |",
        f"| `HALLUCINATION_ERROR` | {breakdown[HALLUCINATION_ERROR]} | {breakdown[HALLUCINATION_ERROR]/total_err*100:.1f}% | Extractor tự bịa giá trị không có trong văn bản OCR |",
        "",
        "## 2. Chi tiết Từng Hóa đơn Có Lỗi / Cảnh báo",
        "",
    ]

    if not error_report["error_records"]:
        lines.append("*Không có hóa đơn nào bị lỗi trong lần chạy này.*")
    else:
        for idx, rec in enumerate(error_report["error_records"], start=1):
            lines.extend(
                [
                    f"### {idx}. Hóa đơn `{rec['sample_id']}` (Mode: `{rec['adaptation_mode']}`, Retry: `{rec['retry_count']}`)",
                    "",
                    "**Danh sách trường bị sai (`field_errors`):**",
                ]
            )
            if not rec["field_errors"]:
                lines.append("- *(Không có lỗi trích xuất trường so với Ground-Truth, chỉ có cảnh báo kiểm định toán học trên hóa đơn gốc)*")
            else:
                for fe in rec["field_errors"]:
                    lines.append(
                        f"- **`{fe['field_name']}`** → [`{fe['error_type']}`]: "
                        f"GT = `{fe['gt_value']}` | Pred = `{fe['pred_value']}` — *{fe['explanation']}*"
                    )

            if rec["validation_issues"]:
                lines.append("")
                lines.append("**Cảnh báo kiểm định (`validation_issues`):**")
                for vi in rec["validation_issues"]:
                    lines.append(f"- `[{vi['severity']}] {vi['code']}`: {vi['message']}")

            lines.extend(
                [
                    "",
                    "<details>",
                    "<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>",
                    "",
                    "**`raw_ocr_text`**:",
                    "```text",
                    rec["raw_ocr_text"],
                    "```",
                    "**`ground_truth_json`**:",
                    "```json",
                    json.dumps(rec["ground_truth_json"], ensure_ascii=False, indent=2),
                    "```",
                    "**`predicted_json`**:",
                    "```json",
                    json.dumps(rec["predicted_json"], ensure_ascii=False, indent=2),
                    "```",
                    "</details>",
                    "",
                ]
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"markdown": md_path, "jsonl": jsonl_path}

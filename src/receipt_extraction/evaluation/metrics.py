"""Khung đánh giá định lượng toàn diện cho bài toán trích xuất thông tin hóa đơn OCR-LLM.

Bao phủ Mục 6.6 của Đề cương NCKH và các giả thuyết từ bài báo định hướng
"Nested Learning: The Illusion of Deep Learning Architecture" (Behrouz et al., NeurIPS 2025):
- Nhóm 1: Chất lượng OCR (CER, WER qua jiwer với chuẩn hóa Unicode NFC).
- Nhóm 2: Trích xuất Trường đơn (4 trường MC-OCR 2021 & 12 trường mở rộng, Exact-Match P/R/F1, Field CER, Bootstrap 95% CI).
- Nhóm 3: Danh sách Mặt hàng (Line-level F1 và Cell-level F1 qua ghép cặp tối ưu).
- Nhóm 4: Độ tin cậy, Ảo giác & Hiệu quả LSS (Hallucination rate, Total cost math error rate, Exact receipt accuracy, Flagged rate).
- Nhóm 5: Chi phí & Hiệu năng (Latency mean/p95, Input/Output tokens phục vụ so sánh f=inf vs f_low).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jiwer
import numpy as np

from receipt_extraction.extraction.regex_baseline import parse_currency
from receipt_extraction.schemas.receipt import (
    ExtractionResult,
    ReceiptData,
    ReceiptItem,
)
from receipt_extraction.validation.receipt_validator import (
    normalize_text_for_comparison,
    validate_arithmetic_consistency,
)

SCALAR_11_FIELDS: Tuple[str, ...] = (
    "seller_name",
    "seller_address",
    "seller_phone",
    "tax_id",
    "invoice_number",
    "timestamp",
    "subtotal",
    "discount",
    "vat_amount",
    "total_cost",
    "payment_method",
)

ALL_12_FIELDS: Tuple[str, ...] = SCALAR_11_FIELDS + ("items",)

MC_OCR_4_FIELDS: Tuple[str, ...] = (
    "SELLER",
    "SELLER_ADDRESS",
    "TIMESTAMP",
    "TOTAL_COST",
)

CURRENCY_FIELDS = {"subtotal", "discount", "vat_amount", "total_cost", "TOTAL_COST"}
ID_PHONE_FIELDS = {"seller_phone", "tax_id", "invoice_number"}
TIMESTAMP_FIELDS = {"timestamp", "TIMESTAMP"}


def normalize_ocr_text(text: Optional[str]) -> str:
    """Chuẩn hóa văn bản OCR về Unicode dựng sẵn (NFC) và gộp khoảng trắng thừa."""
    if not text:
        return ""
    nfc = unicodedata.normalize("NFC", str(text))
    # Chuẩn hóa khoảng trắng liên tiếp nhưng giữ cấu trúc từ
    return re.sub(r"\s+", " ", nfc).strip()


def compute_ocr_metrics(
    pred_texts: Union[str, Sequence[str]],
    gt_texts: Union[str, Sequence[str]],
) -> Dict[str, float]:
    """Nhóm 1: Tính Character Error Rate (CER) và Word Error Rate (WER) bằng jiwer.

    Chuẩn hóa Unicode NFC và khoảng trắng trước khi tính toán.
    Hỗ trợ đầu vào là một cặp chuỗi đơn hoặc danh sách các chuỗi.
    """
    if isinstance(pred_texts, str) and isinstance(gt_texts, str):
        preds = [pred_texts]
        gts = [gt_texts]
    else:
        preds = list(pred_texts) if not isinstance(pred_texts, str) else [pred_texts]
        gts = list(gt_texts) if not isinstance(gt_texts, str) else [gt_texts]

    if len(preds) != len(gts):
        raise ValueError(f"Số lượng pred_texts ({len(preds)}) và gt_texts ({len(gts)}) phải bằng nhau.")

    norm_preds: List[str] = []
    norm_gts: List[str] = []
    for p, g in zip(preds, gts):
        np_str = normalize_ocr_text(p)
        ng_str = normalize_ocr_text(g)
        if ng_str:
            norm_preds.append(np_str)
            norm_gts.append(ng_str)

    if not norm_gts:
        return {"cer": 0.0, "wer": 0.0}

    cer_val = float(jiwer.cer(norm_gts, norm_preds))
    wer_val = float(jiwer.wer(norm_gts, norm_preds))
    return {
        "cer": round(cer_val, 6),
        "wer": round(wer_val, 6),
    }


def _normalize_timestamp(val: str) -> str:
    """Chuẩn hóa chuỗi ngày giờ về định dạng thống nhất để so sánh công bằng."""
    s = normalize_ocr_text(val).lower()
    m_vn = re.search(
        r"(?:ngày|ngay|date)?[^\d]{0,15}(\d{1,2})\s*(?:tháng|thang|month)[^\d]{0,15}(\d{1,2})\s*(?:năm|nam|year)[^\d]{0,15}(\d{4})",
        s,
    )
    if m_vn:
        d_vn, mo_vn, y_vn = int(m_vn.group(1)), int(m_vn.group(2)), m_vn.group(3)
        return f"{d_vn:02d}/{mo_vn:02d}/{y_vn}"
    s = s.replace("-", "/")
    # Chuẩn hóa ngày/tháng 1 chữ số thành 2 chữ số: 7/10/2026 -> 07/10/2026
    def _pad_date(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(y) == 2:
            y = "20" + y
        return f"{d:02d}/{mo:02d}/{y}"

    s = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", _pad_date, s)
    return s


def _normalize_item_for_field(item: ReceiptItem) -> str:
    """Chuẩn hóa một dòng mặt hàng thành chuỗi đại diện chuẩn."""
    name = normalize_ocr_text(item.item_name).lower()
    qty = f"{float(item.quantity or 0.0):.2f}"
    up = str(item.unit_price if item.unit_price is not None else "")
    tp = str(item.total_price if item.total_price is not None else "")
    return f"{name}|{qty}|{up}|{tp}"


def normalize_field_value(field_name: str, value: Any) -> Optional[str]:
    """Chuẩn hóa giá trị của từng loại trường trước khi so khớp Exact-Match hoặc tính CER."""
    if value is None:
        return None

    if field_name == "items":
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            return None
        normalized_items = []
        for it in value:
            if isinstance(it, ReceiptItem):
                normalized_items.append(_normalize_item_for_field(it))
            elif isinstance(it, dict):
                normalized_items.append(_normalize_item_for_field(ReceiptItem(**it)))
        return " ; ".join(sorted(normalized_items)) if normalized_items else None

    if field_name in CURRENCY_FIELDS:
        if isinstance(value, (int, float)):
            return str(int(round(value)))
        parsed = parse_currency(str(value))
        return str(parsed) if parsed is not None else normalize_ocr_text(str(value)).lower()

    str_val = normalize_ocr_text(str(value))
    if not str_val:
        return None

    if field_name in ID_PHONE_FIELDS:
        # Bỏ khoảng trắng, dấu chấm, dấu ngoặc đơn trong SĐT / Mã số thuế / Số HĐ
        cleaned = re.sub(r"[\s\.\(\)]+", "", str_val).lower()
        return cleaned if cleaned else None

    if field_name in TIMESTAMP_FIELDS:
        return _normalize_timestamp(str_val)

    # Các trường văn bản chung (seller_name, seller_address, payment_method)
    cleaned_text = re.sub(r"^[\s\.,:;!\-\_]+|[\s\.,:;!\-\_]+$", "", str_val.lower())
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text if cleaned_text else None


def _extract_field_dict(
    obj: Union[ReceiptData, Dict[str, Any]],
    mode: str,
) -> Dict[str, Optional[str]]:
    """Chuyển đổi ReceiptData hoặc Dict sang từ điển các trường đã chuẩn hóa theo chế độ."""
    if isinstance(obj, dict):
        if mode in ("mcocr_4_fields", "4_fields"):
            # Hỗ trợ cả dict đã ở dạng MC-OCR (SELLER...) hoặc dạng ReceiptData
            if any(k in obj for k in MC_OCR_4_FIELDS):
                return {
                    k: normalize_field_value(k, obj.get(k))
                    for k in MC_OCR_4_FIELDS
                }
            receipt_obj = ReceiptData(**obj)
            raw_mc = receipt_obj.to_mc_ocr_dict()
            return {k: normalize_field_value(k, v) for k, v in raw_mc.items()}
        receipt_obj = ReceiptData(**obj)
    else:
        receipt_obj = obj

    if mode in ("mcocr_4_fields", "4_fields"):
        raw_mc = receipt_obj.to_mc_ocr_dict()
        return {k: normalize_field_value(k, v) for k, v in raw_mc.items()}

    return {
        f: normalize_field_value(f, getattr(receipt_obj, f, None))
        for f in ALL_12_FIELDS
    }


def compute_field_metrics(
    predictions: Sequence[Union[ReceiptData, Dict[str, Any]]],
    ground_truths: Sequence[Union[ReceiptData, Dict[str, Any]]],
    mode: str = "12_fields",
) -> Dict[str, Any]:
    """Nhóm 2: Đánh giá trích xuất trường thông tin (KIE) ở chế độ 4 trường (MC-OCR) hoặc 12 trường.

    Tính toán:
    - Exact-Match Precision, Recall, F1 và Field CER cho từng trường.
    - Micro Precision / Recall / F1 tổng hợp trên toàn bộ các trường.
    - Macro Precision / Recall / F1 (trung bình trên các trường có mặt trong Ground-Truth).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Số lượng predictions ({len(predictions)}) và ground_truths ({len(ground_truths)}) không khớp."
        )

    target_fields = MC_OCR_4_FIELDS if mode in ("mcocr_4_fields", "4_fields") else ALL_12_FIELDS
    pred_dicts = [_extract_field_dict(p, mode) for p in predictions]
    gt_dicts = [_extract_field_dict(g, mode) for g in ground_truths]

    per_field: Dict[str, Dict[str, Any]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    active_f1s: List[float] = []
    active_precisions: List[float] = []
    active_recalls: List[float] = []
    all_field_cers: List[float] = []

    for field in target_fields:
        tp = 0
        fp = 0
        fn = 0
        support = 0
        cer_list: List[float] = []

        for p_map, g_map in zip(pred_dicts, gt_dicts):
            p_val = p_map.get(field)
            g_val = g_map.get(field)

            if g_val is not None:
                support += 1

            if p_val is not None and g_val is not None:
                if p_val == g_val:
                    tp += 1
                    cer_list.append(0.0)
                else:
                    fp += 1
                    fn += 1
                    cer_list.append(float(jiwer.cer(g_val, p_val)))
            elif p_val is not None and g_val is None:
                fp += 1
                cer_list.append(1.0)
            elif p_val is None and g_val is not None:
                fn += 1
                cer_list.append(1.0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if support == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if support == 0 else 0.0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        field_cer = float(np.mean(cer_list)) if cer_list else 0.0

        per_field[field] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "field_cer": round(field_cer, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if support > 0 or (tp + fp) > 0:
            active_precisions.append(precision)
            active_recalls.append(recall)
            active_f1s.append(f1)
            all_field_cers.append(field_cer)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_precision = float(np.mean(active_precisions)) if active_precisions else 0.0
    macro_recall = float(np.mean(active_recalls)) if active_recalls else 0.0
    macro_f1 = float(np.mean(active_f1s)) if active_f1s else 0.0
    mean_field_cer = float(np.mean(all_field_cers)) if all_field_cers else 0.0

    return {
        "mode": mode,
        "num_receipts": len(predictions),
        "per_field": per_field,
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "mean_field_cer": round(mean_field_cer, 6),
    }


def bootstrap_f1_ci(
    preds: Sequence[Union[ReceiptData, Dict[str, Any]]],
    gts: Sequence[Union[ReceiptData, Dict[str, Any]]],
    n_bootstraps: int = 1000,
    seed: int = 42,
    mode: str = "12_fields",
    metric_key: str = "micro_f1",
) -> Dict[str, Any]:
    """Tính khoảng tin cậy 95% (95% Confidence Interval) và mean ± std bằng Non-parametric Bootstrap."""
    if len(preds) != len(gts):
        raise ValueError("Số lượng preds và gts phải bằng nhau.")
    n = len(preds)
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "formatted": "0.0000 ± 0.0000 (95% CI: [0.0000, 0.0000])",
        }

    rng = np.random.default_rng(seed)
    scores: List[float] = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample_preds = [preds[i] for i in indices]
        sample_gts = [gts[i] for i in indices]
        res = compute_field_metrics(sample_preds, sample_gts, mode=mode)
        scores.append(float(res[metric_key]))

    arr = np.array(scores, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n_bootstraps > 1 else 0.0
    ci_lower = float(np.percentile(arr, 2.5))
    ci_upper = float(np.percentile(arr, 97.5))

    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "formatted": f"{mean_val:.4f} ± {std_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])",
    }


def _compare_item_cells(pred_item: ReceiptItem, gt_item: ReceiptItem) -> int:
    """Đếm số ô khớp (tối đa 4 ô: item_name, quantity, unit_price, total_price) giữa 2 dòng mặt hàng."""
    matched_cells = 0

    pred_name = normalize_field_value("item_name", pred_item.item_name)
    gt_name = normalize_field_value("item_name", gt_item.item_name)
    if pred_name is not None and gt_name is not None and pred_name == gt_name:
        matched_cells += 1

    pred_qty = float(pred_item.quantity) if pred_item.quantity is not None else None
    gt_qty = float(gt_item.quantity) if gt_item.quantity is not None else None
    if pred_qty is not None and gt_qty is not None and abs(pred_qty - gt_qty) < 1e-2:
        matched_cells += 1

    if pred_item.unit_price == gt_item.unit_price and pred_item.unit_price is not None:
        matched_cells += 1

    if pred_item.total_price == gt_item.total_price and pred_item.total_price is not None:
        matched_cells += 1

    return matched_cells


def compute_item_metrics(
    pred_items_list: Sequence[Union[ReceiptData, Sequence[ReceiptItem]]],
    gt_items_list: Sequence[Union[ReceiptData, Sequence[ReceiptItem]]],
) -> Dict[str, float]:
    """Nhóm 3: Đánh giá trích xuất danh sách mặt hàng ở mức dòng (line_f1) và mức ô (cell_f1).

    - line_f1: Khớp đồng thời cả 4 thuộc tính (item_name đã chuẩn hóa, quantity, unit_price, total_price).
    - cell_f1: Độ chính xác mức ô (4 ô/dòng) thông qua ghép cặp tối ưu các dòng mặt hàng.
    """
    if len(pred_items_list) != len(gt_items_list):
        raise ValueError("Số lượng hóa đơn dự đoán và nhãn chuẩn trong compute_item_metrics phải bằng nhau.")

    total_pred_lines = 0
    total_gt_lines = 0
    matched_full_lines = 0

    total_pred_cells = 0
    total_gt_cells = 0
    matched_total_cells = 0

    for p_entry, g_entry in zip(pred_items_list, gt_items_list):
        p_items: Sequence[ReceiptItem] = (
            p_entry.items if isinstance(p_entry, ReceiptData) else p_entry
        )
        g_items: Sequence[ReceiptItem] = (
            g_entry.items if isinstance(g_entry, ReceiptData) else g_entry
        )

        total_pred_lines += len(p_items)
        total_gt_lines += len(g_items)
        total_pred_cells += 4 * len(p_items)
        total_gt_cells += 4 * len(g_items)

        if not p_items or not g_items:
            continue

        # Tính ma trận điểm tương đồng số ô khớp (0..4) cho mọi cặp (pred_i, gt_j)
        pair_scores: List[Tuple[int, int, int]] = []
        for i, p_it in enumerate(p_items):
            for j, g_it in enumerate(g_items):
                score = _compare_item_cells(p_it, g_it)
                if score > 0:
                    pair_scores.append((score, i, j))

        # Ghép cặp tham lam theo số ô khớp giảm dần
        pair_scores.sort(key=lambda x: x[0], reverse=True)
        used_p = set()
        used_g = set()

        for score, i, j in pair_scores:
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
                matched_total_cells += score
                if score == 4:
                    matched_full_lines += 1

    line_precision = (
        matched_full_lines / total_pred_lines
        if total_pred_lines > 0
        else (1.0 if total_gt_lines == 0 else 0.0)
    )
    line_recall = (
        matched_full_lines / total_gt_lines
        if total_gt_lines > 0
        else (1.0 if total_pred_lines == 0 else 0.0)
    )
    line_f1 = (
        2 * line_precision * line_recall / (line_precision + line_recall)
        if (line_precision + line_recall) > 0
        else 0.0
    )

    cell_precision = (
        matched_total_cells / total_pred_cells
        if total_pred_cells > 0
        else (1.0 if total_gt_cells == 0 else 0.0)
    )
    cell_recall = (
        matched_total_cells / total_gt_cells
        if total_gt_cells > 0
        else (1.0 if total_pred_cells == 0 else 0.0)
    )
    cell_f1 = (
        2 * cell_precision * cell_recall / (cell_precision + cell_recall)
        if (cell_precision + cell_recall) > 0
        else 0.0
    )

    return {
        "line_precision": round(line_precision, 6),
        "line_recall": round(line_recall, 6),
        "line_f1": round(line_f1, 6),
        "cell_precision": round(cell_precision, 6),
        "cell_recall": round(cell_recall, 6),
        "cell_f1": round(cell_f1, 6),
        "matched_lines": matched_full_lines,
        "total_pred_lines": total_pred_lines,
        "total_gt_lines": total_gt_lines,
    }


def has_evidence_in_raw_ocr(field_name: str, field_value: Any, raw_ocr_text: str) -> bool:
    """Kiểm tra một trường có giá trị khác None có đối chứng thực tế trong raw_ocr_text hay không."""
    if field_value is None:
        return True
    if not raw_ocr_text or not raw_ocr_text.strip():
        return False

    norm_ocr = normalize_text_for_comparison(raw_ocr_text)

    if field_name in CURRENCY_FIELDS:
        val_int = int(field_value) if isinstance(field_value, (int, float)) else parse_currency(str(field_value))
        if val_int is None:
            return False
        abs_str = str(abs(val_int))
        if abs_str in norm_ocr:
            return True
        if abs(val_int) >= 1000 and f"{abs(val_int) // 1000}k" in norm_ocr:
            return True
        return False

    norm_val = normalize_text_for_comparison(str(field_value))
    if not norm_val:
        return True
    if norm_val in norm_ocr:
        return True

    # Đối với các trường văn bản dài (như seller_name, seller_address) hoặc chuẩn hóa payment_method
    # Kiểm tra xem ít nhất 60% các từ có độ dài >= 2 có xuất hiện trong văn bản OCR (không dấu hoặc có dấu)
    tokens = [t for t in re.split(r"\s+", str(field_value).strip()) if len(t) >= 2]
    if tokens:
        matched_tokens = sum(
            1 for t in tokens if normalize_text_for_comparison(t) in norm_ocr
        )
        if matched_tokens / len(tokens) >= 0.6:
            return True

    return False


def compute_reliability_metrics(
    results: Sequence[ExtractionResult],
    ground_truths: Optional[Sequence[ReceiptData]] = None,
    mode: str = "12_fields",
) -> Dict[str, Any]:
    """Nhóm 4: Đánh giá Độ tin cậy, Ảo giác (Hallucination) & Hiệu quả phản hồi LSS.

    Các chỉ số:
    - hallucination_rate: Tỷ lệ các trường có giá trị khác None nhưng không có đối chứng trong raw_ocr_text.
    - total_cost_math_error_rate: Tỷ lệ hóa đơn bị sai trường total_cost hoặc vi phạm ràng buộc số học
      Subtotal - Discount + VAT = Total (phục vụ chứng minh Mục tiêu MT4).
    - exact_receipt_acc: Tỷ lệ hóa đơn trích xuất đúng hoàn toàn mọi trường.
    - flagged_rate: Tỷ lệ hóa đơn bị gắn cờ needs_human_review.
    """
    n = len(results)
    if n == 0:
        return {
            "hallucination_rate": 0.0,
            "hallucinated_fields_count": 0,
            "total_extracted_fields_count": 0,
            "total_cost_math_error_rate": 0.0,
            "exact_receipt_acc": 0.0,
            "flagged_rate": 0.0,
        }

    if ground_truths is not None and len(ground_truths) != n:
        raise ValueError("Số lượng results và ground_truths phải bằng nhau.")

    total_non_null_fields = 0
    hallucinated_fields = 0
    math_or_total_error_receipts = 0
    exact_match_receipts = 0
    flagged_receipts = 0

    for idx, res in enumerate(results):
        data = res.data
        raw_ocr = res.metadata.raw_ocr_text or ""

        # 1. Đếm số trường non-null và số trường bị ảo giác (không có đối chứng trong raw_ocr_text)
        for field in SCALAR_11_FIELDS:
            val = getattr(data, field, None)
            if val is not None and str(val).strip() != "":
                total_non_null_fields += 1
                if not has_evidence_in_raw_ocr(field, val, raw_ocr):
                    hallucinated_fields += 1

        # 2. Kiểm tra lỗi toán học hoặc sai lệch trường total_cost
        arith_issues = validate_arithmetic_consistency(data)
        has_math_violation = any(
            iss.code in ("ARITHMETIC_MISMATCH_TOTAL", "ITEMS_SUM_TOTAL_MISMATCH", "ITEM_MATH_MISMATCH")
            for iss in arith_issues
        )
        missing_or_invalid_total = data.total_cost is None or data.total_cost <= 0
        wrong_total_vs_gt = False
        if ground_truths is not None:
            gt_total = ground_truths[idx].total_cost
            if gt_total is not None and data.total_cost != gt_total:
                wrong_total_vs_gt = True

        if has_math_violation or missing_or_invalid_total or wrong_total_vs_gt:
            math_or_total_error_receipts += 1

        # 3. Kiểm tra Exact Receipt Accuracy
        if ground_truths is not None:
            p_dict = _extract_field_dict(data, mode=mode)
            g_dict = _extract_field_dict(ground_truths[idx], mode=mode)
            if p_dict == g_dict:
                exact_match_receipts += 1
        else:
            if res.is_valid and not arith_issues and hallucinated_fields == 0:
                exact_match_receipts += 1

        # 4. Kiểm tra tỷ lệ gắn cờ cần người duyệt (needs_human_review)
        if res.metadata.needs_human_review:
            flagged_receipts += 1

    hallucination_rate = (
        hallucinated_fields / total_non_null_fields if total_non_null_fields > 0 else 0.0
    )

    return {
        "hallucination_rate": round(hallucination_rate, 6),
        "hallucinated_fields_count": hallucinated_fields,
        "total_extracted_fields_count": total_non_null_fields,
        "total_cost_math_error_rate": round(math_or_total_error_receipts / n, 6),
        "exact_receipt_acc": round(exact_match_receipts / n, 6),
        "flagged_rate": round(flagged_receipts / n, 6),
    }


def compute_cost_metrics(results: Sequence[ExtractionResult]) -> Dict[str, float]:
    """Nhóm 5: Đánh giá Chi phí & Hiệu năng (Latency mean/p95, Input/Output tokens).

    Phục vụ phân tích đánh đổi chi phí - độ chính xác giữa thích nghi ngữ cảnh dài (f=inf)
    và thích nghi trọng số LoRA (f_low) theo khung lý thuyết Nested Learning.
    """
    if not results:
        return {
            "mean_processing_time_sec": 0.0,
            "p95_processing_time_sec": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "mean_input_tokens": 0.0,
            "mean_output_tokens": 0.0,
        }

    latencies = np.array(
        [float(r.metadata.processing_time_sec or 0.0) for r in results],
        dtype=np.float64,
    )
    in_tokens = [int(r.metadata.input_tokens or 0) for r in results]
    out_tokens = [int(r.metadata.output_tokens or 0) for r in results]

    return {
        "mean_processing_time_sec": round(float(np.mean(latencies)), 6),
        "p95_processing_time_sec": round(float(np.percentile(latencies, 95)), 6),
        "total_input_tokens": int(sum(in_tokens)),
        "total_output_tokens": int(sum(out_tokens)),
        "mean_input_tokens": round(float(np.mean(in_tokens)), 2),
        "mean_output_tokens": round(float(np.mean(out_tokens)), 2),
    }

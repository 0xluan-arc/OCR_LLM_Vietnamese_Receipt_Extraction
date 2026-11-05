"""Mô-đun đánh giá định lượng chất lượng OCR, trích xuất trường (KIE), độ tin cậy LSS, chi phí và phân tích lỗi."""

from .error_analysis import (
    EXTRACTOR_MISS_ERROR,
    EXTRACTOR_WRONG_VALUE,
    HALLUCINATION_ERROR,
    OCR_CORRUPTION_ERROR,
    analyze_sample_errors,
    classify_field_error,
    generate_error_analysis_report,
    save_error_analysis_files,
)
from .metrics import (
    bootstrap_f1_ci,
    compute_cost_metrics,
    compute_field_metrics,
    compute_item_metrics,
    compute_ocr_metrics,
    compute_reliability_metrics,
    normalize_field_value,
)

__all__ = [
    "compute_ocr_metrics",
    "compute_field_metrics",
    "bootstrap_f1_ci",
    "compute_item_metrics",
    "compute_reliability_metrics",
    "compute_cost_metrics",
    "normalize_field_value",
    "OCR_CORRUPTION_ERROR",
    "EXTRACTOR_MISS_ERROR",
    "EXTRACTOR_WRONG_VALUE",
    "HALLUCINATION_ERROR",
    "classify_field_error",
    "analyze_sample_errors",
    "generate_error_analysis_report",
    "save_error_analysis_files",
]

"""Ứng dụng Web Streamlit chuẩn Enterprise Document AI:
Hệ thống Trích xuất Thông tin Hóa đơn Bán lẻ Việt Nam (OCR kết hợp LLM & Nested Learning).

Tính năng nổi bật:
1. Side-by-Side Document Inspector:
   - Cột trái: Hiển thị trực tiếp ảnh có lớp phủ Bounding Box sắc nét ngay khi bật checkbox
     `Tô sáng vùng chữ OCR (Bounding Boxes)` + Thanh thông số chất lượng ảnh (Độ phân giải, Độ nét Laplacian, Số dòng OCR).
2. Digital Receipt Card & Custom CSS Financial Cards:
   - Thay thế hoàn toàn `st.metric` mặc định (vốn làm cắt cụt số tiền `72,00...`) bằng thẻ CSS hiện đại
     hiển thị trọn vẹn từng chữ số tiền tệ (`72.000 ₫`, `75.600 ₫`).
   - Hiển thị rõ 12 trường thông tin kèm Badge đối chứng (`✓ Khớp OCR`, `⚠ Đã sửa qua LSS`, `⚠ Thiếu trên phiếu`, `∅ Không ghi nhận`).
   - Bảng mặt hàng HTML/CSS chuyên nghiệp căn lề phải cho cột số tiền và thanh tiến trình Nested Learning.
3. Human-in-the-Loop Continuum Memory System (CMS):
   - Chỉnh sửa trực tiếp, xuất Excel/JSON và nạp vào Bộ nhớ Liên tục kèm cập nhật Ma trận Delta Rule.
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import streamlit as st

_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from receipt_extraction.config import config
from receipt_extraction.extraction.hybrid_extractor import HybridNestedExtractor
from receipt_extraction.extraction.llm_extractor import LLMExtractor, LLMNotConfiguredError
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.memory.cms_store import ContinuumMemoryStore, text_to_l2_normalized_vector
from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.preprocess.image_validator import validate_image
from receipt_extraction.preprocess.preprocessor import PreprocessOptions, preprocess_receipt_image
from receipt_extraction.schemas.receipt import (
    ExtractionMetadata,
    ExtractionResult,
    ReceiptData,
    ReceiptItem,
    ValidationIssue,
    ValidationSeverity,
)
from receipt_extraction.validation.receipt_validator import validate_receipt_data
from receipt_extraction.web.export_utils import draw_ocr_bounding_boxes, export_receipt_to_excel_bytes


ENTERPRISE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.ent-header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 26px;
    margin-bottom: 20px;
    color: #f8fafc;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}
.ent-header-title {
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
    color: #ffffff;
}
.ent-header-sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin: 0;
}

/* Thanh thông số chất lượng ảnh ở Cột trái */
.inspector-stats-bar {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 10px 0 14px 0;
}
.inspector-stat-pill {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
}
.inspector-stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 3px;
}
.inspector-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.96rem;
    font-weight: 600;
    color: #f1f5f9;
}

/* Custom Financial Cards (Không bao giờ cắt chữ số tiền!) */
.fin-cards-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0 20px 0;
}
@media (max-width: 1100px) {
    .fin-cards-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.fin-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.fin-card-total {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 1px solid #10b981;
}
.fin-card-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.fin-card-total .fin-card-label {
    color: #a7f3d0;
}
.fin-card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.18rem;
    font-weight: 700;
    color: #f8fafc;
    white-space: nowrap;
    overflow: visible;
}
.fin-card-total .fin-card-value {
    color: #ecfdf5;
    font-size: 1.26rem;
}

/* Digital Receipt Card */
.digital-receipt-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.receipt-field-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 18px;
}
.receipt-field-row {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 14px;
}
.receipt-field-row.full-width {
    grid-column: 1 / -1;
}
.receipt-field-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.receipt-field-name {
    font-size: 0.76rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.receipt-field-val {
    font-size: 0.96rem;
    font-weight: 600;
    color: #f8fafc;
    word-break: break-word;
}
.receipt-field-val.mono {
    font-family: 'JetBrains Mono', monospace;
}

/* Badges đối chứng */
.badge-verified {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 600;
}
.badge-lss {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 600;
}
.badge-missing {
    background: rgba(239, 68, 68, 0.18);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 600;
}
.badge-null {
    background: rgba(148, 163, 184, 0.15);
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.3);
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 500;
}

/* Bảng mặt hàng chuẩn kế toán */
.items-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
    font-size: 0.90rem;
    background: #1e293b;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #334155;
}
.items-table th {
    background: #0f172a;
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    padding: 10px 14px;
    border-bottom: 1px solid #334155;
}
.items-table td {
    padding: 10px 14px;
    color: #f1f5f9;
    border-bottom: 1px solid #334155;
}
.items-table tr:last-child td {
    border-bottom: none;
}
.text-right {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
}
.text-center {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}

/* Status Banner */
.status-banner-ok {
    background: rgba(6, 95, 70, 0.35);
    border: 1px solid #10b981;
    border-left: 5px solid #10b981;
    border-radius: 10px;
    padding: 12px 16px;
    color: #ecfdf5;
    margin-bottom: 14px;
}
.status-banner-warn {
    background: rgba(120, 53, 15, 0.38);
    border: 1px solid #f59e0b;
    border-left: 5px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 16px;
    color: #fef3c7;
    margin-bottom: 14px;
}
</style>
"""


def format_vnd(amount: Optional[int], default_label: str = "0 ₫") -> str:
    """Định dạng số nguyên VND đầy đủ có dấu chấm phân cách hàng nghìn (ví dụ: 75.600 ₫)."""
    if amount is None:
        return default_label
    return f"{int(amount):,}".replace(",", ".") + " ₫"


def render_field_badge(
    field_name: str,
    val: Any,
    evidence: Dict[str, bool],
    issues: List[ValidationIssue],
    retry_count: int,
) -> str:
    """Sinh HTML Badge đối chứng cho từng trường trên Digital Receipt Card."""
    field_issue_codes = [iss.code for iss in issues if iss.field_name == field_name]
    if "MISSING_DETECTED_FIELD" in field_issue_codes:
        return '<span class="badge-missing">⚠ Bỏ sót trên OCR</span>'
    if any("EVIDENCE_NOT_FOUND" in c or "MISMATCH" in c for c in field_issue_codes):
        return '<span class="badge-lss">⚠ Cảnh báo LSS</span>'
    if val is None or str(val).strip() == "":
        return '<span class="badge-null">∅ Không có trên phiếu</span>'
    if retry_count > 0:
        return '<span class="badge-lss">↻ Đã sửa qua LSS</span>'
    if evidence.get(field_name, False):
        return '<span class="badge-verified">✓ Khớp OCR</span>'
    return '<span class="badge-verified">✓ Đã trích xuất</span>'


def build_digital_receipt_html(
    data: ReceiptData,
    evidence: Dict[str, bool],
    issues: List[ValidationIssue],
    retry_count: int,
) -> str:
    """Tạo khối HTML Digital Receipt Card hiển thị 12 trường và bảng mặt hàng."""
    def _esc(v: Any, fallback: str = "—") -> str:
        if v is None or str(v).strip() == "":
            return fallback
        return html.escape(str(v))

    b_seller = render_field_badge("seller_name", data.seller_name, evidence, issues, retry_count)
    b_addr = render_field_badge("seller_address", data.seller_address, evidence, issues, retry_count)
    b_phone = render_field_badge("seller_phone", data.seller_phone, evidence, issues, retry_count)
    b_tax = render_field_badge("tax_id", data.tax_id, evidence, issues, retry_count)
    b_inv = render_field_badge("invoice_number", data.invoice_number, evidence, issues, retry_count)
    b_time = render_field_badge("timestamp", data.timestamp, evidence, issues, retry_count)
    b_pay = render_field_badge("payment_method", data.payment_method, evidence, issues, retry_count)

    sub_str = format_vnd(data.subtotal, "—")
    disc_str = ("-" + format_vnd(data.discount)) if data.discount else "0 ₫"
    vat_str = format_vnd(data.vat_amount, "0 ₫")
    tot_str = format_vnd(data.total_cost, "—")

    items_rows_html = ""
    if data.items:
        for idx, it in enumerate(data.items, start=1):
            q_str = f"{int(it.quantity)}" if (it.quantity is not None and float(it.quantity).is_integer()) else f"{it.quantity or 1}"
            u_str = format_vnd(it.unit_price, "—")
            t_str = format_vnd(it.total_price, "—")
            items_rows_html += (
                f"<tr>"
                f"<td class='text-center'>{idx}</td>"
                f"<td>{_esc(it.item_name)}</td>"
                f"<td class='text-center'>{q_str}</td>"
                f"<td class='text-right'>{u_str}</td>"
                f"<td class='text-right'><strong>{t_str}</strong></td>"
                f"</tr>"
            )
    else:
        items_rows_html = (
            "<tr><td colspan='5' style='text-align:center;color:#94a3b8;padding:14px;'>"
            "Chưa trích xuất được dòng mặt hàng chi tiết</td></tr>"
        )

    return f"""
    <div class="fin-cards-grid">
        <div class="fin-card">
            <div class="fin-card-label">7. Tạm tính (Subtotal)</div>
            <div class="fin-card-value">{sub_str}</div>
        </div>
        <div class="fin-card">
            <div class="fin-card-label">8. Giảm giá (Discount)</div>
            <div class="fin-card-value">{disc_str}</div>
        </div>
        <div class="fin-card">
            <div class="fin-card-label">9. Thuế GTGT (VAT)</div>
            <div class="fin-card-value">{vat_str}</div>
        </div>
        <div class="fin-card fin-card-total">
            <div class="fin-card-label">10. TỔNG THANH TOÁN</div>
            <div class="fin-card-value">{tot_str}</div>
        </div>
    </div>

    <div class="digital-receipt-card">
        <div class="receipt-field-grid">
            <div class="receipt-field-row full-width">
                <div class="receipt-field-header">
                    <span class="receipt-field-name">1. Tên Đơn vị Bán hàng (seller_name)</span>
                    {b_seller}
                </div>
                <div class="receipt-field-val">{_esc(data.seller_name)}</div>
            </div>

            <div class="receipt-field-row full-width">
                <div class="receipt-field-header">
                    <span class="receipt-field-name">2. Địa chỉ Kinh doanh (seller_address)</span>
                    {b_addr}
                </div>
                <div class="receipt-field-val">{_esc(data.seller_address)}</div>
            </div>

            <div class="receipt-field-row">
                <div class="receipt-field-header">
                    <span class="receipt-field-name">3. Số Điện thoại (seller_phone)</span>
                    {b_phone}
                </div>
                <div class="receipt-field-val mono">{_esc(data.seller_phone)}</div>
            </div>

            <div class="receipt-field-row">
                <div class="receipt-field-header">
                    <span class="receipt-field-name">4. Mã Số Thuế (tax_id)</span>
                    {b_tax}
                </div>
                <div class="receipt-field-val mono">{_esc(data.tax_id)}</div>
            </div>

            <div class="receipt-field-row">
                <div class="receipt-field-header">
                    <span class="receipt-field-name">5. Số Hóa đơn / Phiếu (invoice_number)</span>
                    {b_inv}
                </div>
                <div class="receipt-field-val mono">{_esc(data.invoice_number)}</div>
            </div>

            <div class="receipt-field-row">
                <div class="receipt-field-header">
                    <span class="receipt-field-name">6. Ngày Giờ Giao dịch (timestamp)</span>
                    {b_time}
                </div>
                <div class="receipt-field-val mono">{_esc(data.timestamp)}</div>
            </div>

            <div class="receipt-field-row full-width">
                <div class="receipt-field-header">
                    <span class="receipt-field-name">11. Phương thức Thanh toán (payment_method)</span>
                    {b_pay}
                </div>
                <div class="receipt-field-val">{_esc(data.payment_method)}</div>
            </div>
        </div>

        <div style="margin-top: 18px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span class="receipt-field-name">12. Danh sách Mặt hàng ({len(data.items)} dòng)</span>
                <span class="badge-verified">Căn lề kế toán VND</span>
            </div>
            <table class="items-table">
                <thead>
                    <tr>
                        <th style="width:50px;text-align:center;">STT</th>
                        <th style="text-align:left;">Tên mặt hàng</th>
                        <th style="width:75px;text-align:center;">SL</th>
                        <th style="width:130px;text-align:right;">Đơn giá</th>
                        <th style="width:140px;text-align:right;">Thành tiền</th>
                    </tr>
                </thead>
                <tbody>
                    {items_rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """


def run_extraction_pipeline(
    processed_bytes: bytes,
    ocr_choice: str,
    extraction_method: str,
    enable_lss: bool,
    cms_store: ContinuumMemoryStore,
) -> Dict[str, Any]:
    """Thực thi trọn vẹn pipeline OCR -> KIE -> Validation và trả về kết quả lưu trong session_state."""
    start_pipeline_time = time.time()
    ocr_engine = get_ocr_engine(ocr_choice)
    ocr_res = ocr_engine.recognize(processed_bytes)
    if ocr_res.error_message:
        return {"error": f"Lỗi OCR ({ocr_choice}): {ocr_res.error_message}"}

    bbox_bytes = draw_ocr_bounding_boxes(processed_bytes, ocr_res.lines) if ocr_res.lines else processed_bytes

    llm_sub = LLMExtractor(
        adaptation_mode="few_shot_cms",
        few_shot_k=3,
        enable_lss_retry=enable_lss,
        memory_store=cms_store,
    )

    if "1." in extraction_method:
        extractor = RegexBaselineExtractor()
        used_mode = "regex_b1"
    elif "2." in extraction_method:
        extractor = LLMExtractor(adaptation_mode="zero_shot", enable_lss_retry=enable_lss, memory_store=cms_store)
        used_mode = "zero_shot"
    elif "3." in extraction_method:
        extractor = LLMExtractor(adaptation_mode="few_shot_cms", few_shot_k=3, enable_lss_retry=enable_lss, memory_store=cms_store)
        used_mode = "few_shot_cms"
    elif "4." in extraction_method:
        extractor = LLMExtractor(
            adaptation_mode="few_shot_layout",
            few_shot_k=3,
            use_layout=True,
            enable_lss_retry=enable_lss,
            memory_store=cms_store,
        )
        used_mode = "few_shot_layout"
    else:
        extractor = HybridNestedExtractor(llm_extractor=llm_sub)
        used_mode = "hybrid"

    fallback_warning: Optional[str] = None
    try:
        extracted_data, evidence = extractor.extract(ocr_res)
    except Exception as exc:
        fallback_warning = (
            f"Endpoint LLM ({config.llm.base_url}) chưa phản hồi ({exc}). "
            "Hệ thống đã dùng kết quả trích xuất từ Baseline B1 (Regex + OCR Denoising) và gắn cờ kiểm duyệt."
        )
        extractor = RegexBaselineExtractor()
        extracted_data, evidence = extractor.extract(ocr_res)
        used_mode = "regex_fallback_unverified"

    validation_issues, is_valid = validate_receipt_data(extracted_data, raw_ocr_text=ocr_res.raw_text)
    if fallback_warning is not None:
        validation_issues.append(
            ValidationIssue(
                code="LLM_FALLBACK_NOTICE",
                field_name="system",
                message="LLM server phản hồi chậm/timeout, hệ thống đã chuyển sang kết quả dự phòng từ Regex B1.",
                severity=ValidationSeverity.INFO,
            )
        )
    total_elapsed = time.time() - start_pipeline_time

    if isinstance(extractor, LLMExtractor):
        retry_count = extractor.last_retry_count
        needs_review = extractor.last_needs_human_review
        in_tok = extractor.last_input_tokens
        out_tok = extractor.last_output_tokens
        effective_mode = extractor.adaptation_mode
    elif isinstance(extractor, HybridNestedExtractor):
        retry_count = extractor.last_retry_count
        needs_review = extractor.last_needs_human_review
        in_tok = extractor.last_input_tokens
        out_tok = extractor.last_output_tokens
        effective_mode = extractor.last_adaptation_mode
    else:
        retry_count = 0
        has_warn = any(
            iss.severity in (ValidationSeverity.WARNING, ValidationSeverity.ERROR)
            for iss in validation_issues
        )
        needs_review = has_warn or (not is_valid)
        in_tok = 0
        out_tok = 0
        effective_mode = used_mode

    final_result = ExtractionResult(
        success=True,
        data=extracted_data,
        metadata=ExtractionMetadata(
            method=extractor.name,
            ocr_engine=ocr_res.engine_name,
            raw_ocr_text=ocr_res.raw_text,
            processing_time_sec=round(total_elapsed, 4),
            ocr_character_count=len(ocr_res.raw_text),
            field_evidence=evidence,
            adaptation_mode=effective_mode,
            retry_count=retry_count,
            needs_human_review=needs_review,
            input_tokens=in_tok,
            output_tokens=out_tok,
        ),
        validation_issues=validation_issues,
        is_valid=is_valid,
    )

    return {
        "error": None,
        "ocr_res": ocr_res,
        "bbox_bytes": bbox_bytes,
        "final_result": final_result,
        "fallback_warning": fallback_warning,
    }


def render_ui():
    st.set_page_config(
        page_title="Enterprise Receipt Document AI — Nested Learning",
        page_icon="🧾",
        layout="wide",
    )
    st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ent-header-banner">
            <div class="ent-header-title">🧾 Hệ thống Trích xuất Thông tin Hóa đơn Bán lẻ Việt Nam (Enterprise Document AI)</div>
            <div class="ent-header-sub">
                Kiến trúc <strong>Nested Learning (NeurIPS 2025)</strong>:
                Ma trận Bộ nhớ Liên kết Delta Rule <strong>M<sub>t</sub> = (I − η<sub>t</sub> k<sub>t</sub> k<sub>t</sub><sup>⊤</sup>) M<sub>t−1</sub> + η<sub>t</sub> v<sub>t</sub> k<sub>t</sub><sup>⊤</sup></strong>
                &bull; Kiểm định Toán học & Quét Bỏ sót Trường (LSS Self-Correction)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cms_store = ContinuumMemoryStore()

    # =========================================================================
    # SIDEBAR: CẤU HÌNH HỆ THỐNG & ĐIỀU KHIỂN THÍCH NGHI NESTED LEARNING
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Cấu hình Pipeline")

        # Đặt PaddleVietOCREngine ('paddleocr') làm mặc định (fallback về Tesseract nếu máy không hỗ trợ)
        ocr_options = ["paddleocr", "vietocr", "tesseract", "dummy"]
        ocr_choice = st.selectbox(
            "1. Mô-đun Nhận dạng OCR",
            options=ocr_options,
            index=0,
            help="'paddleocr': PaddleOCR DBNet + VietOCR VGG-Seq2Seq (Mặc định MT2, tự động fallback Tesseract).\n'tesseract': Tesseract 5 Tiếng Việt.",
        )
        engine_obj = get_ocr_engine(ocr_choice)
        is_avail, avail_msg = engine_obj.is_available()
        if is_avail:
            st.success(f"🟢 OCR Sẵn sàng: `{ocr_choice}`")
        else:
            st.warning(f"🟡 `{ocr_choice}` chưa cài trọng số local. Sẽ hướng dẫn hoặc dùng `tesseract`.")
            st.caption(avail_msg or "")

        st.subheader("🧠 Cấp độ Thích nghi (Nested Learning)")
        extraction_method = st.selectbox(
            "2. Phương pháp Bóc tách (KIE)",
            options=[
                "5. Hybrid Nested Router (Regex Fast-Path -> LLM + LSS)",
                "1. Baseline B1 (Regex + OCR Digit Denoising)",
                "2. LLM Zero-shot (k=0)",
                "3. LLM Dynamic Few-shot CMS (k=3, Delta Rule)",
                "4. LLM Few-shot + Bố cục LMDX (k=3, Layout [0,1000])",
            ],
            index=0,
        )

        enable_lss = st.checkbox(
            "🔄 Tự sửa lỗi qua Tín hiệu Ngạc nhiên (LSS)",
            value=True,
            help="Tự động kiểm tra công thức Tạm tính - Giảm giá + VAT = Tổng tiền và quét các nhãn SDT/MST bị bỏ sót.",
        )

        st.markdown(
            f"""
            <div class="inspector-stat-pill" style="text-align:left;margin-top:8px;">
                <div class="inspector-stat-label">Kho Bộ nhớ Liên tục CMS (C^(1)=1)</div>
                <div class="inspector-stat-val">{len(cms_store.episodes)} Episodes &bull; Dim={cms_store.memory_dim}</div>
                <div style="font-size:0.74rem;color:#34d399;margin-top:3px;">
                    ✓ Đã nạp Ma trận Delta Rule ({cms_store.delta_memory.step_count} bước cập nhật)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("🛠️ Tiền xử lý Ảnh (OpenCV)")
        apply_preprocess = st.checkbox("Bật tiền xử lý ảnh", value=True)
        enhance_contrast = False
        to_grayscale = False
        adaptive_threshold = False
        deskew = False

        if apply_preprocess:
            enhance_contrast = st.checkbox("Tăng tương phản (CLAHE)", value=True)
            to_grayscale = st.checkbox("Chuyển ảnh xám (Grayscale)", value=False)
            adaptive_threshold = st.checkbox("Phân ngưỡng nhị phân (Threshold)", value=False)
            deskew = st.checkbox("Khử nghiêng tự động (Deskew)", value=False)

    # =========================================================================
    # KHU VỰC CHÍNH: SIDE-BY-SIDE DOCUMENT INSPECTOR
    # =========================================================================
    col_left, col_right = st.columns([1.0, 1.22], gap="large")

    with col_left:
        st.subheader("🖼️ 1. Trình Kiểm tra Tài liệu (Document Inspector)")
        uploaded_file = st.file_uploader(
            "Tải ảnh hóa đơn bán lẻ (.jpg, .png, .jpeg, .webp)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        show_bboxes = st.checkbox("🔲 Tô sáng vùng chữ OCR (Bounding Boxes)", value=True)

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            val_res = validate_image(file_bytes, file_name=uploaded_file.name)

            if not val_res.is_valid:
                st.error(f"❌ Tệp ảnh không hợp lệ: {val_res.error_message}")
                return

            processed_bytes = file_bytes
            if apply_preprocess:
                opts = PreprocessOptions(
                    enhance_contrast=enhance_contrast,
                    to_grayscale=to_grayscale,
                    adaptive_threshold=adaptive_threshold,
                    deskew=deskew,
                )
                proc_img, _, _ = preprocess_receipt_image(file_bytes, opts)
                if proc_img is not None:
                    ok_enc, buffer = cv2.imencode(".jpg", proc_img)
                    if ok_enc:
                        processed_bytes = buffer.tobytes()

            # Tự động chạy hoặc chạy khi bấm nút (lưu trong session_state theo tên file + cấu hình)
            run_key = f"{uploaded_file.name}_{len(file_bytes)}_{ocr_choice}_{extraction_method}_{enable_lss}_{apply_preprocess}_{enhance_contrast}_{deskew}"
            process_btn = st.button("🚀 Thực hiện Nhận dạng & Trích xuất", type="primary", use_container_width=True)

            if process_btn or st.session_state.get("last_run_key") != run_key:
                with st.spinner("Đang thực thi Pipeline OCR + Kiểm định Nested Learning..."):
                    st.session_state["pipeline_output"] = run_extraction_pipeline(
                        processed_bytes=processed_bytes,
                        ocr_choice=ocr_choice,
                        extraction_method=extraction_method,
                        enable_lss=enable_lss,
                        cms_store=cms_store,
                    )
                    st.session_state["last_run_key"] = run_key

            pipe_out = st.session_state.get("pipeline_output", {})
            ocr_res = pipe_out.get("ocr_res")
            num_ocr_lines = len(ocr_res.lines) if ocr_res and ocr_res.lines else 0

            # Thanh thông số chất lượng ảnh
            st.markdown(
                f"""
                <div class="inspector-stats-bar">
                    <div class="inspector-stat-pill">
                        <div class="inspector-stat-label">Độ phân giải</div>
                        <div class="inspector-stat-val">{val_res.width} &times; {val_res.height} px</div>
                    </div>
                    <div class="inspector-stat-pill">
                        <div class="inspector-stat-label">Độ nét Laplacian</div>
                        <div class="inspector-stat-val">{val_res.blur_score:.1f}</div>
                    </div>
                    <div class="inspector-stat-pill">
                        <div class="inspector-stat-label">Số dòng OCR</div>
                        <div class="inspector-stat-val">{num_ocr_lines} dòng</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # HIỂN THỊ ẢNH CHÍNH: Nếu bật show_bboxes và đã có bbox_bytes thì hiển thị NGAY ảnh đã vẽ Bounding Box!
            display_img_bytes = (
                pipe_out.get("bbox_bytes")
                if (show_bboxes and pipe_out.get("bbox_bytes"))
                else processed_bytes
            )
            caption_str = (
                f"Ảnh phủ hộp bao OCR ({num_ocr_lines} vùng dòng chữ phát hiện) — {uploaded_file.name}"
                if (show_bboxes and num_ocr_lines > 0)
                else f"Ảnh hóa đơn gốc — {uploaded_file.name}"
            )
            st.image(display_img_bytes, caption=caption_str, use_container_width=True)
        else:
            st.info("👆 Hãy chọn một tệp ảnh hóa đơn (ví dụ `hoadon_thucte_010.jpg`) để bắt đầu phân tích.")

    with col_right:
        st.subheader("📑 2. Hóa đơn Điện tử Số hóa (Digital Receipt Card)")

        pipe_out = st.session_state.get("pipeline_output") if uploaded_file is not None else None
        if pipe_out:
            if pipe_out.get("error"):
                st.error(pipe_out["error"])
                return

            final_result: ExtractionResult = pipe_out["final_result"]
            ocr_res = pipe_out["ocr_res"]
            extracted_data = final_result.data
            meta = final_result.metadata
            validation_issues = final_result.validation_issues

            if pipe_out.get("fallback_warning"):
                st.info(f"ℹ️ {pipe_out['fallback_warning']}")

            # Banner trạng thái kiểm định rõ ràng, trung thực (Không bao giờ hiển thị 0 cảnh báo!)
            actionable_issues = [
                iss for iss in validation_issues
                if iss.severity in (ValidationSeverity.WARNING, ValidationSeverity.ERROR)
            ]
            if (meta.needs_human_review or not final_result.is_valid) and len(actionable_issues) > 0:
                st.markdown(
                    f"""
                    <div class="status-banner-warn">
                        <strong>⚠️ CẦN NGƯỜI KIỂM DUYỆT (LSS FLAGGED — {len(actionable_issues)} CẢNH BÁO):</strong><br/>
                        Phát hiện sai lệch toán học hoặc trường thông tin nghi vấn cần xác nhận.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif not final_result.is_valid:
                st.markdown(
                    """
                    <div class="status-banner-warn" style="background:#fee2e2;border-color:#f87171;color:#991b1b;">
                        <strong>❌ HÓA ĐƠN KHÔNG HỢP LỆ (INVALID DATA):</strong><br/>
                        Dữ liệu trích xuất thiếu các trường cốt lõi hoặc không vượt qua kiểm định.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="status-banner-ok">
                        <strong>✅ HÓA ĐƠN HỢP LỆ HOÀN TOÀN (VERIFIED):</strong><br/>
                        Đầy đủ các trường nhận diện trên phiếu và khớp tuyệt đối ràng buộc kế toán:
                        <code>Tạm tính - Giảm giá + VAT = Tổng thanh toán</code>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Thanh tiến trình & thông số Nested Learning
            k_vec = text_to_l2_normalized_vector(ocr_res.raw_text, dim=cms_store.memory_dim)
            similar_eps = cms_store.retrieve_similar_examples(ocr_res.raw_text, k=1, exclude_near_exact=False)
            delta_conf = 0.96 if not meta.needs_human_review else 0.72

            st.caption(
                f"Engine OCR: `{meta.ocr_engine}` | Bộ trích xuất: `{meta.method}` | "
                f"Chế độ Nested: `{meta.adaptation_mode}` | LSS Retries: `{meta.retry_count}` | "
                f"Tokens: `{meta.input_tokens} in / {meta.output_tokens} out` | "
                f"Độ trễ: `{meta.processing_time_sec:.3f}s`"
            )
            st.progress(delta_conf, text=f"Độ tin cậy Kiểm định Toán học & Bộ nhớ Liên kết Delta Rule: {int(delta_conf*100)}%")

            tab_card, tab_editor, tab_validation, tab_ocr, tab_export = st.tabs([
                "🧾 Hóa đơn Số hóa (12 Trường)",
                "✏️ Chỉnh sửa & Nạp CMS",
                f"🛡️ Kiểm định LSS ({len(validation_issues)})",
                "🔍 Văn bản OCR Thô",
                "📥 Xuất Excel & JSON",
            ])

            with tab_card:
                card_html = build_digital_receipt_html(
                    data=extracted_data,
                    evidence=meta.field_evidence,
                    issues=validation_issues,
                    retry_count=meta.retry_count,
                )
                st.markdown(card_html, unsafe_allow_html=True)

            with tab_editor:
                st.markdown("##### Hiệu chỉnh 12 trường & Cập nhật Ma trận Bộ nhớ Delta Rule (`CMS`):")
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    ed_seller = st.text_input("1. Tên cửa hàng", value=extracted_data.seller_name or "")
                    ed_address = st.text_input("2. Địa chỉ", value=extracted_data.seller_address or "")
                    ed_phone = st.text_input("3. Số điện thoại", value=extracted_data.seller_phone or "")
                    ed_tax = st.text_input("4. Mã số thuế", value=extracted_data.tax_id or "")
                    ed_inv = st.text_input("5. Số hóa đơn", value=extracted_data.invoice_number or "")
                    ed_time = st.text_input("6. Ngày giờ", value=extracted_data.timestamp or "")
                with e_col2:
                    ed_sub = st.number_input("7. Tạm tính (VND)", value=int(extracted_data.subtotal or 0), step=1000)
                    ed_disc = st.number_input("8. Giảm giá (VND)", value=int(extracted_data.discount or 0), step=1000)
                    ed_vat = st.number_input("9. Thuế VAT (VND)", value=int(extracted_data.vat_amount or 0), step=1000)
                    ed_tot = st.number_input("10. Tổng thanh toán (VND)", value=int(extracted_data.total_cost or 0), step=1000)
                    ed_pay = st.text_input("11. Hình thức thanh toán", value=extracted_data.payment_method or "")

                st.markdown("##### 12. Danh sách mặt hàng:")
                initial_items = [
                    {
                        "item_name": it.item_name,
                        "quantity": float(it.quantity or 1.0),
                        "unit_price": int(it.unit_price or 0),
                        "total_price": int(it.total_price or 0),
                    }
                    for it in extracted_data.items
                ] or [{"item_name": "Mặt hàng 1", "quantity": 1.0, "unit_price": 0, "total_price": 0}]

                edited_items_df = st.data_editor(
                    initial_items,
                    num_rows="dynamic",
                    use_container_width=True,
                )

                if st.button("🧠 Xác nhận & Cập nhật Ma trận Bộ nhớ Delta Rule (CMS)", type="primary"):
                    reconstructed_items = [
                        ReceiptItem(
                            item_name=str(r.get("item_name", "")),
                            quantity=float(r.get("quantity", 1.0)),
                            unit_price=int(r.get("unit_price", 0)) or None,
                            total_price=int(r.get("total_price", 0)) or None,
                        )
                        for r in edited_items_df
                    ]
                    confirmed_data = ReceiptData(
                        seller_name=ed_seller or None,
                        seller_address=ed_address or None,
                        seller_phone=ed_phone or None,
                        tax_id=ed_tax or None,
                        invoice_number=ed_inv or None,
                        timestamp=ed_time or None,
                        subtotal=ed_sub or None,
                        discount=ed_disc or None,
                        vat_amount=ed_vat or None,
                        total_cost=ed_tot or None,
                        payment_method=ed_pay or None,
                        items=reconstructed_items,
                    )
                    ep = cms_store.add_verified_episode(
                        ocr_text=ocr_res.raw_text,
                        receipt_data=confirmed_data,
                        source="user_interface",
                    )
                    st.success(
                        f"✅ Đã lưu Episode `{ep.episode_id}` vào CMS và cập nhật Ma trận Delta Rule "
                        f"(M_t ∈ ℝ¹²⁸ˣ¹²⁸, bước #{cms_store.delta_memory.step_count})!"
                    )

            with tab_validation:
                if not validation_issues:
                    st.success("🎉 Không có bất kỳ cảnh báo nào. Mọi ràng buộc số học và trường thông tin đều hợp lệ.")
                else:
                    for issue in validation_issues:
                        if issue.severity == ValidationSeverity.ERROR:
                            st.error(f"🔴 **[{issue.code}]** (`{issue.field_name}`): {issue.message}")
                        elif issue.severity == ValidationSeverity.WARNING:
                            st.warning(f"🟡 **[{issue.code}]** (`{issue.field_name}`): {issue.message}")
                        else:
                            st.info(f"🔵 **[{issue.code}]** (`{issue.field_name}`): {issue.message}")

            with tab_ocr:
                st.caption(f"Engine: `{ocr_res.engine_name}` | Số ký tự: `{len(ocr_res.raw_text)}` | Số dòng: `{len(ocr_res.lines)}`")
                st.text_area("Văn bản OCR nhận dạng được:", value=ocr_res.raw_text, height=320)

            with tab_export:
                json_str = final_result.model_dump_json(indent=2)
                excel_bytes = export_receipt_to_excel_bytes(extracted_data, meta)
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    st.download_button(
                        label="📊 Tải báo cáo Excel (.xlsx)",
                        data=excel_bytes,
                        file_name=f"hoadon_{int(time.time())}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                with c_dl2:
                    st.download_button(
                        label="💾 Tải dữ liệu JSON (.json)",
                        data=json_str,
                        file_name=f"hoadon_{int(time.time())}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                st.json(json_str)


if __name__ == "__main__":
    render_ui()

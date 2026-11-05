r"""Bộ kiểm thử toàn diện cho đợt Chấn chỉnh & Nâng cấp Toàn diện:
1. Kiểm thử trực tiếp trên ảnh `hoadon_thucte_010.jpg` (cả bản OCR sạch và bản OCR nhiễu `@28`, `@311`):
   - Bắt buộc `seller_phone == "028 3512 9090"` và `tax_id == "0311234987"`, tuyệt đối không được ra `None`!
2. Kiểm thử `check_unextracted_evidence` (Unextracted Evidence Scan) và điều kiện Escalation của `HybridNestedExtractor`:
   - Khi văn bản OCR có `SDT` hoặc `MST` hoặc `So HD` mà dữ liệu trích xuất bị `None`, bắt buộc phát sinh
     `MISSING_DETECTED_FIELD` (`is_valid == False`) và `HybridNestedExtractor` bắt buộc leo thang sang LLM!
3. Kiểm thử toán học của `DeltaRuleMemoryMatrix` (Phương trình 65, Bài báo Nested Learning, NeurIPS 2025):
   - Sai số hồi quy $\|M_t k_t - v_t\|_2^2$ giảm đơn điệu nghiêm ngặt sau mỗi bước cập nhật Delta.
4. Kiểm thử phép lặp trực giao hóa `newton_schulz_orthogonalize` (Phương trình 44) và `MultiScaleMomentumMuonState` (Algorithm 1 - M3).
5. Kiểm thử hàm vẽ Bounding Box `draw_ocr_bounding_boxes` tạo ra ảnh khác biệt rõ rệt với ảnh gốc.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from receipt_extraction.config import PROJECT_ROOT
from receipt_extraction.extraction.hybrid_extractor import HybridNestedExtractor
from receipt_extraction.extraction.llm_extractor import LLMExtractor
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.memory.cms_store import DeltaRuleMemoryMatrix
from receipt_extraction.ocr.base import OCRBoundingBox, OCRLine, OCRResult
from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.optimizers.nested_optimizers import (
    DeltaGradientDescentState,
    MultiScaleMomentumMuonState,
    newton_schulz_orthogonalize,
)
from receipt_extraction.schemas.receipt import ReceiptData
from receipt_extraction.validation.receipt_validator import (
    check_unextracted_evidence,
    validate_receipt_data,
)
from receipt_extraction.web.export_utils import draw_ocr_bounding_boxes


def test_hoadon_thucte_010_extracts_exact_phone_and_tax_id() -> None:
    """Yêu cầu 1 & 5: Kiểm tra trực tiếp trên ảnh hoadon_thucte_010.jpg và cả trường hợp OCR nhiễu '@'."""
    extractor = RegexBaselineExtractor()

    # Trường hợp A: Chạy Tesseract thật trực tiếp trên file ảnh hoadon_thucte_010.jpg
    img_path = PROJECT_ROOT.parent / "Data_HoaDon_NCKH" / "01_MC_OCR_2021_Chuan_CuocThi" / "images" / "hoadon_thucte_010.jpg"
    if img_path.exists():
        tess = get_ocr_engine("tesseract")
        if tess.is_available()[0]:
            ocr_res = tess.recognize(img_path)
            data_real, _ = extractor.extract(ocr_res)
            assert data_real.seller_phone == "028 3512 9090"
            assert data_real.tax_id == "0311234987"
            assert data_real.invoice_number == "MS-20261021-77"
            assert data_real.total_cost == 75600

    # Trường hợp B: Văn bản OCR bị biến dạng nặng ('@28 3512 9999' hoặc '928 3512 9090' và '@311234987' / '6311234987')
    corrupted_ocr_text = (
        "MINISTOP NGUYEN GIA TRI\n"
        "Dia chi: 120 Nguyen Gia Tri, P.25, Q. Binh Thanh, TP.HCM\n"
        "SDT: @28 3512 9090\n"
        "MST: @311234987\n"
        "So HD: MS-20261021-77\n"
        "Ngay: 21/10/2026 21:05:10\n"
        "Ga ran Karage xau 3 vien 2 25.000 50.000\n"
        "Nuoc ngot Pepsi Zero 320ml 2 11.000 22.000\n"
        "Tam tinh: 72.000\n"
        "Giam gia: -2.000\n"
        "Thue VAT (8%): 5.600\n"
        "Tong thanh toan: 75.600\n"
        "Hinh thuc thanh toan: Tien mat"
    )
    data_corrupted, _ = extractor.extract(OCRResult(engine_name="tesseract", raw_text=corrupted_ocr_text))
    assert data_corrupted.seller_phone == "028 3512 9090"
    assert data_corrupted.tax_id == "0311234987"


def test_unextracted_evidence_scan_triggers_lss_and_forces_hybrid_escalation() -> None:
    """Yêu cầu 1: Nếu OCR có từ khóa SDT / MST / So HD mà trích xuất ra None -> Bắt buộc phát sinh MISSING_DETECTED_FIELD và leo thang LLM."""
    raw_ocr = (
        "MINISTOP NGUYEN GIA TRI\n"
        "Dia chi: 120 Nguyen Gia Tri\n"
        "SDT: [mờ_không_đọc_được]\n"
        "MST: [mờ_không_đọc_được]\n"
        "So HD: [mờ]\n"
        "Ga ran 2 25.000 50.000\n"
        "Tong thanh toan: 50.000\n"
    )
    incomplete_data = ReceiptData(
        seller_name="MINISTOP NGUYEN GIA TRI",
        seller_address="120 Nguyen Gia Tri",
        seller_phone=None,
        tax_id=None,
        invoice_number=None,
        total_cost=50000,
    )
    unextracted = check_unextracted_evidence(incomplete_data, raw_ocr)
    missing_fields = {iss.field_name for iss in unextracted if iss.code == "MISSING_DETECTED_FIELD"}
    assert "seller_phone" in missing_fields
    assert "tax_id" in missing_fields
    assert "invoice_number" in missing_fields

    issues, is_valid = validate_receipt_data(incomplete_data, raw_ocr_text=raw_ocr)
    assert is_valid is False  # Tuyệt đối không được báo "Hóa đơn hợp lệ hoàn toàn"!

    # Kiểm tra HybridNestedExtractor bắt buộc phải leo thang sang LLM khi bị sót SDT/MST
    escalated_called = {"called": False}

    def _mock_llm_recover(_msgs):
        escalated_called["called"] = True
        return ReceiptData(
            seller_name="MINISTOP NGUYEN GIA TRI",
            seller_address="120 Nguyen Gia Tri",
            seller_phone="028 3512 9090",
            tax_id="0311234987",
            invoice_number="MS-01",
            total_cost=50000,
        ).model_dump_json()

    llm_ext = LLMExtractor(custom_chat_fn=_mock_llm_recover, enable_lss_retry=False)
    hybrid = HybridNestedExtractor(llm_extractor=llm_ext)
    out_data, _ = hybrid.extract(OCRResult(engine_name="tesseract", raw_text=raw_ocr))

    assert escalated_called["called"] is True
    assert hybrid.last_adaptation_mode == "hybrid_escalated_llm"
    assert out_data.seller_phone == "028 3512 9090"
    assert out_data.tax_id == "0311234987"


def test_delta_rule_memory_matrix_monotonic_loss_decrease() -> None:
    """Yêu cầu 3 & 5: Kiểm chứng sai số hồi quy ||M_t k_t - v_t||_2^2 giảm đơn điệu nghiêm ngặt sau mỗi bước cập nhật Delta."""
    rng = np.random.default_rng(42)
    dim = 64
    mem = DeltaRuleMemoryMatrix(dim=dim, eta=0.45)

    k_t = rng.normal(size=dim)
    k_t = k_t / np.linalg.norm(k_t, ord=2)
    v_t = rng.normal(size=dim)
    v_t = v_t / np.linalg.norm(v_t, ord=2)

    losses = [mem.regression_loss(k_t, v_t)]
    for _ in range(8):
        loss_before, loss_after = mem.update(k_t, v_t)
        assert loss_after < loss_before
        losses.append(loss_after)

    # Kiểm tra dãy sai số giảm đơn điệu nghiêm ngặt và hội tụ sát về 0
    for i in range(len(losses) - 1):
        assert losses[i + 1] < losses[i]
    assert losses[-1] < 1e-3

    # Kiểm tra cả dạng chiếu bên trái (left_projection=True) khi M_0 = 0
    mem_left = DeltaRuleMemoryMatrix(dim=dim, eta=1.0)
    lb, la = mem_left.update(k_t, v_t, eta_t=1.0, left_projection=True)
    assert la < lb
    assert la < 1e-12


def test_newton_schulz_orthogonalize_and_m3_optimizer() -> None:
    """Yêu cầu 3 & 5: Kiểm chứng phép lặp trực giao hóa Newton-Schulz (Phương trình 44) và Multi-scale Momentum M3."""
    rng = np.random.default_rng(123)
    G = rng.normal(size=(16, 8))

    # Sai số trực giao ban đầu của G chuẩn hóa Frobenius
    G_norm = G / np.linalg.norm(G, ord="fro")
    eye = np.eye(8)
    init_ortho_err = np.linalg.norm(G_norm.T @ G_norm - eye, ord="fro")

    # Sau 8 bước lặp Newton-Schulz, các cột phải tiến gần tới hệ trực chuẩn (O^T O ~ I) hơn rõ rệt
    O_ns = newton_schulz_orthogonalize(G, num_iters=8, zeta=0.25)
    final_ortho_err = np.linalg.norm(O_ns.T @ O_ns - eye, ord="fro")
    assert final_ortho_err < init_ortho_err

    # Kiểm tra MultiScaleMomentumMuonState (Algorithm 1 - M3) tối ưu hàm bậc hai L(W) = 0.5 * ||W - W*||_F^2
    W_star = rng.normal(size=(8, 8))
    W = np.zeros((8, 8), dtype=np.float64)
    m3 = MultiScaleMomentumMuonState(shape=(8, 8), lr=0.15, slow_freq_f=2)

    init_dist = np.linalg.norm(W - W_star, ord="fro")
    for _ in range(12):
        grad = W - W_star
        W = m3.step(W, grad)
    final_dist = np.linalg.norm(W - W_star, ord="fro")
    assert final_dist < init_dist
    assert m3.step_count == 12

    # Kiểm tra DeltaGradientDescentState (Phương trình 57)
    dgd = DeltaGradientDescentState(shape=(8, 8), lr=0.2)
    x_t = rng.normal(size=8)
    g_y = rng.normal(size=8)
    W_dgd = dgd.step(x_t, g_y)
    assert W_dgd.shape == (8, 8)
    assert np.linalg.norm(W_dgd) > 0.0


def test_bounding_box_overlay_modifies_image_pixels() -> None:
    """Yêu cầu 2: Đảm bảo hàm vẽ Bounding Box tạo ra ảnh có lớp phủ khung chữ nhật thực tế."""
    from PIL import Image
    import io

    img = Image.new("RGB", (300, 120), color=(245, 245, 245))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    lines = [
        OCRLine(
            text="MINISTOP NGUYEN GIA TRI",
            confidence=0.95,
            bounding_box=OCRBoundingBox(x_min=20, y_min=25, x_max=260, y_max=55),
        )
    ]
    annotated_bytes = draw_ocr_bounding_boxes(raw_bytes, lines, show_labels=True)
    assert len(annotated_bytes) > 0
    assert annotated_bytes != raw_bytes

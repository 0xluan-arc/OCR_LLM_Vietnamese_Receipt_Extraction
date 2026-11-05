"""Bộ kiểm thử đơn vị (Unit Tests) cho Giai đoạn 5:
1. Kiểm thử 4 nhóm phân loại lỗi tự động trong error_analysis.py:
   - OCR_CORRUPTION_ERROR
   - EXTRACTOR_MISS_ERROR
   - EXTRACTOR_WRONG_VALUE
   - HALLUCINATION_ERROR
2. Kiểm thử nhãn xác thực trung thực (SIMULATION vs VERIFIED REAL MODEL).
3. Kiểm thử chống rò rỉ dữ liệu (0% Data Leakage) khi nạp tập Train vào ContinuumMemoryStore.
4. Kiểm thử xuất dữ liệu huấn luyện QLoRA SFT (ChatML + Delta Rule Denoising) và cấu hình siêu tham số QLoRA.
"""

from __future__ import annotations

import json
from pathlib import Path

from receipt_extraction.data.mcocr_loader import MCOCRRecord, split_dataset
from receipt_extraction.evaluation.error_analysis import (
    EXTRACTOR_MISS_ERROR,
    EXTRACTOR_WRONG_VALUE,
    HALLUCINATION_ERROR,
    OCR_CORRUPTION_ERROR,
    REAL_MODEL_STAMP,
    SIMULATION_STAMP,
    classify_field_error,
    generate_error_analysis_report,
    save_error_analysis_files,
)
from receipt_extraction.schemas.receipt import (
    ExtractionMetadata,
    ExtractionResult,
    ReceiptData,
    ReceiptItem,
)


def test_classify_field_error_all_four_categories() -> None:
    """Kiểm chứng chính xác 4 loại nguyên nhân gốc rễ khi dự đoán trường đơn bị sai."""
    raw_ocr = (
        "SIEU THI TIEN LOI CIRCLE K\n"
        "Dia chi: 123 Nguyen Thi Minh Khai, Q1\n"
        "SDT: 028 3822 1199\n"
        "MST: 0312456789\n"
        "Tam tinh: 100.000\n"
        "Tong thanh toan: 105.000\n"
    )

    # 1. OCR_CORRUPTION_ERROR: Giá trị chuẩn "145000" hoàn toàn không xuất hiện trong raw_ocr (do OCR đọc nhầm thành 105.000)
    err_ocr = classify_field_error(
        field_name="total_cost",
        gt_value=145000,
        pred_value=105000,
        raw_ocr_text=raw_ocr,
    )
    assert err_ocr is not None
    assert err_ocr["error_type"] == OCR_CORRUPTION_ERROR

    # 2. EXTRACTOR_MISS_ERROR: Giá trị chuẩn "0312456789" có nằm trong raw_ocr nhưng Extractor trả về None
    err_miss = classify_field_error(
        field_name="tax_id",
        gt_value="0312456789",
        pred_value=None,
        raw_ocr_text=raw_ocr,
    )
    assert err_miss is not None
    assert err_miss["error_type"] == EXTRACTOR_MISS_ERROR

    # 3. EXTRACTOR_WRONG_VALUE: Giá trị chuẩn "105000" có trong raw_ocr, nhưng Extractor nhặt nhầm số "100000" (cũng có trên hóa đơn)
    err_wrong = classify_field_error(
        field_name="total_cost",
        gt_value=105000,
        pred_value=100000,
        raw_ocr_text=raw_ocr,
    )
    assert err_wrong is not None
    assert err_wrong["error_type"] == EXTRACTOR_WRONG_VALUE

    # 4. HALLUCINATION_ERROR: Extractor tự bịa ra một giá trị "0999888777" không hề tồn tại trong raw_ocr
    err_hallucination = classify_field_error(
        field_name="seller_phone",
        gt_value="028 3822 1199",
        pred_value="0999888777",
        raw_ocr_text=raw_ocr,
    )
    assert err_hallucination is not None
    assert err_hallucination["error_type"] == HALLUCINATION_ERROR

    # Trường hợp khớp hoàn toàn -> None (không có lỗi)
    err_none = classify_field_error(
        field_name="tax_id",
        gt_value="0312456789",
        pred_value="0312456789",
        raw_ocr_text=raw_ocr,
    )
    assert err_none is None


def test_error_analysis_report_and_verification_stamps(tmp_path: Path) -> None:
    """Kiểm chứng báo cáo Error Analysis ghi nhận đúng mẫu lỗi và đóng dấu SIMULATION vs REAL MODEL."""
    raw_ocr = "HIGHLANDS COFFEE\nTong cong: 50.000"
    gt = ReceiptData(
        seller_name="HIGHLANDS COFFEE",
        tax_id="0302088135",
        total_cost=50000,
    )
    pred = ReceiptData(
        seller_name="HIGHLANDS COFFEE",
        tax_id=None,  # Bị mất trong OCR -> OCR_CORRUPTION_ERROR
        total_cost=99000,  # Bịa số không có trong OCR -> HALLUCINATION_ERROR
    )
    res = ExtractionResult(
        success=True,
        data=pred,
        metadata=ExtractionMetadata(method="test", ocr_engine="tesseract", raw_ocr_text=raw_ocr),
        validation_issues=[],
        is_valid=True,
    )

    # Kiểm tra đóng dấu Simulation
    rep_sim = generate_error_analysis_report(
        sample_ids=["sample_01"],
        results=[res],
        ground_truths=[gt],
        is_simulation=True,
    )
    assert rep_sim["verification_stamp"] == SIMULATION_STAMP
    assert rep_sim["samples_with_errors_or_warnings"] == 1
    assert rep_sim["error_type_breakdown"][OCR_CORRUPTION_ERROR] >= 1
    assert rep_sim["error_type_breakdown"][HALLUCINATION_ERROR] >= 1

    # Kiểm tra đóng dấu Real Model và lưu file .md + .jsonl
    rep_real = generate_error_analysis_report(
        sample_ids=["sample_01"],
        results=[res],
        ground_truths=[gt],
        is_simulation=False,
    )
    assert rep_real["verification_stamp"] == REAL_MODEL_STAMP

    saved = save_error_analysis_files(rep_real, tmp_path, run_tag="unit_test")
    assert saved["markdown"].exists()
    assert saved["jsonl"].exists()
    md_text = saved["markdown"].read_text(encoding="utf-8")
    assert REAL_MODEL_STAMP in md_text
    assert HALLUCINATION_ERROR in md_text


def test_cms_train_only_no_data_leakage(tmp_path: Path) -> None:
    """Đảm bảo ContinuumMemoryStore chỉ nạp tập train và không rò rỉ bất kỳ mẫu val/test nào."""
    import sys
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root / "scripts") not in sys.path:
        sys.path.insert(0, str(project_root / "scripts"))

    from download_and_prepare_mcocr import populate_train_only_cms

    records = [
        MCOCRRecord(
            img_id=f"rec_{i:02d}.jpg",
            gt_ocr_text=f"CUA HANG SO {i}\nTong: {i*10000}",
            receipt_data=ReceiptData(seller_name=f"CUA HANG SO {i}", total_cost=i * 10000),
        )
        for i in range(1, 11)
    ]
    splits = split_dataset(records, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=42)
    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 2

    cms_file = tmp_path / "cms_train_only.jsonl"
    store = populate_train_only_cms(splits["train"], cms_file)

    cms_ids = {ep.episode_id for ep in store.episodes}
    val_test_ids = {r.img_id for r in splits["val"]} | {r.img_id for r in splits["test"]}
    assert len(cms_ids) == 7
    assert cms_ids.isdisjoint(val_test_ids)


def test_export_sft_chatml_and_qlora_hyperparams(tmp_path: Path) -> None:
    """Kiểm chứng xuất dữ liệu ChatML JSONL theo Delta Rule và cấu hình QLoRA 4-bit NF4."""
    import sys
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root / "scripts") not in sys.path:
        sys.path.insert(0, str(project_root / "scripts"))

    from export_sft_dataset import export_split_to_chatml, inject_delta_rule_ocr_noise
    from train_qlora import TARGET_LORA_MODULES, build_qlora_hyperparameters

    clean_text = "Ca phe sua da 1 25.000 25.000\nTong cong: 25.000"
    noisy_text = inject_delta_rule_ocr_noise(clean_text)
    assert noisy_text != clean_text
    assert ".O00" in noisy_text or " l " in noisy_text

    rec = MCOCRRecord(
        img_id="sft_01.jpg",
        gt_ocr_text=clean_text,
        receipt_data=ReceiptData(
            seller_name="CAFE TEST",
            items=[ReceiptItem(item_name="Ca phe sua da", quantity=1.0, unit_price=25000, total_price=25000)],
            total_cost=25000,
        ),
    )
    out_jsonl = tmp_path / "train.jsonl"
    n_written = export_split_to_chatml([rec], out_jsonl, augment_delta_noise=True)
    assert n_written == 2
    lines = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert [m["role"] for m in lines[0]["messages"]] == ["system", "user", "assistant"]

    hparams = build_qlora_hyperparameters(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    assert hparams["quantization_config"]["bnb_4bit_quant_type"] == "nf4"
    assert hparams["quantization_config"]["bnb_4bit_use_double_quant"] is True
    assert hparams["lora_config"]["r"] == 16
    assert hparams["lora_config"]["lora_alpha"] == 32
    assert hparams["lora_config"]["target_modules"] == TARGET_LORA_MODULES

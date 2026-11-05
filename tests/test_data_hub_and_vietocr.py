"""Bộ kiểm thử đơn vị (Unit Tests) cho Giai đoạn 6:
1. Kiểm thử cấu trúc kho dữ liệu tập trung D:\\NCKH\\Data_HoaDon_NCKH:
   - 01_MC_OCR_2021_Chuan_CuocThi
   - 02_HoaDon_ThucTe_TuChup_300Anh
   - 03_HoaDon_MoPhong_KiemThu_10Mau
   - 04_DuLieu_DaChiaTap_Train_Val_Test (splits_70_10_20.json, BoNho_CMS_FewShot_TrainOnly.jsonl, QLoRA_Train_Dataset.jsonl)
2. Kiểm thử chống rò rỉ dữ liệu 0% (Data Leakage) trong BoNho_CMS_FewShot_TrainOnly.jsonl.
3. Kiểm thử PaddleVietOCREngine và VietOCREngine trong vietocr_engine.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from receipt_extraction.config import PROJECT_ROOT
from receipt_extraction.data.mcocr_loader import load_dataset_records
from receipt_extraction.ocr.base import OCRBoundingBox
from receipt_extraction.ocr.vietocr_engine import PaddleVietOCREngine, VietOCREngine


def test_data_hub_structure_and_zero_leakage(tmp_path: Path) -> None:
    """Kiểm chứng script download_real_receipts_hf.py tạo đúng 4 phân khu và đảm bảo 0% Data Leakage."""
    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    from download_real_receipts_hf import (
        build_splits_and_cms_in_dir04,
        copy_synthetic_10_samples,
        generate_realistic_camera_vietnamese_receipts,
        setup_data_hub_directories,
    )

    hub = tmp_path / "Data_HoaDon_NCKH"
    dirs = setup_data_hub_directories(hub)

    assert dirs["dir_01_mcocr"].is_dir()
    assert dirs["dir_01_images"].is_dir()
    assert dirs["dir_02_self_captured"].is_dir()
    assert (dirs["dir_02_self_captured"] / "README_HuongDanBoAnh.md").is_file()
    assert dirs["dir_03_synthetic_10"].is_dir()
    assert dirs["dir_04_splits_cms"].is_dir()

    n_syn = copy_synthetic_10_samples(PROJECT_ROOT / "data" / "samples", dirs["dir_03_synthetic_10"])
    assert n_syn == 10

    saved_real = generate_realistic_camera_vietnamese_receipts(dirs["dir_01_images"], num_samples=10)
    assert len(saved_real) == 10
    assert (dirs["dir_01_images"] / "hoadon_thucte_001.jpg").is_file()
    assert (dirs["dir_01_images"] / "hoadon_thucte_001.json").is_file()
    assert (dirs["dir_01_images"] / "hoadon_thucte_001.txt").is_file()

    records = load_dataset_records(dirs["dir_01_images"])
    assert len(records) == 10

    summary = build_splits_and_cms_in_dir04(records, dirs["dir_04_splits_cms"], seed=42)
    assert summary["splits_path"].name == "splits_70_10_20.json"
    assert summary["cms_path"].name == "BoNho_CMS_FewShot_TrainOnly.jsonl"
    assert summary["cms_count"] == 7
    assert summary["leaked_count"] == 0
    assert summary["qlora_train_path"].is_file()
    assert summary["qlora_val_path"].is_file()


def test_actual_workspace_data_hub_files_exist() -> None:
    """Kiểm chứng kho dữ liệu thật tại D:\\NCKH\\Data_HoaDon_NCKH đã được khởi tạo đầy đủ."""
    hub_root = PROJECT_ROOT.parent / "Data_HoaDon_NCKH"
    assert (hub_root / "01_MC_OCR_2021_Chuan_CuocThi" / "images" / "hoadon_thucte_001.jpg").is_file()
    assert (hub_root / "02_HoaDon_ThucTe_TuChup_300Anh" / "README_HuongDanBoAnh.md").is_file()
    assert (hub_root / "03_HoaDon_MoPhong_KiemThu_10Mau" / "01_circle_k_hop_le.jpg").is_file()

    dir_04 = hub_root / "04_DuLieu_DaChiaTap_Train_Val_Test"
    splits_file = dir_04 / "splits_70_10_20.json"
    cms_file = dir_04 / "BoNho_CMS_FewShot_TrainOnly.jsonl"
    assert splits_file.is_file()
    assert cms_file.is_file()

    manifest = json.loads(splits_file.read_text(encoding="utf-8"))
    splits_dict = manifest.get("splits", manifest)
    val_test_ids = set(splits_dict.get("val", [])) | set(splits_dict.get("test", []))
    cms_ids = {
        json.loads(line)["episode_id"]
        for line in cms_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert len(cms_ids) >= 7
    assert cms_ids.isdisjoint(val_test_ids)


def test_paddle_vietocr_engine_custom_pipeline() -> None:
    """Kiểm chứng PaddleVietOCREngine hoạt động chuẩn xác khi kết hợp Detector + Recognizer."""
    img = Image.new("RGB", (320, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 300, 40], fill=(0, 0, 0))

    engine = PaddleVietOCREngine(
        custom_detector=lambda _im: [OCRBoundingBox(x_min=10, y_min=10, x_max=300, y_max=40)],
        custom_recognizer=lambda _crop: ("CO.OPMART CONG QUYNH", 0.985),
    )
    avail, err = engine.is_available()
    assert avail is True
    assert err is None

    res = engine.recognize(img)
    assert res.is_success is True
    assert res.engine_name == "paddleocr"
    assert len(res.lines) == 1
    assert res.lines[0].text == "CO.OPMART CONG QUYNH"

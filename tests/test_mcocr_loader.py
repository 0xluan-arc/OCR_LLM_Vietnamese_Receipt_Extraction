"""Unit tests cho bộ nạp dữ liệu MC-OCR 2021 và hàm chia tập cố định (data/mcocr_loader.py)."""

import json
from pathlib import Path
import pytest

from receipt_extraction.data.mcocr_loader import (
    export_splits_manifest,
    load_dataset_records,
    load_mcocr_csv,
    parse_mcocr_row,
    split_dataset,
)


def test_parse_mcocr_row_from_csv_columns():
    """Kiểm tra chuyển đổi định dạng cột chuẩn của MC-OCR 2021 (anno_texts, anno_labels, anno_image_quality)."""
    row = {
        "img_id": "mcocr_public_145013abc.jpg",
        "anno_texts": "MINIMART AN BINH|||12 Lê Lợi, Q1|||15/08/2021 14:20|||125.000 đ",
        "anno_labels": "SELLER|||ADDRESS|||TIMESTAMP|||TOTAL_COST",
        "anno_image_quality": "0.842",
    }
    rec = parse_mcocr_row(row)
    assert rec.img_id == "mcocr_public_145013abc.jpg"
    assert rec.receipt_data.seller_name == "MINIMART AN BINH"
    assert rec.receipt_data.seller_address == "12 Lê Lợi, Q1"
    assert rec.receipt_data.timestamp == "15/08/2021 14:20"
    assert rec.receipt_data.total_cost == 125000
    assert rec.image_quality == 0.842
    assert "MINIMART AN BINH" in rec.gt_ocr_text

    mc_dict = rec.receipt_data.to_mc_ocr_dict()
    assert mc_dict["SELLER"] == "MINIMART AN BINH"
    assert mc_dict["TOTAL_COST"] == "125000"


def test_parse_mcocr_row_from_polygons():
    """Kiểm tra đọc từ cột anno_polygons sử dụng category_id chuẩn (15..18) của MC-OCR 2021."""
    polygons = [
        {"category_id": 15, "text": "VINMART+"},
        {"category_id": 16, "text": "Số 10 Quang Trung, Gò Vấp"},
        {"category_id": 17, "text": "20/10/2021 09:00"},
        {"category_id": 18, "text": "54.000"},
    ]
    row = {
        "img_id": "mcocr_poly_001.jpg",
        "anno_polygons": json.dumps(polygons, ensure_ascii=False),
        "anno_image_quality": 0.91,
    }
    rec = parse_mcocr_row(row)
    assert rec.receipt_data.seller_name == "VINMART+"
    assert rec.receipt_data.seller_address == "Số 10 Quang Trung, Gò Vấp"
    assert rec.receipt_data.total_cost == 54000
    assert rec.image_quality == 0.91


def test_load_mcocr_csv_and_split_70_10_20(tmp_path: Path):
    """Kiểm tra đọc file CSV MC-OCR 2021 và chia tập 70/10/20 cố định với seed=42."""
    csv_file = tmp_path / "mcocr_train_df.csv"
    lines = ["img_id,anno_texts,anno_labels,anno_image_quality"]
    for i in range(1, 21):
        lines.append(
            f"mcocr_{i:03d}.jpg,SHOP {i}|||Dia chi {i}|||17/10/2026|||{i*10}.000,SELLER|||ADDRESS|||TIMESTAMP|||TOTAL_COST,0.85"
        )
    csv_file.write_text("\n".join(lines), encoding="utf-8")

    records = load_mcocr_csv(csv_file)
    assert len(records) == 20

    splits_1 = split_dataset(records, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=42)
    splits_2 = split_dataset(records, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=42)

    assert len(splits_1["train"]) == 14
    assert len(splits_1["val"]) == 2
    assert len(splits_1["test"]) == 4

    # Kiểm tra tính tất định (Reproducibility) và không rò rỉ dữ liệu giữa các tập
    train_ids = [r.img_id for r in splits_1["train"]]
    val_ids = [r.img_id for r in splits_1["val"]]
    test_ids = [r.img_id for r in splits_1["test"]]

    assert train_ids == [r.img_id for r in splits_2["train"]]
    assert set(train_ids).isdisjoint(set(val_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    assert set(val_ids).isdisjoint(set(test_ids))

    manifest_path = export_splits_manifest(splits_1, tmp_path / "splits.json", seed=42)
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["counts"] == {"train": 14, "val": 2, "test": 4}


def test_load_dataset_records_from_samples_dir():
    """Kiểm tra nạp các bản ghi Ground-Truth từ thư mục data/samples/."""
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    records = load_dataset_records(samples_dir)
    assert len(records) >= 5
    assert records[0].img_id == "01_circle_k_hop_le.jpg"
    assert records[0].receipt_data.total_cost == 102600
    assert len(records[0].receipt_data.items) == 3
    assert "CIRCLE K" in records[0].gt_ocr_text

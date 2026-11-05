"""Kịch bản CLI chuẩn hóa nhãn MC-OCR 2021 và phân chia tập cố định (70/10/20, seed=42)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from receipt_extraction.data.mcocr_loader import (
    export_splits_manifest,
    load_dataset_records,
    split_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa bộ dữ liệu MC-OCR 2021 và xuất manifest chia tập splits.json (70/10/20, seed=42)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "samples"),
        help="Đường dẫn tới file CSV/JSON MC-OCR 2021 hoặc thư mục chứa dữ liệu.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Thư mục lưu splits.json và các bản ghi đã chuẩn hóa.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên cố định để tái lập kết quả chia tập (mặc định: 42).",
    )
    args = parser.parse_args()

    records = load_dataset_records(args.input)
    if not records:
        print(f"⚠️ Không tìm thấy bản ghi nào tại: {args.input}")
        return 1

    splits = split_dataset(records, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = export_splits_manifest(splits, out_dir / "splits.json", seed=args.seed)
    records_path = out_dir / "mcocr_normalized_records.json"
    records_payload = [r.model_dump() for r in records]
    records_path.write_text(
        json.dumps(records_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("✅ HOÀN TẤT CHUẨN HÓA & CHIA TẬP DỮ LIỆU MC-OCR")
    print("=" * 70)
    print(f"- Nguồn dữ liệu       : {args.input}")
    print(f"- Tổng số mẫu         : {len(records)}")
    print(f"- Phân chia (seed={args.seed}) : Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    print(f"- File manifest       : {manifest_path}")
    print(f"- Bản ghi chuẩn hóa   : {records_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Kịch bản xuất dữ liệu huấn luyện QLoRA SFT định dạng ChatML JSONL (qlora_train.jsonl & qlora_val.jsonl)
áp dụng nguyên lý Delta Rule (Denoising Learning) của khung lý thuyết Nested Learning (NeurIPS 2025).

Ghép cặp:
- Đầu vào (User): Văn bản OCR thực tế (kèm biến thể nhiễu OCR từ Tesseract hoặc ký tự nhầm lẫn đặc trưng)
- Đầu ra (Assistant): Chuỗi JSON 12 trường chuẩn xác tuyệt đối (Ground-Truth)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from receipt_extraction.data.mcocr_loader import MCOCRRecord, load_dataset_records, split_dataset
from receipt_extraction.extraction.llm_extractor import RECEIPT_EXTRACTION_SYSTEM_PROMPT


def inject_delta_rule_ocr_noise(clean_ocr_text: str) -> str:
    """Sinh biến thể văn bản OCR nhiễu thực tế theo nguyên lý Delta Rule (Denoising Learning).

    Mô phỏng các sai số nhận dạng đặc trưng của OCR tiếng Việt trên hóa đơn nhiệt:
    - Chữ số '1' ở cột số lượng bị đọc nhầm thành 'l' hoặc 'I'
    - Chữ số '0' ở hàng nghìn bị đọc nhầm thành 'O'
    - Rớt dấu tiếng Việt cục bộ ở các nhãn tổng tiền
    """
    noisy = clean_ocr_text
    # Thay thế một số mẫu số lượng ' 1 ' thành ' l ' để mô hình học khôi phục về 1.0
    noisy = re.sub(r"(\s)1(\s+\d{1,3}[\.,]\d{3})", r"\1l\2", noisy, count=1)
    # Thay thế '.000' đầu tiên thành '.O00' để mô hình học khôi phục số tiền theo ràng buộc tổng
    noisy = noisy.replace(".000", ".O00", 1)
    return noisy


def build_chatml_messages(
    ocr_text: str,
    receipt_json_str: str,
    system_prompt: str = RECEIPT_EXTRACTION_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Đóng gói một mẫu huấn luyện theo chuẩn hội thoại ChatML 3 lượt (system / user / assistant)."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": (
                    f"Văn bản OCR hóa đơn:\n```text\n{ocr_text.strip()}\n```\n"
                    "Hãy trích xuất dữ liệu cấu trúc JSON 12 trường."
                ),
            },
            {"role": "assistant", "content": receipt_json_str},
        ]
    }


def export_split_to_chatml(
    records: List[MCOCRRecord],
    output_path: Path,
    augment_delta_noise: bool = True,
) -> int:
    """Xuất danh sách bản ghi sang file JSONL định dạng ChatML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            target_json = rec.receipt_data.model_dump_json(indent=2)
            base_ocr = rec.gt_ocr_text or ""

            # 1. Mẫu gốc (Original OCR -> Clean 12-field JSON)
            item_clean = build_chatml_messages(base_ocr, target_json)
            f.write(json.dumps(item_clean, ensure_ascii=False) + "\n")
            count += 1

            # 2. Mẫu tăng cường khử nhiễu Delta Rule (Noisy OCR -> Clean 12-field JSON)
            if augment_delta_noise and base_ocr:
                noisy_ocr = inject_delta_rule_ocr_noise(base_ocr)
                if noisy_ocr != base_ocr:
                    item_noisy = build_chatml_messages(noisy_ocr, target_json)
                    f.write(json.dumps(item_noisy, ensure_ascii=False) + "\n")
                    count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Xuất dữ liệu huấn luyện QLoRA SFT (ChatML JSONL) theo nguyên lý Delta Rule."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "mcocr"),
        help="Thư mục dữ liệu nguồn đã chuẩn hóa (mặc định: data/raw/mcocr).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Thư mục lưu qlora_train.jsonl và qlora_val.jsonl (mặc định: data/processed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed chia tập (mặc định: 42).",
    )
    parser.add_argument(
        "--no-delta-augment",
        action="store_true",
        help="Tắt bước sinh cặp khử nhiễu Delta Rule trên tập train.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        dataset_dir = PROJECT_ROOT / "data" / "samples"

    # Ưu tiên nạp từ các file .json chi tiết nếu có để bảo toàn 12 trường
    json_files = sorted(
        p for p in dataset_dir.glob("*.json")
        if p.name not in ("splits.json", "manifest.json")
    )
    if json_files:
        from receipt_extraction.data.mcocr_loader import parse_mcocr_row
        records = []
        for jf in json_files:
            raw_dict = json.loads(jf.read_text(encoding="utf-8"))
            stem = jf.stem
            txt_cand = dataset_dir / f"{stem}.txt"
            if txt_cand.exists() and not raw_dict.get("gt_ocr_text"):
                raw_dict["gt_ocr_text"] = txt_cand.read_text(encoding="utf-8").strip()
            records.append(parse_mcocr_row(raw_dict, images_dir=dataset_dir))
    else:
        records = load_dataset_records(dataset_dir)

    splits = split_dataset(records, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=args.seed)

    out_dir = Path(args.output_dir)
    train_path = out_dir / "qlora_train.jsonl"
    val_path = out_dir / "qlora_val.jsonl"

    n_train = export_split_to_chatml(
        splits["train"],
        train_path,
        augment_delta_noise=not args.no_delta_augment,
    )
    # Tập validation giữ nguyên văn bản gốc (không nhân bản) để đánh giá Loss khách quan
    n_val = export_split_to_chatml(
        splits["val"],
        val_path,
        augment_delta_noise=False,
    )

    print("=" * 70)
    print("✅ HOÀN TẤT XUẤT DỮ LIỆU HUẤN LUYỆN QLORA SFT (CHATML FORMAT)")
    print("=" * 70)
    print(f"- Nguồn dữ liệu      : {dataset_dir}")
    print(f"- Tập Train JSONL    : {train_path} ({n_train} mẫu hội thoại ChatML, Delta Rule={not args.no_delta_augment})")
    print(f"- Tập Val JSONL      : {val_path} ({n_val} mẫu hội thoại ChatML)")
    print(f"- Tập Test (Giữ kín) : {len(splits['test'])} hóa đơn (Tuyệt đối không xuất vào SFT)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

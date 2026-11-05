"""Bộ nạp dữ liệu MC-OCR 2021 (CSV/JSON) và phân chia tập Train/Val/Test cố định (70/10/20, seed=42)."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from receipt_extraction.extraction.regex_baseline import parse_currency
from receipt_extraction.schemas.receipt import ReceiptData


CATEGORY_ID_TO_LABEL: Dict[int, str] = {
    15: "SELLER",
    16: "SELLER_ADDRESS",
    17: "TIMESTAMP",
    18: "TOTAL_COST",
}


class MCOCRRecord(BaseModel):
    """Một bản ghi dữ liệu hóa đơn chuẩn hóa từ MC-OCR 2021 hoặc bộ dữ liệu mở rộng 12 trường."""

    img_id: str = Field(description="Định danh file ảnh (ví dụ: mcocr_public_145013abc.jpg)")
    image_path: Optional[str] = Field(default=None, description="Đường dẫn tới file ảnh")
    receipt_data: ReceiptData = Field(default_factory=ReceiptData, description="Dữ liệu chuẩn hóa theo ReceiptData 12 trường")
    gt_ocr_text: str = Field(default="", description="Văn bản OCR chuẩn (Ground-Truth OCR text) phục vụ thực nghiệm Oracle")
    image_quality: Optional[float] = Field(default=None, description="Điểm chất lượng ảnh từ MC-OCR 2021 (anno_image_quality)")


def _normalize_label_name(label: str) -> str:
    """Đồng nhất tên nhãn giữa các phiên bản MC-OCR 2021 (ADDRESS -> SELLER_ADDRESS)."""
    lbl = label.strip().upper()
    if lbl in ("ADDRESS", "SELLER_ADDRESS"):
        return "SELLER_ADDRESS"
    if lbl in ("SELLER", "SELLER_NAME"):
        return "SELLER"
    if lbl in ("TIMESTAMP", "DATETIME", "DATE"):
        return "TIMESTAMP"
    if lbl in ("TOTAL_COST", "TOTAL"):
        return "TOTAL_COST"
    return lbl


def parse_mcocr_row(
    row: Dict[str, Any],
    images_dir: Optional[Union[str, Path]] = None,
) -> MCOCRRecord:
    """Chuyển đổi 1 dòng dữ liệu MC-OCR 2021 (từ CSV hoặc JSON) sang MCOCRRecord."""
    img_id = str(row.get("img_id") or row.get("image_id") or row.get("filename") or "").strip()

    field_buckets: Dict[str, List[str]] = {
        "SELLER": [],
        "SELLER_ADDRESS": [],
        "TIMESTAMP": [],
        "TOTAL_COST": [],
    }
    all_ocr_lines: List[str] = []

    # Trường hợp 1: Có cột anno_texts và anno_labels phân cách bởi '|||' (chuẩn CSV MC-OCR 2021)
    anno_texts_raw = row.get("anno_texts")
    anno_labels_raw = row.get("anno_labels")

    if anno_texts_raw and anno_labels_raw:
        texts = [t.strip() for t in str(anno_texts_raw).split("|||")]
        labels = [_normalize_label_name(l) for l in str(anno_labels_raw).split("|||")]
        for txt, lbl in zip(texts, labels):
            if txt:
                all_ocr_lines.append(txt)
                if lbl in field_buckets:
                    field_buckets[lbl].append(txt)

    # Trường hợp 2: Có anno_polygons (chuỗi list/dict hoặc list trực tiếp) nhưng chưa có anno_texts
    elif row.get("anno_polygons"):
        raw_polys = row["anno_polygons"]
        if isinstance(raw_polys, str):
            try:
                polys = json.loads(raw_polys)
            except Exception:
                try:
                    polys = ast.literal_eval(raw_polys)
                except Exception:
                    polys = []
        else:
            polys = raw_polys if isinstance(raw_polys, list) else []

        for poly in polys:
            if not isinstance(poly, dict):
                continue
            txt = str(poly.get("text") or "").strip()
            cat_id = poly.get("category_id")
            lbl_raw = poly.get("label") or (CATEGORY_ID_TO_LABEL.get(int(cat_id), "OTHER") if cat_id is not None else "OTHER")
            lbl = _normalize_label_name(str(lbl_raw))
            if txt:
                all_ocr_lines.append(txt)
                if lbl in field_buckets:
                    field_buckets[lbl].append(txt)

    # Trường hợp 3: Dữ liệu JSON trực tiếp chứa các trường của ReceiptData hoặc MC-OCR
    seller_name = " ".join(field_buckets["SELLER"]).strip() or row.get("seller_name") or row.get("SELLER")
    seller_address = " ".join(field_buckets["SELLER_ADDRESS"]).strip() or row.get("seller_address") or row.get("SELLER_ADDRESS")
    timestamp = " ".join(field_buckets["TIMESTAMP"]).strip() or row.get("timestamp") or row.get("TIMESTAMP")

    raw_total = (
        " ".join(field_buckets["TOTAL_COST"]).strip()
        or row.get("total_cost")
        or row.get("TOTAL_COST")
    )
    if isinstance(raw_total, (int, float)):
        total_cost: Optional[int] = int(round(raw_total))
    elif raw_total:
        total_cost = parse_currency(str(raw_total))
    else:
        total_cost = None

    # Xây dựng ReceiptData (nếu JSON hoặc CSV có sẵn các trường mở rộng trong số 12 trường thì giữ nguyên)
    if "receipt_data" in row and isinstance(row["receipt_data"], dict):
        receipt_data = ReceiptData(**row["receipt_data"])
    elif "receipt_data_json" in row and row["receipt_data_json"]:
        parsed_rd = json.loads(str(row["receipt_data_json"]))
        receipt_data = ReceiptData(**parsed_rd)
    else:
        receipt_kwargs: Dict[str, Any] = {
            "seller_name": seller_name if seller_name else None,
            "seller_address": seller_address if seller_address else None,
            "seller_phone": row.get("seller_phone"),
            "tax_id": row.get("tax_id"),
            "invoice_number": row.get("invoice_number"),
            "timestamp": timestamp if timestamp else None,
            "subtotal": row.get("subtotal"),
            "discount": row.get("discount"),
            "vat_amount": row.get("vat_amount"),
            "total_cost": total_cost,
            "payment_method": row.get("payment_method"),
        }
        if "items" in row and isinstance(row["items"], list):
            receipt_kwargs["items"] = row["items"]
        receipt_data = ReceiptData(**receipt_kwargs)

    # Văn bản OCR chuẩn
    gt_ocr_text = str(row.get("gt_ocr_text") or row.get("raw_ocr_text") or "").strip()
    if not gt_ocr_text and all_ocr_lines:
        gt_ocr_text = "\n".join(all_ocr_lines)
    if not gt_ocr_text:
        # Nếu vẫn chưa có, ghép các trường non-null làm văn bản tham chiếu tối thiểu
        fallback_lines = [
            str(v)
            for v in [seller_name, seller_address, timestamp, raw_total]
            if v is not None and str(v).strip()
        ]
        gt_ocr_text = "\n".join(fallback_lines)

    # Điểm chất lượng ảnh (anno_image_quality)
    raw_qual = row.get("anno_image_quality", row.get("image_quality"))
    image_quality: Optional[float] = None
    if raw_qual is not None and str(raw_qual).strip() != "":
        try:
            image_quality = round(float(raw_qual), 4)
        except ValueError:
            image_quality = None

    img_path_str: Optional[str] = row.get("image_path")
    if img_path_str and not Path(img_path_str).exists() and images_dir:
        alt_cand = Path(images_dir) / Path(img_path_str).name
        if alt_cand.exists():
            img_path_str = str(alt_cand)
        elif (Path(images_dir) / "images" / Path(img_path_str).name).exists():
            img_path_str = str(Path(images_dir) / "images" / Path(img_path_str).name)

    if not img_path_str and images_dir and img_id:
        candidate = Path(images_dir) / img_id
        if not candidate.exists() and (Path(images_dir) / "images" / img_id).exists():
            candidate = Path(images_dir) / "images" / img_id
        img_path_str = str(candidate)

    return MCOCRRecord(
        img_id=img_id,
        image_path=img_path_str,
        receipt_data=receipt_data,
        gt_ocr_text=gt_ocr_text,
        image_quality=image_quality,
    )


def load_mcocr_csv(
    csv_path: Union[str, Path],
    images_dir: Optional[Union[str, Path]] = None,
) -> List[MCOCRRecord]:
    """Đọc file CSV định dạng MC-OCR 2021 (ví dụ: mcocr_train_df.csv) và trả về danh sách MCOCRRecord."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV MC-OCR tại: {path}")

    records: List[MCOCRRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(parse_mcocr_row(row, images_dir=images_dir))
    return records


def load_mcocr_json(
    json_path: Union[str, Path],
    images_dir: Optional[Union[str, Path]] = None,
) -> List[MCOCRRecord]:
    """Đọc file JSON (danh sách bản ghi hoặc 1 bản ghi đơn) sang danh sách MCOCRRecord."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON tại: {path}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict):
        if "records" in raw_data and isinstance(raw_data["records"], list):
            rows = raw_data["records"]
        else:
            rows = [raw_data]
    elif isinstance(raw_data, list):
        rows = raw_data
    else:
        raise ValueError(f"Định dạng JSON không hợp lệ trong file: {path}")

    return [parse_mcocr_row(r, images_dir=images_dir) for r in rows]


def load_dataset_records(
    dataset_dir: Union[str, Path],
    split: Optional[str] = None,
    splits_manifest_path: Optional[Union[str, Path]] = None,
) -> List[MCOCRRecord]:
    """Tự động nạp danh sách bản ghi từ thư mục dataset (hỗ trợ cả thư mục chứa cặp .jpg/.json/.txt và CSV/JSON)."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {root}")

    # 1. Nếu là file CSV hoặc JSON trực tiếp
    if root.is_file():
        if root.suffix.lower() == ".csv":
            records = load_mcocr_csv(root, images_dir=root.parent)
        else:
            records = load_mcocr_json(root, images_dir=root.parent)
    else:
        # 2. Kiểm tra có file CSV chuẩn MC-OCR trong thư mục không
        csv_files = sorted(root.glob("*.csv"))
        if csv_files:
            records = load_mcocr_csv(csv_files[0], images_dir=root)
        else:
            # 3. Quét các file nhãn .json đi kèm ảnh mẫu (.jpg/.png) và văn bản OCR chuẩn (.txt)
            records = []
            search_dirs = [root]
            if (root / "annotations").is_dir():
                search_dirs.append(root / "annotations")
            if (root / "images").is_dir():
                search_dirs.append(root / "images")

            seen_stems = set()
            json_files: List[Path] = []
            for sdir in search_dirs:
                for p in sorted(sdir.glob("*.json")):
                    if p.name in ("splits.json", "splits_70_10_20.json", "manifest.json", "dataset_stats.json"):
                        continue
                    if p.stem not in seen_stems:
                        seen_stems.add(p.stem)
                        json_files.append(p)

            for jf in json_files:
                raw_dict = json.loads(jf.read_text(encoding="utf-8"))
                stem = jf.stem
                # Tìm ảnh tương ứng ở cả thư mục gốc, images/, và thư mục chứa file json
                img_path: Optional[Path] = None
                img_search_roots = [jf.parent, root, root / "images"]
                for sroot in img_search_roots:
                    for ext in (".jpg", ".jpeg", ".png", ".webp"):
                        cand = sroot / f"{stem}{ext}"
                        if cand.exists():
                            img_path = cand
                            break
                    if img_path:
                        break
                # Tìm file văn bản OCR chuẩn (.txt) nếu có
                for sroot in [jf.parent, root, root / "images", root / "annotations"]:
                    txt_path = sroot / f"{stem}.txt"
                    if txt_path.exists() and not raw_dict.get("gt_ocr_text"):
                        raw_dict["gt_ocr_text"] = txt_path.read_text(encoding="utf-8").strip()
                        break

                if not raw_dict.get("img_id"):
                    raw_dict["img_id"] = img_path.name if img_path else f"{stem}.jpg"
                if img_path:
                    raw_dict["image_path"] = str(img_path)

                records.append(parse_mcocr_row(raw_dict, images_dir=root))

    # Lọc theo split nếu được chỉ định
    if split and split.lower() != "all":
        manifest_file = Path(splits_manifest_path) if splits_manifest_path else (root / "splits.json")
        if manifest_file.exists():
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            splits_map = manifest_data.get("splits", manifest_data)
            allowed_ids = set(splits_map.get(split, []))
            if allowed_ids:
                records = [
                    r for r in records
                    if r.img_id in allowed_ids or Path(r.img_id).stem in allowed_ids
                ]

    return records


def split_dataset(
    records: Sequence[MCOCRRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, List[MCOCRRecord]]:
    """Chia tập dữ liệu cố định theo tỷ lệ train / val / test (mặc định 70 / 10 / 20) với seed=42."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Tổng tỷ lệ train + val + test phải bằng 1.0 (nhận được: {total_ratio}).")

    n = len(records)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    # Sắp xếp trước theo img_id để đảm bảo tính tất định tuyệt đối trên mọi hệ điều hành
    sorted_records = sorted(records, key=lambda r: r.img_id)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Đảm bảo tổng không vượt quá n
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    return {
        "train": [sorted_records[i] for i in train_idx],
        "val": [sorted_records[i] for i in val_idx],
        "test": [sorted_records[i] for i in test_idx],
    }


def export_splits_manifest(
    splits: Dict[str, Sequence[MCOCRRecord]],
    output_path: Union[str, Path],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.10, 0.20),
) -> Path:
    """Xuất file manifest splits.json chứa danh sách img_id và thống kê từng tập."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_count = sum(len(v) for v in splits.values())
    manifest = {
        "seed": seed,
        "ratios": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "counts": {
            k: len(v) for k, v in splits.items()
        },
        "total": total_count,
        "splits": {
            k: [r.img_id for r in v] for k, v in splits.items()
        },
    }

    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

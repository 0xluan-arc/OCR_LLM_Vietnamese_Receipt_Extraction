r"""Kịch bản quy hoạch kho dữ liệu tập trung tại D:\NCKH\Data_HoaDon_NCKH và tải ảnh hóa đơn chụp thực tế
từ Hugging Face Datasets Server API (hoặc sinh ảnh chụp thực tế có biến dạng quang học/nhiệt khi offline).

Cấu trúc 4 phân khu rõ ràng tại D:\NCKH\Data_HoaDon_NCKH:
1. 01_MC_OCR_2021_Chuan_CuocThi/
   - images/ (hoadon_thucte_001.jpg, hoadon_thucte_001.json, hoadon_thucte_001.txt, ...)
   - Đồng bộ sang D:\NCKH\OCR_LLM_Vietnamese_Receipt_Extraction\data\raw\real_receipts\
2. 02_HoaDon_ThucTe_TuChup_300Anh/
   - images/, annotations/, README_HuongDanBoAnh.md
3. 03_HoaDon_MoPhong_KiemThu_10Mau/
   - Sao chép 10 mẫu kiểm thử phần mềm (01_ đến 10_) tách biệt hoàn toàn với ảnh thật.
4. 04_DuLieu_DaChiaTap_Train_Val_Test/
   - splits_70_10_20.json
   - BoNho_CMS_FewShot_TrainOnly.jsonl (Continuum Memory System - 0% Data Leakage)
   - QLoRA_Train_Dataset.jsonl & QLoRA_Val_Dataset.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_sft_dataset import export_split_to_chatml
from receipt_extraction.config import config
from receipt_extraction.data.mcocr_loader import (
    MCOCRRecord,
    export_splits_manifest,
    load_dataset_records,
    parse_mcocr_row,
    split_dataset,
)
from receipt_extraction.memory.cms_store import ContinuumMemoryStore, MemoryEpisode
from receipt_extraction.ocr.tesseract_ocr import TesseractOCREngine
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem


HF_PUBLIC_RECEIPT_DATASETS = [
    # Các bộ dữ liệu hóa đơn chụp thực tế trên Hugging Face Datasets Server API
    "https://datasets-server.huggingface.co/rows?dataset=darentang%2Fsroie&config=darentang--sroie&split=test&offset=0&length=15",
    "https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix%2Fcord-v2&config=default&split=test&offset=0&length=15",
    "https://datasets-server.huggingface.co/rows?dataset=DThai%2Fmcocr-test-1&config=default&split=train&offset=0&length=15",
]


def setup_data_hub_directories(hub_root: Path) -> Dict[str, Path]:
    r"""Khởi tạo 4 thư mục con chuẩn hóa trong D:\NCKH\Data_HoaDon_NCKH."""
    dir_01 = hub_root / "01_MC_OCR_2021_Chuan_CuocThi"
    dir_01_images = dir_01 / "images"
    dir_02 = hub_root / "02_HoaDon_ThucTe_TuChup_300Anh"
    dir_03 = hub_root / "03_HoaDon_MoPhong_KiemThu_10Mau"
    dir_04 = hub_root / "04_DuLieu_DaChiaTap_Train_Val_Test"

    for d in (
        dir_01,
        dir_01_images,
        dir_02 / "images",
        dir_02 / "annotations",
        dir_03,
        dir_04,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Tạo hướng dẫn bỏ ảnh cho nhóm sinh viên tại 02_HoaDon_ThucTe_TuChup_300Anh
    readme_02 = dir_02 / "README_HuongDanBoAnh.md"
    readme_02.write_text(
        "# HƯỚNG DẪN LƯU TRỮ 300 ẢNH HÓA ĐƠN BÁN LẺ TỰ CHỤP\n\n"
        "Thư mục này dành riêng để lưu trữ **300 ảnh hóa đơn bán lẻ tiếng Việt tự chụp thực tế** "
        "(siêu thị, cửa hàng tiện lợi, quán cà phê, nhà thuốc, quán ăn).\n\n"
        "## 1. Cấu trúc thư mục\n"
        "- `images/`: Nơi bỏ các file ảnh gốc chụp bằng điện thoại (`.jpg`, `.png`).\n"
        "  - Quy tắc đặt tên khuyến nghị: `tuchup_001.jpg`, `tuchup_002.jpg`, ..., `tuchup_300.jpg`.\n"
        "- `annotations/`: Nơi lưu file nhãn chuẩn (`.json`) và văn bản OCR chuẩn (`.txt`) tương ứng.\n\n"
        "## 2. Tiêu chuẩn chụp & ẩn danh dữ liệu\n"
        "1. Chụp rõ toàn bộ mặt hóa đơn (bao gồm cả hóa đơn phẳng và hóa đơn hơi nhăn/nghiêng thực tế).\n"
        "2. Che thông tin cá nhân nhạy cảm (4 số cuối thẻ ngân hàng, tên/SĐT khách hàng cá nhân) trước khi đưa vào kho.\n"
        "3. Tuân thủ quy ước gán nhãn 12 trường tại `docs/annotation_guideline_v1.md`.\n",
        encoding="utf-8",
    )

    return {
        "hub_root": hub_root,
        "dir_01_mcocr": dir_01,
        "dir_01_images": dir_01_images,
        "dir_02_self_captured": dir_02,
        "dir_03_synthetic_10": dir_03,
        "dir_04_splits_cms": dir_04,
    }


def copy_synthetic_10_samples(samples_src: Path, dir_03_dst: Path) -> int:
    """Sao chép 10 mẫu kiểm thử phần mềm (01_ đến 10_) sang 03_HoaDon_MoPhong_KiemThu_10Mau."""
    count = 0
    for ext in ("*.jpg", "*.json", "*.txt"):
        for src_file in sorted(samples_src.glob(ext)):
            if src_file.name in ("splits.json", "manifest.json"):
                continue
            dst_file = dir_03_dst / src_file.name
            shutil.copy2(src_file, dst_file)
            if ext == "*.json":
                count += 1
    return count


def _parse_cord_ground_truth(gt_str: str, idx: int) -> Tuple[ReceiptData, str]:
    """Chuyển đổi nhãn JSON gốc của CORD-v2 (ảnh chụp hóa đơn thực tế) sang ReceiptData 12 trường."""
    obj = json.loads(gt_str)
    gt_parse = obj.get("gt_parse", {})

    items_list: List[ReceiptItem] = []
    raw_lines: List[str] = []

    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]

    for m in menu:
        if not isinstance(m, dict):
            continue
        nm = str(m.get("nm") or "").strip()
        cnt_raw = str(m.get("cnt") or "1").strip()
        price_raw = str(m.get("price") or "0").strip()
        unit_raw = str(m.get("unitprice") or price_raw).strip()

        qty = 1.0
        qty_m = re.search(r"(\d+(?:\.\d+)?)", cnt_raw)
        if qty_m:
            qty = float(qty_m.group(1))

        digits_tot = re.sub(r"[^\d]", "", price_raw)
        tot_p = int(digits_tot) if digits_tot else 0
        digits_unit = re.sub(r"[^\d]", "", unit_raw)
        unit_p = int(digits_unit) if digits_unit else (int(round(tot_p / max(qty, 1.0))) if tot_p else 0)

        if nm and tot_p > 0:
            items_list.append(
                ReceiptItem(
                    item_name=nm,
                    quantity=qty,
                    unit_price=unit_p,
                    total_price=tot_p,
                )
            )
            raw_lines.append(f"{nm} {int(qty) if qty.is_integer() else qty} {unit_p:,} {tot_p:,}")

    sub_dict = gt_parse.get("sub_total", {}) or {}
    tot_dict = gt_parse.get("total", {}) or {}

    def _clean_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        s = re.sub(r"[^\d]", "", str(v))
        return int(s) if s else None

    subtotal = _clean_int(sub_dict.get("subtotal_price"))
    discount = _clean_int(sub_dict.get("discount_price"))
    vat_amount = _clean_int(sub_dict.get("tax_price"))
    total_cost = _clean_int(tot_dict.get("total_price"))

    if total_cost is None and items_list:
        total_cost = sum(it.total_price for it in items_list)
    if subtotal is None and total_cost is not None:
        subtotal = total_cost + (discount or 0) - (vat_amount or 0)

    # Trích xuất các dòng chữ thực tế từ valid_line của CORD
    valid_lines = obj.get("valid_line", [])
    cord_text_lines: List[str] = []
    seller_candidates: List[str] = []
    for vl in valid_lines:
        words = vl.get("words", [])
        line_str = " ".join(str(w.get("text", "")).strip() for w in words if str(w.get("text", "")).strip())
        if line_str:
            cord_text_lines.append(line_str)
            if vl.get("category") in ("store_name", "seller"):
                seller_candidates.append(line_str)

    seller_name = seller_candidates[0] if seller_candidates else (cord_text_lines[0] if cord_text_lines else f"MERCHANT STORE #{idx:03d}")
    gt_ocr_text = "\n".join(cord_text_lines) if cord_text_lines else "\n".join([seller_name] + raw_lines + [f"TOTAL: {total_cost or 0}"])

    rdata = ReceiptData(
        seller_name=seller_name,
        seller_address=None,
        seller_phone=None,
        tax_id=None,
        invoice_number=None,
        timestamp=None,
        items=items_list,
        subtotal=subtotal,
        discount=discount,
        vat_amount=vat_amount,
        total_cost=total_cost,
        payment_method="CASH" if tot_dict.get("cashprice") else None,
    )
    return rdata, gt_ocr_text


def download_from_huggingface_api(
    dst_images_dir: Path,
    max_samples: int = 10,
    timeout: int = 20,
) -> Tuple[bool, str, List[Path]]:
    """Tải trực tiếp các mẫu ảnh hóa đơn chụp thực tế từ Hugging Face Datasets Server API."""
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    saved_json_paths: List[Path] = []

    # Thử tải từ CORD-v2 hoặc MC-OCR trên Hugging Face Datasets Server
    url = "https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix%2Fcord-v2&config=default&split=test&offset=0&length=15"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (NCKH-Receipt-Downloader/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}", []
            payload = json.loads(resp.read().decode("utf-8"))
            rows = payload.get("rows", [])

            count = 0
            for item in rows:
                if count >= max_samples:
                    break
                row = item.get("row", {})
                img_info = row.get("image", {})
                img_src_url = img_info.get("src") if isinstance(img_info, dict) else None
                gt_str = row.get("ground_truth", "")
                if not img_src_url or not gt_str:
                    continue

                idx = count + 1
                stem = f"hoadon_thucte_{idx:03d}"
                img_path = dst_images_dir / f"{stem}.jpg"
                json_path = dst_images_dir / f"{stem}.json"
                txt_path = dst_images_dir / f"{stem}.txt"

                img_req = urllib.request.Request(
                    img_src_url,
                    headers={"User-Agent": "Mozilla/5.0 (NCKH-Receipt-Downloader/1.0)"},
                )
                with urllib.request.urlopen(img_req, timeout=timeout) as img_resp:
                    img_bytes = img_resp.read()
                    pil_im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_im.save(str(img_path), format="JPEG", quality=92)

                rdata, gt_ocr_text = _parse_cord_ground_truth(gt_str, idx)
                txt_path.write_text(gt_ocr_text, encoding="utf-8")

                record_dict = {
                    **rdata.model_dump(),
                    "img_id": img_path.name,
                    "image_path": str(img_path),
                    "gt_ocr_text": gt_ocr_text,
                    "source_dataset": "huggingface_real_camera_receipt",
                }
                json_path.write_text(json.dumps(record_dict, ensure_ascii=False, indent=2), encoding="utf-8")
                saved_json_paths.append(json_path)
                count += 1

            if saved_json_paths:
                return True, f"Đã tải thành công {len(saved_json_paths)} ảnh hóa đơn chụp thực tế từ Hugging Face Hub.", saved_json_paths
    except Exception as exc:
        return False, f"Lỗi kết nối Hugging Face ({exc})", []

    return False, "Không tìm thấy mẫu hợp lệ từ API.", []


def generate_realistic_camera_vietnamese_receipts(
    dst_images_dir: Path,
    num_samples: int = 10,
) -> List[Path]:
    """Tạo bộ ảnh hóa đơn bán lẻ tiếng Việt mô phỏng sát điều kiện chụp thực tế của MC-OCR 2021
    (biến dạng phối cảnh 3D nhẹ, bóng đổ ánh sáng không đều, nhiễu hạt cảm biến camera, mực in nhiệt mờ)
    khi chưa tải được từ Internet hoặc cần bổ sung mẫu tiếng Việt chuẩn 12 trường.
    """
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    real_specs: List[Dict[str, Any]] = [
        {
            "seller_name": "CO.OPMART CONG QUYNH",
            "seller_address": "189C Cong Quynh, P. Nguyen Cu Trinh, Q1, TP.HCM",
            "seller_phone": "028 3925 6789",
            "tax_id": "0301175688",
            "invoice_number": "CM-20261018-0101",
            "timestamp": "18/10/2026 09:15:20",
            "items": [
                {"item_name": "Dau an Neptune Light 1L", "quantity": 1.0, "unit_price": 62000, "total_price": 62000},
                {"item_name": "Trung ga ta Ba Huan H10", "quantity": 2.0, "unit_price": 35000, "total_price": 70000},
                {"item_name": "Mi Hao Hao tom chua cay", "quantity": 5.0, "unit_price": 4500, "total_price": 22500},
            ],
            "subtotal": 154500,
            "discount": 4500,
            "vat_amount": 12000,
            "total_cost": 162000,
            "payment_method": "Tien mat",
            "angle": -1.2,
        },
        {
            "seller_name": "BACH HOA XANH QUAN TAN BINH",
            "seller_address": "312 Cong Hoa, P.13, Q. Tan Binh, TP.HCM",
            "seller_phone": "1900 1908",
            "tax_id": "0313049871",
            "invoice_number": "BHX-882104",
            "timestamp": "18/10/2026 10:42:11",
            "items": [
                {"item_name": "Thit ba roi heo CP 500g", "quantity": 1.0, "unit_price": 89000, "total_price": 89000},
                {"item_name": "Cai ngot VietGAP 500g", "quantity": 2.0, "unit_price": 14000, "total_price": 28000},
            ],
            "subtotal": 117000,
            "discount": 7000,
            "vat_amount": 8800,
            "total_cost": 118800,
            "payment_method": "MoMo",
            "angle": 1.4,
        },
        {
            "seller_name": "PHUC LONG COFFEE & TEA EXPRESS",
            "seller_address": "42 Ngo Duc Ke, P. Ben Nghe, Q1, TP.HCM",
            "seller_phone": "028 6263 0377",
            "tax_id": "0312567890",
            "invoice_number": "PL-2026-0451",
            "timestamp": "18/10/2026 13:20:05",
            "items": [
                {"item_name": "Tra Dao Cam Sa Size L", "quantity": 2.0, "unit_price": 65000, "total_price": 130000},
                {"item_name": "Banh Croissant Bo Phap", "quantity": 1.0, "unit_price": 38000, "total_price": 38000},
            ],
            "subtotal": 168000,
            "discount": 18000,
            "vat_amount": 12000,
            "total_cost": 162000,
            "payment_method": "VNPay",
            "angle": -0.8,
        },
        {
            "seller_name": "NHA THUOC AN KHANG QUAN 10",
            "seller_address": "245 Su Van Hanh, P.9, Q10, TP.HCM",
            "seller_phone": "1900 1572",
            "tax_id": "0314889922",
            "invoice_number": "AK-20261018-09",
            "timestamp": "18/10/2026 16:05:40",
            "items": [
                {"item_name": "Vien sui Berocca Performance", "quantity": 1.0, "unit_price": 98000, "total_price": 98000},
                {"item_name": "Nuoc muoi sinh ly Natri Clorid", "quantity": 3.0, "unit_price": 10000, "total_price": 30000},
            ],
            "subtotal": 128000,
            "discount": 0,
            "vat_amount": 6400,
            "total_cost": 134400,
            "payment_method": "Tien mat",
            "angle": 1.1,
        },
        {
            "seller_name": "FAMILYMART NGUYEN DINH CHIEU",
            "seller_address": "180 Nguyen Dinh Chieu, P.6, Q3, TP.HCM",
            "seller_phone": "028 3930 5566",
            "tax_id": "0311998877",
            "invoice_number": "FM-2026-7712",
            "timestamp": "19/10/2026 07:50:12",
            "items": [
                {"item_name": "Com nam ca hoi Mayo", "quantity": 2.0, "unit_price": 22000, "total_price": 44000},
                {"item_name": "Sua dau nanh Fami 200ml", "quantity": 2.0, "unit_price": 8000, "total_price": 16000},
            ],
            "subtotal": 60000,
            "discount": 0,
            "vat_amount": 4800,
            "total_cost": 64800,
            "payment_method": "Tien mat",
            "angle": -1.5,
        },
        {
            "seller_name": "SIEU THI LOTTE MART NAM SAI GON",
            "seller_address": "469 Nguyen Huu Tho, P. Tan Hung, Q7, TP.HCM",
            "seller_phone": "028 3775 3232",
            "tax_id": "0305678123",
            "invoice_number": "LM-20261019-33",
            "timestamp": "19/10/2026 11:30:00",
            "items": [
                {"item_name": "Nuoc giat OMO Matic 3.6kg", "quantity": 1.0, "unit_price": 195000, "total_price": 195000},
                {"item_name": "Giay ve sinh Pulppy 10 cuon", "quantity": 1.0, "unit_price": 85000, "total_price": 85000},
            ],
            "subtotal": 280000,
            "discount": 20000,
            "vat_amount": 20800,
            "total_cost": 280800,
            "payment_method": "The tin dung",
            "angle": 0.9,
        },
        {
            "seller_name": "THE COFFEE HOUSE CAO THANG",
            "seller_address": "86 Cao Thang, P.4, Q3, TP.HCM",
            "seller_phone": "1800 6936",
            "tax_id": "0312890123",
            "invoice_number": "TCH-2026-5501",
            "timestamp": "19/10/2026 15:12:45",
            "items": [
                {"item_name": "Hi-Tea Vai Size M", "quantity": 2.0, "unit_price": 49000, "total_price": 98000},
                {"item_name": "Banh Mi Que Pate Nong", "quantity": 2.0, "unit_price": 19000, "total_price": 38000},
            ],
            "subtotal": 136000,
            "discount": 16000,
            "vat_amount": 9600,
            "total_cost": 129600,
            "payment_method": "ZaloPay",
            "angle": -1.0,
        },
        {
            "seller_name": "GS25 CONG HOA TAN BINH",
            "seller_address": "19A Cong Hoa, P.12, Q. Tan Binh, TP.HCM",
            "seller_phone": "028 3811 2525",
            "tax_id": "0314556677",
            "invoice_number": "GS-20261020-18",
            "timestamp": "20/10/2026 08:25:30",
            "items": [
                {"item_name": "Mi tron Indomie trung op la", "quantity": 1.0, "unit_price": 32000, "total_price": 32000},
                {"item_name": "Tra xanh Khong Do 455ml", "quantity": 2.0, "unit_price": 12000, "total_price": 24000},
            ],
            "subtotal": 56000,
            "discount": 6000,
            "vat_amount": 4000,
            "total_cost": 54000,
            "payment_method": "Tien mat",
            "angle": 1.3,
        },
        {
            "seller_name": "GUARDIAN HEALTH & BEAUTY Q1",
            "seller_address": "116 Nguyen Hue, P. Ben Nghe, Q1, TP.HCM",
            "seller_phone": "1900 4449",
            "tax_id": "0310987654",
            "invoice_number": "GD-2026-9012",
            "timestamp": "20/10/2026 18:40:00",
            "items": [
                {"item_name": "Sua rua mat Cerave 236ml", "quantity": 1.0, "unit_price": 310000, "total_price": 310000},
                {"item_name": "Bong tay trang Silcot 82M", "quantity": 2.0, "unit_price": 42000, "total_price": 84000},
            ],
            "subtotal": 394000,
            "discount": 34000,
            "vat_amount": 28800,
            "total_cost": 388800,
            "payment_method": "Chuyen khoan",
            "angle": -0.7,
        },
        {
            "seller_name": "MINISTOP NGUYEN GIA TRI",
            "seller_address": "120 Nguyen Gia Tri, P.25, Q. Binh Thanh, TP.HCM",
            "seller_phone": "028 3512 9090",
            "tax_id": "0311234987",
            "invoice_number": "MS-20261021-77",
            "timestamp": "21/10/2026 21:05:10",
            "items": [
                {"item_name": "Ga ran Karage xau 3 vien", "quantity": 2.0, "unit_price": 25000, "total_price": 50000},
                {"item_name": "Nuoc ngot Pepsi Zero 320ml", "quantity": 2.0, "unit_price": 11000, "total_price": 22000},
            ],
            "subtotal": 72000,
            "discount": 2000,
            "vat_amount": 5600,
            "total_cost": 75600,
            "payment_method": "Tien mat",
            "angle": 1.0,
        },
    ]

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    saved_paths: List[Path] = []
    rng = np.random.default_rng(2026)

    for idx, spec in enumerate(real_specs[:num_samples], start=1):
        stem = f"hoadon_thucte_{idx:03d}"
        img_path = dst_images_dir / f"{stem}.jpg"
        json_path = dst_images_dir / f"{stem}.json"
        txt_path = dst_images_dir / f"{stem}.txt"

        def _fmt_vnd(n: int) -> str:
            return f"{n:,}".replace(",", ".")

        lines = [
            spec["seller_name"],
            f"Dia chi: {spec['seller_address']}",
            f"SDT: {spec['seller_phone']}",
            f"MST: {spec['tax_id']}",
            f"So HD: {spec['invoice_number']}",
            f"Ngay: {spec['timestamp']}",
        ]
        for it in spec["items"]:
            q = int(it["quantity"]) if float(it["quantity"]).is_integer() else it["quantity"]
            lines.append(f"{it['item_name']} {q} {_fmt_vnd(it['unit_price'])} {_fmt_vnd(it['total_price'])}")
        lines.append(f"Tam tinh: {_fmt_vnd(spec['subtotal'])}")
        if spec["discount"]:
            lines.append(f"Giam gia: -{_fmt_vnd(spec['discount'])}")
        if spec["vat_amount"]:
            lines.append(f"Thue VAT (8%): {_fmt_vnd(spec['vat_amount'])}")
        lines.append(f"Tong thanh toan: {_fmt_vnd(spec['total_cost'])}")
        lines.append(f"Hinh thuc thanh toan: {spec['payment_method']}")

        gt_ocr_text = "\n".join(lines)

        # Vẽ ảnh hóa đơn nhiệt với hiệu ứng ánh sáng camera thực tế + nghiêng nhẹ + hạt cảm biến
        width, height = 820, 680
        canvas = Image.new("RGB", (width, height), color=(246, 244, 238))
        draw = ImageDraw.Draw(canvas)
        y = 35
        for line in lines:
            draw.text((45, y), line, fill=(28, 28, 32), font=font)
            y += 38

        # Tạo hiệu ứng chiếu sáng không đều (vignette / camera shadow) và xoay nhẹ như ảnh chụp tay
        arr = np.array(canvas, dtype=np.float32)
        yy, xx = np.mgrid[0:height, 0:width]
        illum = 1.0 - 0.10 * ((xx - width * 0.45) ** 2 / (width**2) + (yy - height * 0.5) ** 2 / (height**2))
        arr = arr * illum[:, :, None]
        noise = rng.normal(0.0, 3.5, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        center = (width // 2, height // 2)
        rot_mat = cv2.getRotationMatrix2D(center, float(spec["angle"]), 1.0)
        rotated = cv2.warpAffine(arr, rot_mat, (width, height), borderValue=(235, 233, 228))
        Image.fromarray(rotated).save(str(img_path), format="JPEG", quality=90)

        rdata = ReceiptData(
            seller_name=spec["seller_name"],
            seller_address=spec["seller_address"],
            seller_phone=spec["seller_phone"],
            tax_id=spec["tax_id"],
            invoice_number=spec["invoice_number"],
            timestamp=spec["timestamp"],
            items=[ReceiptItem(**it) for it in spec["items"]],
            subtotal=spec["subtotal"],
            discount=spec["discount"] if spec["discount"] else None,
            vat_amount=spec["vat_amount"] if spec["vat_amount"] else None,
            total_cost=spec["total_cost"],
            payment_method=spec["payment_method"],
        )

        txt_path.write_text(gt_ocr_text, encoding="utf-8")
        payload = {
            **rdata.model_dump(),
            "img_id": img_path.name,
            "image_path": str(img_path),
            "gt_ocr_text": gt_ocr_text,
            "anno_image_quality": 0.88,
            "source_type": "synthetic_test",
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths.append(json_path)

    return saved_paths


def ingest_official_mcocr_archive_if_present(dir_01_mcocr: Path) -> Tuple[bool, int, str]:
    """Tự động giải nén và nạp bộ dữ liệu MC-OCR 2021 thật (1.155 ảnh + mcocr_train_df.csv)
    ngay khi người dùng đặt file .zip hoặc thư mục giải nén vào 01_MC_OCR_2021_Chuan_CuocThi.
    """
    import zipfile
    from receipt_extraction.data.mcocr_loader import load_mcocr_csv

    zip_files = sorted(dir_01_mcocr.glob("*.zip"))
    for zf in zip_files:
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dir_01_mcocr / "extracted_mcocr")
        except Exception:
            continue

    # Tìm file mcocr_train_df.csv thật (có số lượng bản ghi > 50, chuẩn cuộc thi là 1.155 ảnh)
    csv_candidates = list(dir_01_mcocr.rglob("*mcocr*train*.csv"))
    for csv_file in csv_candidates:
        try:
            recs = load_mcocr_csv(csv_file, images_dir=csv_file.parent)
            if len(recs) >= 50:
                return True, len(recs), f"Đã nạp thành công {len(recs)} ảnh MC-OCR 2021 thật từ {csv_file}"
        except Exception:
            continue

    return False, 0, "Chưa phát hiện gói zip 1.155 ảnh gốc của cuộc thi MC-OCR 2021 trong thư mục."


def sync_to_project_real_receipts(src_images_dir: Path, dst_project_dir: Path) -> None:
    """Đồng bộ hóa các mẫu hoadon_thucte_* sang data/raw/real_receipts trong dự án."""
    dst_project_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.jpg", "*.json", "*.txt"):
        for f in sorted(src_images_dir.glob(ext)):
            if f.name.startswith("hoadon_thucte_"):
                shutil.copy2(f, dst_project_dir / f.name)


def build_splits_and_cms_in_dir04(
    records: List[MCOCRRecord],
    dir_04: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Chia tập 70/10/20 (seed=42) và lưu toàn bộ tệp chuẩn hóa vào 04_DuLieu_DaChiaTap_Train_Val_Test."""
    splits = split_dataset(records, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=seed)

    splits_path = export_splits_manifest(splits, dir_04 / "splits_70_10_20.json", seed=seed)

    # Nạp DUY NHẤT tập Train vào BoNho_CMS_FewShot_TrainOnly.jsonl (0% Data Leakage)
    cms_path = dir_04 / "BoNho_CMS_FewShot_TrainOnly.jsonl"
    if cms_path.exists():
        cms_path.unlink()

    store = ContinuumMemoryStore(
        memory_bank_path=cms_path,
        samples_dir=dir_04 / "nonexistent_prevent_seed",
        auto_init=False,
    )
    store.episodes = []
    store._ngram_cache = {}
    for rec in splits["train"]:
        store.episodes.append(
            MemoryEpisode(
                episode_id=rec.img_id,
                ocr_text=rec.gt_ocr_text,
                receipt_data=rec.receipt_data,
                layout_text="",
                source="train_split_70_10_20",
            )
        )
    store.persist()
    store.load()

    # Kiểm chứng 0% rò rỉ dữ liệu
    cms_ids = {ep.episode_id for ep in store.episodes}
    val_test_ids = {r.img_id for r in splits["val"]} | {r.img_id for r in splits["test"]}
    leaked = cms_ids & val_test_ids
    if leaked:
        raise RuntimeError(f"PHÁT HIỆN DATA LEAKAGE TRONG BoNho_CMS_FewShot_TrainOnly.jsonl: {leaked}")

    # Xuất dữ liệu huấn luyện QLoRA SFT
    qlora_train_path = dir_04 / "QLoRA_Train_Dataset.jsonl"
    qlora_val_path = dir_04 / "QLoRA_Val_Dataset.jsonl"
    n_q_train = export_split_to_chatml(splits["train"], qlora_train_path, augment_delta_noise=True)
    n_q_val = export_split_to_chatml(splits["val"], qlora_val_path, augment_delta_noise=False)

    return {
        "splits": splits,
        "splits_path": splits_path,
        "cms_path": cms_path,
        "cms_count": len(store.episodes),
        "leaked_count": len(leaked),
        "qlora_train_path": qlora_train_path,
        "qlora_train_count": n_q_train,
        "qlora_val_path": qlora_val_path,
        "qlora_val_count": n_q_val,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quy hoạch kho dữ liệu D:\\NCKH\\Data_HoaDon_NCKH và chuẩn bị dữ liệu hóa đơn thực tế."
    )
    parser.add_argument(
        "--hub-dir",
        type=str,
        default=str(config.paths.external_data_hub_dir),
        help="Đường dẫn kho dữ liệu tập trung (mặc định: D:\\NCKH\\Data_HoaDon_NCKH).",
    )
    parser.add_argument(
        "--vietnamese-camera-mode",
        action="store_true",
        help="Sử dụng bộ hóa đơn bán lẻ tiếng Việt 12 trường có biến dạng quang học/camera thực tế để đánh giá nhất quán schema tiếng Việt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed cố định chia tập 70/10/20 (mặc định: 42).",
    )
    args = parser.parse_args()

    hub_root = Path(args.hub_dir)
    dirs = setup_data_hub_directories(hub_root)

    # 1. Sao chép 10 mẫu kiểm thử phần mềm sang 03_HoaDon_MoPhong_KiemThu_10Mau
    n_syn = copy_synthetic_10_samples(PROJECT_ROOT / "data" / "samples", dirs["dir_03_synthetic_10"])

    # 2. Kiểm tra xem đã có bộ MC-OCR 2021 thật (1.155 ảnh) trong 01_MC_OCR_2021_Chuan_CuocThi chưa
    has_official, official_cnt, official_msg = ingest_official_mcocr_archive_if_present(dirs["dir_01_mcocr"])
    if has_official:
        source_msg = official_msg
    elif args.vietnamese_camera_mode:
        generate_realistic_camera_vietnamese_receipts(dirs["dir_01_images"], num_samples=10)
        source_msg = "Bộ 10 mẫu synthetic_test mô phỏng hiệu ứng quang học tiếng Việt (hoadon_thucte_001..010) — Chờ nạp 1.155 ảnh MC-OCR 2021 gốc"
    else:
        ok, msg, _ = download_from_huggingface_api(dirs["dir_01_images"], max_samples=10)
        if not ok:
            generate_realistic_camera_vietnamese_receipts(dirs["dir_01_images"], num_samples=10)
            source_msg = f"Bộ 10 mẫu synthetic_test mô phỏng quang học tiếng Việt ({msg})"
        else:
            source_msg = msg

    # Đồng bộ sang data/raw/real_receipts trong dự án
    project_real_dir = PROJECT_ROOT / "data" / "raw" / "real_receipts"
    sync_to_project_real_receipts(dirs["dir_01_images"], project_real_dir)

    # 3. Nạp các bản ghi từ 01_MC_OCR_2021_Chuan_CuocThi/images và chia tập 70/10/20 vào 04_DuLieu_DaChiaTap_Train_Val_Test
    records = load_dataset_records(dirs["dir_01_images"])
    summary_04 = build_splits_and_cms_in_dir04(records, dirs["dir_04_splits_cms"], seed=args.seed)

    # Xuất thêm splits.json ngay tại 01_MC_OCR_2021_Chuan_CuocThi/images để evaluate.py lọc theo --split
    export_splits_manifest(summary_04["splits"], dirs["dir_01_images"] / "splits.json", seed=args.seed)
    export_splits_manifest(summary_04["splits"], project_real_dir / "splits.json", seed=args.seed)

    print("=" * 75)
    print("✅ HOÀN TẤT QUY HOẠCH KHO DỮ LIỆU TẬP TRUNG: D:\\NCKH\\Data_HoaDon_NCKH")
    print("=" * 75)
    print(f"1. [01_MC_OCR_2021_Chuan_CuocThi]     : {dirs['dir_01_images']} ({len(records)} mẫu hoadon_thucte_*)")
    print(f"   - Ghi chú nguồn                    : {source_msg}")
    print(f"   - Đồng bộ trong dự án              : {project_real_dir}")
    print(f"2. [02_HoaDon_ThucTe_TuChup_300Anh]   : {dirs['dir_02_self_captured']} (Sẵn sàng đón 300 ảnh tự chụp)")
    print(f"3. [03_HoaDon_MoPhong_KiemThu_10Mau]  : {dirs['dir_03_synthetic_10']} ({n_syn} mẫu kiểm thử phần mềm)")
    print(f"4. [04_DuLieu_DaChiaTap_Train_Val_Test]: {dirs['dir_04_splits_cms']}")
    print(f"   - File chia tập (70/10/20, seed=42): {summary_04['splits_path'].name}")
    print(f"   - Kho bộ nhớ CMS (Train-Only)      : {summary_04['cms_path'].name} ({summary_04['cms_count']} episodes, {summary_04['leaked_count']} rò rỉ)")
    print(f"   - Tập huấn luyện QLoRA Train/Val   : {summary_04['qlora_train_path'].name} ({summary_04['qlora_train_count']} mẫu) / {summary_04['qlora_val_path'].name} ({summary_04['qlora_val_count']} mẫu)")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

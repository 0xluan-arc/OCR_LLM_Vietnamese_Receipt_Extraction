"""Script tạo bộ mẫu hóa đơn bán lẻ Việt Nam đa dạng kịch bản phục vụ Demo & Kiểm thử."""

import sys
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


SAMPLES_SPECS: List[Dict[str, Any]] = [
    {
        "filename": "01_circle_k_hop_le.jpg",
        "description": "Hóa đơn Cửa hàng tiện lợi Circle K (Tiếng Việt không dấu, đầy đủ trường, khớp số học 100%)",
        "bg_color": (255, 255, 255),
        "lines": [
            "SIEU THI TIEN LOI CIRCLE K",
            "Dia chi: 123 Nguyen Thi Minh Khai, Q1, TP.HCM",
            "MST: 0312456789",
            "So HD: CK-20261017-0892",
            "Ngay: 17/10/2026 08:30:15",
            "----------------------------------------",
            "Ca phe sua da 330ml   2   25.000   50.000",
            "Banh mi que pate      2   15.000   30.000",
            "Nuoc khoang Lavie     2   10.000   20.000",
            "----------------------------------------",
            "Tam tinh:                         100.000",
            "Giam gia:                          -5.000",
            "Thue VAT (8%):                      7.600",
            "Tong thanh toan:                  102.600",
            "Hinh thuc thanh toan: Tien mat"
        ]
    },
    {
        "filename": "02_winmart_co_dau_hop_le.jpg",
        "description": "Hóa đơn Siêu thị WinMart+ (Tiếng Việt có dấu đầy đủ, khớp số học 100%)",
        "bg_color": (252, 252, 248),
        "lines": [
            "SIÊU THỊ WINMART+ QUẬN 7",
            "Địa chỉ: 458 Nguyễn Thị Thập, Tân Quy, Q7, TP.HCM",
            "MST: 0104918404",
            "Số HĐ: WM-20261017-4410",
            "Ngày: 17/10/2026 18:15:00",
            "----------------------------------------",
            "Sữa tươi Vinamilk 1L   2   34.000   68.000",
            "Trứng gà CP hộp 10     2   32.000   64.000",
            "Dầu ăn Neptune chai    2   55.000  110.000",
            "Mì Hảo Hảo tôm chua    5    4.500   22.500",
            "----------------------------------------",
            "Tạm tính:                         264.500",
            "Giảm giá:                         -14.500",
            "Thuế VAT (8%):                     20.000",
            "Tổng thanh toán:                  270.000",
            "Hình thức thanh toán: Chuyển khoản"
        ]
    },
    {
        "filename": "03_highlands_coffee_hop_le.jpg",
        "description": "Hóa đơn Quán Cà phê Highlands Coffee (Không có giảm giá, thanh toán Ví MoMo)",
        "bg_color": (250, 250, 250),
        "lines": [
            "HIGHLANDS COFFEE - BITEXCO TOWER",
            "Địa chỉ: Số 2 Hải Triều, Bến Nghé, Quận 1, TP.HCM",
            "MST: 0302088135",
            "Số phiếu: HL-889210",
            "Ngày: 17/10/2026 12:45:22",
            "----------------------------------------",
            "Phin Sữa Đá Size M     2   39.000   78.000",
            "Trà Sen Vàng Size L    1   55.000   55.000",
            "Bánh Phô Mai Trà Xanh  1   35.000   35.000",
            "----------------------------------------",
            "Tạm tính:                         168.000",
            "Thuế VAT (8%):                     13.440",
            "Tổng cộng:                        181.440",
            "Phương thức TT: MoMo"
        ]
    },
    {
        "filename": "04_nha_thuoc_long_chau_hop_le.jpg",
        "description": "Hóa đơn Nhà thuốc FPT Long Châu (Mã số thuế chi nhánh 13 số)",
        "bg_color": (255, 255, 255),
        "lines": [
            "NHA THUOC FPT LONG CHAU 108",
            "Dia chi: 379 Hai Ba Trung, Phuong Vo Thi Sau, Q3",
            "MST: 0315275368-002",
            "So HD: LC-20261017-019",
            "Ngay: 17/10/2026 15:20:05",
            "----------------------------------------",
            "Panadol Extra Do       2   25.000   50.000",
            "Vitamin C Berocca      1   95.000   95.000",
            "Khau trang Y te 4 lop  1   35.000   35.000",
            "----------------------------------------",
            "Tam tinh:                         180.000",
            "Giam gia:                         -10.000",
            "Thue VAT (5%):                      8.500",
            "Tong thanh toan:                  178.500",
            "Hinh thuc thanh toan: The Visa"
        ]
    },
    {
        "filename": "05_hoa_don_sai_so_hoc_test_validator.jpg",
        "description": "Hóa đơn CỐ TÌNH SAI SỐ HỌC để kiểm thử tính năng phát hiện lỗi của Validator (Tạm tính 100k + VAT 8k nhưng Tổng ghi 150k)",
        "bg_color": (255, 250, 250),
        "lines": [
            "CUA HANG BACH HOA TONG HOP MINH TAM",
            "Dia chi: 99 Le Van Viet, TP. Thu Duc, TP.HCM",
            "MST: 0316677889",
            "So HD: MT-2026-9999",
            "Ngay: 17/10/2026 19:05:00",
            "----------------------------------------",
            "Gao ST25 tui 5kg       1  145.000  145.000",
            "Nuoc mam Nam Ngu       1   45.000   45.000",
            "----------------------------------------",
            "Tam tinh:                         190.000",
            "Giam gia:                         -10.000",
            "Thue VAT (8%):                     14.400",
            "Tong thanh toan:                  250.000",
            "Hinh thuc thanh toan: Tien mat"
        ]
    }
]


def load_vietnamese_font(size: int = 22) -> ImageFont.FreeTypeFont:
    """Ưu tiên chọn font hỗ trợ tiếng Việt có dấu rõ nét và số 0 không gạch chéo."""
    candidates = [
        r"C:\Windows\Fonts\cour.ttf",      # Courier New (Monospace, hỗ trợ tiếng Việt, số 0 chuẩn)
        r"C:\Windows\Fonts\arial.ttf",     # Arial
        r"C:\Windows\Fonts\segoeui.ttf",   # Segoe UI
        "cour.ttf",
        "arial.ttf"
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_all_samples():
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    font = load_vietnamese_font(22)

    print("=" * 70)
    print(f"ĐANG TẠO BỘ HÓA ĐƠN MẪU TẠI: {samples_dir}")
    print("=" * 70)

    for spec in SAMPLES_SPECS:
        width, height = 840, 800
        img = Image.new("RGB", (width, height), color=spec["bg_color"])
        draw = ImageDraw.Draw(img)

        y = 35
        for line in spec["lines"]:
            draw.text((35, y), line, fill=(15, 15, 15), font=font)
            y += 42

        out_path = samples_dir / spec["filename"]
        img.save(out_path, quality=95)
        print(f"✅ Đã tạo: {out_path.name}")
        print(f"   Mô tả : {spec['description']}")

    # Đồng bộ sample_receipt.jpg mặc định với mẫu 01
    default_path = samples_dir / "sample_receipt.jpg"
    img_01 = Image.open(samples_dir / "01_circle_k_hop_le.jpg")
    img_01.save(default_path, quality=95)
    print(f"✅ Đã cập nhật mẫu mặc định: {default_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_samples()

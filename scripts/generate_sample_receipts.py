"""Script tạo bộ mẫu hóa đơn bán lẻ Việt Nam đa dạng kịch bản phục vụ Demo & Kiểm thử."""

import json
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
    },
    {
        "filename": "06_nha_hang_mon_an_xuong_dong.jpg",
        "description": "Hóa đơn Nhà hàng có tên món dài ngắt xuống dòng thứ 2 (Multi-line items phá vỡ biểu thức 1 dòng của Regex)",
        "bg_color": (255, 255, 255),
        "lines": [
            "NHA HANG AM THUC VIET",
            "Dia chi: 72 Vo Van Tan, Phuong Vo Thi Sau, Q3, TP.HCM",
            "SDT: 028 3930 2233",
            "MST: 0311223344",
            "So HD: HD-20261021-06",
            "Ngay: 21/10/2026 19:30:00",
            "----------------------------------------",
            "Lau ca keo la giang dac biet",
            "                       1   180.000  180.000",
            "Bo xao bong thien ly mien Tay",
            "                       2    95.000  190.000",
            "Tra da hoa cuc duong phen",
            "                       3    15.000   45.000",
            "----------------------------------------",
            "Tam tinh:                         415.000",
            "Giam gia:                         -15.000",
            "Thue VAT (8%):                     32.000",
            "Tong thanh toan:                  432.000",
            "Hinh thuc thanh toan: The Visa"
        ]
    },
    {
        "filename": "07_circle_k_nhieu_ky_tu_ocr_lss.jpg",
        "description": "Hóa đơn Circle K bị nhiễu ký tự OCR ở dòng tổng tiền (Kiểm chứng LSS Self-Correction)",
        "bg_color": (254, 254, 252),
        "lines": [
            "SIEU THI TIEN LOI CIRCLE K",
            "Dia chi: 208 Nguyen Huu Canh, Binh Thanh, TP.HCM",
            "Hotline: 1900 6368",
            "MST: 0312456789",
            "So HD: CK-20261021-07",
            "Ngay: 21/10/2026 09:15:00",
            "----------------------------------------",
            "Banh mi sandwich ga xe 1    28.000   28.000",
            "Sua Milo hop 180ml     2    11.000   22.000",
            "Snack khoai tay Lays   2    15.000   30.000",
            "----------------------------------------",
            "Tam tinh:                          80.000",
            "Giam gia:                          -5.000",
            "Thue VAT (8%):                      6.000",
            "Tong thanh toan:                   81.000",
            "Hinh thuc thanh toan: MoMo"
        ]
    },
    {
        "filename": "08_quan_an_viet_tat_k_khong_dau.jpg",
        "description": "Hóa đơn Quán ăn bình dân không dấu, viết tắt tiền tệ bằng k/K và nhãn viết tắt 'TC: 260k'",
        "bg_color": (252, 252, 250),
        "lines": [
            "QUAN LAU NUONG BINH DAN 79",
            "Dia chi: 15 Bo Bao Tan Thang, Tan Phu",
            "SDT: 0903 889900",
            "So ban: B05",
            "Ngay: 21/10/2026 20:00",
            "----------------------------------------",
            "Lau Thai hai san       1      180k     180k",
            "Bia Saigon Special     4       20k      80k",
            "Khan lanh              4        3k      12k",
            "----------------------------------------",
            "Tam tinh:                              272k",
            "Giam gia:                              -12k",
            "TC:                                    260k",
            "Hinh thuc thanh toan: Tien mat"
        ]
    },
    {
        "filename": "09_sieu_thi_khuyen_mai_vat_layout.jpg",
        "description": "Hóa đơn Siêu thị Co.opmart bố cục bảng phức tạp có chiết khấu theo dòng và thuế suất 8%",
        "bg_color": (255, 255, 255),
        "lines": [
            "SIEU THI CO.OPMART NGUYEN DINH CHIEU",
            "Dia chi: 168 Nguyen Dinh Chieu, Phuong 6, Quan 3",
            "MST: 0300624009",
            "So HD: COOP-2026-9011",
            "Ngay: 21/10/2026 10:45:10",
            "----------------------------------------",
            "Gao thom ST25 5kg      1   150.000  150.000",
            "Dau dau nanh Simply    2    65.000  130.000",
            "Nuoc ngot Coca Cola    6     9.000   54.000",
            "----------------------------------------",
            "Tam tinh:                         334.000",
            "Tong chiet khau:                  -16.000",
            "Thue VAT (8%):                     25.440",
            "Tong thanh toan:                  343.440",
            "Hinh thuc thanh toan: Chuyen khoan"
        ]
    },
    {
        "filename": "10_hoa_don_mo_nhan_quang_cao_cuoi_phieu.jpg",
        "description": "Hóa đơn có dòng quảng cáo khuyến mãi voucher 500k và hotline ở cuối phiếu (Thử thách Hallucination)",
        "bg_color": (250, 250, 248),
        "lines": [
            "CUA HANG BACH HOA AN TAM",
            "Dia chi: 55 Nguyen Huu Tho, Quan 7, TP.HCM",
            "MST: 0317899012",
            "Ngay: 21/10/2026 14:10:00",
            "----------------------------------------",
            "Kem danh rang PS       1    35.000   35.000",
            "Ban chai danh rang     2    20.000   40.000",
            "Xa bong tam Lifebuoy   2    15.000   30.000",
            "----------------------------------------",
            "Tam tinh:                         105.000",
            "Tong thanh toan:                  105.000",
            "Hinh thuc thanh toan: Tien mat",
            "----------------------------------------",
            "*** CHUONG TRINH KHUYEN MAI DAC BIET ***",
            "Cai app tich diem nhan voucher 500.000 d",
            "Hotline cskh thuong: 1900 9999",
            "Cam on quy khach va hen gap lai!"
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


SAMPLE_GROUND_TRUTHS: Dict[str, Dict[str, Any]] = {
    "06_nha_hang_mon_an_xuong_dong.jpg": {
        "seller_name": "NHA HANG AM THUC VIET",
        "seller_address": "72 Vo Van Tan, Phuong Vo Thi Sau, Q3, TP.HCM",
        "seller_phone": "028 3930 2233",
        "tax_id": "0311223344",
        "invoice_number": "HD-20261021-06",
        "timestamp": "21/10/2026 19:30:00",
        "items": [
            {"item_name": "Lau ca keo la giang dac biet", "quantity": 1.0, "unit_price": 180000, "total_price": 180000},
            {"item_name": "Bo xao bong thien ly mien Tay", "quantity": 2.0, "unit_price": 95000, "total_price": 190000},
            {"item_name": "Tra da hoa cuc duong phen", "quantity": 3.0, "unit_price": 15000, "total_price": 45000}
        ],
        "subtotal": 415000,
        "discount": 15000,
        "vat_amount": 32000,
        "total_cost": 432000,
        "payment_method": "The Visa"
    },
    "07_circle_k_nhieu_ky_tu_ocr_lss.jpg": {
        "seller_name": "SIEU THI TIEN LOI CIRCLE K",
        "seller_address": "208 Nguyen Huu Canh, Binh Thanh, TP.HCM",
        "seller_phone": "1900 6368",
        "tax_id": "0312456789",
        "invoice_number": "CK-20261021-07",
        "timestamp": "21/10/2026 09:15:00",
        "items": [
            {"item_name": "Banh mi sandwich ga xe", "quantity": 1.0, "unit_price": 28000, "total_price": 28000},
            {"item_name": "Sua Milo hop 180ml", "quantity": 2.0, "unit_price": 11000, "total_price": 22000},
            {"item_name": "Snack khoai tay Lays", "quantity": 2.0, "unit_price": 15000, "total_price": 30000}
        ],
        "subtotal": 80000,
        "discount": 5000,
        "vat_amount": 6000,
        "total_cost": 81000,
        "payment_method": "MoMo"
    },
    "08_quan_an_viet_tat_k_khong_dau.jpg": {
        "seller_name": "QUAN LAU NUONG BINH DAN 79",
        "seller_address": "15 Bo Bao Tan Thang, Tan Phu",
        "seller_phone": "0903 889900",
        "tax_id": None,
        "invoice_number": "B05",
        "timestamp": "21/10/2026 20:00",
        "items": [
            {"item_name": "Lau Thai hai san", "quantity": 1.0, "unit_price": 180000, "total_price": 180000},
            {"item_name": "Bia Saigon Special", "quantity": 4.0, "unit_price": 20000, "total_price": 80000},
            {"item_name": "Khan lanh", "quantity": 4.0, "unit_price": 3000, "total_price": 12000}
        ],
        "subtotal": 272000,
        "discount": 12000,
        "vat_amount": None,
        "total_cost": 260000,
        "payment_method": "Tien mat"
    },
    "09_sieu_thi_khuyen_mai_vat_layout.jpg": {
        "seller_name": "SIEU THI CO.OPMART NGUYEN DINH CHIEU",
        "seller_address": "168 Nguyen Dinh Chieu, Phuong 6, Quan 3",
        "seller_phone": None,
        "tax_id": "0300624009",
        "invoice_number": "COOP-2026-9011",
        "timestamp": "21/10/2026 10:45:10",
        "items": [
            {"item_name": "Gao thom ST25 5kg", "quantity": 1.0, "unit_price": 150000, "total_price": 150000},
            {"item_name": "Dau dau nanh Simply", "quantity": 2.0, "unit_price": 65000, "total_price": 130000},
            {"item_name": "Nuoc ngot Coca Cola", "quantity": 6.0, "unit_price": 9000, "total_price": 54000}
        ],
        "subtotal": 334000,
        "discount": 16000,
        "vat_amount": 25440,
        "total_cost": 343440,
        "payment_method": "Chuyen khoan"
    },
    "10_hoa_don_mo_nhan_quang_cao_cuoi_phieu.jpg": {
        "seller_name": "CUA HANG BACH HOA AN TAM",
        "seller_address": "55 Nguyen Huu Tho, Quan 7, TP.HCM",
        "seller_phone": None,
        "tax_id": "0317899012",
        "invoice_number": None,
        "timestamp": "21/10/2026 14:10:00",
        "items": [
            {"item_name": "Kem danh rang PS", "quantity": 1.0, "unit_price": 35000, "total_price": 35000},
            {"item_name": "Ban chai danh rang", "quantity": 2.0, "unit_price": 20000, "total_price": 40000},
            {"item_name": "Xa bong tam Lifebuoy", "quantity": 2.0, "unit_price": 15000, "total_price": 30000}
        ],
        "subtotal": 105000,
        "discount": None,
        "vat_amount": None,
        "total_cost": 105000,
        "payment_method": "Tien mat"
    }
}


def generate_all_samples():
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    font = load_vietnamese_font(22)

    print("=" * 70)
    print(f"ĐANG TẠO BỘ 10 HÓA ĐƠN MẪU TẠI: {samples_dir}")
    print("=" * 70)

    for spec in SAMPLES_SPECS:
        width, height = 840, 860
        img = Image.new("RGB", (width, height), color=spec["bg_color"])
        draw = ImageDraw.Draw(img)

        y = 35
        for line in spec["lines"]:
            draw.text((35, y), line, fill=(15, 15, 15), font=font)
            y += 42

        fname = spec["filename"]
        out_path = samples_dir / fname
        img.save(out_path, quality=95)

        # Xuất file .txt văn bản OCR chuẩn
        txt_path = samples_dir / f"{out_path.stem}.txt"
        txt_path.write_text("\n".join(spec["lines"]) + "\n", encoding="utf-8")

        # Xuất file .json Ground Truth nếu có định nghĩa
        if fname in SAMPLE_GROUND_TRUTHS:
            gt_dict = {
                "img_id": fname,
                "image_quality": 0.92,
                **SAMPLE_GROUND_TRUTHS[fname]
            }
            json_path = samples_dir / f"{out_path.stem}.json"
            json_path.write_text(json.dumps(gt_dict, ensure_ascii=False, indent=2), encoding="utf-8")

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

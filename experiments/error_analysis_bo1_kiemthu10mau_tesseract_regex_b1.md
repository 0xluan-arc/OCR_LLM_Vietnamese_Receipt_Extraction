# BÁO CÁO PHÂN TÍCH LỖI MINH BẠCH (ERROR ANALYSIS) — `bo1_kiemthu10mau_tesseract_regex_b1`

> **✅ VERIFIED REAL MODEL EVALUATION**

## 1. Thống kê Tổng quan Nguyên nhân Lỗi (Root-Cause Breakdown)

- **Tổng số hóa đơn đánh giá**: `10`
- **Số hóa đơn khớp tuyệt đối & không cảnh báo**: `5`
- **Số hóa đơn có lỗi trích xuất hoặc cảnh báo toán học**: `5`
- **Tổng số trường bị lỗi (`total_field_errors`)**: `7`

| Mã phân loại lỗi (Error Type) | Số lượng trường lỗi | Tỷ trọng (%) | Ý nghĩa Khoa học |
| :--- | :---: | :---: | :--- |
| `OCR_CORRUPTION_ERROR` | 1 | 14.3% | Nhãn chuẩn bị mất/biến dạng từ tầng nhận dạng OCR |
| `EXTRACTOR_MISS_ERROR` | 3 | 42.9% | Có văn bản trong OCR nhưng Extractor bỏ sót (`null`) |
| `EXTRACTOR_WRONG_VALUE` | 3 | 42.9% | Extractor lấy nhầm dòng/giá trị khác trên hóa đơn |
| `HALLUCINATION_ERROR` | 0 | 0.0% | Extractor tự bịa giá trị không có trong văn bản OCR |

## 2. Chi tiết Từng Hóa đơn Có Lỗi / Cảnh báo

### 1. Hóa đơn `01_circle_k_hop_le.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_address`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `123 Nguyen Thi Minh Khai, Q1, TP.HCM` | Pred = `123 Nguyen Thi Minh Khai, Ql, TP.HCM` — *Bộ trích xuất lấy nhầm giá trị '123 Nguyen Thi Minh Khai, Ql, TP.HCM' thay vì nhãn chuẩn '123 Nguyen Thi Minh Khai, Q1, TP.HCM'.*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
SIEU THI TIEN LOI CIRCLE K

Dia chi: 123 Nguyen Thi Minh Khai, Ql, TP.HCM
MST: 0312456789

So HD: CK-20261017-0892

Ngay: 17/10/2026 08:30:15

Ca phe sua da 330ml 2 25.000 50.000
Banh mi que pate 2 15.000 30.000
Nuoc khoang Lavie 2 10.000 20.000
Tam tỉnh: 100.000
Giam gia: -5.000
Thue VAT (8%): 7.600
Tong thanh toan: 102.600
Hinh thuc thanh toan: Tien mat
```
**`ground_truth_json`**:
```json
{
  "seller_name": "SIEU THI TIEN LOI CIRCLE K",
  "seller_address": "123 Nguyen Thi Minh Khai, Q1, TP.HCM",
  "seller_phone": null,
  "tax_id": "0312456789",
  "invoice_number": "CK-20261017-0892",
  "timestamp": "17/10/2026 08:30:15",
  "items": [
    {
      "item_name": "Ca phe sua da 330ml",
      "quantity": 2.0,
      "unit_price": 25000,
      "total_price": 50000
    },
    {
      "item_name": "Banh mi que pate",
      "quantity": 2.0,
      "unit_price": 15000,
      "total_price": 30000
    },
    {
      "item_name": "Nuoc khoang Lavie",
      "quantity": 2.0,
      "unit_price": 10000,
      "total_price": 20000
    }
  ],
  "subtotal": 100000,
  "discount": 5000,
  "vat_amount": 7600,
  "total_cost": 102600,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "SIEU THI TIEN LOI CIRCLE K",
  "seller_address": "123 Nguyen Thi Minh Khai, Ql, TP.HCM",
  "seller_phone": null,
  "tax_id": "0312456789",
  "invoice_number": "CK-20261017-0892",
  "timestamp": "17/10/2026 08:30:15",
  "items": [
    {
      "item_name": "Ca phe sua da 330ml",
      "quantity": 2.0,
      "unit_price": 25000,
      "total_price": 50000
    },
    {
      "item_name": "Banh mi que pate",
      "quantity": 2.0,
      "unit_price": 15000,
      "total_price": 30000
    },
    {
      "item_name": "Nuoc khoang Lavie",
      "quantity": 2.0,
      "unit_price": 10000,
      "total_price": 20000
    }
  ],
  "subtotal": 100000,
  "discount": 5000,
  "vat_amount": 7600,
  "total_cost": 102600,
  "payment_method": "Tien mat"
}
```
</details>

### 2. Hóa đơn `05_hoa_don_sai_so_hoc_test_validator.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- *(Không có lỗi trích xuất trường so với Ground-Truth, chỉ có cảnh báo kiểm định toán học trên hóa đơn gốc)*

**Cảnh báo kiểm định (`validation_issues`):**
- `[WARNING] ARITHMETIC_MISMATCH_TOTAL`: Sai lệch số học tổng thể: Tạm tính (190,000) - Giảm giá (10,000) + VAT (14,400) = 194,400 VND, nhưng Tổng thanh toán thực tế ghi là 250,000 VND (Chênh lệch: 55,600 VND).

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
CUA HANG BACH HOA TONG HOP MINH TAM

Dia chi: 99 Le Van Viet, TP. Thu Duc, TP.HCM
MST: 0316677889

So HD: MT-2026-9999

Ngay: 17/10/2026 19:05:00

Gao ST25 tui 5kg 1 145.000 145.000
Nuoc mam Nam Ngu 1 45.000 45.000
Tam tỉnh: 190.000
Giam gia: -10.000
Thue VAT (8%): 14.400
Tong thanh toan: 250.000
Hinh thuc thanh toan: Tien mat
```
**`ground_truth_json`**:
```json
{
  "seller_name": "CUA HANG BACH HOA TONG HOP MINH TAM",
  "seller_address": "99 Le Van Viet, TP. Thu Duc, TP.HCM",
  "seller_phone": null,
  "tax_id": "0316677889",
  "invoice_number": "MT-2026-9999",
  "timestamp": "17/10/2026 19:05:00",
  "items": [
    {
      "item_name": "Gao ST25 tui 5kg",
      "quantity": 1.0,
      "unit_price": 145000,
      "total_price": 145000
    },
    {
      "item_name": "Nuoc mam Nam Ngu",
      "quantity": 1.0,
      "unit_price": 45000,
      "total_price": 45000
    }
  ],
  "subtotal": 190000,
  "discount": 10000,
  "vat_amount": 14400,
  "total_cost": 250000,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "CUA HANG BACH HOA TONG HOP MINH TAM",
  "seller_address": "99 Le Van Viet, TP. Thu Duc, TP.HCM",
  "seller_phone": null,
  "tax_id": "0316677889",
  "invoice_number": "MT-2026-9999",
  "timestamp": "17/10/2026 19:05:00",
  "items": [
    {
      "item_name": "Gao ST25 tui 5kg",
      "quantity": 1.0,
      "unit_price": 145000,
      "total_price": 145000
    },
    {
      "item_name": "Nuoc mam Nam Ngu",
      "quantity": 1.0,
      "unit_price": 45000,
      "total_price": 45000
    }
  ],
  "subtotal": 190000,
  "discount": 10000,
  "vat_amount": 14400,
  "total_cost": 250000,
  "payment_method": "Tien mat"
}
```
</details>

### 3. Hóa đơn `06_nha_hang_mon_an_xuong_dong.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`items`** → [`EXTRACTOR_MISS_ERROR`]: GT = `[{'item_name': 'Lau ca keo la giang dac biet', 'quantity': 1.0, 'unit_price': 180000, 'total_price': 180000}, {'item_name': 'Bo xao bong thien ly mien Tay', 'quantity': 2.0, 'unit_price': 95000, 'total_price': 190000}, {'item_name': 'Tra da hoa cuc duong phen', 'quantity': 3.0, 'unit_price': 15000, 'total_price': 45000}]` | Pred = `[]` — *Văn bản OCR có chứa thông tin mặt hàng nhưng bộ trích xuất bỏ sót (0/3 dòng).*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
NHA HANG AM THUC VIET
Dia chi: 72 Vo Van Tan, Phuong Vo Thi Sau, Q3, TP.HCM
SDT: 028 3930 2233
MST: 0311223344
So HD: HD-20261021-06
Ngay: 21/10/2026 19:30:00
Lau ca keo la giang dac biet

1 180.000 180.000
Bo xao bong thien ly mien Tay

2 95.000 190.000
Tra da hoa cuc duong phen

8 15.000 45.000
Tam tỉnh: 415.000
Giam gia: -15.000
Thue VAT (8%): 32.000
Tong thanh toan: 432.000
Hinh thuc thanh toan: The Visa
```
**`ground_truth_json`**:
```json
{
  "seller_name": "NHA HANG AM THUC VIET",
  "seller_address": "72 Vo Van Tan, Phuong Vo Thi Sau, Q3, TP.HCM",
  "seller_phone": "028 3930 2233",
  "tax_id": "0311223344",
  "invoice_number": "HD-20261021-06",
  "timestamp": "21/10/2026 19:30:00",
  "items": [
    {
      "item_name": "Lau ca keo la giang dac biet",
      "quantity": 1.0,
      "unit_price": 180000,
      "total_price": 180000
    },
    {
      "item_name": "Bo xao bong thien ly mien Tay",
      "quantity": 2.0,
      "unit_price": 95000,
      "total_price": 190000
    },
    {
      "item_name": "Tra da hoa cuc duong phen",
      "quantity": 3.0,
      "unit_price": 15000,
      "total_price": 45000
    }
  ],
  "subtotal": 415000,
  "discount": 15000,
  "vat_amount": 32000,
  "total_cost": 432000,
  "payment_method": "The Visa"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "NHA HANG AM THUC VIET",
  "seller_address": "72 Vo Van Tan, Phuong Vo Thi Sau, Q3, TP.HCM",
  "seller_phone": "028 3930 2233",
  "tax_id": "0311223344",
  "invoice_number": "HD-20261021-06",
  "timestamp": "21/10/2026 19:30:00",
  "items": [],
  "subtotal": 415000,
  "discount": 15000,
  "vat_amount": 32000,
  "total_cost": 432000,
  "payment_method": "The Visa"
}
```
</details>

### 4. Hóa đơn `08_quan_an_viet_tat_k_khong_dau.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`invoice_number`** → [`OCR_CORRUPTION_ERROR`]: GT = `B05` | Pred = `None` — *Nhãn chuẩn 'B05' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`total_cost`** → [`EXTRACTOR_MISS_ERROR`]: GT = `260000` | Pred = `None` — *Nhãn chuẩn '260000' có xuất hiện trong văn bản OCR nhưng bộ trích xuất trả về null (Bỏ sót).*
- **`items`** → [`EXTRACTOR_MISS_ERROR`]: GT = `[{'item_name': 'Lau Thai hai san', 'quantity': 1.0, 'unit_price': 180000, 'total_price': 180000}, {'item_name': 'Bia Saigon Special', 'quantity': 4.0, 'unit_price': 20000, 'total_price': 80000}, {'item_name': 'Khan lanh', 'quantity': 4.0, 'unit_price': 3000, 'total_price': 12000}]` | Pred = `[]` — *Văn bản OCR có chứa thông tin mặt hàng nhưng bộ trích xuất bỏ sót (0/3 dòng).*

**Cảnh báo kiểm định (`validation_issues`):**
- `[ERROR] MISSING_TOTAL_COST`: Không tìm thấy hoặc không trích xuất được trường Tổng tiền thanh toán (total_cost).

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
QUAN LAU NUONG BINH DAN 79

Dia chỉ: 15 Bo Bao Tan Thang, Tan Phu

SDT: 0903 889900

So ban: BOS

Ngay: 21/10/2026 20:00

Lau Thai hai san 1 180k 180k
Bia Saigon Special 4 20k 80k
Khan lanh 4 3k 12k
Tam tinh: 272k
Giam gia: -12k
TE: 260k
Hinh thuc thanh toan: Tien mat
```
**`ground_truth_json`**:
```json
{
  "seller_name": "QUAN LAU NUONG BINH DAN 79",
  "seller_address": "15 Bo Bao Tan Thang, Tan Phu",
  "seller_phone": "0903 889900",
  "tax_id": null,
  "invoice_number": "B05",
  "timestamp": "21/10/2026 20:00",
  "items": [
    {
      "item_name": "Lau Thai hai san",
      "quantity": 1.0,
      "unit_price": 180000,
      "total_price": 180000
    },
    {
      "item_name": "Bia Saigon Special",
      "quantity": 4.0,
      "unit_price": 20000,
      "total_price": 80000
    },
    {
      "item_name": "Khan lanh",
      "quantity": 4.0,
      "unit_price": 3000,
      "total_price": 12000
    }
  ],
  "subtotal": 272000,
  "discount": 12000,
  "vat_amount": null,
  "total_cost": 260000,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "QUAN LAU NUONG BINH DAN 79",
  "seller_address": "15 Bo Bao Tan Thang, Tan Phu",
  "seller_phone": "0903 889900",
  "tax_id": null,
  "invoice_number": null,
  "timestamp": "21/10/2026 20:00",
  "items": [],
  "subtotal": 272000,
  "discount": 12000,
  "vat_amount": null,
  "total_cost": null,
  "payment_method": "Tien mat"
}
```
</details>

### 5. Hóa đơn `10_hoa_don_mo_nhan_quang_cao_cuoi_phieu.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_phone`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `None` | Pred = `1900 9999` — *Bộ trích xuất lấy nhầm giá trị '1900 9999' thay vì nhãn chuẩn 'None'.*
- **`discount`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `None` | Pred = `500000` — *Bộ trích xuất lấy nhầm giá trị '500000' thay vì nhãn chuẩn 'None'.*

**Cảnh báo kiểm định (`validation_issues`):**
- `[WARNING] ARITHMETIC_MISMATCH_TOTAL`: Sai lệch số học tổng thể: Tạm tính (105,000) - Giảm giá (500,000) + VAT (0) = -395,000 VND, nhưng Tổng thanh toán thực tế ghi là 105,000 VND (Chênh lệch: 500,000 VND).

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
CUA HANG BACH HOA AN TAM

Dia chỉ: 55 Nguyen Huu Tho, Quan 7, TP.HCM
MST: 0317899012

Ngay: 21/10/2026 14:10:00

Kem danh rang PS 1 35.000 35.000
Ban chai danh rang 2 20.000 40.000
Xa bong tam Lifebuoy 2 15.000 30.000
Tam tỉnh: 105.000
Tong thanh toan: 105.000
Hinh thuc thanh toan: Tien mat

*** CHUONG TRINH KHUYEN MAI DAC BIET ***
Cai app tích diem nhan voucher 500.000 d
Hotline cskh thuong: 1900 9999

Cam on quy khach va hen gap lai!
```
**`ground_truth_json`**:
```json
{
  "seller_name": "CUA HANG BACH HOA AN TAM",
  "seller_address": "55 Nguyen Huu Tho, Quan 7, TP.HCM",
  "seller_phone": null,
  "tax_id": "0317899012",
  "invoice_number": null,
  "timestamp": "21/10/2026 14:10:00",
  "items": [
    {
      "item_name": "Kem danh rang PS",
      "quantity": 1.0,
      "unit_price": 35000,
      "total_price": 35000
    },
    {
      "item_name": "Ban chai danh rang",
      "quantity": 2.0,
      "unit_price": 20000,
      "total_price": 40000
    },
    {
      "item_name": "Xa bong tam Lifebuoy",
      "quantity": 2.0,
      "unit_price": 15000,
      "total_price": 30000
    }
  ],
  "subtotal": 105000,
  "discount": null,
  "vat_amount": null,
  "total_cost": 105000,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "CUA HANG BACH HOA AN TAM",
  "seller_address": "55 Nguyen Huu Tho, Quan 7, TP.HCM",
  "seller_phone": "1900 9999",
  "tax_id": "0317899012",
  "invoice_number": null,
  "timestamp": "21/10/2026 14:10:00",
  "items": [
    {
      "item_name": "Kem danh rang PS",
      "quantity": 1.0,
      "unit_price": 35000,
      "total_price": 35000
    },
    {
      "item_name": "Ban chai danh rang",
      "quantity": 2.0,
      "unit_price": 20000,
      "total_price": 40000
    },
    {
      "item_name": "Xa bong tam Lifebuoy",
      "quantity": 2.0,
      "unit_price": 15000,
      "total_price": 30000
    }
  ],
  "subtotal": 105000,
  "discount": 500000,
  "vat_amount": null,
  "total_cost": 105000,
  "payment_method": "Tien mat"
}
```
</details>


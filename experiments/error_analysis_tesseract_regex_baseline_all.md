# BÁO CÁO PHÂN TÍCH LỖI MINH BẠCH (ERROR ANALYSIS) — `tesseract_regex_baseline_all`

> **✅ VERIFIED REAL MODEL EVALUATION**

## 1. Thống kê Tổng quan Nguyên nhân Lỗi (Root-Cause Breakdown)

- **Tổng số hóa đơn đánh giá**: `10`
- **Số hóa đơn khớp tuyệt đối & không cảnh báo**: `0`
- **Số hóa đơn có lỗi trích xuất hoặc cảnh báo toán học**: `10`
- **Tổng số trường bị lỗi (`total_field_errors`)**: `33`

| Mã phân loại lỗi (Error Type) | Số lượng trường lỗi | Tỷ trọng (%) | Ý nghĩa Khoa học |
| :--- | :---: | :---: | :--- |
| `OCR_CORRUPTION_ERROR` | 19 | 57.6% | Nhãn chuẩn bị mất/biến dạng từ tầng nhận dạng OCR |
| `EXTRACTOR_MISS_ERROR` | 3 | 9.1% | Có văn bản trong OCR nhưng Extractor bỏ sót (`null`) |
| `EXTRACTOR_WRONG_VALUE` | 11 | 33.3% | Extractor lấy nhầm dòng/giá trị khác trên hóa đơn |
| `HALLUCINATION_ERROR` | 0 | 0.0% | Extractor tự bịa giá trị không có trong văn bản OCR |

## 2. Chi tiết Từng Hóa đơn Có Lỗi / Cảnh báo

### 1. Hóa đơn `hoadon_thucte_001.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_phone`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `028 3925 6789` | Pred = `928 3925 6789` — *Bộ trích xuất lấy nhầm giá trị '928 3925 6789' thay vì nhãn chuẩn '028 3925 6789'.*
- **`subtotal`** → [`OCR_CORRUPTION_ERROR`]: GT = `154500` | Pred = `154590` — *Nhãn chuẩn '154500' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`discount`** → [`OCR_CORRUPTION_ERROR`]: GT = `4500` | Pred = `4509` — *Nhãn chuẩn '4500' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`vat_amount`** → [`OCR_CORRUPTION_ERROR`]: GT = `12000` | Pred = `12009` — *Nhãn chuẩn '12000' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`items`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `[{'item_name': 'Dau an Neptune Light 1L', 'quantity': 1.0, 'unit_price': 62000, 'total_price': 62000}, {'item_name': 'Trung ga ta Ba Huan H10', 'quantity': 2.0, 'unit_price': 35000, 'total_price': 70000}, {'item_name': 'Mi Hao Hao tom chua cay', 'quantity': 5.0, 'unit_price': 4500, 'total_price': 22500}]` | Pred = `[{'item_name': 'Dau an Neptune Light 1L', 'quantity': 1.0, 'unit_price': 62000, 'total_price': 62000}, {'item_name': 'Trung ga ta Ba Huan H19', 'quantity': 2.0, 'unit_price': 35000, 'total_price': 78068}, {'item_name': 'Mi Hao Hao tom chua cay', 'quantity': 5.0, 'unit_price': 4590, 'total_price': 22500}]` — *Danh sách mặt hàng trích xuất (3 dòng) không khớp hoàn toàn với nhãn chuẩn (3 dòng).*

**Cảnh báo kiểm định (`validation_issues`):**
- `[WARNING] ITEM_MATH_MISMATCH`: Mặt hàng 'Trung ga ta Ba Huan H19': SL (2.0) * Đơn giá (35,000) = 70,000 VND, khác Thành tiền ghi nhận (78,068 VND).
- `[WARNING] ITEMS_SUM_SUBTOTAL_MISMATCH`: Tổng thành tiền các mặt hàng (162,568 VND) không khớp với Tạm tính (154,590 VND).

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
CO.OPMART CONG QUYNH

Dia chỉ: 189C Cong Quynh, P. Nguyen Cu Trinh, Q1, TP.HCM
SDT: 928 3925 6789

MST: 0301175688

So HD: CM-20261018-0101

Ngay: 18/10/2026 09:15:20

Dau an Neptune Light 1L 1 62.000 62.000
Trung ga ta Ba Huan H19 2 35.000 78.068
Mi Hao Hao tom chua cay 5 4.590 22.500
Tam tỉnh: 154.590

Giam gia: -4.509

Thue VAT (8%): 12.009

Tong thanh toan: 162.000

Hinh thuc thanh toan: Tien mat
```
**`ground_truth_json`**:
```json
{
  "seller_name": "CO.OPMART CONG QUYNH",
  "seller_address": "189C Cong Quynh, P. Nguyen Cu Trinh, Q1, TP.HCM",
  "seller_phone": "028 3925 6789",
  "tax_id": "0301175688",
  "invoice_number": "CM-20261018-0101",
  "timestamp": "18/10/2026 09:15:20",
  "items": [
    {
      "item_name": "Dau an Neptune Light 1L",
      "quantity": 1.0,
      "unit_price": 62000,
      "total_price": 62000
    },
    {
      "item_name": "Trung ga ta Ba Huan H10",
      "quantity": 2.0,
      "unit_price": 35000,
      "total_price": 70000
    },
    {
      "item_name": "Mi Hao Hao tom chua cay",
      "quantity": 5.0,
      "unit_price": 4500,
      "total_price": 22500
    }
  ],
  "subtotal": 154500,
  "discount": 4500,
  "vat_amount": 12000,
  "total_cost": 162000,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "CO.OPMART CONG QUYNH",
  "seller_address": "189C Cong Quynh, P. Nguyen Cu Trinh, Q1, TP.HCM",
  "seller_phone": "928 3925 6789",
  "tax_id": "0301175688",
  "invoice_number": "CM-20261018-0101",
  "timestamp": "18/10/2026 09:15:20",
  "items": [
    {
      "item_name": "Dau an Neptune Light 1L",
      "quantity": 1.0,
      "unit_price": 62000,
      "total_price": 62000
    },
    {
      "item_name": "Trung ga ta Ba Huan H19",
      "quantity": 2.0,
      "unit_price": 35000,
      "total_price": 78068
    },
    {
      "item_name": "Mi Hao Hao tom chua cay",
      "quantity": 5.0,
      "unit_price": 4590,
      "total_price": 22500
    }
  ],
  "subtotal": 154590,
  "discount": 4509,
  "vat_amount": 12009,
  "total_cost": 162000,
  "payment_method": "Tien mat"
}
```
</details>

### 2. Hóa đơn `hoadon_thucte_002.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`tax_id`** → [`OCR_CORRUPTION_ERROR`]: GT = `0313049871` | Pred = `None` — *Nhãn chuẩn '0313049871' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`items`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `[{'item_name': 'Thit ba roi heo CP 500g', 'quantity': 1.0, 'unit_price': 89000, 'total_price': 89000}, {'item_name': 'Cai ngot VietGAP 500g', 'quantity': 2.0, 'unit_price': 14000, 'total_price': 28000}]` | Pred = `[{'item_name': 'Thit ba roi heo CP 599g', 'quantity': 1.0, 'unit_price': 89000, 'total_price': 89000}, {'item_name': 'Cai ngot VietGAP 500g', 'quantity': 2.0, 'unit_price': 14000, 'total_price': 28000}]` — *Danh sách mặt hàng trích xuất (2 dòng) không khớp hoàn toàn với nhãn chuẩn (2 dòng).*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
BACH HOA XANH QUAN TAN BINH

Dia chi: 312 Cong Hoa, P.13, Q. Tan Binh, TP.HCM
SDT: 1900 1908

MST: @313049871

So HD: BHX-882104

Ngay: 18/10/2026 10:42:11

Thit ba roi heo CP 599g 1 89.000 89.000
Cai ngot VietGAP 500g 2 14.000 28.000
Tam tinh: 117.000

Giam gia: -7.000

Thue VAT (8%): 8.800

Tong thanh toan: 118.800

Hinh thuc thanh toan: MoMo
```
**`ground_truth_json`**:
```json
{
  "seller_name": "BACH HOA XANH QUAN TAN BINH",
  "seller_address": "312 Cong Hoa, P.13, Q. Tan Binh, TP.HCM",
  "seller_phone": "1900 1908",
  "tax_id": "0313049871",
  "invoice_number": "BHX-882104",
  "timestamp": "18/10/2026 10:42:11",
  "items": [
    {
      "item_name": "Thit ba roi heo CP 500g",
      "quantity": 1.0,
      "unit_price": 89000,
      "total_price": 89000
    },
    {
      "item_name": "Cai ngot VietGAP 500g",
      "quantity": 2.0,
      "unit_price": 14000,
      "total_price": 28000
    }
  ],
  "subtotal": 117000,
  "discount": 7000,
  "vat_amount": 8800,
  "total_cost": 118800,
  "payment_method": "MoMo"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "BACH HOA XANH QUAN TAN BINH",
  "seller_address": "312 Cong Hoa, P.13, Q. Tan Binh, TP.HCM",
  "seller_phone": "1900 1908",
  "tax_id": null,
  "invoice_number": "BHX-882104",
  "timestamp": "18/10/2026 10:42:11",
  "items": [
    {
      "item_name": "Thit ba roi heo CP 599g",
      "quantity": 1.0,
      "unit_price": 89000,
      "total_price": 89000
    },
    {
      "item_name": "Cai ngot VietGAP 500g",
      "quantity": 2.0,
      "unit_price": 14000,
      "total_price": 28000
    }
  ],
  "subtotal": 117000,
  "discount": 7000,
  "vat_amount": 8800,
  "total_cost": 118800,
  "payment_method": "MoMo"
}
```
</details>

### 3. Hóa đơn `hoadon_thucte_003.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_phone`** → [`EXTRACTOR_MISS_ERROR`]: GT = `028 6263 0377` | Pred = `None` — *Nhãn chuẩn '028 6263 0377' có xuất hiện trong văn bản OCR nhưng bộ trích xuất trả về null (Bỏ sót).*
- **`tax_id`** → [`OCR_CORRUPTION_ERROR`]: GT = `0312567890` | Pred = `None` — *Nhãn chuẩn '0312567890' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`timestamp`** → [`OCR_CORRUPTION_ERROR`]: GT = `18/10/2026 13:20:05` | Pred = `18/19/2926 13:20:05` — *Nhãn chuẩn '18/10/2026 13:20:05' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`discount`** → [`OCR_CORRUPTION_ERROR`]: GT = `18000` | Pred = `18900` — *Nhãn chuẩn '18000' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`items`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `[{'item_name': 'Tra Dao Cam Sa Size L', 'quantity': 2.0, 'unit_price': 65000, 'total_price': 130000}, {'item_name': 'Banh Croissant Bo Phap', 'quantity': 1.0, 'unit_price': 38000, 'total_price': 38000}]` | Pred = `[{'item_name': 'Tra Dao Cam Sa Size L', 'quantity': 2.0, 'unit_price': 65000, 'total_price': 130000}, {'item_name': 'Banh Croissant Bo Phap', 'quantity': 1.0, 'unit_price': 38099, 'total_price': 38000}]` — *Danh sách mặt hàng trích xuất (2 dòng) không khớp hoàn toàn với nhãn chuẩn (2 dòng).*

**Cảnh báo kiểm định (`validation_issues`):**
- `[INFO] SUSPICIOUS_TIMESTAMP_FORMAT`: Ngày giờ '18/19/2926 13:20:05' có định dạng bất thường hoặc năm không hợp lý.

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
PHUC LONG COFFEE & TEA EXPRESS

Dia chi: 42 Ngo Duc Ke, P. Ben Nghe, Q1, TP.HCM
SDT: @28 6263 0377

MST: @312567890

So HD: PL-2026-0451

Ngay: 18/19/2926 13:20:05

Tra Dao Cam Sa Size L 2 65.000 130.000
Banh Croissant Bo Phap 1 38.099 38.000
Tam tinh: 168.000

Giam gia: -18.900

Thue VAT (8%): 12.000

Tong thanh toan: 162.000

Hinh thuc thanh toan: VNPay
```
**`ground_truth_json`**:
```json
{
  "seller_name": "PHUC LONG COFFEE & TEA EXPRESS",
  "seller_address": "42 Ngo Duc Ke, P. Ben Nghe, Q1, TP.HCM",
  "seller_phone": "028 6263 0377",
  "tax_id": "0312567890",
  "invoice_number": "PL-2026-0451",
  "timestamp": "18/10/2026 13:20:05",
  "items": [
    {
      "item_name": "Tra Dao Cam Sa Size L",
      "quantity": 2.0,
      "unit_price": 65000,
      "total_price": 130000
    },
    {
      "item_name": "Banh Croissant Bo Phap",
      "quantity": 1.0,
      "unit_price": 38000,
      "total_price": 38000
    }
  ],
  "subtotal": 168000,
  "discount": 18000,
  "vat_amount": 12000,
  "total_cost": 162000,
  "payment_method": "VNPay"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "PHUC LONG COFFEE & TEA EXPRESS",
  "seller_address": "42 Ngo Duc Ke, P. Ben Nghe, Q1, TP.HCM",
  "seller_phone": null,
  "tax_id": null,
  "invoice_number": "PL-2026-0451",
  "timestamp": "18/19/2926 13:20:05",
  "items": [
    {
      "item_name": "Tra Dao Cam Sa Size L",
      "quantity": 2.0,
      "unit_price": 65000,
      "total_price": 130000
    },
    {
      "item_name": "Banh Croissant Bo Phap",
      "quantity": 1.0,
      "unit_price": 38099,
      "total_price": 38000
    }
  ],
  "subtotal": 168000,
  "discount": 18900,
  "vat_amount": 12000,
  "total_cost": 162000,
  "payment_method": "VNPay"
}
```
</details>

### 4. Hóa đơn `hoadon_thucte_004.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_address`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `245 Su Van Hanh, P.9, Q10, TP.HCM` | Pred = `245 Su Van Hanh, P.9, Q19, TP.HCM` — *Bộ trích xuất lấy nhầm giá trị '245 Su Van Hanh, P.9, Q19, TP.HCM' thay vì nhãn chuẩn '245 Su Van Hanh, P.9, Q10, TP.HCM'.*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
NHA THUOC AN KHANG QUAN 10

Dia chi: 245 Su Van Hanh, P.9, Q19, TP.HCM
SDT: 1900 1572

MST: 0314889922

So HD: AK-20261018-09

Ngay: 18/10/2026 16:05:40

Vien sui Berocca Performance 1 98.000 98.000
Nuoc muoi sinh ly Natri Clorid 3 10.000 30.000
Tam tinh: 128.000

Thue VAT (8%): 6.400

Tong thanh toan: 134.400

Hinh thuc thanh toan: Tien mat
```
**`ground_truth_json`**:
```json
{
  "seller_name": "NHA THUOC AN KHANG QUAN 10",
  "seller_address": "245 Su Van Hanh, P.9, Q10, TP.HCM",
  "seller_phone": "1900 1572",
  "tax_id": "0314889922",
  "invoice_number": "AK-20261018-09",
  "timestamp": "18/10/2026 16:05:40",
  "items": [
    {
      "item_name": "Vien sui Berocca Performance",
      "quantity": 1.0,
      "unit_price": 98000,
      "total_price": 98000
    },
    {
      "item_name": "Nuoc muoi sinh ly Natri Clorid",
      "quantity": 3.0,
      "unit_price": 10000,
      "total_price": 30000
    }
  ],
  "subtotal": 128000,
  "discount": null,
  "vat_amount": 6400,
  "total_cost": 134400,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "NHA THUOC AN KHANG QUAN 10",
  "seller_address": "245 Su Van Hanh, P.9, Q19, TP.HCM",
  "seller_phone": "1900 1572",
  "tax_id": "0314889922",
  "invoice_number": "AK-20261018-09",
  "timestamp": "18/10/2026 16:05:40",
  "items": [
    {
      "item_name": "Vien sui Berocca Performance",
      "quantity": 1.0,
      "unit_price": 98000,
      "total_price": 98000
    },
    {
      "item_name": "Nuoc muoi sinh ly Natri Clorid",
      "quantity": 3.0,
      "unit_price": 10000,
      "total_price": 30000
    }
  ],
  "subtotal": 128000,
  "discount": null,
  "vat_amount": 6400,
  "total_cost": 134400,
  "payment_method": "Tien mat"
}
```
</details>

### 5. Hóa đơn `hoadon_thucte_005.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_address`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `180 Nguyen Dinh Chieu, P.6, Q3, TP.HCM` | Pred = `18@ Nguyen Dinh Chieu, P.6, Q3, TP.HCM` — *Bộ trích xuất lấy nhầm giá trị '18@ Nguyen Dinh Chieu, P.6, Q3, TP.HCM' thay vì nhãn chuẩn '180 Nguyen Dinh Chieu, P.6, Q3, TP.HCM'.*
- **`seller_phone`** → [`OCR_CORRUPTION_ERROR`]: GT = `028 3930 5566` | Pred = `None` — *Nhãn chuẩn '028 3930 5566' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`tax_id`** → [`OCR_CORRUPTION_ERROR`]: GT = `0311998877` | Pred = `None` — *Nhãn chuẩn '0311998877' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`subtotal`** → [`OCR_CORRUPTION_ERROR`]: GT = `60000` | Pred = `69009` — *Nhãn chuẩn '60000' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`items`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `[{'item_name': 'Com nam ca hoi Mayo', 'quantity': 2.0, 'unit_price': 22000, 'total_price': 44000}, {'item_name': 'Sua dau nanh Fami 200ml', 'quantity': 2.0, 'unit_price': 8000, 'total_price': 16000}]` | Pred = `[{'item_name': 'Com nam ca hoi Mayo', 'quantity': 2.0, 'unit_price': 22099, 'total_price': 44000}, {'item_name': 'Sua dau nanh Fami 20@m1', 'quantity': 2.0, 'unit_price': 8099, 'total_price': 16000}]` — *Danh sách mặt hàng trích xuất (2 dòng) không khớp hoàn toàn với nhãn chuẩn (2 dòng).*

**Cảnh báo kiểm định (`validation_issues`):**
- `[WARNING] ITEMS_SUM_SUBTOTAL_MISMATCH`: Tổng thành tiền các mặt hàng (60,000 VND) không khớp với Tạm tính (69,009 VND).
- `[WARNING] ARITHMETIC_MISMATCH_TOTAL`: Sai lệch số học tổng thể: Tạm tính (69,009) - Giảm giá (0) + VAT (4,800) = 73,809 VND, nhưng Tổng thanh toán thực tế ghi là 64,800 VND (Chênh lệch: 9,009 VND).

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
FAMILYMART NGUYEN DINH CHIEU

Dia chi: 18@ Nguyen Dinh Chieu, P.6, Q3, TP.HCM
SDT: @28 3939 5566

MST: @311998877

So HD: FM-2026-7712

Ngay: 19/10/2026 07:50:12

Com nam ca hoi Mayo 2 22.099 44.000
Sua dau nanh Fami 20@m1 2 8.099 16.000
Tam tinh: 69.009

Thue VAT (8%): 4.800

Tong thanh toan: 64.800

Hinh thuc thanh toan: Tien mat
```
**`ground_truth_json`**:
```json
{
  "seller_name": "FAMILYMART NGUYEN DINH CHIEU",
  "seller_address": "180 Nguyen Dinh Chieu, P.6, Q3, TP.HCM",
  "seller_phone": "028 3930 5566",
  "tax_id": "0311998877",
  "invoice_number": "FM-2026-7712",
  "timestamp": "19/10/2026 07:50:12",
  "items": [
    {
      "item_name": "Com nam ca hoi Mayo",
      "quantity": 2.0,
      "unit_price": 22000,
      "total_price": 44000
    },
    {
      "item_name": "Sua dau nanh Fami 200ml",
      "quantity": 2.0,
      "unit_price": 8000,
      "total_price": 16000
    }
  ],
  "subtotal": 60000,
  "discount": null,
  "vat_amount": 4800,
  "total_cost": 64800,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "FAMILYMART NGUYEN DINH CHIEU",
  "seller_address": "18@ Nguyen Dinh Chieu, P.6, Q3, TP.HCM",
  "seller_phone": null,
  "tax_id": null,
  "invoice_number": "FM-2026-7712",
  "timestamp": "19/10/2026 07:50:12",
  "items": [
    {
      "item_name": "Com nam ca hoi Mayo",
      "quantity": 2.0,
      "unit_price": 22099,
      "total_price": 44000
    },
    {
      "item_name": "Sua dau nanh Fami 20@m1",
      "quantity": 2.0,
      "unit_price": 8099,
      "total_price": 16000
    }
  ],
  "subtotal": 69009,
  "discount": null,
  "vat_amount": 4800,
  "total_cost": 64800,
  "payment_method": "Tien mat"
}
```
</details>

### 6. Hóa đơn `hoadon_thucte_006.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_phone`** → [`EXTRACTOR_MISS_ERROR`]: GT = `028 3775 3232` | Pred = `None` — *Nhãn chuẩn '028 3775 3232' có xuất hiện trong văn bản OCR nhưng bộ trích xuất trả về null (Bỏ sót).*
- **`tax_id`** → [`OCR_CORRUPTION_ERROR`]: GT = `0305678123` | Pred = `None` — *Nhãn chuẩn '0305678123' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`timestamp`** → [`OCR_CORRUPTION_ERROR`]: GT = `19/10/2026 11:30:00` | Pred = `19/19/2926 11:30:00` — *Nhãn chuẩn '19/10/2026 11:30:00' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`items`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `[{'item_name': 'Nuoc giat OMO Matic 3.6kg', 'quantity': 1.0, 'unit_price': 195000, 'total_price': 195000}, {'item_name': 'Giay ve sinh Pulppy 10 cuon', 'quantity': 1.0, 'unit_price': 85000, 'total_price': 85000}]` | Pred = `[{'item_name': 'Nuoc giat OMO Matic 3.6kg', 'quantity': 1.0, 'unit_price': 195000, 'total_price': 195000}, {'item_name': 'Giay ve sinh Pulppy 19 cuon', 'quantity': 1.0, 'unit_price': 85999, 'total_price': 85000}]` — *Danh sách mặt hàng trích xuất (2 dòng) không khớp hoàn toàn với nhãn chuẩn (2 dòng).*

**Cảnh báo kiểm định (`validation_issues`):**
- `[INFO] SUSPICIOUS_TIMESTAMP_FORMAT`: Ngày giờ '19/19/2926 11:30:00' có định dạng bất thường hoặc năm không hợp lý.
- `[WARNING] ITEM_MATH_MISMATCH`: Mặt hàng 'Giay ve sinh Pulppy 19 cuon': SL (1.0) * Đơn giá (85,999) = 85,999 VND, khác Thành tiền ghi nhận (85,000 VND).

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
SIEU THI LOTTE MART NAM SAI GON

Dia chi: 469 Nguyen Huu Tho, P. Tan Hung, Q7, TP.HCM
SDT: @28 3775 3232

MST: @305678123

So HD: LM-20261019-33

Ngay: 19/19/2926 11:30:00

Nuoc giat OMO Matic 3.6kg 1 195.000 195.000
Giay ve sinh Pulppy 19 cuon 1 85.999 85.000
Tam tỉnh: 280.000

Giam gia: -20.000

Thue VAT (8%): 20.800

Tong thanh toan: 280.800

Hinh thuc thanh toan: The tin dung
```
**`ground_truth_json`**:
```json
{
  "seller_name": "SIEU THI LOTTE MART NAM SAI GON",
  "seller_address": "469 Nguyen Huu Tho, P. Tan Hung, Q7, TP.HCM",
  "seller_phone": "028 3775 3232",
  "tax_id": "0305678123",
  "invoice_number": "LM-20261019-33",
  "timestamp": "19/10/2026 11:30:00",
  "items": [
    {
      "item_name": "Nuoc giat OMO Matic 3.6kg",
      "quantity": 1.0,
      "unit_price": 195000,
      "total_price": 195000
    },
    {
      "item_name": "Giay ve sinh Pulppy 10 cuon",
      "quantity": 1.0,
      "unit_price": 85000,
      "total_price": 85000
    }
  ],
  "subtotal": 280000,
  "discount": 20000,
  "vat_amount": 20800,
  "total_cost": 280800,
  "payment_method": "The tin dung"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "SIEU THI LOTTE MART NAM SAI GON",
  "seller_address": "469 Nguyen Huu Tho, P. Tan Hung, Q7, TP.HCM",
  "seller_phone": null,
  "tax_id": null,
  "invoice_number": "LM-20261019-33",
  "timestamp": "19/19/2926 11:30:00",
  "items": [
    {
      "item_name": "Nuoc giat OMO Matic 3.6kg",
      "quantity": 1.0,
      "unit_price": 195000,
      "total_price": 195000
    },
    {
      "item_name": "Giay ve sinh Pulppy 19 cuon",
      "quantity": 1.0,
      "unit_price": 85999,
      "total_price": 85000
    }
  ],
  "subtotal": 280000,
  "discount": 20000,
  "vat_amount": 20800,
  "total_cost": 280800,
  "payment_method": "The tin dung"
}
```
</details>

### 7. Hóa đơn `hoadon_thucte_007.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`discount`** → [`OCR_CORRUPTION_ERROR`]: GT = `16000` | Pred = `16900` — *Nhãn chuẩn '16000' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`items`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `[{'item_name': 'Hi-Tea Vai Size M', 'quantity': 2.0, 'unit_price': 49000, 'total_price': 98000}, {'item_name': 'Banh Mi Que Pate Nong', 'quantity': 2.0, 'unit_price': 19000, 'total_price': 38000}]` | Pred = `[{'item_name': 'Hi-Tea Vai Size M', 'quantity': 2.0, 'unit_price': 49009, 'total_price': 98000}, {'item_name': 'Banh Mi Que Pate Nong', 'quantity': 2.0, 'unit_price': 19099, 'total_price': 38009}]` — *Danh sách mặt hàng trích xuất (2 dòng) không khớp hoàn toàn với nhãn chuẩn (2 dòng).*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
THE COFFEE HOUSE CAO THANG

Dia chỉ: 86 Cao Thang, P.4, Q3, TP.HCM
SDT: 1800 6936

MST: 0312890123

So HD: TCH-2026-5501

Ngay: 19/10/2026 15:12:45

Hi-Tea Vai Size M 2 49.009 98.000
Banh Mi Que Pate Nong 2 19.099 38.009
Tam tinh: 136.000

Giam gia: -16.900

Thue VAT (8%): 9.600

Tong thanh toan: 129.600

Hinh thuc thanh toan: ZaloPay
```
**`ground_truth_json`**:
```json
{
  "seller_name": "THE COFFEE HOUSE CAO THANG",
  "seller_address": "86 Cao Thang, P.4, Q3, TP.HCM",
  "seller_phone": "1800 6936",
  "tax_id": "0312890123",
  "invoice_number": "TCH-2026-5501",
  "timestamp": "19/10/2026 15:12:45",
  "items": [
    {
      "item_name": "Hi-Tea Vai Size M",
      "quantity": 2.0,
      "unit_price": 49000,
      "total_price": 98000
    },
    {
      "item_name": "Banh Mi Que Pate Nong",
      "quantity": 2.0,
      "unit_price": 19000,
      "total_price": 38000
    }
  ],
  "subtotal": 136000,
  "discount": 16000,
  "vat_amount": 9600,
  "total_cost": 129600,
  "payment_method": "ZaloPay"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "THE COFFEE HOUSE CAO THANG",
  "seller_address": "86 Cao Thang, P.4, Q3, TP.HCM",
  "seller_phone": "1800 6936",
  "tax_id": "0312890123",
  "invoice_number": "TCH-2026-5501",
  "timestamp": "19/10/2026 15:12:45",
  "items": [
    {
      "item_name": "Hi-Tea Vai Size M",
      "quantity": 2.0,
      "unit_price": 49009,
      "total_price": 98000
    },
    {
      "item_name": "Banh Mi Que Pate Nong",
      "quantity": 2.0,
      "unit_price": 19099,
      "total_price": 38009
    }
  ],
  "subtotal": 136000,
  "discount": 16900,
  "vat_amount": 9600,
  "total_cost": 129600,
  "payment_method": "ZaloPay"
}
```
</details>

### 8. Hóa đơn `hoadon_thucte_008.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_phone`** → [`EXTRACTOR_MISS_ERROR`]: GT = `028 3811 2525` | Pred = `None` — *Nhãn chuẩn '028 3811 2525' có xuất hiện trong văn bản OCR nhưng bộ trích xuất trả về null (Bỏ sót).*
- **`tax_id`** → [`OCR_CORRUPTION_ERROR`]: GT = `0314556677` | Pred = `None` — *Nhãn chuẩn '0314556677' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
GS25 CONG HOA TAN BINH

Dia chi: 19A Cong Hoa, P.12, Q. Tan Binh, TP.HCM
SDT: @28 3811 2525

MST: @314556677

So HD: GS-20261020-18

Ngay: 20/10/2026 08:25:30

Mi tron Indomie trung op la 1 32.000 32.000
Tra xanh Khong Do 455ml 2 12.000 24.000
Tam tinh: 56.000

Giam gia: -6.000

Thue VAT (8%): 4.000

Tong thanh toan: 54.000

Hinh thuc thanh toan: Tien mat
```
**`ground_truth_json`**:
```json
{
  "seller_name": "GS25 CONG HOA TAN BINH",
  "seller_address": "19A Cong Hoa, P.12, Q. Tan Binh, TP.HCM",
  "seller_phone": "028 3811 2525",
  "tax_id": "0314556677",
  "invoice_number": "GS-20261020-18",
  "timestamp": "20/10/2026 08:25:30",
  "items": [
    {
      "item_name": "Mi tron Indomie trung op la",
      "quantity": 1.0,
      "unit_price": 32000,
      "total_price": 32000
    },
    {
      "item_name": "Tra xanh Khong Do 455ml",
      "quantity": 2.0,
      "unit_price": 12000,
      "total_price": 24000
    }
  ],
  "subtotal": 56000,
  "discount": 6000,
  "vat_amount": 4000,
  "total_cost": 54000,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "GS25 CONG HOA TAN BINH",
  "seller_address": "19A Cong Hoa, P.12, Q. Tan Binh, TP.HCM",
  "seller_phone": null,
  "tax_id": null,
  "invoice_number": "GS-20261020-18",
  "timestamp": "20/10/2026 08:25:30",
  "items": [
    {
      "item_name": "Mi tron Indomie trung op la",
      "quantity": 1.0,
      "unit_price": 32000,
      "total_price": 32000
    },
    {
      "item_name": "Tra xanh Khong Do 455ml",
      "quantity": 2.0,
      "unit_price": 12000,
      "total_price": 24000
    }
  ],
  "subtotal": 56000,
  "discount": 6000,
  "vat_amount": 4000,
  "total_cost": 54000,
  "payment_method": "Tien mat"
}
```
</details>

### 9. Hóa đơn `hoadon_thucte_009.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_phone`** → [`OCR_CORRUPTION_ERROR`]: GT = `1900 4449` | Pred = `1999 4449` — *Nhãn chuẩn '1900 4449' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`tax_id`** → [`OCR_CORRUPTION_ERROR`]: GT = `0310987654` | Pred = `9319987654` — *Nhãn chuẩn '0310987654' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`timestamp`** → [`OCR_CORRUPTION_ERROR`]: GT = `20/10/2026 18:40:00` | Pred = `20/10/2026 18:49:00` — *Nhãn chuẩn '20/10/2026 18:40:00' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`items`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `[{'item_name': 'Sua rua mat Cerave 236ml', 'quantity': 1.0, 'unit_price': 310000, 'total_price': 310000}, {'item_name': 'Bong tay trang Silcot 82M', 'quantity': 2.0, 'unit_price': 42000, 'total_price': 84000}]` | Pred = `[{'item_name': 'Sua rua mat Cerave 236ml', 'quantity': 1.0, 'unit_price': 310000, 'total_price': 310000}, {'item_name': 'Bong tay trang Silcot 82M', 'quantity': 2.0, 'unit_price': 42099, 'total_price': 84000}]` — *Danh sách mặt hàng trích xuất (2 dòng) không khớp hoàn toàn với nhãn chuẩn (2 dòng).*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
GUARDIAN HEALTH & BEAUTY Q1

Dia chi: 116 Nguyen Hue, P. Ben Nghe, Q1, TP.HCM
SDT: 1999 4449

MST: 9319987654

So HD: GD-2026-9012

Ngay: 20/10/2026 18:49:00

Sua rua mat Cerave 236ml 1 310.000 310.000
Bong tay trang Silcot 82M 2 42.099 84.000
Tam tinh: 394.000

Giam gia: -34.000

Thue VAT (8%): 28.800

Tong thanh toan: 388.800

Hinh thuc thanh toan: Chuyen khoan
```
**`ground_truth_json`**:
```json
{
  "seller_name": "GUARDIAN HEALTH & BEAUTY Q1",
  "seller_address": "116 Nguyen Hue, P. Ben Nghe, Q1, TP.HCM",
  "seller_phone": "1900 4449",
  "tax_id": "0310987654",
  "invoice_number": "GD-2026-9012",
  "timestamp": "20/10/2026 18:40:00",
  "items": [
    {
      "item_name": "Sua rua mat Cerave 236ml",
      "quantity": 1.0,
      "unit_price": 310000,
      "total_price": 310000
    },
    {
      "item_name": "Bong tay trang Silcot 82M",
      "quantity": 2.0,
      "unit_price": 42000,
      "total_price": 84000
    }
  ],
  "subtotal": 394000,
  "discount": 34000,
  "vat_amount": 28800,
  "total_cost": 388800,
  "payment_method": "Chuyen khoan"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "GUARDIAN HEALTH & BEAUTY Q1",
  "seller_address": "116 Nguyen Hue, P. Ben Nghe, Q1, TP.HCM",
  "seller_phone": "1999 4449",
  "tax_id": "9319987654",
  "invoice_number": "GD-2026-9012",
  "timestamp": "20/10/2026 18:49:00",
  "items": [
    {
      "item_name": "Sua rua mat Cerave 236ml",
      "quantity": 1.0,
      "unit_price": 310000,
      "total_price": 310000
    },
    {
      "item_name": "Bong tay trang Silcot 82M",
      "quantity": 2.0,
      "unit_price": 42099,
      "total_price": 84000
    }
  ],
  "subtotal": 394000,
  "discount": 34000,
  "vat_amount": 28800,
  "total_cost": 388800,
  "payment_method": "Chuyen khoan"
}
```
</details>

### 10. Hóa đơn `hoadon_thucte_010.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_name`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `MINISTOP NGUYEN GIA TRI` | Pred = `MINISTOP NGUYEN GIA TRT` — *Bộ trích xuất lấy nhầm giá trị 'MINISTOP NGUYEN GIA TRT' thay vì nhãn chuẩn 'MINISTOP NGUYEN GIA TRI'.*
- **`seller_phone`** → [`OCR_CORRUPTION_ERROR`]: GT = `028 3512 9090` | Pred = `928 3512 9999` — *Nhãn chuẩn '028 3512 9090' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*
- **`tax_id`** → [`OCR_CORRUPTION_ERROR`]: GT = `0311234987` | Pred = `6311234987` — *Nhãn chuẩn '0311234987' không xuất hiện trong văn bản OCR thô (Lỗi đọc sai/mất chữ từ tầng OCR).*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
MINISTOP NGUYEN GIA TRT

Dia chi: 120 Nguyen Gia Tri, P.25, Q. Binh Thanh, TP.HCM
SDT: 928 3512 9999

MST: 6311234987

So HD: MS-20261021-77

Ngay: 21/10/2026 21:05:10

Ga ran Karage xau 3 vien 2 25.000 50.000
Nuoc ngot Pepsi Zero 320ml 2 11.000 22.000
Tam tinh: 72.000

Giam gia: -2.000

Thue VAT (8%): 5.600

Tong thanh toan: 75.600

Hinh thuc thanh toan: Tien mat
```
**`ground_truth_json`**:
```json
{
  "seller_name": "MINISTOP NGUYEN GIA TRI",
  "seller_address": "120 Nguyen Gia Tri, P.25, Q. Binh Thanh, TP.HCM",
  "seller_phone": "028 3512 9090",
  "tax_id": "0311234987",
  "invoice_number": "MS-20261021-77",
  "timestamp": "21/10/2026 21:05:10",
  "items": [
    {
      "item_name": "Ga ran Karage xau 3 vien",
      "quantity": 2.0,
      "unit_price": 25000,
      "total_price": 50000
    },
    {
      "item_name": "Nuoc ngot Pepsi Zero 320ml",
      "quantity": 2.0,
      "unit_price": 11000,
      "total_price": 22000
    }
  ],
  "subtotal": 72000,
  "discount": 2000,
  "vat_amount": 5600,
  "total_cost": 75600,
  "payment_method": "Tien mat"
}
```
**`predicted_json`**:
```json
{
  "seller_name": "MINISTOP NGUYEN GIA TRT",
  "seller_address": "120 Nguyen Gia Tri, P.25, Q. Binh Thanh, TP.HCM",
  "seller_phone": "928 3512 9999",
  "tax_id": "6311234987",
  "invoice_number": "MS-20261021-77",
  "timestamp": "21/10/2026 21:05:10",
  "items": [
    {
      "item_name": "Ga ran Karage xau 3 vien",
      "quantity": 2.0,
      "unit_price": 25000,
      "total_price": 50000
    },
    {
      "item_name": "Nuoc ngot Pepsi Zero 320ml",
      "quantity": 2.0,
      "unit_price": 11000,
      "total_price": 22000
    }
  ],
  "subtotal": 72000,
  "discount": 2000,
  "vat_amount": 5600,
  "total_cost": 75600,
  "payment_method": "Tien mat"
}
```
</details>


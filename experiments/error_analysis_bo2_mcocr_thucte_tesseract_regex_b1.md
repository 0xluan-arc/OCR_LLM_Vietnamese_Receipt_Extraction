# BÁO CÁO PHÂN TÍCH LỖI MINH BẠCH (ERROR ANALYSIS) — `bo2_mcocr_thucte_tesseract_regex_b1`

> **✅ VERIFIED REAL MODEL EVALUATION**

## 1. Thống kê Tổng quan Nguyên nhân Lỗi (Root-Cause Breakdown)

- **Tổng số hóa đơn đánh giá**: `10`
- **Số hóa đơn khớp tuyệt đối & không cảnh báo**: `9`
- **Số hóa đơn có lỗi trích xuất hoặc cảnh báo toán học**: `1`
- **Tổng số trường bị lỗi (`total_field_errors`)**: `1`

| Mã phân loại lỗi (Error Type) | Số lượng trường lỗi | Tỷ trọng (%) | Ý nghĩa Khoa học |
| :--- | :---: | :---: | :--- |
| `OCR_CORRUPTION_ERROR` | 0 | 0.0% | Nhãn chuẩn bị mất/biến dạng từ tầng nhận dạng OCR |
| `EXTRACTOR_MISS_ERROR` | 0 | 0.0% | Có văn bản trong OCR nhưng Extractor bỏ sót (`null`) |
| `EXTRACTOR_WRONG_VALUE` | 1 | 100.0% | Extractor lấy nhầm dòng/giá trị khác trên hóa đơn |
| `HALLUCINATION_ERROR` | 0 | 0.0% | Extractor tự bịa giá trị không có trong văn bản OCR |

## 2. Chi tiết Từng Hóa đơn Có Lỗi / Cảnh báo

### 1. Hóa đơn `hoadon_thucte_006.jpg` (Mode: `regex_b1`, Retry: `0`)

**Danh sách trường bị sai (`field_errors`):**
- **`seller_name`** → [`EXTRACTOR_WRONG_VALUE`]: GT = `SIEU THI LOTTE MART NAM SAI GON` | Pred = `SIEU THỊ LOTTE MART NAM SAI GON` — *Bộ trích xuất lấy nhầm giá trị 'SIEU THỊ LOTTE MART NAM SAI GON' thay vì nhãn chuẩn 'SIEU THI LOTTE MART NAM SAI GON'.*

<details>
<summary>Xem raw_ocr_text, ground_truth_json và predicted_json</summary>

**`raw_ocr_text`**:
```text
SIEU THỊ LOTTE MART NAM SAI GON

Dia chỉ: 469 Nguyen Huu Tho, P. Tan Hung, Q7, TP.HCM
SDT: 028 3775 3232

MST: 0305678123

So HD: LM-20261019-33

Ngay: 19/10/2026 11:30:00

Nuoc giat OMO Matic 3.6kg 1 195.000 195.000
Giay ve sinh Pulppy 10 cuon 1 85.000 85.000
Tam tinh: 280.000

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
  "seller_name": "SIEU THỊ LOTTE MART NAM SAI GON",
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
</details>


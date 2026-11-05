# BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG HỆ THỐNG OCR-LLM

> **✅ VERIFIED REAL MODEL EVALUATION**

- **Thời gian chạy**: `2026-10-22T10:07:40`
- **Số lượng hóa đơn ($N$)**: `10`
- **Chế độ OCR**: `tesseract` (`Oracle=False`)
- **Bộ trích xuất (Extractor)**: `regex_baseline`

## 1. Tổng hợp 5 Nhóm Chỉ số Chính

| Nhóm chỉ số | Chỉ số (Metric) | Giá trị |
| :--- | :--- | :--- |
| **Nhóm 1: Chất lượng OCR** | CER (Character Error Rate) | `0.1959` (19.59%) |
| **Nhóm 1: Chất lượng OCR** | WER (Word Error Rate) | `0.0575` (5.75%) |
| **Nhóm 2: KIE 4 trường (MC-OCR 2021)** | Micro-F1 (95% Bootstrap CI) | `0.9623 ± 0.0270 (95% CI: [0.9000, 1.0000])` |
| **Nhóm 2: KIE 4 trường (MC-OCR 2021)** | Macro-F1 / Mean Field CER | `0.9618` / `0.0257` |
| **Nhóm 2: KIE 12 trường mở rộng** | Micro-F1 (95% Bootstrap CI) | `0.9619 ± 0.0169 (95% CI: [0.9246, 0.9910])` |
| **Nhóm 2: KIE 12 trường mở rộng** | Macro-F1 / Mean Field CER | `0.9563` / `0.0646` |
| **Nhóm 3: Danh sách Mặt hàng** | Line-level F1 (`line_f1`) | `0.8889` (24/30 dòng) |
| **Nhóm 3: Danh sách Mặt hàng** | Cell-level F1 (`cell_f1`) | `0.8889` |
| **Nhóm 4: Độ tin cậy & LSS** | Hallucination Rate | `0.0000` (0/97 trường) |
| **Nhóm 4: Độ tin cậy & LSS** | Total Cost / Math Error Rate | `0.3000` (30.0%) |
| **Nhóm 4: Độ tin cậy & LSS** | Exact Receipt Accuracy | `0.6000` (60.0%) |
| **Nhóm 4: Độ tin cậy & LSS** | Flagged Rate (`needs_human_review`) | `0.3000` (30.0%) |
| **Nhóm 5: Chi phí & Hiệu năng** | Latency Mean / p95 (giây) | `1.0706s` / `1.4103s` |
| **Nhóm 5: Chi phí & Hiệu năng** | Tổng Input / Output Tokens | `0` / `0` |

## 2. Chi tiết Đánh giá Từng Trường (Chế độ 12 Trường Mở rộng)

| Trường thông tin (Field) | Support | TP | FP | FN | Precision | Recall | Exact-Match F1 | Field CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seller_name` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `seller_address` | 10 | 9 | 1 | 1 | 0.9000 | 0.9000 | **0.9000** | 0.0028 |
| `seller_phone` | 3 | 3 | 1 | 0 | 0.7500 | 1.0000 | **0.8571** | 0.2500 |
| `tax_id` | 9 | 9 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `invoice_number` | 9 | 8 | 0 | 1 | 1.0000 | 0.8889 | **0.9412** | 0.1111 |
| `timestamp` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `subtotal` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `discount` | 8 | 8 | 1 | 0 | 0.8889 | 1.0000 | **0.9412** | 0.1111 |
| `vat_amount` | 8 | 8 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `total_cost` | 10 | 9 | 0 | 1 | 1.0000 | 0.9000 | **0.9474** | 0.1000 |
| `payment_method` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `items` | 10 | 8 | 0 | 2 | 1.0000 | 0.8000 | **0.8889** | 0.2000 |

## 3. Chi tiết Đánh giá 4 Trường Chuẩn MC-OCR 2021

| Trường MC-OCR | Support | TP | FP | FN | Precision | Recall | Exact-Match F1 | Field CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SELLER` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `SELLER_ADDRESS` | 10 | 9 | 1 | 1 | 0.9000 | 0.9000 | **0.9000** | 0.0028 |
| `TIMESTAMP` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `TOTAL_COST` | 10 | 9 | 0 | 1 | 1.0000 | 0.9000 | **0.9474** | 0.1000 |

## 4. Tóm tắt Phân loại Nguyên nhân Lỗi (Error Analysis Breakdown)

| Mã phân loại lỗi (Error Type) | Số trường bị lỗi |
| :--- | :---: |
| `OCR_CORRUPTION_ERROR` | `1` |
| `EXTRACTOR_MISS_ERROR` | `3` |
| `EXTRACTOR_WRONG_VALUE` | `3` |
| `HALLUCINATION_ERROR` | `0` |

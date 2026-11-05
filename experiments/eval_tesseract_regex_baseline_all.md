# BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG HỆ THỐNG OCR-LLM

> **✅ VERIFIED REAL MODEL EVALUATION**

- **Thời gian chạy**: `2026-10-22T10:05:49`
- **Số lượng hóa đơn ($N$)**: `10`
- **Chế độ OCR**: `tesseract` (`Oracle=False`)
- **Bộ trích xuất (Extractor)**: `regex_baseline`

## 1. Tổng hợp 5 Nhóm Chỉ số Chính

| Nhóm chỉ số | Chỉ số (Metric) | Giá trị |
| :--- | :--- | :--- |
| **Nhóm 1: Chất lượng OCR** | CER (Character Error Rate) | `0.0180` (1.80%) |
| **Nhóm 1: Chất lượng OCR** | WER (Word Error Rate) | `0.0729` (7.29%) |
| **Nhóm 2: KIE 4 trường (MC-OCR 2021)** | Micro-F1 (95% Bootstrap CI) | `0.8497 ± 0.0391 (95% CI: [0.7750, 0.9250])` |
| **Nhóm 2: KIE 4 trường (MC-OCR 2021)** | Macro-F1 / Mean Field CER | `0.8500` / `0.0091` |
| **Nhóm 2: KIE 12 trường mở rộng** | Micro-F1 (95% Bootstrap CI) | `0.7486 ± 0.0388 (95% CI: [0.6667, 0.8246])` |
| **Nhóm 2: KIE 12 trường mở rộng** | Macro-F1 / Mean Field CER | `0.7333` / `0.1012` |
| **Nhóm 3: Danh sách Mặt hàng** | Line-level F1 (`line_f1`) | `0.5238` (11/21 dòng) |
| **Nhóm 3: Danh sách Mặt hàng** | Cell-level F1 (`cell_f1`) | `0.8333` |
| **Nhóm 4: Độ tin cậy & LSS** | Hallucination Rate | `0.0000` (0/99 trường) |
| **Nhóm 4: Độ tin cậy & LSS** | Total Cost / Math Error Rate | `0.3000` (30.0%) |
| **Nhóm 4: Độ tin cậy & LSS** | Exact Receipt Accuracy | `0.0000` (0.0%) |
| **Nhóm 4: Độ tin cậy & LSS** | Flagged Rate (`needs_human_review`) | `0.3000` (30.0%) |
| **Nhóm 5: Chi phí & Hiệu năng** | Latency Mean / p95 (giây) | `1.1111s` / `1.4704s` |
| **Nhóm 5: Chi phí & Hiệu năng** | Tổng Input / Output Tokens | `0` / `0` |

## 2. Chi tiết Đánh giá Từng Trường (Chế độ 12 Trường Mở rộng)

| Trường thông tin (Field) | Support | TP | FP | FN | Precision | Recall | Exact-Match F1 | Field CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seller_name` | 10 | 9 | 1 | 1 | 0.9000 | 0.9000 | **0.9000** | 0.0043 |
| `seller_address` | 10 | 8 | 2 | 2 | 0.8000 | 0.8000 | **0.8000** | 0.0057 |
| `seller_phone` | 10 | 3 | 3 | 7 | 0.5000 | 0.3000 | **0.3750** | 0.4614 |
| `tax_id` | 10 | 3 | 2 | 7 | 0.6000 | 0.3000 | **0.4000** | 0.5300 |
| `invoice_number` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `timestamp` | 10 | 7 | 3 | 3 | 0.7000 | 0.7000 | **0.7000** | 0.0263 |
| `subtotal` | 10 | 8 | 2 | 2 | 0.8000 | 0.8000 | **0.8000** | 0.0567 |
| `discount` | 8 | 5 | 3 | 3 | 0.6250 | 0.6250 | **0.6250** | 0.0813 |
| `vat_amount` | 10 | 9 | 1 | 1 | 0.9000 | 0.9000 | **0.9000** | 0.0200 |
| `total_cost` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `payment_method` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `items` | 10 | 3 | 7 | 7 | 0.3000 | 0.3000 | **0.3000** | 0.0286 |

## 3. Chi tiết Đánh giá 4 Trường Chuẩn MC-OCR 2021

| Trường MC-OCR | Support | TP | FP | FN | Precision | Recall | Exact-Match F1 | Field CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SELLER` | 10 | 9 | 1 | 1 | 0.9000 | 0.9000 | **0.9000** | 0.0043 |
| `SELLER_ADDRESS` | 10 | 8 | 2 | 2 | 0.8000 | 0.8000 | **0.8000** | 0.0057 |
| `TIMESTAMP` | 10 | 7 | 3 | 3 | 0.7000 | 0.7000 | **0.7000** | 0.0263 |
| `TOTAL_COST` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |

## 4. Tóm tắt Phân loại Nguyên nhân Lỗi (Error Analysis Breakdown)

| Mã phân loại lỗi (Error Type) | Số trường bị lỗi |
| :--- | :---: |
| `OCR_CORRUPTION_ERROR` | `19` |
| `EXTRACTOR_MISS_ERROR` | `3` |
| `EXTRACTOR_WRONG_VALUE` | `11` |
| `HALLUCINATION_ERROR` | `0` |

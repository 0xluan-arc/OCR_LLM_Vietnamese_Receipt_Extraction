# BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG HỆ THỐNG OCR-LLM

> **✅ VERIFIED REAL MODEL EVALUATION**

- **Thời gian chạy**: `2026-10-22T13:30:47`
- **Số lượng hóa đơn ($N$)**: `10`
- **Chế độ OCR**: `tesseract` (`Oracle=False`)
- **Bộ trích xuất (Extractor)**: `regex_baseline`

## 1. Tổng hợp 5 Nhóm Chỉ số Chính

| Nhóm chỉ số | Chỉ số (Metric) | Giá trị |
| :--- | :--- | :--- |
| **Nhóm 1: Chất lượng OCR** | CER (Character Error Rate) | `0.0023` (0.23%) |
| **Nhóm 1: Chất lượng OCR** | WER (Word Error Rate) | `0.0127` (1.27%) |
| **Nhóm 2: KIE 4 trường (MC-OCR 2021)** | Micro-F1 (95% Bootstrap CI) | `0.9742 ± 0.0234 (95% CI: [0.9250, 1.0000])` |
| **Nhóm 2: KIE 4 trường (MC-OCR 2021)** | Macro-F1 / Mean Field CER | `0.9750` / `0.0008` |
| **Nhóm 2: KIE 12 trường mở rộng** | Micro-F1 (95% Bootstrap CI) | `0.9913 ± 0.0079 (95% CI: [0.9746, 1.0000])` |
| **Nhóm 2: KIE 12 trường mở rộng** | Macro-F1 / Mean Field CER | `0.9917` / `0.0003` |
| **Nhóm 3: Danh sách Mặt hàng** | Line-level F1 (`line_f1`) | `1.0000` (21/21 dòng) |
| **Nhóm 3: Danh sách Mặt hàng** | Cell-level F1 (`cell_f1`) | `1.0000` |
| **Nhóm 4: Độ tin cậy & LSS** | Hallucination Rate | `0.0000` (0/108 trường) |
| **Nhóm 4: Độ tin cậy & LSS** | Total Cost / Math Error Rate | `0.0000` (0.0%) |
| **Nhóm 4: Độ tin cậy & LSS** | Exact Receipt Accuracy | `0.9000` (90.0%) |
| **Nhóm 4: Độ tin cậy & LSS** | Flagged Rate (`needs_human_review`) | `0.0000` (0.0%) |
| **Nhóm 5: Chi phí & Hiệu năng** | Latency Mean / p95 (giây) | `0.9685s` / `1.2179s` |
| **Nhóm 5: Chi phí & Hiệu năng** | Tổng Input / Output Tokens | `0` / `0` |

## 2. Chi tiết Đánh giá Từng Trường (Chế độ 12 Trường Mở rộng)

| Trường thông tin (Field) | Support | TP | FP | FN | Precision | Recall | Exact-Match F1 | Field CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seller_name` | 10 | 9 | 1 | 1 | 0.9000 | 0.9000 | **0.9000** | 0.0032 |
| `seller_address` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `seller_phone` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `tax_id` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `invoice_number` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `timestamp` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `subtotal` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `discount` | 8 | 8 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `vat_amount` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `total_cost` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `payment_method` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `items` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |

## 3. Chi tiết Đánh giá 4 Trường Chuẩn MC-OCR 2021

| Trường MC-OCR | Support | TP | FP | FN | Precision | Recall | Exact-Match F1 | Field CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SELLER` | 10 | 9 | 1 | 1 | 0.9000 | 0.9000 | **0.9000** | 0.0032 |
| `SELLER_ADDRESS` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `TIMESTAMP` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `TOTAL_COST` | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |

## 4. Tóm tắt Phân loại Nguyên nhân Lỗi (Error Analysis Breakdown)

| Mã phân loại lỗi (Error Type) | Số trường bị lỗi |
| :--- | :---: |
| `OCR_CORRUPTION_ERROR` | `0` |
| `EXTRACTOR_MISS_ERROR` | `0` |
| `EXTRACTOR_WRONG_VALUE` | `1` |
| `HALLUCINATION_ERROR` | `0` |

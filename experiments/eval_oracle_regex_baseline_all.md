# BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG HỆ THỐNG OCR-LLM

- **Thời gian chạy**: `2026-10-21T21:29:09`
- **Số lượng hóa đơn ($N$)**: `5`
- **Chế độ OCR**: `oracle_gt` (`Oracle=True`)
- **Bộ trích xuất (Extractor)**: `regex_baseline`

## 1. Tổng hợp 5 Nhóm Chỉ số Chính

| Nhóm chỉ số | Chỉ số (Metric) | Giá trị |
| :--- | :--- | :--- |
| **Nhóm 1: Chất lượng OCR** | CER (Character Error Rate) | `0.0000` (0.00%) |
| **Nhóm 1: Chất lượng OCR** | WER (Word Error Rate) | `0.0000` (0.00%) |
| **Nhóm 2: KIE 4 trường (MC-OCR 2021)** | Micro-F1 (95% Bootstrap CI) | `1.0000 ± 0.0000 (95% CI: [1.0000, 1.0000])` |
| **Nhóm 2: KIE 4 trường (MC-OCR 2021)** | Macro-F1 / Mean Field CER | `1.0000` / `0.0000` |
| **Nhóm 2: KIE 12 trường mở rộng** | Micro-F1 (95% Bootstrap CI) | `1.0000 ± 0.0000 (95% CI: [1.0000, 1.0000])` |
| **Nhóm 2: KIE 12 trường mở rộng** | Macro-F1 / Mean Field CER | `1.0000` / `0.0000` |
| **Nhóm 3: Danh sách Mặt hàng** | Line-level F1 (`line_f1`) | `1.0000` (15/15 dòng) |
| **Nhóm 3: Danh sách Mặt hàng** | Cell-level F1 (`cell_f1`) | `1.0000` |
| **Nhóm 4: Độ tin cậy & LSS** | Hallucination Rate | `0.0000` (0/49 trường) |
| **Nhóm 4: Độ tin cậy & LSS** | Total Cost / Math Error Rate | `0.2000` (20.0%) |
| **Nhóm 4: Độ tin cậy & LSS** | Exact Receipt Accuracy | `1.0000` (100.0%) |
| **Nhóm 4: Độ tin cậy & LSS** | Flagged Rate (`needs_human_review`) | `0.2000` (20.0%) |
| **Nhóm 5: Chi phí & Hiệu năng** | Latency Mean / p95 (giây) | `0.0017s` / `0.0042s` |
| **Nhóm 5: Chi phí & Hiệu năng** | Tổng Input / Output Tokens | `0` / `0` |

## 2. Chi tiết Đánh giá Từng Trường (Chế độ 12 Trường Mở rộng)

| Trường thông tin (Field) | Support | TP | FP | FN | Precision | Recall | Exact-Match F1 | Field CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seller_name` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `seller_address` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `seller_phone` | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `tax_id` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `invoice_number` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `timestamp` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `subtotal` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `discount` | 4 | 4 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `vat_amount` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `total_cost` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `payment_method` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `items` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |

## 3. Chi tiết Đánh giá 4 Trường Chuẩn MC-OCR 2021

| Trường MC-OCR | Support | TP | FP | FN | Precision | Recall | Exact-Match F1 | Field CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SELLER` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `SELLER_ADDRESS` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `TIMESTAMP` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| `TOTAL_COST` | 5 | 5 | 0 | 0 | 1.0000 | 1.0000 | **1.0000** | 0.0000 |

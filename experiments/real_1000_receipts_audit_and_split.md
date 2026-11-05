# Báo Cáo Kiểm Toán Kho Dữ Liệu Thực Tế Hóa Đơn Việt Nam MC-OCR 2021

> **Mục tiêu MT1 trong Đề cương**: Xây dựng kho dữ liệu $\ge 1.000$ ảnh hóa đơn bán lẻ tiếng Việt thực tế (MC-OCR 2021) có nhãn chuẩn 12 trường.  
> **Nguồn Hugging Face Repository**: `banhchungtuongot/hybrid-docai-mcocr-ocr`  
> **Thời gian thực hiện**: 2026-10-25 22:15:06  

---

## 1. Kết Quả Kiểm Toán Thực Tế
- **Tổng số ảnh hóa đơn Việt Nam trong kho (`01_MC_OCR_2021_Chuan_CuocThi/images`)**: **1165** ảnh.
- **Số cặp dữ liệu đầy đủ nhãn chuẩn (`.jpg` + `.json` + `.txt`)**: **1165** mẫu.
- **Nguồn dữ liệu**: Bộ dữ liệu Hóa đơn Bán lẻ Tiếng Việt **MC-OCR 2021** (`banhchungtuongot/hybrid-docai-mcocr-ocr`). Tuyệt đối không sử dụng dữ liệu hóa đơn nước ngoài.

---

## 2. Kết Quả Phân Chia Tập Dữ Liệu (70% Train / 10% Val / 20% Test, Seed=42)
| Tập dữ liệu | Số lượng mẫu | Tỷ lệ (%) | Mục đích nghiên cứu |
| :--- | :---: | :---: | :--- |
| **Train Set** | **816** | 70.0% | Nạp vào Bộ nhớ Liên tục CMS & Huấn luyện QLoRA SFT |
| **Validation Set** | **116** | 10.0% | Tinh chỉnh siêu tham số và kiểm tra Early Stopping |
| **Test Set (Held-out)** | **233** | 20.0% | Đánh giá độc lập chất lượng mô hình (0% rò rỉ tri thức) |
| **TỔNG CỘNG** | **1165** | **100.0%** | Chuẩn hóa 100% Hóa đơn Bán lẻ Việt Nam |

---

## 3. Tích Hợp Khung Lý Thuyết Nested Learning (NeurIPS 2025)
- **Tập Train** được nạp riêng biệt vào `BoNho_CMS_FewShot_TrainOnly.jsonl`.
- **Ma trận Bộ nhớ Liên kết Delta Rule** ($M_t \in \mathbb{R}^{128 \times 128}$) đã hoàn thành cập nhật **816 bước**, đạt chuẩn Frobenius $\|M_t\|_F = 1.5989$.

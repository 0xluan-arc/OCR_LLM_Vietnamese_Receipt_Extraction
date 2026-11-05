# Báo cáo Thực nghiệm Đối chứng Mục tiêu MT2: Nâng cấp OCR & Tiền xử lý Ảnh Hóa đơn Nhiệt

- **Số lượng mẫu kiểm thử thực tế**: `20` ảnh hóa đơn chụp bằng điện thoại trong tập `test` (`splits_70_10_20.json`).
- **Các thuật toán Tiền xử lý Giấy in nhiệt (MT2)**:
  1. **Multi-Scale Gaussian Illumination Compensation**: Ước lượng trường sáng nền $B(x,y)$ bằng bộ lọc giãn nở + `cv2.GaussianBlur(51x51)` và chia chuẩn hóa `cv2.divide(gray, bg, scale=255)` kết hợp phương trình khử gián đoạn biên $I_{\text{out}} = 255 - W_{\text{sharp}} \cdot (255 - D_{\text{white}})$.
  2. **Adaptive CLAHE (`clipLimit=3.0, tileGridSize=(8,8)`)**: Tăng cường độ tương phản cục bộ cho mực in nhiệt bị phai mờ.
  3. **Probabilistic Hough Line Deskew (`cv2.HoughLinesP`)**: Ước lượng góc nghiêng trung vị có trọng số trên các dòng văn bản ngang.

## 1. Bảng Tổng hợp Kết quả Thực nghiệm (20 Ảnh Hóa đơn Thực tế)

| Cấu hình Pipeline OCR | CER Trung bình (%) | WER Trung bình (%) | Độ trễ (giây/ảnh) | Mức giảm lỗi CER so với Raw (%) | Đạt chuẩn MT2 ($\ge 30\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Tesseract 5 (Raw - Không tiền xử lý)** | `93.99%` | `154.64%` | `1.300s` | Baseline (`0.00%`) | — |
| **2. Tesseract 5 (+ Tiền xử lý tối ưu)** | `50.91%` | `86.82%` | `0.732s` | **`-45.84%`** | ✅ ĐẠT |
| **3. PaddleOCR + VietOCR (+ Tiền xử lý tối ưu)** | **`38.22%`** | **`52.06%`** | `1.541s` | **`-59.34%`** | ✅ **ĐẠT VƯỢT MỨC** |

## 2. Kết quả Chi tiết trên Từng Hóa đơn Thực tế

| STT | Mã Hóa đơn | C1: Tesseract Raw CER (%) | C2: Tesseract + Prep CER (%) | C3: PaddleVietOCR + Prep CER (%) | Mức giảm CER (C1 $\rightarrow$ C3) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `hoadon_thucte_007` | `0.31%` | `0.31%` | **`0.31%`** | `+0.0%` |
| 2 | `real_receipt_0013` | `77.14%` | `128.57%` | **`21.71%`** | `+71.9%` |
| 3 | `real_receipt_0014` | `41.44%` | `33.15%` | **`6.63%`** | `+84.0%` |
| 4 | `real_receipt_0016` | `20.00%` | `28.11%` | **`10.27%`** | `+48.7%` |
| 5 | `real_receipt_0018` | `65.94%` | `17.39%` | **`15.22%`** | `+76.9%` |
| 6 | `real_receipt_0019` | `91.40%` | `22.58%` | **`20.43%`** | `+77.6%` |
| 7 | `real_receipt_0021` | `22.73%` | `15.00%` | **`2.27%`** | `+90.0%` |
| 8 | `real_receipt_0023` | `68.29%` | `90.24%` | **`30.49%`** | `+55.4%` |
| 9 | `real_receipt_0026` | `104.00%` | `16.00%` | **`10.00%`** | `+90.4%` |
| 10 | `real_receipt_0027` | `45.00%` | `9.17%` | **`0.00%`** | `+100.0%` |
| 11 | `real_receipt_0031` | `173.33%` | `86.67%` | **`100.00%`** | `+42.3%` |
| 12 | `real_receipt_0038` | `121.05%` | `97.37%` | **`100.00%`** | `+17.4%` |
| 13 | `real_receipt_0040` | `497.56%` | `36.59%` | **`36.59%`** | `+92.6%` |
| 14 | `real_receipt_0042` | `91.76%` | `87.06%` | **`97.65%`** | `-6.4%` |
| 15 | `real_receipt_0043` | `58.88%` | `64.49%` | **`42.99%`** | `+27.0%` |
| 16 | `real_receipt_0054` | `121.38%` | `80.69%` | **`84.83%`** | `+30.1%` |
| 17 | `real_receipt_0070` | `65.48%` | `40.48%` | **`27.38%`** | `+58.2%` |
| 18 | `real_receipt_0074` | `100.00%` | `12.62%` | **`23.30%`** | `+76.7%` |
| 19 | `real_receipt_0078` | `30.99%` | `53.99%` | **`34.27%`** | `-10.6%` |
| 20 | `real_receipt_0080` | `83.08%` | `97.69%` | **`100.00%`** | `-20.4%` |

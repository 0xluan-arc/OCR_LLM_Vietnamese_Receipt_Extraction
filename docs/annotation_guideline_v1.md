# HƯỚNG DẪN GÁN NHÃN DỮ LIỆU HÓA ĐƠN BÁN LẺ VIỆT NAM (ANNOTATION GUIDELINE v1.0)

> **Dự án**: Nghiên cứu kết hợp OCR và Mô hình Ngôn ngữ Lớn (LLM) trích xuất thông tin có cấu trúc từ ảnh hóa đơn bán lẻ Việt Nam  
> **Phiên bản**: `v1.0`  
> **Schema tương thích**: `ReceiptData` (12 trường mở rộng) & `MC-OCR 2021` (4 trường chuẩn)  
> **Định hướng lý thuyết**: *Nested Learning: The Illusion of Deep Learning Architecture* (Behrouz et al., NeurIPS 2025)

---

## 1. Mục tiêu & Phạm vi

Tài liệu này quy định quy chuẩn thống nhất cho quá trình gán nhãn Ground-Truth trên:
1. **Tập dữ liệu công khai MC-OCR 2021** (bổ sung từ 4 trường gốc lên 12 trường chi tiết cho tập con mở rộng).
2. **Tập dữ liệu tự thu thập thực tế** (300–500 hóa đơn từ siêu thị, cửa hàng tiện lợi, quán cà phê, nhà thuốc tại Việt Nam).

Mục tiêu cốt lõi là đảm bảo **độ đồng thuận giữa những người gán nhãn (Inter-Annotator Agreement – IAA) đạt $\ge 90\%$** trước khi đưa dữ liệu vào huấn luyện LoRA ($f_{\text{low}}$), chọn mẫu Few-shot ($f=\infty$) và đánh giá độc lập.

---

## 2. Đặc tả Chi tiết 12 Trường Thông tin (`ReceiptData`)

| STT | Tên trường (`ReceiptData`) | Ánh xạ MC-OCR 2021 | Kiểu dữ liệu | Bắt buộc? | Quy tắc gán nhãn & Ví dụ chuẩn |
| :-: | :--- | :---: | :---: | :---: | :--- |
| 1 | `seller_name` | `SELLER` | `str` | Có | Tên thương hiệu / đơn vị bán hàng in ở phần đầu hóa đơn. Giữ nguyên chữ hoa/thường và dấu tiếng Việt theo đúng hóa đơn. **Ví dụ**: `"SIÊU THỊ WINMART+ QUẬN 7"` |
| 2 | `seller_address` | `SELLER_ADDRESS` | `str` | Có | Địa chỉ chi nhánh/cửa hàng (không lấy tiền tố `"Địa chỉ:"`, `"Đ/C:"`). Nếu địa chỉ trải trên nhiều dòng, ghép nối bằng 1 dấu cách. **Ví dụ**: `"458 Nguyễn Thị Thập, Tân Quy, Q7, TP.HCM"` |
| 3 | `seller_phone` | *(Mở rộng)* | `str` | Tùy chọn | Số điện thoại hoặc Hotline của cửa hàng (bỏ tiền tố `"SĐT:"`, `"Tel:"`, `"Hotline:"`). Giữ nguyên dãy số hoặc khoảng cách chuẩn. **Ví dụ**: `"028 3822 8899"`, `"1800 6928"` |
| 4 | `tax_id` | *(Mở rộng)* | `str` | Tùy chọn | Mã số thuế doanh nghiệp (10 chữ số) hoặc chi nhánh (13 chữ số có dấu `-`). Không lấy nhãn `"MST:"`. **Ví dụ**: `"0315275368-002"` |
| 5 | `invoice_number` | *(Mở rộng)* | `str` | Tùy chọn | Mã số hóa đơn / Số phiếu thanh toán / Số biên lai. Không lấy nhãn `"Số HĐ:"`. **Ví dụ**: `"WM-20261017-4410"` |
| 6 | `timestamp` | `TIMESTAMP` | `str` | Có | Ngày giờ phát hành hóa đơn. Chuẩn hóa về định dạng `DD/MM/YYYY HH:MM:SS` hoặc `DD/MM/YYYY HH:MM` (hoặc `DD/MM/YYYY` nếu hóa đơn không in giờ). **Ví dụ**: `"17/10/2026 18:15:00"` |
| 7 | `items` | *(Mở rộng)* | `List[ReceiptItem]` | Tùy chọn | Danh sách từng mặt hàng mua, mỗi phần tử gồm 4 thuộc tính: `item_name` (`str`), `quantity` (`float`), `unit_price` (`int`), `total_price` (`int`). |
| 8 | `subtotal` | *(Mở rộng)* | `int` | Tùy chọn | Tổng tiền hàng tạm tính trước giảm giá và thuế VAT (số nguyên VND dương). **Ví dụ**: `264500` |
| 9 | `discount` | *(Mở rộng)* | `int` | Tùy chọn | Tổng tiền chiết khấu / giảm giá / khuyến mãi. **Quy ước**: Luôn lưu **số nguyên dương** (ví dụ hóa đơn in `-14.500` thì gán `14500`). Nếu không có giảm giá hoặc ghi `0`, gán `null` (hoặc `0` nếu in rõ dòng `Giảm giá: 0`). |
| 10 | `vat_amount` | *(Mở rộng)* | `int` | Tùy chọn | Tiền thuế giá trị gia tăng (VAT) tính bằng số nguyên VND dương. **Ví dụ**: `20000` |
| 11 | `total_cost` | `TOTAL_COST` | `int` | **Bắt buộc** | Tổng số tiền thanh toán cuối cùng khách phải trả (số nguyên VND dương). Không nhầm với `"Tiền khách đưa"` hay `"Tiền thừa"`. **Ví dụ**: `270000` |
| 12 | `payment_method` | *(Mở rộng)* | `str` | Tùy chọn | Hình thức thanh toán ghi trên hóa đơn. **Ví dụ**: `"Tiền mặt"`, `"Chuyển khoản"`, `"MoMo"`, `"Thẻ Visa"` |

---

## 3. Quy ước Chuẩn hóa Dữ liệu (Normalization Rules)

### 3.1. Chuẩn hóa Tiền tệ (Currency Normalization)
- Mọi trường tiền tệ (`subtotal`, `discount`, `vat_amount`, `total_cost`, `unit_price`, `total_price`) trong file JSON Ground-Truth phải được lưu dưới dạng **số nguyên (`int`) tính bằng đồng (VND)**.
- Loại bỏ hoàn toàn dấu phân cách hàng nghìn (`.`, `,`), khoảng trắng và ký hiệu tiền tệ (`đ`, `Đ`, `VND`, `vnđ`).
- Nếu hóa đơn sử dụng hậu tố viết tắt nghìn đồng (`k` hoặc `K`, ví dụ `25k`), quy đổi thành số nguyên đầy đủ (`25000`).
- **Kiểm tra ràng buộc toán học khi gán nhãn**:
  - Với từng dòng mặt hàng: $\text{quantity} \times \text{unit\_price} \approx \text{total\_price}$.
  - Với toàn hóa đơn: $\text{subtotal} - \text{discount} + \text{vat\_amount} \approx \text{total\_cost}$ (dung sai làm tròn $\le 1.000\text{ VND}$).
  - Nếu bản thân tờ hóa đơn gốc in sai toán học hoặc bị mờ mất một phần, **gán nhãn trung thực theo đúng chữ in trên ảnh** (không tự ý bịa số), đồng thời ghi chú vào trường `notes` để phục vụ thực nghiệm kiểm định LSS.

### 3.2. Chuẩn hóa Ngày tháng & Unicode
- Toàn bộ chuỗi tiếng Việt phải được lưu ở bảng mã **UTF-8, chuẩn hóa Unicode dựng sẵn (NFC)**.
- Trường `timestamp` chuẩn hóa dấu phân cách ngày-tháng-năm về dấu gạch chéo `/` (`DD/MM/YYYY`), ngày và tháng luôn đủ 2 chữ số (ví dụ `05/09/2026`), năm đủ 4 chữ số (`2026`).

### 3.3. Quy tắc Gán nhãn Danh sách Mặt hàng (`items`)
- Nếu một mặt hàng bị ngắt xuống 2 dòng trên hóa đơn nhiệt (dòng 1 là tên hàng, dòng 2 là `SL x Đơn giá = Thành tiền`), phải gộp lại thành **1 đối tượng `ReceiptItem` duy nhất**.
- Không đưa các dòng tiêu đề (`TÊN HÀNG`, `SL`, `ĐƠN GIÁ`), dòng khuyến mãi tổng hoặc dòng thuế VAT vào danh sách `items`.

---

## 4. Quy tắc Bảo mật & Ẩn danh Thông tin Cá nhân (PII Anonymization)

Để tuân thủ đạo đức nghiên cứu khoa học và bảo vệ dữ liệu cá nhân khi công bố tập dữ liệu:
1. **Thông tin của Đơn vị bán hàng (Giữ nguyên)**: Tên cửa hàng (`seller_name`), địa chỉ kinh doanh (`seller_address`), số điện thoại/hotline công khai của cửa hàng (`seller_phone`), mã số thuế doanh nghiệp (`tax_id`).
2. **Thông tin Cá nhân của Khách hàng / Thu ngân (Bắt buộc ẩn danh)**:
   - **Trên ảnh hóa đơn tự chụp**: Làm mờ (Gaussian Blur) vùng in tên khách hàng thành viên, số điện thoại cá nhân của khách hàng, mã QR thanh toán cá nhân hoặc 12 số đầu của thẻ ngân hàng trước khi đưa vào kho dữ liệu.
   - **Trong văn bản OCR chuẩn (`gt_ocr_text`)**: Thay thế số điện thoại khách hàng bằng `[PII_PHONE]` và số thẻ ngân hàng bằng `****-****-****-XXXX`.

---

## 5. Quy trình Kiểm soát Chất lượng & Đánh giá Độ đồng thuận (IAA $\ge 90\%$)

1. **Giai đoạn Huấn luyện Thử nghiệm (Pilot Annotation)**:
   - Hai thành viên nghiên cứu cùng gán nhãn độc lập trên 20 hóa đơn đầu tiên.
   - So sánh đối chiếu và thống nhất các trường hợp biên (edge cases).
2. **Gán nhãn Chéo 10% (Cross-Annotation Protocol)**:
   - Trích ngẫu nhiên **10% tổng số mẫu** (với `seed=42`) để cả 2 người gán nhãn độc lập (Double-Blind Annotation).
   - Sử dụng `scripts/evaluate.py` (chế độ so sánh Annotator A vs Annotator B) để tính **Exact-Match Micro-F1** và hệ số **Cohen's Kappa ($\kappa$)**.
3. **Tiêu chí Nghiệm thu Bộ nhãn**:
   - Độ đồng thuận toàn cục (Overall IAA Exact-Match F1) bắt buộc đạt **$\ge 90\%$** (và **$\ge 98\%$** riêng cho trường `total_cost`).
   - Nếu IAA $< 90\%$, toàn bộ các mẫu bất đồng phải được phân xử (Adjudication) bởi thành viên thứ 3 và cập nhật lại hướng dẫn trước khi gán nhãn phần còn lại.

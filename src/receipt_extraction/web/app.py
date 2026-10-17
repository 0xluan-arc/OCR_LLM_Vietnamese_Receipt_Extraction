"""Ứng dụng Web Streamlit: Hệ thống Trích xuất Thông tin Hóa đơn Bán lẻ Việt Nam (MVP)."""

import json
import time
from typing import Optional
from pathlib import Path
import streamlit as st

from receipt_extraction.config import config
from receipt_extraction.preprocess.image_validator import validate_image
from receipt_extraction.preprocess.preprocessor import preprocess_receipt_image, PreprocessOptions
from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.extraction.llm_extractor import LLMExtractor, LLMNotConfiguredError
from receipt_extraction.validation.receipt_validator import validate_receipt_data
from receipt_extraction.schemas.receipt import (
    ExtractionResult,
    ExtractionMetadata,
    ValidationSeverity
)


def render_ui():
    st.set_page_config(
        page_title="Vietnamese Receipt KIE Demo",
        page_icon="🧾",
        layout="wide"
    )

    st.title("🧾 Trích xuất Thông tin Hóa đơn Bán lẻ Việt Nam")
    st.caption("Đề tài NCKH Sinh viên: Nghiên cứu ứng dụng OCR kết hợp Mô hình Ngôn ngữ Lớn (Document AI)")

    # =========================================================================
    # SIDEBAR: CẤU HÌNH HỆ THỐNG & TIỀN XỬ LÝ
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Cấu hình Hệ thống")
        
        # 1. Chọn OCR Engine
        ocr_choice = st.selectbox(
            "Mô-đun OCR",
            options=["tesseract", "dummy", "vietocr", "paddleocr"],
            index=0,
            help="'tesseract': OCR thật nhận dạng văn bản tiếng Việt.\n'dummy': Chế độ giả lập phục vụ kiểm thử."
        )

        # Hiển thị trạng thái engine
        engine_obj = get_ocr_engine(ocr_choice)
        is_avail, avail_msg = engine_obj.is_available()
        if is_avail:
            st.success(f"Trạng thái: 🟢 Khả dụng ({ocr_choice})")
        else:
            st.warning(f"Trạng thái: 🔴 Chưa sẵn sàng")

        # 2. Chọn phương pháp trích xuất KIE
        st.subheader("Trích xuất KIE")
        extraction_method = st.radio(
            "Phương pháp bóc tách",
            options=["Baseline B1 (Regex/Luật)", "LLM Trích xuất (Zero/Few-shot)"],
            index=0
        )

        st.divider()

        # 3. Tùy chọn Tiền xử lý ảnh (OpenCV)
        st.subheader("🛠️ Tiền xử lý ảnh (OpenCV)")
        apply_preprocess = st.checkbox("Bật tiền xử lý ảnh", value=True)
        
        enhance_contrast = False
        to_grayscale = False
        adaptive_threshold = False
        deskew = False

        if apply_preprocess:
            enhance_contrast = st.checkbox("Tăng tương phản (CLAHE)", value=True, help="Tối ưu cho chữ in nhiệt bị mờ")
            to_grayscale = st.checkbox("Chuyển ảnh xám (Grayscale)", value=False)
            adaptive_threshold = st.checkbox("Phân ngưỡng nhị phân (Threshold)", value=False, help="Chuyển ảnh thành đen/trắng")
            deskew = st.checkbox("Khử nghiêng tự động (Deskew)", value=False, help="Chỉnh góc xoay của hóa đơn")

        st.divider()
        st.markdown("**Thông số phần cứng máy trạm:**")
        st.caption("GPU: NVIDIA RTX 3050 Laptop (6 GB VRAM, CUDA 13.1)")
        st.caption("OS: Windows 11 64-bit | Python 3.10.11")

    # =========================================================================
    # KHU VỰC CHÍNH: TẢI ẢNH & HIỂN THỊ KẾT QUẢ
    # =========================================================================
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1. Tải ảnh Hóa đơn")
        uploaded_file = st.file_uploader(
            "Chọn ảnh hóa đơn bán lẻ (.jpg, .png, .jpeg)",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            
            # Bước A: Kiểm định tính hợp lệ của ảnh đầu vào
            val_res = validate_image(file_bytes, file_name=uploaded_file.name)
            
            if not val_res.is_valid:
                st.error(f"❌ Tệp ảnh không hợp lệ: {val_res.error_message}")
                return

            st.image(file_bytes, caption=f"Ảnh gốc: {uploaded_file.name} ({val_res.width}x{val_res.height}px)", use_container_width=True)
            
            if val_res.is_blurry:
                st.warning(f"⚠️ Cảnh báo: Ảnh có dấu hiệu mờ (Độ nét: {val_res.blur_score:.1f} < {config.image.blur_threshold}).")

            # Xử lý ảnh nếu người dùng chọn tiền xử lý
            processed_bytes = file_bytes
            preprocess_meta = {}
            if apply_preprocess:
                opts = PreprocessOptions(
                    enhance_contrast=enhance_contrast,
                    to_grayscale=to_grayscale,
                    adaptive_threshold=adaptive_threshold,
                    deskew=deskew
                )
                proc_img, p_meta, p_err = preprocess_receipt_image(file_bytes, opts)
                if proc_img is not None:
                    preprocess_meta = p_meta
                    import cv2
                    is_success, buffer = cv2.imencode(".jpg", proc_img)
                    if is_success:
                        processed_bytes = buffer.tobytes()
                    with st.expander("👁️ Xem ảnh sau tiền xử lý OpenCV"):
                        st.image(processed_bytes, caption=f"Sau tiền xử lý ({', '.join(p_meta.get('applied_transforms', ['gốc']))})", use_container_width=True)

            process_btn = st.button("🚀 Thực hiện Nhận dạng & Trích xuất", type="primary")
        else:
            st.info("👆 Vui lòng tải lên một ảnh hóa đơn để bắt đầu kiểm nghiệm pipeline.")
            process_btn = False

    with col_right:
        st.subheader("2. Kết quả Trích xuất & Kiểm định")

        if uploaded_file and process_btn:
            # Kiểm tra tính sẵn sàng của engine
            ocr_engine = get_ocr_engine(ocr_choice)
            is_ready, not_ready_reason = ocr_engine.is_available()
            
            if not is_ready:
                st.error(f"❌ Không thể thực thi OCR bằng '{ocr_choice}':")
                st.warning(not_ready_reason)
                st.info("💡 Bạn có thể chọn engine 'dummy' ở thanh menu bên trái để chạy kiểm thử luồng ứng dụng ngoại tuyến.")
                return

            start_pipeline_time = time.time()
            progress_bar = st.progress(0, text="Bắt đầu Pipeline...")

            # 1. Chạy OCR
            progress_bar.progress(30, text=f"Đang nhận dạng văn bản với engine '{ocr_choice}'...")
            ocr_res = ocr_engine.recognize(processed_bytes)
            
            if ocr_res.error_message:
                st.error(f"Lỗi trong quá trình OCR: {ocr_res.error_message}")
                return

            # Cảnh báo minh bạch nếu đang dùng Dummy
            if ocr_choice == "dummy":
                st.warning("⚠️ **CHẾ ĐỘ MÔ PHỎNG (DUMMY OCR):** Văn bản dưới đây là dữ liệu giả lập phục vụ kiểm thử đơn vị, KHÔNG PHẢI kết quả đọc từ ảnh bạn vừa tải lên.")
            else:
                st.success(f"🟢 **OCR THẬT THÀNH CÔNG:** Nhận dạng bằng '{ocr_choice}' ({ocr_res.processing_time_sec:.3f}s)")

            # 2. Thực hiện trích xuất KIE
            progress_bar.progress(70, text=f"Đang trích xuất dữ liệu bằng '{extraction_method}'...")
            
            evidence = {}
            if "LLM" in extraction_method:
                llm_ext = LLMExtractor()
                configured, cfg_err = llm_ext.is_configured()
                if not configured:
                    st.error(f"❌ {cfg_err}")
                    st.info("💡 Hệ thống tự động chuyển về chạy **Baseline B1 (Regex)** để hoàn tất quy trình.")
                    extractor = RegexBaselineExtractor()
                    extracted_data, evidence = extractor.extract(ocr_res)
                    used_method = "regex_baseline (fallback do LLM chưa cấu hình)"
                else:
                    try:
                        extracted_data, evidence = llm_ext.extract(ocr_res)
                        used_method = "llm_extractor"
                    except Exception as e:
                        st.error(f"Lỗi thực thi LLM: {str(e)}")
                        return
            else:
                extractor = RegexBaselineExtractor()
                extracted_data, evidence = extractor.extract(ocr_res)
                used_method = "regex_baseline"

            # 3. Chạy Validator nghiệp vụ & kiểm tra Hallucination
            progress_bar.progress(90, text="Đang chạy kiểm định nghiệp vụ & đối sánh bằng chứng...")
            validation_issues, is_valid = validate_receipt_data(extracted_data, raw_ocr_text=ocr_res.raw_text)

            total_elapsed = time.time() - start_pipeline_time
            progress_bar.progress(100, text="Hoàn tất!")
            time.sleep(0.2)
            progress_bar.empty()

            # Đóng gói kết quả
            final_result = ExtractionResult(
                success=True,
                data=extracted_data,
                metadata=ExtractionMetadata(
                    method=used_method,
                    ocr_engine=ocr_res.engine_name,
                    raw_ocr_text=ocr_res.raw_text,
                    processing_time_sec=round(total_elapsed, 4),
                    ocr_character_count=len(ocr_res.raw_text),
                    field_evidence=evidence
                ),
                validation_issues=validation_issues,
                is_valid=is_valid
            )

            # Hiển thị trạng thái kiểm định
            if is_valid:
                st.success(f"✅ Hóa đơn hợp lệ theo ràng buộc nghiệp vụ (Thời gian tổng: {total_elapsed:.2f}s)")
            else:
                st.error(f"❌ Phát hiện lỗi kiểm định nghiêm ngặt (Thời gian tổng: {total_elapsed:.2f}s)")

            # Tab phân tách các góc nhìn dữ liệu
            tab_summary, tab_json, tab_ocr, tab_validation = st.tabs([
                "📊 Tóm tắt Trích xuất",
                "📄 JSON Output",
                "🔍 Văn bản OCR Thô",
                "🛡️ Báo cáo Kiểm định"
            ])

            with tab_summary:
                st.markdown(f"**Cửa hàng:** {extracted_data.seller_name or '*(Chưa nhận diện)*'}")
                st.markdown(f"**Địa chỉ:** {extracted_data.seller_address or '*(Chưa nhận diện)*'}")
                st.markdown(f"**Mã số thuế:** `{extracted_data.tax_id or 'Chưa có'}`")
                st.markdown(f"**Số hóa đơn:** `{extracted_data.invoice_number or 'Chưa có'}`")
                st.markdown(f"**Ngày mua:** `{extracted_data.timestamp or 'Chưa có'}`")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tạm tính", f"{extracted_data.subtotal:,} đ" if extracted_data.subtotal else "N/A")
                col2.metric("Giảm giá", f"{extracted_data.discount:,} đ" if extracted_data.discount else "0 đ")
                col3.metric("VAT", f"{extracted_data.vat_amount:,} đ" if extracted_data.vat_amount else "N/A")
                col4.metric("TỔNG TIỀN", f"{extracted_data.total_cost:,} đ" if extracted_data.total_cost else "N/A")

                if extracted_data.items:
                    st.write("**Danh sách mặt hàng trích xuất:**")
                    items_table = [
                        {
                            "Tên hàng": it.item_name,
                            "Số lượng": it.quantity,
                            "Đơn giá (đ)": f"{it.unit_price:,}" if it.unit_price else "-",
                            "Thành tiền (đ)": f"{it.total_price:,}" if it.total_price else "-"
                        }
                        for it in extracted_data.items
                    ]
                    st.dataframe(items_table, use_container_width=True)
                else:
                    st.info("Chưa bóc tách được danh sách món hàng chi tiết (chức năng sẽ được hoàn thiện cùng LLM).")

            with tab_json:
                json_str = final_result.model_dump_json(indent=2)
                st.json(json_str)
                st.download_button(
                    label="💾 Tải kết quả JSON",
                    data=json_str,
                    file_name=f"receipt_{int(time.time())}.json",
                    mime="application/json"
                )

            with tab_ocr:
                st.caption(f"Engine: {ocr_res.engine_name} | Số ký tự: {len(ocr_res.raw_text)} | Số dòng: {len(ocr_res.lines)} | Thời gian OCR: {ocr_res.processing_time_sec:.3f}s")
                st.text_area("Toàn bộ văn bản do OCR phát hiện:", value=ocr_res.raw_text, height=300)

            with tab_validation:
                if not validation_issues:
                    st.info("Không có cảnh báo nghiệp vụ nào. Dữ liệu thỏa mãn toàn bộ ràng buộc.")
                else:
                    for issue in validation_issues:
                        if issue.severity == ValidationSeverity.ERROR:
                            st.error(f"🔴 [{issue.code}] {issue.message}")
                        elif issue.severity == ValidationSeverity.WARNING:
                            st.warning(f"🟡 [{issue.code}] {issue.message}")
                        else:
                            st.info(f"🔵 [{issue.code}] {issue.message}")


if __name__ == "__main__":
    render_ui()

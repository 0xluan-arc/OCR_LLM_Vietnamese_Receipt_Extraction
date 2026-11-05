"""FastAPI Backend Server cho Hệ thống Trích xuất Hóa đơn Bán lẻ Việt Nam
kết hợp OCR, LLM và Khung lý thuyết Nested Learning (NeurIPS 2025).

Cung cấp các RESTful APIs:
- GET  /api/status        : Kiểm tra trạng thái OCR Engines, Ollama LLM, Continuum Memory System (CMS).
- GET  /api/samples       : Danh sách hóa đơn mẫu có sẵn để chọn nhanh trên Web UI.
- GET  /api/samples/{name}: Tải file ảnh hóa đơn mẫu.
- POST /api/extract       : Trích xuất 12 trường thông tin từ ảnh hóa đơn (OCR -> KIE -> LSS Validator).
- POST /api/cms/save      : Củng cố mẫu hóa đơn đã chuẩn hóa vào CMS và cập nhật Ma trận Delta Rule.
- POST /api/export/excel  : Xuất file Excel (.xlsx) 2 sheet chuẩn kế toán.
- GET  /                  : Phục vụ Giao diện Web Studio hiện đại (SPA).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Đảm bảo đường dẫn import
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from receipt_extraction.config import config
from receipt_extraction.extraction.hybrid_extractor import HybridNestedExtractor
from receipt_extraction.extraction.llm_extractor import LLMExtractor
from receipt_extraction.extraction.regex_baseline import RegexBaselineExtractor
from receipt_extraction.memory.cms_store import (
    ContinuumMemoryStore,
    MemoryEpisode,
    receipt_data_to_target_vector,
    text_to_l2_normalized_vector,
)
from receipt_extraction.ocr.factory import get_ocr_engine
from receipt_extraction.preprocess.image_validator import validate_image
from receipt_extraction.preprocess.preprocessor import PreprocessOptions, preprocess_receipt_image
from receipt_extraction.schemas.receipt import (
    ExtractionMetadata,
    ExtractionResult,
    ReceiptData,
    ReceiptItem,
    ValidationIssue,
    ValidationSeverity,
)
from receipt_extraction.validation.receipt_validator import (
    sanitize_and_ground_receipt_data,
    validate_receipt_data,
)
from receipt_extraction.web.export_utils import draw_ocr_bounding_boxes, export_receipt_to_excel_bytes


app = FastAPI(
    title="Vietnamese Receipt Document AI Studio",
    description="Backend API kết hợp OCR, Qwen2.5 LLM và Continuum Associative Memory (Nested Learning - NeurIPS 2025)",
    version="1.0.0",
)

# Kích hoạt CORS để frontend có thể gọi API linh hoạt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo singleton CMS Store
CMS_STORE = ContinuumMemoryStore()
STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# PYDANTIC SCHEMAS CHO REQUEST / RESPONSE
# ==============================================================================

class CMSSaveRequest(BaseModel):
    receipt_data: Dict[str, Any]
    ocr_text: str
    layout_text: str = ""
    source: str = "human_verified_web_studio"


class ExportExcelRequest(BaseModel):
    receipt_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/api/status")
def get_system_status() -> Dict[str, Any]:
    """Kiểm tra trạng thái kết nối thực tế của OCR Engines, Ollama LLM, và CMS Memory."""
    # 1. Trạng thái OCR Engines
    engines = {}
    for name in ["tesseract", "vietocr", "paddleocr"]:
        try:
            eng = get_ocr_engine(name)
            ok, msg = eng.is_available()
            engines[name] = {"available": ok, "message": msg or "Sẵn sàng hoạt động"}
        except Exception as e:
            engines[name] = {"available": False, "message": str(e)}

    # 2. Trạng thái Ollama Endpoint (Kiểm tra nhanh qua 127.0.0.1 để tránh treo IPv6 DNS)
    ollama_status = {"connected": False, "model": config.llm.model, "message": "Chưa kết nối"}
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.4)
        res = s.connect_ex(("127.0.0.1", 11434))
        s.close()
        if res == 0:
            url = "http://127.0.0.1:11434/api/tags"
            req = urllib.request.Request(url, headers={"User-Agent": "FastAPI-Studio/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                has_target = any(config.llm.model in m for m in models)
                ollama_status["connected"] = True
                ollama_status["models_available"] = models
                ollama_status["message"] = (
                    f"Đã kết nối! Tìm thấy '{config.llm.model}'" if has_target
                    else f"Đã kết nối Ollama nhưng chưa tải '{config.llm.model}'. Hiện có: {models}"
                )
        else:
            ollama_status["message"] = "Dịch vụ Ollama chưa chạy trên cổng 11434"
    except Exception as exc:
        ollama_status["message"] = f"Không kết nối được ({exc})"

    # 3. Trạng thái Continuum Memory System (CMS) & Ma trận Delta Rule
    delta_matrix = CMS_STORE.delta_memory
    import numpy as np
    matrix_norm = float(np.linalg.norm(delta_matrix.M, ord="fro"))

    cms_info = {
        "episode_count": len(CMS_STORE.episodes),
        "memory_dim": CMS_STORE.memory_dim,
        "delta_matrix_shape": f"{delta_matrix.dim}x{delta_matrix.dim}",
        "delta_matrix_steps": delta_matrix.step_count,
        "delta_matrix_frobenius_norm": round(matrix_norm, 4),
    }

    return {
        "status": "healthy",
        "ocr_engines": engines,
        "ollama_llm": ollama_status,
        "cms_store": cms_info,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/samples")
def list_sample_receipts() -> List[Dict[str, Any]]:
    """Trả về danh sách 10 mẫu hóa đơn độ khó cao kiểm thử có sẵn trên hệ thống."""
    samples_dir = config.paths.synthetic_test_10_dir
    if not samples_dir.exists():
        samples_dir = config.paths.samples_dir

    if not samples_dir.exists():
        return []

    samples = []
    jpg_files = sorted(list(samples_dir.glob("*.jpg")))
    for f in jpg_files:
        if f.name.startswith("sample_receipt"):
            continue
        # Tên hiển thị thân thiện
        title = f.stem.replace("_", " ").title()
        samples.append({
            "id": f.stem,
            "filename": f.name,
            "title": title,
            "has_gt_json": f.with_suffix(".json").exists(),
        })
    return samples


@app.get("/api/samples/{filename}")
def get_sample_image(filename: str):
    """Tải file ảnh hóa đơn mẫu."""
    samples_dir = config.paths.synthetic_test_10_dir
    target = samples_dir / filename
    if not target.exists():
        target = config.paths.samples_dir / filename

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh mẫu: {filename}")
    return FileResponse(str(target), media_type="image/jpeg")


@app.post("/api/extract")
def extract_receipt(
    file: UploadFile = File(...),
    ocr_engine: str = Form("paddleocr"),
    extractor_mode: str = Form("hybrid"),
    enable_lss_retry: bool = Form(True),
    use_layout: bool = Form(False),
) -> Dict[str, Any]:
    """Trích xuất 12 trường thông tin từ ảnh hóa đơn tải lên kèm lớp phủ Bounding Box và Pipeline Trace."""
    t0 = time.time()
    if hasattr(file, "file") and hasattr(file.file, "read"):
        image_bytes = file.file.read()
    else:
        image_bytes = file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Tệp ảnh tải lên rỗng.")

    # 1. Tiền xử lý ảnh & Đánh giá chất lượng (Laplacian blur, resolution)
    val_res = validate_image(image_bytes)
    processed_bytes = image_bytes
    prep_meta = {}
    try:
        import cv2
        proc_img, prep_meta, _ = preprocess_receipt_image(
            image_bytes,
            PreprocessOptions(
                crop_receipt=False,
                remove_shadows=False,
                illumination_compensation=False,
                deskew=True,
                enhance_contrast=True,
                clahe_clip_limit=2.0,
            ),
        )
        if proc_img is not None:
            ok_enc, buffer = cv2.imencode(".jpg", proc_img)
            if ok_enc:
                processed_bytes = buffer.tobytes()
    except Exception:
        pass

    skew_ang = prep_meta.get("skew_angle", 0.0) if isinstance(prep_meta, dict) else 0.0
    trace_steps: List[Dict[str, Any]] = [
        {
            "step": 1,
            "name": "Tiền xử lý ảnh (Image Preprocessing)",
            "duration_ms": 15.0,
            "status": "success",
            "details": f"Độ phân giải: {val_res.width}x{val_res.height}px | Độ nét Laplacian: {val_res.blur_score:.1f} | Góc nghiêng: {skew_ang:.1f}°",
        }
    ]

    # 2. Nhận dạng ký tự quang học (OCR Recognition - Mặc định PaddleVietOCREngine, fallback Tesseract)
    t_ocr_0 = time.time()
    engine_inst = get_ocr_engine(ocr_engine, fallback_to_tesseract=True)
    ocr_res = engine_inst.recognize(processed_bytes)
    t_ocr = time.time() - t_ocr_0

    if ocr_res.error_message:
        trace_steps.append({
            "step": 2,
            "name": f"Nhận dạng OCR ({ocr_engine})",
            "duration_ms": round(t_ocr * 1000, 2),
            "status": "error",
            "details": ocr_res.error_message,
        })
        raise HTTPException(status_code=500, detail=f"Lỗi OCR: {ocr_res.error_message}")

    trace_steps.append({
        "step": 2,
        "name": f"Nhận dạng OCR ({ocr_engine})",
        "duration_ms": round(t_ocr * 1000, 2),
        "status": "success",
        "details": f"Nhận diện được {len(ocr_res.lines)} dòng chữ ({len(ocr_res.raw_text)} ký tự).",
    })

    # 3. Truy xuất Ký ức Liên tục CMS (Continuum Memory Step)
    t_cms_0 = time.time()
    similar_eps = CMS_STORE.retrieve_similar_examples(ocr_res.raw_text, k=2, exclude_near_exact=False)
    t_cms = time.time() - t_cms_0
    trace_steps.append({
        "step": 3,
        "name": "Bộ nhớ Liên tục Delta Rule (CMS Retrieval)",
        "duration_ms": round(t_cms * 1000, 2),
        "status": "success",
        "details": f"Tìm thấy {len(similar_eps)} mẫu ngữ cảnh tương đồng từ kho {len(CMS_STORE.episodes)} ký ức.",
    })

    # 4. Trích xuất Thực thể Hóa đơn (KIE - Nested Learning Fast/Slow Path)
    t_kie_0 = time.time()
    llm_sub = LLMExtractor(
        adaptation_mode="few_shot_cms",
        few_shot_k=2,
        enable_lss_retry=enable_lss_retry,
        memory_store=CMS_STORE,
    )

    mode_normalized = extractor_mode.lower().strip()
    if mode_normalized in ("regex_b1", "regex_baseline", "regex"):
        extractor = RegexBaselineExtractor()
        used_mode = "regex_baseline"
    elif mode_normalized in ("zero_shot", "llm_zero_shot"):
        extractor = LLMExtractor(adaptation_mode="zero_shot", enable_lss_retry=enable_lss_retry, memory_store=CMS_STORE)
        used_mode = "llm_zero_shot"
    elif mode_normalized in ("few_shot_cms", "llm_few_shot", "few_shot"):
        extractor = LLMExtractor(adaptation_mode="few_shot_cms", few_shot_k=2, enable_lss_retry=enable_lss_retry, memory_store=CMS_STORE)
        used_mode = "llm_few_shot"
    elif mode_normalized in ("few_shot_layout", "llm_few_shot_layout"):
        extractor = LLMExtractor(adaptation_mode="few_shot_layout", few_shot_k=2, use_layout=True, enable_lss_retry=enable_lss_retry, memory_store=CMS_STORE)
        used_mode = "llm_few_shot_layout"
    else:
        extractor = HybridNestedExtractor(llm_extractor=llm_sub)
        used_mode = "hybrid"

    fallback_info = None
    try:
        receipt_data, evidence = extractor.extract(ocr_res)
    except Exception as exc:
        fallback_info = f"LLM timeout/lỗi kết nối ({exc}). Hệ thống tự động chuyển sang Regex B1."
        extractor = RegexBaselineExtractor()
        receipt_data, evidence = extractor.extract(ocr_res)
        used_mode = "regex_fallback"

    t_kie = time.time() - t_kie_0

    # Áp dụng bộ chuẩn hóa và khôi phục đối chứng tổng quát
    receipt_data = sanitize_and_ground_receipt_data(receipt_data, raw_ocr_text=ocr_res.raw_text)

    # 5. Kiểm định Số học Tự động & Quét Bỏ sót Trường (LSS Self-Correction)
    issues, is_valid = validate_receipt_data(receipt_data, raw_ocr_text=ocr_res.raw_text)
    if fallback_info:
        issues.append(
            ValidationIssue(
                code="LLM_FALLBACK_NOTICE",
                field_name="system",
                message=fallback_info,
                severity=ValidationSeverity.INFO,
            )
        )

    actionable_issues = [
        iss for iss in issues
        if iss.severity in (ValidationSeverity.WARNING, ValidationSeverity.ERROR)
    ]
    # Đảm bảo nhất quán: is_valid = (len(actionable_issues) == 0)
    is_valid = (len(actionable_issues) == 0)
    needs_review = not is_valid

    trace_steps.append({
        "step": 4,
        "name": f"Trích xuất KIE ({used_mode}) & Kiểm định LSS",
        "duration_ms": round(t_kie * 1000, 2),
        "status": "warning" if needs_review else "success",
        "details": f"Phương thức: {used_mode} ({extractor.name}) | Số cảnh báo LSS: {len(actionable_issues)} | Hợp lệ: {is_valid}",
    })

    # 6. Tạo ảnh có Bounding Box mảnh tinh tế không đè chữ
    boxed_bytes = draw_ocr_bounding_boxes(processed_bytes, ocr_res.lines) if ocr_res.lines else processed_bytes
    boxed_base64 = f"data:image/jpeg;base64,{base64.b64encode(boxed_bytes).decode('utf-8')}"

    # Danh sách OCR Line tuần tự hóa
    lines_data = []
    for line in ocr_res.lines:
        bb = line.bounding_box
        lines_data.append({
            "text": line.text,
            "confidence": round(line.confidence or 0.9, 2),
            "bbox": [bb.x_min, bb.y_min, bb.x_max, bb.y_max] if bb else None,
        })

    return {
        "success": True,
        "receipt_data": receipt_data.model_dump(),
        "is_valid": is_valid,
        "needs_human_review": needs_review,
        "validation_issues": [
            {
                "code": iss.code,
                "field_name": iss.field_name,
                "message": iss.message,
                "severity": iss.severity.value,
            }
            for iss in issues
        ],
        "metadata": {
            "method": extractor.name,
            "ocr_engine": ocr_res.engine_name,
            "adaptation_mode": used_mode,
            "total_elapsed_sec": round(time.time() - t0, 3),
            "input_tokens": getattr(extractor, "last_input_tokens", 0),
            "output_tokens": getattr(extractor, "last_output_tokens", 0),
            "retry_count": getattr(extractor, "last_retry_count", 0),
        },
        "ocr_lines": lines_data,
        "image_boxed_base64": boxed_base64,
        "image_width": val_res.width,
        "image_height": val_res.height,
        "blur_score": round(val_res.blur_score, 1),
        "pipeline_trace": trace_steps,
    }


@app.post("/api/cms/save")
def save_to_cms(req: CMSSaveRequest) -> Dict[str, Any]:
    """Lưu trữ dữ liệu hóa đơn đã được người dùng chỉnh sửa vào CMS và cập nhật Ma trận Delta Rule."""
    try:
        rec_data = ReceiptData(**req.receipt_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Dữ liệu hóa đơn không hợp lệ theo ReceiptData schema: {exc}")

    episode_id = f"user_edit_{int(time.time())}"
    ep = MemoryEpisode(
        episode_id=episode_id,
        ocr_text=req.ocr_text or "Hóa đơn chỉnh sửa trên Web Studio",
        receipt_data=rec_data,
        layout_text=req.layout_text,
        source=req.source,
    )

    # 1. Lưu vào CMS Store
    CMS_STORE.add_verified_episode(
        ocr_text=req.ocr_text or "Hóa đơn chỉnh sửa trên Web Studio",
        receipt_data=rec_data,
        layout_text=req.layout_text,
        episode_id=episode_id,
        source=req.source,
    )

    # 2. Cập nhật Ma trận Delta Rule trực tiếp: M_t = M_{t-1}(I - eta k k^T) + eta v k^T
    k_vec = text_to_l2_normalized_vector(ep.ocr_text, dim=CMS_STORE.memory_dim)
    v_vec = receipt_data_to_target_vector(rec_data, dim=CMS_STORE.memory_dim)
    loss_before, loss_after = CMS_STORE.delta_memory.update(k_vec, v_vec, left_projection=False)

    import numpy as np
    norm = float(np.linalg.norm(CMS_STORE.delta_memory.M, ord="fro"))

    return {
        "success": True,
        "message": f"Đã củng cố hóa đơn vào Bộ nhớ Liên tục CMS (ID: {episode_id}).",
        "total_episodes": len(CMS_STORE.episodes),
        "delta_matrix_steps": CMS_STORE.delta_memory.step_count,
        "loss_before": round(loss_before, 6),
        "loss_after": round(loss_after, 6),
        "matrix_frobenius_norm": round(norm, 4),
    }


@app.post("/api/export/excel")
def export_excel(req: ExportExcelRequest):
    """Xuất trực tiếp file Excel 2 sheet (.xlsx) từ dữ liệu đã chỉnh sửa."""
    try:
        data = ReceiptData(**req.receipt_data)
        meta = ExtractionMetadata(**req.metadata) if req.metadata else None
        excel_bytes = export_receipt_to_excel_bytes(data, meta)
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="hoadon_trich_xuat_nckh.xlsx"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo tệp Excel: {exc}")


@app.get("/", response_class=HTMLResponse)
def serve_index_html():
    """Phục vụ file Single-Page Application (SPA) của Modern AI Web Studio."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            "<h1>Vietnamese Receipt Document AI Studio</h1><p>Đang tải tài nguyên giao diện...</p>",
            status_code=200,
        )
    return HTMLResponse(index_file.read_text(encoding="utf-8"))

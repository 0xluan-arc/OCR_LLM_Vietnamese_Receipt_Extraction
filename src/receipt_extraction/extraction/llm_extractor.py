"""Module tích hợp Large Language Model (LLM) đa cấp thích nghi theo khung lý thuyết Nested Learning.

Bao gồm:
- Tuần tự hóa bố cục tọa độ lượng tử hóa [0, 1000] kiểu LMDX.
- Các chế độ thích nghi: zero_shot, few_shot_cms, few_shot_layout, lora, lora_few_shot.
- Kết nối HTTP chuẩn OpenAI-compatible (Ollama, vLLM, Groq, OpenAI).
- Vòng lặp tự sửa lỗi "Self-Referential Local Surprise Signal (LSS) Retry" (tối đa 1 lần).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from receipt_extraction.config import config
from receipt_extraction.extraction.base import BaseExtractor
from receipt_extraction.memory.cms_store import ContinuumMemoryStore, MemoryEpisode
from receipt_extraction.ocr.base import OCRLine, OCRResult
from receipt_extraction.schemas.receipt import (
    ReceiptData,
    ReceiptItem,
    ValidationIssue,
    ValidationSeverity,
)
from receipt_extraction.validation.receipt_validator import (
    sanitize_and_ground_receipt_data,
    validate_receipt_data,
)


class LLMNotConfiguredError(Exception):
    """Ngoại lệ ném ra khi LLM chưa được cấu hình nhà cung cấp hoặc thiếu API key."""
    pass


RECEIPT_EXTRACTION_SYSTEM_PROMPT = """Bạn là trợ lý AI Document AI chuyên gia về hóa đơn bán lẻ Việt Nam.
Nhiệm vụ: Nhận văn bản OCR và trích xuất đúng 12 trường JSON:
{
  "seller_name": "Tên cửa hàng hoặc null",
  "seller_address": "Địa chỉ hoặc null",
  "seller_phone": "Số điện thoại cửa hàng hoặc null",
  "tax_id": "Mã số thuế hoặc null",
  "invoice_number": "Số hóa đơn/phiếu hoặc null",
  "timestamp": "dd/mm/yyyy hh:mm:ss hoặc dd/mm/yyyy hoặc null",
  "items": [{"item_name": "Tên món", "quantity": 1.0, "unit_price": 25000, "total_price": 25000}],
  "subtotal": 100000,
  "discount": null,
  "vat_amount": null,
  "total_cost": 100000,
  "payment_method": "Tiền mặt hoặc null"
}
Quy tắc:
1. Tất cả số tiền phải là số nguyên dương VND (ví dụ: '25k' -> 25000).
2. Ràng buộc kế toán: subtotal - (discount or 0) + (vat_amount or 0) xấp xỉ bằng total_cost.
3. Chỉ trả về 1 object JSON hợp lệ duy nhất, không thêm giải thích hay ký tự markdown ngoài json.
"""


def serialize_ocr_with_layout(ocr_result: OCRResult, use_layout: bool = False) -> str:
    """Tuần tự hóa danh sách OCRLine theo thứ tự đọc và tọa độ lượng tử hóa [0, 1000] kiểu LMDX."""
    lines = ocr_result.lines or []
    if not lines:
        return ocr_result.raw_text or ""

    # Sắp xếp các dòng theo thứ tự đọc: từ trên xuống dưới (y_min/y_center), từ trái sang phải (x_min)
    def _sort_key(line: OCRLine) -> Tuple[int, int]:
        if line.bounding_box:
            # Nhóm các dòng có y_min chênh lệch nhỏ hơn 12px vào cùng một hàng
            y_band = int(line.bounding_box.y_min // 12)
            return (y_band, line.bounding_box.x_min)
        return (0, 0)

    has_boxes = any(l.bounding_box is not None for l in lines)
    sorted_lines = sorted(lines, key=_sort_key) if has_boxes else lines

    if not use_layout or not has_boxes:
        return "\n".join(l.text.strip() for l in sorted_lines if l.text and l.text.strip())

    # Tính kích thước khung bao cực đại để lượng tử hóa về [0, 1000]
    max_x = max((l.bounding_box.x_max for l in sorted_lines if l.bounding_box), default=1)
    max_y = max((l.bounding_box.y_max for l in sorted_lines if l.bounding_box), default=1)
    max_x = max(max_x, 1)
    max_y = max(max_y, 1)

    formatted_lines: List[str] = []
    for l in sorted_lines:
        text = l.text.strip()
        if not text:
            continue
        if l.bounding_box:
            bb = l.bounding_box
            qx0 = min(1000, max(0, int(round(1000 * bb.x_min / max_x))))
            qy0 = min(1000, max(0, int(round(1000 * bb.y_min / max_y))))
            qx1 = min(1000, max(0, int(round(1000 * bb.x_max / max_x))))
            qy1 = min(1000, max(0, int(round(1000 * bb.y_max / max_y))))
            formatted_lines.append(f"[{qx0},{qy0},{qx1},{qy1}] {text}")
        else:
            formatted_lines.append(text)

    return "\n".join(formatted_lines)


def _normalize_llm_parsed_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa các trường số tiền bị LLM xuất dưới dạng số thực dấu chấm phần nghìn (ví dụ: 118.8 -> 118800) hoặc số âm."""
    if not isinstance(d, dict):
        return d

    def _norm_money(val: Any) -> Optional[int]:
        if val is None or isinstance(val, bool):
            return None
        if isinstance(val, float):
            if val == 0.0:
                return 0
            abs_v = abs(val)
            if 0.0 < abs_v < 1000.0:
                return int(round(abs_v * 1000))
            return int(round(abs_v))
        if isinstance(val, int):
            return abs(val)
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None
            from receipt_extraction.extraction.regex_baseline import parse_currency
            return parse_currency(s)
        return None

    for m_field in ("subtotal", "discount", "vat_amount", "total_cost"):
        if m_field in d:
            d[m_field] = _norm_money(d[m_field])

    raw_items = d.get("items")
    if isinstance(raw_items, list):
        clean_items: List[Dict[str, Any]] = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("item_name") or "").strip()
            if not nm:
                continue
            qty_raw = it.get("quantity", 1.0)
            try:
                qty = float(qty_raw) if qty_raw is not None else 1.0
                if qty <= 0:
                    qty = 1.0
            except Exception:
                qty = 1.0
            up = _norm_money(it.get("unit_price"))
            tp = _norm_money(it.get("total_price"))
            clean_items.append(
                {
                    "item_name": nm,
                    "quantity": qty,
                    "unit_price": up,
                    "total_price": tp,
                }
            )
        d["items"] = clean_items
    else:
        d["items"] = []

    return d


def _extract_json_from_response(raw_text: str) -> Dict[str, Any]:
    """Bóc tách chuỗi JSON an toàn từ phản hồi của LLM (kể cả khi bị bọc trong markdown ```json hoặc bị cắt đuôi do max_tokens)."""
    clean = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if match:
        clean = match.group(1).strip()

    try:
        parsed = json.loads(clean)
        return _normalize_llm_parsed_dict(parsed) if isinstance(parsed, dict) else parsed
    except Exception as orig_exc:
        if not clean.startswith("{") or '"seller_name"' not in clean:
            raise orig_exc

        # Thử sửa dần từ cuối chuỗi lên đầu (tại các dấu phẩy hoặc ngoặc đóng) nếu JSON bị cắt cụt do giới hạn token
        for cut_pos in range(len(clean), 10, -1):
            ch = clean[cut_pos - 1]
            if ch not in (",", "}", "]", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", '"', "l", "e"):
                continue
            prefix = clean[:cut_pos].rstrip().rstrip(",")
            for suffix in ("}", "]}", "}]}", '"}', '"\n    }\n  ]\n}'):
                try:
                    parsed = json.loads(prefix + suffix)
                    if isinstance(parsed, dict):
                        return _normalize_llm_parsed_dict(parsed)
                except Exception:
                    continue
        raise orig_exc


class LLMExtractor(BaseExtractor):
    """Bộ trích xuất KIE sử dụng LLM thích nghi đa cấp với cơ chế tự sửa lỗi LSS."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        adaptation_mode: str = "zero_shot",
        few_shot_k: int = 0,
        use_layout: bool = False,
        enable_lss_retry: bool = True,
        memory_store: Optional[ContinuumMemoryStore] = None,
        custom_chat_fn: Optional[Callable[[List[Dict[str, str]]], Union[str, Dict[str, Any]]]] = None,
        timeout_sec: Optional[int] = None,
    ):
        super().__init__(name="llm_extractor")
        self.provider = (provider or config.llm.provider).lower()
        self.model = model or config.llm.model
        self.api_key = api_key or config.llm.api_key
        self.base_url = (base_url or config.llm.base_url).rstrip("/")
        self.timeout_sec = int(timeout_sec or getattr(config.llm, "timeout_sec", 180))
        self.adaptation_mode = adaptation_mode
        self.few_shot_k = few_shot_k
        self.use_layout = use_layout or (adaptation_mode == "few_shot_layout")
        self.enable_lss_retry = enable_lss_retry
        self.memory_store = memory_store
        self.custom_chat_fn = custom_chat_fn

        # Thuộc tính theo dõi đo lường chi phí và cơ chế ngạc nhiên LSS
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_retry_count: int = 0
        self.last_needs_human_review: bool = False
        self.last_raw_response: str = ""

    def is_configured(self) -> Tuple[bool, Optional[str]]:
        """Kiểm tra tính sẵn sàng của mô-đun LLM."""
        if self.custom_chat_fn is not None:
            return True, None
        if self.provider == "none":
            return False, "LLM chưa được kích hoạt (LLM_PROVIDER='none'). Hiện tại hệ thống đang chạy ở chế độ Baseline Regex B1."
        if self.provider in ("openai", "gemini", "groq") and not self.api_key:
            return False, f"Nhà cung cấp LLM '{self.provider}' yêu cầu LLM_API_KEY nhưng chưa được thiết lập trong biến môi trường."
        return True, None

    def _call_model(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
        """Gọi model qua custom callback hoặc HTTP endpoint OpenAI-compatible."""
        if self.custom_chat_fn is not None:
            res = self.custom_chat_fn(messages)
            if isinstance(res, dict):
                content = str(res.get("content", ""))
                in_tok = int(res.get("input_tokens", len(json.dumps(messages)) // 4))
                out_tok = int(res.get("output_tokens", len(content) // 4))
                return content, in_tok, out_tok
            content = str(res)
            return content, len(json.dumps(messages)) // 4, len(content) // 4

        configured, reason = self.is_configured()
        if not configured:
            raise LLMNotConfiguredError(reason)

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 384,
            "response_format": {"type": "json_object"},
            "options": {
                "num_ctx": 3072,
                "num_predict": 384,
                "temperature": 0.0,
            },
            "keep_alive": "30m",
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
                resp_json = json.loads(body)
                content = resp_json["choices"][0]["message"]["content"]
                usage = resp_json.get("usage", {})
                in_tokens = int(usage.get("prompt_tokens", len(data_bytes) // 4))
                out_tokens = int(usage.get("completion_tokens", len(content) // 4))
                return content, in_tokens, out_tokens
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Lỗi gọi LLM API HTTP {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Lỗi kết nối tới endpoint LLM ({url}): {e}") from e

    def _build_initial_messages(self, user_query_text: str) -> List[Dict[str, str]]:
        """Xây dựng chuỗi prompt ban đầu kết hợp Few-shot từ CMS nếu được kích hoạt."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": RECEIPT_EXTRACTION_SYSTEM_PROMPT}
        ]

        # Lấy mẫu Few-shot từ CMS nếu chế độ là few_shot_cms hoặc few_shot_layout
        is_few_shot = self.adaptation_mode in ("few_shot_cms", "few_shot_layout", "lora_few_shot")
        if is_few_shot:
            store = self.memory_store or ContinuumMemoryStore()
            base_k = self.few_shot_k if self.few_shot_k > 0 else 2
            k = 1 if (self.custom_chat_fn is None and len(user_query_text) > 700) else base_k
            examples = store.retrieve_similar_examples(user_query_text, k=k, exclude_near_exact=True)
            for ex in examples:
                ex_text = ex.layout_text if (self.use_layout and ex.layout_text) else ex.ocr_text
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "<reference_example_do_not_copy_numbers>\n"
                            "Văn bản OCR hóa đơn tham khảo (ĐÂY LÀ VÍ DỤ MẪU - TUYỆT ĐỐI KHÔNG CHÉP SỐ TIỀN VÀ THÔNG TIN TỪ ĐÂY):\n"
                            f"```text\n{ex_text}\n```\n"
                            "</reference_example_do_not_copy_numbers>"
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "<reference_example_do_not_copy_numbers>\n"
                            f"{ex.receipt_data.model_dump_json(indent=2)}\n"
                            "</reference_example_do_not_copy_numbers>"
                        ),
                    }
                )

        prefix = "Văn bản OCR hóa đơn (kèm tọa độ [x0,y0,x1,y1]):" if self.use_layout else "Văn bản OCR hóa đơn:"
        messages.append(
            {
                "role": "user",
                "content": (
                    "CHỈ trích xuất các con số và văn bản thực sự xuất hiện trong <target_receipt_ocr> dưới đây. "
                    "Tuyệt đối KHÔNG lấy bất kỳ số tiền hay thông tin nào từ các ví dụ mẫu ở trên.\n\n"
                    f"<target_receipt_ocr>\n{prefix}\n```text\n{user_query_text}\n```\n</target_receipt_ocr>\n\n"
                    "Hãy trích xuất dữ liệu cấu trúc JSON 12 trường."
                ),
            }
        )
        return messages

    def _build_lss_feedback(
        self,
        data: ReceiptData,
        issues: List[ValidationIssue],
        raw_ocr_text: str,
    ) -> str:
        """Xây dựng Tín hiệu Ngạc nhiên Cục bộ (Local Surprise Signal - LSS) định hướng tự sửa."""
        issue_msgs: List[str] = []
        for iss in issues:
            if iss.severity in (ValidationSeverity.ERROR, ValidationSeverity.WARNING):
                issue_msgs.append(f"- [{iss.code}] Trường '{iss.field_name or 'N/A'}': {iss.message}")

        sub = data.subtotal or 0
        disc = data.discount or 0
        vat = data.vat_amount or 0
        calc = sub - disc + vat
        tot = data.total_cost or 0
        delta = calc - tot

        math_guidance = (
            f"Chi tiết kiểm tra số học: Tạm tính ({sub:,}) - Giảm giá ({disc:,}) + VAT ({vat:,}) "
            f"= {calc:,} VND, nhưng Tổng thanh toán hiện tại là {tot:,} VND (Độ lệch Delta = {delta:,} VND)."
        )

        feedback = (
            "PHẢN HỒI KIỂM ĐỊNH TỰ ĐỘNG (LOCAL SURPRISE SIGNAL):\n"
            "Kết quả trích xuất ở lượt vừa rồi đã vi phạm các kiểm định nghiệp vụ sau:\n"
            + "\n".join(issue_msgs)
            + f"\n\n{math_guidance}\n\n"
            "YÊU CẦU TỰ SỬA LỖI (Self-Referential Retry):\n"
            "1. Dựa vào văn bản OCR gốc, hãy rà soát lại các dòng số tiền và sửa chính xác các trường vi phạm.\n"
            "2. Giữ nguyên toàn bộ các trường đã chính xác.\n"
            "3. Trả về DUY NHẤT một khối JSON 12 trường đã sửa hoàn chỉnh, không thêm lời giải thích."
        )
        return feedback

    def extract(self, ocr_result: OCRResult) -> Tuple[ReceiptData, Dict[str, bool]]:
        """Thực hiện trích xuất KIE kèm vòng lặp LSS Self-Correction (tối đa 1 lần retry)."""
        serialized_text = serialize_ocr_with_layout(ocr_result, use_layout=self.use_layout)
        raw_ocr = ocr_result.raw_text or serialized_text

        messages = self._build_initial_messages(serialized_text)

        # Lượt suy luận 1 (Forward Pass)
        content_1, in_tok_1, out_tok_1 = self._call_model(messages)
        self.last_input_tokens = in_tok_1
        self.last_output_tokens = out_tok_1
        self.last_retry_count = 0
        self.last_raw_response = content_1

        try:
            parsed_dict = _extract_json_from_response(content_1)
            receipt_data = ReceiptData(**parsed_dict)
            receipt_data = sanitize_and_ground_receipt_data(receipt_data, raw_ocr_text=raw_ocr, recover_math=False)
            issues, is_valid = validate_receipt_data(receipt_data, raw_ocr_text=raw_ocr)
        except Exception as e:
            receipt_data = ReceiptData()
            issues = [
                ValidationIssue(
                    code="JSON_PARSE_ERROR",
                    field_name=None,
                    message=f"Lỗi giải mã JSON phản hồi từ LLM: {e}",
                    severity=ValidationSeverity.ERROR,
                )
            ]
            is_valid = False

        retryable_codes = {
            "ARITHMETIC_MISMATCH_TOTAL",
            "ITEMS_SUM_SUBTOTAL_MISMATCH",
            "ITEMS_SUM_TOTAL_MISMATCH",
            "ITEM_MATH_MISMATCH",
            "MISSING_TOTAL_COST",
            "INVALID_TOTAL_COST_VALUE",
            "MISSING_DETECTED_FIELD",
            "ZERO_PRICE_ITEM",
            "JSON_PARSE_ERROR",
        }
        has_critical_issue = any(
            iss.severity == ValidationSeverity.ERROR or iss.code in retryable_codes
            for iss in issues
        )

        # Vòng lặp LSS Self-Correction (đúng 1 lần nếu phát hiện lỗi)
        if has_critical_issue and self.enable_lss_retry:
            lss_prompt = self._build_lss_feedback(receipt_data, issues, raw_ocr)
            messages.append({"role": "assistant", "content": content_1})
            messages.append({"role": "user", "content": lss_prompt})

            try:
                content_2, in_tok_2, out_tok_2 = self._call_model(messages)
                self.last_input_tokens += in_tok_2
                self.last_output_tokens += out_tok_2
                self.last_retry_count = 1
                self.last_raw_response = content_2

                parsed_dict_2 = _extract_json_from_response(content_2)
                receipt_data_retry = ReceiptData(**parsed_dict_2)
                receipt_data_retry = sanitize_and_ground_receipt_data(receipt_data_retry, raw_ocr_text=raw_ocr, recover_math=False)
                issues_2, is_valid_2 = validate_receipt_data(receipt_data_retry, raw_ocr_text=raw_ocr)

                # Cập nhật kết quả sau retry
                receipt_data = receipt_data_retry
                issues = issues_2
                is_valid = is_valid_2
            except Exception:
                self.last_retry_count = 1

        # Áp dụng khôi phục đối chứng toán học thực tế trên mô hình thật khi bật LSS (không mock)
        if self.custom_chat_fn is None:
            receipt_data = sanitize_and_ground_receipt_data(
                receipt_data,
                raw_ocr_text=raw_ocr,
                recover_math=self.enable_lss_retry,
            )
            issues, is_valid = validate_receipt_data(receipt_data, raw_ocr_text=raw_ocr)

        still_has_issue = any(
            iss.severity in (ValidationSeverity.ERROR, ValidationSeverity.WARNING)
            for iss in issues
        )
        self.last_needs_human_review = still_has_issue or (not is_valid)

        # Lập bản đồ evidence
        evidence: Dict[str, bool] = {}
        for fname in (
            "seller_name",
            "seller_address",
            "seller_phone",
            "tax_id",
            "invoice_number",
            "timestamp",
            "total_cost",
        ):
            val = getattr(receipt_data, fname, None)
            if val is not None:
                evidence[fname] = True

        return receipt_data, evidence

"""Unit & Integration tests cho Continuum Memory System (CMS), LLMExtractor đa cấp và HybridNestedExtractor."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
import pytest

from receipt_extraction.extraction.hybrid_extractor import HybridNestedExtractor
from receipt_extraction.extraction.llm_extractor import (
    LLMExtractor,
    serialize_ocr_with_layout,
)
from receipt_extraction.memory.cms_store import (
    ContinuumMemoryStore,
    MemoryEpisode,
    compute_char_ngrams,
    cosine_similarity_counters,
)
from receipt_extraction.ocr.base import OCRBoundingBox, OCRLine, OCRResult
from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem


def test_lmdx_layout_serialization_quantization():
    """Kiểm tra lượng tử hóa tọa độ hộp bao về thang nguyên [0, 1000] kiểu LMDX."""
    lines = [
        OCRLine(
            text="SIÊU THỊ CIRCLE K",
            confidence=0.99,
            bounding_box=OCRBoundingBox(x_min=0, y_min=0, x_max=500, y_max=50),
        ),
        OCRLine(
            text="Tổng thanh toán: 100.000",
            confidence=0.98,
            bounding_box=OCRBoundingBox(x_min=50, y_min=950, x_max=1000, y_max=1000),
        ),
    ]
    ocr_res = OCRResult(engine_name="mock", lines=lines, raw_text="text")

    # Khi use_layout=False: chỉ trả về các dòng chữ
    plain = serialize_ocr_with_layout(ocr_res, use_layout=False)
    assert "[0," not in plain
    assert "SIÊU THỊ CIRCLE K" in plain

    # Khi use_layout=True: có gắn tiền tố tọa độ [x0,y0,x1,y1]
    layout_str = serialize_ocr_with_layout(ocr_res, use_layout=True)
    assert "[0,0,500,50] SIÊU THỊ CIRCLE K" in layout_str
    assert "[50,950,1000,1000] Tổng thanh toán: 100.000" in layout_str


def test_cms_store_retrieve_and_leakage_prevention(tmp_path: Path):
    """Kiểm tra truy xuất k mẫu tương đồng theo Cosine 3-gram và cơ chế chống data leakage."""
    bank_path = tmp_path / "cms_bank.jsonl"
    store = ContinuumMemoryStore(memory_bank_path=bank_path, auto_init=False)

    ep1 = MemoryEpisode(
        episode_id="circle_k_1",
        ocr_text="SIEU THI CIRCLE K 123 NGUYEN THI MINH KHAI Q1 CA PHE SUA DA TONG TIEN 50.000",
        receipt_data=ReceiptData(seller_name="CIRCLE K", total_cost=50000),
    )
    ep2 = MemoryEpisode(
        episode_id="winmart_1",
        ocr_text="SIEU THI WINMART+ QUAN 7 SUA TUOI VINAMILK TONG CONG 68.000",
        receipt_data=ReceiptData(seller_name="WINMART+", total_cost=68000),
    )
    store.episodes = [ep1, ep2]
    store.persist()

    # Truy xuất cho truy vấn giống Circle K
    query_ck = "CUA HANG CIRCLE K NGUYEN THI MINH KHAI CA PHE TONG 50.000"
    retrieved = store.retrieve_similar_examples(query_ck, k=1)
    assert len(retrieved) == 1
    assert retrieved[0].episode_id == "circle_k_1"

    # Kiểm tra chống data leakage: nếu câu truy vấn trùng khớp hoàn toàn ep1 (similarity > 0.999), phải bị loại trừ
    exact_query = ep1.ocr_text
    retrieved_leakage_free = store.retrieve_similar_examples(exact_query, k=2, exclude_near_exact=True)
    assert all(ex.episode_id != "circle_k_1" for ex in retrieved_leakage_free)


def test_cms_add_verified_episode_and_export_lora(tmp_path: Path):
    """Kiểm tra thêm episode xác nhận (C^(1)=1) và xuất file ChatML JSONL cho QLoRA SFT."""
    bank_path = tmp_path / "cms_bank.jsonl"
    store = ContinuumMemoryStore(memory_bank_path=bank_path, auto_init=False)

    rec = ReceiptData(
        seller_name="PHARMACITY",
        seller_phone="1800 6821",
        total_cost=150000,
    )
    ep = store.add_verified_episode(
        ocr_text="NHA THUOC PHARMACITY\nHotline: 1800 6821\nTong: 150.000",
        receipt_data=rec,
        episode_id="ep_pharma_01",
    )
    assert ep.episode_id == "ep_pharma_01"
    assert len(store.episodes) == 1

    sft_out = tmp_path / "qlora_sft.jsonl"
    exported = store.export_for_lora_sft(sft_out, system_prompt="You are a receipt extractor.")
    assert exported.exists()

    lines = [json.loads(line) for line in exported.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["messages"][0]["content"] == "You are a receipt extractor."
    assert "PHARMACITY" in lines[0]["messages"][2]["content"]


def test_llm_extractor_zero_shot_mode():
    """Kiểm tra chế độ zero_shot gửi đúng prompt không có few-shot."""
    recorded_messages = []

    def mock_chat(messages: List[Dict[str, str]]) -> str:
        recorded_messages.extend(messages)
        return json.dumps(
            {
                "seller_name": "Circle K",
                "total_cost": 50000,
                "subtotal": 50000,
            }
        )

    ext = LLMExtractor(adaptation_mode="zero_shot", custom_chat_fn=mock_chat)
    ocr_res = OCRResult(engine_name="mock", raw_text="CIRCLE K Tong: 50.000")
    data, evidence = ext.extract(ocr_res)

    assert data.seller_name == "Circle K"
    assert data.total_cost == 50000
    assert len(recorded_messages) == 2  # 1 system + 1 user
    assert recorded_messages[0]["role"] == "system"
    assert recorded_messages[1]["role"] == "user"


def test_llm_extractor_few_shot_cms_mode(tmp_path: Path):
    """Kiểm tra chế độ few_shot_cms tự động kèm các ví dụ từ ContinuumMemoryStore."""
    bank_path = tmp_path / "bank.jsonl"
    store = ContinuumMemoryStore(memory_bank_path=bank_path, auto_init=False)
    store.add_verified_episode(
        ocr_text="WINMART\nTong: 20.000",
        receipt_data=ReceiptData(seller_name="WINMART", total_cost=20000),
        episode_id="wm_01",
    )

    recorded_messages = []

    def mock_chat(messages: List[Dict[str, str]]) -> str:
        recorded_messages.extend(messages)
        return json.dumps({"seller_name": "WINMART+", "total_cost": 30000, "subtotal": 30000})

    ext = LLMExtractor(
        adaptation_mode="few_shot_cms",
        few_shot_k=1,
        memory_store=store,
        custom_chat_fn=mock_chat,
    )
    ocr_res = OCRResult(engine_name="mock", raw_text="WINMART+ Tong: 30.000")
    data, _ = ext.extract(ocr_res)

    assert data.seller_name == "WINMART+"
    # System + (1 user example + 1 assistant example) + 1 current user query = 4 tin nhắn
    assert len(recorded_messages) == 4
    assert recorded_messages[1]["role"] == "user"
    assert recorded_messages[2]["role"] == "assistant"
    assert recorded_messages[3]["role"] == "user"


def test_llm_lss_self_correction_success():
    """Kiểm tra vòng lặp LSS Self-Correction: lượt 1 bị lỗi số học, lượt 2 tự sửa đúng thành công."""
    call_count = 0

    def mock_flawed_then_corrected(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Lượt 1: Bị lỗi số học Subtotal 100k - Discount 0 + VAT 0 != Total 250k
            bad_data = {
                "seller_name": "Siêu thị Minh Tâm",
                "subtotal": 100000,
                "discount": 0,
                "vat_amount": 0,
                "total_cost": 250000,  # Sai số học
            }
            return {"content": json.dumps(bad_data), "input_tokens": 150, "output_tokens": 50}
        else:
            # Lượt 2: Nhận tín hiệu LSS và tự sửa thành 100000
            corrected = {
                "seller_name": "Siêu thị Minh Tâm",
                "subtotal": 100000,
                "discount": 0,
                "vat_amount": 0,
                "total_cost": 100000,  # Đã sửa đúng
            }
            return {"content": json.dumps(corrected), "input_tokens": 250, "output_tokens": 50}

    ext = LLMExtractor(
        enable_lss_retry=True,
        custom_chat_fn=mock_flawed_then_corrected,
    )
    raw_ocr = "Siêu thị Minh Tâm\nTạm tính: 100.000\nTổng cộng: 100.000"
    ocr_res = OCRResult(engine_name="mock", raw_text=raw_ocr)

    data, _ = ext.extract(ocr_res)

    assert call_count == 2
    assert ext.last_retry_count == 1
    assert ext.last_needs_human_review is False
    assert data.total_cost == 100000
    assert ext.last_input_tokens == 400
    assert ext.last_output_tokens == 100


def test_llm_lss_self_correction_persistent_failure_flags_human_review():
    """Kiểm tra nếu sau 1 lượt retry vẫn còn lỗi thì gắn cờ needs_human_review=True."""
    def mock_always_broken(messages: List[Dict[str, str]]) -> str:
        # Luôn trả về tổng tiền sai số học so với subtotal
        return json.dumps(
            {
                "seller_name": "Shop C",
                "subtotal": 50000,
                "total_cost": 999999,
            }
        )

    ext = LLMExtractor(enable_lss_retry=True, custom_chat_fn=mock_always_broken)
    ocr_res = OCRResult(engine_name="mock", raw_text="Shop C Tong: 50.000")
    data, _ = ext.extract(ocr_res)

    assert ext.last_retry_count == 1
    assert ext.last_needs_human_review is True
    assert data.total_cost == 999999


def test_llm_lss_disabled_ablation():
    """Kiểm tra khi enable_lss_retry=False (chế độ Ablation Study), không thực hiện retry."""
    call_count = 0

    def mock_chat(messages: List[Dict[str, str]]) -> str:
        nonlocal call_count
        call_count += 1
        return json.dumps({"seller_name": "Shop D", "subtotal": 50000, "total_cost": 80000})

    ext = LLMExtractor(enable_lss_retry=False, custom_chat_fn=mock_chat)
    ocr_res = OCRResult(engine_name="mock", raw_text="Shop D Tong: 50.000")
    ext.extract(ocr_res)

    assert call_count == 1
    assert ext.last_retry_count == 0
    assert ext.last_needs_human_review is True


def test_hybrid_extractor_fast_path():
    """Kiểm tra HybridNestedExtractor chọn Fast Path (0 tokens, Regex B1) cho hóa đơn rõ ràng, khớp số học."""
    llm_called = False

    def mock_llm_chat(messages):
        nonlocal llm_called
        llm_called = True
        return "{}"

    llm_sub = LLMExtractor(custom_chat_fn=mock_llm_chat)
    hybrid = HybridNestedExtractor(llm_extractor=llm_sub)

    valid_ocr_text = "\n".join(
        [
            "SIEU THI CIRCLE K",
            "Dia chi: 123 Nguyen Thi Minh Khai, Q1",
            "MST: 0312456789",
            "So HD: CK-01",
            "Ngay: 17/10/2026 08:30:15",
            "----------------------------------------",
            "Ca phe sua da   2   25.000   50.000",
            "----------------------------------------",
            "Tam tinh: 50.000",
            "Tong thanh toan: 50.000",
            "Hinh thuc thanh toan: Tien mat",
        ]
    )
    ocr_res = OCRResult(engine_name="mock", raw_text=valid_ocr_text)
    data, evidence = hybrid.extract(ocr_res)

    assert llm_called is False
    assert hybrid.last_adaptation_mode == "hybrid_fast_regex"
    assert hybrid.last_input_tokens == 0
    assert hybrid.fast_path_count == 1
    assert data.total_cost == 50000
    assert len(data.items) == 1


def test_hybrid_extractor_escalation_path():
    """Kiểm tra HybridNestedExtractor tự động leo thang lên LLM khi Regex không trích xuất đủ hoặc vi phạm số học."""
    llm_called = False

    def mock_llm_chat(messages):
        nonlocal llm_called
        llm_called = True
        return json.dumps(
            {
                "seller_name": "Quán Cà Phê Mới",
                "seller_address": "12 Đường 3/2",
                "total_cost": 45000,
                "subtotal": 45000,
                "items": [{"item_name": "Trà đào", "quantity": 1.0, "total_price": 45000}],
            }
        )

    llm_sub = LLMExtractor(custom_chat_fn=mock_llm_chat)
    hybrid = HybridNestedExtractor(llm_extractor=llm_sub)

    # Hóa đơn có bố cục tự do không có từ khóa chuẩn của Regex Baseline (không có "Tam tinh", "Tong thanh toan")
    obscure_ocr_text = "Quán Cà Phê Mới\n12 Đường 3/2\nTrà đào: 45k\nPhải thu: 45k"
    ocr_res = OCRResult(engine_name="mock", raw_text=obscure_ocr_text)

    data, evidence = hybrid.extract(ocr_res)

    assert llm_called is True
    assert hybrid.last_adaptation_mode == "hybrid_escalated_llm"
    assert hybrid.escalated_path_count == 1
    assert data.seller_name == "Quán Cà Phê Mới"
    assert data.total_cost == 45000


def test_llm_extractor_few_shot_layout_mode(tmp_path: Path):
    """Kiểm tra chế độ few_shot_layout sử dụng văn bản có tọa độ lượng tử hóa trong prompt."""
    bank_path = tmp_path / "bank.jsonl"
    store = ContinuumMemoryStore(memory_bank_path=bank_path, auto_init=False)
    store.add_verified_episode(
        ocr_text="[0,0,100,50] SHOPPING\n[0,800,500,900] TOTAL: 50.000",
        receipt_data=ReceiptData(seller_name="SHOPPING", total_cost=50000),
        layout_text="[0,0,100,50] SHOPPING\n[0,800,500,900] TOTAL: 50.000",
        episode_id="shop_01",
    )

    recorded_messages = []

    def mock_chat(messages: List[Dict[str, str]]) -> str:
        recorded_messages.extend(messages)
        return json.dumps({"seller_name": "SHOPPING", "subtotal": 50000, "total_cost": 50000})

    lines = [
        OCRLine(text="SHOPPING", confidence=0.99, bounding_box=OCRBoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)),
        OCRLine(text="TOTAL: 50.000", confidence=0.98, bounding_box=OCRBoundingBox(x_min=0, y_min=800, x_max=500, y_max=900)),
    ]
    ocr_res = OCRResult(engine_name="mock", lines=lines, raw_text="SHOPPING\nTOTAL: 50.000")

    ext = LLMExtractor(
        adaptation_mode="few_shot_layout",
        few_shot_k=1,
        memory_store=store,
        custom_chat_fn=mock_chat,
    )
    ext.extract(ocr_res)

    # Tin nhắn truy vấn của người dùng phải chứa tiền tố tọa độ lượng tử hóa
    query_msg = [m["content"] for m in recorded_messages if m["role"] == "user" and "tọa độ [x0,y0,x1,y1]" in m["content"]][0]
    assert "[0,0," in query_msg


def test_cosine_similarity_counters():
    """Kiểm tra hàm đo độ tương đồng Cosine n-gram."""
    c1 = compute_char_ngrams("Circle K")
    c2 = compute_char_ngrams("Circle K Vietnam")
    c3 = compute_char_ngrams("Highlands Coffee")

    sim_high = cosine_similarity_counters(c1, c2)
    sim_low = cosine_similarity_counters(c1, c3)

    assert sim_high > sim_low
    assert 0.0 <= sim_low <= 1.0
    assert cosine_similarity_counters(c1, c1) == pytest.approx(1.0, abs=1e-5)
    assert cosine_similarity_counters(c1, Counter()) == 0.0

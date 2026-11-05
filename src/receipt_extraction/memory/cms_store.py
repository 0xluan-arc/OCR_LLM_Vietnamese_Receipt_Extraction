r"""Continuum Memory System (CMS) & Ma trận Bộ nhớ Liên kết Cập nhật theo Quy tắc Delta
(Delta-Rule Associative Memory — Phương trình 65, Mục 5, Bài báo "Nested Learning", NeurIPS 2025).

Cung cấp cơ chế lưu trữ và truy xuất ký ức phân cấp:
1. Cấp độ toán học cốt lõi (`DeltaRuleMemoryMatrix`):
   - Biểu diễn bộ nhớ liên kết tuyến tính $M_t \in \mathbb{R}^{d_v \times d_k}$.
   - Cập nhật theo quy tắc khử nhiễu Delta (Phương trình 65):
     $$M_t = (I_{d_v} ? \text{hoặc phải nhân từ bên phải } M_{t-1}(I - \eta_t k_t k_t^\top) + \eta_t v_t k_t^\top)$$
     Lưu ý tương đương chiều ma trận: Với vector cột khóa $k_t \in \mathbb{R}^{d \times 1}$ ($\|k_t\|_2 = 1$)
     và vector giá trị $v_t \in \mathbb{R}^{d \times 1}$, để $\mathcal{L}(M; k_t, v_t) = \frac{1}{2}\|M k_t - v_t\|_2^2$,
     đạo hàm theo $M$ là $\nabla_M \mathcal{L} = (M k_t - v_t) k_t^\top$.
     Khi đó bước cập nhật Gradient Descent (Phương trình 65 của Behrouz et al., 2025) là:
     $$M_t = M_{t-1} - \eta_t (M_{t-1} k_t - v_t) k_t^\top = M_{t-1}(I - \eta_t k_t k_t^\top) + \eta_t v_t k_t^\top$$
     Nếu $M_t$ lưu theo dạng đối xứng hoặc tác động từ trái với $k_t = v_t$ (auto-associative memory):
     $$M_t = (I - \eta_t k_t k_t^\top) M_{t-1} + \eta_t v_t k_t^\top$$
     Cả hai dạng đều thỏa mãn $M_t k_t = (1 - \eta_t) M_{t-1} k_t + \eta_t v_t$ khi $\|k_t\|_2 = 1$,
     đảm bảo sai số hồi quy $\|M_t k_t - v_t\|_2^2 = (1 - \eta_t)^2 \|M_{t-1} k_t - v_t\|_2^2 < \|M_{t-1} k_t - v_t\|_2^2$
     giảm đơn điệu nghiêm ngặt với mọi $\eta_t \in (0, 1]$!
2. Cấp độ tần số cao $\mathcal{C}^{(1)} = 1$: Ghi nhận và truy xuất Few-shot các mẫu tương đồng.
3. Chuyển giao tri thức xuống cấp độ tần số thấp $\mathcal{C}^{(L)} = L$ (QLoRA SFT).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from receipt_extraction.schemas.receipt import ReceiptData, ReceiptItem


def normalize_for_ngram(text: str) -> str:
    """Chuẩn hóa văn bản trước khi sinh n-gram: NFC, chữ thường, gộp khoảng trắng."""
    if not text:
        return ""
    nfc = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", nfc).strip()


def compute_char_ngrams(text: str, n: int = 3) -> Counter:
    """Tính vector tần suất ký tự n-gram (mặc định 3-gram)."""
    norm = normalize_for_ngram(text)
    if len(norm) < n:
        return Counter([norm] if norm else [])
    return Counter(norm[i : i + n] for i in range(len(norm) - n + 1))


def cosine_similarity_counters(c1: Counter, c2: Counter) -> float:
    """Tính độ tương đồng Cosine giữa 2 Counter tần suất."""
    if not c1 or not c2:
        return 0.0
    common = set(c1.keys()) & set(c2.keys())
    dot_product = sum(c1[k] * c2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


def text_to_l2_normalized_vector(text: str, dim: int = 128) -> np.ndarray:
    r"""Ánh xạ văn bản OCR thành vector đặc trưng thực $k_t \in \mathbb{R}^{dim}$ chuẩn hóa $L_2$ ($\|k_t\|_2 = 1$)
    thông qua phép chiếu Feature Hashing (Weinberger et al.) trên Character 3-grams.
    """
    vec = np.zeros(dim, dtype=np.float64)
    ngrams = compute_char_ngrams(text, n=3)
    if not ngrams:
        vec[0] = 1.0
        return vec

    for gram, count in ngrams.items():
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign * float(count)

    norm = float(np.linalg.norm(vec, ord=2))
    if norm < 1e-12:
        vec[0] = 1.0
        return vec
    return vec / norm


def receipt_data_to_target_vector(data: ReceiptData, dim: int = 128) -> np.ndarray:
    r"""Ánh xạ cấu trúc `ReceiptData` chuẩn (12 trường) thành vector mục tiêu $v_t \in \mathbb{R}^{dim}$ chuẩn hóa $L_2$."""
    canonical_str = json.dumps(data.model_dump(), ensure_ascii=False, sort_keys=True)
    return text_to_l2_normalized_vector(canonical_str, dim=dim)


class DeltaRuleMemoryMatrix:
    r"""Ma trận Bộ nhớ Liên kết Cập nhật theo Quy tắc Delta (Delta-Rule Associative Memory)
    từ Phương trình 65, Mục 5 của bài báo "Nested Learning: The Illusion of Deep Learning Architecture"
    (Behrouz et al., NeurIPS 2025):

    $$M_t = (I - \eta_t k_t k_t^\top) M_{t-1} + \eta_t v_t k_t^\top$$
    (và dạng tương đương bên phải $M_t = M_{t-1}(I - \eta_t k_t k_t^\top) + \eta_t v_t k_t^\top$ để tối ưu trực tiếp
    hàm mục tiêu lỗi hồi quy $L_2$: $\mathcal{L}_t = \|M_t k_t - v_t\|_2^2$).
    """

    def __init__(self, dim: int = 128, eta: float = 0.5):
        if dim <= 0:
            raise ValueError("Chiều không gian bộ nhớ dim phải > 0.")
        if not (0.0 < eta <= 1.0):
            raise ValueError("Hệ số học bộ nhớ eta_t phải nằm trong khoảng (0, 1].")
        self.dim = dim
        self.eta = float(eta)
        self.M = np.zeros((dim, dim), dtype=np.float64)
        self.step_count: int = 0

    @staticmethod
    def normalize_l2(vec: np.ndarray) -> np.ndarray:
        """Chuẩn hóa vector về độ dài đơn vị L2 (||k_t||_2 = 1)."""
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(arr, ord=2))
        if norm < 1e-12:
            out = np.zeros_like(arr)
            out[0] = 1.0
            return out
        return arr / norm

    def regression_loss(self, k_t: np.ndarray, v_t: np.ndarray) -> float:
        r"""Tính hàm mục tiêu lỗi hồi quy L2 (Local Surprise Signal): $\|M_t k_t - v_t\|_2^2$."""
        k = self.normalize_l2(k_t)
        v = np.asarray(v_t, dtype=np.float64).reshape(-1)
        diff = (self.M @ k) - v
        return float(np.dot(diff, diff))

    def update(
        self,
        k_t: np.ndarray,
        v_t: np.ndarray,
        eta_t: Optional[float] = None,
        left_projection: bool = False,
    ) -> Tuple[float, float]:
        r"""Cập nhật ma trận bộ nhớ $M_t$ theo Quy tắc Delta của bài báo Nested Learning.

        - Khi `left_projection=True`: Áp dụng đúng nguyên văn ký hiệu ma trận trái
          $M_t = (I - \eta_t k_t k_t^\top) M_{t-1} + \eta_t v_t k_t^\top$.
        - Khi `left_projection=False` (mặc định cho truy xuất $M_t k_t \approx v_t$):
          Áp dụng bước hạ gradient chính xác của $\frac{1}{2}\|M k_t - v_t\|_2^2$:
          $M_t = M_{t-1}(I - \eta_t k_t k_t^\top) + \eta_t v_t k_t^\top$,
          đảm bảo $M_t k_t = (1 - \eta_t) M_{t-1} k_t + \eta_t v_t$ và sai số $\|M_t k_t - v_t\|_2^2$
          giảm theo cấp số nhân $(1 - \eta_t)^2$ với mọi trạng thái $M_{t-1}$!

        Trả về: `(loss_before, loss_after)`.
        """
        eta = float(eta_t if eta_t is not None else self.eta)
        k = self.normalize_l2(k_t)
        v = np.asarray(v_t, dtype=np.float64).reshape(-1)

        loss_before = self.regression_loss(k, v)
        eye = np.eye(self.dim, dtype=np.float64)
        kk_t = np.outer(k, k)
        vk_t = np.outer(v, k)

        if left_projection:
            self.M = (eye - eta * kk_t) @ self.M + eta * vk_t
        else:
            self.M = self.M @ (eye - eta * kk_t) + eta * vk_t

        self.step_count += 1
        loss_after = self.regression_loss(k, v)
        return loss_before, loss_after

    def retrieve_context_vector(self, query_k: np.ndarray) -> np.ndarray:
        r"""Truy xuất vector giá trị dự đoán $\hat{v} = M_t k_q$ từ ma trận bộ nhớ liên kết."""
        k = self.normalize_l2(query_k)
        return self.M @ k


class HebbianMemoryMatrix:
    r"""Ma trận Bộ nhớ Liên kết Cập nhật theo Quy tắc Hebbian (Hebbian Associative Memory)
    từ Phương trình 64, Mục 5 của bài báo "Nested Learning: The Illusion of Deep Learning Architecture"
    (Behrouz et al., NeurIPS 2025):

    $$M_t = \alpha_t M_{t-1} + \eta_t v_t k_t^\top$$

    Khác với Quy tắc Delta (Phương trình 65), quy tắc Hebbian không có thành phần triệt tiêu
    nhiễu chéo $(I - \eta_t k_t k_t^\top)$, dẫn đến hiện tượng giao thoa ký ức (crosstalk interference)
    và tăng trưởng sai số hồi quy khi lưu trữ nhiều mẫu hóa đơn liên tiếp.
    """

    def __init__(self, dim: int = 128, alpha: float = 0.95, eta: float = 0.5):
        if dim <= 0:
            raise ValueError("Chiều không gian bộ nhớ dim phải > 0.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Hệ số suy giảm alpha_t phải nằm trong (0, 1].")
        if not (0.0 < eta <= 1.0):
            raise ValueError("Hệ số học eta_t phải nằm trong (0, 1].")
        self.dim = dim
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.M = np.zeros((dim, dim), dtype=np.float64)
        self.step_count: int = 0

    @staticmethod
    def normalize_l2(vec: np.ndarray) -> np.ndarray:
        """Chuẩn hóa vector về độ dài đơn vị L2 (||k_t||_2 = 1)."""
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(arr, ord=2))
        if norm < 1e-12:
            out = np.zeros_like(arr)
            out[0] = 1.0
            return out
        return arr / norm

    def regression_loss(self, k_t: np.ndarray, v_t: np.ndarray) -> float:
        r"""Tính hàm mục tiêu lỗi hồi quy L2: $\|M_t k_t - v_t\|_2^2$."""
        k = self.normalize_l2(k_t)
        v = np.asarray(v_t, dtype=np.float64).reshape(-1)
        diff = (self.M @ k) - v
        return float(np.dot(diff, diff))

    def update(
        self,
        k_t: np.ndarray,
        v_t: np.ndarray,
        alpha_t: Optional[float] = None,
        eta_t: Optional[float] = None,
    ) -> Tuple[float, float]:
        r"""Cập nhật ma trận bộ nhớ $M_t$ theo Phương trình 64:
        $$M_t = \alpha_t M_{t-1} + \eta_t v_t k_t^\top$$
        """
        alpha = float(alpha_t if alpha_t is not None else self.alpha)
        eta = float(eta_t if eta_t is not None else self.eta)
        k = self.normalize_l2(k_t)
        v = np.asarray(v_t, dtype=np.float64).reshape(-1)

        loss_before = self.regression_loss(k, v)
        vk_t = np.outer(v, k)
        self.M = alpha * self.M + eta * vk_t
        self.step_count += 1
        loss_after = self.regression_loss(k, v)
        return loss_before, loss_after

    def retrieve_context_vector(self, query_k: np.ndarray) -> np.ndarray:
        r"""Truy xuất vector giá trị dự đoán $\hat{v} = M_t k_q$."""
        k = self.normalize_l2(query_k)
        return self.M @ k


@dataclass
class MemoryEpisode:
    """Một tập hợp ký ức (Episode) gồm văn bản OCR và nhãn cấu trúc tương ứng."""

    episode_id: str
    ocr_text: str
    receipt_data: ReceiptData
    layout_text: str = ""
    source: str = "seed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "ocr_text": self.ocr_text,
            "receipt_data": self.receipt_data.model_dump(),
            "layout_text": self.layout_text,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEpisode:
        rec_data = data.get("receipt_data", {})
        if isinstance(rec_data, dict):
            receipt_obj = ReceiptData(**rec_data)
        elif isinstance(rec_data, ReceiptData):
            receipt_obj = rec_data
        else:
            receipt_obj = ReceiptData()

        return cls(
            episode_id=str(data.get("episode_id", "")),
            ocr_text=str(data.get("ocr_text", "")),
            receipt_data=receipt_obj,
            layout_text=str(data.get("layout_text", "")),
            source=str(data.get("source", "seed")),
        )


class ContinuumMemoryStore:
    """Kho lưu trữ Bộ nhớ Liên tục (CMS Store) kết hợp Ma trận Bộ nhớ Liên kết Delta Rule
    và Character 3-gram để truy xuất Few-shot khử nhiễu ngữ cảnh.
    """

    def __init__(
        self,
        memory_bank_path: Optional[Union[str, Path]] = None,
        samples_dir: Optional[Union[str, Path]] = None,
        auto_init: bool = True,
        memory_dim: int = 128,
        delta_eta: float = 0.5,
    ):
        self.memory_bank_path = (
            Path(memory_bank_path)
            if memory_bank_path
            else Path(__file__).resolve().parents[3] / "data" / "processed" / "cms_memory_bank.jsonl"
        )
        self.samples_dir = (
            Path(samples_dir)
            if samples_dir
            else Path(__file__).resolve().parents[3] / "data" / "samples"
        )
        self.memory_dim = memory_dim
        self.delta_memory = DeltaRuleMemoryMatrix(dim=memory_dim, eta=delta_eta)
        self.episodes: List[MemoryEpisode] = []
        self._ngram_cache: Dict[str, Counter] = {}
        self._key_vec_cache: Dict[str, np.ndarray] = {}
        self._val_vec_cache: Dict[str, np.ndarray] = {}

        if auto_init:
            self.load()

    def _register_episode_in_memory(self, ep: MemoryEpisode) -> None:
        """Đưa một episode vào cả bộ đệm n-gram và cập nhật Ma trận Bộ nhớ Delta Rule."""
        self._ngram_cache[ep.episode_id] = compute_char_ngrams(ep.ocr_text)
        k_vec = text_to_l2_normalized_vector(ep.ocr_text, dim=self.memory_dim)
        v_vec = receipt_data_to_target_vector(ep.receipt_data, dim=self.memory_dim)
        self._key_vec_cache[ep.episode_id] = k_vec
        self._val_vec_cache[ep.episode_id] = v_vec
        self.delta_memory.update(k_vec, v_vec)

    def load(self) -> None:
        """Nạp các episode từ file bộ nhớ bền vững và tái lập trạng thái Ma trận Delta Rule."""
        self.episodes = []
        self._ngram_cache = {}
        self._key_vec_cache = {}
        self._val_vec_cache = {}
        self.delta_memory = DeltaRuleMemoryMatrix(dim=self.memory_dim, eta=self.delta_memory.eta)

        if self.memory_bank_path.exists():
            with self.memory_bank_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        ep = MemoryEpisode.from_dict(json.loads(line_str))
                        self.episodes.append(ep)
                        self._register_episode_in_memory(ep)

        if not self.episodes and self.samples_dir.exists():
            self._load_from_samples_dir()
            self.persist()

        if not self.episodes:
            self._init_seed_episodes()
            self.persist()

    def _load_from_samples_dir(self) -> None:
        """Nạp các cặp .json và .txt trong samples_dir làm hạt giống."""
        json_files = sorted(self.samples_dir.glob("*.json"))
        for jf in json_files:
            if jf.name in ("splits.json", "manifest.json"):
                continue
            try:
                data_dict = json.loads(jf.read_text(encoding="utf-8"))
                stem = jf.stem
                txt_path = self.samples_dir / f"{stem}.txt"
                ocr_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
                receipt_obj = ReceiptData(**data_dict)
                ep = MemoryEpisode(
                    episode_id=stem,
                    ocr_text=ocr_text,
                    receipt_data=receipt_obj,
                    layout_text="",
                    source="samples",
                )
                self.episodes.append(ep)
                self._register_episode_in_memory(ep)
            except Exception:
                continue

    def _init_seed_episodes(self) -> None:
        """Khởi tạo 3 episode hạt giống cơ bản cho cửa hàng tiện lợi, siêu thị và cà phê."""
        seed_specs = [
            (
                "seed_circle_k",
                "SIEU THI TIEN LOI CIRCLE K\nDia chi: 123 Nguyen Thi Minh Khai, Q1\nMST: 0312456789\nNgay: 17/10/2026 08:30:15\nCa phe sua da 2 25.000 50.000\nTam tinh: 50.000\nTong thanh toan: 50.000\nHinh thuc thanh toan: Tien mat",
                ReceiptData(
                    seller_name="SIEU THI TIEN LOI CIRCLE K",
                    seller_address="123 Nguyen Thi Minh Khai, Q1",
                    tax_id="0312456789",
                    timestamp="17/10/2026 08:30:15",
                    items=[ReceiptItem(item_name="Ca phe sua da", quantity=2.0, unit_price=25000, total_price=50000)],
                    subtotal=50000,
                    total_cost=50000,
                    payment_method="Tien mat",
                ),
            ),
            (
                "seed_winmart",
                "SIÊU THỊ WINMART+ QUẬN 7\nĐịa chỉ: 458 Nguyễn Thị Thập, Q7\nMST: 0104918404\nNgày: 17/10/2026 18:15:00\nSữa tươi Vinamilk 2 34.000 68.000\nTạm tính: 68.000\nTổng thanh toán: 68.000\nHình thức thanh toán: Chuyển khoản",
                ReceiptData(
                    seller_name="SIÊU THỊ WINMART+ QUẬN 7",
                    seller_address="458 Nguyễn Thị Thập, Q7",
                    tax_id="0104918404",
                    timestamp="17/10/2026 18:15:00",
                    items=[ReceiptItem(item_name="Sữa tươi Vinamilk", quantity=2.0, unit_price=34000, total_price=68000)],
                    subtotal=68000,
                    total_cost=68000,
                    payment_method="Chuyển khoản",
                ),
            ),
            (
                "seed_highlands",
                "HIGHLANDS COFFEE\nĐịa chỉ: Số 2 Hải Triều, Q1\nMST: 0302088135\nNgày: 17/10/2026 12:45:22\nPhin Sữa Đá 1 39.000 39.000\nTạm tính: 39.000\nTổng thanh toán: 39.000\nHình thức thanh toán: MoMo",
                ReceiptData(
                    seller_name="HIGHLANDS COFFEE",
                    seller_address="Số 2 Hải Triều, Q1",
                    tax_id="0302088135",
                    timestamp="17/10/2026 12:45:22",
                    items=[ReceiptItem(item_name="Phin Sữa Đá", quantity=1.0, unit_price=39000, total_price=39000)],
                    subtotal=39000,
                    total_cost=39000,
                    payment_method="MoMo",
                ),
            ),
        ]
        for ep_id, text, rdata in seed_specs:
            ep = MemoryEpisode(
                episode_id=ep_id,
                ocr_text=text,
                receipt_data=rdata,
                source="seed",
            )
            self.episodes.append(ep)
            self._register_episode_in_memory(ep)

    def persist(self) -> None:
        """Ghi toàn bộ episode xuống file JSONL bền vững."""
        self.memory_bank_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_bank_path.open("w", encoding="utf-8") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")

    def add_verified_episode(
        self,
        ocr_text: str,
        receipt_data: ReceiptData,
        layout_text: str = "",
        episode_id: Optional[str] = None,
        source: str = "user_verified",
    ) -> MemoryEpisode:
        """Thêm một episode đã được kiểm chứng (C^(1)=1) và cập nhật Ma trận Bộ nhớ Delta Rule."""
        ep_id = episode_id or f"ep_{len(self.episodes) + 1:04d}"
        ep = MemoryEpisode(
            episode_id=ep_id,
            ocr_text=ocr_text,
            receipt_data=receipt_data,
            layout_text=layout_text,
            source=source,
        )
        self.episodes.append(ep)
        self._register_episode_in_memory(ep)

        self.memory_bank_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_bank_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")

        return ep

    def retrieve_similar_examples(
        self,
        query_ocr_text: str,
        k: int = 3,
        exclude_near_exact: bool = True,
    ) -> List[MemoryEpisode]:
        r"""Truy xuất $k$ mẫu ký ức tương đồng nhất bằng kết hợp:
        1. Độ tương đồng Cosine trên Character 3-gram.
        2. Điểm khớp hồi quy của Ma trận Bộ nhớ Liên kết Delta Rule:
           $\text{score}_{\text{Delta}}(ep) = -\frac{1}{2}\|M_t k_q - v_{ep}\|_2^2 + \langle M_t k_q, v_{ep} \rangle$.

        Loại trừ mẫu có similarity > 0.999 khi `exclude_near_exact=True` để chống data leakage.
        """
        if not self.episodes or k <= 0:
            return []

        query_ngram = compute_char_ngrams(query_ocr_text)
        query_k = text_to_l2_normalized_vector(query_ocr_text, dim=self.memory_dim)
        predicted_v = self.delta_memory.retrieve_context_vector(query_k)

        scored: List[Tuple[float, MemoryEpisode]] = []

        for ep in self.episodes:
            ep_ngram = self._ngram_cache.get(ep.episode_id)
            if ep_ngram is None:
                self._register_episode_in_memory(ep)
                ep_ngram = self._ngram_cache[ep.episode_id]

            sim_ngram = cosine_similarity_counters(query_ngram, ep_ngram)
            if exclude_near_exact and sim_ngram > 0.999:
                continue

            v_ep = self._val_vec_cache.get(ep.episode_id)
            if v_ep is not None:
                # Điểm khử nhiễu liên kết từ ma trận Delta Rule: tương quan giữa dự đoán M_t k_q và v_ep
                delta_assoc_score = float(np.dot(predicted_v, v_ep))
            else:
                delta_assoc_score = 0.0

            combined_score = 0.70 * sim_ngram + 0.30 * delta_assoc_score
            scored.append((combined_score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def export_for_lora_sft(
        self,
        output_path: Union[str, Path],
        system_prompt: str,
    ) -> Path:
        """Xuất dữ liệu dưới định dạng ChatML JSONL chuẩn bị cho huấn luyện thích nghi tần số thấp (QLoRA SFT)."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w", encoding="utf-8") as f:
            for ep in self.episodes:
                assistant_json = ep.receipt_data.model_dump_json(indent=2)
                user_content = ep.layout_text if ep.layout_text else ep.ocr_text
                sample_payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Văn bản OCR hóa đơn:\n```text\n{user_content}\n```"},
                        {"role": "assistant", "content": assistant_json},
                    ]
                }
                f.write(json.dumps(sample_payload, ensure_ascii=False) + "\n")

        return out

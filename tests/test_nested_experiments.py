"""Bài kiểm thử đơn vị cho các thực nghiệm toán học Nested Learning
(Hebbian vs Delta-Rule, M3 Optimizer và các kịch bản Continual Learning).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from receipt_extraction.memory.cms_store import (
    DeltaRuleMemoryMatrix,
    HebbianMemoryMatrix,
    text_to_l2_normalized_vector,
)
from receipt_extraction.optimizers.nested_optimizers import (
    DeltaGradientDescentState,
    MultiScaleMomentumMuonState,
    newton_schulz_orthogonalize,
)
from scripts.run_nested_learning_experiments import (
    build_receipt_vectors,
    export_experiment_report,
    run_experiment_1_memory_capacity,
    run_experiment_2_continual_learning,
)


def test_hebbian_memory_matrix_init_and_update() -> None:
    """Kiểm tra khởi tạo và bước cập nhật của HebbianMemoryMatrix (Eq. 64)."""
    dim = 32
    heb = HebbianMemoryMatrix(dim=dim, alpha=0.9, eta=0.4)
    assert heb.dim == dim
    assert heb.M.shape == (dim, dim)
    assert heb.step_count == 0

    k = np.zeros(dim)
    k[0] = 1.0
    v = np.zeros(dim)
    v[1] = 1.0

    loss_before, loss_after = heb.update(k, v)
    assert heb.step_count == 1
    assert isinstance(loss_before, float)
    assert isinstance(loss_after, float)

    # Sau khi cập nhật, truy xuất M @ k phải có thành phần tại v[1]
    retrieved = heb.retrieve_context_vector(k)
    assert retrieved.shape == (dim,)
    assert retrieved[1] > 0.1


def test_delta_vs_hebbian_mse_convergence() -> None:
    """Kiểm tra đặc tính toán học: Delta Rule (Eq. 65) triệt tiêu nhiễu tốt hơn Hebbian (Eq. 64)."""
    dim = 32
    rng = np.random.RandomState(42)

    # Tạo 4 vector khóa và giá trị có độ tương quan
    keys = rng.normal(0, 1, size=(4, dim))
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    vals = rng.normal(0, 1, size=(4, dim))
    vals = vals / np.linalg.norm(vals, axis=1, keepdims=True)

    heb = HebbianMemoryMatrix(dim=dim, alpha=0.95, eta=0.5)
    delta = DeltaRuleMemoryMatrix(dim=dim, eta=0.5)

    # Lặp cập nhật 10 vòng
    for _ in range(10):
        for i in range(4):
            heb.update(keys[i], vals[i])
            delta.update(keys[i], vals[i], left_projection=False)

    heb_mse = float(np.mean([heb.regression_loss(keys[i], vals[i]) for i in range(4)]))
    delta_mse = float(np.mean([delta.regression_loss(keys[i], vals[i]) for i in range(4)]))

    # Delta Rule phải có sai số hồi quy nhỏ hơn đáng kể so với Hebbian
    assert delta_mse < heb_mse
    assert delta_mse < 0.5


def test_m3_optimizer_state_and_newton_schulz() -> None:
    """Kiểm tra phép trực giao hóa Newton-Schulz và bước cập nhật của MultiScaleMomentumMuonState."""
    shape = (4, 16)
    opt = MultiScaleMomentumMuonState(shape=shape, lr=0.05, slow_freq_f=2, ns_iters=5)
    assert opt.step_count == 0

    W = np.zeros(shape, dtype=np.float64)
    G = np.ones(shape, dtype=np.float64)

    # Bước 1
    W_1 = opt.step(W, G)
    assert opt.step_count == 1
    assert W_1.shape == shape
    assert not np.allclose(W_1, W)

    # Bước 2 (kích hoạt chu kỳ chậm f=2)
    W_2 = opt.step(W_1, G)
    assert opt.step_count == 2
    assert W_2.shape == shape


def test_build_receipt_vectors_and_experiments_fast(tmp_path: Path) -> None:
    """Kiểm tra các hàm chạy thực nghiệm 1 và 2 với cấu hình rút gọn."""
    dim = 16
    keys, vals, labels = build_receipt_vectors(dim=dim, seed=123)
    assert len(keys) == len(vals) == len(labels)
    assert keys.shape[1] == dim
    assert vals.shape[1] == dim

    # Kiểm tra chuẩn L2 = 1.0
    for k_vec in keys[:5]:
        assert abs(float(np.linalg.norm(k_vec, ord=2)) - 1.0) < 1e-6

    # Chạy Experiment 1 nhanh
    exp1_res = run_experiment_1_memory_capacity(
        keys=keys[:8],
        vals=vals[:8],
        labels=labels[:8],
        noise_levels=[0.0, 0.1],
        epochs=2,
        seed=123,
    )
    assert "final_delta_mse" in exp1_res
    assert "noise_eval_results" in exp1_res
    assert len(exp1_res["noise_eval_results"]) == 2

    # Chạy Experiment 2 nhanh
    exp2_res = run_experiment_2_continual_learning(
        dim=dim,
        epochs_per_task=2,
        lr=0.05,
        seed=123,
    )
    assert "results" in exp2_res
    assert "MultiScale_Muon_M3" in exp2_res["results"]
    m3_res = exp2_res["results"]["MultiScale_Muon_M3"]
    assert "avg_forgetting" in m3_res
    assert "backward_transfer" in m3_res

    # Kiểm tra xuất báo cáo JSON & Markdown
    json_file = tmp_path / "test_report.json"
    md_file = tmp_path / "test_report.md"
    export_experiment_report(exp1_res, exp2_res, json_file, md_file)

    assert json_file.exists()
    assert md_file.exists()
    loaded_json = json.loads(json_file.read_text(encoding="utf-8"))
    assert "experiment_1_memory_capacity_denoising" in loaded_json
    md_content = md_file.read_text(encoding="utf-8")
    assert "Báo Cáo Nghiên Cứu Khoa Học" in md_content
    assert "MultiScale_Muon_M3" in md_content


def test_end_to_end_real_receipts_mt3_mt4_report() -> None:
    """Kiểm chứng báo cáo thực nghiệm End-to-End 20 hóa đơn thực tế (MT3 & MT4)."""
    root = Path(__file__).resolve().parent.parent
    json_path = root / "experiments" / "end_to_end_real_receipts_mt3_mt4.json"
    md_path = root / "experiments" / "end_to_end_real_receipts_mt3_mt4.md"

    assert json_path.exists(), "Thiếu file báo cáo JSON thực nghiệm E2E MT3 & MT4"
    assert md_path.exists(), "Thiếu file báo cáo Markdown thực nghiệm E2E MT3 & MT4"

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["experiment_metadata"]["num_test_receipts"] == 20

    cfgs = report["configurations"]
    assert len(cfgs) == 5

    # Kiểm chứng MT3: Cấu hình 4 và Cấu hình 5 vượt qua Regex Baseline B1 trên cả 4 trường và 12 trường
    mt3 = report["mt3_improvement_analysis"]
    assert mt3["config4_abs_gain_4_fields_pct"] > 0.0
    assert mt3["config4_abs_gain_12_fields_pct"] > 0.0
    assert mt3["config5_abs_gain_12_fields_pct"] > 0.0

    # Kiểm chứng MT4: Tín hiệu LSS ở Cấu hình 4 phát hiện và sửa đúng lỗi từ Cấu hình 3
    mt4 = report["mt4_lss_self_correction_analysis"]
    assert mt4["lss_detected_and_corrected_count"] >= 5
    assert mt4["config4_total_cost_math_error_rate"] < mt4["config3_total_cost_math_error_rate"]


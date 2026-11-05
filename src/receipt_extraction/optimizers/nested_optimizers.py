r"""Cài đặt thuật toán tối ưu hóa từ bài báo "Nested Learning: The Illusion of Deep Learning Architecture"
(Behrouz et al., NeurIPS 2025):

1. **Phép lặp trực giao hóa đa thức Newton-Schulz (Phương trình 44 & Algorithm 1):**
   $$O_{i+1} = O_i - \zeta \left( O_i - g_t + 2 O_i (O_i^\top O_i - I) \right)$$
   (với tiền chuẩn hóa Frobenius $O_0 = G / \|G\|_F$ để đảm bảo bán kính phổ hội tụ về ma trận trực giao $O^\top O \approx I$).

2. **Thuật toán Delta Gradient Descent (DGD — Phương trình 57):**
   $$W_{t+1} = W_t (I - \eta'_t x_t x_t^\top) - \eta'_t \nabla_{y_t} \mathcal{L}(W_t; x_t) \otimes x_t$$
   trong đó $\eta'_t = \frac{\eta_t}{1 + \eta_t}$ khi $\|x_t\|_2 = 1$.

3. **Thuật toán Multi-scale Momentum Muon (M3 — Algorithm 1, Mục 7.2):**
   - Bộ nhớ tần số cao (Fast Momentum $\mathcal{C}^{(1)} = 1$):
     $$M_t^{(1)} = \beta_1 M_{t-1}^{(1)} + (1 - \beta_1) G_t$$
   - Bộ nhớ tần số thấp (Slow Momentum $\mathcal{C}^{(2)} = f$, cập nhật mỗi $f$ bước):
     $$M_t^{(2)} = \begin{cases} \beta_2 M_{t-1}^{(2)} + (1 - \beta_2) \bar{G}_t & \text{nếu } t \bmod f = 0 \\ M_{t-1}^{(2)} & \text{ngược lại} \end{cases}$$
   - Trực giao hóa đa thang thông qua Newton-Schulz:
     $$O_t^{(1)} = \text{NewtonSchulz}(M_t^{(1)}), \quad O_t^{(2)} = \text{NewtonSchulz}(M_t^{(2)})$$
   - Cập nhật tham số kết hợp hai thang tần số:
     $$W_t = W_{t-1} - \eta \left( O_t^{(1)} + \alpha O_t^{(2)} \right) / \sqrt{V_t + \epsilon}$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def newton_schulz_orthogonalize(
    grad_matrix: np.ndarray,
    num_iters: int = 5,
    zeta: float = 0.25,
    eps: float = 1e-8,
) -> np.ndarray:
    r"""Phép lặp đa thức Newton-Schulz trực giao hóa ma trận gradient/momentum
    theo Phương trình 44 và Algorithm 1 của bài báo *Nested Learning (NeurIPS 2025)*:

    $$O_{i+1} = O_i - \zeta \left( O_i - \hat{G}_t + 2 O_i (O_i^\top O_i - I) \right)$$

    Trong đó $\hat{G}_t = G_t / (\|G_t\|_F + \epsilon)$ là ma trận đã chuẩn hóa chuẩn Frobenius
    để đảm bảo toàn bộ trị kỳ dị ban đầu nằm trong miền hội tụ $(0, 1]$.
    Nếu số hàng nhỏ hơn số cột ($m < n$), phép lặp được thực hiện trên ma trận chuyển vị rồi chuyển vị lại
    để $O_i O_i^\top \approx I_m$.
    """
    G = np.asarray(grad_matrix, dtype=np.float64)
    orig_ndim = G.ndim
    if orig_ndim == 1:
        norm = float(np.linalg.norm(G, ord=2))
        return G / (norm + eps)

    transposed = False
    if G.shape[0] < G.shape[1]:
        G = G.T
        transposed = True

    frob_norm = float(np.linalg.norm(G, ord="fro"))
    if frob_norm < eps:
        return np.zeros_like(grad_matrix, dtype=np.float64)

    g_hat = G / (frob_norm + eps)
    O_i = g_hat.copy()
    cols = O_i.shape[1]
    eye = np.eye(cols, dtype=np.float64)

    for _ in range(num_iters):
        ot_o = O_i.T @ O_i
        # Phương trình 44 của bài báo: O_{i+1} = O_i - zeta * (O_i - g_t + 2 * O_i @ (O_i^T O_i - I))
        O_i = O_i - zeta * (O_i - g_hat + 2.0 * (O_i @ (ot_o - eye)))

    return O_i.T if transposed else O_i


class DeltaGradientDescentState:
    r"""Trạng thái tối ưu hóa Delta Gradient Descent (DGD — Phương trình 57, Behrouz et al., 2025):

    $$W_{t+1} = W_t (I - \eta'_t x_t x_t^\top) - \eta'_t \nabla_{y_t} \mathcal{L}(W_t; x_t) \otimes x_t$$
    với $\eta'_t = \frac{\eta_t}{1 + \eta_t}$.
    """

    def __init__(self, shape: Tuple[int, int], lr: float = 0.1):
        self.W = np.zeros(shape, dtype=np.float64)
        self.lr = float(lr)
        self.step_count: int = 0

    def step(self, x_t: np.ndarray, grad_y_t: np.ndarray) -> np.ndarray:
        """Thực hiện 1 bước cập nhật DGD theo Phương trình 57."""
        x = np.asarray(x_t, dtype=np.float64).reshape(-1)
        x_norm = float(np.linalg.norm(x, ord=2))
        if x_norm > 1e-12:
            x = x / x_norm

        g_y = np.asarray(grad_y_t, dtype=np.float64).reshape(-1)
        eta_prime = self.lr / (1.0 + self.lr)
        eye = np.eye(x.shape[0], dtype=np.float64)

        self.W = self.W @ (eye - eta_prime * np.outer(x, x)) - eta_prime * np.outer(g_y, x)
        self.step_count += 1
        return self.W


class MultiScaleMomentumMuonState:
    r"""Bộ tối ưu hóa Multi-scale Momentum Muon (M3 — Algorithm 1, Mục 7.2 của bài báo Nested Learning).

    Kết hợp:
    - Bộ nhớ động lượng tần số cao $M_t^{(1)}$ (cập nhật mỗi bước $t$, chu kỳ $C^{(1)} = 1$).
    - Bộ nhớ động lượng tần số thấp $M_t^{(2)}$ (cập nhật mỗi $f$ bước, chu kỳ $C^{(2)} = f$, tích lũy gradient trung bình $\bar{G}$).
    - Bộ ước lượng phương sai bậc hai $V_t$ (giống Adam) và phép trực giao hóa Newton-Schulz trên cả hai thang tần số.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        beta3: float = 0.95,
        alpha: float = 0.5,
        slow_freq_f: int = 4,
        ns_iters: int = 5,
        zeta: float = 0.25,
        eps: float = 1e-8,
    ):
        self.shape = shape
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.beta3 = float(beta3)
        self.alpha = float(alpha)
        self.slow_freq_f = max(1, int(slow_freq_f))
        self.ns_iters = int(ns_iters)
        self.zeta = float(zeta)
        self.eps = float(eps)

        self.M1 = np.zeros(shape, dtype=np.float64)  # Fast momentum (C^(1) = 1)
        self.M2 = np.zeros(shape, dtype=np.float64)  # Slow momentum (C^(2) = f)
        self.O2 = np.zeros(shape, dtype=np.float64)  # Cached orthogonal slow momentum
        self.V = np.zeros(shape, dtype=np.float64)   # Second moment
        self._grad_chunk_accum = np.zeros(shape, dtype=np.float64)
        self._chunk_count: int = 0
        self.step_count: int = 0

    def step(self, weights: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Thực hiện 1 bước cập nhật trọng số theo đúng Algorithm 1 (Multi-scale Momentum Muon - M3)."""
        self.step_count += 1
        G_t = np.asarray(grad, dtype=np.float64)
        W = np.asarray(weights, dtype=np.float64).copy()

        # Tích lũy gradient cho chu kỳ chậm f
        self._grad_chunk_accum += G_t
        self._chunk_count += 1

        # 1. Cập nhật bộ nhớ chậm M^(2) nếu đến chu kỳ t % f == 0 (hoặc bước đầu tiên)
        if self.step_count == 1 or (self.step_count % self.slow_freq_f == 0):
            avg_slow_grad = self._grad_chunk_accum / float(max(1, self._chunk_count))
            self.M2 = self.beta3 * self.M2 + (1.0 - self.beta3) * avg_slow_grad
            self.O2 = newton_schulz_orthogonalize(self.M2, num_iters=self.ns_iters, zeta=self.zeta, eps=self.eps)
            self._grad_chunk_accum.fill(0.0)
            self._chunk_count = 0

        # 2. Cập nhật bộ nhớ nhanh M^(1) và phương sai bậc hai V_t ở mọi bước (C^(1) = 1)
        self.M1 = self.beta1 * self.M1 + (1.0 - self.beta1) * G_t
        self.V = self.beta2 * self.V + (1.0 - self.beta2) * (G_t * G_t)

        O1 = newton_schulz_orthogonalize(self.M1, num_iters=self.ns_iters, zeta=self.zeta, eps=self.eps)

        # Hiệu chỉnh độ lệch bước đầu cho V_t
        v_hat = self.V / (1.0 - (self.beta2 ** self.step_count))
        denom = np.sqrt(v_hat) + self.eps

        # 3. Bước cập nhật M3: W_t = W_{t-1} - eta * (O_t^(1) + alpha * O_t^(2)) / (sqrt(V_t) + eps)
        W_next = W - self.lr * (O1 + self.alpha * self.O2) / denom
        return W_next


def build_pytorch_m3_optimizer_class() -> Any:
    """Trả về lớp `torch.optim.Optimizer` cài đặt Multi-scale Momentum Muon (M3) khi môi trường có PyTorch."""
    import torch

    class NestedM3Optimizer(torch.optim.Optimizer):
        """Bộ tối ưu hóa PyTorch thực thi Algorithm 1 (Multi-scale Momentum Muon - M3, NeurIPS 2025)."""

        def __init__(
            self,
            params: Any,
            lr: float = 2e-4,
            betas: Tuple[float, float, float] = (0.9, 0.99, 0.95),
            alpha: float = 0.5,
            slow_freq_f: int = 4,
            ns_iters: int = 5,
            zeta: float = 0.25,
            eps: float = 1e-8,
        ):
            defaults = dict(
                lr=lr,
                betas=betas,
                alpha=alpha,
                slow_freq_f=slow_freq_f,
                ns_iters=ns_iters,
                zeta=zeta,
                eps=eps,
            )
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure: Optional[Any] = None) -> Optional[Any]:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2, beta3 = group["betas"]
                alpha = group["alpha"]
                freq_f = group["slow_freq_f"]
                ns_iters = group["ns_iters"]
                zeta = group["zeta"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["m1"] = torch.zeros_like(p)
                        state["m2"] = torch.zeros_like(p)
                        state["o2"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                        state["grad_accum"] = torch.zeros_like(p)
                        state["accum_cnt"] = 0

                    state["step"] += 1
                    t = state["step"]
                    state["grad_accum"].add_(grad)
                    state["accum_cnt"] += 1

                    # Cập nhật bộ nhớ nhanh M1 và V
                    state["m1"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    state["v"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    # Cập nhật bộ nhớ chậm M2 mỗi chu kỳ f
                    if t == 1 or (t % freq_f == 0):
                        avg_g = state["grad_accum"] / float(max(1, state["accum_cnt"]))
                        state["m2"].mul_(beta3).add_(avg_g, alpha=1.0 - beta3)
                        if p.ndim >= 2:
                            m2_np = state["m2"].detach().float().cpu().numpy().reshape(p.shape[0], -1)
                            o2_np = newton_schulz_orthogonalize(m2_np, num_iters=ns_iters, zeta=zeta, eps=eps)
                            state["o2"].copy_(torch.from_numpy(o2_np).to(p.device, dtype=p.dtype).view_as(p))
                        else:
                            state["o2"].copy_(state["m2"])
                        state["grad_accum"].zero_()
                        state["accum_cnt"] = 0

                    if p.ndim >= 2:
                        m1_np = state["m1"].detach().float().cpu().numpy().reshape(p.shape[0], -1)
                        o1_np = newton_schulz_orthogonalize(m1_np, num_iters=ns_iters, zeta=zeta, eps=eps)
                        o1 = torch.from_numpy(o1_np).to(p.device, dtype=p.dtype).view_as(p)
                    else:
                        o1 = state["m1"]

                    v_hat = state["v"] / (1.0 - (beta2 ** t))
                    denom = v_hat.sqrt().add_(eps)
                    combined_dir = (o1 + alpha * state["o2"]) / denom
                    p.add_(combined_dir, alpha=-lr)

            return loss

    return NestedM3Optimizer

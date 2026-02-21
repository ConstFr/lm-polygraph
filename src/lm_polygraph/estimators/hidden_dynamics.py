import numpy as np
import logging
from typing import Dict, Optional, Tuple

from .estimator import Estimator

log = logging.getLogger(__name__)


def _try_savgol(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """
    Savitzky–Golay smoothing if scipy is available, else a simple moving average fallback.
    """
    if window_length <= 1:
        return y.copy()

    try:
        from scipy.signal import savgol_filter  # type: ignore
        if len(y) < window_length:
            return y.copy()
        return savgol_filter(y, window_length=window_length, polyorder=polyorder, mode="interp")
    except Exception:
        if len(y) < window_length:
            return y.copy()
        kernel = np.ones(window_length, dtype=np.float32) / float(window_length)
        return np.convolve(y, kernel, mode="same")


class HiddenDynamics(Estimator):
    """
    Hidden Dynamics UQ (Sampling-free uncertainty via hidden state dynamics).

    Returns:
      - sequence-level score (shape [1]) when aggregation is enabled.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        sg_window_halfwidth: int = 5,
        sg_polyorder: int = 3,
        delta_layer: int = 10,
        curvature_eps: float = 1e-6,
        layer_idx = -1,
        head_idx = -1,
        use_last_step_attention: bool = True,
    ):
        dependencies = ["all_hidden_states", "all_attentions", "greedy_tokens", "prompt_len"]
        super().__init__(dependencies, "sequence")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.w = int(sg_window_halfwidth)
        self.polyorder = int(sg_polyorder)
        self.delta = int(delta_layer)
        self.eps = float(curvature_eps)
        self.layer_idx = int(layer_idx)
        self.head_idx = int(head_idx)
        self.use_last_step_attention = use_last_step_attention

    def __str__(self) -> str:
        return f"HiddenDynamics_layer_{self.layer_idx}_head_{self.head_idx}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        hs = stats["all_hidden_states"]  # [L+1, B, T, D]
        att = stats["all_attentions"]   # [L, B, H, T, T]
        prompt_len = int(stats["prompt_len"])

        log.info(f"HiddenDynamics: all_hidden_states shape {hs.shape}, all_attentions shape {att.shape}, prompt_len {prompt_len}")
        log.info(f"{self.layer_idx=}, {self.head_idx=}")

        if hs.ndim != 4:
            raise ValueError(f"Expected all_hidden_states shape [L+1,B,T,D], got {hs.shape}")
        if att.ndim != 5:
            raise ValueError(f"Expected all_attentions shape [L,B,H,T,T], got {att.shape}")

        Lp1, B, T, D = hs.shape
        L = Lp1 - 1

        if B != 1:
            raise NotImplementedError("Batch size > 1 not supported")

        # Answer span indices
        ans_start = min(max(prompt_len, 0), T)

        # Drop embedding layer: hs_layers [L, T, D]
        hs_layers = hs[1:, 0, :, :]  # [L, T, D]
        h_final = hs_layers[-1]  # [T, D]

        # LSR score s[l, t] = cos(h_l(t), h_final(t)) -> [L, T]
        # cosine similarity
        num = np.sum(hs_layers * h_final[None, :, :], axis=-1)  # [L, T]
        den = (np.linalg.norm(hs_layers, axis=-1) * np.linalg.norm(h_final, axis=-1)[None, :])  # [L, T]
        s_scores = num / np.clip(den, 1e-12, None)

        # Token UQ only for answer tokens
        token_uq = np.zeros((T - ans_start,), dtype=np.float32)

        for i, t in enumerate(range(ans_start, T)):
            traj = s_scores[:, t].astype(np.float64)  # [L]
            tau = self._compute_tau(traj)  # 1-based

            sss = self._sss(traj, tau)
            ccs = self._ccs(traj, tau)

            token_uq[i] = self.alpha * sss + self.beta * ccs

        if self.layer_idx == -2:
            # If no attention head is provided, return mean over answer tokens
            return np.array([token_uq.mean()])
        
        useq = self._sequence_uq_from_attention(
            token_uq=token_uq,
            attentions=att,
            ans_start=ans_start,
        )
        return np.array([useq])

    def _compute_tau(self, s_traj) -> int:
        L = int(s_traj.shape[0])

        win = 2 * self.w + 1
        s_hat = _try_savgol(s_traj, window_length=win, polyorder=self.polyorder)

        s1 = np.zeros_like(s_hat)
        s2 = np.zeros_like(s_hat)
        s1[:-1] = s_hat[1:] - s_hat[:-1]
        s2[:-2] = s1[1:-1] - s1[:-2]

        kappa = np.abs(s2) / np.power(1.0 + (s1 * s1) + self.eps, 1.5)

        start_idx = max(self.delta - 1, 0)
        if start_idx >= L:
            start_idx = 0

        tau_idx = np.argmax(kappa[start_idx:]) + start_idx  # 0-based
        return tau_idx + 1  # 1-based

    def _sss(self, s_traj, tau_1based) -> float:
        tau = max(int(tau_1based), 1)
        pre = s_traj[:tau]
        mu = float(np.mean(pre))
        var = float(np.mean((pre - mu) ** 2))
        return float(1.0 - var)

    def _ccs(self, s_traj, tau_1based) -> float:
        L = int(s_traj.shape[0])
        tau = min(max(int(tau_1based), 1), L)
        if tau >= L:
            return 0.0

        diffs = s_traj[tau:] - s_traj[tau - 1 : L - 1]
        return float(np.mean(diffs))

    def _sequence_uq_from_attention(
        self,
        token_uq,  # [T_answer]
        attentions,  # [L, B, H, T, T]
        ans_start,
    ) -> float:
        if self.layer_idx == -2:
            return float(token_uq.mean())
        A = attentions[self.layer_idx, 0, self.head_idx]  # [T, T]

        if self.use_last_step_attention:
            attn_vec = A[-1, :]  # [T]
        else:
            attn_vec = A[ans_start:, :].mean(axis=0)  # [T]

        attn_ans = attn_vec[ans_start:]  # [T_answer]

        w = np.exp(attn_ans - np.max(attn_ans))
        w = w / np.clip(w.sum(), 1e-12, None)

        return float(np.sum(w * token_uq))

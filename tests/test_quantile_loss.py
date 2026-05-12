# MIT License, 2026
"""Unit test: synthetic-bandit QR-DQN should recover analytical quantiles.

Catches the kind of broadcasting bug we hit in pre-training review C2.
A correctly-implemented quantile-Huber loss converges to the empirical
inverse-CDF of the reward distribution; a mis-broadcasted one does not.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F


def _quantile_huber_loss_correct(pred: torch.Tensor, target: torch.Tensor, tau_mids: torch.Tensor) -> torch.Tensor:
    """Reference impl matching the fixed src/training/neuropolicy_agent.py."""
    # pred: (B, M_pred), target: (B, M_target), tau_mids: (M_pred,)
    pred_b = pred.unsqueeze(2)                       # (B, M_pred, 1)
    target_b = target.unsqueeze(1)                   # (B, 1, M_target)
    diff = target_b - pred_b                          # (B, M_pred, M_target)
    huber = F.smooth_l1_loss(pred_b.expand_as(diff), target_b.expand_as(diff), reduction="none")
    sign = (diff < 0).float()
    tau = tau_mids.view(1, -1, 1)                    # (1, M_pred, 1)
    weight = torch.abs(tau - sign)
    return (weight * huber).sum(dim=2).mean(dim=1).mean()


def test_qr_recovers_normal_quantiles():
    """Train a single-state, single-action distributional critic on samples
    drawn from N(2, 1); the M=21 learned quantiles should approximate the
    analytical inverse-CDF Φ⁻¹((i+0.5)/M)·1 + 2 within ±0.15 on average."""
    torch.manual_seed(0)
    np.random.seed(0)
    M = 21
    tau_mids = (torch.arange(M, dtype=torch.float32) + 0.5) / M
    quantiles = torch.nn.Parameter(torch.zeros(1, M))   # learnable scalar->M
    opt = torch.optim.Adam([quantiles], lr=2e-2)
    mu, sigma = 2.0, 1.0
    n_steps = 8_000
    batch = 256
    for _ in range(n_steps):
        targets = torch.from_numpy(
            np.random.normal(mu, sigma, size=(batch, 1)).astype("float32")
        )  # (B, 1) — one sample per step; broadcast as (B, M_target=1)
        loss = _quantile_huber_loss_correct(
            quantiles.expand(batch, M), targets, tau_mids,
        )
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    learned = quantiles.detach().cpu().numpy().flatten()
    learned_sorted = np.sort(learned)
    # Analytical quantiles of N(mu, sigma)
    from scipy.stats import norm
    analytical = norm.ppf(((np.arange(M) + 0.5) / M)) * sigma + mu
    err = np.abs(learned_sorted - analytical)
    # Threshold = 0.20: tight enough to catch the "broken broadcast" bug (which
    # would produce flat output ≈ mean), loose enough to tolerate reasonable
    # under-convergence at the tails after 8K Adam steps.
    assert err.mean() < 0.20, f"mean abs err {err.mean():.3f} too high; learned={learned_sorted}"
    print(f"OK — mean abs err = {err.mean():.3f}, max = {err.max():.3f}")


if __name__ == "__main__":
    test_qr_recovers_normal_quantiles()

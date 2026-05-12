# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""Policy / critic networks for NeuroPolicy.

v1 (this file):
  * Scalar discrete Q-network. Action selection by argmax (or softmax
    sampling at temperature τ for stochastic evaluation).
  * Distributional QR-style critic outputting M quantiles per action.

The agent that combines these (NeuroPolicyAgent) lives in
src/training/neuropolicy_agent.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3,
                 activation=nn.ReLU, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarQNet(nn.Module):
    """Q(s, a) for discrete actions. Returns (B, n_actions)."""
    def __init__(self, state_dim: int, n_actions: int,
                 hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_actions = n_actions
        self.body = MLP(state_dim, n_actions, hidden=hidden, depth=depth, dropout=dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.body(s)


class QuantileQNet(nn.Module):
    """Distributional Z(s, a) returning (B, n_actions, M) quantile values.

    Used for CVaR-CQL (v2).
    """
    def __init__(self, state_dim: int, n_actions: int, n_quantiles: int = 31,
                 hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.body = MLP(state_dim, n_actions * n_quantiles,
                        hidden=hidden, depth=depth, dropout=dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.body(s)
        return z.view(-1, self.n_actions, self.n_quantiles)

    @staticmethod
    def cvar(z: torch.Tensor, alpha: float) -> torch.Tensor:
        """CVaR_alpha aggregator over quantile dim. z: (..., M) -> (...,).

        Lower-tail mean over the lowest ceil(alpha*M) quantiles.
        alpha=1.0 returns the simple mean (= scalar Q via mid-quantile estimate).
        """
        if alpha >= 1.0 - 1e-6:
            return z.mean(dim=-1)
        m = z.shape[-1]
        k = max(1, int(round(alpha * m)))
        # Quantiles are unordered; sort then take lower-k mean.
        z_sorted, _ = torch.sort(z, dim=-1)
        return z_sorted[..., :k].mean(dim=-1)

# MIT License - Bryan Cheng, 2026
# Part of NeuroPolicy / ieeeICIST
"""NeuroPolicy agent: discrete CQL with optional CVaR-distributional critic
and CMDP safety constraint via Lagrangian dual ascent.

References:
  * Kumar et al. 2020, "Conservative Q-Learning for Offline RL" (NeurIPS 2020).
  * Dabney et al. 2018, "Distributional RL with Quantile Regression" (AAAI).
  * Achiam et al. 2017, "Constrained Policy Optimization" (CPO).
  * RESEARCH_PLAN.md §3.2.3.

The agent supports two modes:
  - "scalar"        : ScalarQNet, standard discrete CQL.
  - "distributional": QuantileQNet, CVaR-CQL.

CMDP penalty (always applied; pass cmdp_eps=None to disable):
  L_cmdp = β · max(0, E_dataset[1{wrong commit}] - ε) where β is a
  Lagrange multiplier updated by dual ascent.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.policy_critic import QuantileQNet, ScalarQNet
from src.training.replay_buffer import OfflineBuffer
from src.utils.config import TRAIN_CFG, TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainStepStats:
    td_loss: float
    cql_loss: float
    cmdp_loss: float
    total_loss: float
    q_data_mean: float
    q_logsumexp_mean: float
    cmdp_lambda: float
    grad_norm: float

    def asdict(self) -> dict:
        return {k: float(getattr(self, k)) for k in vars(self)}


@dataclass
class NeuroPolicyAgent:
    state_dim: int
    n_actions: int
    n_classes: int
    cfg: TrainConfig = field(default_factory=lambda: TRAIN_CFG)
    mode: Literal["scalar", "distributional"] = "scalar"
    cmdp_eps: float | None = None
    gamma: float = 0.99
    polyak_tau: float = 0.005
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    hidden: int = 256
    depth: int = 3
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.mode == "scalar":
            self.q = ScalarQNet(self.state_dim, self.n_actions, hidden=self.hidden, depth=self.depth).to(self.device)
            self.q_target = deepcopy(self.q).to(self.device)
        elif self.mode == "distributional":
            self.q = QuantileQNet(self.state_dim, self.n_actions,
                                  n_quantiles=self.cfg.n_quantiles,
                                  hidden=self.hidden, depth=self.depth).to(self.device)
            self.q_target = deepcopy(self.q).to(self.device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.opt = torch.optim.AdamW(
            self.q.parameters(), lr=self.cfg.critic_lr, weight_decay=self.cfg.weight_decay,
        )

        # CMDP Lagrange multiplier (log-parametrized so it stays >= 0)
        self._log_lambda = torch.tensor(np.log(0.1), dtype=torch.float32,
                                         device=self.device, requires_grad=True)
        self._cmdp_opt = torch.optim.Adam([self._log_lambda], lr=self.cfg.cmdp_lr)

        # Quantile midpoints for distributional critic
        if self.mode == "distributional":
            mids = (torch.arange(self.cfg.n_quantiles, dtype=torch.float32, device=self.device) + 0.5) / self.cfg.n_quantiles
            self._tau_mids = mids  # (M,)

        self.rng = np.random.default_rng(self.seed)

    # ----- Q-aggregation helpers -----
    def _q_values(self, states: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Returns (B, n_actions) scalar Q estimates.

        For distributional critic: aggregator is mean over quantiles when
        target=True (Bellman target), but CVaR_alpha when computing the
        action-selection greedy values used in policy improvement. We use
        `cvar` only at policy-eval time; the Bellman target uses mean for
        consistency with QR-DQN.
        """
        net = self.q_target if target else self.q
        if self.mode == "scalar":
            return net(states)
        z = net(states)  # (B, A, M)
        return z.mean(dim=-1)

    def _z_values(self, states: torch.Tensor, target: bool = False) -> torch.Tensor:
        """(B, n_actions, n_quantiles) — distributional only."""
        if self.mode != "distributional":
            raise RuntimeError("Z values only available in distributional mode.")
        net = self.q_target if target else self.q
        return net(states)

    # ----- Action selection -----
    @torch.no_grad()
    def select_action(self, state: np.ndarray | torch.Tensor,
                      stochastic_temperature: float | None = None,
                      cvar_alpha: float | None = None) -> int:
        if isinstance(state, np.ndarray):
            s = torch.from_numpy(state).to(self.device).float().unsqueeze(0)
        else:
            s = state.to(self.device).float()
            if s.ndim == 1: s = s.unsqueeze(0)
        if self.mode == "distributional" and (cvar_alpha is not None and cvar_alpha < 1.0):
            z = self._z_values(s)  # (1, A, M)
            q = QuantileQNet.cvar(z, alpha=cvar_alpha)  # (1, A)
        else:
            q = self._q_values(s)
        if stochastic_temperature is None or stochastic_temperature <= 0:
            return int(q.argmax(dim=1).item())
        probs = F.softmax(q / stochastic_temperature, dim=1).cpu().numpy().flatten()
        return int(self.rng.choice(self.n_actions, p=probs))

    @torch.no_grad()
    def action_probs(self, state: np.ndarray, stochastic_temperature: float = 1.0,
                     cvar_alpha: float | None = None) -> np.ndarray:
        s = torch.from_numpy(state).to(self.device).float().unsqueeze(0)
        if self.mode == "distributional" and (cvar_alpha is not None and cvar_alpha < 1.0):
            z = self._z_values(s)
            q = QuantileQNet.cvar(z, alpha=cvar_alpha)
        else:
            q = self._q_values(s)
        return F.softmax(q / stochastic_temperature, dim=1).cpu().numpy().flatten()

    # ----- Bellman target -----
    def _bellman_target(self, batch: dict) -> torch.Tensor:
        sn, r, d = batch["sn"], batch["r"], batch["d"]
        with torch.no_grad():
            if self.mode == "scalar":
                q_next = self._q_values(sn, target=True)         # (B, A)
                q_next_max, _ = q_next.max(dim=1)                  # (B,)
                target = r + self.gamma * (1.0 - d) * q_next_max
                return target
            # Distributional: select action by mean-Q (consistent with QR-DQN), bootstrap quantile dist.
            z_next = self._z_values(sn, target=True)               # (B, A, M)
            q_next = z_next.mean(dim=-1)                           # (B, A)
            a_next = q_next.argmax(dim=1)                          # (B,)
            z_next_a = z_next[torch.arange(sn.shape[0], device=sn.device), a_next]  # (B, M)
            target = r.unsqueeze(1) + self.gamma * (1.0 - d).unsqueeze(1) * z_next_a
            return target  # (B, M)

    # ----- Update -----
    def update(self, batch: dict) -> TrainStepStats:
        s, a = batch["s"], batch["a"]
        wrong = batch["wrong"]
        target = self._bellman_target(batch)

        if self.mode == "scalar":
            q = self.q(s)                                          # (B, A)
            q_a = q.gather(1, a.unsqueeze(1)).squeeze(1)            # (B,)
            td_loss = F.smooth_l1_loss(q_a, target)
            # CQL conservative penalty
            logsumexp = torch.logsumexp(q, dim=1)
            cql_term = (logsumexp - q_a)
            cql_loss = self.cfg.cql_alpha * cql_term.mean()
            q_data_mean = q_a.mean()
            q_lse_mean = logsumexp.mean()
        else:
            z = self.q(s)                                          # (B, A, M)
            z_a = z[torch.arange(s.shape[0], device=s.device), a]  # (B, M_pred)
            # Quantile Huber regression. tau_i is the i-th prediction's quantile,
            # so it must broadcast over the prediction axis (1, M_pred, 1) — NOT
            # over the target axis. Earlier code had unsqueeze(0).unsqueeze(0) =
            # (1, 1, M) which mis-aligned tau with targets; corrected here.
            tau = self._tau_mids.view(1, -1, 1)                    # (1, M_pred, 1)
            target_b = target.unsqueeze(1)                         # (B, 1, M_target)
            pred_b = z_a.unsqueeze(2)                              # (B, M_pred, 1)
            diff = target_b - pred_b                               # (B, M_pred, M_target)
            huber = F.smooth_l1_loss(pred_b.expand_as(diff), target_b.expand_as(diff), reduction="none")
            sign = (diff < 0).float()
            quantile_weight = torch.abs(tau - sign)                # (B, M_pred, M_target)
            td_loss = (quantile_weight * huber).sum(dim=2).mean(dim=1).mean()
            # CQL on mean-Q
            q = z.mean(dim=-1)
            q_a_mean = q.gather(1, a.unsqueeze(1)).squeeze(1)
            logsumexp = torch.logsumexp(q, dim=1)
            cql_term = (logsumexp - q_a_mean)
            cql_loss = self.cfg.cql_alpha * cql_term.mean()
            q_data_mean = q_a_mean.mean()
            q_lse_mean = logsumexp.mean()

        # CMDP: encourage policy whose argmax actions don't include wrong commits.
        # Soft proxy: if data action is a wrong commit, push down its Q so policy avoids picking it.
        # The Lagrangian is over the empirical wrong-commit rate.
        if self.cmdp_eps is not None:
            lam = self._log_lambda.exp()
            # Pull Q of (state, wrong-action) lower:
            cmdp_loss = lam * (wrong * (q_a if self.mode == "scalar" else q_a_mean)).mean()
        else:
            cmdp_loss = torch.tensor(0.0, device=self.device)

        loss = td_loss + cql_loss + cmdp_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        # Dual ascent on lambda (run after critic step). Maximize the Lagrangian
        # w.r.t. lambda ⇒ gradient of -lambda * (E[wrong|policy] - eps).
        if self.cmdp_eps is not None:
            self._cmdp_opt.zero_grad(set_to_none=True)
            # Approximate "policy wrong rate" by Q-weighted probability of wrong commits at sampled states.
            # We treat actions 0..n_classes-1 as commits; the env "wrongness" depends on label which we
            # don't have at general states. We instead use the dataset's empirical wrong rate as proxy.
            wrong_rate = wrong.mean().detach()
            dual_loss = -(self._log_lambda.exp() * (wrong_rate - self.cmdp_eps))
            dual_loss.backward()
            self._cmdp_opt.step()

        # Polyak update
        with torch.no_grad():
            for tp, p in zip(self.q_target.parameters(), self.q.parameters()):
                tp.data.mul_(1.0 - self.polyak_tau).add_(p.data, alpha=self.polyak_tau)

        return TrainStepStats(
            td_loss=float(td_loss.item()),
            cql_loss=float(cql_loss.item()),
            cmdp_loss=float(cmdp_loss.item()),
            total_loss=float(loss.item()),
            q_data_mean=float(q_data_mean.item()),
            q_logsumexp_mean=float(q_lse_mean.item()),
            cmdp_lambda=float(self._log_lambda.exp().item()),
            grad_norm=float(gn.item() if isinstance(gn, torch.Tensor) else gn),
        )

    # ----- Train loop -----
    def train(self, buffer: OfflineBuffer, n_steps: int, batch_size: int,
              log_every: int = 1000) -> list[TrainStepStats]:
        history: list[TrainStepStats] = []
        for step in range(n_steps):
            batch = buffer.sample(batch_size, self.device, self.rng)
            stats = self.update(batch)
            if (step + 1) % log_every == 0 or step == 0:
                logger.info(
                    "step=%6d  td=%.4f cql=%.4f cmdp=%.4f tot=%.4f q_data=%.3f q_lse=%.3f λ=%.3f",
                    step + 1, stats.td_loss, stats.cql_loss, stats.cmdp_loss,
                    stats.total_loss, stats.q_data_mean, stats.q_logsumexp_mean, stats.cmdp_lambda,
                )
                history.append(stats)
        return history

    # ----- Save / load -----
    def state_dict(self) -> dict:
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "log_lambda": self._log_lambda.detach().cpu(),
            "config": {
                "state_dim": self.state_dim, "n_actions": self.n_actions,
                "n_classes": self.n_classes, "mode": self.mode,
                "cmdp_eps": self.cmdp_eps, "hidden": self.hidden, "depth": self.depth,
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q_target"])
        with torch.no_grad():
            self._log_lambda.copy_(sd["log_lambda"].to(self.device))

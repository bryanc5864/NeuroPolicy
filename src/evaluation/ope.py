# MIT License - Bryan Cheng, 2026
# Part of NeuroPolicy / ieeeICIST
"""Off-Policy Evaluation estimators for the BCI MDP.

Provides:
  * FQEEstimator        : Fitted-Q Evaluation (model-based, no propensity).
  * pdis_estimate       : Per-Decision Importance Sampling (requires
                          non-degenerate behavior log-probs).
  * dr_estimate         : Doubly-Robust estimator combining FQE + PDIS.
  * bootstrap_ci        : trial-level non-parametric bootstrap CIs.

Per RESEARCH_PLAN.md §3.2.5 the gating metric is FQE Pearson r ≥ 0.85 vs.
on-policy ground-truth value on a panel of ≥30 policies for BCI-IV-2b.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.policy_critic import ScalarQNet
from src.training.behavior_policies import Transition
from src.training.replay_buffer import OfflineBuffer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PolicyActionProbsFn = Callable[[np.ndarray], np.ndarray]
"""Maps a (B, S) state batch -> (B, A) action probabilities."""


# ---------------------------------------------------------------------------
# Fitted-Q Evaluation
# ---------------------------------------------------------------------------
@dataclass
class FQEEstimator:
    """Train Q̂(s, a) to satisfy the Bellman equation under target policy π:
        Q̂(s, a) ≈ r + γ · E_{a' ~ π(s')}[Q̂(s', a')]
    Then estimate V(π) = E_{s_0 ∈ initial states}[E_{a ~ π(s_0)}[Q̂(s_0, a)]].
    Does NOT require behavior propensity — works for deterministic μ.
    """
    state_dim: int
    n_actions: int
    gamma: float = 0.99
    hidden: int = 256
    depth: int = 3
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    n_iters: int = 5_000
    batch_size: int = 256
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        self.q = ScalarQNet(self.state_dim, self.n_actions,
                            hidden=self.hidden, depth=self.depth).to(self.device)
        self.q_target = ScalarQNet(self.state_dim, self.n_actions,
                                    hidden=self.hidden, depth=self.depth).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.AdamW(self.q.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.rng = np.random.default_rng(self.seed)

    def _next_value_under_pi(self, sn: torch.Tensor, target_action_probs_fn: PolicyActionProbsFn) -> torch.Tensor:
        """Compute V_target(s') = E_{a ~ π(s')}[Q_target(s', a)]."""
        sn_np = sn.detach().cpu().numpy()
        probs_np = target_action_probs_fn(sn_np)             # (B, A)
        probs = torch.from_numpy(probs_np.astype(np.float32)).to(self.device)
        with torch.no_grad():
            q_next = self.q_target(sn)                        # (B, A)
        return (probs * q_next).sum(dim=1)                    # (B,)

    def fit(self, buffer: OfflineBuffer, target_action_probs_fn: PolicyActionProbsFn,
            polyak_tau: float = 0.005, log_every: int = 1000) -> dict:
        """Train Q̂ to satisfy the Bellman equation under target policy."""
        history = []
        for it in range(self.n_iters):
            batch = buffer.sample(self.batch_size, self.device, self.rng)
            s, a, r, sn, d = batch["s"], batch["a"], batch["r"], batch["sn"], batch["d"]
            with torch.no_grad():
                v_next = self._next_value_under_pi(sn, target_action_probs_fn)
                target = r + self.gamma * (1.0 - d) * v_next
            q_pred = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_pred, target)
            self.opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.opt.step()
            with torch.no_grad():
                for tp, p in zip(self.q_target.parameters(), self.q.parameters()):
                    tp.data.mul_(1.0 - polyak_tau).add_(p.data, alpha=polyak_tau)
            if (it + 1) % log_every == 0:
                history.append({"iter": it + 1, "loss": float(loss.item())})
        return {"history": history}

    @torch.no_grad()
    def estimate_value(self, initial_states: np.ndarray,
                       target_action_probs_fn: PolicyActionProbsFn) -> float:
        """V̂(π) = mean over initial states of E_{a ~ π(s)}[Q̂(s, a)]."""
        s = torch.from_numpy(initial_states.astype(np.float32)).to(self.device)
        probs_np = target_action_probs_fn(initial_states)
        probs = torch.from_numpy(probs_np.astype(np.float32)).to(self.device)
        q = self.q(s)
        v = (probs * q).sum(dim=1)
        return float(v.mean().item())

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        s = torch.from_numpy(states.astype(np.float32)).to(self.device)
        return self.q(s).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Per-Decision Importance Sampling
# ---------------------------------------------------------------------------
def pdis_estimate(
    trajectories: list[list[Transition]],
    target_action_probs_fn: PolicyActionProbsFn,
    gamma: float = 0.99,
    weighted: bool = True,
    eps: float = 1e-12,
) -> float:
    """Per-Decision Importance Sampling estimate of V(π).

    For each trajectory τ:
        contribution = Σ_t γ^t · (Π_{t'≤t} π(a_{t'}|s_{t'}) / μ(a_{t'}|s_{t'})) · r_t
    The weighted variant divides each timestep by the empirical mean weight at
    that timestep across trajectories (variance reduction; consistent under
    mild regularity).

    Behavior log-probs come from `tr.log_prob`. For deterministic μ₁ where
    log_prob = 0 (probability 1), the importance ratio = π(a|s) / 1 = π(a|s),
    which is well-defined; only when μ assigns probability 0 to the chosen
    action would IS blow up — which doesn't happen under our μ₁/μ₂ designs
    because both can produce every commit-or-defer action.
    """
    if not trajectories:
        return 0.0

    n_traj = len(trajectories)
    max_T = max(len(t) for t in trajectories)
    rho_cum = np.zeros((n_traj, max_T), dtype=np.float64)
    rewards = np.zeros((n_traj, max_T), dtype=np.float64)

    # 1) Compute per-step importance ratios.
    all_states = []
    all_actions = []
    all_logmu = []
    traj_step_idx = []  # (traj_idx, step_idx)
    for i, traj in enumerate(trajectories):
        for t, tr in enumerate(traj):
            all_states.append(tr.obs)
            all_actions.append(tr.action)
            all_logmu.append(tr.log_prob)
            traj_step_idx.append((i, t))
            rewards[i, t] = tr.reward
    all_states = np.stack(all_states).astype(np.float32)
    all_actions = np.asarray(all_actions, dtype=np.int64)
    all_logmu = np.asarray(all_logmu, dtype=np.float64)
    pi_probs = target_action_probs_fn(all_states)             # (N, A)
    pi_a = pi_probs[np.arange(pi_probs.shape[0]), all_actions]  # (N,)
    log_pi_a = np.log(pi_a + eps)
    rho = np.exp(log_pi_a - all_logmu)                         # (N,)

    # 2) Build cumulative product per trajectory.
    for k, (i, t) in enumerate(traj_step_idx):
        prev = rho_cum[i, t - 1] if t > 0 else 1.0
        rho_cum[i, t] = prev * rho[k]

    # 3) PDIS estimate.
    discounts = (gamma ** np.arange(max_T))[None, :]            # (1, T)
    weighted_returns = rho_cum * discounts * rewards           # (n_traj, T)

    if weighted:
        # Weighted IS (Rubin's normalization at each timestep)
        # Estimate per-timestep mean rho_cum across trajectories that reached that t.
        T_per_traj = np.array([len(t) for t in trajectories])
        valid = (np.arange(max_T)[None, :] < T_per_traj[:, None])
        mean_rho_t = np.where(valid.sum(axis=0) > 0,
                               (rho_cum * valid).sum(axis=0) / np.maximum(valid.sum(axis=0), 1),
                               1.0)
        norm = np.maximum(mean_rho_t[None, :], eps)
        per_step = (rho_cum * discounts * rewards) / norm
        per_step = per_step * valid
        return float(per_step.sum(axis=1).mean())
    return float(weighted_returns.sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Doubly-Robust
# ---------------------------------------------------------------------------
def dr_estimate(
    trajectories: list[list[Transition]],
    fqe: FQEEstimator,
    target_action_probs_fn: PolicyActionProbsFn,
    gamma: float = 0.99,
    eps: float = 1e-12,
) -> float:
    """Doubly-robust estimator: V̂_FQE(s_0) + Σ_t γ^t · ρ_{1:t} · (r_t + γ V̂(s_{t+1}) − Q̂(s_t, a_t))."""
    if not trajectories:
        return 0.0
    n = len(trajectories)
    estimates = np.zeros(n, dtype=np.float64)
    device = fqe.device

    # Pre-compute Q and V values for all states needed.
    all_obs = []
    all_next = []
    all_actions = []
    all_logmu = []
    all_rewards = []
    all_terminated = []
    bounds = []  # for slicing
    for traj in trajectories:
        for tr in traj:
            all_obs.append(tr.obs); all_next.append(tr.next_obs)
            all_actions.append(tr.action); all_logmu.append(tr.log_prob)
            all_rewards.append(tr.reward); all_terminated.append(1.0 if tr.terminated else 0.0)
        bounds.append(len(all_obs))
    all_obs = np.stack(all_obs).astype(np.float32)
    all_next = np.stack(all_next).astype(np.float32)
    all_actions = np.asarray(all_actions, dtype=np.int64)
    all_logmu = np.asarray(all_logmu, dtype=np.float64)
    all_rewards = np.asarray(all_rewards, dtype=np.float64)
    all_terminated = np.asarray(all_terminated, dtype=np.float64)

    pi_probs = target_action_probs_fn(all_obs)                      # (N, A)
    pi_probs_next = target_action_probs_fn(all_next)                # (N, A)
    pi_a = pi_probs[np.arange(pi_probs.shape[0]), all_actions]
    log_pi_a = np.log(pi_a + eps)
    rho_step = np.exp(log_pi_a - all_logmu)                         # (N,)

    with torch.no_grad():
        s_t = torch.from_numpy(all_obs).to(device)
        sn_t = torch.from_numpy(all_next).to(device)
        q_s = fqe.q(s_t).cpu().numpy()                              # (N, A)
        q_sn = fqe.q(sn_t).cpu().numpy()
    q_sa = q_s[np.arange(q_s.shape[0]), all_actions]
    v_s = (pi_probs * q_s).sum(axis=1)
    v_sn = (pi_probs_next * q_sn).sum(axis=1)

    # Iterate trajectory by trajectory using bounds.
    start = 0
    for i, end in enumerate(bounds):
        T = end - start
        rho_cum = 1.0
        est = float(v_s[start])           # V̂(s_0)
        for t in range(T):
            k = start + t
            rho_cum *= rho_step[k]
            bootstrap_v = 0.0 if all_terminated[k] else gamma * v_sn[k]
            est += (gamma ** t) * rho_cum * (all_rewards[k] + bootstrap_v - q_sa[k])
        estimates[i] = est
        start = end
    return float(estimates.mean())


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------
def bootstrap_ci(
    estimator_fn: Callable[[list], float],
    items: list,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Trial-level bootstrap CI for an estimator that takes a list of trajectories
    (or an analogous unit) and returns a scalar estimate.
    Returns (point_estimate, low_ci, high_ci)."""
    point = estimator_fn(items)
    rng = np.random.default_rng(seed)
    n = len(items)
    boots = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [items[i] for i in idx]
        boots[b] = estimator_fn(sample)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(point), lo, hi

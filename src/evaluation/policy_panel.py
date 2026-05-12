# MIT License - Bryan Cheng, 2026
# Part of NeuroPolicy / ieeeICIST
"""Policy interface + panel of candidate policies for OPE calibration.

A policy must provide:
    - action_probs(states: np.ndarray of shape (B, S)) -> np.ndarray (B, A)
    - select_action(state: np.ndarray of shape (S,))    -> int

Policy panel for M6 calibration:
    * RandomPolicy
    * AgentPolicy(agent, temperature, mix_random_frac)
    * (Optional, plug-in) Mu1Policy / Mu2Policy with state-based reconstruction.

We deliberately keep μ₁/μ₂ logic out of the OPE panel for now: those policies
read running-mean from the state and require parsing the state vector layout
(doable but bug-prone). For the M6 gate we use a panel of trained-agent
variants which already give a wide spread of values via temperature/mixing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PolicyInterface(ABC):
    """All OPE-evaluable policies implement this protocol."""

    n_actions: int
    name: str

    @abstractmethod
    def action_probs(self, states: np.ndarray) -> np.ndarray: ...

    def select_action(self, state: np.ndarray, rng: np.random.Generator | None = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        probs = self.action_probs(state[None])[0]
        return int(rng.choice(self.n_actions, p=probs))


class RandomPolicy(PolicyInterface):
    def __init__(self, n_actions: int, name: str = "random"):
        self.n_actions = n_actions
        self.name = name

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        b = states.shape[0]
        return np.full((b, self.n_actions), 1.0 / self.n_actions, dtype=np.float32)


class AgentPolicy(PolicyInterface):
    """Wrap a trained NeuroPolicyAgent as a softmax-of-Q policy with optional
    uniform-random mixing. temperature → ∞ ⇒ ε-greedy (uniform); → 0 ⇒ argmax.
    """
    def __init__(self, agent, temperature: float = 1.0, mix_random_frac: float = 0.0,
                 name: str | None = None, cvar_alpha: float | None = None):
        self.agent = agent
        self.temperature = float(temperature)
        self.mix = float(mix_random_frac)
        self.n_actions = agent.n_actions
        self.cvar_alpha = cvar_alpha
        self.name = name or f"agent_T{self.temperature}_mix{self.mix}"

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        # Vectorize over the batch.
        import torch
        import torch.nn.functional as F
        s = torch.from_numpy(states.astype(np.float32)).to(self.agent.device)
        with torch.no_grad():
            if self.agent.mode == "distributional" and self.cvar_alpha is not None and self.cvar_alpha < 1.0:
                from src.models.policy_critic import QuantileQNet
                z = self.agent.q(s)
                q = QuantileQNet.cvar(z, alpha=self.cvar_alpha)
            else:
                q = self.agent._q_values(s)
            if self.temperature <= 1e-6:
                # argmax
                probs = torch.zeros_like(q)
                probs.scatter_(1, q.argmax(dim=1, keepdim=True), 1.0)
            else:
                probs = F.softmax(q / self.temperature, dim=1)
        probs_np = probs.detach().cpu().numpy().astype(np.float32)
        if self.mix > 0:
            uniform = np.full_like(probs_np, 1.0 / self.n_actions)
            probs_np = (1.0 - self.mix) * probs_np + self.mix * uniform
        # numerical guard
        probs_np = np.clip(probs_np, 1e-12, 1.0)
        probs_np /= probs_np.sum(axis=1, keepdims=True)
        return probs_np

    def select_action(self, state, rng=None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        probs = self.action_probs(state[None])[0]
        return int(rng.choice(self.n_actions, p=probs))

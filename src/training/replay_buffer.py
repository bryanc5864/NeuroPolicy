# MIT License - Bryan Cheng, 2026
# Part of NeuroPolicy / ieeeICIST
"""Tensor-backed offline replay buffer for the NeuroPolicy agent.

We store transitions (s, a, r, s', done, log_pi_b) in CPU-pinned tensors and
stream batches to GPU per gradient step. For the dataset sizes at hand
(< 1M transitions) this comfortably fits in RAM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.training.behavior_policies import Transition


@dataclass
class OfflineBuffer:
    states: torch.Tensor       # (N, S)  float32
    actions: torch.Tensor      # (N,)    int64
    rewards: torch.Tensor      # (N,)    float32
    next_states: torch.Tensor  # (N, S)  float32
    dones: torch.Tensor        # (N,)    float32
    log_probs_b: torch.Tensor  # (N,)    float32  (log π_b(a|s))
    # Auxiliary (for OPE / CMDP):
    is_wrong_commit: torch.Tensor  # (N,) float32  1 iff commit and wrong
    info_actions: torch.Tensor     # (N,) int64    a_DEFER/a_RECAL/a_ABSTAIN/0..K-1
    # Episode boundaries (for episode-level OPE):
    episode_id: torch.Tensor   # (N,) int64

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def sample(self, batch_size: int, device: torch.device,
               rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(low=0, high=len(self), size=batch_size)
        idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.long)
        return {
            "s": self.states.index_select(0, idx_t.cpu()).to(device, non_blocking=True),
            "a": self.actions.index_select(0, idx_t.cpu()).to(device, non_blocking=True),
            "r": self.rewards.index_select(0, idx_t.cpu()).to(device, non_blocking=True),
            "sn": self.next_states.index_select(0, idx_t.cpu()).to(device, non_blocking=True),
            "d": self.dones.index_select(0, idx_t.cpu()).to(device, non_blocking=True),
            "log_pi_b": self.log_probs_b.index_select(0, idx_t.cpu()).to(device, non_blocking=True),
            "wrong": self.is_wrong_commit.index_select(0, idx_t.cpu()).to(device, non_blocking=True),
        }


def build_buffer_from_rollouts(
    all_trajectories: list[list[Transition]],
    a_defer: int,
    a_recal: int,
    a_abstain: int,
    n_classes: int,
) -> OfflineBuffer:
    """Pack a list-of-list-of-Transitions into tensors."""
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
    is_wrong = []
    info_actions = []
    episode_ids = []
    for ep_id, traj in enumerate(all_trajectories):
        for tr in traj:
            states.append(tr.obs)
            actions.append(tr.action)
            rewards.append(tr.reward)
            next_states.append(tr.next_obs)
            dones.append(1.0 if tr.terminated else 0.0)
            log_probs.append(tr.log_prob)
            is_wrong.append(1.0 if tr.info.get("outcome") == "wrong_commit" else 0.0)
            info_actions.append(tr.action)
            episode_ids.append(ep_id)

    return OfflineBuffer(
        states=torch.from_numpy(np.stack(states).astype(np.float32)).pin_memory(),
        actions=torch.tensor(actions, dtype=torch.long).pin_memory(),
        rewards=torch.tensor(rewards, dtype=torch.float32).pin_memory(),
        next_states=torch.from_numpy(np.stack(next_states).astype(np.float32)).pin_memory(),
        dones=torch.tensor(dones, dtype=torch.float32).pin_memory(),
        log_probs_b=torch.tensor(log_probs, dtype=torch.float32).pin_memory(),
        is_wrong_commit=torch.tensor(is_wrong, dtype=torch.float32).pin_memory(),
        info_actions=torch.tensor(info_actions, dtype=torch.long).pin_memory(),
        episode_id=torch.tensor(episode_ids, dtype=torch.long).pin_memory(),
    )

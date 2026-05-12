# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""BCI-as-MDP simulator.

Single-trial deterministic environment. The agent observes a rolling window
embedding plus running stats, time index, recal flag, and subject embedding.
At each step the agent picks {commit-class_i, DEFER, REQUEST_RECAL, ABSTAIN}.

Design choices:
* The encoder is OUTSIDE the env: the env consumes pre-computed per-window
  features so it stays cheap, deterministic, and trivially batch-able.
* Subject embedding is also passed in (default + post-recal precomputed).
* Action space is Discrete(K + 3): the last 3 are DEFER, REQUEST_RECAL, ABSTAIN.
* Episode = one trial. `reset` requires an episode bundle in `options`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import MDP_CFG, MDPConfig

# Sentinel action indices computed at env construction.


@dataclass
class TrialEpisode:
    """One pre-encoded trial that the MDP can be reset onto.

    Fields:
        window_features      : (T, embed_dim) np.float32 — encoder output per window
        label                : int — ground-truth class index
        subject_id           : int — for indexing into per-subject calibration buffers
        subj_embed_default   : (D_s,) np.float32 — initial subject embedding
        subj_embed_post_recal: (D_s,) np.float32 — embedding after REQUEST_RECAL
        stride_seconds       : float — wall time advanced per DEFER step
    """
    window_features: np.ndarray
    label: int
    subject_id: int
    subj_embed_default: np.ndarray
    subj_embed_post_recal: np.ndarray
    stride_seconds: float

    def __post_init__(self):
        assert self.window_features.ndim == 2
        assert self.window_features.dtype == np.float32
        assert self.subj_embed_default.shape == self.subj_embed_post_recal.shape
        assert self.subj_embed_default.dtype == np.float32

    @property
    def T(self) -> int:
        return int(self.window_features.shape[0])

    @property
    def embed_dim(self) -> int:
        return int(self.window_features.shape[1])

    @property
    def subj_dim(self) -> int:
        return int(self.subj_embed_default.shape[0])


def state_dim(embed_dim: int, subj_dim: int) -> int:
    """Layout: [feat_t (D), running_mean (D), running_var (D), t/Tmax (1), r_used (1), e_subj (Ds)]."""
    return 3 * embed_dim + 2 + subj_dim


class BCIEnv(gym.Env):
    """Per-trial offline BCI MDP. .2.1.

    Action layout (Discrete):
        0..K-1     : commit class i
        K          : DEFER
        K + 1      : REQUEST_RECAL
        K + 2      : ABSTAIN
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_classes: int,
        embed_dim: int,
        subj_dim: int,
        mdp_cfg: MDPConfig = MDP_CFG,
        max_T: int = 64,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.embed_dim = int(embed_dim)
        self.subj_dim = int(subj_dim)
        self.cfg = mdp_cfg
        self.max_T = int(max_T)

        self.A_DEFER = self.n_classes
        self.A_RECAL = self.n_classes + 1
        self.A_ABSTAIN = self.n_classes + 2
        self.action_space = spaces.Discrete(self.n_classes + 3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim(self.embed_dim, self.subj_dim),),
            dtype=np.float32,
        )

        # Per-episode state (set in reset)
        self._episode: TrialEpisode | None = None
        self._t: int = 0
        self._r_used: int = 0
        self._sum: np.ndarray | None = None
        self._sumsq: np.ndarray | None = None
        self._n_observed: int = 0
        self._terminated: bool = False
        self._recal_advance_steps: int = max(
            1, int(round(self.cfg.recal_time_cost_seconds /
                          max(self.cfg.stride_seconds, 1e-6)))
        )

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options is None or "episode" not in options:
            raise ValueError("BCIEnv.reset requires options={'episode': TrialEpisode}")
        ep: TrialEpisode = options["episode"]
        if ep.embed_dim != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: env={self.embed_dim} ep={ep.embed_dim}")
        if ep.subj_dim != self.subj_dim:
            raise ValueError(f"subj_dim mismatch: env={self.subj_dim} ep={ep.subj_dim}")
        if ep.T > self.max_T:
            raise ValueError(f"Episode length {ep.T} exceeds max_T={self.max_T}")

        self._episode = ep
        self._t = 0
        self._r_used = 0
        self._sum = np.zeros(self.embed_dim, dtype=np.float64)
        self._sumsq = np.zeros(self.embed_dim, dtype=np.float64)
        self._n_observed = 0
        self._terminated = False

        # First window already observed at t=0
        self._observe_current_window()

        return self._make_state(), {"label": ep.label, "subject_id": ep.subject_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("step called after termination; call reset.")
        ep = self._episode
        assert ep is not None
        action = int(action)
        info: dict[str, Any] = {"label": ep.label, "subject_id": ep.subject_id, "t": self._t}

        if 0 <= action < self.n_classes:                          # commit
            correct = (action == ep.label)
            elapsed_seconds = self._t * self.cfg.stride_seconds
            if correct:
                reward = self.cfg.r_correct - self.cfg.lambda_per_sec * elapsed_seconds
                info["outcome"] = "correct_commit"
            else:
                reward = -self.cfg.r_wrong - self.cfg.lambda_per_sec * elapsed_seconds
                info["outcome"] = "wrong_commit"
            self._terminated = True
            return self._make_state(), float(reward), True, False, info

        if action == self.A_DEFER:
            self._t += 1
            if self._t >= ep.T:                                    # ran out of windows
                # forced to abstain
                self._terminated = True
                info["outcome"] = "abstain_timeout"
                reward = -self.cfg.r_abstain - self.cfg.lambda_per_sec * (
                    self._t * self.cfg.stride_seconds
                )
                return self._make_state(), float(reward), True, False, info
            self._observe_current_window()
            reward = -self.cfg.lambda_per_sec * self.cfg.stride_seconds
            info["outcome"] = "defer"
            return self._make_state(), float(reward), False, False, info

        if action == self.A_RECAL:
            if self._r_used:                                       # already used; soft-illegal
                # Treat as no-op DEFER with extra penalty so policy learns to avoid.
                reward = -self.cfg.r_recal - self.cfg.lambda_per_sec * self.cfg.stride_seconds
                self._t += 1
                if self._t >= ep.T:
                    self._terminated = True
                    info["outcome"] = "abstain_timeout"
                    return self._make_state(), float(reward), True, False, info
                self._observe_current_window()
                info["outcome"] = "recal_repeated_no_op"
                return self._make_state(), float(reward), False, False, info
            self._r_used = 1
            self._t += self._recal_advance_steps
            if self._t >= ep.T:
                self._terminated = True
                info["outcome"] = "abstain_timeout_recal"
                reward = -self.cfg.r_recal - self.cfg.r_abstain - self.cfg.lambda_per_sec * (
                    self._t * self.cfg.stride_seconds
                )
                return self._make_state(), float(reward), True, False, info
            self._observe_current_window()
            reward = -self.cfg.r_recal
            info["outcome"] = "recal"
            return self._make_state(), float(reward), False, False, info

        if action == self.A_ABSTAIN:
            self._terminated = True
            info["outcome"] = "abstain"
            reward = -self.cfg.r_abstain - self.cfg.lambda_per_sec * (
                self._t * self.cfg.stride_seconds
            )
            return self._make_state(), float(reward), True, False, info

        raise ValueError(f"Invalid action {action}; expected 0..{self.n_classes + 2}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _observe_current_window(self) -> None:
        """Fold the window at index self._t into running stats."""
        ep = self._episode
        assert ep is not None
        feat = ep.window_features[self._t].astype(np.float64, copy=False)
        self._sum += feat
        self._sumsq += feat * feat
        self._n_observed += 1

    def _make_state(self) -> np.ndarray:
        ep = self._episode
        assert ep is not None
        feat_t = (ep.window_features[min(self._t, ep.T - 1)]
                  if not self._terminated else np.zeros(self.embed_dim, dtype=np.float32))
        if self._n_observed > 0:
            mean = (self._sum / self._n_observed).astype(np.float32)
            var = ((self._sumsq / self._n_observed) - mean.astype(np.float64) ** 2)
            var = np.maximum(var, 0.0).astype(np.float32)
        else:
            mean = np.zeros(self.embed_dim, dtype=np.float32)
            var = np.zeros(self.embed_dim, dtype=np.float32)
        elapsed_norm = np.float32(self._t / max(ep.T - 1, 1))
        recal = np.float32(self._r_used)
        subj = (ep.subj_embed_post_recal if self._r_used else ep.subj_embed_default)

        s = np.concatenate(
            [
                feat_t.astype(np.float32, copy=False),
                mean,
                var,
                np.array([elapsed_norm, recal], dtype=np.float32),
                subj.astype(np.float32, copy=False),
            ],
            axis=0,
        )
        return s

    @property
    def state_dim(self) -> int:
        return state_dim(self.embed_dim, self.subj_dim)

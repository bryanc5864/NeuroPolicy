# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""Heuristic / non-RL stopping baselines (RESEARCH_PLAN §4.2 entries iv-vi, x).

These baselines all consume the MDP state via the running-mean slot and a
classifier (LogReg / EEGNet linear head) trained on training trials.

Implemented:
  * EEGNetThresholdStoppingPolicy : commit when classifier max-prob >= τ,
    else DEFER. Forces commit at t = T-1.   (Verschore & Kindermans 2012;
    Lawhern et al. 2018 backbone.)
  * BTSPRTPolicy                  : log-likelihood-ratio sequential
    probability ratio test with Bonferroni-balanced thresholds (Liu 2017).
    For two classes uses log p(c=1)/p(c=0); for K>2, max log p(c)/p_avg.
  * OraclePolicy                  : commits the ground-truth label at t=0.
    Used as the upper bound only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import ClassifierMixin

from src.evaluation.policy_panel import PolicyInterface


def _state_layout(state_dim: int, n_actions: int, n_classes: int):
    """Returns (embed_dim, subj_dim) such that
    state = [feat_t (D), running_mean (D), running_var (D), t/T (1), recal (1), e_subj (Ds)].
    Plain inversion: 3*D + 2 + Ds = state_dim. We assume a fixed convention
    consistent with src/training/bci_env.py.
    """
    # Without explicit metadata we cannot infer D and Ds from state_dim alone.
    # The caller must pass them in.
    raise NotImplementedError("Use the explicit (embed_dim, subj_dim) constructor args.")


@dataclass
class EEGNetThresholdStoppingPolicy(PolicyInterface):
    """Commit when classifier(running_mean).max_prob >= threshold; else DEFER.

    Forces commit on the running-mean classifier prediction at t = T-1.
    """
    classifier: ClassifierMixin
    n_actions: int
    n_classes: int
    embed_dim: int
    subj_dim: int
    T_total: int
    threshold: float = 0.55
    a_defer_id: int | None = None
    name: str = "eegnet_threshold"

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        B = states.shape[0]
        probs = np.zeros((B, self.n_actions), dtype=np.float32)
        D = self.embed_dim
        a_defer = self.a_defer_id if self.a_defer_id is not None else self.n_classes
        running_mean = states[:, D:2 * D]
        elapsed_norm = states[:, 3 * D]
        # running-mean classifier posterior
        p = self.classifier.predict_proba(running_mean)            # (B, K)
        max_p = p.max(axis=1)
        argmax = p.argmax(axis=1)
        force_commit = (elapsed_norm >= 1.0 - 1e-6)
        commit_mask = (max_p >= self.threshold) | force_commit
        for i in range(B):
            if commit_mask[i]:
                probs[i, int(argmax[i])] = 1.0
            else:
                probs[i, a_defer] = 1.0
        return probs


@dataclass
class BTSPRTPolicy(PolicyInterface):
    """Balanced-threshold SPRT on classifier log-likelihood ratios.

    For two classes, evidence_t = log[p_t(1)/p_t(0)]. Stop when |evidence_t| > A.
    For K > 2 we use max-class margin log[p_t(c*)/p_t(2nd best)].
    """
    classifier: ClassifierMixin
    n_actions: int
    n_classes: int
    embed_dim: int
    subj_dim: int
    T_total: int
    threshold_A: float = 1.5  # log-odds threshold; ≈ probabilistic 0.82 prob
    a_defer_id: int | None = None
    name: str = "btsprt"

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        B = states.shape[0]
        probs = np.zeros((B, self.n_actions), dtype=np.float32)
        D = self.embed_dim
        a_defer = self.a_defer_id if self.a_defer_id is not None else self.n_classes
        running_mean = states[:, D:2 * D]
        elapsed_norm = states[:, 3 * D]
        p = self.classifier.predict_proba(running_mean)            # (B, K)
        if self.n_classes == 2:
            llr = np.log(p[:, 1] + 1e-12) - np.log(p[:, 0] + 1e-12)
            commit_mask = np.abs(llr) > self.threshold_A
            argmax = (llr > 0).astype(np.int64)
        else:
            sort_p = -np.sort(-p, axis=1)
            margin = np.log(sort_p[:, 0] + 1e-12) - np.log(sort_p[:, 1] + 1e-12)
            commit_mask = margin > self.threshold_A
            argmax = p.argmax(axis=1)
        force_commit = (elapsed_norm >= 1.0 - 1e-6)
        commit_mask = commit_mask | force_commit
        for i in range(B):
            if commit_mask[i]:
                probs[i, int(argmax[i])] = 1.0
            else:
                probs[i, a_defer] = 1.0
        return probs


@dataclass
class OraclePolicy(PolicyInterface):
    """Commits the ground-truth class at t = 0.

    Cannot be used in OPE (it requires the label at action time, which is
    not in the state). Provided only as an upper-bound reference for
    on-policy MC evaluation; the env expose label via reset() info.
    """
    n_actions: int
    n_classes: int
    name: str = "oracle"

    # action_probs cannot be computed from state alone — Oracle is "cheating"
    # so it does not take part in OPE comparisons.
    def action_probs(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Oracle requires the trial label; not OPE-compatible.")

    def select_action_with_label(self, label: int) -> int:
        return int(label)

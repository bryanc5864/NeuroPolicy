# MIT License - Bryan Cheng, 2026
# Part of NeuroPolicy / ieeeICIST
"""Behavior policies that generate the offline RL dataset.

Per RESEARCH_PLAN.md §3.2.4 we build TWO logging policies:
  μ₁ (deterministic-fixed-window): DEFER until t = T_fix, then commit using
      a per-subject EEGNet-on-mean classifier on the running-mean features.
      T_fix sweeps across {1.0, 2.0, 3.0, 4.0} s.
      Action propensity is 1 for the chosen action; degenerate for IS-based
      OPE (we rely on FQE for those rollouts).

  μ₂ (SPRT-stochastic): Defer until evidence accumulator crosses a threshold
      OR a stochastic stop fires; commit with prob 1 - eps to the per-subject
      classifier prediction, otherwise uniform random. This gives a non-
      degenerate propensity log_prob suitable for PDIS/DR estimators.

The policies are simple by design — they exist to populate the offline buffer,
not to be strong baselines (we have a separate baselines suite for that).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.training.bci_env import BCIEnv, TrialEpisode


@dataclass
class _PolicyContext:
    """Per-subject helpers that any logging policy needs."""
    classifier: ClassifierMixin
    n_classes: int


def fit_running_mean_classifier(
    train_episodes: list[TrialEpisode],
    n_classes: int,
    seed: int = 0,
) -> _PolicyContext:
    """Train a logistic-regression classifier on running-mean features.

    Each training point is the running mean of encoder features over the
    *full* trial (label = ground-truth class). This is the simplest
    classifier we can use for the logging policy without overfitting; it is
    NOT meant to be a strong decoder.
    """
    Xs, ys = [], []
    for ep in train_episodes:
        Xs.append(ep.window_features.mean(axis=0))
        ys.append(ep.label)
    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X, y)
    return _PolicyContext(classifier=clf, n_classes=n_classes)


# ---------------------------------------------------------------------------
# μ₁ — deterministic fixed-window
# ---------------------------------------------------------------------------
@dataclass
class FixedWindowPolicy:
    """μ₁: DEFER until t = T_fix, then commit using the running-mean classifier."""
    ctx: _PolicyContext
    T_fix: int           # number of strides to defer before committing
    env: BCIEnv

    def reset(self): pass

    def act(self, episode: TrialEpisode, env: BCIEnv, t: int,
            running_mean: np.ndarray) -> tuple[int, float]:
        """Return (action, log_prob_under_behavior). Deterministic μ₁."""
        if t < self.T_fix and t < episode.T - 1:
            return env.A_DEFER, 0.0  # log prob = log(1) = 0
        # commit with classifier
        feat = running_mean.reshape(1, -1)
        pred = int(self.ctx.classifier.predict(feat)[0])
        return pred, 0.0


# ---------------------------------------------------------------------------
# μ₂ — SPRT-stochastic
# ---------------------------------------------------------------------------
@dataclass
class SPRTStochasticPolicy:
    """μ₂: probabilistic stop on evidence + commit with epsilon-noise.

    At each step compute classifier posterior on the running-mean features.
    Evidence statistic = max class prob. If evidence > threshold, COMMIT
    with prob 1 - eps_explore to argmax prediction, else uniform among classes.
    Otherwise DEFER. Forces commit by step T-1.

    `commit_temp` and `evidence_threshold` are tunable knobs.
    """
    ctx: _PolicyContext
    env: BCIEnv
    evidence_threshold: float = 0.55
    eps_explore: float = 0.10
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def reset(self): pass

    def act(self, episode: TrialEpisode, env: BCIEnv, t: int,
            running_mean: np.ndarray) -> tuple[int, float]:
        K = self.ctx.n_classes
        proba = self.ctx.classifier.predict_proba(running_mean.reshape(1, -1))[0]
        confident = proba.max() >= self.evidence_threshold
        force_commit = (t >= episode.T - 1)

        if not (confident or force_commit):
            return env.A_DEFER, float(np.log(1.0))

        # Build commit distribution: argmax with prob 1 - eps + eps/K, others eps/K
        argmax = int(np.argmax(proba))
        commit_p = np.full(K, self.eps_explore / K, dtype=np.float64)
        commit_p[argmax] += (1.0 - self.eps_explore)

        # Sample
        action = int(self.rng.choice(K, p=commit_p))
        log_prob = float(np.log(commit_p[action] + 1e-12))
        return action, log_prob


# ---------------------------------------------------------------------------
# Rollout generator
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    log_prob: float
    info: dict


def rollout_episode(
    env: BCIEnv,
    episode: TrialEpisode,
    policy,
    seed: int | None = None,
) -> list[Transition]:
    """Run policy on one episode, returning the trajectory."""
    obs, info = env.reset(options={"episode": episode}, seed=seed)
    transitions: list[Transition] = []
    embed_dim = env.embed_dim
    while True:
        # The state's [D : 2D] slice is the running mean.
        running_mean = obs[embed_dim:2 * embed_dim]
        t = env._t
        action, log_prob = policy.act(episode, env, t, running_mean)
        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append(Transition(
            obs=obs.copy(), action=int(action), reward=float(reward),
            next_obs=next_obs.copy(), terminated=bool(terminated),
            log_prob=float(log_prob), info=info,
        ))
        if terminated or truncated:
            break
        obs = next_obs
    return transitions

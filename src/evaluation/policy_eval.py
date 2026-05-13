# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""On-policy Monte-Carlo evaluation of any policy on a list of TrialEpisodes.

Because the env is deterministic given an episode and our policies are
deterministic at eval time (argmax-Q or fixed thresholds), this gives the
ground-truth value of a policy. We use this as the reference for the OPE
calibration in  and as the headline evaluation metric.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.training.bci_env import BCIEnv, TrialEpisode


@dataclass
class PolicyMetrics:
    return_mean: float
    return_std: float
    returns: np.ndarray
    accuracy_on_commits: float    # acc among commit episodes
    n_commits: int
    n_correct_commits: int
    n_wrong_commits: int
    n_abstains: int
    n_abstain_timeouts: int
    n_recals: int
    mean_decision_seconds: float  # over commit episodes only
    median_decision_seconds: float
    outcome_counts: dict[str, int]
    wrong_commit_rate: float       # over all episodes (for CMDP feasibility)
    information_transfer_rate: float  # bits/min, over commit episodes
    decision_seconds: np.ndarray | None = None  # per-commit decision times (s)
    commit_correct: np.ndarray | None = None    # per-commit correct/incorrect flag

    def asdict(self) -> dict:
        d = {k: getattr(self, k) for k in vars(self) if k != "returns"}
        d["returns"] = [float(x) for x in self.returns]
        if self.decision_seconds is not None:
            d["decision_seconds"] = [float(x) for x in self.decision_seconds]
        if self.commit_correct is not None:
            d["commit_correct"] = [int(x) for x in self.commit_correct]
        return d


def evaluate_policy(
    env: BCIEnv,
    episodes: list[TrialEpisode],
    select_action,
    n_classes: int,
    seed: int = 0,
) -> PolicyMetrics:
    """Run `select_action(state)` on every episode and aggregate metrics.

    `select_action` may be:
      * agent.select_action     (NeuroPolicy)
      * lambda s: env.A_DEFER ... (heuristic)
    """
    returns = np.zeros(len(episodes), dtype=np.float64)
    decision_steps = []
    commit_correct_flags: list[int] = []
    n_commits = n_correct = n_wrong = n_abstain = n_abstain_to = n_recals = 0
    outcomes: Counter = Counter()

    for i, ep in enumerate(episodes):
        obs, info = env.reset(options={"episode": ep}, seed=seed + i)
        episode_return = 0.0
        last_outcome = None
        last_t = 0
        while True:
            a = int(select_action(obs))
            next_obs, r, term, trunc, info = env.step(a)
            episode_return += r
            outc = info.get("outcome", "")
            outcomes[outc] += 1
            if outc == "recal" or outc == "recal_repeated_no_op":
                n_recals += 1
            if term or trunc:
                last_outcome = outc
                last_t = info.get("t", 0)
                break
            obs = next_obs
        returns[i] = episode_return
        if last_outcome == "correct_commit":
            n_correct += 1; n_commits += 1
            decision_steps.append(last_t)
            commit_correct_flags.append(1)
        elif last_outcome == "wrong_commit":
            n_wrong += 1; n_commits += 1
            decision_steps.append(last_t)
            commit_correct_flags.append(0)
        elif last_outcome == "abstain":
            n_abstain += 1
        elif "abstain_timeout" in (last_outcome or ""):
            n_abstain_to += 1

    accuracy = (n_correct / n_commits) if n_commits else 0.0

    # ITR (bits / min) using Wolpaw formula:
    # B = log2(N) + acc*log2(acc) + (1-acc)*log2((1-acc)/(N-1))
    # multiplied by commits/min implied by mean decision time.
    if n_commits and 1.0 / n_classes < accuracy < 1.0:
        bits_per_decision = (
            np.log2(n_classes)
            + accuracy * np.log2(accuracy + 1e-12)
            + (1 - accuracy) * np.log2((1 - accuracy + 1e-12) / (n_classes - 1))
        )
    elif accuracy >= 1.0 and n_commits:
        bits_per_decision = np.log2(n_classes)
    else:
        bits_per_decision = 0.0
    mean_decision_sec = (
        float(np.mean(decision_steps) * env.cfg.stride_seconds) if decision_steps else float("nan")
    )
    itr = float(bits_per_decision / max(mean_decision_sec, 1e-3) * 60.0) if decision_steps else 0.0

    return PolicyMetrics(
        return_mean=float(returns.mean()),
        return_std=float(returns.std()),
        returns=returns,
        accuracy_on_commits=float(accuracy),
        n_commits=int(n_commits),
        n_correct_commits=int(n_correct),
        n_wrong_commits=int(n_wrong),
        n_abstains=int(n_abstain),
        n_abstain_timeouts=int(n_abstain_to),
        n_recals=int(n_recals),
        mean_decision_seconds=mean_decision_sec,
        median_decision_seconds=float(np.median(decision_steps) * env.cfg.stride_seconds) if decision_steps else float("nan"),
        outcome_counts=dict(outcomes),
        wrong_commit_rate=float(n_wrong / max(len(episodes), 1)),
        information_transfer_rate=itr,
        decision_seconds=(np.array(decision_steps) * env.cfg.stride_seconds) if decision_steps else None,
        commit_correct=np.array(commit_correct_flags) if commit_correct_flags else None,
    )


def random_policy(n_actions: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return lambda s: int(rng.integers(n_actions))

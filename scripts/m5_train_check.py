# MIT License - Bryan Cheng, 2026
"""Milestone 5 verification: train scalar CQL on bci2b sub 4 and confirm
the agent improves over a random policy within a budget that fits the
laptop GPU.

Per RESEARCH_PLAN.md §6 Milestone 5: 'sanity check that the agent improves
over the random policy and converges within 100k steps.' We run 20k steps
here for a quick verification and document the curve.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data.preprocess import preprocess_subject
from src.evaluation.policy_eval import evaluate_policy, random_policy
from src.models.encoder import build_encoder, pretrain_encoder_supervised
from src.training.behavior_policies import (
    FixedWindowPolicy,
    SPRTStochasticPolicy,
    fit_running_mean_classifier,
    rollout_episode,
)
from src.training.bci_env import BCIEnv
from src.training.episode_builder import (
    SubjectEmbeddingFactory,
    build_episodes_for_trial_batch,
    windows_from_trial,
)
from src.training.neuropolicy_agent import NeuroPolicyAgent
from src.training.replay_buffer import build_buffer_from_rollouts
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG, TRAIN_CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("m5_train")


def _slice_trial_batch(tb, idx):
    """Return a TrialBatch containing only the trials at indices `idx`."""
    from src.data.moabb_loader import TrialBatch
    return TrialBatch(
        X=tb.X[idx].copy(), y=tb.y[idx].copy(),
        session=tb.session[idx].copy(), run=tb.run[idx].copy(),
        sfreq=tb.sfreq, ch_names=list(tb.ch_names),
        class_labels=list(tb.class_labels),
        subject_id=tb.subject_id, dataset_key=tb.dataset_key,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # 1) Data
    tb = preprocess_subject("bci2b", subject_id=4)
    n_classes = len(tb.class_labels)
    log.info("Loaded bci2b sub 4: X=%s n_classes=%d", tb.X.shape, n_classes)

    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    enc = build_encoder("eegnet", n_channels=tb.n_channels,
                        n_samples=win_samples, sfreq=tb.sfreq).to(device)

    # 1a) Trial-level split BEFORE encoding so the encoder is pretrained only
    # on training trials (no leakage into val/test feature space).
    rng = np.random.default_rng(0)
    perm = rng.permutation(tb.n_trials)
    n_tr = int(tb.n_trials * 0.70)
    n_va = int(tb.n_trials * 0.15)
    tr_idx = np.sort(perm[:n_tr]); va_idx = np.sort(perm[n_tr:n_tr + n_va])
    te_idx = np.sort(perm[n_tr + n_va:])
    log.info("Trial split: train=%d val=%d test=%d", len(tr_idx), len(va_idx), len(te_idx))

    # 1b) Supervised pretrain encoder on TRAIN trials' 1-s windows
    # (each window inherits its trial label). This stands in for LaBraM
    # pretrained features until the foundation-model checkpoint is integrated.
    win_X, win_y = [], []
    for i in tr_idx:
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
        win_X.append(W)
        win_y.extend([int(tb.y[i])] * W.shape[0])
    win_X = np.concatenate(win_X, axis=0)
    win_y = np.asarray(win_y, dtype=np.int64)
    log.info("Encoder pretrain set: windows=%s labels=%s", win_X.shape, win_y.shape)
    pre_info = pretrain_encoder_supervised(
        enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
        n_epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.15,
    )
    log.info("Encoder pretrain done: best_val_acc=%.3f", pre_info["best_val_acc"])

    # 2) Build episodes from each split using the now-frozen encoder.
    # LEAK-SAFE: build train_eps first so its (default, post_recal) embedding
    # pair is derived from training data, then pass that pair into val/test
    # builds via `subj_embeds=` so all splits share identical embeddings.
    subj_factory = SubjectEmbeddingFactory(encoder_embed_dim=enc.spec.embed_dim, subj_dim=16, seed=0)
    train_eps, _, cal_idx_tr = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, tr_idx), encoder=enc, device=device, subj_factory=subj_factory,
    )
    train_subj_embeds = (
        train_eps[0].subj_embed_default.copy(),
        train_eps[0].subj_embed_post_recal.copy(),
    )
    val_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, va_idx), encoder=enc, device=device, subj_factory=subj_factory,
        subj_embeds=train_subj_embeds,
        calibration_indices=np.array([], dtype=np.int64),  # don't reserve val cal trials
    )
    test_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, te_idx), encoder=enc, device=device, subj_factory=subj_factory,
        subj_embeds=train_subj_embeds,
        calibration_indices=np.array([], dtype=np.int64),
    )
    log.info("Episodes: train=%d val=%d test=%d", len(train_eps), len(val_eps), len(test_eps))

    # 3) Env
    env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                 subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)

    # 4) Logging classifier + behavior policies
    ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)
    log.info("Logging-classifier train acc on running-mean: %.3f",
             float(ctx.classifier.score(
                 np.stack([e.window_features.mean(0) for e in train_eps]),
                 np.array([e.label for e in train_eps]))))

    # 5) Generate offline dataset: μ₁ at multiple T_fix + μ₂ stochastic
    rollouts = []
    for T_fix in [4, 8, 12]:
        mu1 = FixedWindowPolicy(ctx=ctx, T_fix=T_fix, env=env)
        for i, ep in enumerate(train_eps):
            rollouts.append(rollout_episode(env, ep, mu1, seed=i))
    rng_mu = np.random.default_rng(0)
    mu2 = SPRTStochasticPolicy(ctx=ctx, env=env, evidence_threshold=0.55, eps_explore=0.10, rng=rng_mu)
    for i, ep in enumerate(train_eps):
        rollouts.append(rollout_episode(env, ep, mu2, seed=10_000 + i))
    n_transitions = sum(len(r) for r in rollouts)
    log.info("Generated %d trajectories (%d transitions)", len(rollouts), n_transitions)

    buffer = build_buffer_from_rollouts(
        rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL, a_abstain=env.A_ABSTAIN, n_classes=n_classes,
    )

    # 6) Train scalar CQL agent
    agent = NeuroPolicyAgent(
        state_dim=env.state_dim, n_actions=env.action_space.n, n_classes=n_classes,
        cfg=TRAIN_CFG, mode="scalar", cmdp_eps=None, gamma=0.99, polyak_tau=0.005,
        device=device, hidden=256, depth=3, seed=0,
    )
    log.info("Agent: scalar mode, hidden=256 depth=3, params=%d",
             sum(p.numel() for p in agent.q.parameters()))

    # Baseline eval (random) before training
    pre_rand = evaluate_policy(env, val_eps, random_policy(env.action_space.n, 0), n_classes=n_classes)
    log.info("Pre-train RANDOM on val: return=%.3f acc=%.3f n_commits=%d ITR=%.2f",
             pre_rand.return_mean, pre_rand.accuracy_on_commits, pre_rand.n_commits,
             pre_rand.information_transfer_rate)

    pre_agent = evaluate_policy(env, val_eps, lambda s: agent.select_action(s), n_classes=n_classes)
    log.info("Pre-train AGENT (random init) on val: return=%.3f acc=%.3f n_commits=%d ITR=%.2f",
             pre_agent.return_mean, pre_agent.accuracy_on_commits, pre_agent.n_commits,
             pre_agent.information_transfer_rate)

    t0 = time.time()
    history = agent.train(buffer, n_steps=20_000, batch_size=256, log_every=2_000)
    train_seconds = time.time() - t0

    # 7) Eval after training
    post_agent = evaluate_policy(env, val_eps, lambda s: agent.select_action(s), n_classes=n_classes)
    log.info("Post-train AGENT on val: return=%.3f acc=%.3f n_commits=%d ITR=%.2f outcomes=%s",
             post_agent.return_mean, post_agent.accuracy_on_commits, post_agent.n_commits,
             post_agent.information_transfer_rate, post_agent.outcome_counts)

    test_agent = evaluate_policy(env, test_eps, lambda s: agent.select_action(s), n_classes=n_classes)
    log.info("Post-train AGENT on TEST: return=%.3f acc=%.3f n_commits=%d ITR=%.2f outcomes=%s",
             test_agent.return_mean, test_agent.accuracy_on_commits, test_agent.n_commits,
             test_agent.information_transfer_rate, test_agent.outcome_counts)

    # 8) Save artefacts
    out_dir = EXPERIMENTS_DIR / "m5_train_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), out_dir / "agent.pt")
    summary = {
        "train_seconds": train_seconds,
        "n_train_steps": 20_000,
        "n_transitions": n_transitions,
        "pre_random": pre_rand.asdict(),
        "pre_agent": pre_agent.asdict(),
        "post_agent_val": post_agent.asdict(),
        "post_agent_test": test_agent.asdict(),
        "history": [s.asdict() for s in history],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")

    log.info("Train wall time: %.1fs (%.1f kstep/s)", train_seconds, 20_000 / max(train_seconds, 1e-3) / 1000)
    log.info("M5 verification: agent return %.3f vs random %.3f (Δ=%+.3f)",
             post_agent.return_mean, pre_rand.return_mean,
             post_agent.return_mean - pre_rand.return_mean)


if __name__ == "__main__":
    main()

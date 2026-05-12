# MIT License, 2026
""": extend the within-subject ablation to Lee2019 (62-ch, 2-class).

Same protocol as cvar_cmdp_within_subject.py, but on Lee2019 (54-subject MI corpus,
62 EEG channels, 2-class). We pick subject 1 by default (index into the
moabb subject list). Lee2019 has more channels and a richer recording setup
than bci2a/bci2b, so this is a valuable third-dataset confirmation.

Configs (4):
  scalar_a1                   : scalar CQL, alpha=1, no CMDP
  scalar_a1_cmdp_eps0.10      : scalar CQL + CMDP eps=0.10
  cvar_a0.25                  : distributional CVaR-CQL alpha=0.25
  cvar_a0.25_cmdp_eps0.10     : combined

Output: experiments/lee2019_within/summary.json
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("lee2019_within")


def _slice_trial_batch(tb, idx):
    from src.data.moabb_loader import TrialBatch
    return TrialBatch(
        X=tb.X[idx].copy(), y=tb.y[idx].copy(),
        session=tb.session[idx].copy(), run=tb.run[idx].copy(),
        sfreq=tb.sfreq, ch_names=list(tb.ch_names),
        class_labels=list(tb.class_labels),
        subject_id=tb.subject_id, dataset_key=tb.dataset_key,
    )


def main(subject_id: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    tb = preprocess_subject("lee2019", subject_id=subject_id)
    n_classes = len(tb.class_labels)
    log.info("Loaded lee2019 sub %d: X=%s n_classes=%d (%s)",
             subject_id, tb.X.shape, n_classes, tb.class_labels)
    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))

    enc = build_encoder("eegnet", n_channels=tb.n_channels,
                        n_samples=win_samples, sfreq=tb.sfreq).to(device)
    rng = np.random.default_rng(0)
    perm = rng.permutation(tb.n_trials)
    n_tr = int(tb.n_trials * 0.70)
    n_va = int(tb.n_trials * 0.15)
    tr_idx = np.sort(perm[:n_tr]); va_idx = np.sort(perm[n_tr:n_tr + n_va])
    te_idx = np.sort(perm[n_tr + n_va:])
    log.info("Trial split: train=%d val=%d test=%d", len(tr_idx), len(va_idx), len(te_idx))

    win_X, win_y = [], []
    for i in tr_idx:
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
        win_X.append(W); win_y.extend([int(tb.y[i])] * W.shape[0])
    win_X = np.concatenate(win_X, axis=0)
    win_y = np.asarray(win_y, dtype=np.int64)
    log.info("Encoder pretrain: windows=%s", win_X.shape)
    pre_info = pretrain_encoder_supervised(
        enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
        n_epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.15,
    )
    log.info("Encoder pretrain val acc: %.3f", pre_info["best_val_acc"])

    subj_factory = SubjectEmbeddingFactory(encoder_embed_dim=enc.spec.embed_dim, subj_dim=16, seed=0)
    train_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, tr_idx), encoder=enc, device=device, subj_factory=subj_factory,
    )
    train_subj_embeds = (train_eps[0].subj_embed_default.copy(),
                         train_eps[0].subj_embed_post_recal.copy())
    val_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, va_idx), encoder=enc, device=device, subj_factory=subj_factory,
        subj_embeds=train_subj_embeds, calibration_indices=np.array([], dtype=np.int64),
    )
    test_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, te_idx), encoder=enc, device=device, subj_factory=subj_factory,
        subj_embeds=train_subj_embeds, calibration_indices=np.array([], dtype=np.int64),
    )
    log.info("Episodes: train=%d val=%d test=%d", len(train_eps), len(val_eps), len(test_eps))

    env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                 subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
    ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)

    rollouts = []
    for T_fix in [4, 8, 12]:
        mu1 = FixedWindowPolicy(ctx=ctx, T_fix=T_fix, env=env)
        for i, ep in enumerate(train_eps):
            rollouts.append(rollout_episode(env, ep, mu1, seed=i))
    rng_mu = np.random.default_rng(0)
    mu2 = SPRTStochasticPolicy(ctx=ctx, env=env, evidence_threshold=0.55,
                                eps_explore=0.10, rng=rng_mu)
    for i, ep in enumerate(train_eps):
        rollouts.append(rollout_episode(env, ep, mu2, seed=10_000 + i))
    n_transitions = sum(len(r) for r in rollouts)
    log.info("buffer: %d trajectories, %d transitions", len(rollouts), n_transitions)
    buffer = build_buffer_from_rollouts(
        rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL,
        a_abstain=env.A_ABSTAIN, n_classes=n_classes,
    )

    rand_val = evaluate_policy(env, val_eps, random_policy(env.action_space.n, 0),
                                n_classes=n_classes)
    rand_test = evaluate_policy(env, test_eps, random_policy(env.action_space.n, 0),
                                 n_classes=n_classes)
    log.info("RANDOM val:  ret=%+.3f acc=%.3f ITR=%.2f n_commit=%d",
             rand_val.return_mean, rand_val.accuracy_on_commits,
             rand_val.information_transfer_rate, rand_val.n_commits)
    log.info("RANDOM test: ret=%+.3f acc=%.3f ITR=%.2f n_commit=%d",
             rand_test.return_mean, rand_test.accuracy_on_commits,
             rand_test.information_transfer_rate, rand_test.n_commits)

    configs = [
        {"name": "scalar_a1",                "mode": "scalar",         "cql_alpha": 1.0, "cmdp_eps": None},
        {"name": "scalar_a1_cmdp_eps0.10",   "mode": "scalar",         "cql_alpha": 1.0, "cmdp_eps": 0.10},
        {"name": "cvar_a0.25",               "mode": "distributional", "cql_alpha": 1.0, "cmdp_eps": None},
        {"name": "cvar_a0.25_cmdp_eps0.10",  "mode": "distributional", "cql_alpha": 1.0, "cmdp_eps": 0.10},
    ]
    results = {
        "subject_id": int(tb.subject_id),
        "dataset_key": tb.dataset_key,
        "n_classes": int(n_classes),
        "class_labels": list(tb.class_labels),
        "n_channels": int(tb.n_channels),
        "n_trials": int(tb.n_trials),
        "encoder_val_acc": float(pre_info["best_val_acc"]),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_test_episodes": len(test_eps),
        "n_transitions": int(n_transitions),
        "random_val": rand_val.asdict(),
        "random_test": rand_test.asdict(),
    }

    for cfg in configs:
        cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG),
                                          "cql_alpha": cfg["cql_alpha"],
                                          "seed": 0})
        agent = NeuroPolicyAgent(
            state_dim=env.state_dim, n_actions=env.action_space.n,
            n_classes=n_classes, cfg=cfg_obj,
            mode=cfg["mode"], cmdp_eps=cfg["cmdp_eps"],
            gamma=0.99, polyak_tau=0.005, device=device,
            hidden=256, depth=3, seed=0,
        )
        t0 = time.time()
        agent.train(buffer, n_steps=20_000, batch_size=256, log_every=20_000)
        elapsed = time.time() - t0

        val = evaluate_policy(env, val_eps, lambda s: agent.select_action(s),
                              n_classes=n_classes)
        test = evaluate_policy(env, test_eps, lambda s: agent.select_action(s),
                               n_classes=n_classes)
        log.info("[%-25s] val: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d  test: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d  (%.1fs)",
                 cfg["name"],
                 val.return_mean, val.accuracy_on_commits,
                 val.information_transfer_rate, val.wrong_commit_rate, val.n_commits,
                 test.return_mean, test.accuracy_on_commits,
                 test.information_transfer_rate, test.wrong_commit_rate, test.n_commits,
                 elapsed)
        results[cfg["name"]] = {
            "config": cfg,
            "val": val.asdict(),
            "test": test.asdict(),
            "elapsed_s": float(elapsed),
        }

    out_dir = EXPERIMENTS_DIR / "lee2019_within"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    import sys
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(subject_id=sid)

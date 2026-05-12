# MIT License, 2026
"""Threshold-baseline benchmark on bci2a 4-class canonical protocol.

Goal: solidify the "~2× typical ITR" claim by running EEGNet + confidence-
threshold (the strongest hand-tuned dynamic-stopping baseline) under the
EXACT same canonical session-T → session-E split, same encoder, same env,
and same Wolpaw-formula ITR convention (commit-time basis) used for
NeuroPolicy. If NeuroPolicy still wins after this apples-to-apples
comparison, the SOTA claim is solid; if a threshold baseline ties or
beats NeuroPolicy, the claim is honestly deflated.

Threshold values: τ ∈ {0.55, 0.65, 0.75, 0.85, 0.90}. Also includes
EEGNet-fixed (commit at trial end) as the canonical mandatory-classifier
ITR reference.

3 subjects × 5 seeds (matches neuropolicy_canonical_bci2a.py).

Output: experiments/canonical_bci2a_threshold_baselines/summary.json
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
    FixedWindowPolicy, SPRTStochasticPolicy, fit_running_mean_classifier,
    rollout_episode,
)
from src.training.bci_env import BCIEnv
from src.training.episode_builder import (
    SubjectEmbeddingFactory, build_episodes_for_trial_batch, windows_from_trial,
)
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("canonical_threshold_baselines")

SUBJECTS = [1, 3, 7]
SEEDS = [0, 1, 2, 3, 4]
THRESHOLDS = [0.55, 0.65, 0.75, 0.85, 0.90]


def _slice_trial_batch(tb, idx):
    from src.data.moabb_loader import TrialBatch
    return TrialBatch(
        X=tb.X[idx].copy(), y=tb.y[idx].copy(),
        session=tb.session[idx].copy(), run=tb.run[idx].copy(),
        sfreq=tb.sfreq, ch_names=list(tb.ch_names),
        class_labels=list(tb.class_labels),
        subject_id=tb.subject_id, dataset_key=tb.dataset_key,
    )


def canonical_split(tb, val_frac=0.20, seed=0):
    train_pool = np.where(tb.session == 0)[0]
    test_idx = np.where(tb.session == 1)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_pool)
    n_val = int(round(val_frac * len(train_pool)))
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    return tr_idx, val_idx, np.sort(test_idx)


def make_eegnet_fixed_policy(ctx, env):
    """Mandatory: defer until last step, then commit argmax."""
    embed_dim = env.embed_dim
    elapsed_idx = 3 * embed_dim
    def select(state):
        elapsed = float(state[elapsed_idx])
        if elapsed < 0.999:
            return env.A_DEFER
        running_mean = state[embed_dim:2*embed_dim].reshape(1, -1)
        return int(ctx.classifier.predict(running_mean)[0])
    return select


def make_eegnet_threshold_policy(ctx, env, threshold):
    """Confidence-threshold dynamic stopping."""
    embed_dim = env.embed_dim
    elapsed_idx = 3 * embed_dim
    def select(state):
        running_mean = state[embed_dim:2*embed_dim].reshape(1, -1)
        proba = ctx.classifier.predict_proba(running_mean)[0]
        elapsed = float(state[elapsed_idx])
        force = elapsed >= 0.999
        if proba.max() >= threshold or force:
            return int(np.argmax(proba))
        return env.A_DEFER
    return select


def build_subject_episodes(tb, device, seed):
    """Canonical-protocol episode building."""
    n_classes = len(tb.class_labels)
    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    tr_idx, va_idx, te_idx = canonical_split(tb, val_frac=0.20, seed=0)

    enc = build_encoder("eegnet", n_channels=tb.n_channels,
                          n_samples=win_samples, sfreq=tb.sfreq).to(device)
    win_X, win_y = [], []
    for i in tr_idx:
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
        win_X.append(W); win_y.extend([int(tb.y[i])] * W.shape[0])
    win_X = np.concatenate(win_X, axis=0); win_y = np.asarray(win_y, dtype=np.int64)
    pre_info = pretrain_encoder_supervised(
        enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
        n_epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=seed, val_frac=0.15,
    )

    subj_factory = SubjectEmbeddingFactory(encoder_embed_dim=enc.spec.embed_dim,
                                            subj_dim=16, seed=0)
    train_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, tr_idx), encoder=enc, device=device, subj_factory=subj_factory,
    )
    train_subj_embeds = (train_eps[0].subj_embed_default.copy(),
                          train_eps[0].subj_embed_post_recal.copy())
    test_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, te_idx), encoder=enc, device=device, subj_factory=subj_factory,
        subj_embeds=train_subj_embeds, calibration_indices=np.array([], dtype=np.int64),
    )
    env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                  subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
    ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)
    return env, ctx, test_eps, n_classes, float(pre_info["best_val_acc"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "canonical_bci2a_threshold_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"subjects": SUBJECTS, "seeds": SEEDS, "thresholds": THRESHOLDS,
                "per_subject": {}}

    for sid in SUBJECTS:
        log.info("\n========== SUBJECT %d ==========", sid)
        tb = preprocess_subject("bci2a", subject_id=sid)
        sub_results = {"per_seed": {}}
        for seed in SEEDS:
            log.info("  seed %d", seed)
            torch.manual_seed(seed); np.random.seed(seed)
            env, ctx, test_eps, n_classes, val_acc = build_subject_episodes(tb, device, seed)
            seed_res = {"encoder_val_acc": val_acc}
            # Mandatory baseline (commit at trial end)
            fixed_pol = make_eegnet_fixed_policy(ctx, env)
            m_fix = evaluate_policy(env, test_eps, fixed_pol, n_classes=n_classes)
            log.info("    eegnet_fixed:   ret=%+.3f acc=%.3f ITR=%.2f commit_rate=%.2f wrong=%.3f",
                      m_fix.return_mean, m_fix.accuracy_on_commits,
                      m_fix.information_transfer_rate,
                      m_fix.n_commits / len(test_eps), m_fix.wrong_commit_rate)
            seed_res["eegnet_fixed"] = m_fix.asdict()
            # Threshold variants
            for tau in THRESHOLDS:
                pol = make_eegnet_threshold_policy(ctx, env, threshold=tau)
                m = evaluate_policy(env, test_eps, pol, n_classes=n_classes)
                log.info("    eegnet_thr_%.2f: ret=%+.3f acc=%.3f ITR=%.2f commit_rate=%.2f wrong=%.3f",
                          tau, m.return_mean, m.accuracy_on_commits,
                          m.information_transfer_rate,
                          m.n_commits / len(test_eps), m.wrong_commit_rate)
                seed_res[f"eegnet_thr_{tau}"] = m.asdict()
            sub_results["per_seed"][seed] = seed_res
            results["per_subject"][sid] = sub_results
            (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

        # Per-subject aggregate across seeds
        agg = {}
        for cfg in ["eegnet_fixed"] + [f"eegnet_thr_{t}" for t in THRESHOLDS]:
            accs = [sub_results["per_seed"][s][cfg]["accuracy_on_commits"] for s in SEEDS]
            itrs = [sub_results["per_seed"][s][cfg]["information_transfer_rate"] for s in SEEDS]
            rets = [sub_results["per_seed"][s][cfg]["return_mean"] for s in SEEDS]
            wrong = [sub_results["per_seed"][s][cfg]["wrong_commit_rate"] for s in SEEDS]
            n_commits = [sub_results["per_seed"][s][cfg]["n_commits"] for s in SEEDS]
            n_te = len(test_eps)  # last value, all seeds same test set
            agg[cfg] = {
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)),
                "itr_mean": float(np.mean(itrs)), "itr_std": float(np.std(itrs, ddof=1)),
                "ret_mean": float(np.mean(rets)), "ret_std": float(np.std(rets, ddof=1)),
                "wrong_mean": float(np.mean(wrong)),
                "commit_rate_mean": float(np.mean([c / n_te for c in n_commits])),
            }
        sub_results["aggregate"] = agg
        results["per_subject"][sid] = sub_results
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    # Cross-subject aggregate
    log.info("\n=== CROSS-SUBJECT AGGREGATE ===")
    cross = {}
    for cfg in ["eegnet_fixed"] + [f"eegnet_thr_{t}" for t in THRESHOLDS]:
        accs = [results["per_subject"][s]["aggregate"][cfg]["acc_mean"] for s in SUBJECTS]
        itrs = [results["per_subject"][s]["aggregate"][cfg]["itr_mean"] for s in SUBJECTS]
        crates = [results["per_subject"][s]["aggregate"][cfg]["commit_rate_mean"] for s in SUBJECTS]
        cross[cfg] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)),
            "itr_mean": float(np.mean(itrs)), "itr_std": float(np.std(itrs, ddof=1)),
            "commit_rate_mean": float(np.mean(crates)),
        }
        log.info("  %s: acc=%.3f±%.3f ITR=%.2f±%.2f commit_rate=%.2f",
                  cfg, cross[cfg]["acc_mean"], cross[cfg]["acc_std"],
                  cross[cfg]["itr_mean"], cross[cfg]["itr_std"],
                  cross[cfg]["commit_rate_mean"])
    results["cross_subject"] = cross
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("\nWrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

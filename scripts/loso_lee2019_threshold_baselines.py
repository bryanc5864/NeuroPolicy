# MIT License, 2026
"""Threshold-baseline LOSO benchmark on Lee2019 (62-ch) — apples-to-apples
comparison to NeuroPolicy's Lee2019 LOSO claim.

The Lee2019 LOSO classification benchmark (loso_classification_lee2019.py)
currently compares NeuroPolicy only to a random baseline. A reviewer will
ask: does a simple EEGNet+threshold classifier ALSO beat random on
Lee2019 LOSO? If so, the "10/10 wins p=0.001" claim looks less novel.

This script runs the same EEGNet encoder pretraining and same canonical
LOSO setup, then evaluates 6 baselines through the same BCIEnv:
  - random
  - EEGNet-fixed (mandatory commit at trial end)
  - EEGNet+threshold tau in {0.55, 0.65, 0.75, 0.85, 0.90}

Same 10-subject LOSO subset as NeuroPolicy. Direct apples-to-apples
commit-time-ITR comparison.

Output: experiments/loso_lee2019_threshold_baselines/summary.json
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
from src.training.behavior_policies import fit_running_mean_classifier
from src.training.bci_env import BCIEnv
from src.training.episode_builder import (
    SubjectEmbeddingFactory, build_episodes_for_trial_batch, windows_from_trial,
)
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("loso_lee2019_baselines")

SUBJECTS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
THRESHOLDS = [0.55, 0.65, 0.75, 0.85, 0.90]


def make_eegnet_fixed_policy(ctx, env):
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


def _windowize(tb_list):
    Xs, ys = [], []
    for tb in tb_list:
        for i in range(tb.n_trials):
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
            Xs.append(W); ys.extend([int(tb.y[i])] * W.shape[0])
    return np.concatenate(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s subjects=%s", device, SUBJECTS)
    out_dir = EXPERIMENTS_DIR / "loso_lee2019_threshold_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Preprocessing %d Lee2019 subjects...", len(SUBJECTS))
    tbs = {}
    for sid in SUBJECTS:
        try:
            t0 = time.time()
            tbs[sid] = preprocess_subject("lee2019", subject_id=sid)
            log.info("  sub %d: %d trials (%.1fs)", sid, tbs[sid].n_trials, time.time() - t0)
        except Exception as exc:
            log.error("FAILED sub %d: %s", sid, exc)

    sids = sorted(tbs.keys())
    n_classes = len(tbs[sids[0]].class_labels)
    sfreq = tbs[sids[0]].sfreq
    n_channels = tbs[sids[0]].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))

    results = {"subjects": sids, "thresholds": THRESHOLDS,
                "encoder": "eegnet", "dataset": "lee2019",
                "per_subject": []}

    for held_out in sids:
        train_subjs = [s for s in sids if s != held_out]
        log.info("\n=== LOSO held-out=%d ===", held_out)
        train_tb_list = [tbs[s] for s in train_subjs]
        win_X, win_y = _windowize(train_tb_list)
        log.info("  pretrain set: %d windows", win_X.shape[0])

        enc = build_encoder("eegnet", n_channels=n_channels,
                              n_samples=win_samples, sfreq=sfreq).to(device)
        t0 = time.time()
        pre_info = pretrain_encoder_supervised(
            enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
            n_epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.10,
        )
        log.info("  encoder val acc: %.3f (%.1fs)", pre_info["best_val_acc"], time.time() - t0)

        subj_factory = SubjectEmbeddingFactory(encoder_embed_dim=enc.spec.embed_dim,
                                                subj_dim=16, seed=0)
        train_eps_per_subj = []
        for tb in train_tb_list:
            eps, _, _ = build_episodes_for_trial_batch(tb, encoder=enc, device=device,
                                                         subj_factory=subj_factory)
            train_eps_per_subj.append(eps)
        defaults = np.stack([eps[0].subj_embed_default for eps in train_eps_per_subj])
        post_recals = np.stack([eps[0].subj_embed_post_recal for eps in train_eps_per_subj])
        train_subj_embeds = (defaults.mean(0).astype(np.float32),
                              post_recals.mean(0).astype(np.float32))
        train_eps = []
        for tb in train_tb_list:
            eps, _, _ = build_episodes_for_trial_batch(
                tb, encoder=enc, device=device, subj_factory=subj_factory,
                subj_embeds=train_subj_embeds,
                calibration_indices=np.array([], dtype=np.int64),
            )
            train_eps.extend(eps)
        held_eps, _, _ = build_episodes_for_trial_batch(
            tbs[held_out], encoder=enc, device=device, subj_factory=subj_factory,
            subj_embeds=train_subj_embeds, calibration_indices=np.array([], dtype=np.int64),
        )

        env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                      subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
        ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)

        sub_results = {"held_out": held_out, "encoder_val_acc": float(pre_info["best_val_acc"]),
                        "baselines": {}}

        # Random baseline
        m = evaluate_policy(env, held_eps, random_policy(env.action_space.n, 0),
                              n_classes=n_classes)
        sub_results["baselines"]["random"] = m.asdict()
        log.info("    random:           acc=%.3f ITR=%.2f", m.accuracy_on_commits, m.information_transfer_rate)

        # EEGNet-fixed
        m = evaluate_policy(env, held_eps, make_eegnet_fixed_policy(ctx, env),
                              n_classes=n_classes)
        sub_results["baselines"]["eegnet_fixed"] = m.asdict()
        log.info("    eegnet_fixed:     acc=%.3f ITR=%.2f commit_rate=%.2f wrong=%.3f",
                  m.accuracy_on_commits, m.information_transfer_rate,
                  m.n_commits / len(held_eps), m.wrong_commit_rate)

        # Thresholds
        for tau in THRESHOLDS:
            m = evaluate_policy(env, held_eps,
                                  make_eegnet_threshold_policy(ctx, env, tau),
                                  n_classes=n_classes)
            sub_results["baselines"][f"eegnet_thr_{tau}"] = m.asdict()
            log.info("    eegnet_thr_%.2f:   acc=%.3f ITR=%.2f commit_rate=%.2f wrong=%.3f",
                      tau, m.accuracy_on_commits, m.information_transfer_rate,
                      m.n_commits / len(held_eps), m.wrong_commit_rate)

        results["per_subject"].append(sub_results)
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    # Aggregate
    log.info("\n=== AGGREGATE Lee2019 LOSO baselines ===")
    n_te = len(held_eps)
    agg = {}
    for cfg in ["random", "eegnet_fixed"] + [f"eegnet_thr_{t}" for t in THRESHOLDS]:
        accs = [r["baselines"][cfg]["accuracy_on_commits"] for r in results["per_subject"]]
        itrs = [r["baselines"][cfg]["information_transfer_rate"] for r in results["per_subject"]]
        wrong = [r["baselines"][cfg]["wrong_commit_rate"] for r in results["per_subject"]]
        ncm = [r["baselines"][cfg]["n_commits"] / n_te for r in results["per_subject"]]
        agg[cfg] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)),
            "itr_mean": float(np.mean(itrs)), "itr_std": float(np.std(itrs, ddof=1)),
            "wrong_mean": float(np.mean(wrong)),
            "commit_rate_mean": float(np.mean(ncm)),
        }
        log.info("  %-20s: acc=%.3f+-%.3f ITR=%.2f+-%.2f wrong=%.3f commit_rate=%.2f",
                  cfg, agg[cfg]["acc_mean"], agg[cfg]["acc_std"],
                  agg[cfg]["itr_mean"], agg[cfg]["itr_std"],
                  agg[cfg]["wrong_mean"], agg[cfg]["commit_rate_mean"])
    results["aggregate"] = agg
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

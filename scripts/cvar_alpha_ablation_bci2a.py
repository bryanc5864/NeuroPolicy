# MIT License, 2026
"""CVaR alpha ablation on canonical bci2a.

CVaR alpha is an *eval-time* risk parameter for distributional agents:
the same trained Z-value distribution is collapsed to scalar Q under
different left-tail fractions alpha in {0.10, 0.25, 0.50, 1.00}.
alpha = 1.0 reduces to plain expected-value Q (no risk aversion).

This script trains the standard distributional-CQL + CMDP agent once
per (subject, seed) with Conformer encoder under the same canonical
session-T -> session-E protocol used in neuropolicy_canonical_bci2a_conformer.py
and reports the Pareto frontier across alpha values, characterising
how risk aversion trades commit accuracy for throughput.

Subjects: {1, 3, 7} x 5 seeds, ~3h GPU.

Output: experiments/cvar_alpha_ablation_bci2a/summary.json
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

from src.data.moabb_loader import TrialBatch
from src.data.preprocess import preprocess_subject
from src.evaluation.policy_eval import evaluate_policy
from src.models.encoder import build_encoder, pretrain_encoder_supervised
from src.training.bci_env import BCIEnv
from src.training.episode_builder import (
    SubjectEmbeddingFactory, build_episodes_for_trial_batch, windows_from_trial,
)
from src.training.neuropolicy_agent import NeuroPolicyAgent
from src.training.replay_buffer import build_buffer_from_rollouts
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG, TRAIN_CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("cvar_alpha_ablation")

SUBJECTS = [1, 3, 7]
SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [0.10, 0.25, 0.50, 1.00]
ENCODER = "conformer"


def _slice_trial_batch(tb, idx):
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


def build_buffer_for_subject(tb, device, split_seed=0, encoder_name=ENCODER):
    n_classes = len(tb.class_labels)
    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    tr_idx, va_idx, te_idx = canonical_split(tb, val_frac=0.20, seed=split_seed)
    log.info("Canonical split: train=%d val=%d test=%d",
              len(tr_idx), len(va_idx), len(te_idx))

    enc = build_encoder(encoder_name, n_channels=tb.n_channels,
                          n_samples=win_samples, sfreq=tb.sfreq).to(device)
    win_X, win_y = [], []
    for i in tr_idx:
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
        win_X.append(W); win_y.extend([int(tb.y[i])] * W.shape[0])
    win_X = np.concatenate(win_X, axis=0); win_y = np.asarray(win_y, dtype=np.int64)
    pre_info = pretrain_encoder_supervised(
        enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
        n_epochs=20, batch_size=128, lr=2e-4, weight_decay=1e-4, seed=0, val_frac=0.15,
    )
    log.info("Encoder: %s, embed_dim=%d", enc.spec.name, enc.spec.embed_dim)
    log.info("Encoder val acc: %.3f", pre_info["best_val_acc"])

    subj_factory = SubjectEmbeddingFactory(encoder_embed_dim=enc.spec.embed_dim,
                                            subj_dim=16, seed=0)
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

    env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                  subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
    buffer = build_buffer_from_rollouts(train_eps, env=env, n_classes=n_classes, seed=0)
    return env, buffer, train_eps, val_eps, test_eps, n_classes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s, encoder=%s", device, ENCODER)
    out_dir = EXPERIMENTS_DIR / "cvar_alpha_ablation_bci2a"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("%d subjects x %d seeds x %d alphas (eval-time) = %d (sub,seed) trainings",
              len(SUBJECTS), len(SEEDS), len(ALPHAS),
              len(SUBJECTS) * len(SEEDS))
    results = {"protocol": "canonical_T_to_E", "encoder": ENCODER,
                "subjects": SUBJECTS, "seeds": SEEDS, "alphas": ALPHAS,
                "per_subject": {}}

    for sid in SUBJECTS:
        log.info("\n========== SUBJECT %d ==========", sid)
        tb = preprocess_subject("bci2a", subject_id=sid)
        env, buffer, train_eps, val_eps, test_eps, n_classes = build_buffer_for_subject(tb, device)
        sub_results = {"per_seed": {}}
        for seed in SEEDS:
            log.info("\n--- subject %d seed %d ---", sid, seed)
            torch.manual_seed(seed); np.random.seed(seed)
            cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG),
                                              "cql_alpha": 1.0,
                                              "seed": seed})
            agent = NeuroPolicyAgent(
                state_dim=env.state_dim, n_actions=env.action_space.n,
                n_classes=n_classes, cfg=cfg_obj,
                mode="distributional", cmdp_eps=0.10,
                gamma=0.99, polyak_tau=0.005, device=device,
                hidden=256, depth=3, seed=seed,
            )
            t0 = time.time()
            agent.train(buffer, n_steps=20_000, batch_size=256, log_every=100_000)
            elapsed = time.time() - t0
            log.info("  training done in %.1fs", elapsed)
            sub_results["per_seed"][seed] = {"elapsed_s": elapsed, "alphas": {}}
            for alpha in ALPHAS:
                test = evaluate_policy(env, test_eps,
                                          lambda s, a=alpha: agent.select_action(s, cvar_alpha=a),
                                          n_classes=n_classes)
                log.info("  alpha=%.2f test: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d/%d",
                          alpha, test.return_mean, test.accuracy_on_commits,
                          test.information_transfer_rate, test.wrong_commit_rate,
                          test.n_commits, len(test_eps))
                sub_results["per_seed"][seed]["alphas"][f"{alpha:.2f}"] = test.asdict()
            results["per_subject"][sid] = sub_results
            (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

        # Per-subject aggregate.
        agg = {}
        for alpha in ALPHAS:
            key = f"{alpha:.2f}"
            accs = [sub_results["per_seed"][s]["alphas"][key]["accuracy_on_commits"] for s in SEEDS]
            itrs = [sub_results["per_seed"][s]["alphas"][key]["information_transfer_rate"] for s in SEEDS]
            wrongs = [sub_results["per_seed"][s]["alphas"][key]["wrong_commit_rate"] for s in SEEDS]
            n_te = len(test_eps)
            ncm = [sub_results["per_seed"][s]["alphas"][key]["n_commits"] / n_te for s in SEEDS]
            agg[key] = {
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0),
                "itr_mean": float(np.mean(itrs)), "itr_std": float(np.std(itrs, ddof=1) if len(itrs) > 1 else 0),
                "wrong_mean": float(np.mean(wrongs)),
                "commit_rate_mean": float(np.mean(ncm)),
            }
        sub_results["aggregate"] = agg
        results["per_subject"][sid] = sub_results
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    # Cross-subject aggregate.
    log.info("\n=== CROSS-SUBJECT AGGREGATE ===")
    cross = {}
    for alpha in ALPHAS:
        key = f"{alpha:.2f}"
        accs = [results["per_subject"][s]["aggregate"][key]["acc_mean"] for s in SUBJECTS]
        itrs = [results["per_subject"][s]["aggregate"][key]["itr_mean"] for s in SUBJECTS]
        wrongs = [results["per_subject"][s]["aggregate"][key]["wrong_mean"] for s in SUBJECTS]
        crates = [results["per_subject"][s]["aggregate"][key]["commit_rate_mean"] for s in SUBJECTS]
        cross[key] = {
            "acc_mean_of_means": float(np.mean(accs)),
            "acc_std_of_means": float(np.std(accs, ddof=1) if len(accs) > 1 else 0),
            "itr_mean_of_means": float(np.mean(itrs)),
            "itr_std_of_means": float(np.std(itrs, ddof=1) if len(itrs) > 1 else 0),
            "wrong_mean_of_means": float(np.mean(wrongs)),
            "commit_rate_mean_of_means": float(np.mean(crates)),
        }
        log.info("  alpha=%s: acc=%.3f+-%.3f ITR=%.2f+-%.2f wrong=%.3f commit_rate=%.2f",
                  key, cross[key]["acc_mean_of_means"], cross[key]["acc_std_of_means"],
                  cross[key]["itr_mean_of_means"], cross[key]["itr_std_of_means"],
                  cross[key]["wrong_mean_of_means"], cross[key]["commit_rate_mean_of_means"])
    results["cross_subject"] = cross
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

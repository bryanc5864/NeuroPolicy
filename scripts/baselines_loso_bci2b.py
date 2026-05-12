# MIT License, 2026
"""Baseline LOSO sweep on BCI-IV-2b.

Same pipeline as  (encoder pretrain → episode build → eval) but evaluating
non-RL baselines instead of NeuroPolicy. Apples-to-apples with 's cql_a1 row
of summary.json (same encoder weights per held-out subject given fixed seed).

Baselines evaluated (all consume the same held_eps via env.evaluate_policy):
  random              : uniform over A_DEFER, K commit actions, A_RECAL, A_ABSTAIN
  csp_lda_fixed       : CSP+LDA on training subjects' trials → predict held-out
                        trial → emit DEFER until last step, then commit prediction.
  eegnet_fixed        : EEGNet running-mean-classifier; defer until last step,
                        then commit argmax. Uses the SAME encoder NeuroPolicy uses.
  eegnet_threshold_T  : EEGNet running-mean classifier with confidence threshold τ
                        for early commit; if not reached by t=T-1, commit forced.
                        We sweep τ ∈ {0.55, 0.65, 0.75} and report all three; the
                        per-subject best is the upper-bound for hand-tuned dynamic
                        stopping baselines.

The trained-encoder running-mean classifier is the SAME `_PolicyContext` we use
in 's behavior policies — so eegnet_fixed and eegnet_threshold are literally
"the  logging-policy heuristics evaluated as final policies."

Output: experiments/baselines_loso_bci2b/summary.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from src.data.preprocess import preprocess_subject
from src.evaluation.policy_eval import evaluate_policy, random_policy
from src.models.encoder import build_encoder, pretrain_encoder_supervised
from src.training.behavior_policies import fit_running_mean_classifier
from src.training.bci_env import BCIEnv
from src.training.episode_builder import (
    SubjectEmbeddingFactory,
    build_episodes_for_trial_batch,
    windows_from_trial,
)
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("baselines_loso_bci2b")


# ---------------------------------------------------------------------------
# State-only baseline policies (compatible with evaluate_policy)
# ---------------------------------------------------------------------------
def make_eegnet_fixed_policy(ctx, env: BCIEnv):
    """Defer until elapsed_norm == 1.0 (last step), then commit argmax."""
    embed_dim = env.embed_dim
    elapsed_idx = 3 * embed_dim

    def select(state):
        elapsed = float(state[elapsed_idx])
        if elapsed < 0.999:
            return env.A_DEFER
        running_mean = state[embed_dim:2 * embed_dim].reshape(1, -1)
        return int(ctx.classifier.predict(running_mean)[0])
    return select


def make_eegnet_threshold_policy(ctx, env: BCIEnv, threshold: float):
    """Commit argmax as soon as max class prob ≥ threshold; else DEFER."""
    embed_dim = env.embed_dim
    elapsed_idx = 3 * embed_dim

    def select(state):
        running_mean = state[embed_dim:2 * embed_dim].reshape(1, -1)
        proba = ctx.classifier.predict_proba(running_mean)[0]
        elapsed = float(state[elapsed_idx])
        force = elapsed >= 0.999
        if proba.max() >= threshold or force:
            return int(np.argmax(proba))
        return env.A_DEFER
    return select


def make_csp_lda_fixed_policy(env: BCIEnv, pred_by_episode_id: dict):
    """Defer until last step, then commit precomputed CSP+LDA prediction.

    `pred_by_episode_id` maps `id(env._episode)` → predicted class.
    """
    embed_dim = env.embed_dim
    elapsed_idx = 3 * embed_dim

    def select(state):
        elapsed = float(state[elapsed_idx])
        if elapsed < 0.999:
            return env.A_DEFER
        ep_id = id(env._episode)
        return int(pred_by_episode_id[ep_id])
    return select


# ---------------------------------------------------------------------------
# Helpers shared with LOSO baseline runner
# ---------------------------------------------------------------------------
def _windowize_for_pretrain(tb_list, mdp_cfg):
    Xs, ys = [], []
    for tb in tb_list:
        for i in range(tb.n_trials):
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=mdp_cfg)
            Xs.append(W); ys.extend([int(tb.y[i])] * W.shape[0])
    return np.concatenate(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def _pool_trials(tb_list):
    """Concat raw trials across subjects for CSP fitting."""
    X = np.concatenate([tb.X for tb in tb_list], axis=0)
    y = np.concatenate([tb.y for tb in tb_list], axis=0)
    return X, y


def fit_csp_lda(X_train: np.ndarray, y_train: np.ndarray,
                 n_components: int = 4, seed: int = 0) -> Pipeline:
    n_comp = min(n_components, X_train.shape[1] - 1)
    n_comp = max(n_comp, 2)
    pipe = Pipeline([
        ("csp", CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------------
# Main LOSO loop
# ---------------------------------------------------------------------------
def run_loso(subjects: list[int], dataset_key: str = "bci2b",
              csp_n_components: int = 2,
              eegnet_thresholds: tuple[float, ...] = (0.55, 0.65, 0.75)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    out_dir = EXPERIMENTS_DIR / "baselines_loso_bci2b"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    log.info("Preprocessing %d subjects...", len(subjects))
    subject_tbs = {}
    for sid in subjects:
        t0 = time.time()
        subject_tbs[sid] = preprocess_subject(dataset_key, sid)
        log.info("  sub %d: %d trials (%.1fs)", sid, subject_tbs[sid].n_trials,
                 time.time() - t0)

    n_classes = len(subject_tbs[subjects[0]].class_labels)
    sfreq = subject_tbs[subjects[0]].sfreq
    n_channels = subject_tbs[subjects[0]].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))

    all_results: list[dict] = []
    for held_out in subjects:
        train_subjs = [s for s in subjects if s != held_out]
        log.info("\n=== LOSO held-out=%d  train_subjs=%s ===", held_out, train_subjs)
        run_dir = out_dir / f"sub{held_out:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- 1) Encoder pretrain (matches  exactly: same seed, same hparams) ---
        train_tb_list = [subject_tbs[s] for s in train_subjs]
        win_X, win_y = _windowize_for_pretrain(train_tb_list, MDP_CFG)
        enc = build_encoder("eegnet", n_channels=n_channels, n_samples=win_samples,
                            sfreq=sfreq).to(device)
        pre_info = pretrain_encoder_supervised(
            enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
            n_epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-4,
            seed=0, val_frac=0.10,
        )
        log.info("  encoder val acc: %.3f", pre_info["best_val_acc"])

        # --- 2) Build episodes (same as ) ---
        subj_factory = SubjectEmbeddingFactory(
            encoder_embed_dim=enc.spec.embed_dim, subj_dim=16, seed=0,
        )
        train_eps_per_subj = []
        for tb in train_tb_list:
            eps, _, _ = build_episodes_for_trial_batch(
                tb, encoder=enc, device=device, subj_factory=subj_factory,
            )
            train_eps_per_subj.append(eps)
        defaults = np.stack([eps[0].subj_embed_default for eps in train_eps_per_subj])
        post_recals = np.stack([eps[0].subj_embed_post_recal for eps in train_eps_per_subj])
        pooled_default = defaults.mean(axis=0).astype(np.float32)
        pooled_post_recal = post_recals.mean(axis=0).astype(np.float32)
        train_subj_embeds = (pooled_default, pooled_post_recal)

        train_eps = []
        for tb in train_tb_list:
            eps, _, _ = build_episodes_for_trial_batch(
                tb, encoder=enc, device=device, subj_factory=subj_factory,
                subj_embeds=train_subj_embeds,
                calibration_indices=np.array([], dtype=np.int64),
            )
            train_eps.extend(eps)
        held_eps, _, _ = build_episodes_for_trial_batch(
            subject_tbs[held_out], encoder=enc, device=device, subj_factory=subj_factory,
            subj_embeds=train_subj_embeds,
            calibration_indices=np.array([], dtype=np.int64),
        )
        log.info("  episodes: train=%d  held=%d", len(train_eps), len(held_eps))

        # --- 3) Env + EEGNet running-mean classifier ---
        env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                     subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
        ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)

        # --- 4) Fit CSP+LDA on raw training subjects' trials, predict held-out ---
        Xtr, ytr = _pool_trials(train_tb_list)
        Xte, yte = subject_tbs[held_out].X, subject_tbs[held_out].y
        # held_eps is a subset of held-out trials (skip cal_indices=empty → all)
        # but episode_builder may have filtered — check via subject id ordering.
        # Our build_episodes_for_trial_batch with calibration_indices=[] returns
        # one episode per trial in tb order. So ep i ↔ Xte[i].
        try:
            t0 = time.time()
            csp_pipe = fit_csp_lda(Xtr, ytr, n_components=csp_n_components)
            te_preds = csp_pipe.predict(Xte)
            csp_train_acc = float(csp_pipe.score(Xtr, ytr))
            csp_te_acc_offline = float((te_preds == yte).mean())
            log.info("  CSP+LDA: train_acc=%.3f, held-out raw acc=%.3f (%.1fs)",
                     csp_train_acc, csp_te_acc_offline, time.time() - t0)
            # Map id(ep) → prediction for held_eps
            pred_by_id = {id(ep): int(te_preds[i]) for i, ep in enumerate(held_eps)}
            csp_lda_ok = True
        except Exception as exc:
            log.warning("CSP+LDA failed: %s", exc)
            csp_train_acc = float("nan"); csp_te_acc_offline = float("nan")
            pred_by_id = {}
            csp_lda_ok = False

        # --- 5) Evaluate baselines via env (apples-to-apples with ) ---
        per_subj: dict[str, dict] = {}

        rand_metrics = evaluate_policy(env, held_eps,
                                        random_policy(env.action_space.n, 0),
                                        n_classes=n_classes)
        per_subj["random"] = rand_metrics.asdict()
        log.info("  RANDOM:                 ret=%+.3f acc=%.3f ITR=%.2f",
                 rand_metrics.return_mean, rand_metrics.accuracy_on_commits,
                 rand_metrics.information_transfer_rate)

        eegnet_fixed = make_eegnet_fixed_policy(ctx, env)
        m = evaluate_policy(env, held_eps, eegnet_fixed, n_classes=n_classes)
        per_subj["eegnet_fixed"] = m.asdict()
        log.info("  eegnet_fixed:            ret=%+.3f acc=%.3f ITR=%.2f n_commit=%d",
                 m.return_mean, m.accuracy_on_commits, m.information_transfer_rate,
                 m.n_commits)

        for thr in eegnet_thresholds:
            pol = make_eegnet_threshold_policy(ctx, env, threshold=thr)
            m = evaluate_policy(env, held_eps, pol, n_classes=n_classes)
            per_subj[f"eegnet_threshold_{thr:.2f}"] = m.asdict()
            log.info("  eegnet_thr=%.2f:        ret=%+.3f acc=%.3f ITR=%.2f n_commit=%d lat=%.2fs",
                     thr, m.return_mean, m.accuracy_on_commits,
                     m.information_transfer_rate, m.n_commits,
                     m.mean_decision_seconds if m.mean_decision_seconds == m.mean_decision_seconds else float("nan"))

        if csp_lda_ok:
            csp_pol = make_csp_lda_fixed_policy(env, pred_by_id)
            m = evaluate_policy(env, held_eps, csp_pol, n_classes=n_classes)
            per_subj["csp_lda_fixed"] = m.asdict()
            log.info("  csp_lda_fixed:           ret=%+.3f acc=%.3f ITR=%.2f n_commit=%d",
                     m.return_mean, m.accuracy_on_commits,
                     m.information_transfer_rate, m.n_commits)

        per_subj["_meta"] = {
            "encoder_val_acc": float(pre_info["best_val_acc"]),
            "csp_train_acc": csp_train_acc,
            "csp_held_offline_acc": csp_te_acc_offline,
            "n_held_episodes": len(held_eps),
        }
        all_results.append({"held_out": held_out, "results": per_subj})
        # Persist after each subject (resumability)
        summary_path.write_text(json.dumps(
            {"per_subject": all_results, "subjects": subjects,
             "dataset": dataset_key,
             "eegnet_thresholds": list(eegnet_thresholds)}, indent=2))

    # --- 6) Aggregate per-baseline ---
    log.info("\n=== AGGREGATE ===")
    baseline_names = [k for k in all_results[0]["results"] if k != "_meta"]
    rows = []
    for name in baseline_names:
        rets = [r["results"][name]["return_mean"] for r in all_results]
        accs = [r["results"][name]["accuracy_on_commits"] for r in all_results]
        itrs = [r["results"][name]["information_transfer_rate"] for r in all_results]
        n_commits = [r["results"][name]["n_commits"] for r in all_results]
        wrong = [r["results"][name]["wrong_commit_rate"] for r in all_results]
        latencies = [r["results"][name]["mean_decision_seconds"] for r in all_results]
        latencies = [x for x in latencies if x == x]  # drop nan
        log.info("[%-22s] ret=%+.3f±%.2f  acc=%.3f±%.3f  ITR=%.1f±%.1f  lat=%.2f±%.2f  wrong=%.3f",
                 name, np.mean(rets), np.std(rets, ddof=1),
                 np.mean(accs), np.std(accs, ddof=1),
                 np.mean(itrs), np.std(itrs, ddof=1),
                 np.mean(latencies) if latencies else float("nan"),
                 np.std(latencies, ddof=1) if len(latencies) > 1 else 0.0,
                 np.mean(wrong))
        rows.append({"baseline": name,
                     "return_mean": float(np.mean(rets)),
                     "return_std": float(np.std(rets, ddof=1)),
                     "acc_mean": float(np.mean(accs)),
                     "acc_std": float(np.std(accs, ddof=1)),
                     "itr_mean": float(np.mean(itrs)),
                     "itr_std": float(np.std(itrs, ddof=1)),
                     "n_commits_mean": float(np.mean(n_commits)),
                     "wrong_rate_mean": float(np.mean(wrong)),
                     "latency_mean": float(np.mean(latencies)) if latencies else float("nan")})

    summary = {"per_subject": all_results, "aggregate": rows,
               "subjects": subjects, "dataset": dataset_key,
               "eegnet_thresholds": list(eegnet_thresholds)}
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", summary_path)


if __name__ == "__main__":
    subjects = list(range(1, 10))
    run_loso(subjects=subjects, dataset_key="bci2b")

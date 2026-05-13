# MIT License, 2026
"""NeuroPolicy with EEG-Conformer encoder on canonical bci2a 4-class, all 9 subjects.

Drop-in encoder swap from EEGNet (~1.5K params) to EEG-Conformer (~25K
params with 6 transformer layers) under the SAME canonical session-T -> E
protocol, SAME 3 subjects, SAME 5 seeds, SAME RL training budget. Tests
whether a stronger encoder lifts NeuroPolicy off the
encoder-quality ceiling (~86% commit accuracy with EEGNet).

Hypothesis: if encoder mandatory accuracy on {1, 3, 7} is ~88% with
Conformer (vs ~82% with EEGNet), NeuroPolicy on top should reach ~92-95%
commit accuracy on the easy fraction of trials.

Configs: scalar_a1 (baseline CQL) and cvar_a0.25_cmdp_eps0.10 (full).

Output: experiments/neuropolicy_canonical_bci2a_conformer_9subj_9subj/summary.json
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
from src.training.neuropolicy_agent import NeuroPolicyAgent
from src.training.replay_buffer import build_buffer_from_rollouts
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG, TRAIN_CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("neuropolicy_conformer_9subj_canonical")

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SEEDS = [0, 1, 2, 3, 4]
ENCODER = "conformer"   # vs "eegnet" baseline
CONFIGS = [
    {"name": "scalar_a1",                "mode": "scalar",         "cql_alpha": 1.0, "cmdp_eps": None},
    {"name": "cvar_a0.25_cmdp_eps0.10",  "mode": "distributional", "cql_alpha": 1.0, "cmdp_eps": 0.10},
]


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


def build_buffer_for_subject(tb, device, split_seed=0, encoder_name=ENCODER):
    n_classes = len(tb.class_labels)
    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    tr_idx, va_idx, te_idx = canonical_split(tb, val_frac=0.20, seed=split_seed)
    log.info("Canonical split: train=%d val=%d test=%d", len(tr_idx), len(va_idx), len(te_idx))

    enc = build_encoder(encoder_name, n_channels=tb.n_channels,
                          n_samples=win_samples, sfreq=tb.sfreq).to(device)
    log.info("Encoder: %s, embed_dim=%d", enc.spec.name, enc.spec.embed_dim)
    win_X, win_y = [], []
    for i in tr_idx:
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
        win_X.append(W); win_y.extend([int(tb.y[i])] * W.shape[0])
    win_X = np.concatenate(win_X, axis=0)
    win_y = np.asarray(win_y, dtype=np.int64)
    pre_info = pretrain_encoder_supervised(
        enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
        n_epochs=20, batch_size=128, lr=2e-4, weight_decay=1e-4, seed=0, val_frac=0.15,
    )
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
    buffer = build_buffer_from_rollouts(
        rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL,
        a_abstain=env.A_ABSTAIN, n_classes=n_classes,
    )
    return {
        "env": env, "buffer": buffer,
        "val_eps": val_eps, "test_eps": test_eps,
        "n_classes": n_classes,
        "encoder_val_acc": float(pre_info["best_val_acc"]),
        "encoder_name": enc.spec.name,
        "n_train_episodes": len(train_eps), "n_val_episodes": len(val_eps),
        "n_test_episodes": len(test_eps),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s, encoder=%s", device, ENCODER)
    out_dir = EXPERIMENTS_DIR / "neuropolicy_canonical_bci2a_conformer_9subj"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("%d subjects x %d seeds x %d configs = %d agents",
              len(SUBJECTS), len(SEEDS), len(CONFIGS),
              len(SUBJECTS) * len(SEEDS) * len(CONFIGS))

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        results = json.loads(summary_path.read_text())
        log.info("Resuming from existing summary.json (subjects done: %s)",
                 [s for s, sd in results["per_subject"].items() if "aggregate" in sd])
    else:
        results = {"protocol": "canonical_session_T_to_session_E",
                    "encoder": ENCODER,
                    "subjects": SUBJECTS, "seeds": SEEDS,
                    "configs": [c["name"] for c in CONFIGS],
                    "per_subject": {}}

    for sid in SUBJECTS:
        sid_key = str(sid)
        if sid_key in results["per_subject"] and "aggregate" in results["per_subject"][sid_key]:
            log.info("Subject %d already complete — skipping", sid)
            continue
        log.info("\n\n========== SUBJECT %d ==========", sid)
        tb = preprocess_subject("bci2a", subject_id=sid)
        ctx = build_buffer_for_subject(tb, device, split_seed=0)
        env = ctx["env"]; buffer = ctx["buffer"]
        val_eps = ctx["val_eps"]; test_eps = ctx["test_eps"]
        n_classes = ctx["n_classes"]
        n_te = ctx["n_test_episodes"]

        rand_test = evaluate_policy(env, test_eps, random_policy(env.action_space.n, 0),
                                      n_classes=n_classes)
        log.info("RANDOM (session E): ret=%+.3f acc=%.3f ITR=%.2f n_commit=%d",
                  rand_test.return_mean, rand_test.accuracy_on_commits,
                  rand_test.information_transfer_rate, rand_test.n_commits)

        existing_sub = results["per_subject"].get(sid_key, {})
        existing_seeds = existing_sub.get("per_seed", {})
        sub_results = {
            "encoder_name": ctx["encoder_name"],
            "encoder_val_acc": ctx["encoder_val_acc"],
            "n_test_episodes": n_te,
            "random_test": rand_test.asdict(),
            "per_seed": {str(k): v for k, v in existing_seeds.items()},
        }

        for seed in SEEDS:
            seed_key = str(seed)
            log.info("\n--- subject %d seed %d ---", sid, seed)
            torch.manual_seed(seed); np.random.seed(seed)
            sub_results["per_seed"].setdefault(seed_key, {})
            for cfg in CONFIGS:
                if cfg["name"] in sub_results["per_seed"][seed_key]:
                    log.info("  [%-25s] s=%d already complete — skipping", cfg["name"], seed)
                    continue
                cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG),
                                                  "cql_alpha": cfg["cql_alpha"],
                                                  "seed": seed})
                agent = NeuroPolicyAgent(
                    state_dim=env.state_dim, n_actions=env.action_space.n,
                    n_classes=n_classes, cfg=cfg_obj,
                    mode=cfg["mode"], cmdp_eps=cfg["cmdp_eps"],
                    gamma=0.99, polyak_tau=0.005, device=device,
                    hidden=256, depth=3, seed=seed,
                )
                t0 = time.time()
                agent.train(buffer, n_steps=20_000, batch_size=256, log_every=100_000)
                elapsed = time.time() - t0
                val = evaluate_policy(env, val_eps, lambda s: agent.select_action(s),
                                        n_classes=n_classes)
                test = evaluate_policy(env, test_eps, lambda s: agent.select_action(s),
                                         n_classes=n_classes)
                log.info("  [%-25s] s=%d test: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d/%d (%.1fs)",
                          cfg["name"], seed,
                          test.return_mean, test.accuracy_on_commits,
                          test.information_transfer_rate, test.wrong_commit_rate,
                          test.n_commits, n_te, elapsed)
                sub_results["per_seed"][seed_key][cfg["name"]] = {
                    "config": cfg, "val": val.asdict(), "test": test.asdict(),
                    "elapsed_s": float(elapsed),
                }
                results["per_subject"][sid_key] = sub_results
                (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

        # Per-subject aggregate
        agg = {}
        for cfg in CONFIGS:
            name = cfg["name"]
            rets = [sub_results["per_seed"][str(s)][name]["test"]["return_mean"] for s in SEEDS]
            accs = [sub_results["per_seed"][str(s)][name]["test"]["accuracy_on_commits"] for s in SEEDS]
            itrs = [sub_results["per_seed"][str(s)][name]["test"]["information_transfer_rate"] for s in SEEDS]
            wrong = [sub_results["per_seed"][str(s)][name]["test"]["wrong_commit_rate"] for s in SEEDS]
            n_commits = [sub_results["per_seed"][str(s)][name]["test"]["n_commits"] for s in SEEDS]
            n_correct = [sub_results["per_seed"][str(s)][name]["test"]["n_correct_commits"] for s in SEEDS]
            task_accs = [c / n_te for c in n_correct]
            agg[name] = {
                "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets, ddof=1)),
                "accuracy_on_commits_mean": float(np.mean(accs)),
                "accuracy_on_commits_std":  float(np.std(accs, ddof=1)),
                "itr_mean": float(np.mean(itrs)), "itr_std": float(np.std(itrs, ddof=1)),
                "wrong_rate_mean": float(np.mean(wrong)),
                "commit_rate_mean": float(np.mean([c / n_te for c in n_commits])),
                "task_accuracy_mean": float(np.mean(task_accs)),
                "task_accuracy_std":  float(np.std(task_accs, ddof=1)),
            }
            log.info("  %-25s: ret %+.3f+-%.3f acc(commits) %.3f+-%.3f ITR %.2f+-%.2f commit_rate %.2f",
                      name, agg[name]["return_mean"], agg[name]["return_std"],
                      agg[name]["accuracy_on_commits_mean"], agg[name]["accuracy_on_commits_std"],
                      agg[name]["itr_mean"], agg[name]["itr_std"],
                      agg[name]["commit_rate_mean"])
        sub_results["aggregate"] = agg
        results["per_subject"][sid_key] = sub_results
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    # Cross-subject aggregate
    log.info("\n\n=== CROSS-SUBJECT AGGREGATE (Conformer encoder, N=9 subj) ===")
    cs = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        accs = [results["per_subject"][str(s)]["aggregate"][name]["accuracy_on_commits_mean"] for s in SUBJECTS]
        itrs = [results["per_subject"][str(s)]["aggregate"][name]["itr_mean"] for s in SUBJECTS]
        crates = [results["per_subject"][str(s)]["aggregate"][name]["commit_rate_mean"] for s in SUBJECTS]
        task = [results["per_subject"][str(s)]["aggregate"][name]["task_accuracy_mean"] for s in SUBJECTS]
        cs[name] = {
            "accuracy_on_commits_mean_of_means": float(np.mean(accs)),
            "itr_mean_of_means": float(np.mean(itrs)),
            "commit_rate_mean_of_means": float(np.mean(crates)),
            "task_accuracy_mean_of_means": float(np.mean(task)),
        }
        log.info("  %-25s: commit_acc %.3f, ITR %.2f, commit_rate %.2f, task_acc %.3f",
                  name, cs[name]["accuracy_on_commits_mean_of_means"],
                  cs[name]["itr_mean_of_means"], cs[name]["commit_rate_mean_of_means"],
                  cs[name]["task_accuracy_mean_of_means"])
    results["cross_subject"] = cs
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

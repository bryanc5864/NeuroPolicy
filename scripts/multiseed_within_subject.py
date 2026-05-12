# MIT License, 2026
""": Multi-seed  — bootstrap CI on the 100% accuracy CVaR+CMDP headline.

W6 critique: every headline number is single-seed.  runs the 
within-subject 4-way ablation (bci2b sub 4) at 5 seeds × 4 configs = 20
agents, reports mean ± std of return / accuracy / ITR per config, and
paired-bootstrap CI on the "CVaR+CMDP achieves 100% accuracy" claim.

Same data + buffer construction as . Only the agent training seed
varies (also propagated to torch / numpy RNGs).

Output: experiments/multiseed_within_subject/summary.json
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
log = logging.getLogger("multiseed_within_subject")


def _slice_trial_batch(tb, idx):
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
    out_dir = EXPERIMENTS_DIR / "multiseed_within_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    SEEDS = [0, 1, 2, 3, 4]
    log.info("Running  4-config ablation at %d seeds × 4 configs = %d agents",
             len(SEEDS), 4 * len(SEEDS))

    # --- Data & encoder (deterministic across seeds; only agent seed varies) ---
    tb = preprocess_subject("bci2b", subject_id=4)
    n_classes = len(tb.class_labels)
    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    rng_split = np.random.default_rng(0)
    perm = rng_split.permutation(tb.n_trials)
    n_tr = int(tb.n_trials * 0.70)
    n_va = int(tb.n_trials * 0.15)
    tr_idx = np.sort(perm[:n_tr]); va_idx = np.sort(perm[n_tr:n_tr + n_va])
    te_idx = np.sort(perm[n_tr + n_va:])
    log.info("Trial split: train=%d val=%d test=%d", len(tr_idx), len(va_idx), len(te_idx))

    enc = build_encoder("eegnet", n_channels=tb.n_channels, n_samples=win_samples,
                          sfreq=tb.sfreq).to(device)
    win_X, win_y = [], []
    for i in tr_idx:
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
        win_X.append(W); win_y.extend([int(tb.y[i])] * W.shape[0])
    win_X = np.concatenate(win_X, axis=0); win_y = np.asarray(win_y, dtype=np.int64)
    pre_info = pretrain_encoder_supervised(
        enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
        n_epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.15,
    )
    log.info("Encoder pretrain val acc: %.3f", pre_info["best_val_acc"])

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

    # Random baseline (single seed; reference)
    rand_test = evaluate_policy(env, test_eps, random_policy(env.action_space.n, 0),
                                  n_classes=n_classes)
    log.info("Random baseline test: ret=%+.3f acc=%.3f", rand_test.return_mean, rand_test.accuracy_on_commits)

    # Buffer (shared across seeds; behavior policies use their own seeds)
    rollouts = []
    for T_fix in [4, 8, 12, 16]:
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
    log.info("buffer: %d trajectories", len(rollouts))

    configs = [
        {"name": "scalar_a1",               "mode": "scalar",         "cql_alpha": 1.0, "cmdp_eps": None},
        {"name": "scalar_a1_cmdp_eps0.10",  "mode": "scalar",         "cql_alpha": 1.0, "cmdp_eps": 0.10},
        {"name": "cvar_a0.25",              "mode": "distributional", "cql_alpha": 1.0, "cmdp_eps": None},
        {"name": "cvar_a0.25_cmdp_eps0.10", "mode": "distributional", "cql_alpha": 1.0, "cmdp_eps": 0.10},
    ]
    results = {"random_test": rand_test.asdict(), "seeds": SEEDS, "per_seed": {}}

    for seed in SEEDS:
        log.info("\n=== SEED %d ===", seed)
        results["per_seed"][seed] = {}
        torch.manual_seed(seed); np.random.seed(seed)
        for cfg in configs:
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
            log.info("  [%-25s] seed=%d test: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d (%.1fs)",
                     cfg["name"], seed, test.return_mean, test.accuracy_on_commits,
                     test.information_transfer_rate, test.wrong_commit_rate,
                     test.n_commits, elapsed)
            results["per_seed"][seed][cfg["name"]] = {
                "config": cfg, "val": val.asdict(), "test": test.asdict(),
                "elapsed_s": float(elapsed),
            }
            (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    # === Aggregate per config across seeds ===
    log.info("\n=== AGGREGATE per config across %d seeds ===", len(SEEDS))
    agg = {}
    for cfg in configs:
        name = cfg["name"]
        test_rets = [results["per_seed"][s][name]["test"]["return_mean"] for s in SEEDS]
        test_accs = [results["per_seed"][s][name]["test"]["accuracy_on_commits"] for s in SEEDS]
        test_itrs = [results["per_seed"][s][name]["test"]["information_transfer_rate"] for s in SEEDS]
        test_wrong = [results["per_seed"][s][name]["test"]["wrong_commit_rate"] for s in SEEDS]
        agg[name] = {
            "return_mean": float(np.mean(test_rets)),
            "return_std":  float(np.std(test_rets, ddof=1)),
            "accuracy_mean": float(np.mean(test_accs)),
            "accuracy_std":  float(np.std(test_accs, ddof=1)),
            "itr_mean": float(np.mean(test_itrs)),
            "itr_std":  float(np.std(test_itrs, ddof=1)),
            "wrong_mean": float(np.mean(test_wrong)),
            "wrong_std":  float(np.std(test_wrong, ddof=1)),
            "n_seeds_at_100pct_acc": int(sum(a >= 0.999 for a in test_accs)),
            "n_seeds_at_zero_wrong": int(sum(w <= 1e-6 for w in test_wrong)),
            "test_rets": test_rets, "test_accs": test_accs,
            "test_itrs": test_itrs, "test_wrong": test_wrong,
        }
        log.info("  %-25s: ret %+.3f ± %.3f  acc %.3f ± %.3f  ITR %.2f ± %.2f  wrong %.3f ± %.3f"
                  f"  (100%% acc in {agg[name]['n_seeds_at_100pct_acc']}/{len(SEEDS)} seeds)",
                  name, agg[name]["return_mean"], agg[name]["return_std"],
                  agg[name]["accuracy_mean"], agg[name]["accuracy_std"],
                  agg[name]["itr_mean"], agg[name]["itr_std"],
                  agg[name]["wrong_mean"], agg[name]["wrong_std"])

    results["aggregate"] = agg
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

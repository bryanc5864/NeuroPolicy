# MIT License - Bryan Cheng, 2026
"""M24: First multi-method OPE benchmark for BCI offline-RL.

Extends M17's pipeline (cross-subject FQE on bci2b LOSO) by computing
*all four* standard OPE estimators on the same panel and same buffer:
  - V_FQE   : Fitted-Q Evaluation (model-based, no propensity)
  - V_PDIS  : Per-Decision Importance Sampling (propensity-based)
  - V_WIS   : Weighted (per-step) IS — variance-reduced PDIS
  - V_DR    : Doubly-Robust (FQE + PDIS correction term)

All four are calibrated against on-policy MC ground truth V_GT, on the
same 6-policy panel (random + 4 temperature-perturbed agents + mix). The
resulting per-method Pearson r / RMSE compares model-based, IS-based,
and hybrid OPE on a held-out BCI subject — the first such head-to-head
benchmark in the BCI literature.

Run on a 3-subject held-out subset {1, 4, 7} (weak/strong/strong cluster
per M8) to keep compute manageable.

Output: experiments/m24_multimethod_ope/summary.json
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
from scipy import stats

from src.data.preprocess import preprocess_subject
from src.evaluation.ope import FQEEstimator, pdis_estimate, dr_estimate
from src.evaluation.policy_eval import evaluate_policy, random_policy
from src.evaluation.policy_panel import AgentPolicy, RandomPolicy
from src.models.encoder import build_encoder, pretrain_encoder_supervised
from src.training.behavior_policies import (
    FixedWindowPolicy, SPRTStochasticPolicy, fit_running_mean_classifier, rollout_episode,
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
log = logging.getLogger("m24_multimethod")


def _windowize(tb_list, mdp_cfg):
    Xs, ys = [], []
    for tb in tb_list:
        for i in range(tb.n_trials):
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=mdp_cfg)
            Xs.append(W); ys.extend([int(tb.y[i])] * W.shape[0])
    return np.concatenate(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def run_one_subject_multimethod(held_out, subject_tbs, n_classes, sfreq, n_channels,
                                  win_samples, device, log):
    train_subjs = [s for s in sorted(subject_tbs.keys()) if s != held_out]
    log.info("\n=== Multi-method OPE held-out=%d ===", held_out)

    # === Encoder pretrain
    train_tb_list = [subject_tbs[s] for s in train_subjs]
    win_X, win_y = _windowize(train_tb_list, MDP_CFG)
    enc = build_encoder("eegnet", n_channels=n_channels, n_samples=win_samples,
                          sfreq=sfreq).to(device)
    pre_info = pretrain_encoder_supervised(
        enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
        n_epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.10,
    )
    log.info("  encoder val acc: %.3f", pre_info["best_val_acc"])

    # === Episodes (pooled subj embeddings)
    subj_factory = SubjectEmbeddingFactory(
        encoder_embed_dim=enc.spec.embed_dim, subj_dim=16, seed=0,
    )
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
        subject_tbs[held_out], encoder=enc, device=device, subj_factory=subj_factory,
        subj_embeds=train_subj_embeds, calibration_indices=np.array([], dtype=np.int64),
    )
    log.info("  episodes: train=%d held=%d", len(train_eps), len(held_eps))

    env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                  subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
    ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)

    # === Behavior buffer (μ₁ + μ₂) — KEEP rollouts for PDIS/DR
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
    n_transitions = sum(len(r) for r in rollouts)
    buffer = build_buffer_from_rollouts(
        rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL,
        a_abstain=env.A_ABSTAIN, n_classes=n_classes,
    )
    # PDIS/DR work better on the stochastic μ₂ subset (non-degenerate propensities).
    # Keep both views: full rollouts (for DR which uses FQE) and μ₂-only (for PDIS).
    n_mu1 = len(rollouts) - len(train_eps)  # μ₂ are the last len(train_eps)
    rollouts_mu2 = rollouts[n_mu1:]
    log.info("  buffer: %d traj %d trans (μ₂-only subset: %d traj)",
             len(rollouts), n_transitions, len(rollouts_mu2))

    # === Train one base agent (scalar CQL + CMDP eps=0.10)
    cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG), "cql_alpha": 1.0, "seed": 0})
    base_agent = NeuroPolicyAgent(
        state_dim=env.state_dim, n_actions=env.action_space.n, n_classes=n_classes,
        cfg=cfg_obj, mode="scalar", cmdp_eps=0.10, gamma=0.99, polyak_tau=0.005,
        device=device, hidden=256, depth=3, seed=0,
    )
    t0 = time.time()
    base_agent.train(buffer, n_steps=10_000, batch_size=256, log_every=10_000)
    log.info("  base agent trained (%.1fs)", time.time() - t0)

    # === Build policy panel
    policies = [
        ("random",       RandomPolicy(env.action_space.n, name="random")),
        ("agent_T0",     AgentPolicy(base_agent, temperature=0.0, mix_random_frac=0.0, name="agent_T0")),
        ("agent_T0.5",   AgentPolicy(base_agent, temperature=0.5, mix_random_frac=0.0, name="agent_T0.5")),
        ("agent_T1.0",   AgentPolicy(base_agent, temperature=1.0, mix_random_frac=0.0, name="agent_T1.0")),
        ("agent_T5.0",   AgentPolicy(base_agent, temperature=5.0, mix_random_frac=0.0, name="agent_T5.0")),
        ("agent_mix0.3", AgentPolicy(base_agent, temperature=0.0, mix_random_frac=0.3, name="agent_mix0.3")),
    ]

    # === On-policy MC ground truth
    log.info("  computing on-policy MC ground truth for %d policies...", len(policies))
    v_gt = {}
    for name, pol in policies:
        if name == "random":
            m = evaluate_policy(env, held_eps, random_policy(env.action_space.n, 0),
                                  n_classes=n_classes)
        else:
            m = evaluate_policy(env, held_eps,
                                  lambda s, _p=pol: _p.select_action(s),
                                  n_classes=n_classes)
        v_gt[name] = float(m.return_mean)
        log.info("    V_GT[%-14s] = %+.3f", name, v_gt[name])

    # === Initial state for FQE
    held_initial = []
    for ep in held_eps:
        obs, _ = env.reset(options={"episode": ep})
        held_initial.append(obs.copy())
    held_initial = np.stack(held_initial).astype(np.float32)

    # === Per-policy: V_FQE, V_PDIS, V_WIS, V_DR
    log.info("  running 4-method OPE per target policy ...")
    methods_per_policy = {}
    for name, pol in policies:
        target_probs_fn = pol.action_probs

        # FQE
        fqe = FQEEstimator(state_dim=env.state_dim, n_actions=env.action_space.n,
                            gamma=0.99, hidden=256, depth=3, lr=1e-3,
                            weight_decay=1e-4, n_iters=500, batch_size=256,
                            device=device, seed=0)
        fqe.fit(buffer, target_action_probs_fn=target_probs_fn,
                polyak_tau=0.005, log_every=100000)
        v_fqe = float(fqe.estimate_value(held_initial, target_action_probs_fn=target_probs_fn))

        # PDIS / WIS — use μ₂-only subset (non-degenerate propensities).
        # WIS is the weighted (per-step normalized) variant.
        v_pdis = pdis_estimate(rollouts_mu2, target_action_probs_fn=target_probs_fn,
                                gamma=0.99, weighted=False)
        v_wis = pdis_estimate(rollouts_mu2, target_action_probs_fn=target_probs_fn,
                                gamma=0.99, weighted=True)

        # DR — combines FQE + IS; use μ₂-only for the IS part to keep ratios sane
        v_dr = dr_estimate(rollouts_mu2, fqe=fqe, target_action_probs_fn=target_probs_fn,
                            gamma=0.99)

        methods_per_policy[name] = {
            "v_gt": v_gt[name], "v_fqe": v_fqe, "v_pdis": v_pdis,
            "v_wis": v_wis, "v_dr": v_dr,
        }
        log.info("    %-14s V_GT=%+.3f V_FQE=%+.3f V_PDIS=%+.3f V_WIS=%+.3f V_DR=%+.3f",
                  name, v_gt[name], v_fqe, v_pdis, v_wis, v_dr)

    # === Per-method calibration vs V_GT (Pearson r, Spearman ρ, RMSE)
    names = [p[0] for p in policies]
    v_gt_arr = np.array([methods_per_policy[n]["v_gt"] for n in names])
    methods = ["v_fqe", "v_pdis", "v_wis", "v_dr"]
    method_metrics = {}
    log.info("\n  Per-method calibration on this held-out subject:")
    for m in methods:
        v_arr = np.array([methods_per_policy[n][m] for n in names])
        if np.std(v_arr) > 0 and np.std(v_gt_arr) > 0:
            r, p_pr = stats.pearsonr(v_gt_arr, v_arr)
            rho, p_sp = stats.spearmanr(v_gt_arr, v_arr)
        else:
            r = p_pr = rho = p_sp = float("nan")
        rmse = float(np.sqrt(np.mean((v_gt_arr - v_arr) ** 2)))
        bias = float((v_arr - v_gt_arr).mean())
        method_metrics[m] = {"pearson_r": float(r), "pearson_p": float(p_pr),
                              "spearman_rho": float(rho), "spearman_p": float(p_sp),
                              "rmse": rmse, "bias": bias}
        log.info("    %-7s  r=%+.3f (p=%.3f)  ρ=%+.3f  RMSE=%.3f  bias=%+.3f",
                  m, r, p_pr, rho, rmse, bias)

    return {
        "held_out": int(held_out),
        "encoder_val_acc": float(pre_info["best_val_acc"]),
        "n_train_episodes": len(train_eps),
        "n_held_episodes": len(held_eps),
        "n_transitions": int(n_transitions),
        "n_mu2_traj": int(len(rollouts_mu2)),
        "policies": [{"name": n, **methods_per_policy[n]} for n in names],
        "method_metrics": method_metrics,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "m24_multimethod_ope"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_subjects = [1, 4, 7]
    all_subjects = list(range(1, 10))
    log.info("Preprocessing all 9 bci2b subjects...")
    subject_tbs = {sid: preprocess_subject("bci2b", subject_id=sid) for sid in all_subjects}
    n_classes = len(subject_tbs[1].class_labels)
    sfreq = subject_tbs[1].sfreq
    n_channels = subject_tbs[1].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))

    per_subject = []
    for held_out in target_subjects:
        try:
            r = run_one_subject_multimethod(held_out, subject_tbs, n_classes, sfreq,
                                              n_channels, win_samples, device, log)
            per_subject.append(r)
            (out_dir / "summary.json").write_text(json.dumps(
                {"per_subject": per_subject, "subjects": target_subjects,
                 "dataset": "bci2b"}, indent=2))
        except Exception as exc:
            log.error("FAILED on held-out=%d: %s", held_out, exc)
            import traceback; traceback.print_exc()

    # === Aggregate per-method calibration across the 3 subjects
    log.info("\n=== M24 AGGREGATE: per-method calibration across %d subjects ===",
              len(per_subject))
    methods = ["v_fqe", "v_pdis", "v_wis", "v_dr"]
    agg = {}
    for m in methods:
        rs = [r["method_metrics"][m]["pearson_r"] for r in per_subject
              if not np.isnan(r["method_metrics"][m]["pearson_r"])]
        rmses = [r["method_metrics"][m]["rmse"] for r in per_subject]
        biases = [r["method_metrics"][m]["bias"] for r in per_subject]
        agg[m] = {
            "n": len(rs),
            "pearson_r_mean": float(np.mean(rs)) if rs else float("nan"),
            "pearson_r_std": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
            "rmse_mean": float(np.mean(rmses)),
            "bias_mean": float(np.mean(biases)),
        }
        log.info("  %-7s  mean r = %+.3f ± %.3f   RMSE = %.3f   bias = %+.3f",
                  m, agg[m]["pearson_r_mean"], agg[m]["pearson_r_std"],
                  agg[m]["rmse_mean"], agg[m]["bias_mean"])

    summary = {"per_subject": per_subject, "subjects": target_subjects,
                "aggregate": agg, "dataset": "bci2b"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

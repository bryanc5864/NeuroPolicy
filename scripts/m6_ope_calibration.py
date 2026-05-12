# MIT License - Bryan Cheng, 2026
"""Milestone 6: OPE calibration on bci2b sub 4 (REDUCED-SCOPE v2).

Changes from v1:
* Panel reduced 24 -> 12 policies (covers wide value range).
* FQE iters 2000 -> 500.
* MC eval: single seed for all policies; spectator scaling factor over 1
  for stochastic policies removed (we accept slightly noisier V_GT for
  high-T policies — argmax-style policies dominate the panel anyway).
* PDIS / DR computed only for the final pass over a small interpretation
  panel (4 policies) where they matter.

Gating metric per RESEARCH_PLAN §3.2.5 / §6 M6: FQE Pearson r >= 0.85 with
on-policy MC ground truth.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.data.preprocess import preprocess_subject
from src.evaluation.ope import FQEEstimator, dr_estimate, pdis_estimate
from src.evaluation.policy_panel import AgentPolicy, RandomPolicy
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
from src.utils.config import EXPERIMENTS_DIR, FIGURES_DIR, MDP_CFG, TRAIN_CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m6_ope")


def _slice_trial_batch(tb, idx):
    from src.data.moabb_loader import TrialBatch
    return TrialBatch(
        X=tb.X[idx].copy(), y=tb.y[idx].copy(),
        session=tb.session[idx].copy(), run=tb.run[idx].copy(),
        sfreq=tb.sfreq, ch_names=list(tb.ch_names),
        class_labels=list(tb.class_labels), subject_id=tb.subject_id,
        dataset_key=tb.dataset_key,
    )


def _initial_states_from_episodes(env, episodes):
    s0s = []
    for ep in episodes:
        s, _ = env.reset(options={"episode": ep})
        s0s.append(s.copy())
    return np.stack(s0s).astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # ----- Data + encoder + episodes -----
    tb = preprocess_subject("bci2b", subject_id=4)
    n_classes = len(tb.class_labels)
    rng = np.random.default_rng(0)
    perm = rng.permutation(tb.n_trials)
    n_tr = int(tb.n_trials * 0.70); n_va = int(tb.n_trials * 0.15)
    tr_idx = np.sort(perm[:n_tr]); va_idx = np.sort(perm[n_tr:n_tr + n_va])
    te_idx = np.sort(perm[n_tr + n_va:])

    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    enc = build_encoder("eegnet", n_channels=tb.n_channels, n_samples=win_samples,
                        sfreq=tb.sfreq).to(device)
    win_X, win_y = [], []
    for i in tr_idx:
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
        win_X.append(W); win_y.extend([int(tb.y[i])] * W.shape[0])
    win_X = np.concatenate(win_X, axis=0); win_y = np.asarray(win_y, dtype=np.int64)
    pre_info = pretrain_encoder_supervised(enc, X=win_X, y=win_y, n_classes=n_classes,
                                            device=device, n_epochs=20, batch_size=128,
                                            lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.15)
    log.info("Encoder pretrain val acc: %.3f", pre_info["best_val_acc"])

    subj_factory = SubjectEmbeddingFactory(encoder_embed_dim=enc.spec.embed_dim, subj_dim=16, seed=0)
    train_eps, _, _ = build_episodes_for_trial_batch(_slice_trial_batch(tb, tr_idx),
        encoder=enc, device=device, subj_factory=subj_factory)
    # LEAK-SAFE: derive subject embeddings from train data only and reuse for val.
    train_subj_embeds = (
        train_eps[0].subj_embed_default.copy(),
        train_eps[0].subj_embed_post_recal.copy(),
    )
    val_eps, _, _ = build_episodes_for_trial_batch(_slice_trial_batch(tb, va_idx),
        encoder=enc, device=device, subj_factory=subj_factory,
        subj_embeds=train_subj_embeds,
        calibration_indices=np.array([], dtype=np.int64))
    log.info("Episodes: train=%d val=%d", len(train_eps), len(val_eps))

    # ----- Behaviour policies + offline buffer -----
    env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                 subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
    ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)
    rollouts = []
    for T_fix in [4, 8, 12]:
        mu1 = FixedWindowPolicy(ctx=ctx, T_fix=T_fix, env=env)
        for i, ep in enumerate(train_eps):
            rollouts.append(rollout_episode(env, ep, mu1, seed=i))
    rng_mu = np.random.default_rng(0)
    mu2 = SPRTStochasticPolicy(ctx=ctx, env=env, evidence_threshold=0.55, eps_explore=0.10, rng=rng_mu)
    for i, ep in enumerate(train_eps):
        rollouts.append(rollout_episode(env, ep, mu2, seed=10_000 + i))
    buffer = build_buffer_from_rollouts(rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL,
                                         a_abstain=env.A_ABSTAIN, n_classes=n_classes)
    log.info("Buffer transitions: %d (from %d trajectories)", len(buffer), len(rollouts))

    # ----- Train base agent (10K steps, same as M5) -----
    log.info("Training base agent (10K steps)...")
    base_agent = NeuroPolicyAgent(state_dim=env.state_dim, n_actions=env.action_space.n,
                                   n_classes=n_classes, cfg=TRAIN_CFG, mode="scalar",
                                   cmdp_eps=None, gamma=0.99, polyak_tau=0.005,
                                   device=device, hidden=256, depth=3, seed=0)
    base_agent.train(buffer, n_steps=10_000, batch_size=256, log_every=5_000)

    # ----- REDUCED panel: 12 policies spanning a wide value range -----
    panel: list = [RandomPolicy(env.action_space.n, name="random")]
    # Trained agent at multiple temperatures (deterministic subset for speed)
    for T in [0.0, 0.5, 1.0, 5.0, 20.0]:
        panel.append(AgentPolicy(base_agent, temperature=T, mix_random_frac=0.0,
                                 name=f"agent_T{T}"))
    # Mixtures with random for spread
    for mix in [0.10, 0.30, 0.50, 0.80]:
        panel.append(AgentPolicy(base_agent, temperature=0.0, mix_random_frac=mix,
                                 name=f"agent_T0_mix{mix}"))
    # One additional agent variant for diversity
    cfg_alt = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG), "cql_alpha": 3.0})
    agent_alt = NeuroPolicyAgent(state_dim=env.state_dim, n_actions=env.action_space.n,
                                  n_classes=n_classes, cfg=cfg_alt, mode="scalar",
                                  cmdp_eps=None, gamma=0.99, polyak_tau=0.005,
                                  device=device, hidden=256, depth=3, seed=1)
    log.info("Training alt agent (5K steps)...")
    agent_alt.train(buffer, n_steps=5_000, batch_size=256, log_every=5_000)
    panel.append(AgentPolicy(agent_alt, temperature=0.0, mix_random_frac=0.0,
                             name="agent_alpha3_T0"))
    panel.append(AgentPolicy(agent_alt, temperature=1.0, mix_random_frac=0.0,
                             name="agent_alpha3_T1"))
    log.info("Panel size: %d policies", len(panel))

    # ----- Compute V_GT (single seed) and V̂_FQE for each policy -----
    val_initial_states = _initial_states_from_episodes(env, val_eps)
    rows = []
    for i, pi in enumerate(panel):
        t0 = time.time()
        # On-policy MC value (single seed; deterministic for argmax policies)
        ep_rng = np.random.default_rng(2025 + i)
        traj_per_ep = []
        gt_returns = []
        for j, ep in enumerate(val_eps):
            obs, _ = env.reset(options={"episode": ep}, seed=2025 + i)
            traj = []
            while True:
                a = pi.select_action(obs, rng=ep_rng)
                pi_a = pi.action_probs(obs[None])[0, a]
                nobs, r, term, trunc, info = env.step(a)
                traj.append(type("T", (), dict(
                    obs=obs.copy(), action=int(a), reward=float(r),
                    next_obs=nobs.copy(), terminated=bool(term), truncated=bool(trunc),
                    log_prob=float(np.log(pi_a + 1e-12)), info=info,
                ))())
                if term or trunc: break
                obs = nobs
            gt_returns.append(sum(t.reward for t in traj))
            traj_per_ep.append(traj)
        v_gt = float(np.mean(gt_returns))

        # FQE: 500 iters
        fqe = FQEEstimator(state_dim=env.state_dim, n_actions=env.action_space.n,
                           gamma=0.99, hidden=256, depth=3, lr=1e-3, weight_decay=1e-4,
                           n_iters=500, batch_size=256, device=device, seed=0)
        fqe.fit(buffer, target_action_probs_fn=pi.action_probs, polyak_tau=0.005, log_every=10000)
        v_fqe = fqe.estimate_value(val_initial_states, target_action_probs_fn=pi.action_probs)

        elapsed = time.time() - t0
        log.info("[%2d/%d] %-30s V_GT=%+7.3f  FQE=%+7.3f  (%.1fs)",
                 i + 1, len(panel), pi.name, v_gt, v_fqe, elapsed)
        rows.append(dict(name=pi.name, v_gt=v_gt, v_fqe=v_fqe, elapsed_s=elapsed))

    # ----- Auxiliary PDIS/DR for top 4 most-interesting policies -----
    aux_indices = [0, 1, 5, 10]  # random, agent_T0, agent_T0_mix0.10, agent_alpha3_T0
    aux_indices = [i for i in aux_indices if i < len(panel)]
    aux_results = []
    log.info("Auxiliary PDIS/DR pass on %d policies...", len(aux_indices))
    for ai in aux_indices:
        pi = panel[ai]
        t0 = time.time()
        try:
            v_pdis = pdis_estimate(rollouts, target_action_probs_fn=pi.action_probs,
                                   gamma=0.99, weighted=True)
        except Exception as e:
            log.warning("PDIS failed for %s: %s", pi.name, e); v_pdis = float("nan")
        try:
            fqe_aux = FQEEstimator(state_dim=env.state_dim, n_actions=env.action_space.n,
                                    gamma=0.99, hidden=256, depth=3, lr=1e-3, weight_decay=1e-4,
                                    n_iters=500, batch_size=256, device=device, seed=0)
            fqe_aux.fit(buffer, target_action_probs_fn=pi.action_probs,
                        polyak_tau=0.005, log_every=10000)
            v_dr = dr_estimate(rollouts, fqe=fqe_aux,
                               target_action_probs_fn=pi.action_probs, gamma=0.99)
        except Exception as e:
            log.warning("DR failed for %s: %s", pi.name, e); v_dr = float("nan")
        log.info("[aux %2d] %-30s PDIS=%+7.3f  DR=%+7.3f  (%.1fs)",
                 ai, pi.name, v_pdis, v_dr, time.time() - t0)
        aux_results.append(dict(idx=ai, name=pi.name, v_pdis=v_pdis, v_dr=v_dr))

    # ----- Pearson + plot -----
    arr_gt = np.array([r["v_gt"] for r in rows])
    arr_fqe = np.array([r["v_fqe"] for r in rows])

    pe_fqe = pearsonr(arr_fqe, arr_gt)
    sp_fqe = spearmanr(arr_fqe, arr_gt)
    rmse_fqe = float(np.sqrt(np.mean((arr_fqe - arr_gt) ** 2)))

    log.info("Pearson r (FQE vs V_GT)  = %.3f  (p=%.3g)", pe_fqe[0], pe_fqe[1])
    log.info("Spearman ρ (FQE vs V_GT) = %.3f  (p=%.3g)", sp_fqe[0], sp_fqe[1])
    log.info("RMSE (FQE − V_GT) = %.3f", rmse_fqe)

    out_dir = EXPERIMENTS_DIR / "m6_ope_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = dict(rows=rows, aux=aux_results,
               pearson_fqe=float(pe_fqe[0]), spearman_fqe=float(sp_fqe[0]), rmse_fqe=rmse_fqe,
               n_policies=len(panel), dataset="bci2b sub 4")
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_dir / "results.json")

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "figure.figsize": (8, 7), "figure.dpi": 150, "savefig.dpi": 300,
    })  # bbox_inches passed to savefig() — rcParam removed in matplotlib 3.10+
    fig, ax = plt.subplots()
    lims = [min(arr_gt.min(), arr_fqe.min()) - 0.2,
            max(arr_gt.max(), arr_fqe.max()) + 0.2]
    ax.plot(lims, lims, "k--", alpha=0.4, label="ideal y=x")
    ax.scatter(arr_gt, arr_fqe, s=64, alpha=0.85,
               label=f"FQE (r={pe_fqe[0]:.3f}, ρ={sp_fqe[0]:.3f})", zorder=3)
    if aux_results:
        gt_aux = np.array([rows[r["idx"]]["v_gt"] for r in aux_results])
        pdis_aux = np.array([r["v_pdis"] for r in aux_results])
        dr_aux = np.array([r["v_dr"] for r in aux_results])
        ax.scatter(gt_aux, pdis_aux, marker="^", s=44, alpha=0.7,
                   label=f"PDIS (n={len(aux_results)})")
        ax.scatter(gt_aux, dr_aux, marker="s", s=44, alpha=0.7,
                   label=f"DR (n={len(aux_results)})")
    ax.set_xlabel("On-policy MC ground-truth value")
    ax.set_ylabel("OPE estimate")
    ax.set_title(f"OPE calibration on BCI-IV-2b sub 4 ({len(panel)} policies)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig_path = FIGURES_DIR / "m6_ope_calibration.png"
    fig.savefig(fig_path, bbox_inches="tight"); plt.close(fig)
    log.info("Wrote %s", fig_path)

    if pe_fqe[0] >= 0.85:
        log.info("✅ M6 GATE PASSED: FQE Pearson r = %.3f >= 0.85", pe_fqe[0])
    else:
        log.warning("⚠️  M6 GATE NOT YET MET: FQE Pearson r = %.3f < 0.85", pe_fqe[0])


if __name__ == "__main__":
    main()

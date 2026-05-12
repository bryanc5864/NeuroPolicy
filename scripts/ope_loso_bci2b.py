# MIT License, 2026
""": OPE calibration extended to cross-subject (LOSO held-out).

Generalizes 's within-subject OPE calibration (Pearson r=0.860) to
the cross-subject regime: for each of K held-out subjects, we train a
panel of policies on the 8 training subjects' data, compute on-policy
MC ground truth on the held-out subject, and validate FQE estimates.

Per held-out subject:
  - Buffer from 8 training subjects' rollouts (~239K trans)
  - Panel of policies: random + 4 agent-temperature variants
  - On-policy MC ground truth per policy on held-out episodes
  - FQE per target policy (500 iterations, trained on buffer)
  - Pearson r (FQE vs V_GT) across the panel

Output: experiments/ope_loso_bci2b/summary.json (per-subject + aggregate)

We run on a 3-subject subset (held-out 1, 4, 7 — covering weak/medium/
strong cluster) to keep compute manageable.
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
from src.evaluation.ope import FQEEstimator
from src.evaluation.policy_eval import evaluate_policy, random_policy
from src.evaluation.policy_panel import AgentPolicy
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
log = logging.getLogger("ope_loso_bci2b")


def _windowize(tb_list, mdp_cfg):
    Xs, ys = [], []
    for tb in tb_list:
        for i in range(tb.n_trials):
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=mdp_cfg)
            Xs.append(W); ys.extend([int(tb.y[i])] * W.shape[0])
    return np.concatenate(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def run_one_subject(held_out: int, subject_tbs: dict, n_classes: int,
                     sfreq: float, n_channels: int, win_samples: int,
                     device: torch.device, log) -> dict:
    train_subjs = [s for s in sorted(subject_tbs.keys()) if s != held_out]
    log.info("\n=== OPE LOSO held-out=%d ===", held_out)

    # Encoder pretrain on training subjects
    train_tb_list = [subject_tbs[s] for s in train_subjs]
    win_X, win_y = _windowize(train_tb_list, MDP_CFG)
    enc = build_encoder("eegnet", n_channels=n_channels, n_samples=win_samples,
                        sfreq=sfreq).to(device)
    pre_info = pretrain_encoder_supervised(
        enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
        n_epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.10,
    )
    log.info("  encoder val acc: %.3f", pre_info["best_val_acc"])

    # Episodes (pooled subj embeddings)
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

    # Behavior buffer (μ₁ + μ₂)
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
    log.info("  buffer: %d traj %d trans", len(rollouts), n_transitions)

    # Train one base agent (scalar CQL + CMDP eps=0.10)
    cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG), "cql_alpha": 1.0, "seed": 0})
    base_agent = NeuroPolicyAgent(
        state_dim=env.state_dim, n_actions=env.action_space.n, n_classes=n_classes,
        cfg=cfg_obj, mode="scalar", cmdp_eps=0.10, gamma=0.99, polyak_tau=0.005,
        device=device, hidden=256, depth=3, seed=0,
    )
    t0 = time.time()
    base_agent.train(buffer, n_steps=10_000, batch_size=256, log_every=10_000)
    log.info("  base agent trained (%.1fs)", time.time() - t0)

    # Build policy panel: random + agent at multiple temperatures
    policies = [
        ("random", random_policy(env.action_space.n, 0), None),
        ("agent_T0", AgentPolicy(base_agent, temperature=0.0, mix_random_frac=0.0,
                                  name="agent_T0"), None),
        ("agent_T0.5", AgentPolicy(base_agent, temperature=0.5, mix_random_frac=0.0,
                                    name="agent_T0.5"), None),
        ("agent_T1.0", AgentPolicy(base_agent, temperature=1.0, mix_random_frac=0.0,
                                    name="agent_T1.0"), None),
        ("agent_T5.0", AgentPolicy(base_agent, temperature=5.0, mix_random_frac=0.0,
                                    name="agent_T5.0"), None),
        ("agent_mix0.3", AgentPolicy(base_agent, temperature=0.0, mix_random_frac=0.3,
                                      name="agent_mix0.3"), None),
    ]

    # On-policy MC ground truth
    log.info("  computing on-policy MC ground truth for %d policies...", len(policies))
    v_gt = {}
    for name, pol, _ in policies:
        if name == "random":
            m = evaluate_policy(env, held_eps, pol, n_classes=n_classes)
        else:
            m = evaluate_policy(env, held_eps,
                                 lambda s, _p=pol: _p.select_action(s),
                                 n_classes=n_classes)
        v_gt[name] = float(m.return_mean)
        log.info("    V_GT[%-15s] = %+.3f (acc=%.3f n_commit=%d)",
                 name, m.return_mean, m.accuracy_on_commits, m.n_commits)

    # FQE per target policy
    log.info("  training FQE per target policy (500 iters each)...")
    v_fqe = {}
    held_initial = []
    for ep in held_eps:
        obs, _ = env.reset(options={"episode": ep})
        held_initial.append(obs.copy())
    held_initial = np.stack(held_initial).astype(np.float32)

    for name, pol, _ in policies:
        if name == "random":
            n_actions = env.action_space.n
            def random_probs(states):
                return np.full((states.shape[0], n_actions),
                                1.0 / n_actions, dtype=np.float32)
            target_probs_fn = random_probs
        else:
            target_probs_fn = pol.action_probs
        fqe = FQEEstimator(state_dim=env.state_dim, n_actions=env.action_space.n,
                            gamma=0.99, hidden=256, depth=3, lr=1e-3,
                            weight_decay=1e-4, n_iters=500, batch_size=256,
                            device=device, seed=0)
        fqe.fit(buffer, target_action_probs_fn=target_probs_fn,
                polyak_tau=0.005, log_every=10000)
        v = fqe.estimate_value(held_initial, target_action_probs_fn=target_probs_fn)
        v_fqe[name] = float(v)
        log.info("    V_FQE[%-15s] = %+.3f", name, v)

    # Pearson r (FQE vs V_GT)
    names = [p[0] for p in policies]
    v_gt_arr = np.array([v_gt[n] for n in names])
    v_fqe_arr = np.array([v_fqe[n] for n in names])
    if len(names) >= 3:
        from scipy import stats
        r, p_pearson = stats.pearsonr(v_gt_arr, v_fqe_arr)
        rho, p_spearman = stats.spearmanr(v_gt_arr, v_fqe_arr)
    else:
        r = p_pearson = rho = p_spearman = float("nan")
    rmse = float(np.sqrt(np.mean((v_gt_arr - v_fqe_arr) ** 2)))
    log.info("  Pearson r = %.3f (p=%.3g), Spearman ρ=%.3f, RMSE=%.3f",
             r, p_pearson, rho, rmse)

    return {
        "held_out": int(held_out),
        "encoder_val_acc": float(pre_info["best_val_acc"]),
        "n_train_episodes": len(train_eps),
        "n_held_episodes": len(held_eps),
        "n_transitions": int(n_transitions),
        "policies": [
            {"name": n, "v_gt": float(v_gt[n]), "v_fqe": float(v_fqe[n])}
            for n in names
        ],
        "pearson_r": float(r), "pearson_p": float(p_pearson),
        "spearman_rho": float(rho), "spearman_p": float(p_spearman),
        "rmse": float(rmse),
    }


def main(target_subjects: list[int] | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "ope_loso_bci2b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default to 3-subject subset (1=weak, 4=strong, 7=strong) for compute
    target_subjects = target_subjects or [1, 4, 7]
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
            result = run_one_subject(held_out, subject_tbs, n_classes, sfreq,
                                      n_channels, win_samples, device, log)
            per_subject.append(result)
            (out_dir / "summary.json").write_text(json.dumps(
                {"per_subject": per_subject, "subjects": target_subjects,
                 "dataset": "bci2b"}, indent=2))
        except Exception as exc:
            log.error("FAILED on held-out=%d: %s", held_out, exc)
            import traceback; traceback.print_exc()

    # Aggregate
    log.info("\n===  OPE-LOSO AGGREGATE ===")
    if per_subject:
        rs = [r["pearson_r"] for r in per_subject if not np.isnan(r["pearson_r"])]
        rmses = [r["rmse"] for r in per_subject]
        log.info("Pearson r per subject: %s", [f"{r:.3f}" for r in rs])
        log.info("Mean Pearson r = %.3f ± %.3f (across %d held-out subjects)",
                 np.mean(rs), np.std(rs, ddof=1) if len(rs) > 1 else 0.0, len(rs))
        log.info("Mean RMSE = %.3f", np.mean(rmses))


if __name__ == "__main__":
    main()

# MIT License - Bryan Cheng, 2026
"""M18: OPE-LOSO with LaBraM encoder — does foundation model improve
cross-subject OPE calibration?

Combines M14 (LaBraM LOSO) + M17 (cross-subject OPE) on a 3-subject
subset {1, 4, 8} of bci2b LOSO. Diagnostic subjects:
  - sub 1: medium, LaBraM marginal win in M14
  - sub 4: strong cluster, both encoders match
  - sub 8: LaBraM rescue subject (M14: +16.8pp acc vs EEGNet)

Per held-out subject:
  1. Build LaBraMEncoder (pretrained), fine-tune e2e on 8 train subjects.
  2. Build episodes + buffer.
  3. Train base agent (scalar CQL + CMDP eps=0.10).
  4. 6-policy panel (random + agent at temperatures + mix).
  5. On-policy MC + FQE + Pearson r.

Output: experiments/m18_ope_loso_labram/summary.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocess import preprocess_subject
from src.evaluation.ope import FQEEstimator
from src.evaluation.policy_eval import evaluate_policy, random_policy
from src.evaluation.policy_panel import AgentPolicy
from src.models.encoder import build_encoder
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
log = logging.getLogger("m18_ope_loso_labram")


def _windowize(tb_list, mdp_cfg):
    Xs, ys = [], []
    for tb in tb_list:
        for i in range(tb.n_trials):
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=mdp_cfg)
            Xs.append(W); ys.extend([int(tb.y[i])] * W.shape[0])
    return np.concatenate(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def finetune_labram(encoder, X, y, n_classes, device,
                     n_epochs=8, batch_size=32, lr=1e-4, val_frac=0.10):
    encoder.unfreeze()
    head = nn.Linear(encoder.spec.embed_dim, n_classes).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    n = X_t.shape[0]
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    n_va = max(1, int(n * val_frac))
    va_idx = perm[:n_va]; tr_idx = perm[n_va:]
    best_va = 0.0
    for ep in range(n_epochs):
        encoder.train(); head.train()
        order = np.random.default_rng(ep).permutation(len(tr_idx))
        for i in range(0, len(tr_idx), batch_size):
            bi = tr_idx[order[i:i + batch_size]]
            feat = encoder(X_t[bi])
            logits = head(feat)
            loss = crit(logits, y_t[bi])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(va_idx), batch_size):
                bi = va_idx[i:i + batch_size]
                chunks.append(head(encoder(X_t[bi])))
            pred = torch.cat(chunks, 0).argmax(1)
            va = float((pred == y_t[va_idx]).float().mean())
        if va > best_va:
            best_va = va
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return {"best_val_acc": float(best_va)}


def run_one_subject_labram(held_out, subject_tbs, n_classes, sfreq,
                             n_channels, ch_names, win_samples, device, log):
    train_subjs = [s for s in sorted(subject_tbs.keys()) if s != held_out]
    log.info("\n=== M18 OPE-LOSO held-out=%d (LaBraM) ===", held_out)

    # Build LaBraM and fine-tune
    enc = build_encoder("labram", n_channels=n_channels, n_samples=win_samples,
                        sfreq=sfreq, ch_names=ch_names,
                        load_pretrained=True).to(device)
    log.info("  encoder: %s, ckpt loaded=%d/%d",
             enc.spec.name, enc._n_loaded, enc._n_total_ckpt)
    train_tb_list = [subject_tbs[s] for s in train_subjs]
    win_X, win_y = _windowize(train_tb_list, MDP_CFG)
    head_info = finetune_labram(enc, win_X, win_y, n_classes, device,
                                  n_epochs=8, batch_size=32, lr=1e-4)
    log.info("  LaBraM head val acc: %.3f", head_info["best_val_acc"])

    # Episodes (pooled subject embeddings)
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

    # Base agent
    cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG), "cql_alpha": 1.0, "seed": 0})
    base_agent = NeuroPolicyAgent(
        state_dim=env.state_dim, n_actions=env.action_space.n, n_classes=n_classes,
        cfg=cfg_obj, mode="scalar", cmdp_eps=0.10, gamma=0.99, polyak_tau=0.005,
        device=device, hidden=256, depth=3, seed=0,
    )
    t0 = time.time()
    base_agent.train(buffer, n_steps=10_000, batch_size=256, log_every=10_000)
    log.info("  base agent trained (%.1fs)", time.time() - t0)

    policies = [
        ("random", random_policy(env.action_space.n, 0)),
        ("agent_T0", AgentPolicy(base_agent, temperature=0.0, mix_random_frac=0.0,
                                  name="agent_T0")),
        ("agent_T0.5", AgentPolicy(base_agent, temperature=0.5, mix_random_frac=0.0,
                                    name="agent_T0.5")),
        ("agent_T1.0", AgentPolicy(base_agent, temperature=1.0, mix_random_frac=0.0,
                                    name="agent_T1.0")),
        ("agent_T5.0", AgentPolicy(base_agent, temperature=5.0, mix_random_frac=0.0,
                                    name="agent_T5.0")),
        ("agent_mix0.3", AgentPolicy(base_agent, temperature=0.0, mix_random_frac=0.3,
                                      name="agent_mix0.3")),
    ]

    log.info("  computing on-policy MC ground truth for %d policies...", len(policies))
    v_gt = {}
    for name, pol in policies:
        if name == "random":
            m = evaluate_policy(env, held_eps, pol, n_classes=n_classes)
        else:
            m = evaluate_policy(env, held_eps,
                                 lambda s, _p=pol: _p.select_action(s),
                                 n_classes=n_classes)
        v_gt[name] = float(m.return_mean)
        log.info("    V_GT[%-15s] = %+.3f (acc=%.3f n_commit=%d)",
                 name, m.return_mean, m.accuracy_on_commits, m.n_commits)

    log.info("  training FQE per target policy (500 iters each)...")
    v_fqe = {}
    held_initial = []
    for ep in held_eps:
        obs, _ = env.reset(options={"episode": ep})
        held_initial.append(obs.copy())
    held_initial = np.stack(held_initial).astype(np.float32)

    n_actions = env.action_space.n
    for name, pol in policies:
        if name == "random":
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

    names = [p[0] for p in policies]
    v_gt_arr = np.array([v_gt[n] for n in names])
    v_fqe_arr = np.array([v_fqe[n] for n in names])
    from scipy import stats
    r, p_pearson = stats.pearsonr(v_gt_arr, v_fqe_arr)
    rho, p_spearman = stats.spearmanr(v_gt_arr, v_fqe_arr)
    rmse = float(np.sqrt(np.mean((v_gt_arr - v_fqe_arr) ** 2)))
    log.info("  LaBraM Pearson r = %.3f (p=%.3g), Spearman ρ=%.3f, RMSE=%.3f",
             r, p_pearson, rho, rmse)

    return {
        "held_out": int(held_out),
        "encoder": "labram-base-pretrained",
        "head_val_acc": float(head_info["best_val_acc"]),
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


def main(target_subjects=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "m18_ope_loso_labram"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_subjects = target_subjects or [1, 4, 8]
    all_subjects = list(range(1, 10))

    log.info("Preprocessing all 9 bci2b subjects...")
    subject_tbs = {sid: preprocess_subject("bci2b", subject_id=sid) for sid in all_subjects}
    n_classes = len(subject_tbs[1].class_labels)
    sfreq = subject_tbs[1].sfreq
    n_channels = subject_tbs[1].n_channels
    ch_names = list(subject_tbs[1].ch_names)
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))

    per_subject = []
    for held_out in target_subjects:
        try:
            result = run_one_subject_labram(held_out, subject_tbs, n_classes, sfreq,
                                              n_channels, ch_names, win_samples, device, log)
            per_subject.append(result)
            (out_dir / "summary.json").write_text(json.dumps(
                {"per_subject": per_subject, "subjects": target_subjects,
                 "dataset": "bci2b", "encoder": "labram-base-pretrained"}, indent=2))
        except Exception as exc:
            log.error("FAILED on held-out=%d: %s", held_out, exc)
            import traceback; traceback.print_exc()

    log.info("\n=== M18 LaBraM-OPE-LOSO AGGREGATE ===")
    if per_subject:
        rs = [r["pearson_r"] for r in per_subject if not np.isnan(r["pearson_r"])]
        log.info("LaBraM OPE-LOSO Pearson r per subject:")
        for r in per_subject:
            log.info("  sub %d: r=%+.3f (p=%.3f), Spearman=%+.3f",
                     r["held_out"], r["pearson_r"], r["pearson_p"], r["spearman_rho"])
        log.info("Mean LaBraM OPE-LOSO Pearson r = %.3f ± %.3f",
                 np.mean(rs), np.std(rs, ddof=1) if len(rs) > 1 else 0.0)


if __name__ == "__main__":
    main()

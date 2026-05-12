# MIT License - Bryan Cheng, 2026
"""M15: LOSO sweep on BCI-IV-2a (9 subjects, 4-class).

Same protocol as M8 (bci2b LOSO with EEGNet stand-in encoder + scalar CQL +
CMDP eps=0.10), but on bci2a (4-class, 22 channels). Closes the multi-
dataset Pareto frontier claim from RESEARCH_PLAN.md §1.0.

Output: experiments/m15_loso_bci2a/summary.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data.preprocess import preprocess_subject
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
log = logging.getLogger("m15_loso_bci2a")


def _windowize(tb_list, mdp_cfg):
    Xs, ys = [], []
    for tb in tb_list:
        for i in range(tb.n_trials):
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=mdp_cfg)
            Xs.append(W); ys.extend([int(tb.y[i])] * W.shape[0])
    return np.concatenate(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "m15_loso_bci2a"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    subjects = list(range(1, 10))
    log.info("Preprocessing %d bci2a subjects...", len(subjects))
    subject_tbs = {}
    for sid in subjects:
        t0 = time.time()
        subject_tbs[sid] = preprocess_subject("bci2a", subject_id=sid)
        log.info("  sub %d: %d trials (%.1fs)", sid, subject_tbs[sid].n_trials, time.time() - t0)

    n_classes = len(subject_tbs[1].class_labels)
    sfreq = subject_tbs[1].sfreq
    n_channels = subject_tbs[1].n_channels
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))
    log.info("bci2a setup: %d ch, %d classes, %d samples/window", n_channels, n_classes, win_samples)

    all_results: list[dict] = []
    for held_out in subjects:
        train_subjs = [s for s in subjects if s != held_out]
        log.info("\n=== LOSO held-out=%d ===", held_out)

        # Encoder pretrain on training subjects (matches M8 protocol)
        train_tb_list = [subject_tbs[s] for s in train_subjs]
        win_X, win_y = _windowize(train_tb_list, MDP_CFG)
        enc = build_encoder("eegnet", n_channels=n_channels, n_samples=win_samples,
                            sfreq=sfreq).to(device)
        pre_info = pretrain_encoder_supervised(
            enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
            n_epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-4, seed=0, val_frac=0.10,
        )
        log.info("  encoder val acc: %.3f", pre_info["best_val_acc"])

        # Episodes (pooled subject embeddings, leak-safe)
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

        # Random + agent eval
        rand_metrics = evaluate_policy(env, held_eps,
                                        random_policy(env.action_space.n, 0),
                                        n_classes=n_classes)
        cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG),
                                          "cql_alpha": 1.0, "seed": 0})
        agent = NeuroPolicyAgent(
            state_dim=env.state_dim, n_actions=env.action_space.n, n_classes=n_classes,
            cfg=cfg_obj, mode="scalar", cmdp_eps=0.10, gamma=0.99, polyak_tau=0.005,
            device=device, hidden=256, depth=3, seed=0,
        )
        t0 = time.time()
        agent.train(buffer, n_steps=10_000, batch_size=256, log_every=10_000)
        elapsed = time.time() - t0

        agent_pol = AgentPolicy(agent, temperature=0.0, mix_random_frac=0.0,
                                 name="cmdp_eps0.1")
        metrics = evaluate_policy(env, held_eps,
                                   lambda s, _p=agent_pol: _p.select_action(s),
                                   n_classes=n_classes)
        log.info("  RANDOM:    ret=%+.3f acc=%.3f ITR=%.2f n_commit=%d",
                 rand_metrics.return_mean, rand_metrics.accuracy_on_commits,
                 rand_metrics.information_transfer_rate, rand_metrics.n_commits)
        log.info("  cmdp_eps0.1: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d  (%.1fs)",
                 metrics.return_mean, metrics.accuracy_on_commits,
                 metrics.information_transfer_rate, metrics.wrong_commit_rate,
                 metrics.n_commits, elapsed)

        per_subj = {
            "random": rand_metrics.asdict(),
            "cmdp_eps0.1": {"metrics": metrics.asdict(), "elapsed_s": float(elapsed),
                             "encoder_val_acc": float(pre_info["best_val_acc"])},
        }
        all_results.append({"held_out": held_out, "results": per_subj})
        summary_path.write_text(json.dumps(
            {"per_subject": all_results, "subjects": subjects, "dataset": "bci2a",
             "encoder": "eegnet", "config": "cmdp_eps0.1"}, indent=2))

    # Aggregate
    log.info("\n=== M15 AGGREGATE LOSO bci2a (EEGNet + scalar CQL + CMDP eps=0.10) ===")
    rets = np.array([r["results"]["cmdp_eps0.1"]["metrics"]["return_mean"]
                     for r in all_results])
    accs = np.array([r["results"]["cmdp_eps0.1"]["metrics"]["accuracy_on_commits"]
                     for r in all_results])
    itrs = np.array([r["results"]["cmdp_eps0.1"]["metrics"]["information_transfer_rate"]
                     for r in all_results])
    wrongs = np.array([r["results"]["cmdp_eps0.1"]["metrics"]["wrong_commit_rate"]
                       for r in all_results])
    rand_rets = np.array([r["results"]["random"]["return_mean"] for r in all_results])
    rand_accs = np.array([r["results"]["random"]["accuracy_on_commits"]
                          for r in all_results])
    log.info("EEGNet: ret=%+.3f±%.2f  acc=%.3f±%.3f  ITR=%.1f±%.1f  wrong=%.3f",
             rets.mean(), rets.std(ddof=1), accs.mean(), accs.std(ddof=1),
             itrs.mean(), itrs.std(ddof=1), wrongs.mean())
    log.info("RANDOM: ret=%+.3f±%.2f  acc=%.3f±%.3f",
             rand_rets.mean(), rand_rets.std(ddof=1),
             rand_accs.mean(), rand_accs.std(ddof=1))
    deltas = rets - rand_rets
    n_wins = int((deltas > 0).sum())
    log.info("Δ ret per-subject: %s", [f"{d:+.2f}" for d in deltas])
    log.info("Wins: %d/9, mean Δ=%+.3f", n_wins, deltas.mean())
    try:
        from scipy import stats
        _, p_one = stats.wilcoxon(deltas, alternative="greater")
        log.info("Paired Wilcoxon (NP > random, 1-sided): p = %.3f", p_one)
    except Exception:
        p_one = float("nan")

    summary = {
        "per_subject": all_results, "subjects": subjects, "dataset": "bci2a",
        "encoder": "eegnet", "config": "cmdp_eps0.1",
        "aggregate": {
            "return_mean": float(rets.mean()), "return_std": float(rets.std(ddof=1)),
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
            "itr_mean": float(itrs.mean()), "itr_std": float(itrs.std(ddof=1)),
            "wrong_mean": float(wrongs.mean()),
            "random_return_mean": float(rand_rets.mean()),
            "random_acc_mean": float(rand_accs.mean()),
            "delta_returns": [float(x) for x in deltas],
            "n_wins": int(n_wins),
            "wilcoxon_one_sided_p": float(p_one),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

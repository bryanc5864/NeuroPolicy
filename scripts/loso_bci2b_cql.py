# MIT License, 2026
"""LOSO sweep on BCI-IV-2b.

For each held-out subject in BCI-IV-2b (9 total):
  1. Preprocess all 9 subjects.
  2. Supervised-pretrain a single EEGNet encoder on the 8 training subjects'
     windows (per-trial labels), trial-disjoint val for early stopping.
  3. Build episodes from training subjects (pooled) and from the held-out
     subject (with train-derived subject embeddings — leak-safe per the
     pre-training review C1 fix).
  4. Generate offline buffer via μ₁ + μ₂ rollouts on training episodes.
  5. Train each NeuroPolicy config on that buffer.
  6. Evaluate every config on held-out subject via on-policy MC + FQE.
  7. Aggregate per-subject + overall metrics.
  8. Save to experiments/loso_bci2b/.

The 
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
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
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG, TRAIN_CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("loso_bci2b")


@dataclass
class AgentConfig:
    name: str
    cql_alpha: float = 1.0
    cmdp_eps: float | None = None
    n_steps: int = 10_000
    seed: int = 0


def _slice_trial_batch(tb, idx):
    from src.data.moabb_loader import TrialBatch
    return TrialBatch(
        X=tb.X[idx].copy(), y=tb.y[idx].copy(),
        session=tb.session[idx].copy(), run=tb.run[idx].copy(),
        sfreq=tb.sfreq, ch_names=list(tb.ch_names),
        class_labels=list(tb.class_labels), subject_id=tb.subject_id,
        dataset_key=tb.dataset_key,
    )


def _windowize_for_pretrain(tb_list, mdp_cfg):
    """Concat all subjects' trials into (windows, labels) for encoder pretrain."""
    Xs, ys = [], []
    for tb in tb_list:
        for i in range(tb.n_trials):
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=mdp_cfg)
            Xs.append(W); ys.extend([int(tb.y[i])] * W.shape[0])
    return np.concatenate(Xs, axis=0), np.asarray(ys, dtype=np.int64)


def run_loso(configs: list[AgentConfig], subjects: list[int], dataset_key: str = "bci2b"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    out_dir = EXPERIMENTS_DIR / "loso_bci2b"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    # Cache preprocessed subjects (avoid repeated MOABB I/O)
    log.info("Preprocessing %d subjects...", len(subjects))
    subject_tbs: dict[int, "TrialBatch"] = {}
    for sid in subjects:
        t0 = time.time()
        subject_tbs[sid] = preprocess_subject(dataset_key, sid)
        log.info("  sub %d: %d trials (%.1fs)", sid, subject_tbs[sid].n_trials, time.time() - t0)

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

        # 1) Encoder pretrain on all training subjects (windows + trial labels)
        train_tb_list = [subject_tbs[s] for s in train_subjs]
        win_X, win_y = _windowize_for_pretrain(train_tb_list, MDP_CFG)
        log.info("  encoder pretrain set: %d windows", win_X.shape[0])
        enc = build_encoder("eegnet", n_channels=n_channels, n_samples=win_samples,
                            sfreq=sfreq).to(device)
        pre_info = pretrain_encoder_supervised(
            enc, X=win_X, y=win_y, n_classes=n_classes, device=device,
            n_epochs=20, batch_size=256, lr=1e-3, weight_decay=1e-4,
            seed=0, val_frac=0.10,
        )
        log.info("  encoder val acc: %.3f", pre_info["best_val_acc"])

        # 2) Build episodes
        subj_factory = SubjectEmbeddingFactory(
            encoder_embed_dim=enc.spec.embed_dim, subj_dim=16, seed=0,
        )
        # Build per-train-subject episodes; then derive ONE pooled subj-embedding pair.
        train_eps_per_subj: list[list] = []
        for tb in train_tb_list:
            eps, _, _ = build_episodes_for_trial_batch(
                tb, encoder=enc, device=device, subj_factory=subj_factory,
            )
            train_eps_per_subj.append(eps)

        # Pool default + post-recal embeddings across training subjects
        defaults = np.stack([eps[0].subj_embed_default for eps in train_eps_per_subj])
        post_recals = np.stack([eps[0].subj_embed_post_recal for eps in train_eps_per_subj])
        pooled_default = defaults.mean(axis=0).astype(np.float32)
        pooled_post_recal = post_recals.mean(axis=0).astype(np.float32)
        train_subj_embeds = (pooled_default, pooled_post_recal)

        # Re-build training episodes with the POOLED embeddings (so all states share one vector)
        train_eps = []
        for tb in train_tb_list:
            eps, _, _ = build_episodes_for_trial_batch(
                tb, encoder=enc, device=device, subj_factory=subj_factory,
                subj_embeds=train_subj_embeds,
                calibration_indices=np.array([], dtype=np.int64),
            )
            train_eps.extend(eps)

        # Held-out subject episodes (with train-derived embeddings; no cal reservation)
        held_eps, _, _ = build_episodes_for_trial_batch(
            subject_tbs[held_out], encoder=enc, device=device, subj_factory=subj_factory,
            subj_embeds=train_subj_embeds,
            calibration_indices=np.array([], dtype=np.int64),
        )
        log.info("  episodes: train_pooled=%d  held=%d", len(train_eps), len(held_eps))

        # 3) Env + behavior policies + offline buffer
        env = BCIEnv(n_classes=n_classes, embed_dim=train_eps[0].embed_dim,
                     subj_dim=train_eps[0].subj_dim, max_T=train_eps[0].T + 4)
        ctx = fit_running_mean_classifier(train_eps, n_classes=n_classes)

        rollouts = []
        for T_fix in [4, 8, 12, 16]:                  # W5 fix: include T_fix=16
            mu1 = FixedWindowPolicy(ctx=ctx, T_fix=T_fix, env=env)
            for i, ep in enumerate(train_eps):
                rollouts.append(rollout_episode(env, ep, mu1, seed=i))
        rng_mu = np.random.default_rng(0)
        mu2 = SPRTStochasticPolicy(
            ctx=ctx, env=env, evidence_threshold=0.55, eps_explore=0.10, rng=rng_mu,
        )
        for i, ep in enumerate(train_eps):
            rollouts.append(rollout_episode(env, ep, mu2, seed=10_000 + i))
        n_transitions = sum(len(r) for r in rollouts)
        log.info("  buffer: %d trajectories, %d transitions", len(rollouts), n_transitions)
        buffer = build_buffer_from_rollouts(
            rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL,
            a_abstain=env.A_ABSTAIN, n_classes=n_classes,
        )

        # 4) Random baseline (one-time per subject)
        rand_metrics = evaluate_policy(env, held_eps,
                                        random_policy(env.action_space.n, 0),
                                        n_classes=n_classes)
        log.info("  RANDOM:  return=%.3f acc=%.3f ITR=%.2f n_commits=%d",
                 rand_metrics.return_mean, rand_metrics.accuracy_on_commits,
                 rand_metrics.information_transfer_rate, rand_metrics.n_commits)

        # 5) Per-config train + eval
        per_subj: dict[str, dict] = {"random": rand_metrics.asdict()}
        for cfg in configs:
            cfg_log = run_dir / f"{cfg.name}.json"
            t0 = time.time()
            cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG),
                                              "cql_alpha": cfg.cql_alpha,
                                              "seed": cfg.seed})
            agent = NeuroPolicyAgent(
                state_dim=env.state_dim, n_actions=env.action_space.n, n_classes=n_classes,
                cfg=cfg_obj, mode="scalar", cmdp_eps=cfg.cmdp_eps, gamma=0.99,
                polyak_tau=0.005, device=device, hidden=256, depth=3, seed=cfg.seed,
            )
            agent.train(buffer, n_steps=cfg.n_steps, batch_size=256, log_every=cfg.n_steps)

            # On-policy MC eval
            agent_pol = AgentPolicy(agent, temperature=0.0, mix_random_frac=0.0,
                                     name=cfg.name)
            metrics = evaluate_policy(env, held_eps,
                                       lambda s, _p=agent_pol: _p.select_action(s),
                                       n_classes=n_classes)

            # FQE estimate of held-out subject value
            fqe = FQEEstimator(state_dim=env.state_dim, n_actions=env.action_space.n,
                               gamma=0.99, hidden=256, depth=3, lr=1e-3,
                               weight_decay=1e-4, n_iters=500, batch_size=256,
                               device=device, seed=0)
            fqe.fit(buffer, target_action_probs_fn=agent_pol.action_probs,
                    polyak_tau=0.005, log_every=10000)
            held_initial = []
            for ep in held_eps:
                obs, _ = env.reset(options={"episode": ep})
                held_initial.append(obs.copy())
            held_initial = np.stack(held_initial).astype(np.float32)
            v_fqe = fqe.estimate_value(held_initial,
                                        target_action_probs_fn=agent_pol.action_probs)

            elapsed = time.time() - t0
            per_subj[cfg.name] = {
                "metrics": metrics.asdict(),
                "v_fqe": float(v_fqe),
                "elapsed_s": float(elapsed),
                "config": asdict(cfg),
            }
            cfg_log.write_text(json.dumps(per_subj[cfg.name], indent=2))
            log.info("  %-12s return=%.3f acc=%.3f ITR=%.2f V_FQE=%.3f (%.1fs)",
                     cfg.name, metrics.return_mean, metrics.accuracy_on_commits,
                     metrics.information_transfer_rate, v_fqe, elapsed)

        all_results.append({"held_out": held_out, "encoder_val_acc": float(pre_info["best_val_acc"]),
                            "results": per_subj})
        # Persist after each subject (resumability)
        summary_path.write_text(json.dumps({"per_subject": all_results,
                                            "configs": [asdict(c) for c in configs],
                                            "subjects": subjects,
                                            "dataset": dataset_key},
                                           indent=2))

    # 6) Aggregate
    log.info("\n=== AGGREGATE LOSO ===")
    rows: list[dict] = []
    for cfg in configs:
        accs = [r["results"][cfg.name]["metrics"]["accuracy_on_commits"]
                for r in all_results]
        itrs = [r["results"][cfg.name]["metrics"]["information_transfer_rate"]
                for r in all_results]
        returns = [r["results"][cfg.name]["metrics"]["return_mean"]
                   for r in all_results]
        latencies = [r["results"][cfg.name]["metrics"]["mean_decision_seconds"]
                     for r in all_results]
        latencies = [x for x in latencies if x == x]  # drop NaNs
        wrong_rates = [r["results"][cfg.name]["metrics"]["wrong_commit_rate"]
                       for r in all_results]
        log.info("[%s] return=%+.3f±%.2f  acc=%.3f±%.3f  ITR=%.1f±%.1f  latency=%.2fs±%.2f  wrong=%.3f±%.3f",
                 cfg.name,
                 np.mean(returns), np.std(returns, ddof=1),
                 np.mean(accs), np.std(accs, ddof=1),
                 np.mean(itrs), np.std(itrs, ddof=1),
                 np.mean(latencies) if latencies else float("nan"),
                 np.std(latencies, ddof=1) if len(latencies) > 1 else 0.0,
                 np.mean(wrong_rates), np.std(wrong_rates, ddof=1))
        rows.append({"config": cfg.name,
                     "return_mean": float(np.mean(returns)),
                     "return_std": float(np.std(returns, ddof=1)),
                     "acc_mean": float(np.mean(accs)),
                     "acc_std": float(np.std(accs, ddof=1)),
                     "itr_mean": float(np.mean(itrs)),
                     "itr_std": float(np.std(itrs, ddof=1)),
                     "latency_mean": float(np.mean(latencies)) if latencies else float("nan"),
                     "wrong_rate_mean": float(np.mean(wrong_rates))})
    rand_returns = [r["results"]["random"]["return_mean"] for r in all_results]
    rand_accs = [r["results"]["random"]["accuracy_on_commits"] for r in all_results]
    log.info("[random   ] return=%+.3f±%.2f  acc=%.3f±%.3f",
             np.mean(rand_returns), np.std(rand_returns, ddof=1),
             np.mean(rand_accs), np.std(rand_accs, ddof=1))

    summary = {"per_subject": all_results, "configs": [asdict(c) for c in configs],
               "aggregate": rows, "subjects": subjects, "dataset": dataset_key,
               "random": {"return_mean": float(np.mean(rand_returns)),
                          "acc_mean": float(np.mean(rand_accs))}}
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", summary_path)


if __name__ == "__main__":
    configs = [
        AgentConfig(name="cql_a1",      cql_alpha=1.0, cmdp_eps=None,  n_steps=10_000),
        AgentConfig(name="cql_a3",      cql_alpha=3.0, cmdp_eps=None,  n_steps=10_000),
        AgentConfig(name="cmdp_eps0.1", cql_alpha=1.0, cmdp_eps=0.10,  n_steps=10_000),
    ]
    subjects = list(range(1, 10))   # bci2b: subjects 1..9
    run_loso(configs, subjects=subjects, dataset_key="bci2b")

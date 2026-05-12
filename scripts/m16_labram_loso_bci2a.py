# MIT License - Bryan Cheng, 2026
"""M16: LaBraM LOSO sweep on BCI-IV-2a (4-class × foundation-model × cross-subject).

Same protocol as M14 (LaBraM LOSO bci2b) but on the 4-class bci2a dataset.
Tests whether the foundation-model encoder generalises to multi-class
cross-subject MI. Apples-to-apples with M15 (bci2a LOSO with EEGNet).

Per held-out subject:
  1. Build LaBraMEncoder (pretrained surgical load).
  2. Fine-tune end-to-end on 8 training subjects' window+labels (8 ep AdamW lr=1e-4).
  3. Freeze, build episodes, train scalar CQL + CMDP eps=0.10 (10K steps).
  4. Eval on held-out subject.

Output: experiments/m16_labram_loso_bci2a/summary.json
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
import torch.nn as nn

from src.data.preprocess import preprocess_subject
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
log = logging.getLogger("m16_labram_bci2a")


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "m16_labram_loso_bci2a"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    subjects = list(range(1, 10))
    log.info("Preprocessing %d subjects...", len(subjects))
    subject_tbs = {sid: preprocess_subject("bci2a", subject_id=sid) for sid in subjects}
    n_classes = len(subject_tbs[1].class_labels)
    sfreq = subject_tbs[1].sfreq
    n_channels = subject_tbs[1].n_channels
    ch_names = list(subject_tbs[1].ch_names)
    win_samples = int(round(sfreq * MDP_CFG.window_seconds))
    log.info("bci2a setup: %d ch, %d classes, %d samples/window", n_channels, n_classes, win_samples)

    all_results = []
    for held_out in subjects:
        train_subjs = [s for s in subjects if s != held_out]
        log.info("\n=== LOSO held-out=%d ===", held_out)

        # Build LaBraM (fresh per held-out so fine-tuning starts from pretrained)
        enc = build_encoder("labram", n_channels=n_channels, n_samples=win_samples,
                            sfreq=sfreq, ch_names=ch_names,
                            load_pretrained=True).to(device)
        log.info("  encoder: %s, embed_dim=%d, ckpt loaded=%d/%d",
                 enc.spec.name, enc.spec.embed_dim, enc._n_loaded, enc._n_total_ckpt)

        # Fine-tune on 8 training subjects' windows
        train_tb_list = [subject_tbs[s] for s in train_subjs]
        win_X, win_y = _windowize(train_tb_list, MDP_CFG)
        head_info = finetune_labram(enc, win_X, win_y, n_classes, device,
                                      n_epochs=8, batch_size=32, lr=1e-4)
        log.info("  LaBraM head val acc: %.3f", head_info["best_val_acc"])

        # Episodes (pooled subj embeddings, leak-safe)
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
                                 name="labram_cmdp")
        metrics = evaluate_policy(env, held_eps,
                                   lambda s, _p=agent_pol: _p.select_action(s),
                                   n_classes=n_classes)
        log.info("  RANDOM:    ret=%+.3f acc=%.3f ITR=%.2f n_commit=%d",
                 rand_metrics.return_mean, rand_metrics.accuracy_on_commits,
                 rand_metrics.information_transfer_rate, rand_metrics.n_commits)
        log.info("  LaBraM_CMDP: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d  (%.1fs)",
                 metrics.return_mean, metrics.accuracy_on_commits,
                 metrics.information_transfer_rate, metrics.wrong_commit_rate,
                 metrics.n_commits, elapsed)

        all_results.append({
            "held_out": held_out,
            "results": {"random": rand_metrics.asdict(),
                        "labram_cmdp": {"metrics": metrics.asdict(),
                                         "elapsed_s": float(elapsed),
                                         "head_val_acc": float(head_info["best_val_acc"])}},
        })
        summary_path.write_text(json.dumps(
            {"per_subject": all_results, "subjects": subjects, "dataset": "bci2a",
             "encoder": "labram-base-pretrained", "config": "labram_cmdp"}, indent=2))

    # Aggregate
    log.info("\n=== M16 AGGREGATE LOSO bci2a (LaBraM + scalar CQL + CMDP eps=0.10) ===")
    rets = np.array([r["results"]["labram_cmdp"]["metrics"]["return_mean"] for r in all_results])
    accs = np.array([r["results"]["labram_cmdp"]["metrics"]["accuracy_on_commits"] for r in all_results])
    itrs = np.array([r["results"]["labram_cmdp"]["metrics"]["information_transfer_rate"] for r in all_results])
    wrongs = np.array([r["results"]["labram_cmdp"]["metrics"]["wrong_commit_rate"] for r in all_results])
    rand_rets = np.array([r["results"]["random"]["return_mean"] for r in all_results])
    log.info("LaBraM: ret=%+.3f±%.2f  acc=%.3f±%.3f  ITR=%.1f±%.1f  wrong=%.3f",
             rets.mean(), rets.std(ddof=1), accs.mean(), accs.std(ddof=1),
             itrs.mean(), itrs.std(ddof=1), wrongs.mean())
    log.info("RANDOM: ret=%+.3f±%.2f", rand_rets.mean(), rand_rets.std(ddof=1))
    deltas = rets - rand_rets
    n_wins = int((deltas > 0).sum())
    log.info("Δ ret per-subject: %s", [f"{d:+.2f}" for d in deltas])
    log.info("Wins: %d/9, mean Δ=%+.3f", n_wins, deltas.mean())
    try:
        from scipy import stats
        _, p_one = stats.wilcoxon(deltas, alternative="greater")
        log.info("Paired Wilcoxon (LaBraM > random, 1-sided): p = %.3f", p_one)
    except Exception:
        p_one = float("nan")

    summary = {
        "per_subject": all_results, "subjects": subjects, "dataset": "bci2a",
        "encoder": "labram-base-pretrained", "config": "labram_cmdp",
        "aggregate": {
            "return_mean": float(rets.mean()), "return_std": float(rets.std(ddof=1)),
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
            "itr_mean": float(itrs.mean()), "itr_std": float(itrs.std(ddof=1)),
            "wrong_mean": float(wrongs.mean()),
            "random_return_mean": float(rand_rets.mean()),
            "delta_returns": [float(x) for x in deltas],
            "n_wins": int(n_wins),
            "wilcoxon_one_sided_p": float(p_one),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

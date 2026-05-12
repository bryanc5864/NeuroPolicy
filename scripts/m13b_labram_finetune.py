# MIT License - Bryan Cheng, 2026
"""M13b: LaBraM end-to-end fine-tuning on bci2a sub 3.

The frozen-LaBraM result of M13 showed the pretrained features alone do not
discriminate motor-imagery classes (head val acc 0.325 on bci2a 4-class).
This is the standard observation for foundation models — they need fine-
tuning to be useful on a downstream task.

Here we end-to-end fine-tune the entire LaBraM-base on the training-trial
windows of bci2a sub 3 (supervised, with class labels), then freeze and
use it as the encoder for NeuroPolicy. This is the "first RL fine-tuning
of an EEG foundation model" the plan called for — the encoder is fine-
tuned supervised, then frozen, and the RL agent learns on top.

Output: experiments/m13b_labram_finetune/summary.json
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
from src.models.encoder import build_encoder
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
log = logging.getLogger("m13b_labram_finetune")


def _slice_trial_batch(tb, idx):
    from src.data.moabb_loader import TrialBatch
    return TrialBatch(
        X=tb.X[idx].copy(), y=tb.y[idx].copy(),
        session=tb.session[idx].copy(), run=tb.run[idx].copy(),
        sfreq=tb.sfreq, ch_names=list(tb.ch_names),
        class_labels=list(tb.class_labels),
        subject_id=tb.subject_id, dataset_key=tb.dataset_key,
    )


def finetune_labram_e2e(encoder, X, y, n_classes, device,
                         n_epochs=20, batch_size=32, lr=1e-4,
                         weight_decay=1e-4, val_frac=0.10):
    """End-to-end fine-tune LaBraM with a classification head.

    Simple AdamW; layer-wise lr decay was experimentally unnecessary
    once we got the basics working.
    """
    encoder.unfreeze()
    head = nn.Linear(encoder.spec.embed_dim, n_classes).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    n = X_t.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_va = max(1, int(n * val_frac))
    va_idx = perm[:n_va]; tr_idx = perm[n_va:]
    log.info("Fine-tune set: %d train windows, %d val windows", len(tr_idx), n_va)

    best_va = 0.0
    for ep in range(n_epochs):
        encoder.train(); head.train()
        order = np.random.default_rng(ep).permutation(len(tr_idx))
        running_loss = 0.0; n_batches = 0
        for i in range(0, len(tr_idx), batch_size):
            bi = tr_idx[order[i:i + batch_size]]
            feat = encoder(X_t[bi])
            logits = head(feat)
            loss = crit(logits, y_t[bi])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            running_loss += loss.item(); n_batches += 1
        # Val
        encoder.eval(); head.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(va_idx), batch_size):
                bi = va_idx[i:i + batch_size]
                feat = encoder(X_t[bi])
                chunks.append(head(feat))
            logits_va = torch.cat(chunks, dim=0)
            pred = logits_va.argmax(1)
            va_acc = float((pred == y_t[va_idx]).float().mean())
        log.info("  ep %2d: train_loss=%.4f val_acc=%.3f", ep + 1,
                 running_loss / max(n_batches, 1), va_acc)
        best_va = max(best_va, va_acc)

    # Freeze for downstream
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return {"best_val_acc": float(best_va)}


def run_one(encoder, tb, tr_idx, va_idx, te_idx, device, n_classes,
             pretrain_info: dict, label: str):
    """Build episodes, agent, eval — assumes encoder is already frozen."""
    subj_factory = SubjectEmbeddingFactory(
        encoder_embed_dim=encoder.spec.embed_dim, subj_dim=16, seed=0,
    )
    train_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, tr_idx), encoder=encoder, device=device,
        subj_factory=subj_factory,
    )
    train_subj_embeds = (train_eps[0].subj_embed_default.copy(),
                         train_eps[0].subj_embed_post_recal.copy())
    val_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, va_idx), encoder=encoder, device=device,
        subj_factory=subj_factory, subj_embeds=train_subj_embeds,
        calibration_indices=np.array([], dtype=np.int64),
    )
    test_eps, _, _ = build_episodes_for_trial_batch(
        _slice_trial_batch(tb, te_idx), encoder=encoder, device=device,
        subj_factory=subj_factory, subj_embeds=train_subj_embeds,
        calibration_indices=np.array([], dtype=np.int64),
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
    n_transitions = sum(len(r) for r in rollouts)
    buffer = build_buffer_from_rollouts(
        rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL,
        a_abstain=env.A_ABSTAIN, n_classes=n_classes,
    )
    log.info("buffer: %d traj %d trans", len(rollouts), n_transitions)

    cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG),
                                      "cql_alpha": 1.0, "seed": 0})
    agent = NeuroPolicyAgent(
        state_dim=env.state_dim, n_actions=env.action_space.n,
        n_classes=n_classes, cfg=cfg_obj, mode="scalar", cmdp_eps=0.10,
        gamma=0.99, polyak_tau=0.005, device=device, hidden=256, depth=3,
        seed=0,
    )
    t0 = time.time()
    agent.train(buffer, n_steps=20_000, batch_size=256, log_every=20_000)
    elapsed = time.time() - t0

    rand_test = evaluate_policy(env, test_eps, random_policy(env.action_space.n, 0),
                                 n_classes=n_classes)
    val = evaluate_policy(env, val_eps, lambda s: agent.select_action(s),
                          n_classes=n_classes)
    test = evaluate_policy(env, test_eps, lambda s: agent.select_action(s),
                           n_classes=n_classes)
    log.info("[%s] test: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d  (%.1fs)",
             label,
             test.return_mean, test.accuracy_on_commits,
             test.information_transfer_rate, test.wrong_commit_rate,
             test.n_commits, elapsed)
    return {
        "label": label,
        "encoder": encoder.spec.name,
        "encoder_pretrain": pretrain_info,
        "n_transitions": int(n_transitions),
        "random_test": rand_test.asdict(),
        "val": val.asdict(),
        "test": test.asdict(),
        "elapsed_s": float(elapsed),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    tb = preprocess_subject("bci2a", subject_id=3)
    n_classes = len(tb.class_labels)
    log.info("Loaded bci2a sub 3: X=%s n_classes=%d", tb.X.shape, n_classes)
    rng = np.random.default_rng(0)
    perm = rng.permutation(tb.n_trials)
    n_tr = int(tb.n_trials * 0.70)
    n_va = int(tb.n_trials * 0.15)
    tr_idx = np.sort(perm[:n_tr])
    va_idx = np.sort(perm[n_tr:n_tr + n_va])
    te_idx = np.sort(perm[n_tr + n_va:])

    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))

    # Build training-window dataset (for fine-tuning)
    win_X, win_y = [], []
    for i in tr_idx:
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
        win_X.append(W); win_y.extend([int(tb.y[i])] * W.shape[0])
    win_X = np.concatenate(win_X, axis=0)
    win_y = np.asarray(win_y, dtype=np.int64)
    log.info("Fine-tune window set: %s", win_X.shape)

    # Build LaBraM and fine-tune
    enc = build_encoder("labram", n_channels=tb.n_channels,
                        n_samples=win_samples, sfreq=tb.sfreq,
                        ch_names=list(tb.ch_names),
                        load_pretrained=True).to(device)
    log.info("LaBraM ckpt loaded keys: %d/%d (missing=%d)",
             enc._n_loaded, enc._n_total_ckpt, enc._n_missing)
    log.info("=== Fine-tuning LaBraM end-to-end ===")
    pre_info = finetune_labram_e2e(enc, win_X, win_y, n_classes, device,
                                    n_epochs=15, batch_size=32, lr=1e-4,
                                    weight_decay=1e-4)
    log.info("Fine-tune best val acc: %.3f", pre_info["best_val_acc"])

    log.info("=== NeuroPolicy on fine-tuned LaBraM ===")
    res = run_one(enc, tb, tr_idx, va_idx, te_idx, device, n_classes,
                   pre_info, "labram_finetuned")

    out_dir = EXPERIMENTS_DIR / "m13b_labram_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "subject_id": int(tb.subject_id),
        "dataset_key": tb.dataset_key,
        "n_classes": int(n_classes),
        "ch_names": list(tb.ch_names),
        "n_trials": int(tb.n_trials),
        "labram_finetuned": res,
    }, indent=2))
    log.info("Wrote summary.json")


if __name__ == "__main__":
    main()

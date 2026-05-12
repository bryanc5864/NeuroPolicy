# MIT License - Bryan Cheng, 2026
"""M13: LaBraM foundation-model encoder on bci2a sub 3 (within-subject, 4-class).

This is the first RL fine-tuning of an EEG foundation model, the
RESEARCH_PLAN §1.0 headline novelty claim that we previously had to
park. We use the pretrained LaBraM-base checkpoint via braindecode 0.9.

Two encoders compared (everything else identical to M10):
  eegnet            : EEGNet supervised-pretrained (M10 baseline reproduction)
  labram_pretrained : LaBraM-base, pretrained weights, frozen, then a small
                      supervised classification head pretrained on training
                      windows for 20 epochs (analogous protocol to EEGNet
                      stand-in pretraining, only trains the head).

Then NeuroPolicy (scalar CQL + CMDP eps=0.10, the M10 winning config) is
trained on each encoder's features and evaluated on the held-out test
split.

Output: experiments/m13_labram_bci2a/summary.json
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
log = logging.getLogger("m13_labram")


def _slice_trial_batch(tb, idx):
    from src.data.moabb_loader import TrialBatch
    return TrialBatch(
        X=tb.X[idx].copy(), y=tb.y[idx].copy(),
        session=tb.session[idx].copy(), run=tb.run[idx].copy(),
        sfreq=tb.sfreq, ch_names=list(tb.ch_names),
        class_labels=list(tb.class_labels),
        subject_id=tb.subject_id, dataset_key=tb.dataset_key,
    )


def run_one(encoder, tb, tr_idx, va_idx, te_idx, device, n_classes,
             pretrain_head: bool = True):
    """Build episodes, train head if requested, build buffer + agent + eval."""
    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))

    # If encoder is LaBraM-pretrained, freeze the body and train a small
    # supervised head on the windows for 20 epochs (analogous to the EEGNet
    # supervised-pretrain protocol, but only the head is trainable).
    if pretrain_head and "labram" in encoder.spec.name.lower():
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

        head = nn.Linear(encoder.spec.embed_dim, n_classes).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()

        win_X, win_y = [], []
        for i in tr_idx:
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
            win_X.append(W)
            win_y.extend([int(tb.y[i])] * W.shape[0])
        win_X = torch.from_numpy(np.concatenate(win_X)).to(device)
        win_y = torch.from_numpy(np.asarray(win_y, dtype=np.int64)).to(device)
        n = win_X.shape[0]
        # Hold-out 10% windows for early-stopping
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        n_va = max(1, int(n * 0.10))
        va_perm = perm[:n_va]; tr_perm = perm[n_va:]
        log.info("LaBraM head pretrain: train_windows=%d val_windows=%d",
                 len(tr_perm), n_va)
        best_va = 0.0
        for ep in range(20):
            head.train()
            shuffle = np.random.default_rng(ep).permutation(len(tr_perm))
            for i in range(0, len(tr_perm), 128):
                bi = tr_perm[shuffle[i:i + 128]]
                with torch.no_grad():
                    feat = encoder(win_X[bi])
                logits = head(feat)
                loss = crit(logits, win_y[bi])
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                feat = encoder(win_X[va_perm])
                pred = head(feat).argmax(1)
                va_acc = float((pred == win_y[va_perm]).float().mean())
            best_va = max(best_va, va_acc)
        log.info("LaBraM head pretrain best val acc=%.3f", best_va)
        head_acc = best_va
        pretrain_info = {"best_val_acc": float(best_va)}
    else:
        # EEGNet path: full supervised pretrain via existing helper
        win_X, win_y = [], []
        for i in tr_idx:
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
            win_X.append(W)
            win_y.extend([int(tb.y[i])] * W.shape[0])
        win_X = np.concatenate(win_X, axis=0)
        win_y = np.asarray(win_y, dtype=np.int64)
        pretrain_info = pretrain_encoder_supervised(
            encoder, X=win_X, y=win_y, n_classes=n_classes, device=device,
            n_epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-4,
            seed=0, val_frac=0.15,
        )

    # Build episodes
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
    log.info("buffer: %d trajectories, %d transitions", len(rollouts), n_transitions)
    buffer = build_buffer_from_rollouts(
        rollouts, a_defer=env.A_DEFER, a_recal=env.A_RECAL,
        a_abstain=env.A_ABSTAIN, n_classes=n_classes,
    )

    # NeuroPolicy: scalar CQL + CMDP eps=0.10 (M10's winning config)
    cfg_obj = TRAIN_CFG.__class__(**{**vars(TRAIN_CFG),
                                      "cql_alpha": 1.0, "seed": 0})
    agent = NeuroPolicyAgent(
        state_dim=env.state_dim, n_actions=env.action_space.n,
        n_classes=n_classes, cfg=cfg_obj, mode="scalar",
        cmdp_eps=0.10, gamma=0.99, polyak_tau=0.005, device=device,
        hidden=256, depth=3, seed=0,
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
    return {
        "encoder": encoder.spec.name,
        "encoder_pretrain": pretrain_info,
        "n_transitions": int(n_transitions),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_test_episodes": len(test_eps),
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
    log.info("Loaded bci2a sub 3: X=%s n_classes=%d ch_names=%s",
             tb.X.shape, n_classes, tb.ch_names[:5])
    rng = np.random.default_rng(0)
    perm = rng.permutation(tb.n_trials)
    n_tr = int(tb.n_trials * 0.70)
    n_va = int(tb.n_trials * 0.15)
    tr_idx = np.sort(perm[:n_tr])
    va_idx = np.sort(perm[n_tr:n_tr + n_va])
    te_idx = np.sort(perm[n_tr + n_va:])

    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    results = {
        "subject_id": int(tb.subject_id),
        "dataset_key": tb.dataset_key,
        "n_classes": int(n_classes),
        "ch_names": list(tb.ch_names),
        "n_trials": int(tb.n_trials),
    }

    # 1) EEGNet baseline reproduction (M10 scalar+CMDP config)
    log.info("\n=== EEGNet baseline ===")
    enc_eegnet = build_encoder("eegnet", n_channels=tb.n_channels,
                                n_samples=win_samples, sfreq=tb.sfreq).to(device)
    results["eegnet"] = run_one(enc_eegnet, tb, tr_idx, va_idx, te_idx,
                                  device, n_classes, pretrain_head=False)
    r = results["eegnet"]; t = r["test"]
    log.info("[eegnet]            test: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d",
             t["return_mean"], t["accuracy_on_commits"],
             t["information_transfer_rate"], t["wrong_commit_rate"], t["n_commits"])

    # 2) LaBraM pretrained
    log.info("\n=== LaBraM pretrained ===")
    enc_labram = build_encoder("labram", n_channels=tb.n_channels,
                                n_samples=win_samples, sfreq=tb.sfreq,
                                ch_names=list(tb.ch_names),
                                load_pretrained=True).to(device)
    log.info("LaBraM ckpt loaded keys: %d/%d (missing=%d)",
             enc_labram._n_loaded, enc_labram._n_total_ckpt, enc_labram._n_missing)
    results["labram_pretrained"] = run_one(enc_labram, tb, tr_idx, va_idx, te_idx,
                                            device, n_classes, pretrain_head=True)
    r = results["labram_pretrained"]; t = r["test"]
    log.info("[labram_pretrained] test: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d",
             t["return_mean"], t["accuracy_on_commits"],
             t["information_transfer_rate"], t["wrong_commit_rate"], t["n_commits"])

    # 3) LaBraM random-init (ablation: pretraining vs not)
    log.info("\n=== LaBraM random-init (no pretraining) ===")
    enc_labram_init = build_encoder("labram", n_channels=tb.n_channels,
                                     n_samples=win_samples, sfreq=tb.sfreq,
                                     ch_names=list(tb.ch_names),
                                     load_pretrained=False).to(device)
    results["labram_init"] = run_one(enc_labram_init, tb, tr_idx, va_idx, te_idx,
                                      device, n_classes, pretrain_head=True)
    r = results["labram_init"]; t = r["test"]
    log.info("[labram_init]       test: ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f n_commit=%d",
             t["return_mean"], t["accuracy_on_commits"],
             t["information_transfer_rate"], t["wrong_commit_rate"], t["n_commits"])

    out_dir = EXPERIMENTS_DIR / "m13_labram_bci2a"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")

    log.info("\n=== M13 SUMMARY (test split, scalar CQL + CMDP eps=0.10) ===")
    for k in ["eegnet", "labram_pretrained", "labram_init"]:
        t = results[k]["test"]
        log.info("%-20s ret=%+.3f acc=%.3f ITR=%.2f wrong=%.3f",
                 k, t["return_mean"], t["accuracy_on_commits"],
                 t["information_transfer_rate"], t["wrong_commit_rate"])


if __name__ == "__main__":
    main()

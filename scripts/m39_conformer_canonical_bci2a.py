# MIT License - Bryan Cheng, 2026
"""M39: EEG-Conformer (Song et al. 2023) mandatory classification on bci2a
4-class — canonical session-T → session-E protocol — SOTA-push experiment.

Goal: beat our M35 EEGNet-mandatory baseline (0.821 ± 0.07) and push toward
the leaderboard top (CTNet 0.825, Transformer-2025 0.865).

EEG-Conformer reproduces the published 78.66% number when run at 4-s full-
trial inputs (T=1000 samples at 250 Hz). We test two variants:
  * `conformer_full_trial`: original protocol, single forward pass per trial.
  * `conformer_windowavg`: our window-averaging recipe — 12 rolling 1-s
    windows per trial, average head logits, argmax. Tests whether the
    window-averaging trick that boosted EEGNet (+10pp) also helps the
    stronger encoder.

Both variants: 3 subjects {1, 3, 7} × 5 seeds. Canonical T→E split for
direct comparison to M35 EEGNet-mandatory and to published literature.

Output: experiments/m39_conformer_canonical_bci2a/summary.json
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
from src.models.encoder import build_encoder
from src.training.episode_builder import windows_from_trial
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m39_conformer")

SUBJECTS = [1, 3, 7]
SEEDS = [0, 1, 2, 3, 4]


def canonical_split(tb, val_frac=0.20, seed=0):
    train_pool = np.where(tb.session == 0)[0]
    test_idx = np.where(tb.session == 1)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_pool)
    n_val = int(round(val_frac * len(train_pool)))
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    return tr_idx, val_idx, np.sort(test_idx)


def train_classifier_full_trial(tb, tr_idx, va_idx, te_idx, n_classes, seed, device,
                                 n_epochs=80, batch_size=64, lr=2e-4, weight_decay=0):
    """Train EEG-Conformer at full-trial T (4-s @ 250 Hz). Returns test_acc, val_acc."""
    n_samples = tb.X.shape[-1]
    enc = build_encoder("conformer", n_channels=tb.n_channels,
                          n_samples=n_samples, sfreq=tb.sfreq).to(device)
    head = nn.Sequential(
        nn.Linear(enc.spec.embed_dim, 256), nn.ELU(), nn.Dropout(0.5),
        nn.Linear(256, 32), nn.ELU(), nn.Dropout(0.3),
        nn.Linear(32, n_classes),
    ).to(device)

    Xtr = torch.from_numpy(tb.X[tr_idx].astype("float32")).to(device)
    ytr = torch.from_numpy(tb.y[tr_idx].astype("int64")).to(device)
    Xva = torch.from_numpy(tb.X[va_idx].astype("float32")).to(device)
    yva = torch.from_numpy(tb.y[va_idx].astype("int64")).to(device)
    Xte = torch.from_numpy(tb.X[te_idx].astype("float32")).to(device)
    yte = torch.from_numpy(tb.y[te_idx].astype("int64")).to(device)

    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                            lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None

    for epoch in range(n_epochs):
        enc.train(); head.train()
        p = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], batch_size):
            idx = p[i:i + batch_size]
            feat = enc(Xtr[idx])
            logits = head(feat)
            loss = crit(logits, ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            va_pred = head(enc(Xva)).argmax(1)
            va_acc = float((va_pred == yva).float().mean())
        if va_acc > best_val:
            best_val = va_acc
            best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                           {k: v.detach().clone() for k, v in head.state_dict().items()})
    if best_state is not None:
        enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    enc.eval(); head.eval()
    with torch.no_grad():
        te_pred = head(enc(Xte)).argmax(1)
        te_acc = float((te_pred == yte).float().mean())
    return te_acc, best_val


def train_classifier_windowavg(tb, tr_idx, va_idx, te_idx, n_classes, seed, device,
                                 n_epochs=80, batch_size=128, lr=2e-4, weight_decay=0):
    """Train EEG-Conformer at 1-s rolling windows; train on windows, test by
    averaging head logits across windows of a trial."""
    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    enc = build_encoder("conformer", n_channels=tb.n_channels,
                          n_samples=win_samples, sfreq=tb.sfreq).to(device)
    head = nn.Sequential(
        nn.Linear(enc.spec.embed_dim, 256), nn.ELU(), nn.Dropout(0.5),
        nn.Linear(256, 32), nn.ELU(), nn.Dropout(0.3),
        nn.Linear(32, n_classes),
    ).to(device)

    def windows_for(idx):
        XW, yW = [], []
        for i in idx:
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
            XW.append(W); yW.extend([int(tb.y[i])] * W.shape[0])
        return (np.concatenate(XW, axis=0),
                np.asarray(yW, dtype=np.int64))

    Xw_tr, yw_tr = windows_for(tr_idx)
    Xw_va, yw_va = windows_for(va_idx)
    Xt_tr = torch.from_numpy(Xw_tr.astype("float32")).to(device)
    yt_tr = torch.from_numpy(yw_tr).to(device)
    Xt_va = torch.from_numpy(Xw_va.astype("float32")).to(device)
    yt_va = torch.from_numpy(yw_va).to(device)

    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                            lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None
    for epoch in range(n_epochs):
        enc.train(); head.train()
        p = torch.randperm(Xt_tr.shape[0], device=device)
        for i in range(0, Xt_tr.shape[0], batch_size):
            idx = p[i:i + batch_size]
            logits = head(enc(Xt_tr[idx]))
            loss = crit(logits, yt_tr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            preds = head(enc(Xt_va)).argmax(1)
            va_acc = float((preds == yt_va).float().mean())
        if va_acc > best_val:
            best_val = va_acc
            best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                           {k: v.detach().clone() for k, v in head.state_dict().items()})
    if best_state is not None:
        enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])

    enc.eval(); head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in te_idx:
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
            Xw = torch.from_numpy(W.astype("float32")).to(device)
            trial_logits = head(enc(Xw)).mean(dim=0)
            pred = int(trial_logits.argmax().item())
            if pred == int(tb.y[i]):
                correct += 1
            total += 1
    return correct / total, best_val


def run_subject(sid, device):
    tb = preprocess_subject("bci2a", subject_id=sid)
    n_classes = len(tb.class_labels)
    tr_idx, va_idx, te_idx = canonical_split(tb, val_frac=0.20, seed=0)
    log.info("Loaded bci2a sub %d: X=%s sessions=%s. Split: train=%d val=%d test=%d",
              sid, tb.X.shape,
              {int(s): int((tb.session == s).sum()) for s in np.unique(tb.session)},
              len(tr_idx), len(va_idx), len(te_idx))

    results = {
        "subject_id": sid, "n_trials": int(tb.n_trials),
        "n_classes": n_classes,
        "n_train_trials": len(tr_idx), "n_val_trials": len(va_idx),
        "n_test_trials": len(te_idx),
        "per_seed": {},
    }
    full_accs, win_accs = [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        full_acc, full_val = train_classifier_full_trial(
            tb, tr_idx, va_idx, te_idx, n_classes, seed, device,
        )
        t1 = time.time()
        log.info("  [%s] s=%d conformer_full_trial  test_acc=%.4f  val=%.3f  (%.0fs)",
                  f"sub{sid}", seed, full_acc, full_val, t1 - t0)

        win_acc, win_val = train_classifier_windowavg(
            tb, tr_idx, va_idx, te_idx, n_classes, seed, device,
        )
        t2 = time.time()
        log.info("  [%s] s=%d conformer_windowavg  test_acc=%.4f  val=%.3f  (%.0fs)",
                  f"sub{sid}", seed, win_acc, win_val, t2 - t1)

        results["per_seed"][seed] = {
            "conformer_full_trial": {"test_acc": float(full_acc), "val_acc": float(full_val),
                                       "elapsed_s": float(t1 - t0)},
            "conformer_windowavg":  {"test_acc": float(win_acc), "val_acc": float(win_val),
                                       "elapsed_s": float(t2 - t1)},
        }
        full_accs.append(full_acc); win_accs.append(win_acc)

    results["conformer_full_trial_mean"] = float(np.mean(full_accs))
    results["conformer_full_trial_std"] = float(np.std(full_accs, ddof=1))
    results["conformer_windowavg_mean"] = float(np.mean(win_accs))
    results["conformer_windowavg_std"] = float(np.std(win_accs, ddof=1))
    results["conformer_full_trial_per_seed"] = full_accs
    results["conformer_windowavg_per_seed"] = win_accs
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "m39_conformer_canonical_bci2a"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"subjects": SUBJECTS, "seeds": SEEDS, "per_subject": {}}
    for sid in SUBJECTS:
        log.info("\n\n========== SUBJECT %d ==========", sid)
        r = run_subject(sid, device)
        all_results["per_subject"][sid] = r
        (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))

    log.info("\n\n=== CROSS-SUBJECT AGGREGATE ===")
    full_subj = [all_results["per_subject"][s]["conformer_full_trial_mean"] for s in SUBJECTS]
    win_subj = [all_results["per_subject"][s]["conformer_windowavg_mean"] for s in SUBJECTS]
    log.info("  conformer_full_trial : per-sub %s  mean %.4f ± %.4f",
              [f"{a:.3f}" for a in full_subj], np.mean(full_subj), np.std(full_subj, ddof=1))
    log.info("  conformer_windowavg  : per-sub %s  mean %.4f ± %.4f",
              [f"{a:.3f}" for a in win_subj], np.mean(win_subj), np.std(win_subj, ddof=1))
    all_results["aggregate"] = {
        "conformer_full_trial_mean": float(np.mean(full_subj)),
        "conformer_full_trial_std_subjects": float(np.std(full_subj, ddof=1)),
        "conformer_windowavg_mean": float(np.mean(win_subj)),
        "conformer_windowavg_std_subjects": float(np.std(win_subj, ddof=1)),
        "conformer_full_trial_per_subject": full_subj,
        "conformer_windowavg_per_subject": win_subj,
    }
    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

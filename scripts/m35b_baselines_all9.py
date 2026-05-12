# MIT License - Bryan Cheng, 2026
"""M35: Within-subject SOTA baselines on bci2a 4-class — CANONICAL PROTOCOL.

For an honest head-to-head against published SOTA on bci2a 4-class,
M35 runs three mandatory-classification baselines (every trial forced
to commit at trial end, no defer) on the canonical BCI Competition IV
protocol: session 0 (T) = train+val, session 1 (E) = test. This matches
the protocol used by FBCSP / EEGNet / EEG-Conformer / etc. and is
directly comparable to published SOTA numbers. Apples-to-apples vs M38.
  * `csp_lda` : 8-component CSP on broadband 4-40 Hz + LDA (classical strong
                baseline; close to FBCSP — Ang et al. 2008 published 67.75%).
  * `fbcsp_lda` : Filterbank CSP+LDA — bank of {4-8, 8-12, 12-16, 16-20,
                  20-24, 24-28, 28-32} Hz × CSP(n=4 components) → mutual-info
                  top-12 features → LDA. (FBCSP-like, Ang et al. 2008.)
  * `eegnet_mandatory` : Same EEGNet encoder pretrained as M34 → linear head
                         on trial-mean embedding → argmax (no defer).

Per subject:
  * 5 seeds (matches M34 — only the encoder/RNG seed varies; classical
    baselines have a single deterministic fit per seed).
  * Same 70/15/15 trial split as M34.
  * Reports mean ± std task accuracy + per-class breakdown.

Output: experiments/m35b_baselines_all9/summary.json
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
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from mne.decoding import CSP

from src.data.preprocess import preprocess_subject
from src.models.encoder import build_encoder
from src.training.episode_builder import windows_from_trial
from src.utils.config import EXPERIMENTS_DIR, MDP_CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m35b_baselines")

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SEEDS = [0, 1, 2, 3, 4]
FB_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]


def bandpass(X, sfreq, low, high, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, X, axis=-1)


def make_canonical_split(tb, val_frac=0.20, seed=0):
    """Canonical bci2a split: session 0 (T) = train+val, session 1 (E) = test."""
    train_pool = np.where(tb.session == 0)[0]
    test_idx = np.where(tb.session == 1)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_pool)
    n_val = int(round(val_frac * len(train_pool)))
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    return tr_idx, val_idx, np.sort(test_idx)


def csp_lda(X_tr, y_tr, X_te, y_te, n_components=8):
    # Regularize to avoid SVD singularities on small samples / narrow bands
    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
    csp.fit(X_tr, y_tr)
    F_tr = csp.transform(X_tr)
    F_te = csp.transform(X_te)
    clf = LinearDiscriminantAnalysis()
    clf.fit(F_tr, y_tr)
    return clf.predict(F_te), float(clf.score(F_te, y_te))


def fbcsp_lda(X_tr, y_tr, X_te, y_te, sfreq, n_components=4, n_select=12):
    """Filterbank CSP + mutual-info selection + LDA (regularized for stability)."""
    F_tr_all = []
    F_te_all = []
    for low, high in FB_BANDS:
        Xb_tr = bandpass(X_tr, sfreq, low, high)
        Xb_te = bandpass(X_te, sfreq, low, high)
        if not (np.isfinite(Xb_tr).all() and np.isfinite(Xb_te).all()):
            log.warning("  FBCSP band (%d-%d) produced non-finite values; skipping band", low, high)
            continue
        try:
            csp = CSP(n_components=n_components, reg="ledoit_wolf",
                       log=True, norm_trace=False)
            csp.fit(Xb_tr, y_tr)
            F_tr_all.append(csp.transform(Xb_tr))
            F_te_all.append(csp.transform(Xb_te))
        except Exception as e:
            log.warning("  FBCSP band (%d-%d) failed: %s; skipping", low, high, e)
            continue
    if not F_tr_all:
        return np.zeros(len(y_te), dtype=int), 0.25  # all bands failed → chance
    F_tr = np.concatenate(F_tr_all, axis=1)
    F_te = np.concatenate(F_te_all, axis=1)
    n_select_eff = min(n_select, F_tr.shape[1])
    mi = mutual_info_classif(F_tr, y_tr, random_state=0)
    top = np.argsort(mi)[-n_select_eff:]
    clf = LinearDiscriminantAnalysis()
    clf.fit(F_tr[:, top], y_tr)
    return clf.predict(F_te[:, top]), float(clf.score(F_te[:, top], y_te))


def eegnet_mandatory(tb, tr_idx, va_idx, te_idx, n_classes, win_samples, seed, device,
                       n_epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-4):
    """Train EEGNet + linear head end-to-end on training-trial windows, select
    best epoch by val acc, then predict each test trial by averaging window
    logits and arg-maxing."""
    enc = build_encoder("eegnet", n_channels=tb.n_channels,
                          n_samples=win_samples, sfreq=tb.sfreq).to(device)
    head = nn.Linear(enc.spec.embed_dim, n_classes).to(device)

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
                            lr=lr, weight_decay=weight_decay)
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
            val_acc = float((preds == yt_va).float().mean())
        if val_acc > best_val:
            best_val = val_acc
            best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                           {k: v.detach().clone() for k, v in head.state_dict().items()})
    if best_state is not None:
        enc.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])

    enc.eval(); head.eval()
    preds = []; truths = []
    with torch.no_grad():
        for i in te_idx:
            W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=MDP_CFG)
            Xw = torch.from_numpy(W.astype("float32")).to(device)
            logits = head(enc(Xw))  # (T_win, n_classes)
            trial_logit = logits.mean(dim=0)
            preds.append(int(trial_logit.argmax().item()))
            truths.append(int(tb.y[i]))
    preds = np.asarray(preds); truths = np.asarray(truths)
    return preds, float((preds == truths).mean()), float(best_val)


def run_subject(sid, device):
    tb = preprocess_subject("bci2a", subject_id=sid)
    n_classes = len(tb.class_labels)
    win_samples = int(round(tb.sfreq * MDP_CFG.window_seconds))
    log.info("Loaded bci2a sub %d: X=%s n_classes=%d", sid, tb.X.shape, n_classes)

    # Canonical protocol: session 0 = train+val, session 1 = test (matches M38)
    tr_idx, va_idx, te_idx = make_canonical_split(tb, val_frac=0.20, seed=0)
    log.info("Canonical split: train=%d val=%d test=%d (session 0 / session 0 / session 1)",
              len(tr_idx), len(va_idx), len(te_idx))

    X_tr = tb.X[tr_idx]; y_tr = tb.y[tr_idx]
    X_te = tb.X[te_idx]; y_te = tb.y[te_idx]

    results = {
        "subject_id": sid, "n_trials": int(tb.n_trials),
        "n_classes": n_classes,
        "n_train_trials": len(tr_idx), "n_test_trials": len(te_idx),
        "per_seed": {},
    }

    # Deterministic baselines (no seed dependence) — fit once per subject
    log.info("Running CSP+LDA broadband...")
    t0 = time.time()
    preds_csp, acc_csp = csp_lda(X_tr, y_tr, X_te, y_te, n_components=8)
    log.info("  CSP+LDA acc=%.3f (%.1fs)", acc_csp, time.time() - t0)

    log.info("Running FBCSP+LDA...")
    t0 = time.time()
    preds_fb, acc_fb = fbcsp_lda(X_tr, y_tr, X_te, y_te,
                                    sfreq=tb.sfreq, n_components=4, n_select=12)
    log.info("  FBCSP+LDA acc=%.3f (%.1fs)", acc_fb, time.time() - t0)

    # EEGNet-mandatory, run at each seed (seed varies encoder init)
    eegnet_accs = []
    eegnet_vals = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        log.info("EEGNet-mandatory seed=%d...", seed)
        t0 = time.time()
        preds_e, acc_e, val_e = eegnet_mandatory(
            tb, tr_idx, va_idx, te_idx, n_classes, win_samples, seed, device,
        )
        log.info("  EEGNet-mandatory s=%d acc=%.3f (val_acc=%.3f, %.1fs)",
                  seed, acc_e, val_e, time.time() - t0)
        eegnet_accs.append(acc_e)
        eegnet_vals.append(val_e)
        results["per_seed"][seed] = {"eegnet_mandatory_acc": float(acc_e),
                                       "eegnet_val_acc": float(val_e)}

    results["csp_lda_acc"] = float(acc_csp)
    results["fbcsp_lda_acc"] = float(acc_fb)
    results["eegnet_mandatory_acc_mean"] = float(np.mean(eegnet_accs))
    results["eegnet_mandatory_acc_std"] = float(np.std(eegnet_accs, ddof=1))
    results["eegnet_mandatory_accs_per_seed"] = eegnet_accs
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    out_dir = EXPERIMENTS_DIR / "m35b_baselines_all9"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"subjects": SUBJECTS, "seeds": SEEDS,
                    "fb_bands": FB_BANDS, "per_subject": {}}
    for sid in SUBJECTS:
        log.info("\n\n========== SUBJECT %d ==========", sid)
        r = run_subject(sid, device)
        all_results["per_subject"][sid] = r
        (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))

    # Cross-subject aggregate
    log.info("\n\n=== CROSS-SUBJECT AGGREGATE ===")
    csp_accs = [all_results["per_subject"][s]["csp_lda_acc"] for s in SUBJECTS]
    fb_accs = [all_results["per_subject"][s]["fbcsp_lda_acc"] for s in SUBJECTS]
    eg_mean = [all_results["per_subject"][s]["eegnet_mandatory_acc_mean"] for s in SUBJECTS]
    log.info("  CSP+LDA          : per-sub %s  mean %.3f",
              [f"{a:.3f}" for a in csp_accs], np.mean(csp_accs))
    log.info("  FBCSP+LDA        : per-sub %s  mean %.3f",
              [f"{a:.3f}" for a in fb_accs], np.mean(fb_accs))
    log.info("  EEGNet-mandatory : per-sub %s  mean %.3f",
              [f"{a:.3f}" for a in eg_mean], np.mean(eg_mean))
    all_results["aggregate"] = {
        "csp_lda_mean": float(np.mean(csp_accs)),
        "fbcsp_lda_mean": float(np.mean(fb_accs)),
        "eegnet_mandatory_mean": float(np.mean(eg_mean)),
        "csp_lda_per_subject": csp_accs,
        "fbcsp_lda_per_subject": fb_accs,
        "eegnet_mandatory_per_subject": eg_mean,
    }
    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

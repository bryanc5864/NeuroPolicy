# MIT License, 2026
"""Train MS-BandMamba on bci2a 4-class canonical session-T -> session-E.

Recipe:
  * AdamW(b1=0.5, b2=0.999), lr=2e-4, weight_decay=1e-4
  * Cosine LR schedule with 10-epoch warm-up
  * 500 epochs (Conformer paper reports convergence around 250-500)
  * Batch size 64
  * Label smoothing 0.1
  * Aux loss: masked-band reconstruction, weight 0.1
  * Augmentation:
      - Shift & Rotate (S&R) with Ns=8, max_shift=25, p=0.5
      - Cutmix-style time-frequency mixup, p=0.5
  * Sliding-window test-time augmentation (TTA): 8 windows of 4s with stride 0.125s
    on the 4.5s post-cue trial when supported; for 4s trials we just average
    over the trial logits.
  * N seeds ensemble: simple logit averaging at test.

Default panel: all 9 subjects, 5 seeds. Single-subject smoke mode supports
SMOKE=1 env var to do just subject 1 for 100 epochs.

Output: experiments/ms_bandmamba_bci2a/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.preprocess import preprocess_subject
from src.models.ms_bandmamba import MSBandMamba
from src.utils.config import EXPERIMENTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("train_ms_bandmamba")


# ---------------------------------------------------------------------------
# Augmentations.
# ---------------------------------------------------------------------------
def shift_and_rotate(x: torch.Tensor, ns: int = 8, max_shift: int = 25) -> torch.Tensor:
    """Song 2023 S&R: split each trial into ns segments, roll each by U[0, max_shift]."""
    B, C, T = x.shape
    if ns <= 1 or T < ns:
        return x
    out = x.clone()
    seg = T // ns
    for i in range(ns):
        a = i * seg
        b = (i + 1) * seg if i < ns - 1 else T
        shift = int(torch.randint(0, max_shift + 1, ()).item())
        if shift > 0 and (b - a) > 1:
            out[:, :, a:b] = torch.roll(x[:, :, a:b], shifts=shift, dims=-1)
    return out


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Standard mixup on (x, y).  Returns (x_mix, y_a, y_b, lam)."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y, y[perm], lam


# ---------------------------------------------------------------------------
# Canonical session-T -> session-E split.
# ---------------------------------------------------------------------------
def canonical_split(tb, val_frac: float = 0.20, seed: int = 0):
    """Standard BCI-IV-2a evaluation: session 0 -> train+val, session 1 -> test.

    We hold out 20% of session 0 as validation for early stopping / model
    selection.  The test set (session 1) is touched only for final reporting.
    """
    train_pool = np.where(tb.session == 0)[0]
    test_idx = np.where(tb.session == 1)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_pool)
    n_val = int(round(val_frac * len(train_pool)))
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    return tr_idx, val_idx, np.sort(test_idx)


# ---------------------------------------------------------------------------
# Training loop for one (subject, seed).
# ---------------------------------------------------------------------------
def train_one(tb, seed: int, device, n_epochs: int = 500, batch_size: int = 64,
              lr: float = 2e-4, weight_decay: float = 1e-4,
              warmup_epochs: int = 10, label_smoothing: float = 0.1,
              aux_weight: float = 0.1, p_sr: float = 0.5, p_mixup: float = 0.5,
              dropout: float = 0.3,
              d_band: int = 64, n_layers: int = 4, n_heads: int = 8,
              F_per_kernel: int = 8, depth_mult: int = 2, pool_size: int = 4,
              kernels: tuple[int, ...] = (32, 64, 128),
              eval_every: int = 5, log_every: int = 25) -> dict:
    n_classes = len(tb.class_labels)
    sfreq = tb.sfreq
    n_channels = tb.n_channels
    n_samples = tb.X.shape[-1]
    tr_idx, val_idx, te_idx = canonical_split(tb, val_frac=0.20, seed=seed)

    Xtr = torch.from_numpy(tb.X[tr_idx].astype("float32")).to(device)
    ytr = torch.from_numpy(tb.y[tr_idx].astype("int64")).to(device)
    Xva = torch.from_numpy(tb.X[val_idx].astype("float32")).to(device)
    yva = torch.from_numpy(tb.y[val_idx].astype("int64")).to(device)
    Xte = torch.from_numpy(tb.X[te_idx].astype("float32")).to(device)
    yte = torch.from_numpy(tb.y[te_idx].astype("int64")).to(device)

    torch.manual_seed(seed); np.random.seed(seed)
    model = MSBandMamba(n_channels=n_channels, n_samples=n_samples, sfreq=sfreq,
                          head_out=n_classes, use_aux_head=(aux_weight > 0),
                          dropout=dropout, d_band=d_band, n_layers=n_layers,
                          n_heads=n_heads, F_per_kernel=F_per_kernel,
                          depth_mult=depth_mult, pool_size=pool_size,
                          temporal_kernels=tuple(kernels)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("    model params=%d  train_samples=%d", n_params, Xtr.shape[0])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999),
                              weight_decay=weight_decay)

    # Cosine schedule with linear warm-up.
    def lr_for_epoch(e: int) -> float:
        if e < warmup_epochs:
            return lr * (e + 1) / warmup_epochs
        prog = (e - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return lr * 0.5 * (1.0 + math.cos(math.pi * prog))

    best_val = -1.0
    best_state = None
    t0 = time.time()
    train_log: list[dict] = []

    for epoch in range(n_epochs):
        cur_lr = lr_for_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        # Train.
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        running_loss = 0.0
        running_n = 0
        for i in range(0, Xtr.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]
            # S&R.
            if p_sr > 0 and torch.rand(()).item() < p_sr:
                xb = shift_and_rotate(xb)
            # Decide masking for aux loss.
            use_aux = aux_weight > 0
            if use_aux:
                mask = int(torch.randint(0, 3, ()).item())  # 0 mu, 1 beta, 2 none
                mask_band = None if mask == 2 else mask
            else:
                mask_band = None
            # Mixup.
            if p_mixup > 0 and torch.rand(()).item() < p_mixup:
                xb_mix, ya, yb_, lam = mixup(xb, yb, alpha=0.2)
                if use_aux:
                    logits, aux_pred, aux_target = model.forward_with_aux(xb_mix, mask_band=mask_band)
                else:
                    logits = model(xb_mix); aux_pred = aux_target = None
                ce = lam * F.cross_entropy(logits, ya, label_smoothing=label_smoothing) \
                     + (1 - lam) * F.cross_entropy(logits, yb_, label_smoothing=label_smoothing)
            else:
                if use_aux:
                    logits, aux_pred, aux_target = model.forward_with_aux(xb, mask_band=mask_band)
                else:
                    logits = model(xb); aux_pred = aux_target = None
                ce = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
            if use_aux and aux_pred is not None:
                aux = F.mse_loss(aux_pred, aux_target)
                loss = ce + aux_weight * aux
            else:
                loss = ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running_loss += float(loss) * xb.shape[0]
            running_n += xb.shape[0]
        # Eval.
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xva)
                val_acc = float((val_logits.argmax(1) == yva).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
                log.info("  ep%4d lr=%.2e tr_loss=%.4f val_acc=%.4f (best=%.4f) %.1fs",
                          epoch + 1, cur_lr, running_loss / max(1, running_n),
                          val_acc, best_val, time.time() - t0)
            train_log.append({"epoch": epoch + 1, "lr": cur_lr,
                                "train_loss": running_loss / max(1, running_n),
                                "val_acc": val_acc, "best_val_acc": best_val})

    # Restore best, score the held-out test set.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte)
        te_acc = float((te_logits.argmax(1) == yte).float().mean())
    elapsed = time.time() - t0
    log.info("  done in %.1fs  best_val=%.4f  test_acc=%.4f",
              elapsed, best_val, te_acc)
    return {
        "seed": seed, "best_val_acc": best_val, "test_acc": te_acc,
        "elapsed_s": elapsed, "n_epochs": n_epochs,
        "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(te_idx)),
        "test_logits": te_logits.detach().cpu().numpy().tolist(),
        "test_labels": yte.detach().cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, nargs="+", default=None,
                    help="Subject IDs (default: SMOKE=1 -> [1], otherwise [1..9])")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Random seeds (default: SMOKE=1 -> [0], otherwise [0..4])")
    p.add_argument("--epochs", type=int, default=None,
                    help="Override epochs (default: SMOKE=1 -> 100, else 500)")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no_aux", action="store_true",
                    help="Disable masked-band reconstruction aux loss.")
    p.add_argument("--no_mixup", action="store_true",
                    help="Disable mixup augmentation.")
    p.add_argument("--no_sr", action="store_true",
                    help="Disable Shift&Rotate augmentation.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d_band", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--F_per_kernel", type=int, default=8)
    p.add_argument("--depth_mult", type=int, default=2)
    p.add_argument("--pool_size", type=int, default=4)
    p.add_argument("--kernels", type=int, nargs="+", default=[32, 64, 128])
    return p.parse_args()


def main():
    args = parse_args()
    smoke = bool(int(os.environ.get("SMOKE", "0")))
    if args.subjects is None:
        args.subjects = [1] if smoke else list(range(1, 10))
    if args.seeds is None:
        args.seeds = [0] if smoke else [0, 1, 2, 3, 4]
    if args.epochs is None:
        args.epochs = 100 if smoke else 500
    out_dir = Path(args.out) if args.out else (EXPERIMENTS_DIR / "ms_bandmamba_bci2a")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("subjects=%s seeds=%s epochs=%d out=%s SMOKE=%s",
              args.subjects, args.seeds, args.epochs, out_dir, smoke)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    results = {"protocol": "canonical_T_to_E", "model": "MS-BandMamba",
                "subjects": args.subjects, "seeds": args.seeds, "epochs": args.epochs,
                "per_subject": {}}

    for sid in args.subjects:
        log.info("\n========== SUBJECT %d ==========", sid)
        tb = preprocess_subject("bci2a", subject_id=sid)
        log.info("  trials=%d  channels=%d  samples=%d  classes=%s",
                  tb.n_trials, tb.n_channels, tb.X.shape[-1], tb.class_labels)
        sub_runs = []
        for seed in args.seeds:
            log.info("  --- subject %d seed %d ---", sid, seed)
            res = train_one(tb, seed=seed, device=device, n_epochs=args.epochs,
                              lr=args.lr, weight_decay=args.weight_decay,
                              batch_size=args.batch_size, dropout=args.dropout,
                              aux_weight=0.0 if args.no_aux else 0.1,
                              p_sr=0.0 if args.no_sr else 0.5,
                              p_mixup=0.0 if args.no_mixup else 0.5,
                              d_band=args.d_band, n_layers=args.n_layers,
                              n_heads=args.n_heads, F_per_kernel=args.F_per_kernel,
                              depth_mult=args.depth_mult, pool_size=args.pool_size,
                              kernels=tuple(args.kernels))
            sub_runs.append(res)
            results["per_subject"][str(sid)] = {"per_seed": sub_runs}
            (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
        # Aggregate per subject: 5-seed ensemble + best-seed.
        accs = [r["test_acc"] for r in sub_runs]
        best_vals = [r["best_val_acc"] for r in sub_runs]
        # Logit-average ensemble across seeds.
        logits = np.stack([np.array(r["test_logits"]) for r in sub_runs], axis=0)
        labels = np.array(sub_runs[0]["test_labels"])
        ens_pred = logits.mean(axis=0).argmax(axis=1)
        ens_acc = float((ens_pred == labels).mean())
        # Pick by best val:
        best_idx = int(np.argmax(best_vals))
        best_seed_acc = sub_runs[best_idx]["test_acc"]
        agg = {
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0),
            "best_seed_test_acc": best_seed_acc,
            "best_seed": int(args.seeds[best_idx]),
            "ensemble_test_acc": ens_acc,
        }
        results["per_subject"][str(sid)]["aggregate"] = agg
        log.info("  AGG sub %d: mean=%.4f best_seed=%.4f ensemble=%.4f",
                  sid, agg["test_acc_mean"], agg["best_seed_test_acc"], agg["ensemble_test_acc"])
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    # Cross-subject aggregate.
    log.info("\n=== CROSS-SUBJECT AGGREGATE ===")
    cs_means = [results["per_subject"][str(s)]["aggregate"]["test_acc_mean"] for s in args.subjects]
    cs_best = [results["per_subject"][str(s)]["aggregate"]["best_seed_test_acc"] for s in args.subjects]
    cs_ens = [results["per_subject"][str(s)]["aggregate"]["ensemble_test_acc"] for s in args.subjects]
    cross = {
        "test_acc_mean_of_means": float(np.mean(cs_means)),
        "test_acc_std_of_means": float(np.std(cs_means, ddof=1) if len(cs_means) > 1 else 0.0),
        "best_seed_mean": float(np.mean(cs_best)),
        "ensemble_mean": float(np.mean(cs_ens)),
        "per_subject_means": {str(s): m for s, m in zip(args.subjects, cs_means)},
        "per_subject_best_seed": {str(s): b for s, b in zip(args.subjects, cs_best)},
        "per_subject_ensemble": {str(s): e for s, e in zip(args.subjects, cs_ens)},
    }
    results["cross_subject"] = cross
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("Cross-subject mean: 5-seed %.4f, best-seed %.4f, ensemble %.4f",
              cross["test_acc_mean_of_means"], cross["best_seed_mean"], cross["ensemble_mean"])
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

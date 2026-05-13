# MIT License, 2026
"""Generic mandatory-classifier trainer on bci2a 4-class canonical session-T -> session-E.

Designed to compare encoders head-to-head under identical training conditions
(Song 2023 recipe + S&R augmentation).  Pass --encoder to select between
``conformer`` (baseline), ``banded_conformer`` (mu/beta IFNet-style front-end
+ Conformer), or ``ms_bandmamba``.

Output: experiments/mandatory_bci2a_<encoder>/summary.json
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
import torch.nn as nn
import torch.nn.functional as F

from src.data.preprocess import preprocess_subject
from src.models.encoder import build_encoder
from src.utils.config import EXPERIMENTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("train_mandatory")


def shift_and_rotate(x: torch.Tensor, ns: int = 8, max_shift: int = 25) -> torch.Tensor:
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


def canonical_split(tb, val_frac: float = 0.20, seed: int = 0):
    train_pool = np.where(tb.session == 0)[0]
    test_idx = np.where(tb.session == 1)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_pool)
    n_val = int(round(val_frac * len(train_pool)))
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    return tr_idx, val_idx, np.sort(test_idx)


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)


def train_one(tb, encoder_name: str, seed: int, device, n_epochs: int = 250,
              batch_size: int = 72, lr: float = 2e-4, weight_decay: float = 1e-4,
              warmup_epochs: int = 10, label_smoothing: float = 0.1,
              p_sr: float = 0.5, eval_every: int = 5, log_every: int = 25,
              dropout: float | None = None) -> dict:
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
    enc = build_encoder(encoder_name, n_channels=n_channels, n_samples=n_samples,
                         sfreq=sfreq, ch_names=list(tb.ch_names) if hasattr(tb, "ch_names") else None)
    enc = enc.to(device)
    model = EncoderClassifier(enc, embed_dim=enc.spec.embed_dim, n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("    model=%s params=%d  train=%d val=%d test=%d",
              enc.spec.name, n_params, Xtr.shape[0], Xva.shape[0], Xte.shape[0])

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999),
                            weight_decay=weight_decay)

    def lr_for_epoch(e: int) -> float:
        if e < warmup_epochs:
            return lr * (e + 1) / warmup_epochs
        prog = (e - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return lr * 0.5 * (1.0 + math.cos(math.pi * prog))

    best_val = -1.0
    best_state = None
    t0 = time.time()
    for epoch in range(n_epochs):
        cur_lr = lr_for_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        running_loss = 0.0
        running_n = 0
        for i in range(0, Xtr.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx]; yb = ytr[idx]
            if p_sr > 0 and torch.rand(()).item() < p_sr:
                xb = shift_and_rotate(xb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += float(loss) * xb.shape[0]
            running_n += xb.shape[0]
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xva)
                val_acc = float((val_logits.argmax(1) == yva).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
                log.info("    ep%4d lr=%.2e tr_loss=%.4f val_acc=%.4f (best=%.4f) %.1fs",
                          epoch + 1, cur_lr, running_loss / max(1, running_n),
                          val_acc, best_val, time.time() - t0)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte)
        te_acc = float((te_logits.argmax(1) == yte).float().mean())
    elapsed = time.time() - t0
    log.info("    done in %.1fs  best_val=%.4f  test_acc=%.4f", elapsed, best_val, te_acc)
    return {
        "seed": seed, "encoder": enc.spec.name, "best_val_acc": best_val,
        "test_acc": te_acc, "elapsed_s": elapsed, "n_epochs": n_epochs,
        "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)), "n_test": int(len(te_idx)),
        "n_params": n_params,
        "test_logits": te_logits.detach().cpu().numpy().tolist(),
        "test_labels": yte.detach().cpu().numpy().tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True,
                    choices=["conformer", "banded_conformer", "ms_bandmamba",
                              "eegnet", "ctnet"])
    p.add_argument("--subjects", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=72)
    p.add_argument("--no_sr", action="store_true")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    smoke = bool(int(os.environ.get("SMOKE", "0")))
    if args.subjects is None:
        args.subjects = [1] if smoke else list(range(1, 10))
    if args.seeds is None:
        args.seeds = [0] if smoke else [0, 1, 2, 3, 4]
    if args.epochs is None:
        args.epochs = 100 if smoke else 250
    out_dir = Path(args.out) if args.out else (EXPERIMENTS_DIR / f"mandatory_bci2a_{args.encoder}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("encoder=%s subjects=%s seeds=%s epochs=%d out=%s SMOKE=%s",
              args.encoder, args.subjects, args.seeds, args.epochs, out_dir, smoke)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    results = {"protocol": "canonical_T_to_E", "encoder": args.encoder,
                "subjects": args.subjects, "seeds": args.seeds, "epochs": args.epochs,
                "per_subject": {}}

    for sid in args.subjects:
        log.info("\n========== SUBJECT %d ==========", sid)
        tb = preprocess_subject("bci2a", subject_id=sid)
        log.info("  trials=%d  channels=%d  samples=%d",
                  tb.n_trials, tb.n_channels, tb.X.shape[-1])
        sub_runs = []
        for seed in args.seeds:
            log.info("  --- subject %d seed %d ---", sid, seed)
            res = train_one(tb, encoder_name=args.encoder, seed=seed, device=device,
                              n_epochs=args.epochs, lr=args.lr,
                              weight_decay=args.weight_decay,
                              batch_size=args.batch_size,
                              p_sr=0.0 if args.no_sr else 0.5)
            sub_runs.append(res)
            results["per_subject"][str(sid)] = {"per_seed": sub_runs}
            (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
        accs = [r["test_acc"] for r in sub_runs]
        best_vals = [r["best_val_acc"] for r in sub_runs]
        logits = np.stack([np.array(r["test_logits"]) for r in sub_runs], axis=0)
        labels = np.array(sub_runs[0]["test_labels"])
        ens_pred = logits.mean(axis=0).argmax(axis=1)
        ens_acc = float((ens_pred == labels).mean())
        best_idx = int(np.argmax(best_vals))
        agg = {
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0),
            "best_seed_test_acc": sub_runs[best_idx]["test_acc"],
            "best_seed": int(args.seeds[best_idx]),
            "ensemble_test_acc": ens_acc,
        }
        results["per_subject"][str(sid)]["aggregate"] = agg
        log.info("  AGG sub %d: mean=%.4f best_seed=%.4f ensemble=%.4f",
                  sid, agg["test_acc_mean"], agg["best_seed_test_acc"], agg["ensemble_test_acc"])
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    log.info("\n=== CROSS-SUBJECT AGGREGATE (%s) ===", args.encoder)
    cs_means = [results["per_subject"][str(s)]["aggregate"]["test_acc_mean"] for s in args.subjects]
    cs_best = [results["per_subject"][str(s)]["aggregate"]["best_seed_test_acc"] for s in args.subjects]
    cs_ens = [results["per_subject"][str(s)]["aggregate"]["ensemble_test_acc"] for s in args.subjects]
    results["cross_subject"] = {
        "test_acc_mean_of_means": float(np.mean(cs_means)),
        "test_acc_std_of_means": float(np.std(cs_means, ddof=1) if len(cs_means) > 1 else 0.0),
        "best_seed_mean": float(np.mean(cs_best)),
        "ensemble_mean": float(np.mean(cs_ens)),
        "per_subject_means": {str(s): m for s, m in zip(args.subjects, cs_means)},
    }
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    log.info("Cross-subject mean (5-seed)=%.4f  ensemble=%.4f",
              results["cross_subject"]["test_acc_mean_of_means"], results["cross_subject"]["ensemble_mean"])
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

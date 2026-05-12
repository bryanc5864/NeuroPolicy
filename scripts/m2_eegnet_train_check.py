# MIT License - Bryan Cheng, 2026
"""Train EEGNet supervised on bci2a sub 3 to verify the encoder can learn.

Not a baseline — just a backbone-validity check before building the MDP/RL
pipeline on top. Reference target: within-subject 5-fold CV ~ 0.65-0.85
for sub 3 (CSP+LDA achieved 0.82 in M1 sanity).
"""
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data.preprocess import preprocess_subject
from src.models.encoder import build_encoder
from src.utils.config import EXPERIMENTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("m2_eegnet")


class EEGNetClassifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.spec.embed_dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def train_one_fold(X_tr, y_tr, X_te, y_te, sfreq, n_classes, device,
                   n_epochs=80, batch_size=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = build_encoder("eegnet", n_channels=X_tr.shape[1], n_samples=X_tr.shape[2], sfreq=sfreq).to(device)
    model = EEGNetClassifier(enc, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)
    n = X_tr_t.shape[0]

    best_test = 0.0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_tr_t[idx])
            loss = crit(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        # eval
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(dim=1)
            acc = float((preds == y_te_t).float().mean())
        best_test = max(best_test, acc)
    return best_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    tb = preprocess_subject("bci2a", subject_id=3)
    log.info("bci2a sub 3: X=%s y=%s n_classes=%d", tb.X.shape, tb.y.shape, len(tb.class_labels))

    # 1) Untrained-encoder linear probe
    enc_un = build_encoder("eegnet", n_channels=tb.n_channels,
                            n_samples=tb.n_times, sfreq=tb.sfreq).to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, tb.n_trials, 64):
            xb = torch.from_numpy(tb.X[i:i + 64]).to(device)
            feats.append(enc_un(xb).cpu().numpy())
    Z = np.concatenate(feats, axis=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    probe_accs = []
    for tr, te in skf.split(Z, tb.y):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Z[tr], tb.y[tr])
        probe_accs.append(clf.score(Z[te], tb.y[te]))
    probe_mean, probe_std = float(np.mean(probe_accs)), float(np.std(probe_accs))
    log.info("Untrained linear probe acc = %.3f ± %.3f", probe_mean, probe_std)

    # 2) End-to-end EEGNet (best-of-epoch test acc)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    fold_times = []
    for f, (tr, te) in enumerate(skf.split(tb.X, tb.y)):
        t0 = time.time()
        acc = train_one_fold(tb.X[tr], tb.y[tr], tb.X[te], tb.y[te],
                             sfreq=tb.sfreq, n_classes=len(tb.class_labels),
                             device=device, n_epochs=80, batch_size=64, lr=1e-3, seed=f)
        elapsed = time.time() - t0
        log.info("fold %d  best test acc = %.3f  (%.1fs)", f, acc, elapsed)
        accs.append(acc); fold_times.append(elapsed)
    end_mean, end_std = float(np.mean(accs)), float(np.std(accs))
    log.info("EEGNet end-to-end CV acc = %.3f ± %.3f  (CSP+LDA was 0.821 on this subject)",
             end_mean, end_std)

    # 3) Save JSON
    out_dir = EXPERIMENTS_DIR / "m2_eegnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "subject_id": tb.subject_id, "dataset_key": tb.dataset_key,
        "n_trials": int(tb.n_trials), "n_channels": int(tb.n_channels),
        "n_times": int(tb.n_times), "sfreq": float(tb.sfreq),
        "n_classes": len(tb.class_labels),
        "untrained_probe": {
            "fold_accs": [float(x) for x in probe_accs],
            "mean": probe_mean, "std": probe_std,
        },
        "end_to_end": {
            "fold_accs": [float(x) for x in accs],
            "mean": end_mean, "std": end_std,
            "fold_times_s": [float(x) for x in fold_times],
            "config": {"n_epochs": 80, "batch_size": 64, "lr": 1e-3,
                       "weight_decay": 1e-4, "best_of_epoch": True},
        },
        "csp_lda_reference": 0.821,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

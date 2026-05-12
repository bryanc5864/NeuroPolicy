# MIT License - Bryan Cheng, 2026
"""Smoke test EEGNetEncoder forward shapes + linear-probe accuracy."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data.preprocess import preprocess_subject
from src.models.encoder import build_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("smoke_encoder")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    tb = preprocess_subject("bci2a", subject_id=3)  # the strong-MI subject
    log.info("Loaded bci2a subj 3: X=%s y=%s", tb.X.shape, tb.y.shape)

    enc = build_encoder("eegnet", n_channels=tb.n_channels, n_samples=tb.n_times, sfreq=tb.sfreq).to(device)
    log.info("Built %s; embed_dim=%d  total_params=%d",
             enc.spec.name, enc.spec.embed_dim, sum(p.numel() for p in enc.parameters()))

    # Forward shape check
    x = torch.from_numpy(tb.X[:8]).to(device)  # (8, C, T)
    with torch.no_grad():
        z = enc(x)
    log.info("Forward: in=%s -> out=%s", tuple(x.shape), tuple(z.shape))
    assert z.shape == (8, enc.spec.embed_dim)

    # Linear-probe with **untrained** encoder + LogReg, 5-fold CV.
    # Untrained EEGNet should still beat chance via random projection of band-passed signal.
    enc.eval()
    feats = []
    with torch.no_grad():
        for batch_start in range(0, tb.n_trials, 64):
            xb = torch.from_numpy(tb.X[batch_start:batch_start + 64]).to(device)
            zb = enc(xb).cpu().numpy()
            feats.append(zb)
    Z = np.concatenate(feats, axis=0)
    log.info("Untrained-EEGNet features: shape=%s mean=%.3f std=%.3f",
             Z.shape, Z.mean(), Z.std())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(Z, tb.y):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Z[tr], tb.y[tr])
        accs.append(clf.score(Z[te], tb.y[te]))
    log.info("Linear-probe (random-init EEGNet) acc = %.3f ± %.3f  (chance=0.25 for 4-class)",
             float(np.mean(accs)), float(np.std(accs)))


if __name__ == "__main__":
    main()

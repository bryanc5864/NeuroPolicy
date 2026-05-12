# MIT License - Bryan Cheng, 2026
"""Verify per-channel mean/std after preprocessing equals (0, 1)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.preprocess import preprocess_subject

tb = preprocess_subject("bci2b", subject_id=1)
X = tb.X  # (n_trials, n_channels, n_times)
per_ch = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
print("per-channel mean:", per_ch.mean(axis=1))
print("per-channel std :", per_ch.std(axis=1))
print("global mean     :", X.mean())
print("global std      :", X.std())
print("global var      :", X.var())
print("per-channel var :", per_ch.var(axis=1))
print("mean of per-ch var:", per_ch.var(axis=1).mean())

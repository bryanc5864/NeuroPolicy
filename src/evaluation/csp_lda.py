# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""CSP + LDA: classical baseline used for  data-pipeline validation.

Per RESEARCH_PLAN.md , we verify that our preprocessing yields
decodable signal by reproducing within-subject CSP+LDA accuracy in line with
literature ranges:
    BCI-IV-2a (4-class) : 60-72% within-subject
    BCI-IV-2b (2-class) : 70-80% within-subject
    Lee2019  (2-class)  : 65-75% within-subject

Note: this is plain CSP, not the filter-bank (FBCSP) variant. FBCSP is built
in src/evaluation/baselines/fbcsp.py for the  baselines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.moabb_loader import TrialBatch

logger = logging.getLogger(__name__)


@dataclass
class CVAccuracyResult:
    subject_id: int
    dataset_key: str
    fold_accs: list[float]
    mean_acc: float
    std_acc: float

    def asdict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "dataset_key": self.dataset_key,
            "fold_accs": [float(x) for x in self.fold_accs],
            "mean_acc": float(self.mean_acc),
            "std_acc": float(self.std_acc),
        }


def csp_lda_cv(
    tb: TrialBatch,
    n_components: int = 6,
    n_splits: int = 5,
    seed: int = 0,
) -> CVAccuracyResult:
    """Stratified k-fold within-subject accuracy of CSP+LDA on a TrialBatch.

    Splits are at the trial level (no window-level leakage).
    """
    X, y = tb.X, tb.y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs: list[float] = []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        n_comp = min(n_components, X.shape[1] - 1)  # CSP needs C > components
        n_comp = max(n_comp, 2)
        pipe = Pipeline([
            ("csp", CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        pipe.fit(X[tr], y[tr])
        acc = float(pipe.score(X[te], y[te]))
        fold_accs.append(acc)
        logger.debug(
            "subj=%d fold=%d acc=%.3f", tb.subject_id, fold_idx, acc,
        )
    return CVAccuracyResult(
        subject_id=tb.subject_id,
        dataset_key=tb.dataset_key,
        fold_accs=fold_accs,
        mean_acc=float(np.mean(fold_accs)),
        std_acc=float(np.std(fold_accs, ddof=0)),
    )

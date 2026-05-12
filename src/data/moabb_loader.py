# MIT License - Bryan Cheng, 2026
# Part of NeuroPolicy / ieeeICIST
"""Wrappers around MOABB datasets that expose raw MNE recordings.

Per RESEARCH_PLAN.md §5.1, we work with three datasets accessed via MOABB:
- BNCI2014_001 (BCI Competition IV-2a, 4-class MI)
- BNCI2014_004 (BCI Competition IV-2b, 2-class MI)
- Lee2019_MI    (54-subject 2-class MI)

We bypass MOABB's `Paradigm.get_data()` short-cut so we can apply our own
preprocessing (matching the plan's §5.2). Instead we use `dataset.get_data()`
which returns raw MNE objects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mne
import numpy as np

from src.utils.config import DATA_CACHE, DATASETS, DatasetSpec

mne.set_log_level("ERROR")
logger = logging.getLogger(__name__)


@dataclass
class TrialBatch:
    """Container for a subject's epoched, preprocessed trials.

    Shape conventions (numpy):
      X       : (n_trials, n_channels, n_times)  float32
      y       : (n_trials,) int64
      session : (n_trials,) int (session index, 0-based)
      run     : (n_trials,) int (run index within session, 0-based)
      sfreq   : float, sampling rate after resampling
      ch_names: list[str]
      class_labels : list[str], len = n_classes
    """
    X: np.ndarray
    y: np.ndarray
    session: np.ndarray
    run: np.ndarray
    sfreq: float
    ch_names: list[str]
    class_labels: list[str]
    subject_id: int
    dataset_key: str

    def __post_init__(self):
        assert self.X.ndim == 3, f"X must be 3-D, got {self.X.shape}"
        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.dtype == np.float32, f"X dtype {self.X.dtype}, want float32"
        assert self.y.dtype == np.int64, f"y dtype {self.y.dtype}, want int64"

    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.X.shape[2])


def _moabb_dataset(spec: DatasetSpec):
    """Instantiate a MOABB dataset object by ID."""
    from moabb.datasets import BNCI2014_001, BNCI2014_004, Lee2019_MI

    table = {
        "BNCI2014_001": BNCI2014_001,
        "BNCI2014_004": BNCI2014_004,
        "Lee2019_MI": Lee2019_MI,
    }
    if spec.moabb_id not in table:
        raise ValueError(f"Unknown MOABB id: {spec.moabb_id}")
    return table[spec.moabb_id]()


def list_subjects(dataset_key: str) -> list[int]:
    """Return the list of subject IDs MOABB defines for this dataset."""
    spec = DATASETS[dataset_key]
    ds = _moabb_dataset(spec)
    return list(ds.subject_list)


def iter_subject_recordings(
    dataset_key: str,
    subjects: list[int] | None = None,
    cache_root: Path = DATA_CACHE,
) -> Iterator[tuple[int, dict[str, dict[str, mne.io.BaseRaw]]]]:
    """Yield (subject_id, sessions_dict) for each requested subject.

    sessions_dict has the MOABB structure {session_id: {run_id: Raw}}.
    Downloads on first access; cached under cache_root/mne_data.
    """
    import os
    mne_dir = cache_root / "mne_data"
    moabb_dir = cache_root / "moabb_data"
    mne_dir.mkdir(parents=True, exist_ok=True)
    moabb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DATA", str(mne_dir))
    os.environ.setdefault("MOABB_DATA", str(moabb_dir))
    # Also set MNE config so MOABB's download paths work
    try:
        import mne as _mne
        _mne.set_config("MNE_DATA", str(mne_dir), set_env=True)
    except Exception:
        pass

    spec = DATASETS[dataset_key]
    ds = _moabb_dataset(spec)
    subjects = subjects or list(ds.subject_list)
    for sid in subjects:
        logger.info("Fetching %s subject %d", dataset_key, sid)
        data = ds.get_data(subjects=[sid])
        # data: {subject_id: {session_id: {run_id: Raw}}}
        if sid not in data:
            raise RuntimeError(f"MOABB returned no data for subject {sid}")
        yield sid, data[sid]

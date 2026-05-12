# MIT License - Bryan Cheng, 2026
# Part of NeuroPolicy / ieeeICIST
"""EEG preprocessing pipeline (RESEARCH_PLAN.md §5.2).

Steps applied to each Raw recording before epoching:
  1. Resample to PRE_CFG.target_fs (250 Hz) if needed.
  2. Band-pass 4-40 Hz (zero-phase IIR Butterworth, order 4).
  3. Common-average re-reference.
  4. (Optional) ICA-based ocular artifact removal via mne-icalabel.
  5. Epoch to [0, 4] s post-cue using event annotations.
  6. Per-subject z-score across (channel, time) per recording.

The intentional default for `apply_ica` is False during pipeline shake-down
because BCI-IV-2a/2b have varying EOG-channel availability and ICA can fail
silently on too-few channels. Lee2019 (62 ch) is well-suited to ICA and we
flip the default for that dataset in `preprocess_subject(...)`.
This is documented as an explicit deviation from §5.2 step 5; we will toggle
ICA back on for the full sweep once the rest of the pipeline is verified.
"""
from __future__ import annotations

import logging

import mne
import numpy as np

from src.data.moabb_loader import TrialBatch, iter_subject_recordings
from src.utils.config import DATASETS, PRE_CFG, DatasetSpec, PreprocessConfig

mne.set_log_level("ERROR")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class-label canonicalization
# ---------------------------------------------------------------------------
# MOABB encodes labels as strings. We map to int64 in a deterministic order
# per dataset so y is comparable across subjects.
CLASS_ORDER: dict[str, list[str]] = {
    "bci2a": ["left_hand", "right_hand", "feet", "tongue"],
    "bci2b": ["left_hand", "right_hand"],
    "lee2019": ["left_hand", "right_hand"],
}


def _label_to_int(dataset_key: str, label: str) -> int:
    order = CLASS_ORDER[dataset_key]
    if label not in order:
        raise ValueError(f"Unexpected class label {label!r} for {dataset_key}")
    return order.index(label)


# ---------------------------------------------------------------------------
# Per-Raw preprocessing
# ---------------------------------------------------------------------------
def preprocess_raw(
    raw: mne.io.BaseRaw,
    cfg: PreprocessConfig = PRE_CFG,
    apply_ica: bool = False,
) -> mne.io.BaseRaw:
    """Filter / resample / re-reference one MNE Raw; returns a new Raw."""
    raw = raw.copy().load_data()

    # 1. Resample
    if abs(raw.info["sfreq"] - cfg.target_fs) > 1e-3:
        raw = raw.resample(cfg.target_fs)

    # 2. Band-pass filter (zero-phase IIR Butterworth)
    raw = raw.filter(
        l_freq=cfg.bandpass_low,
        h_freq=cfg.bandpass_high,
        method="iir",
        iir_params=dict(order=cfg.filter_order, ftype="butter"),
        verbose="ERROR",
    )

    # 3. Common-average re-reference (EEG only; pick eligible channels)
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
    if len(eeg_picks) == 0:
        # Some MOABB datasets register channels as misc; fall back.
        raw = raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw = raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # 4. ICA-based ocular removal (optional, off by default)
    if apply_ica:
        raw = _apply_ica_ocular(raw, cfg)

    return raw


def _apply_ica_ocular(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> mne.io.BaseRaw:
    """Best-effort ICA + EOG component removal via mne-icalabel.

    Wrapped in a wide try/except: on any failure we log and return the input
    unchanged. The plan calls for ICA, but a failure must not silently corrupt
    data — better to skip and document.
    """
    try:
        from mne.preprocessing import ICA
        from mne_icalabel import label_components

        n_components = min(cfg.ica_n_components, raw.info["nchan"] - 1)
        ica = ICA(
            n_components=n_components,
            random_state=cfg.ica_random_state,
            method="infomax",
            fit_params=dict(extended=True),
        )
        ica.fit(raw)
        labels = label_components(raw, ica, method="iclabel")
        # ICLabel categories: ['brain','muscle artifact','eye blink',
        #                      'heart beat','line noise','channel noise','other']
        cat = labels["labels"]
        proba = labels["y_pred_proba"]
        eog_indices = [
            i for i, (c, p) in enumerate(zip(cat, proba))
            if c == "eye blink" and p >= cfg.eog_label_threshold
        ]
        if eog_indices:
            ica.exclude = eog_indices
            raw = ica.apply(raw)
            logger.info("ICA removed %d EOG components", len(eog_indices))
    except Exception as e:
        logger.warning("ICA step skipped: %s", e)
    return raw


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------
def epoch_raw(
    raw: mne.io.BaseRaw,
    dataset_key: str,
    cfg: PreprocessConfig = PRE_CFG,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cut Raw into trial epochs in [tmin, tmax] post-cue.

    Returns (X, y, ch_names) where X has shape (n_trials, n_channels, n_times)
    in float32, y in int64.
    """
    spec = DATASETS[dataset_key]
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if not event_id:
        raise RuntimeError(f"No events found for {dataset_key}")

    # Filter event_id to the target classes only.
    target_labels = CLASS_ORDER[dataset_key]
    keep_event_id = {k: v for k, v in event_id.items() if k in target_labels}
    if not keep_event_id:
        raise RuntimeError(
            f"No target events {target_labels} found in raw events {list(event_id)}"
        )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=keep_event_id,
        tmin=cfg.epoch_tmin,
        tmax=cfg.epoch_tmax,
        baseline=None,
        preload=True,
        picks="eeg",
        reject_by_annotation=False,
        verbose="ERROR",
    )

    # Build aligned y.
    inv_id = {v: k for k, v in keep_event_id.items()}
    y = np.array(
        [_label_to_int(dataset_key, inv_id[ev[2]]) for ev in epochs.events],
        dtype=np.int64,
    )
    X = epochs.get_data(copy=True).astype(np.float32, copy=False)
    # Rescale volts -> microvolts so downstream code sees ~unit-scale signals.
    X *= np.float32(cfg.volts_to_uv)
    if X.shape[0] != y.shape[0]:
        raise RuntimeError("Epoch/label length mismatch")
    return X, y, list(epochs.ch_names)


# ---------------------------------------------------------------------------
# Per-subject normalization
# ---------------------------------------------------------------------------
def zscore_per_recording(X: np.ndarray, eps: float = PRE_CFG.zscore_eps) -> np.ndarray:
    """Per-(channel) z-score across the recording's time samples.

    NOTE: We normalize per-channel using stats pooled across trials (treating
    the epoch concatenation as one stream). This avoids leaking trial-level
    means but uses subject-pooled stats. No leakage of label info.
    """
    # X: (n_trials, n_channels, n_times)
    flat = X.transpose(1, 0, 2).reshape(X.shape[1], -1)  # (C, T_all)
    mu = flat.mean(axis=1, keepdims=True)
    sigma = flat.std(axis=1, keepdims=True) + eps
    flat = (flat - mu) / sigma
    return flat.reshape(X.shape[1], X.shape[0], X.shape[2]).transpose(1, 0, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Driver: full per-subject preprocessing
# ---------------------------------------------------------------------------
def preprocess_subject(
    dataset_key: str,
    subject_id: int,
    apply_ica: bool | None = None,
    cfg: PreprocessConfig = PRE_CFG,
) -> TrialBatch:
    """Run the full preprocessing pipeline on one subject and return a TrialBatch."""
    spec = DATASETS[dataset_key]
    if apply_ica is None:
        apply_ica = (dataset_key == "lee2019")  # ICA default-on for 62-ch dataset only

    X_parts, y_parts, sess_parts, run_parts = [], [], [], []
    ch_names_ref: list[str] | None = None
    sfreq_ref: float | None = None

    for sid, sessions in iter_subject_recordings(dataset_key, subjects=[subject_id]):
        for sess_idx, (sess_name, runs) in enumerate(sessions.items()):
            for run_idx, (run_name, raw) in enumerate(runs.items()):
                raw_pp = preprocess_raw(raw, cfg=cfg, apply_ica=apply_ica)
                X, y, chs = epoch_raw(raw_pp, dataset_key, cfg=cfg)
                if X.shape[0] == 0:
                    logger.warning(
                        "Subject %d session %s run %s yielded 0 epochs",
                        sid, sess_name, run_name,
                    )
                    continue
                X_parts.append(X)
                y_parts.append(y)
                sess_parts.append(np.full(X.shape[0], sess_idx, dtype=np.int64))
                run_parts.append(np.full(X.shape[0], run_idx, dtype=np.int64))
                ch_names_ref = ch_names_ref or chs
                sfreq_ref = sfreq_ref or float(raw_pp.info["sfreq"])

    if not X_parts:
        raise RuntimeError(f"No usable trials for {dataset_key} subject {subject_id}")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    sess = np.concatenate(sess_parts, axis=0)
    run = np.concatenate(run_parts, axis=0)
    X = zscore_per_recording(X, eps=cfg.zscore_eps)

    return TrialBatch(
        X=X,
        y=y,
        session=sess,
        run=run,
        sfreq=sfreq_ref,
        ch_names=ch_names_ref,
        class_labels=CLASS_ORDER[dataset_key],
        subject_id=subject_id,
        dataset_key=dataset_key,
    )

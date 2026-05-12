# MIT License, 2026
# Part of NeuroPolicy / ieeeICIST
"""Convert preprocessed TrialBatch -> rolling windows -> encoded TrialEpisodes.

2.1 we discretize each trial into windows of length
L = 1.0 s with stride Δ = 0.25 s. For a 4 s trial @ 250 Hz this yields T = 13
windows. We encode all windows up front so the MDP env can be cheap.

Subject embeddings:
* `subj_embed_default`     : mean-encoded features over the train-split of
                             the trial's subject (or pooled mean across subjects
                             when the subject is unseen).
* `subj_embed_post_recal`  : mean-encoded features over the per-subject
                             calibration buffer (held-out trials).
The default and post-recal vectors live in the same R^{D_s} space; we use a
fixed random projection so subj_dim is decoupled from encoder embed_dim and
the env can be agnostic to the encoder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from src.data.moabb_loader import TrialBatch
from src.models.encoder import BaseEncoder
from src.training.bci_env import TrialEpisode
from src.utils.config import MDP_CFG, MDPConfig

logger = logging.getLogger(__name__)


def windows_from_trial(
    X_trial: np.ndarray, sfreq: float, cfg: MDPConfig = MDP_CFG,
) -> np.ndarray:
    """Cut one (C, T_total) trial into (T_windows, C, T_window).

    L = cfg.window_seconds, stride = cfg.stride_seconds.
    """
    n_channels, n_samples = X_trial.shape
    win_samples = int(round(cfg.window_seconds * sfreq))
    stride_samples = int(round(cfg.stride_seconds * sfreq))
    if win_samples > n_samples:
        raise ValueError(
            f"Window {win_samples} exceeds trial length {n_samples}"
        )
    starts = list(range(0, n_samples - win_samples + 1, stride_samples))
    out = np.stack([X_trial[:, s:s + win_samples] for s in starts], axis=0)
    return out.astype(np.float32, copy=False)


@torch.no_grad()
def encode_windows(
    X_windows: np.ndarray,
    encoder: BaseEncoder,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run encoder on (B, C, T_window) -> (B, embed_dim) numpy."""
    encoder.eval()
    feats = []
    for i in range(0, X_windows.shape[0], batch_size):
        xb = torch.from_numpy(X_windows[i:i + batch_size]).to(device)
        zb = encoder(xb).detach().cpu().numpy()
        feats.append(zb)
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)


@dataclass
class SubjectEmbeddingFactory:
    """Builds D_s-dim subject embeddings from encoder features.

    A fixed random projection from encoder embed_dim -> subj_dim, applied to
    the mean encoder feature over a set of trials. Frozen at construction.
    Same projection used for default and post-recal so they live in the
    same space.
    """
    encoder_embed_dim: int
    subj_dim: int = 16
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # Orthonormal-ish projection (Gaussian rows, normalized).
        proj = rng.standard_normal((self.subj_dim, self.encoder_embed_dim)).astype(np.float32)
        proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)
        self._proj = proj  # (D_s, D_enc)

    def project_mean(self, encoder_feats: np.ndarray) -> np.ndarray:
        """encoder_feats: (n, D_enc) -> (D_s,)."""
        if encoder_feats.size == 0:
            return np.zeros(self.subj_dim, dtype=np.float32)
        mu = encoder_feats.mean(axis=0).astype(np.float32)
        return (self._proj @ mu).astype(np.float32)


def build_episodes_for_trial_batch(
    tb: TrialBatch,
    encoder: BaseEncoder,
    device: torch.device,
    subj_factory: SubjectEmbeddingFactory,
    pooled_default_feats: np.ndarray | None = None,
    calibration_indices: np.ndarray | None = None,
    cfg: MDPConfig = MDP_CFG,
    subj_embeds: tuple[np.ndarray, np.ndarray] | None = None,
    return_train_episodes_only: bool = False,
) -> tuple[list[TrialEpisode], np.ndarray, np.ndarray]:
    """Encode all trials into TrialEpisode objects.

    Args:
        tb                        : preprocessed trial batch (typically one
                                    train/val/test slice for one subject).
        encoder                   : Encoder module (in eval mode).
        device                    : torch device.
        subj_factory              : produces subject embeddings.
        pooled_default_feats      : optional (n_pool, D_enc) — features pooled
                                    over training trials, used to compute the
                                    default subject embedding shared across
                                    train/val/test.
        calibration_indices       : indices of `tb` reserved for the recal
                                    action. They will form
                                    subj_embed_post_recal and be EXCLUDED
                                    from the returned episodes. Pass an
                                    empty array to keep all trials as
                                    episodes (typical for val/test).
        subj_embeds               : explicit (e_subj_default, e_subj_post_recal)
                                    tuple. **Takes precedence** over
                                    pooled_default_feats and calibration_indices
                                    for embedding construction. This is the
                                    leak-safe path: pass the train-derived
                                    pair to all val/test builds so episodes
                                    across splits share identical embeddings.
        return_train_episodes_only: convenience flag — currently unused.

    Returns:
        (episodes, encoded_feats_all_trials, calibration_indices_used)
    """
    n_trials = tb.n_trials
    if calibration_indices is None:
        # Use last `holdout_frac` of trials per class as calibration.
        from src.utils.config import PRE_CFG
        holdout = []
        for c in range(int(tb.y.max()) + 1):
            idx_c = np.where(tb.y == c)[0]
            n_hold = max(1, int(round(len(idx_c) * PRE_CFG.calibration_holdout_frac)))
            holdout.extend(idx_c[-n_hold:].tolist())
        calibration_indices = np.array(sorted(holdout), dtype=np.int64)

    cal_set = set(calibration_indices.tolist())

    # 1) Slice each trial into windows.
    all_windows: list[np.ndarray] = []
    trial_index_by_window: list[int] = []
    for i in range(n_trials):
        W = windows_from_trial(tb.X[i], sfreq=tb.sfreq, cfg=cfg)
        all_windows.append(W)
        trial_index_by_window.extend([i] * W.shape[0])
    W_concat = np.concatenate(all_windows, axis=0)  # (sum_T, C, T_win)
    trial_index_by_window = np.asarray(trial_index_by_window, dtype=np.int64)

    # 2) Encode all windows in one shot.
    feats = encode_windows(W_concat, encoder, device)
    # split feats back per trial (T_i, D_enc)
    feats_per_trial: list[np.ndarray] = []
    cursor = 0
    for W in all_windows:
        n_t = W.shape[0]
        feats_per_trial.append(feats[cursor:cursor + n_t])
        cursor += n_t
    assert cursor == feats.shape[0]

    # 3) Build subject embeddings.
    if subj_embeds is not None:
        # LEAK-SAFE PATH: caller supplies precomputed embeddings (typically
        # derived from training data only). Used by val/test builds to ensure
        # their states share the train-derived default and post-recal vectors.
        e_subj_default, e_subj_post_recal = subj_embeds
        e_subj_default = np.asarray(e_subj_default, dtype=np.float32)
        e_subj_post_recal = np.asarray(e_subj_post_recal, dtype=np.float32)
        if e_subj_default.shape != (subj_factory.subj_dim,):
            raise ValueError(
                f"e_subj_default shape {e_subj_default.shape} != ({subj_factory.subj_dim},)"
            )
        if e_subj_post_recal.shape != (subj_factory.subj_dim,):
            raise ValueError(
                f"e_subj_post_recal shape {e_subj_post_recal.shape} != ({subj_factory.subj_dim},)"
            )
    else:
        # Train-time path: compute embeddings from THIS batch's data.
        cal_feat_pool = []
        for idx in calibration_indices:
            cal_feat_pool.append(feats_per_trial[idx])
        cal_feats = (np.concatenate(cal_feat_pool, axis=0)
                     if cal_feat_pool else np.zeros((0, feats.shape[1]), np.float32))
        e_subj_post_recal = subj_factory.project_mean(cal_feats)

        if pooled_default_feats is not None and pooled_default_feats.size > 0:
            e_subj_default = subj_factory.project_mean(pooled_default_feats)
        else:
            non_cal_feats = []
            for idx in range(n_trials):
                if idx in cal_set:
                    continue
                non_cal_feats.append(feats_per_trial[idx])
            if non_cal_feats:
                e_subj_default = subj_factory.project_mean(np.concatenate(non_cal_feats, axis=0))
            else:
                e_subj_default = np.zeros(subj_factory.subj_dim, dtype=np.float32)

    # 4) Construct TrialEpisodes (excluding calibration trials).
    episodes: list[TrialEpisode] = []
    for i in range(n_trials):
        if i in cal_set:
            continue
        episodes.append(TrialEpisode(
            window_features=feats_per_trial[i].astype(np.float32, copy=False),
            label=int(tb.y[i]),
            subject_id=int(tb.subject_id),
            subj_embed_default=e_subj_default.astype(np.float32, copy=False),
            subj_embed_post_recal=e_subj_post_recal.astype(np.float32, copy=False),
            stride_seconds=cfg.stride_seconds,
        ))
    return episodes, feats, calibration_indices

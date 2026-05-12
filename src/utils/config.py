# MIT License - Bryan Cheng, 2026
# Part of NeuroPolicy / ieeeICIST
"""Global config constants for NeuroPolicy.

Centralizes paths and signal-processing defaults so every module agrees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE = PROJECT_ROOT / "data_cache"        # MOABB downloads here
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

for _d in (DATA_CACHE, EXPERIMENTS_DIR, LOGS_DIR, FIGURES_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing knobs from RESEARCH_PLAN.md §5.2."""
    target_fs: float = 250.0
    bandpass_low: float = 4.0
    bandpass_high: float = 40.0
    filter_order: int = 4
    epoch_tmin: float = 0.0
    epoch_tmax: float = 4.0
    reference: str = "average"
    ica_n_components: int = 20
    ica_random_state: int = 97
    eog_label_threshold: float = 0.8
    calibration_holdout_frac: float = 0.10
    # MNE returns EEG in volts (~1e-5 magnitude). We rescale to microvolts up
    # front so all downstream code sees ~unit-scale signals; eps here is a
    # true numerical guard, not a competitor with the data scale.
    volts_to_uv: float = 1.0e6
    zscore_eps: float = 1e-8


@dataclass(frozen=True)
class DatasetSpec:
    """One MOABB dataset config."""
    moabb_id: str
    paradigm: str            # "MotorImagery" | "LeftRightImagery"
    n_classes: int
    n_channels: int
    name: str
    sfreq_native: float
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "bci2a": DatasetSpec(
        moabb_id="BNCI2014_001",
        paradigm="MotorImagery",
        n_classes=4,
        n_channels=22,
        name="BCI Competition IV - 2a",
        sfreq_native=250.0,
    ),
    "bci2b": DatasetSpec(
        moabb_id="BNCI2014_004",
        paradigm="LeftRightImagery",
        n_classes=2,
        n_channels=3,
        name="BCI Competition IV - 2b",
        sfreq_native=250.0,
    ),
    "lee2019": DatasetSpec(
        moabb_id="Lee2019_MI",
        paradigm="LeftRightImagery",
        n_classes=2,
        n_channels=62,
        name="Lee2019 Motor Imagery",
        sfreq_native=1000.0,
        notes="Downsampled to 250 Hz",
    ),
}


@dataclass(frozen=True)
class MDPConfig:
    """BCI-as-MDP simulator parameters from RESEARCH_PLAN.md §3.2.1."""
    window_seconds: float = 1.0
    stride_seconds: float = 0.25
    recal_time_cost_seconds: float = 0.5

    # Reward shaping defaults (sweepable)
    r_correct: float = 1.0
    r_wrong: float = 5.0
    r_abstain: float = 0.5
    r_recal: float = 0.3
    lambda_per_sec: float = 0.05


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 256
    n_grad_steps: int = 100_000
    encoder_lr: float = 1.0e-4
    policy_lr: float = 1.0e-4
    critic_lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    n_quantiles: int = 31
    cvar_alpha: float = 0.25
    cmdp_eps: float = 0.10
    cmdp_lr: float = 1.0e-3
    cql_alpha: float = 1.0
    bf16_encoder: bool = True


# Singletons
PRE_CFG = PreprocessConfig()
MDP_CFG = MDPConfig()
TRAIN_CFG = TrainConfig()

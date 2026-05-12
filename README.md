# NeuroPolicy

**Risk-Aware Offline RL with Calibrated Off-Policy Evaluation for Motor-Imagery BCI Decoding**

A complete codebase for an offline-reinforcement-learning framework that
turns single-trial motor-imagery EEG decoding into a sequential decision
problem with a learned action space
`{Commit_k, Defer, Recal, Abstain}`, trained with Conservative Q-Learning
(scalar and CVaR variants) under an optional CMDP commit-error
constraint, and evaluated under a calibrated off-policy-evaluation (OPE)
pipeline validated against on-policy Monte-Carlo ground truth.

The codebase reproduces every figure and table in `paper/main.pdf`
(IEEE conference format, 10 pages, anonymized for blind review).

---

## Installation

Tested on Python 3.11, PyTorch 2.6+cu124, single NVIDIA RTX 3080 (10 GB
VRAM), Linux / Windows / WSL.

```bash
git clone <repo-url>
cd NeuroPolicy
pip install -r requirements.txt
```

System requirements:
- Python 3.11
- NVIDIA GPU with CUDA 12.4 (for training) — runs on CPU for inference / data analysis
- ~10 GB free disk for MOABB / MNE data caches (auto-downloaded on first run)

---

## Datasets

All three datasets are public motor-imagery EEG corpora, fetched
automatically via [MOABB](https://moabb.neurotechx.com) on first run:

| Dataset            | MOABB key            | Subjects | Channels | Classes |
|-------------------|----------------------|----------|----------|---------|
| BCI Competition IV-2b | `BNCI2014_004`    | 9        | 3        | 2       |
| BCI Competition IV-2a | `BNCI2014_001`    | 9        | 22       | 4       |
| Lee2019 MI            | `Lee2019_MI`      | 54       | 62       | 2       |

The first run will download ~5 GB of EEG data into `data_cache/`.

---

## Reproducing all experiments

The full sequence below reproduces every numerical claim in the paper.
Total wall-time is ~24 GPU-hours on an RTX 3080.

### Setup (sanity checks)

```bash
# Data pipeline sanity (CSP+LDA baseline, ~3 min)
python scripts/sanity_check_csp_lda.py

# EEGNet backbone verification (~5 min)
python scripts/eegnet_backbone_check.py

# MDP environment + behavior policies smoke test (~30 s)
python scripts/mdp_env_smoke_test.py
```

### Within-subject experiments

```bash
# Scalar CQL agent on BCI-IV-2b sub. 4 (~5 min)
python scripts/within_subject_train.py

# Calibrated OPE — 12-policy panel (~30 min)
python scripts/ope_calibration.py

# CVaR-CQL + CMDP within-subject 4-way ablation (~10 min)
python scripts/cvar_cmdp_within_subject.py

# Multi-class within-subject on BCI-IV-2a sub. 3 (~10 min)
python scripts/bci2a_within_subject.py

# Lee2019 sub. 1 within-subject (~5 min)
python scripts/lee2019_within_subject.py

# Multi-seed within-subject (5 seeds × 4 configs) (~25 min)
python scripts/multiseed_within_subject.py

# CVaR-alpha sensitivity sweep (~10 min)
python scripts/cvar_alpha_sweep.py
```

### Cross-subject LOSO classification

```bash
# BCI-IV-2b LOSO (9 subjects, 3 CQL configs, ~40 min)
python scripts/loso_bci2b_cql.py

# BCI-IV-2a 4-class LOSO (~40 min)
python scripts/loso_bci2a_cql.py

# Lee2019 (62-ch) LOSO — first published benchmark (~30 min)
python scripts/loso_classification_lee2019.py

# Strong-baseline LOSO comparison (CSP+LDA, EEGNet fixed/threshold) (~3 min)
python scripts/baselines_loso_bci2b.py
```

### Foundation-model (LaBraM) integration

```bash
# LaBraM within-subject (frozen + fine-tuned) (~20 min)
python scripts/labram_bci2a_within.py
python scripts/labram_finetune_within.py

# LaBraM LOSO on both datasets (~3 h total)
python scripts/labram_loso_bci2b.py
python scripts/labram_loso_bci2a.py

# Compare LaBraM vs EEGNet head-to-head
python scripts/compare_labram_vs_eegnet_bci2b.py
python scripts/compare_labram_vs_eegnet_bci2a.py
```

### Cross-subject OPE benchmark

```bash
# OPE LOSO on BCI-IV-2b (~20 min)
python scripts/ope_loso_bci2b.py
python scripts/ope_loso_bci2b_full.py

# OPE LOSO on BCI-IV-2a (~25 min)
python scripts/ope_loso_bci2a.py

# OPE LOSO on Lee2019 (~30 min)
python scripts/ope_loso_lee2019.py
python scripts/ope_loso_lee2019_extended.py

# Multi-method OPE benchmark (FQE/PDIS/WIS/DR), full 9-subject (~15 min)
python scripts/multimethod_ope_loso_full.py

# Deployment-rule analyses
python scripts/deployment_rule_validation.py
python scripts/bootstrap_ci_deployment_rule.py
python scripts/fqe_bias_decomposition.py
python scripts/slope_corrected_fqe.py
python scripts/wis_deployment_full_panel.py
python scripts/wis_stochastic_subset.py
python scripts/unified_ope_predictor.py

# Action-utilization audit (88 trained agents)
python scripts/action_utilization_audit.py
```

### SOTA comparison on BCI-IV-2a 4-class canonical protocol

```bash
# NeuroPolicy on canonical session-T -> session-E split (~70 min)
python scripts/neuropolicy_canonical_bci2a.py

# Strong-baseline 9-subject baselines (CSP, FBCSP, EEGNet+winavg) (~5 min)
python scripts/within_subject_baselines_all9.py

# EEG-Conformer encoder, canonical protocol, all 9 subjects (~60 min)
python scripts/conformer_canonical_bci2a_all9.py

# Aggregate the SOTA leaderboard
python scripts/sota_leaderboard_aggregator.py
```

### Build the paper PDF

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# -> paper/main.pdf (10 pages, IEEE conference format, anonymized)
```

---

## Repository layout

```
NeuroPolicy/
├── LICENSE                # MIT
├── README.md              # this file
├── requirements.txt       # pinned Python deps
├── src/
│   ├── data/              # preprocessing + MOABB loader
│   ├── models/            # encoders (EEGNet, EEG-Conformer, LaBraM)
│   ├── training/          # MDP env, behavior policies, RL agent, replay buffer
│   ├── evaluation/        # policy_eval, OPE estimators (FQE/PDIS/WIS/DR), calibration
│   └── utils/             # central config (datasets, MDP params, training params)
├── scripts/               # reproducible experiment scripts (~40 entries)
├── experiments/           # per-experiment summary.json outputs
├── figures/               # paper figures (PNG + PDF)
├── paper/
│   ├── main.tex           # IEEE conference paper (10 pages with appendices)
│   ├── sections/          # introduction, methods, results, discussion, related work
│   ├── refs.bib           # bibliography (28 verified entries)
│   └── main.pdf           # built paper
└── tests/                 # unit tests (quantile-loss check, etc.)
```

---

## Key results

- **Calibrated OPE**: FQE attains Pearson r = 0.860 vs on-policy ground truth on a 12-policy panel (first published calibrated OPE for BCI motor imagery)
- **Canonical-protocol selective decoding**: 86.2 ± 5.8% commit accuracy at 34.2 bits/min ITR on BCI-IV-2a 4-class with `Defer` as a learned action
- **Lee2019 LOSO**: 0.696 ± 0.104 commit accuracy (random 0.481), 10/10 subjects win, paired Wilcoxon p = 0.001 — first published Lee2019 LOSO benchmark for offline-RL BCI
- **Multi-method OPE benchmark**: WIS gives 3× lower bias than FQE on RMSE (9/9 subjects, p = 0.002), revealing a calibration-rank decoupling
- **LaBraM RL fine-tuning**: first offline-RL fine-tuning of an EEG foundation model in BCI; ties EEGNet on aggregate, with a subject-specific rescue effect (~+16pp)

Every numerical claim traces to a specific `summary.json` in `experiments/`.

---

## License

MIT — see `LICENSE`.

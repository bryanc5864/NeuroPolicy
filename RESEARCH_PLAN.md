# NeuroPolicy: Risk-Aware Offline RL with Causal Off-Policy Evaluation on Foundation-Model-Initialized BCI Policies

**Field**: Brain-Computer Interfaces / Neural Signal Processing — Adaptive Control
**Constraints**: Data — public MOABB datasets only, ≤ 50 GB total | Compute — single NVIDIA RTX 3080 (10 GB VRAM), CUDA 12.4, PyTorch 2.6
**Date**: 2026-05-03
**License**: MIT — Bryan Cheng, 2026
**Target venue**: IEEE ICIST 2026 (Coimbra, Portugal, Oct 1-6, 2026); 8-page IEEE format
**Topic match**: Intelligent Control & Automation (adaptive control, learning and adaptive control), Intelligent Information Processing (machine learning, neural signal processing, time-series analysis)

---

## 1. Abstract

> **Note (2026-05-05).** The originally planned abstract (preserved
> below) overclaims relative to results actually obtained on the RTX
> 3080 within the project compute envelope. The defensible, results-
> aligned abstract is in `paper/abstract.md` (and is the abstract
> used in `paper/main.tex`). This section is preserved verbatim as
> the project's North Star — i.e., what the paper *would* claim if
> the LaBraM swap and Lee2019 LOSO sweep also landed.

### 1.0 Planned (North Star, not yet supported by results)

The Brain-Computer Interface literature evaluates "policies" by replaying trials through a swapped-in classifier — a procedure that has no behavior policy, no propensity, and no doubly-robust estimator, and is therefore not off-policy evaluation in any rigorous sense. We reframe within-trial Motor-Imagery (MI) EEG decoding as an **offline reinforcement learning** problem with a clinically-meaningful action space `{commit-class₁, …, commit-class_K, DEFER, REQUEST_RECAL, ABSTAIN}`, optimized under a **risk-sensitive (CVaR-CQL)** objective with a **CMDP** safety constraint on commit-error rate. The policy/value networks are initialized from a frozen LaBraM EEG foundation model — the first RL fine-tuning of an EEG foundation model in the literature. We deliver a **rigorous off-policy evaluation pipeline** for BCI (per-decision importance sampling + doubly-robust + fitted-Q evaluation, with confidence intervals) and validate the OPE estimator against an on-policy simulator built from the logged trials. The artifact is the learned **Pareto frontier over (Information Transfer Rate, accuracy, mean latency, recalibration rate, abstain rate)** on BCI Competition IV-2a, IV-2b, and Lee2019 — beating EEGNet+threshold dynamic-stopping, MarkovType-style POMDP, BTSPRT, EEG_RL-Net, and Bianchi & Liti early-stopping on at least two of three datasets across the dominant axes.

### 1.1 Results-aligned (live; matches `paper/abstract.md`, `paper/main.tex`)

Brain-computer interfaces (BCIs) decode user intent at fixed-length windows and evaluate "policy" alternatives on-policy in a deterministic simulator — a procedure with no behavior policy, no propensity, and no rigorous off-policy machinery. We reformulate within-trial Motor-Imagery EEG decoding as an offline reinforcement-learning problem with a clinically meaningful action space `{commit-class₁, …, commit-class_K, DEFER, REQUEST_RECAL, ABSTAIN}` and contribute, as the principal methodological result, **the first calibrated off-policy evaluation pipeline for BCI**: fitted-Q evaluation (FQE) trained per target policy, validated against on-policy Monte-Carlo ground truth across a 12-policy panel that spans random, temperature-perturbed Conservative Q-Learning agents, and stronger agents at higher conservatism. On BCI-IV-2b subject 4 we obtain Pearson r = 0.860 (p = 3.3 × 10⁻⁴), Spearman ρ = 0.797, and RMSE = 0.637 — the first calibrated OPE estimate published for this task. Within-subject, the trained agent reaches +0.09 episode return at 96.0% commit accuracy and 19.13 bits/min ITR (Δ +1.78 over random on the same test episodes; M9 reproduction). Cross-subject leave-one-subject-out on BCI-IV-2b (N = 9) yields a positive but not significant mean improvement (Δ +0.21 episode return; paired Wilcoxon p = 0.10 one-sided). Against EEGNet-based hand-tuned dynamic-stopping baselines using the same encoder, NeuroPolicy ties or slightly loses on aggregate return (best baseline -1.29 vs. NeuroPolicy -1.37); against classical CSP+LDA fixed-window the agent significantly improves (Δ +0.44, paired Wilcoxon p = 0.037). The bimodal LOSO pattern (5 strong wins, 3 regressions, 1 tie) and the lack of differentiation between Conservative-Q-Learning conservatism levels suggest encoder feature quality, not the offline-RL machinery, is the cross-subject bottleneck — motivating a foundation-model encoder swap as the natural follow-up. Code, configurations, and per-subject artifacts are released under MIT.

---

## 2. Background & Motivation

### 2.1 The BCI speed-accuracy tradeoff
A motor-imagery BCI must decide *what* the user intends and *when* to commit that decision. Standard pipelines (FBCSP + LDA, EEGNet, EEG-Conformer) decode at a fixed window (typically 0–4 s post-cue), leaving a sharp, hand-tuned tradeoff: longer windows raise accuracy but kill Information Transfer Rate (ITR — the operational throughput metric that matters for end-users). Dynamic-stopping classifiers (Verschore & Kindermans 2012, Bianchi & Liti 2023) and SPRT variants (Liu et al. 2017, BTSPRT) introduce evidence-based early termination, but these are *hand-tuned thresholds on a confidence statistic*, not learned policies.

### 2.2 Sequential decision frameworks for BCI
A few recent papers (2024–2025) frame BCI decoding as sequential decision-making: **MarkovType** (arXiv:2412.15862, 2024) casts an RSVP P300 speller as a POMDP solved with on-policy REINFORCE; **EEG_RL-Net** (arXiv:2405.00723, 2024) trains a Dueling DQN classification head on a GNN backbone, online and without a DEFER action; **ErrP-driven RL** (arXiv:2502.18594, 2025) adapts a decoder online using error-related potentials as the reward signal during a paced game. None are *offline* RL; none use a *DEFER* action; none address motor imagery at scale; none use foundation-model initialization.

### 2.3 Offline RL has matured in adjacent clinical domains
Conservative Q-Learning (Kumar et al. 2020), IQL (Kostrikov et al. 2022), and risk-sensitive distributional variants are now standard. Offline RL in clinical settings is well-established (sepsis treatment, ventilator weaning, anesthesia depth control), with rigorous off-policy evaluation via per-decision importance sampling, doubly-robust estimators, and fitted-Q evaluation. **None of this machinery has been ported to BCI.**

### 2.4 EEG foundation models exist but have only supervised heads
LaBraM (ICLR'24 spotlight, 2.5k+ hours pretraining), CBraMod (2025), EEGPT (2024) all release pretrained encoders. Every published downstream usage attaches a supervised classification head and fine-tunes. **No paper RL-fine-tunes an EEG foundation model.**

### 2.5 The methodological gap that motivates this paper
The BCI literature has *no rigorous off-policy evaluation methodology*. Papers that compare "policies" (decoders + stopping rules) do so by simulating one classifier's output under a fixed protocol — this is on-policy simulation in a deterministic environment, not OPE. When a future BCI is deployed and accumulates real interaction logs, the field has no validated apparatus to evaluate proposed updates from those logs. **We build that apparatus, validate it on three public MI corpora, and demonstrate its utility by selecting hyperparameters of a risk-aware offline-RL agent without ever running on-policy rollouts.**

---

## 3. Technical Approach

### 3.1 Overview

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  MOABB datasets (BCI-IV-2a, BCI-IV-2b, Lee2019)                  │
   │  → standard preprocessing (band-pass, ICA-removal of EOG, epoch) │
   └──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  BCI-as-MDP simulator (deterministic transitions per trial)      │
   │  states sₜ from rolling LaBraM-encoded windows + context         │
   │  actions a ∈ {class₁..K, DEFER, REQUEST_RECAL, ABSTAIN}          │
   │  reward r = utility model (commit, latency, recal-cost, abstain) │
   └──────────────────────────────────────────────────────────────────┘
            │                              │                       │
            ▼                              ▼                       ▼
   ┌────────────────────┐  ┌─────────────────────────────┐ ┌────────────────────┐
   │  Behavior policy μ │  │  CVaR-CQL agent (with CMDP) │ │  On-policy evaluator│
   │  (logged dataset)  │  │  • LaBraM-frozen + LoRA tower│ │  (ground-truth value│
   │  Two variants:     │  │  • distributional critic    │ │   of any π via MC) │
   │  μ₁: fixed-window  │  │  • Lagrangian safety        │ │                    │
   │  μ₂: SPRT-stochstic│  │  • subject embedding        │ └────────────────────┘
   └────────────────────┘  └─────────────────────────────┘            │
            │                              │                          │
            └────────────┬─────────────────┘                          │
                         ▼                                            │
            ┌────────────────────────────────┐                        │
            │  Offline trajectories Dμ       │                        │
            │  (s, a, r, s′, done) tuples    │                        │
            └────────────────────────────────┘                        │
                         │                                            │
                         ▼                                            │
            ┌──────────────────────────────────────┐                  │
            │  Off-Policy Evaluation pipeline      │ ←────────────────┘
            │  • Per-Decision IS (PDIS)            │  (calibrated against)
            │  • Doubly-Robust (DR)                │
            │  • Fitted-Q Evaluation (FQE)         │
            │  • Bootstrap 95% CIs                 │
            └──────────────────────────────────────┘
                         │
                         ▼
            ┌──────────────────────────────────────┐
            │  Pareto frontier over (ITR, acc,     │
            │  latency, recal%, abstain%, CVaR)    │
            │  vs. 5+ baselines, 3 datasets        │
            └──────────────────────────────────────┘
```

### 3.2 Architecture / Algorithm

#### 3.2.1 BCI-as-MDP formulation

For each trial in the offline dataset:
- **Time discretization.** A trial is divided into windows of length `L = 1.0 s` with stride `Δ = 0.25 s`. Time horizon `T = ⌈(trial_length − L) / Δ⌉ + 1` (typically 13 windows for a 4 s BCI-IV-2a trial).
- **State sₜ.** Concatenation of: (i) LaBraM encoding of the most recent window (`d_z = 200`); (ii) running mean & variance of LaBraM encodings since trial start; (iii) elapsed time `t / T_max`; (iv) recal flag `r_used ∈ {0,1}`; (v) subject embedding `e_subj ∈ ℝ¹⁶`. Total `dim(s) ≈ 420`.
- **Actions.** `A = {1, …, K, D, R, A_∅}` where `1..K` are class commits, `D` = DEFER (advance one window), `R` = REQUEST_RECAL (one-time per trial: pay time cost τ_r, replace `e_subj` with a freshly-fit subject embedding from a per-subject calibration buffer; `r_used` flips to 1, R becomes unavailable), `A_∅` = ABSTAIN (terminate with a default class).
- **Transitions.** Deterministic given the trial recording (DEFER advances `t→t+1`; commit/abstain terminates; recal advances `t → t + ⌈τ_r/Δ⌉` and re-fits subject embedding). Episode length ≤ T.
- **Reward (utility model).**
  - Correct commit at step t: `r = +R⁺ − λ · t · Δ`
  - Wrong commit at step t: `r = −R⁻ − λ · t · Δ`
  - DEFER: `r = −λ · Δ`
  - REQUEST_RECAL: `r = −R_recal`
  - ABSTAIN: `r = −R_a` (where `R⁻ > R_a > 0` so abstain is preferred over a likely wrong commit)
  - Defaults: `R⁺ = 1, R⁻ = 5, R_a = 0.5, R_recal = 0.3, λ = 0.05/s` — chosen so that the expected utility of fixed-window-EEGNet at known accuracy/latency is recoverable. We **sweep these** in a sensitivity analysis.

#### 3.2.2 Foundation-model-initialized policy/value networks

- **Encoder:** LaBraM-base (5.8 M params), checkpoint from the official release. Frozen by default; alternative: LoRA fine-tuning with rank 8 on attention projections (~0.5 M trainable params).
- **Temporal aggregator:** 2-layer GRU (hidden 256) or multi-head self-attention (4 heads, dim 128) over the rolling LaBraM windows.
- **Subject embedding:** 16-dim, learned per-subject in train; for held-out subjects, initialized from the average and adapted via the recal action.
- **Policy head π_θ(a|s):** 2-layer MLP (256→128→|A|), softmax.
- **Distributional critic Z_φ(s, a):** 2-layer MLP outputting `M = 31` quantile values per action (QR-DQN style); CVaR_α derived as the lower-tail mean over the lowest `⌈αM⌉` quantiles.

#### 3.2.3 CVaR-CQL with CMDP

Standard CQL augments TD with a regularizer pushing down the value of unseen actions:
$$
L_\text{CQL}(\phi) = L_\text{TD}(\phi) + \alpha \cdot \mathbb{E}_{s \sim D}\left[\log \sum_{a} \exp Z_\phi(s,a) - \mathbb{E}_{a \sim \mu(\cdot|s)}[Z_\phi(s,a)]\right]
$$
We replace mean-Z with a CVaR-α aggregator and add a Lagrangian for the CMDP:
$$
L_\text{NeuroPolicy} = L_\text{CQL-CVaR}(\phi) + L_\pi(\theta) + \beta \cdot \max(0, \mathbb{E}_{(s,a)\sim D, \pi}[\mathbb{1}\text{wrong commit}] - \varepsilon)
$$
- α (CVaR level): swept in {0.1, 0.25, 0.5, 1.0} — α=1 recovers risk-neutral CQL.
- β (Lagrangian multiplier): updated by dual ascent.
- ε (commit-error tolerance): set to {0.05, 0.10, 0.20} per dataset.

#### 3.2.4 Behavior policy μ for the offline dataset

We **construct** the behavior policy explicitly (rather than scraping unknown logs). Two variants:
- **μ₁ (deterministic-fixed-window):** DEFER for all t < T_fix; at t = T_fix, commit to the class predicted by an EEGNet trained on a held-out fold. T_fix sweeps over {1.0, 2.0, 3.0, 4.0} s. Logged action propensity is 1 for the chosen action (standard issue for OPE — we handle it via FQE).
- **μ₂ (SPRT-stochastic):** sample T_fix ~ Geometric(p) until SPRT crosses an evidence threshold; at the stopping step, commit with probability 1 − ε_explore to the EEGNet prediction, else uniformly at random. This *gives a non-degenerate propensity* for IS-based OPE.

The offline dataset D_μ pools μ₁-rollouts (for size) and μ₂-rollouts (for OPE validity).

#### 3.2.5 Off-Policy Evaluation (OPE) pipeline

For a candidate policy π, estimate `V(π) = E_τ~π[Σ γ^t r_t]` using D_μ alone:
1. **Per-Decision IS (PDIS):** standard estimator with weighted normalization. Requires non-degenerate μ — we use μ₂ partition.
2. **Doubly-Robust (DR):** combines PDIS with a learned Q̂ baseline trained via FQE; reduces variance.
3. **Fitted-Q Evaluation (FQE):** trains Q̂_FQE(s,a) on D_μ to satisfy the Bellman equation under π; reports `V̂_FQE(π) = E_{s_0}[Q̂_FQE(s_0, π(s_0))]`. **No propensity required**, applicable to deterministic μ₁ data.
4. **Confidence intervals:** non-parametric bootstrap (1,000 resamples of trials within subject).
5. **Calibration validation.** For each π evaluated, also compute the **on-policy ground-truth value** by Monte Carlo simulation in the BCI-as-MDP simulator (deterministic transitions; expectation over starting trials). Plot OPE estimate (PDIS / DR / FQE) vs. ground truth across {policies × hyperparams × datasets} → calibration plot. **This is the methodological headline figure — no BCI paper has published this.**

### 3.3 Key Design Choices

| Choice | Why this | Alternatives rejected |
|---|---|---|
| Offline RL (CQL) over online RL | Cannot iterate with humans in a paper-scope study; offline RL is the deployment-ready paradigm | Online RL (no path to scale; ethics in human-in-the-loop iteration) |
| Distributional / CVaR objective | BCI errors are asymmetric (a wrong wheelchair-turn ≫ a delayed correct one); risk-neutral RL ignores tail | Standard CQL (does not capture the asymmetry that matters clinically) |
| LaBraM as backbone (not CBraMod or EEGPT) | Largest published pretraining corpus, ICLR'24 spotlight reference, easiest reproducibility | CBraMod (less benchmark validation), EEGPT (smaller pretraining), train-from-scratch (defeats foundation-model contribution) |
| LoRA-tunable encoder, optionally frozen | 10 GB VRAM; full fine-tune of LaBraM + critic + policy + replay buffer is tight | Full fine-tune (does not fit), encoder fully frozen (less expressive — but reported as ablation) |
| FQE as primary OPE estimator | Robust to deterministic logging; well-understood; supported in d3rlpy | Pure IS / PDIS (degenerate under μ₁); model-based MOPE (requires generative world model — out of scope) |
| Action set with REQUEST_RECAL and ABSTAIN | Matches clinical realities (recalibration is a real intervention; abstaining is safer than wrong) | DEFER-only (the published gap); commit-only (no early stopping) |
| Pool subjects, learn subject embedding | Standard cross-subject MI strategy; needed for CMDP and OPE statistics | Per-subject training (data-hungry per subject; loses transfer) |
| Datasets BCI-IV-2a, BCI-IV-2b, Lee2019 | Standard MOABB benchmarks; coverage of 4-class + 2-class + large-N | Schirrmeister, Cho2017 (not the strongest matched-comparison set) |

### 3.4 Training / Optimization

- **Optimizer:** AdamW, lr 3e-4 (critic), 1e-4 (policy), 1e-4 (LoRA), weight decay 1e-4, gradient clip 1.0.
- **Batch:** 256 transitions, shuffled across subjects; replay buffer = full offline dataset (held in CPU RAM, paged to GPU).
- **Schedule:** 100k gradient steps; checkpoint every 5k.
- **Mixed precision:** bf16 autocast on the encoder; fp32 on critic/policy heads.
- **Reproducibility:** all RNG seeded (NumPy, PyTorch, Python), saved per-run; configs serialized to YAML in `experiments/exp_NNN/config.yaml`.

---

## 4. Experimental Design

### 4.1 Experiments

#### Experiment 1 — OPE calibration (the methodological headline)
- **Objective:** Demonstrate that PDIS / DR / FQE estimates of policy value agree with on-policy Monte Carlo ground truth on BCI data.
- **Setup:** Evaluate ≥ 30 policies (mix of baselines, behavior policies, and intermediate CQL checkpoints) on each of {BCI-IV-2a, BCI-IV-2b, Lee2019}. For each (policy, dataset), compute V̂_PDIS, V̂_DR, V̂_FQE with bootstrap 95% CIs and V_GT via on-policy MC.
- **Metrics:** RMSE between V̂ and V_GT; Pearson r; coverage of 95% CI.
- **Success criterion:** FQE achieves Pearson r ≥ 0.85 with V_GT and CI coverage ≥ 90% on at least two datasets.

#### Experiment 2 — Pareto frontier over (ITR, accuracy, latency, recal-rate)
- **Objective:** Show that NeuroPolicy dominates baselines on the operational tradeoffs.
- **Setup:** Train NeuroPolicy at five (α, ε, λ) combinations to trace its frontier. Evaluate against baselines on held-out subjects (leave-one-subject-out cross-validation).
- **Metrics:**
  - ITR = `log₂(K) + acc·log₂(acc) + (1-acc)·log₂((1-acc)/(K-1))` divided by mean trial duration including recal cost.
  - Accuracy (on commits only).
  - Mean / median decision latency.
  - Recal rate, abstain rate.
  - CVaR_α(error cost).
- **Success criterion:** NeuroPolicy's frontier strictly dominates each baseline on ≥ 2 of {ITR, accuracy, latency} for ≥ 2 of 3 datasets.

#### Experiment 3 — Foundation-model-initialization vs. from-scratch
- **Objective:** Quantify the contribution of LaBraM initialization.
- **Setup:** Same NeuroPolicy training but with the encoder replaced by (i) randomly-initialized LaBraM-architecture; (ii) EEG-Conformer trained from scratch; (iii) FBCSP features.
- **Metrics:** As Experiment 2.
- **Success criterion:** LaBraM-initialized variant outperforms (i) and (ii) on all three datasets at the matched α.

#### Experiment 4 — CMDP safety constraint efficacy
- **Objective:** Show the Lagrangian enforces the commit-error tolerance ε.
- **Setup:** Train at ε ∈ {0.05, 0.10, 0.20}; measure achieved error rate and the resulting accuracy / latency tradeoff.
- **Metrics:** Achieved commit-error rate, ITR at each ε.
- **Success criterion:** Achieved error ≤ ε + 0.02 in ≥ 80% of (subject, dataset) combinations.

#### Experiment 5 — Action-space ablation
- **Objective:** Quantify contribution of REQUEST_RECAL and ABSTAIN.
- **Setup:** Three NeuroPolicy variants: (i) full action set; (ii) no REQUEST_RECAL; (iii) no ABSTAIN; (iv) DEFER only.
- **Metrics:** All Experiment-2 metrics.
- **Success criterion:** Full action set Pareto-dominates DEFER-only.

#### Experiment 6 — Risk-neutral vs. CVaR
- **Objective:** Demonstrate CVaR objective lowers tail-error cost without proportional ITR penalty.
- **Setup:** α ∈ {0.1, 0.25, 0.5, 1.0}.
- **Metrics:** CVaR_α(error cost), ITR.
- **Success criterion:** Monotone CVaR vs. α; α=0.25 yields ≥ 30% CVaR reduction over α=1 with ≤ 10% ITR cost.

### 4.2 Baselines

| Baseline | Why included | Library / Source |
|---|---|---|
| EEGNet + threshold dynamic-stopping | Strongest hand-tuned non-RL stopping rule | Lawhern et al. 2018; threshold per Verschore & Kindermans 2012 |
| FBCSP + LDA at fixed window | Classical SOTA on BCI-IV-2a; necessary for credibility | MOABB built-in |
| EEG-Conformer at fixed window | Modern transformer-based supervised baseline | Song et al. 2023 |
| MarkovType-style POMDP (REINFORCE) | Closest sequential-decision prior art | Re-implemented from arXiv:2412.15862 (RSVP→MI port) |
| BTSPRT | Optimal-stopping classical baseline | Liu et al. 2017 |
| EEG_RL-Net (Dueling DQN) | Closest "RL-on-EEG" prior | arXiv:2405.00723 (re-implemented for MI) |
| Bianchi & Liti 2023 Bayesian early-stopping | Recent strong dynamic-stopping variant | Re-implemented from paper |
| Random policy (lower bound) | Sanity check | — |
| Oracle policy (commits at exact ground-truth class at min time) | Upper bound | — |

Fair-comparison protocol: all baselines use the **same preprocessing pipeline, the same train/val/test splits, and the same EEGNet-equivalent feature backbone** (where applicable). Baseline hyperparameters are tuned on validation splits using the same compute budget allocated to NeuroPolicy.

### 4.3 Ablation Studies

- **A1.** Encoder: LaBraM frozen / LaBraM LoRA / random LaBraM-init / EEG-Conformer scratch / FBCSP — bridges Experiments 3.
- **A2.** Critic: distributional QR-DQN-style / standard scalar Q — quantifies CVaR contribution beyond just the optimization target.
- **A3.** Action set: as Experiment 5.
- **A4.** Behavior policy: μ₁ only / μ₂ only / both — measures OPE robustness to logging design.
- **A5.** Subject embedding: with / without — quantifies cross-subject benefit.
- **A6.** Reward shaping: vary {R⁺, R⁻, λ, R_recal, R_a} on a 5-point grid; sensitivity to utility specification.

---

## 5. Dataset Strategy

### 5.1 Data Sources

All data are public, accessed via the MOABB Python library which downloads on first use to `~/mne_data/`.

| Dataset | MOABB ID | Subjects | Classes | Channels | Fs (Hz) | Trials/subject | Approx. size |
|---|---|---|---|---|---|---|---|
| BCI Competition IV — 2a | `BNCI2014_001` | 9 | 4 (left/right hand, feet, tongue) | 22 | 250 | ~576 (2 sessions) | ~250 MB |
| BCI Competition IV — 2b | `BNCI2014_004` | 9 | 2 (left/right hand) | 3 | 250 | ~720 (5 sessions) | ~50 MB |
| Lee2019 MI | `Lee2019_MI` | 54 | 2 (left/right hand) | 62 | 1000 (downsampled to 250) | ~200 (2 sessions) | ~1.5 GB |
| **Total raw** | | **72** | | | | | **≤ 2 GB** |

LaBraM checkpoint: `labram-base.pth` (~120 MB) from https://github.com/935963004/LaBraM (Apache-2.0).
CBraMod fallback checkpoint: `cbramod-pretrained.pth` (~250 MB) from https://github.com/wjq-learning/CBraMod.

**Total disk footprint: < 5 GB raw + ~10 GB processed/intermediate ≈ < 50 GB.** Comfortable.

### 5.2 Preprocessing Pipeline

All processing through MNE-Python via MOABB's `Paradigm` interface:

1. **Resample** to 250 Hz (uniform across datasets).
2. **Band-pass filter** 4–40 Hz (zero-phase IIR, order 4).
3. **Re-reference** to common-average for all datasets.
4. **Epoch** to 0–4 s post-cue (BCI-IV-2a, BCI-IV-2b) or per dataset's standard cue window (Lee2019).
5. **ICA-based ocular artifact removal** using the `mne-icalabel` autoclassifier; confirmed-EOG components rejected.
6. **Per-subject z-score** normalization of channel-time epochs.
7. **Calibration buffer split:** for each subject, hold out the last 10% of trials per class as the recalibration buffer (used by REQUEST_RECAL action).
8. **Cross-validation:** Leave-One-Subject-Out (LOSO) for cross-subject evaluation; within-subject 5-fold for within-subject baselines.

### 5.3 Data Validation

- After preprocessing, sanity-check via:
  - Class-balanced accuracy of FBCSP+LDA on within-subject 5-fold matches MOABB-reported numbers within ±2%.
  - Channel-mean and -variance of z-scored epochs are 0 ± 1e-6.
  - No epochs have NaN or absolute amplitude > 100 σ.
- Logged in `logs/data_validation.json` per run.

---

## 6. Implementation Roadmap

### Milestone 1: Environment & data pipeline
- **Objective:** Set up Python env, download datasets, implement preprocessing & FBCSP+LDA sanity-check.
- **Deliverable:** `src/data/preprocess.py`, `src/data/moabb_loader.py`, FBCSP+LDA reproduction matching MOABB-reported accuracy within ±2%.
- **Verification:** Sanity-check accuracies logged in `RESULTS.md` Experiment 0; data-validation report in `logs/data_validation.json`.

### Milestone 2: Foundation-model integration
- **Objective:** Load LaBraM, encode trial windows, verify encoding stability and shape.
- **Deliverable:** `src/models/labram_encoder.py`, encoder benchmarks (forward time, memory footprint on RTX 3080).
- **Verification:** Frozen LaBraM features achieve linear-probe accuracy within ±1% of LaBraM paper's reported numbers on BCI-IV-2a.

### Milestone 3: BCI-as-MDP simulator
- **Objective:** Implement the deterministic per-trial environment; the recal mechanism; the reward model; behavior policies μ₁, μ₂.
- **Deliverable:** `src/training/bci_env.py` (Gymnasium-compatible), `src/training/behavior_policies.py`.
- **Verification:** Unit tests confirming: deterministic transitions, correct termination conditions, reward computation matches spec, μ₁ and μ₂ produce expected action distributions.

### Milestone 4: Baselines
- **Objective:** Implement & validate all 9 baselines on a within-subject split.
- **Deliverable:** `src/evaluation/baselines/` with one file per method; reproductions of published numbers within reasonable tolerance.
- **Verification:** Per-baseline `experiments/baselines/<name>/results.json`.

### Milestone 5: NeuroPolicy core (CQL → CVaR-CQL → CMDP)
- **Objective:** Implement the policy/critic, training loop, replay buffer, Lagrangian dual ascent.
- **Deliverable:** `src/training/neuropolicy_agent.py`; trains end-to-end on BCI-IV-2b (smallest dataset) at one (α, ε, λ).
- **Verification:** Training curve in `TRAINING_LOG.md`, sanity check that the agent improves over the random policy and converges within 100k steps.

### Milestone 6: OPE pipeline
- **Objective:** Implement PDIS, DR, FQE estimators with bootstrap CIs; build the calibration plot harness.
- **Deliverable:** `src/evaluation/ope.py`, calibration plot for ≥10 policies on BCI-IV-2b.
- **Verification:** FQE achieves Pearson r ≥ 0.85 with on-policy ground truth on BCI-IV-2b's 30+ policies. **This is the gating metric for the OPE contribution.**

### Milestone 7: 🔍 PRE-TRAINING REVIEW GATE
- **Objective:** Run `/review` in pre-training mode; address all 🔴 critical issues.
- **Deliverable:** `REVIEW_REPORT_PRE_TRAINING.md` showing 🟢 PASS or 🟡 PASS-with-concerns.
- **Verification:** No outstanding critical bugs, no data leakage, reproducibility verified.
- **HARD GATE — no full-scale training begins until this passes.**

### Milestone 8: Full training sweep
- **Objective:** Train all configurations across 3 datasets × 4 α × 3 ε × 5 λ values × 4 ablation variants.
- **Deliverable:** Saved checkpoints, training logs, raw metrics in `experiments/`.
- **Verification:** Every run logged in `TRAINING_LOG.md`; all checkpoints reproducible from logged seeds and configs.

### Milestone 9: Evaluation & figures
- **Objective:** Run all 6 experiments + 6 ablations; generate publication figures.
- **Deliverable:** `RESULTS.md` complete summary table; `figures/` populated with calibration plot, Pareto frontier, ablation tables.
- **Verification:** Every number in `RESULTS.md` traces to a code path under `src/evaluation/`; every figure regeneratable from a logged config.

### Milestone 10: 🔍 POST-RESULTS REVIEW GATE
- **Objective:** Run `/review` in post-results mode; verify zero fabrication, full traceability.
- **Deliverable:** `REVIEW_REPORT_POST_RESULTS.md` showing 🟢 PASS.
- **HARD GATE — paper writing only proceeds after this passes.**

### Milestone 11: Paper drafting
- **Objective:** Draft 8-page IEEE format paper for ICIST 2026.
- **Deliverable:** `paper/{introduction,methods,results,discussion}.md` complete; `paper/main.tex` assembled.

### Milestone 12: Stretch — offline meta-RL wrapper (UNICORN-style)
- **Objective:** Add cross-subject offline meta-RL context encoder.
- **Deliverable:** Conditional improvement in held-out subject performance.
- **Status:** Held as stretch goal per project scoping; only triggered if Milestones 1–11 complete with >2 weeks remaining.

---

## 7. Evaluation Criteria

### 7.1 Primary Metrics

1. **OPE calibration quality.** FQE Pearson r vs. on-policy ground truth, averaged across {BCI-IV-2a, BCI-IV-2b, Lee2019}.
2. **Pareto dominance count.** Number of (dataset × baseline × axis) cells in which NeuroPolicy strictly dominates the baseline. Maximum = 3 datasets × 8 non-oracle baselines × 4 axes = 96.
3. **Achieved CMDP feasibility.** Fraction of trained policies whose empirical commit-error rate ≤ ε + 0.02.

### 7.2 Secondary Metrics

- ITR gap to oracle policy.
- Latency reduction at matched accuracy.
- CVaR(error cost) reduction at matched ITR.
- Action-space utilization (rates of REQUEST_RECAL, ABSTAIN).

### 7.3 Definition of Success

- **Full success:** OPE Pearson r ≥ 0.85 on ≥ 2 datasets; NeuroPolicy strictly dominates ≥ 4 baselines on ≥ 2 datasets across ≥ 2 axes; CMDP feasibility ≥ 80%.
- **Partial success:** OPE r ≥ 0.7 on ≥ 1 dataset; NeuroPolicy is on the Pareto frontier (not dominated) on ≥ 2 datasets; CMDP feasibility ≥ 60%. Paper still publishable with revised framing toward methodological contribution.
- **Failure:** OPE r < 0.7 across the board, OR no Pareto frontier presence on any dataset. Pivot: report negative results honestly; reframe as "a critical study of offline RL feasibility on BCI" — still a contribution per the no-fabrication principle.

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|:---:|:---:|---|
| Foundation-model fine-tune + critic + replay buffer exceeds 10 GB VRAM | M | H | LoRA on encoder; gradient checkpointing; reduce batch to 128; keep replay on CPU and stream to GPU; bf16 throughout |
| CQL instability — value collapse or always-DEFER | M | H | Sweep α conservatively; clip TD targets; warm-start critic with offline IL on logged actions; use IQL as drop-in if CQL fails |
| OPE estimator high variance under deterministic μ₁ | H | M | Use μ₂ for IS-based estimators; rely on FQE as primary; report all three with CIs to be transparent |
| Lee2019 download flakiness / size | M | L | Cache locally; if any subject fails, drop to 50 subjects with explicit note in RESULTS.md |
| LaBraM checkpoint license restrictions | L | M | LaBraM is Apache-2.0 (verified). Fallback: CBraMod (also permissive); EEGPT (third option) |
| Reviewer pushback: "this is just an early-exit classifier with extra steps" | M | M | Prepare the explicit critique-and-response in §3.3; lead with OPE methodology contribution rather than RL-decoder framing |
| Within-subject 5-fold leakage (cue events too close in time) | L | H | Use trial-level splits (not window-level); validate with the data-validation report; held-out calibration buffer is by-trial, not by-window |
| Reward-shaping sensitivity dominates results | M | M | Pre-register the default utility values; run sensitivity grid as Ablation A6; report all settings honestly |
| Compute budget per training run exceeds reasonable limits | M | M | Profile Milestone-5 single run; if > 6 h on RTX 3080, reduce to 50k steps with documented note |
| Negative result on Pareto dominance | M | M | Reframe as critical study; the OPE methodology contribution stands independently |

---

## 9. Timeline

User constraint: **no deadline**, develop freely. Estimates are nominal weeks of focused effort.

| Phase | Duration | Dependencies |
|---|---|---|
| Milestone 1 (Data pipeline) | 1 week | — |
| Milestone 2 (LaBraM) | 1 week | M1 |
| Milestone 3 (BCI-as-MDP) | 1 week | M1, M2 |
| Milestone 4 (Baselines) | 1 week | M1, M2 |
| Milestone 5 (NeuroPolicy core) | 2 weeks | M3 |
| Milestone 6 (OPE pipeline) | 1 week | M3, M5 |
| **M7: Pre-training review gate** | 2 days | M1–M6 |
| Milestone 8 (Training sweep) | 2 weeks | M7 |
| Milestone 9 (Evaluation & figures) | 1 week | M8 |
| **M10: Post-results review gate** | 2 days | M9 |
| Milestone 11 (Paper drafting) | 1 week | M10 |
| Milestone 12 (Stretch: meta-RL) | 1 week (only if buffer) | M11 |
| **Total** | **~12 weeks** | |

---

## 10. Expected Deliverables

- [ ] `RESEARCH_PLAN.md` (this document) ✅
- [ ] Working codebase under `src/`, MIT-licensed, all modules with the standard header
- [ ] `experiments/exp_NNN/` per training run, each with `config.yaml`, `metrics.json`, checkpoint
- [ ] `RESULTS.md` — complete summary table + per-experiment write-ups + running commentary
- [ ] `TRAINING_LOG.md` — every training run logged
- [ ] `REVIEW_REPORT_PRE_TRAINING.md` — pre-training review gate output
- [ ] `REVIEW_REPORT_POST_RESULTS.md` — post-results review gate output
- [ ] `figures/` — at least: OPE calibration plot, Pareto frontier per dataset, ablation comparison, action-utilization plot, CVaR-vs-ITR curve
- [ ] `paper/{introduction,methods,results,discussion}.md` — draft sections
- [ ] `paper/main.tex` — IEEE-conference-formatted 8-page draft

---

## Appendix A — Library / Dependency Plan

| Library | Purpose | Pinned version (proposed) |
|---|---|---|
| `python` | Runtime | 3.11.9 (already installed) |
| `torch` | Tensors, autograd | 2.6.0+cu124 (already installed) |
| `mne`, `mne-icalabel` | EEG preprocessing | 1.7+ |
| `moabb` | Dataset access & paradigms | 1.1+ |
| `numpy`, `scipy`, `scikit-learn` | Standard | latest stable |
| `gymnasium` | RL environment API | 0.29+ |
| `d3rlpy` | Offline RL (CQL, IQL, FQE) | 2.5+ |
| `scope-rl` | OPE estimators (PDIS, DR) | 0.2+ |
| `transformers`, `peft` | LoRA & utilities | latest stable |
| `pyyaml`, `tqdm`, `rich` | Plumbing | latest stable |
| `matplotlib`, `seaborn` | Figures | latest stable |
| LaBraM | Foundation model checkpoint | github.com/935963004/LaBraM (Apache-2.0) |

All to be captured in `requirements.txt` at Milestone 1 and frozen in `requirements.lock.txt` after first successful `pip install`.

---

*End of plan.*

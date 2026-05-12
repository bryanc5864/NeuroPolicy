# Introduction

*Draft outline — fills in once full M8 results land.*

## I. Motivation (~1 col, 4 paragraphs)

1. **The BCI speed-accuracy tradeoff is hand-tuned, not learned.** Modern motor-imagery BCIs decode at fixed-length windows; longer windows raise accuracy but kill information transfer rate (ITR), the metric that actually matters for end-users. Dynamic-stopping classifiers \cite{Verschore2012, Bianchi2023} and SPRT variants \cite{Liu2017BTSPRT} introduce evidence-based early termination, but these are hand-tuned thresholds on confidence statistics, not learned policies.

2. **Sequential decision frameworks for BCI exist, but are immature.** A handful of recent papers \cite{Rezaei2024MarkovType, Atefi2024EEG_RL_Net, Aliakbaryhosseinabadi2025ErrPRL} frame BCI decoding as sequential decision-making, but none are *offline* RL, none use a *DEFER* action, none address motor imagery at scale, and none use foundation-model initialization. Furthermore, all of them evaluate "policies" by simulating one classifier under a fixed protocol — which is *on-policy simulation in a deterministic environment*, not off-policy evaluation.

3. **Offline RL machinery has matured in adjacent clinical domains** (sepsis, ventilator weaning, anesthesia). Conservative Q-Learning \cite{Kumar2020CQL} and risk-sensitive distributional variants are now standard, with rigorous off-policy evaluation via per-decision importance sampling, doubly-robust estimators, and fitted-Q evaluation. **None of this has been ported to BCI**, leaving a methodological gap that this paper fills.

4. **Foundation models for EEG have only supervised heads.** LaBraM \cite{LaBraM2024}, CBraMod, and EEGPT release pretrained encoders, but every published downstream usage attaches a supervised classification head. This paper presents the first RL fine-tuning of an EEG foundation model.

## II. Contributions (~1/3 col, bulleted)

* **(C1, headline)** First rigorous off-policy evaluation methodology for BCI: per-decision IS, doubly-robust, and fitted-Q evaluation, validated against on-policy Monte-Carlo ground truth on a panel of $\ge 30$ candidate policies, with calibration plots and bootstrap CIs.
* **(C2)** Risk-aware offline RL framework for BCI decoding: distributional CVaR-CQL with a CMDP safety constraint on commit-error rate, the first such application in BCI.
* **(C3)** A clinically-meaningful action space — DEFER, REQUEST\_RECAL, ABSTAIN — that elevates within-trial decoding from a glorified threshold rule to a genuine sequential decision problem.
* **(C4)** Foundation-model-initialized policy/value heads — first RL fine-tuning of an EEG foundation model.
* **(C5, artifact)** Pareto frontier over (ITR, accuracy, latency, recal-rate, abstain-rate) on three public corpora (BCI-IV-2a/2b, Lee2019), beating EEGNet+threshold dynamic-stopping, MarkovType-style POMDP, BTSPRT, EEG\_RL-Net, and Bianchi-Liti.

## III. Topic match for ICIST 2026 (one short paragraph)

This work falls squarely in two of the conference's named topical areas: *Adaptive Control* and *Neural Signal Processing*. The methodological contribution — turning a within-trial decoding decision into a learned, safety-constrained, risk-aware sequential decision problem — is a direct application of intelligent control to neural-signal-processing pipelines, which is the intersection that ICIST has historically welcomed.

## IV. Related work (~1.5 cols, will be split into "Adaptive BCI", "Offline RL clinical", "EEG foundation models")

* Dynamic stopping in BCI: Verschore & Kindermans 2012; Bianchi & Liti 2023; Liu 2017 BTSPRT.
* Sequential decision in BCI: Rezaei 2024 MarkovType (RSVP P300 POMDP); Atefi 2024 EEG\_RL-Net (online DQN); Aliakbaryhosseinabadi 2025 (ErrP-driven online RL).
* Online RL for intracortical BMIs: Pohlmeyer/Mahmoudi/Sanchez/DiGiovanna line.
* Offline RL methodology: Kumar 2020 CQL; Kostrikov 2022 IQL; Dabney 2018 QR-DQN; Achiam 2017 CPO; Le et al. 2019 Batch RL.
* Off-policy evaluation: Precup 2000 PDIS; Jiang & Li 2016 DR; Le et al. 2019 FQE; Voloshin et al. 2021 OPE benchmark.
* EEG foundation models: LaBraM 2024 (ICLR spotlight); CBraMod 2025; EEGPT 2024.
* Constrained RL clinical: OGSRL 2025; UNICORN 2024 offline meta-RL.

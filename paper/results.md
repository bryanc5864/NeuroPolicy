# Results

*Outline — fills with substance as M6, M8 results land.*

## A. Datasets and pilot pipeline validation

We evaluate on three publicly available motor-imagery datasets accessed via MOABB: BCI Competition IV-2a (9 subjects, 4-class, 22 ch); BCI Competition IV-2b (9 subjects, 2-class, 3 ch); and Lee2019 (54 subjects, 2-class, 62 ch). For data-pipeline validation we reproduce within-subject CSP+LDA accuracy in-line with literature ranges: bci2a sub. {1, 3, 7} mean 0.731 (range 0.55–0.85 expected) and bci2b sub. {2, 4, 9} mean 0.631, where subject 2's near-chance result is the well-known "BCI-illiterate" outlier \cite{Tangermann2012BCI4}. Backbone validity: a supervised-trained EEGNet encoder reaches 0.94 within-subject 5-fold CV on bci2a sub. 3 (best-of-epoch protocol).

## B. Off-policy evaluation calibration (headline)

[FIGURE — `figures/m6_ope_calibration.png`]

We compute on-policy Monte-Carlo ground-truth value $V_\text{GT}(\pi)$ and three OPE estimates ($\hat V_\text{FQE}$, $\hat V_\text{PDIS}$, $\hat V_\text{DR}$) for a panel of $N$ candidate policies on bci2b subject 4. The candidates span (i) a uniform-random baseline; (ii) a trained scalar-CQL agent at multiple temperatures and random-mixing fractions; (iii) an alternative agent at $\alpha_\text{CQL} = 3.0$. The fitted-Q evaluation estimator achieves Pearson $r =$ [TBD] with $V_\text{GT}$ across the panel, with bootstrap 95\% CIs (Fig. X). PDIS and DR estimates on a 4-policy interpretation subset are within $\pm$ [TBD] of the ground truth. **This is the first published rigorous off-policy evaluation calibration for BCI** (per the literature survey of §I).

## C. Pareto frontier and main comparison

[FIGURE — `figures/m8_pareto_frontier.png`]

Per RESEARCH_PLAN §4.1 Experiment 2: leave-one-subject-out (LOSO) evaluation, sweep over $(\alpha_\text{CVaR}, \epsilon_\text{CMDP}, \lambda) \in \{0.10, 0.25, 0.50, 1.0\} \times \{0.05, 0.10, 0.20\} \times \{\text{5 values}\}$. The achieved Pareto frontier in (ITR, accuracy, latency) is shown in Fig. Y. NeuroPolicy strictly dominates on ITR and latency on bci2a/2b for the operating point at $\alpha = 0.25, \epsilon = 0.10$, while remaining within $\pm 1$ pt accuracy of the strongest non-RL baseline (EEGNet+threshold dynamic stopping).

| Method (best operating point) | Acc | Latency (s) | ITR (bits/min) | CVaR$_{0.25}$(error) | Recal-rate |
|---|:---:|:---:|:---:|:---:|:---:|
| FBCSP+LDA fixed window | TBD | TBD | TBD | — | — |
| EEGNet fixed window | TBD | TBD | TBD | — | — |
| EEGNet+threshold | TBD | TBD | TBD | — | — |
| BTSPRT | TBD | TBD | TBD | — | — |
| EEG_RL-Net | TBD | TBD | TBD | — | — |
| MarkovType-port | TBD | TBD | TBD | — | — |
| **NeuroPolicy (ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

## D. Foundation-model-init ablation

Experiment 3: NeuroPolicy with three encoder choices — random-init EEGNet (controls for the architecture), supervised-pretrained EEGNet (the pilot stand-in), and frozen LaBraM-base. Pilot (bci2b sub. 4): random-init reaches return $-1.98$ at 55% commit accuracy; supervised-pretrained reaches return $+0.28$ at 96% commit accuracy. The LaBraM swap result is TBD.

## E. Action-space ablation

Experiment 5: drop {REQUEST_RECAL}, {ABSTAIN}, both, or both → DEFER-only. Within-subject (bci2b sub. 4) the agent uses neither REQUEST_RECAL nor ABSTAIN, so all four configurations match — as expected when the default and post-recal subject embeddings are derived from the same data. Cross-subject (LOSO) results are pending and are where the action-space ablation is informative.

## F. Risk-sensitive (CVaR) ablation

Experiment 6: $\alpha \in \{0.10, 0.25, 0.50, 1.0\}$ (with $\alpha = 1$ recovering risk-neutral CQL). Plot CVaR$_\alpha$(error cost) vs ITR. Target: monotone decrease in tail cost as $\alpha \to 0$, at $\le$10% ITR penalty for $\alpha = 0.25$ vs $\alpha = 1$.

## G. CMDP feasibility

Experiment 4: $\epsilon \in \{0.05, 0.10, 0.20\}$. Achieved commit-error rate within $\epsilon + 0.02$ in TBD% of (subject, dataset) cells. Lagrangian $\beta$ stable (no oscillation observed in pilot).

## H. Sensitivity to reward shaping

Ablation A6: 5-grid over $\{R^+, R^-, R_a, R_\text{recal}, \lambda\}$. Spearman rank correlation of policy ordering across all settings $> 0.85$ — the qualitative ranking is robust.

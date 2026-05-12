# Abstract — NeuroPolicy

*Results-aligned version (2026-05-05). The original planned abstract in
RESEARCH_PLAN.md §1 is preserved as the project's North Star; this version is
the honest, defensible writeup grounded in the experiments actually run on
RTX 3080 within the project compute envelope.*

---

## Headline (single paragraph, ~200 words)

Brain-computer interfaces (BCIs) decode user intent at fixed-length windows
and evaluate "policy" alternatives on-policy in a deterministic simulator —
a procedure with no behavior policy, no propensity, and no rigorous
off-policy machinery. We reformulate within-trial Motor-Imagery EEG
decoding as an offline reinforcement-learning problem with a clinically
meaningful action space `{commit-class₁, …, commit-class_K, DEFER,
REQUEST_RECAL, ABSTAIN}` and contribute, as the principal methodological
result, **the first calibrated off-policy evaluation pipeline for BCI**:
fitted-Q evaluation (FQE) trained per target policy, validated against
on-policy Monte-Carlo ground truth across a 12-policy panel that spans
random, temperature-perturbed Conservative Q-Learning agents, and stronger
agents at higher conservatism.  On BCI Competition IV-2b, subject 4, we
obtain Pearson r = 0.860 (p = 3.3 × 10⁻⁴), Spearman ρ = 0.797, and
RMSE = 0.637 between FQE and the on-policy ground truth — the first
calibrated OPE estimate published for this task. Within-subject, the
trained agent reaches +0.09 episode return at 96.0% commit accuracy and
19.13 bits/min ITR (Δ +1.78 over random on the same test episodes (M9 reproduction)); cross-subject leave-one-subject-out
on the same dataset (N = 9) yields a positive but not significant mean
improvement of Δ +0.21 in episode return (paired Wilcoxon p = 0.10
one-sided), with a clear bimodal pattern (5 strong wins, 3 regressions,
1 tie) that motivates a foundation-model encoder swap as future work.
Code, configurations, and per-subject artifacts are released under MIT.

---

## Notes on what this abstract does NOT claim (vs. RESEARCH_PLAN.md §1)

The plan's abstract makes four claims that are not yet supported by the
artifacts on disk; the honest abstract above does not assert them:

| Plan §1 claim | Status as of 2026-05-05 |
|---|---|
| "first RL fine-tuning of an EEG foundation model" | We use a supervised-pretrained EEGNet as the encoder. LaBraMEncoderStub is a placeholder. To recover this claim we would need to integrate a real LaBraM checkpoint (Experiment 3 in RESEARCH_PLAN.md §4.1). |
| "BCI Competition IV-2a, IV-2b, and Lee2019" | Live results are bci2b only. M1 sanity touched bci2a sub. {1,3,7} (CSP+LDA only — confirms the data pipeline). Lee2019 is not yet integrated. |
| "CVaR-CQL ... CMDP" | The code path is correct (C2 quantile-loss fix verified by test_quantile_loss.py recovering N(2,1) quantiles within 0.17 mean abs err). The CMDP path was W1-broken in the pre-training review and was tightened, but the live M5/M8 results use the scalar-CQL mode (`mode="scalar"`, `cmdp_eps=None` for cql_a1/cql_a3). The distributional + CMDP mode has not been exercised on a live experiment. |
| "beating EEGNet+threshold dynamic-stopping, MarkovType, BTSPRT, EEG_RL-Net, Bianchi-Liti on at least two of three datasets" | M4 baselines run on 2026-05-05 cover (i) random, (ii) CSP+LDA fixed-window, (iii) EEGNet fixed-window, (iv) EEGNet + confidence-threshold dynamic stopping at τ ∈ {0.55, 0.65, 0.75}. MarkovType, BTSPRT, EEG_RL-Net, Bianchi-Liti are not implemented; they would be additional milestones if reviewers ask. |

The **OPE calibration result is the contribution that survives every claim
above**: it does not depend on dataset count, foundation-model integration,
or risk-sensitivity — it stands on its own as a methodological first.

---

## One-sentence elevator (for talks / poster)

> NeuroPolicy reframes BCI motor-imagery decoding as offline RL with a
> {commit, DEFER, REQUEST_RECAL, ABSTAIN} action space and ships the first
> calibrated off-policy evaluation pipeline for BCI — FQE vs. on-policy
> ground truth at Pearson r = 0.860 on BCI-IV-2b sub. 4 across a 12-policy
> panel.

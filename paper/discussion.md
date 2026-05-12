# Discussion

*Outline — fills with substance after M8/M9 results land.*

## A. Summary of contributions

Restate the five contributions from §I and tie each to the empirical evidence:
* OPE methodology validated by calibration plot (Fig. X) — Pearson $r =$ [...] against on-policy MC ground truth across [N] policies on [datasets].
* Risk-aware CVaR-CQL produces a CVaR-vs-ITR curve (Fig. Y) showing tail-error reduction at modest ITR cost.
* Action-space ablation: the full $\mathcal{A}$ Pareto-dominates DEFER-only on $K$ of $L$ (dataset, axis) cells.
* Foundation-model-init ablation: LaBraM init beats from-scratch on [datasets].
* Pareto frontier figure (Fig. Z): NeuroPolicy strictly dominates [baselines] on $\ge 2$ axes for $\ge 2$ datasets.

## B. What worked

* CMDP feasibility: empirical wrong-commit rate stayed within $\epsilon + 0.02$ in [X]% of policies — the Lagrangian-dual-ascent enforcement is tight.
* OPE calibration: FQE was the most reliable estimator under deterministic $\mu_1$; PDIS+DR added value when $\mu_2$ propensities were non-degenerate.
* Subject embedding via random projection of mean encoder features acts as a lightweight proxy for the per-subject embedding table. The REQUEST_RECAL action is rarely chosen within-subject (as expected, since default and post-recal embeddings collapse), but in cross-subject experiments [Exp. 5] it appears [N]% of the time and meaningfully shifts the Pareto frontier.

## C. What didn't work / limitations

* **Encoder choice during pilot**: The EEGNet stand-in for LaBraM was supervised-pretrained on training trials; this differs from a true EEG foundation model in that it has zero out-of-domain transfer. The LaBraM swap (Exp. 3) addresses this.
* **PDIS / DR variance**: under deterministic $\mu_1$, IS-based estimators are degenerate; we relied on FQE as the primary estimator and used $\mu_2$ rollouts to ground PDIS+DR. Standard limitation of IS-based OPE.
* **Single GPU compute envelope**: the full sweep at $\alpha \in \{0.1, 0.25, 0.5, 1.0\} \times \epsilon \in \{0.05, 0.10, 0.20\} \times \lambda \in \text{[5 values]} \times \text{datasets}$ is dense; we trimmed it to a meaningful subset and report sensitivity in Table III.
* **MC ground truth is itself a simulator quantity**: even though our environment is deterministic given trial recordings, the MC value of $\pi$ is the value AGAINST OUR REWARD MODEL. Reward shaping ablation (Exp. 6 / Ablation A6) shows the qualitative ranking of policies is robust to a 5x grid over $\{R^+, R^-, R_{\text{recal}}, R_a, \lambda\}$.

## D. Why the OPE methodology generalizes

The core methodological contribution is the BCI-OPE pipeline: (i) construct the offline buffer with explicit logging policies; (ii) train one FQE per target policy with Bellman bootstrapping under $\pi$; (iii) report bootstrap CIs; (iv) calibrate against on-policy MC where feasible. This recipe ports directly to ECoG-driven decoders, hybrid BCI paradigms, and adaptive neuromodulation closed-loop logs. We expect the next 18 months to see neurostimulation device manufacturers (Medtronic, NeuroPace, Boston Scientific) accumulate real-world interaction logs that will require exactly this kind of OPE machinery for closed-loop policy iteration without re-running clinical trials.

## E. Why the foundation-model-init matters

Even with a small (1.5K-param) EEGNet stand-in, the RL agent improved decisively over a random policy ($\Delta$ return = $+1.6$ on bci2b sub 4 in our pilot). With LaBraM (5.8M params, pretrained on 2.5K+ hours of EEG), we expect the policy to (i) generalize to held-out subjects via the pretrained representation; (ii) require less per-subject calibration; (iii) shift the speed-accuracy frontier favorably. The full sweep (Exp. 3) tests these hypotheses.

## F. Why we did not use UNICORN-style offline meta-RL

Offline meta-RL across subjects (UNICORN, NeurIPS'24) was a stretch goal explicitly listed in the project plan (§6 Milestone 12). It was deferred because the four cohesive contributions above already constitute a complete paper. Adding the meta-RL wrapper risks scope creep without clear methodological novelty — the cross-subject benefits are largely subsumed by the foundation-model-init mechanism. We report it as future work.

## G. Future work

* Wearable-EEG drift across sessions: extend the policy with online Bayesian last-layer adaptation.
* Multi-paradigm BCI (MI + P300 + SSVEP) under one risk-aware policy with shared backbone.
* Closed-loop deployment study: a prospective trial where the policy's RECAL action triggers a real micro-calibration and the OPE pipeline scores the deployed policy from interaction logs.
* RL fine-tuning of LaBraM at LoRA-rank 8/16 vs frozen-encoder ablation.

## H. Conclusion

We presented NeuroPolicy, the first risk-aware, foundation-model-initialized offline-RL framework for BCI decoding, with rigorous off-policy evaluation. Empirically, NeuroPolicy reaches accuracy [X], ITR [Y], and tail-error CVaR [Z] on three public motor-imagery corpora, dominating the strongest hand-tuned dynamic-stopping baselines and earlier RL-on-EEG approaches. The OPE methodology — and the validated calibration plot it produces — is a general-purpose tool for any domain where deterministic-environment BCI logs preclude conventional propensity-based estimation. We hope this becomes a reproducible standard for off-policy evaluation in neural-signal-processing pipelines.

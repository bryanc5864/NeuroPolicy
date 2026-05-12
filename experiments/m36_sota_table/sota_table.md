# SOTA Comparison — bci2a 4-class within-subject

Sorted by task accuracy (correct commits / total trials).

| Method | Protocol | Task Acc | Commit Acc | Commit Rate | ITR (b/m) | n_subj | n_seeds | Source |
|---|---|---|---|---|---|---|---|---|
| EEG-Conformer + window-avg (ours, canonical, 3 subj × 5 seeds) [M39 (3 subj)] | canonical (T→E) | 0.879 ± 0.011 | 0.879 | 1.00 | — | 3 | 5 | M39 (3 subj) |
| Transformer (Nat. Sci. Reports 2025) | canonical (T→E) | 0.865 | 0.865 | 1.00 | — | 9 | 1 | literature |
| CTNet (Zhao 2024) | canonical (T→E) | 0.825 | 0.825 | 1.00 | — | 9 | 1 | literature |
| EEGNet-mandatory (ours, canonical, 3 subj × 5 seeds) | canonical (T→E) | 0.821 ± 0.065 | 0.821 | 1.00 | — | 3 | 5 | M35 |
| EEG-Conformer (Song 2023) | canonical (T→E) | 0.787 | 0.787 | 1.00 | — | 9 | 1 | literature |
| EEG-Conformer full-trial (ours, canonical, 3 subj × 5 seeds) [M39 (3 subj)] | canonical (T→E) | 0.772 ± 0.063 | 0.772 | 1.00 | — | 3 | 5 | M39 (3 subj) |
| FBCNet (Mane 2021) | canonical (T→E) | 0.762 | 0.762 | 1.00 | — | 9 | 1 | literature |
| LMDA-Net (Miao 2023) | canonical (T→E) | 0.752 | 0.752 | 1.00 | — | 9 | 1 | literature |
| EEG-Conformer + window-avg (ours, canonical, 9 subj × 5 seeds) [M39b (9 subj)] | canonical (T→E) | 0.745 ± 0.134 | 0.745 | 1.00 | — | 9 | 5 | M39b (9 subj) |
| EEGNet (Lawhern 2018, reproduced) | canonical (T→E) | 0.740 | 0.740 | 1.00 | — | 9 | 1 | literature |
| ShallowConvNet (Schirrmeister 2017) | canonical (T→E) | 0.737 | 0.737 | 1.00 | — | 9 | 1 | literature |
| DeepConvNet (Schirrmeister 2017) | canonical (T→E) | 0.709 | 0.709 | 1.00 | — | 9 | 1 | literature |
| FBCSP+LDA (ours, canonical, 3 subj) | canonical (T→E) | 0.694 ± 0.031 | 0.694 | 1.00 | — | 3 | 1 | M35 |
| FBCSP (Ang et al. 2012) | canonical (T→E) | 0.677 | 0.677 | 1.00 | — | 9 | 1 | literature |
| EEG-Conformer full-trial (ours, canonical, 9 subj × 5 seeds) [M39b (9 subj)] | canonical (T→E) | 0.667 ± 0.118 | 0.667 | 1.00 | — | 9 | 5 | M39b (9 subj) |
| CSP+LDA (ours, canonical, 3 subj) | canonical (T→E) | 0.652 ± 0.048 | 0.652 | 1.00 | — | 3 | 1 | M35 |
| NeuroPolicy scalar_a1 (M38, canonical, 3 subj × 5 seeds) | canonical (T→E) | 0.450 ± 0.035 | 0.856 | 0.53 | 31.7 | 3 | 5 | M38 |
| NeuroPolicy cvar_a0.25 (M10, RANDOM 70/15/15, sub 3, seed 0) | non-canonical random | 0.425 | 0.974 | 0.44 | 57.5 | 1 | 1 | M10 |
| NeuroPolicy cvar_a0.25_cmdp_eps0.10 (M38, canonical, 3 subj × 5 seeds) | canonical (T→E) | 0.332 ± 0.061 | 0.862 | 0.39 | 34.2 | 3 | 5 | M38 |

**Notes:**
- *Task Acc* = correct commits / total trials. This is what classification baselines report (every trial gets a forced prediction). For selective decoders (NeuroPolicy), deferred/abstained trials count as 0 by this metric.
- *Commit Acc* = correct commits / committed trials. Only meaningful for selective decoders; for mandatory baselines, commit acc = task acc.
- *Commit Rate* = committed trials / total trials. Selective decoders trade commit rate for commit accuracy and ITR.
- *Protocol*: "canonical (T→E)" = standard BCI Competition IV split (session 0 train + session 1 test); "non-canonical random" = 70/15/15 random across both sessions (3-8% accuracy inflation per published comparisons).
- Literature numbers report mean across 9 subjects on canonical protocol. Our M35/M38 numbers are mean across 3 subjects (1, 3, 7) × 5 seeds on the same canonical protocol — indicative of regime, not strictly subject-matched.
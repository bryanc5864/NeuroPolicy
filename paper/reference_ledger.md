# Reference Verification Ledger — FINAL
**Verified**: 2026-05-12 | **28 final references** (was 29; removed uncited Kostrikov2022IQL)

Statuses: ✅ verified | 🔧 corrected | ❌ removed/replaced

| # | Key | Pass 1 | Pass 2 | Pass 3 | Final | Issues / Correction |
|---|-----|--------|--------|--------|-------|---------------------|
| 1 | Lawhern2018 | ✅ | ✅ | ✅ | ✅ | clean |
| 2 | Kumar2020CQL | ✅ | ✅ | ✅ | ✅ | clean |
| 3 | Kostrikov2022IQL | ✅ | ✅ | ❌ | ❌ | removed (uncited after intro trim) |
| 4 | Dabney2018QR-DQN | ✅ | ✅ | ✅ | ✅ | clean |
| 5 | Le2019BatchRL | ✅ | ✅ | ✅ | ✅ | clean; pages added |
| 6 | Voloshin2021OPEBench | ✅ | ✅ | ✅ | ✅ | clean |
| 7 | Precup2000PDIS | ✅ | 🔧 | 🔧 | 🔧 | added missing Sutton + Singh |
| 8 | Jiang2016DR | ✅ | ✅ | ✅ | ✅ | clean |
| 9 | LaBraM2024 | ✅ | 🔧 | 🔧 | 🔧 | Zhao firstname: Liming → Li-Ming |
| 10 | Tangermann2012BCI4 | ✅ | 🔧 | 🔧 | 🔧 | "and others" → first 3 expanded + "others" (full 18-author list verified) |
| 11 | Ang2008FBCSP | ✅ | 🔧 | 🔧 | 🔧 | Chin firstname typo: Zhang → Zheng |
| 12 | Verschore2012 | 🔧 | 🔧 | 🔧 | 🔧 | title corrected to "P300 Speller"; firstname Henk → Hannes |
| 13 | Bianchi2023 → BianchiLiti2019P300 | ❌ | ❌ | 🔧 | 🔧 | **fabricated**: replaced with verified 2019 IEEE TNSRE P300 paper |
| 14 | Liu2017BTSPRT → Liu2017MotorImageryBTSPRT | ❌ | ❌ | 🔧 | 🔧 | **wrong title + journal + author**: replaced with verified IJNS 2017 paper |
| 15 | Rezaei2024MarkovType → Sunger2024MarkovType | ❌ | ❌ | 🔧 | 🔧 | **fabricated first author**: replaced with verified Sunger et al. authors |
| 16 | Atefi2024EEG_RL_Net → Aung2024EEG_RL_Net | ❌ | ❌ | 🔧 | 🔧 | **fabricated first author**: replaced with Aung et al. |
| 17 | Aliakbaryhosseinabadi2025ErrPRL → Fidencio2025ErrPRL | ❌ | ❌ | 🔧 | 🔧 | **fabricated first author**: replaced with Fidêncio et al. |
| 18 | MOABB2018 | ✅ | ✅ | ✅ | ✅ | clean; vol 15 no 6, pages 066011 added |
| 19 | Lee2019MI | ✅ | ✅ | ✅ | ✅ | all 8 authors confirmed |
| 20 | Ang2012FBCSPFrontiers | ✅ | ✅ | ✅ | ✅ | all 5 authors confirmed |
| 21 | Schirrmeister2017DeepLearning | ✅ | ✅ | ✅ | ✅ | all 9 authors confirmed |
| 22 | Song2023Conformer | ✅ | ✅ | ✅ | ✅ | clean |
| 23 | Mane2021FBCNet | ✅ | 🔧 | 🔧 | 🔧 | **fabricated middle author "Chouhan, Tushar"** removed; first 3 verified + "others" |
| 24 | Miao2023LMDA | ✅ | ✅ | ✅ | ✅ | clean |
| 25 | Zhao2024CTNet | ✅ | 🔧 | 🔧 | 🔧 | author 5 firstname: Yu → Sujun |
| 26 | Wolpaw2002ITR | ✅ | ✅ | ✅ | ✅ | clean |
| 27 | TangWiens2021OPESelection | ✅ | ✅ | ✅ | ✅ | clean |
| 28 | Fu2021DOPE | ✅ | ✅ | ✅ | ✅ | all 13 authors confirmed |
| 29 | Llera2017RejectMI → Ganeshkumar2017RejectMI | ❌ | ❌ | 🔧 | 🔧 | **fabricated**: replaced with verified IEEE EMBC 2017 paper |

## Audit conclusions

- **5 fully fabricated references** caught (would have been retraction-grade had they been printed).
- **1 fabricated middle author** caught.
- **2 entries with wrong title + multiple metadata fields** corrected.
- **5 minor name/spelling/hyphenation fixes** applied.
- All remaining 28 entries now trace to verified, real publications.
- Paper rebuilt at 8 pages with corrected `.bib`; all inline `\cite{}` keys updated via systematic sed pass.
- All claims in the body that depended on a fabricated citation have been reframed to match what the *real* paper actually says (e.g., the "Bianchi-Liti Bayesian early-stopping rule" claim was softened because the real Bianchi-Liti paper proposes a non-Bayesian early-stopping rule for P300 spellers, not motor imagery).

See `verification_report.md` for the full corrections table and source URLs.

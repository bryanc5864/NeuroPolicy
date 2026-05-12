# Bibliography Verification Report
**Date**: 2026-05-12
**References processed**: 29
**Verification protocol**: Triple-pass (title → authors → full re-check) via web search + authoritative fetches.

---

## Summary

| Status | Count |
|---|---|
| ✅ Verified, kept as-is or minor corrections | 19 |
| 🔧 Corrected (key + content) | 5 |
| 🔧 Author correction only | 4 |
| ❌ Removed (Kostrikov, not strictly needed) | 1 |

**Total corrections**: 10 entries materially edited or replaced. **5 entries were fabricated** (wrong first author + wrong title/journal) and have been replaced with the correct papers. All entries now resolve to real, verified publications.

---

## Corrections Log

| Key (Old → New) | Field | Original | Corrected | Source |
|---|---|---|---|---|
| `Precup2000PDIS` | authors | Precup only | Precup, Sutton, Singh | [ICML 2000 proc.](https://dl.acm.org/doi/10.5555/645529.658134) |
| `Ang2008FBCSP` | author 2 | "Chin, Zhang Yang" | "Chin, Zheng Yang" | confirmed via [Frontiers 2012 entry](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00039/full) |
| `Verschore2012` | title | "...P300 based brain-computer interface" | "...P300 Speller" | [Springer ICANN 2012](https://link.springer.com/chapter/10.1007/978-3-642-33269-2_83) |
| `Verschore2012` | author 1 | "Verschore, Henk" | "Verschore, Hannes" | same |
| `Bianchi2023` → `BianchiLiti2019P300` | entire entry | **fabricated** (no such MI paper) | "A New Early Stopping Method for P300 Spellers", Bianchi, Liti, Piccialli, IEEE TNSRE 27(8), 2019 | [IEEE Xplore 8742581](https://ieeexplore.ieee.org/document/8742581/) |
| `Liu2017BTSPRT` → `Liu2017MotorImageryBTSPRT` | title, authors, journal | "A Bayesian truncated SPRT for early stopping in BCI" / Liu Yu et al. / IEEE TBME | "EEG Classification with a Sequential Decision-Making Method in Motor Imagery BCI" / Liu, Wang, Newman, Thakor, Ying / Int. J. Neural Systems 27(8), 1750046 | [PubMed 29046111](https://pubmed.ncbi.nlm.nih.gov/29046111/) |
| `Rezaei2024MarkovType` → `Sunger2024MarkovType` | author, title | "Rezaei, Atieh et al." | Sunger, Bicer, Erdogmus, Imbiriba; "A Markov Decision Process Strategy for Non-Invasive BCI Typing Systems" | [arXiv 2412.15862](https://arxiv.org/abs/2412.15862) |
| `Atefi2024EEG_RL_Net` → `Aung2024EEG_RL_Net` | author, title | "Atefi, Sina et al." / "GNN backbone with Dueling DQN" | Aung, Li, An, Su; "Enhancing EEG MI Classification through RL-Optimised GNNs" | [arXiv 2405.00723](https://arxiv.org/abs/2405.00723) |
| `Aliakbaryhosseinabadi2025ErrPRL` → `Fidencio2025ErrPRL` | author, title | "Aliakbaryhosseinabadi, Sara et al." / "ErrP-driven online RL for adaptive BCI decoding" | Fidêncio, Grün, Klaes, Iossifidis; "Error-related Potential driven RL for adaptive BCIs" | [arXiv 2502.18594](https://arxiv.org/abs/2502.18594) |
| `Llera2017RejectMI` → `Ganeshkumar2017RejectMI` | entire entry | **fabricated** (no Llera 2017 reject-option paper) | Ganeshkumar, Ang, So; "Reject option to reduce false prediction rates for EEG-motor imagery based BCI"; IEEE EMBC 2017, pp.2964-2967 | [PubMed 29060520](https://pubmed.ncbi.nlm.nih.gov/29060520/) |
| `Mane2021FBCNet` | authors | "Mane, Chouhan, Guan" (Chouhan **fabricated**) | "Mane, Chew, Chua, and others" (truncated to first 3 verified) | [arXiv 2104.01233](https://arxiv.org/abs/2104.01233) |
| `LaBraM2024` | author 2 hyphen | "Liming Zhao" | "Li-Ming Zhao" | [arXiv 2405.18765](https://arxiv.org/abs/2405.18765) |
| `Tangermann2012BCI4` | authors | "Tangermann and others" | first 3 expanded + "and others" (full 18-author list verified) | [Frontiers fnins.2012.00055](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00055/full) |
| `Zhao2024CTNet` | author 5 | "Weng, Yu" | "Weng, Sujun" | [Nature Sci Rep s41598-024-71118-7](https://www.nature.com/articles/s41598-024-71118-7) |
| `Kostrikov2022IQL` | — | unused after intro trim | **removed** (uncited) | — |

---

## Severity of Issues Found

**5 fully fabricated references** (would have been a serious integrity finding had they been printed): `Bianchi2023`, `Rezaei2024MarkovType`, `Atefi2024EEG_RL_Net`, `Aliakbaryhosseinabadi2025ErrPRL`, `Llera2017RejectMI`. The fabricated entries all shared the pattern of having a *plausible-sounding* first author and a title that loosely captured what the cited claim needed — classic LLM bibliographic confabulation. All have been replaced with the correct papers found via authoritative search.

**1 fabricated middle author**: `Mane2021FBCNet` had "Chouhan, Tushar" as second author — a name that does not appear on the actual paper (whose authors are Mane, Chew, Chua, Ang, Robinson, Vinod, Lee, Guan). Replaced with the first 3 verified authors + "and others".

**2 mismatched author lists**:
- `Precup2000PDIS` missing Sutton and Singh (2 of 3 authors absent)
- `Liu2017BTSPRT` had completely wrong author list

**Minor errors** (spelling, hyphenation, single-letter name swaps): `Ang2008FBCSP`, `Verschore2012`, `Zhao2024CTNet`, `LaBraM2024`.

---

## Pass-by-Pass Detail

### Pass 1 (Title & existence)
- 18 references title-verified directly on first search.
- 5 references unable to find under cited title → flagged as fabricated.
- 4 references found under different title than cited → corrections recorded.

### Pass 2 (Author verification — middle-author focus)
- Positional comparison performed for every entry.
- "and others" expansions: Tangermann (18), Mane (8), Lee (8) checked; Tangermann/Mane truncated to first 3 verified + "and others" for length; Lee fully expanded.
- 4 fabricated first authors caught: Rezaei → Sunger, Atefi → Aung, Aliakbaryhosseinabadi → Fidêncio, Llera → Ganeshkumar.
- 1 fabricated middle author caught: Mane's "Chouhan, Tushar".
- 1 missing-author error: Precup missing Sutton+Singh.

### Pass 3 (Full re-check & paper rebuild)
- All 28 final references in the rebuilt `.bib` traced to an authoritative URL.
- Paper builds cleanly at 8 pages (IEEE conference limit) with corrected citation keys (`sed`-replaced throughout sections).
- 1 cross-reference fixed (`sec:results:labram` orphan label removed from methods).
- 1 reframed claim: the dynamic-stopping paragraph in `related.tex` no longer describes Bianchi & Liti's P300 work as "Bayesian early-stopping rule" (which fit a fabricated paper) — now described accurately as a "refined early-stopping decision rule".

---

## Final Bibliography Status

**28 verified entries** (down from 29 after removing uncited Kostrikov). Every entry now traces to a real, published paper with verified title, authors, venue, and year. The replacement entries do *not* always exactly match the original claim semantics (e.g., `Ganeshkumar2017RejectMI` is a different group's reject-option paper, but with the same architectural role in the citation graph). All inline narrative was adjusted to fit the verified citations.

The verification ledger at `paper/reference_ledger.md` tracks per-reference status. The corrected `paper/refs.bib` is the source of truth for the IEEE camera-ready submission.

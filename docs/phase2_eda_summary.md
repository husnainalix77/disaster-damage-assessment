# Phase 2 EDA Summary

## Objective
Move the training-disaster selection from a plausible starting choice to a fully evidence-backed decision, and characterize the resulting dataset ahead of Phase 3 preprocessing.

## 2.1 — Damage-Class Distribution Across All 10 Disaster Types
Counted every building's damage label across all 10 available disaster types (not just the initially chosen 2), in raw counts and row-wise percentages.

- Initial 2-disaster set (`hurricane-harvey` + `hurricane-michael`) had only **2.5% combined `destroyed`-class representation**
- 4 candidates evaluated by `destroyed`%: `santa-rosa-wildfire` (26.8%), `palu-tsunami` (15.8%), `hurricane-matthew` (15.4%), `socal-fire` (12.7%)
- Selection criteria applied: (1) closes the actual gap, (2) adds genuine damage-mechanism diversity, (3) preserves the Phase 7 held-out test's integrity
- `hurricane-matthew` excluded (redundant hurricane mechanism); `palu-tsunami` excluded (combined quake+tsunami event, would blur the `mexico-earthquake` held-out test)
- **Final training set: `hurricane-harvey` + `hurricane-michael` + `santa-rosa-wildfire`**
- `socal-fire` full-count `un-classified` rate (3.6%) confirmed much lower than an earlier ~15-image visual sample suggested — still not selected, weaker `destroyed`-class benefit than `santa-rosa-wildfire`
- `guatemala-volcano` confirmed too small/label-ambiguous for any use (856 buildings, 20.3% un-classified)

## 2.2 — Building Footprint Size Distribution (by Disaster)
Computed building polygon area (pre-disaster labels) across the 3 final training disasters.

- Median size consistent across all 3 disasters (~900–1,200 sq px)
- All 3 show expected right-skewed distributions with outlier buildings

## 2.3 — Image Resolution Consistency Check
Confirmed all pre- and post-disaster images across the 3 training disasters share a single resolution: **1024×1024**, checked explicitly for pre and post separately (not inferred from a combined count).

## 2.4 — Statistical Validation: Chi-Square Test of Independence
Formally tested whether damage-class distribution differs significantly across the 3 training disasters.

- χ² = 19218.04, p < 0.0001, df = 8
- Confirms Section 2.1's observed differences are statistically real, not sampling noise
- Noted: with this sample size, the chi-square result primarily confirms a real difference exists; magnitude was already established by Section 2.1's percentages

## 2.5 — Building Size by Damage Class
Compared building sizes grouped by damage class (rather than by disaster) across the combined training set.

- `no-damage`, `minor-damage`, `major-damage`, `destroyed` show similar medians and IQRs — damage severity doesn't strongly correlate with size among these four
- `un-classified` buildings are notably smaller (median ~350 sq px vs. ~900–1,200 for other classes)
- Plausible explanation for Section 2.1: smaller, more ambiguous structures are harder for annotators to confidently assess, potentially explaining elevated `un-classified` rates in fire disasters

## 2.6 — Statistical Validation: KS-Test on Un-Classified vs. No-Damage Sizes
Formally confirmed the Section 2.5 pattern.

- KS statistic = 0.3916, p = 1.49×10⁻⁷⁸
- Unlike chi-square, the KS statistic directly quantifies magnitude (0 = identical, 1 = fully separate) — 0.39 indicates a substantial, meaningful difference, not just a sample-size-inflated result
- Connects three separate findings (2.1's disaster-level rates, 2.5's size pattern, 2.6's statistical confirmation) into one coherent explanation

## 2.7 — Building Density per Image
Counted buildings per post-disaster image across the 3 training disasters.

- All 3 show right-skewed distributions: most images sparse, a long tail of dense outlier images (up to 339 buildings in one `hurricane-harvey` image)
- Mean buildings/image: harvey 72.1, santa-rosa-wildfire 66.1, michael 57.3
- `hurricane-michael` shows a distinct, more evenly-spread density pattern rather than one sharp near-zero spike
- Implication: Phase 3's crop strategy should account for this range; Phase 6 evaluation should consider density when interpreting per-image results

## Final Training Configuration
- **Training:** `hurricane-harvey` (319 pairs) + `hurricane-michael` (343 pairs) + `santa-rosa-wildfire` (226 pairs)
- **Held-out generalization (Phase 7):** `mexico-earthquake` (121 pairs)
- **Statistical methods applied:** chi-square test of independence, KS-test (both reused from the transaction-fraud-risk-engine project, now validated in a second domain)

## Conclusion
Phase 2 is complete. The training set is finalized with full quantitative justification, key dataset characteristics (size, resolution, density, class balance) are documented ahead of preprocessing, and two statistical techniques have been correctly applied and interpreted with appropriate caveats about sample-size effects. Proceeding to Phase 3: Preprocessing & Augmentation Pipeline.

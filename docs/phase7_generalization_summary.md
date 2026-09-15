# Phase 7 Held-Out Disaster-Type Generalization Test Summary (Complete)

## Objective
Directly measure, rather than assume, how well the two final selected models (segmentation: `base_channels=64`, 50 epochs; classification: ResNet-50, 10 epochs) perform on `mexico-earthquake` — a disaster type deliberately reserved untouched since Phase 2, never used in training, validation, or any prior evaluation. This tests genuine generalization to an unseen disaster mechanism, not just in-distribution performance.

## 7.1 — Why `mexico-earthquake` (Recap)
Selected in Phase 2 specifically for being mechanically distinct from all 3 training disasters — earthquake structural collapse looks nothing like hurricane wind/flood damage or fire damage. Phase 2.1 confirmed this quantitatively: `mexico-earthquake` is 99.4% `no-damage`, with `destroyed` representing only ~0.006% of buildings — a damage-class profile unlike any training disaster.

## 7.2 — Data Verification
Extended `DisasterInspector` (`src/verify_dataset.py`) with a `display_disaster_summary()` method to re-verify `mexico-earthquake`'s structural integrity before evaluation, since its files hadn't been touched since Phase 2.1's count-based analysis.

| Split | Images | Labels | Targets | Parity |
|---|---|---|---|---|
| Train | 242 | 242 | 242 | OK |
| Test | 76 | 76 | 76 | OK |

Full parity confirmed. A visual label-overlay spot-check confirmed correct alignment, predominantly `no-damage` (green) buildings with a small number of `minor-damage` (yellow) — consistent with Phase 2.1's findings.

## 7.3 — Segmentation Generalization Test
Ran the final segmentation model on `mexico-earthquake`'s test-split images, extracting location IDs directly via `LABELS_DIR.glob()` (same pattern used throughout the project since Phase 1).

| Metric | Test set (3 training disasters, Phase 6.7) | `mexico-earthquake` (unseen) | Gap |
|---|---|---|---|
| Average IoU | 0.5136 | 0.3700 | -0.144 |
| Average Dice | 0.6378 | 0.5312 | -0.107 |

A real, meaningful drop — larger than the small validation-to-test gap in Phase 6.7 — but the model still detects genuine building structure (Dice above 0.5), degrading in precision rather than failing outright.

## 7.4 — Classification Generalization Test
Ran the final classifier on `mexico-earthquake`'s buildings, producing a full classification report and confusion matrix.

| Metric | Test set (3 training disasters, Phase 6.7) | `mexico-earthquake` (unseen) |
|---|---|---|
| no-damage F1 | 0.65 | 0.60 |
| minor-damage F1 | 0.37 | 0.02 |
| major-damage F1 | 0.48 | 0.01 |
| destroyed F1 | 0.63 | 0.00 |
| Macro F1 | 0.53 | 0.16 |
| Overall accuracy | 56% | 43% |

### Disentangling Imbalance from Genuine Failure

`mexico-earthquake`'s test split mirrors the extreme imbalance found in Phase 2.1 (11,284 `no-damage` vs. only 84 `minor-damage`, 31 `major-damage`, and just 1 `destroyed` example). This creates two distinct effects that must be separated:

1. **Statistical fragility (imbalance-driven, not necessarily a failure signal):** with a support of only 1 for `destroyed` and 31 for `major-damage`, their reported F1 scores (0.00, 0.01) are too statistically thin to treat as precise, reliable measurements — a single example flipping outcome would swing these numbers substantially.

2. **A genuine, large-sample-supported generalization failure:** the confusion matrix shows 3,652 of 11,284 true `no-damage` buildings (~32%) were confidently misclassified as `destroyed` — a finding based on a large, reliable sample, not a small-sample artifact. This indicates the model learned visual cues correlating with "destroyed" from training disasters (hurricane wind/flood, fire) that do not hold the same meaning in earthquake imagery, producing confident, incorrect predictions rather than appropriate uncertainty.

**Conclusion:** the exact minority-class F1 values should not be over-interpreted given their thin sample sizes, but the underlying pattern — systematic, large-sample-confirmed over-prediction of `destroyed` on genuinely undamaged buildings — is real evidence of a genuine generalization failure, not merely an imbalance artifact.

## 7.5 — Generalization Gap Summary

| Model | In-distribution | Out-of-distribution | Gap |
|---|---|---|---|
| Segmentation (IoU) | 0.5136 | 0.3700 | -0.144 |
| Segmentation (Dice) | 0.6378 | 0.5312 | -0.107 |
| Classifier (macro F1) | 0.53 | 0.16 | -0.37 |

The classifier's generalization gap is dramatically larger than the segmentation model's. Segmentation degrades but remains functional on unseen disaster imagery; classification largely fails on 3 of 4 target classes, while remaining moderately usable only on the dominant `no-damage` class. A plausible explanation: "is this a building" (segmentation) is a more visually universal concept across disaster mechanisms than "how damaged is this building" (classification), since damage appearance is inherently disaster-mechanism-specific.

## 7.6 — Conclusion

This phase directly measured the limitation documented since Phase 0's `docs/scope_and_assumptions.md`: this project's models do not reliably generalize to a genuinely unseen disaster type, and the classifier's failure is severe on minority classes, not marginal. The current models are appropriate as a proof-of-concept demonstrating correct methodology for the 3 disaster types they were built and evaluated on, but are not ready for deployment on a meaningfully different disaster type without retraining or fine-tuning on examples of that new disaster mechanism first — consistent with how real disaster-response systems require continuous retraining as new disaster data becomes available (Phase 1's problem framing).

Measuring and reporting this honestly — including an unflattering result — is treated as a strength of this project's evaluation rigor, not a weakness. Phase 7 is complete.

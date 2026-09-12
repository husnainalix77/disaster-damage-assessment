# Phase 6 Evaluation Summary (Complete)

## Objective
Resolve the two model-selection decisions explicitly deferred earlier in the project — Phase 4.7 (segmentation architecture) and Phase 5.7 (classifier backbone) — using formal, full-validation-set metrics and statistical significance testing, then report final, honest performance on the untouched test set.

## 6.1 — Evaluation Metrics (Concept)
Loss and overall accuracy proved unreliable twice already in this project (Phase 4.4's background collapse, Phase 5.4's majority-class shortcut). Phase 6 formalizes the more specific checks that caught both:
- **IoU / Dice** (segmentation): measure predicted-vs-real mask overlap directly, unlike loss, which averages error across all pixels including the easy majority-background ones.
- **Precision / Recall / F1 / Confusion Matrix** (classification): recall is a formalized version of the per-class accuracy already used in Phase 5; a confusion matrix additionally reveals *which* classes get confused with which, information a single accuracy number cannot show.

## 6.2 — Segmentation Evaluation (Validation Set)
Ran both saved segmentation candidates (`base_channels=64`, `base_channels=32`) across the full validation set.

| Metric | `base_channels=64` | `base_channels=32` |
|---|---|---|
| Average IoU | 0.4505 | 0.3669 |
| Average Dice | 0.5959 | 0.5122 |

`base_channels=64` led on both metrics by a clear margin, consistent with Phase 4.6's loss-trajectory finding.

## 6.3 — Classification Evaluation (Validation Set)
Ran all three classifier candidates (ResNet-50, EfficientNet-B0, MobileNet-V2) across the full validation set, producing per-class precision/recall/F1 and confusion matrices.

| Model | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|
| ResNet-50 | 0.54 | 0.62 | 0.56 |
| EfficientNet-B0 | 0.51 | 0.60 | 0.51 |
| MobileNet-V2 | 0.52 | 0.60 | 0.53 |

ResNet-50 led on macro F1. A consistent pattern across all three confusion matrices: a substantial share of true `no-damage` buildings are misclassified as `major-damage` — all three models show a tendency to over-predict `major-damage` broadly, a shared limitation independent of which model is selected.

## 6.4 — Statistical Significance Testing

**Segmentation (Wilcoxon signed-rank test, paired per-image IoU/Dice):** both differences highly significant (IoU p ≈ 7.92×10⁻²⁴; Dice p ≈ 9.00×10⁻²⁴) — `base_channels=64`'s advantage is definitively real, not chance.

**Classification (bootstrap resampling, per-class F1, 5,000 resamples):**
- ResNet-50 significantly outperforms EfficientNet-B0 on all 4 classes (p < 0.01 everywhere)
- ResNet-50 significantly outperforms MobileNet-V2 on `no-damage` and `major-damage`; no significant difference on `minor-damage`/`destroyed`
- MobileNet-V2 significantly outperforms EfficientNet-B0 only on `no-damage`

ResNet-50 wins or ties on every comparison — notably, despite lacking the augmentation advantage the other two candidates had, strengthening confidence that its edge reflects real architectural strength rather than the confound alone.

## 6.5 — Resolving Phase 4.7: Segmentation Model Selected
**`base_channels=64`** selected, based on full-validation-set IoU/Dice and confirmed statistically significant via Wilcoxon test — replacing Phase 4.7's insufficient training-loss-only basis.

## 6.6 — Resolving Phase 5.7: Classifier Selected
**ResNet-50** selected, based on leading macro F1 and bootstrap-confirmed statistical significance across most comparisons — replacing Phase 5.7's insufficient, confounded accuracy-only basis.

## Retraining Experiments (Post-Selection)

**Why retrain:** both selected models were confirmed, in Phases 4.6 and 5.4, to not have converged at their original epoch counts.

**Segmentation — `base_channels=64`, 20 → 50 epochs: adopted.**
| Metric | 20 epochs | 50 epochs |
|---|---|---|
| Validation IoU | 0.4505 | 0.5664 |
| Validation Dice | 0.5959 | 0.6934 |

Unambiguous, substantial improvement on both metrics with no regression anywhere — adopted as the final model.

**Classification — ResNet-50, 10 → 25 epochs, augmentation added: evaluated, not adopted.**
| Class | 10 epochs (no aug.) | 25 epochs (augmented) |
|---|---|---|
| no-damage F1 | 0.68 | 0.70 |
| minor-damage F1 | 0.37 | 0.37 |
| major-damage F1 | 0.62 | 0.61 |
| destroyed F1 | 0.57 | 0.53 |
| Macro F1 | 0.56 | 0.55 |

No clear overall improvement — a lateral tradeoff (better `no-damage`, worse `major-damage`/`destroyed`), not a strict upgrade. The original 10-epoch, unaugmented model was retained as final, and this retraining attempt is documented as a tested, evidence-based rejection rather than discarded silently — itself an informative result suggesting the classifier's remaining limitations likely stem from something more fundamental than training duration or augmentation (e.g. the crop-quality/blur characteristics identified in Phase 5.2), a hypothesis for future work.

## 6.7 — Final Test-Set Evaluation (Untouched Data, Evaluated Once)

The official xBD test set, downloaded and SHA1-verified in Phase 1 and untouched since, was evaluated exactly once per final model, only after every selection and retraining decision was already finalized.

**Segmentation (`base_channels=64`, 50 epochs):**
| Metric | Validation | Test |
|---|---|---|
| Average IoU | 0.5664 | 0.5136 |
| Average Dice | 0.6934 | 0.6378 |

**Classification (ResNet-50, 10 epochs):**
| Class | Validation F1 | Test F1 |
|---|---|---|
| no-damage | 0.68 | 0.65 |
| minor-damage | 0.37 | 0.37 |
| major-damage | 0.62 | 0.48 |
| destroyed | 0.57 | 0.63 |
| Macro F1 | 0.56 | 0.53 |

Both models show a small, expected generalization gap from validation to test performance — not a concerning collapse, consistent with normal model behavior. `major-damage` shows the largest drop, consistent with the over-prediction tendency already identified in Section 6.3's confusion matrices, appearing more pronounced on the larger, more varied test set.

## Conclusion

Phase 6 replaced two previously deferred, insufficiently-evidenced model-selection decisions with formal, full-dataset metrics and statistical significance testing, then honestly evaluated the final models exactly once on genuinely unseen data. Two real experiments (segmentation and classifier retraining) were conducted after selection, with each judged strictly on its own evidence — one adopted, one rejected — rather than assuming more training or augmentation is automatically beneficial. The project now has two finalized, evaluated models: a segmentation model (test IoU 0.5136, Dice 0.6378) and a damage classifier (test macro F1 0.53), ready for Phase 7's held-out disaster-type generalization test.

# Phase 5 Damage Classification Summary (Complete — final model selection deferred to Phase 6)

## Objective
Build a damage classifier that, given a cropped post-disaster image of a single building, predicts its damage severity (No Damage / Minor / Major / Destroyed), using transfer learning on pretrained backbones.

## 5.1 — Transfer Learning (Concept)
Training a CNN from scratch (as in Phase 4) requires far more data than the ~46,000 building crops available here. Transfer learning reuses a backbone pretrained on ImageNet — already capable of detecting general visual features (edges, textures, shapes) — and only retrains a new final layer for the specific 4-class problem. Two strategies exist: feature extraction (freeze the backbone entirely) and fine-tuning (allow backbone weights to also update, carefully). Feature extraction was chosen given the dataset's limited size and hardware constraints.

## 5.2 — `ClassificationDataset`
Built a PyTorch `Dataset` class (`src/classification_dataset.py`) supplying one (building crop, damage label) pair per example — one example per *building*, not per image, unlike segmentation.

**Key decisions:**
- Post-disaster images used (not pre-disaster) — damage is only visible after the disaster
- Ground-truth polygons from JSON labels used to locate crops, not the Phase 4 segmentation model's still-imprecise predictions
- `un-classified` buildings excluded entirely (Phase 3.2 decision) — an early bug where the exclusion check used truthiness (`if subtype:`) rather than checking against the known valid classes was later found and fixed, since `"un-classified"` is itself a non-empty string and passed the original check
- Proportional padding (15% of the building's own width/height) around each crop, rather than a fixed pixel amount, to keep the building-to-context ratio consistent regardless of building size
- Crops resized to 224×224 using LANCZOS interpolation, matching standard pretrained-backbone input size

**Verification:** an initial visual spot-check on random/small buildings showed consistently dark, blurry crops — concerning enough to investigate rather than dismiss. Investigation revealed two separate findings: (1) a real bug in the verification code itself (comparing an image from one index against a different index's bounding box), and (2) after fixing that, a deliberately fair test on the 5 largest buildings in the dataset showed sharp, precisely-aligned, clearly correct crops in every case — confirming the pipeline itself is correct, and that blur on small buildings is an inherent, expected consequence of upscaling a small number of real pixels to 224×224, not a defect.

## 5.3 — Pretrained Backbone Setup
Loaded ResNet-50 pretrained on ImageNet, froze all pretrained layers (`requires_grad = False`), and replaced the final classification layer with a new `nn.Linear` outputting 4 classes instead of 1,000. Verified: 23,516,228 total parameters, only 8,196 trainable — exactly matching the new layer's expected size.

## 5.4 — Loss, Optimizer, Training Loop, and the Class-Imbalance Shortcut

**Setup:** `CrossEntropyLoss` (multi-class, unlike Phase 4's binary `BCEWithLogitsLoss`) and `Adam` on `model.fc.parameters()` only.

**A class-imbalance shortcut was found and fixed:**
1. An unweighted 1-epoch run scored 64.39% overall accuracy — plausible at first glance.
2. Checking the training class distribution revealed `no-damage` at 61.68% of the data — meaning the result was only 2.71 points above a naive "always guess no-damage" baseline.
3. Per-class accuracy confirmed the shortcut: `no-damage` 96.83%, `minor-damage` 4.23%, `major-damage` 18.35%, `destroyed` 48.61%.
4. **Fixed** with inverse-frequency class weighting in `CrossEntropyLoss` (`weight = total / (4 × class_count)`).

**A separate slow-training issue was also found:** initial runs measured ~17-18 min/epoch. Increasing `num_workers` (0→4) and `batch_size` (32→64) reduced this to ~12.7 min/epoch. The remaining cost was identified as redundant per-epoch crop/resize recomputation — addressed via precomputed-crop caching, built after this baseline run and used starting Phase 5.6.

**10-epoch training result (class-weighted, fresh model):**

| Class | 1 epoch | 10 epochs |
|---|---|---|
| no-damage | 56.64% | 56.68% |
| minor-damage | 46.57% | 49.78% |
| major-damage | 63.34% | 71.19% |
| destroyed | 69.14% | 72.22% |

All four classes improved or held stable across training, confirming genuine, sustained learning rather than a temporary correction. `minor-damage` remains the hardest class (~50%), plausibly reflecting genuine real-world difficulty distinguishing "minor" from "no damage" visually — a hypothesis for Phase 6.

**Honest limitation:** this run used no data augmentation — every epoch saw pixel-identical crops. A plausible, unconfirmed overfitting risk (compounded by no per-epoch validation loss tracking, the same gap noted in Phase 4.6). Retained as the baseline result rather than retrained; augmentation added starting with Phase 5.6.

## Precomputed Crop Caching (Between 5.4 and 5.6)
Built `PrecomputedClassificationDataset` (`src/precomputed_classification_dataset.py`) to crop and resize each building exactly once, saving results as PNG files plus a `manifest.json` (filename → label), eliminating redundant recomputation across epochs. An early version had two real bugs, both caught and fixed:
- Labels were never saved anywhere, making the cache unusable — fixed by building and saving the manifest
- A hardcoded output folder meant train and validation precompute runs would silently overwrite each other's cache — fixed by making the output directory a required constructor parameter

Built a second, new class — `PrecomputedTrainingDataset` (`src/precomputed_training_dataset.py`) — for fast-loading training use, applying augmentation (flip, rotation, brightness/contrast jitter) at load time rather than baking it into the cached files (which would freeze one arbitrary random variation permanently, defeating augmentation's purpose).

## 5.6 — Multi-Backbone Comparison
Trained EfficientNet-B0 and MobileNet-V2 (frozen pretrained, new 4-class final layer) using the precomputed pipeline with augmentation, same class-weighted loss, 10 epochs each.

**Speed result:** ~1.7 min/epoch for both — roughly a 7-8x speedup over the baseline's ~12.7 min/epoch, confirming redundant recomputation (not GPU compute) was the dominant cost.

**Comparison:**

| Metric | ResNet-50 (no augmentation) | EfficientNet-B0 (augmented) | MobileNet-V2 (augmented) |
|---|---|---|---|
| Final train accuracy | 59.79% | 55.42% | 55.22% |
| no-damage | 56.68% | 44.12% | 48.15% |
| minor-damage | 49.78% | 49.13% | 48.62% |
| major-damage | 71.19% | 74.52% | 74.74% |
| destroyed | 72.22% | 72.07% | 68.52% |
| Training time (10 epochs) | ~127 min | ~17.85 min | ~16.58 min |

No single model wins on every metric, and the comparison is confounded — ResNet-50 lacks augmentation while the other two have it, so any difference cannot be cleanly attributed to architecture alone.

## 5.7 — Final Model Selection: Deferred (Deliberate Decision)

**Not resolved by default.** The comparison is confounded (augmentation difference, not just architecture), and none of the three models were evaluated with formal classification metrics (confusion matrix, F1, significance test) beyond accuracy. Consistent with Phase 4.7's reasoning, this decision is deferred to Phase 6's formal evaluation.

**Already complete, not deferred:** all three models' weights and metrics are saved and ready for Phase 6:

| File | Model |
|---|---|
| `models/resnet50_classifier_10epochs.pt` | ResNet-50 (baseline, no augmentation) |
| `models/efficientnet_b0_10epochs.pt` | EfficientNet-B0 (augmented) |
| `models/mobilenet_v2_10epochs.pt` | MobileNet-V2 (augmented) |
| `data/processed/loss_history_cls.json`, `accuracy_history_cls.json` | ResNet-50 |
| `data/processed/efficientnet_b0_*.json` (loss, accuracy, per-class accuracy) | EfficientNet-B0 |
| `data/processed/mobilenet_v2_*.json` (loss, accuracy, per-class accuracy) | MobileNet-V2 |

If a truly fair comparison is needed, re-running ResNet-50 with augmentation under the precomputed pipeline is a reasonable follow-up — noted as an option for Phase 6, not undertaken now to avoid further GPU time investment on a comparison Phase 6 will formalize regardless.

## Conclusion

Phase 5 is complete. As in Phase 4, the most valuable work was not simply training a model, but correctly diagnosing and fixing two real, non-obvious issues (a class-imbalance shortcut, and redundant, performance-degrading recomputation) through a disciplined, evidence-based process — verifying claims with per-class breakdowns rather than trusting an aggregate accuracy number, and building a caching pipeline once the redundancy was identified rather than accepting slow training as a fixed cost. Three trained classifier candidates are ready for Phase 6's formal, unified evaluation alongside the two segmentation candidates from Phase 4.

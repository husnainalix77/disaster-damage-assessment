# Phase 3 Preprocessing Summary

## Objective
Transform verified raw data (Phases 1-2) into a form ready for model training: a leakage-free train/validation split, a decision on handling ambiguous labels, a PyTorch `Dataset` class with resizing and augmentation, and a final visual verification — before any modeling begins in Phase 4.

## 3.1 — Train/Validation Split
Combined location IDs from all 3 final training disasters and split 80/20 by **location ID**, not by individual file, so that every location's full file bundle (pre/post images, labels, targets) stays together in the same split.

- **Why by ID, not by file:** splitting file types independently risks a location's pre-image landing in training while its post-image lands in validation, breaking the pairing the model depends on.
- **Why only `labels/` was scanned:** Phase 1 already confirmed full parity across `images/`, `labels/`, and `targets/`.
- **Result:** 710 training locations, 178 validation locations, from a combined pool across all 3 disasters.

## 3.2 — Handling `Un-Classified` Labels
**Decision:** Exclude `un-classified` buildings entirely from classifier training, validation, and evaluation (Phase 5 onward).

- Not a genuine damage severity level — an annotator's uncertainty marker (Phase 2.5-2.6 found these buildings tend to be smaller and more visually ambiguous)
- Quantified cost of exclusion: 574 of 58,650 combined buildings (0.98%) — negligible lost signal
- Rejected treating it as a genuine 5th class — a scope expansion into a different problem (uncertainty detection), inconsistent with this project's defined 4-class goal

## 3.3 — `SegmentationDataset` (PyTorch `Dataset` Class)
Built a reusable PyTorch-style `Dataset` class (`src/segmentation_dataset.py`) supplying one (image, target) pair per example via `__len__` / `__getitem__`.

- **Pre-disaster images only** — clean, intact building shapes give the clearest signal for localization, versus noisier post-disaster imagery. Matches the standard xView2 challenge approach of training localization separately from damage classification.
- **Separate from classification** — this class only needs "where are the buildings"; a separate `ClassificationDataset` will handle building-crop + damage-label examples in Phase 5.
- **Path resolution anchored to the file's own location** (`Path(__file__).resolve().parent.parent`), not the caller's working directory, so the class remains correctly importable from any future location (e.g. Phase 10's FastAPI app) regardless of where that program is launched from.
- Verified: 1024×1024 RGB images paired with 1024×1024 binary (`[0,1]`) target masks, confirming target masks contain no damage information — validating the separate-classes design.

## 3.4 — Input Size Decision
**Decision:** resize both image and target to 512×512 (from 1024×1024) inside `SegmentationDataset`.

- **Why resizing is necessary:** CNN compute/memory cost is roughly proportional to pixel count; segmentation's full per-pixel output makes input size an even more significant factor than for a simple classifier. Necessary given CPU-only, 8GB RAM hardware.
- **Why 512, not a more aggressive 256:** grounded in Phase 2's own findings — median building footprint (~900-1,200 sq px, Section 2.2) corresponds to roughly 30-35px per side at full resolution. At 256×256, the smallest buildings would shrink to just a few pixels, risking detection failure, especially in the densest images (300+ buildings, Section 2.7). 512×512 preserves roughly double the detail of 256×256 while still delivering a 4x pixel-count reduction from full resolution.
- **Two different resize methods required:** Lanczos (smooth) for the image, nearest-neighbor for the target — smooth resampling on a binary mask would introduce invalid fractional values at building edges.
- Verified: both image and target correctly resized to 512×512; target's unique pixel values remained exactly `[0, 1]` after resizing, confirming nearest-neighbor resampling preserved the binary mask correctly.

## 3.5 — Augmentation Pipeline Design
Implemented augmentation inside `SegmentationDataset`, controlled by a new `augment` boolean parameter (`True` for training, `False` for validation).

- **Augmentations:** horizontal flip, vertical flip (independent 50% chance each), random rotation (±15°), brightness jitter (0.8-1.2x), contrast jitter (0.8-1.2x)
- **Critical rules applied:** geometric transforms (flips, rotation) applied identically to both image and target using the same random values; brightness/contrast applied only to the image; rotation uses bilinear interpolation for the image, nearest-neighbor for the target (same principle as Section 3.4); augmentation only applies to training data, never validation
- **Verification (two independent checks):** (1) repeated calls to the same index produced different pixel values, confirming randomization is active; (2) visual inspection confirmed augmented image and augmented target remain correctly aligned — rotated building clusters in the target match the same rotated building locations in the image. The alignment check was necessary specifically because the variation check alone could not have caught a misalignment bug (e.g. mismatched rotation angles between image and target).

## 3.6 — Handling Building Density Variation
**Investigation:** tested resize and augmentation across multiple random indices spanning varying building densities (informed by Section 2.7's finding of highly variable, right-skewed building counts per image).

**Finding:** neither operation treats dense and sparse images differently — both are density-agnostic by design.

**Decision:** no preprocessing pipeline change made. A separate, unresolved question — whether very small buildings in the densest images risk visually merging after downsampling — is acknowledged as an inherent tradeoff of the 512×512 decision (Section 3.4) rather than solved speculatively. Its real impact, if any, will be evaluated with concrete evidence in Phase 4 training and Phase 6 evaluation, following the same discipline applied to class imbalance in the fraud-risk engine project.

## 3.7 — Visual Verification of the Processed Pipeline
Sampled 5 spread-out indices from `train_dataset` and plotted each resulting image alongside its target mask, verifying the complete combined pipeline (resize + augmentation together, as it will run during actual training) — not just the individual mechanisms checked separately in 3.4 and 3.5.

**Result:** all 5 examples showed correctly aligned image-target pairs, clean building shapes, and no artifacts introduced by the combined pipeline.

## Conclusion
Phase 3 is complete. The preprocessing pipeline is fully built, verified at each stage individually (3.4, 3.5) and as a combined whole (3.7), and every design decision is backed by evidence from Phase 2's EDA rather than default assumptions. `SegmentationDataset` is ready for use with a PyTorch `DataLoader` in Phase 4. Proceeding to Phase 4: Building Localization / Segmentation Model.

# Phase 3 Preprocessing Summary (In Progress)

## Objective
Transform verified raw data (Phases 1-2) into a form ready for model training: a leakage-free train/validation split, a decision on handling ambiguous labels, and a PyTorch `Dataset` class for lazy data loading — before any modeling begins in Phase 4.

## 3.1 — Train/Validation Split
Combined location IDs from all 3 final training disasters (`hurricane-harvey`, `hurricane-michael`, `santa-rosa-wildfire`) and split 80/20 by **location ID**, not by individual file.

- **Why by ID, not by file:** each location's 6 related files (pre/post images, labels, targets) must stay together in the same split. Splitting file types independently risks a location's pre-image landing in training while its post-image lands in validation, breaking the pairing the model depends on.
- **Why only `labels/` was scanned:** Phase 1 already confirmed full parity (equal counts, no mismatches) across `images/`, `labels/`, and `targets/`. A location ID's existence in `labels/` guarantees its matching image and target files exist too.
- **Result:** 710 training locations, remaining ~20% held as `val_ids`, from a combined pool across all 3 disasters.

## 3.2 — Handling `Un-Classified` Labels
**Decision:** Exclude `un-classified` buildings entirely from classifier training, validation, and evaluation (Phase 5 onward).

- `un-classified` is not a genuine damage severity level — it's an annotator's uncertainty marker (Phase 2.5-2.6 found these buildings tend to be smaller and more visually ambiguous than properly labeled ones)
- Quantified the cost of exclusion: only 574 of 58,650 combined buildings (0.98%) — negligible lost training signal
- Rejected the alternative (treating `un-classified` as a genuine 5th class) as a scope expansion into a different problem (uncertainty/out-of-distribution detection), inconsistent with this project's defined 4-class goal

## 3.3 — `SegmentationDataset` (PyTorch `Dataset` Class)
Built a reusable PyTorch-style `Dataset` class (`src/segmentation_dataset.py`) supplying one (image, target) pair per training example, following the standard `__len__` / `__getitem__` contract for lazy, on-demand loading.

**Design decisions:**
- **Pre-disaster images only** — buildings are intact and cleanly shaped pre-disaster, giving the clearest signal for "what does a building look like," versus noisier post-disaster imagery (collapse, smoke, debris). Matches the standard xView2 challenge approach of training localization separately from damage classification.
- **Separate from classification** — this class only needs "where are the buildings," not damage information. A separate `ClassificationDataset` will be built in Phase 5 for building-crop + damage-label examples.
- **`un-classified` exclusion (3.2) does not apply here** — target masks are purely binary (building present or not), confirmed to contain no damage-class information at all.
- **Path resolution anchored to the file's own location** (`Path(__file__).resolve().parent.parent`), not the caller's working directory, so the class remains correctly importable from any future location (e.g. Phase 10's FastAPI app).

**Verification:**
- `len(train_dataset)` correctly returned 710, matching `train_ids`
- `train_dataset[0]` returned a 1024×1024 RGB pre-disaster image paired with a 1024×1024 single-channel (`"L"` mode) target mask
- Target mask confirmed binary: unique pixel values `[0, 1]` — building present or not, no damage information

## Remaining Phase 3 Tasks
- **3.4** — Input size decision given hardware constraints (downsizing 1024×1024 for CPU-feasible training)
- **3.5** — Augmentation pipeline design (fit on training data only, to avoid leakage)
- **3.6** — Handling building density variation (informed by Phase 2.7's finding of highly variable buildings-per-image)
- **3.7** — Visual verification of a processed batch (images + augmentations + labels correctly aligned)

*This document will be updated as Phase 3 continues.*

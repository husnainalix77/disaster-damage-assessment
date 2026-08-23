# Data Verification Log — Phase 1

## Source
- Dataset: xBD / xView2 Challenge dataset
- Downloaded: Challenge training set (7.8 GB) + Challenge test set (2.6 GB)
- Integrity: both archives SHA1-verified against official checksums before extraction — confirmed match

## Folder Structure Confirmed
Both `data/raw/train` and `data/raw/test` contain three subfolders: `images/` (pre/post PNG pairs), `labels/` (JSON polygon + damage annotations), `targets/` (rasterized damage mask PNGs). Filenames follow the pattern `<disaster-name>_<id>_<pre|post>_disaster.<ext>`.

## Count Verification (via src/verify_dataset.py)

### Training set — image pre/post pair counts, no mismatches
| Disaster | Pairs |
|---|---|
| socal-fire | 823 |
| hurricane-michael | 343 |
| hurricane-florence | 319 |
| hurricane-harvey | 319 |
| midwest-flooding | 279 |
| hurricane-matthew | 238 |
| santa-rosa-wildfire | 226 |
| mexico-earthquake | 121 |
| palu-tsunami | 113 |
| guatemala-volcano | 18 |
| **Total** | **2,799 pairs** |

### Images / Labels / Targets parity — train and test
Confirmed full parity (images = labels = targets count) across all 10 disaster types, in both the training set and the test set. No mismatches found in either split.

## Disaster Type Selection

**Final selection:**
- **Training disasters**: `hurricane-harvey` (319 pairs, flood/wind damage) + `hurricane-michael` (343 pairs, wind/structural damage) — selected for being visually/mechanically distinct damage types
- **Held-out generalization test (Phase 7)**: `mexico-earthquake` (121 pairs, structural collapse) — mechanically distinct from both training disasters, making it a genuine test of generalization rather than a soft one
- **Excluded**: `guatemala-volcano` (only 18 pairs, too small to be useful)

**Revision history — why `socal-fire` was dropped:**
`socal-fire` (823 pairs) was the original first-choice training disaster based on pair count and visual/mechanical distinctness from `hurricane-harvey`. During visual label-overlay verification, however, sampling ~15 `socal-fire` images showed buildings predominantly labeled `un-classified` rather than one of the 4 target damage classes (No Damage/Minor/Major/Destroyed). This appears to be a fire-damage-specific annotation characteristic — char/smoke damage is plausibly harder to visually classify into discrete damage levels than structural collapse or water damage, which may be why annotators marked many fire-damaged buildings as `un-classified` in the source dataset. Since `un-classified` labels don't map to any of this project's target classes, `socal-fire` would have provided limited useful training signal for the damage classifier.

`hurricane-michael` was evaluated as a replacement using the same visual sampling method and confirmed to show a healthy mix across all 4 target damage classes (no-damage, minor-damage, major-damage, destroyed) across multiple sampled images — selected as the final second training disaster.

## Visual Label-Overlay Verification
Polygon labels (from the `xy` key in each JSON, pixel coordinates) were plotted directly on top of their matching post-disaster images using shapely + matplotlib, color-coded by damage class.

- **hurricane-harvey**: outlines precisely traced real building shapes; damage classes (no-damage/minor/major) visually consistent with flood damage visible in imagery
- **hurricane-michael**: outlines correctly aligned; sampled images showed a healthy spread across no-damage, minor-damage, major-damage, and destroyed classes
- **mexico-earthquake**: outlines correctly aligned; sampled buildings showed no-damage in the checked images (expected variation, not an issue)
- **socal-fire** *(evaluated, not selected)*: outlines correctly aligned, but predominantly `un-classified` damage labels — see revision history above

## Conclusion
Dataset integrity, structure, and label alignment confirmed across both training and test sets. No blocking issues found. Disaster type selection was revised once during Phase 1 based on real visual evidence (not assumption), resulting in a more useful training set for the damage classifier. Phase 1 is complete.

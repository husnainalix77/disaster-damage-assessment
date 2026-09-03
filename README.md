<div align="center">

# 🛰️ Disaster Damage Assessment — Post-Disaster Building Damage Detection from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Build-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%204%20Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-xBD%20%2F%20xView2-blueviolet)

*A deep learning pipeline that compares pre- and post-disaster satellite images to automatically detect and classify building damage — the analysis step that currently takes human responders 24–48 hours to do by hand.*

</div>

---

## 📖 Table of Contents
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Progress](#-project-progress)
- [Dataset & Training Set Rationale](#-dataset--training-set-rationale)
- [Repository Structure](#-repository-structure)
- [Problems Faced & How They Were Solved](#-problems-faced--how-they-were-solved)
- [Key Engineering Decisions](#-key-engineering-decisions)
- [How to Run](#-how-to-run)
- [About the Author](#-about-the-author)

---

## 🌍 Problem Statement

When a major disaster strikes — an earthquake, flood, hurricane, or wildfire — emergency response agencies need to know **immediately** which buildings are destroyed, damaged, or safe, to decide where rescue teams and aid go first.

Today, this is done through **in-person, manual assessment**, which is why current response strategies still require in-person damage assessments within 24–48 hours of a disaster — a delay that costs lives.

Satellites already capture before/after imagery of disaster zones quickly. **The bottleneck isn't the photo — it's the interpretation.** A human analyst manually compares pre/post images and marks damage, building by building. This project automates that interpretation step.

**What this project builds:** a pipeline that takes a pre-disaster and post-disaster satellite image pair, locates every building, and classifies its damage level — `No Damage → Minor → Major → Destroyed` — producing a structured, prioritizable damage report.

> 📌 **Honest scope note:** this is a solo, single-laptop project (8GB RAM, no dedicated GPU) built on a small, deliberately chosen disaster-type subset of the real xBD/xView2 benchmark dataset — a proof of concept demonstrating industry-standard *method*, not industry-scale *infrastructure*. See [Key Engineering Decisions](#-key-engineering-decisions) for the full reasoning.

---

## 🏗️ System Architecture

```
   ┌─────────────────────┐        ┌─────────────────────┐
   │  Pre-Disaster Image  │        │ Post-Disaster Image │
   └──────────┬───────────┘        └──────────┬───────────┘
              │                               │
              └───────────────┬───────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │   Preprocessing &        │
                  │   Augmentation Pipeline  │
                  └────────────┬─────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │  Building Segmentation   │
                  │      (U-Net CNN)         │
                  └────────────┬─────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │  Damage Classification   │
                  │ (Transfer-Learned CNN)   │
                  │  No Dmg / Minor / Major  │
                  │      / Destroyed         │
                  └────────────┬─────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │  FastAPI Inference API   │
                  │      (Dockerized)        │
                  └────────────┬─────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │   Streamlit Dashboard    │
                  │   (Interactive Demo)     │
                  └─────────────────────────┘
```

*This is the planned end-to-end architecture per the project brief. Components are implemented and verified progressively, phase by phase — see [Project Progress](#-project-progress).*

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **Deep Learning** | PyTorch, torchvision |
| **Data Handling** | pandas, numpy, shapely |
| **Statistical Validation** | scipy — chi-square test of independence, KS-test |
| **Visualization** | matplotlib, seaborn |
| **Classical ML** | scikit-learn |
| **Deployment** | FastAPI, Docker |
| **Experiment Tracking** | MLflow |
| **Demo/UI** | Streamlit |
| **Dataset** | [xBD / xView2](https://xview2.org/dataset) — CMU SEI & U.S. Defense Innovation Unit |
| **Compute** | Local (Dell Latitude 5590, i5, 8GB RAM, CPU-only) for development/verification + free-tier Google Colab / Kaggle GPU (Tesla T4) for actual training runs |

---

## 📊 Project Progress

| Phase | Title | Status |
|---|---|---|
| 0 | Environment, Repo Structure & Scoping Decisions | ✅ Complete |
| 1 | Dataset Acquisition & Verification | ✅ Complete |
| 2 | Exploratory Data Analysis | ✅ Complete |
| 3 | Preprocessing & Augmentation Pipeline | ✅ Complete |
| 4 | Building Localization / Segmentation Model | ✅ Complete (final model selection deferred to Phase 6) |
| 5 | Damage Classification (Transfer Learning) | ⬜ Not Started |
| 6 | Evaluation — Segmentation & Classification Metrics | ⬜ Not Started |
| 7 | Held-Out Disaster-Type Generalization Test | ⬜ Not Started |
| 8 | Benchmark Comparison Against Published Results | ⬜ Not Started |
| 9 | Experiment Tracking & Explainability | ⬜ Not Started |
| 10 | Deployment — FastAPI + Docker | ⬜ Not Started |
| 11 | Streamlit Demo Dashboard | ⬜ Not Started |
| 12 | Documentation & README | 🔄 Ongoing (this file) |

### ✅ Phase 0-3
Environment/repo setup, dataset verification (2,799 pairs, zero mismatches), evidence-based EDA and training-disaster selection, and a verified preprocessing pipeline (`SegmentationDataset`, 80/20 split, training-only augmentation). Full details in `docs/scope_and_assumptions.md`, `docs/phase1_dataset_verification.md`, `docs/phase2_eda_summary.md`, `docs/phase3_preprocessing_summary.md`.

### ✅ Phase 4 — Building Localization / Segmentation Model

**4.1-4.2 — CNN/U-Net fundamentals and architecture.** U-Net built and verified in `src/unet.py` (`EncoderBlock`, `Bottleneck`, `DecoderBlock`, `UNet`), parameterized by `base_channels` for multi-architecture comparison. Verified end-to-end: 512×512×3 input → 512×512×1 output.

**4.3 — Loss function, optimizer, training loop.** `BCEWithLogitsLoss` + `Adam` (lr=1e-4). Bug found and fixed: `SegmentationDataset` returned PIL Images instead of tensors, breaking `DataLoader` batching — fixed with `torchvision.transforms.ToTensor()`. Local CPU speed measured at ~2 min/batch (~3 hr/epoch projected) — confirmed training must move to GPU.

**4.4 — GPU training (Colab).** A serious debugging arc:
- First GPU training attempt showed a suspiciously fast loss collapse and, visually, the model predicted pure black (no buildings) despite a building-dense input.
- **Hypothesis 1 (pixel class imbalance, 90.77% background):** `pos_weight=9.84` applied — model became *more* confident in all-background over 8 epochs. Rejected by evidence.
- **Hypothesis 2 (BCE unsuited to severe imbalance):** Dice Loss added — same declining trend. Rejected by evidence.
- **Root cause found:** `ToTensor()` was silently dividing the mask's clean `0`/`1` values by 255 (correct for photos, wrong for label data), corrupting the ground truth itself.
- **Fixed** with a direct, non-rescaling tensor conversion for the mask. A fresh 20-epoch run (BCE + Dice, no `pos_weight`) showed smooth, consistent loss decline (1.28 → 0.56) and genuine, if imprecise, building-shaped predictions.
- Measured GPU speedup: ~2.3 min/epoch vs. ~3 hr/epoch projected on CPU — roughly a **230x speedup**.

**4.5 — Multi-architecture comparison (Kaggle, after Colab's free GPU quota was exhausted).** Trained `UNet(base_channels=32)` under identical conditions as the `base_channels=64` baseline.
- `base_channels=64`: final loss 0.5625, ~46 min training time
- `base_channels=32`: final loss 0.6862, ~34 min training time (~25% faster)
- Both candidates' weights saved (`models/unet_baseline_20epochs.pt`, `models/unet_light_20epochs.pt`); a re-uploaded, pre-fix copy of `segmentation_dataset.py` was caught and corrected via a data-integrity check before this run.

**4.6 — Training loss monitoring.** Neither model had plateaued by epoch 20 — both still improving meaningfully. `base_channels=32` declined faster early but slowed more in the second half; `base_channels=64` sustained a steadier improvement rate and pulled ahead after epoch ~9. **Honest limitation:** validation loss was not tracked per epoch in these runs, so a proper overfitting check isn't yet possible.

**4.7 — Final model selection: explicitly deferred to Phase 6**, not left unresolved by default. Training-loss evidence alone, from two non-converged models and a single validation image, was judged insufficient for a confident final choice — Phase 6's planned IoU/Dice evaluation and paired significance test is the appropriate basis for this decision.

**4.8 — Model weight saving: partially complete.** Both candidates' weights are already saved and loadable. Finalizing which is "the" model (e.g. renaming to `unet_final.pt`) is deferred alongside 4.7, pending Phase 6's decision — the non-selected candidate will be retained, not discarded, since the comparison itself is a documented part of this project's evaluation process.

**4.9 (visual sanity check) was intentionally not run as a separate step** — its substance was already covered informally during the 4.4-4.5 debugging process (repeated input/ground-truth/prediction visualizations), and a final, formal version on the selected model fits more naturally as part of Phase 6's evaluation write-up than as a repeated Phase 4 step.

---

## 🗂️ Dataset & Training Set Rationale

**Final training set:** `hurricane-harvey` (319 pairs) + `hurricane-michael` (343 pairs) + `santa-rosa-wildfire` (226 pairs)
**Held-out generalization disaster (Phase 7):** `mexico-earthquake` (121 pairs)

| Step | Finding |
|---|---|
| Initial 2-disaster set (`harvey` + `michael`) | Only 2.5% combined `destroyed`-class representation |
| Full 10-disaster count analysis | Identified `santa-rosa-wildfire` (26.8% destroyed) as the best 3rd training disaster |
| `hurricane-matthew`, `palu-tsunami` | Evaluated, excluded — redundant damage mechanism / would contaminate the held-out test |
| `socal-fire` | Re-evaluated with full counts (3.6% un-classified, not the majority a small sample suggested); still not selected |
| Statistical confirmation | Chi-square test confirms damage-class distributions differ significantly across the 3 training disasters (χ² = 19218.04, p < 0.0001) |
| Building size, by disaster & by damage class | Consistent across disasters and classes, except `un-classified` buildings (significantly smaller, KS statistic = 0.3916) |
| Image resolution | Uniformly 1024×1024 across all training data, pre and post |
| Building density per image | Right-skewed across all 3 disasters; `hurricane-michael` shows a distinct, more evenly-spread pattern |
| `un-classified` labels | 0.98% of combined training set — excluded from classifier training |

---

## 📁 Repository Structure

```
disaster-damage-assessment/
├── app/                          # FastAPI + Streamlit application code (Phase 10-11)
├── data/
│   ├── raw/                      # Downloaded xBD imagery (gitignored)
│   └── processed/                # train_ids.txt, val_ids.txt, loss_history*.json (gitignored)
├── docs/
│   ├── scope_and_assumptions.md
│   ├── phase1_dataset_verification.md
│   ├── phase2_eda_summary.md
│   ├── phase3_preprocessing_summary.md
│   └── phase4_segmentation_summary.md   # Phase 4 model architecture, training, comparison
├── models/                       # unet_baseline_20epochs.pt, unet_light_20epochs.pt (gitignored)
├── notebooks/
│   ├── 01_dataset_verification.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_segmentation.ipynb
├── src/
│   ├── verify_dataset.py
│   ├── segmentation_dataset.py
│   └── unet.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🐛 Problems Faced & How They Were Solved

**1-9.** Windows path-length limits, `.gitignore` gaps, `shasum` unavailable on Windows, a silent validation bug, an overstated `socal-fire` visual sample, `ModuleNotFoundError` from `src/` imports, fragile relative paths, a `DataLoader` batching failure (fixed with `ToTensor()`), and infeasible local CPU training speed — all detailed in earlier phase docs.

**10. A critical, multi-step data corruption bug in the segmentation mask pipeline.** The model appeared to be learning (loss dropping fast) but was actually predicting pure background everywhere. Two principled, evidence-based fixes (`pos_weight` for class imbalance, then Dice Loss) both failed — the repeated failure of two independent fixes was the signal to look upstream. Root cause: `ToTensor()`'s automatic 0-255→0-1 rescaling, applied uniformly to both image and mask, silently corrupted the mask's clean `0`/`1` labels into `0`/`0.0039`. Fixed with a direct, non-rescaling tensor conversion for the mask only.

**11. Colab free-tier GPU quota exhausted mid-project.** Training moved to Kaggle Notebooks (separate free GPU quota) for Phase 4.5 — a pre-fix copy of `segmentation_dataset.py` was initially re-uploaded there and caught via a data-integrity check before retraining, preventing a repeat of problem #10 on the new platform.

---

## 🎯 Key Engineering Decisions

**Q: Why not use the full xBD dataset?**
A: Impractical on an 8GB RAM, GPU-less laptop within a reasonable iteration loop. A focused subset allows honest, fast iteration.

**Q: Why train on Colab/Kaggle for some phases instead of fully locally?**
A: Confirmed with a live measurement (~2 min/batch locally, ~3 hr/epoch projected) vs. ~2.3 min/epoch on a free Tesla T4 GPU — roughly a 230x speedup. When Colab's free-tier quota was exhausted mid-project, training moved to Kaggle's separate free GPU quota rather than waiting or reverting to CPU.

**Q: Why 3 training disasters instead of 2? Why these specific ones?**
A: See [Dataset & Training Set Rationale](#-dataset--training-set-rationale) above.

**Q: Why does the segmentation model use pre-disaster images only?**
A: Segmentation's only job is locating buildings — pre-disaster images show buildings intact and cleanly shaped, giving the clearest training signal. The pre/post *comparison* is deliberately deferred to Phase 5.

**Q: Why combine BCE with Dice Loss?**
A: Standard practice for imbalanced segmentation — BCE gives stable per-pixel gradients, Dice directly penalizes poor shape overlap and is far less exploitable by an "always predict background" shortcut than BCE alone.

**Q: Why was `pos_weight` removed after fixing the mask bug, rather than kept alongside it?**
A: The severe class imbalance that originally justified `pos_weight=9.84` was partly an artifact of debugging a broken pipeline. Once the real bug was fixed, re-testing without `pos_weight` showed healthy, smooth convergence — confirming the extra weighting was unnecessary and could have caused a different overcorrection if kept.

**Q: Why defer final model selection (4.7) and weight finalization (4.8) to Phase 6, rather than deciding now?**
A: Neither candidate model had plateaued by epoch 20, and only training-loss and single-image visual evidence were available. Committing to a final architecture on this basis would be less rigorous than the standard applied throughout this project. Phase 6's planned IoU/Dice evaluation on the full validation set, plus a paired statistical significance test, is the appropriate and intended basis for this decision — both candidates' weights are already saved and ready for that evaluation.

**Q: Why wasn't a formal Phase 4.9 visual check performed?**
A: Its substance was already covered repeatedly and informally during the 4.4-4.5 debugging process. A final, formal visual check on the selected model fits more naturally as part of Phase 6's evaluation write-up, once a winner is actually chosen, than as a separate, repeated Phase 4 step.

**Q: Why is the U-Net architecture parameterized by `base_channels` instead of hardcoded?**
A: Enables creating a lighter comparison variant without duplicating the class — used directly in Phase 4.5.

**Q: Why MIT license, and why isn't the dataset (or a personal cloud copy of it) linked anywhere in this repo?**
A: MIT covers the project's own code only. The xBD dataset is subject to xView2/DIU's own terms; even a private cloud storage link used for personal Colab/Kaggle access is deliberately never published in this repo, to avoid any appearance of unauthorized redistribution.

---

## ▶️ How to Run

> Phases 0-4 are complete (final model selection pending Phase 6) — full inference/demo instructions will be added as later phases are finished.

```bash
git clone https://github.com/husnainalix77/disaster-damage-assessment.git
cd disaster-damage-assessment

python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Dataset (not included in repo — download separately):
- [xView2 / xBD official dataset page](https://xview2.org/dataset) — download the **Challenge training set** and **Challenge test set**, verify SHA1 checksums, then extract into `data/raw/train/` and `data/raw/test/`

```bash
python src/verify_dataset.py
```

> **Note on Phase 4 training cells:** the training loop code in `notebooks/04_segmentation.ipynb` reflects the local project's file structure for reproducibility, but was actually executed on Google Colab / Kaggle (GPU) due to hardware constraints established in Phase 0 — running it locally as-is would work but take significantly longer (~3 hours/epoch on CPU vs. ~2.3 minutes/epoch on a free GPU).

---

## 👤 About the Author

<div align="center">

### Husnain Maroof

**Mechatronics & Control Engineering Student** · UET Lahore

Self-taught in Python and data science for 2+ years — building applied ML/DL projects end-to-end, from raw data through deployment, outside a formal data science curriculum. Currently deepening a background in imbalanced classification, statistical validation, and explainability (via the [transaction-fraud-risk-engine](https://github.com/husnainalix77/transaction-fraud-risk-engine) project) into deep learning and computer vision with this project.

Also researching SUPARCO-sponsored optical beacon tracking as part of a Final Year Project, and building toward a career in applied data science and machine learning.

[![GitHub](https://img.shields.io/badge/GitHub-husnainalix77-181717?logo=github&logoColor=white)](https://github.com/husnainalix77)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/)

</div>

<div align="center">

# 🛰️ Disaster Damage Assessment — Post-Disaster Building Damage Detection from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Build-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%205%20Complete-brightgreen)
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
                  │  FastAPI Inference API  │
                  │      (Dockerized)       │
                  └────────────┬────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │   Streamlit Dashboard   │
                  │   (Interactive Demo)    │
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
| 5 | Damage Classification (Transfer Learning) | ✅ Complete (final model selection deferred to Phase 6) |
| 6 | Evaluation — Segmentation & Classification Metrics | ✅ Complete |
| 7 | Held-Out Disaster-Type Generalization Test | ✅ Complete |
| 8 | Benchmark Comparison Against Published Results | ⬜ Not Started |
| 9 | Experiment Tracking & Explainability | ⬜ Not Started |
| 10 | Deployment — FastAPI + Docker | ⬜ Not Started |
| 11 | Streamlit Demo Dashboard | ⬜ Not Started |
| 12 | Documentation & README | 🔄 Ongoing (this file) |

### ✅ Phase 0 — Environment, Repo Structure & Scoping Decisions
Set up the repo, Python environment, and PyTorch (CPU build), and documented the project's hardware constraints and honest scope upfront — before any modeling began.
📓 *(no notebook — environment setup)* · 📄 [scope_and_assumptions.md](docs/scope_and_assumptions.md)

### ✅ Phase 1 — Dataset Acquisition & Verification
Downloaded and SHA1-verified the xBD dataset, confirmed folder structure and full count parity across all 10 disaster types (2,799 pairs, zero mismatches), and visually verified label alignment.
📓 [01_dataset_verification.ipynb](notebooks/01_dataset_verification.ipynb) · 📄 [phase1_dataset_verification.md](docs/phase1_dataset_verification.md)

### ✅ Phase 2 — Exploratory Data Analysis
Made an evidence-based, full-count comparison across all 10 disaster types to select the final 3-disaster training set, characterized building size/resolution/density, and applied statistical validation (chi-square, KS-test) throughout.
📓 [02_eda.ipynb](notebooks/02_eda.ipynb) · 📄 [phase2_eda_summary.md](docs/phase2_eda_summary.md)

### ✅ Phase 3 — Preprocessing & Augmentation Pipeline
Built a leakage-free 80/20 train/validation split by location ID, decided how to handle ambiguous `un-classified` labels, and built/verified the `SegmentationDataset` pipeline with training-only augmentation.
📓 [03_preprocessing.ipynb](notebooks/03_preprocessing.ipynb) · 📄 [phase3_preprocessing_summary.md](docs/phase3_preprocessing_summary.md)

### ✅ Phase 4 — Building Localization / Segmentation Model
Built and trained a U-Net segmentation model from scratch on GPU (Colab/Kaggle), diagnosing and fixing a serious data-corruption bug along the way (a mask-scaling error that caused the model to collapse to all-background predictions), then trained and compared two architecture variants. Final model selection is explicitly deferred to Phase 6, pending formal IoU/Dice evaluation.
📓 [04_segmentation.ipynb](notebooks/04_segmentation.ipynb) · 📄 [phase4_segmentation_summary.md](docs/phase4_segmentation_summary.md)

### ✅ Phase 5 — Damage Classification (Transfer Learning)
Built a building-crop classifier using transfer learning (frozen pretrained backbones with a new final layer), diagnosed and fixed a class-imbalance shortcut using weighted loss, built a precomputed-crop caching pipeline for a ~7-8x training speedup, and trained three candidate backbones (ResNet-50, EfficientNet-B0, MobileNet-V2). Final model selection is explicitly deferred to Phase 6, since the comparison is confounded by an augmentation difference between the baseline and comparison models.
📓 [05_classification.ipynb](notebooks/05_classification.ipynb) · 📄 [phase5_classification_summary.md](docs/phase5_classification_summary.md)

### ✅ Phase 6 — Evaluation, Model Selection & Final Test-Set Results
Resolved both deferred model-selection decisions (Phase 4.7, Phase 5.7) using full-validation-set IoU/Dice, precision/recall/F1, confusion matrices, and statistical significance testing. Retrained and adopted an improved segmentation model; retrained and evaluated (but did not adopt) an augmented classifier, since it showed no clear improvement. Final models evaluated exactly once on the untouched test set: segmentation IoU 0.5136 / Dice 0.6378; classification macro F1 0.53.
📓 [06_evaluation.ipynb](notebooks/06_evaluation.ipynb) · 📄 [phase6_evaluation_summary.md](docs/phase6_evaluation_summary.md)

### ✅ Phase 7 — Held-Out Disaster-Type Generalization Test
Evaluated both final models on `mexico-earthquake`, reserved untouched since Phase 2 specifically for this purpose. The segmentation model degrades moderately on unseen earthquake imagery (IoU 0.51 → 0.37, Dice 0.64 → 0.53) but remains functional. The classifier fails severely on 3 of 4 damage classes (macro F1 0.53 → 0.16), while retaining moderate performance on the dominant `no-damage` class — an honest, directly-measured limitation, consistent with the project's stated scope since Phase 0.
📓 [07_generalization_test.ipynb](notebooks/07_generalization_test.ipynb) · 📄 [phase7_generalization_summary.md](docs/phase7_generalization_summary.md)

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
│   └── processed/                # IDs, loss/accuracy histories, precomputed crops (gitignored)
├── docs/
│   ├── scope_and_assumptions.md
│   ├── phase1_dataset_verification.md
│   ├── phase2_eda_summary.md
│   ├── phase3_preprocessing_summary.md
│   ├── phase4_segmentation_summary.md
│   └── phase5_classification_summary.md
├── models/                       # Trained model weights (gitignored)
├── notebooks/
│   ├── 01_dataset_verification.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_segmentation.ipynb
│   └── 05_classification.ipynb
├── src/
│   ├── verify_dataset.py
│   ├── segmentation_dataset.py
│   ├── unet.py
│   ├── classification_dataset.py
│   ├── precomputed_classification_dataset.py
│   └── precomputed_training_dataset.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🐛 Problems Faced & How They Were Solved

**1-9 (Phases 0-4).** Windows path-length limits, `.gitignore` gaps, `shasum` unavailable on Windows, a silent validation bug, an overstated `socal-fire` visual sample, `ModuleNotFoundError` from `src/` imports, fragile relative paths, a `DataLoader` batching failure, and infeasible local CPU training speed — all detailed in `docs/phase1_dataset_verification.md` through `docs/phase4_segmentation_summary.md`.

**10. A critical, multi-step data corruption bug in the segmentation mask pipeline (Phase 4.4).** The model appeared to be learning but was actually predicting pure background everywhere. Two principled fixes (`pos_weight`, then Dice Loss) both failed by evidence before the real cause — `ToTensor()` silently rescaling the mask's `0`/`1` labels by dividing by 255 — was found and fixed.

**11. Colab free-tier GPU quota exhausted mid-project (Phase 4.5).** Training moved to Kaggle Notebooks — a pre-fix copy of `segmentation_dataset.py` was caught and corrected there before retraining.

**12. A class-imbalance shortcut in the damage classifier (Phase 5.4).** An unweighted classifier scored a deceptively reasonable 64.39% overall accuracy after 1 epoch, but per-class accuracy revealed it had learned to almost always predict the majority class (`no-damage`: 96.83%, `minor-damage`: 4.23%). Fixed with inverse-frequency class weighting in `CrossEntropyLoss`.

**13. Redundant, repeated crop computation slowed classifier training (Phase 5.4-5.6).** `ClassificationDataset` recomputed the same crop/resize operation for the same building on every epoch, despite the result never changing. Fixed with a one-time precomputed-crop caching pipeline, reducing per-epoch training time by roughly 7-8x for the Phase 5.6 comparison models.

**14. A precomputation class silently discarded labels.** An early version of `PrecomputedClassificationDataset` saved cropped images but never recorded their damage labels anywhere, making the cache unusable for training. Fixed by building and saving an explicit `manifest.json`.

**15. A hardcoded output folder risked silently overwriting train/validation caches.** `PrecomputedClassificationDataset` originally used one fixed `PROCESSED_DIR`, meaning a validation precompute run would silently overwrite the training run's manifest and crops. Fixed by making the output directory a required constructor parameter, so train and validation caches are kept in separate, explicit locations.

---

## 🎯 Key Engineering Decisions

**Q: Why not use the full xBD dataset?**
A: Impractical on an 8GB RAM, GPU-less laptop within a reasonable iteration loop. A focused subset allows honest, fast iteration.

**Q: Why train on Colab/Kaggle instead of fully locally?**
A: Confirmed with live measurements — segmentation projected ~3 hr/epoch locally vs. ~2.3 min/epoch on a free GPU (~230x speedup); classifier training showed a similar CPU/GPU gap. When Colab's free-tier quota was exhausted mid-project, training moved to Kaggle's separate free GPU quota.

**Q: Why 3 training disasters instead of 2? Why these specific ones?**
A: See [Dataset & Training Set Rationale](#-dataset--training-set-rationale) above.

**Q: Why does the segmentation model use pre-disaster images, but the classifier use post-disaster images?**
A: Segmentation's only job is locating buildings — pre-disaster imagery shows them intact and cleanly shaped. Classification's job is judging damage, which is only visible in post-disaster imagery. The pre/post *comparison* itself is a natural extension for future work, using each model for its respective strength.

**Q: Why transfer learning (frozen pretrained backbones) for the classifier, rather than training a CNN from scratch as in Phase 4?**
A: The classifier's dataset (~46,000 building crops) is far smaller than what's needed to learn general visual features from scratch. Reusing ImageNet-pretrained features and only training a new final layer is standard practice for small-data image tasks, and dramatically reduces both overfitting risk and required compute.

**Q: Why defer final model selection for both the segmentation model (Phase 4.7) and the classifier (Phase 5.7) to Phase 6?**
A: In both cases, the available comparison evidence was judged insufficient for a confident, defensible choice — non-converged segmentation models with only training-loss evidence, and classifier candidates trained under inconsistent conditions (an augmentation difference between the baseline and comparison models). Phase 6's planned formal evaluation (IoU/Dice, per-class F1, confusion matrices, and a paired statistical significance test) is the appropriate and intended basis for both decisions.

**Q: Why MIT license, and why isn't the dataset (or a personal cloud copy of it) linked anywhere in this repo?**
A: MIT covers the project's own code only. The xBD dataset is subject to xView2/DIU's own terms; even a private cloud storage link used for personal Colab/Kaggle access is deliberately never published in this repo, to avoid any appearance of unauthorized redistribution.

---

## ▶️ How to Run

> Phases 0-5 are complete (final model selections pending Phase 6) — full inference/demo instructions will be added as later phases are finished.

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

> **Note on GPU training cells:** the training loop code in `notebooks/04_segmentation.ipynb` and `notebooks/05_classification.ipynb` reflects the local project's file structure for reproducibility, but was actually executed on Google Colab / Kaggle (GPU) due to hardware constraints established in Phase 0.

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

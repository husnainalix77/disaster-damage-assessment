<div align="center">

# 🛰️ Disaster Damage Assessment — Post-Disaster Building Damage Detection from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Build-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%204%20In%20Progress-yellow)
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
| **Deep Learning** | PyTorch (CPU build locally; GPU via Colab/Kaggle for training), torchvision |
| **Data Handling** | pandas, numpy, shapely |
| **Statistical Validation** | scipy — chi-square test of independence, KS-test |
| **Visualization** | matplotlib, seaborn |
| **Classical ML** | scikit-learn |
| **Deployment** | FastAPI, Docker |
| **Experiment Tracking** | MLflow |
| **Demo/UI** | Streamlit |
| **Dataset** | [xBD / xView2](https://xview2.org/dataset) — CMU SEI & U.S. Defense Innovation Unit |
| **Compute** | Local (Dell Latitude 5590, i5, 8GB RAM, CPU-only) for development/verification + free-tier Google Colab / Kaggle GPU for actual training runs |

---

## 📊 Project Progress

| Phase | Title | Status |
|---|---|---|
| 0 | Environment, Repo Structure & Scoping Decisions | ✅ Complete |
| 1 | Dataset Acquisition & Verification | ✅ Complete |
| 2 | Exploratory Data Analysis | ✅ Complete |
| 3 | Preprocessing & Augmentation Pipeline | ✅ Complete |
| 4 | Building Localization / Segmentation Model | 🔄 In Progress |
| 5 | Damage Classification (Transfer Learning) | ⬜ Not Started |
| 6 | Evaluation — Segmentation & Classification Metrics | ⬜ Not Started |
| 7 | Held-Out Disaster-Type Generalization Test | ⬜ Not Started |
| 8 | Benchmark Comparison Against Published Results | ⬜ Not Started |
| 9 | Experiment Tracking & Explainability | ⬜ Not Started |
| 10 | Deployment — FastAPI + Docker | ⬜ Not Started |
| 11 | Streamlit Demo Dashboard | ⬜ Not Started |
| 12 | Documentation & README | 🔄 Ongoing (this file) |

### ✅ Phase 0 — Environment, Repo Structure & Scoping Decisions
Repo, venv, PyTorch (CPU build) and full stack installed and verified; scope/assumptions documented upfront; dataset downloaded and SHA1-verified; `.gitignore` extended for large files.

### ✅ Phase 1 — Dataset Acquisition & Verification
Folder structure, count parity (2,799 pairs, 10 disaster types, zero mismatches), and visual label-overlay alignment all confirmed. Full findings in `docs/phase1_dataset_verification.md`.

### ✅ Phase 2 — Exploratory Data Analysis
Full-count, evidence-based training-disaster selection; building size, resolution, and density characterization; two statistical validation techniques (chi-square, KS-test) applied and correctly interpreted. See [Dataset & Training Set Rationale](#-dataset--training-set-rationale) and `docs/phase2_eda_summary.md`.

### ✅ Phase 3 — Preprocessing & Augmentation Pipeline
80/20 train/validation split by location ID (710 training locations), saved to disk (`train_ids.txt`/`val_ids.txt`) for reuse across all later phases; `un-classified` labels excluded from classifier training (0.98% of data); `SegmentationDataset` PyTorch class built and verified, supplying resized (512×512) pre-disaster image + binary building-mask pairs with training-only augmentation. Full details in `docs/phase3_preprocessing_summary.md`.

### 🔄 Phase 4 — Building Localization / Segmentation Model (in progress)

**4.1 — CNN & segmentation fundamentals (concept only).** Convolution (learned, reusable pattern-detectors), pooling (spatial compression preserving strong signals), encoder-decoder structure, and U-Net's defining feature — skip connections passing detail-rich encoder feature maps directly to matching decoder stages.

**4.2 — U-Net architecture built and verified in PyTorch (`src/unet.py`).** Built incrementally — `EncoderBlock`, `Bottleneck`, `DecoderBlock`, assembled into `UNet` — each verified against hand-calculated expected shapes on a dummy input before proceeding. Parameterized by `base_channels` (default 64) to support a lighter comparison variant in Phase 4.5 without duplicating the class. Verified end-to-end: 512×512×3 input correctly produces 512×512×1 output.

**4.3 — Loss function, optimizer, and training loop.** `BCEWithLogitsLoss` (matches the model's raw-logit output and the binary per-pixel target) and `Adam` (lr=1e-4) set up; training loop built (zero gradients → forward pass → compute loss → backpropagate → optimizer step) and validated on live data.

- **Bug found and fixed:** `SegmentationDataset.__getitem__` was returning PIL Image objects rather than tensors — unnoticed in Phase 3.3's single-item testing, since `DataLoader`'s batching only breaks when combining *multiple* examples. Fixed by adding `torchvision.transforms.ToTensor()` conversion after resizing/augmentation.
- **Verified healthy starting behavior:** first-batch loss (0.7508) close to the theoretical random-guessing baseline for BCE (≈0.693) — confirming the pipeline is correctly wired, not just error-free.
- **Critical hardware finding:** measured ~2 minutes per batch on local CPU, projecting ~3 hours per epoch and ~90 hours for a full ~30-epoch training run — confirming the Phase 0 scope decision that real training must move to free-tier GPU (Colab/Kaggle), with local CPU reserved for pipeline development and verification only.

**Remaining:** 4.4 (Colab/Kaggle GPU training setup and execution), 4.5 (multi-architecture comparison), 4.6-4.9 (monitoring, model selection, saving, visual verification).

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
│   └── processed/                # train_ids.txt / val_ids.txt (gitignored)
├── docs/
│   ├── scope_and_assumptions.md      # Honest project scope, written in Phase 0
│   ├── phase1_dataset_verification.md # Phase 1 findings and disaster selection reasoning
│   ├── phase2_eda_summary.md         # Phase 2 EDA findings and statistical validation
│   ├── phase3_preprocessing_summary.md # Phase 3 preprocessing decisions
│   └── phase4_segmentation_summary.md # Phase 4 model architecture and training (in progress)
├── models/                       # Trained model weights (gitignored)
├── notebooks/
│   ├── 01_dataset_verification.ipynb # Phase 1 verification + visual label-overlay checks
│   ├── 02_eda.ipynb                  # Phase 2 EDA — full analysis, sections 2.1-2.8
│   ├── 03_preprocessing.ipynb        # Phase 3 preprocessing pipeline
│   └── 04_segmentation.ipynb         # Phase 4 U-Net segmentation model (in progress)
├── src/
│   ├── verify_dataset.py         # Reusable dataset count/parity verification utility
│   ├── segmentation_dataset.py   # PyTorch Dataset class for segmentation
│   └── unet.py                   # U-Net architecture (EncoderBlock, Bottleneck, DecoderBlock, UNet)
├── .gitignore
├── LICENSE                       # MIT (project code only — not the dataset)
├── README.md
└── requirements.txt
```

---

## 🐛 Problems Faced & How They Were Solved

**1. Windows path-length limit broke the PyTorch install.**
Fixed by enabling Windows long-path support and Git's long-path config, then reinstalling cleanly.

**2. `.gitignore` didn't cover project-specific large files.**
Fixed by manually extending it to exclude `data/raw/`, `data/processed/`, model weight files, and MLflow logs.

**3. SHA1 verification via `shasum` doesn't exist natively on Windows.**
Fixed using PowerShell's built-in `Get-FileHash -Algorithm SHA1`.

**4. A silent validation bug in the dataset-structure checker.**
`_validate_structure()` returned success after checking only the first folder due to an indentation error. Fixed by correcting the loop structure.

**5. A small visual sample overstated `socal-fire`'s data quality issue.**
An ~15-image visual check suggested `socal-fire` was predominantly `un-classified`; full-count analysis showed the true rate was only 3.6%. Lesson applied: visual samples flag possible issues, final decisions rest on full counts.

**6. `ModuleNotFoundError` importing from `src/` into a notebook.**
Fixed by explicitly appending the project root to `sys.path` at the top of the notebook before importing.

**7. Relative paths inside `src/` files were fragile.**
Fixed by anchoring paths to each file's own location (`Path(__file__).resolve().parent.parent`), which resolves correctly regardless of where the code is called from — e.g. `streamlit run app/app.py` sets the working directory to the project root, not `app/`, which would break a naive `../` path.

**8. `DataLoader` couldn't batch `SegmentationDataset`'s output.**
`__getitem__` returned raw PIL Image objects, which `DataLoader`'s batching logic can't stack into tensors. Unnoticed during Phase 3.3's single-item testing, since batching only occurs when combining *multiple* examples. Fixed by adding `torchvision.transforms.ToTensor()` conversion as the final step in `__getitem__`.

**9. Local CPU training speed made real experimentation infeasible.**
A live-timed test showed ~2 minutes per batch, projecting ~3 hours per epoch (~90 hours for a full training run). Rather than attempting to push through locally, training was stopped after confirming the pipeline works correctly end-to-end, and moved to free-tier Colab/Kaggle GPU — the exact contingency planned for in `docs/scope_and_assumptions.md`.

---

## 🎯 Key Engineering Decisions

**Q: Why not use the full xBD dataset?**
A: Impractical on an 8GB RAM, GPU-less laptop within a reasonable iteration loop. A focused subset allows honest, fast iteration.

**Q: Why train on Colab/Kaggle for some phases instead of fully locally?**
A: Confirmed with a live measurement (~2 min/batch locally) — a full epoch would take ~3 hours on CPU. Phases 0-3 and pipeline verification run locally; actual training runs (Phase 4.4 onward) use free-tier cloud GPU.

**Q: Why 3 training disasters instead of 2? Why these specific ones?**
A: See [Dataset & Training Set Rationale](#-dataset--training-set-rationale) above.

**Q: Why exclude `un-classified` labels rather than treating them as a 5th class?**
A: `un-classified` isn't a genuine damage severity level — it's an annotator's uncertainty marker. Treating it as a predictable class would expand the project into a different problem for negligible lost signal (under 1% of data).

**Q: Why resize to 512×512 rather than training at full 1024×1024 resolution?**
A: A quarter of the pixel count at 512×512 gives a real speed/memory improvement, while preserving enough detail for typical building sizes to remain clearly learnable — unlike a more aggressive downsize (256×256), which risks shrinking small buildings to just a few pixels.

**Q: Why does the segmentation model use pre-disaster images only?**
A: Segmentation's only job is locating buildings — pre-disaster images show buildings intact and cleanly shaped, giving the clearest training signal. The pre/post *comparison* is deliberately deferred to Phase 5.

**Q: Why is the U-Net architecture parameterized by `base_channels` instead of hardcoded?**
A: Enables creating a lighter comparison variant (e.g. `base_channels=32`) for Phase 4.5's planned multi-architecture comparison, by reusing the same class rather than duplicating it with different hardcoded values.

**Q: Why `BCEWithLogitsLoss` specifically, rather than plain `BCELoss`?**
A: The model's final layer outputs raw logits (unbounded values), not probabilities. `BCEWithLogitsLoss` applies sigmoid and computes cross-entropy in one numerically stable step, avoiding precision issues from doing sigmoid and BCE separately.

**Q: Why isn't a link to the training data provided anywhere in this repo?**
A: Per `docs/scope_and_assumptions.md`, the xBD dataset is not covered by this repo's MIT license and is subject to xView2/DIU's own terms. Even a personal cloud storage link would constitute unauthorized redistribution — the dataset is referenced only via the official xView2 download page.

**Q: Why MIT license, and why isn't the dataset included in the repo?**
A: MIT covers the project's own code. xBD has its own usage terms set by CMU SEI/DIU — dataset files are excluded from version control and credited via link instead.

---

## ▶️ How to Run

> Phases 0-3 are complete and Phase 4 is in progress — full inference/demo instructions will be added as later phases are finished.

```bash
# Clone the repository
git clone https://github.com/husnainalix77/disaster-damage-assessment.git
cd disaster-damage-assessment

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Verify PyTorch installed correctly (CPU build)
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Dataset (not included in repo — download separately):
- [xView2 / xBD official dataset page](https://xview2.org/dataset) — download the **Challenge training set** and **Challenge test set**, verify SHA1 checksums, then extract into `data/raw/train/` and `data/raw/test/`

Run dataset verification:
```bash
python src/verify_dataset.py
```

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

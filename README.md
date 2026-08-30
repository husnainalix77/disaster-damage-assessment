<div align="center">

# 🛰️ Disaster Damage Assessment — Post-Disaster Building Damage Detection from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Build-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%203%20Complete-brightgreen)
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
| **Deep Learning** | PyTorch (CPU build), torchvision |
| **Data Handling** | pandas, numpy, shapely |
| **Statistical Validation** | scipy — chi-square test of independence, KS-test |
| **Visualization** | matplotlib, seaborn |
| **Classical ML** | scikit-learn |
| **Deployment** | FastAPI, Docker |
| **Experiment Tracking** | MLflow |
| **Demo/UI** | Streamlit |
| **Dataset** | [xBD / xView2](https://xview2.org/dataset) — CMU SEI & U.S. Defense Innovation Unit |
| **Compute** | Local (Dell Latitude 5590, i5, 8GB RAM, CPU-only) + free-tier Google Colab / Kaggle GPU for training phases |

---

## 📊 Project Progress

| Phase | Title | Status |
|---|---|---|
| 0 | Environment, Repo Structure & Scoping Decisions | ✅ Complete |
| 1 | Dataset Acquisition & Verification | ✅ Complete |
| 2 | Exploratory Data Analysis | ✅ Complete |
| 3 | Preprocessing & Augmentation Pipeline | ✅ Complete |
| 4 | Building Localization / Segmentation Model | ⬜ Not Started |
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
Built the complete data pipeline feeding Phase 4's model training. Full findings in `docs/phase3_preprocessing_summary.md`.

- **3.1** — Split 710 training / 178 validation locations by unique location ID (not by file), preserving every location's full file bundle together
- **3.2** — Decided to exclude `un-classified` labels from classifier training (0.98% of data, not a genuine damage class)
- **3.3** — Built `SegmentationDataset` (`src/segmentation_dataset.py`), a PyTorch `Dataset` class using pre-disaster images + binary building masks, with paths anchored to the file's own location for reliable reuse from any future caller (notebook, script, or FastAPI app)
- **3.4** — Resized images/targets from 1024×1024 to 512×512, justified using Phase 2's own building-size findings (not a default choice); used different interpolation methods for image (Lanczos) vs. target (nearest-neighbor) to preserve the mask's binary values
- **3.5** — Implemented an augmentation pipeline (flips, rotation, brightness/contrast jitter) for training data only, with geometric transforms applied identically to image and target to preserve alignment — verified both by variation testing and direct visual inspection
- **3.6** — Investigated whether building density variation (Section 2.7) required pipeline changes; confirmed resize/augmentation are density-agnostic, deferred a related open question (small-building merging in dense images) to Phase 4/6 evaluation rather than solving it speculatively
- **3.7** — Final visual verification of the complete combined pipeline across 5 varied examples, confirming correct alignment and no artifacts before proceeding to Phase 4

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
| Image resolution | Uniformly 1024×1024 across all training data, pre and post — resized to 512×512 for training (Phase 3.4) |
| Building density per image | Right-skewed across all 3 disasters; `hurricane-michael` shows a distinct, more evenly-spread pattern |
| `un-classified` labels | 0.98% of combined training set — excluded from classifier training (Phase 3.2) |

---

## 📁 Repository Structure

```
disaster-damage-assessment/
├── app/                          # FastAPI + Streamlit application code (Phase 10-11)
├── data/
│   ├── raw/                      # Downloaded xBD imagery (gitignored)
│   └── processed/                # Preprocessed data (gitignored)
├── docs/
│   ├── scope_and_assumptions.md      # Honest project scope, written in Phase 0
│   ├── phase1_dataset_verification.md # Phase 1 findings and disaster selection reasoning
│   ├── phase2_eda_summary.md         # Phase 2 EDA findings and statistical validation
│   └── phase3_preprocessing_summary.md # Phase 3 preprocessing pipeline and decisions
├── models/                       # Trained model weights (gitignored)
├── notebooks/
│   ├── 01_dataset_verification.ipynb # Phase 1 verification + visual label-overlay checks
│   ├── 02_eda.ipynb                  # Phase 2 EDA — full analysis, sections 2.1-2.8
│   └── 03_preprocessing.ipynb        # Phase 3 preprocessing pipeline, sections 3.1-3.7
├── src/
│   ├── verify_dataset.py         # Reusable dataset count/parity verification utility
│   └── segmentation_dataset.py   # PyTorch Dataset class — resize, augmentation, for Phase 4
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

**7. Relative paths inside `src/segmentation_dataset.py` were fragile.**
Fixed by anchoring paths to the file's own location (`Path(__file__).resolve().parent.parent`) instead of the caller's working directory, so the class works correctly regardless of whether it's called from a notebook (`notebooks/`), a future script, or a future FastAPI app (`app/`) — each of which has a different working directory.

---

## 🎯 Key Engineering Decisions

**Q: Why not use the full xBD dataset?**
A: Impractical on an 8GB RAM, GPU-less laptop within a reasonable iteration loop. A focused subset allows honest, fast iteration.

**Q: Why train on Colab/Kaggle for some phases instead of fully locally?**
A: No dedicated GPU — CNN training is dramatically slower on CPU. Phases 0-3 run locally; Phases 4-5 use free-tier cloud GPU.

**Q: Why 3 training disasters instead of 2? Why these specific ones?**
A: See [Dataset & Training Set Rationale](#-dataset--training-set-rationale) above.

**Q: Why split by location ID instead of splitting each file type independently?**
A: Each location's 6 related files (pre/post images, labels, targets) must stay together in the same train/validation split — splitting independently risks breaking the pairing a model depends on.

**Q: Why exclude `un-classified` labels rather than treating them as a 5th class?**
A: Not a genuine damage severity level — an annotator's uncertainty marker, confirmed smaller/more ambiguous in Phase 2.5-2.6. Under 1% of the training set, so exclusion costs negligible signal.

**Q: Why separate `Dataset` classes for segmentation and classification?**
A: The two tasks need structurally different data — segmentation needs a full image + binary building mask (verified to contain no damage information at all), while classification needs individual building crops + damage labels.

**Q: Why pre-disaster images for segmentation, not post-disaster or a pre/post pair?**
A: Segmentation's only job is locating buildings, which doesn't require damage information. Pre-disaster buildings are clean and intact, giving the clearest possible training signal — post-disaster imagery (collapse, debris, smoke) would introduce unnecessary noise. The pre/post *comparison* that assesses damage happens later, in Phase 5, on individually-cropped buildings whose locations came from this segmentation model.

**Q: Why resize to 512×512, and why two different interpolation methods?**
A: Justified using Phase 2's building-size findings (not a default choice) — 512×512 balances CPU/memory feasibility against preserving enough detail for the smallest buildings. Lanczos interpolation is used for the image (smooth resampling suits photos); nearest-neighbor is used for the target mask, since smooth resampling would introduce invalid fractional pixel values into what must remain a strictly binary (0/1) mask.

**Q: Why U-Net for segmentation instead of a more complex architecture?**
A: Better learning curve for a first segmentation project — a default choice, open to revision after hands-on experience in earlier phases.

**Q: Why MIT license, and why isn't the dataset included in the repo?**
A: MIT covers the project's own code. xBD has its own usage terms set by CMU SEI/DIU — dataset files are excluded from version control and credited via link instead.

---

## ▶️ How to Run

> Phases 0-3 are complete — full inference/demo instructions will be added as later phases are finished.

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

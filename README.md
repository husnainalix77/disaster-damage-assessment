<div align="center">

# 🛰️ Disaster Damage Assessment — Post-Disaster Building Damage Detection from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Build-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen)
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
| 3 | Preprocessing & Augmentation Pipeline | ⬜ Not Started |
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
- Repo created (`disaster-damage-assessment`), structured into `data/`, `docs/`, `models/`, `notebooks/`, `src/`, `app/`
- Python virtual environment created; PyTorch **2.13.0+cpu** installed and verified as the correct CPU build for this hardware
- Full stack installed and frozen into `requirements.txt`; `docs/scope_and_assumptions.md` written upfront, documenting hardware constraints and methodology choices before any modeling began
- xBD Challenge training set (7.8 GB) and test set (2.6 GB) downloaded and **SHA1-verified**
- `.gitignore` extended to exclude large/non-redistributable data and model files

### ✅ Phase 1 — Dataset Acquisition & Verification
- Confirmed folder structure (`images/`, `labels/`, `targets/`) and full parity across all three, in both train and test splits, across 10 disaster types (2,799 total training pairs, zero mismatches)
- Built a reusable verification utility (`src/verify_dataset.py`)
- Performed visual label-overlay verification confirming polygon labels are correctly aligned to real buildings
- Full findings in `docs/phase1_dataset_verification.md`

### ✅ Phase 2 — Exploratory Data Analysis
Moved the project's training-disaster selection from a plausible starting guess to a fully evidence-backed decision, and characterized the resulting dataset ahead of preprocessing. See [Dataset & Training Set Rationale](#-dataset--training-set-rationale) below for the full breakdown, and `docs/phase2_eda_summary.md` for complete findings.

**Highlights:**
- Full-count damage-class analysis across all 10 disaster types identified a critical `destroyed`-class gap in the original 2-disaster set and resolved it with a statistically justified 3rd training disaster
- Building size, image resolution, and building density were all characterized across the final training set, directly informing Phase 3's preprocessing strategy
- Two statistical techniques (chi-square test, KS-test) were applied to validate observed patterns rather than relying on visual inspection alone — the same techniques used in the [transaction-fraud-risk-engine](https://github.com/husnainalix77/transaction-fraud-risk-engine) project, now reused in a second, different domain

---

## 🗂️ Dataset & Training Set Rationale

**Final training set:** `hurricane-harvey` (319 pairs) + `hurricane-michael` (343 pairs) + `santa-rosa-wildfire` (226 pairs)
**Held-out generalization disaster (Phase 7):** `mexico-earthquake` (121 pairs)

| Step | Finding |
|---|---|
| Initial 2-disaster set (`harvey` + `michael`) | Only 2.5% combined `destroyed`-class representation — a critical gap for a disaster-response tool |
| Full 10-disaster count analysis | 4 candidates identified by `destroyed`%: `santa-rosa-wildfire` (26.8%), `palu-tsunami` (15.8%), `hurricane-matthew` (15.4%), `socal-fire` (12.7%) |
| `hurricane-matthew` — excluded | Duplicates the existing hurricane wind/flood damage mechanism; adds volume, not diversity |
| `palu-tsunami` — excluded | Real-world combined earthquake+tsunami event; would blur the "unseen mechanism" boundary against the `mexico-earthquake` held-out disaster |
| `socal-fire` — re-evaluated, not selected | Full-count `un-classified` rate is only 3.6% (an earlier small visual sample overstated this); still a weaker `destroyed`-class benefit than `santa-rosa-wildfire` |
| `santa-rosa-wildfire` — **selected** | 26.8% `destroyed` (highest by a wide margin), mechanically distinct (fire vs. wind/flood), doesn't compromise the held-out test |
| Statistical confirmation | Chi-square test confirms damage-class distributions differ significantly across the 3 training disasters (χ² = 19218.04, p < 0.0001) |
| Building size, by disaster | Consistent across all 3 disasters (median ~900–1,200 sq px) — no conflicting scale introduced |
| Building size, by damage class | Consistent across damage classes, except `un-classified` buildings, which are significantly smaller (KS statistic = 0.3916, p < 0.0001) — a plausible explanation for elevated `un-classified` rates in fire disasters |
| Image resolution | Uniformly 1024×1024 across all training data, pre and post |
| Building density per image | Right-skewed across all 3 disasters (mean 57–72 buildings/image); `hurricane-michael` shows a distinct, more evenly-spread pattern |

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
│   └── phase2_eda_summary.md         # Phase 2 EDA findings and statistical validation
├── models/                       # Trained model weights (gitignored)
├── notebooks/
│   ├── 01_dataset_verification.ipynb # Phase 1 verification + visual label-overlay checks
│   └── 02_eda.ipynb                  # Phase 2 EDA — full analysis, sections 2.1-2.8
├── src/
│   └── verify_dataset.py         # Reusable dataset count/parity verification utility
├── .gitignore
├── LICENSE                       # MIT (project code only — not the dataset)
├── README.md
└── requirements.txt
```

---

## 🐛 Problems Faced & How They Were Solved

**1. Windows path-length limit broke the PyTorch install.**
PyTorch's deeply nested internal files, combined with the project's folder path, exceeded Windows' 260-character path limit, causing a corrupted install. **Fixed** by enabling Windows long-path support and Git's long-path config, then reinstalling cleanly.

**2. `.gitignore` didn't cover project-specific large files.**
GitHub's default Python template doesn't account for large data/model files. **Fixed** by manually extending it to exclude `data/raw/`, `data/processed/`, model weight files, and MLflow logs.

**3. SHA1 verification via `shasum` doesn't exist natively on Windows.**
**Fixed** using PowerShell's built-in `Get-FileHash -Algorithm SHA1`.

**4. A silent validation bug in the dataset-structure checker.**
`_validate_structure()` returned success after checking only the first folder due to an indentation error. **Fixed** by correcting the loop structure so all folders are genuinely checked.

**5. A small visual sample overstated `socal-fire`'s data quality issue.**
An ~15-image visual check suggested `socal-fire` was predominantly `un-classified`; Phase 2's full-count analysis (10,475 buildings) showed the true rate was only 3.6%. **Lesson applied:** visual samples flag possible issues, but final decisions rest on full counts — which is why Phase 2 checked all 10 disaster types rather than stopping at the initial candidates.

---

## 🎯 Key Engineering Decisions

**Q: Why not use the full xBD dataset?**
A: The full dataset (22,068 images, 850,736 annotations) is impractical on an 8GB RAM, GPU-less laptop within a reasonable iteration loop. A focused subset allows honest, fast iteration.

**Q: Why train on Colab/Kaggle for some phases instead of fully locally?**
A: Integrated graphics only, no dedicated GPU — CNN training is dramatically slower on CPU. Phases 0-3 run locally; Phases 4-5 use free-tier cloud GPU.

**Q: Why 3 training disasters instead of 2? Why these specific ones?**
A: See [Dataset & Training Set Rationale](#-dataset--training-set-rationale) above for the full, evidence-based breakdown.

**Q: Why U-Net for segmentation instead of a more complex architecture?**
A: Better learning curve for a first segmentation project — a default choice, open to revision after hands-on experience in earlier phases.

**Q: Why MIT license, and why isn't the dataset included in the repo?**
A: MIT covers the project's own code. xBD has its own usage terms set by CMU SEI/DIU — dataset files are excluded from version control and credited via link instead.

---

## ▶️ How to Run

> Phases 0-2 are complete — full inference/demo instructions will be added as later phases are finished.

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

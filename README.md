<div align="center">

# 🛰️ Disaster Damage Assessment — Post-Disaster Building Damage Detection from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Build-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-yellow)
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

> 📌 **Honest scope note:** this is a solo, single-laptop project (8GB RAM, no dedicated GPU) built on a 2–3 disaster-type subset of the real xBD/xView2 benchmark dataset — a proof of concept demonstrating industry-standard *method*, not industry-scale *infrastructure*. See [Key Engineering Decisions](#-key-engineering-decisions) for the full reasoning.

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
| 2 | Exploratory Data Analysis | 🔄 Ongoing |
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
- Python virtual environment (`venv`) created and activated
- PyTorch **2.13.0+cpu** installed and verified (`torch.cuda.is_available()` correctly returns `False`, confirming the CPU build matches the hardware)
- Full stack installed: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter, fastapi, uvicorn, mlflow — frozen into `requirements.txt`
- `docs/scope_and_assumptions.md` written upfront, documenting the dataset subset decision, hardware constraints, methodology choices, and known generalization limitations *before* any modeling began
- xBD Challenge training set (7.8 GB) and test set (2.6 GB) downloaded and **SHA1-verified** against official checksums
- `.gitignore` extended beyond the default Python template to exclude `data/raw/`, `data/processed/`, model weight files, and MLflow tracking logs — keeping large/non-redistributable files out of version control

### ✅ Phase 1 — Dataset Acquisition & Verification
- Confirmed folder structure in both `data/raw/train` and `data/raw/test`: `images/` (pre/post PNG pairs), `labels/` (JSON polygon + damage annotations), `targets/` (rasterized damage mask PNGs)
- Built a reusable `DisasterInspector` verification utility (`src/verify_dataset.py`) that scans filenames, extracts disaster type, and cross-checks counts across all three folders
- Confirmed **2,799 pre/post image pairs** across 10 disaster types in the training set, with full images/labels/targets parity (no mismatches) in both the training and test splits
- **Selected training disasters: `hurricane-harvey`** (319 pairs, flood/wind damage) **+ `hurricane-michael`** (343 pairs, wind/structural damage) — chosen for being visually and mechanically distinct damage types
- **Selected held-out generalization disaster (for Phase 7): `mexico-earthquake`** (121 pairs, structural collapse) — mechanically distinct from both training types, making it a genuine generalization test rather than a soft one
- Performed **visual label-overlay verification**: parsed WKT building polygons (via `shapely`) from label JSONs and plotted them directly on matching post-disaster imagery, color-coded by damage class — confirmed polygons are correctly aligned to real building shapes across all selected disaster types
- **Disaster selection was revised mid-phase**: `socal-fire` was the original training choice but was dropped after visual sampling of ~15 images showed buildings predominantly labeled `un-classified` rather than a specific damage class — likely a fire-damage-specific annotation characteristic. `hurricane-michael` was verified via the same method to show a healthy spread across all 4 target damage classes and selected as the replacement
- Full findings and revision history documented in `docs/phase1_dataset_verification.md`

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
│   └── phase1_dataset_verification.md # Phase 1 findings and disaster selection reasoning
├── models/                       # Trained model weights (gitignored)
├── notebooks/
│   └── 01_dataset_verification.ipynb # Phase 1 verification + visual label-overlay checks
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
PyTorch's package includes deeply nested internal files (e.g. `torch/.../third_party/kineto/.../duktape-1.5.2`), which combined with the project's folder path exceeded Windows' default 260-character path limit, causing `WinError 206` mid-install and leaving a corrupted, partial installation (`ModuleNotFoundError: No module named 'torchgen'`). **Fixed** by enabling Windows long-path support via a registry change (`LongPathsEnabled`), enabling `git config --system core.longpaths true`, restarting, then reinstalling cleanly — verified afterward with `torch.__version__` and `torch.cuda.is_available()`.

**2. `.gitignore` didn't cover project-specific large files.**
GitHub's default Python `.gitignore` template covers standard Python artifacts (`__pycache__/`, `.pyc`, etc.) but knows nothing about this project's large data/model files. **Fixed** by manually extending `.gitignore` with `data/raw/`, `data/processed/`, `models/*.pt`, `mlruns/`, and `*.tar.gz` before any large files could be accidentally committed.

**3. SHA1 verification via `shasum` doesn't exist natively on Windows.**
The xView2 download page's verification instructions assume a Unix-like `shasum` command, unavailable by default in Windows PowerShell. **Fixed** by using PowerShell's built-in `Get-FileHash -Algorithm SHA1` instead — confirmed both training and test archive hashes matched the official checksums exactly.

**4. A silent validation bug in the dataset-structure checker.**
An early version of `_validate_structure()` had a `return True` statement indented inside the folder-existence loop instead of after it — meaning the function returned success after checking only the *first* folder, never actually verifying the other two. It went unnoticed initially because all folders genuinely existed. **Fixed** by correcting the indentation so all three folders are checked before returning success.

**5. Initial disaster type selection (`socal-fire`) turned out to be a poor training choice.**
Counts and label/target parity for `socal-fire` all checked out cleanly — the problem only surfaced during visual label-overlay verification, where ~15 sampled images showed buildings predominantly labeled `un-classified` rather than a specific damage class. Numeric verification alone would not have caught this. **Fixed** by visually re-evaluating alternative disaster types the same way and confirming `hurricane-michael` shows a healthy spread across all 4 target damage classes before selecting it as the replacement — a reminder that "the counts match" is necessary but not sufficient verification for a dataset intended to teach a model real damage patterns.

---

## 🎯 Key Engineering Decisions

**Q: Why not use the full xBD dataset?**
A: The full dataset (22,068 images, 850,736 building annotations) is impractical to train on an 8GB RAM, GPU-less laptop within a reasonable iteration loop. A focused 2–3 disaster-type subset allows honest, fast iteration — more data would mainly buy marginal accuracy, not new understanding, which isn't this project's goal.

**Q: Why train on Colab/Kaggle for some phases instead of fully locally?**
A: The laptop has integrated graphics only — no dedicated GPU — making CNN training (especially segmentation) dramatically slower on CPU. Phases 0–3 (setup, verification, EDA, preprocessing) run locally; Phases 4–5 (the actual model training) use free-tier cloud GPU, with trained weights brought back locally afterward.

**Q: Why U-Net for segmentation instead of a more complex architecture?**
A: U-Net offers a better learning curve for a first segmentation project and is well-documented, appropriate given this is the author's first deep learning project. This is a default choice, open to revision after hands-on experience in earlier phases.

**Q: Why hurricane-harvey + hurricane-michael for training, and mexico-earthquake for the held-out test?**
A: Training disasters were chosen to be visually and mechanically distinct from each other (flood/wind vs. wind/structural), so the model learns genuinely different damage patterns rather than near-duplicate ones. `socal-fire` was the original choice for its larger pair count and distinct damage mechanism (fire), but was replaced after visual verification showed most of its buildings were labeled `un-classified` rather than a usable damage class — a real data-quality finding, not an assumption. The held-out disaster was deliberately chosen to be mechanically distinct from *both* training types (earthquake structural collapse vs. flood/wind), making the Phase 7 generalization test genuinely meaningful rather than artificially easy.

**Q: Why MIT license, and why isn't the dataset included in the repo?**
A: MIT license covers the project's own code, consistent with prior portfolio projects. The xBD dataset has its own usage terms set by CMU SEI/DIU, separate from this repo's license — dataset files are excluded from version control (`.gitignore`) and credited via link in this README instead.

---

## ▶️ How to Run

> Phases 0-1 (environment setup + dataset verification) are complete — full inference/demo instructions will be added as later phases are finished.

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

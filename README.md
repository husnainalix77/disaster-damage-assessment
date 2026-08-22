<div align="center">

# 🛰️ Disaster Damage Assessment — Post-Disaster Building Damage Detection from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Build-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%200%20Complete-yellow)
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
| **Data Handling** | pandas, numpy |
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
| 1 | Dataset Acquisition & Verification | ⬜ Not Started |
| 2 | Exploratory Data Analysis | ⬜ Not Started |
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

---

## 📁 Repository Structure

```
disaster-damage-assessment/
├── app/                          # FastAPI + Streamlit application code (Phase 10-11)
├── data/
│   ├── raw/                      # Downloaded xBD imagery (gitignored)
│   └── processed/                # Preprocessed data (gitignored)
├── docs/
│   └── scope_and_assumptions.md  # Honest project scope, written in Phase 0
├── models/                       # Trained model weights (gitignored)
├── notebooks/                    # Jupyter notebooks per phase
├── src/                          # Reusable pipeline code
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

---

## 🎯 Key Engineering Decisions

**Q: Why not use the full xBD dataset?**
A: The full dataset (22,068 images, 850,736 building annotations) is impractical to train on an 8GB RAM, GPU-less laptop within a reasonable iteration loop. A focused 2–3 disaster-type subset allows honest, fast iteration — more data would mainly buy marginal accuracy, not new understanding, which isn't this project's goal.

**Q: Why train on Colab/Kaggle for some phases instead of fully locally?**
A: The laptop has integrated graphics only — no dedicated GPU — making CNN training (especially segmentation) dramatically slower on CPU. Phases 0–3 (setup, verification, EDA, preprocessing) run locally; Phases 4–5 (the actual model training) use free-tier cloud GPU, with trained weights brought back locally afterward.

**Q: Why U-Net for segmentation instead of a more complex architecture?**
A: U-Net offers a better learning curve for a first segmentation project and is well-documented, appropriate given this is the author's first deep learning project. This is a default choice, open to revision after hands-on experience in earlier phases.

**Q: Why MIT license, and why isn't the dataset included in the repo?**
A: MIT license covers the project's own code, consistent with prior portfolio projects. The xBD dataset has its own usage terms set by CMU SEI/DIU, separate from this repo's license — dataset files are excluded from version control (`.gitignore`) and credited via link in this README instead.

---

## ▶️ How to Run

> Only Phase 0 (environment setup) is complete — full inference/demo instructions will be added as later phases are finished.

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
- [xView2 / xBD official dataset page](https://xview2.org/dataset) — download the **Challenge training set** and **Challenge test set**, verify SHA1 checksums, then extract into `data/raw/`

---

## 👤 About the Author

<div align="center">

**Husnain Maroof**
Mechatronics & Control Engineering Student, UET Lahore

[![GitHub](https://img.shields.io/badge/GitHub-husnainalix77-181717?logo=github&logoColor=white)](https://github.com/husnainalix77)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/)

</div>

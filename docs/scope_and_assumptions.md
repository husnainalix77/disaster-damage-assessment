# Scope & Assumptions

This document states upfront what this project does and does not claim, so it's never oversold — in the README, in an interview, or to myself mid-project.

## Problem
Automate the interpretation step of post-disaster damage assessment: given a pre-disaster and post-disaster satellite image pair, locate buildings and classify each one's damage level (No Damage / Minor Damage / Major Damage / Destroyed). Satellites already capture the imagery quickly — the bottleneck this project addresses is the currently manual, slow human interpretation of that imagery.

## Dataset Scope
- Source: xBD / xView2 Challenge dataset (train + test sets, SHA1-verified on download)
- Training subset: 2 disaster types selected from the Challenge training set (not the full dataset, not the full "with metadata" GeoTIFF release, not Tier3 additional data)
- **Selected training disasters: `hurricane-harvey` (319 pairs, flood/wind damage) + `hurricane-michael` (343 pairs, wind/structural damage)** — chosen for being visually/mechanically distinct damage types
- **Held-out generalization disaster (Phase 7): `mexico-earthquake` (121 pairs, structural collapse)** — mechanically distinct from both training types
- **Revision note:** `socal-fire` (823 pairs, fire damage) was the original training choice but was dropped after visual sampling (~15 images) showed buildings predominantly labeled `un-classified` rather than a specific damage class — likely a fire-damage-specific annotation characteristic (char/smoke damage is harder to visually classify than structural collapse or water damage), not a data-quality bug. `hurricane-michael` was confirmed via the same visual sampling method to show a healthy spread across all 4 target damage classes before being selected as the replacement.
- Reason for a subset at all: hardware constraints (see below) make the full ~22,068-image dataset impractical to iterate on locally; a focused subset allows faster, honest iteration
- A 3rd disaster type may be added later only if Phases 0-4 go faster than expected — not assumed upfront

## Hardware & Compute Scope
- Local machine: Dell Latitude 5590, i5 8th gen, 8GB DDR4 RAM, integrated GPU only (no dedicated/NVIDIA GPU), 512GB NVMe SSD
- Implication: all CNN training is CPU-bound locally, which is slow for segmentation/classification workloads
- Plan: Phases 0-3 (setup, data verification, EDA, preprocessing) run locally. Phases 4-5 (segmentation and classification training) run on free-tier Google Colab / Kaggle GPU notebooks; trained model weights are brought back locally for Phases 6 onward
- This is a proof-of-concept demonstrating method at industry standard, not infrastructure at industry scale

## Methodology Scope
- Segmentation architecture: U-Net-style, chosen for its learning curve as a first segmentation project (default choice — may revisit after Phase 0-3 experience)
- Damage classification: transfer learning on a pretrained CNN backbone, not training from scratch (correct choice for limited data/compute, not a shortcut)
- Multiple architectures will be compared per stage (2-3 segmentation architectures, 2-3 classification backbones) rather than training one model each, with the final choice backed by a paired statistical significance test (Wilcoxon/bootstrap), not just the higher raw metric
- Evaluation includes segmentation metrics (IoU/Dice), per-class F1, weighted Cohen's Kappa (for the ordinal damage scale), and standard distribution tests (chi-square, KS-test) reused from prior project experience

## Generalization Scope — Known Limitation
- The model will generalize reasonably to data resembling its training distribution (similar disaster types, regions, imagery source)
- The model will NOT reliably generalize to a genuinely unseen disaster type, region, or satellite source — this is measured directly via a held-out disaster-type test (not assumed, not hidden)
- Production systems handle this via continuous retraining; this project measures and reports the gap honestly rather than solving it, which is outside a portfolio project's realistic scope

## Deployment Scope
- FastAPI + minimal Docker containerization, proving the model works end-to-end outside a notebook
- No CI/CD, no orchestration (Kubernetes), no load balancing, no uptime guarantees — intentionally out of scope for a solo project

## Licensing Note
- Project code is MIT licensed
- The xBD dataset itself is NOT covered by this repo's license and is subject to xView2/DIU's own terms — dataset files are excluded from version control (see `.gitignore`) and credited in the README

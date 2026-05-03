# TackleNet

**A Biomechanical Practice-Video Benchmark for Tackle-Safety Assessment in American Football**

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/ahsanzaidi786/tacklenet)

This repository is the companion code release for the **TackleNet** benchmark dataset,
accompanying the NeurIPS 2026 Datasets and Benchmarks submission.

> **Run all commands from the repository root.**
> Scripts use package-relative imports (`from eval.metrics import ...`) that
> require the repo root to be on the Python path. Running from a subdirectory
> will produce `ModuleNotFoundError`.

**Full dataset (~25 GB):** https://www.kaggle.com/datasets/ahsanzaidi786/tacklenet  
**30-clip sample (~1 GB):** https://www.kaggle.com/datasets/ahsanzaidi786/tacklenet-sample

---

## Dataset at a Glance

| Property | Value |
|---|---|
| Total clips | 737 de-identified MP4 clips |
| Risky / Safe | 259 risky (35.1%) / 478 safe (64.9%) |
| Practice sessions | 30 sessions, 3 collection sites (PA, KS, FL) |
| Benchmark label | `satt3_risk_binary` derived from SATT3 component score |
| FPOC label | `fpoc_frame` — 0-indexed frame of First Point of Contact |
| Train / Val / Test | 515 / 111 / 111 (stratified clip-level split, seed 42) |
| De-identification | RetinaFace + SAM 3.1 (face regions) |
| Data license | CC BY 4.0 |
| Code license | MIT |

---

## Benchmark Tasks

| Task | Goal | Primary Metric |
|---|---|---|
| **Task A — Risk Classification** | Predict tackle safety (risky/safe) from video clip | Risky recall |
| **Task B — Contact Onset Localization** | Predict FPOC frame index | Acc@±20 frames |
| **Task C — Contact Window Analysis** | Oracle analysis: GT FPOC window vs. full clip | PR-AUC delta |

Task A and Task C use the frozen train/val/test split (`splits/splits_frozen.csv`).
Task B methods are zero-shot and are evaluated on all 737 clips.

---

## Repository Structure

```
tacklenet/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── CONTRIBUTING.md
├── .gitignore
├── data/
│   ├── __init__.py
│   └── loader.py                    # PyTorch Dataset for TackleNet clips
├── splits/
│   ├── annotations_satt_fpoc.csv    # Frozen annotation table (737 clips)
│   ├── splits_frozen.csv            # Official train/val/test split
│   ├── split_policy.md              # Split rationale and limitations
│   └── tacklenet_create_splits_v2.py
├── baselines/
│   ├── majority_baseline.py         # Task A: majority-class baseline
│   └── center_frame_baseline.py     # Task B: center-frame FPOC baseline
├── eval/
│   ├── metrics.py                   # Shared metric utilities
│   ├── evaluate_task_a.py           # Task A evaluator
│   └── evaluate_task_b.py           # Task B evaluator
├── examples/
│   ├── task_a_predictions_example.csv   # 10-row format demo — not full eval
│   └── task_b_predictions_example.csv   # 10-row format demo — not full eval
└── notebooks/
    └── tacklenet_usage.ipynb        # Training-free usage notebook
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Full dataset (~25 GB)
kaggle datasets download ahsanzaidi786/tacklenet --unzip -p ./tacklenet_data

# 30-clip sample (~1 GB) — for pipeline verification only
kaggle datasets download ahsanzaidi786/tacklenet-sample --unzip -p ./tacklenet_sample

**Interactive notebook (no setup required):**
https://www.kaggle.com/code/ahsanzaidi786/tacklenet-usage

```

---

## Task A — Risk Classification

### Run the majority baseline

```bash
python baselines/majority_baseline.py \
    --annot   splits/annotations_satt_fpoc.csv \
    --splits  splits/splits_frozen.csv \
    --split   test \
    --out_dir outputs/majority_baseline
```

Outputs: `majority_predictions.csv`, `majority_metrics.json`

### Evaluate Task A predictions

```bash
python eval/evaluate_task_a.py \
    --predictions outputs/majority_baseline/majority_predictions.csv \
    --splits      splits/splits_frozen.csv \
    --split       test \
    --out_dir     outputs/majority_eval
```

Outputs: `task_a_metrics.json`, `task_a_classification_report.txt`, `task_a_confusion_matrix.png`

### Prediction file format

One row per test clip:

| Column | Type | Description |
|---|---|---|
| `clip_id` | int | Clip identifier matching `splits_frozen.csv` |
| `prob_risky` | float [0,1] | Predicted risky-class probability |
| `pred_label` | int 0/1 | Predicted binary label |

See `examples/task_a_predictions_example.csv` for a 10-row format demo.

---

## Task B — Contact Onset Localization

Task B methods are zero-shot and are evaluated on all 737 clips.
The GRAZE pipeline (reference Task B method) is maintained separately:

> https://github.com/AhsanZaidi12/GRAZE

GRAZE predictions can be evaluated directly with `eval/evaluate_task_b.py`.

### Run the center-frame baseline

```bash
python baselines/center_frame_baseline.py \
    --annot     splits/annotations_satt_fpoc.csv \
    --video_dir path/to/tacklenet_data/videos \
    --out_dir   outputs/center_frame
```

Outputs: `center_frame_predictions.csv` (for the evaluator),
`center_frame_results.csv` (per-clip detail)

### Evaluate Task B predictions

```bash
python eval/evaluate_task_b.py \
    --predictions outputs/center_frame/center_frame_predictions.csv \
    --annot       splits/annotations_satt_fpoc.csv \
    --splits      splits/splits_frozen.csv \
    --split       all \
    --out_dir     outputs/center_frame_eval
```

Outputs: `task_b_metrics.json`, `task_b_error_distribution.png`

### Smoke-test the evaluator format (no videos needed)

The example file contains only 10 rows and is intended for format
validation only — not for benchmark reporting. Running against
`--split test` (111 clips) will correctly show partial coverage.

```bash
python eval/evaluate_task_b.py \
    --predictions examples/task_b_predictions_example.csv \
    --annot       splits/annotations_satt_fpoc.csv \
    --splits      splits/splits_frozen.csv \
    --split       test \
    --out_dir     /tmp/test_eval_b
```

Expected output: `n_covered=10, n_total=111, coverage=9.0%`.
This confirms the evaluator runs correctly; the low coverage is
expected because the example file contains only 10 rows.

### Prediction file format

One row per clip:

| Column | Type | Description |
|---|---|---|
| `clip_id` | int | Clip identifier |
| `pred_fpoc_frame` | int | Predicted 0-indexed FPOC frame |

See `examples/task_b_predictions_example.csv` for a 10-row format demo.

---

## Task C — Contact Window Analysis

Task C is an oracle analysis comparing classifiers trained and evaluated
on full clips against classifiers trained and evaluated on the GT FPOC
32-frame window (23 pre-contact + FPOC frame + 8 post-contact).

It uses the same Task A evaluation protocol. Train a classifier under
each input condition, save its test predictions, and run
`evaluate_task_a.py` on each. Compare the PR-AUC values.

---

## Split Policy

Always use `splits/splits_frozen.csv`. Do not regenerate or substitute
a custom split for benchmark-comparable results.
See `splits/split_policy.md` for full details.

The split is stratified at the clip level (seed 42, 70/15/15). It is
not session-grouped because reliable session identifiers were not
available at split construction time. `Session_ID` is provided in the
annotation file for transparency and post-hoc analysis only.

### Verify the frozen split

```bash
python splits/tacklenet_create_splits_v2.py \
    --input  splits/annotations_satt_fpoc.csv \
    --output splits_check.csv \
    --seed   42
```

`splits_check.csv` must match `splits/splits_frozen.csv` exactly.

---

## Label Convention

A clip is labeled **risky** (`satt3_risk_binary = 1`) if its SATT3
component score is 0 or 1, and **safe** (`satt3_risk_binary = 0`) if
the score is 2 or 3.

---

## FPOC Window Convention

All FPOC-windowed models use the locked asymmetric 32-frame window:

- **23 frames before FPOC** + FPOC frame + **8 frames after** = 32 frames total
- FPOC is placed at window index 23 (pre-contact heavy by design)
- Boundary handling: repeat-first / repeat-last frame clamping (never zero-pad)

Implemented in `data/loader.py` and demonstrated in
`notebooks/tacklenet_usage.ipynb`.

---

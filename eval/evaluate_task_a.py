#!/usr/bin/env python3
"""
TackleNet Task A — Risk Classification Evaluator

Evaluates a predictions CSV against ground-truth labels from
splits_frozen.csv, restricted to the requested split (default: test).

Prediction CSV format — one row per clip:
    clip_id      int    matching splits_frozen.csv
    prob_risky   float  predicted risky-class probability in [0, 1]
    pred_label   int    predicted binary label (0=safe, 1=risky)

Usage
-----
python eval/evaluate_task_a.py \\
    --predictions examples/task_a_predictions_example.csv \\
    --splits      splits/splits_frozen.csv \\
    --split       test \\
    --out_dir     .

Outputs
-------
task_a_metrics.json               all Task A metrics
task_a_classification_report.txt  per-class precision / recall / F1
task_a_confusion_matrix.png       row-normalised confusion matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.metrics import (
    classification_report_str,
    compute_confusion_matrix,
    compute_task_a_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TackleNet Task A Evaluation")
    p.add_argument("--predictions", required=True,
                   help="CSV with columns: clip_id, prob_risky, pred_label")
    p.add_argument("--splits", required=True,
                   help="Path to splits_frozen.csv")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Which split to evaluate (default: test)")
    p.add_argument("--out_dir", default=".",
                   help="Output directory (default: current dir)")
    return p.parse_args()


def load_and_merge(
    pred_path: str,
    splits_path: str,
    split: str,
) -> pd.DataFrame:
    """
    Load predictions and merge with ground-truth labels from splits_frozen.csv.

    splits_frozen.csv contains satt3_risk_binary (identical to the annotation
    file — 0 mismatches verified). This avoids requiring --annot as a second
    argument while keeping a single source of truth for the frozen split.
    """
    preds  = pd.read_csv(pred_path)
    splits = pd.read_csv(splits_path)

    # Validate required columns
    required_pred   = {"clip_id", "prob_risky", "pred_label"}
    required_splits = {"clip_id", "satt3_risk_binary", "split"}
    missing_pred    = required_pred   - set(preds.columns)
    missing_splits  = required_splits - set(splits.columns)
    if missing_pred:
        raise ValueError(f"Prediction file missing columns: {sorted(missing_pred)}")
    if missing_splits:
        raise ValueError(f"Splits file missing columns: {sorted(missing_splits)}")

    # Filter to requested split
    split_df = splits[splits["split"] == split][["clip_id", "satt3_risk_binary"]].copy()
    if len(split_df) == 0:
        raise ValueError(f"No clips found for split='{split}' in {splits_path}")

    # Merge predictions with ground truth
    merged = split_df.merge(
        preds[["clip_id", "prob_risky", "pred_label"]],
        on="clip_id",
        how="inner",
    )

    n_gt   = len(split_df)
    n_pred = len(merged)
    if n_pred < n_gt:
        print(
            f"WARNING: predictions cover {n_pred}/{n_gt} clips in "
            f"split='{split}'. Missing clips are excluded from evaluation."
        )
    if n_pred == 0:
        raise ValueError(
            f"No clip_ids matched between predictions and split='{split}'. "
            f"Check that clip_ids in the prediction file match splits_frozen.csv."
        )

    return merged


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "#222222"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                ha="center", va="center",
                fontsize=11, color=color,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Safe", "Risky"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Safe", "Risky"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True",      fontsize=10)
    ax.set_title("Task A — Confusion Matrix", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(args.predictions, args.splits, args.split)

    y_true     = df["satt3_risk_binary"].to_numpy(dtype=int)
    y_pred     = df["pred_label"].to_numpy(dtype=int)
    prob_risky = df["prob_risky"].to_numpy(dtype=float)

    metrics = compute_task_a_metrics(y_true, y_pred, prob_risky)
    metrics.update({
        "split":           args.split,
        "n_total":         int(len(y_true)),
        "n_risky_true":    int(y_true.sum()),
        "n_risky_pred":    int(y_pred.sum()),
        "prediction_file": str(args.predictions),
    })

    print(f"\n=== Task A Results ({args.split} split, n={metrics['n_total']}) ===")
    print(f"  Risky recall      : {metrics['risky_recall']:.4f}  [primary metric]")
    print(f"  Risky precision   : {metrics['risky_precision']:.4f}")
    print(f"  Risky F1          : {metrics['risky_f1']:.4f}")
    print(f"  Macro-F1          : {metrics['macro_f1']:.4f}")
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  PR-AUC            : {metrics['pr_auc']}")
    print(f"  Risky true / pred : {metrics['n_risky_true']} / {metrics['n_risky_pred']}")

    # Save JSON
    json_path = out_dir / "task_a_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save classification report
    report_path = out_dir / "task_a_classification_report.txt"
    report_path.write_text(classification_report_str(y_true, y_pred))
    print(f"Saved: {report_path}")

    # Save confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, out_dir / "task_a_confusion_matrix.png")


if __name__ == "__main__":
    main()

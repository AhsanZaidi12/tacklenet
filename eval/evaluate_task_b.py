#!/usr/bin/env python3
"""
TackleNet Task B — Contact Onset Localization Evaluator

Evaluates predicted FPOC frames against ground-truth fpoc_frame from
the annotation file. Task B methods are zero-shot and are evaluated on
all 737 clips (--split all, the default). Missing predictions count as
incorrect toward Acc@k.

Prediction CSV format — one row per clip:
    clip_id          int   matching annotations_satt_fpoc.csv
    pred_fpoc_frame  int   predicted 0-indexed FPOC frame

Usage
-----
python eval/evaluate_task_b.py \\
    --predictions examples/task_b_predictions_example.csv \\
    --annot       splits/annotations_satt_fpoc.csv \\
    --splits      splits/splits_frozen.csv \\
    --split       all \\
    --out_dir     .

Outputs
-------
task_b_metrics.json              all Task B metrics
task_b_error_distribution.png    absolute error histogram
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

from eval.metrics import compute_task_b_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TackleNet Task B Evaluation")
    p.add_argument("--predictions", required=True,
                   help="CSV with columns: clip_id, pred_fpoc_frame")
    p.add_argument("--annot", required=True,
                   help="Path to annotations_satt_fpoc.csv")
    p.add_argument("--splits", required=True,
                   help="Path to splits_frozen.csv")
    p.add_argument("--split", default="all",
                   choices=["all", "train", "val", "test"],
                   help="Clip set to evaluate. 'all' = all 737 clips "
                        "(correct for zero-shot Task B). Default: all")
    p.add_argument("--out_dir", default=".",
                   help="Output directory (default: current dir)")
    return p.parse_args()


def load_and_merge(
    pred_path: str,
    annot_path: str,
    splits_path: str,
    split: str,
) -> tuple[pd.DataFrame, int]:
    """
    Returns (merged_df, n_total).
    merged_df contains only clips where a prediction exists.
    n_total is the full size of the evaluated set (coverage denominator).
    Missing predictions count as incorrect in Acc@k.
    """
    preds  = pd.read_csv(pred_path)
    annot  = pd.read_csv(annot_path)
    splits = pd.read_csv(splits_path)

    # Validate required columns
    required_pred   = {"clip_id", "pred_fpoc_frame"}
    required_annot  = {"clip_id", "fpoc_frame"}
    required_splits = {"clip_id", "split"}
    missing_pred    = required_pred   - set(preds.columns)
    missing_annot   = required_annot  - set(annot.columns)
    missing_splits  = required_splits - set(splits.columns)
    if missing_pred:
        raise ValueError(f"Prediction file missing columns: {sorted(missing_pred)}")
    if missing_annot:
        raise ValueError(f"Annotation file missing columns: {sorted(missing_annot)}")
    if missing_splits:
        raise ValueError(f"Splits file missing columns: {sorted(missing_splits)}")

    # Select the right clip set
    if split == "all":
        annot_subset = annot[["clip_id", "fpoc_frame"]].copy()
    else:
        split_ids    = splits.loc[splits["split"] == split, "clip_id"]
        annot_subset = annot.loc[
            annot["clip_id"].isin(split_ids), ["clip_id", "fpoc_frame"]
        ].copy()

    n_total = len(annot_subset)
    if n_total == 0:
        raise ValueError(f"No annotation rows for split='{split}'")

    # Inner join — clips without predictions are excluded from error stats
    # but counted in n_total for coverage and Acc@k
    merged = annot_subset.merge(
        preds[["clip_id", "pred_fpoc_frame"]],
        on="clip_id",
        how="inner",
    )

    n_covered = len(merged)
    if n_covered == 0:
        raise ValueError(
            "No clip_ids matched between predictions and the annotation set. "
            "Check that clip_ids in the prediction file match annotations_satt_fpoc.csv."
        )
    if n_covered < n_total:
        print(
            f"INFO: {n_covered}/{n_total} clips have predictions "
            f"(coverage={n_covered/n_total:.1%}). "
            f"Missing predictions count as incorrect in Acc@k."
        )

    return merged, n_total


def plot_error_distribution(abs_errors: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(abs_errors, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.median(abs_errors)), color="#DD8452", linewidth=1.5,
               label=f"Median = {np.median(abs_errors):.1f} frames")
    ax.axvline(float(np.mean(abs_errors)),   color="#55A868", linewidth=1.5,
               linestyle="--",
               label=f"Mean = {np.mean(abs_errors):.1f} frames")
    for tol in [5, 10, 20]:
        ax.axvline(tol, color="gray", linewidth=0.8, linestyle=":",
                   label=f"±{tol}f")
    ax.set_title("Task B — Absolute Frame Error Distribution")
    ax.set_xlabel("Absolute frame error (frames)")
    ax.set_ylabel("Clips")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, n_total = load_and_merge(
        args.predictions, args.annot, args.splits, args.split
    )

    gt_fpoc   = merged["fpoc_frame"].to_numpy(dtype=int)
    pred_fpoc = merged["pred_fpoc_frame"].to_numpy(dtype=int)

    metrics = compute_task_b_metrics(gt_fpoc, pred_fpoc, n_total)
    metrics["split"]           = args.split
    metrics["prediction_file"] = str(args.predictions)

    n_cov = metrics["n_covered"]
    print(f"\n=== Task B Results (split='{args.split}', "
          f"n_covered={n_cov}, n_total={n_total}) ===")
    print(f"  Coverage          : {metrics['coverage']:.1%}")
    print(f"  Median Abs Error  : {metrics['median_abs_error']}  [primary metric]")
    print(f"  MAE               : {metrics['mae']}")
    print(f"  Acc @ ±5  frames  : {metrics['acc_5']:.1%}  "
          f"({int(metrics['acc_5']*n_total)}/{n_total})")
    print(f"  Acc @ ±10 frames  : {metrics['acc_10']:.1%}  "
          f"({int(metrics['acc_10']*n_total)}/{n_total})")
    print(f"  Acc @ ±15 frames  : {metrics['acc_15']:.1%}  "
          f"({int(metrics['acc_15']*n_total)}/{n_total})")
    print(f"  Acc @ ±20 frames  : {metrics['acc_20']:.1%}  "
          f"({int(metrics['acc_20']*n_total)}/{n_total})")

    json_path = out_dir / "task_b_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {json_path}")

    abs_errors = np.abs(gt_fpoc.astype(float) - pred_fpoc.astype(float))
    plot_error_distribution(abs_errors, out_dir / "task_b_error_distribution.png")


if __name__ == "__main__":
    main()

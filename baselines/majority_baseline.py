#!/usr/bin/env python3
"""
TackleNet Task A — Majority-Class Baseline (A0)

Predicts safe (0) for every clip. No video loading required.
Metrics are computed via eval.metrics.compute_task_a_metrics so that
baseline numbers are guaranteed consistent with evaluate_task_a.py.

Usage
-----
python baselines/majority_baseline.py \\
    --annot   splits/annotations_satt_fpoc.csv \\
    --splits  splits/splits_frozen.csv \\
    --split   test \\
    --out_dir .

Outputs
-------
majority_predictions.csv   clip_id, prob_risky, pred_label
                           (compatible with evaluate_task_a.py)
majority_metrics.json      Task A metrics on the requested split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import compute_task_a_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TackleNet Task A majority-class baseline"
    )
    p.add_argument("--annot",   required=True,
                   help="Path to annotations_satt_fpoc.csv")
    p.add_argument("--splits",  required=True,
                   help="Path to splits_frozen.csv")
    p.add_argument("--split",   default="test",
                   choices=["train", "val", "test"],
                   help="Split to evaluate (default: test)")
    p.add_argument("--out_dir", default=".",
                   help="Output directory (default: current dir)")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann    = pd.read_csv(args.annot)
    splits = pd.read_csv(args.splits)

    # Validate required columns
    for col in ("clip_id", "satt3_risk_binary"):
        if col not in ann.columns:
            raise ValueError(f"annotations CSV missing column: '{col}'")
    for col in ("clip_id", "split"):
        if col not in splits.columns:
            raise ValueError(f"splits CSV missing column: '{col}'")

    # Merge — left join so missing split assignments surface as errors
    data = ann.merge(splits[["clip_id", "split"]], on="clip_id", how="left")
    missing_split = data["split"].isna().sum()
    if missing_split > 0:
        raise ValueError(
            f"{missing_split} clip(s) in annotations have no split assignment"
        )

    subset = data[data["split"] == args.split].reset_index(drop=True)
    if len(subset) == 0:
        raise ValueError(f"No clips found for split='{args.split}'")

    clip_ids = subset["clip_id"].tolist()
    labels   = subset["satt3_risk_binary"].tolist()
    preds    = [0]   * len(labels)   # always predict safe
    scores   = [0.0] * len(labels)   # P(risky) = 0.0 for all clips

    la = np.array(labels, dtype=int)
    pa = np.array(preds,  dtype=int)
    sa = np.array(scores, dtype=float)

    # Use shared metric function — identical to what evaluate_task_a.py reports
    metrics = compute_task_a_metrics(la, pa, sa)
    metrics.update({
        "split":        args.split,
        "threshold":    0.5,
        "n_total":      int(len(la)),
        "n_risky_true": int(la.sum()),
        "n_risky_pred": int(pa.sum()),
    })

    # Predictions file — column names match evaluate_task_a.py expected format
    preds_path = out_dir / "majority_predictions.csv"
    pd.DataFrame({
        "clip_id":    clip_ids,
        "prob_risky": scores,
        "pred_label": preds,
    }).to_csv(preds_path, index=False)

    metrics_path = out_dir / "majority_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"=== A0 Majority Baseline ({args.split} split, n={metrics['n_total']}) ===")
    for k, v in metrics.items():
        print(f"  {k:22s}: {v}")
    print(f"\nSaved: {preds_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
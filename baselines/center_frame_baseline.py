#!/usr/bin/env python3
"""
TackleNet Task B — Center-Frame FPOC Baseline

Predicts FPOC = total_frames // 2 for every clip.
Operates zero-shot across all clips in the annotation file.
Coverage is 100% when all released videos are present and readable.

Usage
-----
python baselines/center_frame_baseline.py \\
    --annot     splits/annotations_satt_fpoc.csv \\
    --video_dir path/to/tacklenet_data/videos \\
    --out_dir   .

Outputs
-------
center_frame_predictions.csv   clip_id, pred_fpoc_frame
                                (compatible with eval/evaluate_task_b.py)
center_frame_results.csv        per-clip detail: clip_id, gt_fpoc,
                                pred_fpoc_frame, total_frames, abs_error
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def get_total_frames(video_path: Path) -> int | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total if total > 0 else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TackleNet Task B center-frame FPOC baseline"
    )
    p.add_argument("--annot",     required=True,
                   help="Path to annotations_satt_fpoc.csv")
    p.add_argument("--video_dir", required=True,
                   help="Directory containing MP4 clips")
    p.add_argument("--out_dir",   default=".",
                   help="Output directory (default: current dir)")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    out_dir   = Path(args.out_dir)
    video_dir = Path(args.video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(args.annot)

    for col in ("clip_id", "clip_filename", "fpoc_frame"):
        if col not in ann.columns:
            raise ValueError(f"annotations CSV missing column: '{col}'")

    print(f"Loaded {len(ann)} annotations")

    results = []
    failed  = []

    for _, row in ann.iterrows():
        clip_id    = int(row["clip_id"])
        gt_fpoc    = int(row["fpoc_frame"])
        video_path = video_dir / row["clip_filename"]

        if not video_path.exists():
            failed.append(clip_id)
            print(f"  WARNING: {video_path.name} not found")
            continue

        total = get_total_frames(video_path)
        if total is None:
            failed.append(clip_id)
            print(f"  WARNING: cannot read frame count for {video_path.name}")
            continue

        pred_fpoc = total // 2
        results.append({
            "clip_id":         clip_id,
            "gt_fpoc":         gt_fpoc,
            "pred_fpoc_frame": pred_fpoc,
            "total_frames":    total,
            "abs_error":       abs(pred_fpoc - gt_fpoc),
        })

    df      = pd.DataFrame(results)
    n_total = len(ann)
    n_ok    = len(df)

    print(f"\nProcessed : {n_ok}/{n_total} clips")
    if failed:
        print(f"Failed    : {len(failed)} clips — {failed}")

    print(f"\n{'='*60}")
    print("CENTER-FRAME BASELINE — TASK B")
    print(f"{'='*60}")
    print(f"  Coverage          : {n_ok}/{n_total} = {n_ok/n_total:.1%}")
    print(f"  Median Abs Error  : {df['abs_error'].median():.1f} frames")
    print(f"  Mean Abs Error    : {df['abs_error'].mean():.1f} frames")
    print()
    for tol in [5, 10, 15, 20]:
        correct = (df["abs_error"] <= tol).sum()
        print(f"  Acc @ ±{tol:2d}f : {correct}/{n_total} = {correct/n_total:.1%}")
    print(f"{'='*60}")

    # Predictions file — compatible with eval/evaluate_task_b.py
    preds_path = out_dir / "center_frame_predictions.csv"
    df[["clip_id", "pred_fpoc_frame"]].to_csv(preds_path, index=False)
    print(f"\nSaved predictions : {preds_path}")

    # Full per-clip detail
    detail_path = out_dir / "center_frame_results.csv"
    df.to_csv(detail_path, index=False)
    print(f"Saved detail      : {detail_path}")


if __name__ == "__main__":
    main()

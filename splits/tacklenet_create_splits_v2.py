#!/usr/bin/env python3
"""
Create a reproducible frozen candidate split for TackleNet.

Usage:
    python tacklenet_create_splits_v2.py \
        --input "annotations_satt_fpoc.csv" \
        --output "splits_frozen.csv" \
        --seed 42

What this script does
---------------------
1. Validates the annotation CSV
2. Creates clip_filename from clip_id
3. Generates a stratified clip-level 70/15/15 split on satt3_risk_binary
4. Writes the split CSV
5. Prints a concise audit summary

Notes
-----
- This version does NOT use session_id.
- The script uses the provided satt3_risk_binary column as the source-of-truth label.
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to annotation CSV")
    parser.add_argument("--output", default="splits_frozen.csv", help="Path to output split CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_cols = ["clip_id", "satt3_risk_binary"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["clip_id"].isnull().any():
        raise ValueError(f"clip_id has {int(df['clip_id'].isnull().sum())} missing values")
    if df["satt3_risk_binary"].isnull().any():
        raise ValueError(f"satt3_risk_binary has {int(df['satt3_risk_binary'].isnull().sum())} missing values")

    if df["clip_id"].duplicated().any():
        dupes = df.loc[df["clip_id"].duplicated(), "clip_id"].tolist()
        raise ValueError(f"Duplicate clip_id values found: {dupes[:20]}")

    label_values = sorted(df["satt3_risk_binary"].dropna().unique().tolist())
    if any(v not in [0, 1] for v in label_values):
        raise ValueError(f"satt3_risk_binary must be binary 0/1; found {label_values}")

    df = df.copy()
    df["clip_id"] = df["clip_id"].astype(int)
    df["satt3_risk_binary"] = df["satt3_risk_binary"].astype(int)

    width = max(3, len(str(df["clip_id"].max())))
    df["clip_filename"] = df["clip_id"].astype(str).str.zfill(width) + ".mp4"

    split_source = df[["clip_id", "clip_filename", "satt3_risk_binary"]].copy()

    train_df, temp_df = train_test_split(
        split_source,
        test_size=0.30,
        stratify=split_source["satt3_risk_binary"],
        random_state=args.seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["satt3_risk_binary"],
        random_state=args.seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    out_df = out_df[["clip_id", "clip_filename", "satt3_risk_binary", "split"]]
    out_df = out_df.sort_values("clip_id").reset_index(drop=True)

    if len(out_df) != len(split_source):
        raise RuntimeError("Row count changed after splitting")
    if out_df["clip_id"].nunique() != len(split_source):
        raise RuntimeError("clip_id uniqueness check failed")

    out_df.to_csv(args.output, index=False)

    print(f"Saved split file to: {args.output}")
    print("\nSplit sizes:")
    print(out_df["split"].value_counts().reindex(["train", "val", "test"]))
    print("\nPer-split class counts:")
    print(out_df.groupby(["split", "satt3_risk_binary"]).size().unstack(fill_value=0).reindex(["train", "val", "test"]))


if __name__ == "__main__":
    main()

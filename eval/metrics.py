"""
eval/metrics.py — Shared metric utilities for TackleNet Task A and Task B.

All scripts in this repo import from here. Do not compute metrics
inline in baselines or evaluators — use these functions to keep
reported numbers consistent across all tools.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ── Task A ────────────────────────────────────────────────────────────────────

def compute_task_a_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_risky: np.ndarray,
) -> dict:
    """
    Compute all Task A (Risk Classification) metrics.

    Parameters
    ----------
    y_true     : (N,) int array, ground-truth labels (0=safe, 1=risky)
    y_pred     : (N,) int array, predicted labels
    prob_risky : (N,) float array, predicted risky-class probabilities

    Returns
    -------
    dict with keys: macro_f1, balanced_accuracy, risky_recall,
                    risky_precision, risky_f1, pr_auc
    pr_auc is None if y_true contains only one class.
    """
    y_true     = np.asarray(y_true,     dtype=int)
    y_pred     = np.asarray(y_pred,     dtype=int)
    prob_risky = np.asarray(prob_risky, dtype=float)

    if len(y_true) != len(y_pred) or len(y_true) != len(prob_risky):
        raise ValueError(
            f"y_true, y_pred, and prob_risky must have the same length; "
            f"got {len(y_true)}, {len(y_pred)}, {len(prob_risky)}"
        )

    n_classes = len(np.unique(y_true))
    pr_auc = (
        round(float(average_precision_score(y_true, prob_risky)), 4)
        if n_classes >= 2
        else None
    )

    return {
        "macro_f1":          round(float(f1_score(y_true, y_pred, average="macro",  zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "risky_recall":      round(float(recall_score(y_true,    y_pred, pos_label=1, zero_division=0)), 4),
        "risky_precision":   round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "risky_f1":          round(float(f1_score(y_true,        y_pred, pos_label=1, zero_division=0)), 4),
        "pr_auc":            pr_auc,
    }


def classification_report_str(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["safe (0)", "risky (1)"],
        zero_division=0,
    )


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


# ── Task B ────────────────────────────────────────────────────────────────────

def compute_task_b_metrics(
    gt_fpoc: np.ndarray,
    pred_fpoc: np.ndarray,
    n_total: int,
) -> dict:
    """
    Compute all Task B (Contact Onset Localization) metrics.

    Acc@k is end-to-end over all n_total clips. Missing predictions
    count as incorrect (they are excluded from gt_fpoc / pred_fpoc
    by the caller, so the denominator stays n_total).

    Parameters
    ----------
    gt_fpoc   : (N,) int array, ground-truth FPOC frames for covered clips
    pred_fpoc : (N,) int array, predicted FPOC frames for covered clips
    n_total   : total clips in the evaluated set (coverage denominator)

    Returns
    -------
    dict with keys: coverage, median_abs_error, mae,
                    acc_5, acc_10, acc_15, acc_20, n_covered, n_total
    median_abs_error and mae are None when n_covered == 0.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive; got {n_total}")

    gt_fpoc   = np.asarray(gt_fpoc,   dtype=int)
    pred_fpoc = np.asarray(pred_fpoc, dtype=int)

    if len(gt_fpoc) != len(pred_fpoc):
        raise ValueError(
            f"gt_fpoc and pred_fpoc must have the same length; "
            f"got {len(gt_fpoc)} and {len(pred_fpoc)}"
        )

    n_covered = len(gt_fpoc)

    if n_covered == 0:
        return {
            "coverage":         0.0,
            "median_abs_error": None,
            "mae":              None,
            "acc_5":            0.0,
            "acc_10":           0.0,
            "acc_15":           0.0,
            "acc_20":           0.0,
            "n_covered":        0,
            "n_total":          int(n_total),
        }

    abs_err = np.abs(gt_fpoc.astype(float) - pred_fpoc.astype(float))
    return {
        "coverage":         round(n_covered / n_total, 4),
        "median_abs_error": round(float(np.median(abs_err)), 4),
        "mae":              round(float(np.mean(abs_err)), 4),
        "acc_5":            round(float((abs_err <=  5).sum() / n_total), 4),
        "acc_10":           round(float((abs_err <= 10).sum() / n_total), 4),
        "acc_15":           round(float((abs_err <= 15).sum() / n_total), 4),
        "acc_20":           round(float((abs_err <= 20).sum() / n_total), 4),
        "n_covered":        int(n_covered),
        "n_total":          int(n_total),
    }
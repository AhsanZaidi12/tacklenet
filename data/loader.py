#!/usr/bin/env python3
"""
data/loader.py — TackleNet PyTorch Dataset

Loads de-identified MP4 clips and annotations for TackleNet benchmark use,
including Task A risk classification and Task C contact-window analysis.
FPOC metadata can also be returned for Task B localization evaluation.

Two input modes
---------------
full  — uniformly subsample the entire clip to num_frames
fpoc  — extract the locked 32-frame asymmetric FPOC window
        (pre=23 frames + FPOC frame + post=8 frames)

FPOC window boundary handling (locked convention)
-------------------------------------------------
- Indices before frame 0   → repeat frame 0        (repeat-first)
- Indices past last frame  → repeat last frame      (repeat-last)
- Never zero-pad: black frames degrade temporal attention in
  Kinetics-pretrained transformers.

Frame array shape returned: (T, H, W, 3) uint8 numpy array (RGB).
Normalization and tensor conversion are left to the caller's transform
or model processor, keeping this loader framework-agnostic.

Usage
-----
from data.loader import TackleNetDataset

dataset = TackleNetDataset(
    annotations_csv="splits/annotations_satt_fpoc.csv",
    splits_csv="splits/splits_frozen.csv",
    video_dir="path/to/videos",
    split="train",         # "train" | "val" | "test"
    mode="full",           # "full"  | "fpoc"
    num_frames=32,
    image_size=224,
)

# Each item is a dict: clip_id (int), frames (T,H,W,3) uint8, label (int)
item = dataset[0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# ── Locked benchmark constants ────────────────────────────────────────────────

FPOC_PRE        = 23   # frames before FPOC anchor (locked for TackleNet)
FPOC_POST       = 8    # frames after  FPOC anchor (locked for TackleNet)
FPOC_WINDOW_LEN = FPOC_PRE + 1 + FPOC_POST   # = 32

DEFAULT_NUM_FRAMES = 32
DEFAULT_IMAGE_SIZE = 224


# ── Video loading helpers ─────────────────────────────────────────────────────

def _decode_video(path: str) -> list[np.ndarray]:
    """Decode all frames from an MP4. Returns list of (H, W, 3) RGB uint8 arrays."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Decoded zero frames from: {path}")
    return frames


def _resize_frames(frames: list[np.ndarray], size: int) -> list[np.ndarray]:
    return [cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in frames]


def _repeat_last_pad(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    """Extend frame list to target length by repeating the last frame."""
    while len(frames) < target:
        frames.append(frames[-1].copy())
    return frames


def _uniform_subsample(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    """Uniformly subsample frame list to exactly target frames."""
    indices = np.linspace(0, len(frames) - 1, target, dtype=int)
    return [frames[i] for i in indices]


def load_full_clip(
    path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Load a full clip, subsample to num_frames, resize to image_size.

    Short clips (fewer frames than num_frames) are padded with
    repeat-last before subsampling.

    Returns
    -------
    np.ndarray of shape (num_frames, image_size, image_size, 3), dtype uint8
    """
    frames = _decode_video(path)
    frames = _resize_frames(frames, image_size)
    if len(frames) < num_frames:
        frames = _repeat_last_pad(frames, num_frames)
    if len(frames) > num_frames:
        frames = _uniform_subsample(frames, num_frames)
    return np.stack(frames)  # (T, H, W, 3)


def load_fpoc_window(
    path: str,
    fpoc_frame: int,
    pre: int = FPOC_PRE,
    post: int = FPOC_POST,
    num_frames: int = FPOC_WINDOW_LEN,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Extract the locked asymmetric FPOC window from a clip.

    The window targets pre+1+post frames with FPOC at index `pre`.
    Boundary handling: index clamping (repeat-first / repeat-last).
    The FPOC anchor is always preserved at window position `pre`.

    If num_frames != pre+1+post (e.g. TimeSformer uses 8 frames),
    the natural window is uniformly subsampled to num_frames.

    Returns
    -------
    np.ndarray of shape (num_frames, image_size, image_size, 3), dtype uint8
    """
    frames = _decode_video(path)
    frames = _resize_frames(frames, image_size)
    T = len(frames)

    if fpoc_frame < 0 or fpoc_frame >= T:
        raise ValueError(
            f"fpoc_frame={fpoc_frame} out of range [0, {T - 1}] for {path}"
        )

    natural_len = pre + 1 + post
    raw_indices = np.arange(fpoc_frame - pre, fpoc_frame + post + 1)  # length = natural_len
    clamped     = np.clip(raw_indices, 0, T - 1)
    window      = [frames[i] for i in clamped]

    assert len(window) == natural_len
    assert np.array_equal(window[pre], frames[fpoc_frame]), \
        f"FPOC frame not at window index {pre}"

    if num_frames != natural_len:
        window = _uniform_subsample(window, num_frames)

    return np.stack(window)  # (num_frames, H, W, 3)


# ── Dataset class ─────────────────────────────────────────────────────────────

class TackleNetDataset(Dataset):
    """
    PyTorch Dataset for TackleNet benchmark clips.

    Parameters
    ----------
    annotations_csv : path to annotations_satt_fpoc.csv
    splits_csv      : path to splits_frozen.csv
    video_dir       : directory containing MP4 files (e.g. videos/)
    split           : "train", "val", or "test"
    mode            : "full" (entire clip) or "fpoc" (32-frame contact window)
    num_frames      : frames per clip fed to the model (default 32)
    image_size      : spatial resize in pixels (default 224)
    transform       : optional callable applied to the (T,H,W,3) uint8 array
                      before returning; use for normalization and tensor
                      conversion specific to your model's preprocessor.

    Item returned
    -------------
    dict with keys:
        clip_id (int)        : clip identifier
        frames  (np.ndarray) : (num_frames, image_size, image_size, 3) uint8
        label   (int)        : 0 = safe, 1 = risky  (satt3_risk_binary)
    """

    def __init__(
        self,
        annotations_csv: str,
        splits_csv: str,
        video_dir: str,
        split: str,
        mode: str = "full",
        num_frames: int = DEFAULT_NUM_FRAMES,
        image_size: int = DEFAULT_IMAGE_SIZE,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")
        if mode not in ("full", "fpoc"):
            raise ValueError(f"mode must be 'full' or 'fpoc'; got '{mode}'")

        ann    = pd.read_csv(annotations_csv)
        splits = pd.read_csv(splits_csv)

        # Validate required columns
        for col in ("clip_id", "clip_filename", "satt3_risk_binary", "fpoc_frame"):
            if col not in ann.columns:
                raise ValueError(f"annotations_csv missing column: '{col}'")
        for col in ("clip_id", "split"):
            if col not in splits.columns:
                raise ValueError(f"splits_csv missing column: '{col}'")

        data = ann.merge(splits[["clip_id", "split"]], on="clip_id", how="left")
        missing = data["split"].isna().sum()
        if missing > 0:
            raise ValueError(f"{missing} clip(s) have no split assignment")

        self.records = (
            data[data["split"] == split]
            .reset_index(drop=True)
            .copy()
        )

        if len(self.records) == 0:
            raise ValueError(f"No records found for split='{split}'")

        if mode == "fpoc":
            missing_fpoc = self.records["fpoc_frame"].isna().sum()
            if missing_fpoc > 0:
                bad = self.records.loc[
                    self.records["fpoc_frame"].isna(), "clip_id"
                ].tolist()
                raise ValueError(
                    f"mode='fpoc' requires fpoc_frame for all clips; "
                    f"{missing_fpoc} missing (clip_ids: {bad[:10]})"
                )

        self.video_dir  = Path(video_dir)
        self.mode       = mode
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row   = self.records.iloc[idx]
        path  = str(self.video_dir / row["clip_filename"])
        label = int(row["satt3_risk_binary"])

        if self.mode == "full":
            frames = load_full_clip(path, self.num_frames, self.image_size)
        else:
            frames = load_fpoc_window(
                path,
                fpoc_frame=int(row["fpoc_frame"]),
                num_frames=self.num_frames,
                image_size=self.image_size,
            )

        if self.transform is not None:
            frames = self.transform(frames)

        return {
            "clip_id": int(row["clip_id"]),
            "frames":  frames,
            "label":   label,
        }

    def __repr__(self) -> str:
        return (
            f"TackleNetDataset(split='{self.records['split'].iloc[0]}', "
            f"n={len(self)}, mode='{self.mode}', "
            f"num_frames={self.num_frames}, image_size={self.image_size})"
        )

"""Loading the competition annotations into flat tables and building clip-level folds.

Conventions used across the project:
- class ids are 0-based (0=person ... 7=motor), as in the submission format;
- boxes are absolute pixels [x, y, w, h] with (x, y) the top-left corner;
- every frame has a string key "<clip>_<frame:07d>" (matches test image_id).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

COMP = "multiple-object-video-detection-hw-2-ucu-ai-2026"


def find_data_root() -> Path:
    """Competition data root: ./data locally, the mounted input on Kaggle."""
    env = os.environ.get("HW2_DATA")
    if env:
        return Path(env)
    candidates = ["data", "../data", f"/kaggle/input/{COMP}", f"/kaggle/input/competitions/{COMP}"]
    candidates += glob.glob(f"/kaggle/input/*/{COMP}") + glob.glob(f"/kaggle/input/*/*/{COMP}")
    for c in candidates:
        if (Path(c) / "test_images.csv").exists():
            return Path(c).resolve()
    raise FileNotFoundError("competition data not found; set HW2_DATA")


def frame_key(clip: str, frame: int) -> str:
    return f"{clip}_{int(frame):07d}"


def load_split(root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (frames, boxes) for 'train' or 'val'.

    frames: key, split, clip, frame, path, width, height
    boxes:  key, cls, x, y, w, h, ignore, occlusion, truncation, track_id, orig_cls
            ignore=1 rows are "don't care" regions (crowds, 'others', ...); their cls is meaningless.
    """
    coco = json.loads((root / split / "annotations_coco.json").read_text())
    frames = pd.DataFrame(coco["images"])
    frames["key"] = [frame_key(v, f) for v, f in zip(frames.video, frames.frame_id)]
    frames["path"] = [str(root / split / fn) for fn in frames.file_name]
    frames = frames.rename(columns={"video": "clip", "frame_id": "frame"})
    frames["split"] = split

    ann = pd.DataFrame(coco["annotations"])
    id2key = dict(zip(frames.id, frames.key))
    xywh = np.array(ann.bbox.tolist(), dtype=np.float32)
    boxes = pd.DataFrame({
        "key": ann.image_id.map(id2key),
        "cls": ann.category_id - 1,
        "x": xywh[:, 0], "y": xywh[:, 1], "w": xywh[:, 2], "h": xywh[:, 3],
        "ignore": ann.iscrowd.astype(int),
        "occlusion": ann.get("occlusion", pd.Series(-1, index=ann.index)).fillna(-1).astype(int),
        "truncation": ann.get("truncation", pd.Series(-1, index=ann.index)).fillna(-1).astype(int),
        "track_id": ann.get("track_id", pd.Series(-1, index=ann.index)).fillna(-1).astype(int),
        "orig_cls": ann.orig_category_id.astype(int),
    })
    frames = frames[["key", "split", "clip", "frame", "path", "width", "height"]]
    return frames, boxes


def load_all(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and val pooled together: folds are re-drawn over all 63 labelled clips."""
    root = root or find_data_root()
    parts = [load_split(root, s) for s in ("train", "val")]
    frames = pd.concat([p[0] for p in parts], ignore_index=True)
    boxes = pd.concat([p[1] for p in parts], ignore_index=True)
    return frames, boxes


def load_test(root: Path | None = None) -> pd.DataFrame:
    root = root or find_data_root()
    t = pd.read_csv(root / "test_images.csv")
    t["key"] = t.image_id
    t["path"] = [str(root / f) for f in t.file_name]
    t["split"] = "test"
    return t[["key", "split", "clip", "frame", "path", "width", "height"]]


def make_folds(frames: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Clip-level GroupKFold: all frames of a sequence land in the same fold.

    GroupKFold runs on frames grouped by clip, so it balances the number of frames per fold.
    """
    clips = frames.groupby("clip").size().rename("n_frames").reset_index()
    clip_fold = {}
    gkf = GroupKFold(n_splits=n_folds)
    for f, (_, va) in enumerate(gkf.split(frames, groups=frames["clip"])):
        clip_fold.update({c: f for c in frames["clip"].iloc[va].unique()})
    clips["fold"] = clips["clip"].map(clip_fold)
    return clips[["clip", "n_frames", "fold"]]


def subsample(frames: pd.DataFrame, step: int) -> pd.DataFrame:
    """Keep every `step`-th frame of each clip (frames are consecutive, 1-based)."""
    return frames[(frames.frame - 1) % step == 0]


def fold_frames(frames: pd.DataFrame, folds: pd.DataFrame, fold: int, inside: bool = True) -> pd.DataFrame:
    """Frames whose clip is (inside=True) or is not (inside=False) in `fold`."""
    clips = set(folds.loc[folds["fold"] == fold, "clip"])
    return frames[frames["clip"].isin(clips) == inside]

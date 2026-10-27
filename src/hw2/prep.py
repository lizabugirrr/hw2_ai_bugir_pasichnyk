"""Builds the training image set shared by all frameworks.

For every kept frame (every `step`-th frame of all 63 labelled clips):
- resize so the long side is <= 1920 (the largest test resolution);
- fill ignored regions with gray (114) *except* where they overlap a valid box,
  so the detector is never shown unlabelled objects as background;
- write YOLO labels (class cx cy w h, normalised) for the non-ignored boxes.

Evaluation never uses these images: it runs on the original frames and handles
ignored regions inside the COCO evaluator (see evaluate.py).

Usage: python -m hw2.prep --out prep --step 2 [--clips N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import find_data_root, load_all, make_folds, subsample

MAX_SIDE = 1920
FILL = 114


def _process(args):
    path, out_img, out_lbl, valid, ignore = args
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]

    if len(ignore):
        mask = np.zeros((H, W), np.uint8)
        for x, y, bw, bh in ignore * s:
            mask[int(y):int(np.ceil(y + bh)), int(x):int(np.ceil(x + bw))] = 1
        for _, x, y, bw, bh in valid:
            mask[int(y * s):int(np.ceil((y + bh) * s)), int(x * s):int(np.ceil((x + bw) * s))] = 0
        img[mask.astype(bool)] = FILL
    cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    lines = []
    for c, x, y, bw, bh in valid:
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
    Path(out_lbl).write_text("\n".join(lines) + ("\n" if lines else ""))
    return W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prep")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--clips", type=int, default=0, help="debug: only first N clips")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    root = find_data_root()
    frames, boxes = load_all(root)
    folds = make_folds(frames)
    if a.clips:
        frames = frames[frames["clip"].isin(sorted(frames["clip"].unique())[: a.clips])]
    frames = subsample(frames, a.step).merge(folds[["clip", "fold"]], on="clip")

    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    by_key = {k: g for k, g in boxes[boxes.key.isin(frames.key)].groupby("key")}
    empty = boxes.iloc[:0]
    jobs = []
    for r in frames.itertuples():
        g = by_key.get(r.key, empty)
        valid = g.loc[g.ignore == 0, ["cls", "x", "y", "w", "h"]].to_numpy(np.float32)
        ignore = g.loc[g.ignore == 1, ["x", "y", "w", "h"]].to_numpy(np.float32)
        jobs.append((r.path, str(out / "images" / f"{r.key}.jpg"), str(out / "labels" / f"{r.key}.txt"), valid, ignore))

    with ProcessPoolExecutor(a.workers) as ex:
        sizes = list(ex.map(_process, jobs, chunksize=16))
    frames["prep_w"], frames["prep_h"] = zip(*sizes)
    frames["prep_path"] = [f"images/{k}.jpg" for k in frames.key]
    frames.to_csv(out / "frames.csv", index=False)
    folds.to_csv(out / "folds.csv", index=False)
    print(f"wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()

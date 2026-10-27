"""Inference strategies for trained Ultralytics models (used by the final pipeline).

- full frame at a chosen `imgsz`;
- optional SAHI-style slicing (Akyon et al., 2022): an n x n grid of overlapping tiles, each
  resized to `tile_imgsz`, merged with the full-frame pass by class-wise NMS;
- optional test-time augmentation (Ultralytics `augment=True`: flips + scales).

Usage:
  python -m hw2.infer --weights runs/x/train/weights/last.pt --out runs/x --tag t2 --tiles 2 --split val test
Writes <out>/<split>_preds_<tag>.parquet.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms

from .data import find_data_root, fold_frames, load_all, load_test, make_folds, subsample


def tiles_for(w: int, h: int, n: int, overlap: float):
    """n x n grid of tiles covering the frame with relative overlap."""
    tw, th = int(np.ceil(w / (n - (n - 1) * overlap))), int(np.ceil(h / (n - (n - 1) * overlap)))
    xs = np.linspace(0, w - tw, n).astype(int)
    ys = np.linspace(0, h - th, n).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def run(model, frames: pd.DataFrame, imgsz: int, tiles: int, tile_imgsz: int, overlap: float,
        tta: bool, conf: float, iou: float, merge_iou: float) -> pd.DataFrame:
    out = []
    kw = dict(conf=conf, iou=iou, max_det=500, half=True, verbose=False, augment=tta)
    for r in frames.itertuples():
        img = cv2.imread(r.path)
        H, W = img.shape[:2]
        parts = []
        res = model.predict(img, imgsz=imgsz, **kw)[0].boxes
        parts.append((res.xyxy.cpu(), res.conf.cpu(), res.cls.cpu()))
        if tiles > 1:
            crops = tiles_for(W, H, tiles, overlap)
            batch = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in crops]
            for (x0, y0, _, _), rr in zip(crops, model.predict(batch, imgsz=tile_imgsz, **kw)):
                b = rr.boxes
                xy = b.xyxy.cpu() + torch.tensor([x0, y0, x0, y0], dtype=torch.float32)
                parts.append((xy, b.conf.cpu(), b.cls.cpu()))
        xy = torch.cat([p[0] for p in parts]); sc = torch.cat([p[1] for p in parts]); cl = torch.cat([p[2] for p in parts])
        if tiles > 1 and len(xy):
            keep = batched_nms(xy.float(), sc.float(), cl.long(), merge_iou)[:500]
            xy, sc, cl = xy[keep], sc[keep], cl[keep]
        xy = xy.numpy()
        out.append(pd.DataFrame({"key": r.key, "cls": cl.numpy().astype(int), "score": sc.numpy(),
                                 "x": xy[:, 0], "y": xy[:, 1], "w": xy[:, 2] - xy[:, 0], "h": xy[:, 3] - xy[:, 1]}))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=1344)
    ap.add_argument("--tiles", type=int, default=1)
    ap.add_argument("--tile_imgsz", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--merge_iou", type=float, default=0.6)
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--val_step", type=int, default=4)
    a = ap.parse_args()

    from .ultra_train import load_model
    model = load_model(a.weights, None)
    root = find_data_root()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for split in a.split:
        if split == "test":
            frames = load_test(root)
        else:
            fr, _ = load_all(root)
            frames = subsample(fold_frames(fr, make_folds(fr), a.val_fold), a.val_step)
        t = time.time()
        p = run(model, frames, a.imgsz, a.tiles, a.tile_imgsz, a.overlap, a.tta, a.conf, a.iou, a.merge_iou)
        p.to_parquet(out / f"{split}_preds_{a.tag}.parquet")
        info = {**vars(a), "split": split, "n_frames": len(frames), "sec_per_frame": round((time.time() - t) / len(frames), 4)}
        (out / f"{split}_preds_{a.tag}.json").write_text(json.dumps(info, indent=1))
        print(split, info["sec_per_frame"], "s/frame", flush=True)


if __name__ == "__main__":
    main()

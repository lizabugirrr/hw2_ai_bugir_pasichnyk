"""Training / evaluation of Ultralytics detectors (YOLO = anchor-free, RT-DETR = query-based).

Same protocol as tv_train.py: train on prepared frames of folds != val_fold (every
`step`-th frame), evaluate our own COCO metric on original frames of val_fold.
Ultralytics' own per-epoch validation runs on the masked val frames and is only used
for training curves.

Usage:
  python -m hw2.ultra_train --model yolo11s.pt --name yolo_base --prep PREP --out runs \
      --cfg '{"imgsz": 1344}' [--hw2 '{"nwd_ratio": 0.5}']
--cfg goes to model.train(); --hw2 holds options handled here (see EXTRA).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import find_data_root, fold_frames, load_all, load_test, make_folds, subsample
from .evaluate import build_gt, evaluate

TRAIN_DEFAULTS = dict(
    epochs=12, imgsz=1344, batch=8, workers=2, seed=0, deterministic=False, close_mosaic=3,
    patience=100, plots=True, max_det=500, cache=False, amp=True,
)
EXTRA = dict(
    val_fold=0, step=10, val_step=4, curve_step=20,
    weights=None,          # load these weights into a custom model yaml
    test_imgsz=None,       # defaults to imgsz
    conf=0.001, iou=0.6,   # NMS settings for prediction (ignored by NMS-free heads)
    nwd_ratio=0.0, nwd_c=32.0,  # box loss = (1-r)*(1-CIoU) + r*(1-NWD)
    tal_topk=None,         # task-aligned assigner top-k (Ultralytics default 10)
    matcher_cost=None,     # RT-DETR Hungarian matcher cost gains, e.g. {"class": 2, "bbox": 5, "giou": 2}
)


# ---------------------------------------------------------------- loss patches
def nwd(b1: torch.Tensor, b2: torch.Tensor, c: float) -> torch.Tensor:
    """Normalized Wasserstein Distance between xyxy boxes (Wang et al., 2021)."""
    cx1, cy1 = (b1[:, 0] + b1[:, 2]) / 2, (b1[:, 1] + b1[:, 3]) / 2
    cx2, cy2 = (b2[:, 0] + b2[:, 2]) / 2, (b2[:, 1] + b2[:, 3]) / 2
    w1, h1 = b1[:, 2] - b1[:, 0], b1[:, 3] - b1[:, 1]
    w2, h2 = b2[:, 2] - b2[:, 0], b2[:, 3] - b2[:, 1]
    w2_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    return torch.exp(-torch.sqrt(w2_dist.clamp(min=1e-7)) / c)


def patch_losses(extra: dict):
    from ultralytics.utils import loss as uloss
    from ultralytics.utils.metrics import bbox_iou

    r, c = extra["nwd_ratio"], extra["nwd_c"]
    if r > 0:
        orig = uloss.BboxLoss.forward

        def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                    target_scores_sum, fg_mask, imgsz, stride):
            loss_iou, loss_dfl = orig(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                      target_scores, target_scores_sum, fg_mask, imgsz, stride)
            idx = fg_mask.nonzero(as_tuple=True)
            weight = target_scores[idx].sum(-1, keepdim=True)
            s = stride[idx[1]]  # boxes are in grid units of their level -> pixels
            n = nwd(pred_bboxes[idx] * s, target_bboxes[idx] * s, c).unsqueeze(-1)
            loss_nwd = ((1.0 - n) * weight).sum() / target_scores_sum
            return (1 - r) * loss_iou + r * loss_nwd, loss_dfl

        uloss.BboxLoss.forward = forward

    if extra["tal_topk"]:
        d = uloss.v8DetectionLoss.__init__.__defaults__
        uloss.v8DetectionLoss.__init__.__defaults__ = (extra["tal_topk"],) + d[1:]

    if extra["matcher_cost"]:
        from ultralytics.models.utils import ops as rtops
        orig_init = rtops.HungarianMatcher.__init__

        def init(self, *a, **kw):
            kw["cost_gain"] = extra["matcher_cost"]
            orig_init(self, *a, **kw)

        rtops.HungarianMatcher.__init__ = init


# ---------------------------------------------------------------- inference
def predict(model, frames: pd.DataFrame, imgsz: int, extra: dict, batch: int = 8) -> pd.DataFrame:
    out = []
    paths, keys = frames.path.tolist(), frames.key.tolist()
    for i in range(0, len(paths), batch):
        res = model.predict(paths[i:i + batch], imgsz=imgsz, conf=extra["conf"], iou=extra["iou"],
                            max_det=500, half=True, verbose=False)
        for k, r in zip(keys[i:i + batch], res):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()
            out.append(pd.DataFrame({
                "key": k, "cls": b.cls.cpu().numpy().astype(int), "score": b.conf.cpu().numpy(),
                "x": xyxy[:, 0], "y": xyxy[:, 1], "w": xyxy[:, 2] - xyxy[:, 0], "h": xyxy[:, 3] - xyxy[:, 1]}))
    return pd.concat(out, ignore_index=True)


def measure_fps(model, frames: pd.DataFrame, imgsz: int, extra: dict, n: int = 100) -> float:
    """Batch-1 latency incl. letterbox/normalise/NMS, excluding JPEG decoding (same as tv_train)."""
    import cv2
    imgs = [cv2.imread(p) for p in frames.path.iloc[:n]]
    times = []
    for i, im in enumerate(imgs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.predict(im, imgsz=imgsz, conf=extra["conf"], iou=extra["iou"], max_det=500, half=True, verbose=False)
        torch.cuda.synchronize()
        if i >= 10:
            times.append(time.perf_counter() - t)
    return round(1 / float(np.mean(times)), 2)


def load_model(name: str, weights: str | None):
    from ultralytics import RTDETR, YOLO
    cls = RTDETR if "rtdetr" in Path(name).name else YOLO
    m = cls(name)
    if weights:
        m.load(weights)
    return m


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="weights (yolo11s.pt, rtdetr-l.pt) or model yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--cfg", default="{}", help="JSON overrides for model.train()")
    ap.add_argument("--hw2", default="{}", help="JSON overrides of EXTRA")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max_train", type=int, default=0, help="debug: cap number of train frames")
    ap.add_argument("--no_test", action="store_true")
    a = ap.parse_args()

    tcfg = {**TRAIN_DEFAULTS, **json.loads(a.cfg)}
    extra = {**EXTRA, **json.loads(a.hw2)}
    out = (Path(a.out) / a.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": a.model, "train": tcfg, "hw2": extra}, indent=1))
    random.seed(tcfg["seed"]); np.random.seed(tcfg["seed"]); torch.manual_seed(tcfg["seed"])

    # dataset lists: Ultralytics finds labels by replacing /images/ with /labels/
    prep = Path(a.prep).resolve()
    pf = pd.read_csv(prep / "frames.csv")
    train = subsample(pf[pf.fold != extra["val_fold"]], extra["step"])
    if a.max_train:
        train = train.sample(a.max_train, random_state=0)
    curve_masked = subsample(pf[pf.fold == extra["val_fold"]], extra["curve_step"])
    (out / "train.txt").write_text("\n".join(str(prep / p) for p in train.prep_path) + "\n")
    (out / "val.txt").write_text("\n".join(str(prep / p) for p in curve_masked.prep_path) + "\n")
    from . import CLASSES
    (out / "data.yaml").write_text(json.dumps({
        "path": str(out), "train": str(out / "train.txt"), "val": str(out / "val.txt"),
        "nc": len(CLASSES), "names": CLASSES}))

    frames, boxes = load_all(find_data_root())
    folds = make_folds(frames)
    val = subsample(fold_frames(frames, folds, extra["val_fold"]), extra["val_step"])
    if a.max_train:
        val = val.iloc[:: max(1, len(val) // 40)]
    print(f"train {len(train)} frames, val {len(val)} frames", flush=True)

    patch_losses(extra)
    model = load_model(a.model, extra["weights"])
    t0 = time.time()
    model.train(data=str(out / "data.yaml"), project=str(out), name="train", exist_ok=True,
                device=a.device, **tcfg)
    train_min = round((time.time() - t0) / 60, 1)

    best = out / "train" / "weights" / "best.pt"
    last = out / "train" / "weights" / "last.pt"
    model = load_model(str(last if last.exists() else best), None)  # fixed budget: use the final epoch
    imgsz = extra["test_imgsz"] or tcfg["imgsz"]
    vp = predict(model, val, imgsz, extra)
    vp.to_parquet(out / "val_preds.parquet")
    metrics = evaluate(vp, build_gt(val, boxes), per_class=True)
    metrics.update(fps=measure_fps(model, val, imgsz, extra), train_min=train_min,
                   n_train=len(train), n_val=len(val), gpu=torch.cuda.get_device_name(0))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("VAL", metrics, flush=True)
    if not a.no_test:
        predict(model, load_test(), imgsz, extra).to_parquet(out / "test_preds.parquet")
    print("done", flush=True)


if __name__ == "__main__":
    main()

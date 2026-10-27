"""COCO-style evaluation on original frames, with ignored regions as "don't care".

Predictions everywhere are a DataFrame with columns key, cls, score, x, y, w, h
(original-image pixels, 0-based class ids).

pycocotools only matches a crowd (iscrowd=1) box against detections of the *same*
category, so each ignored region is replicated once per class. A detection whose area
lies mostly inside an ignored region is then matched to it and neither rewarded nor
penalised. maxDets is raised to 500 to match the competition (up to 500 boxes per image
are scored; dense frames have up to ~180 objects, so the COCO default of 100 would cap recall).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from . import CLASSES, NUM_CLASSES

MAX_DETS = [1, 100, 500]


def build_gt(frames: pd.DataFrame, boxes: pd.DataFrame) -> COCO:
    img_ids = {k: i + 1 for i, k in enumerate(frames.key)}
    images = [{"id": img_ids[r.key], "width": int(r.width), "height": int(r.height)} for r in frames.itertuples()]
    b = boxes[boxes.key.isin(img_ids)]
    anns = []
    for r in b.itertuples():
        cats = range(NUM_CLASSES) if r.ignore else [r.cls]
        for c in cats:
            anns.append({
                "id": len(anns) + 1, "image_id": img_ids[r.key], "category_id": int(c) + 1,
                "bbox": [float(r.x), float(r.y), float(r.w), float(r.h)],
                "area": float(r.w * r.h), "iscrowd": int(r.ignore),
            })
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    gt.key2id = img_ids
    return gt


def evaluate(preds: pd.DataFrame, gt: COCO, per_class: bool = False) -> dict:
    """Returns mAP (0.5:0.95), AP50, AP75, AP_small/medium/large, AR@500 [, per-class AP]."""
    p = preds[preds.key.isin(gt.key2id)]
    p = p.sort_values("score", ascending=False).groupby("key").head(MAX_DETS[-1])
    if len(p) == 0:
        return {"mAP": 0.0}
    dets = np.column_stack([
        p.key.map(gt.key2id).to_numpy(), p.x, p.y, p.w, p.h, p.score, p.cls.to_numpy() + 1,
    ]).astype(np.float64)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = MAX_DETS
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    res = {"mAP": s[0], "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4],
           "AP_large": s[5], "AR500": s[8]}
    if per_class:
        prec = ev.eval["precision"][:, :, :, 0, -1]  # [iou, recall, cls] at area=all, maxDets=500
        for c, name in enumerate(CLASSES):
            v = prec[:, :, c]
            res[f"AP_{name}"] = float(v[v > -1].mean()) if (v > -1).any() else float("nan")
    return {k: round(float(v), 4) for k, v in res.items()}

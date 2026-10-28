"""Helpers for result analysis: loading runs, error breakdown, drawing predictions."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import CLASSES
from .postprocess import iou_matrix

RUNS = Path(__file__).resolve().parents[2] / "runs"


def load_run(name: str, runs: Path = RUNS) -> dict:
    d = runs / name
    r = {"name": name, "config": json.loads((d / "config.json").read_text())}
    for f, key in [("metrics.json", "metrics")]:
        if (d / f).exists():
            r[key] = json.loads((d / f).read_text())
    if (d / "history.csv").exists():
        r["history"] = pd.read_csv(d / "history.csv")
    elif (d / "train" / "results.csv").exists():  # Ultralytics
        h = pd.read_csv(d / "train" / "results.csv")
        h.columns = [c.strip() for c in h.columns]
        r["history"] = h
    if (d / "val_preds.parquet").exists():
        r["val_preds"] = pd.read_parquet(d / "val_preds.parquet")
    return r


def metrics_table(names: list[str], runs: Path = RUNS) -> pd.DataFrame:
    rows = []
    for n in names:
        p = runs / n / "metrics.json"
        if p.exists():
            rows.append({"run": n, **json.loads(p.read_text())})
    cols = ["run", "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR500", "fps", "train_min"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df]].set_index("run") if len(df) else df


def _xyxy(df):
    return np.stack([df.x, df.y, df.x + df.w, df.y + df.h], 1) if len(df) else np.zeros((0, 4))


def error_breakdown(preds: pd.DataFrame, boxes: pd.DataFrame, keys, score_thr: float = 0.3,
                    iou_thr: float = 0.5) -> dict:
    """Simplified TIDE-style counts of detections above `score_thr` and of missed GT.

    Each detection (high score first) is labelled:
      tp         IoU >= iou_thr with an unmatched GT of the same class
      duplicate  IoU >= iou_thr with an already matched GT of the same class
      cls        IoU >= iou_thr with a GT of another class
      loc        0.1 <= IoU < iou_thr with a GT of the same class
      ignored    mostly (>= 50%) inside an ignored region
      background otherwise
    GT objects not matched by any detection are 'missed' (split by COCO size).
    """
    keys = set(keys)
    p = preds[preds.key.isin(keys) & (preds.score >= score_thr)].sort_values("score", ascending=False)
    b = boxes[boxes.key.isin(keys)]
    counts = dict.fromkeys(["tp", "duplicate", "cls", "loc", "ignored", "background"], 0)
    missed = {"small": 0, "medium": 0, "large": 0}
    total = {"small": 0, "medium": 0, "large": 0}
    pg, bg = dict(tuple(p.groupby("key"))), dict(tuple(b.groupby("key")))
    for k in keys:
        g = bg.get(k, b.iloc[:0])
        gt, ign = g[g.ignore == 0], g[g.ignore == 1]
        gxy, ixy = _xyxy(gt), _xyxy(ign)
        gcls = gt.cls.to_numpy()
        matched = np.zeros(len(gt), bool)
        d = pg.get(k)
        if d is not None and len(d):
            dxy = _xyxy(d)
            ious = iou_matrix(dxy, gxy) if len(gt) else np.zeros((len(d), 0))
            if len(ign):
                ix1 = np.maximum(dxy[:, None, 0], ixy[None, :, 0]); iy1 = np.maximum(dxy[:, None, 1], ixy[None, :, 1])
                ix2 = np.minimum(dxy[:, None, 2], ixy[None, :, 2]); iy2 = np.minimum(dxy[:, None, 3], ixy[None, :, 3])
                inside = (np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)).max(1) / (d.w * d.h).to_numpy()
            else:
                inside = np.zeros(len(d))
            for i, c in enumerate(d.cls.to_numpy()):
                same = np.where(gcls == c, ious[i], 0) if len(gt) else np.zeros(0)
                free = np.where(~matched, same, 0)
                if len(gt) and free.max(initial=0) >= iou_thr:
                    matched[free.argmax()] = True; counts["tp"] += 1
                elif len(gt) and same.max(initial=0) >= iou_thr:
                    counts["duplicate"] += 1
                elif len(gt) and ious[i].max(initial=0) >= iou_thr:
                    counts["cls"] += 1
                elif inside[i] >= 0.5:
                    counts["ignored"] += 1
                elif len(gt) and same.max(initial=0) >= 0.1:
                    counts["loc"] += 1
                else:
                    counts["background"] += 1
        area = (gt.w * gt.h).to_numpy()
        size = np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))
        for s, m in zip(size, matched):
            total[s] += 1
            missed[s] += int(not m)
    out = {**counts, **{f"missed_{s}": missed[s] for s in missed},
           **{f"recall_{s}": round(1 - missed[s] / max(total[s], 1), 3) for s in total}}
    return out


def draw(path: str, preds: pd.DataFrame | None, gt: pd.DataFrame | None, score_thr: float = 0.3,
         crop: tuple | None = None) -> np.ndarray:
    """RGB image with GT (green), ignored regions (gray) and predictions (red, with class/score)."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    t = max(1, img.shape[1] // 900)
    if gt is not None:
        for r in gt.itertuples():
            col = (160, 160, 160) if r.ignore else (0, 220, 0)
            cv2.rectangle(img, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), col, t)
    if preds is not None:
        for r in preds[preds.score >= score_thr].itertuples():
            cv2.rectangle(img, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), (230, 30, 30), t)
            cv2.putText(img, f"{CLASSES[int(r.cls)][:3]} {r.score:.2f}", (int(r.x), int(r.y) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (230, 30, 30), t)
    if crop:
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1]
    return img


def curve(run: dict) -> pd.DataFrame:
    """Validation mAP over training for one run (epoch, mAP, minutes) from the run's own history.

    torchvision runs evaluate our COCO metric every 2 epochs on every 20th frame of fold 0.
    Ultralytics runs report their metric after every epoch on the same frames (ignored regions masked).
    """
    h = run["history"]
    if "curve_mAP" in h:
        h = h.dropna(subset=["curve_mAP"])
        return pd.DataFrame({"epoch": h.epoch, "mAP": h.curve_mAP, "minutes": h.time_min})
    return pd.DataFrame({"epoch": h.epoch, "mAP": h["metrics/mAP50-95(B)"], "minutes": h.time / 60})


def worst_frames(preds: pd.DataFrame, boxes: pd.DataFrame, keys, score_thr: float = 0.3, n_sample: int = 150,
                 seed: int = 0) -> pd.DataFrame:
    """Per-frame error counts for a random sample of frames, to pick failure cases from."""
    keys = list(pd.Series(sorted(keys)).sample(min(n_sample, len(keys)), random_state=seed))
    rows = []
    for k in keys:
        r = error_breakdown(preds, boxes, [k], score_thr=score_thr)
        rows.append({"key": k, **r, "n_gt": int(((boxes.key == k) & (boxes.ignore == 0)).sum())})
    return pd.DataFrame(rows).set_index("key")

"""Logs finished runs (curves, config, metrics, validation predictions with boxes) to Weights & Biases.

The runs were trained on Kaggle without network access to W&B, so they are logged after the fact from
runs/<name>/ (history.csv or Ultralytics results.csv, config.json, metrics.json, val_preds.parquet).

Usage:
  wandb login
  python -m hw2.wandb_log --project hw2-visdrone-vid [--runs frcnn_base yolo_base ...] [--n_images 8]
Offline: WANDB_MODE=offline python -m hw2.wandb_log ...   then   wandb sync wandb/offline-run-*
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import CLASSES
from .analysis import RUNS, curve, load_run
from .data import find_data_root, fold_frames, load_all, make_folds, subsample

FAMILY = {"frcnn": "two-stage", "retina": "one-stage anchor-based", "yolo": "one-stage anchor-free", "rtdetr": "query-based"}
CLASS_LABELS = {i: c for i, c in enumerate(CLASSES)}


def boxes_dict(df: pd.DataFrame, w: int, h: int, pred: bool, thr: float) -> dict:
    rows = df[df.score >= thr] if pred else df
    out = []
    for r in rows.itertuples():
        out.append({"position": {"minX": max(r.x, 0) / w, "minY": max(r.y, 0) / h,
                                 "maxX": min(r.x + r.w, w) / w, "maxY": min(r.y + r.h, h) / h},
                    "class_id": int(r.cls), "box_caption": f"{CLASSES[int(r.cls)]} {r.score:.2f}" if pred else CLASSES[int(r.cls)],
                    **({"scores": {"score": float(r.score)}} if pred else {})})
    return {"box_data": out, "class_labels": CLASS_LABELS}


def log_run(name: str, project: str, val: pd.DataFrame, gt: pd.DataFrame, keys: list, thr: float, entity: str | None):
    import wandb
    r = load_run(name)
    fam = next((v for k, v in FAMILY.items() if name.startswith(k)), "other")
    run = wandb.init(project=project, entity=entity, name=name, group=fam, job_type="train", reinit=True,
                     config={**r["config"], "family": fam}, tags=[fam])
    c = curve(r)
    for row in c.itertuples():
        wandb.log({"epoch": row.epoch, "val/mAP_curve": row.mAP, "train_minutes": row.minutes})
    h = r.get("history")
    if h is not None:
        for _, row in h.iterrows():
            ep = int(row["epoch"])
            wandb.log({"epoch": ep, **{f"train/{k}": float(v) for k, v in row.items()
                                       if k not in ("epoch",) and "loss" in k.lower() and pd.notna(v)}})
    if "metrics" in r:
        wandb.summary.update({k: v for k, v in r["metrics"].items() if isinstance(v, (int, float))})
    if "val_preds" in r:
        p = r["val_preds"]
        imgs = []
        for k in keys:
            fr = val[val.key == k].iloc[0]
            im = wandb.Image(fr.path, boxes={"predictions": boxes_dict(p[p.key == k], fr.width, fr.height, True, thr),
                                             "ground_truth": boxes_dict(gt[(gt.key == k) & (gt.ignore == 0)].assign(score=1.0), fr.width, fr.height, False, 0)},
                             caption=f"{k} (score >= {thr})")
            imgs.append(im)
        wandb.log({"val/predictions": imgs})
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="hw2-visdrone-vid")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--n_images", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.3)
    a = ap.parse_args()
    names = a.runs or sorted(p.parent.name for p in RUNS.glob("*/metrics.json") if (p.parent / "config.json").exists())
    frames, boxes = load_all(find_data_root())
    val = subsample(fold_frames(frames, make_folds(frames), 0), 4)
    keys = sorted(val.key.sample(a.n_images, random_state=1))   # the same frames for every run
    gt = boxes[boxes.key.isin(keys)]
    for n in names:
        print("logging", n, flush=True)
        log_run(n, a.project, val, gt, keys, a.thr, a.entity)


if __name__ == "__main__":
    main()

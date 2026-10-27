"""Training / evaluation of torchvision detectors (Faster R-CNN v2, RetinaNet v2, FCOS).

All runs share one protocol: train on the prepared (masked) frames of folds != val_fold
subsampled with `step`, evaluate on the *original* frames of val_fold with evaluate.py.

Usage:
  python -m hw2.tv_train --model fasterrcnn --name frcnn_base --prep PREP --out runs \
      --cfg '{"anchor_sizes": [16, 32, 64, 128, 256]}'
Every run writes to <out>/<name>/: config.json, history.csv, metrics.json,
val_preds.parquet, test_preds.parquet, model.pt.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection import retinanet as tv_retinanet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import generalized_box_iou_loss

from . import NUM_CLASSES
from .data import find_data_root, fold_frames, load_all, load_test, make_folds, subsample
from .evaluate import build_gt, evaluate

DEFAULTS = dict(
    val_fold=0, step=10, val_step=4, curve_step=20,
    epochs=12, batch=4, lr=0.01, weight_decay=1e-4, warmup_iters=500, workers=2, seed=0,
    # input size: train multi-scale on the short side, evaluate at 756x1344 (16:9)
    train_min_sizes=[672, 704, 736, 768, 800, 832], train_max_size=1500,
    test_min_size=756, test_max_size=1344,
    hflip=0.5, color_jitter=0.2,
    # shared detector settings
    anchor_sizes=None,            # None = torchvision default
    anchor_ratios=[0.5, 1.0, 2.0],
    detections_per_img=500, score_thresh=0.001, nms_thresh=0.5,
    # Faster R-CNN
    rpn_pre_nms_train=2000, rpn_post_nms_train=2000, rpn_pre_nms_test=1000, rpn_post_nms_test=1000,
    roi_batch=512, roi_pos_fraction=0.25, roi_box_loss="smooth_l1",
    # RetinaNet
    focal_alpha=0.25, focal_gamma=2.0, fg_iou=0.5, bg_iou=0.4, retina_box_loss="giou",
)


# ---------------------------------------------------------------- data
def read_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


class TrainSet(Dataset):
    """Prepared frames + YOLO labels -> (image tensor, target) with flip / colour jitter."""

    def __init__(self, prep: Path, frames: pd.DataFrame, cfg: dict):
        self.prep, self.frames, self.cfg = prep, frames.reset_index(drop=True), cfg

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        r = self.frames.iloc[i]
        img = read_rgb(str(self.prep / r.prep_path))
        H, W = img.shape[:2]
        lbl = np.loadtxt(self.prep / "labels" / f"{r.key}.txt", ndmin=2).reshape(-1, 5)
        cls = lbl[:, 0].astype(np.int64)
        cx, cy, w, h = lbl[:, 1] * W, lbl[:, 2] * H, lbl[:, 3] * W, lbl[:, 4] * H
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).clip(0, [W, H, W, H])

        if random.random() < self.cfg["hflip"]:
            img = img[:, ::-1]
            boxes = np.stack([W - boxes[:, 2], boxes[:, 1], W - boxes[:, 0], boxes[:, 3]], 1)
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
        j = self.cfg["color_jitter"]
        if j:
            img = (img * random.uniform(1 - j, 1 + j) + random.uniform(-j, j) / 2).clamp(0, 1)

        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        target = {"boxes": torch.as_tensor(boxes[keep], dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(cls[keep] + 1)}  # 0 is background in torchvision
        return img, target


class ImageSet(Dataset):
    """Original frames for evaluation / test inference."""

    def __init__(self, frames: pd.DataFrame):
        self.frames = frames.reset_index(drop=True)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        r = self.frames.iloc[i]
        img = torch.from_numpy(read_rgb(r.path)).permute(2, 0, 1)
        return img, r.key


def collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------- losses
def giou_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, weight=10.0):
    """Drop-in for torchvision's fastrcnn_loss with GIoU instead of smooth-L1.

    Both the predicted and the target deltas are decoded relative to the same unit
    proposal. Axis-aligned scaling and translation leave (G)IoU unchanged, so this equals
    the GIoU computed against the real proposals without needing them here.
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    pos = torch.where(labels > 0)[0]
    N = class_logits.shape[0]
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    pred = box_regression[pos, labels[pos]].float()
    tgt = regression_targets[pos].float()

    def decode(d):  # BoxCoder weights (10, 10, 5, 5), reference box centred at 0 with size 1
        cx, cy = d[:, 0] / 10, d[:, 1] / 10
        w = torch.exp(torch.clamp(d[:, 2] / 5, max=math.log(1000 / 16)))
        h = torch.exp(torch.clamp(d[:, 3] / 5, max=math.log(1000 / 16)))
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)

    box_loss = weight * generalized_box_iou_loss(decode(pred), decode(tgt), reduction="sum") / labels.numel()
    return classification_loss, box_loss


# ---------------------------------------------------------------- models
def build_model(name: str, cfg: dict) -> torch.nn.Module:
    det = torchvision.models.detection
    common = dict(min_size=cfg["train_min_sizes"], max_size=cfg["train_max_size"],
                  detections_per_img=cfg["detections_per_img"])

    if name == "fasterrcnn":
        if cfg["roi_box_loss"] == "giou":
            tv_roi_heads.fastrcnn_loss = giou_fastrcnn_loss
        kw = dict(
            rpn_pre_nms_top_n_train=cfg["rpn_pre_nms_train"], rpn_post_nms_top_n_train=cfg["rpn_post_nms_train"],
            rpn_pre_nms_top_n_test=cfg["rpn_pre_nms_test"], rpn_post_nms_top_n_test=cfg["rpn_post_nms_test"],
            box_batch_size_per_image=cfg["roi_batch"], box_positive_fraction=cfg["roi_pos_fraction"],
            box_score_thresh=cfg["score_thresh"], box_nms_thresh=cfg["nms_thresh"],
            box_detections_per_img=cfg["detections_per_img"], min_size=common["min_size"], max_size=common["max_size"],
        )
        m = det.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", **kw)
        if cfg["anchor_sizes"]:  # one size x 3 ratios per FPN level, so the COCO RPN head still fits
            m.rpn.anchor_generator = AnchorGenerator(
                tuple((s,) for s in cfg["anchor_sizes"]), (tuple(cfg["anchor_ratios"]),) * len(cfg["anchor_sizes"]))
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = det.faster_rcnn.FastRCNNPredictor(in_f, NUM_CLASSES + 1)
        return m

    if name == "retinanet":
        tv_retinanet.sigmoid_focal_loss = functools.partial(
            torchvision.ops.sigmoid_focal_loss, alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
        kw = dict(score_thresh=cfg["score_thresh"], nms_thresh=cfg["nms_thresh"],
                  fg_iou_thresh=cfg["fg_iou"], bg_iou_thresh=cfg["bg_iou"], topk_candidates=1000, **common)
        m = det.retinanet_resnet50_fpn_v2(weights="DEFAULT", **kw)
        if cfg["anchor_sizes"]:  # 3 octave scales per level, like the default generator
            sizes = tuple((s, int(s * 2 ** (1 / 3)), int(s * 2 ** (2 / 3))) for s in cfg["anchor_sizes"])
            m.anchor_generator = AnchorGenerator(sizes, (tuple(cfg["anchor_ratios"]),) * len(sizes))
        na = m.anchor_generator.num_anchors_per_location()[0]
        m.head.classification_head = tv_retinanet.RetinaNetClassificationHead(
            256, na, NUM_CLASSES + 1, norm_layer=functools.partial(torch.nn.GroupNorm, 32))
        m.head.regression_head._loss_type = cfg["retina_box_loss"]
        if na != 9:  # anchor count changed: regression head can't reuse COCO weights
            m.head.regression_head = tv_retinanet.RetinaNetRegressionHead(
                256, na, norm_layer=functools.partial(torch.nn.GroupNorm, 32))
            m.head.regression_head._loss_type = cfg["retina_box_loss"]
        return m

    if name == "fcos":
        m = det.fcos_resnet50_fpn(weights="DEFAULT", score_thresh=cfg["score_thresh"],
                                  nms_thresh=cfg["nms_thresh"], **common)
        m.head.classification_head = det.fcos.FCOSClassificationHead(256, 1, NUM_CLASSES + 1)
        return m

    raise ValueError(name)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, frames: pd.DataFrame, cfg: dict, device, workers=2) -> pd.DataFrame:
    model.eval()
    model.transform.min_size = (cfg["test_min_size"],)
    model.transform.max_size = cfg["test_max_size"]
    dl = DataLoader(ImageSet(frames), batch_size=2, num_workers=workers, collate_fn=collate)
    out = []
    for imgs, keys in dl:
        imgs = [i.to(device, non_blocking=True).float() / 255 for i in imgs]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            res = model(imgs)
        for k, r in zip(keys, res):
            b = r["boxes"].float().cpu().numpy()
            out.append(pd.DataFrame({
                "key": k, "cls": r["labels"].cpu().numpy() - 1, "score": r["scores"].float().cpu().numpy(),
                "x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1]}))
    model.transform.min_size = tuple(cfg["train_min_sizes"])
    model.transform.max_size = cfg["train_max_size"]
    return pd.concat(out, ignore_index=True)


@torch.no_grad()
def measure_fps(model, frames: pd.DataFrame, cfg: dict, device, n=100) -> float:
    """Batch-1 latency incl. resize/normalise/postprocess, excluding JPEG decoding."""
    model.eval()
    model.transform.min_size, model.transform.max_size = (cfg["test_min_size"],), cfg["test_max_size"]
    imgs = [torch.from_numpy(read_rgb(p)).permute(2, 0, 1) for p in frames.path.iloc[:n]]
    times = []
    for i, img in enumerate(imgs):
        x = img.to(device).float() / 255
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            model([x])
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        if i >= 10:
            times.append(time.perf_counter() - t)
    return round(1 / float(np.mean(times)), 2)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["fasterrcnn", "retinanet", "fcos"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--cfg", default="{}", help="JSON overrides of DEFAULTS")
    ap.add_argument("--max_train", type=int, default=0, help="debug: cap number of train frames")
    ap.add_argument("--no_test", action="store_true")
    a = ap.parse_args()

    cfg = {**DEFAULTS, **json.loads(a.cfg), "model": a.model}
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prep = Path(a.prep)
    pf = pd.read_csv(prep / "frames.csv")
    train = subsample(pf[pf.fold != cfg["val_fold"]], cfg["step"])
    if a.max_train:
        train = train.sample(a.max_train, random_state=0)
    frames, boxes = load_all(find_data_root())
    folds = make_folds(frames)
    val_frames = fold_frames(frames, folds, cfg["val_fold"])
    val = subsample(val_frames, cfg["val_step"])
    curve = subsample(val_frames, cfg["curve_step"])
    if a.max_train:
        val, curve = val.iloc[:: max(1, len(val) // 40)], curve.iloc[:: max(1, len(curve) // 20)]
    gt_val, gt_curve = build_gt(val, boxes), build_gt(curve, boxes)
    print(f"train {len(train)} frames, val {len(val)} frames, device {device}", flush=True)

    model = build_model(a.model, cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(cfg["seed"])
    dl = DataLoader(TrainSet(prep, train, cfg), batch_size=cfg["batch"], shuffle=True, generator=g,
                    num_workers=cfg["workers"], collate_fn=collate, drop_last=True, persistent_workers=True)
    total = cfg["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / cfg["warmup_iters"])
                                              * 0.5 * (1 + math.cos(math.pi * it / total)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, t0 = [], time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        sums, n = {}, 0
        for imgs, tg in dl:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tg = [{k: v.to(device) for k, v in t.items()} for t in tg]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                losses = model(imgs, tg)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch", {k: float(v) for k, v in losses.items()})
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
        row = {"epoch": ep + 1, "time_min": round((time.time() - t0) / 60, 1), "lr": opt.param_groups[0]["lr"],
               **{k: round(v / max(n, 1), 4) for k, v in sums.items()}}
        if (ep + 1) % 2 == 0 or ep + 1 == cfg["epochs"]:
            row.update({f"curve_{k}": v for k, v in evaluate(predict(model, curve, cfg, device), gt_curve).items()})
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(row, flush=True)

    torch.save(model.state_dict(), out / "model.pt")
    vp = predict(model, val, cfg, device)
    vp.to_parquet(out / "val_preds.parquet")
    metrics = evaluate(vp, gt_val, per_class=True)
    metrics.update(fps=measure_fps(model, val, cfg, device), train_min=round((time.time() - t0) / 60, 1),
                   n_train=len(train), n_val=len(val), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("VAL", metrics, flush=True)
    if not a.no_test:
        predict(model, load_test(), cfg, device).to_parquet(out / "test_preds.parquet")
    print("done", flush=True)


if __name__ == "__main__":
    main()

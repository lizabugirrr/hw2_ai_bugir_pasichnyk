"""Model-internals visualisation: Grad-CAM for a CNN detector (YOLO11) and deformable attention / query
embeddings of RT-DETR. Both work on a single image so they run on a small GPU.

YOLO:    gradcam_yolo(net, img, box_xyxy, cls)  -> {layer name: heat map in image coordinates}
RT-DETR: rtdetr_run(net, img)                    -> boxes, scores, classes, decoder embeddings and per-layer
                                                   attention sampling points of every query
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

YOLO_LAYERS = {"backbone P3": 4, "backbone P4": 6, "backbone P5": 10, "neck P3": 16, "neck P4": 19, "neck P5": 22}


def load_net(path: str, kind: str, device: str = "cuda"):
    from ultralytics import RTDETR, YOLO
    net = (RTDETR if kind == "rtdetr" else YOLO)(path).model.float().to(device).eval()
    for p in net.parameters():
        p.requires_grad_(kind == "yolo")
    return net


# ------------------------------------------------------------------ YOLO + Grad-CAM
def _letterbox(img_bgr: np.ndarray, imgsz: int):
    h, w = img_bgr.shape[:2]
    s = imgsz / max(h, w)
    nh, nw = round(h * s), round(w * s)
    im = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = -nh % 32, -nw % 32
    im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
    return x[None], s, (nh, nw)


def gradcam_yolo(net, img_bgr: np.ndarray, box_xyxy, cls: int, imgsz: int = 1280, layers: dict | None = None,
                 device: str = "cuda"):
    """Grad-CAM of the class score of the prediction that best covers `box_xyxy` (original pixels).

    Returns (maps, info): maps[name] is a [0, 1] heat map with the size of the image, info holds the score of the
    chosen prediction and the stride level it came from.
    """
    layers = layers or YOLO_LAYERS
    x, s, (nh, nw) = _letterbox(img_bgr, imgsz)
    x = x.to(device).requires_grad_(True)
    acts = {}
    hooks = [net.model[i].register_forward_hook(lambda m, i_, o, n=n: acts.__setitem__(n, o)) for n, i in layers.items()]
    try:
        with torch.enable_grad():
            y = net(x)[0][0]                                    # (4 + nc, anchors); xywh in input pixels, then sigmoid scores
            bx = y[:4].detach()
            cx, cy = bx[0] / s, bx[1] / s
            x0, y0, x1, y1 = box_xyxy
            inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
            score = y[4 + cls]
            cand = torch.where(inside, score, torch.full_like(score, -1))
            a = int(cand.argmax())
            target = score[a]
            grads = torch.autograd.grad(target, list(acts.values()), retain_graph=False, allow_unused=True)
    finally:
        for h in hooks:
            h.remove()
    H, W = img_bgr.shape[:2]
    maps = {}
    for (name, act), g in zip(acts.items(), grads):
        if g is None:
            maps[name] = np.zeros((H, W), np.float32)
            continue
        wgt = g.mean((2, 3), keepdim=True)
        cam = torch.relu((wgt * act).sum(1))[0].detach().cpu().numpy()
        full = cv2.resize(cam, (x.shape[3], x.shape[2]))[:nh, :nw]
        full = cv2.resize(full, (W, H))
        full = full - full.min()
        maps[name] = (full / full.max() if full.max() > 0 else full).astype(np.float32)
    n = [(x.shape[2] // st) * (x.shape[3] // st) for st in (8, 16, 32)]
    level = int(np.searchsorted(np.cumsum(n), a, side="right"))     # 0 = stride 8, 1 = stride 16, 2 = stride 32
    return maps, {"score": float(target.detach()), "anchor": a, "level": level,
                  "anchor_xywh": (bx[:, a] / s).cpu().numpy().tolist()}


def overlay(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.5, crop=None) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    out = (img_bgr[:, :, ::-1] * (1 - alpha) + heat * alpha).astype(np.uint8)
    if crop:
        x0, y0, x1, y1 = [int(v) for v in crop]
        out = out[y0:y1, x0:x1]
    return out


# ------------------------------------------------------------------ RT-DETR
@torch.no_grad()
def rtdetr_run(net, img_bgr: np.ndarray, imgsz: int = 1536, device: str = "cuda", half: bool = True):
    """One forward pass keeping what RT-DETR normally throws away.

    Returns dict with, for each of the nq queries: xyxy box (original pixels), score, class, the last decoder layer's
    embedding, and for every decoder layer the attention sampling points (x, y in original pixels), their weights and
    the feature level each point is taken from.
    """
    import ultralytics.nn.modules.transformer as T
    H, W = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    x = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float().div(255).to(device)
    cap, emb = [], []
    orig = T.multi_scale_deformable_attn_pytorch

    def wrapped(value, shapes, loc, w):
        cap.append((loc.detach().float().cpu(), w.detach().float().cpu(), [tuple(s) for s in shapes]))
        return orig(value, shapes, loc, w)

    T.multi_scale_deformable_attn_pytorch = wrapped
    h = net.model[-1].decoder.layers[-1].register_forward_hook(lambda m, i, o: emb.append(o.detach().float().cpu()))
    try:
        with torch.autocast("cuda", enabled=half):
            raw = net(x)[1]                                      # (dec_bboxes, dec_scores, ...) of the last decoder layer
    finally:
        T.multi_scale_deformable_attn_pytorch = orig
        h.remove()
    nq = emb[-1].shape[1]
    boxes = raw[0].float().cpu().reshape(-1, 4)[-nq:].numpy()      # normalised xywh
    probs = raw[1].float().cpu().reshape(-1, raw[1].shape[-1])[-nq:].sigmoid()
    xyxy = np.stack([(boxes[:, 0] - boxes[:, 2] / 2) * W, (boxes[:, 1] - boxes[:, 3] / 2) * H,
                     (boxes[:, 0] + boxes[:, 2] / 2) * W, (boxes[:, 1] + boxes[:, 3] / 2) * H], 1)
    sc, cl = probs.max(1)
    layers = []
    for loc, w, shapes in cap:
        nq, heads, tot, _ = loc[0].shape
        n_levels = len(shapes)
        pts = loc[0].numpy().copy()                               # (nq, heads, levels * points, 2) in [0, 1]
        pts[..., 0] *= W
        pts[..., 1] *= H
        level = np.repeat(np.arange(n_levels), tot // n_levels)
        layers.append({"xy": pts, "w": w[0].numpy(), "level": level})
    return {"xyxy": xyxy, "score": sc.numpy(), "cls": cl.numpy(), "emb": emb[-1][0].numpy(), "layers": layers,
            "shapes": cap[0][2] if cap else []}


def attention_spread(layer: dict, q: int) -> float:
    """Weighted standard deviation (original pixels) of where query q samples, averaged over heads."""
    xy, w = layer["xy"][q], layer["w"][q]                          # (heads, P, 2), (heads, P)
    w = w / w.sum(1, keepdims=True)
    mu = (xy * w[..., None]).sum(1, keepdims=True)
    var = (w[..., None] * (xy - mu) ** 2).sum(1)                   # (heads, 2)
    return float(np.sqrt(var.sum(-1)).mean())

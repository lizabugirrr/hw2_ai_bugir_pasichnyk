"""Inference-time post-processing on prediction tables (key, cls, score, x, y, w, h).

- weighted_boxes_fusion: merge predictions of several models / TTA passes (Solovyev et al., 2019).
- temporal_rescore: use detections of neighbouring frames of the same clip (Seq-NMS-like
  rescoring, Han et al., 2016) to boost temporally consistent boxes and damp flickering ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _iou_one_to_many(b: np.ndarray, bs: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    x1 = np.maximum(b[0], bs[:, 0]); y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2]); y2 = np.minimum(b[3], bs[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (a + (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1]) - inter + 1e-9)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes a [N,4] and b [M,4]."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def _wbf_one(boxes, scores, model_ids, n_models, weights, iou_thr):
    """WBF for one image and one class. boxes xyxy sorted by score desc."""
    clusters, fused = [], []  # members (indices) and current fused box per cluster
    for i in range(len(boxes)):
        if fused:
            ious = _iou_one_to_many(boxes[i], np.array([f[:4] for f in fused]))
            j = int(ious.argmax())
            if ious[j] > iou_thr:
                clusters[j].append(i)
                m = clusters[j]
                w = scores[m] * weights[model_ids[m]]
                fused[j][:4] = (boxes[m] * w[:, None]).sum(0) / w.sum()
                continue
        clusters.append([i])
        fused.append(np.concatenate([boxes[i], [0.0]]))
    out = []
    for m, f in zip(clusters, fused):
        w = weights[model_ids[m]]
        # average weighted score over the models that could have voted; penalise boxes few models agree on
        score = (scores[m] * w).sum() / weights.sum()
        out.append([*f[:4], min(score, 1.0)])
    return np.array(out).reshape(-1, 5)


def weighted_boxes_fusion(preds: list[pd.DataFrame], weights=None, iou_thr: float = 0.55,
                          skip_thr: float = 0.001, max_per_image: int = 500) -> pd.DataFrame:
    weights = np.ones(len(preds)) if weights is None else np.asarray(weights, float)
    allp = pd.concat([p.assign(model=i) for i, p in enumerate(preds)], ignore_index=True)
    allp = allp[allp.score >= skip_thr]
    out = []
    for (key, c), g in allp.groupby(["key", "cls"], sort=False):
        g = g.sort_values("score", ascending=False)
        b = np.stack([g.x, g.y, g.x + g.w, g.y + g.h], 1)
        f = _wbf_one(b, g.score.to_numpy(), g.model.to_numpy(), len(preds), weights, iou_thr)
        out.append(pd.DataFrame({"key": key, "cls": c, "score": f[:, 4], "x": f[:, 0], "y": f[:, 1],
                                 "w": f[:, 2] - f[:, 0], "h": f[:, 3] - f[:, 1]}))
    res = pd.concat(out, ignore_index=True)
    return res.sort_values("score", ascending=False).groupby("key").head(max_per_image).reset_index(drop=True)


def temporal_rescore(preds: pd.DataFrame, frames: pd.DataFrame, radius: int = 2, iou_thr: float = 0.5,
                     alpha: float = 0.5, class_agnostic: bool = False) -> pd.DataFrame:
    """Rescores each detection with its best matches in the +-radius neighbouring frames.

    For a detection with score s, let m_k be the highest score among same-class detections
    in frame t+k with IoU > iou_thr (0 if none). The new score is
        s' = (1 - alpha) * s + alpha * mean_k(max(s, m_k) if matched else 0) ... averaged over k != 0,
    i.e. a box seen in its neighbours keeps (or raises) its score, an isolated box is damped.
    Frames are matched by their number within the clip, so it works on any subsampling step;
    `radius` counts available neighbours, not raw frame numbers.
    """
    f = frames.set_index("key")[["clip", "frame"]]
    p = preds.join(f, on="key")
    out = []
    for clip, g in p.groupby("clip", sort=False):
        frame_ids = np.sort(g.frame.unique())
        per = {fr: gg for fr, gg in g.groupby("frame")}
        xyxy = {fr: np.stack([gg.x, gg.y, gg.x + gg.w, gg.y + gg.h], 1) for fr, gg in per.items()}
        for i, fr in enumerate(frame_ids):
            cur = per[fr]
            s = cur.score.to_numpy()
            support, n_nb = np.zeros(len(cur)), 0
            for j in range(max(0, i - radius), min(len(frame_ids), i + radius + 1)):
                if j == i:
                    continue
                nb = per[frame_ids[j]]
                n_nb += 1
                iou = iou_matrix(xyxy[fr], xyxy[frame_ids[j]])
                if not class_agnostic:
                    iou = iou * (cur.cls.to_numpy()[:, None] == nb.cls.to_numpy()[None, :])
                ok = iou > iou_thr
                best = np.where(ok, nb.score.to_numpy()[None, :], 0).max(1) if len(nb) else np.zeros(len(cur))
                support += np.where(best > 0, np.maximum(best, s), 0)
            if n_nb:
                s = (1 - alpha) * s + alpha * support / n_nb
            out.append(cur.assign(score=s))
    res = pd.concat(out, ignore_index=True)
    return res[preds.columns.tolist()]


def seq_rescore(preds: pd.DataFrame, frames: pd.DataFrame, iou_thr: float = 0.5, beta: float = 0.5,
                mode: str = "mean", max_len: int = 0) -> pd.DataFrame:
    """Seq-NMS-style rescoring (Han et al., 2016) without the suppression step.

    Same-class detections of consecutive frames are linked when their IoU exceeds `iou_thr`. A dynamic
    programme finds, for every detection, the best path (highest score sum) through it going back in time and
    forward in time; the detection is rescored with the mean (or max) score of that path:
        s' = (1 - beta) * s + beta * path_score.
    Frames are linked in order of frame number within a clip; gaps are not bridged, so it expects
    consecutive frames. `max_len` > 0 limits how far the path may extend in each direction.
    """
    import torch
    from torchvision.ops import box_iou

    f = frames.set_index("key")[["clip", "frame"]]
    p = preds.join(f, on="key").reset_index(drop=True)
    new = p.score.to_numpy().copy()
    for _, g in p.groupby("clip", sort=False):
        order = sorted(g.frame.unique())
        idx = [g.index[g.frame == fr].to_numpy() for fr in order]
        n = len(idx)
        xyxy = [torch.tensor(np.stack([p.x[i], p.y[i], p.x[i] + p.w[i], p.y[i] + p.h[i]], 1), dtype=torch.float32) for i in idx]
        cls = [p.cls.to_numpy()[i] for i in idx]
        sc = [p.score.to_numpy()[i] for i in idx]
        links = [None] * (n - 1)
        for t in range(n - 1):
            if len(idx[t]) and len(idx[t + 1]):
                ok = (box_iou(xyxy[t], xyxy[t + 1]).numpy() > iou_thr) & (cls[t][:, None] == cls[t + 1][None, :])
                links[t] = ok
        # forward: best path ending at each detection; backward: best path starting at each detection
        fs, fl, bs, bl = [None] * n, [None] * n, [None] * n, [None] * n
        for t in range(n):
            fs[t], fl[t] = sc[t].copy(), np.ones(len(sc[t]))
            if t and links[t - 1] is not None:
                m = np.where(links[t - 1], fs[t - 1][:, None], -np.inf)
                j, best = m.argmax(0), m.max(0)
                has = np.isfinite(best)
                fs[t] = sc[t] + np.where(has, best, 0)
                fl[t] = 1 + np.where(has, fl[t - 1][j], 0)
        for t in range(n - 1, -1, -1):
            bs[t], bl[t] = sc[t].copy(), np.ones(len(sc[t]))
            if t < n - 1 and links[t] is not None:
                m = np.where(links[t], bs[t + 1][None, :], -np.inf)
                j, best = m.argmax(1), m.max(1)
                has = np.isfinite(best)
                bs[t] = sc[t] + np.where(has, best, 0)
                bl[t] = 1 + np.where(has, bl[t + 1][j], 0)
        for t in range(n):
            total, length = fs[t] + bs[t] - sc[t], fl[t] + bl[t] - 1
            path = total / length if mode == "mean" else np.maximum(sc[t], total / length)
            new[idx[t]] = (1 - beta) * sc[t] + beta * path
    return preds.assign(score=new)[preds.columns.tolist()]


def propagate(preds: pd.DataFrame, frames: pd.DataFrame, conf: float = 0.6, decay: float = 0.5,
              dup_iou: float = 0.5, nms_iou: float = 0.6) -> pd.DataFrame:
    """Copies confident detections of the previous and next frame into the current one (T-CNN style propagation).

    A neighbour detection with score >= `conf` is added with score * `decay` unless the current frame
    already has a same-class box with IoU > `dup_iou`. Added boxes then go through class-wise NMS.
    """
    import torch
    from torchvision.ops import batched_nms, box_iou

    f = frames.set_index("key")[["clip", "frame"]]
    p = preds.join(f, on="key")
    keys = {(c, fr): k for (c, fr), k in zip(zip(frames["clip"], frames["frame"]), frames["key"])}
    by = {k: g for k, g in p.groupby("key")}
    out = []
    for key, cur in by.items():
        c, fr = cur["clip"].iloc[0], cur["frame"].iloc[0]
        parts = [cur]
        cb = torch.tensor(np.stack([cur.x, cur.y, cur.x + cur.w, cur.y + cur.h], 1), dtype=torch.float32)
        for d in (-1, 1):
            nk = keys.get((c, fr + d))
            if nk is None or nk not in by:
                continue
            nb = by[nk]
            nb = nb[nb.score >= conf]
            if not len(nb):
                continue
            nbx = torch.tensor(np.stack([nb.x, nb.y, nb.x + nb.w, nb.y + nb.h], 1), dtype=torch.float32)
            same = torch.tensor(nb.cls.to_numpy()[:, None] == cur.cls.to_numpy()[None, :])
            dup = ((box_iou(nbx, cb) > dup_iou) & same).any(1).numpy() if len(cur) else np.zeros(len(nb), bool)
            add = nb[~dup].copy()
            add["score"] *= decay
            add["key"] = key
            parts.append(add)
        m = pd.concat(parts, ignore_index=True)
        if len(m) > len(cur):
            b = torch.tensor(np.stack([m.x, m.y, m.x + m.w, m.y + m.h], 1), dtype=torch.float32)
            keep = batched_nms(b, torch.tensor(m.score.to_numpy(), dtype=torch.float32), torch.tensor(m.cls.to_numpy()), nms_iou)
            m = m.iloc[keep.numpy()]
        out.append(m)
    res = pd.concat(out, ignore_index=True)
    return res[preds.columns.tolist()]

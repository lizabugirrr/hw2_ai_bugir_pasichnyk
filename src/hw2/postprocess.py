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

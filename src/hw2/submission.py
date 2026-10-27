"""Predictions DataFrame (key, cls, score, x, y, w, h) -> Kaggle submission CSV."""
from __future__ import annotations

import pandas as pd

MAX_PER_IMAGE = 500


def to_submission(preds: pd.DataFrame, test_frames: pd.DataFrame, path: str) -> pd.DataFrame:
    p = preds.sort_values("score", ascending=False).groupby("key").head(MAX_PER_IMAGE)
    rows = {}
    for key, g in p.groupby("key"):
        rows[key] = " ".join(
            f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}"
            for c, s, x, y, w, h in zip(g.cls, g.score, g.x, g.y, g.w, g.h)
        )
    sub = pd.DataFrame({"image_id": test_frames.key})
    sub["PredictionString"] = sub.image_id.map(rows).fillna("")
    sub.to_csv(path, index=False)
    return sub

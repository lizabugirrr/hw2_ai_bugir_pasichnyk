"""Final test submission: WBF of RT-DETR models, then Seq-NMS-style rescoring along each clip.

Members (all RT-DETR at 1536 px, each trained on a different set of clips):
  rtdetr_all_1536, rtdetr_x_all (all 63 clips, weight 2 each),
  rtdetr_s5_1536, rtdetr_cv_f1..f4 (clip-level folds 0-4 held out, weight 1 each).
Fold models have no clean validation set for this ensemble (each saw four of the five folds), so its only
measurement is the public leaderboard.

  python -m hw2.final_ensemble --out sub_final.csv [--no_seq]
"""
from __future__ import annotations

import argparse

import pandas as pd

from .data import load_test
from .postprocess import seq_rescore, weighted_boxes_fusion
from .submission import to_submission

MEMBERS = [("rtdetr_all_1536", 2), ("rtdetr_x_all", 2), ("rtdetr_s5_1536", 1), ("rtdetr_cv_f1", 1),
           ("rtdetr_cv_f2", 1), ("rtdetr_cv_f3", 1), ("rtdetr_cv_f4", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.7, help="WBF IoU threshold")
    ap.add_argument("--no_seq", action="store_true", help="skip the temporal rescoring")
    a = ap.parse_args()

    test = load_test()
    preds = [pd.read_parquet(f"{a.runs}/{n}/test_preds.parquet") for n, _ in MEMBERS]
    fused = weighted_boxes_fusion(preds, weights=[w for _, w in MEMBERS], iou_thr=a.iou)
    fused = fused[fused.score >= 0.001]
    if not a.no_seq:
        fused = seq_rescore(fused, test, iou_thr=0.7, beta=0.5, mode="mean")
    to_submission(fused, test, a.out)
    print(f"wrote {a.out}: {len(fused)} detections")


if __name__ == "__main__":
    main()

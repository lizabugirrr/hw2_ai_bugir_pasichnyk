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
# Eight more all-clip members, each trained on a different frame offset and seed (see kernels/launch_ens_more.sh)
EXTRA = [(f"rtdetr_all_o{o}_s{s}", 1) for o, s in [(2, 1), (4, 2), (2, 3), (4, 4), (6, 5), (8, 6), (6, 7), (8, 8)]]
SETS = {"base": MEMBERS, "all": MEMBERS + EXTRA, "extra": MEMBERS[:2] + EXTRA}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.7, help="WBF IoU threshold")
    ap.add_argument("--set", default="base", choices=sorted(SETS), help="which ensemble members to fuse")
    ap.add_argument("--no_seq", action="store_true", help="skip the temporal rescoring")
    a = ap.parse_args()

    members = SETS[a.set]
    test = load_test()
    preds = [pd.read_parquet(f"{a.runs}/{n}/test_preds.parquet") for n, _ in members]
    fused = weighted_boxes_fusion(preds, weights=[w for _, w in members], iou_thr=a.iou)
    fused = fused[fused.score >= 0.001]
    if not a.no_seq:
        fused = seq_rescore(fused, test, iou_thr=0.7, beta=0.5, mode="mean")
    to_submission(fused, test, a.out)
    print(f"wrote {a.out}: {len(fused)} detections")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Cross-validation (folds 1-4) of the extra fold-0 variants: Faster R-CNN with GIoU, YOLO at 1920 px and the YOLO
# head variants. One queue per fold (about 6 h each), two kernels with two queues each, so two GPU sessions.
#
# Run from the repository root with the Kaggle account that should spend its quota:
#   KAGGLE_USER=<kaggle username> bash kernels/launch_cv_extra.sh
# That account needs its own hw2-prep kernel (see kernels/launch.py) and must have joined the competition.
# Fetch the results with:  KAGGLE_USER=<username> python kernels/fetch.py hw2-cvx-a   (and hw2-cvx-b)
set -euo pipefail

ULTRA="python -m hw2.ultra_train --prep {PREP} --out {OUT} --no_test"
TV="python -m hw2.tv_train --prep {PREP} --out {OUT} --no_test"

queue() {
  local k=$1
  local yolo='"epochs":12,"imgsz":1344,"batch":8,"seed":0'
  echo "$TV --model fasterrcnn --name frcnn_giou_cv_f$k --cfg '{\"val_fold\":$k,\"roi_box_loss\":\"giou\",\"rpn_pre_nms_test\":2000,\"rpn_post_nms_test\":2000}'" \
       "&& $ULTRA --model yolo11s.pt --name yolo_1920_cv_f$k --cfg '{\"epochs\":12,\"imgsz\":1920,\"batch\":4,\"seed\":0}' --hw2 '{\"val_fold\":$k,\"step\":10}'" \
       "&& $ULTRA --model hw2:yolo11s-p2.yaml --name yolo_p2_cv_f$k --cfg '{$yolo}' --hw2 '{\"val_fold\":$k,\"step\":10,\"weights\":\"yolo11s.pt\"}'" \
       "&& $ULTRA --model hw2:yolo11s-p2.yaml --name yolo_p2_hinit_cv_f$k --cfg '{$yolo}' --hw2 '{\"val_fold\":$k,\"step\":10,\"weights\":\"yolo11s.pt\",\"head_shift\":1}'" \
       "&& $ULTRA --model hw2:yolo11s-p2-nop5.yaml --name yolo_p2_nop5_cv_f$k --cfg '{$yolo}' --hw2 '{\"val_fold\":$k,\"step\":10,\"weights\":\"yolo11s.pt\",\"head_shift\":1}'" \
       "&& $ULTRA --model hw2:yolo11s-p3p4.yaml --name yolo_p3p4_cv_f$k --cfg '{$yolo}' --hw2 '{\"val_fold\":$k,\"step\":10,\"weights\":\"yolo11s.pt\",\"head_shift\":0}'"
}

python kernels/launch.py --slug hw2-cvx-a --queue "$(queue 1)" --queue "$(queue 2)"
python kernels/launch.py --slug hw2-cvx-b --queue "$(queue 3)" --queue "$(queue 4)"

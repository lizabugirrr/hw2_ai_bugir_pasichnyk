#!/usr/bin/env bash
# Cross-validation (folds 1-4) of the tuned variants: Faster R-CNN with 16 px anchors, RetinaNet with 12 px anchors,
# YOLO11s with NWD loss and RT-DETR-l with a re-weighted Hungarian cost. Two kernels, two fold queues each.
set -euo pipefail
ULTRA="python -m hw2.ultra_train --prep {PREP} --out {OUT} --no_test"
TV="python -m hw2.tv_train --prep {PREP} --out {OUT} --no_test"
queue() {
  local k=$1
  echo "$TV --model fasterrcnn --name frcnn_a16_cv_f$k --cfg '{\"val_fold\":$k,\"anchor_sizes\":[16,32,64,128,256]}'" \
       "&& $TV --model retinanet --name retina_a12_cv_f$k --cfg '{\"val_fold\":$k,\"anchor_sizes\":[12,24,48,96,192]}'" \
       "&& $ULTRA --model yolo11s.pt --name yolo_nwd_cv_f$k --cfg '{\"epochs\":12,\"imgsz\":1344,\"batch\":8,\"seed\":0}' --hw2 '{\"val_fold\":$k,\"step\":10,\"nwd_ratio\":0.5}'" \
       "&& $ULTRA --model rtdetr-l.pt --name rtdetr_cost_cv_f$k --cfg '{\"epochs\":12,\"imgsz\":1344,\"batch\":4,\"seed\":0}' --hw2 '{\"val_fold\":$k,\"step\":10,\"matcher_cost\":{\"class\":2,\"bbox\":5,\"giou\":5,\"no_object\":0.1}}'"
}
python kernels/launch.py --slug hw2-cv3a --queue "$(queue 1)" --queue "$(queue 2)"
python kernels/launch.py --slug hw2-cv3b --queue "$(queue 3)" --queue "$(queue 4)"

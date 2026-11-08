#!/usr/bin/env bash
# YOLO26s (end-to-end, NMS-free) with the same protocol as YOLO11s: 5-fold CV, plus a P2-head variant on fold 0.
# One kernel, two queues (one per GPU), about 2 h wall time.
set -euo pipefail
ULTRA="python -m hw2.ultra_train --prep {PREP} --out {OUT} --no_test"
y() {  # fold
  echo "$ULTRA --model yolo26s.pt --name yolo26_cv_f$1 --cfg '{\"epochs\":12,\"imgsz\":1344,\"batch\":8,\"seed\":0}' --hw2 '{\"val_fold\":$1,\"step\":10}'"
}
p2="$ULTRA --model yolo26s-p2.yaml --name yolo26_p2_cv_f0 --cfg '{\"epochs\":12,\"imgsz\":1344,\"batch\":8,\"seed\":0}' --hw2 '{\"val_fold\":0,\"step\":10,\"weights\":\"yolo26s.pt\"}'"
python kernels/launch.py --slug hw2-y26 --queue "$(y 0) && $(y 1) && $(y 2)" --queue "$p2 && $(y 3) && $(y 4)"

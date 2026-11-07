#!/usr/bin/env bash
# More RT-DETR-l 1536 members for the final ensemble: trained on all 63 clips, each on a different frame offset
# (frames 3, 5, 7, 9 of every 10) and seed, so the members see different frames. Two kernels, two queues each.
set -euo pipefail
ULTRA="python -m hw2.ultra_train --prep {PREP} --out {OUT}"
m() {  # offset seed
  echo "$ULTRA --model rtdetr-l.pt --name rtdetr_all_o$1_s$2 --cfg '{\"epochs\":12,\"imgsz\":1536,\"batch\":3,\"seed\":$2}'" \
       "--hw2 '{\"val_fold\":0,\"step\":10,\"frame_offset\":$1,\"train_all\":true,\"matcher_cost\":{\"class\":2,\"bbox\":5,\"giou\":5,\"no_object\":0.1}}'"
}
python kernels/launch.py --slug hw2-ens-a --queue "$(m 2 1) && $(m 6 5)" --queue "$(m 4 2) && $(m 8 6)"
python kernels/launch.py --slug hw2-ens-b --queue "$(m 2 3) && $(m 6 7)" --queue "$(m 4 4) && $(m 8 8)"

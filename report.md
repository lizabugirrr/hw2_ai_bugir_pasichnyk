---
title: Homework 2 -- Multi-object Detection in Drone Video (VisDrone-VID)
author: Yelyzaveta Bugir, Nazar Pasichnyk
course: Artificial Intelligence · UCU
date: 2026-11-03
---

## Summary

We detect 8 object classes (pedestrian and people merged into `person`, awning-tricycle merged into `tricycle`, plus bicycle, car, van, truck, bus, motor) in drone video frames. The data are 63 labelled VisDrone-VID clips (27,047 frames, 1.22 M boxes) and a hidden test set of 48 clips (3,811 frames). The metric is COCO mAP@[.5:.95].

Our final model is **RT-DETR-l at 1536 px trained on all 63 clips**. It scores **0.3465** on the public leaderboard, first place at the time of writing. All model comparisons are made on one clip-level validation fold, so differences smaller than about 0.015 mAP should not be read as real (see [Noise](#how-much-to-trust-the-numbers)). Several things that looked good on validation did not help on the leaderboard, and we report these as they are.

## 1. Data and validation protocol

**Conversion.** VisDrone `ignored region` and `others` become don't-care regions. In training they are filled with gray, except where they overlap a valid box. In evaluation they are crowd regions (`iscrowd=1`), so detections inside them are neither rewarded nor penalised. Our evaluator is COCO-style with `maxDets=[1, 100, 500]`, matching the 500-detections-per-image limit of the submission.

![Original frame with ground truth and ignored regions (left) next to the prepared training frame with boxes read back from the label file (right)](images/eda_conversion.png)

**What the EDA showed** (figures below):

- Objects are tiny: the median box is about 32 px at the 1344 px model input and 50% of boxes are COCO-small. Scenes are dense (median 42 objects per frame, up to 180).
- Classes are imbalanced: car 44%, person 31%, motor 9%, bus only 0.8%.
- Consecutive frames are nearly identical: the same object has a median IoU of 0.89 with itself one frame later, 0.59 after five frames and 0.09 after twenty. We therefore train on every 10th frame, since adding more neighbouring frames adds almost no information (confirmed in §4).
- The default torchvision anchors (32-512) reach IoU ≥ 0.5 with only 75% of the boxes. Anchors 16-256 reach 98%, and 12-192 reach 99% (k-means on the box sizes gives 14 to 95 px).
- **Test clips differ from the labelled ones.** The test set contains low-resolution clips (680x382, 960x540) that do not exist among the labelled clips, while some labelled clips are 2.7K-4K, and test clips are much shorter (35-148 frames).

![Box sizes](images/eda_sizes.png)
![Class distribution](images/eda_classes.png)
![Objects per frame](images/eda_density.png)

**Split.** Frames of one clip must not appear on both sides, because neighbouring frames are near-duplicates. We use 5-fold `GroupKFold` by clip and, because of the compute budget (Kaggle T4, 30 GPU-hours per week), validate every model on **fold 0 only** (1,354 validation frames, one frame in four of the held-out clips). Training uses 2,189 frames (every 10th frame of the other folds) unless stated otherwise. Every detector is trained for 12 epochs. Fold assignments are stored in `src/hw2/configs/folds.csv`, because `GroupKFold` returns different folds in different scikit-learn versions.

Models trained on all 63 clips have no clean validation set. Their "fold 0" numbers (about 0.54) are in-sample and are not used in any comparison below.

## 2. Detectors and tuning experiments

Every family has a baseline and at least two modified runs. "fps" is the throughput logged by our validation script on a Kaggle T4 and is only meant for comparing the models with each other.

| Family | Run | What changed | mAP | AP50 | AP_small | fps |
|---|---|---|---|---|---|---|
| Faster R-CNN (R50-FPN v2) | frcnn_base | torchvision defaults | 0.2595 | 0.4813 | 0.0811 | 14.4 |
| | frcnn_a16 | anchors 16-256 | 0.2736 | 0.5149 | 0.0802 | 14.4 |
| | frcnn_giou | GIoU RoI box loss, 2000 test proposals | 0.2608 | 0.4849 | 0.0778 | 10.9 |
| RetinaNet (anchor-based) | retina_base | defaults | 0.2287 | 0.4175 | 0.0454 | 18.6 |
| | retina_a12 | anchors 12-192 | 0.1834 | 0.3992 | 0.0664 | 19.2 |
| YOLO11s (anchor-free) | yolo_base | 1344 px | 0.2449 | 0.4427 | 0.0797 | 74.9 |
| | yolo_base_s1 | same, other seed | 0.2294 | 0.4123 | 0.0769 | 71.3 |
| | yolo_nwd | box loss 0.5·CIoU + 0.5·NWD | 0.2489 | 0.4496 | 0.0739 | 76.3 |
| | yolo_1920 | input 1920 px | 0.2511 | 0.4517 | 0.0837 | 37.3 |
| RT-DETR-l (query-based) | rtdetr_base | 300 queries, 1344 px | 0.2971 | 0.5424 | 0.1085 | 10.0 |
| | rtdetr_q500 | 500 queries / 200 denoising queries | 0.2973 | 0.5414 | 0.0970 | 12.2 |
| | rtdetr_cost | Hungarian cost class 2, L1 5, GIoU 5, no-object 0.1 | 0.3004 | 0.5436 | 0.1059 | 12.1 |
| | rtdetr_s5 | same as `rtdetr_cost`, other frame subsample (noise check) | 0.2945 | 0.5291 | 0.0973 | 12.6 |
| | rtdetr_s5_1536 | input 1536 px | **0.3096** | 0.5510 | 0.1024 | 10.2 |
| | rtdetr_x | RT-DETR-x, 1536 px | 0.3135 | 0.5520 | 0.1082 | 6.7 |

![Per-epoch validation mAP: RT-DETR versus the CNN detectors](images/curve_rtdetr_vs_cnn.png)

(The curves use each framework's own per-epoch validation on every 20th fold-0 frame, so they show trends, not exact values.)

**Loss modifications**

- *GIoU in the Faster R-CNN RoI head* replaces smooth-L1 with $1-\mathrm{GIoU}$. It raised small-object recall (0.353 to 0.374) but also produced more detections inside ignored regions and in background, and mAP did not change (0.2595 to 0.2608).
- *NWD in YOLO* uses the normalised Wasserstein distance, $\mathrm{NWD}=\exp\!\big(-W_2(\mathcal N_a,\mathcal N_b)/C\big)$ with $C=32$, to give a smoother signal for tiny boxes. mAP rose by 0.004 but AP_small fell (0.0797 to 0.0739), so it did not do what it was meant to do.
- *Hungarian matcher cost in RT-DETR* gave 0.3004 against 0.2971, which is smaller than the spread between two runs of the same configuration (0.3004 versus 0.2945).

**Reading the table**

1. The query-based detector is the clear winner: RT-DETR-l at 1344 px (about 0.295-0.300 over three runs) is 0.02-0.03 mAP above the best CNN detector (Faster R-CNN with tuned anchors, 0.2736) and 0.05 above YOLO11s, and its AP_small is higher (about 0.10 versus 0.08). Its speed (10-12 fps) is close to Faster R-CNN (11-14 fps) and far below YOLO (75 fps).
2. Matching the anchors to the data helped the two-stage model (+0.014, borderline given the noise below) but hurt RetinaNet badly (-0.045). We did not investigate why.
3. Resolution is the most useful single change for RT-DETR: 1344 px to 1536 px gave +0.01 to +0.015 on validation and +0.029 on the leaderboard (0.3137 to 0.3426). For YOLO, 1920 px gave +0.006, within noise.
4. RT-DETR-x (0.3135) is no better than RT-DETR-l (0.3096) at the same resolution, at about two thirds of the speed.

### How much to trust the numbers

Two YOLO runs with an identical configuration and different seeds differ by 0.015 mAP (0.2449 versus 0.2294), and two RT-DETR runs by 0.006. Since we have a single validation fold of about 12 clips, we treat differences below roughly 0.015 as inconclusive. Only the family ranking (RT-DETR above the rest) and the effect of the input resolution are clearly outside this noise.

## 3. Where the errors are

For each model we match detections with score ≥ 0.3 to the ground truth and label them as correct, duplicate, wrong class, badly localised, inside an ignored region, or background. We also count misses by object size. The full tables and the three automatically selected failure frames per family are in notebooks 02-04.

![Failure cases, Faster R-CNN](images/fail_frcnn.png)
![Failure cases, one-stage detectors](images/fail_onestage.png)
![Failure cases, RT-DETR](images/fail_rtdetr.png)

- **Small objects are missed.** Recall on small objects is 0.29-0.37 for the CNN detectors and about 0.52 for RT-DETR, while recall on large objects is 0.85-0.93 for all of them. The frames with the most missed small objects come from the same clip (`uav0000124_00944`) for all three families. In the one selected for Faster R-CNN, 29 of the 132 ground-truth objects are small and unmatched.
- **Class confusions come from similar vehicle types.** The frame with the most class confusions (`uav0000288_00001`) is the same for all three families. Wrong-class detections are 13% of the correct ones for Faster R-CNN and YOLO, 17% for RT-DETR and 26% for RetinaNet. Per-class AP is lowest for bicycle (0.05-0.12), motor (0.07-0.16) and person (0.15-0.24), the smallest classes.
- **Many false alarms are inside ignored regions.** At score ≥ 0.3 the number of detections in ignored regions is large for every model and is lowest for the two-stage and one-stage detectors (11k-16k) and highest for RT-DETR (29k-34k). These are not penalised by the metric. The counts are not directly comparable between frameworks, because RT-DETR scores are distributed differently, so we do not draw conclusions from the false-alarm totals.
- **RetinaNet and Faster R-CNN produce more duplicates** than the others (1,850 and 425 against 355-406 for YOLO), as expected from their dense anchors plus NMS, whereas RT-DETR's one-to-one matching removes most of this need.

## 4. Final pipeline and what did not help

Final model: RT-DETR-l at 1536 px, trained on all 63 clips, with inference at 1536 px and `conf=0.001`.

| Submission | Description | Fold 0 mAP | Public LB |
|---|---|---|---|
| frcnn_base | Faster R-CNN baseline | 0.2595 | 0.2125 |
| rtdetr_base | RT-DETR-l 1344 | 0.2971 | 0.3137 |
| rtdetr_1536 | RT-DETR-l 1536 (fold-0 training only) | 0.3096 | 0.3426 |
| **rtdetr_all_1536** | RT-DETR-l 1536, all 63 clips | n/a (trained on the fold) | **0.3465** |
| + flip TTA | same model, horizontal-flip TTA | 0.3180 | 0.3458 |
| ensemble | RT-DETR (all) + TTA with Faster R-CNN (16-256 anchors), WBF 2:1, IoU 0.7 | 0.3322 | 0.3251 |
| low-res augmentation | all-clips model trained with random down-sampling | 0.2961 | 0.3245 |

![Fold-0 validation mAP against the public leaderboard score](images/val_vs_lb.png)

Inference variants on fold 0 (RT-DETR-l 1536):

| Strategy | mAP | AP_small | s/frame |
|---|---|---|---|
| full frame | 0.3095 | 0.1022 | 0.13 |
| + flip TTA | 0.3180 | 0.1037 | 0.23 |
| 2x2 tiles (SAHI-like) | 0.2865 | 0.0895 | 0.29 |
| 2x2 tiles + TTA | 0.2956 | 0.0921 | 0.39 |
| 3x3 tiles | 0.2712 | 0.0833 | 0.39 |

- **Tiling hurt.** Cutting the frame into tiles lowered mAP, including AP_small. The model already sees the frame at 1536 px, so tiles mostly add duplicated and truncated boxes.
- **More training frames hurt.** Training on 2.5 times more frames (every 4th instead of every 10th) lowered validation mAP (0.2911 against 0.3096 at 1536 px). Those frames are near-duplicates and the model saturates after one or two epochs, so the new frames mostly add overfitting to the training clips. The diversity is in the number of clips, not frames.
- **The Faster R-CNN ensemble disagrees between validation and leaderboard.** On fold 0 it is the best result (+0.014 over TTA alone, 0.3322), but on the leaderboard it is 0.021 *worse* than the single model. TTA alone gives +0.0085 on fold 0 and -0.0007 on the leaderboard. We do not have a verified explanation. One candidate is the distribution gap below; another is that a single fold of about 12 clips is simply a poor proxy for the 48 test clips.
- **A resolution gap exists, but we cannot prove it is the cause.** The test set contains low-resolution clips that the labelled data lacks (§1). Degrading fold 0 to 800 px lowers the base model from 0.3096 to 0.2776 (AP_small 0.1024 to 0.0713), so the gap is real. We tried to close it with low-resolution augmentation (30% of images at 640-1000 px). It helped on degraded input (0.2776 to 0.2784, AP_small 0.0713 to 0.0814) but cost 0.0135 on clean input, and on the leaderboard the augmented all-clips model scored 0.3245 against 0.3465. So augmentation was not the fix, and whether this gap explains the ensemble's drop remains a hypothesis.

![Fold-0 mAP of the base model on clean and degraded frames](images/resolution_shift.png)

## 5. Bonus tasks

### 5.1 Temporal information

We implemented Seq-NMS-style rescoring (Han et al., 2016) without the suppression step. Same-class detections in consecutive frames are linked when their IoU exceeds a threshold, and dynamic programming finds the highest-scoring path through each detection backwards and forwards in time. The score becomes $s'=(1-\beta)\,s+\beta\,\bar s_{\text{path}}$. We also tried T-CNN-style propagation, which copies confident neighbour detections into the current frame with a decayed score. Both run as post-processing on the RT-DETR predictions for consecutive frames of the fold-0 clips (RT-DETR-l 1536, score ≥ 0.01).

| Method | mAP | AP50 | AP_small |
|---|---|---|---|
| no temporal step | 0.3095 | 0.5511 | 0.1022 |
| first attempt: averaging scores over a window of ±1/±2/±4 frames (best: ±2, α=0.3) | 0.3090 | 0.5490 | 0.1028 |
| Seq-rescore, mean path, β=0.5, IoU 0.5 | 0.3141 | 0.5597 | 0.0974 |
| Seq-rescore, mean path, β=0.3, IoU 0.5 | 0.3152 | 0.5617 | 0.1013 |
| Seq-rescore, max path, β=0.5, IoU 0.5 | 0.3110 | 0.5509 | 0.1000 |
| **Seq-rescore, mean path, β=0.5, IoU 0.7** | **0.3176** | 0.5635 | 0.0988 |
| Propagation (conf 0.6, decay 0.5) | 0.3097 | 0.5524 | 0.1023 |
| Propagation (conf 0.8, decay 0.9) | 0.3088 | 0.5496 | 0.1022 |

The naive window average gives nothing and rescoring along linked paths gives about +0.008 mAP, with the gain in AP50 (+0.012) larger than in AP_small, which falls slightly. Propagation gives nothing because the recall (AR500) does not increase. **Caveats:** the settings were chosen on the same fold they are evaluated on, and the gain is about half of the run-to-run noise from §2, so it is weak evidence at best. We did not put it into the final submission, and we have no leaderboard measurement for it.

### 5.2 Visualisation and Weights & Biases

Notebook 06 looks inside both detector types on fold-0 frames. Code is in `src/hw2/viz.py`. Training curves, losses, configurations and validation predictions of every run are logged to Weights & Biases by `src/hw2/wandb_log.py`: <!-- TODO (authors): paste the public W&B project link here -->

**Grad-CAM for YOLO11s.** The target is the class score of the prediction that best covers a ground-truth box. Maps are taken at three backbone stages and three neck outputs.

![Grad-CAM for a detected and a missed small car](images/viz_gradcam_found_missed.png)
![Grad-CAM for a small and a large object](images/viz_gradcam_small_large.png)

Over 90 ground-truth objects (30 per size bin) the prediction chosen for an object comes from a level that matches its size:

| Object size | stride 8 (P3) | stride 16 (P4) | stride 32 (P5) |
|---|---|---|---|
| small (<32 px) | 77% | 20% | 3% |
| medium | 80% | 20% | 0% |
| large (>96 px) | 0% | 87% | 13% |

![Level used per object size, and Grad-CAM concentration per layer](images/viz_gradcam_levels.png)

However, the maps themselves are **not sharply focused on the object**. We measured the share of the heat-map mass inside the box (dilated by 16 px) divided by the share of the image the window covers, where 1 means no better than uniform. The median is between 0.7 and 1.3 for the backbone layers and between 0.5 and 2.1 for the neck layers, except 3.8 for small objects at the stride-32 neck output. So Grad-CAM at these layers shows which scale is used, but only weakly shows where the evidence is. In the first figure above the neck P4 and P5 maps are almost empty. We did not investigate why and did not try other CAM variants.

**RT-DETR deformable attention.** We keep the sampling locations and weights of all six decoder layers for 60 random validation frames (4,098 queries with score ≥ 0.5).

![Attention sampling points of a small and a large query, decoder layers 1, 3 and 6](images/viz_rtdetr_attention_points.png)
![Sampling spread and the feature level used, per decoder layer](images/viz_rtdetr_attention.png)

- The sampling points get **more spread out with depth**, not less. Measured relative to the object size, the median spread of small objects grows from 0.21 in layer 1 to 0.54 in layer 6 (large objects: 0.19 to 0.40). In the first four layers queries sample tightly inside the object (about a quarter of its size), and in the last two layers they take wider context.
- Small objects rely on the finest feature map: the share of attention weight on the stride-8 level for small objects rises from 0.58 (layer 1) to 0.65 (layers 3-4) and then falls to 0.38 in layer 6, where weight moves to the coarser levels. For large objects the stride-8 share is only 0.21 to 0.38. This agrees with the YOLO result above.

**Query embeddings.** We project the 256-d last-layer embedding of 5,937 queries (score ≥ 0.3) with t-SNE.

![t-SNE of decoder query embeddings by class, size and outcome](images/viz_rtdetr_tsne.png)

A $k$-nearest-neighbour check (k = 10) in the original space finds that the majority class of the neighbours equals the predicted class for 99.5% of queries. This is expected, since the embedding feeds the class head directly, so it shows that the embedding is consistent with the output rather than that it is a good representation. Per class the agreement is lowest for van (0.87), truck (0.92) and bus (0.92), the three rarest vehicle classes.

### 5.3 Custom architectures (negative result)

Small objects are the bottleneck, so we modified the YOLO11s detection head. `P2` adds a stride-4 level, `P2-noP5` also removes the stride-32 level, and `P3P4` keeps only strides 8 and 16. Because the new head layers have no pretrained weights, `P2-hinit` copies the weights of the matching levels of the pretrained head (297 of 593 tensors transfer).

| Run | mAP | AP50 | AP_small | AP_large | fps |
|---|---|---|---|---|---|
| yolo_base (reference) | 0.2449 | 0.4427 | 0.0797 | 0.4004 | 74.9 |
| yolo_base_s1 (other seed) | 0.2294 | 0.4123 | 0.0769 | 0.3859 | 71.3 |
| P2 head, new weights | 0.2178 | 0.4301 | 0.0756 | 0.3524 | 50.9 |
| P2 head, transferred head weights | 0.2446 | 0.4656 | 0.0797 | 0.3893 | 54.1 |
| P2, no P5 | 0.2210 | 0.4359 | 0.0763 | 0.3551 | 56.9 |
| P3 + P4 only | 0.2308 | 0.4274 | 0.0733 | 0.3802 | 74.5 |

None of the variants beats the reference, and the differences are at the level of the seed-to-seed noise. The P2 head did not improve AP_small (0.0797 at best, the same as the reference). Transferring the head weights recovers the loss from random initialisation (0.2178 to 0.2446) but does not go beyond the pretrained model. The P2 head costs about 30% of the speed. Our conclusion is limited to this setting (12 epochs, a single seed per variant). We cannot exclude that more training would change the picture.

## 6. Conclusions

- A query-based detector (RT-DETR) at high input resolution was the best choice for this dataset. The advantage over CNN detectors (+0.02 to +0.03 mAP over the best CNN detector) and the benefit of 1536 px over 1344 px are the only effects clearly above the noise of our protocol.
- Most of our other ideas (tiling, more frames, loss changes, extra detection levels, an ensemble with Faster R-CNN, low-resolution augmentation) gave nothing or hurt on the leaderboard. A lesson from this is that **a single validation fold is a weak guide**: the ensemble had the best fold-0 score of any inference variant and was our worst use of a submission.
- The final result is 0.3465 mAP on the public leaderboard.

**Limitations.** One validation fold, one or two seeds per configuration, a hidden test set that differs from the labelled data in resolution, and a temporal gain (+0.008) that is not confirmed on the leaderboard. All numbers are COCO mAP computed with our own evaluator on fold 0, except the leaderboard column.

**Reproducing.** Code is in `src/hw2/` (data preparation `prep.py`, training `tv_train.py` and `ultra_train.py`, inference `infer.py`, post-processing `postprocess.py`, evaluation `evaluate.py`). Training ran on Kaggle T4 GPUs through the scripts in `kernels/`. All metrics are in `results/` and the analysis notebooks are in `notebooks/`.

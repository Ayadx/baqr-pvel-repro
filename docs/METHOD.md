# Method and evaluation protocol

## Detector and training design

The experiments used Co-DETR/Co-DINO with a ViT-L backbone and twelve PV-EL defect classes. The detector's native query classification uses Quality Focal Loss, so unmatched queries already contribute negative classification supervision.

All five models were trained with seed **4** for **30 epochs**, using **715 defect-annotated training images** and **175 defect-annotated validation images**. The defect-only baseline used no additional good-labelled training images; Clean control, Plain BAQR, Entropy BAQR and top-50 BCE used the same **100 good-labelled training images**. Checkpoints were selected according to the highest validation AP50 within each run. The recorded environment used Python 3.8.10, PyTorch 1.11.0, CUDA 11.3, `mmcv-full` 1.5.0 and MMDetection 2.25.3.

Training used AdamW (learning rate `1e-4`, weight decay `0.01`), a cosine learning-rate schedule, a nominal effective batch size of 16, and a 720-pixel nominal input scale. Additional training details and augmentation settings are specified in the manuscript.

## BAQR auxiliary

On each training image with an empty ground-truth annotation list, let `p_q = max_c sigmoid(z_qc)` be the maximum foreground confidence of matching query `q` from the final decoder layer. For the top **K = 50** queries by `p_q`, the additional loss is

`L_BAQR = 0.10 * mean_{empty images, selected queries} max(p_q - 0.05, 0)`.

Denoising queries are excluded; all native detector losses are retained. The original loss function is available in [`losses/baqr_loss.py`](../losses/baqr_loss.py). It can be integrated into the Co-DINO query head under the archived settings `baqr_enabled=True`, `baqr_topk=50`, `baqr_margin=0.05` and `baqr_lambda=0.10`.

## Conventional BCE comparison

For identical empty-image activation and top-50 matching-query selection, the BCE comparator adds a target-zero winner-class binary cross-entropy:

`L_BCE50 = 0.10 * mean_{empty images, selected queries} softplus(max_c z_qc)`.

The implementation is in [`losses/bce_neg_auxiliary.py`](../losses/bce_neg_auxiliary.py). The nominal auxiliary coefficients are equal; however, the two loss functions are **not matched in gradient magnitude**. The manuscript therefore treats BCE as a complementary training comparison, not a controlled test establishing hinge-specific superiority.

## Evaluation protocol

The evaluation uses 500 dataset-labelled good images and 2,000 sampled official defect-test images containing 3,550 annotations. At score `>= 0.25`, postprocessing uses classwise hard NMS at IoU `0.50` (up to 300 predictions per image). Greedy class-aware annotation matching uses IoU `>= 0.50`.

The analysis distinguishes the proportion of good-labelled images with at least one prediction, retained boxes on those images, correct annotation matches, unmatched boxes on defect-annotated images and native COCO AP. The Python evaluation tools recalculate fixed-score outcomes and paired events from saved post-NMS predictions. COCO AP uses the archived native evaluator outputs supplied in [`reference/native_coco_ap.json`](../reference/native_coco_ap.json).

## Interpretation of the exploratory model

Entropy BAQR is reported as an exploratory configuration. The original development AP50 report (`0.885`) differs from a later evaluation (`0.899`); the discrepancy is not resolved. The manuscript therefore does not rely on this variant as evidence that an entropy gate improves the primary Clean-control–Plain-BAQR result.

The reported comparisons concern fixed seed-4 checkpoints and the stated within-dataset evaluation cohorts. The original complete detector training environment is not bundled in this repository; its loss implementations and the reproducible offline metric-evaluation tools are supplied separately.

# BAQR — photovoltaic electroluminescence inspection

Code and evaluation tools associated with the manuscript *Background-aware query regularization in photovoltaic electroluminescence inspection: alarm incidence and defect sensitivity* (target journal: *Solar Energy Materials and Solar Cells*).

The repository provides the BAQR loss, a top-50 negative-query BCE comparator, and scripts to recalculate detection metrics from saved predictions. The five corresponding model checkpoints are identified in the [Zenodo record](https://doi.org/10.5281/zenodo.24160247) (DOI reserved; record accessible after publication).

## Method

BAQR supplements the native Co-DETR/Co-DINO classification objective on training images with empty defect annotations. For each such image, the 50 final matching queries with the highest foreground confidence receive a margin penalty (margin 0.05; coefficient 0.10). The detector architecture and inference procedure remain unchanged.

The primary comparison uses a **Clean control** trained on the same images and schedule as **Plain BAQR**. A secondary top-50 winner-class BCE model provides a comparison with another negative-query penalty. Detailed definitions and training settings appear in [docs/METHOD.md](docs/METHOD.md).

## Results

All reported checkpoints were evaluated on 500 dataset-labelled good images and 2,000 defect-annotated images containing 3,550 annotations. Fixed-operating-point metrics use score 0.25, classwise hard NMS at IoU 0.50 and class-aware annotation matching at IoU 0.50.

| Model | Good-image alarms / 500 | Matched objects / 3,550 | Unmatched boxes on defect images | Matched cracks / 279 | COCO AP |
|---|---:|---:|---:|---:|---:|
| Defect-only baseline | 282 | 3,273 | 1,468 | 228 | 0.606 |
| Clean control | 130 | 3,312 | 1,978 | 241 | 0.569 |
| Plain BAQR | 103 | 3,300 | 1,622 | 235 | 0.576 |
| Entropy BAQR (exploratory) | 83 | 3,288 | 1,372 | 225 | 0.566 |
| Top-50 BCE | 70 | 3,221 | 1,040 | 211 | 0.558 |

Compared with Clean control, Plain BAQR reduced good-image alarm incidence from **26.0% to 20.6%** (37 resolved and 10 introduced alarms). The same comparison yielded 12 fewer matched object annotations, including six fewer matched cracks. The BCE comparator reduced alarm incidence further, with a larger reduction in object and crack matches. These results characterize the alarm–localization trade-off at the specified operating point.

Native COCO AP values were calculated with the original detector evaluator. This repository's offline evaluation code reconstructs the fixed-score and paired-event metrics from saved predictions; it does not recalculate native COCO AP.

## Install and test

The offline evaluation tools support Python 3.9 or newer; this environment is separate from the original model-training stack.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[test]'
python -m pytest -q
```

To run the synthetic loss-gradient tests, also install `.[loss]`.

## Recalculate the results

With authorized access to the evaluation manifests and saved predictions, arrange the input files outside the repository:

```text
validation-data/
  manifests/
    defects.json
    good.json
  predictions/
    baseline/{defects,normals}.jsonl.gz
    clean_control/{defects,normals}.jsonl.gz
    plain_baqr/{defects,normals}.jsonl.gz
    entropy_baqr/{defects,normals}.jsonl.gz
    bce_top50/{defects,normals}.jsonl.gz
```

Run:

```bash
baqr-recompute \
  --data-root ../validation-data \
  --out results/recomputed \
  --expected reference/expected_results.json
```

The command generates model summaries, per-class recall, threshold-filtered results and paired comparisons. For the threshold plot:

```bash
python -m pip install -e '.[plot]'
python scripts/plot_thresholds.py \
  --csv results/recomputed/threshold_points.csv \
  --out results/recomputed/thresholds.png
```

See [docs/DATA.md](docs/DATA.md) for the input formats and [reference/manuscript_results.csv](reference/manuscript_results.csv) for aggregated values.

## Checkpoints and data access

The five selected checkpoints are identified by SHA-256, independently of their filenames. Model identifiers, selected epochs and hashes are listed in [docs/MODELS.md](docs/MODELS.md) and in the Zenodo companion manifest. The PVEL-AD images are obtained from the dataset provider and are not redistributed here. Saved image-level detections and annotations are maintained separately according to their access conditions.

The evaluation reflects the stated PVEL-AD cohorts and seed-4 checkpoints. Uncertainty estimates, the exploratory variant and the BCE comparison are interpreted as specified in the accompanying manuscript and [methods documentation](docs/METHOD.md).

## Checkpoint filenames

The checkpoint files in the Zenodo release use these names:

| Model | Filename |
|---|---|
| `baseline` | `0.baseline.pth` |
| `clean_control` | `1.clean_control.pth` |
| `plain_baqr` | `2.baqr_only.pth` |
| `entropy_baqr` | `3.baqr_entropy_only.pth` |
| `bce_top50` | `Top-50 negative BCE.pth` |

The filenames above are recorded in [reference/checkpoints.json](reference/checkpoints.json) and the Zenodo `model_manifest.json`. Model identity is verified by SHA-256 rather than by the name alone.

```bash
python scripts/verify_checkpoint_files.py --directory /path/to/checkpoints
```

To generate metadata for a future release from local, hash-verified weights:

```bash
python scripts/prepare_zenodo_files.py \
  --checkpoint-dir /path/to/checkpoints \
  --out /path/to/zenodo-companion
```

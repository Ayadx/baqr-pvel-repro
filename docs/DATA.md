# Evaluation inputs

PVEL-AD image data should be obtained from the [original dataset project](https://github.com/binyisu/PVEL-AD) under the provider's access conditions. This repository distributes neither the original EL images nor image-level experimental data.

To recalculate manuscript metrics, supply the fixed study cohorts as COCO-style JSON manifests:

- `manifests/defects.json` — 2,000 images with 3,550 annotations.
- `manifests/good.json` — 500 images without defect annotations.

For each of the five study configurations, supply two gzip-compressed JSON Lines files in `predictions/<model>/`: `defects.jsonl.gz` and `normals.jsonl.gz`. Each line contains one image identifier and its retained predictions:

```json
{"image_id": 1, "file_name": "example.jpg", "predictions": [{"category_id": 0, "score": 0.9, "bbox_xyxy": [10, 20, 50, 80]}]}
```

This is a format illustration, not a PVEL-AD example. Prediction records must match the manifest image order and IDs. Predictions must have undergone score filtering at `>= 0.25` and classwise hard NMS at IoU `0.50` before being saved; the script does not rerun detector inference or NMS. Results at stricter score thresholds use only the preserved predictions.

The command checks the expected study-cohort sizes, annotation identifiers, prediction order, class IDs, and box geometry. The optional `--allow-other-cohorts` flag permits running the same evaluation code on other datasets without treating the output as a replication of the published cohorts.

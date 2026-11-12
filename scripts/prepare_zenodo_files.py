#!/usr/bin/env python3
"""Build Zenodo metadata using the *actual* names of verified .pth checkpoints.

Example:
    python scripts/prepare_zenodo_files.py \
        --checkpoint-dir /path/to/checkpoints --out /path/to/zenodo-companion

Neither checkpoint contents nor filenames are changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from verify_checkpoint_files import DEFAULT_REFERENCE, match_folder

LABELS = {
    'baseline': 'Defect-only baseline',
    'clean_control': 'Clean control',
    'plain_baqr': 'Plain BAQR',
    'entropy_baqr': 'Entropy BAQR (exploratory)',
    'bce_top50': 'Top-50 negative BCE',
}


def generate(reference_path: Path, checkpoint_dir: Path, output_dir: Path) -> dict:
    reference = json.loads(Path(reference_path).read_text(encoding='utf-8'))
    matched = match_folder(checkpoint_dir, reference)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
      'resource_type': 'Model',
      'title': 'BAQR: Trained Co-DINO Checkpoints for Photovoltaic Electroluminescence Inspection',
      'doi': reference['doi'],
      'version': '1.0.0',
      'model_family': 'Co-DETR / Co-DINO ViT-L',
      'training_seed': 4,
      'checkpoint_selection': 'Best validation AP50 within the 30-epoch run',
      'models': {}
    }
    tsv_lines = ['model_id\tfilename\tsha256']
    rows = []
    for model, data in reference['models'].items():
        path = matched[model]
        filename = path.name
        sha = data['sha256']
        epoch = int(data['epoch'])
        manifest['models'][model] = {
            'filename': filename,
            'epoch': epoch,
            'sha256': sha,
            'size_bytes': path.stat().st_size,
        }
        tsv_lines.append(f'{model}\t{filename}\t{sha}')
        rows.append(f'| {LABELS[model]} | `{filename}` | {epoch} |')
    readme = f"""# BAQR trained Co-DINO checkpoints

Model weights for the manuscript *Background-aware query regularization in photovoltaic electroluminescence inspection: alarm incidence and defect sensitivity* (target: *Solar Energy Materials and Solar Cells*).

**Zenodo DOI:** `{reference['doi']}`  
**Release version:** `1.0.0`  
**Training seed:** `4`

## Checkpoint files

| Configuration | File | Selected epoch |
|---|---|---:|
{chr(10).join(rows)}

All five models were trained for 30 epochs; their selected checkpoints maximize validation AP50. The defect-only baseline used 715 labelled training images. The other four configurations used the same 715 labelled and 100 good-labelled training images, with 175 annotated validation images.

The principal Clean-control–Plain-BAQR comparison produced 130 versus 103 alarms on the same 500 good-labelled images and 3,312 versus 3,300 matched annotations in the 2,000-image defect-test sample. Details and full results are provided in the associated manuscript and the GitHub repository `baqr-pvel-repro`.

## Verify file integrity

From the accompanying GitHub checkout run:

```bash
python scripts/verify_checkpoint_files.py --directory /path/to/checkpoints
```

`SHA256_BY_MODEL.tsv` and `model_manifest.json` record the model IDs, names and SHA-256 hashes. Filenames are read from verified checkpoint files and do not affect model identity. The PVEL-AD images and image-level predictions are not part of this archive; obtain access to dataset images from the original provider.
"""
    (output_dir/'README.md').write_text(readme, encoding='utf-8')
    (output_dir/'model_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (output_dir/'SHA256_BY_MODEL.tsv').write_text('\n'.join(tsv_lines)+'\n', encoding='utf-8')
    return manifest


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    args=ap.parse_args()
    try:
        manifest=generate(args.reference, args.checkpoint_dir, args.out)
    except ValueError as exc:
        ap.exit(1, f'FAIL: {exc}\n')
    for model, item in manifest['models'].items():
        print(f'{model:<15} {item["filename"]}')
    print('PASS: all five checkpoints matched by SHA-256; companion files generated.')


if __name__ == '__main__':
    main()

# Model checkpoints

Five ViT-L Co-DETR/Co-DINO model checkpoints accompany the BAQR manuscript. The files are associated with [Zenodo DOI 10.5281/zenodo.24160247](https://doi.org/10.5281/zenodo.24160247).

| Model | Zenodo filename | Selected epoch | Expected SHA-256 |
|---|---|---:|---|
| `baseline` | `0.baseline.pth` | 18 | `c4c6812a3ca46bc404571c3fa694afc76e39801f3e0305b0bb79727b2be91d16` |
| `clean_control` | `1.clean_control.pth` | 16 | `3fc5eea6b2d5f415b57c5082a7cf90ca96da509d1eeb3e754046f4f7ec6a54a8` |
| `plain_baqr` | `2.baqr_only.pth` | 16 | `3150b1b8d615fa5ffa29e941d9d8ee733ade3b000faa950b67f129a8aab5b5bd` |
| `entropy_baqr` | `3.baqr_entropy_only.pth` | 9 | `a3c159d28c0d816d073822832da5923ebd24c9f243f94adac99577d3a04582b2` |
| `bce_top50` | `Top-50 negative BCE.pth` | 9 | `852b8aaef7f098581ae7579986fce65adc3a63b3fc0fddab90d18c90efc45bdb` |

The checksum, rather than the filename, identifies the checkpoint bytes. The names above correspond to the files shown in the Zenodo upload; the fingerprints originate from the archived training checkpoints and require confirmation against the uploaded files.

## Check files after download

```bash
python scripts/verify_checkpoint_files.py --directory /path/to/checkpoints
```

Or check a single model:

```bash
python scripts/verify_checkpoint_files.py --model plain_baqr --file /path/to/2.baqr_only.pth
```

For future versions or renamed files, regenerate the companion metadata from verified checkpoints:

```bash
python scripts/prepare_zenodo_files.py \
  --checkpoint-dir /path/to/checkpoints \
  --out /path/to/zenodo-companion
```

The generator uses SHA-256 to identify each model, and preserves the original `.pth` files. Images and image-level predictions are not distributed as part of the public code repository.

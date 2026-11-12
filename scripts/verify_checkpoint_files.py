#!/usr/bin/env python3
"""Check downloaded model files against the five published SHA-256 fingerprints.

Works with renamed checkpoint files; no filename assumptions are made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_REFERENCE = Path(__file__).resolve().parents[1] / 'reference/checkpoints.json'


def sha256_file(path: Path, block_size: int = 4*1024*1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(block_size), b''):
            h.update(chunk)
    return h.hexdigest()


def match_folder(folder: Path, reference: dict, require_all: bool = True) -> dict[str, Path]:
    """Identify model IDs by file hash; reject duplicate, unknown, and missing weights."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f'Checkpoint folder not found: {folder}')
    expected = {entry['sha256'].lower(): model for model, entry in reference['models'].items()}
    found: dict[str, Path] = {}
    candidates = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.pth')
    if not candidates:
        raise ValueError(f'No .pth checkpoint files found in {folder}')
    for path in candidates:
        digest = sha256_file(path).lower()
        model = expected.get(digest)
        if model is None:
            raise ValueError(f'Unknown checkpoint contents for {path.name} (SHA-256: {digest})')
        if model in found:
            raise ValueError(f'Multiple files match {model}: {found[model].name}, {path.name}')
        found[model] = path
    if require_all and set(found) != set(reference['models']):
        missing = sorted(set(reference['models']) - set(found))
        raise ValueError(f'Missing checkpoints for: {", ".join(missing)}')
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--directory', type=Path, help='Folder containing the five .pth files')
    group.add_argument('--file', type=Path, help='Single .pth checkpoint')
    parser.add_argument('--model', help='Model ID, required with --file')
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text(encoding='utf-8'))
    if args.directory:
        if args.model:
            parser.error('--model can only be used together with --file')
        try:
            found = match_folder(args.directory, ref)
        except ValueError as exc:
            parser.exit(1, f'FAIL: {exc}\n')
        for model in ref['models']:
            print(f'PASS  {model:<15} {found[model].name}')
        print(f'All {len(found)} expected models verified by SHA-256.')
        return
    if not args.model or args.model not in ref['models']:
        parser.error('--file requires --model among: '+', '.join(ref['models']))
    if not args.file.is_file():
        parser.error(f'File not found: {args.file}')
    got = sha256_file(args.file)
    expected = ref['models'][args.model]['sha256'].lower()
    if got != expected:
        parser.exit(1, f'FAIL: wrong weights for {args.model}; SHA-256 {got}\n')
    print(f'PASS: {args.file.name} contains the archived {args.model} model.')


if __name__ == '__main__':
    main()

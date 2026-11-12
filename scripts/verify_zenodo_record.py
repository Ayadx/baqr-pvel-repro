#!/usr/bin/env python3
"""Inspect Zenodo checkpoint-record metadata without downloading model weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_DOI = "10.5281/zenodo.24160247"
DEFAULT_RECORD_ID = "24160247"


def get_record_files(record):
    f = record.get("files", [])
    if isinstance(f, dict):
        f = f.get("entries", [])
    if isinstance(f, dict):
        return list(f.values())
    return list(f)


def record_doi(record):
    identifier = record.get("doi")
    if identifier:
        return str(identifier).lower()
    pids = record.get("pids") or {}
    return str((pids.get("doi") or {}).get("identifier", "")).lower()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    p.add_argument("--doi", default=DEFAULT_DOI)
    p.add_argument("--record-json", type=Path, help="Offline metadata JSON, if Zenodo is inaccessible")
    args = p.parse_args()
    if args.record_json:
        record = json.loads(args.record_json.read_text(encoding="utf-8"))
    else:
        url = "https://zenodo.org/api/records/" + args.record_id
        req = Request(url, headers={"User-Agent": "BAQR-repro/0.3 (metadata check)"})
        try:
            with urlopen(req, timeout=20) as response:
                record = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            p.exit(2, f"Zenodo record cannot be verified at present: {exc}\n"
                      "A reserved DOI may not yet be a publicly published record.\n")
    observed = record_doi(record)
    if observed != args.doi.lower():
        p.exit(1, f"DOI mismatch: expected {args.doi!r}, observed {observed!r}\n")
    files = get_record_files(record)
    print("DOI record metadata verified:", observed)
    print("Title:", (record.get("metadata") or {}).get("title", "[unknown]"))
    print("Number of files listed:", len(files))
    for entry in files:
        print(" -", entry.get("key") or entry.get("filename") or entry.get("name", "[unnamed]"),
              "| bytes", entry.get("size") or entry.get("filesize", "unknown"),
              "|", entry.get("checksum", "checksum unavailable"))
    print("IMPORTANT: metadata listing is not a SHA-256 check of checkpoint bytes.")
    print("Compare downloaded checkpoint files with reference/checkpoints.json.")
    if not files:
        p.exit(3, "Record has no listed files accessible in the metadata.\n")


if __name__ == "__main__":
    main()

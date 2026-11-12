"""Tests of utility code without accessing Zenodo or checkpoint data."""
import importlib.util
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checkpoint_sha_works_on_synthetic_blob(tmp_path):
    mod = module("ckpt", "scripts/verify_checkpoint_files.py")
    p = tmp_path / "synthetic.bin"
    p.write_bytes(b"BAQR synthetic test only")
    assert mod.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_zenodo_file_metadata_shapes():
    mod = module("zenodo", "scripts/verify_zenodo_record.py")
    assert mod.record_doi({"doi": "10.5281/zenodo.24160247"}) == "10.5281/zenodo.24160247"
    assert mod.record_doi({"pids":{"doi":{"identifier":"10.5281/zenodo.24160247"}}}) == "10.5281/zenodo.24160247"
    item = {"key": "model.pth", "size": 100}
    assert mod.get_record_files({"files": [item]}) == [item]
    assert mod.get_record_files({"files": {"entries": {"model.pth": item}}}) == [item]

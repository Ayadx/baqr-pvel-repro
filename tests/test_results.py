"""Consistency checks for the five configurations reported in the manuscript."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "reference"
NAMES = {"baseline", "clean_control", "plain_baqr", "entropy_baqr", "bce_top50"}


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def manuscript_rows():
    with (ROOT / "manuscript_results.csv").open(newline="", encoding="utf-8") as f:
        return {row["model"]: row for row in csv.DictReader(f)}


def test_results_match_five_model_reference():
    rows = manuscript_rows()
    expected = read_json("expected_results.json")["models"]
    assert set(rows) == set(expected) == NAMES
    for model in NAMES:
        for key in ("good_alarms", "good_boxes", "tp", "fn", "fp_defect"):
            assert int(rows[model][key]) == expected[model][key]
        assert expected[model]["tp"] + expected[model]["fn"] == 3550


def test_native_ap_matches_manuscript_table():
    rows = manuscript_rows()
    ap = read_json("native_coco_ap.json")["models"]
    assert set(ap) == NAMES
    for model in NAMES:
        for key in ("AP", "AP50", "AP75"):
            assert float(rows[model][key]) == ap[model][key]


def test_crack_counts_and_primary_tradeoff():
    r = manuscript_rows()
    assert {name: int(r[name]["matched_cracks"]) for name in NAMES} == {
        "baseline": 228, "clean_control": 241, "plain_baqr": 235,
        "entropy_baqr": 225, "bce_top50": 211,
    }
    assert int(r["clean_control"]["good_alarms"]) - int(r["plain_baqr"]["good_alarms"]) == 27
    assert int(r["clean_control"]["tp"]) - int(r["plain_baqr"]["tp"]) == 12
    assert int(r["clean_control"]["matched_cracks"]) - int(r["plain_baqr"]["matched_cracks"]) == 6


def test_model_fingerprints_are_well_formed():
    models = read_json("checkpoints.json")
    assert models["doi"] == "10.5281/zenodo.24160247"
    assert set(models["models"]) == NAMES
    for item in models["models"].values():
        assert len(item["sha256"]) == 64
        assert item["epoch"] > 0

import json
from pathlib import Path


def test_expected_counts_are_self_consistent():
    root = Path(__file__).resolve().parents[1]
    r = json.loads((root/'reference/expected_results.json').read_text())
    for model, v in r['models'].items():
        assert v['tp'] + v['fn'] == 3550, model
        assert 0 <= v['good_alarms'] <= 500
    for pair, info in r['pairs'].items():
        assert sum(info['good_images'].values()) == 500, pair
        assert sum(info['annotated_objects'].values()) == 3550, pair
        assert sum(info['cracks'].values()) == 279, pair

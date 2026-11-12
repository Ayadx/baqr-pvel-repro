"""End-to-end analysis on a synthetic two-image example."""
import gzip
import json
from pathlib import Path

from baqr_repro.cli import run

def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')

def _pred(path, image_id, filename, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write(json.dumps({'image_id': image_id,
                            'file_name': filename,
                            'predictions': boxes}) + '\n')

def test_cli_with_small_synthetic_cohort(tmp_path):
    root = tmp_path / 'data'
    _save(root/'manifests/defects.json', {
        'images': [{'id': 10,'file_name': 'synthetic_defect'}],
        'annotations': [{'id': 22,'image_id': 10,'category_id': 0,
                         'bbox': [0,0,10,10]}]})
    _save(root/'manifests/good.json', {
        'images': [{'id': 11,'file_name': 'synthetic_good'}], 'annotations': []})
    box = {'category_id': 0,'score': 0.9,'bbox_xyxy': [0,0,10,10]}
    for name in ('clean_control','plain_baqr'):
        defect_predictions = [box] if name == 'clean_control' else []
        good_predictions = [box] if name == 'clean_control' else []
        _pred(root/'predictions'/name/'defects.jsonl.gz',10,'synthetic_defect',defect_predictions)
        _pred(root/'predictions'/name/'normals.jsonl.gz',11,'synthetic_good',good_predictions)
    out=tmp_path/'out'
    assert run(root,out,models=('clean_control','plain_baqr'),strict=False,bootstrap=False) == 0
    data=json.loads((out/'recalculated.json').read_text())
    assert data['models']['clean_control']['good_alarms'] == 1
    assert data['models']['plain_baqr']['tp'] == 0
    diff=data['pairs']['clean_control_vs_plain_baqr']
    assert diff['good_images']['reference_only'] == 1
    assert diff['cracks']['reference_only'] == 1

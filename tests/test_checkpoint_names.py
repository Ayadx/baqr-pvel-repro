"""Tests that checkpoint identities survive arbitrary renaming."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

REPO=Path(__file__).resolve().parents[1]


def load_script(name):
    import sys
    scripts=REPO/'scripts'
    sys.path.insert(0,str(scripts))
    p=scripts/name
    spec=importlib.util.spec_from_file_location(p.stem,p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_release(tmp_path):
    blobs={
        'baseline': b'BAQR test baseline',
        'clean_control': b'BAQR test control',
        'plain_baqr': b'BAQR test BAQR',
        'entropy_baqr': b'BAQR test entropy',
        'bce_top50': b'BAQR test BCE',
    }
    reference={'doi':'10.5281/zenodo.24160247','models':{}}
    for idx,(name,data) in enumerate(blobs.items()):
        file=tmp_path/('renamed weights '+str(idx)+'.pth')
        file.write_bytes(data)
        reference['models'][name]={'epoch':idx+1,'sha256':hashlib.sha256(data).hexdigest()}
    rp=tmp_path/'ref.json'
    rp.write_text(json.dumps(reference),encoding='utf-8')
    return rp,reference


def test_hash_matching_ignores_filename(tmp_path):
    rp,reference=synthetic_release(tmp_path)
    verify=load_script('verify_checkpoint_files.py')
    matched=verify.match_folder(tmp_path,reference)
    assert set(matched)==set(reference['models'])
    (tmp_path/'renamed weights 0.pth').rename(tmp_path/'even more professional.pth')
    matched2=verify.match_folder(tmp_path,reference)
    assert 'even more professional.pth'==matched2['baseline'].name


def test_generator_produces_exact_local_file_names(tmp_path):
    rp,_=synthetic_release(tmp_path)
    gen=load_script('prepare_zenodo_files.py')
    output=tmp_path/'output'
    manifest=gen.generate(rp,tmp_path,output)
    assert len(manifest['models'])==5
    for model,item in manifest['models'].items():
        assert (tmp_path/item['filename']).is_file()
    assert (output/'SHA256_BY_MODEL.tsv').is_file()
    checks=(output/'SHA256_BY_MODEL.tsv').read_text()
    assert 'renamed weights 2.pth' in checks
    assert 'renamed weights 2.pth' in (output/'README.md').read_text()


def test_unknown_weight_rejected(tmp_path):
    rp,reference=synthetic_release(tmp_path)
    (tmp_path/'irrelevant.pth').write_bytes(b'unexpected weights')
    verify=load_script('verify_checkpoint_files.py')
    with pytest.raises(ValueError,match='Unknown checkpoint'):
        verify.match_folder(tmp_path,reference)


def test_missing_checkpoint_rejected(tmp_path):
    rp,reference=synthetic_release(tmp_path)
    (tmp_path/'renamed weights 3.pth').unlink()
    verify=load_script('verify_checkpoint_files.py')
    with pytest.raises(ValueError,match='Missing checkpoints'):
        verify.match_folder(tmp_path,reference)


def test_release_filenames_are_recorded_without_changing_hashes():
    r = json.loads((REPO / 'reference/checkpoints.json').read_text(encoding='utf-8'))
    assert {k: v['filename'] for k, v in r['models'].items()} == {
        'baseline': '0.baseline.pth',
        'clean_control': '1.clean_control.pth',
        'plain_baqr': '2.baqr_only.pth',
        'entropy_baqr': '3.baqr_entropy_only.pth',
        'bce_top50': 'Top-50 negative BCE.pth',
    }

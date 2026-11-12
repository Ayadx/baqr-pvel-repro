"""CLI for reconstructing evaluation metrics from private frozen predictions."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .evaluation import (CLASSES, MODELS, THRESHOLDS, cluster_recall_bootstrap,
                         compact_model, load_manifest, paired_events,
                         read_predictions, sha256, summarize_model,
                         validate_manifest, validate_rows, write_csv)


def verify_hashes(expected_file, data_root):
    expected = json.loads(Path(expected_file).read_text(encoding='utf8'))
    for relative_path, checksum in expected.items():
        path = data_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f'Private input missing: {relative_path}')
        observed = sha256(path)
        if observed != checksum:
            raise ValueError(f'Hash mismatch for {relative_path}: expected {checksum}, got {observed}')


def compare_expected(recomputed, reference):
    discrepancies = []
    for name, expected in reference['models'].items():
        if name not in recomputed['models']:
            discrepancies.append(f'Missing required model: {name}')
            continue
        for k, value in expected.items():
            got = recomputed['models'][name].get(k)
            if got != value:
                discrepancies.append(f'{name}.{k} expected={value}, observed={got}')
    for pair, expected in reference['pairs'].items():
        if pair not in recomputed['pairs']:
            discrepancies.append(f'Missing paired comparison: {pair}')
            continue
        for subset, values in expected.items():
            for k, value in values.items():
                got = recomputed['pairs'][pair][subset].get(k)
                if got != value:
                    discrepancies.append(f'{pair}.{subset}.{k} expected={value}, observed={got}')
    return discrepancies


def run(data_root, out, *, models=MODELS, strict=True, bootstrap=True,
        expected_path=None, hash_path=None, replicates=10000):
    data_root = Path(data_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    defect_file = data_root / 'manifests/defects.json'
    good_file = data_root / 'manifests/good.json'
    if hash_path is not None:
        verify_hashes(hash_path, data_root)
    defects = load_manifest(defect_file)
    good = load_manifest(good_file)
    validate_manifest(defects, good, strict=strict)
    byimage = defaultdict(list)
    for ann in defects['annotations']:
        byimage[int(ann['image_id'])].append(ann)
    report = {
        'analysis': 'post-NMS stored predictions; not training or native COCO AP re-evaluation',
        'fixed_confidence': 0.25,
        'classwise_NMS_IoU_used_when_predictions_were_saved': 0.50,
        'matching_IoU': 0.50,
        'models': {}, 'pairs': {},
        'evaluation_scope': {
            'training_seed': 4,
            'sampling': '500 dataset-labelled good images and 2,000 annotated test images',
            'bootstrap_unit': 'evaluation image; models held fixed',
            'input_predictions': 'post-NMS, retained scores >= 0.25',
        },
    }
    model_rows = {}
    details = {}
    summary_rows = []
    curve_rows = []
    perclass_rows = []
    for name in models:
        folder = data_root / 'predictions' / name
        ds = read_predictions(folder / 'defects.jsonl.gz')
        ns = read_predictions(folder / 'normals.jsonl.gz')
        validate_rows(ds, defects['images'])
        validate_rows(ns, good['images'])
        model_rows[name] = (ds, ns)
        summary, img_stats, alarms = summarize_model(ds, ns, byimage)
        details[name] = (img_stats, alarms)
        report['models'][name] = compact_model(summary)
        summary_rows.append({'model': name, **compact_model(summary)})
        for cls in CLASSES:
            perclass_rows.append({'model': name, 'class': cls, **summary['per_class'][cls]})
        for threshold in THRESHOLDS:
            s, _, _ = summarize_model(ds, ns, byimage, threshold=threshold)
            curve_rows.append({
                'model': name, 'threshold': threshold,
                'tp': s['tp'], 'fp_defect': s['fp_defect'],
                'fn': s['fn'], 'recall': s['recall'],
                'good_alarms': s['good_alarms'], 'good_alarm_rate': s['good_alarm_rate'],
            })
    for ref, comp in (('clean_control', 'plain_baqr'), ('plain_baqr', 'bce_top50')):
        if ref not in details or comp not in details:
            continue
        a_stats, a_good = details[ref]
        b_stats, b_good = details[comp]
        d_rows = model_rows[ref][0]
        event = paired_events(a_stats, b_stats, a_good, b_good, byimage, d_rows,
                              bootstrap=bootstrap, replicates=replicates)
        report['pairs'][f'{ref}_vs_{comp}'] = event
    write_csv(out / 'model_summary.csv', summary_rows,
              ('model', 'good_alarms', 'good_boxes', 'tp', 'fp_defect', 'fn',
               'recall', 'precision_defect_cohort'))
    write_csv(out / 'per_class.csv', perclass_rows,
              ('model', 'class', 'gt', 'tp', 'fn', 'fp_defect', 'recall'))
    write_csv(out / 'threshold_points.csv', curve_rows,
              ('model', 'threshold', 'tp', 'fp_defect', 'fn', 'recall',
               'good_alarms', 'good_alarm_rate'))
    if expected_path:
        reference = json.loads(Path(expected_path).read_text(encoding='utf8'))
        discrepancies = compare_expected(report, reference)
    else:
        discrepancies = []
    report['expected_comparison'] = {
        'reference_provided': expected_path is not None,
        'passed': len(discrepancies) == 0 if expected_path else None,
        'discrepancies': discrepancies,
    }
    (out / 'recalculated.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    with open(out / 'report.md', 'w', encoding='utf8') as f:
        f.write('# BAQR evaluation summary\n\n')
        f.write('Metrics calculated from the supplied, postprocessed predictions at the common evaluation operating point.\n\n')
        f.write('| Model | Good alarms /500 | Good boxes | TP /3550 | FP on defective images | FN |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        for s in summary_rows:
            f.write(f"| {s['model']} | {s['good_alarms']} | {s['good_boxes']} | {s['tp']} | {s['fp_defect']} | {s['fn']} |\n")
        for name, ev in report['pairs'].items():
            f.write(f'\n## {name}\n\n')
            for kind in ('good_images', 'annotated_objects', 'cracks'):
                f.write(f'- {kind}: {ev[kind]}\n')
            f.write(f"- Two-sided exact McNemar test (conditional on image pairing): {ev['mcnemar_exact_two_sided_p']:.7g}\n")
        f.write('\n## Evaluation context\n\n')
        f.write('Training seed: 4. Bootstrap resampling uses images and holds the checkpoints fixed.\n')
        f.write('\nNative COCO AP values are recorded separately in `reference/native_coco_ap.json`.\n')
    print('Saved offline reconstruction in', out)
    if discrepancies:
        print('FAIL: frozen reference mismatch:')
        for diff in discrepancies:
            print(' -', diff)
        return 2
    if expected_path:
        print('PASS: all checked frozen fixed-operating-point and paired counts match.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Private validation-data directory (not for GitHub)')
    parser.add_argument('--out', type=Path, required=True, help='Output directory for regenerated tables')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--expected', type=Path, help='Public aggregated reference/expected_results.json')
    parser.add_argument('--verify-hashes', type=Path, help='Public SHA-256 hashes of restricted inputs')
    parser.add_argument('--allow-other-cohorts', action='store_true', help='Disable locked 500/2000 cohort size check (exploratory only)')
    parser.add_argument('--no-bootstrap', action='store_true', help='Fast deterministic tests without bootstrap CIs')
    parser.add_argument('--replicates', type=int, default=10000)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error('--replicates must be positive')
    raise SystemExit(run(args.data_root, args.out, models=args.models,
                         strict=not args.allow_other_cohorts,
                         bootstrap=not args.no_bootstrap,
                         expected_path=args.expected,
                         hash_path=args.verify_hashes, replicates=args.replicates))

if __name__ == '__main__':
    main()

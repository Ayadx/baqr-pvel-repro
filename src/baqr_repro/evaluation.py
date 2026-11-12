"""Code-based reconstruction of archived PV-EL fixed-operating-point metrics.

This is a secondary evaluation of *stored post-NMS predictions*, not a native
COCO AP recomputation. Inputs contain PVEL-AD-derived private information and
are intentionally excluded from the public repository.

Matching policy follows the frozen ``fixed_operating_point.py`` implementation:
confidence-ordered, class-aware greedy matching with an IoU>=0.5 threshold.
The first maximum in annotation-list order resolves exact equal-IoU ties.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import binomtest

CLASSES = (
    'crack', 'finger', 'black_core', 'thick_line', 'star_crack', 'corner',
    'fragment', 'scratch', 'horizontal_dislocation', 'vertical_dislocation',
    'printing_error', 'short_circuit',
)
MODELS = ('baseline', 'clean_control', 'plain_baqr', 'entropy_baqr', 'bce_top50')
THRESHOLDS = (0.25, 0.35, 0.50, 0.65, 0.75)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest(path):
    with open(path, encoding='utf8') as handle:
        return json.load(handle)


def read_predictions(path):
    with gzip.open(path, 'rt', encoding='utf8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_manifest(defects, good, *, strict=True):
    if not isinstance(defects.get('images'), list) or not isinstance(defects.get('annotations'), list):
        raise ValueError('Defect manifest must be COCO JSON with images and annotations.')
    if not isinstance(good.get('images'), list) or not isinstance(good.get('annotations'), list):
        raise ValueError('Good manifest must be COCO JSON with images and annotations.')
    if good['annotations']:
        raise ValueError('Good-labelled evaluation manifest must have zero GT annotations.')
    for man in (defects, good):
        ids = [int(x['id']) for x in man['images']]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate image id in manifest.')
        names = [x['file_name'] for x in man['images']]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate image name in manifest.')
    d_ids = {int(x['id']) for x in defects['images']}
    gt_ids = [int(x['id']) for x in defects['annotations']]
    if len(gt_ids) != len(set(gt_ids)):
        raise ValueError('Duplicate annotation ID.')
    for x in defects['annotations']:
        if int(x['image_id']) not in d_ids or int(x['category_id']) not in range(len(CLASSES)):
            raise ValueError('Invalid annotation image or category id.')
        if len(x['bbox']) != 4 or x['bbox'][2] < 0 or x['bbox'][3] < 0:
            raise ValueError('Invalid annotation bbox.')
    if strict and (len(defects['images']), len(defects['annotations']), len(good['images'])) != (2000, 3550, 500):
        raise ValueError('Expected frozen 2,000-image defect cohort, 3,550 GT and 500 good images.')


def validate_rows(rows, images, *, threshold=0.25):
    if len(rows) != len(images):
        raise ValueError(f'Prediction length {len(rows)} != manifest length {len(images)}')
    for row, image in zip(rows, images):
        if int(row['image_id']) != int(image['id']) or row['file_name'] != image['file_name']:
            raise ValueError('Prediction order, image ID or filename differs from manifest.')
        if len(row['predictions']) > 300:
            raise ValueError('Maximum 300 predictions per image exceeded.')
        for pred in row['predictions']:
            box = pred['bbox_xyxy']
            if int(pred['category_id']) not in range(len(CLASSES)):
                raise ValueError('Invalid prediction class.')
            if not threshold <= float(pred['score']) <= 1.0 or len(box) != 4:
                raise ValueError('Prediction score or box inconsistent with frozen operating point.')
            if not np.isfinite(np.asarray(box, dtype=float)).all() or box[2] < box[0] or box[3] < box[1]:
                raise ValueError('Invalid or nonfinite prediction bbox.')
            if not np.isfinite(float(pred['score'])):
                raise ValueError('Nonfinite prediction score.')


def iou_xyxy_coco(box, annotation):
    x, y, w, h = map(float, annotation['bbox'])
    x2, y2 = x + w, y + h
    left = max(0.0, min(float(box[2]), x2) - max(float(box[0]), x))
    top = max(0.0, min(float(box[3]), y2) - max(float(box[1]), y))
    overlap = left * top
    a = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
    return overlap / max(a + w * h - overlap, 1e-12)


def match_image(predictions, annotations, *, score_threshold=0.25, matching_iou=0.5):
    """Return counts, matched ground-truth IDs and per-class TP/FP/FN.

    Uses stored detector prediction order, as in the historical fixed evaluator;
    simply filters scores >= requested threshold. It does not rerun NMS.
    """
    preds = [p for p in predictions if float(p['score']) >= score_threshold]
    matched = set()
    by_class = {k: Counter() for k in range(len(CLASSES))}
    tp = fp = 0
    for p in preds:
        cat = int(p['category_id'])
        best = -1
        best_iou = matching_iou - 1e-12
        for i, gt in enumerate(annotations):
            if i in matched or int(gt['category_id']) != cat:
                continue
            overlap = iou_xyxy_coco(p['bbox_xyxy'], gt)
            if overlap > best_iou:  # original first-maximum IoU tie policy
                best_iou, best = overlap, i
        if best >= 0:
            matched.add(best)
            tp += 1
            by_class[cat]['tp'] += 1
        else:
            fp += 1
            by_class[cat]['fp'] += 1
    fn = len(annotations) - len(matched)
    for i, gt in enumerate(annotations):
        if i not in matched:
            by_class[int(gt['category_id'])]['fn'] += 1
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'matched_ids': {int(annotations[i]['id']) for i in matched},
        'by_class': by_class,
    }


def summarize_model(defect_rows, good_rows, annotations_by_image, *, threshold=0.25):
    per_image = []
    totals = Counter()
    cls = {i: Counter() for i in range(len(CLASSES))}
    for row in defect_rows:
        r = match_image(row['predictions'], annotations_by_image[int(row['image_id'])],
                        score_threshold=threshold)
        per_image.append(r)
        for key in ('tp', 'fp', 'fn'):
            totals[key] += r[key]
        for c, vals in r['by_class'].items():
            cls[int(c)].update(vals)
    g_count = [sum(float(p['score']) >= threshold for p in row['predictions']) for row in good_rows]
    summary = {
        'tp': totals['tp'], 'fp_defect': totals['fp'], 'fn': totals['fn'],
        'recall': totals['tp'] / (totals['tp'] + totals['fn']) if totals['tp'] + totals['fn'] else 0.,
        'precision_defect_cohort': totals['tp'] / (totals['tp'] + totals['fp']) if totals['tp'] + totals['fp'] else 0.,
        'good_alarms': int(sum(n > 0 for n in g_count)),
        'good_boxes': int(sum(g_count)),
        'good_alarm_rate': float(np.mean(np.array(g_count) > 0)),
        'per_class': {CLASSES[c]: {
            'gt': int(v['tp'] + v['fn']),
            'tp': int(v['tp']), 'fn': int(v['fn']), 'fp_defect': int(v['fp']),
            'recall': v['tp'] / (v['tp'] + v['fn']) if v['tp'] + v['fn'] else None,
        } for c, v in cls.items()},
    }
    return summary, per_image, np.asarray(g_count, dtype=int)


def percentile_bootstrap(values, *, seed=20261011, replicates=10000):
    diff = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates)
    for i in range(replicates):
        ind = rng.integers(0, len(diff), len(diff))
        bootstrap_means[i] = diff[ind].mean()
    return [float(x) for x in np.quantile(bootstrap_means, [0.025, 0.975])]


def cluster_recall_bootstrap(a, b, *, seed=20261011, replicates=10000):
    d_tp = np.asarray([x['tp'] - y['tp'] for x, y in zip(b, a)], dtype=float)
    den = np.asarray([x['tp'] + x['fn'] for x in a], dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(replicates):
        sel = rng.integers(0, len(den), len(den))
        total = den[sel].sum()
        if total:
            diffs.append(float(d_tp[sel].sum() / total))
    return [float(x) for x in np.quantile(diffs, [0.025, 0.975])]


def paired_events(a_stats, b_stats, a_good, b_good, annotations_by_image, defect_rows,
                  *, bootstrap=True, seed=20261011, replicates=10000):
    if len(a_stats) != len(b_stats) or len(a_good) != len(b_good):
        raise ValueError('Models must have the same image cohorts.')
    good_a = a_good > 0
    good_b = b_good > 0
    good = {
        'both': int(np.sum(good_a & good_b)),
        'reference_only': int(np.sum(good_a & ~good_b)),
        'comparator_only': int(np.sum(~good_a & good_b)),
        'neither': int(np.sum(~good_a & ~good_b)),
    }
    gt_events = Counter()
    crack_events = Counter()
    for i, row in enumerate(defect_rows):
        aa = a_stats[i]['matched_ids']
        bb = b_stats[i]['matched_ids']
        gt = annotations_by_image[int(row['image_id'])]
        for tag, s in (
            ('both', aa & bb), ('reference_only', aa - bb),
            ('comparator_only', bb - aa),
            ('neither', {int(x['id']) for x in gt} - (aa | bb)),
        ):
            gt_events[tag] += len(s)
            cracks = {int(x['id']) for x in gt if int(x['category_id']) == 0}
            crack_events[tag] += len(s & cracks)
    discord = good['reference_only'] + good['comparator_only']
    out = {
        'good_images': dict(good),
        'annotated_objects': dict(gt_events),
        'cracks': dict(crack_events),
        'good_alarm_difference_pp': float((good_b.mean() - good_a.mean()) * 100),
        'recall_difference_pp': float(
            (sum(x['tp'] for x in b_stats) - sum(x['tp'] for x in a_stats)) /
            sum(x['tp'] + x['fn'] for x in a_stats) * 100),
        'mcnemar_exact_two_sided_p': float(binomtest(min(good['reference_only'], good['comparator_only']), discord).pvalue) if discord else 1.0,
    }
    if bootstrap:
        out['good_alarm_diff_CI95_pp'] = [100 * x for x in percentile_bootstrap(good_b.astype(int) - good_a.astype(int), seed=seed, replicates=replicates)]
        out['recall_diff_CI95_pp'] = [100 * x for x in cluster_recall_bootstrap(a_stats, b_stats, seed=seed, replicates=replicates)]
    return out


def compact_model(summary):
    return {key: summary[key] for key in
            ('good_alarms', 'good_boxes', 'tp', 'fp_defect', 'fn', 'recall', 'precision_defect_cohort')}


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

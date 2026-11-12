import pytest
from baqr_repro.evaluation import iou_xyxy_coco, match_image, paired_events


def pred(cat, s, box):
    return dict(category_id=cat, score=s, bbox_xyxy=box)


def ann(i, cat, box):
    return dict(id=i, category_id=cat, bbox=box)


def test_iou_matching_class_and_threshold():
    gt = [ann(1, 0, [0, 0, 10, 10]), ann(2, 1, [10, 10, 10, 10])]
    ps = [pred(0, 0.8, [0, 0, 10, 10]),
          pred(1, 0.7, [10, 10, 20, 20]),
          pred(0, 0.6, [0, 0, 10, 10])]
    r = match_image(ps, gt)
    assert (r['tp'], r['fp'], r['fn']) == (2, 1, 0)
    assert r['matched_ids'] == {1, 2}
    assert iou_xyxy_coco([0, 0, 10, 10], gt[0]) == 1.0


def test_threshold_does_not_recover_discarded_predictions():
    gt = [ann(2, 0, [0, 0, 10, 10])]
    ps = [pred(0, 0.27, [0, 0, 10, 10])]
    assert match_image(ps, gt, score_threshold=.25)['tp'] == 1
    assert match_image(ps, gt, score_threshold=.35)['tp'] == 0


def test_first_maximum_tie_stays_with_first_annotation():
    gt = [ann(15, 0, [0, 0, 10, 10]), ann(16, 0, [0, 0, 10, 10])]
    ps = [pred(0, .9, [0, 0, 10, 10])]
    assert match_image(ps, gt)['matched_ids'] == {15}


def test_zero_support_is_not_equivalent_to_perfect_class_recall():
    assert match_image([], [], score_threshold=.25)['tp'] == 0


def test_pair_decomposition():
    gt = {0: [ann(2, 0, [0, 0, 10, 10])]}
    # evaluator sees one good image with a control-only alarm and one defect annotation
    a = [match_image([pred(0, .9, [0, 0, 10, 10])], gt[0])]
    b = [match_image([], gt[0])]
    import numpy as np
    events = paired_events(a, b, np.array([1]), np.array([0]), gt,
                           [{'image_id': 0}], bootstrap=False)
    assert events['good_images']['reference_only'] == 1
    assert events['annotated_objects']['reference_only'] == 1
    assert events['cracks']['reference_only'] == 1

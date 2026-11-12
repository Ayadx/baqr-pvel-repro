"""Top-K winner-class negative BCE comparator for BAQR.

Comparison is with the original BAQR's same selected final matching queries:
  K=50; empty-labelled training images only; max sigmoid foreground confidence
  determines the winning foreground class and the selected query indices.
The hinge is replaced by BCEWithLogits(z_winning_class, target=0).
No native detector loss is replaced and no denoising query is included.
"""
from functools import wraps
import torch
import torch.nn.functional as F


def compute_negative_topk_bce(all_matching_cls_scores, gt_labels_list,
                              k=50, weight=0.10):
    """Return scalar auxiliary or None when the batch contains no empty-GT images.

    all_matching_cls_scores: [decoder_layers, batch, matching_queries, classes]
    gt_labels_list: a list of batch-length label tensors

    BCE = weight * mean_{empty image, selected query} softplus(max_c(z_iqc)),
    with top-K selected exactly as in original BAQR on max_c sigmoid(z_iqc).
    """
    if all_matching_cls_scores is None:
        raise RuntimeError('Missing matching-query logits; comparator cannot run.')
    if all_matching_cls_scores.dim() != 4:
        raise RuntimeError('Expected [layers, batch, queries, classes] logits; got %s' %
                           (tuple(all_matching_cls_scores.shape),))
    last_logits = all_matching_cls_scores[-1]
    B, Q, C = last_logits.shape
    if B != len(gt_labels_list) or Q <= 0 or C <= 1:
        raise RuntimeError('Unexpected BCE input shape or GT list length')
    if not (0 < int(k) <= Q):
        raise RuntimeError('BCE top-K must be positive and <= matching queries.')
    empty_idx = [i for i, lbl in enumerate(gt_labels_list) if len(lbl) == 0]
    if not empty_idx:
        return None
    empty_logits = last_logits[empty_idx]
    # The original BAQR implementation sorts by this exact confidence.
    fg_probs = empty_logits.sigmoid().max(dim=-1).values
    idx = fg_probs.topk(k=min(int(k), Q), dim=1).indices
    # max over classes is the same winning-class logit selected by max(sigmoid).
    max_logits = empty_logits.max(dim=-1).values
    selected_logits = max_logits.gather(1, idx)
    # Stable negative BCE on the winning logit, not BCE averaged over all classes.
    # This isolates the hinge-versus-BCE penalty on otherwise identical query targets.
    loss = F.binary_cross_entropy_with_logits(
        selected_logits, torch.zeros_like(selected_logits), reduction='mean'
    )
    return float(weight) * loss


def install_on_codino_head():
    """Class-level wrapper is safe under model replication; no repo files mutated."""
    from projects.models.co_dino_head import CoDINOHead
    if getattr(CoDINOHead.loss, '_bce_topk_wrapped', False):
        return CoDINOHead
    original = CoDINOHead.loss

    @wraps(original)
    def loss_with_bce(self, all_cls_scores, all_bbox_preds,
                      enc_topk_scores, enc_topk_anchors, enc_outputs,
                      gt_bboxes_list, gt_labels_list, img_metas,
                      dn_meta=None, gt_bboxes_ignore=None):
        loss_dict = original(
            self, all_cls_scores, all_bbox_preds,
            enc_topk_scores, enc_topk_anchors, enc_outputs,
            gt_bboxes_list, gt_labels_list, img_metas,
            dn_meta=dn_meta, gt_bboxes_ignore=gt_bboxes_ignore
        )
        if getattr(self, 'bce_neg_topk_enabled', False):
            if getattr(self, 'baqr_enabled', False):
                raise RuntimeError('BAQR and negative BCE must not both be enabled.')
            # Use precisely the same matching-query partition as the archived BAQR.
            matching_cls, _, _, _ = self.extract_dn_outputs(
                all_cls_scores, all_bbox_preds, dn_meta
            )
            if matching_cls.shape[2] != int(getattr(self, 'num_query', -1)):
                raise RuntimeError('Matching-query count differs from configured query count')
            aux = compute_negative_topk_bce(
                matching_cls, gt_labels_list,
                k=int(getattr(self, 'bce_neg_topk', 50)),
                weight=float(getattr(self, 'bce_neg_weight', 0.10))
            )
            if aux is not None:
                if 'loss_baqr' in loss_dict or 'loss_bce_neg' in loss_dict:
                    raise RuntimeError('Duplicate or simultaneous auxiliary loss.')
                loss_dict['loss_bce_neg'] = aux
        return loss_dict

    loss_with_bce._bce_topk_wrapped = True
    CoDINOHead.loss = loss_with_bce
    return CoDINOHead


def run_synthetic_tests():
    """CPU-only algebra and gradient checks; no model weights or data needed."""
    # 1 image with GT, 1 empty; only empty image must influence gradients.
    logits = torch.tensor([[[[100., 100.], [-100., -100.], [2., -1.]],
                           [[-3., -4.], [0., -10.], [2., -1.]]]],
                          dtype=torch.float64, requires_grad=True)
    labels = [torch.tensor([0]), torch.empty(0, dtype=torch.long)]
    loss = compute_negative_topk_bce(logits, labels, k=2, weight=0.1)
    # Largest max logits in empty example are 2 and 0.
    expected = 0.1 * (F.softplus(torch.tensor(2.,dtype=torch.float64)) +
                       F.softplus(torch.tensor(0.,dtype=torch.float64))) / 2
    assert torch.allclose(loss, expected, atol=1e-12), (loss, expected)
    loss.backward()
    assert logits.grad[0, 0].abs().sum().item() == 0., 'Positive-image gradient leaked.'
    assert logits.grad[0, 1, 0].abs().sum().item() == 0., 'Unselected-query gradient leaked.'
    assert logits.grad[0, 1, 1, 0] > 0, 'Selected winner class must receive gradient.'
    assert logits.grad[0, 1, 2, 0] > 0, 'Selected winner class must receive gradient.'
    assert logits.grad[0, 1, 1, 1] == 0, 'Non-winning class unexpectedly trained.'
    assert compute_negative_topk_bce(logits.detach(), [torch.tensor([1]), torch.tensor([0])],
                                     k=2, weight=0.1) is None
    # Check original BAQR selection in a clean-margin-free case (identity of indices).
    scores = logits.detach()[-1, 1]
    ids_baqr = scores.sigmoid().max(-1).values.topk(2).indices
    ids_bce = scores.sigmoid().max(-1).values.topk(2).indices
    assert torch.equal(ids_baqr, ids_bce)
    return {'loss': float(loss.detach()), 'expected': float(expected), 'status':'PASS'}

"""BAQR auxiliary for final-decoder matching-query logits on empty-labelled images.

Integrate with a compatible Co-DINO head after excluding denoising queries.
"""

import torch
import torch.nn.functional as F

def compute_baqr_loss(all_cls_scores,
                      gt_labels_list,
                      baqr_lambda=0.10,
                      baqr_topk=50,
                      baqr_margin=0.05,
                      baqr_entropy_enabled=False,
                      baqr_entropy_gamma=2.0,
                      baqr_entropy_temperature=1.0):
    """
    BAQR loss for empty-GT images.

    all_cls_scores: Tensor [num_dec_layers, bs, num_queries, num_classes]
    gt_labels_list: list[Tensor], one per image

    Standard BAQR:
        lambda * mean( relu(max(sigmoid(logits)) - margin) )

    Entropy-gated BAQR:
        lambda * mean( ((1 - H)^gamma) * relu(max(sigmoid(logits)) - margin) )
    where H is the normalized entropy of softmax(logits / T).
    """
    if all_cls_scores is None:
        return None

    cls_scores = all_cls_scores[-1]  # [bs, num_queries, num_classes]
    if cls_scores.numel() == 0:
        return None

    device = cls_scores.device

    empty_mask = torch.tensor(
        [len(labels) == 0 for labels in gt_labels_list],
        dtype=torch.bool,
        device=device
    )

    if empty_mask.sum() == 0:
        return None

    empty_scores = cls_scores[empty_mask]  # [B_empty, Nq, C]

    # foreground confidence
    fg_probs = empty_scores.sigmoid().max(dim=-1).values  # [B_empty, Nq]

    # top-k hardest queries
    if baqr_topk is not None and baqr_topk > 0 and baqr_topk < fg_probs.size(1):
        topk_vals, topk_idx = torch.topk(fg_probs, k=baqr_topk, dim=1)
        fg_probs = topk_vals
        if baqr_entropy_enabled:
            empty_scores = torch.gather(
                empty_scores,
                dim=1,
                index=topk_idx.unsqueeze(-1).expand(-1, -1, empty_scores.size(-1))
            )

    hinge_term = F.relu(fg_probs - baqr_margin)

    if baqr_entropy_enabled:
        temp = max(float(baqr_entropy_temperature), 1e-6)
        soft = F.softmax(empty_scores / temp, dim=-1).clamp(min=1e-12, max=1.0)
        entropy = -(soft * soft.log()).sum(dim=-1)
        num_classes = max(int(empty_scores.size(-1)), 2)
        entropy = entropy / float(torch.log(torch.tensor(float(num_classes), device=soft.device)))
        gate = (1.0 - entropy).clamp(min=0.0, max=1.0).pow(float(baqr_entropy_gamma))
        loss = (gate * hinge_term).mean()
    else:
        loss = hinge_term.mean()

    return float(baqr_lambda) * loss


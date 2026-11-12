"""Optional CPU-only tests: pytest.importorskip('torch') avoids training hardware."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]

def load(name, file):
    spec = spec_from_file_location(name, ROOT / file)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

baqr = load('baqr_archived','losses/baqr_loss.py')
bce = load('bce_original','losses/bce_neg_auxiliary.py')


def test_baqr_only_empty_image_gets_direct_gradient():
    logits = torch.tensor([[[[-10., -9.], [7., 4.]],
                            [[.0, -4.], [2., -10.]]]],
                          dtype=torch.float64, requires_grad=True)
    labels = [torch.tensor([1]), torch.empty(0, dtype=torch.long)]
    result = baqr.compute_baqr_loss(logits, labels, baqr_topk=1,
                                    baqr_margin=.05, baqr_lambda=.1)
    expected = .1 * torch.relu(torch.sigmoid(torch.tensor(2.,dtype=torch.float64))-.05)
    assert torch.allclose(result, expected, atol=1e-12)
    result.backward()
    assert logits.grad[0, 0].abs().sum().item() == 0.
    assert logits.grad[0, 1, 1, 0] > 0
    assert logits.grad[0, 1, 0].abs().sum().item() == 0.


def test_baqr_none_without_empty_gt():
    logits = torch.zeros(1, 2, 3, 2)
    assert baqr.compute_baqr_loss(logits,[torch.tensor([1]),torch.tensor([0])]) is None


def test_bce_archived_synthetic_validation():
    bce.run_synthetic_tests()


def test_baqr_entropy_zero_for_uniform_high_logits():
    logits = torch.tensor([[[[10.,10.,10.],[9.,9.,9.]]]],dtype=torch.float32)
    result = baqr.compute_baqr_loss(logits, [torch.empty(0,dtype=torch.long)],
                                    baqr_entropy_enabled=True,
                                    baqr_topk=1)
    assert result.item() < 1e-6

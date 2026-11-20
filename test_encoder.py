"""Unit tests for the Attn-LOB encoder (encoder.py)."""
import torch

from encoder import AttnLOB, count_parameters


def test_forward_shapes():
    m = AttnLOB()
    x = torch.randn(8, 50, 40)
    assert m(x).shape == (8, 3)
    assert m.forward_features(x).shape == (8, 64)


def test_forward_features_deterministic_eval():
    m = AttnLOB()
    m.eval()
    x = torch.randn(4, 50, 40)
    with torch.no_grad():
        assert torch.allclose(m.forward_features(x), m.forward_features(x))


def test_gradient_flows():
    m = AttnLOB()
    x = torch.randn(4, 50, 40, requires_grad=True)
    loss = m(x).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_parameter_count_reported():
    m = AttnLOB()
    n = count_parameters(m)
    print(f"Attn-LOB parameters: {n}")
    assert 10_000 < n < 500_000

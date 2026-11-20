"""Unit tests for the PPO implementation (ppo.py)."""
import numpy as np
import torch

from encoder import AttnLOB
from ppo import (ActorCritic, compute_gae, episode_features, ppo_update)


def _policy():
    torch.manual_seed(0)
    enc = AttnLOB()
    enc.eval()
    return ActorCritic(enc)


def test_act_shapes_and_range():
    pol = _policy()
    x = torch.randn(4, 68)
    a, lp, v = pol.act_from_features(x)
    assert a.shape == (4, 2) and lp.shape == (4,) and v.shape == (4,)
    assert bool(((a >= 0) & (a <= 1)).all())
    assert torch.isfinite(lp).all() and torch.isfinite(v).all()


def test_act_deterministic_is_mean():
    pol = _policy()
    x = torch.randn(2, 68)
    a1 = pol.act_deterministic_from_features(x)
    a2 = pol.act_deterministic_from_features(x)
    assert torch.allclose(a1, a2)
    assert bool(((a1 >= 0) & (a1 <= 1)).all())


def test_gae_hand_computed():
    # rewards [1,1,1], values 0, gamma=1, lam=1 -> adv [3,2,1]
    adv = compute_gae(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]),
                      gamma=1.0, lam=1.0)
    assert np.allclose(adv, [3.0, 2.0, 1.0])
    # gamma=0 -> adv = rewards - values
    adv0 = compute_gae(np.array([1.0, 2.0]), np.array([0.5, 0.5]),
                       gamma=0.0, lam=1.0)
    assert np.allclose(adv0, [0.5, 1.5])


def test_ppo_update_runs_and_returns_stats():
    pol = _policy()
    opt = torch.optim.Adam([p for p in pol.parameters() if p.requires_grad],
                           lr=3e-4)
    rng = np.random.default_rng(0)
    n = 512
    buf = {
        "features": rng.standard_normal((n, 68)).astype(np.float32),
        "actions": rng.random((n, 2)).astype(np.float32),
        "logprobs": rng.standard_normal(n).astype(np.float32) * 0.1,
        "advantages": rng.standard_normal(n).astype(np.float32),
        "returns": rng.standard_normal(n).astype(np.float32),
    }
    stats = ppo_update(pol, opt, buf, epochs=1, minibatch=256)
    assert set(stats) == {"clip_frac", "entropy", "pg_loss", "vf_loss"}
    assert all(np.isfinite(v) for v in stats.values())
    assert 0.0 <= stats["clip_frac"] <= 1.0


def test_episode_features_shape():
    torch.manual_seed(0)
    enc = AttnLOB()
    enc.eval()
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((40, 500)).astype(np.float32)
    mid = np.cumsum(rng.standard_normal(500) * 0.001)
    f = episode_features(enc, feats, mid, start=100, episode_len=200,
                         window=50, device="cpu")
    assert f.shape == (200, 64)
    assert torch.isfinite(f).all()

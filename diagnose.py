"""Diagnose the trained PPO agent vs fixed/AS baselines.

Runs the deterministic PPO policy (results/ppo_agent.pt) plus the
fixed-spread and AS baselines on the SAME 20 held-out test episodes
(first 20 of evaluate.py's seed-7 starts) and logs, per step:
  action (a1, a2), quoted half-spread, fills, inventory,
  reward decomposition (dp / tp / ip from env.step info).
Also reports the pretrained encoder's predicted-class distribution
on the same test windows (is the trunk's classifier still collapsed?).

Usage: nice -n 10 ~/workspace/.tlob-venv/bin/python diagnose.py
"""
from __future__ import annotations

import os

import numpy as np
import torch

from encoder import AttnLOB
from env import (EnvParams, MMEnv, as_policy_factory, evaluate_policy,
                 fixed_policy_factory, sample_starts, valid_starts)
from lob_data import load_lob
from ppo import ActorCritic, episode_features

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))


def instrumented_run(policy_fn, feats, mid, starts, params, ppo=None,
                     device="cpu"):
    """Run policy_fn(env)->action on starts; return per-step aggregates."""
    a1s, a2s, halves = [], [], []
    dp_sum, tp_sum, ip_sum = 0.0, 0.0, 0.0
    fills_ep, invs, pnls = [], [], []
    for s in starts:
        env = MMEnv(feats, mid, int(s), params)
        feats_t = None
        if ppo is not None:
            feats_t = episode_features(ppo.encoder, feats, mid, int(s),
                                       params.episode_len, params.window,
                                       device)
        obs = env.reset()
        for t in range(params.episode_len):
            if ppo is not None:
                x = torch.from_numpy(
                    np.concatenate([feats_t[t].numpy(),
                                    obs["agent_state"]])[None]).to(device)
                with torch.no_grad():
                    a = ppo.act_deterministic_from_features(x)
                act = tuple(a.cpu().numpy()[0])
            else:
                act = policy_fn(env)
            a1s.append(act[0])
            a2s.append(act[1])
            halves.append(act[1] * params.max_spread / 2.0)
            obs, _, done, info = env.step(act)
            dp_sum += info["dp"]
            tp_sum += info["tp"]
            ip_sum += info["ip"]
            if done:
                break
        fills_ep.append(env.fills)
        invs.append(float(np.mean(np.abs(env.inv_curve))))
        pnls.append(env.pnl)
    n = len(starts)
    return {
        "mean_a1": float(np.mean(a1s)),
        "mean_a2": float(np.mean(a2s)),
        "a2_p10": float(np.percentile(a2s, 10)),
        "a2_p50": float(np.percentile(a2s, 50)),
        "a2_p90": float(np.percentile(a2s, 90)),
        "frac_a2_ge_0.9": float(np.mean(np.array(a2s) >= 0.9)),
        "frac_a2_le_0.2": float(np.mean(np.array(a2s) <= 0.2)),
        "mean_half_spread": float(np.mean(halves)),
        "mean_fills_ep": float(np.mean(fills_ep)),
        "mean_abs_inv": float(np.mean(invs)),
        "mean_pnl": float(np.mean(pnls)),
        "dp_per_ep": dp_sum / n,
        "tp_per_ep": tp_sum / n,
        "ip_per_ep": ip_sum / n,
        "reward_per_ep": (dp_sum + tp_sum - ip_sum) / n,
    }


def main():
    device = torch.device("cpu")
    params = EnvParams()
    print("loading FI-2010 test ...", flush=True)
    d = load_lob(horizon=10)
    te_f, te_mid = d["test_feats"], d["test_mid"]
    starts = sample_starts(50, len(te_mid), params.episode_len,
                           params.window, seed=7,
                           valid=valid_starts(te_mid, params.episode_len,
                                              params.window))[:20]
    print(f"diagnosing on {len(starts)} test episodes", flush=True)

    # --- encoder classifier check (collapsed?) ---
    enc = AttnLOB()
    enc.load_state_dict(torch.load(os.path.join(HERE, "results/attn_lob_pretrained.pt"),
                                   map_location="cpu"))
    enc.eval()
    from lob_data import LOBWindows
    tw = LOBWindows(te_f, d["test_labels"], seq_size=50, stride=10,
                    max_windows=2000)
    xs = torch.stack([torch.from_numpy(tw[i][0]) for i in range(len(tw))])
    with torch.no_grad():
        logits = enc(xs)
        preds = logits.argmax(dim=1).numpy()
        feats64 = enc.forward_features(xs).numpy()
    print("encoder predicted-class shares (up/flat/down):",
          [round(float(np.mean(preds == c)), 3) for c in (0, 1, 2)])
    print("encoder feature std per dim: mean %.4f min %.4f max %.4f" % (
        feats64.std(axis=0).mean(), feats64.std(axis=0).min(),
        feats64.std(axis=0).max()))
    print("encoder feature |mean| per dim: %.4f" % np.abs(feats64.mean(axis=0)).mean())

    # --- baselines ---
    for name, pol in [("fixed(half 0.001)", fixed_policy_factory(0.001)),
                      ("as(g=0.5,k=1000)", as_policy_factory(0.5, 1000.0))]:
        r = instrumented_run(pol, te_f, te_mid, starts, params)
        print(f"{name}: {r}", flush=True)

    # --- trained PPO (deterministic) ---
    policy = ActorCritic(enc)
    policy.load_state_dict(torch.load(os.path.join(HERE, "results/ppo_agent.pt"),
                                      map_location="cpu"))
    policy.eval()
    r = instrumented_run(None, te_f, te_mid, starts, params, ppo=policy)
    print(f"ppo-deterministic: {r}", flush=True)

    # --- trained PPO (stochastic, one pass) ---
    import torch as _t
    a1s, a2s = [], []
    for s in starts[:5]:
        env = MMEnv(te_f, te_mid, int(s), params)
        feats_t = episode_features(policy.encoder, te_f, te_mid, int(s),
                                   params.episode_len, params.window, device)
        obs = env.reset()
        for t in range(params.episode_len):
            x = _t.from_numpy(np.concatenate([feats_t[t].numpy(),
                                              obs["agent_state"]])[None]).to(device)
            with _t.no_grad():
                a, _, _ = policy.act_from_features(x)
            a1s.append(float(a[0][0]))
            a2s.append(float(a[0][1]))
            obs, _, done, _ = env.step(tuple(a.cpu().numpy()[0]))
            if done:
                break
    print("ppo-stochastic first-5-eps: mean a1 %.3f mean a2 %.3f "
          "log_std %s" % (np.mean(a1s), np.mean(a2s),
                          np.round(policy.log_std.detach().numpy(), 3)))


if __name__ == "__main__":
    main()

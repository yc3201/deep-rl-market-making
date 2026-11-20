"""Evaluate PPO vs AS vs fixed-spread vs random on identical TEST episodes.

Baselines use the TRAIN-calibrated hyperparameters (see calibrate.py
output — fill them in here after calibration; defaults are the
pre-calibration guesses and must be updated). PPO is evaluated with
its deterministic mean action. All policies run on the same episode
starts, so the comparison is paired.

Metrics (paper Sec. IV-C4 + the AS simulator's): mean PnL, Sharpe,
ND-PnL (PnL / market spread), PnLMAP (PnL / mean |inventory|), PR
(PnL / total volume), mean max drawdown, mean |inventory|, mean
|final inventory|, win rate, mean fills.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from encoder import AttnLOB
from env import (EnvParams, MMEnv, as_policy_factory, evaluate_policy,
                 fixed_policy_factory, random_policy, sample_starts,
                 valid_starts)
from lob_data import load_lob
from ppo import ActorCritic, episode_features

torch.set_num_threads(1)

# TRAIN-calibrated baseline hyperparameters (calibrate.py, 20 train eps).
# Fixed: widest in range wins (monotone: wider -> better PnL, fewer fills,
# less adverse selection); AS: kappa=500/1000 tie (both clip to max width),
# gamma barely matters (the sigma^2 term is negligible in z-units).
FIXED_HALF = 0.001
AS_GAMMA = 0.5
AS_KAPPA = 1000.0


def evaluate_ppo(policy, feats, mid, starts, params, device):
    """Deterministic-mean-action PPO evaluation on fixed TEST episodes."""
    import numpy as np
    from env import _episode_metrics
    recs = []
    for s in starts:
        env = MMEnv(feats, mid, int(s), params)
        feats_t = episode_features(policy.encoder, feats, mid, int(s),
                                   params.episode_len, params.window, device)
        obs = env.reset()
        for t in range(params.episode_len):
            x = torch.from_numpy(
                np.concatenate([feats_t[t].numpy(),
                                obs["agent_state"]])[None]).to(device)
            with torch.no_grad():
                a = policy.act_deterministic_from_features(x)
            obs, _, done, _ = env.step(tuple(a.cpu().numpy()[0]))
            if done:
                break
        recs.append(_episode_metrics(env, params))
    pnls = np.array([r["pnl"] for r in recs])
    return {
        "n_episodes": len(recs),
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "sharpe": float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0.0,
        "mean_nd_pnl": float(np.mean([r["nd_pnl"] for r in recs])),
        "mean_pnlmap": float(np.mean([r["pnlmap"] for r in recs])),
        "mean_pr": float(np.mean([r["pr"] for r in recs])),
        "mean_max_drawdown": float(np.mean([r["max_drawdown"] for r in recs])),
        "mean_abs_inv": float(np.mean([r["mean_abs_inv"] for r in recs])),
        "mean_final_abs_inv": float(np.mean([r["final_abs_inv"] for r in recs])),
        "win_rate": float(np.mean([r["win"] for r in recs])),
        "mean_fills": float(np.mean([r["fills"] for r in recs])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--zeta", type=float, default=None)
    ap.add_argument("--fixed-half", type=float, default=FIXED_HALF)
    ap.add_argument("--min-half-spread", type=float, default=0.0005,
                    help="floor on the quoted half-spread; MUST match the "
                         "value the agent was trained with (the V4 "
                         "validation was invalidated because eval used "
                         "the 0.0005 default while training pinned 0.001)")
    ap.add_argument("--as-gamma", type=float, default=AS_GAMMA)
    ap.add_argument("--as-kappa", type=float, default=AS_KAPPA)
    ap.add_argument("--agent-ckpt", default="results/ppo_agent.pt")
    ap.add_argument("--encoder-ckpt", default="results/attn_lob_pretrained.pt")
    ap.add_argument("--out", default="results/evaluation.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    here = os.path.dirname(os.path.abspath(__file__))
    params = EnvParams()
    params.min_half_spread = args.min_half_spread
    if args.zeta is not None:
        params.zeta = args.zeta

    print("loading FI-2010 test ...", flush=True)
    d = load_lob(horizon=10)
    te_f, te_mid = d["test_feats"], d["test_mid"]
    starts = sample_starts(args.n_episodes, len(te_mid), params.episode_len,
                           params.window, seed=args.seed,
                           valid=valid_starts(te_mid, params.episode_len,
                                              params.window))
    print(f"evaluating on {len(starts)} test episodes", flush=True)

    results = {}
    for name, pol in [
        ("random", random_policy),
        ("fixed", fixed_policy_factory(args.fixed_half)),
        ("as", as_policy_factory(args.as_gamma, args.as_kappa)),
    ]:
        results[name] = evaluate_policy(pol, te_f, te_mid, starts, params)
        print(f"{name}: {results[name]}", flush=True)

    ckpt = os.path.join(here, args.agent_ckpt)
    if os.path.exists(ckpt):
        encoder = AttnLOB()
        encoder.load_state_dict(torch.load(os.path.join(here, args.encoder_ckpt),
                                           map_location="cpu"))
        encoder.eval()
        policy = ActorCritic(encoder)
        policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
        policy.eval()
        results["ppo"] = evaluate_ppo(policy, te_f, te_mid, starts, params,
                                      device)
        print(f"ppo: {results['ppo']}", flush=True)
    else:
        print(f"no agent checkpoint at {ckpt}; baselines only", flush=True)

    keys = ["mean_pnl", "sharpe", "mean_nd_pnl", "mean_pnlmap", "mean_pr",
            "mean_max_drawdown", "mean_abs_inv", "mean_final_abs_inv",
            "win_rate", "mean_fills"]
    print(f"\n{'method':<8}" + "".join(f"{k:>14}" for k in keys))
    for name, r in results.items():
        print(f"{name:<8}" + "".join(f"{r[k]:>14.3f}" for k in keys))
    with open(os.path.join(here, args.out), "w") as f:
        json.dump({"params": {"zeta": params.zeta,
                              "half_spread": params.half_spread,
                              "fixed_half": args.fixed_half,
                              "as_gamma": args.as_gamma,
                              "as_kappa": args.as_kappa,
                              "n_episodes": args.n_episodes,
                              "seed": args.seed},
                   "results": results}, f, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

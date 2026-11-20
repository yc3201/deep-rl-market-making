"""Evaluate PPO vs AS vs fixed-spread vs random on identical TEST
episodes of real LOBSTER data (AAPL, 2012-06-21, last 25% of the day).

Baselines use the TRAIN-calibrated hyperparameters (calibrate_real.py):
fixed half $0.12 (best Sharpe on train), AS gamma 0.5 / kappa 5.0
(best Sharpe on train). PPO is evaluated with its deterministic mean
action. All policies run on the same episode starts (paired).

Money is in dollars; 1 inventory unit = 100 shares.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from encoder import AttnLOB
from env import (as_policy_factory, fixed_policy_factory, random_policy)
from ppo import ActorCritic, episode_features
from real_env import (RealMMEnv, _episode_metrics, evaluate_real_policy,
                      real_env_params)

torch.set_num_threads(1)

# TRAIN-calibrated baseline hyperparameters (calibrate_real.py, 20 train eps).
FIXED_HALF = 0.12
AS_GAMMA = 0.5
AS_KAPPA = 5.0

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "AAPL_10.npz")


def evaluate_ppo(policy, data, starts, params, device):
    """Deterministic-mean-action PPO evaluation on fixed TEST episodes."""
    recs = []
    for s in starts:
        env = RealMMEnv(data["feats"], data["mid"], data["raw_book"],
                        data["exec_price"], data["exec_size"],
                        data["exec_side"], data["exec_count"], int(s), params)
        feats_t = episode_features(policy.encoder, data["feats"], data["mid"],
                                   int(s), params.episode_len, params.window,
                                   device)
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
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--zeta", type=float, default=None)
    ap.add_argument("--fixed-half", type=float, default=FIXED_HALF)
    ap.add_argument("--as-gamma", type=float, default=AS_GAMMA)
    ap.add_argument("--as-kappa", type=float, default=AS_KAPPA)
    ap.add_argument("--agent-ckpt", default="results/ppo_agent_real.pt")
    ap.add_argument("--encoder-ckpt",
                    default="results/attn_lob_real_pretrained.pt")
    ap.add_argument("--out", default="results/evaluation_real.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    params = real_env_params()
    if args.zeta is not None:
        params.zeta = args.zeta

    print("loading AAPL ...", flush=True)
    data = dict(np.load(DATA))
    n = len(data["mid"])
    n_train = int(data["train_mask"].sum())
    rng = np.random.default_rng(args.seed)
    starts = sorted(rng.integers(n_train, n - params.episode_len,
                                 size=args.n_episodes).tolist())
    print(f"evaluating on {len(starts)} test episodes "
          f"(steps {n_train}..{n})", flush=True)

    results = {}
    for name, pol in [
        ("random", random_policy),
        ("fixed", fixed_policy_factory(args.fixed_half)),
        ("as", as_policy_factory(args.as_gamma, args.as_kappa)),
    ]:
        results[name] = evaluate_real_policy(pol, data, starts, params)
        r = results[name]
        print(f"{name}: mean_pnl=${r['mean_pnl']:+.2f} "
              f"sharpe={r['sharpe']:+.2f} fills={r['mean_fills']:.1f} "
              f"win={r['win_rate']:.2f} |inv|={r['mean_abs_inv']:.2f}",
              flush=True)

    ckpt = os.path.join(HERE, args.agent_ckpt)
    if os.path.exists(ckpt):
        encoder = AttnLOB()
        encoder.load_state_dict(torch.load(
            os.path.join(HERE, args.encoder_ckpt), map_location="cpu"))
        encoder.eval()
        policy = ActorCritic(encoder)
        policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
        policy.eval()
        results["ppo"] = evaluate_ppo(policy, data, starts, params, device)
        r = results["ppo"]
        print(f"ppo: mean_pnl=${r['mean_pnl']:+.2f} "
              f"sharpe={r['sharpe']:+.2f} fills={r['mean_fills']:.1f} "
              f"win={r['win_rate']:.2f} |inv|={r['mean_abs_inv']:.2f}",
              flush=True)
    else:
        print(f"no agent checkpoint at {ckpt}; baselines only", flush=True)

    keys = ["mean_pnl", "sharpe", "mean_nd_pnl", "mean_pnlmap", "mean_pr",
            "mean_max_drawdown", "mean_abs_inv", "mean_final_abs_inv",
            "win_rate", "mean_fills"]
    print(f"\n{'method':<8}" + "".join(f"{k:>14}" for k in keys))
    for name, r in results.items():
        print(f"{name:<8}" + "".join(f"{r[k]:>14.3f}" for k in keys))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, args.out), "w") as f:
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

"""Train the PPO market-making agent (smoke run by default).

Each iteration: collect N fresh TRAIN episodes (batched encoder
forward per episode, frozen trunk), compute GAE advantages, run a PPO
update. Logs per-iteration mean PnL (raw z-units), mean fills and
mean |inventory| on train episodes, plus PPO update stats.

The smoke defaults (30 iterations x 16 episodes x 200 steps) are
deliberately modest: the TLOB overnight training shares this box.
The full run (more iterations/episodes) is scripted in run_full.sh.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from encoder import AttnLOB
from env import EnvParams, MMEnv, sample_starts, valid_starts
from lob_data import load_lob
from ppo import (ActorCritic, collect_episode, compute_gae,
                 episode_features, ppo_update)

torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--episodes-per-iter", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4,
                    help="PPO epochs per update (fewer = smaller, more "
                         "stable steps; the smoke/V1/V2 runs used 4 and "
                         "showed clip_frac 0.3-0.5)")
    ap.add_argument("--init-std", type=float, default=0.3,
                    help="initial Gaussian policy std (exploration)")
    ap.add_argument("--init-a1", type=float, default=0.5,
                    help="warm-start mean of the skew action a1 (0..1); "
                         "0.1 = near-zero skew, like the fixed baseline")
    ap.add_argument("--init-a2", type=float, default=0.5,
                    help="warm-start mean of the spread action a2 (0..1); "
                         "0.9 = start quoting wide, near the calibrated "
                         "fixed-spread baseline")
    ap.add_argument("--eta", type=float, default=0.5,
                    help="dampened-PnL parameter (paper: 0.5; 0 = no dampening, "
                         "DP = dPnL, so reward telescopes to PnL)")
    ap.add_argument("--min-half-spread", type=float, default=0.0005,
                    help="floor on the quoted half-spread (env)")
    ap.add_argument("--tp-weight", type=float, default=1.0,
                    help="weight on the trading-PnL reward term (0 drops TP)")
    ap.add_argument("--zeta", type=float, default=None,
                    help="inventory-punishment coefficient (default: env default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder-ckpt", default="results/attn_lob_pretrained.pt")
    ap.add_argument("--out", default="results/ppo_agent.pt")
    ap.add_argument("--log", default="results/ppo_training_log.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    here = os.path.dirname(os.path.abspath(__file__))
    params = EnvParams()
    if args.zeta is not None:
        params.zeta = args.zeta
    params.eta = args.eta
    params.min_half_spread = args.min_half_spread
    params.tp_weight = args.tp_weight
    reward_scale = 1.0 / (2.0 * params.half_spread)
    print(f"params: zeta={params.zeta} eta={params.eta} "
          f"tp_weight={params.tp_weight} "
          f"min_half_spread={params.min_half_spread} "
          f"half_spread={params.half_spread} "
          f"reward_scale={reward_scale:.0f}", flush=True)

    print("loading FI-2010 train ...", flush=True)
    d = load_lob(horizon=10)
    tr_f, tr_mid = d["train_feats"], d["train_mid"]

    encoder = AttnLOB()
    encoder.load_state_dict(torch.load(os.path.join(here, args.encoder_ckpt),
                                       map_location="cpu"))
    encoder.eval()
    policy = ActorCritic(encoder, init_std=args.init_std,
                         init_a1=args.init_a1, init_a2=args.init_a2)
    opt = torch.optim.Adam([p for p in policy.parameters()
                            if p.requires_grad], lr=args.lr)
    print(f"trainable params: "
          f"{sum(p.numel() for p in policy.parameters() if p.requires_grad)}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    # Episodes must not span stock/day boundary jumps in the
    # concatenated FI-2010 files (|d mid| >= 0.02).
    valid = valid_starts(tr_mid, params.episode_len, params.window)
    log = []
    for it in range(args.iterations):
        starts = sample_starts(args.episodes_per_iter, len(tr_mid),
                               params.episode_len, params.window,
                               seed=int(rng.integers(0, 2 ** 31 - 1)),
                               valid=valid)
        bufs, pnls, fills, invs, a1s, a2s = [], [], [], [], [], []
        for s in starts:
            env = MMEnv(tr_f, tr_mid, s, params)
            feats_t = episode_features(encoder, tr_f, tr_mid, s,
                                       params.episode_len, params.window,
                                       device)
            ep = collect_episode(env, policy, feats_t, device, reward_scale)
            adv = compute_gae(ep["rewards"], ep["values"],
                              gamma=args.gamma, lam=args.lam)
            ep["advantages"] = adv
            ep["returns"] = adv + ep["values"]
            bufs.append(ep)
            pnls.append(ep["pnl"])
            fills.append(ep["fills"])
            invs.append(ep["mean_abs_inv"])
            a1s.append(float(np.mean(ep["actions"][:, 0])))
            a2s.append(float(np.mean(ep["actions"][:, 1])))
        buf = {k: np.concatenate([b[k] for b in bufs])
               for k in ("features", "actions", "logprobs", "advantages",
                         "returns")}
        stats = ppo_update(policy, opt, buf, clip=args.clip,
                           ent_coef=args.ent_coef, epochs=args.epochs,
                           device=device)
        rec = {"iter": it, "mean_pnl": float(np.mean(pnls)),
               "std_pnl": float(np.std(pnls)),
               "mean_fills": float(np.mean(fills)),
               "mean_abs_inv": float(np.mean(invs)),
               "mean_a1": float(np.mean(a1s)),
               "mean_a2": float(np.mean(a2s)), **stats}
        log.append(rec)
        print(f"iter {it}: mean_pnl={rec['mean_pnl']:+.4f} "
              f"fills={rec['mean_fills']:.1f} |inv|={rec['mean_abs_inv']:.2f} "
              f"a1={rec['mean_a1']:.2f} a2={rec['mean_a2']:.2f} "
              f"clip={stats['clip_frac']:.2f} ent={stats['entropy']:.2f}",
              flush=True)
    torch.save(policy.state_dict(), os.path.join(here, args.out))
    with open(os.path.join(here, args.log), "w") as f:
        json.dump({"params": {"zeta": params.zeta,
                              "eta": params.eta,
                              "tp_weight": params.tp_weight,
                              "min_half_spread": params.min_half_spread,
                              "init_std": args.init_std,
                              "init_a1": args.init_a1,
                              "init_a2": args.init_a2,
                              "half_spread": params.half_spread,
                              "lr": args.lr, "gamma": args.gamma,
                              "lam": args.lam, "clip": args.clip,
                              "ent_coef": args.ent_coef,
                              "iterations": args.iterations,
                              "episodes_per_iter": args.episodes_per_iter},
                   "log": log}, f, indent=2)
    print(f"saved {args.out} + {args.log}", flush=True)


if __name__ == "__main__":
    main()

"""Train the PPO market-making agent on REAL LOBSTER data (AAPL).

Mirrors train_agent.py (FI-2010) with the data source swapped: the
AAPL .npz + RealMMEnv. V5 recipe defaults (see run_full_real.sh):
lr 1e-5, gamma/lam 0.9, eta 0, zeta 0.1, min-half-spread $0.01,
ent-coef 0.005, init-std 0.3, init-a1 0.05, init-a2 0.9, clip 0.1,
2 PPO epochs. Rewards are scaled by 1 / market spread (dollars).
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
from ppo import (ActorCritic, collect_episode, compute_gae,
                 episode_features, ppo_update)
from real_env import RealMMEnv, real_env_params

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "AAPL_10.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--episodes-per-iter", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--init-a1", type=float, default=0.05)
    ap.add_argument("--init-a2", type=float, default=0.9)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--min-half-spread", type=float, default=0.01)
    ap.add_argument("--tp-weight", type=float, default=1.0)
    ap.add_argument("--zeta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder-ckpt",
                    default="results/attn_lob_real_pretrained.pt")
    ap.add_argument("--init-policy", default=None,
                    help="optional policy checkpoint to warm-start from "
                         "(e.g. results/ppo_agent_real.ckpt.pt left by a "
                         "stalled run); training then runs --iterations "
                         "further iters from those weights")
    ap.add_argument("--out", default="results/ppo_agent_real.pt")
    ap.add_argument("--log", default="results/ppo_training_log_real.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    params = real_env_params()
    params.eta = args.eta
    params.zeta = args.zeta
    params.min_half_spread = args.min_half_spread
    params.tp_weight = args.tp_weight
    reward_scale = 1.0 / (2.0 * params.half_spread)
    print(f"params: zeta={params.zeta} eta={params.eta} "
          f"tp_weight={params.tp_weight} "
          f"min_half_spread={params.min_half_spread} "
          f"half_spread={params.half_spread} "
          f"reward_scale={reward_scale:.2f}", flush=True)

    print("loading AAPL ...", flush=True)
    d = dict(np.load(DATA))
    n_train = int(d["train_mask"].sum())

    encoder = AttnLOB()
    encoder.load_state_dict(torch.load(os.path.join(HERE, args.encoder_ckpt),
                                       map_location="cpu"))
    encoder.eval()
    policy = ActorCritic(encoder, init_std=args.init_std,
                         init_a1=args.init_a1, init_a2=args.init_a2)
    if args.init_policy:
        policy.load_state_dict(torch.load(
            os.path.join(HERE, args.init_policy), map_location="cpu"))
        print(f"warm-started policy from {args.init_policy}", flush=True)
    opt = torch.optim.Adam([p for p in policy.parameters()
                            if p.requires_grad], lr=args.lr)
    print(f"trainable params: "
          f"{sum(p.numel() for p in policy.parameters() if p.requires_grad)}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    log = []
    for it in range(args.iterations):
        starts = sorted(rng.integers(
            params.window, n_train - params.episode_len,
            size=args.episodes_per_iter).tolist())
        bufs, pnls, fills, invs, a1s, a2s = [], [], [], [], [], []
        for s in starts:
            env = RealMMEnv(d["feats"], d["mid"], d["raw_book"],
                            d["exec_price"], d["exec_size"], d["exec_side"],
                            d["exec_count"], int(s), params)
            feats_t = episode_features(encoder, d["feats"], d["mid"], int(s),
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
        print(f"iter {it}: mean_pnl=${rec['mean_pnl']:+.2f} "
              f"fills={rec['mean_fills']:.1f} |inv|={rec['mean_abs_inv']:.2f} "
              f"a1={rec['mean_a1']:.2f} a2={rec['mean_a2']:.2f} "
              f"clip={stats['clip_frac']:.2f} ent={stats['entropy']:.2f}",
              flush=True)
        # Periodic checkpoint (every 10 iters): the 2026-11-20 run died
        # silently at iter 33 with no checkpoint and no log, because
        # train_real.py only saved at the end. A stall now loses at most
        # 10 iters, and the partial policy stays evaluable. Uses a
        # separate .ckpt path so run_full_real.sh's stage-2 skip check
        # (which looks for args.out) is unaffected.
        if (it + 1) % 10 == 0:
            base, ext = os.path.splitext(os.path.join(HERE, args.out))
            ckpt_path = base + ".ckpt" + ext
            torch.save(policy.state_dict(), ckpt_path)
            log_base, log_ext = os.path.splitext(
                os.path.join(HERE, args.log))
            with open(log_base + ".partial" + log_ext, "w") as f:
                json.dump({"params": {"zeta": params.zeta,
                                      "eta": params.eta,
                                      "tp_weight": params.tp_weight,
                                      "min_half_spread":
                                          params.min_half_spread,
                                      "init_std": args.init_std,
                                      "init_a1": args.init_a1,
                                      "init_a2": args.init_a2,
                                      "half_spread": params.half_spread,
                                      "lr": args.lr, "gamma": args.gamma,
                                      "lam": args.lam, "clip": args.clip,
                                      "ent_coef": args.ent_coef,
                                      "iterations": args.iterations,
                                      "episodes_per_iter":
                                          args.episodes_per_iter},
                           "log": log}, f, indent=2)
            print(f"checkpointed {ckpt_path} after iter {it}", flush=True)
    torch.save(policy.state_dict(), os.path.join(HERE, args.out))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, args.log), "w") as f:
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

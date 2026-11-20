"""Calibration on REAL LOBSTER data (AAPL, train slice only).

Mirrors calibrate.py (FI-2010) but in dollars:
1. Mid-move / spread distribution on train steps.
2. Reward-component scales under a random policy -> zeta suggestion
   (IP ~ 20% of a typical |dPnL|).
3. Fixed half-spread grid on 20 train episodes (mean PnL).
4. AS (gamma, kappa) grid on 20 train episodes.
The winners are frozen as the defaults in evaluate_real.py; the
sanity gates are then checked on TEST episodes (never train).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from env import (as_policy_factory, evaluate_policy, fixed_policy_factory,
                 random_policy)
from real_env import RealMMEnv, evaluate_real_policy, real_env_params

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "AAPL_10.npz")


def main():
    d = dict(np.load(DATA))
    n = len(d["mid"])
    n_train = int(d["train_mask"].sum())
    p = real_env_params()
    print(f"AAPL: {n} steps, train {n_train}, test {n - n_train}")

    print("== 1. mid-move / spread (train) ==")
    mid_tr = d["mid"][:n_train]
    dm = np.abs(np.diff(mid_tr))
    print(f"  per-step |d mid|: median {np.median(dm):.4f}$ "
          f"mean {dm.mean():.4f}$ std {dm.std():.4f}$ "
          f"P(0)={np.mean(dm == 0):.3f}")
    print(f"  median half-spread (book): {p.half_spread:.3f}$ "
          f"(= {p.half_spread / max(dm.std(), 1e-9):.1f} x per-step std)")
    print(f"  executions/step (train): "
          f"{d['exec_count'][:n_train].mean():.2f}")

    rng = np.random.default_rng(1)
    starts = sorted(rng.integers(p.window, n_train - p.episode_len,
                                 size=20).tolist())

    print("== 2. reward scales under a random policy (20 train eps) ==")
    dpnls, tps, inv2 = [], [], []
    for s in starts:
        env = RealMMEnv(d["feats"], d["mid"], d["raw_book"], d["exec_price"],
                        d["exec_size"], d["exec_side"], d["exec_count"],
                        int(s), p)
        env.reset()
        while not env.done:
            _, _, _, info = env.step(random_policy(env))
            dpnls.append(abs(info["dpnl"]))
            tps.append(abs(info["tp"]))
            inv2.append(env.inv ** 2)
    print(f"  mean |dPnL|/step = ${np.mean(dpnls):.4f}")
    print(f"  mean |TP|/step   = ${np.mean(tps):.4f}")
    print(f"  mean inv^2       = {np.mean(inv2):.2f}")
    zeta_star = 0.2 * np.mean(dpnls) / max(np.mean(inv2), 1e-9)
    print(f"  -> zeta for IP ~ 20% of |dPnL|: {zeta_star:.4f} "
          f"(env default {p.zeta})")

    print("== 3. fixed-spread grid (20 train eps, mean PnL $) ==")
    best_fixed, best_fixed_pnl = None, -1e18
    for half in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        out = evaluate_real_policy(fixed_policy_factory(half), d, starts, p)
        print(f"  fixed half=${half:.2f}: mean_pnl=${out['mean_pnl']:+.2f} "
              f"sharpe={out['sharpe']:+.2f} fills={out['mean_fills']:.1f} "
              f"win={out['win_rate']:.2f} |inv|={out['mean_abs_inv']:.2f}")
        if out["mean_pnl"] > best_fixed_pnl:
            best_fixed, best_fixed_pnl = half, out["mean_pnl"]
    print(f"  BEST fixed half: ${best_fixed:.2f}")

    print("== 4. AS grid (20 train eps, mean PnL $) ==")
    best_as, best_as_sharpe = None, -1e18
    for gamma in (0.1, 0.5, 2.0):
        for kappa in (5.0, 20.0, 50.0, 200.0):
            out = evaluate_real_policy(as_policy_factory(gamma, kappa), d,
                                       starts, p)
            print(f"  AS g={gamma} k={kappa}: mean_pnl=${out['mean_pnl']:+.2f} "
                  f"sharpe={out['sharpe']:+.2f} fills={out['mean_fills']:.1f} "
                  f"win={out['win_rate']:.2f} |inv|={out['mean_abs_inv']:.2f}")
            if out["sharpe"] > best_as_sharpe:
                best_as, best_as_sharpe = (gamma, kappa), out["sharpe"]
    print(f"  BEST AS (by Sharpe): gamma={best_as[0]} kappa={best_as[1]}")


if __name__ == "__main__":
    main()

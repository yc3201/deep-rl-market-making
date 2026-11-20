"""Calibration: measure the scales the env design depends on, on TRAIN.

1. Fill-probability curve: P(|d mid| >= d) per sample — how often a
   resting quote at distance d from the mid is crossed in one step.
   This sets the market half-spread: it must be small enough that a
   reasonable quote gets filled, large enough that fills are not free.
2. Reward-component scales under a random policy: mean |dPnL|, |DP|,
   |TP| per step and E[inv^2]. zeta is chosen so that a typical IP is
   ~20% of a typical |DP + TP| — the paper's zeta = 0.01 was tuned to
   CNY prices and is meaningless in z-units.
3. Baseline grids on TRAIN episodes (never test): fixed half-spread
   and AS (gamma, kappa). Winners are frozen for evaluation.
"""
from __future__ import annotations

import numpy as np

from env import (EnvParams, as_policy_factory, evaluate_policy,
                 fixed_policy_factory, random_policy, sample_starts,
                 valid_starts)
from lob_data import load_lob


def main():
    d = load_lob(horizon=10)
    tr_mid = d["train_mid"]
    tr_f = d["train_feats"]
    p = EnvParams()
    print("== 1. mid-move distribution (train) ==")
    dm = np.abs(np.diff(tr_mid))
    for thr in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        print(f"  P(|d mid| >= {thr:.3f}) = {np.mean(dm >= thr):.4f}")
    print(f"  per-step |d mid|: median {np.median(dm):.5f} "
          f"mean {dm.mean():.5f} std {dm.std():.5f}")
    print(f"  market half-spread in use: {p.half_spread} "
          f"(= {p.half_spread / dm.std():.1f} x per-step std)")

    print("== 2. reward scales under a random policy (20 train eps) ==")
    starts = sample_starts(20, len(tr_mid), p.episode_len, p.window, seed=1,
                           valid=valid_starts(tr_mid, p.episode_len,
                                              p.window))
    # Re-run random episodes capturing per-step components via a wrapper.
    from env import MMEnv
    dpnls, tps, inv2 = [], [], []
    for s in starts:
        env = MMEnv(tr_f, tr_mid, s, p)
        env.reset()
        while not env.done:
            _, _, _, info = env.step(random_policy(env))
            dpnls.append(abs(info["dpnl"]))
            tps.append(abs(info["tp"]))
            inv2.append(env.inv ** 2)
    print(f"  mean |dPnL|/step = {np.mean(dpnls):.5f}")
    print(f"  mean |TP|/step   = {np.mean(tps):.5f}")
    print(f"  mean inv^2       = {np.mean(inv2):.2f}")
    zeta_star = 0.2 * np.mean(dpnls) / max(np.mean(inv2), 1e-9)
    print(f"  -> zeta for IP ~ 20% of |dPnL|: {zeta_star:.5f} "
          f"(paper value 0.01; env default {p.zeta})")

    print("== 3. fixed-spread grid (20 train eps, mean PnL) ==")
    for half in (0.00025, 0.0005, 0.00075, 0.001):
        out = evaluate_policy(fixed_policy_factory(half), tr_f, tr_mid,
                              starts, p)
        print(f"  fixed half={half:.3f}: mean_pnl={out['mean_pnl']:+.4f} "
              f"sharpe={out['sharpe']:+.2f} fills={out['mean_fills']:.1f} "
              f"win={out['win_rate']:.2f}")

    print("== 4. AS grid (20 train eps, mean PnL) ==")
    for gamma in (0.5, 2.0, 5.0):
        for kappa in (500.0, 1000.0, 2000.0):
            out = evaluate_policy(as_policy_factory(gamma, kappa), tr_f,
                                  tr_mid, starts, p)
            print(f"  AS g={gamma} k={kappa}: mean_pnl={out['mean_pnl']:+.4f} "
                  f"sharpe={out['sharpe']:+.2f} fills={out['mean_fills']:.1f} "
                  f"win={out['win_rate']:.2f} |inv|={out['mean_abs_inv']:.2f}")


if __name__ == "__main__":
    main()

"""Probe: fixed-spread grid on TRAIN episodes (diagnostic only).

The V2 validation showed PPO warm-started at a2=0.82 drifting back
to a2~0.5 during training, even though the fixed grid on TEST says
wider is better. Hypothesis: on TRAIN episodes (a more volatile
regime than test) tight quoting is genuinely better in reward terms,
so the agent is correctly optimizing the TRAIN objective and the
problem is train/test regime transfer, not the optimizer. This probe
measures the fixed grid on train episodes to test that hypothesis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lob_data import load_lob
from env import (EnvParams, evaluate_policy, fixed_policy_factory,
                 sample_starts, valid_starts)


def main():
    d = load_lob(horizon=10)
    tr_f, tr_mid = d["train_feats"], d["train_mid"]
    p = EnvParams()
    starts = sample_starts(50, len(tr_mid), p.episode_len, p.window,
                           seed=7,
                           valid=valid_starts(tr_mid, p.episode_len,
                                              p.window))
    for half in (0.0005, 0.00075, 0.001):
        r = evaluate_policy(fixed_policy_factory(half), tr_f, tr_mid,
                            starts, p)
        print(f"train fixed half={half}: mean_pnl={r['mean_pnl']:+.5f} "
              f"sharpe={r['sharpe']:+.2f} fills={r['mean_fills']:.2f} "
              f"win={r['win_rate']:.2f} |inv|={r['mean_abs_inv']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()

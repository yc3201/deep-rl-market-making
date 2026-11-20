"""One-off data probe: scales of the FI-2010 z-scored LOB series.

Reads ONLY rows 0-3 (level-1 ask/bid price+volume) and the h=10 label
row (row 144) of the train file, so it stays light on memory.
Reports the numbers the env design needs: typical spread in z-units,
mid-price step size, class shares, and the ask>bid sanity check.
"""
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "tlob-midprice-prediction", "data", "FI_2010")
TRAIN_FILE = "Train_Dst_NoAuction_ZScore_CF_7.txt"


def read_rows(path, rows):
    """Read selected 0-indexed lines of a whitespace text file as float64."""
    out = {}
    wanted = set(rows)
    with open(path) as f:
        for i, line in enumerate(f):
            if i in wanted:
                out[i] = np.fromstring(line, sep=" ")
            if i > max(wanted):
                break
    return [out[r] for r in rows]


def main():
    path = os.path.join(DATA_DIR, TRAIN_FILE)
    ask1, ask1v, bid1, bid1v, lab10 = read_rows(path, [0, 1, 2, 3, 144])
    n = len(ask1)
    print(f"train samples: {n}")
    print(f"ask1_z > bid1_z fraction: {np.mean(ask1 > bid1):.4f}")
    mid = (ask1 + bid1) / 2.0
    spread = ask1 - bid1
    print(f"mid_z: mean {mid.mean():.4f} std {mid.std():.4f} "
          f"min {mid.min():.3f} max {mid.max():.3f}")
    print(f"spread_z: median {np.median(spread):.5f} mean {spread.mean():.5f} "
          f"p10 {np.percentile(spread, 10):.5f} p90 {np.percentile(spread, 90):.5f} "
          f"frac<=0: {np.mean(spread <= 0):.4f}")
    dmid = np.diff(mid)
    print(f"|d mid| per sample: median {np.median(np.abs(dmid)):.5f} "
          f"mean {np.mean(np.abs(dmid)):.5f} std {dmid.std():.5f}")
    print(f"ask1 vol_z: mean {ask1v.mean():.3f} std {ask1v.std():.3f}; "
          f"bid1 vol_z: mean {bid1v.mean():.3f} std {bid1v.std():.3f}")
    for c in (1, 2, 3):
        print(f"h=10 label {c} share: {np.mean(lab10 == c):.3f} "
              f"(1=up, 2=flat, 3=down)")


if __name__ == "__main__":
    main()

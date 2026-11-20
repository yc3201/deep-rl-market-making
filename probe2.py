"""Fine-grained mid-move distribution + boundary detection (train file)."""
import numpy as np
from lob_data import load_lob

d = load_lob(horizon=10)
mid = d["train_mid"]
dm = np.abs(np.diff(mid))
print(f"n = {len(mid)}")
print(f"P(|dmid| == 0)      = {np.mean(dm == 0):.4f}")
for thr in (1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3):
    print(f"P(|dmid| >= {thr:g}) = {np.mean(dm >= thr):.4f}")
nz = dm[dm > 0]
print(f"nonzero |dmid|: n={len(nz)} median={np.median(nz):.6f} "
      f"mean={nz.mean():.6f} p90={np.percentile(nz, 90):.6f}")
# candidate stock/day boundaries: large jumps
for thr in (0.02, 0.05, 0.1):
    idx = np.where(dm >= thr)[0]
    print(f"jumps >= {thr}: count={len(idx)}", end=" ")
    if len(idx):
        gaps = np.diff(idx)
        print(f"median gap={np.median(gaps):.0f} min gap={gaps.min()} "
              f"max gap={gaps.max()}")
    else:
        print()
# within-segment move size: split at |dmid| >= 0.02, look at the longest segment
idx = np.where(dm >= 0.02)[0]
bounds = np.concatenate([[-1], idx, [len(mid) - 1]])
segs = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
segs = [(a, b) for a, b in segs if b - a > 500]
print(f"segments >500 samples (split at 0.02): {len(segs)}")
for a, b in segs[:8]:
    seg = np.abs(np.diff(mid[a:b]))
    print(f"  [{a}:{b}] len={b-a} P(|d|>0)={np.mean(seg>0):.3f} "
          f"median|d|={np.median(seg):.6f} mean|d|={seg.mean():.6f}")

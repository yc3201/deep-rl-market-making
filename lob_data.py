"""FI-2010 data loading for the deep-RL market-making project.

Follows the same conventions as the TLOB project's ``data.py`` (verified
label mapping, standard split) but loads ONLY the rows it needs, because
the train file is 607 MB on disk and this box is memory-constrained
(TLOB's overnight training is running alongside):

* Files: ``Train_Dst_NoAuction_ZScore_CF_7.txt`` (days 1-7) and
  ``Test_Dst_NoAuction_ZScore_CF_7/8/9.txt`` (days 8-10). Each file is
  stored transposed: 149 LINES, each line one feature with n_samples
  values. Rows 0-39 are the LOB (10 levels x ask price, ask volume,
  bid price, bid volume); rows 144-148 are the five horizon labels.
* Label rows: horizon 10 -> row 144, 20 -> 145, 30 -> 146, 50 -> 147,
  100 -> 148. Labels are 1/2/3, shifted by -1 to 0/1/2 =
  up / stationary / down (VERIFIED against the FI-2010 dataset record
  and empirically by the TLOB project — do not flip).
* The file samples every 10th event, so 1 sample = 10 events. The
  paper's episode of 2000 events is therefore 200 samples here.

IMPORTANT — what the z-scored files do NOT preserve (verified in
``probe_data.py``): the bid-ask spread. ``ask1_z - bid1_z`` has median
-0.00008, is <= 0 in 57% of samples, and ``ask1_z > bid1_z`` in only
43%: each price series is standardised with its OWN mean/std, so the
level-1 price difference is symmetric noise around zero, not the true
spread. The market-making env therefore uses the REAL book for its
state (encoder input, imbalance, realised volatility) and a calibrated
constant spread for its execution layer — see ``env.py``. Any claim
that the z-scored spread is the market spread is wrong.

Mid-price proxy: ``mid_z = (ask1_z + bid1_z) / 2`` (features 0 and 2).
Per-sample |change| of mid_z: median 0.00000, mean 0.00021, std 0.00531
— prices are sticky at this sampling, moves come in jumps.
"""

from __future__ import annotations

import os

import numpy as np

HORIZON_ROWS = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
CLASS_NAMES = {0: "up", 1: "stationary", 2: "down"}
N_FEATURES_LOB = 40

TRAIN_FILE = "Train_Dst_NoAuction_ZScore_CF_7.txt"
TEST_FILES = [
    "Test_Dst_NoAuction_ZScore_CF_7.txt",
    "Test_Dst_NoAuction_ZScore_CF_8.txt",
    "Test_Dst_NoAuction_ZScore_CF_9.txt",
]

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tlob-midprice-prediction",
                                "data", "FI_2010")


def _load_rows(path: str, n_rows: int, skip: int = 0) -> np.ndarray:
    """Load ``n_rows`` lines starting after ``skip`` lines, as float32.

    ``np.loadtxt`` with ``max_rows`` stops reading after that many data
    lines, so the 607 MB train file is never fully parsed.
    """
    arr = np.loadtxt(path, dtype=np.float32, max_rows=n_rows,
                     skiprows=skip)
    if arr.ndim == 1:  # a single row comes back 1-D; keep (1, n)
        arr = arr.reshape(1, -1)
    return arr


def load_lob(data_dir: str = DEFAULT_DATA_DIR, horizon: int = 10):
    """Load the 40 LOB features + one horizon's labels, train and test.

    Returns dict with float32 arrays:
      train_feats (40, N_train), train_labels (N_train,) in {0,1,2},
      test_feats  (40, N_test),  test_labels  (N_test,),
      train_mid (N_train,), test_mid (N_test,)  — mid_z proxies.
    """
    if horizon not in HORIZON_ROWS:
        raise ValueError(f"horizon must be one of {sorted(HORIZON_ROWS)}")
    row = HORIZON_ROWS[horizon]

    train_path = os.path.join(data_dir, TRAIN_FILE)
    train_feats = _load_rows(train_path, N_FEATURES_LOB)
    train_labels = (_load_rows(train_path, 1, skip=row)[0] - 1).astype(np.int64)

    test_feats_list, test_labels_list = [], []
    for f in TEST_FILES:
        p = os.path.join(data_dir, f)
        test_feats_list.append(_load_rows(p, N_FEATURES_LOB))
        test_labels_list.append((_load_rows(p, 1, skip=row)[0] - 1).astype(np.int64))
    test_feats = np.hstack(test_feats_list)
    test_labels = np.concatenate(test_labels_list)

    out = {
        "train_feats": train_feats,
        "train_labels": train_labels,
        "test_feats": test_feats,
        "test_labels": test_labels,
        "train_mid": (train_feats[0] + train_feats[2]) / 2.0,
        "test_mid": (test_feats[0] + test_feats[2]) / 2.0,
    }
    return out


def class_shares(labels: np.ndarray) -> dict:
    """Fraction of samples in each class {0: up, 1: stationary, 2: down}."""
    return {c: float(np.mean(labels == c)) for c in (0, 1, 2)}


class LOBWindows:
    """Lazy sliding-window view of one (features, labels) matrix.

    Windows are NOT materialised: ``__getitem__`` slices the stored
    float32 matrix, so a stride-10 view over ~250k samples costs one
    copy of the 40-row file. The label is the label at the window's
    FINAL time step (same convention as the TLOB project's loader).
    """

    def __init__(self, feats: np.ndarray, labels: np.ndarray,
                 seq_size: int, stride: int = 10,
                 max_windows: int | None = None):
        self.feats = feats.astype(np.float32)
        self.labels = labels
        self.seq_size = seq_size
        self.stride = stride
        starts = np.arange(0, len(labels) - seq_size + 1, stride)
        starts = starts[labels[starts + seq_size - 1] >= 0]
        if max_windows is not None and len(starts) > max_windows:
            starts = starts[:max_windows]
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i: int):
        start = int(self.starts[i])
        end = start + self.seq_size
        x = self.feats[:, start:end].T.copy()  # [seq, 40]
        y = int(self.labels[end - 1])
        return x, y


def standard_split(n_cols: int, val_frac: float = 0.2):
    """Repo split: first (1-val_frac) of train-file columns -> train."""
    cut = int(n_cols * (1 - val_frac))
    return slice(0, cut), slice(cut, n_cols)


if __name__ == "__main__":
    d = load_lob()
    for k in ("train_feats", "test_feats"):
        print(k, d[k].shape, d[k].dtype)
    print("train h=10 shares (up/flat/down):",
          [round(class_shares(d["train_labels"])[c], 3) for c in (0, 1, 2)])
    print("test  h=10 shares (up/flat/down):",
          [round(class_shares(d["test_labels"])[c], 3) for c in (0, 1, 2)])
    w = LOBWindows(d["train_feats"], d["train_labels"], seq_size=50)
    x, y = w[0]
    print(f"window sample: x {x.shape} {x.dtype}, y={y} ({CLASS_NAMES[y]})")

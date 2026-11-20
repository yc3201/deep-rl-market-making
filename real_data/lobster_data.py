"""Load LOBSTER sample data (NASDAQ, 2012-06-21) for the deep-RL
market-making project.

Source: free single-day samples from lobsterdata.com (legacy host
php.lobsterdata.com), 10 levels, tickers AAPL / INTC / MSFT.
Full LOBSTER access is a paid subscription (academic GBP 7,499/yr +
GBP 499 setup, price list 2026) — NOT purchased; the free samples are
the only LOBSTER data used here. The paper (Guo et al. 2023) used
LOBSTER Tesla/Intel data; these samples are the same vendor and
format, one day (2012-06-21), regular session 09:30-16:00.

Files per ticker (raw/):
  <T>_2012-06-21_34200000_57600000_message_10.csv   (N x 6)
  <T>_2012-06-21_34200000_57600000_orderbook_10.csv (N x 40)
Message columns: Time (s after midnight), Type (1 submit, 2 cancel,
3 delete, 4 exec visible, 5 exec hidden, 7 halt), OrderID, Size
(shares), Price ($ x 10000), Direction (-1 sell limit, +1 buy limit).
Execution of a SELL limit order (Direction -1) is a BUYER-initiated
trade (a buy market order hit the ask); execution of a BUY limit
order (Direction +1) is a SELLER-initiated trade (hit the bid).
Orderbook columns (interleaved, SAME layout as the FI-2010 40
features): AskPrice1, AskSize1, BidPrice1, BidSize1, AskPrice2, ...,
10 levels. Prices $ x 10000.

Sampling: every ``stride``-th event (default 10, like FI-2010's
1 sample = 10 events). The sampled snapshot at event i is the book
AFTER event i; the step's events are message rows i+1 .. i+stride,
and the next snapshot is at i+stride.

Outputs (per ticker, saved as .npz):
  feats (40, N) float32 — LOB snapshots, standardised per row on the
      TRAIN slice (first 75% of the day), exactly like FI-2010.
  mid (N,) float64 — mid price in DOLLARS at sampled indices
      (execution layer uses raw dollars, NOT standardised values).
  raw_book (40, N) float32 — book at sampled indices in dollars /
      shares (queue accounting needs real sizes).
  exec_price / exec_size / exec_side (N, 10) — executions inside each
      sampled step (max 10 events/step): price $, size shares,
      side -1 = sell trade (hits the bid), +1 = buy trade (hits the
      ask). exec_count (N,) how many are real.
  labels (N,) int64 — mid direction 10 steps ahead: +1 up / 0 flat /
      -1 down with a +/- $0.02 (2-tick) deadband. (The TLOB project's
      labels are the FI-2010 file's own; here the rule is defined
      explicitly because LOBSTER ships no labels.)
  train_mask (N,) bool — first 75% of steps (time split).
  stats — coverage numbers for the README.
"""
from __future__ import annotations

import json
import os

import numpy as np

TICK = 0.01            # $0.01 minimum price increment (NASDAQ)
STRIDE = 10            # events per sampled step (FI-2010 convention)
LABEL_HORIZON = 10     # steps ahead for the pretraining label
LABEL_THRESH = 0.02    # $0.02 = 2 ticks deadband
TRAIN_FRAC = 0.75

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
DATA_DIR = os.path.join(HERE, "data")


def _paths(ticker: str):
    base = f"{ticker}_2012-06-21_34200000_57600000"
    return (os.path.join(RAW_DIR, base + "_message_10.csv"),
            os.path.join(RAW_DIR, base + "_orderbook_10.csv"))


def load_lobster_raw(ticker: str, stride: int = STRIDE):
    """Load one ticker's sample files; return the sampled arrays.

    Returns dict with: feats_raw (40, N) dollars/shares, mid (N,)
    dollars, raw_book (40, N), exec arrays (N, stride, ...),
    times (N,) seconds after midnight at sampled indices, n_events.
    """
    msg_path, ob_path = _paths(ticker)
    msg = np.loadtxt(msg_path, delimiter=",", dtype=np.float64)
    ob = np.loadtxt(ob_path, delimiter=",", dtype=np.float64)
    assert len(msg) == len(ob), f"{ticker}: message/orderbook row mismatch"
    n_events = len(msg)
    # sampled indices: 0, stride, 2*stride, ... (leave one stride at the end
    # so every step has a full stride of events after it)
    idx = np.arange(0, n_events - stride, stride)
    n = len(idx)
    ob_s = ob[idx]                       # (N, 40) $x10000 / shares
    feats_raw = np.empty((40, n), dtype=np.float64)
    feats_raw[:40:4] = ob_s[:, 0::4].T / 10000.0    # ask prices $
    feats_raw[1:40:4] = ob_s[:, 1::4].T              # ask sizes (shares)
    feats_raw[2:40:4] = ob_s[:, 2::4].T / 10000.0   # bid prices $
    feats_raw[3:40:4] = ob_s[:, 3::4].T              # bid sizes (shares)
    mid = (feats_raw[0] + feats_raw[2]) / 2.0
    # executions inside each step: message rows idx+1 .. idx+stride
    exec_price = np.zeros((n, stride), dtype=np.float64)
    exec_size = np.zeros((n, stride), dtype=np.float64)
    exec_side = np.zeros((n, stride), dtype=np.float64)
    exec_count = np.zeros(n, dtype=np.int64)
    for s in range(n):
        rows = msg[idx[s] + 1: idx[s] + stride + 1]
        ex = rows[np.isin(rows[:, 1], (4, 5))]
        k = len(ex)
        exec_count[s] = k
        if k:
            exec_price[s, :k] = ex[:, 4] / 10000.0
            exec_size[s, :k] = ex[:, 3]
            # message Direction -1 (sell limit exec) = BUY trade (+1);
            # Direction +1 (buy limit exec) = SELL trade (-1)
            exec_side[s, :k] = -ex[:, 5]
    return {
        "ticker": ticker,
        "n_events": n_events,
        "times": msg[idx, 0],
        "feats_raw": feats_raw,
        "mid": mid,
        "raw_book": feats_raw.copy(),
        "exec_price": exec_price,
        "exec_size": exec_size,
        "exec_side": exec_side,
        "exec_count": exec_count,
    }


def build_dataset(ticker: str, stride: int = STRIDE) -> dict:
    """Standardise + label + split; save .npz; return stats dict."""
    d = load_lobster_raw(ticker, stride)
    n = len(d["mid"])
    n_train = int(n * TRAIN_FRAC)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:n_train] = True
    # standardise each of the 40 rows on the TRAIN slice (FI-2010 rule)
    feats = d["feats_raw"].astype(np.float64)
    mu = feats[:, train_mask].mean(axis=1, keepdims=True)
    sd = feats[:, train_mask].std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    feats_z = ((feats - mu) / sd).astype(np.float32)
    # labels: mid direction LABEL_HORIZON steps ahead, 2-tick deadband
    move = np.full(n, np.nan)
    move[: n - LABEL_HORIZON] = (d["mid"][LABEL_HORIZON:]
                                 - d["mid"][: n - LABEL_HORIZON])
    labels = np.zeros(n, dtype=np.int64)
    labels[move > LABEL_THRESH] = 1
    labels[move < -LABEL_THRESH] = -1
    labels[n - LABEL_HORIZON:] = 0  # no future: call flat (excluded below)
    # coverage stats
    ask1, bid1 = d["feats_raw"][0], d["feats_raw"][2]
    spread_ticks = np.round((ask1 - bid1) / TICK).astype(int)
    ex_all = d["exec_count"].sum()
    ex_shares = d["exec_size"].sum()
    stats = {
        "ticker": ticker,
        "n_events": int(d["n_events"]),
        "n_steps": int(n),
        "n_train_steps": int(n_train),
        "n_test_steps": int(n - n_train),
        "time_first_s": float(d["times"][0]),
        "time_last_s": float(d["times"][-1]),
        "mid_mean": float(d["mid"].mean()),
        "mid_range": [float(d["mid"].min()), float(d["mid"].max())],
        "spread_ticks_median": float(np.median(spread_ticks)),
        "spread_ticks_mean": float(spread_ticks.mean()),
        "spread_1tick_share": float(np.mean(spread_ticks == 1)),
        "l1_ask_size_median": float(np.median(d["feats_raw"][1])),
        "l1_bid_size_median": float(np.median(d["feats_raw"][3])),
        "executions_total": int(ex_all),
        "executions_per_step_mean": float(ex_all / n),
        "exec_shares_total": float(ex_shares),
        "exec_size_mean": float(d["exec_size"][d["exec_count"] > 0].mean())
        if ex_all else 0.0,
        "label_shares": {str(c): float(np.mean(labels[:n - LABEL_HORIZON] == c))
                         for c in (-1, 0, 1)},
        "mid_abs_move_10steps_median": float(np.nanmedian(np.abs(move))),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{ticker}_10.npz")
    np.savez_compressed(
        out_path,
        feats=feats_z,
        mid=d["mid"],
        raw_book=d["raw_book"].astype(np.float32),
        exec_price=d["exec_price"].astype(np.float32),
        exec_size=d["exec_size"].astype(np.float32),
        exec_side=d["exec_side"].astype(np.float32),
        exec_count=d["exec_count"],
        labels=labels,
        train_mask=train_mask,
        feat_mean=mu[:, 0].astype(np.float64),
        feat_std=sd[:, 0].astype(np.float64),
    )
    stats_path = os.path.join(DATA_DIR, f"{ticker}_10_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"{ticker}: {n} steps ({n_train} train / {n - n_train} test), "
          f"{d['n_events']} events -> saved {out_path}")
    print(f"  spread median {stats['spread_ticks_median']:.0f} ticks, "
          f"L1 sizes ask {stats['l1_ask_size_median']:.0f} / "
          f"bid {stats['l1_bid_size_median']:.0f} shares, "
          f"executions {ex_all} ({ex_all / n:.2f}/step)")
    print(f"  labels (up/flat/down): "
          f"{[round(stats['label_shares'][str(c)], 3) for c in (1, 0, -1)]}")
    return stats


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["AAPL", "INTC", "MSFT"]
    for t in tickers:
        build_dataset(t)

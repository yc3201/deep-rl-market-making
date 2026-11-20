"""Unit tests for the real-data env (real_env.py).

All tests run on HAND-BUILT books (no LOBSTER data), so they are
fast and deterministic. They pin down the parts an interviewer would
probe: tick rounding, queue-ahead accounting, partial fills from
real executions, trade-through fills, taker price improvement on
marketable quotes, crossed-book fills, and liquidation.
"""
import os

import numpy as np
import pytest

from real_env import RealMMEnv, real_env_params

N = 300
MID = 100.04


def make_book(n=N, ask1=100.08, bid1=100.00, ask_size=100, bid_size=100):
    """(40, n) raw book: 10 levels, 2-tick gaps, interleaved layout
    [ask px, ask size, bid px, bid size] per level."""
    book = np.zeros((40, n), dtype=np.float32)
    for lvl in range(10):
        book[4 * lvl] = ask1 + 0.02 * lvl
        book[4 * lvl + 1] = ask_size
        book[4 * lvl + 2] = bid1 - 0.02 * lvl
        book[4 * lvl + 3] = bid_size
    return book


def make_env(mid=None, book=None, execs=None, episode_len=5, **params):
    """execs: dict step_index -> list of (price, size, side)."""
    feats = np.random.default_rng(0).standard_normal((40, N)).astype(np.float32)
    mid = np.full(N, MID) if mid is None else np.asarray(mid, dtype=float)
    book = make_book() if book is None else book
    ep = np.zeros((N, 10), dtype=np.float32)
    es = np.zeros((N, 10), dtype=np.float32)
    sd = np.zeros((N, 10), dtype=np.float32)
    ec = np.zeros(N, dtype=np.int64)
    for s, evs in (execs or {}).items():
        ec[s] = len(evs)
        for k, (price, size, side) in enumerate(evs):
            ep[s, k], es[s, k], sd[s, k] = price, size, side
    p = real_env_params()
    p.episode_len = episode_len
    for k, v in params.items():
        setattr(p, k, v)
    return RealMMEnv(feats, mid, book, ep, es, sd, ec, start=100, params=p)


def test_quotes_are_tick_rounded():
    env = make_env()
    env.reset()
    # mid 100.04, a2 = 0.05 -> qs = max(0.05*0.40, 0.02) = 0.02
    # bid = 100.04 - 0.01 = 100.03, ask = 100.05
    bid, ask = env.quotes_from_action((0.0, 0.05))
    assert bid == pytest.approx(100.03) and ask == pytest.approx(100.05)


def test_queue_ahead_at_best_is_level_size():
    env = make_env()
    assert env._queue_ahead(100, 100.00, "bid") == pytest.approx(100.0)


def test_queue_ahead_inside_spread_is_zero():
    env = make_env()
    # bid at 100.04 is above the best bid (100.00): alone at its price
    assert env._queue_ahead(100, 100.04, "bid") == pytest.approx(0.0)


def test_queue_ahead_below_best_sums_levels():
    env = make_env()
    # bid at 99.98 (2nd level): 100 shares at 100.00 + 100 at 99.98
    assert env._queue_ahead(100, 99.98, "bid") == pytest.approx(200.0)


def test_inside_spread_bid_fills_in_full_on_first_sell():
    # bid at 100.03 is inside the spread (queue 0): the first sell
    # execution (181 shares) fills the whole 100-share order at 100.03
    env = make_env(execs={100: [(100.00, 181, -1)]})
    env.reset()
    env.step((0.0, 0.05))  # a2=0.05 -> qs 0.02 -> bid 100.03, ask 100.05
    assert env.inv == pytest.approx(1.0)
    assert env.cash == pytest.approx(-100.03 * 100)


def test_sell_execution_at_best_bid_partial_fill():
    env = make_env(execs={100: [(100.00, 181, -1)]})
    env.reset()
    # a2 = 0.2 -> qs = 0.08 -> bid = 100.04 - 0.04 = 100.00 (best bid)
    env.step((0.0, 0.2))
    assert env.inv == pytest.approx(0.81)
    assert env.cash == pytest.approx(-100.00 * 81)
    assert env.fills == 1 and env.fill_shares == pytest.approx(81.0)


def test_sell_execution_smaller_than_queue_no_fill():
    env = make_env(execs={100: [(100.00, 50, -1)]})
    env.reset()
    env.step((0.0, 0.2))  # bid at best, queue 100, exec only 50
    assert env.inv == pytest.approx(0.0) and env.fills == 0


def test_trade_through_price_fills_in_full():
    # sell exec at 99.98 (below the 100.00 bid): walked through the level
    env = make_env(execs={100: [(99.98, 50, -1)]})
    env.reset()
    env.step((0.0, 0.2))
    assert env.inv == pytest.approx(1.0)
    assert env.cash == pytest.approx(-100.00 * 100)


def test_marketable_bid_fills_at_best_ask_not_limit():
    # short inventory + max skew pushes the reservation price up so the
    # bid crosses the ask; the agent is the TAKER: fills at best ask.
    env = make_env()
    env.reset()
    env.inv = -5.0
    env.step((1.0, 0.05))  # p_r = 100.04 + 0.20 = 100.24, bid = 100.23
    assert env.inv == pytest.approx(-4.0)
    assert env.cash == pytest.approx(-100.08 * 100)  # best ask, not 100.23


def test_crossed_book_after_step_fills_at_quote():
    # next step's book: best ask collapses to 99.99, below the resting
    # bid 100.00 -> the resting bid is filled at its own price (maker).
    book = make_book()
    book[0, 101] = 99.99  # ask1 at step 101
    env = make_env(book=book, episode_len=2)
    env.reset()
    env.step((0.0, 0.2))  # bid 100.00 resting
    assert env.inv == pytest.approx(1.0)
    assert env.cash == pytest.approx(-100.00 * 100)


def test_episode_end_liquidates_and_done_raises():
    env = make_env(episode_len=2)
    env.reset()
    env.inv = 1.0
    env.cash = -100.00 * 100
    _, _, done, _ = env.step((0.0, 0.2))
    assert not done
    _, _, done, _ = env.step((0.0, 0.2))
    assert done
    assert env.inv == 0.0
    # liquidated 100 shares at final mid 100.04: cash = -10000 + 10004
    assert env.pnl == pytest.approx(4.0)
    with pytest.raises(RuntimeError):
        env.step((0.0, 0.2))


def test_determinism_same_episode_twice():
    from real_env import evaluate_real_policy
    from env import fixed_policy_factory
    data = dict(np.load(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "AAPL_10.npz")))
    starts = [100, 500]
    p = real_env_params()
    out1 = evaluate_real_policy(fixed_policy_factory(0.05), data, starts, p)
    out2 = evaluate_real_policy(fixed_policy_factory(0.05), data, starts, p)
    assert out1["mean_pnl"] == pytest.approx(out2["mean_pnl"])
    assert out1["mean_fills"] == pytest.approx(out2["mean_fills"])

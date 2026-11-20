"""Unit tests for the market-making env (env.py).

All tests run on SYNTHETIC mid-price series (no FI-2010 data), so they
are fast and deterministic. They pin down the parts an interviewer
would probe: quote mapping + inventory skew, resting-quote execution,
adverse selection in the reward, the inventory cap, and episode-end
liquidation.
"""
import numpy as np
import pytest

from env import (EnvParams, MMEnv, as_policy_factory, evaluate_policy,
                 fixed_policy_factory, sample_starts)


def make_env(mid, episode_len=50, **params):
    """Synthetic (40, N) feats + given mid series; start=100."""
    n = len(mid)
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((40, n)).astype(np.float32)
    p = EnvParams(episode_len=episode_len, **params)
    env = MMEnv(feats, np.asarray(mid, dtype=np.float64), start=100, params=p)
    return env


def test_quotes_flat_agent_no_skew():
    env = make_env([0.0] * 300)
    bid, ask = env.quotes_from_action((0.0, 0.5))
    # qs = 0.5 * max_spread (0.002) = 0.001; bid/ask = mid -/+ 0.0005
    assert bid == pytest.approx(-0.0005) and ask == pytest.approx(0.0005)


def test_quotes_skew_long_inventory_shifts_down():
    env = make_env([0.0] * 300)
    env.inv = 5
    bid, ask = env.quotes_from_action((1.0, 0.5))
    # p_r = 0 - 1*0.001 = -0.001; bid/ask = -0.001 -/+ 0.0005
    assert bid == pytest.approx(-0.0015) and ask == pytest.approx(-0.0005)


def test_quotes_skew_short_inventory_shifts_up():
    env = make_env([0.0] * 300)
    env.inv = -5
    bid, ask = env.quotes_from_action((1.0, 0.5))
    assert bid == pytest.approx(0.0005) and ask == pytest.approx(0.0015)


def test_no_fill_when_quotes_inside_spread():
    env = make_env([0.0] * 300)
    env.reset()
    for _ in range(10):
        _, reward, done, _ = env.step((0.0, 0.0))
        assert reward == pytest.approx(0.0)
        assert env.inv == 0 and env.fills == 0
    assert not done


def test_resting_bid_filled_when_market_drops_to_it():
    # Episode starts at global index 100. Mid flat 0.0, then drops to
    # -0.03 at index 103, then back to 0.0.
    mid = [0.0] * 103 + [-0.03] + [0.0] * 196
    # zeta=0.01 set explicitly: this test pins the reward FORMULA, not
    # the calibrated default (0.00007); IP = zeta * inv^2 = 0.01.
    env = make_env(mid, zeta=0.01)
    env.reset()
    env.step((0.0, 0.0))  # step 0: post bid=ask=0.0, mid stays 0.0
    env.step((0.0, 0.0))  # step 1: same
    # step 2 acts on mid[102] = 0.0, then the market advances to -0.03:
    # resting bid 0.0 >= best ask (-0.03 + 0.0005 = -0.0295) -> buy 1 @ 0.0
    _, reward, _, info = env.step((0.0, 0.0))
    assert env.inv == 1 and env.fills == 1
    assert env.cash == pytest.approx(0.0)  # bought at 0.0
    # Hand-computed reward: value = 0 + 1*(-0.03) = -0.03; dPnL = -0.03;
    # DP = -0.03 - max(0, 0.5*-0.03) = -0.03; TP = -0.03 - 0.0 = -0.03
    # (adverse selection: filled just before the drop); IP = 0.01*1.
    assert reward == pytest.approx(-0.03 - 0.03 - 0.01)
    assert info["tp"] == pytest.approx(-0.03)


def test_default_zeta_is_the_calibrated_value():
    # calibrate.py: IP ~ 20% of a step's |dPnL| at the random policy's
    # typical inventory -> zeta = 0.00007 (the paper's 0.01 was tuned
    # to CNY prices and is ~140x too big in z-units).
    assert EnvParams().zeta == pytest.approx(0.00007)


def test_dampened_pnl_dampens_gains_not_losses():
    # Long 2 units; mid jumps UP to 0.05 at index 101 (one step ahead).
    # The resting ask (0.001) is picked off on the rise: adverse
    # selection costs TP = -(0.05 - 0.001) = -0.049, while the holding
    # gain is dampened: dPnL = 0.051, DP = 0.051 - 0.5*0.051 = 0.0255.
    mid = [0.0] * 101 + [0.05] + [0.05] * 198
    # zeta=0.01 set explicitly (see the resting-bid test above).
    env = make_env(mid, zeta=0.01)
    env.reset()
    env.inv = 2
    env.cash = 0.0
    env.value_prev = 0.0
    _, reward, _, info = env.step((0.0, 1.0))
    assert info["dpnl"] == pytest.approx(0.051)
    assert info["dp"] == pytest.approx(0.0255)
    assert info["tp"] == pytest.approx(-0.049)
    assert info["ip"] == pytest.approx(0.01)
    assert reward == pytest.approx(0.0255 - 0.049 - 0.01)


def test_inventory_cap_blocks_crowded_side():
    env = make_env([0.0] * 300, max_inventory=1)
    env.reset()
    env.inv = 1
    bid, ask = env.quotes_from_action((0.0, 0.5))
    assert bid is None and ask is not None
    env.inv = -1
    bid, ask = env.quotes_from_action((0.0, 0.5))
    assert bid is not None and ask is None


def test_inventory_never_exceeds_cap_on_falling_market():
    # Mid falls 0.01/step (>= market half-spread), so a tight resting
    # bid is crossed almost every step and inventory climbs to the cap.
    mid = list(np.linspace(0.0, -3.0, 300))
    env = make_env(mid, max_inventory=3)
    env.reset()
    while not env.done:
        env.step((0.0, 0.0))
    assert max(np.abs(env.inv_curve)) <= 3
    assert max(np.abs(env.inv_curve)) == 3  # the cap actually binds


def test_episode_end_liquidates_and_done_raises():
    env = make_env([0.0] * 300, episode_len=5)
    env.reset()
    for _ in range(5):
        _, _, done, _ = env.step((0.0, 0.0))
    assert done
    assert env.pnl == pytest.approx(0.0)
    with pytest.raises(RuntimeError):
        env.step((0.0, 0.0))


def test_liquidation_is_value_neutral():
    # Mid drops to -0.03 at index 103 and STAYS there. The agent buys
    # 1 @ 0.0 on the drop (adverse selection) and holds it to the close.
    # Liquidation at the final mid (-0.03) costs nothing extra: the
    # -0.03 PnL is the fill loss, value before liquidation is already
    # -0.03, and the final cash equals it.
    mid = [0.0] * 103 + [-0.03] * 197
    env = make_env(mid, episode_len=5)
    env.reset()
    env.step((0.0, 0.0))
    env.step((0.0, 0.0))
    env.step((0.0, 0.0))  # fills 1 @ 0.0 on the drop; inv = 1
    assert env.inv == 1
    for _ in range(2):
        env.step((0.0, 0.0))
    assert env.done and env.inv == 0
    assert env.final_inventory == 1
    assert env.pnl == pytest.approx(-0.03)


def test_as_policy_flat_agent_quotes_near_mid():
    env = make_env([0.0] * 300)
    env.reset()
    pol = as_policy_factory(gamma=0.5, kappa=2000.0)
    a1, a2 = pol(env)
    assert a1 == pytest.approx(0.0)  # no inventory -> no skew
    # h = ln(1 + 0.5/2000)/0.5 = 0.0005; a2 = 2h/0.002 = 0.5
    assert a2 == pytest.approx(0.5, abs=0.01)


def test_evaluate_policy_runs_and_reports():
    rng = np.random.default_rng(1)
    mid = np.cumsum(rng.standard_normal(2000) * 0.005)
    feats = rng.standard_normal((40, 2000)).astype(np.float32)
    p = EnvParams(episode_len=200)
    starts = sample_starts(4, 2000, 200, 50, seed=0)
    out = evaluate_policy(fixed_policy_factory(0.0005), feats, mid, starts, p)
    assert out["n_episodes"] == 4
    assert set(out) >= {"mean_pnl", "sharpe", "mean_nd_pnl", "mean_pnlmap",
                        "mean_pr", "mean_max_drawdown", "mean_abs_inv",
                        "mean_final_abs_inv", "win_rate", "mean_fills"}
    assert 0.0 <= out["win_rate"] <= 1.0

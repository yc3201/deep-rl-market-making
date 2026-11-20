"""Market-making env on REAL LOBSTER data (NASDAQ, 2012-06-21).

The fill model is the point of this env — it is trade-flow driven,
not mid-move driven (the FI-2010 env's weakness):

* State. Same as the FI-2010 env: LOB window (50 x 40, standardised)
  + agent state [inv/omega, time left, realised vol, volume
  imbalance]. The mid is in DOLLARS (raw, not standardised); all
  money (PnL, reward) is in dollars. 1 inventory unit = 100 shares
  (the round lot; the median execution size on AAPL is ~81 shares).
* Action. Same (a1, a2) in [0, 1]^2 -> reservation price skewed
  against inventory, spread qs; quotes are ROUNDED to the tick
  ($0.01) because real books live on the tick grid.
* Execution — RESTING quotes filled by REAL trades. Each step is 10
  events (~0.6 s of clock time on AAPL). The agent's quotes rest for
  the step; the step's message-file executions are replayed in order
  against them with exact queue accounting:
    - queue ahead at posting = total displayed volume at prices
      strictly better than the agent's quote, plus the volume at the
      agent's own price level (from the orderbook snapshot at the
      step's start). A quote inside the spread joins an empty level:
      queue ahead 0. A quote at the best bid joins the back of the
      level-1 queue (median 100 shares on AAPL).
    - a sell trade (message Direction +1) at price P consumes the
      bid queue: P > bid eats the levels above; P == bid eats the
      level's queue; P < bid means the book traded through the bid
      -> the agent's remaining size fills in full. A buy trade is
      symmetric on the ask. Fills are PARTIAL by shares (an 81-share
      execution fills 0.81 units of a 100-share order).
    - new submissions / cancels inside the 10-event step are NOT
      tracked for queue accounting (documented approximation: the
      step is ~0.6 s; ignoring cancels ahead of the agent
      UNDERSTATES its fill rate, ignoring new orders above it
      overstates it — both are second-order at this horizon).
    - a quote that is marketable at posting (bid >= best ask) fills
      immediately, and a quote crossed by the book after the step
      (new best bid >= ask) fills as well — in a real book a crossed
      quote is a trade.
  The old FI-2010 channel (fills ONLY when the mid moves through the
  quote) is therefore replaced: here an uninformed trade can fill
  the quote while the mid is flat — spread capture exists — and an
  informed trade that moves the mid still fills it adversely.
* Episode. 200 steps = 2000 events (~2 min on AAPL). Liquidation
  at the final mid, value-neutral.
* Reward. The paper's hybrid reward with the V5 lessons: eta = 0
  (DP = dPnL, no dampening), zeta recalibrated in dollars by
  calibrate_real.py (the FI-2010 zeta is meaningless here).

Regime note (measured, see lobster_data stats): AAPL is a SMALL-tick
stock — spread median 15 ticks, level-1 queues ~100 shares — so a
quoting agent can price-improve inside the spread and get filled by
real flow. INTC/MSFT are BIG-tick stocks (spread 1 tick 70%+ of the
time, level-1 queues ~12k shares): a cancel-replace agent joining
the back of a 12k-share queue is essentially never filled by
uninformed flow, so they are coverage only, not the training stock.
"""
from __future__ import annotations

import numpy as np

from env import EnvParams, _episode_metrics

TICK = 0.01          # $0.01
UNIT_SHARES = 100     # 1 inventory unit = 100 shares


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _round_tick(price: float) -> float:
    """Round a dollar price to the nearest cent (half up)."""
    return float(np.floor(price / TICK + 0.5) * TICK)


def real_env_params() -> EnvParams:
    """EnvParams in DOLLARS for AAPL (see calibrate_real.py for the
    measured values these defaults come from)."""
    p = EnvParams()
    p.episode_len = 200
    p.window = 50
    p.max_inventory = 10        # units = 1000 shares
    p.half_spread = 0.075       # median half-spread on AAPL (15 ticks)
    p.eta = 0.0                 # V5 lesson: no dampening
    p.zeta = 0.1                # V5-analogous small inventory penalty:
                                # calibrate_real.py's formula (IP ~ 20% of
                                # |dPnL|) gives 0.58; V5 used ~1/5 of its
                                # formula value so the reward stays close
                                # to PnL (the FI-2010 failure was an
                                # inventory penalty that dominated the
                                # reward). PnL itself is zeta-independent.
    p.max_bias = 0.20           # 20 ticks
    p.max_spread = 0.40         # 40 ticks
    p.min_half_spread = 0.01    # 1 tick floor on the quoted half-spread
    p.tp_weight = 1.0
    p.rv_scale = 0.1            # per-step |d mid| ~ $0.02 on AAPL
    p.rv_window = 50
    p.imb_window = 10
    return p


class RealMMEnv:
    """One market-making episode on a slice of LOBSTER data."""

    def __init__(self, feats, mid, raw_book, exec_price, exec_size,
                 exec_side, exec_count, start, params=None):
        """feats: (40, N) float32 standardised LOB; mid: (N,) dollars;
        raw_book: (40, N) float32 dollars/shares; exec_*: (N, 10)
        executions inside each sampled step; start: episode start."""
        self.p = params or real_env_params()
        self.feats = feats
        self.mid = mid
        self.raw_book = raw_book
        self.exec_price = exec_price
        self.exec_size = exec_size
        self.exec_side = exec_side
        self.exec_count = exec_count
        self.start = start
        self.n = len(mid)
        self.reset()

    # -- state ---------------------------------------------------------
    def reset(self):
        self.t = 0
        self.cash = 0.0
        self.inv = 0.0          # units (float; partial fills)
        self.value_prev = 0.0
        self.value_curve = [0.0]
        self.inv_curve = [0.0]
        self.fills = 0          # fill EVENTS (one per execution that hits)
        self.fill_shares = 0.0
        self.done = False
        self.rest_bid = None
        self.rest_ask = None
        self.remain_bid = 0.0   # units still resting on the bid
        self.remain_ask = 0.0
        self.final_inventory = 0.0
        return self._obs()

    def _obs(self):
        p = self.p
        idx = self.start + self.t
        lo = idx - p.window + 1
        lob = self.feats[:, lo:idx + 1].T.copy()  # [window, 40]
        return {
            "lob": lob,
            "inventory": self.inv,
            "time_remaining": (p.episode_len - self.t) / p.episode_len,
            "agent_state": self._agent_state(),
            "mid": float(self.mid[idx]),
        }

    def _agent_state(self) -> np.ndarray:
        p = self.p
        idx = self.start + self.t
        lo = max(1, idx - p.rv_window + 1)
        dmid = np.diff(self.mid[lo:idx + 1])
        rv = float(np.sqrt(np.sum(dmid ** 2))) / p.rv_scale if len(dmid) else 0.0
        lo2 = max(0, idx - p.imb_window + 1)
        imb = float(np.mean(self.feats[3, lo2:idx + 1]
                            - self.feats[1, lo2:idx + 1]))
        return np.array([self.inv / p.max_inventory,
                         (p.episode_len - self.t) / p.episode_len,
                         rv, imb], dtype=np.float32)

    # -- quoting --------------------------------------------------------
    def quotes_from_action(self, action):
        """(a1, a2) in [0, 1]^2 -> (bid, ask) in dollars, tick-rounded."""
        p = self.p
        a1, a2 = float(np.clip(action[0], 0, 1)), float(np.clip(action[1], 0, 1))
        mid = float(self.mid[self.start + self.t])
        delta = a1 * p.max_bias
        p_r = mid - _sign(self.inv) * delta
        qs = max(a2 * p.max_spread, 2.0 * p.min_half_spread)
        bid = None if self.inv >= p.max_inventory else _round_tick(p_r - qs / 2.0)
        ask = None if self.inv <= -p.max_inventory else _round_tick(p_r + qs / 2.0)
        return bid, ask

    def _queue_ahead(self, idx: int, price: float, side: str) -> float:
        """Displayed volume ahead of a quote at ``price`` on ``side``.

        side 'bid': sum of bid-level sizes with price > quote price,
        plus the size at the level equal to the quote price (the agent
        joins the back of that level). side 'ask': ask levels with
        price < quote price, plus the level equal to it.
        """
        book = self.raw_book[:, idx]
        if side == "bid":
            px = book[2::4]   # bid prices, best first
            sz = book[3::4]
            return float(sz[px > price].sum() + sz[px == price].sum())
        px = book[0::4]       # ask prices, best first
        sz = book[1::4]
        return float(sz[px < price].sum() + sz[px == price].sum())

    def _fill(self, side: str, price: float, units: float,
              mid: float, tp_acc: list) -> None:
        """Record a fill of ``units`` (1 unit = 100 shares) at ``price``."""
        shares = units * UNIT_SHARES
        if side == "bid":
            self.inv += units
            self.cash -= price * shares
            self.fills += 1
            self.fill_shares += shares
            tp_acc.append((mid - price) * shares)
        else:
            self.inv -= units
            self.cash += price * shares
            self.fills += 1
            self.fill_shares += shares
            tp_acc.append(-(mid - price) * shares)

    # -- step ----------------------------------------------------------
    def step(self, action):
        p = self.p
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        idx = self.start + self.t
        mid = float(self.mid[idx])
        value_start = self.cash + self.inv * UNIT_SHARES * mid

        # 1. Post fresh quotes (replaces the resting ones).
        bid, ask = self.quotes_from_action(action)
        self.rest_bid, self.rest_ask = bid, ask
        self.remain_bid = 1.0 if bid is not None else 0.0
        self.remain_ask = 1.0 if ask is not None else 0.0
        tp_parts: list[float] = []

        # 2. Immediate crossing against the current book (marketable
        #    quotes pay through the spread). The agent is the TAKER
        #    here, so it executes at the book's best price, not its
        #    own limit (price improvement to the taker).
        best_ask = float(self.raw_book[0, idx])
        best_bid = float(self.raw_book[2, idx])
        if bid is not None and bid >= best_ask:
            self._fill("bid", best_ask, self.remain_bid, mid, tp_parts)
            self.rest_bid, self.remain_bid = None, 0.0
        if ask is not None and ask <= best_bid:
            self._fill("ask", best_bid, self.remain_ask, mid, tp_parts)
            self.rest_ask, self.remain_ask = None, 0.0

        # 3. Queue ahead at posting (frozen for this 10-event step).
        q_bid = self._queue_ahead(idx, bid, "bid") if bid is not None else 0.0
        q_ask = self._queue_ahead(idx, ask, "ask") if ask is not None else 0.0

        # 4. Replay the step's real executions in order.
        for k in range(int(self.exec_count[idx])):
            ep = float(self.exec_price[idx, k])
            es = float(self.exec_size[idx, k])
            sd = float(self.exec_side[idx, k])
            if sd < 0 and self.rest_bid is not None:
                # sell trade: hits the bid.
                if ep < bid:
                    # traded through the bid level -> full fill
                    self._fill("bid", bid, self.remain_bid, mid, tp_parts)
                    self.rest_bid, self.remain_bid = None, 0.0
                elif es > q_bid:
                    fill_units = min((es - q_bid) / UNIT_SHARES,
                                     self.remain_bid)
                    self._fill("bid", bid, fill_units, mid, tp_parts)
                    self.remain_bid -= fill_units
                    q_bid = 0.0
                    if self.remain_bid <= 0:
                        self.rest_bid, self.remain_bid = None, 0.0
                else:
                    q_bid -= es
            elif sd > 0 and self.rest_ask is not None:
                # buy trade: hits the ask.
                if ep > ask:
                    self._fill("ask", ask, self.remain_ask, mid, tp_parts)
                    self.rest_ask, self.remain_ask = None, 0.0
                elif es > q_ask:
                    fill_units = min((es - q_ask) / UNIT_SHARES,
                                     self.remain_ask)
                    self._fill("ask", ask, fill_units, mid, tp_parts)
                    self.remain_ask -= fill_units
                    q_ask = 0.0
                    if self.remain_ask <= 0:
                        self.rest_ask, self.remain_ask = None, 0.0
                else:
                    q_ask -= es

        # 5. Advance the market one step; a quote crossed by the new
        #    book is a trade. On the last step, liquidate instead.
        self.t += 1
        if self.t >= p.episode_len:
            self.done = True
            final_mid = float(self.mid[self.start + p.episode_len - 1])
            self.final_inventory = self.inv
            self.cash += self.inv * UNIT_SHARES * final_mid
            self.inv = 0.0
            value = self.cash
            tp = float(sum(tp_parts))
            dpnl = value - value_start
            dp = dpnl - max(0.0, p.eta * dpnl)
            ip = p.zeta * self.final_inventory ** 2
            reward = dp + p.tp_weight * tp - ip
            self.value_curve.append(value)
            self.inv_curve.append(0.0)
            info = {"value": value, "dpnl": dpnl, "dp": dp, "tp": tp, "ip": ip,
                    "fills": self.fills}
            return None, reward, True, info

        mid_new = float(self.mid[self.start + self.t])
        best_ask_new = float(self.raw_book[0, self.start + self.t])
        best_bid_new = float(self.raw_book[2, self.start + self.t])
        if self.rest_bid is not None and self.rest_bid >= best_ask_new:
            self._fill("bid", self.rest_bid, self.remain_bid, mid_new,
                       tp_parts)
            self.rest_bid, self.remain_bid = None, 0.0
        if self.rest_ask is not None and self.rest_ask <= best_bid_new:
            self._fill("ask", self.rest_ask, self.remain_ask, mid_new,
                       tp_parts)
            self.rest_ask, self.remain_ask = None, 0.0
        value = self.cash + self.inv * UNIT_SHARES * mid_new
        tp = float(sum(tp_parts))
        dpnl = value - value_start
        dp = dpnl - max(0.0, p.eta * dpnl)
        ip = p.zeta * self.inv ** 2
        reward = dp + p.tp_weight * tp - ip
        self.value_prev = value
        self.value_curve.append(value)
        self.inv_curve.append(self.inv)
        info = {"value": value, "dpnl": dpnl, "dp": dp, "tp": tp, "ip": ip,
                "fills": self.fills}
        return self._obs(), reward, False, info

    @property
    def pnl(self) -> float:
        """Final PnL in dollars (= final cash; liquidation is neutral)."""
        return self.cash


# ----------------------------------------------------------------------
# Evaluation on identical episodes (mirrors env.evaluate_policy).
# ----------------------------------------------------------------------

def evaluate_real_policy(policy, data: dict, starts, params=None) -> dict:
    """Run ``policy`` on identical episodes (fixed starts) and
    summarise. ``data`` is the dict loaded from the .npz (feats, mid,
    raw_book, exec_*). Sharpe = mean(PnL)/std(PnL) across episodes."""
    p = params or real_env_params()
    recs = []
    for s in starts:
        env = RealMMEnv(data["feats"], data["mid"], data["raw_book"],
                        data["exec_price"], data["exec_size"],
                        data["exec_side"], data["exec_count"], int(s), p)
        env.reset()
        while not env.done:
            env.step(policy(env))
        recs.append(_episode_metrics(env, p))
    pnls = np.array([r["pnl"] for r in recs])
    out = {
        "n_episodes": len(recs),
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "sharpe": float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0.0,
        "mean_nd_pnl": float(np.mean([r["nd_pnl"] for r in recs])),
        "mean_pnlmap": float(np.mean([r["pnlmap"] for r in recs])),
        "mean_pr": float(np.mean([r["pr"] for r in recs])),
        "mean_max_drawdown": float(np.mean([r["max_drawdown"] for r in recs])),
        "mean_abs_inv": float(np.mean([r["mean_abs_inv"] for r in recs])),
        "mean_final_abs_inv": float(np.mean([r["final_abs_inv"] for r in recs])),
        "win_rate": float(np.mean([r["win"] for r in recs])),
        "mean_fills": float(np.mean([r["fills"] for r in recs])),
    }
    return out

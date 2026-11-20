"""Market-making environment: FI-2010 LOB replay + paper's hybrid reward.

Design (paper: Guo et al. 2023, arXiv 2305.15821, Sec. III):

* State. LOB window of length T=50 snapshots x 40 features (the paper's
  input shape), plus agent state: inventory and remaining time, plus
  dynamic features (realised volatility, volume imbalance). The LOB
  window is REAL FI-2010 data (z-scored); the dynamic features are
  computed from the real mid/volume series.
* Action (continuous, paper Sec. III-C2). a1, a2 in [0, 1]:
      delta  = a1 * max_bias
      p_r    = mid - sign(inventory) * delta
      spread = a2 * max_spread
      bid, ask = p_r -/+ spread / 2
  The reservation price is skewed against the inventory (long -> quotes
  shift down to sell faster), exactly as in Avellaneda-Stoikov.
* Execution — RESTING quotes (see note below). Each step the agent
  posts a bid/ask, which rest in the book. A resting quote is filled
  when the market trades through it: (i) immediately, if the quote is
  marketable against the current book (bid >= best ask / ask <= best
  bid); (ii) after the market advances one step, if the new book
  crosses it. A fill buys/sells 1 unit at the agent's quote price.
  Inventory is capped at +-omega (paper: omega=10); at the cap the
  crowded side stops quoting.
* Episode. 200 samples = 2000 events (the paper's episode length; the
  file samples every 10th event). At the end the agent liquidates its
  inventory at the final mid-price (value-neutral: final PnL = value).

WHY RESTING QUOTES — the naive "check the fresh quote against the
current book" model gives ZERO fills for a flat agent: with inventory
0, p_r = mid and bid = mid - spread/2 < best ask, always. A real
limit order earns fills because it RESTS while the market moves to
it — and that is exactly adverse selection: you are filled when the
price moves against you, so the quoted spread must pay for it. The
paper's simulator replays historical orders and fills the agent when
a real order arrives at its price; this env approximates that with
resting quotes crossed by mid-price movement (one step = 10 events).
Documented here because it is the single most consequential modelling
choice in the file.

EXECUTION SPREAD — the modelling choice forced by the data (see
``lob_data.py`` and ``probe2.py``): the z-scored FI-2010 files do NOT
preserve the bid-ask spread (ask1_z - bid1_z is symmetric noise
around 0). So the execution layer uses a CALIBRATED constant market
spread: best_ask = mid + half_spread, best_bid = mid - half_spread.
The calibration is in ticks, not in the per-step std: the mid is
exactly constant in 75% of samples and moves in discrete ticks when
it moves (median nonzero move 0.0005 z-units; P(|d mid| >= 0.0002) =
0.245). The per-step std (0.00531) is dominated by rare ~0.2 jumps at
stock/day boundaries and is the WRONG scale — an earlier version of
this env used half_spread = 0.01 (~2 stds) and got ZERO fills from
every policy, which the calibration caught. Calibrated value (see
``calibrate.py``): half_spread = 0.0005 z-units = 1 tick, so a resting
quote is crossed when the mid moves ~1 tick toward it. Episodes are
sampled to avoid the boundary jumps (|d mid| >= 0.02, 19 in the train
file) — see ``sample_starts``. All PnL is reported in z-units AND in
spread units (ND-PnL = PnL / spread).

Reward (paper Sec. III-D, eq. 16, verbatim structure):
    R_t = DP_t + TP_t - IP_t
    DP_t = dPnL_t - max(0, eta * dPnL_t)      dampened PnL, eta = 0.5
    TP_t = sum over fills v * (mid_t - price) trading PnL (price edge)
    IP_t = zeta * inv_t^2                      inventory punishment
  where dPnL_t = value_t - value_{t-1}, value = cash + inv * mid.
  DP dampens gains from holding (speculation) but not losses; TP
  rewards only the edge captured at the trade; IP grows quadratically
  with inventory. zeta is unit-dependent: the paper's 0.01 was tuned
  to CNY prices; in z-units it is re-calibrated (see ``calibrate.py``)
  so that IP is on the same scale as a step's |dPnL|.

Baselines (same env, identical episodes):
* AS: closed-form Avellaneda-Stoikov quotes, r = mid - q*gamma*sigma^2
  (T-t), half-spread h = gamma*sigma^2*(T-t)/2 + ln(1+gamma/kappa)/gamma,
  with sigma estimated from rolling mid changes. gamma/kappa are
  unit-dependent and calibrated on TRAIN episodes (see calibrate.py):
  in z-units kappa ~ 100 plays the role kappa = 1.5 plays in price
  units, because for small gamma/kappa the spread term is ~1/kappa.
* Fixed: symmetric quotes mid +/- c, c calibrated on TRAIN.
* Random: uniform a1, a2 (sanity baseline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnvParams:
    episode_len: int = 200       # samples per episode (2000 events, paper)
    window: int = 50             # T, LOB window length (paper)
    max_inventory: int = 10      # omega (paper)
    half_spread: float = 0.0005  # market half-spread, z-units = 1 tick
    eta: float = 0.5             # dampened-PnL parameter (paper)
    zeta: float = 0.00007       # inventory-punishment coefficient, calibrated
                                 # (calibrate.py: IP ~ 20% of a step's |dPnL|
                                 # at the random policy's typical inventory;
                                 # the paper's 0.01 was tuned to CNY prices
                                 # and is ~140x too big in z-units)
    max_bias: float = 0.001      # max reservation-price bias, z-units (2 ticks)
    max_spread: float = 0.002    # max quoted spread, z-units (4 ticks)
    min_half_spread: float = 0.0005  # FLOOR on the quoted half-spread
                                 # (1 tick = the market half-spread).
                                 # The smoke-run policy collapsed to a
                                 # near-constant a2 ~ 0.5 and never learned
                                 # to quote wide; without a floor, sampled
                                 # exploration actions quote inside the
                                 # market spread and get picked off for
                                 # free. The floor bounds the worst case;
                                 # the policy still chooses width in
                                 # [min_half_spread, max_spread/2].
                                 # Baselines are unaffected (fixed/AS quote
                                 # at the max width already).
    tp_weight: float = 1.0       # weight on the trading-PnL reward term
                                 # (1.0 = the paper's reward). Set to 0 to
                                 # drop TP (reward = DP - IP only).
    rv_scale: float = 0.05       # normaliser for the RV state feature
    rv_window: int = 50          # samples for realised volatility
    imb_window: int = 10         # samples for volume imbalance


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


class MMEnv:
    """One market-making episode on a slice of FI-2010 LOB data."""

    def __init__(self, feats: np.ndarray, mid: np.ndarray,
                 start: int, params: EnvParams | None = None):
        """feats: (40, N) float32 LOB; mid: (N,) mid_z proxy; the episode
        runs on global indices [start, start + episode_len)."""
        self.p = params or EnvParams()
        self.feats = feats
        self.mid = mid
        self.start = start
        self.n = len(mid)
        self.reset()

    # -- state ---------------------------------------------------------
    def reset(self):
        self.t = 0
        self.cash = 0.0
        self.inv = 0
        self.value_prev = 0.0
        self.value_curve = [0.0]
        self.inv_curve = [0]
        self.fills = 0
        self.done = False
        self.rest_bid = None
        self.rest_ask = None
        self.final_inventory = 0
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
        """[inv/omega, time_remaining, realised vol, volume imbalance]."""
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
        """Continuous action (a1, a2) in [0, 1]^2 -> (bid, ask) prices."""
        p = self.p
        a1, a2 = float(np.clip(action[0], 0, 1)), float(np.clip(action[1], 0, 1))
        mid = float(self.mid[self.start + self.t])
        delta = a1 * p.max_bias
        p_r = mid - _sign(self.inv) * delta
        # Floor the quoted spread: the agent may not quote tighter than
        # min_half_spread on either side (see EnvParams).
        qs = max(a2 * p.max_spread, 2.0 * p.min_half_spread)
        bid = None if self.inv >= p.max_inventory else p_r - qs / 2.0
        ask = None if self.inv <= -p.max_inventory else p_r + qs / 2.0
        return bid, ask

    def _try_fill(self, mid: float, tp_acc: list) -> None:
        """Check resting quotes against the book at ``mid``; fill at most
        one unit per side. Appends trading-PnL contributions to tp_acc."""
        p = self.p
        best_ask = mid + p.half_spread
        best_bid = mid - p.half_spread
        if self.rest_bid is not None and self.rest_bid >= best_ask:
            self.inv += 1
            self.cash -= self.rest_bid
            self.fills += 1
            tp_acc.append(mid - self.rest_bid)  # buy below mid is good
            self.rest_bid = None
        if self.rest_ask is not None and self.rest_ask <= best_bid:
            self.inv -= 1
            self.cash += self.rest_ask
            self.fills += 1
            tp_acc.append(-(mid - self.rest_ask))  # sell above mid is good
            self.rest_ask = None

    # -- step ----------------------------------------------------------
    def step(self, action):
        p = self.p
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        idx = self.start + self.t
        mid = float(self.mid[idx])
        value_start = self.cash + self.inv * mid

        # 1. Post fresh quotes (replaces the resting ones).
        bid, ask = self.quotes_from_action(action)
        self.rest_bid, self.rest_ask = bid, ask
        # 2. Immediate crossing against the current book (marketable
        #    quotes, e.g. a short agent paying through the spread).
        tp_parts: list[float] = []
        self._try_fill(mid, tp_parts)

        # 3. Advance the market one step; resting quotes may be crossed
        #    by the move. On the last step there is no next snapshot:
        #    liquidate instead.
        self.t += 1
        if self.t >= p.episode_len:
            self.done = True
            final_mid = float(self.mid[self.start + p.episode_len - 1])
            self.final_inventory = self.inv
            self.cash += self.inv * final_mid
            self.inv = 0
            value = self.cash
            tp = float(sum(tp_parts))
            dpnl = value - value_start
            dp = dpnl - max(0.0, p.eta * dpnl)
            ip = p.zeta * self.final_inventory ** 2
            reward = dp + p.tp_weight * tp - ip
            self.value_prev = value
            self.value_curve.append(value)
            self.inv_curve.append(0)
            info = {"value": value, "dpnl": dpnl, "dp": dp, "tp": tp, "ip": ip,
                    "fills": self.fills}
            return None, reward, True, info

        mid_new = float(self.mid[self.start + self.t])
        self._try_fill(mid_new, tp_parts)
        value = self.cash + self.inv * mid_new
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
        """Final PnL in z-units (= final cash; liquidation is neutral)."""
        return self.cash


# ----------------------------------------------------------------------
# Baseline policies. A policy is a function env -> (a1, a2) in [0, 1]^2.
# ----------------------------------------------------------------------

def random_policy(env: MMEnv):
    rng = np.random.default_rng()
    return (float(rng.random()), float(rng.random()))


def fixed_policy_factory(half: float):
    """Symmetric quotes mid +/- half (no skew)."""
    def policy(env: MMEnv):
        p = env.p
        a2 = float(np.clip(2.0 * half / p.max_spread, 0, 1))
        return (0.0, a2)
    return policy


def as_policy_factory(gamma: float, kappa: float, sigma: float | None = None):
    """Avellaneda-Stoikov quotes, converted into the (a1, a2) action.

    r = mid - q*gamma*sigma^2*(T-t); h = gamma*sigma^2*(T-t)/2
        + ln(1+gamma/kappa)/gamma. sigma is the per-sample mid-change
    std; if None it is estimated from the env's mid history (rolling
    last 100 samples, floored at 1e-4).
    """
    def policy(env: MMEnv):
        p = env.p
        idx = env.start + env.t
        mid = float(env.mid[idx])
        if sigma is None:
            lo = max(1, idx - 100)
            d = np.diff(env.mid[lo:idx + 1])
            sig = float(np.std(d)) if len(d) > 1 else 1e-4
            sig = max(sig, 1e-4)
        else:
            sig = sigma
        t_rem = p.episode_len - env.t
        q = env.inv
        r = mid - q * gamma * sig ** 2 * t_rem
        h = gamma * sig ** 2 * t_rem / 2.0 + np.log(1.0 + gamma / kappa) / gamma
        delta = abs(q) * gamma * sig ** 2 * t_rem
        a1 = float(np.clip(delta / p.max_bias, 0, 1))
        a2 = float(np.clip(2.0 * h / p.max_spread, 0, 1))
        return (a1, a2)
    return policy


# ----------------------------------------------------------------------
# Evaluation on identical episodes.
# ----------------------------------------------------------------------

def _episode_metrics(env: MMEnv, params: EnvParams) -> dict:
    eq = np.array(env.value_curve)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(peak - eq)) if len(eq) else 0.0
    mean_abs_inv = float(np.mean(np.abs(env.inv_curve)))
    spread = 2.0 * params.half_spread
    pnl = env.pnl
    return {
        "pnl": pnl,
        "nd_pnl": pnl / spread,
        "pnlmap": pnl / mean_abs_inv if mean_abs_inv > 0 else 0.0,
        "pr": pnl / env.fills if env.fills > 0 else 0.0,
        "max_drawdown": max_dd,
        "mean_abs_inv": mean_abs_inv,
        "final_abs_inv": abs(env.final_inventory),
        "fills": env.fills,
        "win": 1.0 if pnl > 0 else 0.0,
    }


def evaluate_policy(policy, feats, mid, starts, params: EnvParams | None = None) -> dict:
    """Run ``policy`` on identical episodes (fixed starts) and summarise.

    Returns mean/std metrics across episodes. Sharpe = mean(PnL)/std(PnL)
    across episodes (the paper reports Sharpe the same way, per stock).
    """
    p = params or EnvParams()
    recs = []
    for s in starts:
        env = MMEnv(feats, mid, int(s), p)
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


def find_boundaries(mid: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    """Indices i where |mid[i+1] - mid[i]| >= thresh — stock/day
    boundaries in the concatenated FI-2010 files (19 in the train
    file). Episodes must not span one."""
    return np.where(np.abs(np.diff(mid)) >= thresh)[0]


def valid_starts(mid: np.ndarray, episode_len: int, window: int,
                 thresh: float = 0.02) -> np.ndarray:
    """Boolean mask over start positions: True iff the episode
    [start, start + episode_len) contains no boundary jump."""
    n = len(mid)
    valid = np.ones(n, dtype=bool)
    valid[:window] = False
    valid[n - episode_len:] = False
    for b in find_boundaries(mid, thresh):
        # a boundary between i and i+1 forbids starts in (i - L + 1, i]
        lo = max(0, b - episode_len + 1)
        valid[lo:b + 1] = False
    return valid


def sample_starts(n: int, n_samples: int, episode_len: int, window: int,
                  seed: int = 0, valid: np.ndarray | None = None) -> list[int]:
    """Episode starts in [window, n_samples - episode_len], seeded.
    If ``valid`` (boolean mask from ``valid_starts``) is given, sample
    only from valid starts."""
    rng = np.random.default_rng(seed)
    if valid is None:
        lo, hi = window, n_samples - episode_len
        return sorted(rng.integers(lo, hi, size=n).tolist())
    idx = np.where(valid)[0]
    return sorted(rng.choice(idx, size=n, replace=False).tolist())

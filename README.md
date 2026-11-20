# Deep-RL Market Making (Attn-LOB + PPO)

From-scratch reimplementation of Guo et al. 2023, **"Market Making with Deep Reinforcement Learning from Limit Order Books"** (arXiv 2305.15821): an Attn-LOB encoder pretrained on mid-price direction, then a PPO agent that quotes bid/ask in continuous action space. Two environments: an FI-2010 replay env and a real-data env on LOBSTER samples.

## Method

- **Paper**: Attn-LOB (CNN over LOB levels + Inception multi-scale temporal filters + multi-head self-attention) pretrained on mid-price direction, then used as the feature trunk of a PPO agent (C-PPO) quoting reservation-price bias + spread, trained with a hybrid reward. On three Shenzhen stocks the paper reports C-PPO ND-PnL 9.3 / 19.8 / 16.0 (×10⁵) and Sharpe 12.3 / 2.2 / 7.1, vs AS 0.49 / 3.14 / 1.57; removing the LOB state collapses ND-PnL 9.34 → 0.15.
- **State**: LOB window of the last 50 snapshots × 40 features (10 levels × ask/bid price and volume) + agent state (inventory, time remaining, realized volatility, volume imbalance).
- **Action**: continuous a1, a2 ∈ [0, 1]. Reservation price p_r = mid − sign(inventory) · a1 · max_bias; spread = a2 · max_spread; bid, ask = p_r ∓ spread/2.
- **Execution**: resting quotes, one unit per side per step, inventory capped at ±10, episodes of 200 steps ending with liquidation at the final mid. On FI-2010 a quote fills when the mid moves through it; on real data fills are driven by replayed executions with exact queue accounting (see below).
- **Reward** (paper eq. 16): R = DP + TP − IP, with DP = ΔPnL (η = 0), TP = price edge captured at the trade, IP = ζ · inv². ζ = 0.00001 on FI-2010, 0.1 on real data.
- **Encoder**: Attn-LOB reimplementation, 70,243 parameters, pretrained on mid-price direction at horizon 10, frozen during PPO. **PPO**: from scratch, Gaussian policy, GAE (γ = 0.9, λ = 0.9), clip 0.1, lr 1e-5.

## Environment 1 — FI-2010

FI-2010's z-scored files do not preserve the bid-ask spread (ask1 − bid1 ≤ 0 in 57% of samples), so the execution layer uses a calibrated constant spread: half-spread 0.0005 (1 tick), max quoted spread 0.002, max bias 0.001. The mid is exactly flat in 75% of steps and moves in discrete ticks when it moves. Episodes avoid stock/day boundary jumps.

## Results — FI-2010 (100 held-out test episodes)

Encoder pretraining (8 epochs): test macro F1 0.278, per-class [0.0, 0.834, 0.0] — collapsed to predicting "flat", no usable directional signal.

| policy | mean PnL | Sharpe | fills/ep | win | mean \|inv\| |
|---|---|---|---|---|---|
| fixed (half 0.001) | −0.00072 ± 0.00037 | −0.20 | 1.97 | 0.19 | 0.36 |
| AS (γ=0.5, κ=1000) | −0.00103 ± 0.00039 | −0.26 | 2.01 | 0.17 | 0.31 |
| **PPO** | **−0.00121 ± 0.00037** | **−0.33** | **2.19** | **0.12** | **0.26** |
| random | −0.00219 ± 0.00047 | −0.46 | 2.40 | 0.07 | 0.20 |

PnL in z-units per 200-sample episode. PPO improved ~4–5× over the first run (−0.0051 → −0.0012) but does not beat the calibrated fixed-spread baseline, and no policy breaks even. In this replay env fills lose money by construction (adverse selection only — there is no real order flow paying the spread), the width optimum is a corner solution, and the encoder carries no directional signal. That is the result: RL cannot beat the calibrated baseline here.

## Environment 2 — real data (LOBSTER)

- **Data**: free LOBSTER samples, 2012-06-21, 10 levels: AAPL 400,391 events, INTC 624,040, MSFT 668,765 (message + orderbook files). AAPL is the training stock (median spread 15 ticks, level-1 queues ~100 shares, 34,990 executions, 8.7% of events). INTC/MSFT are big-tick (1-tick spread 70%+ of the time, level-1 queues ~12k shares) — coverage only, since a back-of-queue agent is essentially never filled there. Time split: first 75% of the day train (30,029 steps), last 25% test (10,010 steps, afternoon).
- **Fill model** (`real_data/real_env.py`): each step is 10 events (~0.6 s on AAPL). Quotes rest for the step; the step's executions are replayed in order against them with exact queue accounting. Fills are partial by shares (an 81-share execution fills 0.81 units of a 100-share order). Money in dollars, 1 unit = 100 shares, quotes rounded to the tick ($0.01).
- **Calibration**: the width optimum is interior on real data ($0.12–$0.15 half-spread on train; $0.01 loses −$262/ep, $0.20 falls to +$73/ep), so width is a real decision here — unlike FI-2010, where it was a corner.

## Results — real data, AAPL (100 held-out test episodes, afternoon)

Baselines (frozen from train, reported with standard errors):

| policy | mean PnL | Sharpe | fills/ep | win | mean \|inv\| |
|---|---|---|---|---|---|
| random | −$188.26 ± $8.13 | −2.32 | 64.5 | 0.01 | 0.51 |
| fixed ($0.12) | +$14.95 ± $6.67 | +0.22 | 8.1 | 0.51 | 1.22 |
| AS (γ=0.5, κ=5.0) | +$8.38 ± $2.07 | +0.41 | 1.6 | 0.32 | 0.26 |

Encoder pretraining on real data (8 epochs): test macro F1 0.202, per-class [0.0, 0.023, 0.583] — same conclusion as FI-2010, no directional signal.

Full run (100 iterations × 32 episodes):

| policy | mean PnL | Sharpe | fills/ep | win | mean \|inv\| | max DD |
|---|---|---|---|---|---|---|
| random | −$178.39 | −2.31 | 62.3 | 0.01 | 0.49 | $184.80 |
| fixed ($0.12) | +$14.95 ± $6.67 | +0.22 | 8.1 | 0.51 | 1.22 | $62.46 |
| AS (0.5, 5.0) | +$8.38 ± $2.07 | +0.41 | 1.6 | 0.32 | 0.26 | $15.96 |
| **PPO** | **+$19.11 ± $7.38** | **+0.26** | **12.2** | **0.58** | **1.39** | **$68.05** |

PPO leads on mean PnL and win rate — the first time it beats fixed on real data — but the +$4.17/ep edge over fixed is within noise (SE ~$7), not decisive, and AS is still best risk-adjusted (Sharpe +0.41, max drawdown $15.96 vs PPO $68.05, tightest inventory).

## Reproduce

```bash
pip install torch scikit-learn numpy pandas

# FI-2010 data is not included. Place the files next to this repo at
# ../tlob-midprice-prediction/data/FI_2010 (or pass data_dir to load_lob).
python pretrain.py --epochs 8 --out results/attn_lob_pretrained_8ep.pt
python train_agent.py --iterations 100 --episodes-per-iter 32 \
  --gamma 0.9 --lam 0.9 --eta 0 --zeta 0.00001 --min-half-spread 0.001 \
  --init-a1 0.05 --init-a2 0.9 --lr 1e-5 --clip 0.1 --epochs 2 \
  --encoder-ckpt results/attn_lob_pretrained_8ep.pt
python evaluate.py --n-episodes 100

# Real data: LOBSTER sample files go in real_data/raw/
# (free samples from lobsterdata.com, 2012-06-21, 10 levels).
cd real_data
python pretrain_real.py --epochs 8
python train_real.py --iterations 100 --episodes-per-iter 32 \
  --gamma 0.9 --lam 0.9 --eta 0 --zeta 0.1 --min-half-spread 0.01 \
  --init-a1 0.05 --init-a2 0.65 --lr 1e-5 --clip 0.1 --epochs 2
python evaluate_real.py --n-episodes 100
```

Full-run scripts: `run_full_v5.sh` (FI-2010), `real_data/run_full_real.sh` (real data). Tests:

```bash
python -m pytest test_env.py test_encoder.py real_data/test_real_env.py -q  # 13 + 4 + 12 tests
```

## Structure

- `encoder.py` — Attn-LOB (CNN + Inception + multi-head attention)
- `env.py` — FI-2010 resting-quote env
- `ppo.py` — PPO from scratch
- `pretrain.py` / `train_agent.py` / `evaluate.py` / `calibrate.py` — FI-2010 pipeline
- `lob_data.py` — FI-2010 loader
- `real_data/` — LOBSTER env (`real_env.py`, `lobster_data.py`, `train_real.py`, `pretrain_real.py`, `evaluate_real.py`, `calibrate_real.py`)
- `results/` / `real_data/results/` — run outputs

## Limitations

- FI-2010 spread is a calibrated constant, not the real book spread.
- The encoder is frozen during PPO, and it carries no directional signal in either environment (test F1 0.278 / 0.202, collapsed to flat).
- Real data is single-day free samples (2012-06-21) per stock; full LOBSTER access is a paid subscription.
- No latency modelling.

## Reference

- Guo, H. et al. *Market Making with Deep Reinforcement Learning from Limit Order Books.* arXiv:2305.15821, 2023.
- Avellaneda, M. & Stoikov, S. *High-frequency trading in a limit order book.* Quantitative Finance, 2008.

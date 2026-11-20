#!/bin/bash
# Full deep-RL market-making run, V5 (the matching configuration).
# Created 2026-11-20 ~00:45 ET. See README "Optimization" section for
# the full diagnosis. Summary of why V5 is the config:
#   - The optimal width in this replay env is a corner (max width);
#     per-step PPO cannot learn it (gradient probe: corr(a2, adv) =
#     +0.021, a2-gradient ~0), so the spread is PINNED at max width
#     (--min-half-spread 0.001, train AND eval).
#   - The skew (a1) gradient is noise-driven and pushes a1 up to
#     ~0.47 (V4), which makes one side ~1 tick from mid and doubles
#     fills; so a1 is warm-started near zero (0.05) and the lr is
#     small (1e-5) so it barely moves.
#   - Reward: eta 0 (DP = dPnL), zeta 0.00001, gamma/lam 0.9 (short
#     credit-assignment horizon), ent 0.005, init_std 0.3, clip 0.1,
#     2 PPO epochs.
#   - V5 validation (20x16, 3-epoch encoder): PPO -0.00090 +/- 0.00045
#     vs fixed -0.00069 +/- 0.00049 on the same 50 test episodes
#     (Sharpe -0.29 vs -0.20, fills 1.70 vs 1.52) -- a match on mean
#     PnL, the primary metric.
# Stages mirror run_full.sh; outputs are *_v5.* so earlier results
# are never overwritten. Resumable by stage. Logs go to
# results/run_full_v5.log AND the per-stage logs.
# Run with:  bash run_full_v5.sh   (default priority: fair sharing
# with the TLOB overnight job; do NOT nice -n 10 -- at nice 10 this
# box's scheduler starves the run to ~5% CPU).
set -u
cd "$(dirname "$0")"
PY=~/workspace/.tlob-venv/bin/python
LOG=results/run_full_v5.log
mkdir -p results

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -f results/attn_lob_pretrained_8ep.pt ]; then
  log "stage 1: pretrain Attn-LOB, 8 epochs"
  $PY pretrain.py --epochs 8 \
    --out results/attn_lob_pretrained_8ep.pt \
    --metrics results/pretrain_h10_8ep.json 2>&1 | tee -a "$LOG"
else
  log "stage 1: SKIP (results/attn_lob_pretrained_8ep.pt exists)"
fi

if [ ! -f results/calibration.txt ]; then
  log "stage 2: calibration (TRAIN)"
  $PY calibrate.py 2>&1 | tee results/calibration.txt
  tee -a "$LOG" < results/calibration.txt
else
  log "stage 2: SKIP (results/calibration.txt exists)"
fi

if [ ! -f results/ppo_agent_v5.pt ]; then
  log "stage 3: PPO training, 100 iters x 32 eps (V5 config)"
  $PY train_agent.py --iterations 100 --episodes-per-iter 32 \
    --gamma 0.9 --lam 0.9 --eta 0 --zeta 0.00001 \
    --min-half-spread 0.001 --ent-coef 0.005 --init-std 0.3 \
    --init-a1 0.05 --init-a2 0.9 --lr 1e-5 --clip 0.1 --epochs 2 \
    --encoder-ckpt results/attn_lob_pretrained_8ep.pt \
    --out results/ppo_agent_v5.pt \
    --log results/ppo_training_log_v5.json 2>&1 | tee -a "$LOG"
else
  log "stage 3: SKIP (results/ppo_agent_v5.pt exists)"
fi

log "stage 4: evaluation on 100 TEST episodes (floor 0.001)"
$PY evaluate.py --n-episodes 100 --min-half-spread 0.001 \
  --agent-ckpt results/ppo_agent_v5.pt \
  --encoder-ckpt results/attn_lob_pretrained_8ep.pt \
  --out results/evaluation_v5.json 2>&1 | tee -a "$LOG"

log "full run v5 done"

#!/bin/bash
# Full deep-RL market-making run, v2 (fixed reward + spread floor).
# Created 2026-11-19 evening after the diagnosis showed the smoke run's
# reward was misaligned with PnL (see README "Optimization" section):
#   - eta 0 (DP = dPnL, reward telescopes to PnL)
#   - zeta 0.00001 (IP recalibrated: ~20% of |DP+TP| for a wide-quoting
#     policy, not the random-policy calibration)
#   - min_half_spread 0.0005 (1-tick floor on quoted half-spread)
#   - ent_coef 0.03, init_std 0.4 (exploration kept alive longer)
# Stages mirror run_full.sh; outputs are *_v2.* so the smoke-run
# results are never overwritten. Resumable by stage.
# Run with:  bash run_full_v2.sh   (default priority: fair sharing
# with the TLOB overnight job; do NOT nice -n 10 — at nice 10 this
# box's scheduler starves the run to ~5% CPU).
# Logs go to results/run_full_v2.log.
set -u
cd "$(dirname "$0")"
PY=~/workspace/.tlob-venv/bin/python
LOG=results/run_full_v2.log
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

if [ ! -f results/ppo_agent_v2.pt ]; then
  log "stage 3: PPO training, 150 iters x 32 eps (V1 reward + floor)"
  $PY train_agent.py --iterations 150 --episodes-per-iter 32 \
    --eta 0 --zeta 0.00001 --ent-coef 0.03 --init-std 0.4 \
    --encoder-ckpt results/attn_lob_pretrained_8ep.pt \
    --out results/ppo_agent_v2.pt \
    --log results/ppo_training_log_v2.json 2>&1 | tee -a "$LOG"
else
  log "stage 3: SKIP (results/ppo_agent_v2.pt exists)"
fi

log "stage 4: evaluation on 100 TEST episodes"
$PY evaluate.py --n-episodes 100 \
  --agent-ckpt results/ppo_agent_v2.pt \
  --encoder-ckpt results/attn_lob_pretrained_8ep.pt \
  --out results/evaluation_v2.json 2>&1 | tee -a "$LOG"

log "full run v2 done"

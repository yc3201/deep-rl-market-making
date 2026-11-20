#!/bin/bash
# Full deep-RL market-making run (overnight scale, ~4 hours on this box).
# Stages:
#   1. Pretrain Attn-LOB, 8 epochs (the 3-epoch smoke encoder collapses
#      to predicting "stationary" — same undertraining regime as TLOB's
#      3-epoch runs; 8 epochs is the full recipe).
#   2. Calibration (TRAIN only) -> results/calibration.txt.
#   3. PPO training, 150 iterations x 32 episodes.
#   4. Evaluation on 100 TEST episodes: PPO vs AS vs fixed vs random.
# Resumable: stages whose output files already exist are skipped.
# Run with:  nice -n 10 bash run_full.sh
# Logs go to results/run_full.log. Do NOT run while the TLOB overnight
# training is still going if you can avoid it — the box shares 1.5 CPUs.
set -u
cd "$(dirname "$0")"
PY=~/workspace/.tlob-venv/bin/python
LOG=results/run_full.log
mkdir -p results

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -f results/attn_lob_pretrained_8ep.pt ]; then
  log "stage 1: pretrain Attn-LOB, 8 epochs"
  nice -n 10 $PY pretrain.py --epochs 8 \
    --out results/attn_lob_pretrained_8ep.pt \
    --metrics results/pretrain_h10_8ep.json 2>&1 | tee -a "$LOG"
else
  log "stage 1: SKIP (results/attn_lob_pretrained_8ep.pt exists)"
fi

if [ ! -f results/calibration.txt ]; then
  log "stage 2: calibration (TRAIN)"
  nice -n 10 $PY calibrate.py 2>&1 | tee results/calibration.txt
  tee -a "$LOG" < results/calibration.txt
else
  log "stage 2: SKIP (results/calibration.txt exists)"
fi

if [ ! -f results/ppo_agent_full.pt ]; then
  log "stage 3: PPO training, 150 iters x 32 eps"
  nice -n 10 $PY train_agent.py --iterations 150 --episodes-per-iter 32 \
    --encoder-ckpt results/attn_lob_pretrained_8ep.pt \
    --out results/ppo_agent_full.pt \
    --log results/ppo_training_log_full.json 2>&1 | tee -a "$LOG"
else
  log "stage 3: SKIP (results/ppo_agent_full.pt exists)"
fi

log "stage 4: evaluation on 100 TEST episodes"
nice -n 10 $PY evaluate.py --n-episodes 100 \
  --agent-ckpt results/ppo_agent_full.pt \
  --encoder-ckpt results/attn_lob_pretrained_8ep.pt \
  --out results/evaluation_full.json 2>&1 | tee -a "$LOG"

log "full run done"

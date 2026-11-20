#!/bin/bash
# Full real-data deep-RL market-making run (LOBSTER AAPL, 2012-06-21).
# Stages mirror run_full_v5.sh. Stage 0 WAITS for the TLOB overnight
# job (any `train.py --model` process) to finish, so the heavy training
# does not slow it down; the script is launched in the background and
# proceeds on its own once TLOB exits. Then:
#   1: pretrain Attn-LOB, 8 epochs (real AAPL labels)
#   2: PPO training, 100 iters x 32 eps (V5 recipe; init-a2 0.65 =
#      the calibrated real-data optimum: half-spread $0.13, the
#      train-calibrated fixed baseline is $0.12 -- the V5 warm start
#      sits AT the calibrated optimum, it does not assume the answer;
#      PPO must earn any edge over fixed by state-dependent quoting)
#   3: evaluation on 100 TEST episodes (afternoon, held out)
# Resumable by stage (skips stages whose outputs exist). Logs go to
# results/run_full_real.log. Run with:  nohup bash run_full_real.sh
#   > results/run_full_real.nohup 2>&1 &
# (default priority: fair sharing; do NOT nice -n 10 -- at nice 10
# this box's scheduler starves the run to ~5% CPU, per the V5 note).
set -u
cd "$(dirname "$0")"
PY=~/workspace/.tlob-venv/bin/python
LOG=results/run_full_real.log
mkdir -p results

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "stage 0: waiting for the TLOB overnight job to finish"
while pgrep -f "train.py --model" > /dev/null; do
  log "  TLOB train.py still running; sleeping 10 min"
  sleep 600
done
log "stage 0: TLOB job finished; starting real-data full run"

if [ ! -f results/attn_lob_real_pretrained_8ep.pt ]; then
  log "stage 1: pretrain Attn-LOB, 8 epochs"
  $PY pretrain_real.py --epochs 8 \
    --out results/attn_lob_real_pretrained_8ep.pt \
    --metrics results/pretrain_real_h10_8ep.json 2>&1 | tee -a "$LOG"
else
  log "stage 1: SKIP (results/attn_lob_real_pretrained_8ep.pt exists)"
fi

if [ ! -f results/ppo_agent_real.pt ]; then
  log "stage 2: PPO training, 100 iters x 32 eps (V5 recipe, init-a2 0.65)"
  $PY train_real.py --iterations 100 --episodes-per-iter 32 \
    --gamma 0.9 --lam 0.9 --eta 0 --zeta 0.1 \
    --min-half-spread 0.01 --ent-coef 0.005 --init-std 0.3 \
    --init-a1 0.05 --init-a2 0.65 --lr 1e-5 --clip 0.1 --epochs 2 \
    --encoder-ckpt results/attn_lob_real_pretrained_8ep.pt \
    --out results/ppo_agent_real.pt \
    --log results/ppo_training_log_real.json 2>&1 | tee -a "$LOG"
else
  log "stage 2: SKIP (results/ppo_agent_real.pt exists)"
fi

log "stage 3: evaluation on 100 TEST episodes"
$PY evaluate_real.py --n-episodes 100 \
  --agent-ckpt results/ppo_agent_real.pt \
  --encoder-ckpt results/attn_lob_real_pretrained_8ep.pt \
  --out results/evaluation_real.json 2>&1 | tee -a "$LOG"

log "full run real done"

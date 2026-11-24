#!/usr/bin/env bash
# Neutral-prompt Qwen3-VL captions on both GPUs in parallel (s11 --variant neutral).
#   GPU 0: LEVIR-CC test          GPU 1: LEVIR-CC val, then Dubai-CC val + test
# Each split is written to its own file (outputs/qwen3vlnt_<ds>/<split>/predictions.json), resumable.
V=D:/rsicc_v2
PYS="${PYS:-python}"
log() { echo "[$(date +%F' '%T)] $*"; }
log "start"
"$PYS" -u $V/scripts/s11_vlm_caption.py --variant neutral --datasets levir_cc --splits test --gpu 0 \
  > $V/logs/vlm_neutral_gpu0.log 2>&1 &
P0=$!
( "$PYS" -u $V/scripts/s11_vlm_caption.py --variant neutral --datasets levir_cc --splits val --gpu 1 &&
  "$PYS" -u $V/scripts/s11_vlm_caption.py --variant neutral --datasets dubai_cc --splits val test --gpu 1 ) \
  > $V/logs/vlm_neutral_gpu1.log 2>&1 &
P1=$!
wait $P0; log "gpu0 exit $?"
wait $P1; log "gpu1 exit $?"
log "VLM NEUTRAL DONE"

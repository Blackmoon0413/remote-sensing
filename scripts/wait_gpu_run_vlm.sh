#!/usr/bin/env bash
# Wait until a GPU has >= 10 GB free (the user's llama-server is not touched), then run the neutral-prompt
# Qwen3-VL captions (s11 --variant neutral) on that GPU. Results: outputs/qwen3vlnt_<levir|dubai>/<split>/.
V=D:/rsicc_v2
PYS="${PYS:-python}"
NEED=10000
log() { echo "[$(date +%F' '%T)] $*"; }
log "waiting for a GPU with >= ${NEED} MiB free"
while true; do
  while IFS=, read -r idx free; do
    free=${free// /}; free=${free//$'\r'/}   # nvidia-smi on Windows ends lines with CR
    if [ "${free%% *}" -ge "$NEED" ] 2>/dev/null; then GPU=$idx; break 2; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  sleep 30
done
log "GPU $GPU free, starting neutral-prompt run"
"$PYS" -u $V/scripts/s11_vlm_caption.py --variant neutral --datasets levir_cc dubai_cc --splits val test --gpu "$GPU"
log "s11 exit $?"
log "VLM NEUTRAL DONE"

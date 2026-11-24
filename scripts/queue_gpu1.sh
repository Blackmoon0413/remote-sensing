#!/usr/bin/env bash
# Overnight queue for GPU 1 (RTX 5070). Each step is independent: a failure is logged
# and the queue moves on.
V=D:/rsicc_v2
PYV=$V/.venv/Scripts/python.exe                                   # v2 venv (transformers 4.39)
PYS="${PYS:-python}"  # system (transformers 5.x)
CHAIN_OUT="$1"   # output file of the RSCaMa/PromptCC chain
DL_LOG=$V/logs/dl_qwen3vl.log
log() { echo "[$(date +%F' '%T)] $*"; }

until grep -q CHAIN_DONE "$CHAIN_OUT" 2>/dev/null; do sleep 30; done
log "chain done"
until grep -q -E "^DONE|Error|Traceback" "$DL_LOG" 2>/dev/null; do sleep 30; done
log "qwen download: $(grep -E '^DONE|Error' "$DL_LOG" | tail -1)"

if $PYS -u $V/scripts/s11_vlm_caption.py --datasets levir_cc --splits test --limit 3 --gpu 1 > $V/logs/vlm_smoke.log 2>&1; then
  log "vlm smoke ok: $(head -c 300 $V/outputs/qwen3vl_levir/test/predictions.json | tr '\n' ' ')"
  $PYS -u $V/scripts/s11_vlm_caption.py --datasets levir_cc dubai_cc --splits test val --gpu 1 > $V/logs/vlm_full.log 2>&1
  log "vlm full exit $?"
else
  log "vlm smoke FAILED (see logs/vlm_smoke.log), skipping VLM"
fi

$PYV -u $V/scripts/run_in.py $V/third_party/Change3D $V/third_party/Change3D/scripts/train_CC.py \
  --file_root $V/data/processed/c3d_dubai --dataset DUBAI_CC_5_cap_per_img_0_min_word_freq --epochs 200 \
  --batch_size 32 --micro_batch 8 --workers 0 --Split VAL --pretrained $V/weights/x3d/X3D_L.pyth \
  --save_dir $V/outputs/change3d_dubai --gpu_id 1 --print_freq 50 > $V/logs/train_c3d_dubai.log 2>&1
log "change3d dubai training exit $?"
log "QUEUE DONE"

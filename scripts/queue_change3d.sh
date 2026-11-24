#!/usr/bin/env bash
# After a Change3D training run ends: pick the best-val epoch, run inference, run the audits.
# Usage: queue_change3d.sh <levir|dubai> <gpu> <task_output_file_to_wait_for> <log> [<log> ...]
V=D:/rsicc_v2
PYV=$V/.venv/Scripts/python.exe
PYS="${PYS:-python}"
DS=$1; GPU=$2; WAIT=$3; shift 3
log() { echo "[$(date +%F' '%T)] $*"; }
until grep -q "exited with code" "$WAIT" 2>/dev/null; do sleep 60; done
log "training finished"
BEST=$($PYS $V/scripts/s13_best_epoch.py $V/outputs/change3d_$DS "$@")
CK=$(echo "$BEST" | $PYS -c "import json,sys; print(json.load(sys.stdin)['checkpoint'])")
log "best epoch: $(echo "$BEST" | $PYS -c "import json,sys; d=json.load(sys.stdin); print(d['best_epoch'], d['val_bleu4'])")"
echo "$BEST" > $V/outputs/change3d_$DS/best_epoch.json
$PYV -u $V/scripts/s10_infer_change3d.py --dataset $DS --checkpoint "$CK" --splits val test train --gpu $GPU > $V/logs/infer_c3d_$DS.log 2>&1
log "inference exit $?"
DSN=${DS}_cc
$PYV -u $V/scripts/run_in.py $V $V/scripts/s9_audit.py --captioners change3d --dataset $DSN \
  --encoders dinov2_b rclip_l14 rclip_b32 own --evidence lr knn11 knn51 > $V/logs/audit_change3d_$DS.log 2>&1
log "audit exit $?"
log "QUEUE DONE"

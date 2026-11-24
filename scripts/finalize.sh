#!/usr/bin/env bash
# Wait for both Change3D queues, then rebuild main results, tables, figures and the manuscript PDF.
V=D:/rsicc_v2
PYV=$V/.venv/Scripts/python.exe
log() { echo "[$(date +%F' '%T)] $*"; }
until grep -q "QUEUE DONE" $V/logs/queue_c3d_levir.log 2>/dev/null && grep -q "QUEUE DONE" $V/logs/queue_c3d_dubai.log 2>/dev/null; do sleep 60; done
log "both Change3D queues done"
$PYV -u $V/scripts/run_in.py $V $V/scripts/s14_main_results.py > $V/logs/final_s14.log 2>&1; log "s14 exit $?"
$PYV -u $V/scripts/run_in.py $V $V/scripts/s16_tables.py > $V/logs/final_s16.log 2>&1; log "s16 exit $?"
$PYV -u $V/scripts/run_in.py $V $V/scripts/s12_summary.py > $V/logs/final_s12.log 2>&1; log "s12 exit $?"
cp $V/outputs/figs/fig_flag_recovery.pdf $V/outputs/figs/fig_fp_fn.pdf $V/manuscript/figs/
cd $V/manuscript && pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
log "manuscript pages: $(pdfinfo main.pdf 2>/dev/null | grep Pages)"
log "FINAL DONE"

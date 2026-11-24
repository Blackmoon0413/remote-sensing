#!/usr/bin/env python3
"""S13: pick the Change3D epoch with the best VALIDATION BLEU-4 from training logs.

train_CC.py prints "Current Epoch: N" before each epoch and, after the epoch's
validation, several "['Bleu_1', ...] [b1, b2, b3, b4]" lines (no-change subset,
change subset, overall); the last one before the next epoch marker is the overall
val score. Works across a resumed run (pass both logs in order).

Usage: python s13_best_epoch.py <run_dir> <log> [<log> ...]
Prints JSON {best_epoch, val_bleu4, checkpoint, curve}.
"""
import json
import re
import sys
from pathlib import Path

run_dir, logs = Path(sys.argv[1]), sys.argv[2:]
curve = {}
for log in logs:
    epoch = None
    for line in Path(log).read_text(encoding="utf-8", errors="ignore").replace("\r", "\n").splitlines():
        m = re.match(r"Current Epoch: (\d+)", line)
        if m:
            epoch = int(m.group(1)); continue
        m = re.match(r"\['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4'\] \[([^\]]+)\]", line)
        if m and epoch is not None:
            curve[epoch] = float(m.group(1).split(",")[3])  # last one per epoch wins (= overall)
best = max(curve, key=curve.get)
ck = next(run_dir.glob(f"*/checkpoint_*_epoch_{best}.pth.tar"))
print(json.dumps({"best_epoch": best, "val_bleu4": round(curve[best] * 100, 2), "checkpoint": str(ck),
                  "n_epochs_logged": len(curve), "curve": {k: round(v * 100, 2) for k, v in sorted(curve.items())}}))

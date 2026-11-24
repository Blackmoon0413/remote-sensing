#!/usr/bin/env python3
"""S15: simple pixel-difference evidence (baseline for the foundation-model evidence).

Per pair (images resized to 256x256, RGB in [0,1]):
  raw:  mean |B-A|, std |B-A|, fraction of pixels with mean-channel |B-A| > 0.1 / 0.2 / 0.3
  norm: the same after standardising each image per channel (removes global colour /
        illumination shifts), thresholds 0.5 / 1.0 / 1.5
Saved as outputs/features/pixdiff/<ds>_<split>.npz with key "x" (and names, changeflag).
CPU only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2  # noqa: E402
from s7_retrieval_features import image_paths  # noqa: E402


def load(p):
    return np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC), dtype=np.float32) / 255.0


def feats(a, b):
    d = np.abs(b - a).mean(-1)
    za = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6)
    zb = (b - b.mean((0, 1))) / (b.std((0, 1)) + 1e-6)
    dz = np.abs(zb - za).mean(-1)
    return [d.mean(), d.std(), (d > 0.1).mean(), (d > 0.2).mean(), (d > 0.3).mean(),
            dz.mean(), dz.std(), (dz > 0.5).mean(), (dz > 1.0).mean(), (dz > 1.5).mean()]


def main() -> None:
    for ds in ("levir_cc", "dubai_cc"):
        refs = json.loads((V2 / "data" / "processed" / f"refs_{ds}.json").read_text(encoding="utf-8"))
        for split, rows in refs.items():
            out = V2 / "outputs" / "features" / "pixdiff" / f"{ds}_{split}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            x = np.array([feats(*(load(p) for p in image_paths(ds, r["name"], split))) for r in rows], dtype=np.float32)
            np.savez(out, names=np.array([r["name"] for r in rows]), changeflag=np.array([r["changeflag"] for r in rows]), x=x)
            print(f"[pixdiff] {ds} {split}: {x.shape}", flush=True)


if __name__ == "__main__":
    main()

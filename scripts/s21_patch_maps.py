#!/usr/bin/env python3
"""S21: patch-change maps of the frozen evidence encoder, for the figures.

For each pair, DINOv2-B patch tokens (same preprocessing as s7: 224x224, ImageNet normalisation, 16x16 patches)
of the pre- and post-change image are compared patch by patch: change = 1 - cos(t_pre, t_post).
This is the quantity summarised by the 'sorted patch-change profile' in Eq. (1) of the paper, shown at its
spatial position. All maps share one colour scale (single-hue sequential ramp, light = no change).
Writes manuscript/figs/maps/<name>.png (256x256) and outputs/cases/patch_maps.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2  # noqa: E402

spec = importlib.util.spec_from_file_location("s7", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s7_retrieval_features.py"))
s7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s7)

# light (no change) -> dark (strong change); one hue, matches the paper's orange evidence colour
RAMP = [(0.0, (250, 248, 244)), (0.35, (246, 196, 160)), (0.7, (226, 108, 52)), (1.0, (120, 38, 10))]


def colourise(v: np.ndarray, vmax: float) -> np.ndarray:
    t = np.clip(v / vmax, 0, 1)
    out = np.zeros(t.shape + (3,))
    for (a, ca), (b, cb) in zip(RAMP[:-1], RAMP[1:]):
        m = (t >= a) & (t <= b)
        w = ((t - a) / (b - a))[m][:, None]
        out[m] = (1 - w) * np.array(ca) + w * np.array(cb)
    return out.astype(np.uint8)


def main() -> None:
    cases = json.loads((V2 / "outputs" / "cases" / "cases.json").read_text(encoding="utf-8"))["cases"]
    names = ["test_000002"] + [c["name"] for c in cases]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = s7.Encoder("dinov2_b", dev)
    maps = {}
    for n in names:
        a, b = s7.image_paths("levir_cc", n, "test")
        xa = s7.load(a, enc.size, enc.norm)[None].to(dev); xb = s7.load(b, enc.size, enc.norm)[None].to(dev)
        _, ta = enc(xa); _, tb = enc(xb)
        ch = (1 - (ta * tb).sum(-1)).squeeze(0).cpu().numpy()          # (256,) patches
        side = int(round(len(ch) ** 0.5))
        maps[n] = ch.reshape(side, side)
    vmax = float(np.percentile(np.concatenate([m.ravel() for m in maps.values()]), 99))
    out = V2 / "manuscript" / "figs" / "maps"; out.mkdir(parents=True, exist_ok=True)
    for n, m in maps.items():
        # smooth upsampling of the 16x16 values (display only; the values themselves are unchanged)
        up = np.asarray(Image.fromarray(m.astype(np.float32), mode="F").resize((256, 256), Image.BICUBIC))
        Image.fromarray(colourise(up, vmax)).save(out / f"{n}.png")
    # colour bar for the figure legend
    bar = colourise(np.tile(np.linspace(0, vmax, 256), (12, 1)), vmax)
    Image.fromarray(bar).save(out / "colourbar.png")
    stats = {n: {"mean": float(m.mean()), "max": float(m.max()), "share_above_half_vmax": float((m > vmax / 2).mean())}
             for n, m in maps.items()}
    (V2 / "outputs" / "cases" / "patch_maps.json").write_text(json.dumps({"vmax": vmax, "stats": stats}, indent=1),
                                                              encoding="utf-8")
    for n, s in stats.items():
        print(f"{n}: mean {s['mean']:.3f} max {s['max']:.3f} share>{vmax / 2:.2f}: {100 * s['share_above_half_vmax']:.1f}%")
    print(f"vmax (99th pct) = {vmax:.3f}")


if __name__ == "__main__":
    main()

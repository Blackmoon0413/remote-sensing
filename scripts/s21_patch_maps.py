#!/usr/bin/env python3
"""S21: patch-change maps of the frozen evidence encoder, for the figures.

For each pair, DINOv2-B patch tokens (same preprocessing as s7: 224x224, ImageNet normalisation, 16x16 patches)
of the pre- and post-change image are compared patch by patch: change = 1 - cos(t_pre, t_post).
This is the quantity summarised by the 'sorted patch-change profile' in Eq. (1) of the paper, shown at its
spatial position. All maps share one colour scale (matplotlib 'magma': dark = no change, bright = strong change;
perceptually uniform with monotonic lightness, so it also reads in greyscale and for colour-blind readers).

Pairs: the framework example (test_000002), the case-study pairs (outputs/cases/cases.json, s18) and the gallery
pairs selected here by a fixed rule: LEVIR-CC test pairs are split into four groups by label and evidence decision
(unchanged / changed x p < 0.5 / p >= 0.5, main evidence configuration); in each group the two pairs whose
p(change) is closest to the group median are shown (pairs already used elsewhere are skipped).

Writes manuscript/figs/maps/<name>.png (256x256), maps/colourbar.png, outputs/cases/patch_maps.json (vmax, stats)
and outputs/cases/gallery.json.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2  # noqa: E402

spec = importlib.util.spec_from_file_location("s7", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s7_retrieval_features.py"))
s7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s7)

CMAP = colormaps["magma"]
PER_SAMPLE = V2 / "outputs" / "audit_per_sample" / "audit" / "qwen3vl_levir_cc_dinov2_b_lr.csv"  # p is captioner-independent
GROUPS = [("unchanged", False, "low"), ("unchanged", False, "high"), ("changed", True, "high"), ("changed", True, "low")]


def colourise(v: np.ndarray, vmax: float) -> np.ndarray:
    return (CMAP(np.clip(v / vmax, 0, 1))[..., :3] * 255).round().astype(np.uint8)


def select_gallery(exclude: set[str]) -> list[dict]:
    rows = [r for r in csv.DictReader(PER_SAMPLE.open(encoding="utf-8")) if r["split"] == "test"]
    out = []
    for label, changed, side in GROUPS:
        g_all = [r for r in rows if (r["changeflag"] == "1") == changed
                 and ((float(r["p_change"]) >= 0.5) == (side == "high"))]
        med = float(np.median([float(r["p_change"]) for r in g_all]))       # median of the whole group
        g = sorted((r for r in g_all if r["name"] not in exclude),          # pairs shown elsewhere are skipped
                   key=lambda r: (abs(float(r["p_change"]) - med), r["name"]))
        for r in g[:2]:
            out.append({"name": r["name"], "label": label, "evidence": "p >= 0.5" if side == "high" else "p < 0.5",
                        "p": float(r["p_change"]), "group_size": len(g_all), "group_median_p": med})
    return out


def main() -> None:
    cases = json.loads((V2 / "outputs" / "cases" / "cases.json").read_text(encoding="utf-8"))["cases"]
    fixed = ["test_000002"] + [c["name"] for c in cases]
    gallery = select_gallery(set(fixed))
    (V2 / "outputs" / "cases" / "gallery.json").write_text(json.dumps(gallery, indent=1), encoding="utf-8")
    names = fixed + [g["name"] for g in gallery]
    dev = torch.device("cpu")  # the GPUs may be busy; 20 pairs are quick on the CPU
    enc = s7.Encoder("dinov2_b", dev)
    maps = {}
    with torch.no_grad():
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
    bar = colourise(np.tile(np.linspace(0, vmax, 256), (12, 1)), vmax)
    Image.fromarray(bar).save(out / "colourbar.png")
    stats = {n: {"mean": float(m.mean()), "max": float(m.max()), "share_above_half_vmax": float((m > vmax / 2).mean())}
             for n, m in maps.items()}
    (V2 / "outputs" / "cases" / "patch_maps.json").write_text(json.dumps({"vmax": vmax, "cmap": "magma", "stats": stats}, indent=1),
                                                              encoding="utf-8")
    for g in gallery:
        s = stats[g["name"]]
        print(f"gallery {g['name']} {g['label']:9s} {g['evidence']:8s} p={g['p']:.4f} (group n={g['group_size']}, "
              f"median {g['group_median_p']:.4f}) mean change {s['mean']:.3f}")
    print(f"{len(maps)} maps; vmax (99th pct) = {vmax:.3f}")


if __name__ == "__main__":
    main()

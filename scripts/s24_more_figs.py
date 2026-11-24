#!/usr/bin/env python3
"""S24: three more figures, drawn at the text width of the MDPI template (13.85 cm), Palatino as in s22.

fig_gallery.pdf  LEVIR-CC test pairs and their patch-change maps for the four combinations of label and evidence
                 decision (two pairs per group, closest to the group median of p; selection in s21)
fig_cells.pdf    p(change) of the main evidence configuration split by what the captioner claims and what is true,
                 for the four captioners on LEVIR-CC test, with the correction and 5% review thresholds
fig_robust.pdf   heatmaps over all evidence configurations: (a) missed changes found at the 5% review budget,
                 (b) corrections made, coloured by their precision
All numbers come from outputs/audit, outputs/audit_per_sample (s23) and outputs/cases (s21).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("s22", os.path.join(HERE, "s22_extra_figs.py"))
s22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s22)   # shared style (rcParams, colours)
spec7 = importlib.util.spec_from_file_location("s7", os.path.join(HERE, "s7_retrieval_features.py"))
s7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(s7)

V2, FIGS, INK, INK2, MUTED, GRID = s22.V2, s22.FIGS, s22.INK, s22.INK2, s22.MUTED, s22.GRID
W = 13.85 / 2.54                       # text width of the MDPI template, inches
GOOD, BAD, MISS = "#1baf7a", "#e34948", "#7b4fc9"
NAME = {"rscama": "RSCaMa", "promptcc": "PromptCC", "change3d": "Change3D", "qwen3vl": "Qwen3-VL-4B"}


def per_sample(cap, ds="levir_cc", enc="dinov2_b", kind="lr", folder="audit"):
    f = V2 / "outputs" / "audit_per_sample" / folder / f"{cap}_{ds}_{enc}_{kind}.csv"
    rows = [r for r in csv.DictReader(f.open(encoding="utf-8")) if r["split"] == "test"]
    return {k: np.array([float(r[k]) for r in rows]) for k in ("changeflag", "claims_change", "p_change")}


def audit(cap, ds="levir_cc", enc="dinov2_b", kind="lr"):
    f = V2 / "outputs" / "audit" / f"{cap}_{ds}_{enc}_{kind}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def fmt_p(p):
    return "$p<0.001$" if p < 0.001 else ("$p>0.999$" if p > 0.999 else f"$p={p:.3f}$")


# ----------------------------------------------------------------------------------------------- gallery
def gallery():
    gal = json.loads((V2 / "outputs" / "cases" / "gallery.json").read_text(encoding="utf-8"))
    vmax = json.loads((V2 / "outputs" / "cases" / "patch_maps.json").read_text(encoding="utf-8"))["vmax"]
    left, gap_in, gap_out, top, head, plab, cbar_h = 0.30, 0.035, 0.13, 0.03, 0.36, 0.17, 0.30
    tile = (W - left - 4 * gap_in - 3 * gap_out - 0.02) / 8
    H = top + head + 3 * tile + 2 * gap_in + plab + cbar_h
    fig = plt.figure(figsize=(W, H))
    heads = [("Unchanged", "evidence right", GOOD), ("Unchanged", "evidence wrong", BAD),
             ("Changed", "evidence right", GOOD), ("Changed", "evidence wrong", BAD)]
    rows = ["Pre-change", "Post-change", "Patch change"]
    x = left
    for gi in range(4):
        g = gal[2 * gi: 2 * gi + 2]
        x0 = x
        for j, item in enumerate(g):
            a, b = s7.image_paths("levir_cc", item["name"], "test")
            ims = [Image.open(a).convert("RGB"), Image.open(b).convert("RGB"),
                   Image.open(V2 / "manuscript" / "figs" / "maps" / f"{item['name']}.png").convert("RGB")]
            for ri, im in enumerate(ims):
                yb = H - top - head - (ri + 1) * tile - ri * gap_in
                ax = fig.add_axes([x / W, yb / H, tile / W, tile / H])
                ax.imshow(np.asarray(im), interpolation="lanczos")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color("white" if ri < 2 else INK2); s.set_linewidth(0.4)
                if gi == 0 and j == 0:
                    ax.set_ylabel(rows[ri], fontsize=7.2, labelpad=3, color=INK)
            yb = H - top - head - 3 * tile - 2 * gap_in
            fig.text((x + tile / 2) / W, (yb - 0.03) / H, fmt_p(item["p"]), ha="center", va="top", fontsize=6.8, color=INK)
            x += tile + (gap_in if j == 0 else 0)
        x1 = x
        label, verdict, col = heads[gi]
        p_rule = "$p\\geq0.5$" if gi in (1, 2) else "$p<0.5$"
        n = g[0]["group_size"]
        yh = H - top - head
        bar = FancyBboxPatch((x0 / W, (yh + 0.04) / H), (x1 - x0) / W, (head - 0.07) / H, transform=fig.transFigure,
                             boxstyle="round,pad=0.0,rounding_size=0.012", facecolor=col, alpha=0.13, edgecolor=col,
                             linewidth=0.7)
        fig.add_artist(bar)
        fig.text(((x0 + x1) / 2) / W, (yh + head - 0.09) / H, f"{label}, {p_rule}", ha="center", va="center",
                 fontsize=7.4, fontweight="bold", color=INK)
        fig.text(((x0 + x1) / 2) / W, (yh + 0.13) / H, f"{verdict} ({n} pairs)", ha="center", va="center",
                 fontsize=6.6, color=col)
        x = x1 + gap_out
    # colour bar
    cax = fig.add_axes([(W * 0.30) / W, 0.10 / H, (W * 0.40) / W, 0.07 / H])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=colormaps["magma"])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1)
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
    fig.text((W * 0.29) / W, 0.135 / H, "patch change", ha="right", va="center", fontsize=6.8, color=INK2)
    fig.text((W * 0.71) / W, 0.135 / H, "$1-\\cos$ of DINOv2 patch tokens", ha="left", va="center", fontsize=6.8, color=INK2)
    fig.savefig(FIGS / "fig_gallery.pdf", bbox_inches=None); plt.close(fig)
    print("[gallery]", ", ".join(f"{g['name']} {g['label']} p={g['p']:.4f}" for g in gal))


# ----------------------------------------------------------------------------------------------- cells
LO = 1e-5


def logit(p):
    p = np.clip(p, LO, 1 - LO)
    return np.log(p / (1 - p))


def cells():
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.55), gridspec_kw={"hspace": 0.58, "wspace": 0.12}, sharey=True)
    cats = [("Hallucinated\nchange", 0, 1, BAD), ("Correct\nchange", 1, 1, "#8fa9c8"),
            ("Correct\nno change", 0, 0, "#b9b7b1"), ("Missed\nchange", 1, 0, MISS)]
    ticks = [LO, 1e-3, 0.1, 0.5, 0.9, 1 - 1e-3, 1 - LO]
    tlab = ["$10^{-5}$", "$10^{-3}$", "0.1", "0.5", "0.9", "$1-10^{-3}$", "$1-10^{-5}$"]
    rng = np.random.default_rng(0)
    stats = {}
    for ax, cap, tag in zip(axes.ravel(), ["rscama", "promptcc", "change3d", "qwen3vl"], "abcd"):
        d, a = per_sample(cap), audit(cap)
        y, c, p = d["changeflag"].astype(int), d["claims_change"].astype(int), d["p_change"]
        for k, (lab, yy, cc, col) in enumerate(cats):
            v = logit(p[(y == yy) & (c == cc)])
            n = len(v)
            stats[(cap, lab.replace("\n", " "))] = (n, float(np.median(p[(y == yy) & (c == cc)])) if n else None)
            if n >= 3:
                vp = ax.violinplot(v, positions=[k], widths=0.82, showextrema=False, bw_method=0.25)
                for b in vp["bodies"]:
                    b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.28); b.set_linewidth(0.6)
            if n:
                jit = rng.uniform(-0.2, 0.2, n) * (0.6 if n < 30 else 1)
                ax.scatter(k + jit, v, s=2.2 if n > 100 else 5, color=col, alpha=0.45 if n > 100 else 0.85,
                           linewidths=0, zorder=2, rasterized=True)
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.add_patch(Rectangle((k - 0.07, q1), 0.14, max(q3 - q1, 0.08), facecolor="white", edgecolor=INK,
                                       linewidth=0.7, zorder=3))
                ax.plot([k - 0.07, k + 0.07], [med, med], color=INK, lw=1.1, zorder=4, solid_capstyle="butt")
            ax.text(k, logit(1 - LO) + 1.15, f"$n={n}$", ha="center", va="bottom", fontsize=6.4, color=INK2)
        t_c = a["correction"]["threshold_from_val"]
        t_f = a["flags"]["5%"]["threshold_from_val"]
        ax.axhline(logit(t_f), color="#eb6834", lw=0.9, ls=(0, (1.2, 1.2)), zorder=1)
        ax.text(3.62, logit(t_f) + 0.35, "$t_f$", color="#eb6834", fontsize=7, ha="right", va="bottom")
        if t_c is not None:
            ax.axhline(logit(t_c), color=GOOD, lw=0.9, ls=(0, (4, 2)), zorder=1)
            ax.text(-0.62, logit(t_c) + 0.35, "$t_c$", color=GOOD, fontsize=7, ha="left", va="bottom")
        cor, f5 = a["correction"], a["flags"]["5%"]
        lines = (f"corrections {cor['test_acts']} ({cor['test_acts'] - cor['test_acts_correct']} wrong)" if cor["test_acts"]
                 else "no corrections (abstains)")
        lines += f"; flags {f5['n_flagged']} ({f5['missed_found']} missed changes)"
        ax.text(0, 1.012, lines, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, color=INK2)
        ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cats], fontsize=6.6, linespacing=0.95)
        for tl, cc in zip(ax.get_xticklabels(), cats):
            tl.set_color(cc[3] if cc[3] in (BAD, MISS) else INK2)
        ax.set_yticks([logit(t) for t in ticks]); ax.set_yticklabels(tlab, fontsize=6.6)
        ax.set_ylim(logit(LO) - 0.6, logit(1 - LO) + 3.4); ax.set_xlim(-0.65, 3.65)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.tick_params(axis="x", length=0, pad=2)
        ax.set_title(f"$\\bf{{({tag})}}$ {NAME[cap]}", loc="left", pad=13)
        if tag in "ac":
            ax.set_ylabel("evidence $p(\\mathrm{change})$")
    fig.savefig(FIGS / "fig_cells.pdf", dpi=600); plt.close(fig)
    for k, (n, med) in stats.items():
        print(f"[cells] {k[0]:9s} {k[1]:22s} n={n:4d} median p={med if med is None else round(med, 5)}")


# ----------------------------------------------------------------------------------------------- robustness heatmaps
FEATS = [("dinov2_b", "DINOv2-B"), ("rclip_l14", "RemoteCLIP-L/14"), ("rclip_b32", "RemoteCLIP-B/32"),
         ("pixdiff", "Pixel difference"), ("own", "Own features")]
KINDS = [("lr", "LP"), ("knn11", "kNN-11"), ("knn51", "kNN-51")]


def heat(ax, M, T, cols, cmap, vmin, vmax, star=None):
    masked = np.ma.masked_invalid(M)
    cm = colormaps[cmap].copy(); cm.set_bad("#f3f2ef")
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            t = T[i][j]
            if t is None:
                continue
            v = M[i, j]
            dark = (not np.isnan(v)) and (v - vmin) / (vmax - vmin) > 0.55
            ax.text(j, i, t, ha="center", va="center", fontsize=6.1, color="white" if dark else INK)
    for g in range(1, len(FEATS)):
        ax.axhline(g * 3 - 0.5, color="white", lw=2.2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.6, rotation=38, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    if star is not None:
        ax.add_patch(Rectangle((star[1] - 0.5, star[0] - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.0))
    return im


def robust():
    cols_a = [("rscama", "levir_cc"), ("promptcc", "levir_cc"), ("change3d", "levir_cc"), ("qwen3vl", "levir_cc"),
              ("qwen3vl", "dubai_cc")]
    cols_b = cols_a[:4] + [("change3d", "dubai_cc"), ("qwen3vl", "dubai_cc")]
    lab = lambda cap, ds: NAME[cap].replace("Qwen3-VL-4B", "Qwen3-VL")  # noqa: E731
    nrow = len(FEATS) * len(KINDS)
    A = np.full((nrow, len(cols_a)), np.nan); TA = [[None] * len(cols_a) for _ in range(nrow)]
    B = np.full((nrow, len(cols_b)), np.nan); TB = [[None] * len(cols_b) for _ in range(nrow)]
    for fi, (enc, _) in enumerate(FEATS):
        for ki, (kind, _) in enumerate(KINDS):
            i = fi * 3 + ki
            for j, (cap, ds) in enumerate(cols_a):
                r = audit(cap, ds, f"own-{cap}" if enc == "own" else enc, kind)
                if r:
                    f5 = r["flags"]["5%"]
                    A[i, j] = 100 * f5["missed_found"] / max(f5["missed_total"], 1)
                    TA[i][j] = f"{A[i, j]:.0f}"
            for j, (cap, ds) in enumerate(cols_b):
                r = audit(cap, ds, f"own-{cap}" if enc == "own" else enc, kind)
                if r:
                    n, ok = r["correction"]["test_acts"], r["correction"]["test_acts_correct"]
                    if n:
                        B[i, j] = 100 * ok / n; TB[i][j] = str(n)
                    else:
                        TB[i][j] = "–"
    fig = plt.figure(figsize=(W, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(cols_a), len(cols_b)], wspace=0.06, left=0.2, right=0.95,
                          top=0.74, bottom=0.12)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    im1 = heat(ax1, A, TA, [lab(*c) for c in cols_a], "YlGnBu", 0, 100, star=None)
    im2 = heat(ax2, B, TB, [lab(*c) for c in cols_b], "YlOrRd", 60, 100)
    ax1.set_yticks(range(nrow)); ax1.set_yticklabels([k[1] for _ in FEATS for k in KINDS], fontsize=6.4)
    ax2.set_yticks([])
    for fi, (_, fname) in enumerate(FEATS):
        ax1.text(-0.205, 1 - (fi * 3 + 1.5) / nrow, fname, transform=ax1.transAxes, ha="right", va="center",
                 fontsize=6.9, fontweight="bold" if fi == 0 else "normal", color=INK)
    for ax in (ax1, ax2):  # main configuration
        ax.add_patch(Rectangle((-0.5, -0.5), ax.get_xlim()[1] + 0.5, 1, fill=False, edgecolor=INK, lw=0.9, ls=(0, (2, 1))))
    ax1.set_title("$\\bf{(a)}$ Missed changes found (%)", fontsize=7.6, loc="left", pad=58)
    ax2.set_title("$\\bf{(b)}$ Corrections made", fontsize=7.6, loc="left", pad=58)
    for ax, groups in ((ax1, [("LEVIR-CC", 0, 3), ("Dubai", 4, 4)]), (ax2, [("LEVIR-CC", 0, 3), ("Dubai-CC", 4, 5)])):
        for name, j0, j1 in groups:
            ax.annotate("", xy=(j0 - 0.4, 1.235), xytext=(j1 + 0.4, 1.235), xycoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), annotation_clip=False)
            ax.text((j0 + j1) / 2, 1.25, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6.6, color=INK2)
    for ax, im, ticks, lab_ in ((ax1, im1, [0, 25, 50, 75, 100], "% of missed changes"),
                                (ax2, im2, [60, 70, 80, 90, 100], "correction precision (%)")):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + pos.width * 0.15, 0.055, pos.width * 0.7, 0.022])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6.2, length=2, width=0.4, pad=1)
        cb.set_label(lab_, fontsize=6.5, labelpad=1.5)
    fig.savefig(FIGS / "fig_robust.pdf"); plt.close(fig)
    print("[robust] flags recall range per column:",
          [f"{np.nanmin(A[:, j]):.0f}-{np.nanmax(A[:, j]):.0f}" for j in range(A.shape[1])])


if __name__ == "__main__":
    which = sys.argv[1:] or ["gallery", "cells", "robust"]
    for w in which:
        globals()[w]()

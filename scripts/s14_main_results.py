#!/usr/bin/env python3
"""S14: pre-specified main configuration, main table and figures.

1. Evidence configuration for the main results = the (encoder, evidence) pair with the
   highest VALIDATION ROC-AUC for the official change label on LEVIR-CC. This choice is
   captioner-independent; all other settings are reported as robustness (summary.md).
2. Main table (LEVIR-CC test) for every captioner with predictions.
3. Figure: missed changes recovered vs. review budget (test), per captioner, vs random.
4. Figure: hallucinated (FP) and missed (FN) changes before/after the audit.
Outputs: outputs/main_results.json, outputs/figs/fig_flag_recovery.pdf, fig_fp_fn.pdf
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change, norm  # noqa: E402

spec = importlib.util.spec_from_file_location("s9", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s9_audit.py"))
s9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)

DS = "levir_cc"
ENCODERS = ["dinov2_b", "rclip_l14", "rclip_b32"]
KINDS = ["lr", "knn11", "knn51"]
CAPTIONERS = [("rscama", "RSCaMa"), ("promptcc", "PromptCC"), ("change3d", "Change3D (retrained)"), ("qwen3vl", "Qwen3-VL-4B (zero-shot)")]
COLORS = {"rscama": "#2a78d6", "promptcc": "#eb6834", "change3d": "#1baf7a", "qwen3vl": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
BASE_GREY = "#c3c2b7"
# Figures are drawn at their final printed size (no scaling in LaTeX), in Palatino like the MDPI text and the
# TikZ figures (mathpazo).
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "savefig.bbox": "tight",
                     "mathtext.fontset": "custom", "mathtext.rm": "Palatino Linotype",
                     "mathtext.it": "Palatino Linotype:italic", "mathtext.bf": "Palatino Linotype:bold",
                     "savefig.pad_inches": 0.02})


def auc(score, y):
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    y = y.astype(bool)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main() -> None:
    refs = json.loads((V2 / "data" / "processed" / f"refs_{DS}.json").read_text(encoding="utf-8"))
    sel = {}
    for enc in ENCODERS:
        for kind in KINDS:
            out = {}
            for split in ("val", "test"):
                names = [r["name"] for r in refs[split]]; y = np.array([r["changeflag"] for r in refs[split]])
                out[split] = round(auc(s9.evidence(kind, enc, DS, refs, split, names), y) * 100, 2)
            sel[f"{enc}/{kind}"] = out
    main_cfg = max(sel, key=lambda k: sel[k]["val"])
    enc, kind = main_cfg.split("/")
    print("[select] evidence AUC (val/test):", json.dumps(sel))
    print(f"[select] main configuration = {main_cfg} (val AUC {sel[main_cfg]['val']})")

    test = refs["test"]; names = [r["name"] for r in test]; y = np.array([r["changeflag"] == 1 for r in test])
    p = s9.evidence(kind, enc, DS, refs, "test", names)
    rows, curves = [], {}
    for cap, label in CAPTIONERS:
        f = V2 / "outputs" / "audit" / f"{cap}_{DS}_{enc}_{kind}.json"
        pf = V2 / "outputs" / f"{cap}_levir" / "test" / "predictions.json"
        if not (f.exists() and pf.exists()):
            print(f"[skip] {cap}"); continue
        r = json.loads(f.read_text(encoding="utf-8"))
        from scipy.stats import binomtest
        for bud, fr in r["flags"].items():  # flag precision vs random review among no-change claims
            fr["binom_p_vs_random"] = float(binomtest(fr["missed_found"], max(fr["n_flagged"], 1), fr["random_precision"] / 100,
                                                      alternative="greater").pvalue) if fr["n_flagged"] else None
        rows.append({"captioner": label, **{k: r[k] for k in ("correction", "test_scores", "binary", "flags")}})
        # qualitative examples for the paper
        pred_q = json.loads(pf.read_text(encoding="utf-8")); pq = dict(zip(names, p))
        cq = {n: asserts_change(norm(pred_q[n])) for n in names}; yq = dict(zip(names, y))
        thr = r["correction"]["threshold_from_val"]
        fixed = [n for n in names if thr is not None and cq[n] and pq[n] <= thr and not yq[n]][:4]
        broke = [n for n in names if thr is not None and cq[n] and pq[n] <= thr and yq[n]][:3]
        flagged = sorted([n for n in names if not cq[n]], key=lambda n: -pq[n])[:5]
        refmap = {rr["name"]: rr["refs"][0] for rr in test}
        rows[-1]["examples"] = {k: [{"name": n, "caption": pred_q[n], "ref": refmap[n], "p": round(float(pq[n]), 3),
                                     "changed": bool(yq[n])} for n in v] for k, v in
                                (("corrected_ok", fixed), ("corrected_wrong", broke), ("top_flags", flagged))}
        pred = json.loads(pf.read_text(encoding="utf-8"))
        c = np.array([asserts_change(norm(pred[n])) for n in names])
        missed = (~c) & y
        order = np.argsort(-p[~c]); hits = missed[~c][order].cumsum()
        if missed.sum() >= 10:  # same rule as the flag table: too few misses for a recovery curve
            curves[cap] = (np.arange(1, (~c).sum() + 1) / len(names) * 100, hits / max(missed.sum(), 1) * 100,
                           missed.sum() / max((~c).sum(), 1))
    (V2 / "outputs" / "main_results.json").write_text(json.dumps({"main_config": main_cfg, "evidence_auc": sel,
                                                                  "rows": rows}, indent=1), encoding="utf-8")
    figs = V2 / "outputs" / "figs"; figs.mkdir(parents=True, exist_ok=True)
    # ---- recovery curves
    XMAX = 20
    fig, ax = plt.subplots(figsize=(4.9, 2.7))
    ends = []
    for cap, (x, yv, base) in curves.items():
        lab = dict(CAPTIONERS)[cap]
        ax.plot(x, yv, color=COLORS[cap], lw=1.8, label=lab, solid_capstyle="round", zorder=3)
        # random review among the captioner's no-change claims recovers misses linearly
        ax.plot(x, x / x[-1] * 100, color=COLORS[cap], lw=1.0, ls=(0, (1.5, 2)), zorder=2)
        ends.append([float(np.interp(XMAX, x, yv)), lab.split(" (")[0], COLORS[cap]])
    # direct labels at the right end, spread so that they never collide (min gap 7 points of the y scale)
    ends.sort(key=lambda e: e[0])
    true_mean = float(np.mean([e[0] for e in ends]))
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 6.5)
    shift = true_mean - float(np.mean([e[0] for e in ends]))  # re-centre the label block on the curve ends
    for e in ends:
        e[0] = min(e[0] + shift, 98)
    true_end = {lab.split(" (")[0]: float(np.interp(XMAX, x, yv)) for cap, (x, yv, _) in curves.items()
                for lab in [dict(CAPTIONERS)[cap]]}
    for y_lab, name, col in ends:
        y0 = true_end[name]
        ax.plot([XMAX], [y0], marker="o", ms=3.2, color=col, clip_on=False, zorder=4)
        ax.annotate(name, xy=(XMAX, y0), xytext=(XMAX + 0.9, y_lab), textcoords="data", va="center",
                    fontsize=7.5, color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=2, shrinkB=1))
    for budget, txt in ((5, "5% budget"), (10, "10%")):
        ax.axvline(budget, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(budget + 0.25, 3, txt, fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 100); ax.set_xticks(range(0, XMAX + 1, 5))
    ax.set_xlabel("Pairs sent to review (% of test pairs)"); ax.set_ylabel("Missed changes recovered (%)")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.plot([], [], color=INK2, lw=1.8, label="ranked by evidence score")
    ax.plot([], [], color=INK2, lw=1.0, ls=(0, (1.5, 2)), label="random review")
    h, l = ax.get_legend_handles_labels()
    order = [i for i, s in enumerate(l) if s.startswith(("ranked", "random"))] + \
            [i for i, s in enumerate(l) if not s.startswith(("ranked", "random"))]
    # legend below the axes so that no curve is hidden (the VLM curve rises steeply at the left)
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=3, frameon=False, handlelength=2.2, columnspacing=1.4, labelspacing=0.35)
    fig.savefig(figs / "fig_flag_recovery.pdf"); fig.savefig(figs / "fig_flag_recovery.png", dpi=300)
    plt.close(fig)
    # ---- FP / FN bars
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"wspace": 0.2})
    src = {"RSCaMa": "released", "PromptCC": "released", "Change3D": "retrained", "Qwen3-VL-4B": "zero-shot"}
    labs = [f"{n}\n{src.get(n, '')}" for n in (r["captioner"].split(" (")[0] for r in rows)]
    for ax, key, tag, title in ((axes[0], "FP", "a", "Hallucinated changes (FP)"), (axes[1], "FN", "b", "Missed changes (FN)")):
        b = [r["binary"]["base"][key] for r in rows]; a = [r["binary"]["audited"][key] for r in rows]
        if key == "FN":  # flagged-and-confirmed misses at the 5% budget count as recovered
            a = [r["binary"]["audited"]["FN"] - r["flags"]["5%"]["missed_found"] for r in rows]
        xx = np.arange(len(rows)); w = 0.36
        ax.bar(xx - w / 2 - 0.01, b, w, color=BASE_GREY, label="Captioner alone", zorder=2)
        ax.bar(xx + w / 2 + 0.01, a, w, color=COLORS["rscama"],
               label="After correction" + (" and 5% review" if key == "FN" else ""), zorder=2)
        top = max(b + a)
        for i, (u, v) in enumerate(zip(b, a)):
            ax.text(i - w / 2 - 0.01, u + top * 0.015, str(u), ha="center", va="bottom", fontsize=7, color=INK2)
            ax.text(i + w / 2 + 0.01, v + top * 0.015, str(v), ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(xx, labs, fontsize=7, linespacing=1.25)
        for t in ax.get_xticklabels():
            t.set_color(INK)
        ax.tick_params(axis="x", length=0, pad=4); ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_title(f"$\\bf{{({tag})}}$ {title}", loc="left", pad=6)
        ax.set_ylim(0, top * 1.38); ax.set_ylabel("Test pairs")
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper left", handlelength=1.2, handleheight=0.9)
    fig.savefig(figs / "fig_fp_fn.pdf"); fig.savefig(figs / "fig_fp_fn.png", dpi=300)
    plt.close(fig)
    for r in rows:
        b0, b1, f5 = r["binary"]["base"], r["binary"]["audited"], r["flags"]["5%"]
        print(f"{r['captioner']:26s} FP {b0['FP']:4d}->{b1['FP']:4d}  FN {b0['FN']:4d}->{b1['FN']:4d}  "
              f"CIDEr {r['test_scores']['base']['CIDEr']:7.2f}->{r['test_scores']['audited']['CIDEr']:7.2f}  "
              f"acts {r['correction']['test_acts']}({r['correction']['test_acts_correct']})  "
              f"5%: {f5['missed_found']}/{f5['missed_total']} prec {f5['precision']}% rand {f5['random_precision']}%")


if __name__ == "__main__":
    main()

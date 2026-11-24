#!/usr/bin/env python3
"""S22: additional result figures (LEVIR-CC test unless stated; main evidence configuration DINOv2-B + LP).

fig_error_profile.pdf  hallucination rate vs miss rate per captioner and dataset (+ evidence model alone)
fig_evidence.pdf       (a) ROC curves of the evidence features, (b) p(change) for unchanged vs changed pairs
fig_tradeoff.pdf       correction trade-off: hallucinations removed vs real changes silenced as t_c varies,
                       for the zero-shot VLM (a) and RSCaMa (b), with the operating points of Table 5
All numbers come from the prediction / audit files; drawn at printed size in Palatino (like s14).
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

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
COL = {"rscama": "#2a78d6", "promptcc": "#eb6834", "change3d": "#1baf7a", "qwen3vl": "#eda100", "evidence": "#52514e"}
NAME = {"rscama": "RSCaMa", "promptcc": "PromptCC", "change3d": "Change3D", "qwen3vl": "Qwen3-VL-4B"}
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     "mathtext.fontset": "custom", "mathtext.rm": "Palatino Linotype",
                     "mathtext.it": "Palatino Linotype:italic", "mathtext.bf": "Palatino Linotype:bold"})
FIGS = V2 / "outputs" / "figs"
ENC, KIND = "dinov2_b", "lr"


def load(ds):
    refs = json.loads((V2 / "data" / "processed" / f"refs_{ds}.json").read_text(encoding="utf-8"))
    names = [r["name"] for r in refs["test"]]
    y = np.array([r["changeflag"] == 1 for r in refs["test"]])
    return refs, names, y


def claims(cap, ds, names):
    f = V2 / "outputs" / f"{cap}_{ds.split('_')[0]}" / "test" / "predictions.json"
    if not f.exists():
        return None
    pred = json.loads(f.read_text(encoding="utf-8"))
    return np.array([asserts_change(norm(pred[n])) for n in names])


def title(ax, tag, text):
    ax.set_title(f"$\\bf{{({tag})}}$ {text}", loc="left", pad=6)


def error_profile():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    pts = []
    for ds, marker, filled in (("levir_cc", "o", True), ("dubai_cc", "s", False)):
        refs, names, y = load(ds)
        for cap in ("rscama", "promptcc", "change3d", "qwen3vl"):
            c = claims(cap, ds, names)
            if c is None:
                continue
            fp = 100 * (c & ~y).sum() / (~y).sum(); fn = 100 * (~c & y).sum() / y.sum()
            pts.append((cap, ds, fp, fn))
            ax.scatter([fp], [fn], s=46, marker=marker, color=COL[cap] if filled else "white",
                       edgecolor=COL[cap], linewidth=1.6, zorder=3)
        if ds == "levir_cc":
            p = s9.evidence(KIND, ENC, ds, refs, "test", names); ev = p >= 0.5
            fp = 100 * (ev & ~y).sum() / (~y).sum(); fn = 100 * (~ev & y).sum() / y.sum()
            pts.append(("evidence", ds, fp, fn))
            ax.scatter([fp], [fn], s=52, marker="D", color=COL["evidence"], zorder=3)
    lim = (0.2, 100)
    ax.plot(lim, lim, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(30, 42, "equal error rates", rotation=38, color=INK2, fontsize=7, ha="center", va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(lim[0], 160)
    ticks = [0.3, 1, 3, 10, 30, 100]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks]); ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Hallucinated changes (% of unchanged pairs)"); ax.set_ylabel("Missed changes (% of changed pairs)")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    offs = {("rscama", "levir_cc"): (-6, 6, "right"), ("promptcc", "levir_cc"): (6, 6, "left"),
            ("change3d", "levir_cc"): (6, -9, "left"), ("qwen3vl", "levir_cc"): (0, -13, "center"),
            ("change3d", "dubai_cc"): (0, 8, "center"), ("qwen3vl", "dubai_cc"): (8, -3, "left"),
            ("evidence", "levir_cc"): (6, 4, "left")}
    for cap, ds, fp, fn in pts:
        dx, dy, ha = offs[(cap, ds)]
        lab = "evidence alone" if cap == "evidence" else NAME[cap] + (" (Dubai-CC)" if ds == "dubai_cc" else "")
        ax.annotate(lab, (fp, fn), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7.5, color=INK)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=INK2, markersize=6, label="LEVIR-CC"),
         Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=INK2, markersize=6,
                markeredgewidth=1.4, label="Dubai-CC"),
         Line2D([], [], marker="D", ls="", color=COL["evidence"], markersize=5.5, label="evidence model alone")]
    ax.legend(handles=h, loc="lower left", frameon=False)
    fig.savefig(FIGS / "fig_error_profile.pdf"); plt.close(fig)
    return pts


def roc(score, y):
    o = np.argsort(-score); yy = y[o]
    tpr = np.concatenate([[0], np.cumsum(yy) / yy.sum()]); fpr = np.concatenate([[0], np.cumsum(~yy) / (~yy).sum()])
    return fpr, tpr


def evidence_fig():
    refs, names, y = load("levir_cc")
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.45, 2.6), gridspec_kw={"wspace": 0.32})
    series = [("dinov2_b", "DINOv2-B", "#2a78d6"), ("rclip_l14", "RemoteCLIP-L/14", "#1baf7a"),
              ("own:rscama", "RSCaMa's own features", "#eb6834"), ("pixdiff", "pixel difference", "#52514e")]
    aucs = {}
    for enc, lab, col in series:
        p = s9.evidence("lr", enc, "levir_cc", refs, "test", names)
        fpr, tpr = roc(p, y); auc = float(np.trapz(tpr, fpr)); aucs[enc] = auc
        a.plot(100 * fpr, 100 * tpr, color=col, lw=1.6, label=f"{lab} ({100 * auc:.1f})",
               ls="-" if enc != "pixdiff" else (0, (3, 1.5)))
    a.plot([0, 100], [0, 100], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    a.set_xlim(0, 100); a.set_ylim(0, 100); a.set_aspect("equal")
    a.set_xlabel("False positive rate (%)"); a.set_ylabel("True positive rate (%)")
    a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
    a.legend(loc="lower right", frameon=False, title="linear probe (test AUC)", title_fontsize=7.5, handlelength=1.8)
    title(a, "a", "Evidence quality (ROC)")
    p = s9.evidence(KIND, ENC, "levir_cc", refs, "test", names)
    bins = np.linspace(0, 1, 41)
    # explicit baseline below the log-axis floor: no fill path down to log(0) = -inf in the PDF
    for mask, col, lab in ((~y, "#2a78d6", f"unchanged pairs ({(~y).sum()})"), (y, "#eb6834", f"changed pairs ({y.sum()})")):
        cnt, _ = np.histogram(p[mask], bins=bins)
        cnt = np.where(cnt > 0, cnt, 0.7)
        b.stairs(cnt, bins, baseline=0.7, fill=True, color=col, alpha=0.22, zorder=2)
        b.stairs(cnt, bins, baseline=0.7, color=col, lw=1.4, label=lab, zorder=3)
    b.set_yscale("log"); b.set_xlim(0, 1); b.set_ylim(0.7, 8000)
    vlm = json.loads((V2 / "outputs" / "audit" / f"qwen3vl_levir_cc_{ENC}_{KIND}.json").read_text(encoding="utf-8"))
    rsc = json.loads((V2 / "outputs" / "audit" / f"rscama_levir_cc_{ENC}_{KIND}.json").read_text(encoding="utf-8"))
    t_c = vlm["correction"]["threshold_from_val"]; t_f = rsc["flags"]["5%"]["threshold_from_val"]
    for t, lab, col, ha, dx in ((t_c, "$t_c$ (VLM)", "#1baf7a", "left", 0.025),
                                (t_f, "$t_f$ (RSCaMa, 5%)", INK2, "right", -0.025)):
        b.axvline(t, color=col, lw=1.1, ls=(0, (3, 2)), zorder=4)
        b.text(t + dx, 150, lab, color=col, fontsize=6.8, va="top", ha=ha, zorder=5,
               bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))
    b.set_xlabel("Evidence score $p$(change)"); b.set_ylabel("Test pairs (log scale)")
    b.grid(axis="y", color=GRID, lw=0.5); b.set_axisbelow(True)
    b.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.44, 1.02), fontsize=7, handlelength=1.6)
    title(b, "b", "Evidence score by true state")
    fig.savefig(FIGS / "fig_evidence.pdf"); plt.close(fig)
    return aucs


def tradeoff_fig():
    refs, names, y = load("levir_cc")
    p = s9.evidence(KIND, ENC, "levir_cc", refs, "test", names)
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 2.7), gridspec_kw={"wspace": 0.34})
    out = {}
    for ax, cap, tag in ((axes[0], "qwen3vl", "a"), (axes[1], "rscama", "b")):
        c = claims(cap, "levir_cc", names)
        a = json.loads((V2 / "outputs" / "audit" / f"{cap}_levir_cc_{ENC}_{KIND}.json").read_text(encoding="utf-8"))
        idx = np.flatnonzero(c); order = idx[np.argsort(p[idx])]
        removed = np.concatenate([[0], np.cumsum(~y[order])]); silenced = np.concatenate([[0], np.cumsum(y[order])])
        ax.plot(silenced, removed, color=COL[cap], lw=1.8, zorder=2)
        hall = int((c & ~y).sum())
        xmax = 400 if cap == "qwen3vl" else 150
        ax.axhline(hall, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.text(xmax * 0.98, hall, f"all {hall} hallucinated changes", ha="right", va="bottom", fontsize=7, color=INK2)
        ca = a["correction"]; cc = a["correction_cider_opt"]
        fire = c & (p < 0.5)
        ops = [("ours ($\\geq$90% precision)", ca["test_acts"] - ca["test_acts_correct"], ca["test_acts_correct"], "*", 120),
               ("evidence decides ($p<0.5$)", int((fire & y).sum()), int((fire & ~y).sum()), "o", 40),
               ("CIDEr-optimal", cc["test_acts"] - cc["test_acts_correct"], cc["test_acts_correct"], "s", 34)]
        out[cap] = ops
        for lab, xs, ys, mk, sz in ops:
            ax.scatter([xs], [ys], marker=mk, s=sz, color=INK if mk == "*" else "white",
                       edgecolor=INK, linewidth=1.2, zorder=4, clip_on=False)
        if cap == "qwen3vl":  # labels placed in empty space below the curve, joined to their points by leader lines
            spots = {"ours": (0.28, 0.40), "evidence": (0.42, 0.62), "CIDEr": (0.56, 0.80)}
            for lab, xs, ys, mk, sz in ops:
                fx, fy = spots[lab.split()[0].split("-")[0]]
                ax.annotate(lab, (xs, ys), xytext=(fx * xmax, fy * hall), textcoords="data", fontsize=7, color=INK,
                            va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=5))
        else:  # points crowd at the origin: label them with leader lines
            place = {"ours": ((62, 4.2), "ours: no correction\n(abstains)"),
                     "evidence": None, "CIDEr": ((62, 1.7), "CIDEr-optimal:\n3 corrections, all wrong")}
            for lab, xs, ys, mk, sz in ops:
                key = lab.split()[0].split("-")[0]
                if key == "evidence":
                    ax.annotate(lab, (xs, ys), xytext=(7, -11), textcoords="offset points", fontsize=7, color=INK)
                    continue
                (tx, ty), text = place[key]
                ax.annotate(text, (xs, ys), xytext=(tx, ty), textcoords="data", fontsize=7, color=INK, va="center",
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=4))
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, hall * 1.12)
        ax.set_xlabel("Real changes silenced")
        ax.set_ylabel("Hallucinated changes removed")
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        title(ax, tag, f"{NAME[cap]}")
    fig.savefig(FIGS / "fig_tradeoff.pdf"); plt.close(fig)
    return out


def cost_fig():
    """Expected cost of each correction rule as a function of r = cost of a missed change / cost of a false alert.

    cost = FP + r * FN on the LEVIR-CC test split, shown relative to the captioner alone (1 = no change in cost).
    The dotted envelope is the best single threshold chosen on the TEST split for each r (an oracle reference only).
    """
    refs, names, y = load("levir_cc")
    p = s9.evidence(KIND, ENC, "levir_cc", refs, "test", names)
    r = np.logspace(-1, 2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 2.8), gridspec_kw={"wspace": 0.3})
    out = {}
    for ax, cap, tag in ((axes[0], "qwen3vl", "a"), (axes[1], "rscama", "b")):
        c = claims(cap, "levir_cc", names)
        a = json.loads((V2 / "outputs" / "audit" / f"{cap}_levir_cc_{ENC}_{KIND}.json").read_text(encoding="utf-8"))
        fp0, fn0 = int((c & ~y).sum()), int((~c & y).sum())
        fire = c & (p < 0.5)
        rules = {"captioner alone": (fp0, fn0),
                 "ours ($\\geq$90% precision)": (a["binary"]["audited"]["FP"], a["binary"]["audited"]["FN"]),
                 "evidence decides ($p<0.5$)": (int((c & ~fire & ~y).sum()), int(((~c | fire) & y).sum())),
                 "CIDEr-optimal": (a["correction_cider_opt"]["binary"]["FP"], a["correction_cider_opt"]["binary"]["FN"])}
        base = fp0 + r * fn0
        # oracle envelope: every threshold over the captioner's change claims (incl. no correction)
        idx = np.flatnonzero(c); order = idx[np.argsort(p[idx])]
        rem = np.concatenate([[0], np.cumsum(~y[order])]); sil = np.concatenate([[0], np.cumsum(y[order])])
        env = np.min((fp0 - rem)[None, :] + r[:, None] * (fn0 + sil)[None, :], axis=1)
        ax.plot(r, env / base, color=MUTED, lw=1.0, ls=(0, (1, 1.5)), zorder=1)
        style = {"captioner alone": (INK2, 1.0, (0, (4, 2))), "ours ($\\geq$90% precision)": ("#1baf7a", 2.0, "-"),
                 "evidence decides ($p<0.5$)": ("#2a78d6", 1.5, "-"), "CIDEr-optimal": ("#eb6834", 1.5, "-")}
        curves = {}
        for lab, (fp, fn) in rules.items():
            col, lw, ls = style[lab]
            curves[lab] = (fp + r * fn) / base
            ax.plot(r, curves[lab], color=col, lw=lw, ls=ls, label=lab, zorder=3 if "ours" in lab else 2)
        # where is our rule the cheapest of the four?
        stack = np.vstack(list(curves.values()))
        ours_c = curves["ours ($\\geq$90% precision)"]
        win = r[ours_c <= stack.min(axis=0) + 1e-9]   # ties count (for RSCaMa, ours = captioner alone)
        if len(win):
            ax.axvspan(win.min(), win.max(), color="#1baf7a", alpha=0.10, lw=0, zorder=0)
            hi = "" if win.max() >= r.max() * 0.999 else f" < {win.max():.0f}"
            lo = f"{win.min():.1f}" if win.min() >= 1 else f"{win.min():.2f}"
            ax.text(np.sqrt(win.min() * win.max()), 0.04, f"ours cheapest\n({lo} < $r${hi})",
                    ha="center", va="bottom", fontsize=7, color="#147a56", transform=ax.get_xaxis_transform())
        out[cap] = {"rules": rules, "ours_cheapest": [float(win.min()), float(win.max())] if len(win) else None}
        ax.set_xscale("log"); ax.set_xlim(0.1, 100)
        ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100], ["0.1", "0.3", "1", "3", "10", "30", "100"]); ax.minorticks_off()
        ax.set_ylim(0, 2.0 if cap == "qwen3vl" else 1.6)
        ax.set_xlabel("Cost of a missed change / cost of a false alert ($r$)")
        ax.set_ylabel("Cost relative to captioner alone")
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        title(ax, tag, f"{NAME[cap]}")
    axes[1].plot([], [], color=MUTED, lw=1.0, ls=(0, (1, 1.5)), label="best threshold on test (oracle)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, handlelength=2.2,
               columnspacing=1.2)
    fig.savefig(FIGS / "fig_cost.pdf"); plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    cost = cost_fig(); print("[cost]", json.dumps(cost))
    pts = error_profile(); aucs = evidence_fig(); ops = tradeoff_fig()
    print("[error profile]", [(c, d, round(f, 2), round(m, 2)) for c, d, f, m in pts])
    print("[ROC AUC]", {k: round(100 * v, 2) for k, v in aucs.items()})
    print("[operating points]", ops)


if __name__ == "__main__":
    main()

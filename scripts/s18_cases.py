#!/usr/bin/env python3
"""S18: case studies for the paper (LEVIR-CC test split, main evidence configuration).

Six outcomes, three per action, each with a pre-registered selection rule so that no example is hand-picked:
  correction (zero-shot VLM)       A1 hallucination corrected      claim=change, unchanged, p <= t_c
                                   A2 real change wrongly corrected claim=change, changed,   p <= t_c
                                   A3 hallucination not corrected   claim=change, unchanged, p >  t_c
  flagging (RSCaMa, 5% budget)     B1 missed change recovered       claim=no change, changed,   flagged
                                   B2 false alarm                   claim=no change, unchanged, flagged
                                   B3 missed change not flagged     claim=no change, changed,   not flagged
Rule: within each outcome, the pair whose p(change) is closest to the outcome's median p (a typical case);
test_000002 (used in Figure 1) is excluded. Ties are broken by name.

Writes outputs/cases/cases.json, copies the image pairs to manuscript/figs/cases/ and writes
manuscript/figs/fig_cases.tex (standalone; compile with pdflatex in manuscript/figs).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change, norm  # noqa: E402

spec = importlib.util.spec_from_file_location("s9", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s9_audit.py"))
s9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)

DS, ENC, KIND = "levir_cc", "dinov2_b", "lr"
EXCLUDE = {"test_000002"}
OUTCOMES = [
    ("A1", "qwen3vl", "Correct", "hallucinated change corrected", True),
    ("A2", "qwen3vl", "Correct", "real change wrongly corrected", False),
    ("A3", "qwen3vl", "Keep", "hallucinated change not corrected", False),
    ("B1", "rscama", "Flag", "missed change recovered by review", True),
    ("B2", "rscama", "Flag", "false alarm (pair is unchanged)", False),
    ("B3", "rscama", "Keep", "missed change not flagged", False),
]


def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def main() -> None:
    refs = json.loads((V2 / "data" / "processed" / f"refs_{DS}.json").read_text(encoding="utf-8"))
    test = refs["test"]; names = [r["name"] for r in test]
    y = np.array([r["changeflag"] == 1 for r in test])
    p = s9.evidence(KIND, ENC, DS, refs, "test", names)
    aud = {c: json.loads((V2 / "outputs" / "audit" / f"{c}_{DS}_{ENC}_{KIND}.json").read_text(encoding="utf-8"))
           for c in ("qwen3vl", "rscama")}
    t_c = aud["qwen3vl"]["correction"]["threshold_from_val"]
    f5 = aud["rscama"]["flags"]["5%"]; t_f = f5["threshold_from_val"]
    assert f5.get("tie_flag_prob", 1.0) == 1.0, "linear-probe flags should have no ties"
    pred = {c: json.loads((V2 / "outputs" / f"{c}_levir" / "test" / "predictions.json").read_text(encoding="utf-8"))
            for c in ("qwen3vl", "rscama")}
    claim = {c: np.array([asserts_change(norm(pred[c][n])) for n in names]) for c in pred}
    groups = {
        "A1": claim["qwen3vl"] & ~y & (p <= t_c), "A2": claim["qwen3vl"] & y & (p <= t_c),
        "A3": claim["qwen3vl"] & ~y & (p > t_c),
        "B1": ~claim["rscama"] & y & (p >= t_f), "B2": ~claim["rscama"] & ~y & (p >= t_f),
        "B3": ~claim["rscama"] & y & (p < t_f),
    }
    # consistency with the audit files
    assert groups["A1"].sum() + groups["A2"].sum() == aud["qwen3vl"]["correction"]["test_acts"]
    assert groups["B1"].sum() == f5["missed_found"] and groups["B1"].sum() + groups["B2"].sum() == f5["n_flagged"]
    cases = []
    for key, cap, action, outcome, ok in OUTCOMES:
        idx = [i for i in np.flatnonzero(groups[key]) if names[i] not in EXCLUDE]
        med = float(np.median(p[idx]))
        order = sorted(idx, key=lambda i: (abs(p[i] - med), names[i]))
        i = order[0]
        r = test[i]
        cases.append({"key": key, "captioner": cap, "action": action, "outcome": outcome, "correct": ok,
                      "name": names[i], "changed": bool(y[i]), "p": float(p[i]), "group_size": int(groups[key].sum()),
                      "group_median_p": med, "caption": pred[cap][names[i]], "refs": r["refs"],
                      "alternates": [names[j] for j in order[1:4]]})
    out = V2 / "outputs" / "cases"; out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps({"t_c": t_c, "t_f_5pct": t_f, "cases": cases}, indent=1), encoding="utf-8")
    figdir = V2 / "manuscript" / "figs" / "cases"; figdir.mkdir(parents=True, exist_ok=True)
    for c in cases:
        for side in ("A", "B"):
            shutil.copy(V2 / "data" / "levir_cc" / "images" / "test" / side / f"{c['name']}.png", figdir / f"{c['name']}_{side}.png")
    for c in cases:
        print(f"{c['key']} {c['name']} changed={c['changed']} p={c['p']:.3f} (median {c['group_median_p']:.3f}, n={c['group_size']}) "
              f"| {c['captioner']}: {c['caption'][:90]} | ref: {c['refs'][0]}")
    print(f"t_c={t_c} t_f(5%)={t_f}")
    write_figure(cases, t_c, t_f)


def fmt_p(v: float) -> str:
    return "$<$0.001" if v < 0.001 else f"{v:.3f}"


# audit line per outcome: (what the audit did, what that means)
AUDIT = {
    "A1": ("{p} $\\le t_c$ $\\rightarrow$ corrected", "hallucination removed"),
    "A2": ("{p} $\\le t_c$ $\\rightarrow$ corrected", "real change silenced"),
    "A3": ("{p} $> t_c$ $\\rightarrow$ kept", "hallucination kept"),
    "B1": ("{p} $\\ge t_f$ $\\rightarrow$ flagged", "missed change found"),
    "B2": ("{p} $\\ge t_f$ $\\rightarrow$ flagged", "false alarm"),
    "B3": ("{p} $< t_f$ $\\rightarrow$ not flagged", "change stays hidden"),
}


def p_expr(v: float) -> str:
    return "$p < 0.001$" if v < 0.001 else f"$p = {v:.3f}$"

FIG_TEMPLATE = r"""% Figure: case studies (generated by scripts/s18_cases.py; do not edit by hand).
% Selection: within each outcome, the pair whose p(change) is closest to the outcome median (test_000002 excluded).
% Patch-change maps: scripts/s21_patch_maps.py (DINOv2 patch tokens, 1 - cos(pre, post), one colour scale).
\documentclass[border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathpazo}
\usepackage{amsmath,graphicx,xcolor,pifont,array,arydshln}
\setlength{\dashlinedash}{3pt}\setlength{\dashlinegap}{2pt}\setlength{\arrayrulewidth}{0.6pt}
\definecolor{ink}{HTML}{1F1E1C}
\definecolor{ink2}{HTML}{52514E}
\definecolor{refc}{HTML}{C0392B}
\definecolor{capc}{HTML}{2A78D6}
\definecolor{good}{HTML}{1BAF7A}
\definecolor{bad}{HTML}{E34948}
\definecolor{panelA}{HTML}{EEF4FC}
\definecolor{panelB}{HTML}{FDF1E8}
\newcommand{\img}[1]{\includegraphics[width=2.3cm]{#1}}
\newcommand{\colhead}{\makebox[2.3cm]{\textbf{Pre-change}}\hspace{0.12cm}\makebox[2.3cm]{\textbf{Post-change}}\hspace{0.12cm}\makebox[2.3cm]{\textbf{Patch change}}}
\newcommand{\barrow}{\makebox[2.3cm]{}\hspace{0.12cm}\makebox[2.3cm]{}\hspace{0.12cm}\makebox[2.3cm]{\scriptsize\color{ink2}low\,\raisebox{0.2ex}{\includegraphics[width=1.25cm,height=0.11cm]{maps/colourbar.png}}\,high}}
\newcommand{\good}[1]{{\color{good}\bfseries #1}}
\newcommand{\bad}[1]{{\color{bad}\bfseries #1}}
\newcommand{\grouphead}[2]{\multicolumn{2}{@{}l@{}}{\colorbox{#1}{\makebox[13.55cm][l]{#2}}}}
\begin{document}
\color{ink}\footnotesize
% text width of the journal template (13.85 cm): one case per row, image triple left, text right
\begin{tabular}{@{}m{7.18cm}@{\hspace{0.3cm}}>{\raggedright\arraybackslash}m{6.3cm}@{}}
\grouphead{panelA}{\small\bfseries A.\ Correction of Qwen3-VL-4B\enspace\normalfont\footnotesize($t_c = TC$)} \\[2pt]
\colhead & \textbf{Reference, caption and audit decision} \\[-1pt]
\barrow & \\[1pt]
ROWS_A
\grouphead{panelB}{\small\bfseries B.\ Review of RSCaMa\enspace\normalfont\footnotesize(5\% budget, $t_f = TF$)} \\[2pt]
ROWS_B
\end{tabular}
\end{document}
"""


def case_block(c: dict) -> str:
    did, means = AUDIT[c["key"]]
    mark = (r"\good{\ding{51}\ " if c["correct"] else r"\bad{\ding{55}\ ") + means + "}"
    truth = "changed" if c["changed"] else "unchanged"
    # one table row: image triple & text (\par inside the cell, because \\ would end the table row)
    return (
        "\\img{cases/" + c["name"] + "_A.png}\\hspace{0.12cm}\\img{cases/" + c["name"] + "_B.png}\\hspace{0.12cm}"
        "\\img{maps/" + c["name"] + ".png}\n&\n"
        "\\textbf{" + c["key"] + "}\\enspace"
        "{\\color{refc}\\textbf{Reference} (" + truth + "):} " + tex_escape(c["refs"][0]) + "\\par\n"
        "{\\color{capc}\\textbf{Caption:}} " + tex_escape(c["caption"]) + "\\par\n"
        "\\textbf{Audit:} " + did.replace("{p}", p_expr(c["p"])) + "\\par " + mark)


def write_figure(cases, t_c, t_f) -> None:
    rows_a = "".join(case_block(a) + "\n\\\\[5pt]\n" for a in cases[:3])
    rows_b = "".join(case_block(b) + "\n\\\\[5pt]\n" for b in cases[3:])
    rows_b = rows_b[: -len("[5pt]\n")] + "\n"
    tex = (FIG_TEMPLATE.replace("ROWS_A", rows_a).replace("ROWS_B", rows_b)
           .replace("TC", f"{t_c:.4f}").replace("TF", f"{t_f:.3f}"))
    (V2 / "manuscript" / "figs" / "fig_cases.tex").write_text(tex, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

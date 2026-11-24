#!/usr/bin/env python3
"""S16: LaTeX tables for the manuscript, generated from result files only (no hand-copying).

Writes v2/manuscript/tables/*.tex :
  tab_captioners.tex   caption quality (unified protocol) + own-code and published numbers
  tab_errors.tex       change-state error profile (FP / FN) per captioner and dataset
  tab_evidence.tex     evidence quality: val/test ROC-AUC of every evidence configuration (LEVIR-CC)
  tab_correction.tex   correction results, main configuration (+ CIDEr-optimal rule for contrast)
  tab_flags.tex        flagging results at 5% / 10% review budgets, main configuration
  tab_robust.tex       all evidence configurations (appendix), pivoted by evidence model
and v2/manuscript/numbers.json with every number quoted in the text.
Rows are grouped (\\midrule between datasets / feature families; repeated names left blank).
The manuscript defines \\alt{} (grey italic) for the CIDEr-optimal contrast rows.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change, binary, score  # noqa: E402

spec = importlib.util.spec_from_file_location("s9", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s9_audit.py"))
s9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)

OUT = V2 / "manuscript" / "tables"
CAPS = [("rscama", "RSCaMa~\\cite{liu2024rscama}", "released"),
        ("promptcc", "PromptCC~\\cite{liu2023promptcc}", "released"),
        ("change3d", "Change3D~\\cite{zhu2025change3d}", "retrained"),
        ("qwen3vl", "Qwen3-VL-4B~\\cite{qwen3vl}", "zero-shot")]
SHORT = {k: v.split("~")[0] for k, v, _ in CAPS}
PUBLISHED = {"rscama": {"B4": 65.24, "CIDEr": 136.56}, "promptcc": {"B4": 63.54, "CIDEr": 136.44},
             "change3d": {"B4": 64.38, "CIDEr": 138.29}}
DATASETS = [("levir_cc", "LEVIR-CC"), ("dubai_cc", "Dubai-CC")]
M7 = ("B1", "B2", "B3", "B4", "METEOR", "ROUGE_L", "CIDEr")
M_CAP = ("B4", "METEOR", "ROUGE_L", "CIDEr")  # caption-quality table (text width); BLEU-1..3 stay in the result files
ENCS = ["dinov2_b", "rclip_l14", "rclip_b32", "pixdiff", "own"]
ENC_NAMES = {"dinov2_b": "DINOv2-B", "rclip_l14": "RemoteCLIP-L/14", "rclip_b32": "RemoteCLIP-B/32",
             "pixdiff": "Pixel difference", "own": "Captioner's own features"}
KINDS = ["lr", "knn11", "knn51"]
KIND_NAMES = {"lr": "linear probe", "knn11": "$k$NN ($K{=}11$)", "knn51": "$k$NN ($K{=}51$)", "knn3": "$k$NN ($K{=}3$)"}
SPACER = "\\addlinespace[3pt]"


def pred_path(cap, ds, split="test"):
    return V2 / "outputs" / f"{cap}_{ds.split('_')[0]}" / split / "predictions.json"


def auc(score_, y):
    o = np.argsort(score_); r = np.empty(len(score_)); r[o] = np.arange(1, len(score_) + 1); y = y.astype(bool)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def f2(x):
    return f"{x:.2f}"


def arrow(a, b):
    return f"{a} $\\to$ {b}"


def grouped(groups):
    """groups: list of lists of row strings -> rows joined, \\midrule between non-empty groups."""
    return "\n\\midrule\n".join("\n".join(g) for g in groups if g) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nums = {}
    main_cfg = json.loads((V2 / "outputs" / "main_results.json").read_text(encoding="utf-8"))["main_config"]
    m_enc, m_kind = main_cfg.split("/")
    nums["main_config"] = main_cfg

    # ---------- captioner quality + error profile
    cap_groups, err_groups = [], []
    for ds, dsn in DATASETS:
        refs = json.loads((V2 / "data" / "processed" / f"refs_{ds}.json").read_text(encoding="utf-8"))["test"]
        R = [r["refs"] for r in refs]; y = np.array([r["changeflag"] == 1 for r in refs])
        cg, eg, data = [], [], []
        for cap, label, src in CAPS:
            pf = pred_path(cap, ds)
            if not pf.exists():
                continue
            pred = json.loads(pf.read_text(encoding="utf-8")); caps = [pred[r["name"]] for r in refs]
            s = score(R, caps); b = binary(np.array([asserts_change(c) for c in caps]), y)
            nums[f"{cap}|{ds}|scores"] = s; nums[f"{cap}|{ds}|binary"] = b
            pub = PUBLISHED.get(cap) if ds == "levir_cc" else None
            own_f = V2 / "outputs" / f"{cap}_{ds.split('_')[0]}" / "test" / "scores_repo.json"
            own = json.loads(own_f.read_text(encoding="utf-8")) if (own_f.exists() and ds == "levir_cc") else None
            data.append((label, src, s, b, own, pub))
        # best value per column within the dataset is set in bold (as in comparison tables of the field)
        best = {m: max(round(d[2][m], 2) for d in data) for m in M_CAP}
        fp_r = [100 * d[3]["FP"] / max(d[3]["FP"] + d[3]["TN"], 1) for d in data]
        fn_r = [100 * d[3]["FN"] / max(d[3]["FN"] + d[3]["TP"], 1) for d in data]
        acc = [d[3]["acc"] for d in data]
        bold = lambda txt, on: f"\\textbf{{{txt}}}" if on else txt  # noqa: E731
        for i, (label, src, s, b, own, pub) in enumerate(data):
            cg.append(f"{'' if cg else dsn} & {label} & {src} & " +
                      " & ".join(bold(f2(s[m]), round(s[m], 2) == best[m]) for m in M_CAP) +
                      (f" & {own['Bleu_4']:.2f} / {own['CIDEr']:.2f}" if own else " & --") +
                      (f" & {pub['B4']:.2f} / {pub['CIDEr']:.2f}" if pub else " & --") + r" \\")
            eg.append(f"{'' if eg else dsn} & {label} & "
                      + bold(f"{b['FP']} / {b['FP'] + b['TN']} ({fp_r[i]:.1f}\\%)", fp_r[i] == min(fp_r)) + " & "
                      + bold(f"{b['FN']} / {b['FN'] + b['TP']} ({fn_r[i]:.1f}\\%)", fn_r[i] == min(fn_r)) + " & "
                      + bold(f"{b['acc']:.2f}", acc[i] == max(acc)) + " \\\\")
        cap_groups.append(cg); err_groups.append(eg)
    (OUT / "tab_captioners.tex").write_text(grouped(cap_groups), encoding="utf-8")
    (OUT / "tab_errors.tex").write_text(grouped(err_groups), encoding="utf-8")

    # ---------- evidence AUC (LEVIR-CC)
    refs = json.loads((V2 / "data" / "processed" / "refs_levir_cc.json").read_text(encoding="utf-8"))
    ev_groups = []
    for enc in ENCS:
        g = []
        for kind in KINDS:
            e = "own:rscama" if enc == "own" else enc
            try:
                a = {}
                for split in ("val", "test"):
                    names = [r["name"] for r in refs[split]]; yy = np.array([r["changeflag"] for r in refs[split]])
                    a[split] = auc(s9.evidence(kind, e, "levir_cc", refs, split, names), yy) * 100
            except FileNotFoundError:
                continue
            nums[f"auc|{enc}|{kind}"] = a
            star = "$^\\star$" if f"{enc}/{kind}" == main_cfg else ""
            name = "" if g else ENC_NAMES[enc] + (" (RSCaMa)" if enc == "own" else "")
            g.append(f"{name} & {KIND_NAMES[kind]}{star} & {a['val']:.2f} & {a['test']:.2f} \\\\")
        ev_groups.append(g)
    (OUT / "tab_evidence.tex").write_text(grouped(ev_groups), encoding="utf-8")

    # ---------- correction + flags (main configuration)
    corr_groups, flag_groups = [], []
    n_corr = n_flag = 0
    for ds, dsn in DATASETS:
        cg, fg = [], []
        for cap, label, _ in CAPS:
            f = V2 / "outputs" / "audit" / f"{cap}_{ds}_{m_enc}_{m_kind}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text(encoding="utf-8")); nums[f"{cap}|{ds}|audit"] = r
            c, cc = r["correction"], r["correction_cider_opt"]
            b0, b1 = r["binary"]["base"], r["binary"]["audited"]
            sb, sa = r["test_scores"]["base"], r["test_scores"]["audited"]
            pval = ((c.get("significance") or {}).get("CIDEr") or {}).get("perm_p")
            prec = f"{100 * c['test_acts_correct'] / c['test_acts']:.1f}\\%" if c["test_acts"] else "--"
            ptxt = "$<$0.001" if pval is not None and pval < 0.001 else (f"{pval:.3f}" if pval is not None else "--")
            first_in_ds = not cg
            if not first_in_ds:
                cg.append(SPACER)
            cg.append(f"{dsn if first_in_ds else ''} & {label} & {c['test_acts']} & {prec} & "
                      f"{arrow(b0['FP'], b1['FP'])} & {arrow(b0['FN'], b1['FN'])} & "
                      f"{arrow(f2(sb['B4']), f2(sa['B4']))} & {arrow(f2(sb['CIDEr']), f2(sa['CIDEr']))} & {ptxt} \\\\")
            cprec = f"{100 * cc['test_acts_correct'] / cc['test_acts']:.1f}\\%" if cc["test_acts"] else "--"
            cells = ("\\quad CIDEr-optimal", str(cc["test_acts"]), cprec,
                     arrow(b0["FP"], cc["binary"]["FP"]), arrow(b0["FN"], cc["binary"]["FN"]),
                     arrow(f2(sb["B4"]), f2(cc["scores"]["B4"])), arrow(f2(sb["CIDEr"]), f2(cc["scores"]["CIDEr"])))
            cg.append(" & " + " & ".join(f"\\alt{{{t}}}" for t in cells) + " & \\\\")
            n_corr += 1
            first_cap_row = True
            for bud in ("5%", "10%"):
                fr = r["flags"][bud]
                if fr["missed_total"] < 10:
                    continue
                lift = fr["precision"] / fr["random_precision"] if fr["random_precision"] else 0
                budtex = bud.replace("%", r"\%")
                # change-state accuracy after correction and review (a reviewed missed change counts as fixed)
                n_pairs = b0["TP"] + b0["FP"] + b0["TN"] + b0["FN"]
                acc_rev = 100 * (n_pairs - b1["FP"] - (b1["FN"] - fr["missed_found"])) / n_pairs
                # same number of no-change claims reviewed at random (expected value), for a fair comparison
                acc_rand = 100 * (n_pairs - b1["FP"] - (b1["FN"] - fr["n_flagged"] * fr["random_precision"] / 100)) / n_pairs
                nums[f"{cap}|{ds}|acc_after_review|{bud}"] = {"base": b0["acc"], "after": acc_rev, "random": acc_rand}
                if first_cap_row and fg:
                    fg.append(SPACER)
                fg.append(f"{'' if any(not x.startswith(SPACER) for x in fg) else dsn} & {label if first_cap_row else ''} & "
                          f"{budtex} & {fr['n_flagged']} & {fr['missed_found']} / {fr['missed_total']} ({fr['recall']:.1f}\\%) & "
                          f"{fr['precision']:.1f}\\% & {fr['random_precision']:.1f}\\% & {lift:.1f}$\\times$ & "
                          f"{arrow(f2(b0['acc']), f2(acc_rev))} ({f2(acc_rand)}) \\\\")
                first_cap_row = False; n_flag += 1
        corr_groups.append(cg); flag_groups.append(fg)
    (OUT / "tab_correction.tex").write_text(grouped(corr_groups), encoding="utf-8")
    (OUT / "tab_flags.tex").write_text(grouped(flag_groups), encoding="utf-8")
    tradeoff_table(refs, m_enc, m_kind, nums)
    prompt_table(nums)

    # ---------- robustness (appendix): one row per (dataset, captioner, features), columns = evidence models
    audits = {}
    for p in (V2 / "outputs" / "audit").glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        enc = "own" if r["encoder"].startswith("own:") else r["encoder"]
        audits[(r["dataset"], r["captioner"], enc, r["evidence"])] = r
    rob_groups = []
    for ds, dsn in DATASETS:
        rows, last_cap = [], None
        for cap, _, _ in CAPS:
            for enc in ENCS:
                cells = [audits.get((ds, cap, enc, k)) for k in KINDS]
                if not any(cells):
                    continue
                total = next(x for x in cells if x)["flags"]["5%"]["missed_total"]
                star = ["$^\\star$" if (enc, k) == (m_enc, m_kind) else "" for k in KINDS]
                corr = [("--" if x is None else (f"{x['correction']['test_acts']} ({x['correction']['test_acts_correct']})"
                                                 if x["correction"]["test_acts"] else "0")) + s for x, s in zip(cells, star)]
                found = [("--" if x is None else f"{x['flags']['5%']['missed_found']} ({x['flags']['5%']['precision']:.0f})") + s
                         for x, s in zip(cells, star)]
                if last_cap is not None and cap != last_cap:
                    rows.append(SPACER)
                rows.append(f"{'' if last_cap else dsn} & {SHORT[cap] if cap != last_cap else ''} & {'Own features' if enc == 'own' else ENC_NAMES[enc]} & "
                            f"{total} & " + " & ".join(corr) + " & " + " & ".join(found) + " \\\\")
                last_cap = cap
        rob_groups.append(rows)
    (OUT / "tab_robust.tex").write_text(grouped(rob_groups), encoding="utf-8")
    k3 = audits.get(("levir_cc", "rscama", "own", "knn3"))
    if k3:
        nums["pilot_knn3"] = {"acts": k3["correction"]["test_acts"], "flags5": k3["flags"]["5%"]}
    (V2 / "manuscript" / "numbers.json").write_text(json.dumps(nums, indent=1, default=float), encoding="utf-8")
    print(f"[tables] main config {main_cfg}: {n_corr} correction rows, {n_flag} flag rows -> {OUT}")


def tradeoff_table(refs, enc, kind, nums) -> None:
    """Operating points of the correction step on LEVIR-CC test: what the 90% precision rule buys and costs,
    compared with letting the evidence model decide (p < 0.5) and with the CIDEr-optimal threshold."""
    from common import norm  # noqa: E402
    test = refs["test"]; names = [r["name"] for r in test]
    y = np.array([r["changeflag"] == 1 for r in test]); n = len(y)
    p = s9.evidence(kind, enc, "levir_cc", refs, "test", names)
    groups = []

    def row(cap_label, rule, acts, wrong, fp, fn, star=""):
        acc = 100 * (n - fp - fn) / n
        corr = "--" if acts is None else (f"{acts} ({wrong})" if acts else "0")
        return f"{cap_label} & {rule}{star} & {corr} & {fp} & {fn} & {acc:.2f} \\\\"

    for cap, short in (("qwen3vl", "Qwen3-VL-4B"), ("rscama", "RSCaMa")):
        a = json.loads((V2 / "outputs" / "audit" / f"{cap}_levir_cc_{enc}_{kind}.json").read_text(encoding="utf-8"))
        pred = json.loads((V2 / "outputs" / f"{cap}_levir" / "test" / "predictions.json").read_text(encoding="utf-8"))
        c = np.array([asserts_change(norm(pred[x])) for x in names])
        fp0, fn0 = int((c & ~y).sum()), int((~c & y).sum())
        g = [row(short, "captioner alone", 0, 0, fp0, fn0)]
        ca = a["correction"]; b1 = a["binary"]["audited"]
        g.append(row("", "\\textbf{audit, $\\geq$90\\% validation precision}", ca["test_acts"],
                     ca["test_acts"] - ca["test_acts_correct"], b1["FP"], b1["FN"], "$^\\star$"))
        fire = c & (p < 0.5)
        fp, fn = int((c & ~fire & ~y).sum()), int(((~c | fire) & y).sum())
        g.append(row("", "evidence decides ($p<0.5$)", int(fire.sum()), int((fire & y).sum()), fp, fn))
        cc = a["correction_cider_opt"]
        g.append(row("", "\\alt{CIDEr-optimal threshold}", cc["test_acts"], cc["test_acts"] - cc["test_acts_correct"],
                     cc["binary"]["FP"], cc["binary"]["FN"]))
        nums[f"tradeoff|{cap}"] = {"alone": [fp0, fn0], "audit": [b1["FP"], b1["FN"]], "p05": [fp, fn],
                                   "p05_acts": int(fire.sum()), "p05_wrong": int((fire & y).sum())}
        groups.append(g)
    ev = p >= 0.5
    fp, fn = int((ev & ~y).sum()), int((~ev & y).sum())
    groups.append([row("Evidence model alone", "change if $p\\geq0.5$ (no caption)", None, None, fp, fn)])
    nums["tradeoff|evidence_alone"] = [fp, fn]
    (OUT / "tab_tradeoff.tex").write_text(grouped(groups), encoding="utf-8")


def prompt_table(nums) -> None:
    """Prompt sensitivity of the zero-shot VLM (from s20: outputs/prompt_test/summary.json)."""
    f = V2 / "outputs" / "prompt_test" / "summary.json"
    if not f.exists():
        return
    s = json.loads(f.read_text(encoding="utf-8"))
    variants = [("qwen3vlorig", "``the scene is the same as before''", "LEVIR-CC reference"),
                ("qwen3vl", "``no change is visible between the two images''$^\\star$", "in neither dataset")]
    groups = []
    for ds, dsn in DATASETS:
        g = []
        for tag, answer, note in variants:
            v = s.get(f"{tag}|{ds}")
            if v is None:
                continue
            b0, a = v["base"]["binary"], v["audit"]
            acts = f"{a['acts']} ({a['acts'] - a['acts_correct']})" if a["acts"] else "0"
            found = v["flags5"]
            g.append(f"{'' if g else dsn} & {answer} & {b0['TN'] + b0['FN']} & {b0['FP']} / {b0['FN']} & {b0['acc']:.2f} & "
                     f"{acts} & {a['binary']['FP']} / {a['binary']['FN']} & {a['binary']['acc']:.2f} & "
                     f"{arrow(f2(v['base']['all']['CIDEr']), f2(a['all']['CIDEr']))} & "
                     f"{found['missed_found']} / {found['missed_total']} \\\\")
            nums[f"prompt|{tag}|{ds}"] = v
        groups.append(g)
    (OUT / "tab_prompt.tex").write_text(grouped(groups), encoding="utf-8")


if __name__ == "__main__":
    main()

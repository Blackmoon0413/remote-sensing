#!/usr/bin/env python3
"""S9: two-direction change-state audit, thresholds chosen on the validation split.

For a captioner C on a dataset:
  c   = C's caption asserts a change (lexicon)
  p   = evidence probability that the pair changed (from retrieval features of
        frozen encoders; kNN = training-free, LR = linear probe on the same features)
  A) change -> no-change CORRECTION: if c and p <= t_c, replace the caption by C's
     most frequent no-change sentence (from C's own validation outputs). t_c is the
     largest val threshold with correction precision >= PREC_TARGET (>= MIN_ACTS acts);
     otherwise the audit abstains. The CIDEr-optimal threshold is reported for contrast.
  B) no-change -> change FLAGGING: if not c and p >= t_f, send the pair to review
     (caption unchanged). t_f is set on val so that the flags fill a review budget of
     5% / 10% of all pairs; pairs tied at t_f (discrete kNN votes) are flagged with the
     probability that fills the budget exactly (randomised threshold, fixed seed).
     Test reports missed changes recovered and precision.
Test is touched once with the val-selected settings.
Output: outputs/audit/<captioner>_<dataset>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change, binary, norm, score  # noqa: E402

PARTS = ["cls_pre", "cls_post", "cls_delta", "diff_mean", "diff_max", "prof"]
RNG = np.random.default_rng(20261123)
PREC_TARGET = 0.90  # required val precision of corrections (share of acted-on pairs that are unchanged)
MIN_ACTS = 3
NEVER = 2.0  # flag threshold above any probability: flag nothing


def feats(enc: str, ds: str, split: str, names: list[str]) -> np.ndarray:
    if enc.startswith("own:"):  # the captioner's own pooled encoder features (pilot-study memory)
        z = np.load(V2 / "outputs" / f"{enc[4:]}_{ds.split('_')[0]}" / split / "features.npz")
        pos = {str(n).split(".")[0]: i for i, n in enumerate(z["sample_ids"])}  # RSCaMa train ids carry ".png"
        return z["embeddings"][[pos[n] for n in names]].astype("float32")
    z = np.load(V2 / "outputs" / "features" / enc / f"{ds}_{split}.npz")
    pos = {n: i for i, n in enumerate(z["names"])}
    if enc == "pixdiff":  # pixel-difference baseline (s15)
        return z["x"][[pos[n] for n in names]].astype("float32")
    idx = [pos[n] for n in names]
    x = np.concatenate([(z["cls_post"] - z["cls_pre"]) if p == "cls_delta" else z[p] for p in PARTS], 1)
    return x[idx].astype("float32")


def evidence(kind: str, enc: str, ds: str, refs: dict, split: str, names: list[str]) -> np.ndarray:
    tr_names = [r["name"] for r in refs["train"]]
    ytr = np.array([r["changeflag"] for r in refs["train"]])
    xtr, xq = feats(enc, ds, "train", tr_names), feats(enc, ds, split, names)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xq = (xtr - mu) / sd, (xq - mu) / sd
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=0.1, max_iter=5000).fit(xtr, ytr).predict_proba(xq)[:, 1]
    K = int(kind[3:])  # knn<K>
    xtr /= np.linalg.norm(xtr, axis=1, keepdims=True); xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    order = np.argsort(-(xq @ xtr.T), axis=1)[:, :K]
    return ytr[order].mean(1)


def significance(a: dict, b: dict) -> dict:
    out = {}
    for m in ("B4", "METEOR", "ROUGE_L", "CIDEr"):
        d = b[m] - a[m]; n = len(d)
        boot = d[RNG.integers(0, n, size=(1000, n))].mean(1)
        perm = np.where(RNG.random((10000, n)) < 0.5, -d, d).mean(1)
        out[m] = {"delta": round(float(d.mean()), 3), "ci95": [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])],
                  "perm_p": round(float((np.abs(perm) >= abs(d.mean())).mean()), 4), "n_changed": int((d != 0).sum())}
    return out


def run(captioner: str, ds: str, enc: str, kind: str) -> dict:
    refs = json.loads((V2 / "data" / "processed" / f"refs_{ds}.json").read_text(encoding="utf-8"))
    pdir = V2 / "outputs" / f"{captioner}_{ds.split('_')[0]}"
    S = {}
    for split in ("val", "test"):
        rows = refs[split]; names = [r["name"] for r in rows]
        pred = json.loads((pdir / split / "predictions.json").read_text(encoding="utf-8"))
        caps = [norm(pred[n]) for n in names]
        S[split] = {"names": names, "refs": [r["refs"] for r in rows], "y": np.array([r["changeflag"] == 1 for r in rows]),
                    "caps": caps, "c": np.array([asserts_change(x) for x in caps]),
                    "p": evidence(kind, enc, ds, refs, split, names)}
    nc_sentence = Counter(x for x in S["val"]["caps"] if not asserts_change(x)).most_common(1)[0][0]

    # ---- A) correction threshold on val. 'never act' is always allowed.
    #  primary  : risk control - the largest threshold whose val correction precision
    #             (share of acted-on pairs that are truly unchanged) is >= PREC_TARGET
    #  secondary: the threshold that maximises val CIDEr (reported for comparison; it can
    #             silence real changes when the base captions score poorly anyway)
    v = S["val"]; base_val = score(v["refs"], v["caps"], only=("CIDEr",))["CIDEr"]
    pc = v["p"][v["c"]]
    cand = sorted(set(np.round(np.quantile(pc, np.linspace(0, 1, 101)), 4))) if len(pc) else []
    best_t, best_c, prec_t, prec_info = None, base_val, None, None
    for t in cand:
        fire = v["c"] & (v["p"] <= t)
        new = [nc_sentence if f else x for x, f in zip(v["caps"], fire)]
        cid = score(v["refs"], new, only=("CIDEr",))["CIDEr"]
        if cid > best_c + 1e-9:
            best_t, best_c = float(t), cid
        prec = float((fire & ~v["y"]).sum() / max(fire.sum(), 1))
        if fire.sum() >= MIN_ACTS and prec >= PREC_TARGET:
            prec_t, prec_info = float(t), {"val_acts": int(fire.sum()), "val_precision": round(prec * 100, 1)}
    # ---- B) flag thresholds on val for review budgets
    #  randomised threshold: flag p > t, and pairs tied at p == t with probability q, where t and q
    #  make the val flags fill the budget exactly (only matters for the discrete kNN votes)
    flags_t = {}
    cand_p = np.sort(v["p"][~v["c"]])[::-1]
    for budget in (0.05, 0.10):
        k = min(int(round(budget * len(v["names"]))), len(cand_p))
        if k == 0:
            flags_t[budget] = (NEVER, 0.0); continue
        thr = float(cand_p[k - 1])
        n_above, n_tie = int((cand_p > thr).sum()), int((cand_p == thr).sum())
        flags_t[budget] = (thr, (k - n_above) / n_tie)

    # ---- test, touched once per pre-selected rule
    t = S["test"]
    s_base, it_b = score(t["refs"], t["caps"], per_item=True)
    b0 = binary(t["c"], t["y"])

    def apply_rule(thr):
        fire = t["c"] & (t["p"] <= thr) if thr is not None else np.zeros(len(t["names"]), bool)
        audited = [nc_sentence if f else x for x, f in zip(t["caps"], fire)]
        s_aud, it_a = score(t["refs"], audited, per_item=True)
        return {"threshold_from_val": thr, "test_acts": int(fire.sum()), "test_acts_correct": int((fire & ~t["y"]).sum()),
                "scores": s_aud, "binary": binary(np.array([asserts_change(x) for x in audited]), t["y"]),
                "significance": significance(it_b, it_a) if fire.any() else None}

    rule_prec, rule_cider = apply_rule(prec_t), apply_rule(best_t)
    rule_prec["val"] = prec_info
    rule_cider["val"] = {"cider_base": base_val, "cider_best": best_c}
    missed = (~t["c"]) & t["y"]
    flag_res = {}
    for budget, (thr, q) in flags_t.items():
        tie_draw = np.random.default_rng(int(budget * 1000)).random(len(t["names"])) < q
        fl = (~t["c"]) & ((t["p"] > thr) | ((t["p"] == thr) & tie_draw))
        hit = int((fl & missed).sum())
        flag_res[f"{int(budget * 100)}%"] = {
            "threshold_from_val": round(thr, 4), "tie_flag_prob": round(q, 4), "n_flagged": int(fl.sum()), "missed_found": hit,
            "missed_total": int(missed.sum()), "recall": round(hit / max(missed.sum(), 1) * 100, 1),
            "precision": round(hit / max(fl.sum(), 1) * 100, 1),
            "random_precision": round(missed.sum() / max((~t["c"]).sum(), 1) * 100, 1)}
    # perfect audits (upper bounds)
    perfect = [nc_sentence if (c and not y) else x for x, c, y in zip(t["caps"], t["c"], t["y"])]
    res = {"captioner": captioner, "dataset": ds, "encoder": enc, "evidence": kind, "no_change_sentence": nc_sentence,
           "precision_target": PREC_TARGET,
           "correction": rule_prec, "correction_cider_opt": rule_cider,
           "test_scores": {"base": s_base, "audited": rule_prec["scores"], "perfect_c2n": score(t["refs"], perfect)},
           "binary": {"base": b0, "audited": rule_prec["binary"]}, "flags": flag_res}
    out = V2 / "outputs" / "audit"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{captioner}_{ds}_{enc.replace(':', '-')}_{kind}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captioners", nargs="+", default=["rscama", "promptcc"])
    ap.add_argument("--dataset", default="levir_cc")
    ap.add_argument("--encoders", nargs="+", default=["dinov2_b", "rclip_l14", "rclip_b32"])
    ap.add_argument("--evidence", nargs="+", default=["lr", "knn11", "knn51"])
    a = ap.parse_args()
    for cap in a.captioners:
        for enc in a.encoders:
            enc = f"own:{cap}" if enc == "own" else enc
            probe = (V2 / "outputs" / f"{cap}_{a.dataset.split('_')[0]}" / "train" / "features.npz") if enc.startswith("own:") \
                else (V2 / "outputs" / "features" / enc / f"{a.dataset}_test.npz")
            if not probe.exists():
                print(f"[skip] {cap} {enc}: no features"); continue
            for kind in a.evidence:
                r = run(cap, a.dataset, enc, kind)
                f5, f10 = r["flags"]["5%"], r["flags"]["10%"]
                cp, cc = r["correction"], r["correction_cider_opt"]
                b0, b1 = r["binary"]["base"], r["binary"]["audited"]
                print(f"{cap:9s} {enc:12s} {kind:6s} | prec-rule acts={cp['test_acts']}({cp['test_acts_correct']} ok) "
                      f"CIDEr {r['test_scores']['base']['CIDEr']:.2f}->{cp['scores']['CIDEr']:.2f} "
                      f"FP {b0['FP']}->{b1['FP']} FN {b0['FN']}->{b1['FN']} | cider-rule acts={cc['test_acts']}({cc['test_acts_correct']} ok) "
                      f"CIDEr->{cc['scores']['CIDEr']:.2f} | flags5% {f5['missed_found']}/{f5['missed_total']} prec {f5['precision']}% "
                      f"(rand {f5['random_precision']}%) | flags10% {f10['missed_found']} prec {f10['precision']}%", flush=True)


if __name__ == "__main__":
    main()

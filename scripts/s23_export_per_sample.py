#!/usr/bin/env python3
"""S23: per-sample evidence probabilities and audit decisions for every audit configuration.

The evidence p(change) is recomputed with s9's own `evidence()` (linear probe and kNN are
deterministic), the correction threshold is taken from the stored audit JSON, and the flag
thresholds are rebuilt from the validation split exactly as in s9 (randomised tie rule, same seeds).
Every file is checked against the counts stored in its audit JSON before it is written, so the CSVs
reproduce the paper's tables without the frozen-encoder features.

Output: outputs/audit_per_sample/<audit name>.csv with columns
  name, split, changeflag, claims_change, p_change, corrected, flagged_5pct, flagged_10pct
(decision columns are filled for the test split only; validation rows carry p_change for threshold checks)
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change, norm  # noqa: E402
from s9_audit import NEVER, evidence  # noqa: E402

# audit folder -> captioner name in the JSON -> folder holding that captioner's predictions
SETS = {"audit": {}, "audit_origprompt": {"qwen3vl": "qwen3vlorig"}}
OUT = V2 / "outputs" / "audit_per_sample"
_cache: dict = {}
_refs: dict = {}


def refs_of(ds: str) -> dict:
    if ds not in _refs:
        _refs[ds] = json.loads((V2 / "data" / "processed" / f"refs_{ds}.json").read_text(encoding="utf-8"))
    return _refs[ds]


def p_of(kind: str, enc: str, ds: str, split: str) -> np.ndarray:
    key = (kind, enc, ds, split)
    if key not in _cache:
        refs = refs_of(ds)
        _cache[key] = evidence(kind, enc, ds, refs, split, [r["name"] for r in refs[split]])
    return _cache[key]


def export(js_path, pred_map: dict, out_dir) -> str:
    r = json.loads(js_path.read_text(encoding="utf-8"))
    cap, ds, enc, kind = r["captioner"], r["dataset"], r["encoder"], r["evidence"]
    pdir = V2 / "outputs" / f"{pred_map.get(cap, cap)}_{ds.split('_')[0]}"
    S = {}
    for split in ("val", "test"):
        rows = refs_of(ds)[split]
        names = [x["name"] for x in rows]
        pred = json.loads((pdir / split / "predictions.json").read_text(encoding="utf-8"))
        c = np.array([asserts_change(norm(pred[n])) for n in names])
        S[split] = {"names": names, "y": np.array([x["changeflag"] == 1 for x in rows]), "c": c,
                    "p": p_of(kind, enc, ds, split)}
    v, t = S["val"], S["test"]
    # A) correction, threshold stored in the audit JSON
    thr = r["correction"]["threshold_from_val"]
    fire = t["c"] & (t["p"] <= thr) if thr is not None else np.zeros(len(t["names"]), bool)
    assert int(fire.sum()) == r["correction"]["test_acts"], (js_path.name, "acts")
    assert int((fire & ~t["y"]).sum()) == r["correction"]["test_acts_correct"], (js_path.name, "acts_correct")
    # B) flags, thresholds rebuilt on val as in s9
    flags = {}
    cand_p = np.sort(v["p"][~v["c"]])[::-1]
    for budget in (0.05, 0.10):
        k = min(int(round(budget * len(v["names"]))), len(cand_p))
        if k == 0:
            th, q = NEVER, 0.0
        else:
            th = float(cand_p[k - 1])
            q = (k - int((cand_p > th).sum())) / int((cand_p == th).sum())
        tie = np.random.default_rng(int(budget * 1000)).random(len(t["names"])) < q
        fl = (~t["c"]) & ((t["p"] > th) | ((t["p"] == th) & tie))
        ref = r["flags"][f"{int(budget * 100)}%"]
        assert int(fl.sum()) == ref["n_flagged"], (js_path.name, budget, "n_flagged")
        assert int((fl & ~t["c"] & t["y"]).sum()) == ref["missed_found"], (js_path.name, budget, "missed_found")
        flags[budget] = fl
    out = out_dir / js_path.name.replace(".json", ".csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "changeflag", "claims_change", "p_change", "corrected", "flagged_5pct", "flagged_10pct"])
        for i, n in enumerate(v["names"]):
            w.writerow([n, "val", int(v["y"][i]), int(v["c"][i]), repr(float(v["p"][i])), "", "", ""])
        for i, n in enumerate(t["names"]):
            w.writerow([n, "test", int(t["y"][i]), int(t["c"][i]), repr(float(t["p"][i])),
                        int(fire[i]), int(flags[0.05][i]), int(flags[0.10][i])])
    return f"{js_path.parent.name}/{out.name}: acts {int(fire.sum())}, flags {int(flags[0.05].sum())}/{int(flags[0.10].sum())}"


def main() -> None:
    n = 0
    for folder, pred_map in SETS.items():
        out_dir = OUT / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        for js in sorted((V2 / "outputs" / folder).glob("*.json")):
            print(export(js, pred_map, out_dir), flush=True)
            n += 1
    print(f"[s23] {n} files verified against the stored audit counts and written to {OUT}")


if __name__ == "__main__":
    main()

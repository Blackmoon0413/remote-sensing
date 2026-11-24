#!/usr/bin/env python3
"""S19: sanity checks on the reported improvements (LEVIR-CC test, main evidence configuration).

1. Trivial baselines: every caption = the no-change template; evidence model alone (p >= 0.5 -> change).
2. Caption metrics split into changed / unchanged pairs (where does the CIDEr gain come from?).
3. Change-state accuracy after review vs. random review of the same number of no-change claims.
4. Sensitivity of the VLM correction to the wording of the replacement sentence.
Writes outputs/sanity.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change, binary, norm, score  # noqa: E402

spec = importlib.util.spec_from_file_location("s9", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s9_audit.py"))
s9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)

DS, ENC, KIND = "levir_cc", "dinov2_b", "lr"


def sub(R, caps, mask, metrics=("B4", "CIDEr")):
    idx = np.flatnonzero(mask)
    s = score([R[i] for i in idx], [caps[i] for i in idx])
    return {m: round(s[m], 2) for m in metrics}


def main() -> None:
    refs = json.loads((V2 / "data" / "processed" / f"refs_{DS}.json").read_text(encoding="utf-8"))
    test = refs["test"]; names = [r["name"] for r in test]; R = [r["refs"] for r in test]
    y = np.array([r["changeflag"] == 1 for r in test]); n = len(y)
    p = s9.evidence(KIND, ENC, DS, refs, "test", names)
    out = {}
    tmpl = "the scene is the same as before"
    out["template_is_a_reference"] = sum(any(norm(x) == tmpl for x in r) for r in R)
    # 1. trivial baselines
    allnc = [tmpl] * n
    out["const_template"] = {"all": {k: round(v, 2) for k, v in score(R, allnc).items() if k in ("B4", "CIDEr")},
                             "acc": round(100 * (~y).mean(), 2)}
    ev = p >= 0.5
    out["evidence_only"] = binary(ev, y)
    res = {}
    for cap in ("rscama", "promptcc", "change3d", "qwen3vl"):
        pred = json.loads((V2 / "outputs" / f"{cap}_levir" / "test" / "predictions.json").read_text(encoding="utf-8"))
        caps = [norm(pred[n_]) for n_ in names]
        c = np.array([asserts_change(x) for x in caps])
        a = json.loads((V2 / "outputs" / "audit" / f"{cap}_{DS}_{ENC}_{KIND}.json").read_text(encoding="utf-8"))
        t_c = a["correction"]["threshold_from_val"]; nc = a["no_change_sentence"]
        fire = c & (p <= t_c) if t_c is not None else np.zeros(n, bool)
        aud = [nc if f else x for x, f in zip(caps, fire)]
        r = {"base": {"changed": sub(R, caps, y), "unchanged": sub(R, caps, ~y)},
             "audited": {"changed": sub(R, aud, y), "unchanged": sub(R, aud, ~y)},
             "no_change_sentence": nc}
        # evidence-only decision applied to this captioner's claims (p < 0.5 -> say no change)
        ev_fire = c & (p < 0.5)
        r["evidence_p05_rule"] = binary(np.array([asserts_change(nc if f else x) for x, f in zip(caps, ev_fire)]), y)
        # 3. accuracy after review vs random review of the same number of no-change claims
        b0 = binary(c, y); b1 = binary(np.array([asserts_change(x) for x in aud]), y)
        f5 = a["flags"]["5%"]
        acc_rev = 100 * (n - b1["FP"] - (b1["FN"] - f5["missed_found"])) / n
        exp_random_found = f5["n_flagged"] * f5["random_precision"] / 100
        acc_rand = 100 * (n - b1["FP"] - (b1["FN"] - exp_random_found)) / n
        r["accuracy"] = {"base": b0["acc"], "after_correction": b1["acc"], "after_5pct_flags": round(acc_rev, 2),
                         "after_5pct_random_review_expected": round(acc_rand, 2), "reviewed": f5["n_flagged"]}
        if cap == "qwen3vl":
            # 4. wording sensitivity: same corrections, neutral sentences that are not LEVIR-CC references
            r["wording"] = {}
            for alt in ("nothing has changed in this area", "no change is visible between the two images"):
                aud2 = [alt if f else x for x, f in zip(caps, fire)]
                r["wording"][alt] = {"all": {k: round(v, 2) for k, v in score(R, aud2).items() if k in ("B4", "CIDEr")},
                                     "is_reference": sum(any(norm(x) == alt for x in rr) for rr in R)}
        res[cap] = r
    out["captioners"] = res
    (V2 / "outputs" / "sanity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

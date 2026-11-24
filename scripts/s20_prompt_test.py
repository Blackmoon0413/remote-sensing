#!/usr/bin/env python3
"""S20: does the VLM's no-change wording matter? Compare the original prompt (no-change answer = a LEVIR-CC
reference sentence) with the neutral prompt (no-change answer that is in neither dataset's references).

For each prompt and dataset: change-state errors, caption scores (all pairs and changed pairs only),
the audit with the main evidence configuration, and the 'evidence decides (p < 0.5)' rule.
Audit JSONs for the neutral run are written to outputs/prompt_test/ (not to outputs/audit, so the paper's
tables are untouched). Summary: outputs/prompt_test/summary.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change, binary, norm, score  # noqa: E402

spec = importlib.util.spec_from_file_location("s9", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s9_audit.py"))
s9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)

ENC, KIND = "dinov2_b", "lr"
# main results use the neutral prompt (tag qwen3vl); the original prompt (qwen3vlorig) is the sensitivity variant
VARIANTS = [("qwen3vlorig", "original prompt"), ("qwen3vl", "neutral prompt (main)")]
OUT = V2 / "outputs" / "prompt_test"


def scores(R, caps, mask=None):
    idx = range(len(caps)) if mask is None else np.flatnonzero(mask)
    s = score([R[i] for i in idx], [caps[i] for i in idx])
    return {m: round(s[m], 2) for m in ("B4", "CIDEr")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds in ("levir_cc", "dubai_cc"):
        refs = json.loads((V2 / "data" / "processed" / f"refs_{ds}.json").read_text(encoding="utf-8"))
        test = refs["test"]; names = [r["name"] for r in test]; R = [r["refs"] for r in test]
        y = np.array([r["changeflag"] == 1 for r in test])
        p = s9.evidence(KIND, ENC, ds, refs, "test", names)
        for tag, label in VARIANTS:
            f = V2 / "outputs" / f"{tag}_{ds.split('_')[0]}" / "test" / "predictions.json"
            if not f.exists():
                print(f"[skip] {tag} {ds}: no predictions"); continue
            pred = json.loads(f.read_text(encoding="utf-8"))
            caps = [norm(pred[n]) for n in names]
            c = np.array([asserts_change(x) for x in caps])
            nc_out = Counter(x for x, ci in zip(caps, c) if not ci).most_common(3)
            # audit with the main configuration (thresholds from this variant's own validation outputs)
            r = s9.run(tag, ds, ENC, KIND)
            written = V2 / "outputs" / "audit" / f"{tag}_{ds}_{ENC}_{KIND}.json"
            if tag != "qwen3vl" and written.exists():
                shutil.move(str(written), OUT / written.name)  # keep the paper's audit folder unchanged
            fire = c & (p < 0.5)
            ev_rule = binary(np.array([asserts_change(r["no_change_sentence"] if fi else x) for x, fi in zip(caps, fire)]), y)
            t_c = r["correction"]["threshold_from_val"]
            fire_a = c & (p <= t_c) if t_c is not None else np.zeros(len(c), bool)
            aud = [r["no_change_sentence"] if fi else x for x, fi in zip(caps, fire_a)]
            summary[f"{tag}|{ds}"] = {
                "prompt": label, "top_no_change_outputs": nc_out,
                "base": {"binary": binary(c, y), "all": scores(R, caps), "changed": scores(R, caps, y)},
                "audit": {"t_c": t_c, "acts": r["correction"]["test_acts"], "acts_correct": r["correction"]["test_acts_correct"],
                          "binary": r["binary"]["audited"], "all": scores(R, aud), "changed": scores(R, aud, y),
                          "replacement": r["no_change_sentence"]},
                "evidence_p05": ev_rule,
                "flags5": r["flags"]["5%"],
            }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    for k, v in summary.items():
        b, a = v["base"], v["audit"]
        print(f"\n== {k} ({v['prompt']})  most frequent no-change outputs: {v['top_no_change_outputs'][:2]}")
        print(f"   alone : FP {b['binary']['FP']:4d} FN {b['binary']['FN']:4d} acc {b['binary']['acc']:6.2f} | "
              f"CIDEr all {b['all']['CIDEr']:6.2f} changed {b['changed']['CIDEr']:6.2f} | B4 all {b['all']['B4']:5.2f}")
        print(f"   audit : acts {a['acts']} ({a['acts'] - a['acts_correct']} wrong) FP {a['binary']['FP']:4d} FN {a['binary']['FN']:4d} "
              f"acc {a['binary']['acc']:6.2f} | CIDEr all {a['all']['CIDEr']:6.2f} changed {a['changed']['CIDEr']:6.2f} "
              f"| replacement '{a['replacement']}'")
        e = v["evidence_p05"]
        print(f"   p<0.5 : FP {e['FP']:4d} FN {e['FN']:4d} acc {e['acc']:6.2f} | flags5% {v['flags5']['missed_found']}/{v['flags5']['missed_total']}")


if __name__ == "__main__":
    main()

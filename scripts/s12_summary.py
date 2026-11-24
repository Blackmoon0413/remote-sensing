#!/usr/bin/env python3
"""S12: one table over every audit run (outputs/audit/*.json) + captioner error profile.

Prints, per captioner x dataset:
  base error profile on test: hallucinated changes (FP) and missed changes (FN)
  and for each evidence setting: correction acts (correct), CIDEr base -> audited,
  flags at 5% / 10% review budget: missed changes found, precision, x-fold over random.
Writes outputs/summary.md
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2  # noqa: E402


def main() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((V2 / "outputs" / "audit").glob("*.json"))]
    lines = ["| captioner | dataset | evidence | FP | FN | corr. acts (ok) | CIDEr base -> audited | "
             "5% flags: found / FN (prec, x rand) | 10% flags: found (prec) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["dataset"], r["captioner"], r["encoder"], r["evidence"])):
        b = r["binary"]["base"]; f5, f10 = r["flags"]["5%"], r["flags"]["10%"]
        lift = f5["precision"] / f5["random_precision"] if f5["random_precision"] else 0
        lines.append(
            f"| {r['captioner']} | {r['dataset']} | {r['encoder']}/{r['evidence']} | {b['FP']} | {b['FN']} | "
            f"{r['correction']['test_acts']} ({r['correction']['test_acts_correct']}) | "
            f"{r['test_scores']['base']['CIDEr']:.2f} -> {r['test_scores']['audited']['CIDEr']:.2f} | "
            f"{f5['missed_found']} / {f5['missed_total']} ({f5['precision']}%, {lift:.1f}x) | "
            f"{f10['missed_found']} ({f10['precision']}%) |")
    text = "\n".join(lines)
    (V2 / "outputs" / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

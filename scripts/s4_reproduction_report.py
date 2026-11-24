#!/usr/bin/env python3
"""S4: reproduction check + unified-protocol scores for every captioner run so far.

For each outputs/<model>_levir/test/:
  * repo-protocol scores (scores_repo.json) vs the numbers published in the paper
  * unified text-protocol scores (common.score) on the official references
  * binary change detection vs the official changeflag, incl. hallucination rate
Writes outputs/reproduction_report.json and prints a table.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change, binary, levir_split, score  # noqa: E402

PUBLISHED = {  # LEVIR-CC test, from the original papers
    "rscama": {"B1": 85.79, "B2": 77.99, "B3": 71.04, "B4": 65.24, "METEOR": 39.91, "ROUGE_L": 75.24, "CIDEr": 136.56},
    "promptcc": {"B1": 83.66, "B2": 75.73, "B3": 69.10, "B4": 63.54, "METEOR": 38.82, "ROUGE_L": 73.72, "CIDEr": 136.44},
}
KEYS = {"Bleu_1": "B1", "Bleu_2": "B2", "Bleu_3": "B3", "Bleu_4": "B4", "METEOR": "METEOR", "ROUGE_L": "ROUGE_L",
        "CIDEr": "CIDEr"}
M = ("B1", "B2", "B3", "B4", "METEOR", "ROUGE_L", "CIDEr")


def main() -> None:
    test = levir_split("test")
    names = [t["name"] for t in test]
    refs = [t["refs"] for t in test]
    gt = np.array([t["changeflag"] == 1 for t in test])
    report = {}
    for model in ("rscama", "promptcc"):
        d = V2 / "outputs" / f"{model}_levir" / "test"
        if not (d / "predictions.json").exists():
            print(f"[skip] {model}: no predictions yet"); continue
        preds_map = json.loads((d / "predictions.json").read_text(encoding="utf-8"))
        preds = [preds_map[n] for n in names]
        repo = json.loads((d / "scores_repo.json").read_text(encoding="utf-8"))
        repo = {KEYS[k]: v for k, v in repo.items() if k in KEYS}
        uni = score(refs, preds)
        b = binary(np.array([asserts_change(p) for p in preds]), gt)
        report[model] = {"repo_protocol": repo, "published": PUBLISHED[model],
                         "delta_vs_published": {m: round(repo[m] - PUBLISHED[model][m], 2) for m in M},
                         "unified_protocol": uni, "binary": b}
        print(f"\n== {model}")
        print("  metric     published   repo   delta  | unified")
        for m in M:
            print(f"  {m:8s} {PUBLISHED[model][m]:9.2f} {repo[m]:7.2f} {repo[m] - PUBLISHED[model][m]:+6.2f}  | {uni[m]:7.2f}")
        print(f"  binary vs changeflag: acc {b['acc']} prec {b['prec']} rec {b['rec']} "
              f"hallucination rate on unchanged pairs {b['halluc_rate']}% ({b['FP']}/{b['FP'] + b['TN']})")
        flags = d / "changeflag.json"
        if flags.exists():
            f = json.loads(flags.read_text(encoding="utf-8"))
            acc = np.mean([f[n]["pred"] == f[n]["gt"] for n in names]) * 100
            report[model]["own_classifier_acc"] = round(float(acc), 2)
            print(f"  PromptCC's own change classifier accuracy: {acc:.2f}%")
    (V2 / "outputs" / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S9b: re-run existing audit configurations (e.g. after a rule change) from outputs/audit/*.json.

Usage: s9b_rerun.py [--kinds knn11 knn51 knn3] [--skip-captioner-dataset change3d:dubai_cc]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2  # noqa: E402

spec = importlib.util.spec_from_file_location("s9", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s9_audit.py"))
s9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["knn3", "knn11", "knn51"])
    ap.add_argument("--skip", nargs="*", default=[], help="captioner:dataset pairs to skip")
    a = ap.parse_args()
    jobs = []
    for p in sorted((V2 / "outputs" / "audit").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["evidence"] in a.kinds and f"{r['captioner']}:{r['dataset']}" not in a.skip:
            jobs.append((r["captioner"], r["dataset"], r["encoder"], r["evidence"], r["flags"]))
    print(f"[rerun] {len(jobs)} configurations", flush=True)
    for cap, ds, enc, kind, old in jobs:
        new = s9.run(cap, ds, enc, kind)["flags"]
        print(f"{cap:9s} {ds:9s} {enc:13s} {kind:6s} | 5%: {old['5%']['n_flagged']}->{new['5%']['n_flagged']} flagged, "
              f"found {old['5%']['missed_found']}->{new['5%']['missed_found']} | 10%: {old['10%']['n_flagged']}->"
              f"{new['10%']['n_flagged']} flagged, found {old['10%']['missed_found']}->{new['10%']['missed_found']}", flush=True)
    print("[rerun] done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S8: how well does retrieval evidence predict the change state?

Memory = training pairs with their official change labels; query = val / test.
P_knn(change) = fraction of the K nearest training pairs (cosine) that changed.
Reports accuracy at 0.5 and ROC-AUC on val and test for each feature view and K,
plus the captioners' own implicit change state for reference.
Output: outputs/knn_change_state.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change  # noqa: E402

VIEWS = {
    "prof": ["prof"],
    "diff_max": ["diff_max"],
    "diff_mean": ["diff_mean"],
    "cls_delta": ["cls_delta"],
    "prof+diff_max": ["prof", "diff_max"],
    "prof+diff_mean+cls_delta": ["prof", "diff_mean", "cls_delta"],
}
KS = (1, 3, 5, 11, 21, 51)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def view(z, parts, mu=None, sd=None):
    blocks = []
    for p in parts:
        x = (z["cls_post"] - z["cls_pre"]) if p == "cls_delta" else z[p]
        blocks.append(x)
    x = np.concatenate(blocks, 1).astype("float32")
    if mu is None:
        mu, sd = x.mean(0), x.std(0) + 1e-6
    x = (x - mu) / sd
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x, mu, sd


def main() -> None:
    res = {}
    for model in sorted(p.name for p in (V2 / "outputs" / "features").iterdir()):
        for ds in ("levir_cc", "dubai_cc"):
            f = {s: V2 / "outputs" / "features" / model / f"{ds}_{s}.npz" for s in ("train", "val", "test")}
            if not all(p.exists() for p in f.values()):
                continue
            z = {s: np.load(p) for s, p in f.items()}
            ytr = z["train"]["changeflag"].astype(bool)
            for vname, parts in VIEWS.items():
                xtr, mu, sd = view(z["train"], parts)
                for split in ("val", "test"):
                    xq, _, _ = view(z[split], parts, mu, sd)
                    y = z[split]["changeflag"].astype(bool)
                    sims = xq @ xtr.T
                    order = np.argsort(-sims, axis=1)[:, :max(KS)]
                    for K in KS:
                        p = ytr[order[:, :K]].mean(1)
                        res.setdefault(f"{model}|{ds}|{vname}|K{K}", {})[split] = {
                            "acc": round(float(((p > 0.5) == y).mean() * 100), 2), "auc": round(auc(p, y) * 100, 2)}
    # captioners' implicit change state (LEVIR-CC test)
    test = json.loads((V2 / "data" / "processed" / "refs_levir_cc.json").read_text(encoding="utf-8"))["test"]
    for m in ("rscama", "promptcc"):
        p = V2 / "outputs" / f"{m}_levir" / "test" / "predictions.json"
        if p.exists():
            pr = json.loads(p.read_text(encoding="utf-8"))
            acc = np.mean([asserts_change(pr[r["name"]]) == bool(r["changeflag"]) for r in test]) * 100
            res[f"captioner|{m}|levir_cc"] = {"test": {"acc": round(float(acc), 2)}}
    (V2 / "outputs" / "knn_change_state.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    best = {}
    for k, v in res.items():
        if "val" in v:
            model, ds = k.split("|")[:2]
            if v["val"]["auc"] > best.get((model, ds), ("", {"val": {"auc": -1}}))[1]["val"]["auc"]:
                best[(model, ds)] = (k, v)
    for (model, ds), (k, v) in sorted(best.items()):
        print(f"best-on-val {k:45s} val acc {v['val']['acc']:6.2f} auc {v['val']['auc']:6.2f} | "
              f"test acc {v['test']['acc']:6.2f} auc {v['test']['auc']:6.2f}")
    for k, v in res.items():
        if k.startswith("captioner"):
            print(f"{k:45s} test acc {v['test']['acc']}")


if __name__ == "__main__":
    main()

"""Shared utilities for the redesigned (v2) experiments.

* V2 root resolved WITHOUT following junctions (run scripts via D:/rsicc_v2/...;
  METEOR-1.5 fails on non-ASCII paths).
* Unified caption scoring ("text protocol"): raw reference sentences from the
  official LevirCCcaptions.json (or Dubai-CC jsons), lower-cased, final period
  removed; predictions normalised the same way. coco-caption BLEU/METEOR/ROUGE-L/
  CIDEr from the RSCaMa repo's eval_func. All cross-model comparisons use this.
* Ground-truth change state = official `changeflag` annotation.
* Change-state lexicon for *predicted* captions (same families as the pilot study).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

V2 = Path(os.path.abspath(__file__)).parents[1]
EVAL_REPO = V2 / "third_party" / "RSCaMa"

KEYWORD_BANK = {
    "building": ["building", "buildings", "house", "houses", "residential", "roof", "constructed", "construction",
                 "villa", "villas"],
    "road": ["road", "roads", "street", "streets", "highway", "path", "track", "trail"],
    "urban": ["urban", "built-up", "city", "developed"],
    "vegetation": ["vegetation", "green", "trees", "tree", "forest", "grass", "woods", "plants"],
    "water": ["water", "lake", "river", "pond", "sea", "island", "pool"],
    "barren": ["desert", "bare", "barren", "sand", "bare land", "bareland", "landscape"],
    "no_change": ["no change", "unchanged", "same", "nothing", "no difference", "identical", "not changed",
                  "no evident change", "no changes"],
}


def families(text: str) -> set[str]:
    t = " " + str(text).lower() + " "
    return {k for k, terms in KEYWORD_BANK.items() if any(term in t for term in terms)}


# Change-event cues: a caption that mentions "the same area" but also says something was built,
# appeared, was removed, ... describes a change (long free-form VLM outputs do this often).
CHANGE_CUES = ["appear", "built", "build", "construct", "develop", "added", "remov", "replac", " new ", " now ",
               "increas", "decreas", "demolish", "expan", "extend", "grow", "disappear", "turn", "become",
               "convert", "emerg", "install", "paved", "cleared", "widen", "wider", "planted", "plowed",
               "not present", "absent", "not visible"]


def asserts_change(caption: str) -> bool:
    """A caption asserts a change unless it contains a no-change phrase and no change-event cue."""
    if "no_change" not in families(caption):
        return True
    t = " " + norm(caption) + " "
    return any(k in t for k in CHANGE_CUES)


def norm(s: str) -> str:
    """lower-case, punctuation -> space (VLM outputs), collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())


# ---------------------------------------------------------------- data
def levir_split(split: str) -> list[dict]:
    """[{name, changeflag, refs}] in official order."""
    d = json.loads((V2 / "data" / "levir_cc" / "LevirCCcaptions.json").read_text(encoding="utf-8"))
    out = []
    for im in d["images"]:
        if im["split"] == split:
            out.append({"name": im["filename"].split(".")[0], "changeflag": int(im["changeflag"]),
                        "refs": [norm(" ".join(s["tokens"])) for s in im["sentences"]]})
    return out


# ---------------------------------------------------------------- scoring
_SCORERS = None


def _scorers():
    global _SCORERS
    if _SCORERS is None:
        if str(EVAL_REPO) not in sys.path:
            sys.path.insert(0, str(EVAL_REPO))
        from eval_func.bleu.bleu import Bleu
        from eval_func.cider.cider import Cider
        from eval_func.meteor.meteor import Meteor
        from eval_func.rouge.rouge import Rouge
        assert all(ord(c) < 128 for c in str(EVAL_REPO)), "run via the ASCII junction (D:/rsicc_v2)"
        _SCORERS = [(Bleu(4), ["B1", "B2", "B3", "B4"]), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]
    return _SCORERS


def score(refs: list[list[str]], preds: list[str], per_item: bool = False, only: tuple | None = None):
    gts = [[norm(r) for r in rs] for rs in refs]
    res = [[norm(p) if norm(p) else "<empty>"] for p in preds]
    out, items = {}, {}
    for sc, names in _scorers():
        if only and not set([names] if isinstance(names, str) else names) & set(only):
            continue
        v, per = sc.compute_score(gts, res)
        if isinstance(v, list):
            for n, x, p in zip(names, v, per):
                out[n] = round(float(x) * 100, 2); items[n] = np.asarray(p, float) * 100
        else:
            out[names] = round(float(v) * 100, 2); items[names] = np.asarray(per, float) * 100
    return (out, items) if per_item else out


def binary(pred_change: np.ndarray, gt_change: np.ndarray) -> dict:
    p, g = np.asarray(pred_change, bool), np.asarray(gt_change, bool)
    tp, fp = int((p & g).sum()), int((p & ~g).sum())
    tn, fn = int((~p & ~g).sum()), int((~p & g).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "acc": round((tp + tn) / len(g) * 100, 2),
            "prec": round(P * 100, 2), "rec": round(R * 100, 2),
            "f1": round(2 * P * R / max(P + R, 1e-9) * 100, 2),
            "halluc_rate": round(fp / max(fp + tn, 1) * 100, 2)}  # change claimed on an unchanged pair

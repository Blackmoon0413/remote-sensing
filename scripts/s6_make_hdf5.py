#!/usr/bin/env python3
"""S6: build RSICCformer/Change3D-style HDF5 inputs for LEVIR-CC and Dubai-CC.

Same format as RSICC/utils.py::create_input_files (word map: words with frequency
> min_word_freq in insertion order, seed(123) caption sampling, max_len 50,
<start>/<end>/<pad>/<unk>), re-implemented with PIL because the original relies
on the removed scipy.misc.imresize and supports LEVIR-CC only.

Additionally writes, per split:
  {SPLIT}_NAMES_{base}.json   image names in HDF5 order (for per-sample outputs)
and per dataset data/processed/refs_{dataset}.json:
  {split: [{name, changeflag, refs}]}  normalised references + change label
  (LEVIR-CC: official changeflag; Dubai-CC: majority vote, >= 3 of 5 references describe a change)
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from random import choice, sample, seed

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2, asserts_change  # noqa: E402

CPI, MAX_LEN = 5, 50


def tok(raw: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", raw.lower()).split()


def load_levir():
    d = json.loads((V2 / "data" / "levir_cc" / "LevirCCcaptions.json").read_text(encoding="utf-8"))
    root = V2 / "data" / "levir_cc" / "images"
    rows = []
    for im in d["images"]:
        rows.append({"name": im["filename"].split(".")[0], "split": im["split"], "changeflag": int(im["changeflag"]),
                     "tokens": [[w.lower() for w in s["tokens"]] for s in im["sentences"]],
                     "paths": (root / im["split"] / "A" / im["filename"], root / im["split"] / "B" / im["filename"])})
    return rows


def load_dubai():
    base = V2 / "data" / "dubai_cc" / "datasetDubaiCCPublic"
    rows = []
    for f, split in (("Train", "train"), ("Validation", "val"), ("Test", "test")):
        d = json.loads((base / "description_jsontr_te_val" / f"{f}_Dubai_CC.json").read_text(encoding="utf-8"))
        for im in d["images"]:
            toks = [tok(s["raw"]) for s in im["sentences"]]
            flag = int(sum(asserts_change(" ".join(t)) for t in toks) >= 3)  # majority of the 5 annotators
            rows.append({"name": im["filename"].split(".")[0], "split": split, "changeflag": flag, "tokens": toks,
                         "paths": (base / "imgs_tiles" / "RGB" / "500_2000" / im["filename"],
                                   base / "imgs_tiles" / "RGB" / "500_2010" / im["filename"])})
    return rows


def read_img(p: Path) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)


def build(rows, dataset: str, min_word_freq: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{dataset}_{CPI}_cap_per_img_{min_word_freq}_min_word_freq"
    word_freq = Counter()
    for r in rows:
        for t in r["tokens"]:
            word_freq.update(t)
    words = [w for w in word_freq if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map["<unk>"] = len(word_map) + 1
    word_map["<start>"] = len(word_map) + 1
    word_map["<end>"] = len(word_map) + 1
    word_map["<pad>"] = 0
    (out_dir / f"WORDMAP_{base}.json").write_text(json.dumps(word_map), encoding="utf-8")
    seed(123)
    for split, SPLIT in (("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")):
        part = [r for r in rows if r["split"] == split and any(len(t) <= MAX_LEN for t in r["tokens"])]
        h5p = out_dir / f"{SPLIT}_IMAGES_{base}.hdf5"
        if h5p.exists():
            h5p.unlink()
        enc, lens = [], []
        with h5py.File(h5p, "w") as h:
            h.attrs["captions_per_image"] = CPI
            images = h.create_dataset("images", (len(part), 2, 3, 256, 256), dtype="uint8")
            for i, r in enumerate(tqdm(part, desc=f"{dataset} {SPLIT}")):
                caps = [t for t in r["tokens"] if len(t) <= MAX_LEN]
                caps = caps + [choice(caps) for _ in range(CPI - len(caps))] if len(caps) < CPI else sample(caps, k=CPI)
                images[i] = [read_img(r["paths"][0]), read_img(r["paths"][1])]
                for c in caps:
                    enc.append([word_map["<start>"]] + [word_map.get(w, word_map["<unk>"]) for w in c]
                               + [word_map["<end>"]] + [word_map["<pad>"]] * (MAX_LEN - len(c)))
                    lens.append(len(c) + 2)
        assert len(part) * CPI == len(enc) == len(lens)
        (out_dir / f"{SPLIT}_CAPTIONS_{base}.json").write_text(json.dumps(enc), encoding="utf-8")
        (out_dir / f"{SPLIT}_CAPLENS_{base}.json").write_text(json.dumps(lens), encoding="utf-8")
        (out_dir / f"{SPLIT}_NAMES_{base}.json").write_text(json.dumps([r["name"] for r in part]), encoding="utf-8")
    refs = {s: [{"name": r["name"], "changeflag": r["changeflag"], "refs": [" ".join(t) for t in r["tokens"]]}
                for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    (V2 / "data" / "processed" / f"refs_{dataset.lower()}.json").write_text(json.dumps(refs), encoding="utf-8")
    print(f"[{dataset}] vocab={len(word_map)} " + " ".join(
        f"{s}={len(v)} (changed {sum(x['changeflag'] for x in v)})" for s, v in refs.items()), flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["levir", "dubai"]
    if "dubai" in which:
        build(load_dubai(), "DUBAI_CC", 0, V2 / "data" / "processed" / "c3d_dubai")
    if "levir" in which:
        build(load_levir(), "LEVIR_CC", 5, V2 / "data" / "processed" / "c3d_levir")

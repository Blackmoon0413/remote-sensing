#!/usr/bin/env python3
"""S11: zero-shot change captions from a general vision-language model (Qwen3-VL-4B-Instruct).

Runs with the SYSTEM Python (transformers 5.x), not the v2 .venv (pinned to 4.39 for
the RSICC repos). Greedy decoding, one pair per call, resumable.
Output: outputs/qwen3vl_<levir|dubai>/<split>/predictions.json  {name: caption}

Run: python s11_vlm_caption.py --datasets levir_cc dubai_cc --splits val test --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

V2 = Path(os.path.abspath(__file__)).parents[1]
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_BASE = ("The two remote sensing images show the same area at an earlier time (first image) and a later "
         "time (second image). In one short sentence, describe what has changed between them, for example "
         "buildings, roads or vegetation that appeared or disappeared. ")
# variant -> (prompt, output tag). "neutral" is the prompt of the paper's main VLM results: its no-change answer
# appears in neither dataset's references. "orig" (prompt sensitivity analysis) asks for a no-change answer that is
# one of the LEVIR-CC reference sentences; "free" gives no fixed answer at all (not used in the paper).
PROMPTS = {
    # not a reference sentence in LEVIR-CC or Dubai-CC ("nothing has changed in this area" is one in Dubai-CC)
    "neutral": (_BASE + "If nothing has changed, answer exactly: no change is visible between the two images.",
                "qwen3vl"),
    "orig": (_BASE + "If nothing has changed, answer exactly: the scene is the same as before.", "qwen3vlorig"),
    "free": (_BASE + "If nothing has changed, say so in one short sentence.", "qwen3vlff"),
}
PROMPT = PROMPTS["neutral"][0]


def pair_paths(ds: str, split: str, name: str):
    if ds == "levir_cc":
        r = V2 / "data" / "levir_cc" / "images" / split
        return r / "A" / f"{name}.png", r / "B" / f"{name}.png"
    r = V2 / "data" / "dubai_cc" / "datasetDubaiCCPublic" / "imgs_tiles" / "RGB"
    return r / "500_2000" / f"{name}.tif", r / "500_2010" / f"{name}.tif"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["levir_cc", "dubai_cc"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--variant", choices=sorted(PROMPTS), default="neutral")
    args = ap.parse_args()
    prompt, tag = PROMPTS[args.variant]
    print(f"[prompt:{args.variant}] {prompt}", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    device = "cpu" if args.cpu else f"cuda:{args.gpu}"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    for ds in args.datasets:
        refs = json.loads((V2 / "data" / "processed" / f"refs_{ds}.json").read_text(encoding="utf-8"))
        for split in args.splits:
            out = V2 / "outputs" / f"{tag}_{ds.split('_')[0]}" / split / "predictions.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            preds = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
            rows = refs[split][: args.limit] if args.limit else refs[split]
            t0, n_new = time.time(), 0
            for r in rows:
                if r["name"] in preds:
                    continue
                a, b = (Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)
                        for p in pair_paths(ds, split, r["name"]))
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": a}, {"type": "image", "image": b}, {"type": "text", "text": prompt}]}]
                inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
                text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                preds[r["name"]] = " ".join(text.strip().split())
                n_new += 1
                if n_new % 100 == 0:
                    out.write_text(json.dumps(preds, indent=1), encoding="utf-8")
                    print(f"[{ds} {split}] {len(preds)}/{len(rows)} {(time.time() - t0) / n_new:.2f}s/pair", flush=True)
            out.write_text(json.dumps(preds, indent=1), encoding="utf-8")
            print(f"[done] {ds} {split}: {len(preds)} captions, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

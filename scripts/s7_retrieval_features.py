#!/usr/bin/env python3
"""S7: captioner-independent retrieval features from frozen foundation models.

For every bi-temporal pair the patch tokens of the pre- and post-image are taken
from a frozen encoder (no training, no captions). Stored per pair:
  cls_pre, cls_post : global embeddings (L2-normalised)
  diff_mean         : mean over patches of |t_post - t_pre|   (tokens L2-normalised)
  diff_max          : max  over patches of |t_post - t_pre|
  prof              : per-patch cosine change 1 - <t_pre, t_post>, sorted descending
                      (a "how much of the scene changed, how strongly" profile that
                      is invariant to where the change is and insensitive to global
                      colour shifts)
Models: dinov2_b (facebook/dinov2-base), rclip_l14 / rclip_b32 (RemoteCLIP, open_clip).
Output: outputs/features/<model>/<dataset>_<split>.npz

Run: python s7_retrieval_features.py --models dinov2_b rclip_l14 --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import V2  # noqa: E402

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIPN = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def image_paths(dataset: str, name: str, split: str):
    if dataset == "levir_cc":
        root = V2 / "data" / "levir_cc" / "images" / split
        return root / "A" / f"{name}.png", root / "B" / f"{name}.png"
    root = V2 / "data" / "dubai_cc" / "datasetDubaiCCPublic" / "imgs_tiles" / "RGB"
    return root / "500_2000" / f"{name}.tif", root / "500_2010" / f"{name}.tif"


def load(p, size, mean_std):
    im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    m, s = (torch.tensor(v).view(3, 1, 1) for v in mean_std)
    return (x - m) / s


class Encoder:
    def __init__(self, name: str, device):
        self.name, self.device = name, device
        if name == "dinov2_b":
            from transformers import Dinov2Model
            self.m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
            self.size, self.norm = 224, IMNET
        elif name.startswith("rclip"):
            import open_clip
            arch = {"rclip_l14": "ViT-L-14", "rclip_b32": "ViT-B-32"}[name]
            model, _, _ = open_clip.create_model_and_transforms(arch)
            sd = torch.load(V2 / "weights" / "remoteclip" / f"RemoteCLIP-{arch}.pt", map_location="cpu",
                            weights_only=False)
            print(f"[{name}] load_state_dict:", model.load_state_dict(sd), flush=True)
            self.m = model.visual.to(device).eval()
            self.m.output_tokens = True
            self.size, self.norm = 224, CLIPN
        else:
            raise ValueError(name)

    @torch.no_grad()
    def __call__(self, x):
        with torch.autocast("cuda", dtype=torch.float16):
            if self.name == "dinov2_b":
                h = self.m(pixel_values=x).last_hidden_state
                g, t = h[:, 0], h[:, 1:]
            else:
                g, t = self.m(x)
        return F.normalize(g.float(), dim=-1), F.normalize(t.float(), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dinov2_b", "rclip_l14"])
    ap.add_argument("--datasets", nargs="+", default=["levir_cc", "dubai_cc"])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    for mname in args.models:
        enc = Encoder(mname, dev)
        for ds in args.datasets:
            refs = json.loads((V2 / "data" / "processed" / f"refs_{ds}.json").read_text(encoding="utf-8"))
            for split, rows in refs.items():
                out = V2 / "outputs" / "features" / mname / f"{ds}_{split}.npz"
                if out.exists():
                    print(f"[skip] {out.name}"); continue
                out.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                acc = {k: [] for k in ("cls_pre", "cls_post", "diff_mean", "diff_max", "prof")}
                for i in range(0, len(rows), args.bs):
                    chunk = rows[i:i + args.bs]
                    pa = torch.stack([load(image_paths(ds, r["name"], split)[0], enc.size, enc.norm) for r in chunk]).to(dev)
                    pb = torch.stack([load(image_paths(ds, r["name"], split)[1], enc.size, enc.norm) for r in chunk]).to(dev)
                    ga, ta = enc(pa)
                    gb, tb = enc(pb)
                    d = (tb - ta).abs()
                    acc["cls_pre"].append(ga.cpu()); acc["cls_post"].append(gb.cpu())
                    acc["diff_mean"].append(d.mean(1).cpu()); acc["diff_max"].append(d.amax(1).cpu())
                    acc["prof"].append((1 - (ta * tb).sum(-1)).sort(dim=1, descending=True).values.cpu())
                np.savez(out, names=np.array([r["name"] for r in rows]),
                         changeflag=np.array([r["changeflag"] for r in rows]),
                         **{k: torch.cat(v).numpy().astype("float32") for k, v in acc.items()})
                print(f"[{mname}] {ds} {split}: n={len(rows)} {time.time() - t0:.0f}s -> {out}", flush=True)
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

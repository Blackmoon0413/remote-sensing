#!/usr/bin/env python3
"""S0: verify downloads (size, and sha256 against the HF LFS oid when available).

Usage: python scripts/s0_verify.py [relative/path ...]   (default: all known files)
Writes v2/logs/s0_verify.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

V2 = Path(os.path.abspath(__file__)).parents[1]  # abspath keeps the ASCII junction path (D:/rsicc_v2); METEOR breaks on non-ASCII paths
HF_API = "https://huggingface.co/api/{kind}/{repo}/tree/main"
FILES = {  # local path -> (hf kind, repo, remote name) or (None, expected_size)
    "data/raw/levir_cc/Levir-CC-dataset.zip": ("datasets", "lcybuaa/LEVIR-CC", "Levir-CC-dataset.zip"),
    "weights/rscama/RSCaMa.pth": ("models", "lcybuaa/RSCaMa", "RSCaMa.pth"),
    "weights/promptcc/cls_model.pth.tar": ("models", "lcybuaa/PromptCC", "cls_model.pth.tar"),
    "weights/promptcc/BEST_checkpoint_ViT-B_32.pth.tar": ("models", "lcybuaa/PromptCC", "BEST_checkpoint_ViT-B_32.pth.tar"),
    "weights/promptcc/TEST_LEVIR_CC_5_cap_per_img.pkl": ("models", "lcybuaa/PromptCC", "TEST_LEVIR_CC_5_cap_per_img.pkl"),
    "weights/remoteclip/RemoteCLIP-ViT-B-32.pt": ("models", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"),
    "weights/remoteclip/RemoteCLIP-ViT-L-14.pt": ("models", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt"),
    "weights/promptcc/VAL_LEVIR_CC_5_cap_per_img.pkl": ("models", "lcybuaa/PromptCC", "VAL_LEVIR_CC_5_cap_per_img.pkl"),
    "weights/x3d/X3D_L.pyth": (None, 50025453),
    "data/raw/dubai_cc/DubaiCC.rar": (None, 59895268),
}


def remote_meta(kind: str, repo: str) -> dict:
    with urllib.request.urlopen(HF_API.format(kind=kind, repo=repo), timeout=60) as r:
        return {f["path"]: f for f in json.load(r)}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    targets = sys.argv[1:] or list(FILES)
    cache: dict = {}
    report = {}
    for rel in targets:
        p = V2 / rel
        spec = FILES[rel]
        if not p.exists():
            report[rel] = "MISSING"; print(f"[MISSING] {rel}"); continue
        size = p.stat().st_size
        if spec[0] is None:
            ok = size == spec[1]
            report[rel] = {"size": size, "expected": spec[1], "ok": ok}
        else:
            kind, repo, name = spec
            meta = cache.setdefault((kind, repo), remote_meta(kind, repo))[name]
            exp_size = meta["size"]
            exp_sha = (meta.get("lfs") or {}).get("oid")
            got_sha = sha256(p) if size == exp_size else None
            ok = size == exp_size and (exp_sha is None or got_sha == exp_sha)
            report[rel] = {"size": size, "expected": exp_size, "sha256": got_sha, "expected_sha256": exp_sha, "ok": ok}
        print(f"[{'OK' if report[rel]['ok'] else 'BAD'}] {rel}  {size:,} bytes")
    (V2 / "logs").mkdir(exist_ok=True)
    prev = {}
    out = V2 / "logs" / "s0_verify.json"
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
    prev.update(report)
    out.write_text(json.dumps(prev, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

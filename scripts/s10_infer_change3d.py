#!/usr/bin/env python3
"""S10: per-sample outputs of a trained Change3D captioner (BEST checkpoint = best val BLEU-4).

Uses the official evaluate() of Change3D/scripts/train_CC.py (greedy, beam 1) so
the decoding is identical to the authors' code; its res.json is mapped back to
image names with the {SPLIT}_NAMES file written by s6_make_hdf5.py.
Also saves pooled encoder features (captioner-own memory) for train/val/test.

Output: outputs/change3d_<levir|dubai>/<split>/{predictions.json, features.npz, scores_repo.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

V2 = Path(os.path.abspath(__file__)).parents[1]
REPO = V2 / "third_party" / "Change3D"
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import torchvision.transforms as transforms  # noqa: E402
from einops import rearrange  # noqa: E402
import train_CC  # noqa: E402
from data.dataset import CaptionDataset  # noqa: E402
from model.trainer import Trainer  # noqa: E402

DATA = {"levir": ("c3d_levir", "LEVIR_CC_5_cap_per_img_5_min_word_freq"),
        "dubai": ("c3d_dubai", "DUBAI_CC_5_cap_per_img_0_min_word_freq")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="levir", choices=list(DATA))
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no_features", action="store_true")
    a = ap.parse_args()
    sub, base = DATA[a.dataset]
    file_root = V2 / "data" / "processed" / sub
    ckpt = Path(a.checkpoint) if a.checkpoint else next((V2 / "outputs" / f"change3d_{a.dataset}").glob(f"*/BEST_checkpoint_{base}.pth.tar"))
    word_map = json.loads((file_root / f"WORDMAP_{base}.json").read_text(encoding="utf-8"))
    torch.cuda.set_device(a.gpu)
    args = SimpleNamespace(file_root=str(file_root), dataset=base, n_head=8, n_layer=3, decoder_n_layers=1, embed_dim=192,
                           dropout=0.1, num_perception_frame=1, in_height=256, in_width=256,
                           pretrained=str(V2 / "weights" / "x3d" / "X3D_L.pyth"), vocab_size=len(word_map), beam_size=1,
                           fine_tune_encoder=True, gpu_id=a.gpu, Split="TEST", save_path="")
    trainer = Trainer(args).cuda()
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    trainer.encoder.load_state_dict(st["encoder_image"]); trainer.decoder.load_state_dict(st["decoder"])
    print(f"[ckpt] {ckpt.name} epoch={st.get('epoch')} val BLEU-4={st.get('bleu-4')}", flush=True)
    enc, dec = trainer.encoder.eval(), trainer.decoder.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for split in a.splits:
        out = V2 / "outputs" / f"change3d_{a.dataset}" / split
        out.mkdir(parents=True, exist_ok=True)
        names = json.loads((file_root / f"{split.upper()}_NAMES_{base}.json").read_text(encoding="utf-8"))
        if split != "train":
            args.Split, args.save_path = split.upper(), str(out)
            metrics = train_CC.evaluate(args, enc, dec, normalize)
            res = json.loads((out / "eval_results" / "res.json").read_text(encoding="utf-8"))
            assert len(res) == len(names), (len(res), len(names))
            preds = {n: res[str(i)][0].strip() for i, n in enumerate(names)}
            (out / "predictions.json").write_text(json.dumps(preds, indent=1), encoding="utf-8")
            (out / "scores_repo.json").write_text(json.dumps(
                {"n": len(names), "epoch": st.get("epoch"), **{k: round(float(v) * 100, 2) for k, v in metrics.items()}},
                indent=2), encoding="utf-8")
            print(f"[{split}] {len(preds)} predictions; repo scores {metrics}", flush=True)
        if a.no_features:
            continue
        ds = CaptionDataset(str(file_root), base, split.upper(), transform=transforms.Compose([normalize]))
        ds.split = "NOAUG"  # the TRAIN split randomly swaps pre/post images (augmentation); disable it
        cpi = ds.cpi
        feats = []
        with torch.no_grad():
            for j in range(len(names)):
                item = ds[j * cpi]
                img = item[0].unsqueeze(0).cuda()
                f = enc(img[:, 0], img[:, 1], output_final=True)
                feats.append(f.mean(dim=(2, 3)).squeeze(0).float().cpu().numpy())
        np.savez(out / "features.npz", sample_ids=np.array(names), embeddings=np.stack(feats).astype("float32"))
        print(f"[{split}] features {len(feats)}", flush=True)


if __name__ == "__main__":
    main()

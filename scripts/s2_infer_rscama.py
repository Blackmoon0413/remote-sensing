#!/usr/bin/env python3
"""S2: run the released RSCaMa checkpoint on a LEVIR-CC split.

Saves (v2/outputs/rscama_levir/<split>/):
  predictions.json   {image_name: caption}
  features.npz       sample_ids + pooled encoder features (mean over tokens of
                     the AttentiveEncoder output) for the captioner-feature memory
  scores_repo.json   scores from RSCaMa's own evaluator (token-id protocol), used
                     to check reproduction of the published numbers

Run from anywhere:
  .venv/Scripts/python.exe scripts/s2_infer_rscama.py --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

V2 = Path(os.path.abspath(__file__)).parents[1]  # abspath keeps the ASCII junction path (D:/rsicc_v2); METEOR breaks on non-ASCII paths
REPO = V2 / "third_party" / "RSCaMa"
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # eval_func (METEOR jar) and relative imports expect the repo root

from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset  # noqa: E402
from model.model_decoder import DecoderTransformer  # noqa: E402
from model.model_encoder_attMamba import AttentiveEncoder, Encoder  # noqa: E402
from utils_tool.utils import get_eval_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--checkpoint", default=str(V2 / "weights" / "rscama" / "RSCaMa.pth"))
    ap.add_argument("--data_folder", default=str(V2 / "data" / "levir_cc" / "images"))
    ap.add_argument("--list_path", default=str(V2 / "data" / "processed" / "rscama_levir") + "/")
    ap.add_argument("--vocab", default=str(V2 / "data" / "processed" / "rscama_levir" / "vocab.json"))  # 468 tokens, matches the released checkpoint
    ap.add_argument("--out", default=str(V2 / "outputs" / "rscama_levir"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N samples")
    args = ap.parse_args()
    # LEVIRCCDataset builds the post-image path with path.replace('A', 'B') on the WHOLE path
    assert "A" not in args.data_folder, "data_folder must not contain an uppercase 'A'"

    torch.cuda.set_device(args.gpu)
    word_vocab = json.load(open(args.vocab, encoding="utf-8"))
    idx2word = list(word_vocab.keys())

    # released model: CLIP-ViT-B/32 backbone, transformer decoder (see RSCaMa test.py defaults)
    network, encoder_dim, feat_size, embed_dim = "CLIP-ViT-B/32", 768, 7, 768
    encoder = Encoder(network)
    encoder_trans = AttentiveEncoder(n_layers=3, feature_size=[feat_size, feat_size, encoder_dim], heads=8, dropout=0.1)
    decoder = DecoderTransformer(decoder_type="transformer_decoder", embed_dim=embed_dim, vocab_size=len(word_vocab),
                                 max_lengths=42, word_vocab=word_vocab, n_head=8, n_layers=1, dropout=0.1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print("[ckpt] keys:", [k for k in ckpt.keys()][:10], flush=True)
    encoder.load_state_dict(ckpt["encoder_dict"])
    encoder_trans.load_state_dict(ckpt["encoder_trans_dict"])
    decoder.load_state_dict(ckpt["decoder_dict"])
    del ckpt
    encoder, encoder_trans, decoder = encoder.cuda().eval(), encoder_trans.cuda().eval(), decoder.cuda().eval()

    ds = LEVIRCCDataset(network, args.data_folder, args.list_path, args.split, args.list_path + "tokens/",
                        word_vocab, 42, 1)
    if args.limit:
        ds.files = ds.files[: args.limit]
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    except_tokens = {word_vocab["<START>"], word_vocab["<END>"], word_vocab["<NULL>"]}

    preds, feats, names, references, hypotheses = {}, [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"RSCaMa {args.split}"):
            imgA, imgB = batch["imgA"].cuda(), batch["imgB"].cuda()
            f1, f2 = encoder(imgA, imgB)
            feat = encoder_trans(f1, f2)
            seq = decoder.sample(feat, k=1)
            pred = [w for w in seq if w not in except_tokens]
            refs = [[w for w in c if w not in except_tokens] for c in batch["token_all"].squeeze(0).tolist()]
            hypotheses.append(pred)
            references.append(refs)
            name = batch["name"][0]
            names.append(name)
            preds[name] = " ".join(idx2word[i] for i in pred)
            feats.append(feat.reshape(feat.shape[0], feat.shape[1], -1).mean(-1).squeeze(0).float().cpu().numpy())
    elapsed = time.time() - t0

    out = Path(args.out) if Path(args.out).is_absolute() else V2 / args.out  # cwd is the repo (os.chdir above)
    out = out / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(preds, indent=1), encoding="utf-8")
    np.savez(out / "features.npz", sample_ids=np.array(names), embeddings=np.stack(feats).astype("float32"))
    print(f"[saved] predictions + features -> {out}", flush=True)
    if args.split == "train":  # no need to score the memory split
        return
    scores = {k: round(float(v) * 100, 2) for k, v in get_eval_score(references, hypotheses).items()}
    (out / "scores_repo.json").write_text(json.dumps({"n": len(names), "seconds": round(elapsed, 1), **scores},
                                                     indent=2), encoding="utf-8")
    print(f"[done] n={len(names)} {elapsed:.0f}s scores={scores}", flush=True)


if __name__ == "__main__":
    main()

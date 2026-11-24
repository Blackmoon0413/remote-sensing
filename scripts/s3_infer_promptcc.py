#!/usr/bin/env python3
"""S3: run the released PromptCC checkpoint on the LEVIR-CC test split.

Replicates the 'predicted changeflag' pass of PromptCC/eval2.py (beam size 3),
i.e. the setting reported in the paper, but also keeps per-sample outputs.

Saves (v2/outputs/promptcc_levir/test/):
  predictions.json   {image_name: caption}
  changeflag.json    {image_name: {"pred": 0/1, "gt": 0/1}}  (PromptCC's own classifier)
  scores_repo.json   scores from PromptCC's evaluator (GPT-2 token protocol)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

V2 = Path(os.path.abspath(__file__)).parents[1]  # abspath keeps the ASCII junction path (D:/rsicc_v2); METEOR breaks on non-ASCII paths
REPO = V2 / "third_party" / "PromptCC"
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from transformers import GPT2Tokenizer  # noqa: E402
from datasets import CaptionDataset  # noqa: E402  (PromptCC's own datasets.py)
from utils import get_eval_score  # noqa: E402

device = torch.device("cuda")


def upgrade_legacy_gpt2(model) -> int:
    """The checkpoint pickles GPT2LMHeadModel objects created with transformers 4.10.3.
    Newer transformers code reads attributes that did not exist then (e.g.
    `gradient_checkpointing`, `config._attn_implementation`). Copy every missing
    non-tensor attribute from a freshly built module of the same name; the defaults
    of those attributes reproduce the old behaviour. Weights are untouched."""
    from transformers import GPT2Config, GPT2LMHeadModel
    n = 0
    for _, old in model.named_modules():
        if not isinstance(old, GPT2LMHeadModel):
            continue
        fresh_cfg = GPT2Config(**{k: v for k, v in old.config.__dict__.items() if not k.startswith("_")})
        for k, v in fresh_cfg.__dict__.items():
            if not hasattr(old.config, k):
                setattr(old.config, k, v); n += 1
        fresh = GPT2LMHeadModel(fresh_cfg)
        fresh_mods = dict(fresh.named_modules())
        for name, m in old.named_modules():
            ref = fresh_mods.get(name)
            if ref is None:
                continue
            for k, v in ref.__dict__.items():
                if k in ("_parameters", "_buffers", "_modules") or k in m.__dict__:
                    continue
                if isinstance(v, torch.Tensor):
                    continue
                m.__dict__[k] = v; n += 1
    return n


def beam_decode(model, tokenizer, ori_img, changeflag, k=3):
    """Body of eval2.evaluate for xunhuan == 1 (predicted change flag)."""
    _, pre_flag, _ = model.Image_Encoder(changeflag, ori_img)
    pred_flag = torch.argmax(pre_flag, 1)
    _, _, inputs_embeds = model.Image_Encoder(pred_flag, ori_img)
    end_id = tokenizer.encode(".")[0]
    d, n = inputs_embeds.size(-1), inputs_embeds.size(1)
    inputs_embeds = inputs_embeds.expand(k, n, d)
    top_k_scores = torch.zeros(k, 1, device=device)
    complete_seqs, complete_scores = [], []
    step = 1
    while True:
        out = model.gpt_decoder(inputs_embeds=inputs_embeds).logits
        logits, _ = model.dual_branch_func(pred_flag, out)
        scores = F.log_softmax(logits[:, -1, :], dim=-1)
        vocab = logits.size(-1)
        scores = top_k_scores.expand_as(scores) + scores
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        prev = torch.div(top_k_words, vocab, rounding_mode="floor")
        nxt = top_k_words % vocab
        seqs = nxt.unsqueeze(1) if step == 1 else torch.cat([seqs[prev], nxt.unsqueeze(1)], dim=1)
        incomplete = [i for i, w in enumerate(nxt) if w != end_id]
        complete = list(set(range(len(nxt))) - set(incomplete))
        if complete:
            complete_seqs.extend(seqs[complete].tolist())
            complete_scores.extend(top_k_scores[complete])
        k -= len(complete)
        if k == 0:
            break
        seqs = seqs[incomplete]
        inputs_embeds = inputs_embeds[prev[incomplete]]
        top_k_scores = top_k_scores[incomplete].unsqueeze(1)
        prev_words = nxt[incomplete].unsqueeze(1)
        inputs_embeds = torch.cat((inputs_embeds, model.gpt_decoder.transformer.wte(prev_words)), dim=1)
        if step > 50:
            break
        step += 1
    if not complete_scores:
        complete_seqs.extend(seqs[incomplete].tolist())
        complete_scores.extend(top_k_scores[incomplete])
    norm = [s / len(q) for s, q in zip(complete_scores, complete_seqs)]
    seq = complete_seqs[norm.index(max(norm))]
    return [w for w in seq if w != end_id], int(pred_flag.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(V2 / "weights" / "promptcc" / "BEST_checkpoint_ViT-B_32.pth.tar"))
    ap.add_argument("--data_folder", default=str(V2 / "weights" / "promptcc"))
    ap.add_argument("--captions_json", default=str(V2 / "data" / "levir_cc" / "LevirCCcaptions.json"))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)["model_GPT"].eval()
    print(f"[legacy] added {upgrade_legacy_gpt2(model)} missing attributes to pickled GPT-2 modules", flush=True)
    ds = CaptionDataset("ViT-B/32", "LEVIR_CC", args.data_folder, "LEVIR_CC_5_cap_per_img", args.split.upper(),
                        prefix_length=2 * 49 + 5 + 1)
    names = [im["filename"].split(".")[0] for im in json.load(open(args.captions_json, encoding="utf-8"))["images"]
             if im["split"] == args.split]
    n_img = len(ds) // 5
    assert n_img == len(names), (n_img, len(names))
    if args.limit:
        n_img = args.limit

    preds, flags, hyps, refs = {}, {}, [], []
    end_id = tokenizer.encode(".")[0]
    t0 = time.time()
    with torch.no_grad():
        for j in tqdm(range(n_img), desc=f"PromptCC {args.split}"):
            ori_img, changeflag, _caps, _mask, _caplens, all_caps = ds[j * 5 + 4]
            ori_img = ori_img.unsqueeze(0).to(device, dtype=torch.float32)
            changeflag = torch.as_tensor(changeflag).reshape(1).to(device)
            seq, pflag = beam_decode(model, tokenizer, ori_img, changeflag)
            r = []
            for one in all_caps.tolist():
                r.append(one[: one.index(end_id)] if end_id in one else one)
            hyps.append(seq)
            refs.append(r)
            preds[names[j]] = tokenizer.decode(seq).strip()
            flags[names[j]] = {"pred": pflag, "gt": int(changeflag.item())}
    elapsed = time.time() - t0
    scores = {k: round(float(v) * 100, 2) for k, v in get_eval_score(refs, hyps).items()}
    args.out = args.out or str(V2 / "outputs" / "promptcc_levir" / args.split)
    out = Path(args.out) if Path(args.out).is_absolute() else V2 / args.out  # cwd is the repo (os.chdir above)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(preds, indent=1), encoding="utf-8")
    (out / "changeflag.json").write_text(json.dumps(flags, indent=1), encoding="utf-8")
    acc = sum(v["pred"] == v["gt"] for v in flags.values()) / len(flags) * 100
    (out / "scores_repo.json").write_text(json.dumps({"n": len(preds), "seconds": round(elapsed, 1),
                                                      "changeflag_acc": round(acc, 2), **scores}, indent=2),
                                          encoding="utf-8")
    print(f"[done] n={len(preds)} {elapsed:.0f}s flag_acc={acc:.2f} scores={scores}", flush=True)


if __name__ == "__main__":
    main()

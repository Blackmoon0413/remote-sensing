#!/usr/bin/env python3
"""S5 (feasibility): can the captioner itself describe a change it missed?

For RSCaMa on LEVIR-CC test:
  * 'missed'  = official changeflag 1, prediction says no change
  * 'stable'  = official changeflag 0, prediction says no change (cost check)
Re-decode both sets greedily while banning the token at which a no-change
template diverges from every change caption of the TRAINING set
(data-driven, no test information). Report:
  * oracle upper bound: corpus scores if every missed caption is replaced by
    its re-decoded caption (selection by the official label = perfect audit)
  * examples, and what the decoder writes for truly stable scenes
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import V2, asserts_change, levir_split, score  # noqa: E402

REPO = V2 / "third_party" / "RSCaMa"
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset  # noqa: E402
from model.model_decoder import DecoderTransformer  # noqa: E402
from model.model_encoder_attMamba import AttentiveEncoder, Encoder  # noqa: E402


def banned_transitions(train_rows, vocab) -> dict[tuple, set]:
    """prefix(tuple of ids) -> ids whose choice would commit to a no-change template."""
    def ids(s):
        return tuple(vocab.get(w, vocab["<UNK>"]) for w in s.split())
    change_prefixes = set()
    for r in train_rows:
        if r["changeflag"] == 1:
            for s in r["refs"]:
                t = ids(s)
                for j in range(1, len(t) + 1):
                    change_prefixes.add(t[:j])
    ban = defaultdict(set)
    for r in train_rows:
        if r["changeflag"] == 0:
            for s in r["refs"]:
                t = ids(s)
                for j in range(len(t)):
                    if t[: j + 1] not in change_prefixes:  # first point of no return
                        ban[t[:j]].add(t[j]); break
    return dict(ban)


@torch.no_grad()
def greedy_banned(dec, x, ban, start, end, null):
    batch, channel = x.size(0), x.size(1)
    x = x.view(batch, channel, -1).permute(2, 0, 1)
    L = dec.max_lengths
    tgt = torch.zeros(batch, L, dtype=torch.int64, device=x.device)
    mask = torch.triu(torch.ones(L, L) * float("-inf"), diagonal=1).to(x.device)
    tgt[:, 0] = start
    out = []
    for step in range(L):
        emb = dec.position_encoding(dec.vocab_embedding(tgt).transpose(1, 0))
        pred = dec.wdc(dec.transformer(emb, x, tgt_mask=mask, tgt_key_padding_mask=(tgt == null)))
        scores = pred.permute(1, 0, 2)[:, step, :].squeeze(0)
        for t in ban.get(tuple(out), ()):
            scores[t] = -float("inf")
        for t in LEXICAL_BAN:
            scores[t] = -float("inf")
        nxt = int(torch.argmax(scores))
        if nxt == end:
            break
        out.append(nxt)
        if step < L - 1:
            tgt[:, step + 1] = nxt
    return out


LEXICAL_BAN: list = []


def main() -> None:
    test = levir_split("test")
    vocab = json.loads((V2 / "data" / "processed" / "rscama_levir" / "vocab.json").read_text(encoding="utf-8"))
    idx2w = list(vocab)
    ban = banned_transitions(levir_split("train"), vocab)
    global LEXICAL_BAN
    LEXICAL_BAN = [vocab[w] for w in ("same", "identical", "difference", "nothing", "unchanged") if w in vocab]
    print("[ban] lexical:", [idx2w[i] for i in LEXICAL_BAN])
    print("[ban] rules:", {" ".join(idx2w[i] for i in k) or "<start>": [idx2w[t] for t in v] for k, v in ban.items()})
    preds = json.loads((V2 / "outputs" / "rscama_levir" / "test" / "predictions.json").read_text(encoding="utf-8"))
    missed = [r for r in test if r["changeflag"] == 1 and not asserts_change(preds[r["name"]])]
    stable = [r for r in test if r["changeflag"] == 0 and not asserts_change(preds[r["name"]])][:150]
    print(f"[sets] missed={len(missed)} stable(sampled)={len(stable)}")

    enc = Encoder("CLIP-ViT-B/32")
    tr = AttentiveEncoder(n_layers=3, feature_size=[7, 7, 768], heads=8, dropout=0.1)
    dec = DecoderTransformer(decoder_type="transformer_decoder", embed_dim=768, vocab_size=len(vocab), max_lengths=42,
                             word_vocab=vocab, n_head=8, n_layers=1, dropout=0.1)
    ck = torch.load(V2 / "weights" / "rscama" / "RSCaMa.pth", map_location="cpu", weights_only=False)
    enc.load_state_dict(ck["encoder_dict"]); tr.load_state_dict(ck["encoder_trans_dict"]); dec.load_state_dict(ck["decoder_dict"])
    del ck
    enc, tr, dec = enc.cuda().eval(), tr.cuda().eval(), dec.cuda().eval()
    ds = LEVIRCCDataset("CLIP-ViT-B/32", str(V2 / "data" / "levir_cc" / "images"),
                        str(V2 / "data" / "processed" / "rscama_levir") + "/", "test",
                        str(V2 / "data" / "processed" / "rscama_levir" / "tokens") + "/", vocab, 42, 1)
    pos = {f["name"]: i for i, f in enumerate(ds.files)}
    special = {vocab["<START>"], vocab["<END>"], vocab["<NULL>"]}

    def redecode(rows):
        out = {}
        for r in rows:
            item = ds[pos[r["name"]]]
            a, b = item["imgA"].unsqueeze(0).cuda(), item["imgB"].unsqueeze(0).cuda()
            f1, f2 = enc(a, b)
            seq = greedy_banned(dec, tr(f1, f2), ban, vocab["<START>"], vocab["<END>"], vocab["<NULL>"])
            out[r["name"]] = " ".join(idx2w[i] for i in seq if i not in special)
        return out

    new_missed = redecode(missed)
    new_stable = redecode(stable)
    names = [r["name"] for r in test]
    refs = [r["refs"] for r in test]
    base = [preds[n] for n in names]
    oracle = [new_missed.get(n, preds[n]) for n in names]
    s_base, s_or = score(refs, base), score(refs, oracle)
    sub_refs = [r["refs"] for r in missed]
    s_sub_b = score(sub_refs, [preds[r["name"]] for r in missed])
    s_sub_n = score(sub_refs, [new_missed[r["name"]] for r in missed])
    still_nc = sum(not asserts_change(v) for v in new_missed.values())
    res = {"n_missed": len(missed), "redecoded_still_no_change": still_nc,
           "corpus_base": s_base, "corpus_oracle_n2c": s_or,
           "missed_subset_base": s_sub_b, "missed_subset_redecoded": s_sub_n,
           "examples_missed": [{"name": r["name"], "ref": r["refs"][0], "base": preds[r["name"]],
                                "redecoded": new_missed[r["name"]]} for r in missed[:15]],
           "examples_stable_forced": [{"name": r["name"], "forced": new_stable[r["name"]]} for r in stable[:15]]}
    out = V2 / "outputs" / "feasibility"
    out.mkdir(parents=True, exist_ok=True)
    (out / "s5_redecode_rscama_lexban.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_missed", "redecoded_still_no_change", "corpus_base", "corpus_oracle_n2c",
                                          "missed_subset_base", "missed_subset_redecoded")}, indent=1))
    for e in res["examples_missed"][:8]:
        print(f"  {e['name']} | ref: {e['ref']} | redecoded: {e['redecoded']}")
    print("  -- forced on truly stable scenes:")
    for e in res["examples_stable_forced"][:6]:
        print(f"  {e['name']} | {e['forced']}")


if __name__ == "__main__":
    main()

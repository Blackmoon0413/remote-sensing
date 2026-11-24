#!/usr/bin/env python3
"""S1: idempotent patches needed to run the official RSCaMa / PromptCC code.

1. OpenAI CLIP (venv site-packages): VisionTransformer.forward also returns the
   patch-token feature map, exactly as described by the RSCaMa/PromptCC author in
   https://github.com/Chen-Yang-Liu/PromptCC/issues/3 .
2. RSCaMa model/mamba_block.py: when the fused Mamba kernels are unavailable,
   use the pure-PyTorch reference kernels from scripts/mamba_ref_ops.py instead of
   raising NotImplementedError.

Each patch is applied only once; diffs are written to v2/patches/.
"""
from __future__ import annotations

import difflib
import os
import shutil
import sys
from pathlib import Path

V2 = Path(os.path.abspath(__file__)).parents[1]  # abspath keeps the ASCII junction path (D:/rsicc_v2); METEOR breaks on non-ASCII paths
PATCH_DIR = V2 / "patches"
PATCH_DIR.mkdir(exist_ok=True)


def apply(path: Path, edits: list[tuple[str, str]], tag: str, marker: str) -> None:
    src = path.read_text(encoding="utf-8")
    if marker in src:
        print(f"[skip] {tag}: already patched")
        return
    new = src
    for old, rep in edits:
        if new.count(old) != 1:
            raise SystemExit(f"[fail] {tag}: anchor found {new.count(old)} times:\n{old}")
        new = new.replace(old, rep)
    path.write_text(new, encoding="utf-8", newline="")
    diff = difflib.unified_diff(src.splitlines(True), new.splitlines(True), f"a/{path.name}", f"b/{path.name}")
    (PATCH_DIR / f"{tag}.patch").write_text("".join(diff), encoding="utf-8")
    print(f"[ok]   {tag}: patched {path}")


def patch_clip() -> None:
    import clip  # noqa: F401  (located in the active environment)
    path = Path(sys.modules["clip"].__file__).with_name("model.py")
    vit_old = (
        "        x = x.permute(1, 0, 2)  # LND -> NLD\n\n"
        "        x = self.ln_post(x[:, 0, :])\n\n"
        "        if self.proj is not None:\n"
        "            x = x @ self.proj\n\n"
        "        return x\n"
    )
    vit_new = (
        "        x = x.permute(1, 0, 2)  # LND -> NLD\n"
        "        featuremap = x[:, 1:, :]  # PATCH(RSICC): patch tokens, see PromptCC issue #3\n\n"
        "        x = self.ln_post(x[:, 0, :])\n\n"
        "        if self.proj is not None:\n"
        "            x = x @ self.proj\n\n"
        "        return x, featuremap\n"
    )
    apply(path, [(vit_old, vit_new)], "clip_vit_patch_tokens", "PATCH(RSICC)")


def patch_rscama() -> None:
    model_dir = V2 / "third_party" / "RSCaMa" / "model"
    shutil.copy2(V2 / "scripts" / "mamba_ref_ops.py", model_dir / "mamba_ref_ops.py")
    path = model_dir / "mamba_block.py"
    imp_old = ("from transformers.models.mamba.modeling_mamba import logger, is_fast_path_available, "
               "mamba_inner_fn, causal_conv1d_fn, causal_conv1d_update, selective_state_update, selective_scan_fn\n")
    imp_new = imp_old + (
        "# PATCH(RSICC): fall back to pure-PyTorch reference kernels when fused kernels are missing\n"
        "if not is_fast_path_available:\n"
        "    from model.mamba_ref_ops import causal_conv1d_fn, selective_scan_fn\n"
        "    is_fast_path_available = True\n"
    )
    fwd_old = '        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:\n'
    fwd_new = '        if is_fast_path_available:  # PATCH(RSICC): reference kernels also run on CPU\n'
    apply(path, [(imp_old, imp_new), (fwd_old, fwd_new)], "rscama_mamba_ref_kernels", "PATCH(RSICC)")


def patch_promptcc_meteor() -> None:
    """The PromptCC repo ships eval_func/meteor without meteor-1.5.jar; reuse RSCaMa's identical jar."""
    dst = V2 / "third_party" / "PromptCC" / "eval_func" / "meteor" / "meteor-1.5.jar"
    if dst.exists():
        print("[skip] promptcc_meteor_jar: present")
        return
    shutil.copy2(V2 / "third_party" / "RSCaMa" / "eval_func" / "meteor" / "meteor-1.5.jar", dst)
    print(f"[ok]   promptcc_meteor_jar: copied to {dst}")


def patch_change3d_microbatch() -> None:
    """Change3D/scripts/train_CC.py: gradient accumulation over micro-batches.

    The official setting (batch 32, X3D-L, 256x256) needs ~30 GB; a 16 GB GPU cannot
    hold it. Each loader batch of 32 is split into micro-batches; every micro-batch
    loss is weighted by its share of target tokens, so the accumulated gradient equals
    the full-batch token-mean gradient. One optimizer step per loader batch, exactly
    as before. (BatchNorm statistics are computed per micro-batch.)"""
    path = V2 / "third_party" / "Change3D" / "scripts" / "train_CC.py"
    loop_old = (
        "        # Move to GPU, if available\n"
        "        img_pairs = img_pairs.cuda()\n"
        "        caps = caps.cuda()\n"
        "        caplens = caplens.cuda()\n"
        "\n"
        "        # Split image pairs and encode\n"
        "        imgs_A = img_pairs[:, 0, :, :, :]\n"
        "        imgs_B = img_pairs[:, 1, :, :, :]\n"
        "        \n"
        "        # Get fused features from encoder\n"
        "        percep_feat = encoder(imgs_A, imgs_B, output_final=True)\n"
        "        percep_feat = rearrange(percep_feat, 'b c h w -> (h w) b c')\n"
        "\n"
        "        # Decode captions\n"
        "        scores, caps_sorted, decode_lengths, sort_ind = decoder(percep_feat, caps, caplens)\n"
        "\n"
        "        # Set targets (all words after <start>, up to <end>)\n"
        "        targets = caps_sorted[:, 1:]\n"
        "\n"
        "        # Remove timesteps that weren't decoded or are pads\n"
        "        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data\n"
        "        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data\n"
        "\n"
        "        # Calculate loss\n"
        "        loss = criterion(scores, targets)\n"
        "\n"
        "        # Backpropagation\n"
        "        decoder_optimizer.zero_grad()\n"
        "        if encoder_optimizer is not None:\n"
        "            encoder_optimizer.zero_grad()\n"
        "        loss.backward()\n"
    )
    loop_new = (
        "        # Move to GPU, if available\n"
        "        img_pairs = img_pairs.cuda()\n"
        "        caps = caps.cuda()\n"
        "        caplens = caplens.cuda()\n"
        "\n"
        "        # PATCH(RSICC): gradient accumulation over micro-batches (token-weighted, exact full-batch mean)\n"
        "        decoder_optimizer.zero_grad()\n"
        "        if encoder_optimizer is not None:\n"
        "            encoder_optimizer.zero_grad()\n"
        "        micro = args.micro_batch if args.micro_batch > 0 else img_pairs.size(0)\n"
        "        n_tok = float((caplens.view(-1) - 1).sum())\n"
        "        loss_sum = 0.0\n"
        "        decode_lengths_all = []\n"
        "        for s in range(0, img_pairs.size(0), micro):\n"
        "            ip, cp, cl = img_pairs[s:s + micro], caps[s:s + micro], caplens[s:s + micro]\n"
        "            percep_feat = encoder(ip[:, 0], ip[:, 1], output_final=True)\n"
        "            percep_feat = rearrange(percep_feat, 'b c h w -> (h w) b c')\n"
        "            scores, caps_sorted, decode_lengths, sort_ind = decoder(percep_feat, cp, cl)\n"
        "            targets = caps_sorted[:, 1:]\n"
        "            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data\n"
        "            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data\n"
        "            loss = criterion(scores, targets) * (targets.numel() / n_tok)\n"
        "            loss.backward()\n"
        "            loss_sum += float(loss)\n"
        "            decode_lengths_all += decode_lengths\n"
        "        loss = torch.tensor(loss_sum)\n"
        "        decode_lengths = decode_lengths_all\n"
    )
    arg_old = (
        "    parser.add_argument(\n"
        "        '--batch_size', \n"
        "        type=int, \n"
        "        default=32,\n"
        "        help='Batch size'\n"
        "    )\n"
    )
    arg_new = arg_old + (
        "    parser.add_argument('--micro_batch', type=int, default=0,\n"
        "                        help='PATCH(RSICC): micro-batch size for gradient accumulation (0 = off)')\n"
    )
    apply(path, [(loop_old, loop_new), (arg_old, arg_new)], "change3d_microbatch", "PATCH(RSICC)")


def patch_change3d_decoder_kwargs() -> None:
    """torch >= 2.1 nn.TransformerDecoder passes tgt_is_causal / memory_is_causal to each
    layer; Change3D's custom layer predates that. Accept and ignore them (no numeric change)."""
    path = V2 / "third_party" / "Change3D" / "model" / "caption_decoder.py"
    old = ("                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:\n"
           "        \"\"\"\n"
           "        Forward pass of transformer decoder layer.\n")
    new = ("                memory_key_padding_mask: Optional[Tensor] = None,\n"
           "                tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:  # PATCH(RSICC)\n"
           "        \"\"\"\n"
           "        Forward pass of transformer decoder layer.\n")
    apply(path, [(old, new)], "change3d_decoder_kwargs", "PATCH(RSICC)")


def patch_change3d_resume() -> None:
    """The official --checkpoint argument is parsed but never used. Implement resuming:
    restore model + optimizer states (incl. the halved learning rates) and continue from
    the next epoch. Model selection is done afterwards from the per-epoch val BLEU-4 log."""
    path = V2 / "third_party" / "Change3D" / "scripts" / "train_CC.py"
    old = "    # Training loop\n    for epoch in range(start_epoch, args.epochs):\n"
    new = (
        "    # PATCH(RSICC): resume from a per-epoch checkpoint\n"
        "    if args.checkpoint:\n"
        "        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)\n"
        "        trainer.encoder.load_state_dict(ck['encoder_image'])\n"
        "        trainer.decoder.load_state_dict(ck['decoder'])\n"
        "        if encoder_optimizer is not None:\n"
        "            encoder_optimizer.load_state_dict(ck['encoder_image_optimizer'].state_dict())\n"
        "        decoder_optimizer.load_state_dict(ck['decoder_optimizer'].state_dict())\n"
        "        start_epoch = ck['epoch'] + 1\n"
        "        print(f\"[resume] from epoch {ck['epoch']} (val BLEU-4 {ck['bleu-4']:.4f})\", flush=True)\n"
        "        del ck\n"
        + old)
    apply(path, [(old, new)], "change3d_resume", "PATCH(RSICC): resume")


if __name__ == "__main__":
    patch_clip()
    patch_rscama()
    patch_promptcc_meteor()
    patch_change3d_microbatch()
    patch_change3d_decoder_kwargs()
    patch_change3d_resume()

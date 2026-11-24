#!/usr/bin/env bash
# Stage-1 downloads (resumable). Each source goes into its own directory.
# Integrity is checked afterwards by s0_verify.py (size + sha256 against the HF API).
set -u
V2="$(cd "$(dirname "$0")/.." && pwd)"
HF=https://huggingface.co

get() {  # url  dest
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  echo "[$(date +%T)] start $dest"
  for attempt in 1 2 3 4 5; do
    curl -L --fail --retry 5 --retry-delay 10 -C - -s -S -o "$dest" "$url" && { echo "[$(date +%T)] done  $dest"; return 0; }
    echo "[$(date +%T)] retry $attempt for $dest"; sleep 15
  done
  echo "[$(date +%T)] FAILED $dest"
}

# small files first
get "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth" "$V2/weights/x3d/X3D_L.pyth"
get "https://disi.unitn.it/~melgani/DubaiCC.rar"                                "$V2/data/raw/dubai_cc/DubaiCC.rar"
# dataset
get "$HF/datasets/lcybuaa/LEVIR-CC/resolve/main/Levir-CC-dataset.zip"           "$V2/data/raw/levir_cc/Levir-CC-dataset.zip"
# released captioner weights (LEVIR-CC)
get "$HF/lcybuaa/RSCaMa/resolve/main/RSCaMa.pth"                                "$V2/weights/rscama/RSCaMa.pth"
get "$HF/lcybuaa/PromptCC/resolve/main/cls_model.pth.tar"                       "$V2/weights/promptcc/cls_model.pth.tar"
get "$HF/lcybuaa/PromptCC/resolve/main/BEST_checkpoint_ViT-B_32.pth.tar"        "$V2/weights/promptcc/BEST_checkpoint_ViT-B_32.pth.tar"
get "$HF/lcybuaa/PromptCC/resolve/main/TEST_LEVIR_CC_5_cap_per_img.pkl"         "$V2/weights/promptcc/TEST_LEVIR_CC_5_cap_per_img.pkl"
echo "[$(date +%T)] ALL DONE"

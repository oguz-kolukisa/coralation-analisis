#!/usr/bin/env bash
# One-shot bootstrap for a fresh 6×H200 instance.
#
# Assumes a clean Ubuntu box with: bash, git, curl, python3, and the user's
# HuggingFace token written to /coralation-analisis/.token (gated weights).
#
# What this script does, in order:
#   1. apt: ensure unzip + rsync are installed
#   2. uv sync to install Python deps
#   3. download datasets:
#        - NICO++ (12 GB zip from user's tailscale URL → /root/data/nicopp/NICOpp)
#        - Colored MNIST H5 → imagefolder (/root/data/colored_mnist_28)
#        - ImageNet-100 val split (HF cache, no special placement)
#        - Waterbirds (HF cache)
#   4. download model weights (~150 GB, all groups):
#        - torchvision (resnet/vit/swin/convnext)
#        - DINOv2
#        - CLIP variants
#        - SigLIP2 variants
#        - misc (MNIST classifiers, paulgavrikov)
#        - Qwen2.5-VL-7B-Instruct
#        - Qwen-Image-Edit
#        - LADDER Waterbirds .pkl (via download_models.py)
#   5. smoke test (one tiny end-to-end run, optional)
#   6. kick off scripts/run_6gpu.sh
#
# Run as root on the new instance:
#   cd /coralation-analisis
#   bash setup/bootstrap_h200.sh
#
# Total bootstrap time on a fresh box: ~1 hour for downloads, then the
# 6-GPU sweep takes another ~4.5 h.

set -euo pipefail

REPO=/coralation-analisis
TOKEN_FILE="$REPO/.token"

cd "$REPO"

echo "============================================================"
echo "[$(date +%H:%M:%S)] coralation 6×H200 bootstrap"
echo "============================================================"

# -- preflight ---------------------------------------------------------------
if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "ERROR: HuggingFace token not found at $TOKEN_FILE"
    echo "Create it with:  echo hf_xxx... > $TOKEN_FILE"
    echo "(needed for gated models: Qwen2.5-VL, Qwen-Image-Edit, ImageNet-1K, etc.)"
    exit 1
fi

# -- 1. OS packages ----------------------------------------------------------
echo "[$(date +%H:%M:%S)] apt: unzip + rsync"
apt-get update -qq
apt-get install -y -qq unzip rsync curl

# -- 2. Python deps via uv ---------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
echo "[$(date +%H:%M:%S)] uv sync"
uv sync

# -- 3. cache env ------------------------------------------------------------
source "$REPO/setup/env.sh"
export HF_TOKEN="$(cat "$TOKEN_FILE")"

mkdir -p /root/data /root/.cache/huggingface /root/.cache/torch

# -- 4. NICO++ ---------------------------------------------------------------
NICO_DST=/root/data/nicopp
NICO_ZIP=/root/data/nicopp_official.zip
NICO_FINAL="$NICO_DST/NICOpp"
if [[ ! -d "$NICO_FINAL" ]]; then
    if [[ ! -f "$NICO_ZIP" ]]; then
        echo "[$(date +%H:%M:%S)] downloading NICO++ (~12 GB) from tailscale..."
        curl -L --fail -C - --retry 5 --retry-delay 10 \
            "https://ubuntu.tailb5431.ts.net/Downloads/NICO%2B%2B.zip" \
            -o "$NICO_ZIP"
    fi
    echo "[$(date +%H:%M:%S)] extracting NICO++..."
    mkdir -p "$NICO_DST"
    cd "$NICO_DST"
    unzip -q -o "$NICO_ZIP"
    cd "$REPO"
    # If the zip extracts as ".../NICOpp/<domain>/..." we're done. If it has
    # an extra wrapper dir, symlink the inner dir as NICOpp.
    if [[ ! -d "$NICO_FINAL" ]]; then
        INNER=$(find "$NICO_DST" -maxdepth 2 -type d -name "*NICO*" | head -1)
        if [[ -n "$INNER" && "$INNER" != "$NICO_FINAL" ]]; then
            ln -sf "$INNER" "$NICO_FINAL"
        fi
    fi
    echo "[$(date +%H:%M:%S)] NICO++ ready: $(du -sh "$NICO_FINAL" | cut -f1)"
else
    echo "[$(date +%H:%M:%S)] NICO++ already extracted at $NICO_FINAL"
fi

# -- 5. Colored MNIST --------------------------------------------------------
MNIST_DST=/root/data/colored_mnist_28
if [[ ! -d "$MNIST_DST" || -z "$(ls -A "$MNIST_DST" 2>/dev/null)" ]]; then
    echo "[$(date +%H:%M:%S)] building Colored MNIST imagefolder..."
    uv run python "$REPO/setup/download_colored_mnist_h5.py"
fi
echo "[$(date +%H:%M:%S)] Colored MNIST: $(du -sh "$MNIST_DST" 2>/dev/null | cut -f1)"

# -- 6. Model weights (long) -------------------------------------------------
echo "[$(date +%H:%M:%S)] downloading model weights (~150 GB total)..."
# Each group exits non-zero on per-model failures but logs them; we tolerate.
uv run python "$REPO/setup/dl1_torchvision.py" || true
uv run python "$REPO/setup/dl3_resume.py"      || true   # dinov2 / resnet50
uv run python "$REPO/setup/dl2_clip.py"        || true
uv run python "$REPO/setup/dl3_siglip2.py"     || true
uv run python "$REPO/setup/dl4_misc.py"        || true   # mnist + IN-100 val
uv run python "$REPO/setup/dl5_qwen_vlm.py"    || true
uv run python "$REPO/setup/dl6_qwen_editor.py" || true
# LADDER waterbirds .pkl (and any final cleanup)
uv run python "$REPO/setup/download_models.py" || true
echo "[$(date +%H:%M:%S)] HF cache: $(du -sh /root/.cache/huggingface | cut -f1)"

# -- 7. Datasets via HF (Waterbirds, IN-100) ---------------------------------
echo "[$(date +%H:%M:%S)] caching Waterbirds + IN-100 datasets..."
uv run python "$REPO/setup/download_datasets.py" || true

# -- 8. Smoke test (optional, fast) ------------------------------------------
echo "[$(date +%H:%M:%S)] smoke test"
uv run python "$REPO/setup/smoke_test.py" || echo "  (smoke test skipped/failed — proceeding)"

# -- 9. Run the sweep --------------------------------------------------------
echo "============================================================"
echo "[$(date +%H:%M:%S)] launching 6-GPU sweep"
echo "============================================================"
bash "$REPO/scripts/run_6gpu.sh"

echo "[$(date +%H:%M:%S)] === BOOTSTRAP DONE ==="

#!/usr/bin/env bash
# Source this file before running anything that touches HuggingFace caches.
# Convention: /workspace is reserved for OUTPUTS. Caches live on rootfs.

export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export TORCH_HOME=/root/.cache/torch       # torchvision weights cache

# Larger HF download retry / timeout to survive flaky networks
export HF_HUB_DOWNLOAD_TIMEOUT=120

# Confirm
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TORCH_HOME=$TORCH_HOME"

#!/usr/bin/env bash
# Workspace-rooted env: source this on the 6×H200 instance so all caches and
# datasets read from /workspace (the persistent shared volume).

export HF_HOME=/workspace/.huggingface
export HF_DATASETS_CACHE=/workspace/.huggingface/datasets
export HF_HUB_CACHE=/workspace/.huggingface/hub
export TORCH_HOME=/workspace/.torch
export HF_HUB_DOWNLOAD_TIMEOUT=120

# Token (optional — only needed for gated HF datasets like ILSVRC/imagenet-1k)
if [[ -f /workspace/coralation-analisis/.token ]]; then
    export HF_TOKEN="$(cat /workspace/coralation-analisis/.token)"
fi

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TORCH_HOME=$TORCH_HOME"

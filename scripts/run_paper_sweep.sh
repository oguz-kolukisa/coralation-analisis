#!/usr/bin/env bash
# Paper sweep driver: 4 datasets, multi-classifier mode (one main.py call per dataset).
# Outputs land under /workspace/<dataset>/output/ to keep results-only on /workspace.
# Sequential, smallest dataset first, ImageNet-100 last.
set -euo pipefail

# Caches off /workspace per project convention
source /coralation-analisis/setup/env.sh

# Read HF token once
if [[ -f /coralation-analisis/.token ]]; then
    export HF_TOKEN="$(cat /coralation-analisis/.token)"
fi

ROOT=/workspace
mkdir -p "$ROOT"

# Common pipeline knobs (matched to example run reports/04/04.23_2)
COMMON=(
    --generations 3
    --attention scorecam
    --negative-samples 10
    --top-negative-classes 5
)

# ---------------------------------------------------------------------------
# Cell 1: Colored MNIST — 10 classes × 100 samples = 1000 positives, 16 models
# ---------------------------------------------------------------------------
DIGITS=(zero one two three four five six seven eight nine)
COLORED_MNIST_MODELS=(
    mnist_resnet_paulgavrikov mnist_vit_farleyknight mnist_siglip2
    clip_vitb32 clip_vitb16 clip_vitl14 clip_vitl14_336
    siglip2_base siglip2_base_256 siglip2_base_384 siglip2_base_512
    siglip2_large siglip2_large_384 siglip2_large_512
    siglip2_giant_256 siglip2_giant_384
)
echo "[$(date +%H:%M:%S)] === Run 1/4: Colored MNIST (16 models, 10 classes, 100 samples/class) ==="
mkdir -p "$ROOT/colored_mnist/output"
uv run python /coralation-analisis/main.py \
    --dataset colored_mnist \
    --classifiers "${COLORED_MNIST_MODELS[@]}" \
    --class-names "${DIGITS[@]}" \
    --samples 100 \
    "${COMMON[@]}" \
    --output-dir "$ROOT/colored_mnist/output" \
    2>&1 | tee "$ROOT/colored_mnist/run.log"
echo "[$(date +%H:%M:%S)] === Colored MNIST done ==="

# ---------------------------------------------------------------------------
# Cell 2: Waterbirds — 2 classes × 2900 samples (capped) = ~5800 positives, 14 models
# ---------------------------------------------------------------------------
WATERBIRDS_MODELS=(
    waterbirds_resnet_ladder
    clip_vitb32 clip_vitb16 clip_vitl14 clip_vitl14_336
    siglip2_base siglip2_base_256 siglip2_base_384 siglip2_base_512
    siglip2_large siglip2_large_384 siglip2_large_512
    siglip2_giant_256 siglip2_giant_384
)
echo "[$(date +%H:%M:%S)] === Run 2/4: Waterbirds (14 models, 2 classes, 2900 samples/class) ==="
mkdir -p "$ROOT/waterbirds/output"
uv run python /coralation-analisis/main.py \
    --dataset waterbirds \
    --classifiers "${WATERBIRDS_MODELS[@]}" \
    --class-names waterbird landbird \
    --samples 2900 \
    --negative-samples 2900 \
    --top-negative-classes 1 \
    --generations 3 --attention scorecam \
    --output-dir "$ROOT/waterbirds/output" \
    2>&1 | tee "$ROOT/waterbirds/run.log"
echo "[$(date +%H:%M:%S)] === Waterbirds done ==="

# ---------------------------------------------------------------------------
# Cell 3: NICO++ — 20 classes × 100 samples = 2000 positives, 13 models
# ---------------------------------------------------------------------------
NICO_CLASSES=(
    cactus car chair crab dolphin elephant fox giraffe gun kangaroo
    lion mailbox pumpkin racket sailboat sheep spider tent tortoise umbrella
)
NICOPP_MODELS=(
    clip_vitb32 clip_vitb16 clip_vitl14 clip_vitl14_336
    siglip2_base siglip2_base_256 siglip2_base_384 siglip2_base_512
    siglip2_large siglip2_large_384 siglip2_large_512
    siglip2_giant_256 siglip2_giant_384
)
echo "[$(date +%H:%M:%S)] === Run 3/4: NICO++ (13 models, 20 classes, 100 samples/class) ==="
mkdir -p "$ROOT/nicopp/output"
uv run python /coralation-analisis/main.py \
    --dataset nicopp \
    --classifiers "${NICOPP_MODELS[@]}" \
    --class-names "${NICO_CLASSES[@]}" \
    --samples 100 \
    "${COMMON[@]}" \
    --output-dir "$ROOT/nicopp/output" \
    2>&1 | tee "$ROOT/nicopp/run.log"
echo "[$(date +%H:%M:%S)] === NICO++ done ==="

# ---------------------------------------------------------------------------
# Cell 4: ImageNet-100 — 100 classes × 100 samples = 10000 positives, 16 models (LAST, biggest)
# ---------------------------------------------------------------------------
IMAGENET100_MODELS=(
    resnet50 resnet152
    convnext_base convnext_large
    vit_b_16 vit_l_16
    swin_t swin_b swin_v2_t swin_v2_b
    dinov2_vitb14_lc dinov2_vitl14_lc
    clip_vitb32 clip_vitl14
    siglip2_base_256 siglip2_large
)
echo "[$(date +%H:%M:%S)] === Run 4/4: ImageNet-100 (16 models, 100 classes, 100 samples/class) ==="
mkdir -p "$ROOT/imagenet100/output"
uv run python /coralation-analisis/main.py \
    --dataset imagenet100 \
    --classifiers "${IMAGENET100_MODELS[@]}" \
    --samples 100 \
    "${COMMON[@]}" \
    --output-dir "$ROOT/imagenet100/output" \
    2>&1 | tee "$ROOT/imagenet100/run.log"
echo "[$(date +%H:%M:%S)] === ImageNet-100 done ==="

echo "[$(date +%H:%M:%S)] === ALL 4 RUNS COMPLETE ==="
du -sh "$ROOT"/*/output 2>/dev/null

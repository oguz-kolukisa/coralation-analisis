#!/usr/bin/env bash
# 6-GPU parallel paper-sweep driver for a single 6×H200 instance.
#
# Layout:
#   GPU 0: Waterbirds, then Colored MNIST (sequential)  ~4.5 h
#   GPU 1: NICO++                                        ~3 h
#   GPU 2: ImageNet-100 classes 0-24                     ~3.5 h
#   GPU 3: ImageNet-100 classes 25-49                    ~3.5 h
#   GPU 4: ImageNet-100 classes 50-74                    ~3.5 h
#   GPU 5: ImageNet-100 classes 75-99                    ~3.5 h
#
# Wall-clock target: ~4.5 hours.
#
# Usage (from the repo root, after setup/bootstrap_h200.sh has finished):
#   bash scripts/run_6gpu.sh
#
# After all GPUs finish:
#   uv run python scripts/merge_imagenet100_shards.py
set -euo pipefail

# Caches live on rootfs (single-machine; /workspace not shared anymore)
source /coralation-analisis/setup/env.sh

ROOT=/root/runs
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/waterbirds" "$ROOT/colored_mnist" "$ROOT/nicopp" "$ROOT/imagenet100"

# Optional HF token
if [[ -f /coralation-analisis/.token ]]; then
    export HF_TOKEN="$(cat /coralation-analisis/.token)"
fi

# ---------------------------------------------------------------------------
# Class lists
# ---------------------------------------------------------------------------
NICO_CLASSES=(
    cactus car chair crab dolphin elephant fox giraffe gun kangaroo
    lion mailbox pumpkin racket sailboat sheep spider tent tortoise umbrella
)

# ImageNet-100 — 100 names extracted from src/imagenet100_classes.json (first
# synonym only). Order matches that JSON file. Split into 4 contiguous shards
# of 25 classes each.
IN100_ALL=(
    "tench" "goldfish" "great white shark" "tiger shark" "hammerhead"
    "electric ray" "stingray" "cock" "hen" "ostrich"
    "brambling" "goldfinch" "house finch" "junco" "indigo bunting"
    "robin" "bulbul" "jay" "magpie" "chickadee"
    "water ouzel" "kite" "bald eagle" "vulture" "great grey owl"
    "European fire salamander" "common newt" "eft" "spotted salamander" "axolotl"
    "bullfrog" "tree frog" "tailed frog" "loggerhead" "leatherback turtle"
    "mud turtle" "terrapin" "box turtle" "banded gecko" "common iguana"
    "American chameleon" "whiptail" "agama" "frilled lizard" "alligator lizard"
    "Gila monster" "green lizard" "African chameleon" "Komodo dragon" "African crocodile"
    "American alligator" "triceratops" "thunder snake" "ringneck snake" "hognose snake"
    "green snake" "king snake" "garter snake" "water snake" "vine snake"
    "night snake" "boa constrictor" "rock python" "Indian cobra" "green mamba"
    "sea snake" "horned viper" "diamondback" "sidewinder" "trilobite"
    "harvestman" "scorpion" "black and gold garden spider" "barn spider" "garden spider"
    "black widow" "tarantula" "wolf spider" "tick" "centipede"
    "black grouse" "ptarmigan" "ruffed grouse" "prairie chicken" "peacock"
    "quail" "partridge" "African grey" "macaw" "sulphur-crested cockatoo"
    "lorikeet" "coucal" "bee eater" "hornbill" "hummingbird"
    "jacamar" "toucan" "drake" "red-breasted merganser" "goose"
)
IN100_SHARD0=("${IN100_ALL[@]:0:25}")
IN100_SHARD1=("${IN100_ALL[@]:25:25}")
IN100_SHARD2=("${IN100_ALL[@]:50:25}")
IN100_SHARD3=("${IN100_ALL[@]:75:25}")

# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------
WATERBIRDS_MODELS=(
    waterbirds_resnet_ladder
    clip_vitb32 clip_vitb16 clip_vitl14 clip_vitl14_336
    siglip2_base siglip2_base_256 siglip2_base_384 siglip2_base_512
    siglip2_large siglip2_large_384 siglip2_large_512
    siglip2_giant_256 siglip2_giant_384
)
NICOPP_MODELS=(
    clip_vitb32 clip_vitb16 clip_vitl14 clip_vitl14_336
    siglip2_base siglip2_base_256 siglip2_base_384 siglip2_base_512
    siglip2_large siglip2_large_384 siglip2_large_512
    siglip2_giant_256 siglip2_giant_384
)
IMAGENET100_MODELS=(
    resnet50 resnet152
    convnext_base convnext_large
    vit_b_16 vit_l_16
    swin_t swin_b swin_v2_t swin_v2_b
    dinov2_vitb14_lc dinov2_vitl14_lc
    clip_vitb32 clip_vitl14
    siglip2_base_256 siglip2_large
)

COMMON=(
    --generations 3
    --attention scorecam
    --negative-samples 10
    --top-negative-classes 5
)

cd /coralation-analisis

echo "[$(date +%H:%M:%S)] Launching 6 GPU-pinned cells in parallel"

# GPU 0 — Waterbirds, then Colored MNIST (sequential in one bash subshell)
COLORED_MNIST_MODELS=(
    mnist_resnet_paulgavrikov mnist_vit_farleyknight mnist_siglip2
    clip_vitb32 clip_vitb16 clip_vitl14 clip_vitl14_336
    siglip2_base siglip2_base_256 siglip2_base_384 siglip2_base_512
    siglip2_large siglip2_large_384 siglip2_large_512
    siglip2_giant_256 siglip2_giant_384
)
DIGITS=(zero one two three four five six seven eight nine)

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
set -euo pipefail
cd /coralation-analisis
echo '[GPU0] Waterbirds start'
uv run python main.py \
    --dataset waterbirds \
    --classifiers ${WATERBIRDS_MODELS[*]} \
    --class-names waterbird landbird \
    --samples 300 --negative-samples 300 --top-negative-classes 1 \
    --generations 3 --attention scorecam \
    --output-dir $ROOT/waterbirds/output
echo '[GPU0] Waterbirds done; Colored MNIST start'
uv run python main.py \
    --dataset colored_mnist \
    --classifiers ${COLORED_MNIST_MODELS[*]} \
    --class-names ${DIGITS[*]} \
    --samples 100 \
    --generations 3 --attention scorecam \
    --negative-samples 10 --top-negative-classes 5 \
    --output-dir $ROOT/colored_mnist/output
echo '[GPU0] Colored MNIST done'
" > "$LOGS/gpu0_waterbirds_then_mnist.log" 2>&1 &
PID_WB=$!
echo "  GPU0 Waterbirds → Colored MNIST   PID=$PID_WB"

# GPU 1 — NICO++
CUDA_VISIBLE_DEVICES=1 nohup uv run python main.py \
    --dataset nicopp \
    --classifiers "${NICOPP_MODELS[@]}" \
    --class-names "${NICO_CLASSES[@]}" \
    --samples 100 \
    "${COMMON[@]}" \
    --output-dir "$ROOT/nicopp/output" \
    > "$LOGS/gpu1_nicopp.log" 2>&1 &
PID_NICO=$!
echo "  GPU1 NICO++                  PID=$PID_NICO"

# GPU 2 — ImageNet-100 shard 0 (classes 0-24)
mkdir -p "$ROOT/imagenet100/shard_0_25/output"
CUDA_VISIBLE_DEVICES=2 nohup uv run python main.py \
    --dataset imagenet100 \
    --classifiers "${IMAGENET100_MODELS[@]}" \
    --class-names "${IN100_SHARD0[@]}" \
    --samples 100 \
    "${COMMON[@]}" \
    --output-dir "$ROOT/imagenet100/shard_0_25/output" \
    > "$LOGS/gpu2_in100_0_25.log" 2>&1 &
PID_S0=$!
echo "  GPU2 IN-100 shard 0 (0-24)   PID=$PID_S0"

# GPU 3 — ImageNet-100 shard 1 (classes 25-49)
mkdir -p "$ROOT/imagenet100/shard_25_50/output"
CUDA_VISIBLE_DEVICES=3 nohup uv run python main.py \
    --dataset imagenet100 \
    --classifiers "${IMAGENET100_MODELS[@]}" \
    --class-names "${IN100_SHARD1[@]}" \
    --samples 100 \
    "${COMMON[@]}" \
    --output-dir "$ROOT/imagenet100/shard_25_50/output" \
    > "$LOGS/gpu3_in100_25_50.log" 2>&1 &
PID_S1=$!
echo "  GPU3 IN-100 shard 1 (25-49)  PID=$PID_S1"

# GPU 4 — ImageNet-100 shard 2 (classes 50-74)
mkdir -p "$ROOT/imagenet100/shard_50_75/output"
CUDA_VISIBLE_DEVICES=4 nohup uv run python main.py \
    --dataset imagenet100 \
    --classifiers "${IMAGENET100_MODELS[@]}" \
    --class-names "${IN100_SHARD2[@]}" \
    --samples 100 \
    "${COMMON[@]}" \
    --output-dir "$ROOT/imagenet100/shard_50_75/output" \
    > "$LOGS/gpu4_in100_50_75.log" 2>&1 &
PID_S2=$!
echo "  GPU4 IN-100 shard 2 (50-74)  PID=$PID_S2"

# GPU 5 — ImageNet-100 shard 3 (classes 75-99)
mkdir -p "$ROOT/imagenet100/shard_75_100/output"
CUDA_VISIBLE_DEVICES=5 nohup uv run python main.py \
    --dataset imagenet100 \
    --classifiers "${IMAGENET100_MODELS[@]}" \
    --class-names "${IN100_SHARD3[@]}" \
    --samples 100 \
    "${COMMON[@]}" \
    --output-dir "$ROOT/imagenet100/shard_75_100/output" \
    > "$LOGS/gpu5_in100_75_100.log" 2>&1 &
PID_S3=$!
echo "  GPU5 IN-100 shard 3 (75-99)  PID=$PID_S3"

echo "[$(date +%H:%M:%S)] All 6 launched. Waiting..."
echo "  tail -f $LOGS/gpu0_waterbirds.log    # to follow Waterbirds"
echo "  tail -f $LOGS/gpu5_in100_75_100.log  # to follow last IN-100 shard"

# Wait for all
FAIL=0
for p in $PID_WB $PID_NICO $PID_S0 $PID_S1 $PID_S2 $PID_S3; do
    if ! wait "$p"; then
        echo "[$(date +%H:%M:%S)] PID $p FAILED" >&2
        FAIL=$((FAIL+1))
    fi
done

echo "[$(date +%H:%M:%S)] === ALL 6 CELLS COMPLETE ($FAIL failures) ==="
du -sh "$ROOT"/waterbirds/output "$ROOT"/nicopp/output "$ROOT"/imagenet100/*/output 2>/dev/null

if [[ $FAIL -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] Merging ImageNet-100 shards..."
    uv run python scripts/merge_imagenet100_shards.py
fi

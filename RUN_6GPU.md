# 6×H200 Paper Sweep — One-Command Plan

A self-contained plan to run the 4-dataset coralation paper sweep on a fresh
6×H200 box. **No `/workspace`, no shared volume, no other machines.** Just
clone, drop a HuggingFace token, and run one bootstrap script.

## Prerequisites on the new instance

- Ubuntu 22.04+ with NVIDIA drivers + CUDA + 6 H200 GPUs
- Python 3.11+ available (or `uv` will install one)
- Internet access (HF + the user's tailscale URL)
- ~250 GB free space on `/` for: 12 GB NICO++, 158 GB model weights,
  ~30 GB outputs

## One-time bring-up (paste these lines)

```bash
# 1. Clone
git clone <repo-url> /coralation-analisis
cd /coralation-analisis
git checkout feat/probe-pipeline

# 2. Drop your HuggingFace token (gated weights need this)
echo 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx' > .token
chmod 600 .token

# 3. Run the bootstrap (downloads everything + launches the sweep)
bash setup/bootstrap_h200.sh
```

That's it. `bootstrap_h200.sh`:
1. installs `unzip` + `rsync`
2. `uv sync` (Python deps)
3. downloads NICO++ from your tailscale (`ubuntu.tailb5431.ts.net`) — 12 GB → `/root/data/nicopp/NICOpp`
4. builds Colored MNIST imagefolder → `/root/data/colored_mnist_28`
5. caches Waterbirds + ImageNet-100 (val split) via HF
6. downloads ~150 GB of model weights (CLIP, SigLIP2, Qwen2.5-VL, Qwen-Image-Edit, torchvision, DINOv2, MNIST classifiers, LADDER ResNet)
7. runs `scripts/run_6gpu.sh` and merges shards

Total wall clock: **~1 h bootstrap + ~4.5 h sweep = ~5.5 h end-to-end**.

## What `run_6gpu.sh` does

| GPU | Workload | ETA |
|---|---|---|
| 0 | Waterbirds → Colored MNIST (sequential) | ~4.5 h |
| 1 | NICO++ (20 classes × 100 samples × 13 models) | ~3 h |
| 2 | ImageNet-100 classes 0-24 (16 models) | ~3.5 h |
| 3 | ImageNet-100 classes 25-49 | ~3.5 h |
| 4 | ImageNet-100 classes 50-74 | ~3.5 h |
| 5 | ImageNet-100 classes 75-99 | ~3.5 h |
| **Wall clock** | | **~4.5 h** |

Each cell pins to its GPU via `CUDA_VISIBLE_DEVICES=N`. After all 6 finish,
`merge_imagenet100_shards.py` concatenates per-model JSON across the 4 IN-100
shards and regenerates a single per-model + comparison report set.

## Output layout (after the sweep)

```
/root/runs/
├── waterbirds/output/reports/        (14-model report set)
├── colored_mnist/output/reports/     (16-model report set)
├── nicopp/output/reports/            (13-model report set)
├── imagenet100/
│   ├── shard_0_25/output/reports/    (16 models × 25 classes)
│   ├── shard_25_50/output/reports/
│   ├── shard_50_75/output/reports/
│   ├── shard_75_100/output/reports/
│   └── merged/output/reports/        (consolidated 100 classes after merge)
└── logs/
    ├── gpu0_waterbirds_then_mnist.log
    ├── gpu1_nicopp.log
    ├── gpu2_in100_0_25.log
    ├── gpu3_in100_25_50.log
    ├── gpu4_in100_50_75.log
    └── gpu5_in100_75_100.log
```

## Manual restart / monitoring

The sweep script uses `nohup` + `wait`. To follow live:

```bash
tail -f /root/runs/logs/gpu0_waterbirds_then_mnist.log
nvidia-smi -l 5
ps -ef | grep main.py | grep -v grep
```

Each `main.py` invocation has built-in checkpointing
(`pipeline_v2._load_completed_checkpoints`) — if a shard dies, you can re-run
just that one command from the log header to resume from the last checkpointed
class.

## Sample sizes (defaults baked into the script)

| Dataset | classes | samples/class | negative samples | total positives |
|---|---|---|---|---|
| Waterbirds | 2 | 300 | 300 (top-1) | 600 |
| Colored MNIST | 10 | 100 | 10 (top-5) | 1000 |
| NICO++ | 20 | 100 | 10 (top-5) | 2000 |
| ImageNet-100 (per shard) | 25 | 100 | 10 (top-5) | 2500 |

The Waterbirds cell deliberately uses smaller-than-full sampling (the original
paper used the full 5800/class) to fit a 1-day budget — adjust by editing
`scripts/run_6gpu.sh` if you need to.

## If the bootstrap fails partway

Each step is idempotent — re-running `bash setup/bootstrap_h200.sh` will skip
already-completed work (existing dataset folder, populated HF cache).

The model-download steps (`dl1` → `dl6`) tolerate per-model failures and continue.

## Token contents required

The `.token` file must contain a HuggingFace token with read access to:
- `Qwen/Qwen2.5-VL-7B-Instruct` (gated)
- `Qwen/Qwen-Image-Edit` (gated)
- `ILSVRC/imagenet-1k` (gated, for IN-100 fallback / probes)
- `google/siglip2-*` (public but token avoids rate limits)
- `shawn24/Ladder` (LADDER Waterbirds ResNet checkpoint)

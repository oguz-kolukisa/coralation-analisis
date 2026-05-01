# Coralation

Automated bias / shortcut discovery and cross-model robustness evaluation for image classifiers. A single `main.py` invocation runs a 9-phase pipeline that finds the features a model relies on, distills a per-class bias catalogue, generates a synthetic probe image set (including adversarial variants designed to trick the model), and scores every classifier on that probe set.

## Requirements

- Python ≥ 3.11
- NVIDIA GPU with 16 GB+ VRAM (24 GB recommended; RTX 4090 tested)
- [uv](https://docs.astral.sh/uv/) package manager
- HuggingFace account with access to gated models

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync
```

For tests:

```bash
uv sync --extra test
```

### 3. HuggingFace token

```bash
export HF_TOKEN=hf_your_token_here
# or
echo "hf_your_token_here" > .token
```

### 4. Accept gated dataset/model terms

- https://huggingface.co/datasets/ILSVRC/imagenet-1k
- https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv

### 5. Download models (once)

```bash
uv run python download_models.py
```

Approx. 50 GB total: classifiers + VLM (Qwen2.5-VL-7B) + editor (FLUX.2-Klein-9B-KV) + dataset shards.

## Pipeline overview

9 phases (in order):

1. **Classifier baseline** — sample positives + negatives, classify, compute attention maps.
2. **VLM discovery** — identify target / environmental / negative visual features per image.
3. **Dedup + edit generation** — merge duplicates, write concrete edit instructions.
4. **Editor** — apply each edit with FLUX (text-to-image editing).
5. **Classifier measure** — re-classify edited images across every classifier.
6. **VLM verdict** — adjudicate essential vs spurious vs state_bias.
7. **Feature catalog** — aggregate per-class bias/real feature lists with strict (all models agree) and any (union) consensus.
8. **Probe generation** — VLM writes prompts → FLUX renders them in 4 variants:
   - `bias_heavy` — target class in its known biased context.
   - `bias_stripped` — the biased context alone, no target class visible.
   - `real_feature_only` — target class on a plain background, only real features.
   - `adversarial` — deliberately crafted to trick: target shown with cues of a *confusing class* (from phase 1), unusual pose/lighting, on its biased context.
9. **Probe evaluation** — every classifier runs over every probe image. Produces 4 headline metrics per model:
   - `top1_accuracy_overall`
   - `top1_accuracy_bias_heavy`
   - `top1_accuracy_bias_stripped` (LOW is good — model isn't fooled by context alone)
   - `top1_accuracy_adversarial` (LOW = model was fooled)
   - `bias_lift_score` = mean_conf(bias_heavy) − mean_conf(real_feature_only)

Outputs include `analysis_results.json`, `feature_catalog.json`, `probes/manifest.json`, `probe_evaluation.json`, per-model HTML reports, a `comparison_report.html`, and a self-contained **`probe_report.html`** with ranking + per-class image grids.

## Probe sampling

### Feature → prompt distribution

Each probe variant needs prompts that actually use the discovered features. Coralation partitions bias features **round-robin** into `N` groups and tells the VLM to use a different group per prompt, so every feature lands in at least one probe.

CLI flag: `--probe-mode {round_robin, vlm_discretion}` (default: `round_robin`).

- `round_robin`: partition `k` features into `N` groups; prompt *i* uses group *i*. Guarantees coverage.
- `vlm_discretion`: VLM picks which features to emphasise per prompt. Faster but may skip some features.

### How many prompts per variant

`--probe-samples N` (default: **auto**) controls the number of prompts per variant per class.

- Omit the flag → auto formula **`N = ceil(sqrt(k))`** per class, where `k` is the number of biased features discovered for that class. Keeps probe count balanced: more features ⇒ slightly more prompts, but sub-linear.
- Pass an explicit integer → that value is used for every class.

Example: a class with 6 biased features gets `N = ⌈√6⌉ = 3` prompts per variant → `4 × 3 = 12` images; a class with 16 features gets `N = 4` → 16 images.

**Total probe images** in a run with `C` classes: `∑_c (4 × N_c)`. With auto N, that's roughly `∑_c 4·⌈√k_c⌉`.

## Usage

### Defaults (ImageNet, all three classic classifiers)

```bash
uv run python main.py
```

### Common options

```bash
# Quick smoke: 3 classes, 3 samples each
uv run python main.py --classes 3 --samples 3

# Five classifiers (CNN + SSL + ViT + two vision-language)
uv run python main.py \
  --classifiers resnet50 dinov2_vitb14_lc vit_l_16 clip_vitb32 siglip2_base

# Switch dataset to CUB-200 bird species (uses test split automatically)
uv run python main.py --dataset cub --classifiers swin_cub clip_vitb32 siglip2_base

# Skip probe generation + evaluation (phases 8, 9) — phase 7 still runs
uv run python main.py --no-probes

# Override probe sampling
uv run python main.py --probe-samples 5 --probe-mode vlm_discretion

# Use strict consensus (all classifiers agree) instead of union
uv run python main.py --probe-feature-source strict
```

### Supported datasets

- `imagenet` (default) → `ILSVRC/imagenet-1k` validation split
- `cub` → `bentrevett/caltech-ucsd-birds-200-2011` test split (200 bird species)

### Supported classifiers

| Family | Model name | Notes |
|---|---|---|
| CNN (ImageNet) | `resnet50` | torchvision weights |
| SSL (ImageNet) | `dinov2_vitb14_lc` | DINOv2 + linear classifier |
| Transformer (ImageNet) | `vit_l_16` | SWAG end-to-end |
| Transformer (CUB) | `swin_cub` | Emiel/cub-200-bird-classifier-swin |
| Vision-Language | `clip_vitb32`, `clip_vitl14` | zero-shot, label-set configurable |
| Vision-Language | `siglip2_base`, `siglip2_large` | zero-shot sigmoid |

Pass multiple via `--classifiers a b c …`. Label sets switch automatically when `--dataset cub` is set.

### All CLI flags

```
--dataset {imagenet,cub}           Dataset preset (default: imagenet)
--classes N                        Number of classes (default: 100)
--class-names NAMES…               Specific class names
--all                              All classes in the class source
--class-source {json,dataset}      Where class names come from (preset-driven)
--class-file PATH                  Override JSON class list
--output-dir DIR                   Output directory (default: output)
--samples N                        Positive images per class (default: 10)
--negative-samples N               Negative images per confusing class (default: 5)
--top-negative-classes N           Confusing classes per target (default: 5)
--generations N                    Edits generated per hypothesis (default: 1)
--max-hypotheses N                 Edit instructions per image (default: 3)
--delta-threshold F                Min confidence delta (default: 0.15)
--classifier MODEL                 Single classifier (used if --classifiers unset)
--classifiers MODEL…               Multiple classifiers to compare
--vlm MODEL                        VLM model id (default: Qwen/Qwen2.5-VL-7B-Instruct)
--editor MODEL                     Editor model id (default: FLUX.2-klein-9b-kv)
--attention {gradcam,gradcam++,scorecam}   Attention map method
--no-stats                         Disable statistical validation
--verify                           Enable VLM edit verification (slower)
--random-classes                   Random class selection instead of deterministic
--no-probes                        Skip phases 8 + 9 (phase 7 still runs)
--probe-samples N                  Auto (ceil(sqrt(k))) if unset, else explicit N per class
--probe-mode {round_robin,vlm_discretion}   Feature distribution strategy
--probe-feature-source {strict,any} Consensus tier used for biased features (default: any)
--low-vram / --high-vram           Sequential / parallel model loading
--debug                            Write debug logs to file
-v, --verbose                      Debug logs in console too
--log-file PATH                    Log file
--hf-token TOKEN                   HF auth token
```

## Output layout

```
output/
├── images/<class>/                 # sampled positives, negatives, edited images
├── checkpoints/<model>/            # per-class analysis checkpoints
├── probes/<class_slug>/            # generated probe images
│   └── manifest.json               # prompts + seeds + feature groups per class
├── dataset.json                    # run metadata + class list
├── dataset_hf/                     # HuggingFace-format dataset dump of edited images
└── reports/
    ├── analysis_results.json       # phases 1-6 raw data
    ├── feature_catalog.json        # phase 7 per-class bias/real catalogue
    ├── probe_evaluation.json       # phase 9 4-metric scores per model
    ├── probe_report.html           # ranking + image grids + per-class tables
    ├── <model>_report.html         # per-model HTML report
    └── comparison_report.html      # cross-model comparison
```

## Running tests

```bash
# Unit tests (fast, no GPU, no model downloads)
uv run python -m pytest tests/unit/ -v

# GPU integration tests (require --gpu)
uv run python -m pytest tests/gpu/ --gpu -v
```

## Architecture

See [`src/ARCHITECTURE.md`](src/ARCHITECTURE.md) for the full module map and pipeline-phase reference.

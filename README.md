# Coralation

Automated bias/shortcut discovery for image classifiers. Uses a VLM + counterfactual editing pipeline to find features a model incorrectly relies on.

## Requirements

- Python >= 3.11
- NVIDIA GPU with 16GB+ VRAM (24GB recommended)
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

This installs all dependencies (PyTorch, Transformers, Diffusers, etc.) into a virtual environment managed by uv.

For running tests:

```bash
uv sync --extra test
```

### 3. Set up HuggingFace token

You need a HuggingFace token for downloading gated models and datasets. Get one at https://huggingface.co/settings/tokens.

**Option A** - environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

**Option B** - token file:

```bash
echo "hf_your_token_here" > .token
```

### 4. Accept gated model/dataset terms

Visit these pages and accept the terms:

- https://huggingface.co/datasets/ILSVRC/imagenet-1k
- https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv

### 5. Download models

```bash
uv run python download_models.py
```

This downloads and verifies all models (~50GB total):

| Model | Size | Purpose |
|-------|------|---------|
| ResNet-50 | ~100MB | Classifier under analysis |
| Qwen2.5-VL-7B | ~15GB | Vision-language model (feature discovery) |
| FLUX.2-Klein-9B-KV | ~18GB | Image editor (counterfactual edits) |
| ImageNet-1k validation | ~7GB | Dataset |

## Usage

### Run with defaults (20 classes)

```bash
uv run python main.py
```

### Common options

```bash
# Fewer classes for a quick test
uv run python main.py --classes 3

# More samples per class
uv run python main.py --samples 10

# Specific classes
uv run python main.py --class-names "tabby cat" "golden retriever" "sports car"

# Custom output directory
uv run python main.py --output-dir ./my_results

# Debug logging to file
uv run python main.py --debug --log-file debug.log

# Verbose (debug logs in console too)
uv run python main.py -v
```

### All CLI options

```
--classes N              Number of classes to analyze (default: 20)
--class-names CLS [CLS]  Specific class names
--all                    All 1000 ImageNet classes
--output-dir DIR         Output directory (default: output)
--samples N              Images to edit per class (default: 5)
--inspect-samples N      Images to inspect for features (default: 10)
--negative-samples N     Negative samples per class (default: 5)
--iterations N           VLM iterations per class (default: 2)
--generations N          Edited images per edit (default: 3)
--delta-threshold F      Min confidence delta (default: 0.15)
--max-hypotheses N       Max edits per image (default: 3)
--classifier MODEL       Classifier model (default: resnet50)
--vlm MODEL              VLM model (default: Qwen/Qwen2.5-VL-7B-Instruct)
--editor MODEL           Editor model (default: FLUX.2-klein-9b-kv)
--attention METHOD       gradcam, gradcam++, or scorecam (default: scorecam)
--no-stats               Disable statistical validation
--verify                 Enable VLM edit verification (slower)
--high-vram              Keep all models loaded (40GB+ VRAM)
--debug                  Write debug logs to file
-v, --verbose            Show debug logs in console
--log-file PATH          Log file path (default: coralation.log)
--hf-token TOKEN         HuggingFace token
```

## Output

Reports are generated in the output directory:

- `report.html` - Interactive HTML report with tabs
- `report.md` - Markdown summary
- `analysis_results.json` - Machine-readable results
- `<class_name>/` - Per-class images and checkpoints

## Running Tests

```bash
uv run python -m pytest tests/ -v
```

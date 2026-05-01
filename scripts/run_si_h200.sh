#!/usr/bin/env bash
# Salient ImageNet recall comparison run — H200 / H100 / A100 80 GB.
#
# Usage on the rented machine, after `uv sync` and writing .token:
#   bash scripts/run_si_h200.sh
#
# Hyperparameters tuned for recall-only comparison against SI (Singla & Feizi
# ICLR 2022). See SALIENT_IMAGENET_EXPERIMENT.md §0 for rationale.
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-output/salient_imagenet_run}
LOG_FILE=${LOG_FILE:-salient_imagenet_run.log}

if [[ ! -f .token ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: no .token file and HF_TOKEN env var unset." >&2
    echo "  echo 'hf_YOUR_TOKEN' > .token" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not installed. Run: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if ! nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 \
        | awk '$1 < 80000 { exit 1 }'; then
    echo "ERROR: GPU has < 80 GB VRAM. This run requires --no-8bit-editor which" >&2
    echo "       needs 80+ GB. Use the 24 GB command in SALIENT_IMAGENET_EXPERIMENT.md §4." >&2
    exit 1
fi

echo "==> Launching SI recall run -> $OUTPUT_DIR"
echo "==> Log file: $LOG_FILE"
echo "==> Tail with: tail -f $LOG_FILE"

nohup uv run python main.py \
    --class-file src/salient_imagenet_classes.json \
    --all \
    --classifiers resnet50 \
    --no-probes \
    --samples 10 \
    --max-hypotheses 5 \
    --top-negative-classes 2 \
    --negative-samples 2 \
    --delta-threshold 0.10 \
    --attention scorecam \
    --batch-size 30 \
    --no-8bit-editor \
    --high-vram \
    --output-dir "$OUTPUT_DIR" \
    --log-file "$LOG_FILE" \
    --debug \
    > stdout.log 2>&1 &

echo $! > run.pid
echo "==> Started (PID $(cat run.pid))"
echo "==> When it finishes, run: bash scripts/collect_si_results.sh"

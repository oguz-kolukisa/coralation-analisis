#!/usr/bin/env bash
# Run the SI comparison and tar everything needed to ship back.
#
# Usage on the rented machine, after the main run finishes:
#   bash scripts/collect_si_results.sh
#
# Produces si_results_YYYYMMDD.tgz in the project root.
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-output/salient_imagenet_run}
THRESHOLD=${THRESHOLD:-0.45}

CATALOG="$OUTPUT_DIR/reports/feature_catalog.json"
if [[ ! -f "$CATALOG" ]]; then
    echo "ERROR: $CATALOG not found." >&2
    echo "       The main run did not finish; nothing to compare." >&2
    exit 1
fi

echo "==> Running comparison vs SI ground truth (threshold=$THRESHOLD)"
uv run python scripts/compare_salient_imagenet.py \
    --classifier resnet50 \
    --catalog "$CATALOG" \
    --out-json "$OUTPUT_DIR/reports/salient_imagenet_comparison.json" \
    --out-md   "$OUTPUT_DIR/reports/salient_imagenet_comparison.md" \
    --threshold "$THRESHOLD"

ARCHIVE="si_results_$(date +%Y%m%d).tgz"
echo "==> Building $ARCHIVE"
tar -czf "$ARCHIVE" \
    "$OUTPUT_DIR/reports/" \
    "$OUTPUT_DIR/checkpoints/" \
    salient_imagenet_run.log

ls -lh "$ARCHIVE"
echo "==> Ship $ARCHIVE back to your laptop:"
echo "    scp \$USER@\$HOST:$PWD/$ARCHIVE ~/Workspace/"

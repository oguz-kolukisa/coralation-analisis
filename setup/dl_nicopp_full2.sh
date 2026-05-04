#!/usr/bin/env bash
# Resume the NICO++.zip download (curl -C -) with no max-time cap.
set -euo pipefail

SRC="https://ubuntu.tailb5431.ts.net/Downloads/NICO%2B%2B.zip"
DST=/root/data/nicopp_full.zip
DEST_DIR=/root/data/nicopp_full

echo "[$(date +%H:%M:%S)] resuming download (current size: $(du -sh "$DST" 2>/dev/null | cut -f1))..."
# -C - resume from offset, no max-time cap
curl -sL --fail -C - --retry 5 --retry-delay 10 "$SRC" -o "$DST"
echo "[$(date +%H:%M:%S)] download done. final size=$(du -sh "$DST" | cut -f1)"

mkdir -p "$DEST_DIR"
echo "[$(date +%H:%M:%S)] unpacking to $DEST_DIR..."
cd "$DEST_DIR"
unzip -q -o "$DST"
echo "[$(date +%H:%M:%S)] unpacked. layout (top 3 levels):"
find "$DEST_DIR" -maxdepth 3 -type d | head -30
echo "[$(date +%H:%M:%S)] done."

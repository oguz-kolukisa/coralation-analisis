#!/usr/bin/env bash
# Download user's full official NICO++ from their tailscale file server.
set -euo pipefail

SRC="https://ubuntu.tailb5431.ts.net/Downloads/NICO%2B%2B.zip"
DST=/root/data/nicopp_full.zip
DEST_DIR=/root/data/nicopp_full

echo "[$(date +%H:%M:%S)] downloading NICO++.zip (~12GB)..."
curl -sL --fail --max-time 1800 "$SRC" -o "$DST"
echo "[$(date +%H:%M:%S)] downloaded. size=$(du -sh "$DST" | cut -f1)"

mkdir -p "$DEST_DIR"
echo "[$(date +%H:%M:%S)] unpacking to $DEST_DIR..."
cd "$DEST_DIR"
unzip -q -o "$DST"
echo "[$(date +%H:%M:%S)] unpacked. layout (top 3 levels):"
find "$DEST_DIR" -maxdepth 3 -type d | head -30
echo "[$(date +%H:%M:%S)] done."

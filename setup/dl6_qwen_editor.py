"""Download Qwen-Image-Edit (the framework's default image editor, ~20 GB)."""
from __future__ import annotations
import os, sys, traceback
for v in ("HF_HOME", "HF_HUB_CACHE"):
    if v not in os.environ: sys.exit(f"set {v}")

TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)

print("[Qwen-Image-Edit] downloading via snapshot_download...", flush=True)
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id="Qwen/Qwen-Image-Edit", token=TOKEN)
    print(f"[Qwen-Image-Edit] OK: cached at {p}", flush=True)
    sys.exit(0)
except Exception as exc:
    traceback.print_exc()
    print(f"[Qwen-Image-Edit] FAIL: {type(exc).__name__}: {exc}", flush=True)
    sys.exit(1)

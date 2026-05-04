"""Download Qwen2.5-VL-7B-Instruct (the framework's default VLM, ~15 GB, gated)."""
from __future__ import annotations
import os, sys, traceback
for v in ("HF_HOME", "HF_HUB_CACHE"):
    if v not in os.environ: sys.exit(f"set {v}")

TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)

print("[Qwen2.5-VL-7B-Instruct] downloading via snapshot_download (gated)...", flush=True)
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct", token=TOKEN)
    print(f"[Qwen2.5-VL-7B-Instruct] OK: cached at {p}", flush=True)
    # Verify a load works
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", token=TOKEN, trust_remote_code=True)
    print("[Qwen2.5-VL-7B-Instruct] processor verified", flush=True)
    sys.exit(0)
except Exception as exc:
    traceback.print_exc()
    print(f"[Qwen2.5-VL-7B-Instruct] FAIL: {type(exc).__name__}: {exc}", flush=True)
    sys.exit(1)

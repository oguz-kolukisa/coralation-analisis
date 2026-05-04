"""Download new OpenAI CLIP size variants (group 2)."""
from __future__ import annotations
import os, sys, traceback
for v in ("HF_HOME", "HF_HUB_CACHE"):
    if v not in os.environ: sys.exit(f"set {v}")

TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)

from transformers import AutoModel, AutoProcessor

CLIPS = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
]

results = {}
for hf_id in CLIPS:
    try:
        print(f"[{hf_id}] downloading...", flush=True)
        AutoModel.from_pretrained(hf_id, token=TOKEN)
        AutoProcessor.from_pretrained(hf_id, token=TOKEN)
        print(f"[{hf_id}] OK", flush=True)
        results[hf_id] = "OK"
    except Exception as exc:
        traceback.print_exc()
        results[hf_id] = f"FAIL: {type(exc).__name__}"

print("\n=== SUMMARY ===")
for k, v in results.items():
    print(f"  {v[:4]:<4} {k}")
sys.exit(0 if all(v == "OK" for v in results.values()) else 1)

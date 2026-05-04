"""Download Google SigLIP2 size+resolution variants (group 3, ~25GB).

Excludes ``-jax`` duplicates (model-only, no PyTorch weights) and naflex
(variable-length, different ingestion). Excludes so400m (different SoViT
backbone — keep separate decision)."""
from __future__ import annotations
import os, sys, traceback
for v in ("HF_HOME", "HF_HUB_CACHE"):
    if v not in os.environ: sys.exit(f"set {v}")

TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)

from transformers import AutoModel, AutoProcessor

SIGLIPS = [
    # base — 224 already cached, skip duplicate. Add 256/384/512 for resolution variants
    "google/siglip2-base-patch16-256",
    "google/siglip2-base-patch16-384",
    "google/siglip2-base-patch16-512",
    # large
    "google/siglip2-large-patch16-256",
    "google/siglip2-large-patch16-384",
    "google/siglip2-large-patch16-512",
    # giant
    "google/siglip2-giant-opt-patch16-256",
    "google/siglip2-giant-opt-patch16-384",
]

results = {}
for hf_id in SIGLIPS:
    try:
        print(f"[{hf_id}] downloading...", flush=True)
        AutoModel.from_pretrained(hf_id, token=TOKEN)
        AutoProcessor.from_pretrained(hf_id, token=TOKEN, use_fast=False)
        print(f"[{hf_id}] OK", flush=True)
        results[hf_id] = "OK"
    except Exception as exc:
        traceback.print_exc()
        results[hf_id] = f"FAIL: {type(exc).__name__}: {str(exc)[:100]}"

print("\n=== SUMMARY ===")
for k, v in results.items():
    print(f"  {v[:4]:<4} {k}")
sys.exit(0 if all(v == "OK" for v in results.values()) else 1)

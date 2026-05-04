"""Download paulgavrikov MNIST color-aware ResNet + ImageNet-100 val split."""
from __future__ import annotations
import os, sys, traceback
for v in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
    if v not in os.environ: sys.exit(f"set {v}")

TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)

from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset

results = {}

# 1. paulgavrikov color-aware MNIST ResNet
hf_id = "paulgavrikov/mnist-resnet-color-noise-fg"
try:
    print(f"[{hf_id}] downloading...", flush=True)
    AutoModelForImageClassification.from_pretrained(hf_id, token=TOKEN)
    AutoImageProcessor.from_pretrained(hf_id, token=TOKEN)
    print(f"[{hf_id}] OK", flush=True)
    results[hf_id] = "OK"
except Exception as exc:
    traceback.print_exc()
    results[hf_id] = f"FAIL: {type(exc).__name__}: {str(exc)[:120]}"

# 2. ImageNet-100 val split only
print("\n[clane9/imagenet-100 val] downloading...", flush=True)
try:
    ds = load_dataset("clane9/imagenet-100", split="validation", token=TOKEN)
    print(f"[clane9/imagenet-100 val] OK: {len(ds)} rows, fields={list(ds[0].keys())}", flush=True)
    results["clane9/imagenet-100"] = f"OK ({len(ds)} rows)"
except Exception as exc:
    traceback.print_exc()
    results["clane9/imagenet-100"] = f"FAIL: {type(exc).__name__}"

print("\n=== SUMMARY ===")
for k, v in results.items():
    print(f"  {v[:4]:<4} {k}")
sys.exit(0 if all(str(v).startswith("OK") for v in results.values()) else 1)

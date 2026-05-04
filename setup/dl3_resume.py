"""Resume the one SigLIP2 variant that didn't finish: giant-opt-patch16-384."""
import os, sys
for v in ("HF_HOME", "HF_HUB_CACHE"):
    if v not in os.environ: sys.exit(f"set {v}")
TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)
from transformers import AutoModel, AutoProcessor
hf_id = "google/siglip2-giant-opt-patch16-384"
print(f"[{hf_id}] downloading...", flush=True)
AutoModel.from_pretrained(hf_id, token=TOKEN)
AutoProcessor.from_pretrained(hf_id, token=TOKEN, use_fast=False)
print(f"[{hf_id}] OK", flush=True)

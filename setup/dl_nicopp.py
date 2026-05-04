"""Download NICO++ via the 2018LZY/NICOplusplus HF mirror."""
import os, sys
for v in ("HF_HOME", "HF_DATASETS_CACHE"):
    if v not in os.environ: sys.exit(f"set {v}")
TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)
from datasets import load_dataset

print("[2018LZY/NICOplusplus] downloading test split...", flush=True)
ds = load_dataset("2018LZY/NICOplusplus", split="test", token=TOKEN)
print(f"[2018LZY/NICOplusplus] OK: {len(ds)} rows, fields={list(ds[0].keys())}", flush=True)

# Inspect class labels (categories)
if "category" in ds.features:
    cats = sorted(set(ds["category"]))
    print(f"  categories ({len(cats)}): {cats}")
if "domain" in ds.features:
    doms = sorted(set(ds["domain"]))
    print(f"  domains ({len(doms)}): {doms}")

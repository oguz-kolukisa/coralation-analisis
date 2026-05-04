"""Download the two HF datasets and verify each loads with one sample.

NICO++ is not on HuggingFace and must be downloaded manually from
http://nico.thumedialab.com/ — see /workspace/setup/NICOPP_INSTRUCTIONS.md.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Caches must already be set via /workspace/setup/env.sh
for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
    if var not in os.environ:
        sys.exit(f"ERROR: {var} not set. Source /coralation-analisis/setup/env.sh first.")

from datasets import load_dataset   # noqa: E402

DATASETS = [
    {
        "name": "waterbirds",
        "hf_id": "grodino/waterbirds",
        "splits": ["test"],
    },
    {
        "name": "colored_mnist",
        "hf_id": "FrankCCCCC/colored_mnist_28",
        "splits": ["test"],
    },
]


def fetch_one(spec: dict) -> tuple[bool, str]:
    """Load each split, take one sample, report fields. Returns (ok, message)."""
    try:
        for split in spec["splits"]:
            print(f"  -> loading {spec['hf_id']} split={split} ...", flush=True)
            ds = load_dataset(spec["hf_id"], split=split)
            sample = ds[0]
            keys = list(sample.keys())
            print(f"     OK: {len(ds)} rows, fields={keys}", flush=True)
        return True, "ok"
    except Exception as exc:
        traceback.print_exc()
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"HF_HOME = {os.environ['HF_HOME']}")
    print(f"HF_DATASETS_CACHE = {os.environ['HF_DATASETS_CACHE']}")
    print()
    results = {}
    for spec in DATASETS:
        print(f"== {spec['name']} ({spec['hf_id']}) ==", flush=True)
        ok, msg = fetch_one(spec)
        results[spec["name"]] = (ok, msg)
        print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    fail = 0
    for name, (ok, msg) in results.items():
        flag = "OK " if ok else "FAIL"
        print(f"{flag}  {name:20s}  {msg}")
        if not ok:
            fail += 1
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

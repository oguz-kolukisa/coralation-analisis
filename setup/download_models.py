"""Download all model checkpoints to the /workspace HF cache.

Each call below triggers a download (or a no-op if already cached). At the end
we print the final cache size and a per-model OK/FAIL line.

LADDER ResNet (Waterbirds) is shipped as a non-standard .pkl in the
shawn24/Ladder repo. We list the repo's files, locate the Waterbirds
checkpoint path, and download just that file via huggingface_hub.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

for var in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME"):
    if var not in os.environ:
        sys.exit(f"ERROR: {var} not set. Source /coralation-analisis/setup/env.sh first.")

# Read HF token if present
TOKEN_FILE = Path("/coralation-analisis/.token")
HF_TOKEN = None
if TOKEN_FILE.exists():
    HF_TOKEN = TOKEN_FILE.read_text().strip()
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

from huggingface_hub import HfApi, hf_hub_download   # noqa: E402

HF_MODELS = [
    # zero-shot CLIP / SigLIP — used across multiple datasets
    {"name": "clip_vitb32",    "hf_id": "openai/clip-vit-base-patch32",            "kind": "auto"},
    {"name": "siglip2_base",   "hf_id": "google/siglip2-base-patch16-224",         "kind": "auto"},

    # MNIST fine-tuned classifiers
    {"name": "mnist_resnet_tiny",      "hf_id": "fxmarty/resnet-tiny-mnist",                          "kind": "img_cls"},
    {"name": "mnist_vit_farleyknight", "hf_id": "farleyknight/mnist-digit-classification-2022-09-04", "kind": "img_cls"},
    {"name": "mnist_siglip2",          "hf_id": "prithivMLmods/Mnist-Digits-SigLIP2",                 "kind": "img_cls"},
]


def download_hf_model(spec: dict) -> tuple[bool, str]:
    """Trigger a from_pretrained call so HF caches the weights."""
    try:
        if spec["kind"] == "auto":
            from transformers import AutoModel, AutoProcessor
            print(f"  -> AutoModel + AutoProcessor.from_pretrained({spec['hf_id']})", flush=True)
            AutoModel.from_pretrained(spec["hf_id"], token=HF_TOKEN)
            AutoProcessor.from_pretrained(spec["hf_id"], token=HF_TOKEN)
        elif spec["kind"] == "img_cls":
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            print(f"  -> AutoModelForImageClassification.from_pretrained({spec['hf_id']})", flush=True)
            AutoModelForImageClassification.from_pretrained(spec["hf_id"], token=HF_TOKEN)
            AutoImageProcessor.from_pretrained(spec["hf_id"], token=HF_TOKEN)
        else:
            return False, f"unknown kind {spec['kind']}"
        return True, "ok"
    except Exception as exc:
        traceback.print_exc()
        return False, f"{type(exc).__name__}: {exc}"


def download_ladder_pkl() -> tuple[bool, str]:
    """Find and download the LADDER Waterbirds ResNet-50 .pkl checkpoint.

    The shawn24/Ladder repo bundles many experiment outputs; we want the
    'resnet_sup_in1k' Waterbirds 'attrNo' checkpoint per the user's table.
    """
    repo = "shawn24/Ladder"
    try:
        api = HfApi(token=HF_TOKEN)
        print(f"  -> listing files in {repo}", flush=True)
        files = api.list_repo_files(repo, token=HF_TOKEN)
        candidates = [
            f for f in files
            if "Waterbirds" in f
            and "resnet_sup_in1k" in f
            and "attrNo" in f
            and f.endswith(".pkl")
        ]
        if not candidates:
            return False, "no matching LADDER Waterbirds .pkl found in repo"
        # Prefer the shortest path (canonical seed/run dir)
        candidates.sort(key=len)
        target = candidates[0]
        print(f"  -> downloading {target}", flush=True)
        local = hf_hub_download(repo_id=repo, filename=target, token=HF_TOKEN)
        size_mb = Path(local).stat().st_size / (1024 * 1024)
        print(f"     OK: {local} ({size_mb:.1f} MB)", flush=True)
        # Remember the picked path for later loading
        marker = Path("/coralation-analisis/setup/ladder_waterbirds_pkl.path")
        marker.write_text(local + "\n")
        return True, f"ok ({size_mb:.1f} MB)"
    except Exception as exc:
        traceback.print_exc()
        return False, f"{type(exc).__name__}: {exc}"


def download_torchvision_models() -> tuple[bool, str]:
    """Trigger torchvision weight downloads to TORCH_HOME."""
    try:
        from torchvision.models import (
            resnet50, ResNet50_Weights,
            vit_l_16, ViT_L_16_Weights,
        )
        print("  -> torchvision resnet50 IMAGENET1K_V1", flush=True)
        resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        print("  -> torchvision vit_l_16 IMAGENET1K_V1", flush=True)
        vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        return True, "ok"
    except Exception as exc:
        traceback.print_exc()
        return False, f"{type(exc).__name__}: {exc}"


def download_dinov2_lc() -> tuple[bool, str]:
    """DINOv2 with ImageNet linear classifier head (the variant used by the framework)."""
    try:
        import torch
        print("  -> torch.hub dinov2_vitb14_lc", flush=True)
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc")
        return True, "ok"
    except Exception as exc:
        traceback.print_exc()
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"HF_HOME = {os.environ['HF_HOME']}")
    print(f"TORCH_HOME = {os.environ['TORCH_HOME']}")
    print(f"HF token present: {bool(HF_TOKEN)}")
    print()

    results: dict[str, tuple[bool, str]] = {}

    for spec in HF_MODELS:
        print(f"== {spec['name']} ({spec['hf_id']}) ==", flush=True)
        results[spec["name"]] = download_hf_model(spec)
        print()

    print("== ladder_waterbirds (shawn24/Ladder .pkl) ==", flush=True)
    results["ladder_waterbirds"] = download_ladder_pkl()
    print()

    print("== torchvision (resnet50, vit_l_16) ==", flush=True)
    results["torchvision_imnet"] = download_torchvision_models()
    print()

    print("== dinov2_vitb14_lc ==", flush=True)
    results["dinov2_vitb14_lc"] = download_dinov2_lc()
    print()

    print("=" * 70)
    print("MODEL DOWNLOAD SUMMARY")
    print("=" * 70)
    fail = 0
    for name, (ok, msg) in results.items():
        flag = "OK " if ok else "FAIL"
        print(f"{flag}  {name:30s}  {msg}")
        if not ok:
            fail += 1
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

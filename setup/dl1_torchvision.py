"""Download torchvision + torch.hub model weights (group 1)."""
from __future__ import annotations
import os, sys
for v in ("HF_HOME", "TORCH_HOME"):
    if v not in os.environ: sys.exit(f"set {v}")

import torch
from torchvision.models import (
    resnet152, ResNet152_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_t, Swin_T_Weights,
    swin_b, Swin_B_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_b, Swin_V2_B_Weights,
)

JOBS = [
    ("resnet152",       lambda: resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)),
    ("convnext_base",   lambda: convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)),
    ("convnext_large",  lambda: convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)),
    ("vit_b_16",        lambda: vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)),
    ("swin_t",          lambda: swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)),
    ("swin_b",          lambda: swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)),
    ("swin_v2_t",       lambda: swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)),
    ("swin_v2_b",       lambda: swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)),
    ("dinov2_vitl14_lc", lambda: torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_lc")),
]

results = {}
for name, fn in JOBS:
    print(f"[{name}] downloading...", flush=True)
    try:
        fn(); results[name] = "OK"
        print(f"[{name}] OK", flush=True)
    except Exception as exc:
        results[name] = f"FAIL: {type(exc).__name__}: {exc}"
        print(f"[{name}] FAIL: {exc}", flush=True)

print("\n=== SUMMARY ===")
for k, v in results.items():
    print(f"  {v[:4]:<4} {k}")
sys.exit(0 if all(v == "OK" for v in results.values()) else 1)

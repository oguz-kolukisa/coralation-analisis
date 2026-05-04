"""Comprehensive smoke test across all downloaded models.

Loads ImageNet-100 val + Waterbirds + Colored MNIST samples, runs one
inference per (dataset, model) cell. NICO++ skipped (manual download
pending). Qwen VLM/Editor not loaded here (separate VRAM pressure).
"""
from __future__ import annotations

import os, sys, traceback
from pathlib import Path

for v in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME"):
    if v not in os.environ: sys.exit(f"set {v}")
TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)

import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_pil(x):
    if isinstance(x, Image.Image): return x.convert("RGB")
    return Image.fromarray(x).convert("RGB")


def load_imagenet100_sample():
    ds = load_dataset("clane9/imagenet-100", split="validation", token=TOKEN)
    rec = ds[0]
    img = _to_pil(rec.get("image") or rec.get("img"))
    raw = rec.get("label", rec.get("fine_label"))
    label = ds.features["label"].names[int(raw)] if hasattr(ds.features.get("label", None), "names") else str(raw)
    return img, label


def load_waterbirds_sample():
    ds = load_dataset("grodino/waterbirds", split="test", token=TOKEN)
    rec = ds[0]
    img = _to_pil(rec.get("image") or rec.get("img"))
    raw = rec.get("label", rec.get("y"))
    return img, ["landbird", "waterbird"][int(raw)]


def load_colored_mnist_sample():
    p = sorted(Path("/root/data/colored_mnist_28/test/7").glob("*.png"))[0]
    return Image.open(p).convert("RGB"), "seven"


def pred_torchvision(img, builder, weights):
    model = builder(weights=weights).eval().to(DEVICE)
    tx = weights.transforms()
    x = tx(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): logits = model(x)[0]
    probs = F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    return weights.meta["categories"][top], round(probs[top].item(), 4)


def pred_dinov2_lc(img, hub_name):
    from torchvision import transforms
    from torchvision.models import ResNet50_Weights
    model = torch.hub.load("facebookresearch/dinov2", hub_name).eval().to(DEVICE)
    tx = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tx(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): logits = model(x)[0]
    probs = F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    return ResNet50_Weights.IMAGENET1K_V1.meta["categories"][top], round(probs[top].item(), 4)


def pred_clip_zeroshot(img, hf_id, labels):
    from transformers import AutoModel, AutoProcessor
    is_siglip = "siglip" in hf_id.lower()
    model = AutoModel.from_pretrained(hf_id, token=TOKEN).eval().to(DEVICE)
    proc = AutoProcessor.from_pretrained(hf_id, token=TOKEN, use_fast=not is_siglip)
    template = "a photo of a {l}." if is_siglip else "a photo of a {l}"
    inputs = proc(text=[template.format(l=l) for l in labels], images=img,
                  return_tensors="pt", padding=("max_length" if is_siglip else True)).to(DEVICE)
    with torch.no_grad(): out = model(**inputs)
    logits = out.logits_per_image[0]
    probs = torch.sigmoid(logits) if is_siglip else F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    return labels[top], round(probs[top].item(), 4)


def pred_hf_classifier(img, hf_id):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(hf_id, token=TOKEN).eval().to(DEVICE)
    proc = AutoImageProcessor.from_pretrained(hf_id, token=TOKEN)
    n_ch = getattr(model.config, "num_channels", 3)
    img_in = img.convert("L") if n_ch == 1 else img.convert("RGB")
    inputs = proc(images=img_in, return_tensors="pt").to(DEVICE)
    with torch.no_grad(): logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    return str(model.config.id2label.get(top, top)), round(probs[top].item(), 4)


def main() -> int:
    img_in, gold_in = load_imagenet100_sample()
    img_w, gold_w = load_waterbirds_sample()
    img_m, gold_m = load_colored_mnist_sample()
    print(f"ImageNet-100 sample gold: {gold_in!r}")
    print(f"Waterbirds gold: {gold_w!r}")
    print(f"ColoredMNIST gold: {gold_m!r}\n")

    from torchvision.models import (
        resnet50, ResNet50_Weights, resnet152, ResNet152_Weights,
        convnext_base, ConvNeXt_Base_Weights, convnext_large, ConvNeXt_Large_Weights,
        vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights,
        swin_t, Swin_T_Weights, swin_b, Swin_B_Weights,
        swin_v2_t, Swin_V2_T_Weights, swin_v2_b, Swin_V2_B_Weights,
    )

    in100 = [
        ("ImageNet-100", "resnet50",         lambda: pred_torchvision(img_in, resnet50, ResNet50_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "resnet152",        lambda: pred_torchvision(img_in, resnet152, ResNet152_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "convnext_base",    lambda: pred_torchvision(img_in, convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "convnext_large",   lambda: pred_torchvision(img_in, convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "vit_b_16",         lambda: pred_torchvision(img_in, vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "vit_l_16",         lambda: pred_torchvision(img_in, vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "swin_t",           lambda: pred_torchvision(img_in, swin_t, Swin_T_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "swin_b",           lambda: pred_torchvision(img_in, swin_b, Swin_B_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "swin_v2_t",        lambda: pred_torchvision(img_in, swin_v2_t, Swin_V2_T_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "swin_v2_b",        lambda: pred_torchvision(img_in, swin_v2_b, Swin_V2_B_Weights.IMAGENET1K_V1)),
        ("ImageNet-100", "dinov2_b14_lc",    lambda: pred_dinov2_lc(img_in, "dinov2_vitb14_lc")),
        ("ImageNet-100", "dinov2_l14_lc",    lambda: pred_dinov2_lc(img_in, "dinov2_vitl14_lc")),
        ("ImageNet-100", "clip_b32",         lambda: pred_clip_zeroshot(img_in, "openai/clip-vit-base-patch32", ["dog","cat","bird","fish","car"])),
        ("ImageNet-100", "clip_l14",         lambda: pred_clip_zeroshot(img_in, "openai/clip-vit-large-patch14", ["dog","cat","bird","fish","car"])),
        ("ImageNet-100", "siglip2_base@256", lambda: pred_clip_zeroshot(img_in, "google/siglip2-base-patch16-256", ["dog","cat","bird","fish","car"])),
        ("ImageNet-100", "siglip2_large@256",lambda: pred_clip_zeroshot(img_in, "google/siglip2-large-patch16-256", ["dog","cat","bird","fish","car"])),
    ]

    wb_clip_ids = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16",
                    "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14-336"]
    wb_siglip_ids = [
        "google/siglip2-base-patch16-224", "google/siglip2-base-patch16-256",
        "google/siglip2-base-patch16-384", "google/siglip2-base-patch16-512",
        "google/siglip2-large-patch16-256", "google/siglip2-large-patch16-384",
        "google/siglip2-large-patch16-512",
        "google/siglip2-giant-opt-patch16-256",
    ]
    wb_cells = [("Waterbirds", hf, lambda hf=hf: pred_clip_zeroshot(img_w, hf, ["waterbird","landbird"])) for hf in wb_clip_ids + wb_siglip_ids]

    digits = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    mnist_cells = [
        ("ColoredMNIST", "mnist_vit_farleyknight", lambda: pred_hf_classifier(img_m, "farleyknight/mnist-digit-classification-2022-09-04")),
        ("ColoredMNIST", "mnist_siglip2",          lambda: pred_hf_classifier(img_m, "prithivMLmods/Mnist-Digits-SigLIP2")),
        ("ColoredMNIST", "siglip2_base@224 ZS",    lambda: pred_clip_zeroshot(img_m, "google/siglip2-base-patch16-224", digits)),
    ]

    cells = in100 + wb_cells + mnist_cells
    rows = []
    for ds, name, fn in cells:
        print(f"== {ds} x {name} ==", flush=True)
        try:
            pred, conf = fn()
            ok, info = True, f"pred={pred!r} conf={conf}"
        except Exception as exc:
            traceback.print_exc()
            ok, info = False, f"{type(exc).__name__}: {str(exc)[:120]}"
        rows.append((ds, name, ok, info))
        print(f"   {'OK' if ok else 'FAIL'}: {info}\n")

    print("=" * 100)
    print("SMOKE TEST MATRIX")
    print("=" * 100)
    fail = 0
    for ds, m, ok, info in rows:
        flag = "OK  " if ok else "FAIL"
        print(f"{flag}  {ds:14s}  {m:38s}  {info}")
        if not ok: fail += 1
    print(f"\n{len(rows) - fail}/{len(rows)} passed.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

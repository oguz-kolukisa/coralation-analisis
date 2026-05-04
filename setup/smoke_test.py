"""Per-cell smoke test: load each (dataset, model) combo, run one inference.

Goal: prove that every cell in the paper-sweep matrix can actually classify
images from its dataset before we wire the full framework. Only tests the
direct-runnable cells; NICO++ cells are skipped until the dataset is
available locally.

Output: a pass/fail matrix printed at the end + non-zero exit if anything
failed.
"""
from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

for var in ("HF_HOME", "HF_HUB_CACHE"):
    if var not in os.environ:
        sys.exit(f"ERROR: {var} not set. Source /coralation-analisis/setup/env.sh first.")

TOKEN_FILE = Path("/coralation-analisis/.token")
HF_TOKEN = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from datasets import load_dataset  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Dataset loaders -----------------------------------------------------

def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")


def load_waterbirds_sample() -> tuple[Image.Image, str, dict]:
    """Return (image, label_str, full_record) for one Waterbirds test image."""
    ds = load_dataset("grodino/waterbirds", split="test", token=HF_TOKEN)
    rec = ds[0]
    keys = list(rec.keys())
    img = _to_pil(rec.get("image") or rec.get("img"))
    raw_label = rec.get("label", rec.get("y", rec.get("fine_label")))
    label_map = {0: "landbird", 1: "waterbird"}
    label = label_map.get(int(raw_label), str(raw_label))
    return img, label, {"keys": keys, "raw_label": raw_label}


def load_colored_mnist_sample() -> tuple[Image.Image, str, dict]:
    """Load one sample from the local imagefolder we built from the .h5 files."""
    root = Path("/root/data/colored_mnist_28/test")
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist; run download_colored_mnist_h5.py first")
    # Pick the first image of digit 7 (arbitrary, deterministic)
    candidates = sorted((root / "7").glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No PNGs under {root}/7")
    path = candidates[0]
    img = Image.open(path).convert("RGB")
    digit_words = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    label = digit_words[7]
    color = path.stem.split("_", 1)[0]
    return img, label, {"path": str(path), "color_attr": color}


# ----- Model loaders + inference -------------------------------------------

def predict_clip_zeroshot(img: Image.Image, hf_id: str, labels: list[str]) -> dict:
    """CLIP zero-shot classification with custom labels."""
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(hf_id, token=HF_TOKEN).eval().to(DEVICE)
    proc = AutoProcessor.from_pretrained(hf_id, token=HF_TOKEN)
    inputs = proc(text=[f"a photo of a {l}" for l in labels],
                  images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits_per_image[0]
    probs = F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    return {"pred_label": labels[top], "pred_conf": round(probs[top].item(), 4),
            "all_probs": {l: round(p.item(), 4) for l, p in zip(labels, probs)}}


def predict_siglip_zeroshot(img: Image.Image, hf_id: str, labels: list[str]) -> dict:
    """SigLIP zero-shot. Needs padding='max_length'."""
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(hf_id, token=HF_TOKEN).eval().to(DEVICE)
    proc = AutoProcessor.from_pretrained(hf_id, token=HF_TOKEN, use_fast=False)
    inputs = proc(text=[f"a photo of a {l}." for l in labels],
                  images=img, return_tensors="pt", padding="max_length").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits_per_image[0]
    probs = torch.sigmoid(logits)
    top = int(probs.argmax())
    return {"pred_label": labels[top], "pred_conf": round(probs[top].item(), 4),
            "all_probs": {l: round(p.item(), 4) for l, p in zip(labels, probs)}}


def predict_hf_classifier(img: Image.Image, hf_id: str) -> dict:
    """Standard AutoModelForImageClassification flow.

    Some MNIST classifiers (fxmarty/resnet-tiny-mnist) expect 1-channel grayscale
    input. We auto-detect that from config.num_channels and convert.
    """
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(hf_id, token=HF_TOKEN).eval().to(DEVICE)
    proc = AutoImageProcessor.from_pretrained(hf_id, token=HF_TOKEN)
    n_ch = getattr(model.config, "num_channels", 3)
    img_in = img.convert("L") if n_ch == 1 else img.convert("RGB")
    inputs = proc(images=img_in, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    label = model.config.id2label.get(top, str(top))
    return {"pred_label": str(label), "pred_conf": round(probs[top].item(), 4),
            "n_classes": len(model.config.id2label),
            "input_channels": n_ch}


def predict_ladder_resnet(img: Image.Image) -> dict:
    """Load LADDER Waterbirds ResNet-50 .pkl (SubpopBench format) and classify.

    Format (verified by inspection):
      payload = {
          'model_dict': OrderedDict {
             'featurizer.network.<resnet50 layer keys...>',
             'classifier.weight' (2, 2048),
             'classifier.bias' (2),
             'network.0.network.<duplicate ResNet>',  # ignore
             'network.1.weight' (2, 2048),            # ignore (duplicate of classifier)
          },
          'num_labels': 2, 'model_input_shape': (3, 224, 224), ...
      }
    """
    from torchvision.models import resnet50
    from torchvision import transforms
    pkl_path = Path("/coralation-analisis/setup/ladder_waterbirds_pkl.path").read_text().strip()
    payload = _safe_torch_load(pkl_path)
    md = payload["model_dict"]
    n_classes = int(payload.get("num_labels", 2))

    # Build ResNet-50 + 2-class head
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)

    # Map LADDER keys → torchvision ResNet keys
    cleaned: dict = {}
    for k, v in md.items():
        if k.startswith("featurizer.network."):
            cleaned[k[len("featurizer.network."):]] = v
        elif k == "classifier.weight":
            cleaned["fc.weight"] = v
        elif k == "classifier.bias":
            cleaned["fc.bias"] = v
        # ignore network.* duplicate
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if any("conv" in k or "bn" in k for k in missing):
        raise RuntimeError(f"Backbone weights not loaded; missing sample={missing[:5]}")
    model.eval().to(DEVICE)

    tx = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tx(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0]
    probs = F.softmax(logits, dim=-1)
    label_map = ["landbird", "waterbird"]   # SubpopBench Waterbirds convention
    top = int(probs.argmax())
    return {"pred_label": label_map[top] if top < len(label_map) else str(top),
            "pred_conf": round(probs[top].item(), 4),
            "n_classes": n_classes,
            "missing_keys": len(missing), "unexpected_keys": len(unexpected)}


def _safe_torch_load(path: str):
    """torch.load with weights_only=False fallback for older pickle artifacts."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(payload):
    """LADDER pkls may be plain state_dict or wrapped dicts ({'state_dict': ...})."""
    if isinstance(payload, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in payload and isinstance(payload[k], dict):
                return payload[k]
        # If it looks like a state dict (string keys, tensor values), use it directly
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise RuntimeError(f"Unrecognised LADDER payload structure: type={type(payload).__name__}")


def predict_torchvision_resnet50(img: Image.Image) -> dict:
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights).eval().to(DEVICE)
    tx = weights.transforms()
    x = tx(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0]
    probs = F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    label = weights.meta["categories"][top]
    return {"pred_label": label, "pred_conf": round(probs[top].item(), 4),
            "n_classes": 1000}


def predict_torchvision_vit_l_16(img: Image.Image) -> dict:
    from torchvision.models import vit_l_16, ViT_L_16_Weights
    weights = ViT_L_16_Weights.IMAGENET1K_V1
    model = vit_l_16(weights=weights).eval().to(DEVICE)
    tx = weights.transforms()
    x = tx(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0]
    probs = F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    label = weights.meta["categories"][top]
    return {"pred_label": label, "pred_conf": round(probs[top].item(), 4),
            "n_classes": 1000}


def predict_dinov2_lc(img: Image.Image) -> dict:
    from torchvision import transforms
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc").eval().to(DEVICE)
    tx = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tx(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0]
    probs = F.softmax(logits, dim=-1)
    top = int(probs.argmax())
    from torchvision.models import ResNet50_Weights
    label = ResNet50_Weights.IMAGENET1K_V1.meta["categories"][top]
    return {"pred_label": label, "pred_conf": round(probs[top].item(), 4),
            "n_classes": 1000}


# ----- Test cells ----------------------------------------------------------

@dataclass
class Cell:
    dataset: str
    model: str
    fn: Callable[[], dict]


def run_cell(c: Cell) -> tuple[bool, dict | str]:
    try:
        result = c.fn()
        return True, result
    except Exception as exc:
        traceback.print_exc()
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    # Load each dataset independently; allow one to be skipped if not ready.
    try:
        img_w, label_w, meta_w = load_waterbirds_sample()
        print(f"Waterbirds sample: label={label_w!r}, meta={meta_w}")
        wb_ready = True
    except Exception as exc:
        print(f"Waterbirds NOT READY: {type(exc).__name__}: {exc}")
        img_w, label_w, wb_ready = None, None, False
    try:
        img_m, label_m, meta_m = load_colored_mnist_sample()
        print(f"ColoredMNIST sample: label={label_m!r}, meta={meta_m}")
        mnist_ready = True
    except Exception as exc:
        print(f"ColoredMNIST NOT READY: {type(exc).__name__}: {exc}")
        img_m, label_m, mnist_ready = None, None, False
    print()

    cells = [
        # Waterbirds (binary)
        Cell("waterbirds", "clip_vitb32",
             lambda: predict_clip_zeroshot(img_w, "openai/clip-vit-base-patch32", ["waterbird", "landbird"])),
        Cell("waterbirds", "siglip2_base",
             lambda: predict_siglip_zeroshot(img_w, "google/siglip2-base-patch16-224", ["waterbird", "landbird"])),
        Cell("waterbirds", "ladder_resnet",
             lambda: predict_ladder_resnet(img_w)),
        Cell("waterbirds", "dinov2_vitb14_lc",
             lambda: predict_dinov2_lc(img_w)),
        Cell("waterbirds", "imnet_resnet50_caveat",
             lambda: predict_torchvision_resnet50(img_w)),
        Cell("waterbirds", "imnet_vit_l_16_caveat",
             lambda: predict_torchvision_vit_l_16(img_w)),

        # Colored MNIST
        Cell("colored_mnist", "mnist_resnet_tiny",
             lambda: predict_hf_classifier(img_m, "fxmarty/resnet-tiny-mnist")),
        Cell("colored_mnist", "mnist_vit_farleyknight",
             lambda: predict_hf_classifier(img_m, "farleyknight/mnist-digit-classification-2022-09-04")),
        Cell("colored_mnist", "mnist_siglip2",
             lambda: predict_hf_classifier(img_m, "prithivMLmods/Mnist-Digits-SigLIP2")),
        Cell("colored_mnist", "clip_vitb32",
             lambda: predict_clip_zeroshot(img_m, "openai/clip-vit-base-patch32",
                                          ["zero","one","two","three","four","five","six","seven","eight","nine"])),
        Cell("colored_mnist", "siglip2_base",
             lambda: predict_siglip_zeroshot(img_m, "google/siglip2-base-patch16-224",
                                            ["zero","one","two","three","four","five","six","seven","eight","nine"])),
        Cell("colored_mnist", "dinov2_vitb14_lc",
             lambda: predict_dinov2_lc(img_m)),
    ]

    rows = []
    for c in cells:
        print(f"== {c.dataset} × {c.model} ==", flush=True)
        ok, result = run_cell(c)
        rows.append((c.dataset, c.model, ok, result))
        print(f"   {'OK' if ok else 'FAIL'}: {result}")
        print()

    print("=" * 80)
    print("SMOKE TEST RESULTS")
    print("=" * 80)
    fail = 0
    print(f"Waterbirds gold label = {label_w!r}")
    print(f"ColoredMNIST gold label = {label_m!r}")
    print()
    for ds, m, ok, r in rows:
        flag = "OK  " if ok else "FAIL"
        gold = label_w if ds == "waterbirds" else label_m
        if ok:
            pred = r.get("pred_label", "?")
            conf = r.get("pred_conf", 0.0)
            match = "✓" if str(pred).lower().strip().startswith(str(gold).lower()[:3]) else " "
            print(f"{flag}  {ds:14s}  {m:25s}  pred={pred!r} conf={conf}  gold={gold!r} {match}")
        else:
            print(f"{flag}  {ds:14s}  {m:25s}  {r}")
            fail += 1
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

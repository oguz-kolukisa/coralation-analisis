"""LADDER Waterbirds ResNet-50 classifier.

The LADDER repo (shawn24/Ladder) ships its Waterbirds checkpoint as a
``model.pkl`` containing a SubpopBench-format payload with keys:
- ``model_dict``: OrderedDict with prefixes ``featurizer.network.<resnet50_keys>``
  + ``classifier.weight`` (2,2048), ``classifier.bias`` (2). Plus a
  duplicate ``network.0.network.<...>`` we ignore.
- ``num_labels``: 2 (landbird=0, waterbird=1)
- ``model_input_shape``: (3, 224, 224)
"""
from __future__ import annotations

import gc as _gc
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from .classifier import ClassifierResult
from .clip_classifier import LabelLookup, topk_from_probs

logger = logging.getLogger(__name__)


_LADDER_REGISTRY: dict[str, dict] = {
    "waterbirds_resnet_ladder": {
        "hf_repo": "shawn24/Ladder",
        "ckpt_path": "out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/model.pkl",
        "labels": ["landbird", "waterbird"],
    },
}


def available_ladder_classifiers() -> list[str]:
    """Names of registered LADDER classifiers."""
    return list(_LADDER_REGISTRY.keys())


def is_ladder_model(name: str) -> bool:
    """True if ``name`` is a registered LADDER classifier."""
    return name in _LADDER_REGISTRY


def _ladder_preprocess() -> transforms.Compose:
    """ImageNet 224×224 preprocessing — matches LADDER training."""
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _remap_state_dict(model_dict: dict) -> dict:
    """Convert LADDER ``model_dict`` keys to torchvision ResNet-50 keys."""
    out: dict = {}
    for k, v in model_dict.items():
        if k.startswith("featurizer.network."):
            out[k[len("featurizer.network."):]] = v
        elif k == "classifier.weight":
            out["fc.weight"] = v
        elif k == "classifier.bias":
            out["fc.bias"] = v
    return out


def _download_ladder_pkl(spec: dict, token: str | None) -> Path:
    """Fetch the LADDER .pkl from HuggingFace via huggingface_hub."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=spec["hf_repo"], filename=spec["ckpt_path"], token=token)
    return Path(p)


class LadderResNetClassifier:
    """Wraps a SubpopBench-format ResNet-50 .pkl checkpoint with the
    standard ImageNetClassifier surface."""

    def __init__(self, model_name: str = "waterbirds_resnet_ladder", device: str = "cuda"):
        if model_name not in _LADDER_REGISTRY:
            raise ValueError(
                f"Unknown LADDER model {model_name!r}. "
                f"Available: {available_ladder_classifiers()}"
            )
        self._model_name = model_name
        self._spec = _LADDER_REGISTRY[model_name]
        self._device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._preprocess = _ladder_preprocess()
        self._load_model()
        self.labels = list(self._spec["labels"])
        self._lookup = LabelLookup(self.labels)
        self.loaded = True

    def _load_model(self) -> None:
        """Build resnet50 with 2-class head and load LADDER weights."""
        import os
        token = os.environ.get("HF_TOKEN")
        pkl = _download_ladder_pkl(self._spec, token=token)
        payload = torch.load(str(pkl), map_location="cpu", weights_only=False)
        n = int(payload.get("num_labels", 2))
        sd = _remap_state_dict(payload["model_dict"])
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if any("conv" in k or "bn" in k for k in missing):
            raise RuntimeError(f"LADDER weights missing backbone keys: {missing[:5]}")
        self.model = model.eval().to(self.device)

    def offload(self) -> None:
        self.model = None
        _gc.collect()
        torch.cuda.empty_cache()
        self.loaded = False

    def load_to_gpu(self) -> None:
        if self.loaded:
            return
        self.device = torch.device(self._device_str if torch.cuda.is_available() else "cpu")
        self._load_model()
        self.loaded = True

    def predict(self, image: Image.Image, target_class_name: str | None = None,
                top_k: int = 5, compute_gradcam: bool = False) -> ClassifierResult:
        probs = self._image_probs(image)
        best, top_pairs = topk_from_probs(probs, self.labels, top_k)
        return ClassifierResult(
            label_idx=best, label_name=self.labels[best],
            confidence=round(probs[best].item(), 4), top_k=top_pairs,
            gradcam_image=None, attention_map=None,
        )

    def predict_with_gradcam(self, image: Image.Image, class_name: str):
        pred = self.predict(image, target_class_name=class_name, top_k=2)
        return self.get_class_confidence(image, class_name), pred

    def get_class_confidence(self, image: Image.Image, class_name: str) -> float:
        probs = self._image_probs(image)
        idx = self._lookup.resolve(class_name)
        return round(probs[idx].item(), 4)

    def _image_probs(self, image: Image.Image) -> torch.Tensor:
        x = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0]
        return F.softmax(logits, dim=-1)

    def is_valid_class(self, class_name: str) -> bool:
        return self._lookup.is_valid(class_name)

    def resolve_imagenet_labels(self, class_name: str) -> list[int]:
        return self._lookup.matches(class_name)

    def _resolve_label(self, class_name: str) -> int:
        return self._lookup.resolve(class_name)

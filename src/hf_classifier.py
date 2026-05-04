"""Thin wrapper for HuggingFace ``AutoModelForImageClassification`` models.

Used when a classifier's labels come from a HF checkpoint's ``id2label`` map
rather than ImageNet-1K (e.g. CUB-200-2011 bird species). Public surface
matches ``ImageNetClassifier`` / ``CLIPClassifier`` so the pipeline is
backbone-agnostic.
"""
from __future__ import annotations

import gc as _gc
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .classifier import ClassifierResult
from .clip_classifier import LabelLookup, topk_from_probs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_HF_REGISTRY: dict[str, dict] = {
    "swin_cub": {
        # Swin-Large fine-tuned on CUB-200-2011 (200 bird species).
        "hf_id": "Emiel/cub-200-bird-classifier-swin",
        "arch": "swin",
    },
    # Colored MNIST classifiers
    "mnist_vit_farleyknight": {
        "hf_id": "farleyknight/mnist-digit-classification-2022-09-04",
        "arch": "vit",
        # id2label = "0".."9" — remap to digit words for class-name matching
        "label_override": ["zero","one","two","three","four","five","six","seven","eight","nine"],
    },
    "mnist_siglip2": {
        "hf_id": "prithivMLmods/Mnist-Digits-SigLIP2",
        "arch": "siglip",
        "label_override": ["zero","one","two","three","four","five","six","seven","eight","nine"],
    },
    "mnist_resnet_paulgavrikov": {
        # Color-aware MNIST ResNet (3-channel, 28x28). Repo lacks
        # preprocessor_config.json so we supply our own transform.
        "hf_id": "paulgavrikov/mnist-resnet-color-noise-fg",
        "arch": "resnet",
        "label_override": ["zero","one","two","three","four","five","six","seven","eight","nine"],
        "manual_transform": "mnist_28_3ch",
    },
}


def available_hf_classifiers() -> list[str]:
    """Return names of registered HF classifier shortcuts."""
    return list(_HF_REGISTRY.keys())


def is_hf_classifier(name: str) -> bool:
    """True if ``name`` names a HF classification checkpoint."""
    return name in _HF_REGISTRY


def _build_manual_transform(name: str | None):
    """Return a torchvision Compose for the named manual-transform spec, else None."""
    if name is None:
        return None
    from torchvision import transforms
    if name == "mnist_28_3ch":
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    raise ValueError(f"Unknown manual_transform spec: {name!r}")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class HFClassifier:
    """Image classifier backed by a HuggingFace classification model."""

    def __init__(self, model_name: str = "swin_cub", device: str = "cuda"):
        if model_name not in _HF_REGISTRY:
            raise ValueError(
                f"Unknown HF classifier '{model_name}'. "
                f"Available: {available_hf_classifiers()}"
            )
        self._model_name = model_name
        self._spec = _HF_REGISTRY[model_name]
        self._device_str = device
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self._load_model()
        self.labels = self._extract_labels()
        self._lookup = LabelLookup(self.labels)
        self.loaded = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the HF model + processor (or fall back to a manual transform)."""
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        hf_id = self._spec["hf_id"]
        logger.debug("Loading HF classifier %s (%s)", self._model_name, hf_id)
        self.model = AutoModelForImageClassification.from_pretrained(hf_id)
        self.model.eval().to(self.device)
        self._manual_transform = _build_manual_transform(self._spec.get("manual_transform"))
        self.processor = None if self._manual_transform else AutoImageProcessor.from_pretrained(hf_id)

    def _extract_labels(self) -> list[str]:
        """Order-preserving list of class names. Honour label_override if present."""
        if "label_override" in self._spec:
            return list(self._spec["label_override"])
        id2label = self.model.config.id2label
        n = len(id2label)
        return [str(id2label[i]) for i in range(n)]

    def offload(self) -> None:
        """Free VRAM — drop model, processor, label bank."""
        self.model = None
        self.processor = None
        _gc.collect()
        torch.cuda.empty_cache()
        self.loaded = False

    def load_to_gpu(self) -> None:
        """Reload the model if previously offloaded."""
        if self.loaded:
            return
        self.device = torch.device(
            self._device_str if torch.cuda.is_available() else "cpu"
        )
        self._load_model()
        self.loaded = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, image: Image.Image, target_class_name: str | None = None,
        top_k: int = 5, compute_gradcam: bool = False,
    ) -> ClassifierResult:
        """Run inference; return top-k + confidence. No Grad-CAM for HF models."""
        probs = self._image_probs(image)
        best_idx, top_pairs = topk_from_probs(probs, self.labels, top_k)
        return ClassifierResult(
            label_idx=best_idx,
            label_name=self.labels[best_idx],
            confidence=round(probs[best_idx].item(), 4),
            top_k=top_pairs,
            gradcam_image=None,
            attention_map=None,
        )

    def predict_with_gradcam(
        self, image: Image.Image, class_name: str,
    ) -> tuple[float, ClassifierResult]:
        """Return (target_class_conf, prediction) — no Grad-CAM for HF models."""
        pred = self.predict(image, target_class_name=class_name, top_k=5)
        return self.get_class_confidence(image, class_name), pred

    def get_class_confidence(
        self, image: Image.Image, class_name: str,
    ) -> float:
        """Return softmax probability for a specific class."""
        probs = self._image_probs(image)
        idx = self._lookup.resolve(class_name)
        conf = round(probs[idx].item(), 4)
        logger.debug(
            "HF.get_class_confidence: class='%s' (idx=%d) -> %.4f",
            class_name, idx, conf,
        )
        return conf

    def _image_probs(self, image: Image.Image) -> torch.Tensor:
        """Image → softmax distribution over all classes."""
        n_ch = getattr(self.model.config, "num_channels", 3)
        img_in = image.convert("L") if n_ch == 1 else image.convert("RGB")
        if self._manual_transform is not None:
            x = self._manual_transform(img_in).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x).logits
        else:
            inputs = self.processor(images=img_in, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
        return F.softmax(logits[0], dim=-1)

    # ------------------------------------------------------------------
    # Label inspection — matches the ImageNetClassifier surface
    # ------------------------------------------------------------------

    def is_valid_class(self, class_name: str) -> bool:
        """True if the class name resolves to a label."""
        return self._lookup.is_valid(class_name)

    def resolve_imagenet_labels(self, class_name: str) -> list[int]:
        """All label indices matching the class name substring."""
        return self._lookup.matches(class_name)

    def _resolve_label(self, class_name: str) -> int:
        """Internal helper kept for API parity with other classifiers."""
        return self._lookup.resolve(class_name)

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
        # NOTE: HaotianZG/vit-cub-200-2011-bird was rejected — despite the
        # repo name its id2label still exposes ImageNet-1K (1000 classes).
        "hf_id": "Emiel/cub-200-bird-classifier-swin",
        "arch": "swin",
    },
}


def available_hf_classifiers() -> list[str]:
    """Return names of registered HF classifier shortcuts."""
    return list(_HF_REGISTRY.keys())


def is_hf_classifier(name: str) -> bool:
    """True if ``name`` names a HF classification checkpoint."""
    return name in _HF_REGISTRY


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
        """Load the HF model + its image processor."""
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        hf_id = self._spec["hf_id"]
        logger.debug("Loading HF classifier %s (%s)", self._model_name, hf_id)
        self.model = AutoModelForImageClassification.from_pretrained(hf_id)
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.model.eval().to(self.device)

    def _extract_labels(self) -> list[str]:
        """Order-preserving list of class names from the model's id2label map."""
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
        inputs = self.processor(
            images=image.convert("RGB"), return_tensors="pt",
        ).to(self.device)
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

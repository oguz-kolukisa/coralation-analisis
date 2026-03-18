"""
ResNet-50 classifier wrapper with attention map generation.
Supports Grad-CAM, Grad-CAM++, and Score-CAM methods.
"""
from __future__ import annotations
import logging
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from .models.attention_maps import get_attention_generator, AttentionMapGenerator

logger = logging.getLogger(__name__)

# ImageNet normalization
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])


class ClassifierResult(NamedTuple):
    label_idx: int
    label_name: str
    confidence: float
    top_k: list[tuple[str, float]]   # [(label, confidence), ...]
    gradcam_image: Image.Image | None  # heatmap overlaid on original
    attention_map: np.ndarray | None = None  # raw (H,W) in [0,1]


class ImageNetClassifier:
    """Wraps ResNet-50 with configurable attention map support."""

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "cuda",
        attention_method: str = "gradcam++",
    ):
        self._model_name = model_name
        self._device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.attention_method = attention_method
        logger.debug("Loading %s on %s with %s", model_name, self.device, attention_method)

        self._load_model()
        self.loaded = True

    def _load_model(self):
        """Create model, labels, and attention generator from scratch."""
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        self.model.eval().to(self.device)

        self.labels: list[str] = weights.meta["categories"]
        self._label_to_idx: dict[str, int] = {
            lbl.lower(): i for i, lbl in enumerate(self.labels)
        }

        # Attention map generator (uses pluggable method)
        self._attention_generator: Optional[AttentionMapGenerator] = None
        self._init_attention_generator()

    def offload(self):
        """Free VRAM by deleting the model entirely."""
        import gc
        del self.model
        self.model = None
        self._attention_generator = None
        gc.collect()
        torch.cuda.empty_cache()
        self.loaded = False

    def load_to_gpu(self):
        """Reload model to GPU if it was offloaded."""
        if not self.loaded:
            self.device = torch.device(
                self._device_str if torch.cuda.is_available() else "cpu"
            )
            self._load_model()
            self.loaded = True

    def _init_attention_generator(self):
        """Initialize the attention map generator based on configured method."""
        self._attention_generator = get_attention_generator(self.attention_method)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image, target_class_name: str | None = None,
                top_k: int = 5, compute_gradcam: bool = True) -> ClassifierResult:
        """Run inference and optionally compute attention map."""
        tensor = _PREPROCESS(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Forward pass (no gradients needed for basic prediction)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_indices = probs.argsort(descending=True)[:top_k].tolist()
        top_k_results = [(self.labels[i], round(probs[i].item(), 4)) for i in top_indices]

        pred_idx = top_indices[0]
        pred_name = self.labels[pred_idx]
        pred_conf = probs[pred_idx].item()

        gradcam_img = None
        raw_attention = None
        if compute_gradcam and self._attention_generator:
            target_idx = pred_idx
            if target_class_name:
                target_idx = self._resolve_label(target_class_name)

            # Use the pluggable attention generator
            raw_attention = self._attention_generator.generate(
                self.model, tensor, target_idx
            )
            gradcam_img = self._attention_generator.overlay_on_image(
                raw_attention, image
            )

        return ClassifierResult(
            label_idx=pred_idx,
            label_name=pred_name,
            confidence=round(pred_conf, 4),
            top_k=top_k_results,
            gradcam_image=gradcam_img,
            attention_map=raw_attention,
        )

    def predict_with_gradcam(
        self, image: Image.Image, class_name: str,
    ) -> tuple[float, ClassifierResult]:
        """Return (target_class_confidence, full_prediction_with_gradcam)."""
        pred = self.predict(image, target_class_name=class_name, compute_gradcam=True)
        conf = self._target_confidence(image, class_name)
        return conf, pred

    def _target_confidence(self, image: Image.Image, class_name: str) -> float:
        """Compute softmax confidence for a specific class."""
        tensor = _PREPROCESS(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        idx = self._resolve_label(class_name)
        return round(probs[idx].item(), 4)

    def get_class_confidence(self, image: Image.Image, class_name: str) -> float:
        """Return confidence score for a specific class (without Grad-CAM)."""
        tensor = _PREPROCESS(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        idx = self._resolve_label(class_name)
        conf = round(probs[idx].item(), 4)
        logger.debug("Classifier.get_class_confidence: class='%s' (idx=%d) -> %.4f", class_name, idx, conf)
        return conf

    def is_valid_class(self, class_name: str) -> bool:
        """Check if a class name resolves to a valid ImageNet label."""
        try:
            self._resolve_label(class_name)
            return True
        except ValueError:
            return False

    def resolve_imagenet_labels(self, class_name: str) -> list[int]:
        """Return all label indices that match the class name (substring match)."""
        name_lower = class_name.lower()
        return [i for i, lbl in enumerate(self.labels)
                if name_lower in lbl.lower() or lbl.lower() in name_lower]

    # ------------------------------------------------------------------
    # Label resolution
    # ------------------------------------------------------------------

    def _resolve_label(self, class_name: str) -> int:
        name_lower = class_name.lower()
        if name_lower in self._label_to_idx:
            return self._label_to_idx[name_lower]
        # fuzzy match
        for lbl, idx in self._label_to_idx.items():
            if name_lower in lbl or lbl in name_lower:
                return idx
        # No match found - raise with suggestions
        similar = [lbl for lbl in self._label_to_idx.keys()
                   if any(word in lbl for word in name_lower.split())][:5]
        raise ValueError(
            f"Class '{class_name}' not found in ImageNet labels. "
            f"Similar classes: {similar if similar else 'none found'}"
        )

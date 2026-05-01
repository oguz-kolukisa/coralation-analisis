"""CLIP / SigLIP text-image classifier.

Performs zero-shot ImageNet classification by cosine similarity between an
image embedding and a bank of per-class text embeddings. Picks the highest-
similarity class as the prediction.

Exposes the same public surface as ``ImageNetClassifier`` so the rest of the
pipeline is agnostic to the backbone.
"""
from __future__ import annotations

import gc as _gc
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights

from .classifier import ClassifierResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry — add a new CLIP/SigLIP model by appending here
# ---------------------------------------------------------------------------
_CLIP_REGISTRY: dict[str, dict] = {
    "clip_vitb32": {
        "hf_id": "openai/clip-vit-base-patch32",
        "kind": "clip",
        "prompt": "a photo of a {label}",
    },
    "clip_vitl14": {
        "hf_id": "openai/clip-vit-large-patch14",
        "kind": "clip",
        "prompt": "a photo of a {label}",
    },
    "siglip2_base": {
        "hf_id": "google/siglip2-base-patch16-224",
        "kind": "siglip",
        "prompt": "a photo of a {label}.",
    },
    "siglip2_large": {
        "hf_id": "google/siglip2-large-patch16-256",
        "kind": "siglip",
        "prompt": "a photo of a {label}.",
    },
}


def available_clip_classifiers() -> list[str]:
    """Return the list of registered CLIP/SigLIP model names."""
    return list(_CLIP_REGISTRY.keys())


def is_clip_model(name: str) -> bool:
    """True if the given classifier name refers to a CLIP/SigLIP backbone."""
    return name in _CLIP_REGISTRY


# ---------------------------------------------------------------------------
# Label lookup — shared by both classifier backbones
# ---------------------------------------------------------------------------

class LabelLookup:
    """Case-insensitive + fuzzy substring lookup over ImageNet-1K labels."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self._by_lower: dict[str, int] = {
            lbl.lower(): i for i, lbl in enumerate(labels)
        }

    def resolve(self, class_name: str) -> int:
        """Return the label index for a class name, with fuzzy fallback."""
        key = class_name.lower()
        if key in self._by_lower:
            return self._by_lower[key]
        for lbl, idx in self._by_lower.items():
            if key in lbl or lbl in key:
                return idx
        raise ValueError(
            f"Class '{class_name}' not found in ImageNet label set"
        )

    def is_valid(self, class_name: str) -> bool:
        """True if this class name resolves to a label."""
        try:
            self.resolve(class_name)
            return True
        except ValueError:
            return False

    def matches(self, class_name: str) -> list[int]:
        """All label indices whose lower-case name substring-matches."""
        key = class_name.lower()
        return [
            i for i, lbl in enumerate(self.labels)
            if key in lbl.lower() or lbl.lower() in key
        ]


def imagenet_label_lookup() -> LabelLookup:
    """Return a LabelLookup over the standard ImageNet-1K class names."""
    return LabelLookup(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])


def _short_label(label: str) -> str:
    """Keep only the first comma-separated token.

    ImageNet labels like 'goldfish, Carassius auratus' contain Latin names that
    embed poorly in CLIP/SigLIP text encoders; taking just 'goldfish' improves
    zero-shot cosine similarity substantially.
    """
    first = label.split(",", 1)[0].strip()
    return first or label


# ---------------------------------------------------------------------------
# Scoring helpers — pure math, independent of model I/O (unit-testable)
# ---------------------------------------------------------------------------

def similarity_logits(
    image_feat: torch.Tensor, text_bank: torch.Tensor, logit_scale: float,
) -> torch.Tensor:
    """Cosine similarity logits between a normalized image feat and text bank."""
    return (image_feat @ text_bank.T * logit_scale).squeeze(0)


def logits_to_probs(logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Convert similarity logits to per-class probabilities."""
    if kind == "siglip":
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def topk_from_probs(
    probs: torch.Tensor, labels: list[str], k: int,
) -> tuple[int, list[tuple[str, float]]]:
    """Return (best_idx, [(label, score), ...]) from a 1D probability tensor."""
    k = min(k, probs.shape[0])
    top_idx = probs.argsort(descending=True)[:k].tolist()
    pairs = [(labels[i], round(probs[i].item(), 4)) for i in top_idx]
    return top_idx[0], pairs


def unwrap_features(output) -> torch.Tensor:
    """Extract the pooled embedding tensor from a transformers model output.

    Newer transformers versions wrap ``get_text_features`` /
    ``get_image_features`` results in a ``BaseModelOutputWithPooling``.
    Older versions returned the tensor directly. Handle both.
    """
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("pooler_output", "text_embeds", "image_embeds",
                 "last_hidden_state"):
        candidate = getattr(output, attr, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError(
        f"Cannot extract feature tensor from {type(output).__name__}"
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class CLIPClassifier:
    """Zero-shot image classifier over ImageNet labels via CLIP/SigLIP."""

    def __init__(self, model_name: str = "clip_vitb32", device: str = "cuda",
                  label_lookup: "LabelLookup | None" = None):
        if model_name not in _CLIP_REGISTRY:
            raise ValueError(
                f"Unknown CLIP model '{model_name}'. "
                f"Available: {available_clip_classifiers()}"
            )
        self._model_name = model_name
        self._spec = _CLIP_REGISTRY[model_name]
        self._device_str = device
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self._lookup = label_lookup or imagenet_label_lookup()
        self.labels = self._lookup.labels
        self._text_bank: Optional[torch.Tensor] = None
        self._load_model()
        self._encode_text_bank()
        self.loaded = True

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the backbone and processor from HuggingFace.

        SigLIP ships a training-time image processor that the 'fast' default
        silently replaces — the replacement uses a different resize/crop
        interpolation and produces degenerate image embeddings on 768×768
        inputs. Force the slow processor for SigLIP kinds.
        """
        from transformers import AutoModel, AutoProcessor
        hf_id = self._spec["hf_id"]
        logger.debug("Loading CLIP %s (%s)", self._model_name, hf_id)
        self.model = AutoModel.from_pretrained(hf_id)
        self.processor = AutoProcessor.from_pretrained(
            hf_id, use_fast=(self._spec["kind"] != "siglip"),
        )
        self.model.eval().to(self.device)

    def _encode_text_bank(self) -> None:
        """Precompute normalized text embeddings for all class prompts.

        For multi-part labels (e.g. 'goldfish, Carassius auratus') keep only
        the first comma-separated component; long latinised names embed badly
        in CLIP/SigLIP text encoders trained on natural web captions.

        SigLIP requires ``padding='max_length'`` — with dynamic padding its
        text embeddings collapse to ~zero cosine similarity with the image
        encoder (fixed-length training-time positional encoding).
        """
        prompts = [
            self._spec["prompt"].format(label=_short_label(lbl))
            for lbl in self.labels
        ]
        padding = "max_length" if self._spec["kind"] == "siglip" else True
        inputs = self.processor(
            text=prompts, return_tensors="pt", padding=padding,
        ).to(self.device)
        with torch.no_grad():
            output = self.model.get_text_features(**inputs)
        feats = unwrap_features(output)
        self._text_bank = F.normalize(feats, dim=-1)

    def offload(self) -> None:
        """Free VRAM by deleting the model, processor, and text bank."""
        self.model = None
        self.processor = None
        self._text_bank = None
        _gc.collect()
        torch.cuda.empty_cache()
        self.loaded = False

    def load_to_gpu(self) -> None:
        """Reload model and text bank if previously offloaded."""
        if self.loaded:
            return
        self.device = torch.device(
            self._device_str if torch.cuda.is_available() else "cpu"
        )
        self._load_model()
        self._encode_text_bank()
        self.loaded = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, image: Image.Image, target_class_name: str | None = None,
        top_k: int = 5, compute_gradcam: bool = False,
    ) -> ClassifierResult:
        """Return best-class prediction and top-k from image-text similarity."""
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
        """Return (target_class_confidence, prediction). No Grad-CAM for CLIP."""
        pred = self.predict(image, target_class_name=class_name, top_k=5)
        return self.get_class_confidence(image, class_name), pred

    def get_class_confidence(
        self, image: Image.Image, class_name: str,
    ) -> float:
        """Return the probability for a specific class."""
        probs = self._image_probs(image)
        idx = self._lookup.resolve(class_name)
        conf = round(probs[idx].item(), 4)
        logger.debug(
            "CLIP.get_class_confidence: class='%s' (idx=%d) -> %.4f",
            class_name, idx, conf,
        )
        return conf

    def _image_probs(self, image: Image.Image) -> torch.Tensor:
        """Image → probability distribution over all classes."""
        feat = self._encode_image(image)
        logits = similarity_logits(feat, self._text_bank, self._logit_scale())
        logits = logits + self._logit_bias()
        return logits_to_probs(logits, self._spec["kind"])

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image into a normalized embedding."""
        inputs = self.processor(
            images=image.convert("RGB"), return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model.get_image_features(**inputs)
        feat = unwrap_features(output)
        return F.normalize(feat, dim=-1)

    def _logit_scale(self) -> float:
        """Temperature factor learned by the backbone (or 1.0 if absent)."""
        scale = getattr(self.model, "logit_scale", None)
        if scale is None:
            return 1.0
        return scale.exp().item()

    def _logit_bias(self) -> float:
        """SigLIP's learned logit bias (~ −12). Zero for CLIP which has none."""
        bias = getattr(self.model, "logit_bias", None)
        if bias is None:
            return 0.0
        return bias.item()

    # ------------------------------------------------------------------
    # Label inspection — matches ImageNetClassifier surface
    # ------------------------------------------------------------------

    def is_valid_class(self, class_name: str) -> bool:
        """True if the class name resolves to a label."""
        return self._lookup.is_valid(class_name)

    def resolve_imagenet_labels(self, class_name: str) -> list[int]:
        """All label indices matching the class name substring."""
        return self._lookup.matches(class_name)

    def _resolve_label(self, class_name: str) -> int:
        """Internal helper kept for API parity with ImageNetClassifier."""
        return self._lookup.resolve(class_name)
